# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import time
from collections import deque
from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import BatchType, HiddenChannelType, SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class PrefillState(enum.Enum):
    """Edge-side prefill in-flight state machine.

    See Phase5 design in ``PDbatch分离边云协同Phase5&7详细设计.md``.
    """
    IDLE = "idle"       # prefill_inflight_count == 0
    LOW = "low"         # prefill_inflight_count == 1
    HIGH = "high"       # prefill_inflight_count >= prefill_inflight_limit


class HiddenChannelManager:
    """Manages data-plane hidden tensor channels for edge-cloud PD separation.

    Two prefill channels (PREFILL_1 / PREFILL_2) support 2P1D; one decode
    channel (DECODE) supports single in-flight decode. Channels are allocated
    in FIFO order and freed when the tail segment completes.
    """

    def __init__(self) -> None:
        self._free_prefills: deque[HiddenChannelType] = deque([
            HiddenChannelType.PREFILL_1,
            HiddenChannelType.PREFILL_2,
        ])
        # Mapping from head_token to the allocated channel.  Only prefill
        # batches are recorded here; decode batches always use DECODE and
        # do not need a mapping.
        self._head_token_to_channel: dict[str, HiddenChannelType] = {}

    # ------------------------------------------------------------------ #
    # Prefill channel allocation / release                               #
    # ------------------------------------------------------------------ #
    def allocate_prefill(self, head_token: str) -> HiddenChannelType:
        """Allocate a free prefill channel for the batch identified by
        ``head_token``. Raises if none available."""
        if not self._free_prefills:
            raise RuntimeError(
                "No free prefill hidden channel available"
            )
        channel = self._free_prefills.popleft()
        self._head_token_to_channel[head_token] = channel
        print(
            f"\r\n[PD-CHAN] allocate prefill channel={channel.value} "
            f"head_token={head_token}",
            flush=True,
        )
        return channel

    def release_prefill(self, head_token: str) -> HiddenChannelType | None:
        """Release the prefill channel previously allocated for
        ``head_token``. Returns the freed channel (or None if not found)."""
        channel = self._head_token_to_channel.pop(head_token, None)
        if channel is None:
            return None
        self._free_prefills.append(channel)
        print(
            f"[PD-CHAN] release prefill channel={channel.value} "
            f"head_token={head_token}",
            flush=True,
        )
        return channel

    def has_free_prefill(self) -> bool:
        return bool(self._free_prefills)

    # ------------------------------------------------------------------ #
    # Decode channel (always DECODE, no free-list)                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def decode_channel() -> HiddenChannelType:
        return HiddenChannelType.DECODE

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #
    def get_channel(self, head_token: str) -> HiddenChannelType | None:
        return self._head_token_to_channel.get(head_token)

    @property
    def in_use_prefills(self) -> list[HiddenChannelType]:
        return list(self._head_token_to_channel.values())


class _NoChunkSentinel:
    """Minimal ec_connector stand-in that disables chunking in parent Scheduler.

    临时设为 ``Scheduler.ec_connector``，使父 ``Scheduler.schedule()``
    跳过 token_budget / long_prefill_token_threshold / enable_chunked_prefill
    三个chunking限制——与真实ec_connector的bypass行为一致。

    PD分离不使用ec_connector的encoder-cache或metadata功能，
    因此所有方法为no-op。
    """

    def update_state_after_alloc(self, request, index):
        pass

    def build_connector_meta(self, scheduler_output):
        return None

    def has_cache_item(self, identifier):
        return False


class PDSeparatedScheduler(Scheduler):
    """Scheduler that separates prefill and decode into distinct steps.

    In edge-cloud PD-separated mode the four cardinal phases are:
      - PREFILL_FIRST  (edge head segment)
      - PREFILL_LAST   (edge tail segment, sourced from cloud-returned outputs)
      - DECODE_FIRST   (Phase 4)
      - DECODE_LAST    (Phase 4)

    This class owns the request bookkeeping for *first* segments
    (``chunk_prefill_first`` + parent's ``waiting`` / ``running``) and the
    ready queues for *last* segments (``prefills_last_ready`` /
    ``decodes_last_ready``), which are filled by the EngineCore from the
    POST_OUT channel before each ``schedule()`` call.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Requests that have started their P-first segment but have not yet
        # been fully consumed (still chunking, or still in flight on cloud).
        self.chunk_prefill_first: list[Request] = []

        # SchedulerOutputs returned from cloud (POST_OUT channel) carrying
        # the metadata needed to execute the edge tail segment.
        # Populated by EngineCore.step() before calling self.schedule().
        self.prefills_last_ready: deque[SchedulerOutput] = deque()
        self.decodes_last_ready: deque[SchedulerOutput] = deque()

        self._step_counter: int = 0

        # In-flight prefill limit (head-segment batches).
        self.prefill_inflight_limit: int = getattr(
            self.scheduler_config, "pd_prefill_inflight_limit", 1
        )
        self.prefill_inflight_count: int = 0
        self.decode_inflight_limit: int = 1
        self.decode_inflight_count: int = 0

        # Phase6 data-plane channel manager.  Two prefill hidden channels are
        # available for 2P1D; decode uses a dedicated fixed channel.
        self.hidden_channel_manager = HiddenChannelManager()

        # Buffer queue: requests whose P-first segment is done but P-last
        # segment has not yet returned from the cloud.  Not eligible for
        # decode scheduling until PL completes and they are moved to running.
        self.prefill_last_pending: list[Request] = []

    def schedule(self) -> SchedulerOutput:
        return self._schedule_pd_separated()

    def _make_empty_batch(self) -> SchedulerOutput:
        scheduler_output = SchedulerOutput.make_empty()
        scheduler_output.finished_req_ids = self.finished_req_ids
        self.finished_req_ids = set()
        return scheduler_output

    def _schedule_pd_separated(self) -> SchedulerOutput:
        state = self._prefill_state()
        scheduler_output = self._pick_by_state(state)
        has_work = scheduler_output.total_num_scheduled_tokens > 0
        is_tail = scheduler_output.batch_type in (
            BatchType.PREFILL_LAST,
            BatchType.DECODE_LAST,
        )
        if has_work or is_tail:
            self._log_scheduler_state(state, scheduler_output.batch_type)
        return scheduler_output

    def _pick_by_state(self, state: PrefillState) -> SchedulerOutput:
        if state == PrefillState.IDLE:
            # IDLE: P首/chunk0首 > D首 > D尾 > Empty.
            if self._can_schedule_prefill_first():
                return self._pick_prefill_first_batch()
            if self._can_schedule_decode_first():
                return self._pick_decode_first_batch()
            if self.decodes_last_ready:
                return self._pick_decode_last_batch()
            return self._make_empty_batch()

        if state == PrefillState.LOW:
            # LOW: chunk/P首(when slot available) > P尾 > D首 > D尾 > Empty.
            if self._can_schedule_prefill_first():
                return self._pick_prefill_first_batch()
            if self.prefills_last_ready:
                return self._pick_prefill_last_batch()
            if self._can_schedule_decode_first():
                return self._pick_decode_first_batch()
            if self.decodes_last_ready:
                return self._pick_decode_last_batch()
            return self._make_empty_batch()

        # HIGH: P尾 > D首 > D尾 > Empty. New P首 is forbidden.
        if self.prefills_last_ready:
            return self._pick_prefill_last_batch()
        if self._can_schedule_decode_first():
            return self._pick_decode_first_batch()
        if self.decodes_last_ready:
            return self._pick_decode_last_batch()
        return self._make_empty_batch()

    def is_waiting_for_remote_tail(self) -> bool:
        """True when local requests exist only as remote in-flight work.

        In this state the edge has no local batch to execute until POST_OUT
        returns a PREFILL_LAST/DECODE_LAST, so the EngineCore should yield
        instead of tight-loop scheduling EMPTY batches.
        """
        return bool(
            (self.prefill_inflight_count > 0 or self.decode_inflight_count > 0)
            and not self.prefills_last_ready
            and not self.decodes_last_ready
            and not self._can_schedule_prefill_first()
            and not self._can_schedule_decode_first()
        )

    def _prefill_state(self) -> PrefillState:
        if self.prefill_inflight_count <= 0:
            return PrefillState.IDLE
        if (
            self.prefill_inflight_limit > 1
            and self.prefill_inflight_count >= self.prefill_inflight_limit
        ):
            return PrefillState.HIGH
        return PrefillState.LOW

    def _has_prefill_work(self) -> bool:
        return bool(self.chunk_prefill_first or self.waiting)

    def _can_schedule_prefill_first(self) -> bool:
        return (
            self._has_prefill_work()
            and self.prefill_inflight_count < self.prefill_inflight_limit
            and self.hidden_channel_manager.has_free_prefill()
        )

    def _can_schedule_decode_first(self) -> bool:
        return bool(
            self.running
            and self.decode_inflight_count < self.decode_inflight_limit
        )

    def _log_scheduler_state(self, state: PrefillState, batch_type: BatchType) -> None:
        self._step_counter += 1
        print(
            f"[PD] Step{self._step_counter}, state is {state}, batch_type is {batch_type}, "
            f"waiting[]: {len(self.waiting)}, "
            f"chunk_prefill_first[]: {len(self.chunk_prefill_first)}, "
            f"prefill_last_pending[]: {len(self.prefill_last_pending)}, "
            f"running[]: {len(self.running)}, "
            f"prefills_last_ready[]: {len(self.prefills_last_ready)}, "
            f"decodes_last_ready[]: {len(self.decodes_last_ready)}, "
            f"prefill_inflight: {self.prefill_inflight_count}/{self.prefill_inflight_limit}, "
            f"decode_inflight: {self.decode_inflight_count}/{self.decode_inflight_limit}",
            flush=True,
        )
        for req in self.chunk_prefill_first:
            print(
                f"[PD] chunk_prefill_first[{req.request_id}],    "
                f"num_prompt_tokens: {req.num_prompt_tokens}, "
                f"num_tokens: {req.num_tokens}, "
                f"num_computed_tokens: {req.num_computed_tokens}, "
                f"chunk_num: {req.chunk_num}",
                flush=True,
            )
        for req in self.running:
            print(
                f"[PD] running[{req.request_id}],    "
                f"num_prompt_tokens: {req.num_prompt_tokens}, "
                f"num_tokens: {req.num_tokens}, "
                f"num_computed_tokens: {req.num_computed_tokens}, "
                f"chunk_num: {req.chunk_num}",
                flush=True,
            )

    def _pick_prefill_first_batch(self) -> SchedulerOutput:
        saved_running = self.running
        saved_chunk_prefill_first = self.chunk_prefill_first
        saved_max_num_running_reqs = self.max_num_running_reqs

        # 保存父调度器的chunking相关参数，临时禁用以确保
        # 每个prefill请求一步内完整调度（is_prefill_chunk始终为False）。
        # PD分离下chunked prefill需要多次edge→cloud→edge往返，极其低效。
        saved_ec_connector = self.ec_connector
        saved_max_num_scheduled_tokens = self.max_num_scheduled_tokens
        saved_long_prefill_token_threshold = (
            self.scheduler_config.long_prefill_token_threshold
        )

        self.running = list(saved_chunk_prefill_first)
        self.chunk_prefill_first = []
        self.max_num_running_reqs -= len(saved_running)

        # 禁用父调度器的所有chunking限制：
        # 1. ec_connector非None → 跳过token_budget/long_prefill_token_threshold/
        #    enable_chunked_prefill三个限制（与真实ec_connector bypass一致）
        # 2. 扩大max_num_scheduled_tokens → 纵深防御
        # 3. 设long_prefill_token_threshold=0 → 纵深防御
        self.ec_connector = _NoChunkSentinel()
        self.max_num_scheduled_tokens = 2**31
        self.scheduler_config.long_prefill_token_threshold = 0

        scheduler_output = None
        try:
            scheduler_output = super().schedule()
        finally:
            self.max_num_running_reqs = saved_max_num_running_reqs
            # 恢复父调度器的chunking限制
            self.ec_connector = saved_ec_connector
            self.max_num_scheduled_tokens = saved_max_num_scheduled_tokens
            self.scheduler_config.long_prefill_token_threshold = (
                saved_long_prefill_token_threshold
            )
            # 清除sentinel可能设置的ec_connector_metadata
            if scheduler_output is not None:
                scheduler_output.ec_connector_metadata = None

            if scheduler_output is not None:
                if scheduler_output.total_num_scheduled_tokens == 0:
                    scheduler_output.batch_type = BatchType.EMPTY
                else:
                    scheduler_output.batch_type = BatchType.PREFILL_FIRST
                    scheduler_output.head_token = uuid4().hex
                    scheduler_output.hidden_channel = (
                        self.hidden_channel_manager.allocate_prefill(
                            scheduler_output.head_token
                        )
                    )
                    self.prefill_inflight_count += 1
                new_chunk_prefill_first = [
                    req for req in self.running if req.is_prefill_chunk
                ]
                new_prefill_completed = [
                    req for req in self.running if not req.is_prefill_chunk
                ]
                # 防御性检查：PD分离下不应出现chunked prefill
                if new_chunk_prefill_first:
                    logger.warning(
                        "[PD] %d request(s) still have is_prefill_chunk=True "
                        "after disabling chunking limits. This should not "
                        "happen and may indicate a bug or KV cache exhaustion.",
                        len(new_chunk_prefill_first),
                    )
                for req in self.chunk_prefill_first:
                    if req not in new_chunk_prefill_first:
                        new_chunk_prefill_first.append(req)
                self.chunk_prefill_first = new_chunk_prefill_first
                # PF 首段完成但 PL 尾段未完成的请求进入缓冲队列，
                # 不直接加入 running，避免 decode 调度器误调度。
                self.prefill_last_pending.extend(new_prefill_completed)
                self.running = saved_running
                print(
                    f"[PD] _pick_prefill_first_batch done: "
                    f"chunk_prefill_first[]: {len(self.chunk_prefill_first)}, "
                    f"prefill_last_pending[]: {len(self.prefill_last_pending)}, "
                    f"running[]: {len(self.running)}, "
                    f"prefill_inflight: {self.prefill_inflight_count}/{self.prefill_inflight_limit}",
                    flush=True,
                )
                for (
                    req_id,
                    num_scheduled_token,
                ) in scheduler_output.num_scheduled_tokens.items():
                    req = self.requests[req_id]
                    print(
                        f"[PD] Scheduled[{req_id}], "
                        f"num_tokens: {req.num_tokens}, "
                        f"num_scheduled_token: {num_scheduled_token}, "
                        f"num_computed_tokens: {req.num_computed_tokens}, "
                        f"is_prefill_chunk: {req.is_prefill_chunk}, "
                        f"chunk_num: {req.chunk_num}",
                        flush=True,
                    )
            else:
                self.chunk_prefill_first = saved_chunk_prefill_first
                self.running = saved_running

        return scheduler_output  # type: ignore[return-value]

    def _pick_prefill_last_batch(self) -> SchedulerOutput:
        """Pop one cloud-returned SchedulerOutput from prefills_last_ready.

        The cloud has already rewritten ``batch_type=PREFILL_LAST`` and kept
        all original KV / sampling metadata intact, so the edge worker can
        directly run segment_e + sampler on it. We also remove the involved
        requests from ``chunk_prefill_first`` so the parent class's
        ``update_from_output`` does not double-account them.
        """
        if not self.prefills_last_ready:
            return self._make_empty_batch()
        so = self.prefills_last_ready.popleft()
        assert so.batch_type == BatchType.PREFILL_LAST, (
            f"prefills_last_ready expects PREFILL_LAST, got {so.batch_type}"
        )
        # Drop these reqs from chunk_prefill_first. Keep them in
        # prefill_last_pending until update_from_output() moves them to running.
        last_req_ids = set(so.num_scheduled_tokens.keys())
        if last_req_ids:
            self.chunk_prefill_first = [
                req for req in self.chunk_prefill_first
                if req.request_id not in last_req_ids
            ]
        self._validate_prefill_tail_channel(so)
        print(
            f"\r\n[PD] _pick_prefill_last_batch popped {len(last_req_ids)} reqs; "
            f"remaining prefills_last_ready[]: {len(self.prefills_last_ready)}, "
            f"prefill_last_pending[]: {len(self.prefill_last_pending)}, "
            f"hidden_channel: {so.hidden_channel}",
            flush=True,
        )
        return so

    def _validate_prefill_tail_channel(self, scheduler_output: SchedulerOutput) -> None:
        token = scheduler_output.head_token
        channel = scheduler_output.hidden_channel
        if not token:
            raise RuntimeError("PREFILL_LAST missing head_token")
        if channel not in (HiddenChannelType.PREFILL_1, HiddenChannelType.PREFILL_2):
            raise RuntimeError(
                f"PREFILL_LAST expects a prefill hidden channel, got {channel}"
            )
        expected = self.hidden_channel_manager.get_channel(token)
        if expected != channel:
            raise RuntimeError(
                f"PREFILL_LAST hidden channel mismatch: expected {expected}, "
                f"got {channel}, head_token={token}"
            )

    def _validate_decode_tail_channel(self, scheduler_output: SchedulerOutput) -> None:
        if scheduler_output.hidden_channel != HiddenChannelType.DECODE:
            raise RuntimeError(
                "DECODE_LAST expects decode hidden channel, got "
                f"{scheduler_output.hidden_channel}"
            )

    def _pick_decode_last_batch(self) -> SchedulerOutput:
        if not self.decodes_last_ready:
            return self._make_empty_batch()
        so = self.decodes_last_ready.popleft()
        assert so.batch_type == BatchType.DECODE_LAST, (
            f"decodes_last_ready expects DECODE_LAST, got {so.batch_type}"
        )
        self._validate_decode_tail_channel(so)
        print(
            f"\r\n[PD] _pick_decode_last_batch popped "
            f"{len(so.num_scheduled_tokens)} reqs; "
            f"remaining decodes_last_ready[]: {len(self.decodes_last_ready)}",
            flush=True,
        )
        return so

    def _ensure_cached_all_token_ids(
        self, scheduler_output: SchedulerOutput,
    ) -> None:
        """Ensure every cached decode req carries all_token_ids.

        In PD-separated mode the edge worker's persistent input_batch may not
        retain a request across the PF → PL → DF transitions (the tail segment
        may have been skipped by ``_update_states``), yet the async-scheduling
        model runner requires ``all_token_ids`` to reconstruct
        ``output_token_ids`` for any resumed (req_index is None) request with
        ``num_output_tokens > 0``.

        The base ``_make_cached_request_data`` only fills ``all_token_ids``
        when the request was *not* scheduled in the previous step
        (``prev_step_scheduled_req_ids``).  Because PD separation interleaves
        PF / PL / DF / DL phases — and PL/DL are popped from cloud-returned
        queues without going through ``super().schedule()`` — the scheduler's
        ``prev_step_scheduled_req_ids`` can be stale or misleading, causing
        ``all_token_ids`` to be omitted for requests that the worker actually
        needs it for.

        This helper unconditionally back-fills ``all_token_ids`` for any
        cached request that has output tokens but is missing from the dict.
        The extra payload is cheap (control-plane only) and avoids the
        ``KeyError`` in ``gpu_model_runner._update_states``.
        """
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for req_id, num_output_tokens in zip(
            cached_reqs.req_ids,
            cached_reqs.num_output_tokens,
        ):
            if num_output_tokens > 0 and req_id not in cached_reqs.all_token_ids:
                cached_reqs.all_token_ids[req_id] = (
                    self.requests[req_id].all_token_ids.copy()
                )

    def _pick_decode_first_batch(self) -> SchedulerOutput:
        if not self.running:
            return self._make_empty_batch()

        saved_chunk_prefill_first = self.chunk_prefill_first
        saved_waiting = self.waiting
        saved_skipped = self.skipped_waiting

        self.chunk_prefill_first = []
        self.waiting = create_request_queue(self.policy)
        self.skipped_waiting = create_request_queue(self.policy)

        scheduler_output = None
        try:
            scheduler_output = super().schedule()
        finally:
            if scheduler_output is not None:
                if scheduler_output.total_num_scheduled_tokens == 0:
                    scheduler_output.batch_type = BatchType.EMPTY
                else:
                    scheduler_output.batch_type = BatchType.DECODE_FIRST
                    scheduler_output.head_token = uuid4().hex
                    scheduler_output.hidden_channel = (
                        self.hidden_channel_manager.decode_channel()
                    )
                    self._ensure_cached_all_token_ids(scheduler_output)
                    self.decode_inflight_count += 1
                for req in list(self.waiting):
                    saved_waiting.prepend_request(req)
                self.chunk_prefill_first = saved_chunk_prefill_first
                self.waiting = saved_waiting
                self.skipped_waiting = saved_skipped
                print(
                    f"\r\n[PD] _pick_decode_first_batch done: "
                    f"running: {len(self.running)}, "
                    f"chunk_prefill_first: {len(self.chunk_prefill_first)}, "
                    f"prefill_last_pending: {len(self.prefill_last_pending)}",
                    flush=True,
                )
                for (
                    req_id,
                    num_scheduled_token,
                ) in scheduler_output.num_scheduled_tokens.items():
                    req = self.requests[req_id]
                    print(
                        f"[PD] Scheduled[{req_id}],    "
                        f"num_tokens: {req.num_tokens}, "
                        f"num_scheduled_token: {num_scheduled_token}, "
                        f"num_computed_tokens: {req.num_computed_tokens}, "
                        f"is_prefill_chunk: {req.is_prefill_chunk}, "
                        f"chunk_num: {req.chunk_num}",
                        flush=True,
                    )
            else:
                self.chunk_prefill_first = saved_chunk_prefill_first
                self.waiting = saved_waiting
                self.skipped_waiting = saved_skipped

        return scheduler_output  # type: ignore[return-value]

    def _migrate_prefill_to_running(self) -> None:
        completed = [
            req for req in self.chunk_prefill_first if not req.is_prefill_chunk
        ]
        if completed:
            print(
                f"[PD] _migrate_prefill_to_running: moving {len(completed)} "
                f"requests from chunk_prefill_first to running",
                flush=True,
            )
        for req in completed:
            self.chunk_prefill_first.remove(req)
            self.running.append(req)

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_preemptions += 1
        if request.spec_token_ids:
            request.spec_token_ids = []
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        if request.is_prefill_chunk:
            print(
                f"[PD] _preempt_request: request {request.request_id} "
                f"stays in chunk_prefill_first "
                f"(computed={request.num_computed_tokens})",
                flush=True,
            )
            self.chunk_prefill_first.append(request)
        else:
            print(
                f"[PD] _preempt_request: request {request.request_id} "
                f"goes back to waiting (decode or finished prefill)",
                flush=True,
            )
            request.num_computed_tokens = 0
            self.waiting.prepend_request(request)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        was_prefill_map = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            was_prefill_map[req_id] = self.requests[req_id].is_prefill_chunk

        super()._update_after_schedule(scheduler_output)

        for req_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items():
            if was_prefill_map[req_id] and num_scheduled_token > 0:
                self.requests[req_id].chunk_num += 1
                print(
                    f"[PD] _update_after_schedule: request {req_id} "
                    f"chunk_num={self.requests[req_id].chunk_num} "
                    f"tokens={self.requests[req_id].num_tokens} "
                    f"scheduled={num_scheduled_token} "
                    f"computed={self.requests[req_id].num_computed_tokens} "
                    f"is_prefill_chunk={self.requests[req_id].is_prefill_chunk}",
                    flush=True,
                )

        self._migrate_prefill_to_running()
        self.finished_req_ids = set()

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, Any]:
        if scheduler_output.batch_type == BatchType.PREFILL_LAST:
            if self.prefill_inflight_count > 0:
                self.prefill_inflight_count -= 1
            if scheduler_output.head_token:
                self.hidden_channel_manager.release_prefill(
                    scheduler_output.head_token
                )
            # Move completed requests from prefill_last_pending to running.
            completed_req_ids = set(scheduler_output.num_scheduled_tokens.keys())
            newly_running = [
                req for req in self.prefill_last_pending
                if req.request_id in completed_req_ids
            ]
            self.prefill_last_pending = [
                req for req in self.prefill_last_pending
                if req.request_id not in completed_req_ids
            ]
            self.running.extend(newly_running)
            print(
                f"[PD] update_from_output PREFILL_LAST done, "
                f"prefill_inflight: {self.prefill_inflight_count}/{self.prefill_inflight_limit}, "
                f"moved {len(newly_running)} reqs to running[], running[]: {len(self.running)}",
                flush=True,
            )
        if scheduler_output.batch_type == BatchType.DECODE_LAST:
            if self.decode_inflight_count > 0:
                self.decode_inflight_count -= 1
            print(
                f"[PD] update_from_output DECODE_LAST done, "
                f"decode_inflight: {self.decode_inflight_count}/{self.decode_inflight_limit}",
                flush=True,
            )
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        self.chunk_prefill_first = [
            req for req in self.chunk_prefill_first if not req.is_finished()
        ]
        self.prefill_last_pending = [
            req for req in self.prefill_last_pending if not req.is_finished()
        ]
        return outputs

    def get_request_counts(self) -> tuple[int, int]:
        num_running, num_waiting = super().get_request_counts()
        return (
            num_running
            + len(self.chunk_prefill_first)
            + len(self.prefill_last_pending),
            num_waiting,
        )

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        return (
            super().get_num_unfinished_requests()
            + len(self.chunk_prefill_first)
            + len(self.prefill_last_pending)
        )

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        result = super().finish_requests(request_ids, finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        to_remove = set()
        for req_id in request_ids:
            req = self.requests.get(req_id)
            if req and req.is_finished():
                to_remove.add(req)

        if to_remove:
            self.chunk_prefill_first = remove_all(
                self.chunk_prefill_first, to_remove
            )

        return result

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        if reset_running_requests:
            timestamp = time.monotonic()
            while self.chunk_prefill_first:
                request = self.chunk_prefill_first.pop()
                self.kv_cache_manager.free(request)
                self.encoder_cache_manager.free(request)
                request.status = RequestStatus.PREEMPTED
                request.num_computed_tokens = 0
                if request.spec_token_ids:
                    request.spec_token_ids = []
                request.num_preemptions += 1
                if self.log_stats:
                    request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True
                self.waiting.prepend_request(request)

        return super().reset_prefix_cache(reset_running_requests, reset_connector)

    def make_stats(self, *args, **kwargs):
        stats = super().make_stats(*args, **kwargs)
        if stats is not None:
            stats.num_running_reqs += len(self.chunk_prefill_first)
        return stats

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        saved_running = self.running
        self.running = list(self.running) + [
            r for r in self.chunk_prefill_first if r not in self.running
        ]
        try:
            result = super()._handle_invalid_blocks(invalid_block_ids)
        finally:
            self.running = saved_running
        return result


class AsyncPDSeparatedScheduler(AsyncScheduler, PDSeparatedScheduler):
    """Async scheduler with PD separation."""
    pass
