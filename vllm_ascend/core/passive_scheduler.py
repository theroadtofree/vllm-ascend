# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Passive scheduler for non-leader PP ranks.

A `PassiveScheduler` does not make scheduling decisions. It receives
SchedulerOutputs that have already been decided by the leader rank (rank 0)
over a ZMQ subscriber, classifies them by `batch_type`, and emits ready-to-
dispatch payloads — optionally splitting prefill / PD-mix batches into N
layer slices based on a YAML config.

The class is intentionally minimal: it shares no implementation with
`vllm.v1.core.sched.scheduler.Scheduler` and depends only on the public
`SchedulerOutput` / `BatchType` types.
"""
import enum
import math
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.core.sched.output import BatchType, SchedulerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.engine.core import PPSchedulerZmqSubscriber

logger = init_logger(__name__)


class DispatchPolicy(enum.Enum):
    """Order in which phase queues are drained inside :meth:`PassiveScheduler.schedule`.

    The three phase queues — PURE_PREFILL, PD_MIX, PURE_DECODE — are polled
    in the order encoded by the policy. One SchedulerOutput is picked per
    non-empty queue per call.
    """
    EXPECT_ALTERNATION = "expect_alternation"  # Phase7 EEP/EED state machine.
    PREFILL_FIRST = "prefill_first"   # P  → PD-mix → D
    DECODE_FIRST = "decode_first"     # D  → PD-mix → P
    PDMIX_FIRST = "pdmix_first"       # PD-mix → P → D


class CloudSchedulingState(enum.Enum):
    EXPECT_EXECUTE_PREFILL = "expect_execute_prefill"
    EXPECT_EXECUTE_DECODE = "expect_execute_decode"


@dataclass
class LayerSliceInfo:
    """Metadata for a single layer slice in layerwise-disaggregated execution.

    When layer slicing is enabled, the PassiveScheduler splits the local
    layer range of a single SchedulerOutput into N slices. Each slice carries
    this info so the worker / model_runner can run only the assigned layer
    range and decide whether to perform PP communication.
    """
    slice_index: int       # 0, 1, 2, ...
    total_slices: int      # N
    start_layer: int       # local start layer (0-based within local layers)
    end_layer: int         # local end layer
    is_first_slice: bool   # slice_index == 0
    is_last_slice: bool    # slice_index == total_slices - 1


@dataclass
class ScheduledBatch:
    """Output of `PassiveScheduler.schedule()`: one SchedulerOutput plus the
    layer slices to dispatch in this engine tick.

    - For PURE_DECODE / DECODE_FIRST batches, or when slicing is disabled:
      ``slices == [None]`` (single full-layer execution).
    - For sliced PURE_PREFILL / PREFILL_FIRST / PD_MIX batches, ``schedule()``
      returns a single ``LayerSliceInfo`` per call so decode batches can be
      interleaved between prefill middle-layer slices.

    An empty instance (``slices == []``) signals that no SchedulerOutput was
    available to dispatch this round; the caller should typically idle.
    """
    scheduler_output: SchedulerOutput
    slices: list["LayerSliceInfo | None"]

    @classmethod
    def empty(cls) -> "ScheduledBatch":
        return cls(scheduler_output=None, slices=[])  # type: ignore[arg-type]

    def is_empty(self) -> bool:
        return not self.slices


class SliceTask(NamedTuple):
    scheduler_output: SchedulerOutput
    slice_info: LayerSliceInfo | None


class PassiveScheduler:
    """Receive → classify → schedule, for non-leader PP ranks.

    Lifecycle (each tick of the engine-core main loop):

        passive_scheduler.poll_and_classify()
        batch = passive_scheduler.schedule()
        if not batch.is_empty():
            for slice_info in batch.slices:
                executor.rpc_broadcast_mq.enqueue(...)

    `schedule()` returns a `ScheduledBatch` with 1 SchedulerOutput plus
    the slice plan; a single PURE_PREFILL / PD_MIX batch may carry N
    layer slices, while PURE_DECODE / DECODE_FIRST batches always carry
    `[None]` (single full-layer execution).
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        pp_subscriber: "PPSchedulerZmqSubscriber",
        dispatch_policy: DispatchPolicy = DispatchPolicy.EXPECT_ALTERNATION,
        run_subscriber_thread: bool = True,
    ) -> None:
        self.pp_subscriber = pp_subscriber
        self.dispatch_policy = dispatch_policy
        self.cloud_scheduling_state = CloudSchedulingState.EXPECT_EXECUTE_PREFILL

        self.ready_prefills: deque[SchedulerOutput] = deque()
        self.ready_pdmixes: deque[SchedulerOutput] = deque()
        self.ready_decodes: deque[SchedulerOutput] = deque()

        # Active sliced prefill / PD-mix continuation.  Only one sliced
        # prefill-like batch is allowed to be active at a time because the
        # Ascend model runner keeps layerwise continuation state in single
        # ``_layerwise_*`` fields.  Decode batches may be interleaved between
        # these continuation slices; another prefill-like slice-0 may not.
        self._active_sliced_prefill: SchedulerOutput | None = None
        self._active_prefill_slices: deque[SliceTask] = deque()

        # Cloud-side P/D interleave guard. After dispatching one prefill-middle
        # slice, EXPECT_EXECUTE_DECODE waits up to 10ms for a decode-middle
        # batch before falling back to another prefill-middle slice.
        self._prefill_middle_throttle_started_at: float | None = None
        self._prefill_middle_throttle_seconds = 0.010

        # Bridge queue between the (optional) subscriber thread and the
        # main loop. When the thread is enabled, it drains
        # `pp_subscriber.consume_new_outputs()` and pushes each SchedulerOutput
        # into `_inbox`; `poll_and_classify` drains `_inbox` instead of
        # touching the subscriber directly.
        self._inbox: queue.Queue[SchedulerOutput] = queue.Queue()
        self._subscriber_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # Precompute local layer count.  The actual slice count is resolved
        # per-batch from a YAML config (token threshold -> slice count).
        self._num_local_layers = 0
        self._layer_slice_config: dict[int, int] | None = None
        self._layer_slice_config_mtime: float = 0.0
        self._layer_slice_config_path: str | None = None
        num_hidden_layers = (
            vllm_config.model_config.hf_config.num_hidden_layers
        )
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if vllm_config.parallel_config.enable_edge_cloud:
            head_k = tail_k = 1
            additional_config = getattr(
                vllm_config, "additional_config", None
            )
            if isinstance(additional_config, dict):
                ec_cfg = additional_config.get("edge_cloud_config", {})
                htl = ec_cfg.get("edge_head_tail_layers", 1)
                if isinstance(htl, int):
                    head_k = tail_k = htl
                elif isinstance(htl, (list, tuple)) and len(htl) >= 2:
                    head_k = int(htl[0])
                    tail_k = int(htl[1])
            self._num_local_layers = max(
                0, num_hidden_layers - head_k - tail_k
            )
        else:
            from vllm.distributed.utils import get_pp_indices
            start_layer_pp, end_layer = get_pp_indices(
                num_hidden_layers, pp_size - 1, pp_size
            )
            self._num_local_layers = end_layer - start_layer_pp

        if self._num_local_layers > 0:
            cfg = self._load_layer_slice_config()
            if cfg is not None:
                self._layer_slice_config = cfg
                logger.info(
                    f"[PassiveScheduler] Layer-slice config loaded: "
                    f"{self._layer_slice_config}",
                )
            else:
                logger.info(
                    "[PassiveScheduler] Layer-slice YAML not found; "
                    "layer slicing is disabled.",
                )

        if run_subscriber_thread:
            self.start_subscriber_thread()

    # ------------------------------------------------------------------ #
    # Subscriber thread lifecycle                                        #
    # ------------------------------------------------------------------ #
    def start_subscriber_thread(self) -> None:
        """Spawn a daemon thread that pulls from the ZMQ subscriber and
        pushes SchedulerOutputs into `_inbox`. Idempotent.
        """
        if self._subscriber_thread is not None:
            return
        self._shutdown_event.clear()
        self._subscriber_thread = threading.Thread(
            target=self._subscriber_loop,
            name="PassiveScheduler-Subscriber",
            daemon=True,
        )
        self._subscriber_thread.start()
        logger.debug("PassiveScheduler subscriber thread started.")

    def _subscriber_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                new_outputs = self.pp_subscriber.consume_new_outputs()
            except Exception:
                if self._shutdown_event.is_set():
                    return
                logger.exception(
                    "PassiveScheduler subscriber thread failed to consume."
                )
                return
            if not new_outputs:
                # Avoid a tight spin when the subscriber returns nothing.
                self._shutdown_event.wait(0.001)
                continue
            for _seq, scheduler_output in new_outputs:
                self._inbox.put(scheduler_output)

    def shutdown(self) -> None:
        """Signal the subscriber thread to stop and join it."""
        self._shutdown_event.set()
        thread = self._subscriber_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._subscriber_thread = None

    # ------------------------------------------------------------------ #
    # Inbox draining + classification                                    #
    # ------------------------------------------------------------------ #
    def poll_and_classify(self) -> None:
        """Drain SchedulerOutputs from the inbox (fed by the subscriber
        thread, or directly by `_drain_subscriber_inline` when the thread
        is disabled) and route each into its phase-specific ready queue.
        """
        if self._subscriber_thread is None:
            # Inline mode: pull from the subscriber directly into _inbox.
            self._drain_subscriber_inline()

        while True:
            has_ready_work = bool(
                self.ready_prefills
                or self._active_prefill_slices
                or self.ready_pdmixes
                or self.ready_decodes
            )
            try:
                scheduler_output = self._inbox.get_nowait()
            except queue.Empty:
                if has_ready_work:
                    break
                logger.info("poll_and_classify: inbox is empty")
                scheduler_output = self._inbox.get(block=True)
            bt = scheduler_output.batch_type
            logger.info(f"Received scheduler_output from edge, batch_type: {bt}")
            if bt == BatchType.EMPTY:
                continue
            elif bt in (BatchType.PURE_PREFILL, BatchType.PREFILL_FIRST):
                # PREFILL_FIRST = edge-cloud "P first" head segment; from the
                # cloud's perspective it is exactly the same workload as a
                # legacy PURE_PREFILL batch (run middle layers, send hidden
                # state back), so route into the same ready queue.
                self.ready_prefills.append(scheduler_output)
            elif bt in (BatchType.PURE_DECODE, BatchType.DECODE_FIRST):
                # Same reasoning as above for decode head segments.
                self.ready_decodes.append(scheduler_output)
            elif bt in (BatchType.PREFILL_LAST, BatchType.DECODE_LAST):
                # Tail-segment batches are edge-only and must never be
                # dispatched on the cloud. If one shows up here it is a
                # routing bug at the publisher side — drop with a loud log.
                logger.error(
                    "PassiveScheduler received tail-segment batch_type=%s; "
                    "tail segments are edge-only and will be dropped.",
                    bt.value,
                )
                continue
            else:  # PD_MIX (or anything unrecognized — treat as mix)
                self.ready_pdmixes.append(scheduler_output)
            logger.debug(
                "PassiveScheduler classified batch_type=%s "
                "(prefills=%d, pdmixes=%d, decodes=%d)",
                bt.value if bt is not None else "<none>",
                len(self.ready_prefills),
                len(self.ready_pdmixes),
                len(self.ready_decodes),
            )

    def _drain_subscriber_inline(self) -> None:
        """Used only when the subscriber thread is disabled (e.g. tests)."""
        new_outputs = self.pp_subscriber.consume_new_outputs()
        for _seq, scheduler_output in new_outputs:
            self._inbox.put(scheduler_output)

    # ------------------------------------------------------------------ #
    # Layer-slice config loading                                         #
    # ------------------------------------------------------------------ #
    def _load_layer_slice_config(self) -> dict[int, int] | None:
        """Load token-threshold -> slice-count mapping from YAML.

        The YAML is expected to contain entries like::

            16: 24
            8: 10
            4: 4
            1: 5
            0: 5

        where the key is the token count in *thousands* and the value is
        the total number of slices.

        The file path resolution order is:
        1. ``VLLM_LAYER_SLICE_CONFIG`` env var if set.
        2. ``layer_slice_config.yaml`` in the same directory as this module.

        On success the path and mtime are cached on ``self`` for hot-reload
        tracking.  Returns ``None`` when the file does not exist or cannot
        be parsed.
        """
        yaml_path = os.environ.get("VLLM_LAYER_SLICE_CONFIG")
        if yaml_path is None:
            yaml_path = os.path.join(
                os.path.dirname(__file__), "layer_slice_config.yaml"
            )
        if not os.path.exists(yaml_path):
            self._layer_slice_config_path = None
            self._layer_slice_config_mtime = 0.0
            return None
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            if not isinstance(raw, dict):
                logger.warning(
                    "Layer-slice config %s is not a dict; ignoring.", yaml_path
                )
                return None
            # Normalize to int keys / values and sort descending by token threshold.
            config = {
                int(k): int(v) for k, v in raw.items()
            }
            self._layer_slice_config_path = yaml_path
            self._layer_slice_config_mtime = os.path.getmtime(yaml_path)
            return dict(sorted(config.items(), key=lambda item: item[0], reverse=True))
        except Exception:
            logger.exception("Failed to load layer-slice config from %s", yaml_path)
            return None

    def _maybe_hot_reload_layer_slice_config(self) -> None:
        """Check whether the YAML config file has changed on disk and reload."""
        path = self._layer_slice_config_path
        if path is None:
            return
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        if mtime != self._layer_slice_config_mtime:
            new_cfg = self._load_layer_slice_config()
            if new_cfg is not None:
                self._layer_slice_config = new_cfg
                logger.info(
                    f"[PassiveScheduler] Layer-slice config hot-reloaded: "
                    f"{self._layer_slice_config}",
                )

    def _resolve_slice_count(self, total_tokens: int) -> int:
        """Map a prefill batch size (in tokens) to the desired slice count.

        Uses the loaded YAML config (token-threshold in **thousands** ->
        slice-count).  The thresholds are checked from largest to smallest;
        the first threshold that ``total_tokens`` meets or exceeds wins.

        If no YAML config is present, layer slicing is disabled.
        """
        self._maybe_hot_reload_layer_slice_config()
        if self._layer_slice_config is not None:
            for token_k, slice_num in self._layer_slice_config.items():
                if total_tokens >= token_k * 1000:
                    return slice_num
        return 0

    # ------------------------------------------------------------------ #
    # Slicing                                                            #
    # ------------------------------------------------------------------ #
    def _make_slice_info(
        self,
        slice_idx: int,
        total_slices: int,
        boundaries: list[tuple[int, int]],
    ) -> LayerSliceInfo:
        slice_start, slice_end = boundaries[slice_idx]
        return LayerSliceInfo(
            slice_index=slice_idx,
            total_slices=total_slices,
            start_layer=slice_start,
            end_layer=slice_end,
            is_first_slice=(slice_idx == 0),
            is_last_slice=(slice_idx == total_slices - 1),
        )

    def _slice_for(
        self, so: SchedulerOutput
    ) -> list["LayerSliceInfo | None"]:
        # Decode-like and empty batches are never sliced. DECODE_FIRST is the
        # edge-cloud head segment of a decode step — same per-token shape as
        # PURE_DECODE, so it follows the same no-slice rule.
        if so.batch_type in (
            BatchType.PURE_DECODE,
            BatchType.DECODE_FIRST,
        ):
            return [None]

        total_slices = self._resolve_slice_count(
            so.total_num_scheduled_tokens
        )
        # Slicing disabled or trivially 1 slice.
        if total_slices <= 1:
            return [None]

        boundaries = self._compute_slice_boundaries(
            self._num_local_layers, total_slices
        )
        # PURE_PREFILL / PREFILL_FIRST / PD_MIX → expand into N slice payloads.
        return [
            self._make_slice_info(i, total_slices, boundaries)
            for i in range(total_slices)
        ]

    # ------------------------------------------------------------------ #
    # Dispatch                                                           #
    # ------------------------------------------------------------------ #
    _POLICY_ORDER: dict[DispatchPolicy, tuple[str, str, str]] = {
        DispatchPolicy.EXPECT_ALTERNATION: (
            "ready_prefills", "ready_decodes", "ready_pdmixes",
        ),
        DispatchPolicy.PREFILL_FIRST: (
            "ready_prefills", "ready_pdmixes", "ready_decodes",
        ),
        DispatchPolicy.DECODE_FIRST: (
            "ready_decodes", "ready_pdmixes", "ready_prefills",
        ),
        DispatchPolicy.PDMIX_FIRST: (
            "ready_pdmixes", "ready_prefills", "ready_decodes",
        ),
    }

    def _start_prefill_middle_throttle(self) -> None:
        self._prefill_middle_throttle_started_at = time.monotonic()
        logger.info(
            f"[PD-PASSIVE] Prefill throttle started: waiting up to "
            f"{self._prefill_middle_throttle_seconds * 1000:.0f}ms for decode",
        )

    def _clear_prefill_middle_throttle(self) -> None:
        started_at = self._prefill_middle_throttle_started_at
        if started_at is not None:
            elapsed_ms = (time.monotonic() - started_at) * 1000
            logger.info(
                f"[PD-PASSIVE] Prefill throttle cleared after "
                f"{elapsed_ms:.1f}ms",
            )
        self._prefill_middle_throttle_started_at = None

    def _can_fallback_to_prefill_in_decode_state(self) -> bool:
        started_at = self._prefill_middle_throttle_started_at
        if started_at is None:
            return True
        elapsed_ms = (time.monotonic() - started_at) * 1000
        limit_ms = self._prefill_middle_throttle_seconds * 1000
        if elapsed_ms >= limit_ms:
            logger.info(
                f"[PD-PASSIVE] Throttle timeout: waited {elapsed_ms:.1f}ms, "
                f"fallback to prefill",
            )
            self._clear_prefill_middle_throttle()
            return True
        logger.info(
            f"[PD-PASSIVE] Throttle active: {elapsed_ms:.1f}ms / {limit_ms:.0f}ms, "
            f"still waiting for decode",
        )
        return False

    def schedule(self) -> ScheduledBatch:
        """Pick the next SchedulerOutput to dispatch.

        ``EXPECT_ALTERNATION`` implements the Phase7 cloud-side EEP/EED state
        machine.  Sliced prefill-like batches are dispatched one slice per call
        so decode batches can be interleaved between the remaining slices.
        """
        if self.dispatch_policy == DispatchPolicy.EXPECT_ALTERNATION:
            return self._schedule_expect_alternation()

        for queue_name in self._POLICY_ORDER[self.dispatch_policy]:
            batch = self._schedule_from_queue(queue_name)
            if not batch.is_empty():
                return batch

        return ScheduledBatch.empty()

    @staticmethod
    def _compute_slice_boundaries(
        num_local_layers: int, layer_slice_num: int
    ) -> list[tuple[int, int]]:
        """Compute layer slice boundaries for a fixed slice count.

        Distributes ``num_local_layers`` into ``layer_slice_num`` slices
        as evenly as possible.  Larger slices come first; the size
        difference between any two slices is at most 1.

        Returns a list of ``(start_layer, end_layer)`` tuples where
        ``end_layer`` is exclusive.
        """
        if num_local_layers <= 0 or layer_slice_num <= 0:
            return []
        boundaries: list[tuple[int, int]] = []
        base = num_local_layers // layer_slice_num
        rem = num_local_layers % layer_slice_num
        start = 0
        for i in range(layer_slice_num):
            size = base + 1 if i < rem else base
            boundaries.append((start, start + size))
            start += size
        return boundaries

    def _schedule_expect_alternation(self) -> ScheduledBatch:
        state = self.cloud_scheduling_state
        if state == CloudSchedulingState.EXPECT_EXECUTE_PREFILL:
            if self._active_prefill_slices:
                self.cloud_scheduling_state = (
                    CloudSchedulingState.EXPECT_EXECUTE_DECODE
                )
                self._start_prefill_middle_throttle()
                return self._build_active_prefill_slice_batch()
            if self.ready_prefills:
                self.cloud_scheduling_state = (
                    CloudSchedulingState.EXPECT_EXECUTE_DECODE
                )
                self._start_prefill_middle_throttle()
                return self._build_batch(self.ready_prefills.popleft())
            if self.ready_decodes:
                self._clear_prefill_middle_throttle()
                return self._build_batch(self.ready_decodes.popleft())
        else:
            if self.ready_decodes:
                self.cloud_scheduling_state = (
                    CloudSchedulingState.EXPECT_EXECUTE_PREFILL
                )
                self._clear_prefill_middle_throttle()
                return self._build_batch(self.ready_decodes.popleft())
            if self._can_fallback_to_prefill_in_decode_state():
                if self._active_prefill_slices:
                    self._start_prefill_middle_throttle()
                    return self._build_active_prefill_slice_batch()
                if self.ready_prefills:
                    self._start_prefill_middle_throttle()
                    return self._build_batch(self.ready_prefills.popleft())
            else:
                return ScheduledBatch.empty()

        if self.ready_pdmixes:
            if (
                state == CloudSchedulingState.EXPECT_EXECUTE_DECODE
                and not self._can_fallback_to_prefill_in_decode_state()
            ):
                return ScheduledBatch.empty()
            if state == CloudSchedulingState.EXPECT_EXECUTE_DECODE:
                self._start_prefill_middle_throttle()
            return self._build_batch(self.ready_pdmixes.popleft())
        return ScheduledBatch.empty()

    def _schedule_from_queue(self, queue_name: str) -> ScheduledBatch:
        if self._active_prefill_slices:
            if queue_name == "ready_decodes" and self.ready_decodes:
                return self._build_batch(self.ready_decodes.popleft())
            if queue_name in ("ready_prefills", "ready_pdmixes"):
                return self._build_active_prefill_slice_batch()
            return ScheduledBatch.empty()

        q: deque[SchedulerOutput] = getattr(self, queue_name)
        if q:
            return self._build_batch(q.popleft())
        return ScheduledBatch.empty()

    def _build_batch(self, so: SchedulerOutput) -> ScheduledBatch:
        slices = self._slice_for(so)
        if len(slices) <= 1:
            batch = ScheduledBatch(scheduler_output=so, slices=slices)
        else:
            first_slice = slices[0]
            assert isinstance(first_slice, LayerSliceInfo)
            self._active_sliced_prefill = so
            self._active_prefill_slices.extend(
                SliceTask(so, slice_info)
                for slice_info in slices[1:]
                if isinstance(slice_info, LayerSliceInfo)
            )
            batch = ScheduledBatch(scheduler_output=so, slices=[first_slice])

        self._log_picked_batch(batch)
        return batch

    def _build_active_prefill_slice_batch(self) -> ScheduledBatch:
        task = self._active_prefill_slices.popleft()
        if not self._active_prefill_slices:
            self._active_sliced_prefill = None
        batch = ScheduledBatch(
            scheduler_output=task.scheduler_output,
            slices=[task.slice_info],
        )
        self._log_picked_batch(batch)
        return batch

    def _log_picked_batch(self, batch: ScheduledBatch) -> None:
        so = batch.scheduler_output
        logger.debug(
            "PassiveScheduler.schedule[%s] picked batch_type=%s slices=%d; "
            "pending=(prefills=%d, active_prefill_slices=%d, "
            "pdmixes=%d, decodes=%d)",
            self.dispatch_policy.value,
            so.batch_type.value if so.batch_type is not None else "<none>",
            len(batch.slices),
            len(self.ready_prefills),
            len(self._active_prefill_slices),
            len(self.ready_pdmixes),
            len(self.ready_decodes),
        )

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #
    def has_pending(self) -> bool:
        return bool(
            self.ready_prefills
            or self._active_prefill_slices
            or self.ready_pdmixes
            or self.ready_decodes
        )

    @property
    def num_pending(self) -> int:
        return (
            len(self.ready_prefills)
            + len(self._active_prefill_slices)
            + len(self.ready_pdmixes)
            + len(self.ready_decodes)
        )
