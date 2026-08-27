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
from typing import TYPE_CHECKING, Callable, NamedTuple

from vllm import envs
from vllm.logger import logger
from vllm.v1.core.sched.output import BatchType, SchedulerOutput

from vllm_ascend.distributed.edge_cloud_comm.scheduler_link import (
    is_irecv_complete,
)
from vllm_ascend.distributed.edge_cloud_comm.types import CommChannelType

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.engine.core import PPSchedulerZmqSubscriber


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
    EXPECT_EXECUTE_DECODE_OR_DRAFT = "expect_execute_decode_or_draft"


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

    _ARRIVAL_SEQ_ATTR = "_passive_scheduler_arrival_seq"

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
        # Draft batches are split into two lanes by draft_prefill_phase:
        # prefill-phase chains arrive on the dedicated PREFILL_DRAFT channel
        # pair, decode-phase chains share the DECODE pair.  Splitting the
        # queues lets each lane's data-readiness gate peek at its own head
        # without cross-channel head-of-line blocking.
        self.ready_prefill_drafts: deque[SchedulerOutput] = deque()
        self.ready_decode_drafts: deque[SchedulerOutput] = deque()
        self.ready_decodes: deque[SchedulerOutput] = deque()

        # Active sliced prefill / PD-mix continuation.  Only one sliced
        # prefill-like batch is allowed to be active at a time because the
        # Ascend model runner keeps layerwise continuation state in single
        # ``_layerwise_*`` fields.  Decode batches may be interleaved between
        # these continuation slices; another prefill-like slice-0 may not.
        self._active_sliced_prefill: SchedulerOutput | None = None
        self._active_prefill_slices: deque[SliceTask] = deque()

        # Cloud-side P/D interleave guard. After dispatching one prefill-middle
        # slice, EXPECT_EXECUTE_DECODE_OR_DRAFT waits up to 10ms for a decode-middle
        # batch before falling back to another prefill-middle slice.
        self._prefill_middle_throttle_started_at: float | None = None
        self._prefill_middle_throttle_seconds = 0.010
        # Rate limiter for the [PD-STALL-CLOUD] empty-dispatch probe.
        self._last_stall_log_ts: float = 0.0

        # Bridge queue between the (optional) subscriber thread and the
        # main loop. When the thread is enabled, it drains
        # `pp_subscriber.consume_new_outputs()` and pushes each SchedulerOutput
        # into `_inbox`; `poll_and_classify` drains `_inbox` instead of
        # touching the subscriber directly.
        self._inbox: queue.Queue[tuple[int, SchedulerOutput]] = queue.Queue()
        self._subscriber_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # [DIAG] Track DECODE_FIRST arrival intervals on the cloud side.
        self._last_decode_first_arrival_ts: float | None = None

        # Precompute local layer count.  The actual slice count is resolved
        # per-batch from a YAML config (token threshold -> slice count).
        self._num_local_layers = 0
        self._layer_slice_config: dict[int, int] | None = None
        self._layer_slice_config_mtime: float = 0.0
        self._layer_slice_config_path: str | None = None
        # Use hf_text_config (not the root hf_config) so multimodal models
        # whose layer count lives in a nested text sub-config (e.g. KimiK2.5
        # -> DeepseekV3Config) resolve correctly. For plain-text models
        # hf_text_config == hf_config, so this is a no-op there.
        num_hidden_layers = (
            vllm_config.model_config.hf_text_config.num_hidden_layers
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
            for seq, scheduler_output in new_outputs:
                self._inbox.put((seq, scheduler_output))

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
    def poll_and_classify(self) -> list[tuple[int, "SchedulerOutput"]]:
        """Drain SchedulerOutputs from the inbox (fed by the subscriber
        thread, or directly by `_drain_subscriber_inline` when the thread
        is disabled) and route each into its phase-specific ready queue.

        Returns the newly classified ``(seq, SchedulerOutput)`` pairs
        (EMPTY batches excluded) so the engine core can run arrival-time
        data-plane actions — pre-posting irecvs for the head payload and
        its following draft chain — before any dispatch decision.
        """
        arrivals: list[tuple[int, SchedulerOutput]] = []
        if self._subscriber_thread is None:
            # Inline mode: pull from the subscriber directly into _inbox.
            self._drain_subscriber_inline()

        while True:
            try:
                seq, scheduler_output = self._inbox.get_nowait()
            except queue.Empty:
                break
            self._remember_arrival_seq(scheduler_output, seq)
            bt = scheduler_output.batch_type
            # logger.info(
            #     "Received scheduler_output from edge, seq=%d, batch_type: %s",
            #     seq,
            #     bt,
            # )
            if bt == BatchType.EMPTY:
                continue
            arrivals.append((seq, scheduler_output))
            if bt in (BatchType.PURE_PREFILL, BatchType.PREFILL_FIRST):
                # PREFILL_FIRST = edge-cloud "P first" head segment; from the
                # cloud's perspective it is exactly the same workload as a
                # legacy PURE_PREFILL batch (run middle layers, send hidden
                # state back), so route into the same ready queue.
                self.ready_prefills.append(scheduler_output)
            elif bt in (BatchType.PURE_DECODE, BatchType.DECODE_FIRST):
                # Same reasoning as above for decode head segments.
                now = time.monotonic()
                if self._last_decode_first_arrival_ts is not None:
                    interval_ms = (now - self._last_decode_first_arrival_ts) * 1000
                    logger.info(
                        "DECODE_FIRST arrival interval: %.2f ms",
                        interval_ms,
                    )
                self._last_decode_first_arrival_ts = now
                self.ready_decodes.append(scheduler_output)
            elif bt in (
                BatchType.PREFILL_DRAFT_FIRST,
                BatchType.DECODE_DRAFT_FIRST,
            ):
                # The draft chain phase is encoded in the batch type itself:
                # prefill-phase chains route to ready_prefill_drafts
                # (PREFILL_DRAFT pair), decode-phase chains to
                # ready_decode_drafts (DECODE pair).
                if bt == BatchType.PREFILL_DRAFT_FIRST:
                    self.ready_prefill_drafts.append(scheduler_output)
                else:
                    self.ready_decode_drafts.append(scheduler_output)
            elif bt in (
                BatchType.PREFILL_LAST,
                BatchType.DECODE_LAST,
                BatchType.PREFILL_DRAFT_LAST,
                BatchType.DECODE_DRAFT_LAST,
            ):
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
                "PassiveScheduler classified seq=%s batch_type=%s "
                "(prefills=%d, pdmixes=%d, prefill_drafts=%d, "
                "decode_drafts=%d, decodes=%d)",
                self._arrival_seq(scheduler_output),
                bt.value if bt is not None else "<none>",
                len(self.ready_prefills),
                len(self.ready_pdmixes),
                len(self.ready_prefill_drafts),
                len(self.ready_decode_drafts),
                len(self.ready_decodes),
            )
        return arrivals

    def _remember_arrival_seq(
        self, scheduler_output: SchedulerOutput, seq: int
    ) -> None:
        try:
            setattr(scheduler_output, self._ARRIVAL_SEQ_ATTR, seq)
        except Exception:
            logger.debug(
                "Unable to attach arrival seq=%d to SchedulerOutput.",
                seq,
                exc_info=True,
            )

    def _arrival_seq(self, scheduler_output: SchedulerOutput) -> int | None:
        seq = getattr(scheduler_output, self._ARRIVAL_SEQ_ATTR, None)
        return seq if isinstance(seq, int) else None

    def _drain_subscriber_inline(self) -> None:
        """Used only when the subscriber thread is disabled (e.g. tests)."""
        new_outputs = self.pp_subscriber.consume_new_outputs()
        for seq, scheduler_output in new_outputs:
            self._inbox.put((seq, scheduler_output))

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
            # Extract optional prefill_middle_throttle_ms (milliseconds) before filtering.
            _throttle_key = "prefill_middle_throttle_ms"
            if _throttle_key in raw:
                try:
                    self._prefill_middle_throttle_seconds = float(raw[_throttle_key]) / 1000.0
                    logger.info(
                        "[PassiveScheduler] %s set to %.1f ms (%.3f s) from %s",
                        _throttle_key, float(raw[_throttle_key]),
                        self._prefill_middle_throttle_seconds, yaml_path,
                    )
                except (ValueError, TypeError):
                    logger.warning(
                        "Invalid %s value %r in %s; keeping %.3f s",
                        _throttle_key, raw[_throttle_key], yaml_path,
                        self._prefill_middle_throttle_seconds,
                    )

            # Normalize to int keys / values and sort descending by token threshold.
            config = {
                int(k): int(v) for k, v in raw.items() if isinstance(k, (int, str)) and str(k).lstrip('-').isdigit()
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

    def _do_slice(
        self, so: SchedulerOutput
    ) -> list["LayerSliceInfo | None"]:
        """Compute layer slices for a prefill-like batch."""
        total_slices = self._resolve_slice_count(
            so.total_num_scheduled_tokens
        )
        if total_slices <= 1:
            return [None]
        boundaries = self._compute_slice_boundaries(
            self._num_local_layers, total_slices
        )
        return [
            self._make_slice_info(i, total_slices, boundaries)
            for i in range(total_slices)
        ]

    def _slice_for(
        self, so: SchedulerOutput
    ) -> list["LayerSliceInfo | None"]:
        # Decode-like and empty batches are never sliced. DECODE_FIRST is the
        # edge-cloud head segment of a decode step — same per-token shape as
        # PURE_DECODE, so it follows the same no-slice rule.
        if so.batch_type in (
            BatchType.PURE_DECODE,
            BatchType.DECODE_FIRST,
            BatchType.PREFILL_DRAFT_FIRST,
            BatchType.DECODE_DRAFT_FIRST,
        ):
            return [None]

        # [方案B] Cloud 侧决策：
        # 1. 已有 decode 到达 Cloud → 强制切层（确定性收益）
        if self.ready_decodes:
            return self._do_slice(so)

        # 2. Edge 建议切层（decode 正在路上）→ 切层
        if getattr(so, "cloud_suggest_slicing", False):
            return self._do_slice(so)

        # 3. Edge 建议不切层 + Cloud 无 decode → 明确不切层（冷启动优化）
        # 短 prefill（<8k）执行太快，decode 来不及穿插，同样不切层
        return [None]

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
        # logger.info(
        #     f"[PD-PASSIVE] Prefill throttle started: waiting up to "
        #     f"{self._prefill_middle_throttle_seconds * 1000:.0f}ms for decode",
        # )

    def _clear_prefill_middle_throttle(self) -> None:
        started_at = self._prefill_middle_throttle_started_at
        if started_at is not None:
            elapsed_ms = (time.monotonic() - started_at) * 1000
            # logger.info(
            #     f"[PD-PASSIVE] Prefill throttle cleared after "
            #     f"{elapsed_ms:.1f}ms",
            # )
        self._prefill_middle_throttle_started_at = None

    def _can_fallback_to_prefill_in_decode_state(self) -> bool:
        # Optimization: when the next prefill to schedule has
        # cloud_suggest_slicing=False, the edge signaled "no running decode
        # in flight" -> decode is not about to arrive, so waiting (throttling)
        # for it is pure idle.  Skip the throttle and fall back to prefill
        # immediately.  This also means the P-middle is unsliced (see
        # _slice_for), so there is no slice interleaving to protect either.
        if self.ready_prefills and not getattr(
            self.ready_prefills[0], "cloud_suggest_slicing", False
        ):
            self._clear_prefill_middle_throttle()
            return True
        started_at = self._prefill_middle_throttle_started_at
        if started_at is None:
            return True
        elapsed_ms = (time.monotonic() - started_at) * 1000
        limit_ms = self._prefill_middle_throttle_seconds * 1000
        if elapsed_ms >= limit_ms:
            # logger.info(
            #     f"[PD-PASSIVE] Throttle timeout: waited {elapsed_ms:.1f}ms, "
            #     f"fallback to prefill",
            # )
            self._clear_prefill_middle_throttle()
            return True
        logger.debug(
            f"[PD-PASSIVE] Throttle active: {elapsed_ms:.1f}ms / {limit_ms:.0f}ms, "
            f"still waiting for decode",
        )
        return False

    def schedule(self) -> ScheduledBatch:
        """Pick the next SchedulerOutput to dispatch.

        ``EXPECT_ALTERNATION`` implements the Phase7 cloud-side EEP/EED state
        machine.  Sliced prefill-like batches are dispatched one slice per call
        so decode/draft batches can be interleaved between the remaining
        slices.  Draft priority is enforced inside the state machine, not via
        an early out-of-band check.
        """
        if self.dispatch_policy == DispatchPolicy.EXPECT_ALTERNATION:
            batch = self._schedule_expect_alternation(ready_only=True)
            if batch.is_empty() and (
                self.ready_prefills or self.ready_decodes
                or self.ready_prefill_drafts or self.ready_decode_drafts
            ):
                # Nothing is data-ready but work is queued: fall back to
                # the original priority order and dispatch anyway — the
                # payload wait happens device-side (wait_event on the
                # pre-posted recv), never a host block.
                batch = self._schedule_expect_alternation(ready_only=False)
            self._log_stall_if_blocked(batch)
            return batch

        for queue_name in self._POLICY_ORDER[self.dispatch_policy]:
            batch = self._schedule_from_queue(queue_name, ready_only=True)
            if not batch.is_empty():
                return batch
        if (
            self.ready_prefills or self.ready_decodes
            or self.ready_prefill_drafts or self.ready_decode_drafts
        ):
            for queue_name in self._POLICY_ORDER[self.dispatch_policy]:
                batch = self._schedule_from_queue(
                    queue_name, ready_only=False
                )
                if not batch.is_empty():
                    return batch
        batch = ScheduledBatch.empty()
        self._log_stall_if_blocked(batch)
        return batch

    def _log_stall_if_blocked(self, batch: ScheduledBatch) -> None:
        """Rate-limited probe for empty dispatches with queued work.

        Fires at most once per 10ms.  Pure supply gaps (all queues empty)
        are intentionally NOT logged here — those belong to the edge's own
        [PD-STALL] probe.  This probe distinguishes the cloud-local causes:
        watermark-not-ready heads, EED throttle waits, and the single
        sliced-prefill constraint.
        """
        if not batch.is_empty():
            return
        if not (
            self.ready_prefills
            or self.ready_pdmixes
            or self.ready_prefill_drafts
            or self.ready_decode_drafts
            or self.ready_decodes
            or self._active_prefill_slices
        ):
            return
        now = time.monotonic()
        if now - self._last_stall_log_ts < 0.010:
            return
        self._last_stall_log_ts = now
        throttle_remaining_ms: float | None = None
        if self._prefill_middle_throttle_started_at is not None:
            elapsed = now - self._prefill_middle_throttle_started_at
            throttle_remaining_ms = max(
                0.0,
                (self._prefill_middle_throttle_seconds - elapsed) * 1000,
            )
        logger.warning(
            "[PD-STALL-CLOUD] empty dispatch with queued work: "
            "policy=%s state=%s | "
            "queues: prefills=%d pdmixes=%d p_drafts=%d d_drafts=%d "
            "decodes=%d active_slices=%d sliced_pf=%s | "
            "head_ready: pf=%s dd=%s p_drf=%s d_drf=%s | "
            "throttle_remaining_ms=%s",
            self.dispatch_policy.value,
            self.cloud_scheduling_state,
            len(self.ready_prefills),
            len(self.ready_pdmixes),
            len(self.ready_prefill_drafts),
            len(self.ready_decode_drafts),
            len(self.ready_decodes),
            len(self._active_prefill_slices),
            self._active_sliced_prefill is not None,
            self._prefill_head_data_ready(),
            self._decode_head_data_ready(),
            self._prefill_draft_head_data_ready(),
            self._decode_draft_head_data_ready(),
            (
                f"{throttle_remaining_ms:.1f}"
                if throttle_remaining_ms is not None
                else "inactive"
            ),
        )

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

    # ------------------------------------------------------------------ #
    # Pick methods (analogous to edge-side PDSeparatedScheduler)         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _seqno_ready(channel: CommChannelType, so: SchedulerOutput) -> bool:
        """True once the pre-posted irecv for this batch's head payload
        has completed — i.e. the channel's completion watermark covers
        the batch's ``comm_seqno`` (stamped by the edge scheduler and
        carried on the SO).  Batches without a comm_seqno (legacy
        PURE_* path) carry no sequenced traffic and are always ready."""
        seqno = getattr(so, "comm_seqno", None)
        if seqno is None:
            return True
        return is_irecv_complete(channel, seqno)

    def _prefill_head_data_ready(self) -> bool:
        """True once the head-of-queue prefill's payload has arrived.

        Only the fresh-prefill queue is gated: active slice continuations
        reuse an already-received payload, and PD-mix batches do not
        participate in pre-posted irecv.
        """
        if not self.ready_prefills:
            return False
        return self._seqno_ready(
            CommChannelType.PREFILL_UP, self.ready_prefills[0]
        )

    def _decode_head_data_ready(self) -> bool:
        """True once the head-of-queue decode's payload has arrived."""
        if not self.ready_decodes:
            return False
        return self._seqno_ready(
            CommChannelType.DECODE_UP, self.ready_decodes[0]
        )

    def _prefill_draft_head_data_ready(self) -> bool:
        """True once the head prefill-phase draft's payload has arrived.

        Prefill-phase chains travel on the dedicated PREFILL_DRAFT pair.
        """
        if not self.ready_prefill_drafts:
            return False
        return self._seqno_ready(
            CommChannelType.PREFILL_DRAFT_UP, self.ready_prefill_drafts[0]
        )

    def _decode_draft_head_data_ready(self) -> bool:
        """True once the head decode-phase draft's payload has arrived.

        Decode-phase chains share the DECODE pair with plain decode.
        """
        if not self.ready_decode_drafts:
            return False
        return self._seqno_ready(
            CommChannelType.DECODE_UP, self.ready_decode_drafts[0]
        )

    def _pick_prefill_batch(
        self, ready_only: bool = True
    ) -> ScheduledBatch:
        """Pick a prefill or prefill-like batch from the ready queues.

        Checks in priority order: active prefill slices (continuation of
        a previously sliced prefill), fresh prefills from ``ready_prefills``,
        then PD-mix batches from ``ready_pdmixes``.

        Caller must ensure at least one source is non-empty before calling.
        """
        if self._active_prefill_slices:
            return self._build_active_prefill_slice_batch()
        if self.ready_prefills:
            # Data-plane gate: in the ready pass, keep the batch queued
            # (and yield an empty tick) until its pre-posted irecv has
            # completed, so the worker never blocks on recv inside
            # execute_model.  The fallback pass dispatches regardless
            # (device-side wait_event covers the in-flight payload).
            if ready_only and not self._prefill_head_data_ready():
                return ScheduledBatch.empty()
            return self._build_batch(self.ready_prefills.popleft())
        assert self.ready_pdmixes, (
            "_pick_prefill_batch called with no prefill work available"
        )
        return self._build_batch(self.ready_pdmixes.popleft())

    def _pick_decode_batch(self) -> ScheduledBatch:
        """Pick a decode batch from ``ready_decodes``.

        Caller must ensure ``ready_decodes`` is non-empty before calling.
        """
        return self._build_batch(self.ready_decodes.popleft())

    def _pick_prefill_draft_batch(self) -> ScheduledBatch:
        """Pick a draft batch from ``ready_prefill_drafts``.

        Caller must ensure ``ready_prefill_drafts`` is non-empty before
        calling.
        """
        return self._build_batch(self.ready_prefill_drafts.popleft())

    def _pick_decode_draft_batch(self) -> ScheduledBatch:
        """Pick a draft batch from ``ready_decode_drafts``.

        Caller must ensure ``ready_decode_drafts`` is non-empty before
        calling.
        """
        return self._build_batch(self.ready_decode_drafts.popleft())

    def _pick_decode_or_draft_by_arrival(
        self, ready_only: bool = True
    ) -> ScheduledBatch:
        """Pick among the head decode / decode-draft / prefill-draft by
        arrival order.

        Only data-ready heads participate (decode-phase drafts excepted:
        they dispatch on arrival, their payload wait covered device-side
        by wait_event on the pre-posted recv).  With pre-posted irecvs
        the recv order is fixed at SO arrival time, so dispatch order no
        longer affects channel pairing — the old cross-side deadlock
        (cloud posting a recv for one payload while the edge's next
        in-flight message is another) cannot occur once every candidate
        is gated on irecv completion.  Arrival order remains as the
        priority tie-breaker among ready heads.

        Caller must ensure at least one of the three queues is non-empty.
        """
        has_decode = bool(self.ready_decodes) and (
            not ready_only or self._decode_head_data_ready()
        )
        # Decode-phase draft heads dispatch on arrival: the payload wait
        # is covered device-side by wait_event on the pre-posted recv
        # (the edge sends promptly after its DRF head executes), so the
        # DECODE_UP watermark is not consulted here.
        has_decode_draft = bool(self.ready_decode_drafts)
        has_prefill_draft = bool(self.ready_prefill_drafts) and (
            not ready_only or self._prefill_draft_head_data_ready()
        )
        # (arrival_seq, pick) candidates; the lowest seq wins.  The lambda
        # order breaks the (impossible-in-practice) seq tie deterministically.
        # A missing seq (annotation failed) sorts last, mirroring the old
        # pair-wise comparison which never let a None seq win.
        candidates: list[tuple[float, Callable[[], ScheduledBatch]]] = []

        def _seq_of(so: SchedulerOutput) -> float:
            seq = self._arrival_seq(so)
            return float(seq) if seq is not None else math.inf

        if has_decode:
            candidates.append(
                (_seq_of(self.ready_decodes[0]), self._pick_decode_batch)
            )
        if has_decode_draft:
            candidates.append(
                (
                    _seq_of(self.ready_decode_drafts[0]),
                    self._pick_decode_draft_batch,
                )
            )
        if has_prefill_draft:
            candidates.append(
                (
                    _seq_of(self.ready_prefill_drafts[0]),
                    self._pick_prefill_draft_batch,
                )
            )
        if not candidates:
            # No head is data-ready yet — idle this tick.
            return ScheduledBatch.empty()
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]()

    def _schedule_by_arrival(self, ready_only: bool = True) -> ScheduledBatch:
        # Only data-ready heads participate (decode-phase drafts excepted:
        # they dispatch on arrival, see _pick_decode_or_draft_by_arrival).
        # With pre-posted irecvs the
        # recv order was fixed at SO arrival time, so dispatching a ready
        # decode/draft ahead of a not-yet-arrived prefill cannot re-order
        # the channel: the inversion hazard the arrival rule guarded
        # against required the worker to block on recv at execution time,
        # which the readiness gate eliminates.  Arrival order remains the
        # priority tie-breaker among ready heads.
        has_prefill = bool(self.ready_prefills) and (
            not ready_only or self._prefill_head_data_ready()
        )
        has_decode = bool(self.ready_decodes) and (
            not ready_only or self._decode_head_data_ready()
        )
        # Decode-phase draft heads dispatch on arrival: the payload wait
        # is covered device-side by wait_event on the pre-posted recv
        # (the edge sends promptly after its DRF head executes), so the
        # DECODE_UP watermark is not consulted here.
        has_decode_draft = bool(self.ready_decode_drafts)
        has_prefill_draft = bool(self.ready_prefill_drafts) and (
            not ready_only or self._prefill_draft_head_data_ready()
        )
        has_draft = has_decode_draft or has_prefill_draft
        prefill_seq = (
            self._arrival_seq(self.ready_prefills[0])
            if has_prefill
            else None
        )
        decode_seq = (
            self._arrival_seq(self.ready_decodes[0])
            if has_decode
            else None
        )
        draft_seqs = [
            seq
            for seq in (
                (
                    self._arrival_seq(self.ready_decode_drafts[0])
                    if has_decode_draft
                    else None
                ),
                (
                    self._arrival_seq(self.ready_prefill_drafts[0])
                    if has_prefill_draft
                    else None
                ),
            )
            if seq is not None
        ]
        draft_seq = min(draft_seqs) if draft_seqs else None
        channel_seq = decode_seq
        if draft_seq is not None and (
            channel_seq is None or draft_seq < channel_seq
        ):
            channel_seq = draft_seq
        if prefill_seq is None:
            # No ready prefill: run whatever channel work is ready.
            if channel_seq is None:
                return ScheduledBatch.empty()
            self._clear_prefill_middle_throttle()
            return self._pick_decode_or_draft_by_arrival(
                ready_only=ready_only
            )
        if channel_seq is not None and channel_seq < prefill_seq:
            logger.info(
                "[PD-PASSIVE] Decode/draft ready and arrived before "
                "prefill: channel_seq=%d, prefill_seq=%d",
                channel_seq,
                prefill_seq,
            )
            self._clear_prefill_middle_throttle()
            return self._pick_decode_or_draft_by_arrival(
                ready_only=ready_only
            )
        logger.info(
            "[PD-PASSIVE] Prefill ready and arrived before decode/draft: "
            "prefill_seq=%d, channel_seq=%s",
            prefill_seq,
            channel_seq,
        )
        self.cloud_scheduling_state = CloudSchedulingState.EXPECT_EXECUTE_DECODE_OR_DRAFT
        self._start_prefill_middle_throttle()
        return self._build_batch(self.ready_prefills.popleft())

    def _schedule_expect_alternation(
        self, ready_only: bool = True
    ) -> ScheduledBatch:
        state = self.cloud_scheduling_state
        # Data-plane readiness: only heads whose pre-posted irecv has
        # completed participate in this tick's scheduling (decode-phase
        # drafts excepted — they dispatch on arrival, see
        # _pick_decode_or_draft_by_arrival).  An unready
        # head is treated as absent so lower-priority ready work can run
        # — with pre-posted irecvs the recv order was fixed at arrival,
        # so this cannot re-order the channel (see _schedule_by_arrival).
        # The fallback pass (ready_only=False, taken only when nothing
        # was ready) treats queue non-empty as sufficient: the payload
        # wait is covered device-side by wait_event, never a host block.
        has_prefill = bool(self.ready_prefills) and (
            not ready_only or self._prefill_head_data_ready()
        )
        has_decode = bool(self.ready_decodes) and (
            not ready_only or self._decode_head_data_ready()
        )
        has_draft = bool(self.ready_decode_drafts) or (
            bool(self.ready_prefill_drafts)
            and (not ready_only or self._prefill_draft_head_data_ready())
        )
        if state == CloudSchedulingState.EXPECT_EXECUTE_PREFILL:
            if self._active_prefill_slices:
                self.cloud_scheduling_state = (
                    CloudSchedulingState.EXPECT_EXECUTE_DECODE_OR_DRAFT
                )
                self._start_prefill_middle_throttle()
                return self._pick_prefill_batch(ready_only=ready_only)
            if has_prefill:
                # Arrival-order protection applies to every READY prefill:
                # when both kinds of work are ready, dispatch strictly by
                # arrival seq (see _schedule_by_arrival).
                if has_decode or has_draft:
                    return self._schedule_by_arrival(ready_only=ready_only)
                self.cloud_scheduling_state = (
                    CloudSchedulingState.EXPECT_EXECUTE_DECODE_OR_DRAFT
                )
                if getattr(
                    self.ready_prefills[0], "cloud_suggest_slicing", False
                ):
                    self._start_prefill_middle_throttle()
                return self._pick_prefill_batch(ready_only=ready_only)
            # No ready prefill: callback to ready decode/draft.
            if has_draft or has_decode:
                self._clear_prefill_middle_throttle()
                return self._pick_decode_or_draft_by_arrival(
                    ready_only=ready_only
                )
        else:  # EXPECT_EXECUTE_DECODE_OR_DRAFT
            # Decode/Draft in arrival order (see
            # _pick_decode_or_draft_by_arrival).
            if has_draft or has_decode:
                # [INVERSION-HAZARD] 观测（历史遗留）：就绪门控 + 预挂
                # irecv 已从机制上消除该镜像死锁（recv 顺序在 SO 到达时
                # 即固定，不再取决于派发顺序），此处仅在“被超越的 prefill
                # 本身已就绪”时记录，作为状态机优先级的纯观测日志。
                if self.ready_prefills and self._prefill_head_data_ready():
                    _pf_seq = self._arrival_seq(self.ready_prefills[0])
                    _dc_seq = (
                        self._arrival_seq(self.ready_decodes[0])
                        if self.ready_decodes else None
                    )
                    _dr_seqs = [
                        seq
                        for seq in (
                            (
                                self._arrival_seq(self.ready_decode_drafts[0])
                                if self.ready_decode_drafts else None
                            ),
                            (
                                self._arrival_seq(self.ready_prefill_drafts[0])
                                if self.ready_prefill_drafts else None
                            ),
                        )
                        if seq is not None
                    ]
                    _dr_seq = min(_dr_seqs) if _dr_seqs else None
                    _ch_seq = _dc_seq
                    if _dr_seq is not None and (
                            _ch_seq is None or _dr_seq < _ch_seq):
                        _ch_seq = _dr_seq
                    if (_pf_seq is not None and _ch_seq is not None
                            and _pf_seq < _ch_seq):
                        logger.info(
                            "[PD-PASSIVE] DECODE-state dispatch overtakes "
                            "an earlier-arrived ready prefill: "
                            "prefill_seq=%d, channel_seq=%d (state-machine "
                            "priority; channel order fixed at arrival).",
                            _pf_seq, _ch_seq,
                        )
                self.cloud_scheduling_state = (
                    CloudSchedulingState.EXPECT_EXECUTE_PREFILL
                )
                self._clear_prefill_middle_throttle()
                return self._pick_decode_or_draft_by_arrival(
                    ready_only=ready_only
                )
            # No ready Draft/Decode: callback to Prefill.  Stay in the
            # current state — the next schedule() call will check for
            # drafts again at its earliest opportunity.
            if self._can_fallback_to_prefill_in_decode_state():
                if self._active_prefill_slices:
                    self._start_prefill_middle_throttle()
                    return self._pick_prefill_batch(ready_only=ready_only)
                if has_prefill:
                    if getattr(
                        self.ready_prefills[0],
                        "cloud_suggest_slicing", False
                    ):
                        self._start_prefill_middle_throttle()
                    return self._pick_prefill_batch(ready_only=ready_only)
            else:
                return ScheduledBatch.empty()

        if self.ready_pdmixes:
            if (
                state == CloudSchedulingState.EXPECT_EXECUTE_DECODE_OR_DRAFT
                and not self._can_fallback_to_prefill_in_decode_state()
            ):
                return ScheduledBatch.empty()
            if state == CloudSchedulingState.EXPECT_EXECUTE_DECODE_OR_DRAFT:
                self._start_prefill_middle_throttle()
            return self._pick_prefill_batch(ready_only=ready_only)
        return ScheduledBatch.empty()

    def _schedule_from_queue(
        self, queue_name: str, ready_only: bool = True
    ) -> ScheduledBatch:
        if self._active_prefill_slices:
            if queue_name == "ready_decodes" and self.ready_decodes and (
                not ready_only or self._decode_head_data_ready()
            ):
                return self._pick_decode_batch()
            if queue_name in ("ready_prefills", "ready_pdmixes"):
                return self._pick_prefill_batch(ready_only=ready_only)
            return ScheduledBatch.empty()

        if queue_name == "ready_prefills":
            if self.ready_prefills and (
                not ready_only or self._prefill_head_data_ready()
            ):
                return self._build_batch(self.ready_prefills.popleft())
            return ScheduledBatch.empty()
        if queue_name == "ready_decodes":
            if self.ready_decodes and (
                not ready_only or self._decode_head_data_ready()
            ):
                return self._pick_decode_batch()
            return ScheduledBatch.empty()
        if queue_name == "ready_prefill_drafts":
            if self.ready_prefill_drafts and (
                not ready_only or self._prefill_draft_head_data_ready()
            ):
                return self._pick_prefill_draft_batch()
            return ScheduledBatch.empty()
        if queue_name == "ready_decode_drafts":
            # Decode-phase drafts dispatch on arrival (see
            # _pick_decode_or_draft_by_arrival); no watermark check.
            if self.ready_decode_drafts:
                return self._pick_decode_draft_batch()
            return ScheduledBatch.empty()
        if queue_name == "ready_pdmixes":
            if self.ready_pdmixes:
                return self._pick_prefill_batch(ready_only=ready_only)
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
            "pdmixes=%d, prefill_drafts=%d, decode_drafts=%d, decodes=%d) "
            "seq=%s",
            self.dispatch_policy.value,
            so.batch_type.value if so.batch_type is not None else "<none>",
            len(batch.slices),
            len(self.ready_prefills),
            len(self._active_prefill_slices),
            len(self.ready_pdmixes),
            len(self.ready_prefill_drafts),
            len(self.ready_decode_drafts),
            len(self.ready_decodes),
            self._arrival_seq(so),
        )

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #
    def has_pending(self) -> bool:
        return bool(
            self.ready_prefills
            or self._active_prefill_slices
            or self.ready_pdmixes
            or self.ready_prefill_drafts
            or self.ready_decode_drafts
            or self.ready_decodes
        )

    @property
    def num_pending(self) -> int:
        return (
            len(self.ready_prefills)
            + len(self._active_prefill_slices)
            + len(self.ready_pdmixes)
            + len(self.ready_prefill_drafts)
            + len(self.ready_decode_drafts)
            + len(self.ready_decodes)
        )
