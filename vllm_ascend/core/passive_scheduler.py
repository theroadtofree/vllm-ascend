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
import pickle
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from vllm import envs
from vllm.logger import logger
from vllm.v1.core.sched.output import BatchType, SchedulerOutput

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
class SchedulerDecision:
    """Read-only snapshot of what ``schedule()`` *would* do next.

    Produced by :meth:`PassiveScheduler._make_decision` without any side
    effects.  When cross-DP coordination is active, decisions from all DPs
    are exchanged via all-reduce and dp0's decision becomes authoritative.
    """
    batch_type: BatchType | None = None     # None → idle, nothing to dispatch
    dispatch_queue: str | None = None       # "ready_prefills" / "ready_decodes" / "ready_pdmixes" / None
    is_continuation: bool = False           # dispatch from _active_prefill_slices?
    token_count: int = 0                    # total_num_scheduled_tokens (0 for dummy)
    new_state: CloudSchedulingState | None = None  # EEP → EED transition hint
    throttle_action: str | None = None      # "start" / "clear" / None


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
        dp_coord_group=None,  # stateless ProcessGroup for cross-DP coordination
    ) -> None:
        self.pp_subscriber = pp_subscriber
        self.vllm_config = vllm_config
        self.dispatch_policy = dispatch_policy
        self.cloud_scheduling_state = CloudSchedulingState.EXPECT_EXECUTE_PREFILL

        # Optional cross-DP coordination group. Set by the engine core
        # when dp>1 + MoE + PD-separation.  schedule() uses it to
        # coordinate batch_type decisions across cloud DPs.
        self.dp_coord_group = dp_coord_group

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
    def poll_and_classify(self) -> None:
        """Drain SchedulerOutputs from the inbox (fed by the subscriber
        thread, or directly by `_drain_subscriber_inline` when the thread
        is disabled) and route each into its phase-specific ready queue.
        """
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
            # [DPDBG] classify dummy vs real prefill/decode. The dummy zmq
            # (from _patched_execute_dummy_batch / _publish_pd_dummy_zmq) is
            # published with batch_type=DECODE_FIRST but
            # total_num_scheduled_tokens==0 (the is_pd_dummy attr is lost in
            # zmq serialization, so the cloud detects dummy by tokens==0).
            # Distinguish so the dummy-flood vs real-request arrival counts per
            # cloud DP are visible: the dp=2 PP-init-timeout symptom is one
            # cloud DP seeing far more dummies before its real PREFILL_FIRST
            # than the other (e.g. seq=282 vs 68 for the real PREFILL_FIRST).
            _total = scheduler_output.total_num_scheduled_tokens
            if _total == 0:
                _kind = "DUMMY"
            elif bt in (BatchType.PURE_PREFILL, BatchType.PREFILL_FIRST):
                _kind = "REAL_PREFILL"
            elif bt in (BatchType.PURE_DECODE, BatchType.DECODE_FIRST):
                _kind = "REAL_DECODE"
            elif bt in (BatchType.PREFILL_LAST, BatchType.DECODE_LAST):
                _kind = "TAIL(edge-only)"
            else:
                _kind = f"OTHER({bt.value if bt is not None else None})"
            _kc = getattr(self, "_dpdbg_kind_count", None)
            if _kc is None:
                _kc = {}
                self._dpdbg_kind_count = _kc
            _kc[_kind] = _kc.get(_kind, 0) + 1
            _dp_rank = getattr(
                getattr(self.vllm_config, "parallel_config", None),
                "data_parallel_rank", "?",
            )
            logger.info(
                "[DPDBG] PassiveScheduler recv: dp_rank=%s seq=%s kind=%s "
                "batch_type=%s tokens=%s kind_counts=%s",
                _dp_rank, seq, _kind,
                bt.value if bt is not None else None, _total, _kc,
            )
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
                now = time.monotonic()
                if self._last_decode_first_arrival_ts is not None:
                    interval_ms = (now - self._last_decode_first_arrival_ts) * 1000
                    logger.debug(
                        "DECODE_FIRST arrival interval: %.2f ms",
                        interval_ms,
                    )
                self._last_decode_first_arrival_ts = now
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
                "PassiveScheduler classified seq=%s batch_type=%s "
                "(prefills=%d, pdmixes=%d, decodes=%d)",
                self._arrival_seq(scheduler_output),
                bt.value if bt is not None else "<none>",
                len(self.ready_prefills),
                len(self.ready_pdmixes),
                len(self.ready_decodes),
            )

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

    def _slice_for(
        self, so: SchedulerOutput, token_count: int = 0,
    ) -> list["LayerSliceInfo | None"]:
        # Decode-like and empty batches are never sliced. DECODE_FIRST is the
        # edge-cloud head segment of a decode step — same per-token shape as
        # PURE_DECODE, so it follows the same no-slice rule.
        if so.batch_type in (
            BatchType.PURE_DECODE,
            BatchType.DECODE_FIRST,
        ):
            return [None]

        _tk = token_count if token_count > 0 else so.total_num_scheduled_tokens
        total_slices = self._resolve_slice_count(_tk)
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
        return True
        '''
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
        logger.debug(
            f"[PD-PASSIVE] Throttle active: {elapsed_ms:.1f}ms / {limit_ms:.0f}ms, "
            f"still waiting for decode",
        )
        return False
        '''

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

    def _ready_prefill_is_sliced_first_block(self) -> bool:
        if not self.ready_prefills:
            return False
        slices = self._slice_for(self.ready_prefills[0])
        return len(slices) > 1 and isinstance(slices[0], LayerSliceInfo)

    def _schedule_by_arrival(self) -> ScheduledBatch:
        prefill_seq = self._arrival_seq(self.ready_prefills[0])
        decode_seq = self._arrival_seq(self.ready_decodes[0])
        if prefill_seq is None or decode_seq is None:
            self.cloud_scheduling_state = CloudSchedulingState.EXPECT_EXECUTE_DECODE
            self._start_prefill_middle_throttle()
            return self._build_batch(self.ready_prefills.popleft())
        if decode_seq < prefill_seq:
            logger.info(
                "[PD-PASSIVE] Decode arrived before prefill slice-0: "
                "decode_seq=%d, prefill_seq=%d",
                decode_seq,
                prefill_seq,
            )
            self._clear_prefill_middle_throttle()
            return self._build_batch(self.ready_decodes.popleft())
        logger.info(
            "[PD-PASSIVE] Prefill slice-0 arrived before decode: "
            "prefill_seq=%d, decode_seq=%d",
            prefill_seq,
            decode_seq,
        )
        self.cloud_scheduling_state = CloudSchedulingState.EXPECT_EXECUTE_DECODE
        self._start_prefill_middle_throttle()
        return self._build_batch(self.ready_prefills.popleft())

    # ------------------------------------------------------------------ #
    # EXPECT_ALTERNATION: decision / coordination / application           #
    # ------------------------------------------------------------------ #

    def sync_queue_state(self) -> bool:
        """Sync queue lengths across cloud DPs via all_reduce.

        Checks that ready_prefills, ready_decodes, ready_pdmixes,
        _active_prefill_slices, and _active_sliced_prefill have consistent
        lengths/non-None status across all DPs.  If any queue is out of
        sync, the method returns False so the caller can sleep and retry.

        When dp_coord_group is not set or coordination is disabled,
        returns True immediately (no sync needed).
        """
        if self.dp_coord_group is None or not self._is_coordinated_dp():
            return True

        import torch
        import torch.distributed as dist
        _dp_size = getattr(
            self.vllm_config.parallel_config, "data_parallel_size", 1
        )
        _dp_rank = getattr(
            self.vllm_config.parallel_config, "data_parallel_rank", 0
        )

        # 5 fields: ready_prefills, ready_decodes, ready_pdmixes,
        #           _active_prefill_slices, has _active_sliced_prefill
        local = [
            len(self.ready_prefills),
            len(self.ready_decodes),
            len(self.ready_pdmixes),
            len(self._active_prefill_slices),
            1 if self._active_sliced_prefill is not None else 0,
        ]

        # Each rank writes to its own row; SUM leaves values independent.
        tensor = torch.zeros(_dp_size * 5, dtype=torch.int32, device="cpu")
        base = _dp_rank * 5
        for i, v in enumerate(local):
            tensor[base + i] = v
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM,
                         group=self.dp_coord_group)

        # Verify all rank values are identical for each of the 5 queues.
        all_match = True
        for q in range(5):
            vals = [int(tensor[r * 5 + q].item()) for r in range(_dp_size)]
            if len(set(vals)) != 1:
                _names = ["ready_prefills", "ready_decodes", "ready_pdmixes",
                          "_active_prefill_slices", "_active_sliced_prefill"]
                logger.warning(
                    "[SYNC-QUEUE] rank=%s %s mismatch: %s",
                    _dp_rank, _names[q], vals,
                )
                all_match = False
        return all_match

    def _schedule_expect_alternation(self) -> ScheduledBatch:
        """Pick the next batch to dispatch under the EEP/EED state machine.

        When cross-DP coordination is active (``dp_coord_group`` is set), the
        three-phase flow is used: decide → coordinate → apply.  Otherwise
        the original single-DP logic is used.
        """
        if self.dp_coord_group is not None and self._is_coordinated_dp():
            return self._schedule_expect_alternation_coordinated()
        return self._schedule_expect_alternation_simple()

    def _is_coordinated_dp(self) -> bool:
        """True when cross-DP coordination should be active on the cloud side:
        dp_coord_group is set AND model is MoE AND PD-separation is enabled."""
        return (
            bool(getattr(self.vllm_config.model_config, "is_moe", False))
            and getattr(
                self.vllm_config.parallel_config, "enable_edge_cloud", False
            )
        )

    def _schedule_expect_alternation_simple(self) -> ScheduledBatch:
        """Original single-DP EEP/EED state machine (no cross-DP coord)."""
        state = self.cloud_scheduling_state
        if state == CloudSchedulingState.EXPECT_EXECUTE_PREFILL:
            if self._active_prefill_slices:
                self.cloud_scheduling_state = (
                    CloudSchedulingState.EXPECT_EXECUTE_DECODE
                )
                self._start_prefill_middle_throttle()
                return self._build_active_prefill_slice_batch()
            if self.ready_prefills:
                if (
                    self.ready_decodes
                    and self._ready_prefill_is_sliced_first_block()
                ):
                    return self._schedule_by_arrival()
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

    def _schedule_expect_alternation_coordinated(self) -> ScheduledBatch:
        """Three-phase dispatch: decide → coordinate → apply.

        Phase 1 – each DP produces a read-only ``SchedulerDecision``.
        Phase 2 – decisions are exchanged via all-reduce; dp0's wins.
        Phase 3 – the coordinated decision is applied locally: state
        machine transitions, throttle, and queue popping.
        """
        decision = self._make_decision_alternation()
        if decision.batch_type is not None:
            self._log_queue_state("post-local-decision")
            self._log_local_decision(decision, "post-local-decision")
        decision = self._coordinate_decision(decision)
        if decision.batch_type is not None:
            self._log_local_decision(decision, "post-coord-decision")
        if decision.batch_type is None:
            return ScheduledBatch.empty()
        return self._apply_decision_alternation(decision)

    def _log_queue_state(self, tag: str) -> None:
        """Log queue lengths and per-item metadata for debugging."""
        _dp_rank = getattr(
            self.vllm_config.parallel_config, "data_parallel_rank", 0
        )
        # Per-item summaries.
        def _pf_str(so):
            return f"bt={so.batch_type.value if so.batch_type else '?'} tok={so.total_num_scheduled_tokens}"
        def _si_str(task):
            si = task.slice_info
            return f"st={si.start_layer},ed={si.end_layer}" if si else "None"
        pf_items = [_pf_str(so) for so in self.ready_prefills]
        dec_items = [_pf_str(so) for so in self.ready_decodes]
        pdmix_items = [_pf_str(so) for so in self.ready_pdmixes]
        as_items = [_si_str(t) for t in self._active_prefill_slices]
        logger.error(
            "[COORD-DIAG] DP%s %s state=%s "
            "pf[%d]=[%s] dec[%d]=[%s] pdmix[%d]=[%s] "
            "active_slices[%d]=[%s]",
            _dp_rank, tag, self.cloud_scheduling_state.name,
            len(self.ready_prefills), ", ".join(pf_items),
            len(self.ready_decodes), ", ".join(dec_items),
            len(self.ready_pdmixes), ", ".join(pdmix_items),
            len(self._active_prefill_slices), ", ".join(as_items),
        )

    def _log_local_decision(
        self, decision: SchedulerDecision, tag: str
    ) -> None:
        _dp_rank = getattr(
            self.vllm_config.parallel_config, "data_parallel_rank", 0
        )
        logger.error(
            "[COORD-DIAG] DP%s %s bt=%s cont=%s tok=%s "
            "new_state=%s throttle=%s queue=%s",
            _dp_rank, tag,
            decision.batch_type.value if decision.batch_type else "EMPTY",
            decision.is_continuation,
            decision.token_count,
            decision.new_state.name if decision.new_state else "-",
            decision.throttle_action or "-",
            decision.dispatch_queue or "-",
        )

    def _make_decision_alternation(self) -> SchedulerDecision:
        """Read-only: produce a ``SchedulerDecision`` from the current
        state machine, throttle, and ready queues.  Zero side effects."""
        state = self.cloud_scheduling_state

        if state == CloudSchedulingState.EXPECT_EXECUTE_PREFILL:
            if self._active_prefill_slices:
                return SchedulerDecision(
                    batch_type=(
                        self._active_sliced_prefill.batch_type
                        if self._active_sliced_prefill else None
                    ),
                    is_continuation=True,
                    token_count=(
                        self._active_sliced_prefill.total_num_scheduled_tokens
                        if self._active_sliced_prefill else 0
                    ),
                    new_state=CloudSchedulingState.EXPECT_EXECUTE_DECODE,
                    throttle_action="start",
                )
            if self.ready_prefills:
                _so = self.ready_prefills[0]
                return SchedulerDecision(
                    batch_type=_so.batch_type,
                    dispatch_queue="ready_prefills",
                    token_count=_so.total_num_scheduled_tokens,
                    new_state=CloudSchedulingState.EXPECT_EXECUTE_DECODE,
                    throttle_action="start",
                )
            if self.ready_decodes:
                _so = self.ready_decodes[0]
                return SchedulerDecision(
                    batch_type=_so.batch_type,
                    dispatch_queue="ready_decodes",
                    token_count=_so.total_num_scheduled_tokens,
                    throttle_action="clear",
                )
        else:  # EXPECT_EXECUTE_DECODE
            if self.ready_decodes:
                _so = self.ready_decodes[0]
                return SchedulerDecision(
                    batch_type=_so.batch_type,
                    dispatch_queue="ready_decodes",
                    token_count=_so.total_num_scheduled_tokens,
                    new_state=CloudSchedulingState.EXPECT_EXECUTE_PREFILL,
                    throttle_action="clear",
                )
            if self._can_fallback_to_prefill_in_decode_state():
                if self._active_prefill_slices:
                    return SchedulerDecision(
                        batch_type=(
                            self._active_sliced_prefill.batch_type
                            if self._active_sliced_prefill else None
                        ),
                        is_continuation=True,
                        token_count=(
                            self._active_sliced_prefill.total_num_scheduled_tokens
                            if self._active_sliced_prefill else 0
                        ),
                        throttle_action="start",
                    )
                if self.ready_prefills:
                    _so = self.ready_prefills[0]
                    return SchedulerDecision(
                        batch_type=_so.batch_type,
                        dispatch_queue="ready_prefills",
                        token_count=_so.total_num_scheduled_tokens,
                        throttle_action="start",
                    )

        if self.ready_pdmixes:
            _so = self.ready_pdmixes[0]
            _blocked = (
                state == CloudSchedulingState.EXPECT_EXECUTE_DECODE
                and not self._can_fallback_to_prefill_in_decode_state()
            )
            if not _blocked:
                return SchedulerDecision(
                    batch_type=_so.batch_type,
                    dispatch_queue="ready_pdmixes",
                    token_count=_so.total_num_scheduled_tokens,
                    throttle_action=(
                        "start"
                        if state == CloudSchedulingState.EXPECT_EXECUTE_DECODE
                        else None
                    ),
                )

        return SchedulerDecision()

    def _coordinate_decision(
        self, decision: SchedulerDecision
    ) -> SchedulerDecision:
        """Cross-DP coordination: exchange full decisions via all-reduce.

        Phase 1 — all-reduce batch_type categories to pick the winner
        (dp0 preferred).  Phase 2 — all-reduce(MAX) the winner's
        pickled SchedulerDecision so every DP receives the complete
        decision (dispatch_queue, is_continuation, etc.), not just
        the batch_type.
        """
        if self.dp_coord_group is None:
            return decision

        # Check whether coordination should be active.
        _is_moe = bool(getattr(
            self.vllm_config.model_config, "is_moe", False
        ))
        _pd_enabled = getattr(
            self.vllm_config.parallel_config, "enable_edge_cloud", False
        )
        if not _is_moe or not _pd_enabled:
            return decision

        import torch
        import torch.distributed as dist
        _dp_size = getattr(
            self.vllm_config.parallel_config, "data_parallel_size", 1
        )
        _dp_rank = getattr(
            self.vllm_config.parallel_config, "data_parallel_rank", 0
        )

        # Map batch_type → coordination category.
        _bt = decision.batch_type
        _cat = 0  # EMPTY / idle
        if _bt in (BatchType.PURE_PREFILL, BatchType.PREFILL_FIRST):
            _cat = 1
        elif _bt in (BatchType.PURE_DECODE, BatchType.DECODE_FIRST):
            _cat = 2
        elif _bt == BatchType.PD_MIX:
            _cat = 3

        tensor = torch.zeros(_dp_size, dtype=torch.int32, device="cpu")
        tensor[_dp_rank] = _cat
        dist.all_reduce(tensor, group=self.dp_coord_group)

        # Determine the winner rank from the category tensor.
        # dp0 is checked first; if dp0 is idle, dp1 takes over
        # so a request bound to dp1 alone is not starved.
        winner_rank = -1
        for r in range(_dp_size):
            if tensor[r].item() != 0:
                winner_rank = r
                break

        if winner_rank == -1:
            return SchedulerDecision()  # all idle

        # Phase 2: all_reduce(MAX) transmits the winner's FULL
        # SchedulerDecision (batch_type, dispatch_queue, is_continuation,
        # token_count, new_state, throttle_action).
        # Non-winners zero-fill their buffer; MAX(0, bytes) = bytes
        # preserves the winner's data untouched.
        _MAX_SER_LEN = 4096
        buf_tensor = torch.zeros(_MAX_SER_LEN, dtype=torch.uint8,
                                  device="cpu")
        if _dp_rank == winner_rank:
            data = pickle.dumps(decision)
            if len(data) > _MAX_SER_LEN:
                raise RuntimeError(
                    f"SchedulerDecision serialized size "
                    f"{len(data)} > {_MAX_SER_LEN}"
                )
            buf_tensor[:len(data)] = torch.tensor(
                list(data), dtype=torch.uint8
            )

        dist.all_reduce(buf_tensor, op=dist.ReduceOp.MAX,
                         group=self.dp_coord_group)

        data = bytes(buf_tensor.tolist()).rstrip(b'\x00')
        winner_decision: SchedulerDecision = pickle.loads(data)

        # Non-winner DPs inherit the winner's decision wholesale to stay
        # in lockstep.
        if _dp_rank != winner_rank:
            return SchedulerDecision(
                batch_type=winner_decision.batch_type,
                dispatch_queue=winner_decision.dispatch_queue,
                is_continuation=winner_decision.is_continuation,
                token_count=winner_decision.token_count,
                new_state=winner_decision.new_state,
                throttle_action=winner_decision.throttle_action,
            )
        return winner_decision

    def _apply_decision_alternation(
        self, decision: SchedulerDecision
    ) -> ScheduledBatch:
        """Apply the coordinated decision: state transitions, throttle,
        queue popping.

        When the decision matches this DP's natural intent,
        ``_schedule_by_arrival`` / ``_build_batch`` / continuation is
        used.  Otherwise a dummy SO with the coordinated batch_type is
        produced so cross-DP EP all-toall stays paired.
        """
        _bt = decision.batch_type
        _is_prefill = _bt in (BatchType.PURE_PREFILL, BatchType.PREFILL_FIRST)
        _is_decode = _bt in (BatchType.PURE_DECODE, BatchType.DECODE_FIRST)
        _is_pdmix = _bt == BatchType.PD_MIX

        if _bt is not None and (_is_prefill or _is_decode):
            logger.info(
                "[APPLY-DECISION] is_prefill=%s is_decode=%s",
                _is_prefill, _is_decode,
            )

        # --- state-machine transitions ---
        if decision.new_state is not None:
            self.cloud_scheduling_state = decision.new_state
        if decision.throttle_action == "start":
            self._start_prefill_middle_throttle()
        elif decision.throttle_action == "clear":
            self._clear_prefill_middle_throttle()

        # --- dispatch ---
        if decision.is_continuation:
            if self._active_prefill_slices:
                if _bt is not None:
                    logger.info(
                        "[APPLY-DECISION] branch=continuation-real "
                        "bt=%s active_slices=%d",
                        _bt.value, len(self._active_prefill_slices),
                    )
                return self._build_active_prefill_slice_batch()
            # Continuation expected but not available — create dummy.
            if _bt is not None:
                logger.info(
                    "[APPLY-DECISION] branch=continuation-dummy "
                    "bt=%s (no active slices)",
                    _bt.value,
                )
            return self._make_dummy_batch(_bt)

        if _is_prefill:
            if decision.dispatch_queue == "ready_prefills" and self.ready_prefills:
                if _bt is not None:
                    logger.info(
                        "[APPLY-DECISION] branch=prefill-real "
                        "bt=%s queue=%s ready_prefills=%d",
                        _bt.value, decision.dispatch_queue,
                        len(self.ready_prefills),
                    )
                return self._build_batch(self.ready_prefills.popleft(),
                                         decision.token_count)
            if _bt is not None:
                logger.info(
                    "[APPLY-DECISION] branch=prefill-dummy "
                    "bt=%s queue=%s ready_prefills=%d "
                    "dispatch_queue_match=%s ready_nonempty=%s",
                    _bt.value, decision.dispatch_queue,
                    len(self.ready_prefills),
                    decision.dispatch_queue == "ready_prefills",
                    bool(self.ready_prefills),
                )
            return self._make_dummy_batch(_bt)

        if _is_decode:
            if decision.dispatch_queue == "ready_decodes" and self.ready_decodes:
                if _bt is not None:
                    logger.info(
                        "[APPLY-DECISION] branch=decode-real "
                        "bt=%s queue=%s ready_decodes=%d",
                        _bt.value, decision.dispatch_queue,
                        len(self.ready_decodes),
                    )
                return self._build_batch(self.ready_decodes.popleft(),
                                         decision.token_count)
            if _bt is not None:
                logger.info(
                    "[APPLY-DECISION] branch=decode-dummy "
                    "bt=%s queue=%s ready_decodes=%d "
                    "dispatch_queue_match=%s ready_nonempty=%s",
                    _bt.value, decision.dispatch_queue,
                    len(self.ready_decodes),
                    decision.dispatch_queue == "ready_decodes",
                    bool(self.ready_decodes),
                )
            return self._make_dummy_batch(_bt)

        if _is_pdmix:
            if decision.dispatch_queue == "ready_pdmixes" and self.ready_pdmixes:
                if _bt is not None:
                    logger.info(
                        "[APPLY-DECISION] branch=pdmix-real "
                        "bt=%s queue=%s ready_pdmixes=%d",
                        _bt.value, decision.dispatch_queue,
                        len(self.ready_pdmixes),
                    )
                return self._build_batch(self.ready_pdmixes.popleft(),
                                         decision.token_count)
            if _bt is not None:
                logger.info(
                    "[APPLY-DECISION] branch=pdmix-dummy "
                    "bt=%s queue=%s ready_pdmixes=%d "
                    "dispatch_queue_match=%s ready_nonempty=%s",
                    _bt.value, decision.dispatch_queue,
                    len(self.ready_pdmixes),
                    decision.dispatch_queue == "ready_pdmixes",
                    bool(self.ready_pdmixes),
                )
            return self._make_dummy_batch(_bt)

        if _bt is not None:
            logger.info(
                "[APPLY-DECISION] branch=empty-fallback bt=%s", _bt.value,
            )
        return ScheduledBatch.empty()

    def _make_dummy_batch(self, batch_type: BatchType) -> ScheduledBatch:
        """Create a dummy SchedulerOutput (tokens=0, is_pd_dummy=True)
        with the given batch_type, so this DP participates in cross-DP
        EP all-toall without real work.

        Returns a ScheduledBatch directly (bypasses _build_batch) to
        avoid populating _active_prefill_slices with dummy slices."""
        so = SchedulerOutput.make_empty()
        so.batch_type = batch_type
        setattr(so, "is_pd_dummy", True)
        return ScheduledBatch(scheduler_output=so, slices=[None])

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

    def _build_batch(self, so: SchedulerOutput, token_count: int = 0) -> ScheduledBatch:
        slices = self._slice_for(so, token_count)
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
            "pdmixes=%d, decodes=%d) seq=%s",
            self.dispatch_policy.value,
            so.batch_type.value if so.batch_type is not None else "<none>",
            len(batch.slices),
            len(self.ready_prefills),
            len(self._active_prefill_slices),
            len(self.ready_pdmixes),
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
