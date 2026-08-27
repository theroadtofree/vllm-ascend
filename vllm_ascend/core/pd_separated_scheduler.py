# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import time
from collections import deque
from dataclasses import dataclass, replace

import numpy as np
from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from vllm.logger import logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import BatchType, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.distributed.edge_cloud_comm.scheduler_link import (
    is_irecv_complete,
)
from vllm_ascend.distributed.edge_cloud_comm.types import CommChannelType


class PrefillState(enum.Enum):
    """Edge-side prefill in-flight state machine.

    See Phase5 design in ``PDbatch分离边云协同Phase5&7详细设计.md``.
    """
    IDLE = "idle"       # prefill_inflight_count == 0
    LOW = "low"         # prefill_inflight_count == 1
    HIGH = "high"       # prefill_inflight_count >= prefill_inflight_limit


@dataclass
class PrefillChunkFlight:
    """Per-chunk in-flight tracking for chunk-prefill-prior scheduling.

    When ``chunk_prefill_prior_enable`` is True, each PREFILL_FIRST batch
    creates one ``PrefillChunkFlight`` keyed by its ``head_token``.  This
    allows the same request to have multiple chunks in-flight
    simultaneously — the next chunk's PF can be dispatched before the
    previous chunk's PL returns.

    Fields
    ------
    request_id : str
        The owning request.
    head_token : str
        Unique token assigned to this chunk's PREFILL_FIRST batch.
        PREFILL_LAST echoes it back so the flight can be located.
    chunk_index : int
        0-based index of this chunk within the request.
    is_last_chunk : bool
        True when this chunk consumes the last remaining prompt tokens.
    num_scheduled_tokens : int
        Number of tokens scheduled in this chunk.
    """
    request_id: str
    head_token: str
    chunk_index: int
    is_last_chunk: bool
    num_scheduled_tokens: int


@dataclass(frozen=True)
class PDActiveFlight:
    """A protected edge-cloud transaction from First creation to Last finish."""

    request_ids: tuple[str, ...]
    first_batch_type: BatchType
    last_batch_type: BatchType
    flight_id: str
    created_at: float


_PD_FIRST_TO_LAST = {
    BatchType.PREFILL_FIRST: BatchType.PREFILL_LAST,
    BatchType.DECODE_FIRST: BatchType.DECODE_LAST,
    BatchType.PREFILL_DRAFT_FIRST: BatchType.PREFILL_DRAFT_LAST,
    BatchType.DECODE_DRAFT_FIRST: BatchType.DECODE_DRAFT_LAST,
}
_PD_LAST_TO_FIRST = {last: first for first, last in _PD_FIRST_TO_LAST.items()}

# Scheduled-draft batch types, split by chain phase: prefill-phase chains
# travel on the dedicated PREFILL_DRAFT pair, decode-phase chains on the
# DECODE pair (shared with plain decode).
_DRAFT_FIRST_TYPES = (
    BatchType.PREFILL_DRAFT_FIRST,
    BatchType.DECODE_DRAFT_FIRST,
)
_DRAFT_LAST_TYPES = (
    BatchType.PREFILL_DRAFT_LAST,
    BatchType.DECODE_DRAFT_LAST,
)


# Default cap on concurrent PF→PL round trips (2P pipelining).  This is a
# pure scheduler-side counter: with key-tagged pre-posted irecvs the
# FIFO channel pairs multiple in-flight batches by order, so the limit is
# no longer derived from the channel pool size.
_DEFAULT_PREFILL_INFLIGHT_LIMIT = 2


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

    Chunk-prefill-prior (Phase 1)
    -----------------------------
    When ``chunk_prefill_prior_enable`` is True, per-chunk flight tracking
    replaces the request-granularity ``prefill_last_pending`` list.  This
    allows the next chunk's PREFILL_FIRST to be dispatched before the
    previous chunk's PREFILL_LAST returns, achieving the same pipeline
    interleaving that MindIE's ``generate_send_metadata_to_queue()``
    provides.
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
        self.decodes_first_ready: deque[SchedulerOutput] = deque()
        self.decodes_last_ready: deque[SchedulerOutput] = deque()
        # Draft work is split into two independent lanes keyed by
        # SchedulerOutput.draft_prefill_phase.  Prefill-phase chains travel
        # on the dedicated PREFILL_DRAFT channel pair, decode-phase chains
        # share the DECODE pair with plain decode, so the two lanes have no
        # data-plane dependency and are queued/gated separately.
        self.prefill_drafts_first_ready: deque[SchedulerOutput] = deque()
        self.prefill_drafts_last_ready: deque[SchedulerOutput] = deque()
        self.decode_drafts_first_ready: deque[SchedulerOutput] = deque()
        self.decode_drafts_last_ready: deque[SchedulerOutput] = deque()

        self._step_counter: int = 0

        # In-flight prefill limit (head-segment batches): a pure
        # scheduler-side counter, decoupled from the channel pool.
        self.prefill_inflight_limit: int = getattr(
            self.scheduler_config, "pd_prefill_inflight_limit",
            _DEFAULT_PREFILL_INFLIGHT_LIMIT,
        )
        self.prefill_inflight_count: int = 0
        self.decode_or_draft_inflight_limit: int = 1
        # Decode-lane in-flight counter: DECODE_FIRST and decode-phase
        # DRAFT_FIRST heads picked but not yet completed
        # (update_from_output).  They share the DECODE stream, so they must
        # not be in flight simultaneously (the cloud's recv order could
        # mismatch the edge's send order).  Prefill-phase DRAFT_FIRST uses
        # the dedicated PREFILL_DRAFT stream and is NOT counted here.
        self.decode_or_draft_inflight_count: int = 0
        # DECODE_FIRST heads picked but not yet completed (update_from_output).
        # Decode-phase DRAFT_FIRST and DECODE_FIRST use different recv
        # primitives but share the DECODE stream, so they must not be in
        # flight simultaneously.  Decode-phase DRAFT_FIRST+DRAFT_FIRST is
        # safe (same primitive, FIFO), so draft pipelining only needs to
        # gate on decode heads, not total heads.
        self.decode_head_inflight_count: int = 0
        # Per-lane count of draft round trips started (DRF picked) but not
        # yet completed (matching DRL's update_from_output).  Each lane
        # keeps its own intra-chain pipelining credit: a lane may have up
        # to `_draft_remote_pending_limit` outstanding round trips.
        self.prefill_draft_remote_pending_count: int = 0
        self.decode_draft_remote_pending_count: int = 0

        # Comm-layer sequencing: per-channel-pair counters.  The edge
        # scheduler is the single ordering authority — it stamps
        # ``comm_seqno`` at FIRST-batch pick time, FIRST and its matching
        # LAST share the value, and both peers derive the same per-channel
        # order from the SO alone.  DF and decode-phase DRF share the
        # DECODE pair (one counter); prefill-phase DRF uses the dedicated
        # PREFILL_DRAFT pair.
        self._prefill_comm_seqno: int = 0
        self._decode_comm_seqno: int = 0
        self._prefill_draft_comm_seqno: int = 0
        # head_token -> first comm seqno of the not-yet-created draft
        # chain that follows this PF/DF batch.  Reserved at FIRST pick
        # time so both peers can pre-post all n draft recv requests as
        # soon as the parent is published; DRF steps consume
        # base + draft_step_idx at pick time.
        self._reserved_draft_seqno_base: dict[str, int] = {}

        # Buffer queue: requests whose P-first segment is done but P-last
        # segment has not yet returned from the cloud.  Not eligible for
        # decode scheduling until PL completes and they are moved to running.
        # When chunk_prefill_prior_enable is True, this is supplemented by
        # per-chunk flight tracking.
        self.prefill_last_pending: list[Request] = []

        # ------------------------------------------------------------------ #
        # Chunk-prefill-prior fields                                          #
        # ------------------------------------------------------------------ #
        # Enabled via pd_separation.next_prefill_prior_enable. When True the
        # scheduler yields a freed prefill slot to a *different* request
        # (cross-request head-prior, MindIE-style P1首->P2首) instead of
        # ahead-dispatching the same request's next chunk.
        self.next_prefill_prior_enable: bool = getattr(
            self.scheduler_config, "pd_next_prefill_prior_enable", False
        )
        # Enabled via pd_separation.chunk_prefill_prior_enable.
        self.chunk_prefill_prior_enable: bool = getattr(
            self.scheduler_config, "pd_chunk_prefill_prior_enable", False
        )
        self.max_chunk_prefill_ahead: int = getattr(
            self.scheduler_config, "pd_max_chunk_prefill_ahead", 1
        )

        # Per-chunk flight tracking: head_token → PrefillChunkFlight.
        # Populated on PF, consumed on PL.
        self._prefill_flight_by_token: dict[str, PrefillChunkFlight] = {}

        # Per-request count of chunks still waiting for PL.
        # request_id → count.  When count reaches 0, the request is
        # eligible to enter decode.
        self._pending_tail_count: dict[str, int] = {}

        # Per-request count of chunks whose PF was dispatched ahead
        # (before the previous chunk's PL returned).  Decremented on
        # PL return so the request is not re-added to chunk_prefill_first.
        self._ahead_chunk_count: dict[str, int] = {}

        # A request is protected from preemption from the moment a First batch
        # is created until the matching Last batch finishes update_from_output.
        # The count (rather than a bool) supports multiple ahead-scheduled
        # prefill chunks for the same request.
        self._pd_active_flight_count: dict[str, int] = {}
        self._pd_active_flight_by_key: dict[
            tuple[BatchType, str], PDActiveFlight
        ] = {}

        # 限制 PREFILL_FIRST 每个 batch 最多只组 1 个请求。
        # 配置路径: additional_config.edge_cloud_config.pd_separation.limit_prefill_batch_size
        self.limit_prefill_batch_size: bool = False
        _additional = getattr(self.vllm_config, "additional_config", None)
        if isinstance(_additional, dict):
            _ec = _additional.get("edge_cloud_config", {})
            _pd = _ec.get("pd_separation", {})
            self.limit_prefill_batch_size = bool(
                _pd.get("limit_prefill_batch_size", False)
            )

        # [新增] 强制调度 D尾 标记。
        # D首 调度后置 True，D尾 调度后置 False。
        # True 时禁止调度 D首，严格保证 DF -> DL 交替时序。
        self._force_decode_last: bool = False
        # Per-lane DRAFT_FIRST -> DRAFT_LAST alternation flags.  Set when a
        # lane's DRAFT_FIRST is picked, cleared when its matching DRAFT_LAST
        # is picked; while set, no new DRAFT_FIRST of the SAME lane may be
        # picked.  The two lanes alternate independently.
        self._force_prefill_draft_last: bool = False
        self._force_decode_draft_last: bool = False

        # After scheduling a DECODE_LAST or DRAFT_LAST, briefly reserve the
        # next scheduling opportunity for DECODE_FIRST or DRAFT_FIRST only.
        self._decode_or_draft_first_only_start_ts: float | None = None
        self._decode_or_draft_first_only_window_ms: int = 20
        # Rate limiter for the [PD-STALL] empty-schedule probe.
        self._last_stall_log_ts: float = 0.0

        # Async scheduled-MTP keeps real draft token IDs in the edge worker.
        # The scheduler only needs fixed-length placeholder SchedulerOutputs,
        # which can be generated and dispatched before the preceding worker
        # result is returned to EngineCore.  Cloud publication is finalized
        # separately once the target sampling scalars are available.
        # Keyed by draft_task_id: one prefill-phase chain and one
        # decode-phase chain may be pending publication concurrently.
        self._draft_publish_pending: dict[str, SchedulerOutput] = {}
        self._draft_publish_scalars_patched: set[str] = set()
        self._draft_publish_dispatched: set[str] = set()
        self._pregenerated_draft_task_ids: set[str] = set()
        self._pregenerated_draft_req_ids: dict[str, set[str]] = {}
        # Per-lane intra-chain pipelining credit: each draft lane may have
        # up to this many round trips outstanding (DRF picked, DRL not yet
        # completed) at the same time.  Sized to the chain length so a full
        # chain can be dispatched without waiting for DRL updates; the
        # credit then only bounds cross-chain backlog (a new chain's first
        # DRF waits until the previous chain's tail updates land).
        self._draft_remote_pending_limit: int = max(2, self.num_spec_tokens)
        self._decode_first_placeholder_parent: SchedulerOutput | None = None
        # Count of DECODE_LAST tails dispatched but not yet settled through
        # update_from_output.  A DECODE_FIRST created while this is nonzero
        # would record PRE-SETTLE (optimistic) num_computed values into its
        # SchedulerOutput: the queue fill loop schedules several batches per
        # EngineCore turn but processes only one output per turn, so the
        # previous chain's DL settle routinely lags the next DF's creation.
        # On a rejection step the optimistic value is one ahead of the
        # settled truth and the phantom +1 propagates to every later SO
        # (observed: worker kernel positions one ahead of the verified
        # truth -> KV misalignment -> NaN row -> missing-correction crash).
        self._unsettled_decode_tail_count: int = 0

        # ------------------------------------------------------------------ #
        # Edge-cloud deferred-draft KV retention                             #
        # ------------------------------------------------------------------ #
        # Non-edge-cloud proposes draft tokens inside the same execute_model
        # call, i.e. strictly BEFORE update_from_output frees the finished
        # requests' KV blocks.  Edge-cloud defers the draft to a later
        # DRAFT_FIRST batch, so without retention the blocks could be freed
        # and reused before the cloud-side draft steps read/write them.
        # The EngineCore registers each deferred task locally from the parent
        # SchedulerOutput's head_token before update_from_output. _free_request
        # then delays _free_blocks for referenced requests until the draft task
        # completes or is dropped (release_draft_retained_blocks).
        # Everything is gated on `_edge_cloud_draft_retention_enabled` so
        # deployments without the scheduled edge-cloud draft behave exactly
        # as upstream.
        self._edge_cloud_draft_retention_enabled: bool = (
            self._check_scheduled_edge_cloud_draft()
        )
        self._edge_cloud_draft_req_tasks: dict[str, set[str]] = {}
        self._edge_cloud_draft_task_reqs: dict[str, set[str]] = {}
        self._draft_retained_requests: dict[str, dict[str, Request]] = {}
        # Draft task ids this scheduler dropped from its ready queues
        # while the runner still held the (enqueued) context.  Drained by
        # the EngineCore patch, which forwards them to the runner so the
        # orphaned context is reaped and the retention released.
        self._dropped_draft_task_ids_to_report: list[str] = []
        # Draft task ids whose cloud-side cached metadata will never be
        # (fully) consumed.  Stamped onto outgoing SchedulerOutputs so
        # the cloud model runner can purge the entries instead of
        # leaking them until the bounded cache evicts.
        self._pending_cloud_draft_invalidations: list[str] = []
        # Finishes deliberately withheld from the cloud: the edge keeps a
        # finished-but-chain-referenced request in self.requests and
        # retains its KV blocks, but upstream _free_request has already
        # added it to finished_req_ids, so the next published FIRST batch
        # would tell the cloud runner to remove the row mid-chain.  The
        # cloud's cached whole-batch draft metadata (block tables, seq
        # lens, positions) is laid out by the batch at chain creation;
        # removing a row condenses the live batch and shifts every
        # positional lookup behind it (recycled rows reading another
        # request's KV -> shifted/garbage draft rows).  Withhold the
        # finish on the cloud-bound copy until the chain releases the
        # request, then re-emit it on the next published batch.
        self._cloud_withheld_finished_req_ids: set[str] = set()
        self._cloud_released_finished_req_ids: set[str] = set()
        # Draft chains whose covered requests have ALL finished/aborted.
        # Dead chains are never dropped mid-flight: their comm seqnos were
        # reserved and their recvs pre-posted at parent publish time, so
        # dropping a queued step would leave a hole on the channel and
        # stall every later chain.  The worker drains the remaining steps
        # with dummy payloads (draft_chain_dead=True on the SO).
        self._dead_draft_task_ids: set[str] = set()
        # Dead-chain task ids whose deferred cloud control (step-0 awaits
        # scalars a dead parent never produces) must be released for
        # immediate publication.  Drained by the EngineCore patch, which
        # calls _release_deferred_draft_pre_out for each.
        self._dead_chain_publish_to_release: list[str] = []

    def invalidate_cloud_draft_tasks(self, task_ids: list[str]) -> None:
        """Queue cloud-side draft metadata invalidations (edge only)."""
        if not self._edge_cloud_draft_retention_enabled:
            return
        self._pending_cloud_draft_invalidations.extend(task_ids)

    def filter_cloud_finished_req_ids(
        self, finished_req_ids: set[str]
    ) -> set[str]:
        """Compute the finished_req_ids a cloud-bound batch should carry.

        The edge retains finished requests that are still referenced by an
        in-flight draft chain (see _free_request): their KV blocks stay
        allocated and they remain in self.requests until the chain
        releases them.  The cloud must apply the SAME retention for its
        batch rows: its cached whole-batch draft metadata is laid out by
        the batch at chain creation, so removing a row mid-chain shifts
        every positional lookup behind it (condense recycles the row for
        another request, and later chain steps then read/write the wrong
        request's KV -- observed as shifted/garbage draft rows and
        verify-time device faults).  Withhold such finishes here and
        re-emit them once release_draft_retained_blocks() drops the last
        referencing task.

        Only the published copy is filtered; the edge worker keeps
        receiving the unmodified SchedulerOutput and cleans up its own
        batch immediately, exactly as before.
        """
        if not self._edge_cloud_draft_retention_enabled:
            return finished_req_ids
        cloud_finished = set(finished_req_ids)
        for req_id in finished_req_ids:
            if not self._edge_cloud_draft_req_tasks.get(req_id):
                continue
            request = self.requests.get(req_id)
            if request is not None and not request.is_finished():
                # req_id was resubmitted while the old request's finish
                # was pending; the new request owns the row now, so the
                # stale finish must reach the cloud before its PREFILL.
                # Withholding it would strand the old row on the cloud.
                logger.error(
                    "[PD] req=%s finished while referenced by draft "
                    "tasks %s but a live request owns the id; NOT "
                    "withholding the finish from the cloud",
                    req_id,
                    self._edge_cloud_draft_req_tasks.get(req_id),
                )
                continue
            cloud_finished.discard(req_id)
            self._cloud_withheld_finished_req_ids.add(req_id)
            logger.info(
                "[PD] withholding cloud finish for req=%s until draft "
                "tasks %s release it",
                req_id,
                self._edge_cloud_draft_req_tasks.get(req_id),
            )
        if self._cloud_released_finished_req_ids:
            still_valid: set[str] = set()
            for req_id in self._cloud_released_finished_req_ids:
                request = self.requests.get(req_id)
                if request is not None and not request.is_finished():
                    # resubmitted while withheld: the new request adopted
                    # the cloud row; emitting the stale finish would
                    # remove the live request's row instead.
                    logger.error(
                        "[PD] withheld finish for req=%s became stale "
                        "(id resubmitted); the cloud row was adopted by "
                        "the new request",
                        req_id,
                    )
                    continue
                still_valid.add(req_id)
            cloud_finished |= still_valid
            self._cloud_released_finished_req_ids = set()
        return cloud_finished

    def schedule(self) -> SchedulerOutput:
        scheduler_output = self._schedule_pd_separated()
        # Only FIRST-segment batches are published to the cloud over PRE_OUT
        # (the publish hook drops PL/DL/DRL tails), and only batches whose
        # cloud-side execution runs the purge hook can deliver the
        # invalidations.  EMPTY batches are not broadcast either.  Stamping
        # any other batch type would silently discard the pending list, so
        # keep the invalidations queued until a cloud-bound batch can carry
        # them.
        if self._pending_cloud_draft_invalidations and (
            scheduler_output.batch_type
            in (
                BatchType.PREFILL_FIRST,
                BatchType.DECODE_FIRST,
            ) + _DRAFT_FIRST_TYPES
        ):
            scheduler_output.cloud_draft_invalidate_task_ids = (
                self._pending_cloud_draft_invalidations
            )
            self._pending_cloud_draft_invalidations = []
        return scheduler_output

    # ------------------------------------------------------------------ #
    # Chunk-prefill-prior helpers                                         #
    # ------------------------------------------------------------------ #
    def _can_ahead_schedule(self, req_id: str) -> bool:
        """True when the request can have one more chunk PF dispatched ahead."""
        return (
            self._ahead_chunk_count.get(req_id, 0) < self.max_chunk_prefill_ahead
        )

    def _has_other_prefill_request(self, current_req_id: str) -> bool:
        """True if a different request has prefill work ready to fill the next
        prefill slot (a cross-request head-prior candidate).

        Only called from the ahead-decision point inside
        ``_pick_prefill_first_batch``, where ``self.running`` temporarily
        holds the drained ``chunk_prefill_first`` (so other mid-prefill
        requests appear there); the ``is_prefill_chunk`` filter excludes
        decode requests that normally live in ``running``.

        Returns True when any of the following holds:
          - ``chunk_prefill_first`` contains a request with a different id;
          - ``running`` contains a still-prefilling request with a different
            id (drained-window case);
          - ``waiting`` is non-empty (a new request is available).
        """
        for req in self.chunk_prefill_first:
            if req.request_id != current_req_id:
                return True
        running = getattr(self, "running", None)
        if running:
            for req in running:
                if (req.request_id != current_req_id
                        and getattr(req, "is_prefill_chunk", False)):
                    return True
        return len(self.waiting) > 0

    def _should_ahead_schedule(self, req: Request, is_last: bool) -> bool:
        """Decide whether to ahead-dispatch ``req``'s next chunk (intra-request
        pipeline) or yield the prefill slot to another request (cross-request
        head-prior, MindIE-style P1首 -> P2首).

        Ahead (re-add ``req`` to ``chunk_prefill_first``) iff:
          - this is not the last chunk, AND
          - the request's ahead budget allows it, AND
          - next_prefill_prior_enable is off, OR no other request is
            available to fill the slot (single-request pipeline).

        Yield (send ``req`` to ``prefill_last_pending``; the PL-return path
        re-adds it for its next chunk when the tail returns) when
        ``next_prefill_prior_enable`` is on and another request has prefill
        work ready.
        """
        if is_last or not self._can_ahead_schedule(req.request_id):
            return False
        if (self.next_prefill_prior_enable
                and self._has_other_prefill_request(req.request_id)):
            return False
        return True

    def _select_single_prefill_candidate(
        self, candidates: list[Request],
    ) -> tuple[list[Request], list[Request]]:
        """Pick at most one prefill candidate to expose to ``super().schedule()``.

        Enforces the one-request-per-PF-batch invariant. chunk-prefill-prior
        keys one ``PrefillChunkFlight`` per ``head_token`` and the sampler
        consumes a batch-level ``is_last_prefill_chunk`` flag, so a PF batch
        must contain exactly one request. If two requests shared a batch they
        would share one ``head_token`` (the second overwrites the first in
        ``_prefill_flight_by_token``, losing its PL tracking) and the
        batch-level ``is_last`` flag could not represent both requests'
        last-chunk state, stalling the overwritten request.

        Returns ``(exposed, rest)`` where ``exposed`` has 0 or 1 request and
        ``rest`` holds the remaining candidates to be scheduled in their own
        PF batches on subsequent calls.
        """
        if not candidates:
            return [], []
        return [candidates[0]], list(candidates[1:])

    def _select_pf_candidate_head_prior(
        self, candidates: list[Request],
    ) -> tuple[Request | None, list[Request]]:
        """Pick at most one PF candidate for cross-request head-prior.

        Prefers a *fresh* candidate -- one with no in-flight chunk
        (``_pending_tail_count[req] == 0``), i.e. its previous chunk's PL has
        returned and it has no other chunk on the cloud. Refilling a freed
        slot with a fresh candidate keeps one in-flight chunk per request, so
        the two 2P1D prefill slots spread across different requests
        (MindIE-style ``P1首 / P2首`` interleaving).

        Decision order:
          1. First fresh candidate in ``candidates`` -> expose it (refill its
             slot with its next chunk).
          2. No fresh candidate but ``waiting`` is non-empty -> return
             ``(None, candidates)`` so the caller admits a *new* request from
             waiting instead of clustering another chunk on an already
             in-flight request (cross-request head-prior).
          3. No fresh candidate and no waiting request -> fall back to the
             first (in-flight) candidate: ahead-dispatch its next chunk so
             both 2P1D slots serve the single request (intra-request
             pipeline).

        Returns ``(exposed_or_None, rest)`` where ``exposed_or_None`` is 0 or
        1 request. ``rest`` holds the remaining candidates (untouched when
        admitting new, so they stay eligible for later batches).
        """
        for req in candidates:
            if self._pending_tail_count.get(req.request_id, 0) == 0:
                return req, [r for r in candidates if r is not req]
        if len(self.waiting) > 0:
            return None, list(candidates)
        exposed_list, rest = self._select_single_prefill_candidate(candidates)
        return (exposed_list[0] if exposed_list else None), rest

    def _prepare_pf_running_state(
        self,
        saved_chunk_prefill_first: list[Request],
        saved_running: list[Request],
        saved_max_num_running_reqs: int,
    ) -> tuple[list[Request], int, list[Request]]:
        """Decide ``(running, max_num_running_reqs, rest_candidates)`` for the
        ``super().schedule()`` call in ``_pick_prefill_first_batch``.

        - ``chunk_prefill_prior_enable``: enforce one-request-per-PF-batch (the
          flight map keys one flight per ``head_token`` and the sampler reads
          a batch-level ``is_last_prefill_chunk`` flag, so a PF batch must
          contain exactly one request). Candidate selection prefers a fresh
          head (no in-flight chunk) or a new waiting request over clustering
          on an in-flight request -- cross-request head-prior. ``max`` is
          capped at 1 (continue one candidate, no new admission) or a
          capacity-gated 0/1 (admit one new request from waiting).
        - Legacy (``chunk_prefill_prior_enable`` off): no per-chunk flight
          tracking, so multi-request PF batches are safe (PL routes by
          ``req_id``) and preferred for token-budget utilization. Expose all
          candidates and cap by system capacity -- the original behavior.

        The base scheduler caps scheduled running reqs by
        ``max_num_running_reqs`` (vllm ``Scheduler.schedule`` line ~390), so
        the value returned here directly bounds the PF batch size.
        """
        if self.chunk_prefill_prior_enable:
            exposed, rest_candidates = self._select_pf_candidate_head_prior(
                saved_chunk_prefill_first
            )
            if exposed is not None:
                # Continue one candidate; cap at 1 so the base does not admit
                # a new request alongside it (one-per-batch).
                return [exposed], 1, rest_candidates
            # No candidate to continue: admit at most one new request from
            # waiting, gated by system capacity (saved_running occupy their
            # slots) so we never exceed max_num_running_reqs system-wide.
            available = saved_max_num_running_reqs - len(saved_running)
            max_num_running_reqs = 1 if available >= 1 else 0
            return [], max_num_running_reqs, rest_candidates
        if self.limit_prefill_batch_size:
            if saved_chunk_prefill_first:
                return (
                    [saved_chunk_prefill_first[0]],
                    saved_max_num_running_reqs - len(saved_running),
                    [],
                )
            else:
                return (
                    [],
                    saved_max_num_running_reqs - len(saved_running),
                    [],
                )
        return (
            list(saved_chunk_prefill_first),
            saved_max_num_running_reqs - len(saved_running),
            [],
        )

    def _total_pending_tails(self) -> int:
        """Total number of chunks waiting for PL across all requests."""
        return sum(self._pending_tail_count.values())

    def _cleanup_request_flight_state(self, req_id: str) -> None:
        """Remove all tracking state for a finished request."""
        self._pending_tail_count.pop(req_id, None)
        self._ahead_chunk_count.pop(req_id, None)
        # Remove flights for this request.
        to_remove = [
            token for token, flight in self._prefill_flight_by_token.items()
            if flight.request_id == req_id
        ]
        for token in to_remove:
            self._prefill_flight_by_token.pop(token, None)

    @staticmethod
    def _pd_flight_key(
        scheduler_output: SchedulerOutput,
        first_batch_type: BatchType,
    ) -> tuple[BatchType, str]:
        if first_batch_type in _DRAFT_FIRST_TYPES:
            draft_task_id = scheduler_output.draft_task_id
            if not draft_task_id:
                raise RuntimeError("DRAFT flight is missing draft_task_id")
            if scheduler_output.draft_step_idx is None:
                raise RuntimeError("DRAFT flight is missing draft_step_idx")
            # A speculative chain reuses draft_task_id across multiple steps.
            # Include the step so asynchronously pipelined draft flights do not
            # collide in the active-flight map.
            flight_id = f"{draft_task_id}:{scheduler_output.draft_step_idx}"
        else:
            flight_id = scheduler_output.head_token
            if not flight_id:
                raise RuntimeError(
                    f"{first_batch_type.value} flight is missing head_token"
                )
        return first_batch_type, str(flight_id)

    @staticmethod
    def _pd_flight_request_ids(
        scheduler_output: SchedulerOutput,
    ) -> tuple[str, ...]:
        request_ids = dict.fromkeys(scheduler_output.num_scheduled_tokens)
        if scheduler_output.parent_req_id:
            request_ids.setdefault(scheduler_output.parent_req_id, None)
        return tuple(request_ids)

    def _register_pd_flight(self, scheduler_output: SchedulerOutput) -> None:
        """Protect requests in a First batch until its Last batch finishes."""
        first_batch_type = scheduler_output.batch_type
        last_batch_type = _PD_FIRST_TO_LAST.get(first_batch_type)
        if last_batch_type is None:
            raise RuntimeError(
                f"Cannot register non-First PD batch: {first_batch_type}"
            )

        key = self._pd_flight_key(scheduler_output, first_batch_type)
        if key in self._pd_active_flight_by_key:
            raise RuntimeError(f"Duplicate active PD flight: {key}")

        request_ids = self._pd_flight_request_ids(scheduler_output)
        if not request_ids:
            raise RuntimeError(
                f"{first_batch_type.value} flight has no associated requests"
            )

        self._pd_active_flight_by_key[key] = PDActiveFlight(
            request_ids=request_ids,
            first_batch_type=first_batch_type,
            last_batch_type=last_batch_type,
            flight_id=key[1],
            created_at=time.monotonic(),
        )
        for req_id in request_ids:
            self._pd_active_flight_count[req_id] = (
                self._pd_active_flight_count.get(req_id, 0) + 1
            )

    def _complete_pd_flight(self, scheduler_output: SchedulerOutput) -> bool:
        """Unprotect requests after a matching Last batch has completed."""
        last_batch_type = scheduler_output.batch_type
        first_batch_type = _PD_LAST_TO_FIRST.get(last_batch_type)
        if first_batch_type is None:
            raise RuntimeError(
                f"Cannot complete non-Last PD batch: {last_batch_type}"
            )

        key = self._pd_flight_key(scheduler_output, first_batch_type)
        flight = self._pd_active_flight_by_key.pop(key, None)
        if flight is None:
            logger.warning(
                "Ignoring stale or duplicate %s flight completion: key=%s",
                last_batch_type.value,
                key,
            )
            return False
        if flight.last_batch_type != last_batch_type:
            raise RuntimeError(
                "PD flight type mismatch: "
                f"expected={flight.last_batch_type}, got={last_batch_type}, "
                f"key={key}"
            )

        for req_id in flight.request_ids:
            count = self._pd_active_flight_count.get(req_id, 0)
            if count <= 0:
                raise RuntimeError(
                    "PD active flight count underflow: "
                    f"request_id={req_id}, key={key}"
                )
            if count == 1:
                self._pd_active_flight_count.pop(req_id)
            else:
                self._pd_active_flight_count[req_id] = count - 1
        return True

    def _is_request_preemptible(self, request: Request) -> bool:
        """Only idle RUNNING requests are safe preemption candidates."""
        return (
            request.status == RequestStatus.RUNNING
            and self._pd_active_flight_count.get(request.request_id, 0) == 0
        )

    def _select_preemption_candidate(self) -> Request | None:
        """Select an idle request without touching active edge-cloud work."""
        if self.policy == SchedulingPolicy.PRIORITY:
            candidates = (
                request
                for request in self.running
                if self._is_request_preemptible(request)
            )
            return max(
                candidates,
                key=lambda request: (request.priority, request.arrival_time),
                default=None,
            )

        return next(
            (
                request
                for request in reversed(self.running)
                if self._is_request_preemptible(request)
            ),
            None,
        )

    def _make_empty_batch(self) -> SchedulerOutput:
        scheduler_output = SchedulerOutput.make_empty()
        scheduler_output.finished_req_ids = self.finished_req_ids
        self.finished_req_ids = set()
        return scheduler_output

    def _schedule_pd_separated(self) -> SchedulerOutput:
        state = self._prefill_state()
        scheduler_output = self._pick_by_state(state, ready_only=True)
        if scheduler_output.batch_type == BatchType.EMPTY and (
            self.prefills_last_ready
            or self.decodes_last_ready
            or self.prefill_drafts_last_ready
            or self.decode_drafts_last_ready
        ):
            # Nothing is data-ready but tails are in flight: fall back to
            # the original priority order and dispatch anyway — the payload
            # wait happens device-side (wait_event on the pre-posted recv),
            # never a host block.  This keeps single-stream latency off the
            # report->drain->dispatch confirmation chain.
            scheduler_output = self._pick_by_state(state, ready_only=False)
        has_work = scheduler_output.total_num_scheduled_tokens > 0
        is_tail = scheduler_output.batch_type in (
            BatchType.PREFILL_LAST,
            BatchType.DECODE_LAST,
        ) + _DRAFT_LAST_TYPES
        if has_work or is_tail:
            self._log_scheduler_state(state, scheduler_output.batch_type)
        else:
            self._log_stall_if_blocked(scheduler_output)
        # Stamp whether this batch carries any multimodal request so the
        # cloud's early-recv hint (built in PassiveEC.step from this SO)
        # can decide whether to irecv mrope_positions. The passive cloud has
        # NO request registry (mm_features do not cross the edge->cloud SO
        # boundary for cached reqs - scheduled_cached_reqs carries only
        # req_ids), so the edge scheduler - which owns self.requests - is the
        # only place this can be computed. The expression mirrors
        # NPUModelRunner.step_has_multimodal_req exactly; self.requests here
        # (at scheduling time) == model_runner.requests at execute time (both
        # reflect this step), so the cloud hint's has_mrope matches the edge
        # sender's include_mrope bit-for-bit (eliminates the mixed-batch
        # mismatch). Dynamic attr; survives the edge->cloud SO pickle
        # (SchedulerOutput has no __slots__ - PassiveScheduler already relies
        # on this for _ARRIVAL_SEQ_ATTR).
        scheduler_output.has_mrope = (
            any(req.mm_features for req in self.requests.values())
            or any(getattr(nr, "mm_features", None)
                   for nr in scheduler_output.scheduled_new_reqs)
        )
        return scheduler_output

    def _log_stall_if_blocked(
        self, scheduler_output: SchedulerOutput
    ) -> None:
        """Rate-limited probe for EMPTY schedules with outstanding work.

        Fires at most once per 10ms.  The snapshot covers every gate that
        can hold back a dispatch: per-lane counters, force flags, queue
        lengths, the placeholder DF state, and the watermark readiness of
        each tail head — so a stall can be attributed to a specific gate
        from a single log line.
        """
        if scheduler_output.batch_type != BatchType.EMPTY:
            return
        if not (
            self.waiting
            or self.chunk_prefill_first
            or self.prefill_last_pending
            or self.running
            or self.prefills_last_ready
            or self.decodes_first_ready
            or self.decodes_last_ready
            or self.prefill_drafts_first_ready
            or self.prefill_drafts_last_ready
            or self.decode_drafts_first_ready
            or self.decode_drafts_last_ready
            or self.prefill_inflight_count > 0
            or self.decode_or_draft_inflight_count > 0
            or self.decode_head_inflight_count > 0
            or self.prefill_draft_remote_pending_count > 0
            or self.decode_draft_remote_pending_count > 0
        ):
            # Genuinely idle: no requests, nothing in flight.
            return
        now = time.monotonic()
        if now - self._last_stall_log_ts < 0.010:
            return
        self._last_stall_log_ts = now
        logger.warning(
            "[PD-STALL] empty schedule with outstanding work: "
            "state=%s waiting=%d chunk_pf=%d pl_pending=%d running=%d | "
            "counters: pf_inflight=%d/%d decode_inflight=%d "
            "decode_head=%d p_pending=%d d_pending=%d | "
            "force: dl=%s pdl=%s ddl=%s | "
            "queues: pl_last=%d df_ph=%d dl_last=%d p_drf=%d p_drl=%d "
            "d_drf=%d d_drl=%d | "
            "tail_ready: pl=%s dl=%s p_drl=%s d_drl=%s | "
            "placeholder_parent=%s publish_pending=%d",
            self._prefill_state(),
            len(self.waiting),
            len(self.chunk_prefill_first),
            len(self.prefill_last_pending),
            len(self.running),
            self.prefill_inflight_count,
            self.prefill_inflight_limit,
            self.decode_or_draft_inflight_count,
            self.decode_head_inflight_count,
            self.prefill_draft_remote_pending_count,
            self.decode_draft_remote_pending_count,
            self._force_decode_last,
            self._force_prefill_draft_last,
            self._force_decode_draft_last,
            len(self.prefills_last_ready),
            len(self.decodes_first_ready),
            len(self.decodes_last_ready),
            len(self.prefill_drafts_first_ready),
            len(self.prefill_drafts_last_ready),
            len(self.decode_drafts_first_ready),
            len(self.decode_drafts_last_ready),
            self._has_actionable_prefill_tail(),
            self._has_actionable_decode_tail(),
            self._has_actionable_prefill_draft_tail(),
            self._has_actionable_decode_draft_tail(),
            self._decode_first_placeholder_parent is not None,
            len(self._draft_publish_pending),
        )

    def _decode_or_draft_first_only_active(self) -> bool:
        started_at = self._decode_or_draft_first_only_start_ts
        if started_at is None:
            return False
        elapsed_ms = (time.monotonic() - started_at) * 1000
        if elapsed_ms >= self._decode_or_draft_first_only_window_ms:
            self._decode_or_draft_first_only_start_ts = None
            return False
        return True

    def _start_decode_or_draft_first_only_window(self) -> None:
        self._decode_or_draft_first_only_start_ts = time.monotonic()

    def _clear_decode_or_draft_first_only_window(self) -> None:
        self._decode_or_draft_first_only_start_ts = None

    def _pick_decode_or_draft_first_only_or_empty(self) -> SchedulerOutput | None:
        if not self._decode_or_draft_first_only_active():
            return None
        # Window: allow Draft首 (decode lane first — the window exists to
        # keep the cloud-side decode stream busy) or Decode首
        if self._can_schedule_decode_draft_first():
            self._clear_decode_or_draft_first_only_window()
            return self._pick_draft_first_batch(prefill_phase=False)
        if self._can_schedule_prefill_draft_first():
            self._clear_decode_or_draft_first_only_window()
            return self._pick_draft_first_batch(prefill_phase=True)
        if self._can_schedule_decode_first():
            self._clear_decode_or_draft_first_only_window()
            return self._pick_decode_first_batch()
        return self._make_empty_batch()

    def _pick_by_state(
        self, state: PrefillState, ready_only: bool = True
    ) -> SchedulerOutput:
        if self._decode_first_placeholder_parent is not None:
            self._prepare_next_decode_first_placeholder(
                self._decode_first_placeholder_parent
            )
        # A placeholder DECODE_FIRST prepared when the final DRAFT_LAST was
        # dispatched pops immediately: it only needs to stay immediately
        # behind its own draft chain in dispatch order.  Its real draft
        # token IDs are scattered from the worker-local
        # _worker_draft_token_ids_by_req entries at _prepare_inputs time;
        # those per-request rows are recorded when the chain's final
        # DRAFT_LAST executes on the worker and are immune to foreign
        # chains overwriting the global _draft_token_ids buffer.  The
        # worker busy_loop executes batches in dispatch (FIFO) order, so
        # popping right behind the chain guarantees the verify's scatter
        # runs after that final DRAFT_LAST has written the fresh rows --
        # no scheduler-side drain wait is needed for correctness.
        #
        # The earlier drain gate (both lanes' remote-pending == 0) dated
        # from execution-time recv posting on the shared DECODE channel,
        # where a DECODE_FIRST in flight with a DRAFT_FIRST could deadlock
        # the two peers on opposite-direction recvs.  With pre-posted,
        # seqno-keyed recvs the data plane pairs by (channel, seqno): the
        # placeholder's seqno is stamped after its chain's reserved range,
        # so sends and recvs pair off in order even while the chain's last
        # DRAFT_LASTs are still in flight -- this is what removes the
        # pre-DF bubble (decode pipeline depth 1 -> 2).
        #
        # Ordering against *queued* draft steps of an interleaved chain is
        # still enforced at creation time (see
        # _prepare_next_decode_first_placeholder): the placeholder is only
        # pre-built once no other draft work remains, because its creation
        # already claims decode_or_draft_inflight/decode_head_inflight.
        # Do NOT gate on decode_or_draft_inflight_count here either: it was
        # incremented by the placeholder's own creation.  A real
        # (non-placeholder) DECODE_FIRST cannot be in flight while a
        # placeholder exists: placeholder creation requires an active
        # pre-generated draft chain, while _can_schedule_decode_first()
        # requires no decode-lane draft work at all.
        if self.decodes_first_ready:
            placeholder = self.decodes_first_ready.popleft()
            # The placeholder was pre-built at chain end without a comm
            # seqno (a drop in the window would have left a hole on the
            # channel); stamp it now, at dispatch time, along with its
            # self-posted DECODE_LAST copy.
            placeholder.comm_seqno = self._decode_comm_seqno
            self._decode_comm_seqno += 1
            self._reserve_draft_seqnos(placeholder)
            for tail in self.decodes_last_ready:
                if tail.head_token == placeholder.head_token:
                    tail.comm_seqno = placeholder.comm_seqno
                    break
            return placeholder

        if ready_only:
            first_only = self._pick_decode_or_draft_first_only_or_empty()
            if first_only is not None:
                return first_only

        # Two-pass readiness semantics: with ready_only=True an unready
        # tail is treated as absent (schedule among data-ready tasks);
        # the fallback pass (ready_only=False, taken only when nothing
        # was ready) dispatches the highest-priority pending tail by the
        # original order — its payload wait is covered device-side by
        # wait_event on the pre-posted recv, never a host block.
        has_pdrl = (
            self._has_actionable_prefill_draft_tail()
            if ready_only
            else bool(self.prefill_drafts_last_ready)
        )
        has_ddrl = (
            self._has_actionable_decode_draft_tail()
            if ready_only
            else bool(self.decode_drafts_last_ready)
        )
        has_dl = (
            self._has_actionable_decode_tail()
            if ready_only
            else bool(self.decodes_last_ready)
        )
        has_pl = (
            self._has_actionable_prefill_tail()
            if ready_only
            else bool(self.prefills_last_ready)
        )

        if state == PrefillState.IDLE:
            # IDLE: P首/chunk0首 > DDraft尾 > PDraft首 > DDraft首
            #       > Decode尾 > PDraft尾 > Decode首 > P尾 > Empty
            if self._can_schedule_prefill_first():
                so = self._pick_prefill_first_batch()
                if so.total_num_scheduled_tokens > 0:
                    return so
                logger.warning(
                    "PREFILL_FIRST returned empty batch (total_num_scheduled_tokens=0). "
                    "This usually means KV cache blocks are exhausted by running decode "
                    "requests. Prefill work will be deferred until resources are freed."
                )
                self.finished_req_ids.update(so.finished_req_ids)
            if has_ddrl:
                return self._pick_draft_last_batch(prefill_phase=False)
            if self._can_schedule_prefill_draft_first():
                return self._pick_draft_first_batch(prefill_phase=True)
            if self._can_schedule_decode_draft_first():
                return self._pick_draft_first_batch(prefill_phase=False)
            # A queued placeholder DECODE_FIRST already self-posted its tail
            # into decodes_last_ready at creation time; while the head is
            # gated above, the tail must not overtake it (the worker would
            # find no suspended HeadState for it).
            if has_dl and not self.decodes_first_ready:
                return self._pick_decode_last_batch()
            # PDraft尾 sits below Decode尾 in every state: the prefill
            # warmup chain is throughput work and must not delay the
            # decode critical path.
            if has_pdrl:
                return self._pick_draft_last_batch(prefill_phase=True)
            if self._can_schedule_decode_first():
                return self._pick_decode_first_batch()
            if has_pl:
                return self._pick_prefill_last_batch(ready_only=ready_only)
            return self._make_empty_batch()

        if state == PrefillState.LOW:
            # LOW: DDraft尾 > PDraft首 > DDraft首 > P首
            #      > Decode尾 > PDraft尾 > Decode首 > P尾 > Empty
            # One prefill is already in flight; the second PF fills the
            # gaps behind draft work (2P pipelining) without delaying
            # decode tails.  Drafts can never starve the PF here: no new
            # prefill chain can form before the PF (PL ranks below it),
            # and the decode lane's next round starts at DL/DF, which
            # also rank below it.
            if has_ddrl:
                return self._pick_draft_last_batch(prefill_phase=False)
            if self._can_schedule_prefill_draft_first():
                return self._pick_draft_first_batch(prefill_phase=True)
            if self._can_schedule_decode_draft_first():
                return self._pick_draft_first_batch(prefill_phase=False)
            if self._can_schedule_prefill_first():
                so = self._pick_prefill_first_batch()
                if so.total_num_scheduled_tokens > 0:
                    return so
                logger.warning(
                    "PREFILL_FIRST returned empty batch (total_num_scheduled_tokens=0). "
                    "This usually means KV cache blocks are exhausted by running decode "
                    "requests. Prefill work will be deferred until resources are freed."
                )
                self.finished_req_ids.update(so.finished_req_ids)
            # Same overtake guard as the IDLE branch above: a queued
            # placeholder DECODE_FIRST's self-posted tail must wait for
            # its head.
            if has_dl and not self.decodes_first_ready:
                return self._pick_decode_last_batch()
            # PDraft尾 sits below Decode尾 in every state (see IDLE).
            if has_pdrl:
                return self._pick_draft_last_batch(prefill_phase=True)
            if self._can_schedule_decode_first():
                return self._pick_decode_first_batch()
            if has_pl:
                return self._pick_prefill_last_batch(ready_only=ready_only)
            return self._make_empty_batch()

        # HIGH (prefill_inflight >= limit): a data-ready P尾 goes first —
        # executing it frees a prefill slot (its update_from_output
        # decrements prefill_inflight_count) and lets the request start its
        # decode/draft lifecycle; with 2P in flight, deferring a ready PL
        # behind draft/decode work stalls the whole prefill pipeline.  The
        # inversion is bounded: at most `prefill_inflight_limit` PLs can be
        # queued, and PL has no dependency on any draft/decode work.
        # Then DDraft尾 > PDraft首 > DDraft首 > D尾 > PDraft尾 > D首 > P尾
        # > Empty — no P首 slot left.
        if self._has_actionable_prefill_tail():
            return self._pick_prefill_last_batch(ready_only=ready_only)
        if has_ddrl:
            return self._pick_draft_last_batch(prefill_phase=False)
        if self._can_schedule_prefill_draft_first():
            return self._pick_draft_first_batch(prefill_phase=True)
        if self._can_schedule_decode_draft_first():
            return self._pick_draft_first_batch(prefill_phase=False)
        # Same overtake guard as the IDLE branch above: a queued placeholder
        # DECODE_FIRST's self-posted tail must wait for its head.
        if has_dl and not self.decodes_first_ready:
            return self._pick_decode_last_batch()
        # PDraft尾 sits below Decode尾 in every state (see IDLE).
        if has_pdrl:
            return self._pick_draft_last_batch(prefill_phase=True)
        if self._can_schedule_decode_first():
            return self._pick_decode_first_batch()
        if has_pl:
            return self._pick_prefill_last_batch(ready_only=ready_only)
        return self._make_empty_batch()

    def is_waiting_for_remote_tail(self) -> bool:
        """True when local requests exist only as remote in-flight work.

        In this state the edge has no local batch to execute until POST_OUT
        returns a PREFILL_LAST/DECODE_LAST, so the EngineCore should yield
        instead of tight-loop scheduling EMPTY batches.
        """
        return bool(
            (
                self.prefill_inflight_count > 0
                or self.decode_or_draft_inflight_count > 0
                or self.prefill_draft_remote_pending_count > 0
                or self.decode_draft_remote_pending_count > 0
            )
            # A PL whose tail payload is still in flight (decoupled-irecv
            # coordination) must not count as local actionable work —
            # otherwise the engine tight-loops EMPTY batches while the
            # data is arriving.  The same applies to DL/DRL tails.
            and not self._has_actionable_prefill_tail()
            and not self._has_actionable_decode_tail()
            and not self._has_actionable_prefill_draft_tail()
            and not self._has_actionable_decode_draft_tail()
            and not self._can_schedule_prefill_first()
            and not self._can_schedule_prefill_draft_first()
            and not self._can_schedule_decode_draft_first()
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
        # When running decode requests already fill max_num_running_reqs,
        # super().schedule() inside _pick_prefill_first_batch will return an
        # empty batch, causing a tight-loop.  Pre-check capacity to avoid
        # the useless call.
        effective_capacity = self.max_num_running_reqs - len(self.running)
        return (
            self._has_prefill_work()
            and self.prefill_inflight_count < self.prefill_inflight_limit
            and effective_capacity > 0
        )

    def _can_schedule_decode_first(self) -> bool:
        # Scoped to the decode lane: prefill-phase draft work travels on the
        # dedicated PREFILL_DRAFT channel pair and must not hold back DF.
        # `not self.decodes_first_ready` restores a key invariant the shared
        # pending counter used to imply: a queued placeholder verify DF must
        # never race a real DECODE_FIRST.  The placeholder now pops on the
        # very next turn after its creation, so the queued window is a single
        # scheduling turn at most — but without this check a real DF picked
        # in that window would double-verify the same drafts and desync the
        # cloud's request-keyed correction bookkeeping.
        return bool(
            self.running
            and self.decode_or_draft_inflight_count == 0
            and self.decode_draft_remote_pending_count == 0
            and not self.decode_drafts_first_ready
            and not self.decode_drafts_last_ready
            and not self.decodes_first_ready
            and not self._force_decode_last
            and not self._force_decode_draft_last
            # A direct (non-placeholder) pick also records num_computed
            # into the SO; it must likewise wait for the previous chain's
            # DECODE_LAST settle or it bakes in the same optimistic phantom.
            # The phantom only exists with speculative decode (its rejection
            # rewind is what makes the pre-settle value optimistic), so gate
            # only then; otherwise this would just delay the next
            # DECODE_FIRST by one settle turn for no benefit.
            and (
                self.num_spec_tokens <= 0
                or self._unsettled_decode_tail_count == 0
            )
        )

    def _can_schedule_prefill_draft_first(self) -> bool:
        if not self.prefill_drafts_first_ready:
            return False
        next_output = self.prefill_drafts_first_ready[0]
        is_pregenerated = (
            next_output.draft_task_id in self._pregenerated_draft_task_ids
        )
        # Prefill-phase chains travel on the dedicated PREFILL_DRAFT channel
        # pair, so no gating on DECODE-stream state (decode_head_inflight /
        # _force_decode_last / decode_or_draft_inflight) is needed here.
        if is_pregenerated:
            # `not self._force_prefill_draft_last` enforces the lane-local
            # DRAFT_FIRST -> DRAFT_LAST alternation invariant; the
            # remote-pending credit bounds intra-chain pipelining: a second
            # DRAFT_FIRST of this lane may be dispatched while the previous
            # DRAFT_LAST is still in flight.
            return bool(
                self.prefill_draft_remote_pending_count
                < self._draft_remote_pending_limit
                and not self.prefill_drafts_last_ready
                and not self._force_prefill_draft_last
            )
        # Dynamic (non-pre-generated) chains: recvs are not pre-posted for
        # every step, so keep the strict serialization — do not start
        # another head while an earlier head is still remote or its tail is
        # ready locally.
        return bool(
            self.prefill_draft_remote_pending_count == 0
            and not self.prefill_drafts_last_ready
            and not self._force_prefill_draft_last
        )

    def _can_schedule_decode_draft_first(self) -> bool:
        if not self.decode_drafts_first_ready:
            return False
        next_output = self.decode_drafts_first_ready[0]
        is_pregenerated = (
            next_output.draft_task_id in self._pregenerated_draft_task_ids
        )
        if is_pregenerated:
            # The edge and cloud workers consume the pre-generated chain in
            # strict FIFO order.  Do not wait for a DRAFT_FIRST result merely
            # to decrement draft_inflight_count; bound the amount of queued
            # work using the remote-pending credit instead.
            # `not self._force_decode_draft_last` enforces the DRAFT_FIRST ->
            # DRAFT_LAST alternation invariant the same way the legacy
            # branch does: once a DRAFT_FIRST is picked, the next draft head
            # may not be picked until its DRAFT_LAST is picked (which clears
            # the flag).  Without it, a DRAFT_LAST dropped/drained without
            # clearing the flag would let a second DRAFT_FIRST through.
            # `decode_head_inflight_count == 0` (not total inflight == 0)
            # gates only on DECODE_FIRST heads: a DRAFT_FIRST is an
            # edge->cloud send while a DRAFT_LAST is a cloud->edge recv, so
            # a second DRAFT_FIRST may be dispatched while the previous
            # DRAFT_LAST is still in flight (draft pipelining).  But
            # DRAFT_FIRST and DECODE_FIRST use different recv primitives on
            # the same stream, so they must never be in flight together --
            # hence gate on decode heads only, allowing draft+draft but not
            # draft+decode.
            return bool(
                self.decode_head_inflight_count == 0
                and self.decode_draft_remote_pending_count
                < self._draft_remote_pending_limit
                and not self.decode_drafts_last_ready
                and not self._force_decode_last
                and not self._force_decode_draft_last
            )

        # Scheduled draft head/tail payloads share the DECODE channel.
        # Do not start another head while an earlier head is still remote or
        # its tail is ready locally: otherwise edge and cloud can each wait
        # for the opposite-direction send before posting the matching recv.
        return bool(
            self.decode_or_draft_inflight_count == 0
            and self.decode_draft_remote_pending_count == 0
            and not self.decode_drafts_last_ready
            and not self._force_decode_last
            and not self._force_decode_draft_last
        )

    def _log_scheduler_state(self, state: PrefillState, batch_type: BatchType) -> None:
        self._step_counter += 1
        if self.chunk_prefill_prior_enable:
            logger.info(
                f"[PD] Step{self._step_counter}, state is {state}, batch_type is {batch_type}, "
                f"waiting[]: {len(self.waiting)}, "
                f"chunk_prefill_first[]: {len(self.chunk_prefill_first)}, "
                f"prefill_last_pending[]: {len(self.prefill_last_pending)}, "
                f"running[]: {len(self.running)}, "
                f"prefills_last_ready[]: {len(self.prefills_last_ready)}, "
                f"prefill_drafts_first_ready[]: {len(self.prefill_drafts_first_ready)}, "
                f"prefill_drafts_last_ready[]: {len(self.prefill_drafts_last_ready)}, "
                f"decode_drafts_first_ready[]: {len(self.decode_drafts_first_ready)}, "
                f"decode_drafts_last_ready[]: {len(self.decode_drafts_last_ready)}, "
                f"decodes_last_ready[]: {len(self.decodes_last_ready)}, "
                f"prefill_inflight: {self.prefill_inflight_count}/{self.prefill_inflight_limit}, "
                f"prefill_draft_remote_pending: {self.prefill_draft_remote_pending_count}, "
                f"decode_draft_remote_pending: {self.decode_draft_remote_pending_count}, "
                f"decode_or_draft_inflight: {self.decode_or_draft_inflight_count}/{self.decode_or_draft_inflight_limit}, "
                f"chunk_flights: {len(self._prefill_flight_by_token)}, "
                f"pending_tails: {self._total_pending_tails()}, "
                f"ahead_chunks: {sum(self._ahead_chunk_count.values())}",
            )
        else:
            logger.info(
                f"[PD] Step{self._step_counter}, state is {state}, batch_type is {batch_type}, "
                f"waiting[]: {len(self.waiting)}, "
                f"chunk_prefill_first[]: {len(self.chunk_prefill_first)}, "
                f"prefill_last_pending[]: {len(self.prefill_last_pending)}, "
                f"running[]: {len(self.running)}, "
                f"prefills_last_ready[]: {len(self.prefills_last_ready)}, "
                f"prefill_drafts_first_ready[]: {len(self.prefill_drafts_first_ready)}, "
                f"prefill_drafts_last_ready[]: {len(self.prefill_drafts_last_ready)}, "
                f"decode_drafts_first_ready[]: {len(self.decode_drafts_first_ready)}, "
                f"decode_drafts_last_ready[]: {len(self.decode_drafts_last_ready)}, "
                f"decodes_last_ready[]: {len(self.decodes_last_ready)}, "
                f"prefill_inflight: {self.prefill_inflight_count}/{self.prefill_inflight_limit}, "
                f"prefill_draft_remote_pending: {self.prefill_draft_remote_pending_count}, "
                f"decode_draft_remote_pending: {self.decode_draft_remote_pending_count}, "
                f"decode_or_draft_inflight: {self.decode_or_draft_inflight_count}/{self.decode_or_draft_inflight_limit}",
            )

    def _pick_prefill_first_batch(self) -> SchedulerOutput:
        saved_running = self.running
        saved_chunk_prefill_first = self.chunk_prefill_first
        saved_max_num_running_reqs = self.max_num_running_reqs

        # Decide what to expose to super().schedule() and the running cap.
        # With chunk_prefill_prior_enable this enforces one-request-per-PF-batch
        # (prevents head_token collision / is_last ambiguity); legacy keeps the
        # original multi-request batching. See _prepare_pf_running_state.
        self.running, self.max_num_running_reqs, rest_candidates = self._prepare_pf_running_state(
            saved_chunk_prefill_first, saved_running, saved_max_num_running_reqs
        )

        if self.limit_prefill_batch_size:
            # 限制每个 P首 batch 最多只包含 1 个请求。
            # 1. chunk_prefill_first 非空时，取第一个请求到 running，
            #    清空 waiting 防止 super().schedule() 再取更多。
            # 2. chunk_prefill_first 为空时，从 waiting 中只保留 1 个请求，
            #    让 super().schedule() 在 waiting 中正常调度（生成 NewRequestData）。
            #    其余 waiting 请求暂存，在 finally 中恢复。
            if saved_chunk_prefill_first:
                self.chunk_prefill_first = list(saved_chunk_prefill_first[1:])
                saved_waiting_rest = []
            else:
                self.chunk_prefill_first = []
                if len(self.waiting) > 0:
                    first_req = self.waiting.pop_request()
                    saved_waiting_rest = list(self.waiting)
                    self.waiting.clear()
                    self.waiting.append(first_req)
                else:
                    saved_waiting_rest = []
        else:
            self.chunk_prefill_first = []
            saved_waiting_rest = []


        # Snapshot num_computed_tokens before super().schedule() so that
        # the is_last computation below uses the pre-schedule value.
        # (super().schedule() → _update_after_schedule increments
        # num_computed_tokens; without the snapshot we would double-count
        # the current chunk's tokens.)
        _num_computed_before: dict[str, int] = {
            req.request_id: req.num_computed_tokens
            for req in self.running
        }

        scheduler_output = None
        try:
            scheduler_output = super().schedule()
        finally:
            self.max_num_running_reqs = saved_max_num_running_reqs

            if scheduler_output is not None:
                if scheduler_output.total_num_scheduled_tokens == 0:
                    scheduler_output.batch_type = BatchType.EMPTY
                    for req in self.running:
                        if req.is_prefill_chunk:
                            self.chunk_prefill_first.append(req)
                        else:
                            self.prefill_last_pending.append(req)
                    # KV cache exhausted: super().schedule() scheduled nothing.
                    # _prepare_pf_running_state swapped self.running for the
                    # prefill candidate(s), so we MUST restore self.running =
                    # saved_running here too (the non-empty branch restores
                    # below).  Without this restore the decode requests that
                    # were in self.running are lost -- _can_schedule_decode_first
                    # never sees them, KV is never freed by finished decodes, and
                    # prefill keeps returning empty -> deadlock.  Also re-prepend
                    # the candidates that were not exposed to super() this round.
                    self.chunk_prefill_first = (
                        rest_candidates + self.chunk_prefill_first
                    )
                    self.running = saved_running
                else:
                    scheduler_output.batch_type = BatchType.PREFILL_FIRST
                    scheduler_output.head_token = uuid4().hex
                    scheduler_output.comm_seqno = self._prefill_comm_seqno
                    self._prefill_comm_seqno += 1
                    self._reserve_draft_seqnos(scheduler_output)
                    self.prefill_inflight_count += 1
                    self._register_pd_flight(scheduler_output)

                    scheduled_req_ids = set(
                        scheduler_output.num_scheduled_tokens.keys()
                    )

                    if self.chunk_prefill_prior_enable:
                        # === Chunk-prefill-prior routing ===
                        # Each scheduled request gets a PerfillChunkFlight.
                        # If the request still has more chunks after this
                        # PF, it may be re-added to chunk_prefill_first
                        # immediately (ahead), allowing the next chunk's PF
                        # before the current chunk's PL returns.
                        for req in self.running:
                            if req.request_id in scheduled_req_ids:
                                num_scheduled = (
                                    scheduler_output.num_scheduled_tokens[
                                        req.request_id
                                    ]
                                )
                                # Use pre-schedule num_computed_tokens
                                # to avoid double-counting the current
                                # chunk's tokens.
                                num_comp_before = _num_computed_before.get(
                                    req.request_id
                                )
                                if num_comp_before is None:
                                    # A newly admitted waiting request was not
                                    # present when the snapshot was taken.
                                    # Upstream discovers its prefix-cache hit
                                    # inside schedule(), then advances
                                    # num_computed_tokens by this chunk before
                                    # returning. Subtract the scheduled suffix
                                    # to recover the true pre-schedule progress.
                                    num_comp_before = (
                                        req.num_computed_tokens
                                        - num_scheduled
                                    )
                                remaining = (
                                    req.num_prompt_tokens
                                    - num_comp_before
                                    - num_scheduled
                                )
                                is_last = remaining <= 0
                                flight = PrefillChunkFlight(
                                    request_id=req.request_id,
                                    head_token=scheduler_output.head_token,
                                    chunk_index=max(0, req.chunk_num - 1),
                                    is_last_chunk=is_last,
                                    num_scheduled_tokens=num_scheduled,
                                )
                                self._prefill_flight_by_token[
                                    scheduler_output.head_token
                                ] = flight
                                self._pending_tail_count[req.request_id] = (
                                    self._pending_tail_count.get(
                                        req.request_id, 0
                                    )
                                    + 1
                                )

                                if self._should_ahead_schedule(req, is_last):
                                    # Ahead: re-add to chunk_prefill_first
                                    # so the next chunk PF can be dispatched
                                    # before this chunk's PL returns. Used for
                                    # the single-request pipeline (both 2P1D
                                    # slots serve the same request).
                                    self.chunk_prefill_first.append(req)
                                    self._ahead_chunk_count[
                                        req.request_id
                                    ] = (
                                        self._ahead_chunk_count.get(
                                            req.request_id, 0
                                        )
                                        + 1
                                    )
                                    logger.info(
                                        "[PD-CHUNK-PRIOR] Ahead-scheduled "
                                        "chunk %d of request %s "
                                        "(head_token=%s, %d tokens, "
                                        "ahead_count=%d)",
                                        flight.chunk_index,
                                        req.request_id,
                                        scheduler_output.head_token,
                                        num_scheduled,
                                        self._ahead_chunk_count[
                                            req.request_id
                                        ],
                                    )
                                else:
                                    # Wait for PL before next chunk. Reasons:
                                    #   - last chunk (is_last);
                                    #   - ahead budget exhausted;
                                    #   - yield: next_prefill_prior_enable is
                                    #     on and another request can fill the
                                    #     slot (cross-request head-prior).
                                    if (
                                        self.next_prefill_prior_enable
                                        and not is_last
                                        and self._can_ahead_schedule(
                                            req.request_id
                                        )
                                        and self._has_other_prefill_request(
                                            req.request_id
                                        )
                                    ):
                                        wait_reason = "yield"
                                    elif is_last:
                                        wait_reason = "last"
                                    else:
                                        wait_reason = "ahead_full"
                                    self.prefill_last_pending.append(req)
                                    logger.info(
                                        "[PD-CHUNK-PRIOR] Chunk %d of "
                                        "request %s waiting for PL "
                                        "(head_token=%s, %d tokens, "
                                        "is_last=%s, "
                                        "pending_tails=%d, reason=%s)",
                                        flight.chunk_index,
                                        req.request_id,
                                        scheduler_output.head_token,
                                        num_scheduled,
                                        is_last,
                                        self._pending_tail_count.get(
                                            req.request_id, 0
                                        ),
                                        wait_reason,
                                    )
                            elif req.is_prefill_chunk:
                                # Not scheduled this round (token budget
                                # exhausted), keep for next round.
                                self.chunk_prefill_first.append(req)
                            else:
                                # Completed but not scheduled – defensive.
                                self.prefill_last_pending.append(req)
                    else:
                        # === Legacy routing (no chunk-prefill-prior) ===
                        for req in self.running:
                            if req.request_id in scheduled_req_ids:
                                self.prefill_last_pending.append(req)
                            elif req.is_prefill_chunk:
                                # Not scheduled this round (e.g. token budget
                                # exhausted), keep in chunk_prefill_first.
                                self.chunk_prefill_first.append(req)
                            else:
                                # Completed but not scheduled – defensive.
                                self.prefill_last_pending.append(req)

                # Restore prefill candidates not exposed to super() this round
                # so each is scheduled in its own PF batch (one-per-batch).
                self.chunk_prefill_first = (
                    rest_candidates + self.chunk_prefill_first
                )
                self.running = saved_running

                # [方案B] Edge 侧建议 Cloud 是否切层。
                # 必须在 self.running 恢复为 saved_running 之后检查，
                # 否则 self.running 被临时替换为 prefill 请求，永远为 False。
                if scheduler_output.total_num_scheduled_tokens > 0:
                    suggest = len(self.running) > 0
                    scheduler_output.cloud_suggest_slicing = suggest
                    if not suggest:
                        logger.info(
                            "[PD-EDGE-NO-SLICE] PREFILL_FIRST "
                            "cloud_suggest_slicing=False, running=%d, "
                            "chunk_prefill_first=%d, total_tokens=%d",
                            len(self.running),
                            len(self.chunk_prefill_first),
                            scheduler_output.total_num_scheduled_tokens,
                        )

            else:
                self.chunk_prefill_first = saved_chunk_prefill_first
                self.running = saved_running

            # 恢复 waiting 中其余请求（仅在 limit_prefill_batch_size 时）
            if self.limit_prefill_batch_size and saved_waiting_rest:
                for req in saved_waiting_rest:
                    self.waiting.append(req)

        if (
            self.limit_prefill_batch_size
            and scheduler_output is not None
            and scheduler_output.total_num_scheduled_tokens > 0
        ):
            logger.info(
                "[PD-LIMIT-PREFILL] PREFILL_FIRST batch_size=%d, "
                "total_tokens=%d, chunk_first_remaining=%d, waiting_remaining=%d",
                len(scheduler_output.num_scheduled_tokens),
                scheduler_output.total_num_scheduled_tokens,
                len(self.chunk_prefill_first),
                len(self.waiting),
            )

        return scheduler_output  # type: ignore[return-value]

    @staticmethod
    def _seqno_ready(channel: CommChannelType, so: SchedulerOutput) -> bool:
        """True once the pre-posted irecv for this batch's payload has
        completed — i.e. the channel's completion watermark covers the
        batch's ``comm_seqno``.  Batches without a comm_seqno carry no
        cross-node traffic and are always ready."""
        seqno = so.comm_seqno
        if seqno is None:
            return True
        return is_irecv_complete(channel, seqno)

    def _prefill_tail_data_ready(self, so: SchedulerOutput) -> bool:
        return self._seqno_ready(CommChannelType.PREFILL_DOWN, so)

    def _has_actionable_prefill_tail(self) -> bool:
        """True when a queued PREFILL_LAST can actually be dispatched now."""
        return bool(self.prefills_last_ready) and self._prefill_tail_data_ready(
            self.prefills_last_ready[0]
        )

    def _decode_tail_data_ready(self, so: SchedulerOutput) -> bool:
        return self._seqno_ready(CommChannelType.DECODE_DOWN, so)

    def _prefill_draft_tail_data_ready(self, so: SchedulerOutput) -> bool:
        return self._seqno_ready(CommChannelType.PREFILL_DRAFT_DOWN, so)

    def _decode_draft_tail_data_ready(self, so: SchedulerOutput) -> bool:
        return self._seqno_ready(CommChannelType.DECODE_DOWN, so)

    def _has_actionable_decode_tail(self) -> bool:
        """True when a queued DECODE_LAST can actually be dispatched now."""
        return bool(self.decodes_last_ready) and self._decode_tail_data_ready(
            self.decodes_last_ready[0]
        )

    def _has_actionable_prefill_draft_tail(self) -> bool:
        """True when a queued prefill-phase DRAFT_LAST can be dispatched."""
        return bool(
            self.prefill_drafts_last_ready
        ) and self._prefill_draft_tail_data_ready(
            self.prefill_drafts_last_ready[0]
        )

    def _has_actionable_decode_draft_tail(self) -> bool:
        """True when a queued decode-phase DRAFT_LAST can be dispatched."""
        return bool(
            self.decode_drafts_last_ready
        ) and self._decode_draft_tail_data_ready(
            self.decode_drafts_last_ready[0]
        )

    def _pick_prefill_last_batch(
        self, ready_only: bool = True
    ) -> SchedulerOutput:
        """Pop one cloud-returned SchedulerOutput from prefills_last_ready.

        The cloud has already rewritten ``batch_type=PREFILL_LAST`` and kept
        all original KV / sampling metadata intact, so the edge worker can
        directly run segment_e + sampler on it. We also remove the involved
        requests from ``chunk_prefill_first`` so the parent class's
        ``update_from_output`` does not double-account them.
        """
        if not self.prefills_last_ready:
            return self._make_empty_batch()
        # Peek before popping: in the ready pass (ready_only=True) a PL
        # whose tail payload has not finished arriving stays queued and
        # this round yields an EMPTY batch instead of blocking the worker
        # on recv.  In HIGH state a data-ready PL is hoisted to the top of
        # the branch by the caller; in every other state it is the
        # lowest-priority pick, so skipping here needs no caller-side
        # handling.  The fallback pass (ready_only=False) dispatches
        # regardless — the payload wait is covered device-side by
        # wait_event on the pre-posted recv.
        if ready_only and not self._prefill_tail_data_ready(
            self.prefills_last_ready[0]
        ):
            return self._make_empty_batch()
        so = self.prefills_last_ready.popleft()
        assert so.batch_type == BatchType.PREFILL_LAST, (
            f"prefills_last_ready expects PREFILL_LAST, got {so.batch_type}"
        )
        # Mark whether this PL is the request's last prefill chunk.  Mid-chunk
        # PL still has to run the drafter so that the MTP layer populates its
        # KV cache for every prompt chunk; its sampled/draft tokens are only
        # discarded after the draft forward.  The flight is still in the map
        # here and is popped later in
        # _update_from_output_prefill_last_chunk_prior.
        flight = (
            self._prefill_flight_by_token.get(so.head_token)
            if so.head_token else None
        )
        batch_req_ids = tuple(so.num_scheduled_tokens)
        if flight is not None:
            draft_output_req_ids = (
                batch_req_ids if flight.is_last_chunk else ()
            )
        else:
            # Legacy mode may batch several prefill requests.  Preserve the
            # per-request distinction: every row warms draft KV, while only
            # requests whose prompt is complete may publish proposals.
            draft_output_req_ids = tuple(
                req_id
                for req_id in batch_req_ids
                if (request := self.requests.get(req_id)) is not None
                and not request.is_prefill_chunk
            )
        so.draft_output_req_ids = draft_output_req_ids
        so.is_last_prefill_chunk = bool(draft_output_req_ids)
        # Drop these reqs from chunk_prefill_first. Keep them in
        # prefill_last_pending until update_from_output() moves them to running.
        last_req_ids = set(so.num_scheduled_tokens.keys())
        if last_req_ids:
            self.chunk_prefill_first = [
                req for req in self.chunk_prefill_first
                if req.request_id not in last_req_ids
            ]
        # Every chunk needs a draft-prefill pass.  Only the last chunk's draft
        # tokens are published to the following target verify batch; the
        # worker uses draft_output_req_ids to discard mid-chunk outputs.
        self._pregenerate_draft_chain(so)
        return so

    def _pick_draft_first_batch(self, prefill_phase: bool) -> SchedulerOutput:
        if prefill_phase:
            first_ready = self.prefill_drafts_first_ready
            last_ready = self.prefill_drafts_last_ready
        else:
            first_ready = self.decode_drafts_first_ready
            last_ready = self.decode_drafts_last_ready
        while first_ready:
            scheduler_output = first_ready.popleft()
            if self._is_stale_draft_output(scheduler_output):
                # Never drop a stale DRAFT_FIRST: its comm seqno was
                # reserved and its recvs were pre-posted when the parent
                # batch was published, so dropping it would leave a hole
                # on the channel and stall every later chain.  Drain the
                # chain with dummy payloads instead (draft_chain_dead):
                # the draft context is gone, but zeros need no context.
                scheduler_output.draft_chain_dead = True
                if scheduler_output.draft_task_id:
                    self._dead_draft_task_ids.add(
                        scheduler_output.draft_task_id
                    )
                    # Its step-0 cloud control may sit in the deferred
                    # pre-out queue waiting for scalars that a dead parent
                    # never produces — release it (engine core drains this
                    # list and publishes immediately).
                    self._dead_chain_publish_to_release.append(
                        scheduler_output.draft_task_id
                    )
                task_id = scheduler_output.draft_task_id
                if (
                    task_id is not None
                    and self._draft_publish_pending.get(task_id)
                    is scheduler_output
                ):
                    self._draft_publish_pending.pop(task_id, None)
                    self._draft_publish_scalars_patched.discard(task_id)
                    self._draft_publish_dispatched.discard(task_id)
                logger.info(
                    "[PD] stale DRAFT_FIRST drains as dummy: task_id=%s "
                    "step=%s",
                    scheduler_output.draft_task_id,
                    scheduler_output.draft_step_idx,
                )
            break
        else:
            return self._make_empty_batch()

        task_id = scheduler_output.draft_task_id
        if (
            task_id is not None
            and self._draft_publish_pending.get(task_id) is scheduler_output
        ):
            self._draft_publish_dispatched.add(task_id)
            if task_id in self._draft_publish_scalars_patched:
                self._draft_publish_pending.pop(task_id, None)
                self._draft_publish_scalars_patched.discard(task_id)

        scheduler_output.batch_type = (
            BatchType.PREFILL_DRAFT_FIRST
            if prefill_phase
            else BatchType.DECODE_DRAFT_FIRST
        )
        if scheduler_output.head_token is None:
            scheduler_output.head_token = uuid4().hex
        # Dead-chain drain: every remaining step of a chain whose requests
        # all finished executes with dummy payloads (see the stale branch
        # above and _drop_stale_drafts_for_req_ids).
        if scheduler_output.draft_task_id in self._dead_draft_task_ids:
            scheduler_output.draft_chain_dead = True
        # The chain's seqno range was reserved at parent PF/DF pick time
        # (draft_seqno_base on the parent SO); steps consume
        # base + draft_step_idx so the pre-posted recv requests match.
        step_idx = int(scheduler_output.draft_step_idx or 0)
        base = (
            self._reserved_draft_seqno_base.get(task_id)
            if task_id
            else None
        )
        if base is not None:
            scheduler_output.comm_seqno = base + step_idx
            if step_idx >= self.num_spec_tokens - 1:
                self._reserved_draft_seqno_base.pop(task_id, None)
        elif scheduler_output.draft_prefill_phase:
            scheduler_output.comm_seqno = self._prefill_draft_comm_seqno
            self._prefill_draft_comm_seqno += 1
        else:
            scheduler_output.comm_seqno = self._decode_comm_seqno
            self._decode_comm_seqno += 1
        # Draft-first self-posting mirrors the decode path below. Every
        # field needed by DRAFT_LAST is already known on the edge; the cloud
        # used to echo the same SchedulerOutput back through POST_OUT only
        # after its worker acked DRAFT_FIRST. Pre-generating the tail here
        # removes that per-draft-step control-plane round trip and lets the
        # edge post the matching receive as soon as DRAFT_FIRST has completed
        # locally. Worker FIFO ordering plus the per-channel send-work wait
        # preserves DRAFT_FIRST -> DRAFT_LAST data-plane ordering.
        draft_last = replace(
            scheduler_output,
            batch_type=(
                BatchType.PREFILL_DRAFT_LAST
                if prefill_phase
                else BatchType.DECODE_DRAFT_LAST
            ),
            num_accepted_tokens=None,
            valid_sampled_token_count=None,
        )
        # is_last_prefill_chunk is a downstream dynamic SchedulerOutput
        # attribute, so dataclasses.replace() does not preserve it.
        draft_last.is_last_prefill_chunk = getattr(
            scheduler_output, "is_last_prefill_chunk", True
        )
        draft_last.draft_output_req_ids = getattr(
            scheduler_output,
            "draft_output_req_ids",
            tuple(scheduler_output.num_scheduled_tokens),
        )
        self._register_pd_flight(scheduler_output)
        last_ready.append(draft_last)
        if prefill_phase:
            # Prefill-phase drafts use the dedicated PREFILL_DRAFT stream
            # and never touch the decode-lane in-flight counter.
            self.prefill_draft_remote_pending_count += 1
            self._force_prefill_draft_last = True
        else:
            self.decode_or_draft_inflight_count += 1
            self.decode_draft_remote_pending_count += 1
            self._force_decode_draft_last = True

        logger.info(
            "[MTP-DEBUG] scheduler picked DRAFT_FIRST: task_id=%s, "
            "parent_req_id=%s, draft_step_idx=%s, head_token=%s, "
            "prefill_phase=%s, remaining_ready=%d, "
            "decode_or_draft_inflight=%d, prefill_draft_remote_pending=%d, "
            "decode_draft_remote_pending=%d",
            scheduler_output.draft_task_id,
            scheduler_output.parent_req_id,
            scheduler_output.draft_step_idx,
            scheduler_output.head_token,
            prefill_phase,
            len(first_ready),
            self.decode_or_draft_inflight_count,
            self.prefill_draft_remote_pending_count,
            self.decode_draft_remote_pending_count,
        )
        return scheduler_output

    def _reserve_draft_seqnos(
        self, scheduler_output: SchedulerOutput
    ) -> None:
        """Reserve the comm-seqno range for the draft chain that follows
        this PF/DF batch.

        The chain does not exist yet at FIRST pick time, but the
        scheduling invariants (no decode-channel consumer may interleave
        an active/queued draft chain) guarantee its steps will occupy
        exactly ``[base, base + num_spec_tokens)`` on the phase's channel
        pair.  Reserving now lets both peers pre-post all n draft recv
        requests the moment the parent is published, via
        ``draft_seqno_base`` on the SO.
        """
        if not self._check_scheduled_edge_cloud_draft():
            return
        if self.num_spec_tokens <= 0:
            return
        head_token = scheduler_output.head_token
        if not head_token:
            return
        if scheduler_output.batch_type == BatchType.PREFILL_FIRST:
            base = self._prefill_draft_comm_seqno
            self._prefill_draft_comm_seqno += self.num_spec_tokens
        else:
            base = self._decode_comm_seqno
            self._decode_comm_seqno += self.num_spec_tokens
        self._reserved_draft_seqno_base[head_token] = base
        scheduler_output.draft_seqno_base = base

    def _uses_async_scheduled_mtp_placeholders(self) -> bool:
        """Whether scheduled MTP can use native async placeholder semantics."""
        if not getattr(self.scheduler_config, "async_scheduling", False):
            return False
        speculative_config = self.vllm_config.speculative_config
        if speculative_config is None or self.num_spec_tokens <= 0:
            return False
        method = getattr(speculative_config, "method", None)
        if method in ("qwen3_5_mtp", "qwen_mtp"):
            return True
        if method != "mtp":
            return False
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        model_type = str(getattr(hf_config, "model_type", "")).lower()
        return "qwen" in model_type or model_type == "deepseek_v4"

    def _pregenerate_draft_chain(
        self, target_tail: SchedulerOutput
    ) -> None:
        """Create fixed-length placeholder DRF tasks at target-tail pick time.

        The real token IDs remain in the edge worker.  FIFO ordering ensures
        every DRF executes after the DRL that produces its local inputs.  Only
        the step-0 accepted-token scalars are finalized later for the cloud;
        they are not consumed by the edge worker.
        """
        if not self._uses_async_scheduled_mtp_placeholders():
            return
        # Per-lane guard: a new chain is pre-generated only when its OWN
        # lane is idle.  The other lane (different channel pair, different
        # seqno space) does not block pre-generation.
        prefill_phase = target_tail.batch_type == BatchType.PREFILL_LAST
        if prefill_phase:
            first_ready = self.prefill_drafts_first_ready
            last_ready = self.prefill_drafts_last_ready
        else:
            first_ready = self.decode_drafts_first_ready
            last_ready = self.decode_drafts_last_ready
        if first_ready or last_ready:
            logger.info(
                "[PD] skip pre-generation (lane busy), chain goes dynamic: "
                "task_id=%s prefill_phase=%s lane_first=%d lane_last=%d",
                target_tail.head_token,
                prefill_phase,
                len(first_ready),
                len(last_ready),
            )
            return
        if any(
            bool(so.draft_prefill_phase) == prefill_phase
            for so in self._draft_publish_pending.values()
        ):
            logger.info(
                "[PD] skip pre-generation (step-0 publish pending), chain "
                "goes dynamic: task_id=%s prefill_phase=%s",
                target_tail.head_token,
                prefill_phase,
            )
            return
        req_ids = list(target_tail.num_scheduled_tokens)
        if not req_ids:
            return
        task_id = target_tail.head_token
        if not task_id:
            return
        # Requests genuinely gone (aborted before the parent tail was
        # processed): the chain can never produce real payloads, but its
        # reserved seqnos and pre-posted recvs must still be consumed —
        # pre-generate the chain as dead; the worker drains it with dummy
        # payloads.
        chain_dead = any(req_id not in self.requests for req_id in req_ids)
        if chain_dead:
            self._dead_draft_task_ids.add(task_id)

        for step_idx in range(self.num_spec_tokens):
            draft_first = replace(
                target_tail,
                batch_type=(
                    BatchType.PREFILL_DRAFT_FIRST
                    if prefill_phase
                    else BatchType.DECODE_DRAFT_FIRST
                ),
                head_token=None,
                parent_req_id=req_ids[0],
                draft_task_id=task_id,
                draft_step_idx=step_idx,
                draft_prefill_phase=prefill_phase,
                draft_chain_dead=chain_dead,
                num_accepted_tokens=None,
                valid_sampled_token_count=None,
            )
            # Preserve the parent prefill phase across the independently
            # scheduled draft chain.  Mid-chunk chains warm the draft KV but
            # must not seed a target decode placeholder.
            draft_first.is_last_prefill_chunk = getattr(
                target_tail, "is_last_prefill_chunk", True
            )
            draft_first.draft_output_req_ids = getattr(
                target_tail,
                "draft_output_req_ids",
                tuple(target_tail.num_scheduled_tokens),
            )
            # The wire copy of every chain step would carry the parent's
            # full token history: replace() shares the parent's
            # CachedRequestData, so each pickled DRAFT_FIRST re-sends the
            # whole all_token_ids dict (MBs for long contexts) -- 3-4x the
            # same history per decode round over the edge->cloud ZMQ. No
            # draft-step consumer reads it: the cloud draft path never
            # calls _update_states, and the recovery path that does read
            # all_token_ids only fires for requests re-joining a
            # persistent batch, which never happens inside a chain. Ship
            # an empty copy instead of mutating the shared original.
            _cached = draft_first.scheduled_cached_reqs
            if _cached is not None and _cached.all_token_ids:
                draft_first.scheduled_cached_reqs = replace(
                    _cached, all_token_ids={}
                )
            first_ready.append(draft_first)
            if step_idx == 0 and not chain_dead:
                self._draft_publish_pending[task_id] = draft_first
                self._draft_publish_scalars_patched.discard(task_id)
                self._draft_publish_dispatched.discard(task_id)

        self._pregenerated_draft_task_ids.add(task_id)
        self._pregenerated_draft_req_ids[task_id] = set(req_ids)
        logger.info(
            "[PD] pre-generated async MTP placeholders task_id=%s steps=%d",
            task_id,
            self.num_spec_tokens,
        )

    def finalize_pre_generated_draft_first(
        self,
        *,
        draft_task_id: str,
        num_accepted_tokens: list[int] | None,
        valid_sampled_token_count: list[int] | None,
    ) -> SchedulerOutput | None:
        """Patch cloud-only sampling state into the queued step-0 control."""
        pending = self._draft_publish_pending.get(draft_task_id)
        if pending is None:
            return None
        pending.num_accepted_tokens = num_accepted_tokens
        pending.valid_sampled_token_count = valid_sampled_token_count
        if draft_task_id not in self._draft_publish_dispatched:
            self._draft_publish_scalars_patched.add(draft_task_id)
            return pending
        self._draft_publish_pending.pop(draft_task_id, None)
        self._draft_publish_scalars_patched.discard(draft_task_id)
        return pending

    def is_pre_generated_draft(
        self, scheduler_output: SchedulerOutput
    ) -> bool:
        task_id = (
            scheduler_output.draft_task_id
            or scheduler_output.head_token
        )
        return bool(task_id and task_id in self._pregenerated_draft_task_ids)

    def active_pre_generated_draft_req_ids(self) -> set[str]:
        active: set[str] = set()
        for task_id in self._pregenerated_draft_task_ids:
            active.update(
                self._pregenerated_draft_req_ids.get(task_id, ())
            )
        return active

    def enqueue_draft_first(
        self,
        source: SchedulerOutput,
        *,
        draft_task_id: str,
        draft_step_idx: int,
        num_accepted_tokens: list[int] | None = None,
        valid_sampled_token_count: list[int] | None = None,
    ) -> bool:
        """Generate a draft head locally, mirroring decode head generation.

        The worker owns the mutable draft tensors, but the scheduler owns all
        draft control-plane SchedulerOutputs. The initial step receives only
        the rejection-corrected scalar state from the worker; follow-up steps
        are derived directly from the completed DRAFT_LAST.
        """
        req_ids = list(source.num_scheduled_tokens)
        if not req_ids:
            return False
        # With KV retention, finished requests stay in self.requests until
        # their draft chain releases them, so requests being genuinely gone
        # (e.g. aborted before the parent output was processed) means the
        # chain can never produce real payloads.  Enqueue the step as a
        # dead-chain dummy anyway: its comm seqno is reserved and its
        # recvs pre-posted, and the worker drains it with zeros.
        chain_dead = any(req_id not in self.requests for req_id in req_ids)
        if chain_dead and draft_task_id:
            self._dead_draft_task_ids.add(draft_task_id)

        draft_first = replace(
            source,
            batch_type=(
                BatchType.PREFILL_DRAFT_FIRST
                if (
                    source.draft_prefill_phase
                    if source.batch_type in _DRAFT_LAST_TYPES
                    else source.batch_type == BatchType.PREFILL_LAST
                )
                else BatchType.DECODE_DRAFT_FIRST
            ),
            head_token=None,
            parent_req_id=req_ids[0],
            draft_task_id=draft_task_id,
            draft_step_idx=draft_step_idx,
            # Chain continuations inherit the phase from the previous
            # DRAFT_LAST; chain starts derive it from the parent tail.
            draft_prefill_phase=(
                source.draft_prefill_phase
                if source.batch_type in _DRAFT_LAST_TYPES
                else source.batch_type == BatchType.PREFILL_LAST
            ),
            draft_chain_dead=chain_dead,
            num_accepted_tokens=num_accepted_tokens,
            valid_sampled_token_count=valid_sampled_token_count,
        )
        draft_first.is_last_prefill_chunk = getattr(
            source, "is_last_prefill_chunk", True
        )
        draft_first.draft_output_req_ids = getattr(
            source,
            "draft_output_req_ids",
            tuple(source.num_scheduled_tokens),
        )
        # Same wire-size trim as the pre-generated chain: drop the shared
        # parent token history before this DRAFT_FIRST crosses the
        # edge->cloud ZMQ (no draft-step consumer reads all_token_ids).
        _cached = draft_first.scheduled_cached_reqs
        if _cached is not None and _cached.all_token_ids:
            draft_first.scheduled_cached_reqs = replace(
                _cached, all_token_ids={}
            )
        if draft_first.draft_prefill_phase:
            self.prefill_drafts_first_ready.append(draft_first)
        else:
            self.decode_drafts_first_ready.append(draft_first)
        if (
            draft_step_idx == 0
            and draft_task_id not in self._pregenerated_draft_task_ids
        ):
            # A step-0 enqueue for a chain that was never pre-generated
            # means the dynamic fallback path (pregenerated chains take
            # finalize_pre_generated_draft_first instead).  Each of its
            # steps costs a full engine round trip.
            logger.info(
                "[PD] dynamic draft chain start: task_id=%s "
                "prefill_phase=%s parent_batch=%s chain_dead=%s",
                draft_task_id,
                bool(draft_first.draft_prefill_phase),
                source.batch_type,
                chain_dead,
            )
        return True

    def _enqueue_next_draft_first(
        self, draft_last: SchedulerOutput
    ) -> bool:
        draft_step_idx = int(draft_last.draft_step_idx or 0)
        next_step_idx = draft_step_idx + 1
        task_id = draft_last.draft_task_id
        if task_id in self._pregenerated_draft_task_ids:
            if next_step_idx >= self.num_spec_tokens:
                self._pregenerated_draft_task_ids.discard(task_id)
                self._pregenerated_draft_req_ids.pop(task_id, None)
            return False
        if next_step_idx >= self.num_spec_tokens:
            return False
        if task_id is None:
            raise RuntimeError("DRAFT_LAST missing draft_task_id")
        return self.enqueue_draft_first(
            draft_last,
            draft_task_id=task_id,
            draft_step_idx=next_step_idx,
        )

    def _pick_draft_last_batch(self, prefill_phase: bool) -> SchedulerOutput:
        last_ready = (
            self.prefill_drafts_last_ready
            if prefill_phase
            else self.decode_drafts_last_ready
        )
        while last_ready:
            scheduler_output = last_ready.popleft()
            if scheduler_output.batch_type not in _DRAFT_LAST_TYPES:
                raise RuntimeError(
                    "drafts_last_ready expects DRAFT_LAST, got "
                    f"{scheduler_output.batch_type}"
                )
            # A DRAFT_LAST here always has its DRAFT_FIRST already dispatched
            # to the cloud (it was self-posted in _pick_draft_first_batch when
            # the head was picked).  The cloud does not track request
            # lifecycle, so it will isend a response even if the owning
            # request has since finished/aborted.  The edge MUST still execute
            # this tail (recv) to keep the DECODE hidden channel paired --
            # never drop it.  When the request is gone the worker drains the
            # recv and skips the tail-segment compute (see
            # _run_edge_cloud_draft_last_segment); we also must not spawn a
            # verify placeholder for a dead request.
            if prefill_phase:
                self._force_prefill_draft_last = False
            else:
                self._force_decode_draft_last = False
            self._start_decode_or_draft_first_only_window()
            output_req_ids = getattr(
                scheduler_output,
                "draft_output_req_ids",
                tuple(scheduler_output.num_scheduled_tokens),
            )
            has_live_output_req = any(
                (request := self.requests.get(req_id)) is not None
                and not request.is_finished()
                for req_id in output_req_ids
            )
            if has_live_output_req:
                self._prepare_next_decode_first_placeholder(scheduler_output)
            elif not output_req_ids:
                logger.info(
                    "[PD] finish DRAFT_LAST task_id=%s step=%s "
                    "(mid-prefill KV warmup; no verify placeholder)",
                    scheduler_output.draft_task_id,
                    scheduler_output.draft_step_idx,
                )
            else:
                logger.info(
                    "[PD] drain DRAFT_LAST task_id=%s step=%s "
                    "(request gone; worker will drain cloud response)",
                    scheduler_output.draft_task_id,
                    scheduler_output.draft_step_idx,
                )
            return scheduler_output
        return self._make_empty_batch()

    def _prepare_next_decode_first_placeholder(
        self, draft_last: SchedulerOutput
    ) -> None:
        """Prepare the next target verify batch behind the final draft tail.

        Scheduler-side spec token values are placeholders.  The edge worker
        replaces them with its local ``_draft_token_ids`` after this DRL has
        executed, so no DraftTokenIds round-trip through EngineCore is needed.
        """
        if not self._uses_async_scheduled_mtp_placeholders():
            return
        if draft_last.draft_task_id not in self._pregenerated_draft_task_ids:
            self._decode_first_placeholder_parent = None
            return
        step_idx = int(draft_last.draft_step_idx or 0)
        if step_idx + 1 < self.num_spec_tokens:
            return
        self._decode_first_placeholder_parent = draft_last
        if self.decodes_first_ready:
            self._decode_first_placeholder_parent = None
            return
        if not self.running:
            # A chain pre-generated from the final PREFILL_LAST can reach its
            # final DRL before EngineCore has applied the prefill result. Retry
            # on the next schedule turn after that request moves to running.
            return
        if (
            self._force_decode_last
            or self.decode_or_draft_inflight_count > 0
            or self.decode_head_inflight_count > 0
        ):
            # A real DECODE_FIRST is still in flight: picked earlier (the
            # draft-lane split lets a real DF go out while a prefill-phase
            # chain is active), but its DL has not been picked or its
            # update_from_output has not landed yet.  Creating the
            # placeholder now would claim the decode counters a second time
            # and its immediate pop would dispatch a second verify behind
            # the in-flight one — both verify the same scheduler-side spec
            # state, and the cloud's request-keyed corrections are consumed
            # by the first, so the second fails with "missing request-keyed
            # speculative corrections".  Defer; the parent stays set and a
            # later turn retries once the in-flight DF has fully drained
            # (its DL pick clears _force_decode_last, its update clears the
            # counters).
            return
        if (
            self.prefill_drafts_first_ready
            or self.prefill_drafts_last_ready
            or self.decode_drafts_first_ready
            or self.decode_drafts_last_ready
            or self.prefill_draft_remote_pending_count > 1
            or self.decode_draft_remote_pending_count > 1
        ):
            # Another draft chain still has steps queued or a head in flight
            # behind this tail (chains interleave when a newly admitted
            # request's prefill-warmup chain overlaps the decode chain).
            # Pre-building the placeholder now would dispatch the verify
            # before that chain has produced its drafts -- the verify's spec
            # rows would be filled from stale/crossed worker-side entries,
            # permanently poisoning the affected requests' draft KV
            # (observed: zero-valid verify rounds, then [0,0,0] drafts).
            # Deferring is also required for liveness: creation claims
            # decode_or_draft_inflight/decode_head_inflight, which the queued
            # draft picks wait on, so creating now would deadlock.  Keep the
            # parent set so a later turn retries once the queues drain; if
            # the other chain was not pre-generated, its own final DRL clears
            # the parent above and the normal _can_schedule_decode_first()
            # path schedules the verify instead.
            return
        if self._unsettled_decode_tail_count > 0:
            # The parent chain's DECODE_LAST output has not been settled
            # through update_from_output yet (the fill loop schedules several
            # batches per EngineCore turn but processes one output per turn).
            # Building the placeholder now would record pre-settle optimistic
            # num_computed values; on a rejection step that bakes a phantom
            # +1 into this and every later SO for the member requests.  Keep
            # the parent set so a later turn retries after the settle lands.
            return

        next_decode = self._pick_decode_first_batch(defer_seqno=True)
        if (
            next_decode is not None
            and next_decode.batch_type == BatchType.DECODE_FIRST
            and next_decode.total_num_scheduled_tokens > 0
        ):
            self.decodes_first_ready.append(next_decode)
            self._decode_first_placeholder_parent = None

    @staticmethod
    def _scheduler_output_intersects_req_ids(
        scheduler_output: SchedulerOutput, req_ids: set[str]
    ) -> bool:
        if scheduler_output.parent_req_id in req_ids:
            return True
        return any(
            req_id in req_ids
            for req_id in scheduler_output.num_scheduled_tokens
        )

    # ------------------------------------------------------------------ #
    # Edge-cloud deferred-draft KV retention                              #
    # ------------------------------------------------------------------ #
    def _check_scheduled_edge_cloud_draft(self) -> bool:
        """Mirror of the runner/engine-core scheduled edge-cloud draft
        predicate.  Gates the retention logic so schedulers without the
        edge-cloud draft methods behave exactly as upstream.
        """
        speculative_config = self.vllm_config.speculative_config
        if speculative_config is None:
            return False
        if not getattr(
            self.vllm_config.parallel_config, "enable_edge_cloud", False
        ):
            return False
        method = getattr(speculative_config, "method", None)
        if method == "eagle3":
            return True
        if method in ("qwen3_5_mtp", "qwen_mtp"):
            return True
        if method != "mtp":
            return False
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        model_type = str(getattr(hf_config, "model_type", "")).lower()
        return "qwen" in model_type or model_type == "deepseek_v4"

    def register_edge_cloud_draft_task(
        self, task_id: str, req_ids: set[str]
    ) -> None:
        """Register a deferred draft before its parent output is applied."""
        if (
            not self._edge_cloud_draft_retention_enabled
            or not task_id
            or not req_ids
        ):
            return
        previous_req_ids = self._edge_cloud_draft_task_reqs.get(task_id)
        if previous_req_ids is not None:
            if previous_req_ids != req_ids:
                raise RuntimeError(
                    "Deferred draft task registration changed request set: "
                    f"task_id={task_id}, previous={previous_req_ids}, "
                    f"new={req_ids}"
                )
            return
        self._edge_cloud_draft_task_reqs[task_id] = set(req_ids)
        for req_id in req_ids:
            self._edge_cloud_draft_req_tasks.setdefault(req_id, set()).add(
                task_id
            )

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        task_ids = (
            self._edge_cloud_draft_req_tasks.get(request.request_id)
            if self._edge_cloud_draft_retention_enabled
            else None
        )
        if not task_ids:
            return super()._free_request(request, delay_free_blocks)
        # The request is referenced by a pending/in-flight deferred draft.
        # Finish it normally (outputs, finished_req_ids) but keep its KV
        # blocks until the draft chain completes on the cloud — same
        # ordering guarantee the non-edge-cloud path gets for free by
        # proposing drafts inside execute_model.
        for task_id in task_ids:
            self._draft_retained_requests.setdefault(task_id, {})[
                request.request_id
            ] = request
        return super()._free_request(request, delay_free_blocks=True)

    def release_draft_retained_blocks(self, task_id: str) -> None:
        """Free KV blocks retained for a completed/dropped draft task.

        Idempotent.  A request referenced by several in-flight draft
        tasks is only freed once its last referencing task releases.
        """
        task_req_ids = self._edge_cloud_draft_task_reqs.pop(task_id, set())
        retained = self._draft_retained_requests.pop(task_id, {})
        for req_id in task_req_ids:
            req_tasks = self._edge_cloud_draft_req_tasks.get(req_id)
            if req_tasks is not None:
                req_tasks.discard(task_id)
                if req_tasks:
                    # Still referenced by another in-flight draft task.
                    continue
                self._edge_cloud_draft_req_tasks.pop(req_id, None)
            # The request is no longer referenced by any in-flight chain.
            # If its finish was withheld from the cloud (see
            # filter_cloud_finished_req_ids), queue it for re-emission on
            # the next cloud-bound batch so the cloud runner finally
            # removes the row.
            if req_id in self._cloud_withheld_finished_req_ids:
                self._cloud_withheld_finished_req_ids.discard(req_id)
                self._cloud_released_finished_req_ids.add(req_id)
            request = retained.get(req_id)
            if request is None:
                continue
            current = self.requests.get(req_id)
            if current is request and request.is_finished():
                self._free_blocks(request)
            elif current is not None and current is not request:
                # req_id resubmitted after finishing: free the old
                # request's blocks without evicting the new entry.
                self.kv_cache_manager.free(request)

    def _scheduler_output_all_requests_finished(
        self, scheduler_output: SchedulerOutput
    ) -> bool:
        batch_req_ids = self._edge_cloud_draft_task_reqs.get(
            scheduler_output.draft_task_id or ""
        )
        if batch_req_ids is None:
            batch_req_ids = set(scheduler_output.num_scheduled_tokens)
        else:
            batch_req_ids = set(batch_req_ids)
        if scheduler_output.parent_req_id is not None:
            batch_req_ids.add(scheduler_output.parent_req_id)
        return bool(batch_req_ids) and all(
            (request := self.requests.get(req_id)) is None
            or request.is_finished()
            for req_id in batch_req_ids
        )

    def _drop_placeholder_decode_first(
        self,
        decode_first: SchedulerOutput,
        req_ids: set[str],
    ) -> None:
        """Discard a queued placeholder DECODE_FIRST and its self-posted tail.

        The tail was appended to decodes_last_ready at creation time while
        the head was never dispatched, so no worker ever suspended a
        HeadState for this head_token.  Both halves must vanish together,
        along with the scheduling credits claimed at creation time; keeping
        any of them either crashes the worker (orphaned tail) or deadlocks
        future DECODE_FIRST/DRAFT_FIRST picks (stranded inflight credits or
        a stranded _force_decode_last gate).
        """
        head_token = decode_first.head_token
        tail: SchedulerOutput | None = None
        kept_decodes_last: deque[SchedulerOutput] = deque()
        for output in self.decodes_last_ready:
            if tail is None and output.head_token == head_token:
                tail = output
                continue
            kept_decodes_last.append(output)
        gone = {
            rid for rid in decode_first.num_scheduled_tokens if rid in req_ids
        }
        # Roll back the schedule-time mutations made when the placeholder
        # was created (_pick_decode_first_batch -> super().schedule()):
        # _update_after_schedule advanced every member request as if this
        # verify step would really run (num_computed_tokens += num_scheduled,
        # plus 1 + num_spec async output placeholders).  The batch is never
        # dispatched, so no update_from_output will ever settle these
        # counters -- and rejection rewinds are RELATIVE (-= num_rejected),
        # so the phantom advance survives later settlements and every
        # subsequent DECODE_FIRST carries num_computed_tokens that run ahead
        # of the edge worker's verified state (observed on the cloud as
        # "cpu-state mismatch ... optimistic=actual=N, cpu=N+2" for every
        # member request, followed by the missing-corrections crash).
        # KV blocks allocated for the phantom step are intentionally NOT
        # freed: the same positions are re-scheduled by the next real
        # DECODE_FIRST, so the already-appended blocks are used, not leaked.
        scheduled_spec_tokens = (
            decode_first.scheduled_spec_decode_tokens or {}
        )
        for req_id, num_scheduled in decode_first.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            request.num_computed_tokens -= num_scheduled
            num_spec = len(scheduled_spec_tokens.get(req_id, ()))
            request.num_output_placeholders = max(
                0, request.num_output_placeholders - (1 + num_spec)
            )
        if tail is None:
            logger.warning(
                "[PD] placeholder DECODE_FIRST head_token=%s dropped but no "
                "self-posted DECODE_LAST was queued; skipping paired cleanup "
                "(%d member request(s) gone)",
                head_token,
                len(gone),
            )
            return
        self.decodes_last_ready = kept_decodes_last
        if self.decode_or_draft_inflight_count > 0:
            self.decode_or_draft_inflight_count -= 1
        if self.decode_head_inflight_count > 0:
            self.decode_head_inflight_count -= 1
        # The flight was registered for the head at creation; completing it
        # with the self-posted tail copy unprotects the member requests.
        self._complete_pd_flight(tail)
        # The dropped tail never dispatches, so no settle will ever arrive
        # for it; release the unsettled-tail credit it held.
        if self._unsettled_decode_tail_count > 0:
            self._unsettled_decode_tail_count -= 1
        # The dropped head held the decode-last gate and delay timer; its
        # tail will never be picked to clear them (see the _force_draft_last
        # deadlock note below).  Placeholder creation requires all prior
        # decode work drained, so this head is the sole gate holder.
        self._force_decode_last = False
        self._decode_last_delay_start_ts = None
        logger.info(
            "[PD] drop placeholder DECODE_FIRST head_token=%s together with "
            "its self-posted DECODE_LAST (%d member request(s) gone; the "
            "head was never dispatched, so the tail has no suspended "
            "HeadState to resume)",
            head_token,
            len(gone),
        )

    def _drop_stale_drafts_for_req_ids(self, req_ids: set[str]) -> None:
        if not req_ids:
            return
        # Aligned with the model runner's deferred-draft policy: a draft
        # batch is dropped only when EVERY request it covers has finished.
        # Partial finishes keep the draft — the cloud-side cached
        # attention metadata is whole-batch and cannot be re-sliced.
        # Dropped task ids are reported to the runner (which may still
        # hold the enqueued context) via take_dropped_draft_task_ids().
        # Never drop queued DRAFT_FIRSTs of finished requests: their comm
        # seqnos were reserved and their recvs pre-posted at parent publish
        # time, so a drop would leave holes on the channel and stall every
        # later chain.  Mark the chain dead instead — the worker drains
        # the remaining steps with dummy payloads, the reserved seqnos get
        # consumed, and the pre-posted recvs pair off normally.
        for first_ready in (
            self.prefill_drafts_first_ready,
            self.decode_drafts_first_ready,
        ):
            for output in first_ready:
                if (
                    self._scheduler_output_intersects_req_ids(output, req_ids)
                    and self._scheduler_output_all_requests_finished(output)
                ):
                    output.draft_chain_dead = True
                    if output.draft_task_id is not None:
                        self._dead_draft_task_ids.add(output.draft_task_id)
                        # Release any deferred cloud control for this chain —
                        # a dead parent never produces the scalars the deferral
                        # waits for, so without the release the SO would sit in
                        # the deferred queue forever and the cloud would never
                        # run the middle (channel stall).
                        self._dead_chain_publish_to_release.append(
                            output.draft_task_id
                        )
                    task_id = output.draft_task_id
                    if (
                        task_id is not None
                        and self._draft_publish_pending.get(task_id) is output
                    ):
                        self._draft_publish_pending.pop(task_id, None)
                        self._draft_publish_scalars_patched.discard(task_id)
                        self._draft_publish_dispatched.discard(task_id)
                    logger.info(
                        "[PD] mark draft chain dead (drain as dummies): "
                        "task_id=%s step=%s",
                        output.draft_task_id,
                        output.draft_step_idx,
                    )

        # A queued placeholder DECODE_FIRST self-posted its DECODE_LAST tail
        # into decodes_last_ready at creation time
        # (_prepare_next_decode_first_placeholder -> _pick_decode_first_batch).
        # Dropping the head without the tail orphans the tail: once
        # decodes_first_ready is empty, the overtake guard in _pick_by_state
        # stops applying, the orphaned tail gets picked, and the edge worker
        # crashes in _resume_and_validate_head_state because no head ever
        # suspended a HeadState for that head_token.  Drop the pair together
        # and roll back every side effect claimed at creation time.
        kept_decodes_first: deque[SchedulerOutput] = deque()
        for output in self.decodes_first_ready:
            if self._scheduler_output_intersects_req_ids(output, req_ids):
                self._drop_placeholder_decode_first(output, req_ids)
            else:
                kept_decodes_first.append(output)
        self.decodes_first_ready = kept_decodes_first
        pending_decode = self._decode_first_placeholder_parent
        if (
            pending_decode is not None
            and self._scheduler_output_intersects_req_ids(
                pending_decode, req_ids
            )
        ):
            self._decode_first_placeholder_parent = None
        # Never drop a queued DRAFT_LAST.  Tails are self-posted in
        # _pick_draft_first_batch at the moment their head is picked, and
        # a picked DRAFT_FIRST is always dispatched to the cloud.  The
        # cloud does not track request lifecycle, so it will still isend
        # the response; the edge MUST execute (drain) the tail to keep
        # the DECODE hidden channel paired (see _pick_draft_last_batch
        # and the drain path in the worker's
        # _run_edge_cloud_draft_last_segment).  Dropping the tail also
        # strands the lane's _force_*_draft_last=True -- the flag is only
        # cleared when a DRAFT_LAST is picked -- which then blocks every
        # future DRAFT_FIRST of that lane and deadlocks the scheduler.
        for last_ready in (
            self.prefill_drafts_last_ready,
            self.decode_drafts_last_ready,
        ):
            for output in last_ready:
                if self._scheduler_output_intersects_req_ids(output, req_ids):
                    gone = {
                        rid
                        for rid in output.num_scheduled_tokens
                        if rid in req_ids
                    }
                    if output.parent_req_id in req_ids:
                        gone.add(output.parent_req_id)
                    logger.info(
                        "[PD] keep DRAFT_LAST task_id=%s step=%s for drain "
                        "(%d member request(s) gone; its DRAFT_FIRST was "
                        "already dispatched to the cloud)",
                        output.draft_task_id,
                        output.draft_step_idx,
                        len(gone),
                    )

    def take_dropped_draft_task_ids(self) -> list[str]:
        """Drain draft task ids dropped from the ready queues since the
        last call (EngineCore patch forwards them to the runner)."""
        dropped = self._dropped_draft_task_ids_to_report
        self._dropped_draft_task_ids_to_report = []
        return dropped

    def take_dead_chain_publish_releases(self) -> list[str]:
        """Drain dead-chain task ids whose deferred cloud control must be
        released for immediate publication (EngineCore patch forwards them
        to _release_deferred_draft_pre_out)."""
        items = self._dead_chain_publish_to_release
        self._dead_chain_publish_to_release = []
        return items

    def _is_stale_draft_output(
        self, scheduler_output: SchedulerOutput
    ) -> bool:
        # A draft output is stale when EVERY backing request has finished.
        # This drives the DRAFT_FIRST dead-chain marking in
        # _pick_draft_first_batch: once all owning requests are gone, the
        # edge can no longer produce a real payload (the draft context was
        # cleared on finish/abort), so the remaining steps drain as dummies
        # (draft_chain_dead) -- they are never dropped, because their comm
        # seqnos were reserved and their recvs pre-posted at parent publish
        # time, and a drop would hole the channel.
        # Partial finishes keep the draft alive: the cloud-side cached
        # attention metadata is whole-batch and cannot be re-sliced, so the
        # chain runs to completion and the dead rows' draft tokens are
        # discarded by the worker (_run_edge_cloud_draft_last_segment).
        # Already-dispatched heads are handled separately: their matching
        # DRAFT_LAST is always executed (drained) in _pick_draft_last_batch
        # to pair the cloud's response, so this check intentionally does NOT
        # exempt pre-generated dispatched chains the way it used to.
        # NOTE: finished requests may still be present in self.requests
        # while their KV blocks are retained for an in-flight draft chain,
        # so liveness must go through is_finished() (via
        # _scheduler_output_all_requests_finished), not dict membership.
        return self._scheduler_output_all_requests_finished(scheduler_output)

    def _draft_output_reqs_live(
        self, scheduler_output: SchedulerOutput
    ) -> bool:
        """True if any request backing this draft output is still active.

        Used both to decide whether a DRAFT_LAST may spawn a verify
        placeholder and (inverted) whether a DRAFT_FIRST is stale.
        """
        return not self._scheduler_output_all_requests_finished(
            scheduler_output
        )

    def _pick_decode_last_batch(self) -> SchedulerOutput:
        if not self.decodes_last_ready:
            return self._make_empty_batch()
        so = self.decodes_last_ready.popleft()
        assert so.batch_type == BatchType.DECODE_LAST, (
            f"decodes_last_ready expects DECODE_LAST, got {so.batch_type}"
        )
        self._start_decode_or_draft_first_only_window()
        self._force_decode_last = False
        self._pregenerate_draft_chain(so)
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
            if req_id not in cached_reqs.all_token_ids:
                # Use the Request-level cached np.ndarray to avoid repeated
                # np.asarray() conversion of the Python list (dominant
                # bottleneck on long-sequence decode batches).
                cached_reqs.all_token_ids[req_id] = (
                    self.requests[req_id].cached_all_token_ids_np)

    def _pick_decode_first_batch(
        self, defer_seqno: bool = False
    ) -> SchedulerOutput:
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
                    logger.debug(
                        "DECODE_FIRST race: empty batch due to async "
                        "update_from_output delay, running=%d",
                        len(self.running),
                    )
                else:
                    scheduler_output.batch_type = BatchType.DECODE_FIRST
                    scheduler_output.head_token = uuid4().hex
                    # Placeholder verify DFs are pre-built behind a draft
                    # chain and may be dropped before dispatch; stamping
                    # their comm_seqno at creation would leave a hole on
                    # the channel.  defer_seqno defers the stamp (and the
                    # draft-chain reservation) to the pop in _pick_by_state.
                    if not defer_seqno:
                        scheduler_output.comm_seqno = self._decode_comm_seqno
                        self._decode_comm_seqno += 1
                        self._reserve_draft_seqnos(scheduler_output)
                    self._ensure_cached_all_token_ids(scheduler_output)
                    self.decode_or_draft_inflight_count += 1
                    self.decode_head_inflight_count += 1
                    self._register_pd_flight(scheduler_output)
                    self._force_decode_last = True

                    # === Decode-first self-posting optimization ===
                    # Cloud's _maybe_publish_post_out merely replaces
                    # batch_type with DECODE_LAST.  We pre-generate it on
                    # the edge side and stash it in decodes_last_ready so
                    # that scheduling DECODE_LAST needs no round-trip
                    # through POST_OUT.  The cloud unconditionally skips
                    # POST_OUT for all DECODE_FIRST batches.
                    decode_last = replace(
                        scheduler_output,
                        batch_type=BatchType.DECODE_LAST,
                    )
                    self.decodes_last_ready.append(decode_last)
                    self._unsettled_decode_tail_count += 1
                    # ===============================================
                for req in list(self.waiting):
                    saved_waiting.prepend_request(req)
                self.chunk_prefill_first = saved_chunk_prefill_first
                self.waiting = saved_waiting
                self.skipped_waiting = saved_skipped
            else:
                self.chunk_prefill_first = saved_chunk_prefill_first
                self.waiting = saved_waiting
                self.skipped_waiting = saved_skipped

        return scheduler_output  # type: ignore[return-value]

    def _migrate_prefill_to_running(self) -> None:
        """Move fully-prefilled requests from chunk_prefill_first to running.

        When chunk_prefill_prior is enabled, a request stays in
        chunk_prefill_first even after ``is_prefill_chunk`` becomes False
        if it still has pending PL returns.  Only requests with zero
        pending tails are eligible to enter decode.
        """
        completed = [
            req for req in self.chunk_prefill_first
            if not req.is_prefill_chunk
            and self._pending_tail_count.get(req.request_id, 0) == 0
        ]
        for req in completed:
            self.chunk_prefill_first.remove(req)
            self.running.append(req)
            # A preempted request lands in chunk_prefill_first (status
            # stays PREEMPTED) and may be re-added to running here.  If
            # super().schedule() later selects it for preemption again,
            # _preempt_request asserts status == RUNNING.  Restore the
            # status unconditionally: the request was fully prefilled and
            # is entering decode — it must be RUNNING.
            if req.status != RequestStatus.RUNNING:
                req.status = RequestStatus.RUNNING

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        if not self._is_request_preemptible(request):
            raise RuntimeError(
                "Cannot preempt an active edge-cloud request: "
                f"request_id={request.request_id}, status={request.status}, "
                "active_flights="
                f"{self._pd_active_flight_count.get(request.request_id, 0)}"
            )

        # Use the upstream recovery path so KV ownership, computed progress,
        # request status, and resumed-request metadata stay consistent.
        super()._preempt_request(request, timestamp)
        self._cleanup_request_flight_state(request.request_id)
        request.chunk_num = 1
        request.is_prefill_chunk = False

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        was_prefill_map = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            was_prefill_map[req_id] = self.requests[req_id].is_prefill_chunk

        super()._update_after_schedule(scheduler_output)

        for req_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items():
            if was_prefill_map[req_id] and num_scheduled_token > 0:
                self.requests[req_id].chunk_num += 1

        self._migrate_prefill_to_running()
        self.finished_req_ids = set()

    # ------------------------------------------------------------------ #
    # update_from_output — chunk-prefill-prior routing                    #
    # ------------------------------------------------------------------ #
    def _update_from_output_prefill_last_legacy(
        self, scheduler_output: SchedulerOutput
    ) -> None:
        """Legacy PL routing: request-granularity pending list."""
        completed_req_ids = set(scheduler_output.num_scheduled_tokens.keys())
        newly_running: list[Request] = []
        newly_chunked: list[Request] = []
        remaining_pending: list[Request] = []
        for req in self.prefill_last_pending:
            if req.request_id in completed_req_ids:
                if req.is_prefill_chunk:
                    self.chunk_prefill_first.append(req)
                    newly_chunked.append(req)
                else:
                    self.running.append(req)
                    newly_running.append(req)
            else:
                remaining_pending.append(req)
        self.prefill_last_pending = remaining_pending

        # logger.info(
        #     f"[PD] update_from_output PREFILL_LAST done, "
        #     f"prefill_inflight: {self.prefill_inflight_count}/{self.prefill_inflight_limit}, "
        #     f"moved {len(newly_running)} reqs to running[], "
        #     f"moved {len(newly_chunked)} reqs to chunk_prefill_first[], "
        #     f"running[]: {len(self.running)}, "
        #     f"chunk_prefill_first[]: {len(self.chunk_prefill_first)}",
        # )

    def _update_from_output_prefill_last_chunk_prior(
        self, scheduler_output: SchedulerOutput
    ) -> None:
        """Chunk-prefill-prior PL routing: head_token → flight lookup."""
        head_token = scheduler_output.head_token
        if not head_token:
            logger.warning(
                "[PD-CHUNK-PRIOR] PREFILL_LAST missing head_token; "
                "falling back to legacy routing."
            )
            self._update_from_output_prefill_last_legacy(scheduler_output)
            return

        flight = self._prefill_flight_by_token.pop(head_token, None)
        if flight is None:
            logger.warning(
                "[PD-CHUNK-PRIOR] PREFILL_LAST head_token=%s not found "
                "in flight map; falling back to legacy routing.",
                head_token,
            )
            self._update_from_output_prefill_last_legacy(scheduler_output)
            return

        req_id = flight.request_id
        req = self.requests.get(req_id)

        # Decrement pending tail count.
        prev_count = self._pending_tail_count.get(req_id, 0)
        if prev_count > 0:
            self._pending_tail_count[req_id] = prev_count - 1
        remaining = self._pending_tail_count.get(req_id, 0)

        # Decrement ahead count if this chunk was pre-scheduled.
        ahead_before = self._ahead_chunk_count.get(req_id, 0)
        if ahead_before > 0:
            self._ahead_chunk_count[req_id] = ahead_before - 1

        logger.info(
            "[PD-CHUNK-PRIOR] PL returned: request=%s chunk=%d/%s "
            "head_token=%s tokens=%d "
            "pending_tails: %d→%d ahead: %d→%d",
            req_id,
            flight.chunk_index,
            "last" if flight.is_last_chunk else "mid",
            head_token,
            flight.num_scheduled_tokens,
            prev_count,
            remaining,
            ahead_before,
            self._ahead_chunk_count.get(req_id, 0),
        )

        if flight.is_last_chunk and remaining == 0:
            # All chunks complete → request enters decode.
            if req is not None:
                self.running.append(req)
            self._cleanup_request_flight_state(req_id)
            logger.info(
                "[PD-CHUNK-PRIOR] Request %s all chunks done, "
                "moved to running[] (%d total).",
                req_id,
                len(self.running),
            )
        else:
            # Mid-chunk PL returned, or last chunk but other tails still
            # pending.  Re-add for the next chunk IF there are more chunks
            # to schedule AND the request is not already queued.  This keeps
            # the pipeline continuous across >2 chunks: the next chunk's PF
            # fills the prefill slot freed by this PL, overlapping other
            # in-flight PLs (e.g. 4 chunks -> chunk2 PF starts as soon as
            # chunk0 PL returns, overlapping chunk1's PL, instead of waiting
            # for chunk1's PL).  The old logic skipped re-add whenever
            # ahead_before > 0, which forced pair-wise scheduling
            # ((0,1) then (2,3)) and left a slot idle between pairs.
            # Do NOT call _cleanup_request_flight_state here: ahead count
            # and in-flight flights are still needed for outstanding chunks.
            has_more_chunks = (
                req is not None
                and req.num_computed_tokens < req.num_prompt_tokens
            )
            already_queued = (
                req is not None and req in self.chunk_prefill_first
            )
            if has_more_chunks and not already_queued:
                self.chunk_prefill_first.append(req)
                logger.info(
                    "[PD-CHUNK-PRIOR] Request %s chunk %d PL: "
                    "re-added to chunk_prefill_first[] for next chunk "
                    "(remaining=%d, ahead=%d).",
                    req_id,
                    flight.chunk_index,
                    remaining,
                    self._ahead_chunk_count.get(req_id, 0),
                )
            else:
                logger.info(
                    "[PD-CHUNK-PRIOR] Request %s chunk %d PL: "
                    "skip re-add (remaining=%d, ahead=%d, more=%s, "
                    "queued=%s).",
                    req_id,
                    flight.chunk_index,
                    remaining,
                    self._ahead_chunk_count.get(req_id, 0),
                    has_more_chunks,
                    already_queued,
                )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, Any]:
        if scheduler_output.batch_type == BatchType.PREFILL_LAST:
            if self.prefill_inflight_count > 0:
                self.prefill_inflight_count -= 1

            # logger.info(
            #     f"[PD] update_from_output PREFILL_LAST done, "
            #     f"prefill_inflight: {self.prefill_inflight_count}/{self.prefill_inflight_limit}, "
            #     f"moved {len(newly_running)} reqs to running[], "
            #     f"moved {len(newly_chunked)} reqs to chunk_prefill_first[], "
            #     f"running[]: {len(self.running)}, "
            #     f"chunk_prefill_first[]: {len(self.chunk_prefill_first)}",
            # )
            if self.chunk_prefill_prior_enable:
                self._update_from_output_prefill_last_chunk_prior(
                    scheduler_output
                )
            else:
                self._update_from_output_prefill_last_legacy(
                    scheduler_output
                )

        if scheduler_output.batch_type == BatchType.DECODE_FIRST:
            # D首完成后立即释放 inflight 计数，使下一个 D首可以
            # 在 D尾仍在 batch_queue 中时就被调度，消除 Cloud idle gap。
            if self.decode_or_draft_inflight_count > 0:
                self.decode_or_draft_inflight_count -= 1
            if self.decode_head_inflight_count > 0:
                self.decode_head_inflight_count -= 1
            logger.info(
                f"[PD] update_from_output DECODE_FIRST done, "
                f"decode_or_draft_inflight: {self.decode_or_draft_inflight_count}/{self.decode_or_draft_inflight_limit}",
            )
        if scheduler_output.batch_type in _DRAFT_FIRST_TYPES:
            # Only decode-phase drafts claim the decode-lane in-flight
            # counter at pick time; prefill-phase drafts never touch it.
            if not scheduler_output.draft_prefill_phase:
                self.decode_or_draft_inflight_count = max(
                    0, self.decode_or_draft_inflight_count - 1
                )
            logger.info(
                "[PD] update_from_output DRAFT_FIRST done, "
                "prefill_phase=%s, decode_or_draft_inflight: %d/%d",
                bool(scheduler_output.draft_prefill_phase),
                self.decode_or_draft_inflight_count,
                self.decode_or_draft_inflight_limit,
            )
        enqueue_next_draft = (
            scheduler_output.batch_type in _DRAFT_LAST_TYPES
        )
        if enqueue_next_draft:
            if scheduler_output.draft_prefill_phase:
                self.prefill_draft_remote_pending_count = max(
                    0, self.prefill_draft_remote_pending_count - 1
                )
            else:
                self.decode_draft_remote_pending_count = max(
                    0, self.decode_draft_remote_pending_count - 1
                )
        if scheduler_output.batch_type == BatchType.DECODE_LAST:
            # decode_or_draft_inflight_count 已在 DECODE_FIRST 的 update_from_output
            # 中释放，此处不再重复减 1。
            if self._unsettled_decode_tail_count > 0:
                self._unsettled_decode_tail_count -= 1
            logger.info(
                f"[PD] update_from_output DECODE_LAST done, "
                f"decode_or_draft_inflight: {self.decode_or_draft_inflight_count}/{self.decode_or_draft_inflight_limit}",
            )
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        if scheduler_output.batch_type in _PD_LAST_TO_FIRST:
            self._complete_pd_flight(scheduler_output)
        if self.finished_req_ids:
            # Natural completion is handled inside the base update path and
            # does not pass through finish_requests().  Do not dispatch any
            # not-yet-enqueued placeholder tasks for those requests.
            self._drop_stale_drafts_for_req_ids(self.finished_req_ids)
        if enqueue_next_draft:
            next_draft_ready = self._enqueue_next_draft_first(
                scheduler_output
            )
            logger.info(
                "[PD] update_from_output DRAFT_LAST done, "
                "prefill_phase=%s, prefill_draft_remote_pending: %d, "
                "decode_draft_remote_pending: %d, next_draft_ready: %s",
                bool(scheduler_output.draft_prefill_phase),
                self.prefill_draft_remote_pending_count,
                self.decode_draft_remote_pending_count,
                next_draft_ready,
            )
        self.chunk_prefill_first = [
            req for req in self.chunk_prefill_first if not req.is_finished()
        ]
        self.prefill_last_pending = [
            req for req in self.prefill_last_pending if not req.is_finished()
        ]
        # Drain finished requests from running: update_from_output removes
        # completed requests from self.requests, but they may still be in
        # running.  That causes KeyError in super().schedule() /
        # _update_after_schedule when the base scheduler accesses
        # self.requests[req_id].  Clean them up here, once per step.
        self.running = [
            req for req in self.running
            if req.request_id in self.requests
        ]
        # A DECODE_LAST settle is the earliest moment the recorded
        # num_computed values become exact.  If a placeholder prebuild was
        # deferred by the unsettled-tail gate, fire it right here instead of
        # waiting for the next scheduling turn's _pick_by_state retry, so
        # the gate costs no extra pipeline delay.  Must run after the
        # running[] cleanup above: the prebuild calls super().schedule().
        if (
            scheduler_output.batch_type == BatchType.DECODE_LAST
            and self._unsettled_decode_tail_count == 0
            and self._decode_first_placeholder_parent is not None
        ):
            self._prepare_next_decode_first_placeholder(
                self._decode_first_placeholder_parent
            )
        return outputs

    def get_request_counts(self) -> tuple[int, int]:
        num_running, num_waiting = super().get_request_counts()
        if self.chunk_prefill_prior_enable:
            # Use a set to avoid double-counting requests that appear in
            # multiple tracking structures (e.g. a request in
            # prefill_last_pending also has _pending_tail_count > 0).
            pending_ids: set[str] = set()
            pending_ids.update(
                req.request_id for req in self.chunk_prefill_first
            )
            pending_ids.update(
                req.request_id for req in self.prefill_last_pending
            )
            pending_ids.update(self._pending_tail_count.keys())
            return (num_running + len(pending_ids), num_waiting)
        return (
            num_running
            + len(self.chunk_prefill_first)
            + len(self.prefill_last_pending),
            num_waiting,
        )

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        base = super().get_num_unfinished_requests()
        if self.chunk_prefill_prior_enable:
            pending_ids: set[str] = set()
            pending_ids.update(
                req.request_id for req in self.chunk_prefill_first
            )
            pending_ids.update(
                req.request_id for req in self.prefill_last_pending
            )
            pending_ids.update(self._pending_tail_count.keys())
            return base + len(pending_ids)
        return (
            base
            + len(self.chunk_prefill_first)
            + len(self.prefill_last_pending)
        )

    def _has_draft_work(self) -> bool:
        return bool(
            self.decodes_first_ready
            or self.prefill_drafts_first_ready
            or self.prefill_drafts_last_ready
            or self.decode_drafts_first_ready
            or self.decode_drafts_last_ready
            or self.decode_or_draft_inflight_count > 0
            or self.prefill_draft_remote_pending_count > 0
            or self.decode_draft_remote_pending_count > 0
        )

    def has_requests(self) -> bool:
        return super().has_requests() or self._has_draft_work()

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        result = super().finish_requests(request_ids, finished_status)
        finished_req_ids = {req_id for req_id, _client_index in result}
        self._drop_stale_drafts_for_req_ids(finished_req_ids)
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
                # Clean up chunk-prefill-prior flight state.
                self._cleanup_request_flight_state(req_id)

        if to_remove:
            self.chunk_prefill_first = remove_all(
                self.chunk_prefill_first, to_remove
            )

        return result

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        if reset_running_requests:
            if self._pd_active_flight_count:
                logger.warning(
                    "Cannot reset running requests while edge-cloud flights "
                    "are active: %s",
                    self._pd_active_flight_count,
                )
                return False

            if any(
                not self._is_request_preemptible(request)
                for request in self.chunk_prefill_first
            ):
                logger.warning(
                    "Cannot reset edge-cloud prefill requests because at least "
                    "one request is not safely preemptible"
                )
                return False

            timestamp = time.monotonic()
            while self.chunk_prefill_first:
                request = self.chunk_prefill_first.pop()
                self._preempt_request(request, timestamp)
                request.async_tokens_to_discard = (
                    request.num_output_placeholders
                )
                request.num_output_placeholders = 0

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


class AsyncPDSeparatedScheduler(PDSeparatedScheduler, AsyncScheduler):
    """Async scheduler with PD separation."""
    pass
