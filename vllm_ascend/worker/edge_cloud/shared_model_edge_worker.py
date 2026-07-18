# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-model edge worker for the edge-cloud collaboration feature.

This module defines :class:`SharedModelEdgeWorker`, a subclass of
:class:`NPUWorker` that lets multiple DP-rank edge workers live in a
single process and share a single ``nn.Module`` replica in NPU memory.

Key invariants:

- All virtual workers in the process share one ``self.rank`` (one
  distributed process group per process).
- ``self.local_rank`` uniquely identifies a virtual worker inside the
  process and is set to be equal to ``self.parallel_config.data_parallel_rank``.
- The total number of virtual workers is
  ``self.parallel_config.data_parallel_size``.
- Only the virtual worker with ``local_rank == 0`` (the *leader*) runs
  the heavy one-shot initialisation (device init, distributed init,
  workspace, model load, NPU graph capture). Followers (the rest) reuse
  the leader's process-level state and bind to the leader's model
  through :meth:`BatchedModelRunner.bind_to_shared_model`.
- All virtual workers participate in PP communication at runtime.
  ``local_rank == k`` routes its PP messages to the cloud first-worker
  for DP-rank ``k``.

PP group layout (with ``data_parallel_size == N``):

    in-group rank 0:        edge_R (the single edge distributed rank)
    in-group rank 1..N:     cloud_dp_0_first, ..., cloud_dp_{N-1}_first

So:

- The edge (one rank) is at in-group rank 0.
- The cloud's first workers are at in-group ranks 1..N. The cloud
  first-worker for DP instance ``k`` is at in-group rank ``k + 1``.

Each virtual edge worker ``k`` communicates with the cloud peer at
in-group rank ``k + 1``. From the cloud's view, the edge is at
in-group rank 0.

This class only implements the *edge* side. The cloud side keeps the
existing edge-cloud layout (one process per cloud rank, TP within each
DP instance, only the first cloud rank per DP instance participating in
the shared PP group).
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
)
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

from vllm_ascend.distributed.parallel_state import (
    edge_cloud_broadcast_recv,
    edge_cloud_isend_tensor_dict,
    get_edge_cloud_tensor_meta,
    init_ascend_model_parallel,
    init_edge_cloud_tensor_meta,
)
from vllm_ascend.worker.edge_cloud.execute_model_bundle import (
    _ExecuteModelBundle,
)
from vllm_ascend.worker.edge_cloud.batched_model_runner import BatchedModelRunner
from vllm_ascend.worker.worker import NPUWorker, _detect_has_residual
from vllm_ascend.utils import enable_sp

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

# Process-wide list of SharedModelEdgeWorker instances, in local_rank
# order. We use a list rather than a rank-keyed dict because all
# virtual workers in the process share the same distributed ``rank``;
# a dict would silently overwrite earlier entries.
_SHARED_MODEL_REGISTRY: list["SharedModelEdgeWorker"] = []


def get_leader_worker() -> "SharedModelEdgeWorker | None":
    """Return the leader worker (local_rank == 0) in this process."""
    for w in _SHARED_MODEL_REGISTRY:
        if w._is_leader:
            return w
    return None

class DeferredExecutePostprocess(AsyncModelRunnerOutput):
    """Marker returned by
    :meth:`SharedModelEdgeWorker.execute_model` indicating that
    the tail recv + tail forward is deferred to the end of the
    current round.

    The class is both an :class:`AsyncModelRunnerOutput` *and*
    a :class:`collections.abc.Callable` — the
    :class:`vllm.v1.executor.shared_model_multiproc_executor.SharedModelWorkerProc`
    detects markers via the conjunction
    ``isinstance(output, AsyncModelRunnerOutput) and callable(output)``
    and accumulates them in ``self._pending_deferred``,
    invoking ``get_output()`` at the round boundary (which
    runs the postprocess and writes the final result to
    ``response_mqs[dp_rank]``).

    Two call surfaces are supported:

    * :meth:`get_output` — the path the WorkerProc takes via
      :meth:`vllm.v1.executor.shared_model_multiproc_executor.SharedModelWorkerProc.enqueue_output`.
      After running the postprocess, it does **one more type
      check** on the result: if the postprocess happens to
      return another :class:`AsyncModelRunnerOutput` (e.g. a
      future from a nested deferred step), it is unwrapped via
      ``result.get_output()`` recursively. The result the
      WorkerProc sees is therefore guaranteed to be a
      ``ModelRunnerOutput`` / ``None`` / ``IntermediateTensors``
      / ``Exception``, never a still-deferred object.

    * :meth:`__call__` — the direct-call path. The postprocess
      is run and the raw result is returned **without any
      type check**. This is what other code (e.g. unit tests
      or hand-rolled drivers) gets when it bypasses the
      WorkerProc's enqueue_output unwrap and simply calls the
      marker.
    """

    __slots__ = ("postprocess",)

    def __init__(self, postprocess) -> None:
        # ``postprocess`` is a zero-argument callable produced
        # by ``execute_model``. When invoked it runs the tail
        # recv + tail forward and returns the raw result with
        # signature ``ModelRunnerOutput | AsyncModelRunnerOutput
        # | IntermediateTensors | None``.
        self.postprocess = postprocess

    def __call__(self):
        # Direct call: run the postprocess and hand the raw
        # result back to the caller with no further
        # processing. This is the "shortcut" path: it is the
        # caller's responsibility to deal with the result
        # type.
        return self.postprocess()

    def get_output(self):
        # WorkerProc path. Run the postprocess, then do one
        # more type check on the result: if the postprocess
        # itself returned a nested
        # :class:`AsyncModelRunnerOutput` (e.g. a still-deferred
        # future), unwrap it via its own ``get_output()``.
        # ``enqueue_output`` downstream treats this return
        # value as a "ready" output and writes it straight to
        # the response MQ.
        result = self.postprocess()
        if isinstance(result, AsyncModelRunnerOutput):
            result = result.get_output()
        return result

class _BatchedExecuteMarker(DeferredExecutePostprocess):
    """Marker returned by :meth:`SharedModelEdgeWorker.execute_model_batched_pre`
    carrying the per-dp_rank :class:`_ExecuteModelBundle` produced by
    ``BatchedModelRunner.execute_model_pre``, plus a back-reference to the
    :class:`SharedModelEdgeWorker` that produced it.

    The shared model ``worker_busy_loop`` intercepts this via
    ``isinstance(output, _BatchedExecuteMarker)`` in ``_dispatch``,
    stores ``output.bundle`` in ``self._round_bundles[k]`` and at the
    round barrier invokes the marker:

    * :meth:`drive_batched_round` (member function) — per-dp_rank
      phase of the round: install this dp_rank's PP isend + recv
      closure into ``pending_deferred``. The dp_rank comes from
      ``self.worker.local_rank`` (the worker the marker was produced
      on). The batched head itself is run once via
      :meth:`run_batched_head` (class function), which is dp_rank /
      ``self``-agnostic.
    * :meth:`drain_batched_round` (class function) — Phase B/C: collect
      per-dp_rank intermediates from the recv closures, run 1× batched
      tail forward + 1× ``compute_logits`` on the leader runner, slice
      the merged sample hidden states / logits back to per-dp_rank and
      call :meth:`BatchedModelRunner.execute_model_post_batched` on each
      worker's model_runner. Then per-dp_rank ``on_dp_rank_output(k, None)``
      is invoked to signal the engine that ``sample_tokens`` can run.

    The round state (``_round_bundles`` / ``_round_intermediates`` /
    ``pending_deferred``) is passed in as parameters — it is **owned
    by** the upstream ``SharedModelWorkerProc`` so the busy_loop can
    manage its lifetime alongside the rest of the round barrier.
    """

    # Class-level cache of the per-dp_rank head slices produced by
    # ``execute_model_batched_head``; consumed by every marker's
    # per-dp_rank PP isend during a single round. Cleared by
    # :meth:`drain_batched_round` at end of round. Keyed by
    # ``dp_rank`` (not by position in ``batched_dp_ranks``) so
    # that a partial round (e.g. ``batched_dp_ranks = [1, 3]``)
    # still resolves ``per_dp_hidden[1]`` / ``per_dp_hidden[3]``
    # without confusing a position index with the dp_rank
    # value.
    _per_dp_hidden: dict[int, Any] | None = None

    __slots__ = ("bundle", "worker")

    def __init__(
        self,
        bundle: "_ExecuteModelBundle",
        worker: "SharedModelEdgeWorker",
    ) -> None:
        self.bundle = bundle
        self.worker = worker
        # The real work is driven by the busy_loop via
        # ``run_batched_head`` (class function, runs once per round
        # for the batched head) and ``drive_batched_round`` (member
        # function, runs once per marker for the per-dp_rank
        # isend + recv closure). The default ``__call__`` path is a
        # no-op (the marker is not callable in the deferred-tail
        # sense).
        super().__init__(postprocess=lambda: None)

    # ------------------------------------------------------------------
    # Phase A — batched head (1× per round, dp_rank / self-agnostic)
    # ------------------------------------------------------------------
    @classmethod
    def run_batched_head(
        cls,
        batched_dp_ranks: list[int],
        round_bundles: dict[int, "_ExecuteModelBundle"],
    ) -> dict[int, Any]:
        """Run the 1× batched head forward on the leader runner.

        dp_rank- and ``self``-agnostic: takes the ordered list of
        dp_ranks participating in this batched forward plus the
        per-dp_rank bundle dict, and returns a list of per-dp_rank
        hidden slices indexed in the same ``batched_dp_ranks`` order
        (so ``result[i]`` corresponds to ``batched_dp_ranks[i]``).
        Called once per round by the busy_loop (the first marker of
        the round).

        ``batched_dp_ranks`` may be a strict subset of
        ``{0..dp_size-1}`` when some dp_ranks did not produce a
        batched marker in this round (e.g. empty
        ``execute_model`` / ``execute_dummy_batch``); those dp_ranks
        are not merged into the head and have no
        ``_per_dp_hidden`` entry — they are not handled by the
        batched path at all.
        """
        leader = next(
            (w for w in _SHARED_MODEL_REGISTRY
             if getattr(w, "_is_leader", False)), None)
        assert leader is not None, (
            "SharedModelEdgeWorker: no leader worker found in "
            "_SHARED_MODEL_REGISTRY when running batched head")
        leader_runner = leader.model_runner
        # Iterate ``batched_dp_ranks`` in caller-supplied order to
        # keep the merged tensor layout deterministic and
        # independent of the full ``{0..dp_size-1}`` range.
        bundles = [round_bundles[k] for k in batched_dp_ranks]
        # The leader runner handles the merged attn_metadata
        # construction internally in the non-embedding_only path
        # (re-using ``self._get_or_build_merged_attn_ctx``), so
        # the worker only needs to forward ``batched_dp_ranks``.
        per_dp_hidden_list = leader_runner.execute_model_batched_head(
            bundles,
            batched_dp_ranks=batched_dp_ranks,
            pp_send_work_by_channel=getattr(leader, "_pp_send_work_by_channel", None),
        )
        # Key the per-dp_rank hidden slices by dp_rank rather
        # than by their position in ``batched_dp_ranks``: this
        # lets ``drive_batched_round`` resolve
        # ``per_dp_hidden[dp_rank]`` directly (where ``dp_rank``
        # is the worker's ``local_rank``) regardless of
        # whether ``batched_dp_ranks`` is a strict subset of
        # ``{0..dp_size-1}``.
        cls._per_dp_hidden = dict(zip(batched_dp_ranks,
                                       per_dp_hidden_list))
        return cls._per_dp_hidden

    # ------------------------------------------------------------------
    # Phase A — per-dp_rank PP isend + recv closure
    # ------------------------------------------------------------------
    def drive_batched_round(
        self,
        pending_deferred: dict[int, Any],
    ) -> None:
        """Per-dp_rank phase A: install this dp_rank's PP isend + recv
        closure into ``pending_deferred``.

        dp_rank comes from ``self.worker.local_rank`` (the worker the
        marker was produced on). Called once per marker by the
        busy_loop after :meth:`run_batched_head` has populated
        :attr:`_per_dp_hidden`.
        """
        per_dp_hidden = _BatchedExecuteMarker._per_dp_hidden
        assert per_dp_hidden is not None, (
            "_BatchedExecuteMarker.drive_batched_round called before "
            "run_batched_head populated _per_dp_hidden")
        # ``_per_dp_hidden`` is keyed by dp_rank (not by position
        # in ``batched_dp_ranks``), so the per-dp_rank slice for
        # this marker is a direct lookup with the marker's own
        # dp_rank.
        dp_rank = self.worker.local_rank
        hidden_k = per_dp_hidden[dp_rank]
        # Edge-cloud with heterogeneous SP: aggregate SP shards to full
        # sequence before cross-PP send so cloud can re-chunk by its SP.
        if enable_sp() and (self.worker.model_runner.edge_cloud_cfg.mode != "embedding_only"
            or not self.worker.model_runner.supports_mm_inputs):
            _gathered = self.worker._all_gather_tensor_dict(hidden_k.tensors)
        else:
            _gathered = hidden_k.tensors
        # Mirror ``execute_model``: use the edge-cloud-optimised isend
        # with explicit ``dst`` and ``num_tokens`` slicing so the
        # receiver can allocate buffers based on
        # ``SchedulerOutput.total_num_scheduled_tokens`` alone (no
        # metadata wire transfer).
        num_tokens = self.bundle.scheduler_output.total_num_scheduled_tokens
        self.worker._pp_send_work = edge_cloud_isend_tensor_dict(
            _gathered,
            dst=dp_rank + 1,
            num_tokens=num_tokens,
        )
        edge_sp = enable_sp()
        edge_merge = get_edge_cloud_tensor_meta().merge_payload
        pending_deferred[dp_rank] = (
            self.worker.make_batched_recv_closure(
                src=dp_rank + 1,
                num_tokens=num_tokens,
                sp_chunk=edge_sp and edge_merge))

    # ------------------------------------------------------------------
    # Phase B/C: 1× batched tail + per-dp_rank post_batched + handle_output
    # ------------------------------------------------------------------
    @classmethod
    def drain_batched_round(
        cls,
        round_bundles: dict[int, "_ExecuteModelBundle"],
        round_intermediates: dict[int, Any],
        pending_deferred: dict[int, Any],
        on_dp_rank_output,
    ) -> None:
        """Phase B/C of the batched round, driven once at the end of
        the round.

        ``on_dp_rank_output`` is a ``(dp_rank, output) -> None``
        callback the busy_loop supplies to do the response-MQ write
        (i.e. ``worker_proc.handle_output``). This keeps the marker
        free of any upstream handle while still letting it route
        per-dp_rank results.

        For each dp_rank the recv closure in ``pending_deferred`` is
        invoked to fetch the cloud's middle output, then a single
        batched tail ``_model_forward`` + ``compute_logits`` runs on
        the leader runner. The merged sample hidden states / logits
        are sliced back to per-dp_rank and
        :meth:`BatchedModelRunner.execute_model_post_batched` is called
        on each worker's model_runner. Finally
        ``on_dp_rank_output(k, None)`` is invoked for each dp_rank so
        the engine knows to dispatch the independent
        ``sample_tokens`` RPC.

        At end-of-round the class-level cache is cleared so the next
        round can run a fresh batched head.
        """
        # 1. Drain per-dp_rank recv closures. Failures on a single
        #    dp_rank are routed via the callback as FAILURE
        #    responses, matching the upstream busy_loop convention.
        for dp_rank, deferred in list(pending_deferred.items()):
            try:
                if callable(deferred):
                    intermediate_k = deferred()
                else:
                    intermediate_k = deferred
            except Exception as e:
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception(
                    "SharedModelWorkerProc hit an exception running "
                    "batched recv on dp_rank=%d.", dp_rank)
                on_dp_rank_output(dp_rank, e)
                # Don't keep the dp_rank in the round; the rest of
                # the round only batch-processes the surviving
                # dp_ranks.
                round_bundles.pop(dp_rank, None)
                continue
            round_intermediates[dp_rank] = intermediate_k

        pending_deferred.clear()

        # 2. 1× batched tail on the leader runner — only if at
        #    least one dp_rank survived the recv.
        ok_dp_ranks = sorted(round_intermediates.keys())
        if ok_dp_ranks:
            leader = next(
                (w for w in _SHARED_MODEL_REGISTRY
                 if getattr(w, "_is_leader", False)), None)
            assert leader is not None, (
                "SharedModelEdgeWorker: no leader worker found in "
                "_SHARED_MODEL_REGISTRY when draining batched tail")
            leader_runner = leader.model_runner
            bundles = [round_bundles[k] for k in ok_dp_ranks]
            intermediates = [round_intermediates[k] for k in ok_dp_ranks]
            # The leader runner handles the merged attn_metadata
            # construction internally (re-using the cached
            # ``self._get_or_build_merged_attn_ctx`` from the head
            # segment), so the worker only forwards
            # ``batched_dp_ranks``.
            logger.info(
                "[PD] drain_batched_round: calling "
                "execute_model_batched_tail dp_ranks=%s",
                ok_dp_ranks)
            try:
                (merged_hidden, merged_sample_hidden, merged_logits,
                 kv_connector_output) = (
                    leader_runner.execute_model_batched_tail(
                        bundles, intermediates,
                        batched_dp_ranks=ok_dp_ranks,
                        pp_send_work_by_channel=getattr(leader, "_pp_send_work_by_channel", None)))
                logger.info(
                    "[PD] drain_batched_round: "
                    "execute_model_batched_tail returned "
                    "dp_ranks=%s", ok_dp_ranks)

                # 3. Slice merged tensors back to per-dp_rank and
                #    run per-dp_rank post_batched. The slice
                #    indices follow the order of ``bundles``
                #    (== ``ok_dp_ranks``) and match the token
                #    offsets used in ``execute_model_batched_tail``
                #    (one slice per ``bundles[i]``'s actual
                # token count — NOT per ``logits_indices``).
                # Use the per-bundle actual token count (from
                # the bundle's attn metadata) as the source of
                # truth, NOT ``intermediates[i]['hidden_states']
                # .shape[0]`` (which may not reflect the actual
                # count after the cloud cudagraph pass).
                def _per_bundle_actual(b) -> int:
                    md = b.attn_metadata
                    if isinstance(md, list) and md:
                        md = md[0][next(iter(md[0]))]
                    else:
                        md = next(iter(md.values()))
                    return md.num_actual_tokens

                token_offsets = [0]
                for b in bundles:
                    token_offsets.append(
                        token_offsets[-1] + _per_bundle_actual(b))
                logits_offsets = [0]
                for b in bundles:
                    logits_offsets.append(
                        logits_offsets[-1] + b.logits_indices.shape[0])
                # peer-workers (per-dp_rank) are reached via the
                # registry, indexed by local_rank == dp_rank.
                for i, k in enumerate(ok_dp_ranks):
                    hidden_k = merged_hidden[
                        token_offsets[i]:token_offsets[i + 1]]
                    sample_hs_k = merged_sample_hidden[
                        logits_offsets[i]:logits_offsets[i + 1]]
                    logits_k = merged_logits[
                        logits_offsets[i]:logits_offsets[i + 1]]
                    peer_worker = _SHARED_MODEL_REGISTRY[k]
                    peer_worker.model_runner.execute_model_post_batched(
                        bundles[i], sample_hs_k, logits_k,
                        hidden_k, kv_connector_output)
            except Exception as e:
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception(
                    "SharedModelWorkerProc hit an exception running "
                    "batched tail on leader_runner.")
                # Surface the failure to every surviving dp_rank so
                # the engine sees a FAILURE response on every MQ.
                for k in ok_dp_ranks:
                    on_dp_rank_output(k, e)
                ok_dp_ranks = []

        # 4. Notify the engine that this dp_rank's batched
        #    ``execute_model`` completed; the engine then issues
        #    the independent ``sample_tokens`` RPC.
        for k in ok_dp_ranks:
            on_dp_rank_output(k, None)

        # 5. Cleanup round state. The next round starts fresh.
        _BatchedExecuteMarker._per_dp_hidden = None
        # Drop the merged-attn-ctx cache held by the leader runner so
        # the next round rebuilds it from the new bundles.
        leader_worker = next(
            (w for w in _SHARED_MODEL_REGISTRY
             if getattr(w, "_is_leader", False)), None)
        if leader_worker is not None:
            leader_worker.model_runner._merged_attn_ctx_cache = None
        round_bundles.clear()
        round_intermediates.clear()


class _FirstRoundMarker(DeferredExecutePostprocess):
    """PD-separation FIRST-round marker: head-forward + isend only.

    Does NOT participate in Phase B/C (recv + tail + logits).
    The recv closure is registered into the cross-round
    ``_pd_recv_closures`` cache (owned by
    :class:`SharedModelWorkerProc`) and is consumed by the matching
    LAST round.
    """

    # Class-level cache of per-dp_rank head hidden slices (same
    # semantics as ``_BatchedExecuteMarker._per_dp_hidden``).
    _per_dp_hidden: dict[int, Any] | None = None

    __slots__ = ("bundle", "worker")

    def __init__(
        self,
        bundle: "_ExecuteModelBundle",
        worker: "SharedModelEdgeWorker",
    ) -> None:
        self.bundle = bundle
        self.worker = worker
        super().__init__(postprocess=lambda: None)

    # ------------------------------------------------------------------
    # Phase A — batched head (1× per round, dp_rank / self-agnostic)
    # ------------------------------------------------------------------
    @classmethod
    def run_batched_head(
        cls,
        batched_dp_ranks: list[int],
        round_bundles: dict[int, "_ExecuteModelBundle"],
    ) -> dict[int, Any]:
        """Run the 1× batched head forward on the leader runner.

        Identical to :meth:`_BatchedExecuteMarker.run_batched_head`.
        """
        leader = next(
            (w for w in _SHARED_MODEL_REGISTRY
             if getattr(w, "_is_leader", False)), None)
        assert leader is not None, (
            "SharedModelEdgeWorker: no leader worker found in "
            "_SHARED_MODEL_REGISTRY when running FIRST batched head")
        leader_runner = leader.model_runner
        bundles = [round_bundles[k] for k in batched_dp_ranks]
        total_tokens = sum(
            b.scheduler_output.total_num_scheduled_tokens for b in bundles)
        logger.info(
            "[PD] FIRST head: dp_ranks=%s num_tokens=%d (total bundled)",
            batched_dp_ranks, total_tokens)
        per_dp_hidden_list = leader_runner.execute_model_batched_head(
            bundles,
            batched_dp_ranks=batched_dp_ranks,
            pp_send_work_by_channel=getattr(leader, "_pp_send_work_by_channel", None),
        )
        cls._per_dp_hidden = dict(zip(batched_dp_ranks,
                                       per_dp_hidden_list))
        # Clear the merged-attn-ctx cache on the leader runner so
        # the matching LAST round rebuilds it from its own bundles
        # instead of reusing FIRST's stale cache.
        leader.model_runner._merged_attn_ctx_cache = None
        return cls._per_dp_hidden

    # ------------------------------------------------------------------
    # Phase A — per-dp_rank PP isend (no recv closure)
    # ------------------------------------------------------------------
    def drive_head_send(self) -> None:
        """Per-dp_rank isend to cloud.

        Only does the send — recv is deferred to the matching
        LAST round, which directly calls
        :meth:`_LastRoundMarker.do_direct_recv`.
        """
        per_dp_hidden = _FirstRoundMarker._per_dp_hidden
        assert per_dp_hidden is not None, (
            "_FirstRoundMarker.drive_head_send called before "
            "run_batched_head populated _per_dp_hidden")
        dp_rank = self.worker.local_rank
        hidden_k = per_dp_hidden[dp_rank]
        # Edge-cloud with heterogeneous SP: aggregate SP shards to
        # full sequence before cross-PP send.
        if enable_sp() and (
                self.worker.model_runner.edge_cloud_cfg.mode
                != "embedding_only"
                or not self.worker.model_runner.supports_mm_inputs):
            _gathered = self.worker._all_gather_tensor_dict(
                hidden_k.tensors)
        else:
            _gathered = hidden_k.tensors
        num_tokens = (
            self.bundle.scheduler_output.total_num_scheduled_tokens)
        logger.info(
            "[PD] FIRST isend: dp_rank=%d num_tokens=%d dst=%d",
            dp_rank, num_tokens, dp_rank + 1)
        # Wait for previous send on this channel before launching
        # a new one (mirrors NPUWorker.execute_model L564-576).
        channel = self.worker._hidden_channel_for(
            self.bundle.scheduler_output)
        self.worker._wait_pp_send_work(channel)
        handles = edge_cloud_isend_tensor_dict(
            _gathered,
            dst=dp_rank + 1,
            num_tokens=num_tokens,
            channel=channel,
        )
        self.worker._record_pp_send_work(handles, channel)


class _LastRoundMarker(_BatchedExecuteMarker):
    """PD-separation LAST-round marker: recv + tail-forward only.

    Inherits from :class:`_BatchedExecuteMarker` to reuse
    :meth:`drain_batched_round` (Phase B/C). Does NOT define
    ``run_batched_head`` or ``drive_batched_round`` — those are
    skipped by the busy_loop's three-way split.

    Recv is done directly via :meth:`do_direct_recv` (no closure),
    mirroring :meth:`vllm_ascend.worker.worker.NPUWorker._execute_model_edge_tail`.
    """

    __slots__ = ()

    def do_direct_recv(self) -> "AsyncIntermediateTensors":
        """Directly receive cloud intermediate tensors.

        Parameters are derived from this marker's own
        ``scheduler_output`` (num_tokens) and worker config
        (sp_chunk), NOT from a pre-stored closure. This is the
        same approach as
        :meth:`NPUWorker._execute_model_edge_tail`.
        """
        edge_sp = enable_sp()
        edge_merge = get_edge_cloud_tensor_meta().merge_payload
        dp_rank = self.worker.local_rank
        num_tokens = (
            self.bundle.scheduler_output.total_num_scheduled_tokens)
        logger.info(
            "[PD] LAST recv: dp_rank=%d num_tokens=%d src=%d",
            dp_rank, num_tokens, dp_rank + 1)
        # Wait for the FIRST round's isend on this channel to
        # complete before we recv the cloud's response
        # (mirrors NPUWorker.execute_model L564-576).
        channel = self.worker._hidden_channel_for(
            self.bundle.scheduler_output)
        self.worker._wait_pp_send_work(channel)
        tensor_dict, comm_handles, comm_postprocess = (
            edge_cloud_broadcast_recv(
                num_tokens=num_tokens,
                sp_chunk=edge_sp and edge_merge,
                src=dp_rank + 1,
                channel=channel,
            ))
        return AsyncIntermediateTensors(
            tensor_dict,
            comm_handles=comm_handles,
            comm_postprocess=comm_postprocess,
        )


class SharedModelEdgeWorker(NPUWorker):
    """Edge worker that shares one ``nn.Module`` across virtual DP workers.

    See module-level docstring for the design. The class follows the
    same RPC contract as :class:`NPUWorker` so that the existing
    :class:`MultiprocExecutor` can drive it through ``collective_rpc``
    without changes.

    KV cache global remap (non-embedding_only)
    ------------------------------------------
    When the edge runs in ``head_tail`` mode (i.e. not ``embedding_only``),
    every dp_rank worker gets a per-dp_rank ``KVCacheConfig`` whose
    ``num_blocks`` reflects only that dp_rank's share of the global
    pool. To allow a single batched forward to address the union of all
    dp_rank KV blocks (one global buffer):

    1. Each worker's ``BatchedModelRunner.initialize_kv_cache`` registers
       its ``KVCacheConfig`` in the class-level
       :attr:`BatchedModelRunner._KV_CACHE_CONFIGS_PER_DP_RANK[dp_rank]`.
    2. The LAST worker to enter ``initialize_kv_cache`` detects
       ``len(...) == data_parallel_size`` and invokes
       :meth:`allocate_global_kv_cache_tensors` once to build a global
       KV buffer of size ``sum(num_blocks_per_dp)``.
    3. The buffer is then shared across all registered workers via a
       shared ``self.kv_caches`` reference, and the forward context is
       bound exactly once (avoids duplicated work).

    The "last caller" is whichever dp_rank worker happens to be the
    final one to register; it does NOT have to be the leader.

    Note: the sync state lives on ``BatchedModelRunner`` (not on
    ``SharedModelEdgeWorker``) because it conceptually belongs to the
    model runner — the worker class doesn't need to know about
    ``KVCacheConfig`` or the construction protocol.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
            **kwargs,
        )
        # ``SharedModelEdgeWorker`` is only valid in the
        # shared-model edge-cloud topology: the worker must be on
        # the edge side and the edge must have exactly one NPU
        # (i.e. ``is_shared_model_edge`` is True). Using this
        # worker on the cloud side, or on a multi-NPU edge, would
        # silently produce incorrect PP routing.
        if not vllm_config.parallel_config.is_shared_model_edge:
            raise RuntimeError(
                "SharedModelEdgeWorker can only be used in the "
                "shared-model edge-cloud topology "
                "(edge_npu_count == 1 across the whole world). "
                "The current parallel_config has "
                f"is_shared_model_edge=False "
                f"(edge_npu_count="
                f"{vllm_config.parallel_config.edge_npu_count}, "
                f"cloud_npu_count="
                f"{vllm_config.parallel_config.cloud_npu_count}, "
                f"data_parallel_size="
                f"{vllm_config.parallel_config.data_parallel_size}); "
                "use a regular NPUWorker instead.")
        if not vllm_config.parallel_config.is_edge_node:
            raise RuntimeError(
                "SharedModelEdgeWorker is for the edge side of an "
                "edge-cloud configuration; the current process has "
                "is_edge_node=False. Use a regular NPUWorker on the "
                "cloud side.")
        # local_rank doubles as the worker's dp_rank in this design.
        self._is_leader: bool = (self.local_rank == 0)
        # Published by the leader in load_model; read by followers.
        self._shared_model: nn.Module | None = None
        # Published by the leader in determine_available_memory; read
        # by followers so they can return the same per-worker share
        # without redoing the memory profiling.
        self._per_worker_kv_cache_memory: int | None = None
        _SHARED_MODEL_REGISTRY.append(self)

    # --------------------------------------------------------- init_device
    def init_device(self) -> None:
        """Set up the NPU device and (for all virtual workers) the
        :class:`BatchedModelRunner`.

        The leader runs the full one-shot initialisation (device set,
        memory snapshot, distributed init, workspace). Followers reuse
        the leader's ``self.device`` and skip those steps because they
        are process-wide; they only construct their own
        :class:`BatchedModelRunner`. The shared model is bound later in
        :meth:`load_model`.
        """
        if self._is_leader:
            self.device = self._init_device()
            from vllm.v1.worker.workspace import init_workspace_manager
            init_workspace_manager(self.device, num_ubatches=1)
        else:
            # Reuse the leader's device: the process has one NPU
            # card and the leader has already called set_device.
            leader = get_leader_worker()
            assert leader is not None, (
                "SharedModelEdgeWorker follower constructed before "
                "the leader; ensure local_rank=0 is constructed first.")
            self.device = leader.device
            # ``_init_device`` is leader-only, so followers do not
            # have ``init_snapshot`` / ``requested_memory``;
            # inherit them from the leader for
            # ``determine_available_memory``.
            self.init_snapshot = leader.init_snapshot
            self.requested_memory = leader.requested_memory

        # All virtual workers construct their own model_runner; the
        # shared model is bound later in load_model. BatchedModelRunner's
        # constructor does not depend on the model object.
        if self.use_v2_model_runner:
            logger.info("v2 is not supported for SharedModelEdgeWorker")
        self.model_runner = BatchedModelRunner(self.vllm_config, self.device, self.local_rank)
        
        if self._is_leader:
            # Initialize edge-cloud tensor metadata for optimized communication
            # (skips inter-node metadata sync in irecv_tensor_dict/isend_tensor_dict)
            if getattr(self.model_runner, '_edge_cloud_enabled', False):
                hidden_size = self.model_config.hf_text_config.hidden_size
                # Derive dtype directly from model config (same as MindIE's
                # self.config.torch_dtype from config.json), instead of
                # requiring a separate user-configured hidden_dtype.
                # model_config.dtype is a torch.dtype resolved from the
                # model's config.json torch_dtype field by _get_and_verify_dtype().
                hidden_dtype = self.model_config.dtype
                has_residual = _detect_has_residual(self.model_config)
                # DeepSeek V4 uses hc_mult > 1 (HC mechanism produces 3D
                # intermediate tensors).  Standard models (Qwen3.5, Llama,
                # etc.) do not have hc_mult, defaulting to 1 (2D tensors).
                hc_mult = getattr(self.model_config.hf_text_config, 'hc_mult', 1)
                init_edge_cloud_tensor_meta(
                    hidden_size=hidden_size,
                    hidden_dtype=hidden_dtype,
                    has_residual=has_residual,
                    hc_mult=hc_mult,
                    mode=self.model_runner.edge_cloud_cfg.mode,
                )

    # ----------------------------------------------- distributed env (leader)
    def _init_worker_distributed_environment(self) -> None:
        """Run the HCCL-backend distributed init once per process.

        Only the leader invokes the upstream machinery; followers inherit
        the process-level distributed state.
        """
        if not self._is_leader:
            return
        super()._init_worker_distributed_environment()

    # --------------------------------------------------------- model load
    def load_model(self) -> None:
        """Load the model (leader) or bind to the leader's model (followers).

        In the same process, the leader's ``__init__`` → ``init_device``
        → ``load_model`` runs strictly before any follower's, so the
        leader has already assigned :attr:`_shared_model` by the time
        followers need it. There is no polling — followers read
        ``_shared_model`` directly.
        """
        if self._is_leader:
            super().load_model()
            self._shared_model = self.model_runner.model
        else:
            leader = get_leader_worker()
            if leader is None or leader._shared_model is None:
                raise RuntimeError(
                    "Shared model not published by the leader worker. "
                    "Ensure SharedModelEdgeWorker instances are constructed "
                    "in local_rank order so the leader's load_model runs "
                    "before any follower's."
                )
            self.model_runner.bind_to_shared_model(leader._shared_model)
            self._shared_model = leader._shared_model
            # Inherit the leader's measured model memory usage so that
            # determine_available_memory can correctly subtract the
            # shared weight footprint.
            self.model_runner.model_memory_usage = (
                leader.model_runner.model_memory_usage)

    # ------------------------------------------------- execute_model / PP
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """Run one scheduler step and route PP communication to the
        cloud first-worker for this virtual worker's dp_rank.

        Mirrors :meth:`NPUWorker.execute_model` exactly. The only
        divergence is the two ``edge_cloud_broadcast_recv`` calls in
        the original: they are called with an explicit ``src=``
        argument (the shared-model edge worker cannot rely on the
        implicit "previous PP rank" routing — each virtual worker
        has its own cloud peer based on ``local_rank``).

        The head forward + PP send is performed synchronously;
        the tail recv + tail forward is wrapped into a zero-
        argument callable (a closure over ``self`` and
        ``scheduler_output``) and returned in lieu of the final
        result. The
        :class:`vllm.v1.executor.shared_model_multiproc_executor.SharedModelWorkerProc`
        accumulates these callables across dp_ranks and invokes
        them in batch when the round barrier is reached (just
        before the result is enqueued onto the response MQ), so
        per-dp_rank tail processing stays in lockstep. The
        ``method == "execute_model"`` filter on the dispatch
        side keeps the callable-detection unambiguous: no other
        return value of any ``SharedModelEdgeWorker`` method is
        a plain function.
        """
        from types import NoneType
        from vllm.sequence import IntermediateTensors
        from vllm_ascend import envs as envs_ascend

        # enable msMonitor to monitor the performance of vllm-ascend
        if envs_ascend.MSMONITOR_USE_DAEMON:
            from vllm_ascend.profiler.torch_npu_profiler import (
                dynamic_profile as dp,
            )
            dp.step()

        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []

        # SharedModelEdgeWorker always sits at PP rank 0 (the edge is
        # the first stage of the shared PP group), so there is no
        # upstream PP receive before the first forward.

        if self.profiler is not None:
            self.profiler.step()

        output = self.model_runner.execute_model(scheduler_output, None)
        if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput,
                               NoneType)):
            return output

        assert isinstance(output, IntermediateTensors)

        # Edge-cloud with heterogeneous SP: aggregate SP shards to full
        # sequence before cross-PP send so cloud can re-chunk by its SP.
        if enable_sp() and (self.model_runner.edge_cloud_cfg.mode != "embedding_only"
            or not self.model_runner.supports_mm_inputs):
            _gathered = self._all_gather_tensor_dict(output.tensors)
        else:
            _gathered = output.tensors
        # Send the head-layer output to the cloud first-worker of
        # ``local_rank``'s dp_rank (in-group rank
        # ``self.local_rank + 1``). The explicit ``dst=`` is required
        # because the edge sits at in-group rank 0 — without it every
        # virtual worker would send to in-group rank 1, which is only
        # correct for the first virtual worker.
        self._pp_send_work = edge_cloud_isend_tensor_dict(
            _gathered,
            dst=self.local_rank + 1,
            num_tokens=scheduler_output.total_num_scheduled_tokens,
        )

        edge_sp = enable_sp()
        edge_merge = get_edge_cloud_tensor_meta().merge_payload
        # Defer the tail recv + tail forward to the end of the
        # current round. The WorkerProc accumulates these
        # callables in ``_pending_deferred`` and invokes them
        # in batch (one ``postprocess()`` per dp_rank) when the
        # round barrier is reached, just before the result is
        # enqueued onto ``response_mqs[dp_rank]``. This keeps
        # the per-dp_rank streams in lockstep — the cloud's
        # middle forward can run concurrently with the edge's
        # head forwards for subsequent dp_ranks, and the edge's
        # tail recvs happen in lockstep with the round
        # boundary.
        def _tail_postprocess():
            # Receive the cloud's middle-layer result and run
            # the second forward (tail layers). The cloud peer
            # is at in-group rank ``self.local_rank + 1``.
            tensor_dict, comm_handles, comm_postprocess = (
                edge_cloud_broadcast_recv(
                    num_tokens=scheduler_output.total_num_scheduled_tokens,
                    sp_chunk=edge_sp and edge_merge,
                    src=self.local_rank + 1))
            intermediate_tensors = AsyncIntermediateTensors(
                tensor_dict,
                comm_handles=comm_handles,
                comm_postprocess=comm_postprocess,
            )
            tail_output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors)
            if isinstance(tail_output,
                          (ModelRunnerOutput, AsyncModelRunnerOutput,
                           NoneType)):
                return tail_output
            # Edge path in the original NPUWorker.execute_model
            # always returns after the second forward — the
            # trailing KV-connector passthrough is for non-edge/
            # non-cloud middle PP stages, which never run for
            # SharedModelEdgeWorker.
            assert isinstance(tail_output, IntermediateTensors)
            return tail_output

        return DeferredExecutePostprocess(postprocess=_tail_postprocess)

    # ------------------------------------------------- batched path entry
    def execute_model_batched_pre(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> "_BatchedExecuteMarker | ModelRunnerOutput | None":
        """Per-dp_rank preprocess entry for the batched compute path.

        This is a NEW interface added next to the existing
        :meth:`execute_model` (which is kept unchanged for
        backward-compat with ``MultiprocExecutor`` and non-batched
        callers). The shared model ``worker_busy_loop`` routes
        ``execute_model`` RPCs here for batched rounds; the original
        head + send path is **not** taken.

        Behaviour:
        - Drain any in-flight ``_pp_send_work`` and step the profiler.
        - Call :meth:`BatchedModelRunner.execute_model_pre` to update
          ``self.input_batch`` and obtain an
          :class:`_ExecuteModelBundle`.
        - For empty / no-work cases ``execute_model_pre`` returns
          ``EMPTY_MODEL_RUNNER_OUTPUT`` / ``None`` directly (mirroring
          the original ``execute_model`` early returns); the
          ``worker_busy_loop`` routes these through ``handle_output``.
        - Otherwise return a :class:`_BatchedExecuteMarker` carrying the
          bundle; the busy_loop intercepts it and orchestrates the
          batched head / tail / per-dp_rank post via
          ``BatchedModelRunner.execute_model_batched_head`` /
          ``_tail`` / ``_post_batched``.
        """
        from vllm_ascend import envs as envs_ascend

        if envs_ascend.MSMONITOR_USE_DAEMON:
            from vllm_ascend.profiler.torch_npu_profiler import (
                dynamic_profile as dp,
            )
            dp.step()
        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []
        if self.profiler is not None:
            self.profiler.step()

        result = self.model_runner.execute_model_pre(scheduler_output)
        if isinstance(result, _ExecuteModelBundle):
            return _BatchedExecuteMarker(bundle=result, worker=self)
        # ``execute_model_pre`` may short-circuit on the no-work paths
        # (EC transfer producer, empty scheduler, ...) and return
        # ``EMPTY_MODEL_RUNNER_OUTPUT`` / ``None`` directly; pass it
        # through unchanged so the busy_loop routes it via
        # ``handle_output``.
        return result

    # ------------------------------------------------- PD-separation entries
    def execute_model_head_pre(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> "_FirstRoundMarker | ModelRunnerOutput | None":
        """FIRST-round entry for PD separation: full preprocess,
        return :class:`_FirstRoundMarker`.

        Logic mirrors :meth:`execute_model_batched_pre` — the only
        difference is the marker type returned to the busy_loop.
        The FIRST marker signals to the busy_loop that only
        Phase A (head + isend) is needed; no recv / tail / logits.
        """
        from vllm_ascend import envs as envs_ascend

        if envs_ascend.MSMONITOR_USE_DAEMON:
            from vllm_ascend.profiler.torch_npu_profiler import (
                dynamic_profile as dp,
            )
            dp.step()
        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []
        if self.profiler is not None:
            self.profiler.step()

        bt = getattr(scheduler_output, "batch_type", None)
        num_tokens = scheduler_output.total_num_scheduled_tokens
        logger.info(
            "[PD] execute_model_head_pre: dp_rank=%d batch_type=%s "
            "num_tokens=%d",
            self.local_rank, bt, num_tokens)
        result = self.model_runner.execute_model_pre(scheduler_output)
        if isinstance(result, _ExecuteModelBundle):
            return _FirstRoundMarker(bundle=result, worker=self)
        return result

    def execute_model_tail_pre(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> "_LastRoundMarker | ModelRunnerOutput | None":
        """LAST-round entry for PD separation: full preprocess + recv + tail.

        Calls :meth:`BatchedModelRunner.execute_model_pre` to get a
        REAL bundle (with attention metadata built from the LAST
        request's own SO), NOT a placeholder — matching the
        multi-card :meth:`NPUWorker._execute_model_edge_tail`
        approach where ``execute_model`` is called independently
        for each stage.
        """
        # Drain in-flight PP sends from any preceding FIRST round
        # before the LAST round's preprocessing begins.
        if self._pp_send_work:
            logger.info(
                "[PD] execute_model_tail_pre: waiting for %d isend handles "
                "dp_rank=%d",
                len(self._pp_send_work), self.local_rank)
            for handle in self._pp_send_work:
                handle.wait()
            logger.info(
                "[PD] execute_model_tail_pre: all isend handles waited "
                "dp_rank=%d",
                self.local_rank)
            self._pp_send_work = []

        bt = getattr(scheduler_output, "batch_type", None)
        num_tokens = scheduler_output.total_num_scheduled_tokens
        logger.info(
            "[PD] execute_model_tail_pre: dp_rank=%d batch_type=%s "
            "num_tokens=%d",
            self.local_rank, bt, num_tokens)
        result = self.model_runner.execute_model_pre(scheduler_output)
        if isinstance(result, _ExecuteModelBundle):
            return _LastRoundMarker(bundle=result, worker=self)
        return result

    def make_batched_recv_closure(
        self,
        src: int,
        num_tokens: int,
        sp_chunk: bool,
    ):
        """Return a no-arg closure that receives the cloud middle
        output for the batched compute path.

        The closure is created here (in vllm_ascend) and stored by
        the busy_loop in ``_pending_deferred``; the upstream busy_loop
        does not import vllm_ascend and just calls the closure like
        any other deferred marker.

        ``src`` is the in-group rank of the cloud first-worker for
        this dp_rank (``local_rank + 1`` on the shared PP group).
        ``num_tokens`` / ``sp_chunk`` mirror the recv call in
        :meth:`SharedModelEdgeWorker.execute_model`'s
        ``_tail_postprocess``. The closure stores the received
        ``comm_postprocess`` handles on the returned
        ``AsyncIntermediateTensors``; the batched tail call in the
        busy_loop awaits them via ``AsyncIntermediateTensors``
        (mirrors the original ``DeferredExecutePostprocess``
        semantics).
        """
        def _recv():
            tensor_dict, comm_handles, comm_postprocess = (
                edge_cloud_broadcast_recv(
                    num_tokens=num_tokens,
                    sp_chunk=sp_chunk,
                    src=src))
            return AsyncIntermediateTensors(
                tensor_dict,
                comm_handles=comm_handles,
                comm_postprocess=comm_postprocess,
            )
        return _recv

    # ------------------------------------------- memory / compile / warmup
    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Per-virtual-worker share of the available KV-cache memory.

        Whichever virtual worker is called first performs the
        actual memory profiling (via ``profile_run``); all
        subsequent callers (regardless of ``local_rank``) return
        the per-worker share already cached on the registry.
        This avoids requiring any specific ``local_rank`` to be
        called first while still ensuring the expensive profile
        run executes exactly once.
        """
        # Fast path: already computed by some virtual worker in
        # this process. Each virtual worker still needs to run
        # ``profile_run`` on its own model_runner to initialise
        # its compiled artifacts.
        for w in _SHARED_MODEL_REGISTRY:
            if w._per_worker_kv_cache_memory is not None:
                self.model_runner.profile_run()
                return w._per_worker_kv_cache_memory

        # Slow path: we are the first caller. Do the actual
        # profiling and divide by dp_size.
        if self.cache_config.kv_cache_memory_bytes:
            self._per_worker_kv_cache_memory = int(
                self.cache_config.kv_cache_memory_bytes
                // self.parallel_config.data_parallel_size)
            return self._per_worker_kv_cache_memory

        from vllm.utils.mem_utils import memory_profiling

        weights_memory = int(self.model_runner.model_memory_usage)
        with memory_profiling(self.init_snapshot,
                              weights_memory=weights_memory) as result:
            self.model_runner.profile_run()
            profile_torch_peak = torch.npu.memory_stats(
                self.device).get("allocated_bytes.all.peak", 0)

        result.torch_peak_increase = (
            profile_torch_peak - result.before_profile.torch_peak)
        result.non_kv_cache_memory = (
            result.non_torch_increase + result.torch_peak_increase
            + result.weights_memory)

        free_gpu_memory = result.after_profile.free_memory
        if self.init_snapshot.free_memory <= free_gpu_memory:
            raise RuntimeError(
                "Error in memory profiling: free memory increased.")
        available = int(self.requested_memory - result.non_kv_cache_memory)
        self._per_worker_kv_cache_memory = (
            available // self.parallel_config.data_parallel_size)
        # For embedding_only edge, the edge device does not actually store KV
        # cache tensors. Return a very large virtual value so that
        # get_kv_cache_configs() does not clamp num_blocks to the edge's
        # (small) available memory. The real num_blocks is determined by cloud.
        if (
            self.model_runner.edge_cloud_cfg.enabled
            and self.model_runner.edge_cloud_cfg.mode == "embedding_only"
            and self.model_runner.edge_cloud_cfg.role == "edge"
        ):
            self._per_worker_kv_cache_memory = 1 << 40  # 1 TiB virtual
        self.available_kv_cache_memory_bytes = self._per_worker_kv_cache_memory
        logger.info(
            "SharedModelEdgeWorker[local_rank=%d] per-worker KV cache "
            "memory: %.2f GiB", self.local_rank,
            self._per_worker_kv_cache_memory / (1024 ** 3))
        return self._per_worker_kv_cache_memory

    def execute_dummy_batch(self) -> None:
        """No-op for all virtual workers.

        NPU graphs are captured in
        :meth:`compile_or_warm_up_model`; in a single-process design
        there is no need to re-run a decode-only dummy per DP rank.
        """
        return None

    # --------------------------------------------------------- sleep/wake
    # TODO(shared-model): these overrides assume a single dp rank on
    # the edge. With multiple dp ranks sharing one process, the
    # leader's sleep/wake_up only acts on the leader's view of the
    # shared ``nn.Module``; followers' sleep/wake_up calls are
    # silently dropped. Verify that this is consistent with the
    # downstream / upstream CaMemAllocator state for all dp ranks,
    # and add a barrier (e.g. a flag in ``_SHARED_MODEL_REGISTRY``)
    # if a coordinated sleep/wake_up across virtual workers is
    # required by the upper layer.
    def sleep(self, level: int = 1) -> None:
        """Sleep: leader runs the standard offload; followers no-op.

        The model is shared across virtual workers, so weights are
        offloaded exactly once by the leader. Followers inherit the
        offloaded state through the shared ``nn.Module``.
        """
        if not self._is_leader:
            return
        super().sleep(level=level)

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake: leader runs the standard restore; followers no-op.

        See :meth:`sleep` for the rationale on the leader-only
        behaviour.
        """
        if not self._is_leader:
            return
        super().wake_up(tags=tags)

    # --------------------------------------------- static kernel cleanup
    def uninstall_static_kernel(self) -> None:
        """Uninstall the static ATB kernel at shutdown (leader only)."""
        if not self._is_leader:
            return
        super().uninstall_static_kernel()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
        from vllm_ascend.device_allocator.camem import CaMemAllocator
        """Allocate NPU KV cache with the specified kv_cache_config."""
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CaMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext

            context = nullcontext()  # type: ignore
        # Register this virtual worker's ``KVCacheConfig`` under
        # ``self.local_rank`` BEFORE the runner inspects the registry.
        # ``self.local_rank`` (the unique virtual worker identifier in
        # the shared-model edge topology) is the correct key here —
        # ``self.parallel_config.data_parallel_rank`` is identical
        # for every virtual worker on the edge (they all share one
        # NPU and one distributed rank), so it would alias every
        # dp_rank's entry.
        BatchedModelRunner._KV_CACHE_CONFIGS_PER_DP_RANK[self.local_rank] = (
            kv_cache_config
        )
        with context:
            self.model_runner.initialize_kv_cache(kv_cache_config)
            if BatchedModelRunner._KV_CACHE_CONSTRUCTED:
                for w in _SHARED_MODEL_REGISTRY:
                    # Mirror shared last-caller state onto every
                    # follower's model_runner. Followers will use
                    # these for batched forward (slot_mapping /
                    # block_tables need per_dp_offsets; attn_groups
                    # is required for ``use_hybrid_blocks`` etc.).
                    w.model_runner._per_dp_offsets = self.model_runner._per_dp_offsets
                    w.model_runner._per_dp_num_blocks = self.model_runner._per_dp_num_blocks
                    w.model_runner._global_num_blocks = self.model_runner._global_num_blocks
                    w.model_runner.kv_caches = self.model_runner.kv_caches
                    w.model_runner.hybrid_with_attn_and_mamba = self.model_runner.hybrid_with_attn_and_mamba
                    w.model_runner.initialize_kv_cache_post()
