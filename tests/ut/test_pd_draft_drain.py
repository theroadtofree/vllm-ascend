# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DRAFT_FIRST -> DRAFT_LAST alternation invariant and the
drain path that pairs the cloud's DRAFT_LAST response when the owning request
finishes or is aborted mid-chain.

Regression coverage for the edge-side MTP draft deadlock fixes:

  * ``_can_schedule_draft_first`` permits one pre-generated look-ahead head.
    The worker suspends it if its preceding DRAFT_LAST has not materialized
    ``last_draft_*`` yet, then resumes it from that tail without blocking the
    worker queue.
  * ``_is_stale_draft_output`` no longer exempts pre-generated dispatched
    chains: once the owning request is gone, future (not-yet-dispatched)
    DRAFT_FIRST heads are stale and must be skipped -- the edge can no longer
    produce their payload (draft context cleared) and the cloud would otherwise
    wait forever for data.
  * ``_pick_draft_last_batch`` never drops a dispatched DRAFT_LAST; it drains
    it (the cloud always ``isend``s a response, so the edge must ``irecv`` to
    keep the DECODE channel paired) and only spawns a verify placeholder for a
    live request.
  * ``_run_edge_cloud_draft_last_segment`` drains (recv already done by the
    caller, skip tail compute, return a token-less placeholder) when the draft
    context is gone, instead of raising.
  * Every middle prefill chunk runs a complete draft chain to populate MTP KV,
    while its proposals are discarded and no target verify placeholder is
    created.
"""

from collections import deque
from unittest.mock import MagicMock

import pytest

from vllm.v1.core.sched.output import (
    BatchType,
    HiddenChannelType,
    SchedulerOutput,
)


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #


def _make_bare_scheduler():
    from vllm_ascend.core.pd_separated_scheduler import PDSeparatedScheduler

    s = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
    s.drafts_first_ready = deque()
    s.drafts_last_ready = deque()
    s.requests = {}
    s._pregenerated_draft_task_ids = set()
    s._pregenerated_draft_req_ids = {}
    s._draft_first_dispatched = False
    s._draft_first_cloud_publish_pending = None
    s._draft_first_scalars_patched = False
    s._draft_remote_pending_limit = 2
    s.draft_remote_pending_count = 0
    s.decode_or_draft_inflight_count = 0
    s.decode_or_draft_inflight_limit = 1
    s.decode_head_inflight_count = 0
    s._force_draft_last = False
    s._force_decode_last = False
    s.num_spec_tokens = 3
    return s


def _make_draft_first(task_id="task-0", req_id="req-0", step=0):
    so = MagicMock()
    so.batch_type = BatchType.PREFILL_DRAFT_FIRST
    so.draft_task_id = task_id
    so.draft_step_idx = step
    so.head_token = f"tok-{task_id}-{step}"
    so.hidden_channel = HiddenChannelType.DECODE
    so.num_scheduled_tokens = {req_id: 1}
    so.parent_req_id = req_id
    so.num_accepted_tokens = None
    so.valid_sampled_token_count = None
    so.is_last_prefill_chunk = True
    so.draft_output_req_ids = (req_id,)
    return so


def _make_draft_last(task_id="task-0", req_id="req-0", step=0):
    so = MagicMock()
    so.batch_type = BatchType.PREFILL_DRAFT_LAST
    so.draft_task_id = task_id
    so.draft_step_idx = step
    so.head_token = f"tok-{task_id}-{step}"
    so.hidden_channel = HiddenChannelType.DECODE
    so.num_scheduled_tokens = {req_id: 1}
    so.parent_req_id = req_id
    so.is_last_prefill_chunk = True
    so.draft_output_req_ids = (req_id,)
    return so


def _make_real_output(
    batch_type=BatchType.PREFILL_LAST,
    task_id="task-0",
    req_id="req-0",
):
    so = SchedulerOutput.make_empty()
    so.batch_type = batch_type
    so.head_token = task_id
    so.num_scheduled_tokens = {req_id: 8}
    so.total_num_scheduled_tokens = 8
    return so


# ------------------------------------------------------------------ #
# Test: _can_schedule_draft_first honors _force_draft_last (fix 鈶?   #
# ------------------------------------------------------------------ #


class TestCanScheduleDraftFirstForceGuard:
    """The pre-generated branch must gate on _force_draft_last just like the
    legacy branch, so DRAFT_FIRST -> DRAFT_LAST alternation is guaranteed."""

    def _setup(self, pregenerated=True):
        s = _make_bare_scheduler()
        drf = _make_draft_first()
        s.drafts_first_ready.append(drf)
        if pregenerated:
            s._pregenerated_draft_task_ids.add(drf.draft_task_id)
        # Conditions that would otherwise allow scheduling.
        s.drafts_last_ready = deque()
        s._force_decode_last = False
        s._force_draft_last = False
        s.draft_remote_pending_count = 0
        s.decode_or_draft_inflight_count = 0
        return s

    def test_preGen_blocked_when_force_draft_last_true(self):
        s = self._setup(pregenerated=True)
        s._force_draft_last = True
        assert s._can_schedule_draft_first() is False

    def test_preGen_allowed_when_force_draft_last_false(self):
        s = self._setup(pregenerated=True)
        s._force_draft_last = False
        assert s._can_schedule_draft_first() is True

    def test_preGen_blocked_by_drafts_last_ready(self):
        s = self._setup(pregenerated=True)
        s._force_draft_last = False
        s.drafts_last_ready.append(_make_draft_last())
        assert s._can_schedule_draft_first() is False

    def test_preGen_blocked_when_decode_head_in_flight(self):
        """Regression for decode_or_draft_inflight=2/1: DRAFT_FIRST and
        DECODE_FIRST use different recv primitives but share the DECODE
        stream, so a DRAFT_FIRST must not be dispatched while a DECODE_FIRST
        head is in flight (the cloud's recv order could mismatch the edge's
        send order).  Gate on decode heads, not total heads."""
        s = self._setup(pregenerated=True)
        s._force_draft_last = False
        s.decode_head_inflight_count = 1  # a DECODE_FIRST in flight
        assert s._can_schedule_draft_first() is False
        s.decode_head_inflight_count = 0
        assert s._can_schedule_draft_first() is True

    def test_preGen_allows_one_deferred_lookahead_head(self):
        """The worker-side suspended-head state makes one-step lookahead safe."""
        s = self._setup(pregenerated=True)
        s._force_draft_last = False  # previous DRAFT_LAST already picked
        s.drafts_last_ready = deque()  # previous DRAFT_LAST popped (in flight)
        s.decode_head_inflight_count = 0  # no DECODE_FIRST in flight
        s.draft_remote_pending_count = 1  # one slot remains under limit=2
        assert s._can_schedule_draft_first() is True

        s.draft_remote_pending_count = 2
        assert s._can_schedule_draft_first() is False

    def test_legacy_branch_also_blocked_by_force_draft_last(self):
        """Non-pre-generated branch already had the guard (unchanged)."""
        s = self._setup(pregenerated=False)
        s._force_draft_last = True
        assert s._can_schedule_draft_first() is False


class TestDraftFirstLastAlternation:
    """A second head is admitted after the first tail is picked, not before."""

    def test_second_head_blocked_until_tail_picked(self):
        s = _make_bare_scheduler()
        s._pregenerated_draft_task_ids.add("task-0")
        # step-0 head already picked; step-1 is next in drafts_first_ready.
        s.drafts_first_ready.append(_make_draft_first(step=1))
        s._force_decode_last = False
        s.draft_remote_pending_count = 1

        # DRAFT_FIRST step-0 in force + its DRAFT_LAST pending -> step-1 blocked.
        s._force_draft_last = True
        s.drafts_last_ready.append(_make_draft_last(step=0))
        assert s._can_schedule_draft_first() is False

        # Once the tail is picked, worker-side suspension safely carries the
        # causal dependency until that tail actually completes.
        s._force_draft_last = False
        s.drafts_last_ready.clear()
        assert s._can_schedule_draft_first() is True

    def test_lookahead_tail_waits_for_preceding_tail_completion(self):
        s = _make_bare_scheduler()
        s._pregenerated_draft_task_ids.add("task-0")
        s._draft_last_delay_start_ts = None
        s.drafts_last_ready.append(_make_draft_last(step=1))

        # Both step 0 and the look-ahead step 1 are still remote.
        s.draft_remote_pending_count = 2
        assert s._can_schedule_draft_last() is False

        # Completion of step 0 releases only step 1's remote credit.
        s.draft_remote_pending_count = 1
        assert s._can_schedule_draft_last() is True


class TestDeferredDraftHeadPipeline:
    """Worker-side suspended heads preserve E(n) -> A(n+1) causality."""

    @staticmethod
    def _make_worker():
        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        worker.model_runner = MagicMock()
        worker._deferred_edge_draft_heads = {}
        worker._run_and_send_edge_draft_head = MagicMock()
        return worker

    def test_unready_head_is_suspended_without_running_segment_a(self):
        worker = self._make_worker()
        worker.model_runner.is_edge_cloud_draft_step_ready.return_value = (
            False
        )
        head = _make_draft_first(step=1)

        output = worker._execute_model_edge_draft_head(head)

        assert output.req_ids == ["req-0"]
        assert worker._deferred_edge_draft_heads[("task-0", 1)] is head
        worker._run_and_send_edge_draft_head.assert_not_called()
        worker.model_runner.mark_edge_cloud_draft_step_deferred.assert_called_once_with(
            head
        )

    def test_preceding_tail_resumes_suspended_head_once(self):
        worker = self._make_worker()
        worker.model_runner.is_edge_cloud_draft_step_ready.return_value = True
        head = _make_draft_first(step=1)
        worker._deferred_edge_draft_heads[("task-0", 1)] = head

        worker._resume_deferred_edge_draft_head(_make_draft_last(step=0))

        assert worker._deferred_edge_draft_heads == {}
        worker.model_runner.unmark_edge_cloud_draft_step_deferred.assert_called_once_with(
            head
        )
        worker._run_and_send_edge_draft_head.assert_called_once_with(head)

        worker._resume_deferred_edge_draft_head(_make_draft_last(step=0))
        worker._run_and_send_edge_draft_head.assert_called_once_with(head)


class TestMidPrefillDraftChain:
    """Every chunk warms MTP KV, but only the last chunk may seed verify."""

    def test_pick_mid_prefill_tail_starts_draft_chain(self):
        s = _make_bare_scheduler()
        target = _make_real_output()
        flight = MagicMock()
        flight.is_last_chunk = False
        s.prefills_last_ready = deque([target])
        s._prefill_flight_by_token = {target.head_token: flight}
        s.chunk_prefill_first = []
        s._validate_prefill_tail_channel = MagicMock()
        s._pregenerate_draft_chain = MagicMock()

        result = s._pick_prefill_last_batch()

        assert result is target
        assert result.is_last_prefill_chunk is False
        assert result.draft_output_req_ids == ()
        s._pregenerate_draft_chain.assert_called_once_with(target)

    def test_pregenerates_mid_chunk_and_preserves_marker(self):
        s = _make_bare_scheduler()
        request = MagicMock()
        request.is_finished.return_value = False
        s.requests["req-0"] = request
        s._uses_async_scheduled_mtp_placeholders = MagicMock(
            return_value=True
        )

        target = _make_real_output()
        target.is_last_prefill_chunk = False
        target.draft_output_req_ids = ()
        s._pregenerate_draft_chain(target)

        assert len(s.drafts_first_ready) == s.num_spec_tokens
        assert all(
            getattr(output, "is_last_prefill_chunk", True) is False
            for output in s.drafts_first_ready
        )
        assert all(
            output.draft_output_req_ids == ()
            for output in s.drafts_first_ready
        )

    def test_mid_chunk_draft_tail_does_not_prepare_verify(self):
        s = _make_bare_scheduler()
        request = MagicMock()
        request.is_finished.return_value = False
        s.requests["req-0"] = request
        tail = _make_draft_last()
        tail.is_last_prefill_chunk = False
        tail.draft_output_req_ids = ()
        s.drafts_last_ready.append(tail)
        s._force_draft_last = True
        s._validate_draft_tail_channel = MagicMock()
        s._start_decode_or_draft_first_only_window = MagicMock()
        s._prepare_next_decode_first_placeholder = MagicMock()

        assert s._pick_draft_last_batch() is tail
        s._prepare_next_decode_first_placeholder.assert_not_called()

    def test_engine_advances_mid_prefill_draft(self):
        from vllm_ascend.patch.platform.patch_engine_core import (
            _advance_edge_cloud_draft,
        )

        engine = MagicMock()
        engine.use_spec_decode = True
        finalized = _make_real_output(BatchType.PREFILL_DRAFT_FIRST)
        engine.scheduler.finalize_pre_generated_draft_first.return_value = (
            finalized
        )
        completed = _make_real_output()
        completed.is_last_prefill_chunk = False
        model_output = MagicMock()
        model_output.edge_cloud_draft_state = {
            "draft_task_id": completed.head_token,
            "draft_step_idx": 0,
        }
        model_output.sampled_token_ids = [[]]

        _advance_edge_cloud_draft(engine, completed, model_output)

        engine.scheduler.finalize_pre_generated_draft_first.assert_called_once_with(
            draft_task_id=completed.head_token,
            num_accepted_tokens=[0],
            valid_sampled_token_count=[0],
        )
        engine._release_deferred_draft_pre_out.assert_called_once_with(
            completed.head_token
        )
        engine.scheduler.enqueue_draft_first.assert_not_called()

    def test_registers_worker_created_fallback_draft(self):
        from vllm_ascend.patch.platform.patch_engine_core import (
            _register_edge_cloud_draft_parent,
        )

        engine = MagicMock()
        engine.use_spec_decode = True
        engine._uses_scheduled_edge_cloud_draft.return_value = True
        completed = _make_real_output()
        model_output = MagicMock()
        model_output.edge_cloud_draft_state = {
            "draft_task_id": completed.head_token,
            "draft_step_idx": 0,
        }

        _register_edge_cloud_draft_parent(engine, completed, model_output)

        engine.scheduler.register_edge_cloud_draft_task.assert_called_once_with(
            completed.head_token, {"req-0"}
        )

    def test_does_not_register_without_worker_draft_state(self):
        from vllm_ascend.patch.platform.patch_engine_core import (
            _register_edge_cloud_draft_parent,
        )

        engine = MagicMock()
        engine.use_spec_decode = True
        engine._uses_scheduled_edge_cloud_draft.return_value = True
        completed = _make_real_output()
        model_output = MagicMock(spec=[])

        _register_edge_cloud_draft_parent(engine, completed, model_output)

        engine.scheduler.register_edge_cloud_draft_task.assert_not_called()


# ------------------------------------------------------------------ #
# Test: _draft_output_reqs_live / _is_stale_draft_output (fix 鈶)    #
# ------------------------------------------------------------------ #


class TestDraftReqsLiveAndStale:
    def _setup(self, req_present=True, req_id="req-0"):
        s = _make_bare_scheduler()
        if req_present:
            s.requests[req_id] = MagicMock()
        so = _make_draft_last(req_id=req_id)
        return s, so

    def test_live_request_reqs_live_true(self):
        s, so = self._setup(req_present=True)
        assert s._draft_output_reqs_live(so) is True

    def test_dead_request_reqs_live_false(self):
        s, so = self._setup(req_present=False)
        assert s._draft_output_reqs_live(so) is False

    def test_live_request_not_stale(self):
        s, so = self._setup(req_present=True)
        assert s._is_stale_draft_output(so) is False

    def test_dead_request_stale(self):
        s, so = self._setup(req_present=False)
        assert s._is_stale_draft_output(so) is True

    def test_preGen_dispatched_dead_request_still_stale(self):
        """Regression for the removed override: a pre-generated, step-0-
        dispatched chain whose request has since gone must still be treated
        as stale for not-yet-dispatched heads.  Otherwise the edge picks and
        dispatches heads it can no longer produce payload for (context
        cleared) -> cloud waits forever for data that never arrives."""
        s, so = self._setup(req_present=False)
        s._pregenerated_draft_task_ids.add(so.draft_task_id)
        s._draft_first_dispatched = True
        assert s._is_stale_draft_output(so) is True


# ------------------------------------------------------------------ #
# Test: _pick_draft_last_batch drains instead of dropping (fix 鈶)   #
# ------------------------------------------------------------------ #


class TestPickDraftLastBatchDrain:
    """A DRAFT_LAST in drafts_last_ready always has its DRAFT_FIRST already
    dispatched to the cloud, so the cloud will isend a response -- the edge
    must execute (drain) the tail to pair it, never drop it."""

    def _setup(self, req_present=True):
        s = _make_bare_scheduler()
        if req_present:
            s.requests["req-0"] = MagicMock()
        s.drafts_last_ready.append(_make_draft_last())
        s._force_draft_last = True
        s.draft_remote_pending_count = 1
        # Stub side-effecting helpers to isolate the drain decision.
        s._validate_draft_tail_channel = MagicMock()
        s._start_decode_or_draft_first_only_window = MagicMock()
        s._prepare_next_decode_first_placeholder = MagicMock()
        s._make_empty_batch = MagicMock(return_value="EMPTY")
        return s

    def test_dead_request_drained_not_dropped(self):
        s = self._setup(req_present=False)
        before = s.draft_remote_pending_count
        result = s._pick_draft_last_batch()

        # The tail is returned (dispatched to the worker for drain), not
        # dropped -- the cloud's response must be paired on the DECODE channel.
        assert result.batch_type == BatchType.PREFILL_DRAFT_LAST
        assert len(s.drafts_last_ready) == 0
        # _force_draft_last is always reset now (the old stale-drop skipped it).
        assert s._force_draft_last is False
        # The decrement moved to update_from_output; _pick_draft_last_batch no
        # longer decrements (the old stale-drop did).
        assert s.draft_remote_pending_count == before
        # No verify placeholder for a gone request.
        s._prepare_next_decode_first_placeholder.assert_not_called()
        s._validate_draft_tail_channel.assert_called_once()

    def test_live_request_prepares_placeholder(self):
        s = self._setup(req_present=True)
        result = s._pick_draft_last_batch()
        assert result.batch_type == BatchType.PREFILL_DRAFT_LAST
        assert s._force_draft_last is False
        s._prepare_next_decode_first_placeholder.assert_called_once()

    def test_empty_returns_empty_batch(self):
        s = self._setup(req_present=True)
        s.drafts_last_ready.clear()
        assert s._pick_draft_last_batch() == "EMPTY"
        s._make_empty_batch.assert_called_once()


# ------------------------------------------------------------------ #
# Test: worker _run_edge_cloud_draft_last_segment drain (fix 鈶)    #
# ------------------------------------------------------------------ #


class TestRunDraftLastSegmentDrain:
    """When the draft context is gone (request finished/aborted after its
    DRAFT_FIRST was dispatched), the tail segment must drain (the recv in
    _execute_model_edge_draft_tail already paired the cloud response) and
    return a token-less placeholder instead of raising."""

    def _make_runner(self, context_present=False):
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner._pending_edge_cloud_draft_contexts = {}
        if context_present:
            runner._pending_edge_cloud_draft_contexts["task-0"] = MagicMock()
        return runner

    @staticmethod
    def _make_so(task_id="task-0"):
        so = MagicMock()
        so.draft_task_id = task_id
        so.draft_step_idx = 0
        so.num_scheduled_tokens = {"req-0": 1}
        return so

    def test_drains_when_context_gone(self):
        from vllm.v1.outputs import ModelRunnerOutput

        runner = self._make_runner(context_present=False)
        result = runner._run_edge_cloud_draft_last_segment(
            self._make_so(), MagicMock()
        )
        assert isinstance(result, ModelRunnerOutput)
        assert result.req_ids == ["req-0"]
        assert result.req_id_to_index == {"req-0": 0}

    def test_finished_context_is_pinned_by_suspended_head(self):
        runner = self._make_runner(context_present=False)
        runner._worker_draft_token_ids_by_req = {}
        runner._pending_edge_cloud_draft_contexts["task-0"] = {
            "req_ids": ("req-0",),
            "deferred_draft_first_steps": {1},
        }

        runner.clear_pending_edge_cloud_draft_for_req_ids(
            {"req-0"}, force_drop_task_ids={"task-0"}
        )

        assert "task-0" in runner._pending_edge_cloud_draft_contexts

    def test_drains_when_task_id_none(self):
        from vllm.v1.outputs import ModelRunnerOutput

        runner = self._make_runner(context_present=False)
        result = runner._run_edge_cloud_draft_last_segment(
            self._make_so(task_id=None), MagicMock()
        )
        assert isinstance(result, ModelRunnerOutput)
        assert result.req_ids == ["req-0"]
