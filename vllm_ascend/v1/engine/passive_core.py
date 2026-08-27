#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""ZMQ scheduler channels and Passive EngineCore process for vllm-ascend.

This module is the vllm-ascend home of four classes that previously lived in
``vllm/v1/engine/core.py`` of the vllm-pdmix downstream fork:

* :class:`PPSchedulerZmqPublisher` — pp rank0 → pp rank1 SchedulerOutput
  publisher (PUSH socket + background pickling thread).
* :class:`PPSchedulerZmqSubscriber` — pp rank1 receiver counterpart.
* :class:`PPSchedulerZmqChannel` — bidirectional channel that owns one
  publisher + one subscriber for the edge-cloud PD-separation flow.
* :class:`PassiveEngineCoreProc` — non-leader PP rank engine process driver
  that consumes scheduler decisions over ZMQ instead of producing them.

All hooks back into the upstream ``EngineCore`` / ``EngineCoreProc`` lifecycle
are installed by :mod:`vllm_ascend.patch.platform.patch_engine_core` so that
the upstream ``vllm/v1/engine/core.py`` stays untouched.
"""
from __future__ import annotations

import copy
import os
import pickle
import queue
import signal
import threading
import time
from typing import TYPE_CHECKING, Optional

import numpy as np
import zmq
from vllm import envs
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.tracing import maybe_init_worker_tracer
from vllm.v1.core.sched.output import BatchType, SchedulerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Decoupled data-plane irecv coordination.  The scheduler-process glue
# (scheduler_link) forwards recv hints to the local worker's comm thread
# and answers per-channel watermark queries fed by the worker's
# completion reports.  See the schema in scheduler_link's docstring and
# vllm_ascend.patch.platform.patch_engine_core.
from vllm_ascend.distributed.edge_cloud_comm.scheduler_link import (
    post_irecv_hint,
    record_irecv_completions,
    register_hint_sender,
)

logger = init_logger(__name__)


def _import_passive_scheduler_module():
    """Lazily resolve the PassiveScheduler implementation.

    The module path ``vllm.v1.core.sched.passive_scheduler`` is aliased to the
    vllm-ascend implementation by
    :mod:`vllm_ascend.patch.platform.patch_pd_scheduler_shim`. We try that
    canonical alias first (so any callsite that legacy-imports from the vLLM
    path keeps working) and fall back to the direct ascend path otherwise.
    """
    try:
        import vllm.v1.core.sched.passive_scheduler as passive_scheduler
    except ImportError:
        try:
            import vllm_ascend.core.passive_scheduler as passive_scheduler
        except ImportError as err:
            raise RuntimeError(
                "PassiveScheduler is provided by the vllm-ascend plugin. "
                "Make sure vllm_ascend.patch.platform.patch_pd_scheduler_shim "
                "is imported before starting PassiveEngineCore."
            ) from err
    return passive_scheduler


class PPSchedulerZmqPublisher:
    """Publishes SchedulerOutput from pp rank0 EngineCore to pp rank1
    PassiveEngineCore via ZMQ PUSH/PULL pattern.

    Architecture: caller thread (scheduler loop) only enqueues the raw
    `SchedulerOutput` object into `_queue`. A dedicated background thread
    pulls from the queue, pickles, and sends over ZMQ. This keeps the
    scheduler step path free of pickling cost and mirrors the symmetric
    queue.Queue bridge used on the subscriber/PassiveScheduler side.
    """

    SHUTDOWN_TIMEOUT: float = 2.0

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._queue: queue.Queue[Optional[tuple[int, SchedulerOutput]]] = (
            queue.Queue(maxsize=1000)
        )
        self._running = True
        self._seq = 0

        # Set up ZMQ PUSH socket
        self._ctx = zmq.Context.instance()
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.set_hwm(1000)
        # Bind if wildcard (pp rank0), otherwise connect
        if "*" in endpoint or "::" in endpoint:
            self._push.bind(endpoint)
        else:
            self._push.connect(endpoint)

        logger.info("PP Scheduler ZMQ publisher started on %s", endpoint)

        # Start background publisher thread
        self._thread = threading.Thread(
            target=self._publisher_thread,
            daemon=True,
            name="pp-scheduler-zmq-pub",
        )
        self._thread.start()

    def publish(self, scheduler_output: SchedulerOutput) -> None:
        """Queue a SchedulerOutput for publishing. Non-blocking: drops the
        message if the bridge queue is full (back-pressure protection).
        """
        if not self._running or scheduler_output.batch_type is BatchType.EMPTY:
            return
        try:
            seq = self._seq
            self._seq += 1
            self._queue.put_nowait((seq, scheduler_output))
        except queue.Full:
            logger.warning(
                "PP Scheduler ZMQ publish queue full, dropping message"
            )

    # Only the PD-separation decode/draft batch types are value-free on
    # the wire. Legacy mixed types (PURE_DECODE, PD_MIX) are excluded on
    # purpose - their consumers are not guaranteed to be non-last PP
    # ranks. DECODE_LAST/DRAFT_LAST never traverse ZMQ (the edge
    # self-posts them), but listing them keeps the intent explicit.
    _VALUE_FREE_BATCH_TYPES = frozenset(
        {
            BatchType.DECODE_FIRST,
            BatchType.DECODE_LAST,
            BatchType.PREFILL_DRAFT_FIRST,
            BatchType.PREFILL_DRAFT_LAST,
            BatchType.DECODE_DRAFT_FIRST,
            BatchType.DECODE_DRAFT_LAST,
        }
    )

    def _compress_token_history(
        self, scheduler_output: SchedulerOutput
    ) -> SchedulerOutput:
        """[PD-WIRE-SLIM] strip token VALUES from decode-family
        SchedulerOutputs, keeping only the history lengths.

        The edge re-sends `all_token_ids[req_id]` (the whole prompt +
        generated history, ~2 MB) with every decode SchedulerOutput.
        The cloud only ever consumes it as a contiguous array whose
        LENGTH matters: the worker's recovery path slices
        `[-num_output_tokens:]` to rebuild request state, and a non-last
        PP rank never embeds or samples the values. DECODE/DRAFT-family
        SOs also never flow back to the edge (the cloud returns only
        PREFILL_LAST), so no consumer anywhere needs the real values.

        The wire therefore carries `all_token_ids[req_id] =
        len(history)` and the subscriber re-inflates `np.zeros(len)`.
        This shrinks the wire to near-zero AND keeps token ids off it
        (security). No cross-message state, no receive-path
        concatenate - unlike a delta scheme this costs nothing on
        either CPU.

        PREFILL-family SOs pass through untouched: their histories
        travel back inside PREFILL_LAST and the edge consumes the values
        (draft prefill warm-up, sampling penalties).
        """
        bt = getattr(scheduler_output, "batch_type", None)
        if bt not in self._VALUE_FREE_BATCH_TYPES:
            return scheduler_output
        cached = scheduler_output.scheduled_cached_reqs
        all_token_ids = getattr(cached, "all_token_ids", None)
        if not all_token_ids:
            return scheduler_output

        slimmed = {
            req_id: len(tokens) for req_id, tokens in all_token_ids.items()
        }

        # The original SO is shared with local dispatch (the edge worker
        # embeds the real tokens), so never mutate it: ship a shallow
        # copy with a copied CachedRequestData carrying only lengths.
        so_copy = copy.copy(scheduler_output)
        cached_copy = copy.copy(cached)
        cached_copy.all_token_ids = slimmed
        so_copy.scheduled_cached_reqs = cached_copy
        return so_copy

    def _publisher_thread(self) -> None:
        while self._running or self._queue.qsize() > 0:
            try:
                item = self._queue.get(timeout=0.1)
                if item is None:
                    break
                seq, scheduler_output = item
                # [PD-WIRE-SLIM] strip token values from decode-family
                # SOs before pickling, so the size probe below measures
                # the bytes actually put on the wire.
                scheduler_output = self._compress_token_history(
                    scheduler_output
                )
                try:
                    data = pickle.dumps(
                        scheduler_output, protocol=pickle.HIGHEST_PROTOCOL
                    )
                except Exception:
                    logger.exception(
                        "Failed to serialize SchedulerOutput for ZMQ"
                    )
                    continue
                seq_bytes = seq.to_bytes(8, "big")
                self._push.send_multipart((seq_bytes, data))
            except queue.Empty:
                continue
            except Exception:
                logger.exception("Error in PP scheduler ZMQ publisher thread")
                time.sleep(0.1)

    def shutdown(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
        try:
            if self._push is not None:
                self._push.close(linger=0)
        except Exception:
            pass


class PPSchedulerZmqSubscriber:
    """Receives SchedulerOutput from pp rank0 EngineCore on pp rank1
    via ZMQ PUSH/PULL pattern.

    Runs a background thread that receives SchedulerOutput messages,
    saves them locally, and logs a summary.
    """

    SHUTDOWN_TIMEOUT: float = 2.0

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._running = True
        self._received_outputs: list[tuple[int, SchedulerOutput]] = []
        self._lock = threading.Lock()

        # Set up ZMQ PULL socket
        self._ctx = zmq.Context.instance()
        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.set_hwm(1000)
        self._pull.connect(endpoint)

        logger.info("PP Scheduler ZMQ subscriber connecting to %s", endpoint)

        # Start background subscriber thread
        self._thread = threading.Thread(
            target=self._subscriber_thread,
            daemon=True,
            name="pp-scheduler-zmq-sub",
        )
        self._thread.start()

    def _expand_token_history(
        self, scheduler_output: SchedulerOutput
    ) -> None:
        """[PD-WIRE-SLIM] restore the length-only token histories sent
        by the publisher (see PPSchedulerZmqPublisher
        ._compress_token_history).

        Each decode-family entry arrives as a bare length; materialize
        a zero-filled int32 array of that length. Values are unused on
        this side (non-last PP rank: no embedding, no sampling), only
        the length feeds the recovery-path slice. np.zeros is calloc
        backed, so this stays far cheaper than the transfer it replaces.
        Entries that are already arrays (prefill-family SOs, or an
        older publisher) pass through untouched.

        Mutates the deserialized SO in place - this thread is the sole
        owner of the object until it is published via
        `_received_outputs`.
        """
        cached = scheduler_output.scheduled_cached_reqs
        all_token_ids = getattr(cached, "all_token_ids", None)
        if not all_token_ids:
            return

        rebuilt: dict[str, object] = {}
        changed = False
        for req_id, payload in all_token_ids.items():
            if isinstance(payload, int):
                rebuilt[req_id] = np.zeros(payload, dtype=np.int32)
                changed = True
            else:
                rebuilt[req_id] = payload
        if changed:
            cached.all_token_ids = rebuilt

    def _subscriber_thread(self) -> None:
        while self._running:
            try:
                if not self._pull.poll(timeout=100):
                    continue
                seq_bytes, data = self._pull.recv_multipart()
                seq = int.from_bytes(seq_bytes, "big")
                scheduler_output = pickle.loads(data)
                if scheduler_output.batch_type is BatchType.EMPTY:
                    continue
                # [PD-WIRE-SLIM] restore zero-filled token histories
                # before the SO is handed downstream (dispatch loop,
                # trim, worker) - they all keep seeing arrays of the
                # original lengths.
                self._expand_token_history(scheduler_output)

                with self._lock:
                    self._received_outputs.append((seq, scheduler_output))
                # logger.info(
                #     "PP rank1 received SchedulerOutput seq=%d, "
                #     "total_scheduled_tokens=%d, "
                #     "new_reqs=%d, cached_reqs=%d, "
                #     "finished_req_ids=%s",
                #     seq,
                #     scheduler_output.total_num_scheduled_tokens,
                #     len(scheduler_output.scheduled_new_reqs),
                #     scheduler_output.scheduled_cached_reqs.num_reqs,
                #     scheduler_output.finished_req_ids,
                # )
            except zmq.ZMQError:
                if self._running:
                    logger.exception("ZMQ error in PP scheduler subscriber")
            except Exception:
                if self._running:
                    logger.exception(
                        "Error in PP scheduler ZMQ subscriber thread"
                    )

    def get_latest_output(self) -> Optional[SchedulerOutput]:
        """Return the most recently received SchedulerOutput, or None."""
        with self._lock:
            if self._received_outputs:
                return self._received_outputs[-1][1]
        return None

    def get_all_outputs(self) -> list[tuple[int, SchedulerOutput]]:
        """Return all received (seq, SchedulerOutput) pairs."""
        with self._lock:
            return list(self._received_outputs)

    def consume_new_outputs(self) -> list[tuple[int, SchedulerOutput]]:
        """Return and clear all new (seq, SchedulerOutput) pairs since last
        call.
        """
        with self._lock:
            outputs = self._received_outputs
            self._received_outputs = []
            return outputs

    def shutdown(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
        try:
            if self._pull is not None:
                self._pull.close(linger=0)
        except Exception:
            pass


class PPSchedulerZmqChannel:
    """Bidirectional ZMQ channel for SchedulerOutput exchange between two
    PP engines.

    A `PPSchedulerZmqChannel` owns one send side (a `PPSchedulerZmqPublisher`)
    and one receive side (a `PPSchedulerZmqSubscriber`), each backed by its
    own dedicated ZMQ PUSH / PULL socket on independent endpoints. It is the
    symmetric primitive needed by the edge-cloud PD-separation flow:

    - Edge constructs one channel with::

          send_endpoint = "tcp://*:<PRE_OUT_PORT>"          # bind, edge → cloud
          recv_endpoint = "tcp://<cloud_addr>:<POST_OUT_PORT>"   # connect

      and uses ``publish()`` to forward PREFILL_FIRST / DECODE_FIRST
      batches, and ``consume_new_outputs()`` to drain PREFILL_LAST /
      DECODE_LAST batches returned from the cloud.

    - Cloud constructs the mirror channel with::

          send_endpoint = "tcp://*:<POST_OUT_PORT>"          # bind, cloud → edge
          recv_endpoint = "tcp://<master_addr>:<PRE_OUT_PORT>"   # connect

      The same publish/consume API drives the opposite traffic direction.

    Both endpoints use the same PUSH/PULL + background-thread + queue.Queue
    bridge as the legacy unidirectional classes, so no scheduler-thread time
    is spent on pickling or socket I/O.

    Channel naming (``name``) is purely diagnostic; it is included in the
    log lines emitted by the underlying publisher / subscriber so the two
    edge-cloud channels can be told apart in a single combined log.
    """

    def __init__(
        self,
        send_endpoint: str,
        recv_endpoint: str,
        name: str = "pp-channel",
    ) -> None:
        self._name = name
        self._send_endpoint = send_endpoint
        self._recv_endpoint = recv_endpoint
        # Publisher binds-if-wildcard / connects-otherwise (see existing
        # `PPSchedulerZmqPublisher.__init__`); subscriber always connects.
        # The endpoints chosen by the caller therefore fully determine the
        # bind/connect roles of each side.
        self._publisher = PPSchedulerZmqPublisher(send_endpoint)
        self._subscriber = PPSchedulerZmqSubscriber(recv_endpoint)
        logger.info(
            "PPSchedulerZmqChannel[%s] up: send=%s, recv=%s",
            name,
            send_endpoint,
            recv_endpoint,
        )

    def publish(self, scheduler_output: SchedulerOutput) -> None:
        """Queue a SchedulerOutput for the peer. Non-blocking."""
        # logger.info(
        #     f"Send scheduler_output to edge, batch_type: "
        #     f"{scheduler_output.batch_type}",
        # )
        self._publisher.publish(scheduler_output)

    def consume_new_outputs(self) -> list[tuple[int, SchedulerOutput]]:
        """Return and clear all (seq, SchedulerOutput) pairs received since
        the last call. Suitable for use as the ``pp_subscriber`` argument
        of `PassiveScheduler`, which only relies on this method.
        """
        return self._subscriber.consume_new_outputs()

    def shutdown(self) -> None:
        self._publisher.shutdown()
        self._subscriber.shutdown()




def _trim_scheduler_output_for_worker_enqueue(
    scheduler_output: SchedulerOutput,
    prev_dispatch_req_ids: set[str] | None,
) -> SchedulerOutput:
    """Trim large cached token lists before cloud EngineCore -> worker MQ.

    Cloud worker ``_update_states`` only needs ``all_token_ids`` for cached
    requests that are not already in its persistent batch and have output
    tokens.  The best local approximation is the previous cloud dispatch batch:
    continuously dispatched requests can drop ``all_token_ids`` while newly
    appearing / resumed requests keep it.

    Entries that are kept MUST remain the complete token history
    (prompt + output): the worker recovery path slices
    ``all_token_ids[-num_output_tokens:]`` and assumes the prompt is
    included. Truncating entries corrupts the recovered output history and
    causes garbled decode output (see [EDGE-CLOUD-RECOVER]).
    """
    cached = scheduler_output.scheduled_cached_reqs
    if cached is None:
        return scheduler_output

    all_token_ids = getattr(cached, "all_token_ids", None)
    if not all_token_ids:
        return scheduler_output

    prev_dispatch_req_ids = prev_dispatch_req_ids or set()
    resumed_req_ids = getattr(cached, "resumed_req_ids", set()) or set()
    num_output_tokens_by_req = {
        req_id: num_output_tokens
        for req_id, num_output_tokens in zip(
            getattr(cached, "req_ids", ()),
            getattr(cached, "num_output_tokens", ()),
        )
    }
    # Keep every entry the cloud worker may need for the resume/recovery
    # path in _update_states. The recovery trigger on the worker is the
    # wire ``num_output_tokens`` (which INCLUDES async/spec placeholders),
    # so the keep condition must use the same placeholder-inclusive count.
    # NOTE: filtering by prev_dispatch_req_ids proved unsafe — interleaved
    # prefill layer-slices and placeholder accounting can evict a request
    # from the worker's persistent batch even when it appeared in the
    # previous dispatch, and a missing entry crashes the worker with a
    # KeyError. Prefer bandwidth over that risk.
    keep_req_ids = {
        req_id
        for req_id in all_token_ids
        if req_id in resumed_req_ids
        or req_id not in prev_dispatch_req_ids
        or num_output_tokens_by_req.get(req_id, 0) > 0
    }
    # Keep the complete prompt + output history for every retained entry.
    # The v0.23 worker computes the recoverable output length as
    # ``len(all_token_ids) - num_prompt_tokens``; sending only the output tail
    # would therefore make the payload look shorter than the prompt and drop
    # the entire recovered output history.
    trimmed_all_token_ids = {
        req_id: token_ids
        for req_id, token_ids in all_token_ids.items()
        if req_id in keep_req_ids
    }
    if len(trimmed_all_token_ids) == len(all_token_ids):
        return scheduler_output

    before_tokens = sum(len(token_ids) for token_ids in all_token_ids.values())
    after_tokens = sum(
        len(token_ids) for token_ids in trimmed_all_token_ids.values()
    )
    # logger.info(
    #     "[CLOUD-MQ-TRIM] batch_type=%s reqs=%d prev_dispatch_reqs=%d "
    #     "resumed=%d all_token_ids entries %d->%d tokens %d->%d",
    #     scheduler_output.batch_type.value,
    #     len(getattr(cached, "req_ids", ())),
    #     len(prev_dispatch_req_ids),
    #     len(resumed_req_ids),
    #     len(all_token_ids),
    #     len(trimmed_all_token_ids),
    #     before_tokens,
    #     after_tokens,
    # )

    so_copy = copy.copy(scheduler_output)
    cached_copy = copy.copy(cached)
    cached_copy.all_token_ids = trimmed_all_token_ids
    so_copy.scheduled_cached_reqs = cached_copy
    return so_copy


def _updates_worker_persistent_batch(
    scheduler_output: SchedulerOutput,
    slice_info,
) -> bool:
    """Return whether this dispatch updates the cloud worker batch state."""
    return (
        scheduler_output.batch_type
        not in (
            BatchType.PREFILL_DRAFT_FIRST,
            BatchType.DECODE_DRAFT_FIRST,
        )
        and scheduler_output.total_num_scheduled_tokens > 0
        and (slice_info is None or slice_info.is_first_slice)
    )


class PassiveEngineCoreProc:
    """Passive EngineCore process for non-leader PP ranks.

    Mirrors the `EngineCore` / `EngineCoreProc` shape on rank0:

    - `step()` is the single-tick action: poll the ZMQ inbox, ask the
      `PassiveScheduler` for one batch, fan its slice plan out to the
      worker `rpc_broadcast_mq`.
    - `run_busy_loop()` is the long-running driver that keeps calling
      `step()` until the executor reports failure.

    Unlike rank0, there is no local scheduling decision — every batch
    comes pre-decided over the cloud-side scheduler input. The static
    :py:meth:`run_passive_engine_core` is the process entry point that
    constructs the executor + input channel, builds an instance, and hands
    off to `run_busy_loop`.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        executor,  # MultiprocExecutor — duck-typed to avoid heavy import
        scheduler_input,
        dispatch_policy=None,
        pp_pd_channel: Optional["PPSchedulerZmqChannel"] = None,
    ) -> None:
        passive_scheduler_module = _import_passive_scheduler_module()
        if dispatch_policy is None:
            dispatch_policy = (
                passive_scheduler_module.DispatchPolicy.EXPECT_ALTERNATION
            )
        self.vllm_config = vllm_config
        self.executor = executor
        # scheduler_input is any object exposing consume_new_outputs(); in
        # PD-separation mode this is the cloud-side PPSchedulerZmqChannel.
        self.passive_scheduler = passive_scheduler_module.PassiveScheduler(
            vllm_config, scheduler_input, dispatch_policy=dispatch_policy
        )
        # Optional POST_OUT (cloud → edge) channel. Only set on the cloud
        # side in PD-separation mode; left None for the legacy PP path.
        self._pp_pd_channel = pp_pd_channel
        if getattr(vllm_config.parallel_config, "enable_edge_cloud", False):
            # PassiveEngineCore runs in a freshly-spawned subprocess; the
            # ``_ASCEND_CONFIG`` singleton may be empty here. ``init_ascend_config``
            # is idempotent and returns the cached singleton if already set.
            from vllm_ascend.ascend_config import init_ascend_config
            _ascend_config = init_ascend_config(vllm_config)
            _edge_cloud = getattr(_ascend_config, "edge_cloud_config", None)
            _pd_enabled = bool(
                _edge_cloud is not None
                and getattr(_edge_cloud, "enabled", False)
                and getattr(_edge_cloud, "pd_separation", None) is not None
                and _edge_cloud.pd_separation.enabled
            )
            logger.info(
                "PassiveEngineCore: edge-cloud mode enabled "
                "(pd_separation=%s, pd_channel=%s)",
                _pd_enabled,
                "on" if pp_pd_channel is not None else "off",
            )
            # Arrival-time recv pre-posting: hints travel on the
            # executor's hint MQ to the cloud worker's comm thread.
            # Read from vllm_config (parallel_config + additional_config
            # dict, both serialized fields) so the gate does not depend
            # on ascend_config singleton init order.
            _pc = vllm_config.parallel_config
            _ac = getattr(vllm_config, "additional_config", None) or {}
            _ec = _ac.get("edge_cloud_config", {}) if isinstance(_ac, dict) else {}
            _pd = _ec.get("pd_separation", {}) if isinstance(_ec, dict) else {}
            _hint_mq = getattr(self.executor, "cloud_recv_hint_mq", None)
            if (
                getattr(_pc, "enable_edge_cloud", False)
                and not getattr(_pc, "is_edge_node", True)
                and _pd.get("enabled", False)
                and _hint_mq is not None
            ):

                def _send_hint(hint: dict, mq=_hint_mq) -> None:
                    # Blocking enqueue: with readiness gating active a
                    # lost hint deadlocks the pipeline (no pre-post ->
                    # watermark never advances -> the gated batch is
                    # never dispatched).
                    mq.enqueue((b"irecv_hint", (hint,), {}, None))

                register_hint_sender(_send_hint)
        self._idle_sleep_seconds = 0.001

        self._prev_dispatch_req_ids: set[str] = set()
        self._pending_post_out_by_head_token: dict[str, SchedulerOutput] = {}
        self._published_post_out_tokens: set[str] = set()

    def _drain_worker_completion_acks(self) -> None:
        """Publish POST_OUT only after cloud workers complete the middle segment."""
        for mq in getattr(self.executor, "response_mqs", []):
            while True:
                try:
                    _status, result = mq.dequeue(timeout=0)
                except TimeoutError:
                    break
                except Exception:
                    logger.exception("Failed to drain cloud worker completion ack")
                    break

                if not (
                    isinstance(result, dict)
                    and result.get("__pp_scheduler_ack__")
                ):
                    continue

                # Decode and draft tails are self-posted on the edge. Their
                # worker acks are still drained from the response MQ, but do
                # not drive a cloud -> edge POST_OUT control message.
                if result.get("batch_type") != BatchType.PREFILL_FIRST:
                    continue
                head_token = result.get("head_token")
                if not head_token or head_token in self._published_post_out_tokens:
                    continue
                scheduler_output = self._pending_post_out_by_head_token.pop(
                    head_token, None
                )
                if scheduler_output is None:
                    continue
                # NOTE: do NOT add head_token to _published_post_out_tokens
                # here — _maybe_publish_post_out records it after actually
                # publishing (single idempotency point for both PL and DL).
                # logger.info(
                #     "[CLOUD-POST-OUT] Publishing PREFILL_LAST after worker done, "
                #     "head_token=%s",
                #     head_token,
                # )
                self._maybe_publish_post_out(scheduler_output)

    def _num_scheduled_spec_tokens(self) -> int:
        """num_spec_tokens when scheduled edge-cloud draft is active.

        Mirrors the edge-side ``_uses_scheduled_edge_cloud_draft`` method
        gating: Eagle3 and all MTP methods produce a draft chain after every
        PF/DF batch, hence only they need the n pre-posted draft irecvs.
        """
        spec = getattr(self.vllm_config, "speculative_config", None)
        if spec is None:
            return 0
        method = getattr(spec, "method", None)
        if method != "eagle3" and "mtp" not in str(method).lower():
            return 0
        return int(getattr(spec, "num_speculative_tokens", 0) or 0)

    def _post_irecv_hints_for_arrivals(self, arrivals) -> None:
        """Pre-post cloud-side irecvs the moment an edge SO arrives.

        For every PREFILL_FIRST / DECODE_FIRST the cloud will receive
        (edge -> cloud):
          1. this batch's head payload (same comm_seqno, UP channel of
             the pair), and
          2. one DRAFT_FIRST head payload per speculative step of the
             draft chain that follows the batch — the chain's reserved
             seqno range rides on the SO as ``draft_seqno_base``.

        Posting at arrival (instead of at dispatch time) maximizes
        the compute/transfer overlap and lets the passive scheduler gate
        dispatch on irecv completion instead of letting the worker
        block on recv.
        """
        for _seq, so in arrivals:
            bt = so.batch_type
            if bt not in (
                BatchType.PREFILL_FIRST,
                BatchType.DECODE_FIRST,
            ):
                continue
            seqno = getattr(so, "comm_seqno", None)
            if seqno is None:
                continue
            total_tokens = so.total_num_scheduled_tokens
            # Head payloads carry mrope when the edge stamped the SO.
            post_irecv_hint({
                "batch_type": bt,
                "draft_prefill_phase": False,
                "seqno": seqno,
                "num_tokens": total_tokens,
                "has_mrope": bool(getattr(so, "has_mrope", True)),
                "draft_step_idx": None,
            })
            base = getattr(so, "draft_seqno_base", None)
            if base is None:
                continue
            prefill_phase = bt == BatchType.PREFILL_FIRST
            num_spec = self._num_scheduled_spec_tokens()
            # Draft step shapes (mirrors the worker's meta derivation):
            # step 0 runs over the parent batch's full token count,
            # later steps over one token per request.
            num_reqs = len(so.num_scheduled_tokens)
            for step_idx in range(num_spec):
                post_irecv_hint({
                    "batch_type": (
                        BatchType.PREFILL_DRAFT_FIRST
                        if prefill_phase
                        else BatchType.DECODE_DRAFT_FIRST
                    ),
                    "draft_prefill_phase": prefill_phase,
                    "seqno": base + step_idx,
                    "num_tokens": (
                        total_tokens if step_idx == 0 else num_reqs
                    ),
                    "has_mrope": False,
                    "draft_step_idx": step_idx,
                })

    def _drain_irecv_completions(self) -> None:
        """Drain worker completion reports into the per-channel
        watermarks the passive scheduler's readiness gating queries."""
        done_mq = getattr(self.executor, "irecv_done_mq", None)
        if done_mq is None:
            if not getattr(self, "_irecv_done_mq_absent_logged", False):
                self._irecv_done_mq_absent_logged = True
                logger.info(
                    "[early-irecv] PassiveEC drain: irecv_done_mq is None "
                    "(reader not attached) — watermarks will never advance"
                )
            return
        items = []
        while True:
            try:
                items.append(done_mq.dequeue(timeout=0))
            except TimeoutError:
                break
            except Exception:
                if not getattr(self, "_irecv_done_dequeue_err_logged", False):
                    self._irecv_done_dequeue_err_logged = True
                    logger.exception(
                        "[early-irecv] PassiveEC drain: dequeue failed"
                    )
                break
        if items:
            record_irecv_completions(items)

    def step(self) -> bool:
        """Single tick: poll ZMQ → pick batches → enqueue worker payloads.

        Batches are dispatched one phase at a time in the order encoded by
        the configured dispatch policy.

        Returns:
            True if at least one payload was enqueued, False if the
            scheduler had nothing to dispatch.
        """
        _t0 = time.monotonic()
        self._drain_worker_completion_acks()
        # Drain worker recv-completion reports into the watermarks.
        self._drain_irecv_completions()
        _dt_drain = (time.monotonic() - _t0) * 1000

        _t0 = time.monotonic()
        arrivals = self.passive_scheduler.poll_and_classify()
        # Arrival-time data-plane decoupling: pre-post irecvs for every
        # newly arrived PF/DF head payload and its n draft head payloads.
        self._post_irecv_hints_for_arrivals(arrivals)
        _dt_poll = (time.monotonic() - _t0) * 1000

        _t0 = time.monotonic()
        batch = self.passive_scheduler.schedule()
        _dt_sched = (time.monotonic() - _t0) * 1000

        if batch.is_empty():
            if _dt_drain > 1.0 or _dt_poll > 1.0 or _dt_sched > 1.0:
                logger.info(
                    "[CLOUD-STEP-EMPTY] drain_acks=%.3f ms, poll=%.3f ms, "
                    "schedule=%.3f ms",
                    _dt_drain, _dt_poll, _dt_sched,
                )
            return False

        _slice_info_str = "["
        for s in batch.slices:
            if s is not None:
                _slice_info_str += (
                    f"slice_index={s.slice_index},"
                    f"start={s.start_layer},"
                    f"end={s.end_layer},"
                    f"is_last={s.is_last_slice};"
                )
            else:
                _slice_info_str += "None;"
        _slice_info_str += "]"
        # logger.info(
        #     f"\r\n[Cloud] Step dispatched batch_type: "
        #     f"{batch.scheduler_output.batch_type}, "
        #     f"slices_count={len(batch.slices)}, "
        #     f"slice_info={_slice_info_str}",
        # )

        for slice_info in batch.slices:
            _t0 = time.monotonic()
            worker_scheduler_output = _trim_scheduler_output_for_worker_enqueue(
                batch.scheduler_output,
                self._prev_dispatch_req_ids,
            )
            _dt_trim = (time.monotonic() - _t0) * 1000

            payload = (
                (worker_scheduler_output, slice_info)
                if slice_info is not None
                else (worker_scheduler_output,)
            )
            bt = batch.scheduler_output.batch_type.value
            # logger.info("[CLOUD-MQ] About to enqueue batch_type=%s", bt)
            _t0 = time.monotonic()
            self.executor.rpc_broadcast_mq.enqueue(
                (b"pp_scheduler_output", payload, {}, None)
            )
            if _updates_worker_persistent_batch(
                batch.scheduler_output, slice_info
            ):
                self._prev_dispatch_req_ids = set(
                    batch.scheduler_output.num_scheduled_tokens.keys()
                )
            _dt_enqueue = (time.monotonic() - _t0) * 1000
            if _dt_trim > 0.5 or _dt_enqueue > 0.5:
                logger.info(
                    "[CLOUD-STEP] trim=%.3f ms, enqueue=%.3f ms, batch_type=%s, "
                    "drain=%.3f ms, poll=%.3f ms, schedule=%.3f ms",
                    _dt_trim, _dt_enqueue, bt,
                    _dt_drain, _dt_poll, _dt_sched,
                )
            else:
                logger.info(
                    "[CLOUD-ENQUEUE] %s enqueue took %.3f ms",
                    bt,
                    _dt_enqueue,
                )
            # For prefill, POST_OUT must mean the cloud middle
            # segment has completed. Store the original SchedulerOutput here
            # and publish it from _drain_worker_completion_acks() after the
            # worker reports done. Decode-last and Draft-last are prepared on
            # the edge (self-posting), so they are not pending here.
            if (
                batch.scheduler_output.batch_type == BatchType.PREFILL_FIRST
                and (slice_info is None or slice_info.is_last_slice)
            ):
                head_token = getattr(batch.scheduler_output, "head_token", None)
                if head_token:
                    self._pending_post_out_by_head_token[head_token] = (
                        batch.scheduler_output
                    )
        return True

    def _maybe_publish_post_out(
        self, scheduler_output: SchedulerOutput
    ) -> None:
        """Rewrite + publish a head-segment batch as a tail-segment one
        on the POST_OUT (cloud → edge) channel.

        Mapping (cloud-side):
            PREFILL_FIRST → PREFILL_LAST
            DECODE_FIRST  → skipped (edge self-posts DECODE_LAST)
            DRAFT_FIRST   → skipped (edge self-posts DRAFT_LAST)
            anything else → dropped (legacy PP batches don't trigger return)

        Uses a shallow copy via :py:func:`dataclasses.replace` so the original
        SchedulerOutput (still about to be enqueued for the local executor)
        keeps its head-segment ``batch_type``.
        """
        if self._pp_pd_channel is None:
            return
        from dataclasses import replace
        bt = scheduler_output.batch_type
        if bt == BatchType.PREFILL_FIRST:
            tail = replace(
                scheduler_output, batch_type=BatchType.PREFILL_LAST
            )
        elif bt == BatchType.DECODE_FIRST:
            # The edge pre-generates DECODE_LAST, so the cloud does not
            # publish another control-plane response for DECODE_FIRST.
            logger.debug(
                "[Cloud] Skipping POST_OUT for DECODE_FIRST "
                "head_token=%s (edge pre-generates DECODE_LAST)",
                scheduler_output.head_token,
            )
            return
        elif bt in (
            BatchType.PREFILL_DRAFT_FIRST,
            BatchType.DECODE_DRAFT_FIRST,
        ):
            # The edge pre-generates DRAFT_LAST (self-posting, same as
            # DECODE_FIRST -> DECODE_LAST), so the cloud does not publish
            # POST_OUT for DRAFT_FIRST.
            logger.debug(
                "[Cloud] Skipping POST_OUT for DRAFT_FIRST "
                "head_token=%s (edge pre-generates DRAFT_LAST)",
                scheduler_output.head_token,
            )
            return
        else:
            return
        # Idempotency guard: publishing the same head_token twice would make
        # the edge process the tail segment twice. A duplicated DECODE_LAST
        # output subtracts num_output_placeholders a second time and drives
        # it negative (fatal assert in AsyncScheduler._update_request_with_output).
        # PREFILL_LAST is additionally gated by the worker-ack path; the
        # guard is harmless there and essential for DL.
        head_token = getattr(tail, "head_token", None)
        if head_token:
            if head_token in self._published_post_out_tokens:
                logger.warning(
                    "[CLOUD-POST-OUT] Suppressing duplicate %s publish for "
                    "head_token=%s",
                    tail.batch_type,
                    head_token,
                )
                return
            self._published_post_out_tokens.add(head_token)
        # Echo the head_token back so the edge can correlate the tail
        # segment with its suspended head state.
        self._pp_pd_channel.publish(tail)

    def run_busy_loop(self) -> None:
        """Drive `step()` until the executor reports failure or shutdown."""
        try:
            while not self.executor.is_failed:
                if not self.step():
                    time.sleep(self._idle_sleep_seconds)
        finally:
            self.passive_scheduler.shutdown()

    @staticmethod
    def run_passive_engine_core(
        vllm_config: "VllmConfig",
        ready_pipe,  # multiprocessing.Connection for signaling readiness
    ):
        """Entry point for the passive EngineCore process.

        Creates a MultiprocExecutor to spawn workers, wires up the
        cloud-side PD-separation channel as the scheduler input when PD
        separation is enabled, then hands off to
        `PassiveEngineCoreProc.run_busy_loop`.
        """
        # Imported lazily so the patched-by-vllm-ascend
        # ``MultiprocExecutor`` (= ``AscendMultiprocExecutor``) is the one
        # we instantiate when this code runs in a child process.
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor

        maybe_register_config_serialize_by_value()

        # Mark this process as a non-leader PP rank running with passive
        # EngineCore, so that AscendMultiprocExecutor and AscendWorkerProc
        # set up dual message queues (local + cross-node).
        os.environ["VLLM_PP_NON_LEADER_ENGINE_CORE"] = "1"
        envs.disable_envs_cache()

        set_process_title("PassiveEngineCore")
        maybe_init_worker_tracer(
            "vllm.engine_core", "engine_core", "PassiveEngineCore"
        )
        decorate_logs()

        # Cloud-side PD-separation channel is constructed inside the try
        # block below (depends on `vllm_config`); declared here so the
        # `finally` clean-up can reference it unconditionally.
        pp_pd_channel: Optional[PPSchedulerZmqChannel] = None

        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        executor = None
        try:
            executor = MultiprocExecutor(vllm_config, monitor_workers=False)

            ready_pipe.send({"status": "READY"})
            ready_pipe.close()
            ready_pipe = None

            passive_scheduler_module = _import_passive_scheduler_module()
            dispatch_policy_cls = passive_scheduler_module.DispatchPolicy
            # Load PD-separation configuration from environment variables.
            from vllm_ascend.pd_separation_config import PDSeparationConfig
            pd_config = PDSeparationConfig.from_env()
            try:
                policy = dispatch_policy_cls(pd_config.dispatch_policy)
            except ValueError:
                logger.warning(
                    "Unknown VLLM_PP_PASSIVE_DISPATCH_POLICY=%r; "
                    "falling back to expect_alternation.",
                    pd_config.dispatch_policy,
                )
                policy = dispatch_policy_cls.EXPECT_ALTERNATION

            scheduler_input = None

            # Set up edge-cloud PD-separation channel (cloud side). The
            # cloud binds POST_OUT and connects PRE_OUT via master_addr
            # (the edge's IP) so PRE_OUT connects back.
            #
            # PassiveEngineCore runs in a freshly-spawned subprocess where
            # the ``_ASCEND_CONFIG`` singleton is empty; re-init from the
            # ``vllm_config`` we were handed. ``init_ascend_config`` is
            # idempotent on the singleton.
            from vllm_ascend.ascend_config import init_ascend_config
            _ascend_config = init_ascend_config(vllm_config)
            _edge_cloud = getattr(_ascend_config, "edge_cloud_config", None)
            _pd_enabled = bool(
                _edge_cloud is not None
                and getattr(_edge_cloud, "enabled", False)
                and getattr(_edge_cloud, "pd_separation", None) is not None
                and _edge_cloud.pd_separation.enabled
            )
            if _pd_enabled:
                master_addr = vllm_config.parallel_config.master_addr
                master_port = vllm_config.parallel_config.master_port

                # Report this node's reachable IP to the edge so the
                # edge can construct POST_OUT's connect endpoint
                # without a CLI flag. Uses a one-shot TCPStore (edge
                # = master, cloud = client) on ``master_port + 1 +
                # dp_rank`` to avoid colliding with the NCCL
                # rendezvous store on ``master_port``. The cloud
                # connects only to the edge DP rank it is paired
                # with.
                import torch.distributed as dist
                from datetime import timedelta
                from vllm.utils.network_utils import get_ip
                _cloud_ip = get_ip()
                _dp_rank = getattr(
                    vllm_config.parallel_config, "data_parallel_rank", 0
                )
                _addr_store = dist.TCPStore(
                    host_name=master_addr,
                    port=master_port + 1 + _dp_rank,
                    world_size=2,
                    is_master=False,
                    timeout=timedelta(seconds=600),
                )
                _addr_store.set("cloud_ip", _cloud_ip)
                del _addr_store

                # ZMQ ports are offset per DP rank on the edge side
                # (dp_rank * 2). The cloud mirrors this offsetting.
                # NOTE: the cloud currently creates a single
                # PPSchedulerZmqChannel (per dp_rank=0). True
                # multi-DP cloud support requires N channels inside
                # PassiveEngineCoreProc.
                _pre_out_port = pd_config.pre_out_port + _dp_rank * 2
                _post_out_port = pd_config.post_out_port + _dp_rank * 2
                post_out_bind = f"tcp://*:{_post_out_port}"
                pre_out_connect = f"tcp://{master_addr}:{_pre_out_port}"
                pp_pd_channel = PPSchedulerZmqChannel(
                    send_endpoint=post_out_bind,
                    recv_endpoint=pre_out_connect,
                    name=f"pd-cloud-dp{_dp_rank}",
                )
                scheduler_input = pp_pd_channel
                logger.info(
                    "PD-separation cloud channel: POST_OUT=%s, "
                    "PRE_OUT=%s (dp_rank=%d)",
                    post_out_bind, pre_out_connect, _dp_rank,
                )

            if scheduler_input is not None:
                executor.start_worker_monitor(inline=False)
                proc = PassiveEngineCoreProc(
                    vllm_config, executor, scheduler_input,
                    dispatch_policy=policy,
                    pp_pd_channel=pp_pd_channel,
                )
                proc.run_busy_loop()
            else:
                # No scheduler input, just monitor workers inline.
                executor.start_worker_monitor(inline=True)

        except SystemExit:
            logger.debug("PassiveEngineCore exiting.")
        except Exception:
            logger.exception("PassiveEngineCore encountered a fatal error.")
            raise
        finally:
            if ready_pipe is not None:
                try:
                    ready_pipe.send({"status": "FAILED"})
                except Exception:
                    pass
                ready_pipe.close()
            if pp_pd_channel is not None:
                pp_pd_channel.shutdown()
            if executor is not None:
                executor.shutdown()
