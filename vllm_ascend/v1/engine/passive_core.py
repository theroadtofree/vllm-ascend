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

import os
import pickle
import queue
import signal
import threading
import time
from typing import TYPE_CHECKING, Optional

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

    def _publisher_thread(self) -> None:
        while self._running or self._queue.qsize() > 0:
            try:
                item = self._queue.get(timeout=0.1)
                if item is None:
                    break
                seq, scheduler_output = item
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
                with self._lock:
                    self._received_outputs.append((seq, scheduler_output))
                logger.info(
                    "PP rank1 received SchedulerOutput seq=%d, "
                    "total_scheduled_tokens=%d, "
                    "new_reqs=%d, cached_reqs=%d, "
                    "finished_req_ids=%s",
                    seq,
                    scheduler_output.total_num_scheduled_tokens,
                    len(scheduler_output.scheduled_new_reqs),
                    scheduler_output.scheduled_cached_reqs.num_reqs,
                    scheduler_output.finished_req_ids,
                )
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
        logger.info(
            f"Send scheduler_output to edge, batch_type: "
            f"{scheduler_output.batch_type}",
        )
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


class PassiveEngineCoreProc:
    """Passive EngineCore process for non-leader PP ranks.

    Mirrors the `EngineCore` / `EngineCoreProc` shape on rank0:

    - `step()` is the single-tick action: poll the ZMQ inbox, ask the
      `PassiveScheduler` for one batch, fan its slice plan out to the
      worker `rpc_broadcast_mq`.
    - `run_busy_loop()` is the long-running driver that keeps calling
      `step()` until the executor reports failure.

    Unlike rank0, there is no local scheduling decision — every batch
    comes pre-decided over ZMQ from the leader rank. The static
    :py:meth:`run_passive_engine_core` is the process entry point that
    constructs the executor + subscriber, builds an instance, and hands
    off to `run_busy_loop`.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        executor,  # MultiprocExecutor — duck-typed to avoid heavy import
        pp_subscriber: "PPSchedulerZmqSubscriber",
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
        self.passive_scheduler = passive_scheduler_module.PassiveScheduler(
            vllm_config, pp_subscriber, dispatch_policy=dispatch_policy
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
        self._idle_sleep_seconds = 0.001

    def step(self) -> bool:
        """Single tick: poll ZMQ → pick batches → enqueue worker payloads.

        Batches are dispatched one phase at a time in the order encoded by
        the configured dispatch policy.

        Returns:
            True if at least one payload was enqueued, False if the
            scheduler had nothing to dispatch.
        """
        self.passive_scheduler.poll_and_classify()
        batch = self.passive_scheduler.schedule()
        if batch.is_empty():
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
        logger.info(
            f"\r\n[Cloud] Step dispatched batch_type: "
            f"{batch.scheduler_output.batch_type}, "
            f"slices_count={len(batch.slices)}, "
            f"slice_info={_slice_info_str}",
        )

        for slice_info in batch.slices:
            payload = (
                (batch.scheduler_output, slice_info)
                if slice_info is not None
                else (batch.scheduler_output,)
            )
            self.executor.rpc_broadcast_mq.enqueue(
                (b"pp_scheduler_output", payload, {}, None)
            )
            # PD-separation: on the cloud side, publish the rewritten
            # tail-segment SchedulerOutput on POST_OUT only when the dispatched
            # work can produce the final middle-segment hidden state. With
            # slice-aware scheduling, early prefill slices must not wake the
            # edge tail segment because doing so can block the edge on a recv
            # and prevent it from issuing decode head work between P slices.
            if slice_info is None or slice_info.is_last_slice:
                self._maybe_publish_post_out(batch.scheduler_output)
        return True

    def _maybe_publish_post_out(
        self, scheduler_output: SchedulerOutput
    ) -> None:
        """Rewrite + publish a head-segment batch as a tail-segment one
        on the POST_OUT (cloud → edge) channel.

        Mapping (cloud-side):
            PREFILL_FIRST → PREFILL_LAST
            DECODE_FIRST  → DECODE_LAST
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
            tail = replace(
                scheduler_output, batch_type=BatchType.DECODE_LAST
            )
        else:
            return
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

        Creates a MultiprocExecutor to spawn workers, optionally wires up
        a ZMQ subscriber to receive SchedulerOutputs from the leader PP
        rank, then hands off to `PassiveEngineCoreProc.run_busy_loop`.
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

        pp_subscriber: Optional[PPSchedulerZmqSubscriber] = None
        if envs.VLLM_PP_SCHEDULER_ZMQ_ADDR is not None:
            pp_subscriber = PPSchedulerZmqSubscriber(
                envs.VLLM_PP_SCHEDULER_ZMQ_ADDR
            )

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

            if pp_subscriber is not None:
                executor.start_worker_monitor(inline=False)

                passive_scheduler_module = _import_passive_scheduler_module()
                dispatch_policy_cls = passive_scheduler_module.DispatchPolicy
                try:
                    policy = dispatch_policy_cls(
                        envs.VLLM_PP_PASSIVE_DISPATCH_POLICY
                    )
                except ValueError:
                    logger.warning(
                        "Unknown VLLM_PP_PASSIVE_DISPATCH_POLICY=%r; "
                        "falling back to expect_alternation.",
                        envs.VLLM_PP_PASSIVE_DISPATCH_POLICY,
                    )
                    policy = dispatch_policy_cls.EXPECT_ALTERNATION

                # Set up edge-cloud PD-separation channel (cloud side).
                # The cloud binds POST_OUT and connects PRE_OUT via
                # master_addr (the edge's IP) so PRE_OUT connects back.
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
                    # = master, cloud = client) on ``master_port + 1``
                    # to avoid colliding with the NCCL rendezvous store
                    # on ``master_port``.
                    import torch.distributed as dist
                    from datetime import timedelta
                    from vllm.utils.network_utils import get_ip
                    _addr_store = dist.TCPStore(
                        host_name=master_addr,
                        port=master_port + 1,
                        world_size=2,
                        is_master=False,
                        timeout=timedelta(seconds=300),
                    )
                    _addr_store.set("cloud_ip", get_ip())
                    del _addr_store

                    post_out_bind = (
                        f"tcp://*:{envs.VLLM_PP_POST_OUT_ZMQ_PORT}"
                    )
                    pre_out_connect = (
                        f"tcp://{master_addr}:"
                        f"{envs.VLLM_PP_PRE_OUT_ZMQ_PORT}"
                    )
                    pp_pd_channel = PPSchedulerZmqChannel(
                        send_endpoint=post_out_bind,
                        recv_endpoint=pre_out_connect,
                        name="pd-cloud",
                    )
                    logger.info(
                        "PD-separation cloud channel: POST_OUT=%s, "
                        "PRE_OUT=%s",
                        post_out_bind, pre_out_connect,
                    )

                proc = PassiveEngineCoreProc(
                    vllm_config, executor, pp_subscriber,
                    dispatch_policy=policy,
                    pp_pd_channel=pp_pd_channel,
                )
                proc.run_busy_loop()
            else:
                # No ZMQ subscriber, just monitor workers inline.
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
            if pp_subscriber is not None:
                pp_subscriber.shutdown()
            if pp_pd_channel is not None:
                pp_pd_channel.shutdown()
            if executor is not None:
                executor.shutdown()
