#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_worker.py
#

from enum import Enum
from typing import Any
import copy
import gc
import logging
import threading
import time
from dataclasses import replace
from types import NoneType

import torch
import torch.nn as nn
import torch_npu
from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
from torch_npu.profiler import dynamic_profile as dp
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.distributed import ensure_model_parallel_initialized, get_pcp_group, init_distributed_environment
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorHandshakeMetadata
from vllm.distributed.parallel_state import (
    Handle,
    get_pp_group,
    get_tp_group,
    is_cloud_device,
    is_edge_device,
)
from vllm.logger import logger
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot, format_gib, memory_profiling
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import (
    BatchType,
    GrammarOutput,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase
from vllm.v1.worker.workspace import init_workspace_manager

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.batch_invariant import init_batch_invariance
from vllm_ascend.cpu_binding import bind_cpus
from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.device_allocator.sleep_mem_optimized import SleepWakeupManager
from vllm_ascend.distributed.edge_cloud_comm import (
    BatchKind,
    CommChannelType,
    CommFuture,
    CommRequest,
    LoggingSchedulerCommSink,
    channel_for,
    channel_for_direction,
    get_comm_service,
    kind_for_batch_type,
)
from vllm_ascend.distributed.parallel_state import (
    ScheduledDraftTensorMeta,
    build_scheduled_draft_tensor_meta,
    init_ascend_model_parallel,
    init_edge_cloud_tensor_meta,
)
from vllm_ascend.edge_cloud_materialized import (
    supports_materialized_boundary_for_config,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper
from vllm_ascend.utils import (
    AscendDeviceType,
    check_ascend_device_type,
    enable_sp,
    get_ascend_device_type,
    register_ascend_customop,
    setup_ascend_local_comm_res,
    vllm_version_is,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

# [early-irecv] Env-var handles of the executor-created sideband hint MQs.
# The names are owned by patch/platform/patch_multiproc_executor.py (which sets
# them before spawning workers); duplicated here as literals because
# importing the patch module from the worker would apply its class
# replacements as an import side effect.  The reverse completion-report MQ
# needs no env handle: it is created (as writer) by this worker and its
# handle rides the READY handshake back to the executor.
_CLOUD_RECV_HINT_MQ_ENV = "VLLM_ASCEND_CLOUD_RECV_HINT_MQ_HANDLE"
_EDGE_RECV_HINT_MQ_ENV = "VLLM_ASCEND_EDGE_RECV_HINT_MQ_HANDLE"


class SchedulerBatchType(Enum):
    """Enum for the batch type of a SchedulerOutput step."""
    ALL_PREFILL = "ALL_PREFILL"
    ALL_DECODE = "ALL_DECODE"
    PREFILL_DECODE_MIXED = "PREFILL_DECODE_MIXED"


torch._dynamo.trace_rules.clear_lru_cache()  # noqa: E402
from torch._dynamo.variables import TorchInGraphFunctionVariable  # noqa: E402
from vllm.utils.torch_utils import set_random_seed  # noqa: E402

torch_non_c_binding_in_graph_functions_npu = dict.fromkeys(
    ["torch.npu.current_stream"],
    TorchInGraphFunctionVariable,
)  # noqa: E402
torch_non_c_binding_in_graph_functions_npu["torch.npu.stream"] = TorchInGraphFunctionVariable  # noqa: E402
torch._dynamo.trace_rules.torch_name_rule_map.append(torch_non_c_binding_in_graph_functions_npu)  # noqa: E402


def _detect_has_residual(model_config) -> bool:
    """Detect whether the model produces a residual tensor in IntermediateTensors.

    Models with residual connections (most decoder-only LLMs) output
    {"hidden_states": ..., "residual": ...} in IntermediateTensors,
    while models without residual output only {"hidden_states": ...}.

    Detection strategy: check the model's architecture class for the
    presence of residual stream handling.
    """
    hf_config = getattr(model_config, "hf_text_config", None)
    model_type = getattr(hf_config, "model_type", "") if hf_config else ""
    # Qwen3.5 / Qwen3.5-MoE use residual connections
    if "qwen3" in model_type:
        return True
    # DeepSeek V4 uses hc_pre/hc_post internally, but in the edge-cloud
    # no-residual variant the residual is recomputed locally per segment and
    # is no longer transmitted across the network.
    if model_type == "deepseek_v4":
        return False
    # Default: most modern decoder models produce residual
    # Can be made more specific as more models are supported
    return True


def _use_materialized_residual_boundary(model_config) -> bool:
    return supports_materialized_boundary_for_config(model_config)


class NPUWorker(WorkerBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        # Additional parameters for compatibility with vllm
        **kwargs,
    ):
        """Initialize the worker for Ascend."""
        if not envs_ascend.COMPILE_CUSTOM_KERNELS:
            logger.warning(
                "COMPILE_CUSTOM_KERNELS is set to False. "
                "In most scenarios, without custom kernels, vllm-ascend will not function correctly."
            )

        # register patch for vllm
        from vllm_ascend.utils import adapt_patch

        adapt_patch()

        # Register ops when worker init.
        from vllm_ascend import ops

        ops.register_dummy_fusion_op()
        if get_ascend_device_type() != AscendDeviceType.A5:
            _register_atb_extensions()
        register_ascend_customop(vllm_config)
        # init ascend config and soc version
        init_ascend_config(vllm_config)
        from vllm_ascend.logger import configure_ascend_file_logging

        configure_ascend_file_logging()
        check_ascend_device_type()

        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[self.cache_config.cache_dtype]

        # Profiler is lazily initialized on first profile(is_start=True) call (RFC #6954)
        self.profiler_config = vllm_config.profiler_config
        self.profiler: TorchNPUProfilerWrapper | None = None
        self.npugraph_memory_bytes = 0
        if vllm_config.model_config and vllm_config.model_config.enable_sleep_mode:
            # Buffers saved before sleep
            self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
        self.sleep_wakeup_manager = SleepWakeupManager(vllm_config, self, lambda: getattr(self, "model_runner", None))

        # Weight transfer engine is created in `load_model` once the model
        # is available, since the engine needs a reference to the model.
        self.weight_transfer_engine = None
        self._weight_update_active = False
        self._is_checkpoint_format = True

        # FixMe: this is a patch to fix the issue cause by https://github.com/vllm-project/vllm/commit/de94289a98d7ec52a5ef02719e01a1db8b505170
        from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED

        if "UnquantizedLinearMethod" in WEIGHT_LOADER_V2_SUPPORTED:
            WEIGHT_LOADER_V2_SUPPORTED.remove("UnquantizedLinearMethod")

        self.use_v2_model_runner = self.vllm_config.use_v2_model_runner
        if self.use_v2_model_runner and vllm_version_is("0.23.0"):
            logger.warning("VLLM_USE_V2_MODEL_RUNNER is not supported on vllm 0.23.0; falling back to v1 model runner.")
            self.use_v2_model_runner = False
        # Legacy (non-edge-cloud) PP path: one outstanding send, waited at
        # the top of _execute_model_legacy.  Edge-cloud sends are owned by
        # the comm service (EdgeCloudCommService) and never tracked here.
        self._pp_send_work: list[Handle] = []
        # A pre-generated MTP DRAFT_FIRST may overtake the DRAFT_LAST that
        # produces its inputs in the async executor queue. Do not block that
        # head in-place (the tail could be queued behind it); suspend its
        # control state and let the preceding tail resume it locally.
        self._deferred_edge_draft_heads: dict[
            tuple[str, int], SchedulerOutput
        ] = {}

        # [early-irecv] Arrival-time recv pre-posting registry, keyed by
        # (channel, seqno).  This worker's own comm thread turns scheduler
        # recv-hints into submit_recv calls ahead of the batch's
        # execute_model; the consume points attach the cached CommFuture via
        # get_or_post_early_recv() (or submit+register themselves on a hint
        # miss, so completion reporting never gaps).  An entry is dropped
        # only once it is BOTH consumed (a consume point attached it) and
        # reported (the comm thread sent its completion on irecv_done_mq);
        # the reporter keeps polling done() on consumed entries to advance
        # the watermarks.  Hints are a correctness dependency under
        # readiness gating, so they are never dropped for flow control --
        # the registry is bounded by the scheduler's in-flight window.
        self._early_recv_futures: dict[tuple[CommChannelType, int], CommFuture] = {}
        self._early_recv_lock = threading.Lock()
        # (channel, seqno) keys already consumed by busy_loop
        # (get_or_post_early_recv).  Prevents the comm thread from
        # submitting a duplicate (orphan) recv when its hint arrives after
        # busy_loop already submitted its own.
        self._early_recv_consumed: set[tuple[CommChannelType, int]] = set()
        # (channel, seqno) keys whose completion was already reported on
        # irecv_done_mq.
        self._early_recv_reported: set[tuple[CommChannelType, int]] = set()
        # comm -> scheduler completion feedback (design doc 8.3-2):
        # register the logging no-op sink so the chain
        # submit -> reap -> sink is live end-to-end; poll_completions()
        # at the execute_model loop head drives it.  Registration is
        # type-idempotent (shared-model virtual workers share a process).
        get_comm_service().register_sink(LoggingSchedulerCommSink())
        # Comm-thread state (arrival-time irecv pre-posting).  The MQs are
        # rebuilt from their env handles and the daemon thread is started
        # at the end of __init__ by _start_edge_cloud_comm_thread() when
        # this is the PP-first worker of a PD-separated edge/cloud node;
        # _early_recv_comm_active then gates the registry/reporting path.
        self._irecv_hint_mq = None
        self._irecv_done_mq = None
        # Public alias of _irecv_done_mq: WorkerProc picks it up after
        # worker construction and exports its handle in the READY
        # handshake so the engine core can attach the reader.
        self.irecv_done_mq = None
        self._early_recv_comm_active: bool = False
        # Set at the end of init_device: the comm thread must not
        # submit_recv before the model runner and the edge-cloud tensor
        # meta exist (SP gate, draft wire meta, recv-buffer shapes).
        self._edge_cloud_comm_ready = threading.Event()
        self._edge_cloud_comm_thread: threading.Thread | None = None

        ascend_compilation_config = get_ascend_config().ascend_compilation_config
        if ascend_compilation_config.enable_npugraph_ex and ascend_compilation_config.enable_static_kernel:
            # Prevent duplicate triggers, execute the exit logic only once
            shutdown_request = False

            def signal_handler(signum, frame):
                nonlocal shutdown_request
                if not shutdown_request:
                    shutdown_request = True
                    self.uninstall_static_kernel()
                    raise SystemExit()

            # Either SIGTERM or SIGINT will terminate the worker
            import signal

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

        # [early-irecv] Start this worker's comm thread (PD-separated
        # edge/cloud PP-first workers only; no-op otherwise).
        self._start_edge_cloud_comm_thread()

    def uninstall_static_kernel(self):
        import fcntl
        import os
        import subprocess

        ascend_home_path = os.environ["ASCEND_HOME_PATH"]
        static_kernel_dir_path = os.path.join(ascend_home_path, "opp/static_kernel")
        uninstall_script_path = os.path.join(static_kernel_dir_path, "ai_core/uninstall.sh")
        lock_file_path = os.path.join(static_kernel_dir_path, "uninstall.lock")

        if not os.path.exists(uninstall_script_path):
            return
        with open(lock_file_path, "w") as lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                subprocess.Popen(
                    ["bash", uninstall_script_path],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except (BlockingIOError, OSError):
                return
            finally:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    if os.path.exists(lock_file_path):
                        os.remove(lock_file_path)
                except Exception:
                    return

    def sleep(self, level: int = 1) -> None:
        free_bytes_before_sleep = torch.npu.mem_get_info()[0]
        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}

        cleanup_enabled = getattr(get_ascend_config(), "enable_sleep_mode_extra_cleanup", False)
        if cleanup_enabled:
            self.sleep_wakeup_manager.sleep()

        allocator = CaMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.npu.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."

        logger.info(
            "Sleep mode (level=%s) freed %.2f GiB memory, %.2f GiB memory is still in use.",
            level,
            freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes,
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        nz_mode = get_ascend_config().weight_nz_mode
        if nz_mode:
            raise ValueError(
                "FRACTAL_NZ mode is enabled. This may cause model parameter precision issues "
                "in the RL scenarios. Please set weight_nz_mode=0 via --additional-config."
            )
        allocator = CaMemAllocator.get_instance()
        allocator.wake_up(tags=tags)

        hidden_size = self.vllm_config.model_config.hf_text_config.hidden_size
        model = self.model_runner.model
        if self.vllm_config.quant_config is None and (tags is None or "weights" in tags):
            for name, param in model.named_parameters():
                if "w2_weight" in name and param.shape[2] == hidden_size:
                    parts = name.split(".")
                    param_name = parts[-1]
                    parent_module = model.get_submodule(".".join(parts[:-1]))

                    w2_data = param.transpose(1, 2)
                    w2_data = torch.nn.Parameter(w2_data, requires_grad=False)
                    setattr(parent_module, param_name, w2_data)
                elif "w13_weight" in name and param.shape[1] == hidden_size:
                    parts = name.split(".")
                    param_name = parts[-1]
                    parent_module = model.get_submodule(".".join(parts[:-1]))

                    w13_data = param.transpose(1, 2)
                    w13_data = torch.nn.Parameter(w13_data, requires_grad=False)
                    setattr(parent_module, param_name, w13_data)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}
        cleanup_enabled = getattr(get_ascend_config(), "enable_sleep_mode_extra_cleanup", False)
        if cleanup_enabled:
            self.sleep_wakeup_manager.wakeup(tags)

    def _check_weight_transfer_engine(self) -> None:
        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. Please set weight_transfer_config to enable weight transfer."
            )

    def init_weight_transfer_engine(self, init_info: dict) -> None:
        """Initialize the HCCL weight transfer process group with the trainer."""
        self._check_weight_transfer_engine()
        assert self.weight_transfer_engine is not None
        typed_init_info = self.weight_transfer_engine.parse_init_info(init_info)
        self.weight_transfer_engine.init_transfer_engine(typed_init_info)

    def _check_nz_disabled(self) -> None:
        if envs_ascend.VLLM_ASCEND_ENABLE_NZ:
            raise ValueError(
                "FRACTAL_NZ mode is enabled. This may cause model parameter "
                "precision issues in the RL scenarios. Please set "
                "VLLM_ASCEND_ENABLE_NZ=0."
            )

    def start_weight_update(self, is_checkpoint_format: bool = True) -> None:
        """Begin a new weight update; prepares the model for layerwise reload."""
        self._check_weight_transfer_engine()

        if self._weight_update_active:
            raise RuntimeError(
                "start_weight_update called while a weight update is already active. Call finish_weight_update first."
            )

        self._check_nz_disabled()

        if is_checkpoint_format:
            from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

            model = self.model_runner.model
            with torch.device(self.device):
                initialize_layerwise_reload(model)

        self._is_checkpoint_format = is_checkpoint_format
        self._weight_update_active = True

    def update_weights(self, update_info: dict) -> None:
        """Receive a chunk of weights from the trainer and load them in place."""
        self._check_weight_transfer_engine()
        assert self.weight_transfer_engine is not None

        typed_update_info = self.weight_transfer_engine.parse_update_info(update_info)
        model = self.model_runner.model

        # state machine driven by start/finish.
        if not self._weight_update_active:
            raise RuntimeError("start_weight_update must be called before update_weights.")

        with torch.device(self.device):
            if self._is_checkpoint_format:
                self.weight_transfer_engine.receive_weights(
                    typed_update_info,
                    load_weights=model.load_weights,
                )
            else:

                def load_weights_direct(weights: list[tuple[str, torch.Tensor]]) -> None:
                    with torch.no_grad():
                        for name, weight in weights:
                            param = model.get_parameter(name)
                            param.copy_(weight)

                self.weight_transfer_engine.receive_weights(
                    typed_update_info,
                    load_weights=load_weights_direct,
                )

        # HCCL broadcast / packed paths are asynchronous.
        # Sync so the next step uses the new weights.
        torch.npu.synchronize()

    def finish_weight_update(self) -> None:
        """Finish the current weight update; runs layerwise postprocessing."""
        self._check_weight_transfer_engine()

        if not self._weight_update_active:
            raise RuntimeError("start_weight_update must be called before finish_weight_update.")

        if self._is_checkpoint_format:
            from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

            model = self.model_runner.model
            with torch.device(self.device):
                finalize_layerwise_reload(model, self.model_config)

        self._weight_update_active = False
        self._is_checkpoint_format = True

    def shutdown(self) -> None:
        if ensure_kv_transfer_shutdown is not None:
            ensure_kv_transfer_shutdown()

        if self.profiler is not None:
            self.profiler.shutdown()

        if weight_transfer_engine := getattr(self, "weight_transfer_engine", None):
            weight_transfer_engine.shutdown()

        if model_runner := getattr(self, "model_runner", None):
            shutdown_fn = getattr(model_runner, "shutdown", None)
            if callable(shutdown_fn):
                shutdown_fn()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def _init_device(self):
        if not vllm_version_is("0.23.0"):
            # vLLM v0.24.0 (PR #45026) removed automatic per-process device
            # isolation for DP workers. Mirror gpu_worker.py::init_device:
            # shift self.local_rank by dp_local_rank * tp_pp_world_size so
            # that each DP group binds to a distinct set of NPUs.
            parallel_config = self.parallel_config
            if (
                parallel_config.distributed_executor_backend not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and parallel_config.nnodes_within_dp == 1
                # vllm-ascend: when the user pre-shards devices via
                # --device-ids (which becomes assigned_physical_gpu_ids),
                # each child process already binds to its own NPU(s); the
                # DP local_rank shift below would push local_rank past the
                # length of the per-rank device list and trip the assert
                # in this same method. Skip the shift in that case.
                and parallel_config.assigned_physical_gpu_ids is None
            ):
                dp_local_rank = parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = parallel_config.data_parallel_index
                tp_pp_world_size = parallel_config.pipeline_parallel_size * parallel_config.tensor_parallel_size
                self.local_rank += dp_local_rank * tp_pp_world_size

            # Publish the logical-to-physical mapping for topology queries.
            assigned_physical_gpu_ids = parallel_config.assigned_physical_gpu_ids
            if assigned_physical_gpu_ids is not None:
                from vllm.platforms.interface import set_assigned_physical_gpu_ids

                set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)
                assert self.local_rank < len(assigned_physical_gpu_ids), (
                    f"local_rank {self.local_rank} is out of bounds for "
                    f"assigned_physical_gpu_ids {assigned_physical_gpu_ids}"
                )
                if parallel_config.distributed_executor_backend not in ("ray", "external_launcher"):
                    assert parallel_config.local_world_size <= len(assigned_physical_gpu_ids), (
                        f"local_world_size ({parallel_config.local_world_size}) "
                        f"exceeds assigned_physical_gpu_ids count "
                        f"({len(assigned_physical_gpu_ids)})"
                    )
            else:
                visible_device_count = torch.npu.device_count() if torch.npu.is_available() else 0
                assert self.local_rank < visible_device_count, (
                    f"DP adjusted local rank {self.local_rank} is out of bounds for {visible_device_count} devices."
                )

            visible_device_index = current_platform.logical_device_id_to_visible_device_id(self.local_rank)
            device = torch.device(f"{current_platform.device_type}:{visible_device_index}")
        else:
            device = torch.device(f"npu:{self.local_rank}")

        torch.npu.set_device(device)

        # Import _inductor for graph mode execution with triton
        # This lazy import avoids torch_npu re-initialization in patch
        # Note that this should be imported after torch.npu.set_device
        # to avoid repeated set_device in extra processes
        from vllm.triton_utils import HAS_TRITON

        if HAS_TRITON:
            import torch_npu._inductor  # noqa: F401

        gc.collect()
        torch.npu.empty_cache()

        if get_ascend_device_type() == AscendDeviceType.A5:
            setup_ascend_local_comm_res(self.local_rank, self.vllm_config.kv_transfer_config)

        # take current memory snapshot
        if vllm_version_is("0.23.0"):
            self.init_snapshot = MemorySnapshot()
        else:
            self.init_snapshot = MemorySnapshot(device=device)
        self.requested_memory = self.init_snapshot.total_memory * self.cache_config.gpu_memory_utilization
        if self.init_snapshot.free_memory < self.requested_memory:
            GiB = lambda b: round(b / GiB_bytes, 2)
            raise ValueError(
                f"Free memory on device "
                f"({GiB(self.init_snapshot.free_memory)}/"
                f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                f"is less than desired GPU memory utilization "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                f"utilization or reduce GPU memory used by other processes."
            )

        if (
            self.parallel_config.data_parallel_size > 1
            and self.parallel_config.data_parallel_size_local > 0
            and self.parallel_config.distributed_executor_backend not in ["ray", "external_launcher"]
            and self.vllm_config.parallel_config.data_parallel_backend != "ray"
            and self.vllm_config.parallel_config.nnodes_within_dp == 1
        ):
            visible_device_count = torch.npu.device_count() if torch.npu.is_available() else 0
            assert self.parallel_config.local_world_size <= visible_device_count, (
                f"local_world_size ({self.parallel_config.local_world_size}) must "
                f"be less than or equal to the number of visible devices "
                f"({visible_device_count})."
            )

        # Initialize the distributed environment.
        self._init_worker_distributed_environment()
        # Set random seed.
        set_random_seed(self.model_config.seed)
        # Initialize device properties used by triton kernels.
        init_device_properties_triton()

        return device

    def init_device(self):
        # NOTE: KEEP device the member of `NPUWorker`, as it will be checked
        # in ray scenario. see https://github.com/vllm-project/vllm/pull/26845
        # for more details
        self.device = self._init_device()
        # Initialize workspace manager
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)
        # Init ModelRunner here, so that we have access to self.device.
        if self.use_v2_model_runner:
            logger.warning("npu model runner v2 is in developing, some features doesn't work for now.")
            from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2

            self.model_runner = NPUModelRunnerV2(self.vllm_config, self.device)
        else:
            self.model_runner = NPUModelRunner(self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

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
                uses_mrope=self.model_config.uses_mrope,
                materialize_residual_boundary=(
                    _use_materialized_residual_boundary(self.model_config)
                ),
            )

        # [early-irecv] Unblock the comm thread (started in __init__ on
        # PD-separated PP-first workers): the model runner and the
        # edge-cloud tensor meta it needs for submit_recv (SP gate, draft
        # wire meta, recv-buffer shapes) are all initialized by now.
        self._edge_cloud_comm_ready.set()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculates the free memory that can be used for KV cache in
        bytes.
        """
        GiB = lambda b: b / GiB_bytes

        # Fast path: user has explicitly specified KV cache size via
        # --kv-cache-memory. Still run profile_run() to compile the model,
        # but skip the memory profiling calculation entirely.
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            self.model_runner.profile_run()
            logger.info(
                "Initial free memory %.2f GiB, reserved %.2f GiB for KV Cache "
                "as specified by kv_cache_memory_bytes, skipping memory profiling. "
                "This does not respect the gpu_memory_utilization config. "
                "Only use kv_cache_memory_bytes when you want manual control of "
                "KV cache memory size. If OOM'ed, check the difference of initial "
                "free memory between the current run and the previous run where "
                "kv_cache_memory_bytes is suggested and update it correspondingly.",
                GiB(self.init_snapshot.free_memory),
                GiB(kv_cache_memory_bytes),
            )
            return kv_cache_memory_bytes

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

            # Record torch peak INSIDE the context and BEFORE graph capture,
            # so that graph pool allocations don't inflate the activation peak.
            # The memory_profiling context will also compute torch_peak_increase
            # on exit, but we override it below with this pre-graph value.
            profile_torch_peak = torch.npu.memory_stats(self.device).get("allocated_bytes.all.peak", 0)

        # Override torch_peak_increase with the pre-graph-capture value to
        # avoid double-counting graph pool memory as activation memory.
        profile_result.torch_peak_increase = profile_torch_peak - profile_result.before_profile.torch_peak
        profile_result.non_kv_cache_memory = (
            profile_result.non_torch_increase + profile_result.torch_peak_increase + profile_result.weights_memory
        )

        # Save per-category memory for use in compile_or_warm_up_model() (step 5).
        self.peak_activation_memory = profile_result.torch_peak_increase
        self.non_torch_memory = profile_result.non_torch_increase

        free_gpu_memory = profile_result.after_profile.free_memory
        assert self.init_snapshot.free_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {GiB(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {GiB(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container."
        )
        self.available_kv_cache_memory_bytes = self.requested_memory - profile_result.non_kv_cache_memory

        # For embedding_only edge, the edge device does not actually store KV
        # cache tensors. Return a very large virtual value so that
        # get_kv_cache_configs() does not clamp num_blocks to the edge's
        # (small) available memory. The real num_blocks is determined by cloud.
        if (
            self.model_runner.edge_cloud_cfg.enabled
            and self.model_runner.edge_cloud_cfg.mode == "embedding_only"
            and self.model_runner.edge_cloud_cfg.role == "edge"
        ):
            virtual_memory = 1 << 40  # 1 TiB virtual
            logger.info(
                "[EdgeCloud] embedding_only edge using virtual available_memory "
                "(%.2f GiB) instead of real %.2f GiB to avoid limiting cloud "
                "KV cache size.",
                GiB(virtual_memory),
                GiB(self.available_kv_cache_memory_bytes),
            )
            self.available_kv_cache_memory_bytes = virtual_memory

        logger.debug(profile_result)
        logger.info_once(
            "Available KV cache memory: %.2f GiB", GiB(self.available_kv_cache_memory_bytes), scope="local"
        )

        return int(self.available_kv_cache_memory_bytes)

    def _wait_pp_send_work(self) -> None:
        """Legacy (non-edge-cloud) PP path only: wait the single outstanding
        send before this batch's compute may reuse its source buffer.

        Edge-cloud sends are submitted to the comm service, whose
        per-channel FIFO + wait-bridge gives device-side ordering and whose
        CommFuture keeps the send buffer alive until ``event.query()``
        reports completion — no wait is ever needed on the compute path for
        them (see edge_cloud_comm_design.md section 5).
        """
        for handle in self._pp_send_work:
            handle.wait()
        self._pp_send_work = []

    # ------------------------------------------------------------------ #
    # [early-irecv] Arrival-time recv pre-posting primitives             #
    # ------------------------------------------------------------------ #
    # This worker's comm thread (one per PP-first worker, started in
    # __init__) drains the scheduler's recv-hints off the sideband hint MQ
    # and calls start_early_irecv() to submit a recv to the comm service
    # ahead of the batch's execute_model (keyed by (channel, seqno));
    # execute_model calls get_or_post_early_recv() to attach the cached
    # CommFuture.  The comm service posts the irecv immediately on submit,
    # so "early" is purely a matter of when the hint arrives.  The same
    # thread reports every completed recv as (channel, seqno) on
    # irecv_done_mq, feeding the scheduler's per-channel watermarks.
    def _start_edge_cloud_comm_thread(self) -> None:
        """Rebuild the hint MQ from its env handle, create the reverse
        completion-report MQ (writer side), and start the comm thread.

        Arrival-time irecv pre-posting is a built-in part of PD-separation
        masking, so the gate is simply "PD-separated edge/cloud node +
        local_rank==0" (the PP-first rank issuing the cross-node irecv;
        other ranks receive hidden via TP-broadcast from rank0).  The
        shared-model single-NPU edge is excluded: that topology has no
        hint/feedback infrastructure this period and its scheduler runs
        ungated.  PD-enabled is read from vllm_config (parallel_config +
        the serialized additional_config dict) for a uniform,
        init-order-independent source of truth.
        """
        pc = self.parallel_config
        if self.local_rank != 0 or not getattr(pc, "enable_edge_cloud", False):
            return
        if getattr(pc, "is_edge_node", False) and getattr(
            pc, "is_shared_model_edge", False
        ):
            return
        ac = getattr(self.vllm_config, "additional_config", None) or {}
        ec = ac.get("edge_cloud_config", {}) if isinstance(ac, dict) else {}
        pd = ec.get("pd_separation", {}) if isinstance(ec, dict) else {}
        if not pd.get("enabled", False):
            return
        import base64
        import os
        import pickle

        from vllm.distributed.device_communicators.shm_broadcast import (
            MessageQueue,
        )

        hint_env = (
            _EDGE_RECV_HINT_MQ_ENV
            if getattr(pc, "is_edge_node", False)
            else _CLOUD_RECV_HINT_MQ_ENV
        )
        raw = os.environ.get(hint_env)
        if raw is None:
            # The executor did not create the hint MQ (PD hint infra off).
            return
        try:
            handle = pickle.loads(base64.b64decode(raw))
            self._irecv_hint_mq = MessageQueue.create_from_handle(handle, 0)
        except Exception:
            logger.exception(
                "[early-irecv] failed to rebuild %s on worker rank=%s; "
                "early-irecv disabled (consume points submit recv "
                "synchronously)",
                hint_env,
                getattr(self, "rank", "?"),
            )
            return
        # Reverse completion-report channel: THIS worker is the writer
        # (MessageQueue creators are writers; create_from_handle yields a
        # reader).  The engine core attaches a reader via the READY
        # handshake (WorkerProc.irecv_done_mq), so no env handle is needed
        # in this direction.
        self._irecv_done_mq = MessageQueue(
            1, 1, max_chunk_bytes=1024, max_chunks=64,
        )
        self.irecv_done_mq = self._irecv_done_mq
        self._early_recv_comm_active = True
        self._edge_cloud_comm_thread = threading.Thread(
            target=self._edge_cloud_comm_loop,
            name="edge-cloud-comm",
            daemon=True,
        )
        self._edge_cloud_comm_thread.start()
        logger.info(
            "[early-irecv] comm thread started on worker rank=%s (hint=%s)",
            getattr(self, "rank", "?"),
            hint_env,
        )

    def _edge_cloud_comm_loop(self) -> None:
        """Comm thread body: drain recv-hints -> submit_recv; report
        completions.

        This thread ONLY dequeues hints, submits recvs, queries
        future.done() and enqueues completion reports -- it never wait()s a
        future and never touches the model, so busy_loop's HCCL usage is
        undisturbed (HCCL does not tolerate a cross-thread wait on a
        channel irecv while busy_loop issues isend on that same channel).
        """
        self._edge_cloud_comm_ready.wait()
        try:
            current_platform.set_device(self.device)
        except Exception:
            logger.exception("[early-irecv] comm thread failed to set device")
        hint_mq = self._irecv_hint_mq
        while True:
            try:
                method, args, _kwargs, _output_rank = hint_mq.dequeue(
                    timeout=0.0005
                )
            except TimeoutError:
                pass
            except Exception:
                # Anything other than TimeoutError (e.g. a torn-down MQ at
                # shutdown): log and avoid a hot spin; the daemon thread
                # dies with the process.
                logger.exception("[early-irecv] hint MQ dequeue error")
                time.sleep(0.01)
            else:
                if method == b"irecv_hint" and args:
                    try:
                        self.start_early_irecv(args[0])
                    except Exception:
                        logger.exception(
                            "[early-irecv] start_early_irecv failed"
                        )
            self._report_irecv_completions()
            # The shm MessageQueue reader busy-spins (sched_yield) for the
            # whole dequeue timeout when traffic is recent
            # (SpinCondition.busy_loop_s=1s), so a short timeout alone
            # does NOT bound CPU usage -- this sleep does.  Keeps the comm
            # thread off the GIL/driver-lock while preserving ~10ms-scale
            # hint/watermark latency.
            time.sleep(0.002)

    def _report_irecv_completions(self) -> None:
        """Report every completed registered recv on irecv_done_mq.

        Per-channel completion is FIFO in-order, so the engine core's
        max-seqno watermark is an exact readiness predicate.  Entries
        already consumed by busy_loop keep being polled here: consumption
        means the forward is using the data, so done() turns true quickly
        and the watermark can advance.  An entry is dropped once it is
        both reported and consumed.
        """
        done_mq = self._irecv_done_mq
        if done_mq is None:
            return
        with self._early_recv_lock:
            entries = [
                (key, future)
                for key, future in self._early_recv_futures.items()
                if key not in self._early_recv_reported
            ]
        for (channel, seqno), future in entries:
            try:
                completed = future.done()
            except Exception:
                logger.exception(
                    "[early-irecv] done() query failed channel=%s seqno=%d",
                    channel.value,
                    seqno,
                )
                continue
            if not completed:
                continue
            try:
                done_mq.enqueue((channel, seqno))
            except Exception:
                # Retry next round; the watermark simply advances later.
                logger.exception(
                    "[early-irecv] completion report enqueue failed "
                    "channel=%s seqno=%d",
                    channel.value,
                    seqno,
                )
                continue
            logger.info(
                "[early-irecv] completion reported channel=%s seqno=%d",
                channel.value,
                seqno,
            )
            with self._early_recv_lock:
                key = (channel, seqno)
                self._early_recv_reported.add(key)
                if key in self._early_recv_consumed:
                    self._early_recv_futures.pop(key, None)
                    self._early_recv_reported.discard(key)
                    self._early_recv_consumed.discard(key)

    def start_early_irecv(self, hint: dict) -> None:
        """Turn one scheduler recv-hint into a pre-posted irecv.

        Hint schema: ``edge_cloud_comm.scheduler_link`` (batch_type /
        draft_prefill_phase / seqno / num_tokens / has_mrope /
        draft_step_idx).  Called only from this worker's comm thread.
        Atomic check-or-submit under ``_early_recv_lock``: a repeated hint
        for the same (channel, seqno) -- or one whose recv a consume point
        already submitted via get_or_post_early_recv -- is a no-op, so
        exactly one recv is ever submitted per seqno on a channel,
        avoiding the double-post deadlock where two irecvs on the same
        channel would race for the sender's single isend.  Hints are a
        correctness dependency under readiness gating and are never
        dropped for flow control; the registry is bounded by the
        scheduler's in-flight window.
        """
        batch_type = hint.get("batch_type")
        seqno = hint.get("seqno")
        num_tokens = hint.get("num_tokens")
        if batch_type is None or seqno is None or num_tokens is None:
            logger.warning(
                "[early-irecv] malformed hint %s, skipping.",
                {
                    k: hint.get(k)
                    for k in ("batch_type", "seqno", "num_tokens")
                },
            )
            return
        kind = kind_for_batch_type(batch_type)
        channel = channel_for(batch_type, kind)
        draft_meta = None
        sp_chunk = False
        if kind in (BatchKind.PREFILL_DRAFT, BatchKind.DECODE_DRAFT):
            # Draft wire derives its schema from the scheduled-draft meta;
            # FIRST travels edge->cloud, LAST cloud->edge.
            direction = (
                "e2c"
                if batch_type
                in (BatchType.PREFILL_DRAFT_FIRST, BatchType.DECODE_DRAFT_FIRST)
                else "c2e"
            )
            draft_meta = self._build_draft_tensor_meta(
                direction,
                int(hint.get("draft_step_idx") or 0),
                num_tokens,
            )
        elif batch_type in (BatchType.PREFILL_FIRST, BatchType.DECODE_FIRST):
            # Cloud-side head recv: mirror _execute_model_cloud's SP gate.
            sp_chunk = enable_sp() and (
                self.model_runner.edge_cloud_cfg.mode != "embedding_only"
                or not self.model_runner.supports_mm_inputs)
        else:
            # Edge-side tail recv: mirror _execute_model_edge_tail.
            sp_chunk = enable_sp()
        request = CommRequest(
            channel=channel,
            op="recv",
            kind=kind,
            num_tokens=num_tokens,
            seqno=seqno,
            sp_chunk=sp_chunk,
            include_mrope=bool(hint.get("has_mrope", True)),
            draft_meta=draft_meta,
            # Shared-model topology: the edge sits at in-group rank 0.
            src_dst=(
                0 if self.parallel_config.is_shared_model_edge else None
            ),
        )
        key = (channel, seqno)
        with self._early_recv_lock:
            if key in self._early_recv_futures:
                return  # idempotent: another hint already submitted
            if key in self._early_recv_consumed:
                return  # busy_loop already consumed (submitted its own)
            _t0 = time.monotonic()
            future = get_comm_service().submit_recv(request)
            _dt_ms = (time.monotonic() - _t0) * 1000
            self._early_recv_futures[key] = future  # cache for busy_loop
        if _dt_ms > 5.0:
            logger.info(
                "[early-irecv] submit_recv took %.1f ms channel=%s seqno=%d",
                _dt_ms,
                channel.value,
                seqno,
            )
        logger.debug(
            "[early-irecv] early-recv submitted channel=%s seqno=%d "
            "num_tokens=%d",
            channel.value,
            seqno,
            request.num_tokens,
        )

    def get_or_post_early_recv(self, request: CommRequest) -> CommFuture:
        """Attach the comm thread's pre-posted recv future, or submit one.

        execute_model calls this instead of pop-then-fallback: under
        ``_early_recv_lock`` it reuses the cached future if the comm
        thread already submitted one for (channel, seqno), otherwise it
        submits the recv itself with the same request (same seqno) and
        registers it, so completion reporting never gaps.  The entry is
        NOT popped here -- it is marked consumed and dropped once it is
        both consumed and reported (the reporter keeps polling done() on
        consumed entries to advance the watermark).  Workers without the
        comm infrastructure (non-PP-first ranks, PD off) skip the registry
        and submit directly.
        """
        if request.seqno is None or not self._early_recv_comm_active:
            # Non-PP-first ranks sit in a singleton PP group, and the comm
            # service collapses every logical channel onto the singleton
            # default device group (service.py's world_size<=1 shortcut),
            # so ALL of this rank's recvs share one CommChannel and one
            # seqno counter -- the independent per-type seqno spaces
            # (prefill/decode/draft each start at 0) would collide there.
            # These ranks perform no real cross-node op (their futures
            # complete immediately), so ordering needs no seqno: submit
            # unsequenced, keeping the collapsed channel uniformly
            # unsequenced (sequenced/unsequenced must not mix).
            if (
                request.seqno is not None
                and get_pp_group().world_size <= 1
            ):
                request = replace(request, seqno=None)
            return get_comm_service().submit_recv(request)
        key = (request.channel, request.seqno)
        with self._early_recv_lock:
            self._early_recv_consumed.add(key)
            future = self._early_recv_futures.get(key)
            if future is not None:
                if key in self._early_recv_reported:
                    # Completion already reported: consumption ends the
                    # entry's lifecycle.
                    self._early_recv_futures.pop(key, None)
                    self._early_recv_reported.discard(key)
                    self._early_recv_consumed.discard(key)
                return future
            future = get_comm_service().submit_recv(request)
            self._early_recv_futures[key] = future
            return future

    @staticmethod
    def _require_comm_seqno(scheduler_output: "SchedulerOutput") -> int:
        """The per-channel seqno stamped by the edge scheduler.

        Every batch that carries cross-node traffic on the six directional
        channels must be stamped (FIRST/LAST share the value; draft steps
        take draft_seqno_base + draft_step_idx).  A missing stamp is a
        scheduler bug: submitting unsequenced would silently mix with the
        sequenced traffic on the same channel, so fail loudly instead.
        """
        seqno = getattr(scheduler_output, "comm_seqno", None)
        if seqno is None:
            raise RuntimeError(
                "SchedulerOutput missing comm_seqno on a sequenced "
                f"edge-cloud path (batch_type={scheduler_output.batch_type}); "
                "the edge scheduler must stamp every PF/DF/DRF pick."
            )
        return seqno

    def _all_gather_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """All-gather tensors across the local TP group along sequence dim.

        Used in edge-cloud mode when edge and cloud have different SP sizes.
        Before cross-PP send, each side must aggregate its SP shards back to
        the full sequence so the remote side can re-chunk with its own SP size.

        Only the all-gather happens here; the gathered tensor is *not* padded
        to the remote TP size.  The sender transmits only the real
        ``num_tokens`` rows (sliced in edge_cloud_isend_tensor_dict via the
        ``num_tokens`` argument), and the receiver zero-pads its buffer up to
        its own local TP size (see ``_pad_num_tokens_to_tp_multiple``).  So a
        send-side pad to the remote TP size is redundant — its dim-0 rows are
        sliced off before send — and for 3D ``(num_tokens, hc_mult, hidden)``
        tensors (DeepSeek V4) it is actively harmful: ``F.pad(t, (0, 0, 0,
        pad_len))`` pads the hc_mult axis (second-to-last), not the sequence
        axis, corrupting the tensor and tripping the isend non-dim-0 shape
        check.
        """
        tp_group = get_tp_group()
        result = {}
        for key, tensor in tensor_dict.items():
            if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                gathered = tp_group.all_gather(tensor, dim=0)
                result[key] = gathered
            else:
                result[key] = tensor
        return result

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        layer_slice_info: Any = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        batch_type = scheduler_output.batch_type
        use_alt_group = (batch_type == SchedulerBatchType.ALL_DECODE)

        if envs_ascend.MSMONITOR_USE_DAEMON:
            dp.step()

        # Edge-cloud sends/recvs are owned by the comm service: per-channel
        # FIFO + wait-bridge ordering plus send-buffer keepalive make the
        # legacy entry wait unnecessary (see edge_cloud_comm_design.md
        # section 5).  Drive completion callbacks/sinks from this loop head
        # instead (one head-of-line event.query() per active channel).
        if self.model_runner._edge_cloud_enabled:
            get_comm_service().poll_completions()

        # Edge-cloud PD-separation: dispatch by batch_type and role.
        if self.model_runner._edge_cloud_enabled:
            bt = scheduler_output.batch_type
            if is_cloud_device():
                if bt in (
                    BatchType.PREFILL_DRAFT_FIRST,
                    BatchType.DECODE_DRAFT_FIRST,
                ):
                    return self._execute_model_cloud_draft(scheduler_output)
                return self._execute_model_cloud(
                    scheduler_output, layer_slice_info
                )
            if bt in (
                BatchType.PREFILL_DRAFT_FIRST,
                BatchType.DECODE_DRAFT_FIRST,
            ):
                return self._execute_model_edge_draft_head(scheduler_output)
            if bt in (
                BatchType.PREFILL_DRAFT_LAST,
                BatchType.DECODE_DRAFT_LAST,
            ):
                return self._execute_model_edge_draft_tail(scheduler_output)
            if bt in (BatchType.PREFILL_FIRST, BatchType.DECODE_FIRST):
                return self._execute_model_edge_head(
                    scheduler_output, layer_slice_info
                )
            if bt in (BatchType.PREFILL_LAST, BatchType.DECODE_LAST):
                return self._execute_model_edge_tail(
                    scheduler_output, layer_slice_info
                )

        # Fallback: original path for non-edge-cloud or unhandled batch types.
        return self._execute_model_legacy(
            scheduler_output, layer_slice_info, use_alt_group
        )

    def _execute_model_edge_head(
        self,
        scheduler_output: "SchedulerOutput",
        layer_slice_info: Any,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """Edge head segment (PF/DF): segment_a -> isend -> suspend -> return EMPTY."""
        logger.info(
            "[PD-TIMING] edge head enter batch_type=%s ts=%.4f",
            scheduler_output.batch_type,
            time.monotonic(),
        )
        output = self.model_runner.execute_model(
            scheduler_output, intermediate_tensors=None,
            layer_slice_info=layer_slice_info,
        )
        logger.info(
            "[PD-TIMING] edge head forward done batch_type=%s ts=%.4f",
            scheduler_output.batch_type,
            time.monotonic(),
        )
        is_last_slice = (
            layer_slice_info is None or layer_slice_info.is_last_slice
        )
        if not is_last_slice:
            return None

        if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput, NoneType)):
            return output

        assert isinstance(output, IntermediateTensors)
        # Edge-cloud with heterogeneous SP: aggregate SP shards to full
        # sequence before cross-PP send so cloud can re-chunk by its SP.
        if enable_sp() and (self.model_runner.edge_cloud_cfg.mode != "embedding_only"
            or not self.model_runner.supports_mm_inputs):
            _gathered = self._all_gather_tensor_dict(output.tensors)
        else:
            _gathered = output.tensors
        # For M-RoPE VL models, edge has already computed the per-token
        # mrope positions (which needs image_grid_thw that did not cross
        # the edge->cloud mm_features boundary). Push them alongside
        # hidden_states so cloud can reuse them instead of recomputing
        # (and hitting the missing grid_thw). Transpose [3, N] -> [N, 3]
        # so the sequence axis is dim-0, matching hidden_states and the
        # e2c transfer's dim-0 slicing / SP-gather path.
        # Skip for text-only batches: cloud computes M-RoPE locally then
        # (empty mm_features degrades to 1D, no grid_thw needed), saving
        # one P2P RTT.
        # Match the receiver: the cloud recv side derives include_mrope from
        # this SO's `has_mrope` stamp (computed by the edge scheduler from its
        # authoritative request registry).  Do NOT derive it from the edge
        # runner's local registry: finished_req_ids of drained requests can
        # still be locked in a deferred EMPTY batch when this head batch
        # executes, so the local registry LAGS the stamp and would put an
        # mrope_positions message on the wire that the cloud -- receiving by
        # the stamp -- never irecv's -> HCCL rendezvous deadlock on the
        # hidden channel.  Trust the stamp; fall back to the local
        # computation only when the stamp is absent (older peer).
        _stamped_mrope = getattr(scheduler_output, "has_mrope", None)
        if _stamped_mrope is None:
            include_mrope = self.model_runner.step_has_multimodal_req(
                scheduler_output
            )
        else:
            include_mrope = _stamped_mrope
            _local_mrope = self.model_runner.step_has_multimodal_req(
                scheduler_output
            )
            if _local_mrope != include_mrope:
                logger.warning(
                    "[PD] edge sender include_mrope divergence: "
                    "has_mrope stamp=%s but runner-local registry says %s "
                    "(batch_type=%s, head_token=%s); using the stamp.  The "
                    "local registry lag (deferred worker cleanup) should be "
                    "investigated.",
                    include_mrope, _local_mrope,
                    scheduler_output.batch_type,
                    getattr(scheduler_output, "head_token", None),
                )
        if (include_mrope and self.model_runner.uses_mrope
                and "hidden_states" in _gathered):
            n = _gathered["hidden_states"].shape[0]
            _gathered["mrope_positions"] = (
                self.model_runner.mrope_positions.gpu[:, :n].t().contiguous()
            )
        if get_pp_group().world_size == 2:
            _kind = kind_for_batch_type(scheduler_output.batch_type)
            get_comm_service().submit_send(
                CommRequest(
                    channel=channel_for(scheduler_output.batch_type, _kind),
                    op="send",
                    kind=_kind,
                    num_tokens=scheduler_output.total_num_scheduled_tokens,
                    seqno=self._require_comm_seqno(scheduler_output),
                    tensor_dict=_gathered,
                    include_mrope=include_mrope,
                ))
        # Return a placeholder output that carries the request IDs so the
        # scheduler can correlate the batch, but contains no sampled tokens
        # because sampling happens in the tail segment (PL/DL).
        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        )

    def _execute_model_edge_tail(
        self,
        scheduler_output: "SchedulerOutput",
        layer_slice_info: Any,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        edge_sp = enable_sp()
        """Edge tail segment (PL/DL): recv -> segment_e -> return output."""
        logger.info(
            "[PD-TIMING] edge tail enter batch_type=%s ts=%.4f",
            scheduler_output.batch_type,
            time.monotonic(),
        )
        # The cloud->edge reply never carries mrope_positions (the c2e meta
        # is built with uses_mrope=False: only the edge computes M-RoPE and
        # pushes it to the cloud).  Pass include_mrope=False explicitly so
        # this recv stays correct if the c2e meta ever gains an mrope key --
        # with the default True, the edge would irecv a tensor the cloud
        # never sends and deadlock on the channel.
        _kind = kind_for_batch_type(scheduler_output.batch_type)
        # Attach the comm thread's pre-posted recv (hinted when the
        # matching FIRST batch was published), or submit the recv now with
        # the same seqno on a hint miss.
        recv_future = self.get_or_post_early_recv(
            CommRequest(
                channel=channel_for(scheduler_output.batch_type, _kind),
                op="recv",
                kind=_kind,
                num_tokens=scheduler_output.total_num_scheduled_tokens,
                seqno=self._require_comm_seqno(scheduler_output),
                sp_chunk=edge_sp,
                include_mrope=False,
            ))

        intermediate_tensors = recv_future.as_intermediate_tensors()
        output = self.model_runner.execute_model(
            scheduler_output, intermediate_tensors,
            layer_slice_info=layer_slice_info,
        )

        is_last_slice = (
            layer_slice_info is None or layer_slice_info.is_last_slice
        )
        if not is_last_slice:
            return None

        if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput, NoneType)):
            return output
        return output

    def _execute_model_cloud(
        self,
        scheduler_output: "SchedulerOutput",
        layer_slice_info: Any,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """Cloud middle segment: recv -> segment_b/c -> isend -> return."""
        # logger.info(
        #     f"Execute model, batch_type: {scheduler_output.batch_type}, " + (
        #         f"slice: {layer_slice_info.slice_index + 1}/{layer_slice_info.total_slices}, "
        #         f"layers: [{layer_slice_info.start_layer},{layer_slice_info.end_layer})"
        #         if layer_slice_info is not None
        #         else ""
        #     )
        # )
        intermediate_tensors = None
        is_first_slice = (
            layer_slice_info is None or layer_slice_info.is_first_slice
        )
        if is_first_slice:
            logger.info(
                "[PD-TIMING] cloud middle enter batch_type=%s ts=%.4f",
                scheduler_output.batch_type,
                time.monotonic(),
            )
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        # Always run _update_states for the first slice (or unsliced batch),
        # even when total_num_scheduled_tokens==0.  Some requests may not
        # contribute tokens to this slice but their state must still be
        # initialised in the cloud worker's all_token_ids, otherwise a
        # subsequent DECODE_FIRST / DRAFT_FIRST will KeyError in _update_states.
        if is_first_slice:
            self.model_runner.cloud_prepare_early(scheduler_output)
        if forward_pass and is_first_slice:
            # [early-irecv] Attach the comm thread's pre-posted recv future
            # (hinted the moment the edge SO arrived), or submit the recv
            # ourselves with the same seqno on a hint miss.
            # get_or_post_early_recv guarantees at most one recv per
            # (channel, seqno) even when the comm thread's hint dequeue
            # races this consume point.  The comm thread ONLY submits
            # (never wait()s); ordering is done lazily via
            # as_intermediate_tensors() on the busy_loop thread.  Each
            # directional channel owns a dedicated communicator and stream,
            # so an early-posted irecv can never block (or be blocked by)
            # sends of any type.
            # Match the sender. The edge scheduler stamps `has_mrope` on
            # every SO from its authoritative request registry, and the edge
            # sender's include_mrope always equals it. The cloud runner's own
            # registry LAGS behind (finished_req_ids flushed via EMPTY batches
            # are dropped before reaching the cloud, and DECODE_FIRST SOs are
            # published before the pending-finish merge), so computing from
            # the local registry can disagree with the edge sender after an
            # mm->text traffic transition and deadlock the HCCL recv. Trust
            # the stamp; fall back to the local computation only if the stamp
            # is absent (older edge).
            _cloud_include_mrope = getattr(scheduler_output, "has_mrope", None)
            if _cloud_include_mrope is None:
                _cloud_include_mrope = self.model_runner.step_has_multimodal_req(
                    scheduler_output)
            # SP chunking is part of the recv postprocess for both
            # merged and non-merged payloads. It must run only after
            # the receive and TP broadcast have completed.
            do_sp_chunk = enable_sp() and (
                self.model_runner.edge_cloud_cfg.mode != "embedding_only"
                or not self.model_runner.supports_mm_inputs)
            # In the shared-model edge-cloud topology the edge has a single
            # distributed rank at in-group rank 0; the cloud first-worker of
            # each dp_rank must receive from that rank (src=0).  In the
            # standard (non-shared-model) topology src=None suffices: it
            # resolves to the implicit "previous PP rank" which IS the edge.
            _recv_src = 0 if self.parallel_config.is_shared_model_edge else None
            _kind = kind_for_batch_type(scheduler_output.batch_type)
            recv_future = self.get_or_post_early_recv(
                CommRequest(
                    channel=channel_for(scheduler_output.batch_type, _kind),
                    op="recv",
                    kind=_kind,
                    num_tokens=scheduler_output.total_num_scheduled_tokens,
                    seqno=self._require_comm_seqno(scheduler_output),
                    sp_chunk=do_sp_chunk,
                    src_dst=_recv_src,
                    include_mrope=_cloud_include_mrope,
                ))
            # wait_event ordering + comm postprocess (the TP collective)
            # run lazily on first .tensors access inside execute_model,
            # on all ranks synchronized.  Do NOT force completion here:
            # doing so blocks busy_loop on the recv BEFORE
            # execute_model, defeating cloud_prepare_early's overlap
            # and stalling the pipeline (TPOT regression).
            intermediate_tensors = recv_future.as_intermediate_tensors()
        if self.profiler is not None:
            self.profiler.step()

        output = self.model_runner.execute_model(
            scheduler_output, intermediate_tensors,
            layer_slice_info=layer_slice_info,
        )

        is_last_slice = (
            layer_slice_info is None or layer_slice_info.is_last_slice
        )
        if not is_last_slice:
            return None

        if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput, NoneType)):
            return output

        assert isinstance(output, IntermediateTensors)
        # Edge-cloud with heterogeneous SP: aggregate SP shards to full
        # sequence before cross-PP send so edge can re-chunk by its SP.
        if enable_sp():
            _gathered = self._all_gather_tensor_dict(output.tensors)
        else:
            _gathered = output.tensors

        # In the shared-model edge-cloud topology the cloud
        # first-worker of each dp_rank is in the shared PP group
        # with the edge and must send its middle-layer output
        # back to the edge (in-group rank 0). Other cloud
        # workers (TP non-first) are in singleton PP groups
        # Send intermediate tensors to edge.  In the shared-model topology the
        # edge sits at in-group rank 0, so dst=0 is needed.  Otherwise dst=None
        # resolves to the implicit "next PP rank" which IS the edge.
        if get_pp_group().world_size > 1:
            _send_dst = 0 if self.parallel_config.is_shared_model_edge else None
            _kind = kind_for_batch_type(scheduler_output.batch_type)
            get_comm_service().submit_send(
                CommRequest(
                    # The batch arrived on the UP wire; the reply travels on
                    # the DOWN wire of the same family.
                    channel=channel_for_direction(_kind, up=False),
                    op="send",
                    kind=_kind,
                    num_tokens=scheduler_output.total_num_scheduled_tokens,
                    seqno=self._require_comm_seqno(scheduler_output),
                    tensor_dict=_gathered,
                    src_dst=_send_dst,
                ))
        return output

    def _build_draft_tensor_meta(
        self,
        direction: str,
        draft_step_idx: int,
        num_tokens: int,
    ) -> ScheduledDraftTensorMeta | None:
        """Build the scheduled draft wire schema for one draft step.

        DeepSeek-V4 MTP uses a full-sequence wire contract under SP, so its
        shape is independent of either peer's TP size and can use this static
        schema. Other draft adapters still retain sender-side shard shapes and
        remain on the dynamic compatibility path.
        """
        if (
            enable_sp()
            and not self.model_runner._is_deepseek_v4_mtp_edge_cloud_draft()
        ):
            return None

        speculative_config = self.model_runner.speculative_config
        drafter = self.model_runner.drafter
        if (
            speculative_config is None
            or speculative_config.method is None
            or drafter is None
        ):
            return None

        hidden_size = drafter.hidden_size
        hc_mult = 1
        draft_model = getattr(drafter, "model", None)
        if (
            getattr(draft_model, "edge_cloud_draft_kind", None)
            == "deepseek_v4_mtp"
        ):
            draft_model_config = speculative_config.draft_model_config
            draft_hf_config = draft_model_config.hf_config
            hc_mult = int(getattr(draft_hf_config, "hc_mult", 1))
            hidden_size = draft_model_config.get_hidden_size()
        return build_scheduled_draft_tensor_meta(
            method=speculative_config.method,
            direction=direction,
            draft_step_idx=draft_step_idx,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            dtype=self.model_runner.dtype,
            hc_mult=hc_mult,
        )

    @staticmethod
    def _scheduled_draft_num_tokens(
        scheduler_output: "SchedulerOutput",
    ) -> int:
        draft_step_idx = int(scheduler_output.draft_step_idx or 0)
        return int(
            scheduler_output.total_num_scheduled_tokens
            if draft_step_idx == 0
            else len(scheduler_output.num_scheduled_tokens)
        )

    def _scheduled_draft_tensor_meta(
        self,
        scheduler_output: "SchedulerOutput",
        direction: str,
    ) -> ScheduledDraftTensorMeta | None:
        """Derive the scheduled draft wire schema on both peers.

        Step 0 runs over the parent batch's full token count, later steps
        over one token per request.
        """
        draft_step_idx = int(scheduler_output.draft_step_idx or 0)
        num_tokens = (
            scheduler_output.total_num_scheduled_tokens
            if draft_step_idx == 0
            else len(scheduler_output.num_scheduled_tokens)
        )
        return self._build_draft_tensor_meta(
            direction, draft_step_idx, num_tokens
        )

    def _execute_model_cloud_draft(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput:
        """Run one cloud-side independently scheduled draft middle step.

        Owns the cross-PP edge-cloud communication, mirroring
        ``_execute_model_cloud``: recv the edge->cloud draft payload, run
        the cloud target/C segment forward (in the model_runner), then send
        the cloud->edge result. The send is recorded (not waited); the edge
        self-posts DRAFT_LAST when it schedules DRAFT_FIRST, so its matching
        receive can be posted without a worker-ack/POST_OUT round trip.
        """
        logger.info(f"Execute model, batch_type: {scheduler_output.batch_type}")
        # Prefill-phase draft chains travel on the PREFILL_DRAFT channel
        # pair, decode-phase chains on the DECODE pair.
        _kind = kind_for_batch_type(scheduler_output.batch_type)
        _seqno = self._require_comm_seqno(scheduler_output)
        recv_tensor_meta = self._scheduled_draft_tensor_meta(
            scheduler_output,
            "e2c",
        )
        # Attach the comm thread's pre-posted recv (hinted when the parent
        # batch arrived), or submit the recv now with the same seqno.
        recv_future = self.get_or_post_early_recv(
            CommRequest(
                channel=channel_for(scheduler_output.batch_type, _kind),
                op="recv",
                kind=_kind,
                num_tokens=scheduler_output.total_num_scheduled_tokens,
                seqno=_seqno,
                draft_meta=recv_tensor_meta,
            ))
        # Lazy consumption: wait_event ordering + TP-broadcast postprocess
        # run on first .tensors access inside the middle segment.
        output = self.model_runner._run_edge_cloud_draft_middle_segment(
            scheduler_output, recv_future.as_intermediate_tensors()
        )
        out_tensor_dict = (
            self.model_runner._gather_deepseek_v4_mtp_draft_tensors(
                output.tensors,
                self._scheduled_draft_num_tokens(scheduler_output),
            )
        )
        if get_pp_group().world_size == 2:
            out_tensor_dict = {
                key: value.contiguous()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in out_tensor_dict.items()
            }
            # Submit only -- the service owns completion; do NOT wait.  See
            # method docstring.
            send_tensor_meta = self._scheduled_draft_tensor_meta(
                scheduler_output,
                "c2e",
            )
            get_comm_service().submit_send(
                CommRequest(
                    channel=channel_for_direction(_kind, up=False),
                    op="send",
                    kind=_kind,
                    num_tokens=scheduler_output.total_num_scheduled_tokens,
                    seqno=_seqno,
                    tensor_dict=out_tensor_dict,
                    draft_meta=send_tensor_meta,
                ))
            logger.info(
                "Send intermediate tensors to edge, "
                f"hidden_channel: {channel_for_direction(_kind, up=False).value}"
            )
        logger.info(
            f"Execute model, batch_type: {scheduler_output.batch_type}, after."
        )
        req_ids = list(scheduler_output.num_scheduled_tokens)
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        )

    def _dummy_draft_head_payload(
        self, scheduler_output: "SchedulerOutput"
    ) -> IntermediateTensors:
        """Zero payload matching the scheduled-draft e2c wire shape.

        Used for dead draft chains (all covered requests finished or were
        aborted): the draft context is already gone, but the cloud's
        pre-posted recv and the channel's seqno sequence must still be
        satisfied — zeros need no context.
        """
        meta = self._scheduled_draft_tensor_meta(scheduler_output, "e2c")
        if meta is None:
            raise RuntimeError(
                "dummy draft head payload requires a static draft wire "
                "meta (unavailable with SP/dynamic draft transport)"
            )
        metas = dict(meta.metadata_list)
        tensor_dict = {}
        for key in meta.send_tensor_keys:
            tm = metas[key]
            tensor_dict[key] = torch.zeros(
                tm.size, dtype=tm.dtype, device=self.device
            )
        return IntermediateTensors(tensor_dict)

    def _execute_model_edge_draft_head(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput:
        """Run and send one edge-side scheduled draft first segment."""
        logger.info(f"Execute model, batch_type: {scheduler_output.batch_type}")
        if scheduler_output.draft_chain_dead:
            # Dead chain (all covered requests finished/aborted): skip the
            # draft forward — its context is gone — and send zeros of the
            # exact wire shape so the channel seqno sequence and the
            # cloud's pre-posted recvs stay paired.
            self._publish_edge_draft_head(
                scheduler_output,
                self._dummy_draft_head_payload(scheduler_output),
            )
        elif not self.model_runner.is_edge_cloud_draft_step_ready(
            scheduler_output
        ):
            # A pre-generated MTP DRAFT_FIRST may overtake the DRAFT_LAST
            # that produces its inputs in the async executor queue.  Do not
            # block that head in-place (the tail could be queued behind it);
            # suspend its control state and let the preceding tail resume it
            # locally (see _resume_deferred_edge_draft_head).
            task_id = scheduler_output.draft_task_id
            if task_id is None:
                raise RuntimeError("DRAFT_FIRST missing draft_task_id")
            step = int(scheduler_output.draft_step_idx or 0)
            key = (task_id, step)
            if key in self._deferred_edge_draft_heads:
                raise RuntimeError(
                    "Duplicate deferred DRAFT_FIRST: "
                    f"task_id={task_id}, step={step}"
                )
            self.model_runner.mark_edge_cloud_draft_step_deferred(
                scheduler_output
            )
            self._deferred_edge_draft_heads[key] = scheduler_output
        else:
            self._run_and_send_edge_draft_head(scheduler_output)
        req_ids = list(scheduler_output.num_scheduled_tokens)
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        )

    def _run_and_send_edge_draft_head(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Execute a ready draft head and publish its cloud-bound payload."""
        output = self.model_runner._run_edge_cloud_draft_first_segment(
            scheduler_output
        )
        if not isinstance(output, IntermediateTensors):
            raise RuntimeError("DRAFT_FIRST did not produce intermediates")
        self._publish_edge_draft_head(scheduler_output, output)

    def _publish_edge_draft_head(
        self,
        scheduler_output: "SchedulerOutput",
        output: IntermediateTensors,
    ) -> None:
        """Gather (DSV4-MTP under SP) and send one draft-head payload.

        Shared by the live path (real first-segment output) and the
        dead-chain path (zero dummy payload of the exact wire shape).
        """
        tensor_dict = (
            self.model_runner._gather_deepseek_v4_mtp_draft_tensors(
                output.tensors,
                self._scheduled_draft_num_tokens(scheduler_output),
            )
        )
        if get_pp_group().world_size == 2:
            tensor_dict = {
                key: value.contiguous()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in tensor_dict.items()
            }
            send_tensor_meta = self._scheduled_draft_tensor_meta(
                scheduler_output,
                "e2c",
            )
            # Prefill-phase draft chains travel on the PREFILL_DRAFT
            # channel pair, decode-phase chains on the DECODE pair.
            _kind = kind_for_batch_type(scheduler_output.batch_type)
            get_comm_service().submit_send(
                CommRequest(
                    channel=channel_for(scheduler_output.batch_type, _kind),
                    op="send",
                    kind=_kind,
                    num_tokens=scheduler_output.total_num_scheduled_tokens,
                    seqno=self._require_comm_seqno(scheduler_output),
                    tensor_dict=tensor_dict,
                    draft_meta=send_tensor_meta,
                ))
            logger.info(
                "Send intermediate tensors to cloud, "
                f"hidden_channel: "
                f"{channel_for(scheduler_output.batch_type, _kind).value}"
            )

    def _resume_deferred_edge_draft_head(
        self, completed_tail: "SchedulerOutput"
    ) -> None:
        task_id = completed_tail.draft_task_id
        if task_id is None:
            return
        next_step = int(completed_tail.draft_step_idx or 0) + 1
        key = (task_id, next_step)
        deferred = self._deferred_edge_draft_heads.get(key)
        if deferred is None:
            return
        if not self.model_runner.is_edge_cloud_draft_step_ready(deferred):
            raise RuntimeError(
                "Deferred DRAFT_FIRST is still not ready after its "
                "preceding tail completed: "
                f"task_id={task_id}, step={next_step}"
            )
        self._deferred_edge_draft_heads.pop(key)
        self.model_runner.unmark_edge_cloud_draft_step_deferred(deferred)
        self._run_and_send_edge_draft_head(deferred)

    def _execute_model_edge_draft_tail(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput:
        """Receive and finish one edge-side scheduled draft step."""
        logger.info(f"Execute model, batch_type: {scheduler_output.batch_type}")
        # Prefill-phase draft chains travel on the PREFILL_DRAFT channel
        # pair, decode-phase chains on the DECODE pair.
        _kind = kind_for_batch_type(scheduler_output.batch_type)
        recv_tensor_meta = self._scheduled_draft_tensor_meta(
            scheduler_output,
            "c2e",
        )
        # Attach the comm thread's pre-posted recv (hinted when the parent
        # batch was published), or submit the recv now with the same seqno.
        recv_future = self.get_or_post_early_recv(
            CommRequest(
                channel=channel_for(scheduler_output.batch_type, _kind),
                op="recv",
                kind=_kind,
                num_tokens=scheduler_output.total_num_scheduled_tokens,
                seqno=self._require_comm_seqno(scheduler_output),
                draft_meta=recv_tensor_meta,
            ))
        logger.info(
            "Receive intermediate tensors from cloud after, "
            f"hidden_channel: "
            f"{channel_for(scheduler_output.batch_type, _kind).value}"
        )
        # Lazy consumption: wait_event ordering + postprocess run on first
        # .tensors access inside the last segment.
        output = self.model_runner._run_edge_cloud_draft_last_segment(
            scheduler_output, recv_future.as_intermediate_tensors()
        )
        self._resume_deferred_edge_draft_head(scheduler_output)
        return output

    def _execute_model_legacy(
        self,
        scheduler_output: "SchedulerOutput",
        layer_slice_info: Any,
        use_alt_group: bool,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """Original non-edge-cloud path (standard PP, layer-slicing, etc.)."""
        # Wait the one outstanding legacy send before this batch's compute
        # may reuse its source buffer (edge-cloud sends never land here;
        # they are owned by the comm service).
        self._wait_pp_send_work()
        # Only receive intermediate tensors on the first slice.
        is_first_slice = (
            layer_slice_info is None or layer_slice_info.is_first_slice
        )

        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        if forward_pass and is_first_slice:
            if not get_pp_group().is_first_rank:
                if enable_sp():
                    all_gather_group = None
                else:
                    all_gather_group = get_tp_group()
                tensor_dict, comm_handles, comm_postprocess = get_pp_group().irecv_tensor_dict(
                    all_gather_group=all_gather_group,
                    use_alt_group=use_alt_group,
                )
                assert tensor_dict is not None, (
                    "worker irecv_tensor_dict returned None, "
                    "previous stage may have failed to send."
                )
                intermediate_tensors = AsyncIntermediateTensors(
                    tensor_dict,
                    comm_handles=comm_handles,
                    comm_postprocess=comm_postprocess,
                )

        if self.profiler is not None:
            self.profiler.step()

        output = self.model_runner.execute_model(
            scheduler_output, intermediate_tensors,
            layer_slice_info=layer_slice_info,
        )

        is_last_slice = (
            layer_slice_info is None or layer_slice_info.is_last_slice
        )
        if not is_last_slice:
            return None

        if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput, NoneType)):
            return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        if not get_pp_group().is_last_rank:
            assert parallel_config.distributed_executor_backend != "external_launcher"
            if enable_sp():
                all_gather_group = None
            else:
                all_gather_group = get_tp_group()
            self._pp_send_work = get_pp_group().isend_tensor_dict(
                output.tensors,
                all_gather_group=all_gather_group,
                use_alt_group=use_alt_group,
            )

        kv_connector_output = output.kv_connector_output
        if not kv_connector_output:
            return None

        if not kv_connector_output.finished_sending and not kv_connector_output.finished_recving:
            return EMPTY_MODEL_RUNNER_OUTPUT
        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: "GrammarOutput") -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    def load_model(self) -> None:
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CaMemAllocator.get_instance()
            assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."
            context = allocator.use_memory_pool(tag="weights")
        else:
            from contextlib import nullcontext

            context = nullcontext()  # type: ignore

        with context, set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model()

        if self.vllm_config.weight_transfer_config is not None:
            from vllm.distributed.weight_transfer.factory import (
                WeightTransferEngineFactory,
            )

            # main: create_engine takes (config, parallel_config, model)
            self.weight_transfer_engine = WeightTransferEngineFactory.create_engine(
                self.vllm_config.weight_transfer_config,
                self.vllm_config.parallel_config,
                self.model_runner.get_model(),
            )

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # Note: need to adapt for graph mode.
        warmup_sizes = (self.vllm_config.compilation_config.compile_sizes or []).copy()
        if not self.model_config.enforce_eager:
            cg_capture_sizes: list[int] = []
            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]

            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()
            # For each compile_range, if none of the batch sizes
            # in warmup_sizes or cudagraph_capture_sizes are in the range,
            # add the end of the range to ensure compilation/warmup.
            all_sizes = set(cg_capture_sizes)
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
            for compile_range in compile_ranges:
                if not any(x in compile_range for x in all_sizes):
                    warmup_sizes.append(compile_range.end)

        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size)

        npugraph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            npugraph_memory_bytes = self.model_runner.capture_model()

        # Suggest an optimal --kv-cache-memory value for future runs.
        # Only emitted when we ran full profiling (kv_cache_memory_bytes was not
        # pre-specified) so that peak_activation_memory etc. are available.
        # non_kv_memory already includes NPU graph memory, so the suggestion
        # accounts for all measured memory categories. A 150 MiB buffer is kept
        # because memory_profiling may slightly underestimate non-torch
        # allocations (ACL context, HCCL buffers, driver layer, etc.).
        if self.cache_config.kv_cache_memory_bytes is None and hasattr(self, "peak_activation_memory"):
            redundancy_buffer = 150 * (1 << 20)  # 150 MiB safety margin
            non_kv_memory = (
                self.model_runner.model_memory_usage
                + self.peak_activation_memory
                + self.non_torch_memory
                + npugraph_memory_bytes
            )
            self.npugraph_memory_bytes = npugraph_memory_bytes
            suggested_to_requested = int(self.requested_memory) - non_kv_memory - redundancy_buffer
            suggested_to_gpu_limit = int(self.init_snapshot.free_memory) - non_kv_memory - redundancy_buffer
            msg = (
                f"Free memory on device "
                f"({format_gib(self.init_snapshot.free_memory)}/"
                f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_memory)} GiB). "
                f"Actual usage: {format_gib(self.model_runner.model_memory_usage)} GiB "
                f"for weights, {format_gib(self.peak_activation_memory)} GiB for peak "
                f"activation, {format_gib(self.non_torch_memory)} GiB for non-torch "
                f"memory, {format_gib(npugraph_memory_bytes)} GiB for NPU graph memory. "
                f"Replace gpu_memory_utilization with "
                f"`--kv-cache-memory={suggested_to_requested}` "
                f"({format_gib(suggested_to_requested)} GiB) to fit into requested "
                f"memory, or `--kv-cache-memory={suggested_to_gpu_limit}` "
                f"({format_gib(suggested_to_gpu_limit)} GiB) to fully utilize NPU "
                f"free memory. Current KV cache memory: "
                f"{format_gib(self.available_kv_cache_memory_bytes)} GiB."
            )
            logger.info(msg)

        # Call ATB matmul to warm up; otherwise, the first operation (ReshapeAndCache)
        # may cause performance degradation at runtime.
        if get_ascend_device_type() != AscendDeviceType.A5:
            self._warm_up_atb()
        # Bind after warmup so hot allocations are already materialized on the
        # worker process before migratepages/taskset run.
        if get_ascend_config().enable_cpu_binding:
            try:
                bind_cpus(self.local_rank)
            except Exception as e:
                logger.warning("Bind cpus failed in rank%s: %s Skip binding cpu.", self.local_rank, e)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        return CompilationTimes(
            language_model=self.vllm_config.compilation_config.compilation_time,
            # `encoder_compilation_time` was added after v0.19.1 (vLLM #39240); fall
            # back to 0.0 so the older release still constructs CompilationTimes.
            encoder=getattr(
                self.vllm_config.compilation_config,
                "encoder_compilation_time",
                0.0,
            ),
        )

    def _warm_up_atb(self):
        x = torch.rand((2, 4), dtype=torch.float16).npu()
        weight = torch.rand((2, 4), dtype=torch.float16).npu()
        c = torch.rand((4, 4), dtype=torch.float32).npu()
        torch_npu._npu_matmul_add_fp32(x, weight, c)

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    @torch.inference_mode()
    def profile_prefill_latency(self, num_tokens: int) -> float:
        """
        Profile prefill latency for a given number of tokens.

        This runs a real model forward pass and measures the execution time.
        Used for profiling-based dynamic chunk sizing.

        In PP (Pipeline Parallelism) mode:
        - All workers execute the forward pass to stay synchronized
        - Only the timing from PP0 (first rank) is meaningful for scheduling
        - PP0 includes all the pipeline stages' latency when using async scheduling

        Args:
            num_tokens: Number of tokens to profile

        Returns:
            Latency in milliseconds
        """
        import time

        # Clamp to valid range
        num_tokens = min(num_tokens, self.scheduler_config.max_num_batched_tokens)
        num_tokens = max(num_tokens, 1)

        # Synchronize all devices before timing
        # This ensures clean measurement in PP/TP scenarios
        torch.npu.synchronize()

        # In PP mode, we still run on all ranks to keep them synchronized
        # but only the first rank's timing is used for scheduling decisions
        is_first_pp_rank = get_pp_group().is_first_rank

        start = time.perf_counter()

        # Run real model forward with force_attention=True
        # This ensures attention is actually executed, not skipped.
        # Without force_attention, attn_metadata may be None and attention
        # won't run, making profiling results inaccurate.
        # _dummy_run handles PP internally (intermediate tensors, etc.)
        self.model_runner._dummy_run(
            num_tokens=num_tokens,
            force_attention=True,  # Critical: ensure attention is executed
            profile_cpp=True,
        )

        # Synchronize after forward to ensure NPU operations complete
        torch.npu.synchronize()

        latency_ms = (time.perf_counter() - start) * 1000

        # Log for debugging in PP mode
        if not is_first_pp_rank:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[ProfilingChunk] PP rank %s: profiled %s tokens, latency=%.2f ms (not used)",
                    get_pp_group().rank_in_group,
                    num_tokens,
                    latency_ms,
                )

        return latency_ms

    def get_kv_connector_handshake_metadata(
        self,
    ) -> dict[tuple[int, ...], KVConnectorHandshakeMetadata] | None:
        """Get KV connector metadata from this worker if available."""
        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()

        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None
        tp_rank = get_tp_group().rank_in_group
        pp_rank = get_pp_group().rank_in_group
        pcp_size = get_pcp_group().world_size
        if pcp_size > 1:
            pcp_rank = get_pcp_group().rank_in_group
            return {(pp_rank, pcp_rank, tp_rank): metadata}
        return {(pp_rank, tp_rank): metadata}

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def update_max_model_len(self, max_model_len: int) -> None:
        """Update max_model_len after auto-fit to NPU memory.

        This is called when max_model_len=-1 is used and the engine
        automatically determines the maximum context length that fits
        in GPU memory. Workers need to update their cached max_model_len
        to match the engine's decision.
        """
        self.model_config.max_model_len = max_model_len
        if self.model_runner is not None:
            self.model_runner.update_max_model_len(max_model_len)
        logger.debug("Updated max_model_len to %s", max_model_len)

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate NPU KV cache with the specified kv_cache_config."""
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CaMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext

            context = nullcontext()  # type: ignore
        with context:
            self.model_runner.initialize_kv_cache(kv_cache_config)

            # Restrict to mamba and full attn hybrid models (e.g. Qwen3.x).
            #
            # When eagle3 is enabled with num_speculative_tokens>1, mamba blocks may be reallocated to full blocks if
            # the target and draft models share the same kv cache tensor (e.g. unaligned full attn layers with
            # different num_kv_heads and head_size). In addition, for performance reasons, the current mtp/eagle path
            # does not update seq_lens_cpu with num_rejected_tokens for step>1, since it would require d2h sync. As a
            # result, seq_lens_cpu can become stale and some blocks will be unintentionally used.
            #
            # If an uncleared mamba block is later reused, the stale state combined with the incorrect seq_lens_cpu may
            # lead to NaNs and reduced acceptance rate.
            if (
                kv_cache_config.needs_kv_cache_zeroing
                and hasattr(self.model_runner, "_init_kv_zero_meta")
                and self.vllm_config is not None
                and self.vllm_config.speculative_config is not None
                and self.vllm_config.speculative_config.method == "eagle3"
                and self.vllm_config.speculative_config.num_speculative_tokens > 1
            ):
                self.model_runner._init_kv_zero_meta()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # Check if profiling is enabled (RFC #6954 - align with upstream vLLM)
        if self.profiler_config is None or self.profiler_config.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Please set --profiler-config to enable "
                "profiling. Example: "
                "'--profiler-config.profiler=torch --profiler-config.torch_profiler_dir"
                "=YOUR_DIR_PATH_TO_DUMP_TRACE'"
            )

        if is_start:
            from vllm.distributed.utils import get_worker_rank_suffix

            rank_suffix = get_worker_rank_suffix(global_rank=self.rank)
            trace_name = f"{profile_prefix}_{rank_suffix}" if profile_prefix else rank_suffix

            if self.profiler is None:
                self.profiler = TorchNPUProfilerWrapper(self.profiler_config, trace_name)
                logger.debug("Starting torch profiler with trace name: %s", trace_name)
                self.profiler.start()  # type: ignore[attr-defined]
            else:
                # Profiler already initialized. Restart profiling but keep
                # the original trace name from the first initialization.
                self.profiler.start()
        else:
            if self.profiler is None:
                logger.warning("Profiler was not started, nothing to stop.")
                return
            self.profiler.stop()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def reset_encoder_cache(self) -> None:
        self.model_runner.reset_encoder_cache()

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(num_tokens=self.model_runner.decode_token_per_req, uniform_decode=True)

    def _init_worker_distributed_environment(self) -> None:
        """Initialize the distributed environment."""
        init_batch_invariance()
        # NOTE: `self.local_rank` is also consumed by `bind_cpus` for CPU
        # binding, so it must stay as the original TP local rank. Compute the
        # adjusted local rank locally and pass it to `init_distributed_environment`.
        local_rank = self.local_rank
        parallel_config = self.parallel_config
        if (
            parallel_config.distributed_executor_backend
            not in ("ray", "external_launcher")
            and parallel_config.data_parallel_backend != "ray"
            and parallel_config.data_parallel_size > 1
        ):
            # Use local DP rank if available, otherwise use global DP rank.
            dp_local_rank = parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = parallel_config.data_parallel_index

            # In edge-cloud mode, local_world_size = edge_npu_count or cloud_npu_count
            # Use local_world_size as the stride per DP instance
            local_world_size = parallel_config.local_world_size
            # DP_LOCAL_RANK * LOCAL_WORLD_SIZE + TP_LOCAL_RANK
            local_rank += dp_local_rank * local_world_size
        init_distributed_environment(
            self.parallel_config.world_size, self.rank, self.distributed_init_method, local_rank, "hccl"
        )
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
            self.parallel_config.prefill_context_parallel_size,
            self.parallel_config.decode_context_parallel_size,
        )
        init_ascend_model_parallel(self.parallel_config)
        ensure_ec_transfer_initialized(self.vllm_config)

    def get_supported_pooling_tasks(self):
        return self.model_runner.get_supported_pooling_tasks()

    def get_supported_tasks(self) -> "tuple[SupportedTask, ...]":
        return self.model_runner.get_supported_tasks()

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()

    def clear_pending_edge_cloud_draft_for_req_ids(
        self,
        req_ids: set[str] | list[str],
        force_drop_task_ids: set[str] | list[str] = (),
    ) -> None:
        self.model_runner.clear_pending_edge_cloud_draft_for_req_ids(
            req_ids, force_drop_task_ids
        )

    def check_health(self) -> None:
        import subprocess

        logger.debug("check_health starting for rank %s...", self.local_rank)
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-i", str(self.local_rank), "-t", "health"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                parse_text_output(result.stdout)
                logger.debug("check_health success for rank %s.", self.local_rank)
            else:
                logger.warning("query NPU card %s fail: %s", self.local_rank, result.stderr)
        except subprocess.TimeoutExpired:
            logger.warning("query NPU card %s timeout.", self.local_rank)
        except FileNotFoundError:
            logger.warning("npu-smi tool not found.")
        except Exception as e:
            logger.error("query NPU card %s fail: %s", self.local_rank, e)
        return


def parse_text_output(output) -> None:
    lines = output.strip().split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if "Health" in line:
            if line.split(":")[-1].strip() != "OK":
                raise RuntimeError("NPU card health status is not OK")
    return
