#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
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
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#

import math
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from functools import partial
from multiprocessing import Manager
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.cuda_graph import CUDAGraphStat
import vllm.compilation.monitor as _monitor
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_dp_group,
    get_edge_cloud_layer_range,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    is_edge_cloud_pp_mode,
    is_edge_device,
    set_edge_cloud_layer_range,
)
from vllm.forward_context import BatchDescriptor, ForwardContext, get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionLayer
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import LazyLoader
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.attention.selector import get_attn_backend  # type: ignore
from vllm.v1.core.sched.output import BatchType, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ECConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.worker.utils import select_common_block_size

from vllm_ascend.utils import vllm_version_is

if not vllm_version_is("0.20.2"):
    from vllm.v1.outputs import RoutedExpertsLists
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer_gpu import copy_num_valid_draft_tokens
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.cp_utils import (
    get_total_cp_world_size,
)
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput, GPUModelRunner
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    maybe_create_ubatch_slices,
)
from vllm.v1.worker.utils import AttentionGroup

# yapf: enable
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionState
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, using_paged_attention

# yapf conflicts with isort for this block
# yapf: disable
from vllm_ascend.compilation.acl_graph import (
    ACLGraphWrapper,
    set_draft_graph_params,
    set_graph_params,
    update_full_graph_params,
)
from vllm_ascend.compilation.acl_graph_edge_cloud import (
    EdgeCloudACLGraphWrapper,
    make_graph_params,
)
from vllm_ascend.compilation.edge_cloud_compiler import (
    EdgeCloudCompiledSegment,
)
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_device_transfer_loader import D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_worker import EplbProcess
from vllm_ascend.eplb.eplb_updator import EplbUpdator
from vllm_ascend.eplb.utils import model_register
from vllm_ascend.ops.rotary_embedding import set_cos_and_sin, update_cos_sin
from vllm_ascend.patch.worker.patch_draft_quarot import patch_load_weights
from vllm_ascend.quantization.utils import enable_fa_quant
from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.spec_decode.extract_hidden_states_proposer import (
    AscendExtractHiddenStatesProposer,
)
from vllm_ascend.spec_decode.medusa_proposer import AscendMedusaProposer
from vllm_ascend.spec_decode.ngram_proposer import AscendNgramProposer
from vllm_ascend.spec_decode.ngram_proposer_npu import AscendNgramProposerNPU
from vllm_ascend.spec_decode.suffix_proposer import AscendSuffixDecodingProposer
from vllm_ascend.spec_decode.utils import update_num_computed_tokens_for_batch_change
from vllm_ascend.utils import (
    calc_split_factor,
    check_gdn_layer,
    enable_sp,
    enable_sp_by_pass,
    get_c_env,
    get_compressed_pos_and_indices,
    global_stream,
    kv_cache_spec_uses_sparse_c8,
    lmhead_tp_enable,
    set_weight_prefetch_method,
    should_skip_allreduce_across_dp_group,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
from vllm_ascend.worker.pcp_utils import PCPManager



from vllm_ascend.ascend_forward_context import (  # isort: skip
    MoECommType,
    get_mc2_tokens_capacity,
    select_moe_comm_method,
    set_ascend_forward_context,
    set_mc2_mask,
    set_mc2_tokens_capacity,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer

from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

from vllm.model_executor.layers.attention import Attention, MLAAttention

# if true, allow tensor initialization and casting with internal format (e.g., NZ)
torch.npu.config.allow_internal_format = True

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict

SEQ_LEN_WITH_MAX_PA_WORKSPACE = 6144

@dataclass
class GraphCaptureContext:
    stream: torch.npu.Stream

@contextmanager
def graph_capture(device: torch.device):
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the NPU graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current NPU stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    graph_capture_context = GraphCaptureContext(torch.npu.Stream(device=device))
    stream = graph_capture_context.stream

    # we use nullcontext now
    maybe_ca_context = nullcontext()

    # ensure all initialization operations complete before attempting to
    # capture the graph on another stream
    curr_stream = torch.npu.current_stream()
    if curr_stream != stream:
        stream.wait_stream(curr_stream)

    with torch.npu.stream(stream), maybe_ca_context:
        yield graph_capture_context

def get_tp_context(drafter):
    return getattr(drafter, "tp_group_context", nullcontext())

class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: AscendCommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    attn_metadata: "PerLayerAttnMetadata"
    positions: torch.Tensor
    ec_connector_output: "ECConnectorOutput | None"
    cudagraph_stats: CUDAGraphStat | None
    batch_desc: BatchDescriptor

class EdgeCloudSegment(torch.nn.Module):
    """执行指定层区间 [start_layer, end_layer) 的轻量 nn.Module。

    将基础模型的 ``forward_edge_cloud_segment``（模型加载阶段通过
    monkey-patch 注入）包装为标准 nn.Module，使 ACLGraphWrapper 可以像
    标准流程包裹完整模型一样包裹 segment：:

        ACLGraphWrapper(EdgeCloudSegment(model, N-1, N), ...)

    使用 nn.Module 而非函数闭包，确保 ``torch.npu.graph`` 的参数追踪、
    缓冲区管理、模块分发与标准（非边云）流程完全一致。

    注意：model 作为 nn.Module 属性注册为子模块，torch.npu.graph 捕获时
    通过模块层级静态识别参数张量，确保图回放时参数正确处理。
    模型在传入前已完成 PPMissingLayer 裁剪，子模块注册不会引入额外显存。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        start_layer: int,
        end_layer: int,
        is_first_segment: bool | None = None,
        is_last_segment: bool | None = None,
    ):
        super().__init__()
        self._edge_model = model
        self._start_layer = start_layer
        self._end_layer = end_layer
        self._is_first_segment = is_first_segment
        self._is_last_segment = is_last_segment

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **extra_layer_kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        # Layer-sliced execution (non-ACLGraph mode): allow dynamic override
        # of the layer range via kwargs from PassiveScheduler slice dispatch.
        start_layer = extra_layer_kwargs.pop(
            "layer_slice_start", self._start_layer
        )
        end_layer = extra_layer_kwargs.pop(
            "layer_slice_end", self._end_layer
        )
        return self._edge_model.forward_edge_cloud_segment(
            start_layer,
            end_layer,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            is_first_segment=self._is_first_segment,
            is_last_segment=self._is_last_segment,
            **extra_layer_kwargs,
        )

@dataclass
class HeadState:
    """Minimal suspended state for an edge-cloud head-segment batch.

    The heavy intermediate tensors (hidden states) are sent to the cloud
    immediately; we only keep enough metadata to correlate the tail-segment
    batch with its head segment via ``head_token``.
    """
    head_token: str
    scheduler_output: "SchedulerOutput"
    req_ids: tuple[str, ...]


def _clone_gdn_attn_metadata(meta):
    """Deep-clone device tensors inside GDNAttentionMetadata.

    GDN metadata holds device tensors that are views into shared
    ``common_attn_metadata`` buffers (e.g. ``query_start_loc``,
    ``state_indices``).  These buffers are rebuilt on every batch, so a
    decode batch interleaved between two prefill slices would silently
    overwrite the data still referenced by the saved
    ``_layerwise_attn_metadata``.  Cloning at **save time** preserves
    the correct prefill-length values.
    """
    import copy
    from dataclasses import fields

    # Use dataclass replacement to create a shallow copy first,
    # then deep-clone the device tensor fields.
    cloned = copy.copy(meta)

    # Device tensor fields that must be cloned to decouple from
    # shared common_attn_metadata buffers.
    _DEVICE_TENSOR_FIELDS = (
        "has_initial_state",
        "spec_query_start_loc",
        "non_spec_query_start_loc",
        "spec_state_indices_tensor",
        "non_spec_state_indices_tensor",
        "spec_sequence_masks",
        "spec_token_indx",
        "non_spec_token_indx",
        "num_accepted_tokens",
        "chunk_indices",
        "chunk_offsets",
    )
    for field_name in _DEVICE_TENSOR_FIELDS:
        tensor = getattr(cloned, field_name, None)
        if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.device.type != "cpu":
            setattr(cloned, field_name, tensor.clone())

    # The non_spec_prefill_fallback_meta contains pooled device tensors
    # for causal_conv1d host args and chunked prefill metadata.
    fallback_meta = getattr(cloned, "non_spec_prefill_fallback_meta", None)
    if fallback_meta is not None:
        cloned_fallback = copy.copy(fallback_meta)

        # Clone causal_conv1d host metadata (CPU pinned tensors)
        causal_conv1d = getattr(cloned_fallback, "causal_conv1d", None)
        if causal_conv1d is not None:
            cloned_causal = copy.copy(causal_conv1d)
            for attr in ("query_start_loc_cpu", "cache_indices_cpu", "has_initial_state_cpu"):
                t = getattr(cloned_causal, attr, None)
                if t is not None and isinstance(t, torch.Tensor):
                    setattr(cloned_causal, attr, t.clone())
            cloned_fallback.causal_conv1d = cloned_causal

        # Clone chunked prefill metadata (device tensors from 2-slot pool)
        chunk_meta = getattr(cloned_fallback, "chunk", None)
        if chunk_meta is not None:
            cloned_chunk = copy.copy(chunk_meta)
            for attr in (
                "chunk_indices_chunk64",
                "chunk_offsets_chunk64",
                "update_chunk_offsets_chunk64",
                "final_chunk_indices_chunk64",
                "chunk_indices_large_block",
                "block_indices_cumsum",
            ):
                t = getattr(cloned_chunk, attr, None)
                if t is not None and isinstance(t, torch.Tensor) and t.device.type != "cpu":
                    setattr(cloned_chunk, attr, t.clone())
            # Decouple from pool so the pool slot can be reused safely
            cloned_chunk._buffer_slot = None
            cloned_fallback.chunk = cloned_chunk

        cloned.non_spec_prefill_fallback_meta = cloned_fallback

    return cloned


class NPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # TODO(qcs): These manual pad and unpad for GPUModelRunner are
        # used to expand some buffers, which need to be reverted after
        # the following PR is merged:
        # https://github.com/vllm-project/vllm/pull/28988
        max_pcp_pad_tokens = (
            vllm_config.parallel_config.prefill_context_parallel_size * 2 * vllm_config.scheduler_config.max_num_seqs
        )
        vllm_config.scheduler_config.max_num_batched_tokens += max_pcp_pad_tokens

        # Must be set before super().__init__() because parent init may call
        # _allocate_kv_cache_tensors which accesses self.use_compress.
        model_config = getattr(vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None) if model_config else None
        self.use_compress = (
            hf_config is not None and hasattr(hf_config, "compress_ratios")
        )

        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
        # See _pad_query_start_loc_for_fia.
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 2,  # type: ignore[has-type]
            dtype=torch.int32,
        )

        # Now, query_start_loc is padded.
        # But gdn needs an unpadded one.
        # gdn_query_start_loc is an unpadded version of query_start_loc.
        # TODO delete it if fia's check is removed.
        self._has_gdn = check_gdn_layer(vllm_config)
        self._has_sinks = False
        if self._has_gdn:
            self.gdn_query_start_loc = self._make_buffer(
                self.max_num_reqs + 1,  # type: ignore[has-type]
                dtype=torch.int32,
            )

        vllm_config.scheduler_config.max_num_batched_tokens -= max_pcp_pad_tokens
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        # PD-separation: batch_type_id for cross-DP sync (0=DUMMY, 1=PF,
        # 2=PL, 3=DF, 4=DL). Packed into sync_metadata all_reduce so the
        # idle DP's DUMMY can match the real DP's segment (head or tail).
        self._dp_batch_type_id = 0
        self._peer_batch_type_id = 0

        self.sampler = AscendSampler()
        self.attn_state: AscendAttentionState | None = None

        # Layerwise chunking: state saved by chunk 0 and consumed by
        # continuation chunks (1..N-1).
        self._layerwise_intermediate: IntermediateTensors | None = None
        self._layerwise_positions: torch.Tensor | None = None
        self._layerwise_attn_metadata: Any = None
        self._layerwise_num_tokens_padded: int = 0
        self._layerwise_num_tokens_across_dp: int | None = None
        self._layerwise_batch_desc: Any = None
        self._layerwise_scheduler_output: Any = None
        # Additional state needed by the last chunk on the last PP rank
        # for logits computation and execute_model_state setup.
        self._layerwise_logits_indices: torch.Tensor | None = None
        self._layerwise_spec_decode_metadata: Any = None
        self._layerwise_spec_decode_common_attn_metadata: Any = None
        self._layerwise_ec_connector_output: Any = None
        self._layerwise_cudagraph_stats: Any = None

        # Edge-cloud PD-separation: suspended head-segment states keyed by
        # head_token.  Each entry holds the minimal context needed to verify
        # that a later tail-segment batch matches its head segment.
        self._pending_head_states: dict[str, "HeadState"] = {}

        # Ascend-specific configurations
        self.ascend_config = get_ascend_config()
        set_weight_prefetch_method(self.ascend_config.weight_prefetch_config)

        # Edge-cloud initialization must happen before _set_up_drafter
        # because the drafter setup needs to know whether edge-cloud is enabled.
        self.edge_cloud_cfg = self.ascend_config.edge_cloud_config
        self._edge_cloud_enabled = self.edge_cloud_cfg.enabled
        # This flag is set per-step in execute_model; initialize it here so
        # that code paths reaching _prepare_inputs before the first execute_model
        # call (e.g. profile_run or unit tests) do not hit AttributeError.
        self._is_edge_cloud_embed_only_tail = False
        if self._edge_cloud_enabled:
            if not self.parallel_config.enable_edge_cloud:
                raise ValueError(
                    "additional_config.edge_cloud_config.enabled requires "
                    "--enable-edge-cloud."
                )
            expected_role = "edge" if self.parallel_config.is_edge_node else "cloud"
            if self.edge_cloud_cfg.role != expected_role:
                raise ValueError(
                    "additional_config.edge_cloud_config.role must match the "
                    f"process role inferred from --headless. Expected "
                    f"{expected_role!r}, got {self.edge_cloud_cfg.role!r}."
                )
            self.head_k, self.tail_k = self.edge_cloud_cfg.head_tail_k
            if self.edge_cloud_cfg.mode == "embedding_only":
                self.head_k = 0
                self.tail_k = 0
                logger.info(
                    "Edge-cloud mode is 'embedding_only', forcing head_k=0, tail_k=0"
                )
            hf_config = getattr(self.model_config, "hf_text_config", None)
            model_type = getattr(hf_config, "model_type", "")
            outer_model_type = getattr(
                getattr(self.model_config, "hf_config", None), "model_type", ""
            )
            self._is_deepseek_v4 = (
                    model_type == "deepseek_v4" or hasattr(hf_config, "hc_mult")
                )
            self._is_qwen3_5 = "qwen3_5" in model_type
            self._is_deepseek_v2 = "deepseek" in model_type
            self._is_kimi_k25 = "kimi_k25" in outer_model_type or "kimi_k25" in model_type
            self._is_glm4_moe = "glm4_moe" in model_type or "glm_moe_dsa" in model_type
            self._is_minimax_m2 = "minimax_m2" in model_type
            self.num_layers = 0
            self.segment_a: Any = None
            self.segment_e: Any = None
            self.segment_c: Any = None
            self.segment_a_wrapper: Any = None
            self.segment_e_wrapper: Any = None
            self.segment_c_wrapper: Any = None
            # Cache segment_a prepare results for segment_e reuse (edge-cloud only).
            # Keyed by head_token (not single-slot) so 2P1D - two prefills in
            # flight (P1首/P2首) with possibly out-of-order P尾 returns - matches
            # each tail to its own head's cached attn_metadata/batch_desc, not
            # whichever prefill wrote a single-slot cache last.
            self._edge_prepare_cache: dict[str, dict] = {}
            # Cache cloud-side prepare results to overlap with edge segment_a
            self._cloud_prepare_cache: dict | None = None
        else:
            self.head_k = 0
            self.tail_k = 0
            self._is_deepseek_v4  = False
            self._is_qwen3_5 = False
            self._is_deepseek_v2 = False
            self._is_kimi_k25 = False
            self._is_glm4_moe = False
            self._is_minimax_m2 = False
            if self.parallel_config.enable_edge_cloud:
                raise ValueError(
                    "--enable-edge-cloud requires "
                    "additional_config.edge_cloud_config.enabled=true."
                )

        # Dump / PrecisionDebugger configuration now comes from AscendConfig
        dump_cfg = self.ascend_config.dump_config_path
        self.debugger = None
        if dump_cfg is not None:
            self._debugger_started = False
            if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
                from msprobe.pytorch import PrecisionDebugger

                self.debugger = PrecisionDebugger(dump_cfg)
            else:
                try:
                    from msprobe.pytorch import AclGraphDumper
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to import AclGraphDumper from msprobe. "
                        "Please install/rebuild msprobe with aclgraph_dump enabled."
                    ) from exc

                self.debugger = AclGraphDumper(dump_cfg)
        # use_hybrid_blocks: if hybrid blocks is used.
        self.use_hybrid_blocks: bool = False
        self.need_accepted_tokens: bool = False

        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.block_size = vllm_config.cache_config.block_size
        # Set up Attention
        self.use_sparse = hasattr(vllm_config.model_config, "hf_text_config") and hasattr(
            vllm_config.model_config.hf_text_config, "index_topk"
        ) and not hasattr(
            vllm_config.model_config.hf_text_config, "compress_ratios"
        )
        if self.use_sparse:
            self.sparse_head_dim = (
                self.model_config.hf_text_config.kv_lora_rank,
                self.model_config.hf_text_config.qk_rope_head_dim,
                self.model_config.hf_text_config.index_head_dim,
            )
        # dsa c8
        self.use_sparse_c8_indexer = self.ascend_config.enable_sparse_c8
        if self.use_sparse_c8_indexer:
            self.c8_k_cache_dtype = torch.int8
            self.c8_k_scale_cache_dtype = torch.float16

        self.attn_backend = get_attn_backend(
            0,
            self.dtype,
            None,
            use_mla=self.model_config.use_mla,
            use_sparse=self.use_sparse,
            use_mm_prefix=self.model_config is not None
            and self.model_config.is_mm_prefix_lm,
        )

        try:
            self.dcp_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
            self.pcp_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        except Exception:
            self.dcp_size = 1
            self.dcp_rank = 0
            self.pcp_size = 1
            self.pcp_rank = 0
        if self.pcp_size > 1:
            self.model_config.max_model_len += 2 * self.pcp_size * self.max_num_reqs
        max_buffer_num_tokens = self.max_num_tokens
        if self.pcp_size * self.dcp_size > 1:
            max_buffer_num_tokens = self.max_num_tokens + self.max_num_reqs * 2 * self.pcp_size
            self.pcp_manager = PCPManager(
                self.pcp_size,
                self.pcp_rank,
                self.dcp_size,
                self.dcp_rank,
                max_buffer_num_tokens,
                self.max_num_reqs,
                self.device,
                self.vllm_config,
                self.use_async_scheduling,
                self.pin_memory,
                self.use_sparse,
            )
            # TODO(zhenwenqi) after https://github.com/vllm-project/vllm/pull/28988 is merged, we can delete this
            self.input_ids = self._make_buffer(max_buffer_num_tokens, dtype=torch.int32)
            self.positions = torch.zeros(
                max_buffer_num_tokens, dtype=torch.int64, device=self.device)
            
        # Create a CPU numpy buffer for positions computation when
        # self.positions is a plain tensor (non-CpuGpuBuffer case).
        self._positions_cpu_buf = torch.zeros(
            max_buffer_num_tokens, dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        self._positions_np_buf = self._positions_cpu_buf.numpy()
        # For deepseek-v4 use only
        self._dsa_positions_cpu_buf = torch.zeros(
            max_buffer_num_tokens, dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        self._dsa_positions_np_buf = self._dsa_positions_cpu_buf.numpy()

        self.use_eagle = (
            vllm_config.speculative_config.use_eagle()
            if vllm_config.speculative_config
            else None
        )
        # When True, run update_full_graph_params before self.model (ENPU / graph capture order).
        # Internal / non-public toggle: read C getenv ``ENPU_ENABLE`` from enpu code (not in envs.py).
        _enpu = get_c_env("ENPU_ENABLE")
        self.enable_enpu = _enpu is not None and _enpu.lower() == "true"

        self._set_up_drafter()

        # Event for async GPU→CPU copy of corrected seq_lens in async
        # spec decode mode. Recorded in _prepare_inputs, synchronized
        # in _build_attention_metadata. Created once, reused each iteration.
        # Only backends that consume CPU seq_lens (AscendAttentionBackend,
        # AscendMLABackend, and DSV4 compressed attention metadata) need this;
        # others (SFA, GDN, etc.) do not.
        self._needs_seq_lens_cpu_sync = self.use_compress or issubclass(
            self.attn_backend, (AscendAttentionBackend, AscendMLABackend)
        )
        self._seq_lens_cpu_event: torch.npu.Event | None = None
        self._seq_lens_cpu_event_pending = False

        # kv role
        self.is_kv_producer = False
        self.is_kv_consumer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
            self.is_kv_consumer = vllm_config.kv_transfer_config.is_kv_consumer

        set_cos_and_sin(vllm_config, self.max_num_reqs, self.uniform_decode_query_len, self.dtype, self.device)
        set_mc2_tokens_capacity(vllm_config, self.max_num_reqs, self.uniform_decode_query_len)
        set_mc2_mask(vllm_config, self.device)
        self.decode_threshold = 1 + (self.speculative_config.num_speculative_tokens if self.speculative_config else 0)

        self.use_aclgraph = self._use_aclgraph()

        eplb_config = self.ascend_config.eplb_config
        self.dynamic_eplb = eplb_config.dynamic_eplb
        self.eplb_enable = self.dynamic_eplb or (eplb_config.expert_map_path is not None)
        if self.dynamic_eplb:
            self.is_eplb_warmuped = False
            self.policy_type = eplb_config.eplb_policy_type
            self.eplb_loader = D2DExpertWeightLoader()
            self.manager = Manager()
            self.shared_dict = self.manager.dict({"expert_map": None, "moe_load": None, "expert_maps": None})
            self.eplb_process = EplbProcess(shared_dict=self.shared_dict, policy_type=self.policy_type, enable_d2d=True)
            self.process = self.eplb_process._launch_process()
            self.eplb_updator = EplbUpdator(eplb_config, self.eplb_loader, self.eplb_process, self.process)
        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        self.input_batch = NPUInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=max(self.model_config.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            kernel_block_sizes=[[self.cache_config.block_size]],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                self.pin_memory,
                self.is_pooling_model,
                self.vllm_config.model_config.logits_processors,
            ),
            logitsprocs_need_output_token_ids=bool(
                self.vllm_config.model_config.logits_processors
            ),
            is_pooling_model=self.is_pooling_model,
            num_speculative_tokens=(
                self.vllm_config.speculative_config.num_speculative_tokens if self.vllm_config.speculative_config else 0
            ),
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
        )
        self.num_draft_tokens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        # here we use int32
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        # for cleancode , actually the three attrs is defined in gpu_model_runner
        self.execute_model_state: ExecuteModelState | None = None
        # [PD-FIX] Set by execute_model when a stale tail segment (PL/DL) is
        # discarded (all reqs already popped from self.requests). sample_tokens
        # checks this to return EMPTY instead of None (which would trigger
        # "unexpected error" in _patched_step_with_batch_queue).
        self._tail_segment_discarded: bool = False
        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None
        self.reorder_batch_threshold: int | None = None
        self.long_seq_metadata = None
        self.query_lens: torch.Tensor | None = None
        self.cpu_slot_mapping = None
        self.sampling_done_event: torch.npu.Event | None = None

        # self.cudagraph_batch_sizes sorts in ascending order.
        if (
            self.compilation_config.cudagraph_capture_sizes
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            self.cudagraph_batch_sizes = sorted(self.compilation_config.cudagraph_capture_sizes)
        else:
            self.cudagraph_batch_sizes = []
        self.mamba_state_idx: dict[str, int] = {}
        self._mamba_bufs: Any | None = None
        self._mamba_copy_bufs: Any | None = None

        # Saved in execute_model() so sample_tokens() can access scheduler_output
        # for edge-cloud mamba state sync (especially on the cloud side).
        self._last_scheduler_output: "SchedulerOutput | None" = None

        # Saved on the cloud side during execute_model() for use by
        # _run_mtp_cloud_segment() which runs later in sample_tokens().
        self._cloud_spec_decode_common_attn_metadata: AscendCommonAttentionMetadata | None = None
        self._cloud_spec_decode_num_reqs: int = 0
        self.enable_hamming_sparse = (self.ascend_config.enable_hamming_sparse is True)
        self.enable_hamming_sparse = self.enable_hamming_sparse and not vllm_config.speculative_config
        if self.enable_hamming_sparse is True:
            from vllm_ascend.worker.kvcomp_utils import initialize_kvcomp_metadata
            self.kvcomp_meta_data = initialize_kvcomp_metadata(max_num_reqs=self.max_num_reqs,
                block_size=self.block_size, device=self.device, vllm_config=self.vllm_config,
                parallel_config=self.parallel_config, dtype=self.dtype)

    @property
    def use_cp(self) -> bool:
        return self.pcp_size * self.dcp_size > 1

    def _init_device_properties(self) -> None:
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.npu.synchronize()

    def _set_up_drafter(self):
        # Set up speculative decoding.
        self.drafter: (
            AscendNgramProposer
            | AscendNgramProposerNPU
            | AscendEagleProposer
            | AscendDraftModelProposer
            | AscendDflashProposer
            | AscendSuffixDecodingProposer
            | AscendMedusaProposer
            | AscendExtractHiddenStatesProposer
            | None
        ) = None
        self.actual_seq_lengths_q: list[int] = []
        self.decode_token_per_req = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            assert spec_token_num > 0
            self.decode_token_per_req = 1 + spec_token_num
            if get_pp_group().is_last_rank or (
                self._edge_cloud_enabled
                and self.speculative_config.method == "mtp"
            ):
                self.drafter = self._get_drafter()
                if self.speculative_config.method == "eagle3":
                    assert isinstance(self.drafter, AscendEagleProposer)
                    self.use_aux_hidden_state_outputs = self.drafter.eagle3_use_aux_hidden_state
                elif self.speculative_config.method == "extract_hidden_states":
                    assert isinstance(self.drafter, AscendExtractHiddenStatesProposer)
                    self.use_aux_hidden_state_outputs = True
                self.rejection_sampler = AscendRejectionSampler(self.sampler)
        self.discard_request_indices = self._make_buffer(self.max_num_reqs, dtype=torch.int64)
        self.num_discarded_requests = 0

    def _get_drafter(self):
        return get_spec_decode_method(self.speculative_config.method, self.vllm_config, self.device, self)

    def _use_aclgraph(self) -> bool:
        return (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and self.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not self.model_config.enforce_eager
        )

    def _is_dummy_or_profile_run(self) -> bool:
        try:
            forward_context = get_forward_context()
        except AssertionError:
            return False
        return bool(getattr(forward_context, "in_profile_run", False))

    def _create_raw_segment_callable(
        self,
        model: torch.nn.Module,
        start_layer: int,
        end_layer: int,
        is_first_segment: bool | None = None,
        is_last_segment: bool | None = None,
    ) -> EdgeCloudSegment:
        return EdgeCloudSegment(
            model, start_layer, end_layer, is_first_segment, is_last_segment
        )

    def _maybe_compile_segment_callable(
        self,
        segment: Any,
        start_layer: int,
        end_layer: int,
    ) -> Any:
        # 若全局 enable_npugraph_ex 开启且当前处于全图模式，
        # 对 segment 应用 npugraph_ex 编译时优化（第1层）。
        # 第2层（ACLGraphWrapper 运行时捕获）由 _wrap_segment_if_needed 负责。
        ascend_compilation_config = get_ascend_config().ascend_compilation_config
        if (
            ascend_compilation_config.enable_npugraph_ex
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            logger.info(
                "EdgeCloudCompiledSegment wrapping segment [%d, %d) "
                "with npugraph_ex compile-time optimization.",
                start_layer,
                end_layer,
            )
            return EdgeCloudCompiledSegment(
                segment,
                self.vllm_config,
                ascend_compilation_config,
            )

        return segment

    def _create_segment_callable(
        self,
        model: torch.nn.Module,
        start_layer: int,
        end_layer: int,
        is_first_segment: bool | None = None,
        is_last_segment: bool | None = None,
    ) -> Any:
        """创建一个仅执行指定层区间 [start_layer, end_layer) 的 nn.Module。

        返回 EdgeCloudSegment（nn.Module 子类）而非函数闭包，
        确保 ACLGraphWrapper 包裹的是标准 nn.Module，与标准流程的
        图捕获方式对齐（torch.npu.graph 捕获 nn.Module.forward()）。

        当 enable_npugraph_ex 开启且 cudagraph_mode 支持全图编译时，
        额外包裹 EdgeCloudCompiledSegment，使 segment 先经过
        torch.compile → npugraph_ex_compile 编译时优化，
        再由 ACLGraphWrapper 进行运行时图捕获，对齐标准流程的两层图优化。

        边云场景下所有模型均已在加载阶段通过对应 patch 文件注入
        forward_edge_cloud_segment，因此直接委托即可，无需额外 fallback。
        """
        segment = self._create_raw_segment_callable(
            model, start_layer, end_layer, is_first_segment, is_last_segment
        )
        return self._maybe_compile_segment_callable(segment, start_layer, end_layer)

    def _wrap_segment_if_needed(
        self,
        segment: Any,
        runtime_mode: CUDAGraphMode = CUDAGraphMode.FULL,
    ) -> Any:
        if not self.edge_cloud_cfg.enable_decode_graph:
            return segment
        if not self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            return segment
        if self._is_dummy_or_profile_run():
            return segment
        return EdgeCloudACLGraphWrapper(
            segment,
            self.vllm_config,
            runtime_mode=runtime_mode,
            cudagraph_options=None,
        )

    def _load_model_edge_cloud(self) -> None:
        """边云场景的模型加载流程（复用 vLLM 标准 PP 初始化，直接加载到 NPU）。

        核心思路：
          通过 set_edge_cloud_layer_range() 存储 head_k/tail_k 到全局变量，
          在模型 __init__ 阶段 make_layers() → get_edge_cloud_layer_range()
          读取后直接按边云非连续层范围创建 PPMissingLayer 占位，
          使非本地层在初始化时就是占位层，权重加载阶段自动跳过。

        流程：
          1. set_edge_cloud_layer_range(head_k, tail_k) 存储层范围
          2. get_model → BaseModelLoader.load_model（标准 NPU 上初始化+加载）
          3. 创建分段 callable 并按需包装 ACLGraphWrapper
        """
        if not (self._is_qwen3_5 or self._is_deepseek_v2 or self._is_kimi_k25 
                or self._is_glm4_moe or self._is_minimax_m2):
            raise NotImplementedError(
                "edge-cloud mode currently supports Qwen3.5, DeepseekV2/V3, "
                "Kimi-K2.5/K2.6, GLM-4/GLM-5 models, and MiniMax-M2 models."
            )

        logger.info(
            "Starting to load model in edge-cloud mode: role=%s, mode=%s, head_k=%d, tail_k=%d",
            self.edge_cloud_cfg.role,
            self.edge_cloud_cfg.mode,
            self.head_k,
            self.tail_k,
        )
        if self._is_deepseek_v4:
            import vllm_ascend.patch.models.deepseek_v4_edge_cloud
        if self._is_qwen3_5:
            import vllm_ascend.patch.models.qwen3_5_edge_cloud  # noqa: F401
        if self._is_deepseek_v2:
            import vllm_ascend.patch.models.deepseek_v2_edge_cloud  # noqa: F401
        if self._is_kimi_k25:
            import vllm_ascend.patch.models.kimi_k25_edge_cloud  # noqa: F401
        if self._is_glm4_moe:
            hf_text_config = getattr(self.model_config, "hf_text_config", None)
            text_model_type = getattr(hf_text_config, "model_type", "")
            if "glm_moe_dsa" in text_model_type:
                import vllm_ascend.patch.models.deepseek_v2_edge_cloud  # noqa: F401
        if self._is_minimax_m2:
            import vllm_ascend.patch.models.minimax_m2_edge_cloud  # noqa: F401

        # 1. 存储 head_k / tail_k 到 parallel_state 全局变量，
        #    使 make_layers() 在模型 __init__ 中能直接读取并创建正确的
        #    PPMissingLayer 占位（非本地层）和真实层（本地层）。
        set_edge_cloud_layer_range(self.head_k, self.tail_k)

        # 2. 复用标准 vLLM 加载流程：init on device + load_weights on device
        #    BaseModelLoader.load_model 内部：
        #      with target_device: initialize_model()  → 模型创建在 NPU
        #      load_weights()                          → 权重直接到 NPU（跳过占位层）
        #      process_weights_after_loading()        → 量化/格式调整在 NPU
        #      return model.eval()
        #
        #    make_layers() → _get_edge_cloud_local_indices()
        #    → get_edge_cloud_layer_range() 读取 head_k/tail_k
        #    → 根据 is_edge_device() 计算本侧本地层索引
        #    → 非本地层初始化为 PPMissingLayer（无参数，不占显存）。
        self.model = get_model(vllm_config=self.vllm_config)

        # Locate the transformer layers — the model may be wrapped in a
        # multimodal ConditionalGeneration (language_model.model.layers)
        # or be a plain CausalLM (model.layers).
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            transformer_layers = self.model.model.layers
        else:
            transformer_layers = self.model.language_model.model.layers
        self.num_layers = len(transformer_layers)

        # set_moe_parameters() 在 __init__ 中只遍历到本地层，因此不存在
        # 非本地层 stale 引用问题。但为确保 moe_layers/moe_mlp_layers
        # 引用正确，重新收集一次（与标准 PP 行为对齐）。
        if hasattr(self.model, 'set_moe_parameters'):
            self.model.set_moe_parameters()

        # 打印每层最终状态（诊断用）
        layer_states = []
        for idx, layer in enumerate(transformer_layers):
            layer_states.append(
                f"{idx}:{'REAL' if not isinstance(layer, PPMissingLayer) else 'SKIP'}"
            )
        logger.info("[EdgeCloud] Final layer states: %s", ", ".join(layer_states))

        # 3. 创建分段 callable 并按需包装 ACLGraphWrapper
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.update_stream = torch.npu.Stream()

        if self.edge_cloud_cfg.role == "edge":
            self.segment_a = self._create_segment_callable(
                self.model, 0, self.head_k, is_first_segment=True, is_last_segment=False
            )
            self.segment_e = self._create_segment_callable(
                self.model,
                self.num_layers - self.tail_k,
                self.num_layers,
                is_first_segment=False,
                is_last_segment=True,
            )
            self.segment_a_wrapper = self._wrap_segment_if_needed(self.segment_a)
            self.segment_e_wrapper = self._wrap_segment_if_needed(self.segment_e)
        else:
            self.segment_c_raw = self._create_raw_segment_callable(
                self.model,
                self.head_k,
                self.num_layers - self.tail_k,
                is_first_segment=False,
                is_last_segment=False,
            )
            self.segment_c = self._maybe_compile_segment_callable(
                self.segment_c_raw,
                self.head_k,
                self.num_layers - self.tail_k,
            )
            self.segment_c_wrapper = self._wrap_segment_if_needed(self.segment_c)

        logger.info(
            "[EdgeCloud] Model loaded. num_layers=%d role=%s",
            self.num_layers,
            self.edge_cloud_cfg.role,
        )

        if self.drafter is not None:
            logger.info("[EdgeCloud] Loading drafter model...")
            if self.vllm_config.quant_config is not None:
                patch_load_weights(self.vllm_config)

            is_mtp_drafter = (
                self.speculative_config is not None
                and self.speculative_config.method == "mtp"
            )
            if is_mtp_drafter:
                # MTP draft models use the same edge-cloud layer range mechanism
                # as the main model. For MTP, all decoder layers run on the cloud
                # side (embedding_only mode), so head_k=tail_k=0.
                set_edge_cloud_layer_range(0, 0)

            with get_tp_context(self.drafter):
                self.drafter.load_model(self.model)

            if (
                is_mtp_drafter
                and hasattr(self.drafter, "model")
                and self.drafter.model is not None
            ):
                self._setup_edge_cloud_mtp(self.drafter.model)

    def _get_mtp_predictor(self, mtp_model: nn.Module) -> nn.Module | None:
        """Locate the MTP predictor module inside the draft model.

        Supports both upstream ``DeepSeekMTP`` style models (predictor exposes
        ``fc``/``norm``/``embed_tokens``/``layers``) and the vLLM-Ascend
        ``DeepSeekV4MTP`` style (predictor inside ``mtp_model.model``).
        """
        if hasattr(mtp_model, "model"):
            inner = mtp_model.model
            if hasattr(inner, "layers") and hasattr(inner, "embed_tokens"):
                return inner
        if hasattr(mtp_model, "layers") and hasattr(mtp_model, "embed_tokens"):
            return mtp_model
        return None

    def _clean_mtp_compilation_config(
        self,
        mtp_model: nn.Module,
        mtp_module_ids: set[int],
    ) -> None:
        """Remove stale static_forward_context entries pointing to MTP layers
        that were replaced by ``PPMissingLayer`` during edge-cloud sharding."""
        compilation_config = self.vllm_config.compilation_config
        if compilation_config is None:
            return

        current_mtp_module_ids = {
            id(module) for _, module in mtp_model.named_modules()
        }
        removed_prefixes: list[str] = []
        for prefix in list(compilation_config.static_forward_context.keys()):
            module = compilation_config.static_forward_context[prefix]
            if id(module) in mtp_module_ids and id(module) not in current_mtp_module_ids:
                del compilation_config.static_forward_context[prefix]
                removed_prefixes.append(prefix)

        if removed_prefixes:
            logger.info(
                "[EdgeCloud] MTP removed %d stale static_forward_context "
                "entries: %s",
                len(removed_prefixes),
                removed_prefixes,
            )

        if hasattr(compilation_config, "static_all_moe_layers"):
            compilation_config.static_all_moe_layers[:] = [
                prefix
                for prefix in compilation_config.static_all_moe_layers
                if prefix not in removed_prefixes
            ]

    def _setup_edge_cloud_mtp(self, mtp_model: nn.Module) -> None:
        predictor = self._get_mtp_predictor(mtp_model)
        if predictor is None:
            logger.warning("[EdgeCloud] Cannot find MTP predictor for sharding")
            return

        num_mtp_layers = len(predictor.layers)

        # Capture MTP module ids before sharding so we can clean stale
        # static_forward_context entries that point to removed layers.
        mtp_module_ids = {id(module) for _, module in mtp_model.named_modules()}

        # Use the same edge-cloud layer range mechanism as the main model.
        # For MTP this was set to head_k=tail_k=0 before the drafter was loaded.
        head_k, tail_k = get_edge_cloud_layer_range()

        local_layers: set[int] = set()
        if is_edge_device():
            if head_k > 0:
                local_layers.update(range(head_k))
            if tail_k > 0:
                local_layers.update(range(num_mtp_layers - tail_k, num_mtp_layers))
        else:
            local_layers.update(range(head_k, num_mtp_layers - tail_k))

        layer_keys = (
            list(predictor.layers.keys())
            if isinstance(predictor.layers, nn.ModuleDict)
            else list(range(num_mtp_layers))
        )
        for idx, key in enumerate(layer_keys):
            if idx not in local_layers and not isinstance(
                predictor.layers[key], PPMissingLayer
            ):
                predictor.layers[key] = PPMissingLayer()

        # Cloud side does not need embedding/fc/norm modules; edge keeps them.
        if not is_edge_device():
            for module_name in (
                "embed_tokens",
                "fc",
                "norm",
                "pre_fc_norm_hidden",
                "pre_fc_norm_embedding",
            ):
                module = getattr(predictor, module_name, None)
                if module is not None and not isinstance(module, PPMissingLayer):
                    setattr(predictor, module_name, PPMissingLayer())
            if (
                hasattr(mtp_model, "lm_head")
                and not isinstance(mtp_model.lm_head, PPMissingLayer)
            ):
                mtp_model.lm_head = PPMissingLayer()

        # Re-collect MoE parameters now that some layers may be placeholders.
        if hasattr(mtp_model, "set_moe_parameters"):
            mtp_model.set_moe_parameters()

        self._clean_mtp_compilation_config(mtp_model, mtp_module_ids)

        if hasattr(self, "_edge_cloud_mtp_segments"):
            delattr(self, "_edge_cloud_mtp_segments")
        self._edge_cloud_mtp_segments = {}

        # Pre-allocate persistent intermediate buffers for MTP edge-cloud
        # segments. ACLGraphWrapper requires stable input tensor addresses
        # across graph replay, but edge_cloud_broadcast_recv_mtp() allocates
        # fresh tensors every iteration. Copying received tensors into these
        # buffers before calling graph-wrapped segments avoids stale-address
        # crashes such as ACL error 507011.
        if hasattr(self, "_edge_cloud_mtp_intermediate_buffers"):
            delattr(self, "_edge_cloud_mtp_intermediate_buffers")
        if hasattr(predictor, "make_empty_intermediate_tensors"):
            max_mtp_tokens = self.max_num_tokens
            if enable_sp():
                tp_size = self.vllm_config.parallel_config.tensor_parallel_size
                max_mtp_tokens = (self.max_num_tokens + tp_size - 1) // tp_size
            self._edge_cloud_mtp_intermediate_buffers = (
                predictor.make_empty_intermediate_tensors(
                    batch_size=max_mtp_tokens,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        else:
            self._edge_cloud_mtp_intermediate_buffers = None

        if self.edge_cloud_cfg.role == "edge":
            seg_a = self._create_segment_callable(
                mtp_model, 0, 0, is_first_segment=True, is_last_segment=False
            )
            seg_e = self._create_segment_callable(
                mtp_model, 0, 0, is_first_segment=False, is_last_segment=True
            )
            self._edge_cloud_mtp_segments["a"] = self._wrap_segment_if_needed(seg_a)
            self._edge_cloud_mtp_segments["e"] = self._wrap_segment_if_needed(seg_e)
        else:
            seg_c = self._create_segment_callable(
                mtp_model, 0, 0, is_first_segment=False, is_last_segment=False
            )
            self._edge_cloud_mtp_segments["c"] = self._wrap_segment_if_needed(seg_c)

    def _sync_edge_cloud_mtp_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors,
    ) -> IntermediateTensors:
        """Copy received MTP intermediate tensors into persistent buffers.

        ACLGraphWrapper captures and replays graphs against fixed input
        addresses. edge_cloud_broadcast_recv_mtp() returns freshly-allocated
        tensors each iteration, so we copy them into pre-allocated buffers
        (sized to max_num_tokens) and return sliced views with stable
        addresses for the current num_tokens.
        """
        buffers = self._edge_cloud_mtp_intermediate_buffers
        if buffers is None:
            return intermediate_tensors

        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        copy_len = (num_tokens + tp_size - 1) // tp_size if enable_sp() else num_tokens

        synced: dict[str, torch.Tensor | Any] = {}
        for key, value in intermediate_tensors.items():
            if key not in buffers.tensors or not isinstance(value, torch.Tensor):
                # positions/spec_step_idx or any non-tensor metadata pass through
                synced[key] = value
                continue
            dst = buffers[key][:copy_len]
            recv_len = min(value.shape[0], copy_len)
            if recv_len:
                # Use synchronous copy on the NPU to avoid async hangs that
                # have been observed on the cloud side when non_blocking=True
                # is combined with ACL graph replay.
                dst[:recv_len].copy_(value[:recv_len])
            if recv_len < copy_len:
                dst[recv_len:].zero_()
            synced[key] = dst

        return IntermediateTensors(synced)

    def _sync_metadata_across_dp(
        self,
        num_tokens: int,
        is_draft_model: bool = False,
        cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        allow_dp_padding: bool = False,
    ) -> tuple[int, torch.Tensor | None, CUDAGraphMode]:
        # TODO: In vLLM, the only thing that needs to be synced is num_tokens, but in
        # our case, we still need to sync the other two flags as well. So we need to
        # include them in the all_reduce operation, and more over, we CANNOT skip it
        # even if we are running in eager mode, which harms performance.
        # FIXME: Restore the `or self.vllm_config.model_config.enforce_eager` here
        # immediately once the other two flags are no longer needed.
        if self.dp_size == 1:
            return num_tokens, None, cudagraph_mode

        if should_skip_allreduce_across_dp_group(self.vllm_config, is_draft_model):
            num_tokens_after_padding = torch.tensor([num_tokens] * self.dp_size, device="cpu", dtype=torch.int32)
            return num_tokens, num_tokens_after_padding, cudagraph_mode

        packed_tensor = torch.zeros(3, self.dp_size, device="cpu", dtype=torch.int32)
        packed_tensor[0][self.dp_rank] = num_tokens
        packed_tensor[1][self.dp_rank] = cudagraph_mode.value
        packed_tensor[2][self.dp_rank] = self._dp_batch_type_id
        dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)

        # Extract peer's batch_type_id (for DUMMY to match real's segment)
        self._peer_batch_type_id = int(packed_tensor[2, 1 - self.dp_rank].item())

        # [DPDBG] trace sync_metadata pairing to locate cross-DP misalignment.
        # Compare rank1 (cloud dp0) vs rank6 (cloud dp1) call sequences: the
        # all_reduce is a barrier so call N on both sides must pair. my_bt=0
        # means this side is a DUMMY, 1/2/3/4 = real PF/PL/DF/DL. peer_bt is
        # what the OTHER cloud contributed this round.
        _smc = getattr(self, "_sync_meta_count", 0) + 1
        self._sync_meta_count = _smc
        try:
            _sm_rank = dist.get_rank()
        except Exception:
            _sm_rank = -1
        logger.error(
            "[PP-EVT][DPDBG] sync_meta: rank=%s dp_rank=%s call=%s my_bt=%s peer_bt=%s "
            "num_tokens=%s",
            _sm_rank, self.dp_rank, _smc, self._dp_batch_type_id,
            self._peer_batch_type_id, num_tokens,
        )

        # Unpack the results
        num_tokens_across_dp = packed_tensor[0, :]
        max_tokens_across_dp = int(num_tokens_across_dp.max().item())
        synced_cudagraph_mode = CUDAGraphMode(_post_process_cudagraph_mode(packed_tensor))

        # Create a tensor for num_tokens_after_padding
        if allow_dp_padding or is_draft_model:
            num_tokens_after_padding = torch.tensor(
                [max_tokens_across_dp] * self.dp_size, device="cpu", dtype=torch.int32
            )
        else:
            num_tokens_after_padding = num_tokens_across_dp.cpu()

        return max_tokens_across_dp, num_tokens_after_padding, synced_cudagraph_mode

    def get_model(self) -> nn.Module:
        # get raw model out of the aclgraph wrapper.
        if isinstance(self.model, ACLGraphWrapper):
            return self.model.unwrap()
        return self.model

    def _pad_query_start_loc_for_fia(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        batch_desc_num_reqs: int | None = None,
    ) -> int:
        """
        This function is only designed to satisfied the constraint that when the layout is TND,
        the first dimension of `hidden_states` must equal the last element of `actual_seq_lengths_q`.
        """
        # TODO: need refactor later, related to vllm PR #34043 this pr delete func
        # relax_for_mixed_batch_cudagraphs, num_reqs no longer equals the actual number of requests.
        if cudagraph_runtime_mode == CUDAGraphMode.FULL and \
            self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs_padded = num_reqs
        else:
            num_reqs_padded = batch_desc_num_reqs if batch_desc_num_reqs is not None else num_reqs

        if num_tokens_padded == num_reqs_padded * self.uniform_decode_query_len:
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            last_loc = self.query_start_loc.np[num_reqs]
            self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1] = (
                self.arange_np[1 : num_reqs_padded + 1 - num_reqs] * self.uniform_decode_query_len + last_loc
            )
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded

            # Do not insert if the last value already equals the num_tokens
            if self.query_start_loc.np[num_reqs_padded] < num_tokens_padded:
                # Insert a dummy request instead of change the last value directly
                self.query_start_loc.np[num_reqs_padded + 1] = num_tokens_padded
                num_reqs_padded = num_reqs_padded + 1

        self.query_start_loc.copy_to_gpu()

        return num_reqs_padded

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None, int, list[np.ndarray[Any, Any]]]:
        """
        :return: tuple[
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # Get the attention state.
        if not scheduler_output.scheduled_spec_decode_tokens:
            num_valid_tokens = num_scheduled_tokens
        else:
            num_valid_tokens = np.array(
                [
                    scheduler_output.num_scheduled_tokens[i]
                    - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                    for i in self.input_batch.req_ids
                ],
                dtype=np.int32,
            )
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)

        # Determine if it's a splitfuse batch
        with_prefill = attn_state not in [AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding]
        self.with_prefill = with_prefill

        # Get positions.
        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )
        positions_np = self._positions_np_buf[:total_num_scheduled_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            self.query_pos.np[: cu_num_tokens[-1]],
            out=positions_np,
        )

        # For PCP, compute slot_mapping on GPU using pre-PCP-split positions.
        # Use blocking .to(device) to ensure data lands on GPU before PCP
        # modifies CPU position buffers. PCP and async spec decode are
        # mutually exclusive, so the sync is acceptable.
        if self.pcp_size > 1:
            pre_pcp_positions = torch.from_numpy(
                positions_np[:total_num_scheduled_tokens]
            ).to(self.device)
            pre_pcp_qsl = torch.zeros(
                num_reqs + 1, dtype=torch.int32, device=self.device)
            pre_pcp_qsl[1:num_reqs + 1] = torch.from_numpy(
                cu_num_tokens
            ).to(dtype=torch.int32, device=self.device)
            self.input_batch.block_table.compute_slot_mapping(
                num_reqs, pre_pcp_qsl, pre_pcp_positions)

        if self.use_cp:
            self.pcp_manager.init_batch_info(
                num_scheduled_tokens,
                self.input_batch.num_reqs,
            )

        # for pcp, prefill mtp should use origin scheduleroutput ,
        if self.speculative_config and self.use_cp:
            self.pcp_manager.generate_pcp_mtp_input(
                total_num_scheduled_tokens,
                scheduler_output.num_scheduled_tokens,
                with_prefill,
                self.input_batch,
                self.arange_np,
                req_indices,
                positions_np,
                cu_num_tokens,
                self._draft_token_ids,  # type: ignore[has-type]
                scheduler_output,
                self.num_spec_tokens,
            )

        if self.pcp_size > 1:
            num_scheduled_tokens[:num_reqs], position_pcp = self.pcp_manager.update_tokens_for_pcp(
                num_scheduled_tokens[:num_reqs], self.arange_np
            )
            # Re-update after PCP split sequences.
            total_num_scheduled_tokens = sum(num_scheduled_tokens[:num_reqs])
            req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
            cu_num_tokens = self._get_cumsum_and_arange(num_scheduled_tokens, self.query_pos.np)
            positions_np = self._positions_np_buf[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                position_pcp[:total_num_scheduled_tokens],
                out=positions_np,
            )
        if self.pcp_size > 1 and self.pcp_manager.pcp_use_hybrid_attn:
            assert self.pcp_manager.num_scheduled_tokens_padded is not None
            self.query_lens = torch.from_numpy(self.pcp_manager.num_scheduled_tokens_padded)
        else:
            self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        token_indices_tensor = torch.from_numpy(token_indices)
        # Prepare input_ids.
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids, 0, token_indices_tensor, out=self.is_token_ids.cpu[:total_num_scheduled_tokens]
            )

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds and (self.is_multimodal_model or self.enable_prompt_embeds):
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                if self.pcp_size > 1:
                    # PCP can split one request into non-contiguous token positions.
                    # We must gather prompt embeds by actual scheduled positions.
                    req_positions_np = positions_np[output_idx : output_idx + num_sched]
                    dst_slice = self.inputs_embeds.cpu[output_idx : output_idx + num_sched]
                    self.pcp_manager.fill_prompt_embeds_for_pcp(
                        req_embeds=req_embeds,
                        req_positions_np=req_positions_np,
                        dst_slice=dst_slice,
                    )
                else:
                    start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                    # Skip if trying to read beyond available embeddings
                    if start_pos >= req_embeds.shape[0]:
                        output_idx += num_sched
                        continue

                    # Copy available embeddings
                    end_pos = start_pos + num_sched
                    actual_end = min(end_pos, req_embeds.shape[0])
                    actual_num_sched = actual_end - start_pos

                    if actual_num_sched > 0:
                        self.inputs_embeds.cpu[output_idx : output_idx + actual_num_sched].copy_(
                            req_embeds[start_pos:actual_end]
                        )

                output_idx += num_sched

        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        self.query_start_loc.copy_to_gpu()

        # Now, query_start_loc is padded.
        # But gdn needs an unpadded one.
        # gdn_query_start_loc is an unpadded version of query_start_loc.
        # TODO delete it if fia's check is removed.
        if self._has_gdn:
            self.gdn_query_start_loc.np[0] = 0
            self.gdn_query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
            self.gdn_query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
            self.gdn_query_start_loc.copy_to_gpu()

        # Compute optimistic seq_lens (assumes all draft tokens from previous
        # iteration accepted). Store in optimistic_seq_lens_cpu for use by
        # _build_attention_metadata (max_seq_len) and discard_request_mask.
        # seq_lens (GPU) will be computed later using the same optimistic values.
        torch.add(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            torch.from_numpy(num_scheduled_tokens),
            out=self.optimistic_seq_lens_cpu[:num_reqs],
        )
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)

        # Build prev_positions mapping: current pos -> prev pos (-1 if new).
        # Used for gathering from previous iteration's GPU tensors.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        self._compute_prev_positions(num_reqs)

        # Fill unused with -1. Needed for reshape_and_cache in attention_cp
        self.query_start_loc.gpu[num_reqs + 1 :].fill_(-1)

        # Copy the tensors to the NPU.
        self._prepare_input_ids(scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens)
        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self._calc_mrope_positions(scheduler_output)
            if self.pcp_size > 1:
                self.pcp_manager.remap_mrope_positions_for_pcp(
                    positions_np,
                    num_scheduled_tokens,
                    num_reqs,
                    self.input_batch,
                    self.requests,
                    self.mrope_positions,
                )
            self.mrope_positions.gpu.copy_(
                self.mrope_positions.cpu,
                non_blocking=True,
            )
        elif self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output)
            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )

        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)
        base_num_reqs = self.input_batch.num_reqs
        num_reqs = base_num_reqs
        tokens_original = None
        if self.pcp_size > 1:
            # while pcp > 1, we need the original num_scheduled_tokens before split
            # to calculate discard_requests_mask
            tokens_original = [scheduler_output.num_scheduled_tokens[i] for i in self.input_batch.req_ids]
            original_seq_lens_np = self.input_batch.num_computed_tokens_cpu[:num_reqs] + np.array(
                tokens_original, dtype=np.int32
            )
            discard_requests_mask = original_seq_lens_np < num_tokens_np
        else:
            discard_requests_mask = self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np

        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[: self.num_discarded_requests] = discard_request_indices
        self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

        # Sync num_accepted_tokens from CPU (set by
        # _update_states_after_model_execute for hybrid models).
        if self.num_accepted_tokens_event is not None:
            self.num_accepted_tokens_event.synchronize()
            # Async mode: condense() reordered indices, use prev_positions mapping
            if self.use_async_scheduling and prev_req_id_to_index:
                prev_idx = self.prev_positions.np[:num_reqs]
                new_mask = prev_idx < 0
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[
                        np.where(new_mask, 0, prev_idx)
                    ]
                )
                self.num_accepted_tokens.np[:num_reqs][new_mask] = 1
                self.input_batch.num_accepted_tokens_cpu[:num_reqs] = (
                    self.num_accepted_tokens.np[:num_reqs]
                )
            else:
                # Non-async mode: use values directly
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()
        else:
            self.num_accepted_tokens.np.fill(1)
            self.num_accepted_tokens.gpu.fill_(1)

        # Update num_computed_tokens on GPU. In async spec decode,
        # CPU values are optimistic (all drafts accepted). The kernel
        # corrects on GPU using the previous step's
        # valid_sampled_token_count_gpu. Otherwise, just copy from CPU.
        # In edge-cloud embedding_only mode, the tail segment reuses the
        # num_computed_tokens already corrected by the head segment, so skip
        # both the kernel and the CPU fallback copy to avoid re-introducing
        # the scheduler's optimistic value.
        if not self._is_edge_cloud_embed_only_tail:
            if (
                self.use_async_spec_decode
                and self.valid_sampled_token_count_gpu is not None
                and prev_req_id_to_index
            ):
                self.prev_positions.copy_to_gpu(num_reqs)
                self.prev_num_draft_tokens.copy_to_gpu()
                cpu_values = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].to(
                    device=self.device, non_blocking=True
                )
                update_num_computed_tokens_for_batch_change(
                    self.num_computed_tokens,
                    self.num_accepted_tokens.gpu[:num_reqs],
                    self.prev_positions.gpu[:num_reqs],
                    self.valid_sampled_token_count_gpu,
                    self.prev_num_draft_tokens.gpu,
                    cpu_values,
                )
            else:
                self.num_computed_tokens[:num_reqs].copy_(
                    self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
                    non_blocking=True,
                )

        self.req_indices.np[:total_num_scheduled_tokens] = req_indices
        self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
        req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]

        self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
        self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
        self.num_scheduled_tokens.copy_to_gpu(num_reqs)
        num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]
        # fix prefix cache ci test
        if self.pcp_size > 1:
            # When PCP (Prefill Context Parallel) is enabled, positions use
            # special PCP offsets (position_pcp) that are only computed on CPU.
            # Copy the correctly-computed CPU positions to GPU instead of
            # recomputing on GPU (which would miss the PCP offsets).
            self.positions[:total_num_scheduled_tokens].copy_(
                torch.from_numpy(
                    positions_np[:total_num_scheduled_tokens]
                ).to(self.device),
                non_blocking=True,
            )
        else:
            self.positions[:total_num_scheduled_tokens] = (
                self.num_computed_tokens[req_indices_gpu].to(torch.int64)
                + self.query_pos.gpu[:total_num_scheduled_tokens]
            )
        self.seq_lens[:num_reqs] = (
            self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
        )
        self.seq_lens[num_reqs:].fill_(0)

        # In async spec decode mode, num_computed_tokens was corrected on GPU
        # by update_num_computed_tokens_for_batch_change, so seq_lens (GPU) is
        # correct but optimistic_seq_lens_cpu is stale (it assumed all drafts
        # were accepted). Sync the corrected values back to CPU so that NPU
        # attention backends (which use _seq_lens_cpu) get the right values.
        # Use non_blocking copy to pinned memory and record an event;
        # _build_attention_metadata will synchronize before reading.
        if (
            self._needs_seq_lens_cpu_sync
            and self.use_async_spec_decode
            and self.valid_sampled_token_count_gpu is not None
            and prev_req_id_to_index
        ):
            self.optimistic_seq_lens_cpu[:num_reqs].copy_(
                self.seq_lens[:num_reqs], non_blocking=True
            )
            if self._seq_lens_cpu_event is None:
                self._seq_lens_cpu_event = torch.npu.Event()
            self._seq_lens_cpu_event.record()
            self._seq_lens_cpu_event_pending = True
        else:
            self._seq_lens_cpu_event_pending = False

        num_computed_tokens_for_compress = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs]
        )
        if (
            self.use_compress
            and self.use_async_spec_decode
            and self.valid_sampled_token_count_gpu is not None
            and prev_req_id_to_index
        ):
            # Async spec decode keeps the CPU counter optimistic until after
            # target forward is launched. DSV4 compressed KV slot mapping is
            # CPU-built today, so use the GPU-corrected counter before
            # computing compressed cache positions.
            if (
                self._seq_lens_cpu_event_pending
                and self._seq_lens_cpu_event is not None
            ):
                self._seq_lens_cpu_event.synchronize()
                self._seq_lens_cpu_event_pending = False
                num_computed_tokens_for_compress = (
                    self.optimistic_seq_lens_cpu[:num_reqs].numpy()
                    - num_scheduled_tokens[:num_reqs]
                )
            else:
                num_computed_tokens_for_compress = (
                    self.num_computed_tokens[:num_reqs].to("cpu").numpy()
                )

        (positions_compressed_list, req_indices_compressed_list,
         num_scheduled_tokens_compressed_list) = get_compressed_pos_and_indices(
            num_computed_tokens_for_compress,
            num_scheduled_tokens[:num_reqs], self.arange_np[:num_reqs],
            self.use_compress,
            self.kv_cache_config.kv_cache_groups)
        # For non-PCP, compute slot_mapping on GPU. PCP slot_mapping was
        # already computed on GPU before PCP split the positions.
        if self.pcp_size <= 1:
            self.input_batch.block_table.compute_slot_mapping(
                num_reqs,
                self.query_start_loc.gpu[: num_reqs + 1],
                self.positions[:total_num_scheduled_tokens],
                positions_compressed_list,
                req_indices_compressed_list,
            )

        if self.use_async_spec_decode and (self.uses_mrope or self.uses_xdrope_dim > 0):
            drift = self.num_computed_tokens[req_indices_gpu].to(
                torch.int64
            ) - self.input_batch.num_computed_tokens_cpu_tensor[req_indices].to(
                device=self.device, dtype=torch.int64, non_blocking=True
            )
            target = self.mrope_positions if self.uses_mrope else self.xdrope_positions
            target.gpu[:, :total_num_scheduled_tokens] += drift

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
            num_draft_tokens = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
            if self.use_cp:
                logits_indices = self.pcp_manager.get_logits_indices(cu_num_tokens, num_reqs, tokens_original)
                logits_indices = logits_indices.pin_memory().to(self.device, non_blocking=True)
            else:
                logits_indices = self.query_start_loc.gpu[1 : num_reqs + 1] - 1
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            new_schedule_reqs = [x.req_id for x in scheduler_output.scheduled_new_reqs]
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                draft_len = len(draft_token_ids)
                num_draft_tokens[req_idx] = draft_len
                if (self.is_kv_consumer and req_id in new_schedule_reqs) or \
                   (self.input_batch.num_computed_tokens_cpu[req_idx] >= \
                    self.input_batch.num_prompt_tokens[req_idx]):
                    num_decode_draft_tokens[req_idx] = draft_len
                else:
                    num_decode_draft_tokens[req_idx] = -1

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens,
                cu_num_tokens,
                num_pcp_pads=self.pcp_manager.num_pcp_pads_cpu[:num_reqs] if self.pcp_size > 1 else None,
            )
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1

            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()
        # save logits_indices for pcp spec decode usage
        self.logits_indices = logits_indices

        # Hot-Swap lora model
        if self.lora_config:
            assert np.sum(num_sampled_tokens) <= self.vllm_config.scheduler_config.max_num_batched_tokens
            self.set_active_loras(self.input_batch, num_scheduled_tokens, num_sampled_tokens)
        if lmhead_tp_enable():
            max_num_reqs_across_dp = self.max_num_reqs * self.uniform_decode_query_len
            logits_indices = nn.functional.pad(logits_indices, (0, max_num_reqs_across_dp - logits_indices.shape[0]))

        # Cache local scheduled token layout for PCP-aware multimodal preprocess.
        if (
            self.pcp_size > 1
            and self.supports_mm_inputs
            and get_pp_group().is_first_rank
            and not self.model_config.is_encoder_decoder
        ):
            self.pcp_manager.cache_local_schedule_layout(
                num_scheduled_tokens=num_scheduled_tokens,
                num_reqs=base_num_reqs,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
            )

        return (
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
            num_scheduled_tokens_compressed_list
        )

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:
        restore_state = None

        # For PCP, local worker token count can differ from scheduler global count.
        # Multimodal preprocessing must use local scheduled token count.
        if (
            self.pcp_size > 1
            and self.supports_mm_inputs
            and get_pp_group().is_first_rank
            and not self.model_config.is_encoder_decoder
        ):
            positions_np = (
                self.positions.np
                if hasattr(self.positions, "np")
                else self._positions_np_buf
            )
            local_num_sched, local_total = self.pcp_manager.get_local_schedule_layout()
            restore_state = self.pcp_manager.maybe_localize_scheduler_output_for_mm_preprocess(
                scheduler_output=scheduler_output,
                req_ids=self.input_batch.req_ids,
                requests=self.requests,
                positions_np=positions_np,
                local_num_scheduled_tokens=local_num_sched,
                local_total_num_scheduled_tokens=local_total,
                encoder_cache=self.encoder_cache,
            )

        try:
            return super()._preprocess(
                scheduler_output, num_input_tokens, intermediate_tensors
            )
        finally:
            if (
                self.pcp_size > 1
                and self.supports_mm_inputs
                and get_pp_group().is_first_rank
                and not self.model_config.is_encoder_decoder
            ):
                self.pcp_manager.restore_scheduler_output_after_mm_preprocess(
                    scheduler_output, restore_state
                )

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
        shift_computed_tokens: int = 0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if self.pcp_size <= 1:
            return super()._gather_mm_embeddings(scheduler_output, shift_computed_tokens)

        local_num_scheduled_tokens, _ = self.pcp_manager.get_local_schedule_layout()
        if local_num_scheduled_tokens is None:
            return super()._gather_mm_embeddings(scheduler_output, shift_computed_tokens)

        total_num_scheduled_tokens = int(np.sum(local_num_scheduled_tokens))
        positions_np = self.positions.np if hasattr(self.positions, "np") else self._positions_np_buf
        mm_embeds = list[torch.Tensor]()
        is_mm_embed = torch.zeros(
            total_num_scheduled_tokens, dtype=torch.bool, device="cpu"
        )

        (
            mm_embeds,
            should_sync_mrope_positions,
            should_sync_xdrope_positions,
        ) = self.pcp_manager.gather_mm_embeddings_for_pcp(
            req_ids=self.input_batch.req_ids,
            requests=self.requests,
            positions_np=positions_np,
            local_num_scheduled_tokens=local_num_scheduled_tokens,
            shift_computed_tokens=shift_computed_tokens,
            encoder_cache=self.encoder_cache,
            is_mm_embed=is_mm_embed,
            model=self.model,
            is_multimodal_pruning_enabled=self.is_multimodal_pruning_enabled,
            uses_mrope=self.uses_mrope,
            warning_once=logger.warning_once,
        )

        if should_sync_mrope_positions:
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        if should_sync_xdrope_positions:
            self._calc_xdrope_positions(scheduler_output)
            self.xdrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        return mm_embeds, is_mm_embed

    def _build_attn_state(self, num_reqs, num_scheduled_tokens, num_valid_tokens):
        if np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] == 0):
            attn_state = AscendAttentionState.PrefillNoCache
        # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
        elif np.all(num_scheduled_tokens == 1):
            attn_state = AscendAttentionState.DecodeOnly
            if self.speculative_config and self.speculative_config.method == "mtp":
                # SpecDecoding now supports seq_len=1 and seq_len=2
                # In Prefilling Decoding Disaggregation scenario, SpecDecoding need to supports seq_len=1
                attn_state = AscendAttentionState.SpecDecoding
        # Speculative decoding.
        elif np.all(num_valid_tokens == 1):
            if self.speculative_config:
                attn_state = AscendAttentionState.SpecDecoding
            else:
                attn_state = AscendAttentionState.ChunkedPrefill
        # splitfuse
        elif self.scheduler_config.enable_chunked_prefill:
            attn_state = AscendAttentionState.ChunkedPrefill
        else:
            attn_state = AscendAttentionState.PrefillCacheHit

        # For the overlay of the PCP feature and the eagle3, attn_state needs to be recovered
        # TODO: Resolved the conflict between the sunset of attn_state and the PCP that requires this interface.
        if attn_state == AscendAttentionState.SpecDecoding and self.speculative_config.method != "mtp":
            self.attn_state = AscendAttentionState.ChunkedPrefill  # type: ignore
        else:
            self.attn_state = attn_state  # type: ignore

        return attn_state

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        num_pcp_pads: np.ndarray | None,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1
        # Step 1.
        # cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # _arange_scratch[:11]: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens = self._get_cumsum_and_arange(
            num_sampled_tokens, self._arange_scratch, cumsum_dtype=np.int32
        )
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += self._arange_scratch[: cu_num_sampled_tokens[-1]]

        # while pcp > 1, decode results may contain padding (from pcp all-gather),
        # update logits_indices after getting draft_token_ids from ori logits_indices
        if self.pcp_size > 1:
            cu_num_scheduled_tokens = cu_num_scheduled_tokens * self.pcp_size - num_pcp_pads
            logits_indices_pcp = np.repeat(cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
            logits_indices_pcp += self._arange_scratch[: cu_num_sampled_tokens[-1]]
            logits_indices_pcp = torch.from_numpy(logits_indices_pcp).pin_memory().to(self.device, non_blocking=True)

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # [3, 3, 5, 5, 6]
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        total_num_draft_tokens = cu_num_draft_tokens[-1]
        # [0, 0, 0, 3, 3, 5]
        cumsums_offsets = np.repeat(cu_num_draft_tokens - num_draft_tokens, num_draft_tokens)
        # [0, 1, 2, 0, 1, 0]
        arange = self.arange_np[:total_num_draft_tokens] - cumsums_offsets
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> NPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).pin_memory().to(self.device, non_blocking=True)
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).pin_memory().to(self.device, non_blocking=True)
        logits_indices = torch.from_numpy(logits_indices).pin_memory().to(self.device, non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).pin_memory().to(self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).pin_memory().to(self.device, non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]
        if self.pcp_size > 1:
            logits_indices = logits_indices_pcp
        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        if self.valid_sampled_token_count_event is None:
            return

        # Initialize a new stream to overlap the copy operation with
        # prepare_input of draft model.
        default_stream = torch.npu.current_stream()
        with torch.npu.stream(self.valid_sampled_token_count_copy_stream):
            self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)
            counts = valid_sampled_tokens_count
            counts_cpu = self.valid_sampled_token_count_cpu
            assert counts_cpu is not None
            counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)
            self.valid_sampled_token_count_event.record()

        if self.use_async_spec_decode:
            # Stash for GPU-side correction in _prepare_inputs.
            self.valid_sampled_token_count_gpu = valid_sampled_tokens_count # type: ignore[no-redef]
        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

    # TODO: Once the PCP features are complete, it will fully inherit the classes from the VLLM community.
    def propose_draft_token_ids(
        self,
        valid_sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        scheduler_output: "SchedulerOutput",
        spec_decode_metadata: SpecDecodeMetadata,
        spec_decode_common_attn_metadata: AscendCommonAttentionMetadata,
        positions: torch.Tensor,
        num_scheduled_tokens: int,
        hidden_states: torch.Tensor,
        aux_hidden_states: torch.Tensor = None,
        sample_hidden_states: torch.Tensor = None,
        target_model_batch_desc: BatchDescriptor = None,
    ) -> list[list[int]] | None:
        if not self.drafter:
            # Speculative decoding is not enabled.
            draft_token_ids = None
        elif isinstance(self.drafter, (AscendNgramProposer, AscendSuffixDecodingProposer)):
            draft_token_ids = self.drafter.propose(valid_sampled_token_ids)
        elif isinstance(self.drafter, AscendNgramProposerNPU):
            batch_size = min(self.input_batch.num_reqs, self.token_ids_gpu_tensor.shape[0])

            # prepare sampled_token_ids tensor（list → padded tensor）
            sampled_token_ids = valid_sampled_token_ids
            if isinstance(sampled_token_ids, list):
                max_len = max((len(sublist) for sublist in sampled_token_ids), default=0)
                max_len = max(max_len, 1)
                padded_list = [
                    sublist + [-1] * (max_len - len(sublist))
                    for sublist in sampled_token_ids
                ]
                sampled_token_ids_tensor = torch.tensor(
                    padded_list, dtype=torch.int32, device=self.device
                )
            else:
                sampled_token_ids_tensor = sampled_token_ids

            (_token_ids, next_token_ids, draft_token_ids,
             num_valid_draft_tokens) = torch.ops._C_ascend.npu_ngram_spec_decode(
                self.token_ids_gpu_tensor[:batch_size],       # [B, max_seq_len], in-place
                self.num_tokens_no_spec_gpu[:batch_size],      # [B]
                sampled_token_ids_tensor[:batch_size],         # [B, max_new_tokens]
                self.discard_request_mask.gpu[:batch_size],    # [B]
                vocab_size=self.model_config.get_vocab_size(),
                min_n=self.drafter.min_n,
                max_n=self.drafter.max_n,
                k=self.drafter.k,
            )

            # only async scheduling, set prev_sampled_token_ids，
            if self.use_async_scheduling:
                self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

            # save num_valid_draft_tokens for scheduler trim
            self._num_valid_draft_tokens = num_valid_draft_tokens

            # async D2H copy num_valid_draft_tokens
            copy_num_valid_draft_tokens(
                self._num_valid_draft_tokens_cpu,
                self._num_valid_draft_tokens_copy_stream,
                self._num_valid_draft_tokens_event,
                self._num_valid_draft_tokens,
                batch_size,
            )
        elif isinstance(self.drafter, AscendMedusaProposer):
            draft_token_ids = self.drafter.propose(
                valid_sampled_token_ids, sampling_metadata, spec_decode_metadata, sample_hidden_states
            )
        elif self.speculative_config.uses_extract_hidden_states():
            # Handle extract_hidden_states method
            assert isinstance(self.drafter, AscendExtractHiddenStatesProposer)
            assert isinstance(valid_sampled_token_ids, torch.Tensor), (
                "sampled_token_ids should be a torch.Tensor for "
                "extract_hidden_states method."
            )
            if not self.use_aux_hidden_state_outputs or aux_hidden_states is None:
                raise ValueError(
                    "aux_hidden_states are required when using `extract_hidden_states`"
                )
            common_attn_metadata = spec_decode_common_attn_metadata
            target_hidden_states = [h[:num_scheduled_tokens] for h in aux_hidden_states]

            draft_token_ids = self.drafter.propose(
                sampled_token_ids=valid_sampled_token_ids,
                target_hidden_states=target_hidden_states,
                common_attn_metadata=common_attn_metadata,
            )
            next_token_ids, valid_sampled_tokens_count = (
                self.drafter.prepare_next_token_ids_padded(
                    valid_sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_indices.gpu,
                    self.num_discarded_requests,
                )
            )
            self._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)
        elif self.speculative_config.use_eagle() or self.speculative_config.uses_draft_model():
            common_attn_metadata = spec_decode_common_attn_metadata
            sampled_token_ids = valid_sampled_token_ids

            if self.vllm_config.speculative_config.disable_padded_drafter_batch:
                # When padded-batch is disabled, the sampled_token_ids should be
                # the cpu-side list[list[int]] of valid sampled tokens for each
                # request, with invalid requests having empty lists.
                assert isinstance(sampled_token_ids, list), (
                    "sampled_token_ids should be a python list whenpadded-batch is disabled."
                )
                assert self.drafter is not None
                next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                    sampled_token_ids, self.requests, self.input_batch, scheduler_output.num_scheduled_tokens
                )
            else:
                # When using padded-batch, the sampled_token_ids should be
                # the gpu tensor of sampled tokens for each request, of shape
                # (num_reqs, num_spec_tokens + 1) with rejected tokens having
                # value -1.
                assert isinstance(sampled_token_ids, torch.Tensor), (
                    "sampled_token_ids should be a torch.Tensor whenpadded-batch is enabled."
                )
                assert self.drafter is not None
                next_token_ids, valid_sampled_tokens_count = self.drafter.prepare_next_token_ids_padded(
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_indices.gpu,
                    self.num_discarded_requests,
                )

            req_scheduled_tokens = scheduler_output.num_scheduled_tokens
            if self.use_cp:
                long_seq_metadata = self.long_seq_metadata  # type: ignore
                input_ids_pcp_full = self.pcp_manager.input_ids_pcp_full.gpu
                query_start_loc_pcp_full = self.pcp_manager.query_start_loc_pcp_full.gpu
                query_start_loc_pcp_full_cpu = self.pcp_manager.query_start_loc_pcp_full.cpu
                num_reqs = self.input_batch.num_reqs
                num_prefill_reqs = self.pcp_manager.num_prefill_reqs
                num_decode_reqs = self.pcp_manager.num_decode_reqs
            else:
                long_seq_metadata = None  # type: ignore
                num_prefill_reqs = 0
                num_decode_reqs = 0

            # Let the target override the hidden state fed to the drafter
            # (e.g. DeepSeek V4 MTP needs the pre-hc_head residual). Safe to
            # rebind here: hidden_states was already consumed for sampling
            # above and is not used again in this branch.
            mtp_hidden_states = getattr(
                self.get_model(), "get_mtp_target_hidden_states", lambda: None
            )()
            if mtp_hidden_states is not None:
                hidden_states = mtp_hidden_states

            num_rejected_tokens_gpu = None
            if spec_decode_metadata is None:
                # update pcp related params
                if self.pcp_size > 1:
                    token_indices_to_sample = query_start_loc_pcp_full[1 : num_reqs + 1] - 1
                    target_token_ids = input_ids_pcp_full[:num_scheduled_tokens]
                    target_positions = self._get_positions(num_scheduled_tokens)
                    target_hidden_states = hidden_states
                    if self.use_aux_hidden_state_outputs:
                        target_hidden_states = torch.cat([h for h in aux_hidden_states], dim=-1)
                else:
                    token_indices_to_sample = None
                    # input_ids can be None for multimodal models.
                    target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                    target_positions = self._get_positions(num_scheduled_tokens)
                    if self.use_aux_hidden_state_outputs:
                        target_hidden_states = torch.cat([h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1)
                    else:
                        target_hidden_states = hidden_states[:num_scheduled_tokens]
            else:
                if self.pcp_size > 1:
                    assert common_attn_metadata is not None
                    common_attn_metadata.query_start_loc_cpu[: num_reqs + 1] = query_start_loc_pcp_full_cpu[
                        : num_reqs + 1
                    ]
                    assert common_attn_metadata is not None
                    common_attn_metadata.query_start_loc[: num_reqs + 1] = query_start_loc_pcp_full[: num_reqs + 1]
                if self.vllm_config.speculative_config.disable_padded_drafter_batch:
                    # NOTE: Currently, MTP-fullgraph is incompatibility with pcp
                    token_indices_to_sample = None
                    assert self.drafter is not None
                    common_attn_metadata, token_indices = self.drafter.prepare_inputs(
                        common_attn_metadata, sampled_token_ids, spec_decode_metadata.num_draft_tokens
                    )
                else:
                    assert self.drafter is not None
                    common_attn_metadata, token_indices, token_indices_to_sample, num_rejected_tokens_gpu = (
                        self.drafter.prepare_inputs_padded(
                            common_attn_metadata, spec_decode_metadata, valid_sampled_tokens_count
                        )
                    )
                if self.pcp_size > 1:
                    target_token_ids = input_ids_pcp_full[token_indices]
                    target_positions = positions
                    target_hidden_states = hidden_states
                    if self.use_aux_hidden_state_outputs:
                        target_hidden_states = torch.cat([h for h in aux_hidden_states], dim=-1)
                else:
                    target_token_ids = self.input_ids.gpu[token_indices]
                    target_positions = self._get_positions(token_indices)
                    if self.use_aux_hidden_state_outputs:
                        target_hidden_states = torch.cat([h[token_indices] for h in aux_hidden_states], dim=-1)
                    else:
                        target_hidden_states = hidden_states[token_indices]
            assert self.drafter is not None
            draft_token_ids = self.drafter._propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                common_attn_metadata=common_attn_metadata,
                target_model_batch_desc=target_model_batch_desc,
                sampling_metadata=sampling_metadata,
                req_scheduled_tokens=req_scheduled_tokens,
                long_seq_metadata=long_seq_metadata,
                num_prefill_reqs=num_prefill_reqs,
                num_decode_reqs=num_decode_reqs,
                scheduler_output=scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
            if not self.vllm_config.speculative_config.disable_padded_drafter_batch:
                self._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)
        else:
            raise ValueError(f"Unknown speculative decoding method: {self.speculative_config.method}")

        return draft_token_ids

    def _copy_draft_token_ids_to_cpu(
        self, scheduler_output: "SchedulerOutput", zeros_only: bool = False
    ) -> None:
        if not self.num_spec_tokens:
            return
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests
            or self.input_batch.sampling_metadata.output_token_ids
        ):
            return
        self._draft_token_req_ids = self.input_batch.req_ids.copy()

        draft_token_ids: torch.Tensor = self._draft_token_ids  # type: ignore[has-type]
        if not torch.is_tensor(draft_token_ids):
            return
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_copy_stream is not None
        assert self.draft_token_ids_cpu is not None
        default_stream = torch.npu.current_stream()
        num_reqs = draft_token_ids.shape[0]
        with torch.npu.stream(self.draft_token_ids_copy_stream):
            if not zeros_only:
                self.draft_token_ids_copy_stream.wait_stream(default_stream)
                self.draft_token_ids_cpu[:num_reqs].copy_(
                    draft_token_ids, non_blocking=True
                )
            else:
                self.draft_token_ids_cpu[:num_reqs] = 0
            self.draft_token_ids_event.record()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
        layer_slice_info: Any = None,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        # PD-separation: encode batch_type for cross-DP sync (0=EMPTY,
        # 1=PF, 2=PL, 3=DF, 4=DL). Packed into sync_metadata all_reduce
        # so the idle DP's DUMMY can match this forward's segment.
        _bt_map = {
            BatchType.EMPTY: 0, BatchType.PREFILL_FIRST: 1,
            BatchType.PREFILL_LAST: 2, BatchType.DECODE_FIRST: 3,
            BatchType.DECODE_LAST: 4,
        }
        self._dp_batch_type_id = _bt_map.get(scheduler_output.batch_type, 0)

        if self.vllm_config.model_config.enable_return_routed_experts:
            if vllm_version_is("0.20.2"):
                capturer = RoutedExpertsCapturer.get_instance()
                if capturer is not None:
                    capturer.clear_buffer()
            elif self.routed_experts_initialized:
                self.routed_experts_capturer.clear_buffer()

        if self.ascend_config.profiling_chunk_config.need_timing:
            # Check if the scheduler signaled that calibration is complete.
            # This flag is set cross-process via scheduler_output because
            # modifying the config singleton in the scheduler process does
            # not affect this worker process.
            if getattr(scheduler_output, "disable_profiling_timing", False):
                self.ascend_config.profiling_chunk_config.need_timing = False
            else:
                self._sync_device()
                self._execution_start_time = time.perf_counter()
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called after execute_model() returns None.")

        # --- Layer slice: non-first slice fast path ---
        # For slices 1..N-1, the batch state (requests, attention metadata,
        # positions, etc.) was already set up by slice 0.  We only need to
        # load the saved intermediate state and run the model forward for
        # the current slice's layer range.
        if (
            layer_slice_info is not None
            and not layer_slice_info.is_first_slice
        ):
            # Slice 0 may have returned early (e.g. 0 scheduled tokens)
            # without saving intermediate state.  In that case all
            # subsequent slices should also return early.
            if self._layerwise_intermediate is None:
                return EMPTY_MODEL_RUNNER_OUTPUT
            return self._execute_layerwise_continuation(
                layer_slice_info
            )
        # --- End layer slice fast path ---

        # Edge-cloud tail-segment validation: use the control-plane
        # head_token to resume the suspended HeadState. The scheduler and
        # hidden channel selection guarantee data-plane alignment.
        if (
            self._edge_cloud_enabled
            and is_edge_device()
            and scheduler_output.batch_type in (
                BatchType.PREFILL_LAST, BatchType.DECODE_LAST
            )
            and intermediate_tensors is not None
        ):
            self._resume_and_validate_head_state(scheduler_output)

            # [PD-FIX] Stale tail segment (PL/DL): cloud shipped a tail batch
            # whose reqs already finished on the edge and were popped from
            # self.requests during the head->tail window. Discard segment_e to
            # avoid _update_states crashing on self.requests[req_id]. HeadState
            # already popped above; hidden already received by
            # _execute_model_edge_tail (data-plane contract preserved).
            # update_from_output's `request is None or is_finished()` guard
            # skips these reqs before req_id_to_index[req_id], so returning
            # EMPTY is safe.
            tail_req_ids = list(scheduler_output.num_scheduled_tokens.keys())
            if tail_req_ids:
                stale = [r for r in tail_req_ids if r not in self.requests]
                if len(stale) == len(tail_req_ids):
                    logger.error(
                        "[EDGE-TAIL-STALE-DISCARD] batch_type=%s "
                        "head_token=%s req_ids=%s all popped from "
                        "self.requests; skip segment_e.",
                        scheduler_output.batch_type,
                        scheduler_output.head_token,
                        tail_req_ids,
                    )
                    # Signal sample_tokens to also skip (return EMPTY, not None).
                    self._tail_segment_discarded = True
                    # [DP-FIX] A discarded tail runs no real forward, but in
                    # DP>1 the peer DP's step still enters the cross-DP
                    # all_reduce (_sync_metadata_across_dp) and may run a
                    # matching segment forward. Returning EMPTY here skips
                    # both -> the peer's all_reduce deadlocks waiting for a
                    # collective this DP never enters (the "edge dummy stuck
                    # at all_reduce" symptom); and if the peer is a real
                    # tail/head, its segment all_gather would then wait for a
                    # forward this DP never runs. Run a DUMMY instead (same
                    # as worker._execute_model_edge_dummy for the idle DP):
                    # _dummy_run advertises batch_type_id=0 (DUMMY) and calls
                    # _sync_metadata_across_dp, keeping the all_reduce 1:1.
                    # Via _peer_batch_type_id it then self-adapts: peer idle
                    # (peer_bt==0) -> skip the forward (sync only); peer real
                    # head/tail -> run the matching segment_a/segment_e to
                    # pair the forward collectives. The tail's real hidden
                    # was already received by worker._execute_model_edge_tail
                    # (data-plane contract preserved); _dummy_run on the edge
                    # uses intermediate_tensors=None, so it ignores that
                    # hidden and runs a pure dummy. dp_size==1 has no peer to
                    # pair with, so it keeps the original early return.
                    if self.dp_size > 1:
                        self._dummy_run(
                            num_tokens=self.decode_token_per_req,
                            uniform_decode=False,
                        )
                    return EMPTY_MODEL_RUNNER_OUTPUT
                if stale:
                    # Partial-stale: alive reqs need the step, stale reqs would
                    # crash. Cannot safely discard nor proceed. Full fix needs
                    # per-req hidden slicing -- not yet implemented.
                    logger.error(
                        "[EDGE-TAIL-PARTIAL-STALE] batch_type=%s "
                        "head_token=%s req_ids=%s stale=%s alive=%s; "
                        "NOT handled, will crash.",
                        scheduler_output.batch_type,
                        scheduler_output.head_token,
                        tail_req_ids,
                        stale,
                        [r for r in tail_req_ids if r in self.requests],
                    )
        # Save scheduler_output for edge-cloud mamba state sync in sample_tokens().
        self._last_scheduler_output = scheduler_output

        # In edge-cloud embedding_only mode, execute_model is called twice for the
        # same scheduler_output: head segment (intermediate_tensors is None) and
        # tail segment (intermediate_tensors is not None). The tail segment should
        # reuse the num_computed_tokens corrected by the head segment, instead of
        # re-running update_num_computed_tokens_for_batch_change or copying the
        # potentially optimistic CPU value back to GPU.
        self._is_edge_cloud_embed_only_tail = (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.mode == "embedding_only"
            and self.edge_cloud_cfg.role == "edge"
            and intermediate_tensors is not None
        )

        # If ngram_gpu is used, we need to copy the scheduler_output to avoid
        # the modification has influence on the scheduler_output in engine core process.
        # The replace is much faster than deepcopy.
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()
            spec_decode_tokens_copy = (
                scheduler_output.scheduled_spec_decode_tokens.copy()
            )
            scheduler_output = replace(
                scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens_copy,
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,
            )

        self._start_dump_data()
        # self._draft_token_ids is None when `input_fits_in_drafter=False`
        # and there is no draft tokens scheduled. so it need to update the
        # spec_decoding info in scheduler_output with async_scheduling.
        # use deepcopy to avoid the modification has influence on the
        # scheduler_output in engine core process.
        # TODO(Ronald1995): deepcopy is expensive when there is a large
        # number of requests, optimize it later.
        if ((
            self.use_async_scheduling and self.num_spec_tokens and self._draft_token_ids is None  # type: ignore[has-type]
        ) or (
            # NOTE: This branch specifically triggers a deepcopy during the prefill phase 
            # only for PCP (Parallel Context Processing) + Multi-Modal (MM) scenarios. 
            # It does not affect other use cases. This is a temporary workaround and 
            # will be removed once upstream vLLM provides native support for PCP + MM.
            self.pcp_size > 1 and self.supports_mm_inputs and get_pp_group().is_first_rank
            and not self.model_config.is_encoder_decoder
        )):
            scheduler_output = deepcopy(scheduler_output)
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        # ---- segment_e fast path: reuse segment_a's cached prepare results ----
        # NOTE: if an intervening EMPTY batch cleared input_batch, we must
        # fall through to the normal path so _update_states can re-add the
        # requests before sampling.
        # 2P1D: segment_e (P尾/D尾) must reuse the prepare result cached by
        # *its own* head segment (matched by head_token), not whichever
        # prefill wrote the single-slot cache last. With two prefills in
        # flight (P1首/P2首) and out-of-order PL returns, a req_ids-only check
        # can match the wrong tail to the wrong cache.
        _tail_head_token = getattr(scheduler_output, "head_token", None)
        _fast_path = (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "edge"
            and intermediate_tensors is not None
            and _tail_head_token is not None
            and _tail_head_token in self._edge_prepare_cache
            and self.input_batch.num_reqs > 0
            and tuple(self.input_batch.req_ids)
            == tuple(scheduler_output.num_scheduled_tokens)
        )
        # 2P1D: a tail that misses fast path (e.g. req_ids mismatch when two
        # prefills are in flight) re-prepares on the slow path, so its cached
        # entry is no longer needed - drop it to avoid leaking the
        # attn_metadata / device tensors it holds. head_token is unique per
        # batch, so this is hygiene, not correctness.
        if (self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "edge"
            and intermediate_tensors is not None
            and not _fast_path
            and _tail_head_token is not None):
            self._edge_prepare_cache.pop(_tail_head_token, None)
        # ---- cloud fast path: reuse pre-computed prepare results ----
        _cloud_fast_path = (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "cloud"
            and intermediate_tensors is not None
            and self._cloud_prepare_cache is not None
        )
        if _fast_path:
            cache = self._edge_prepare_cache.pop(_tail_head_token)
            # consumed: only this head_token's entry is removed; other
            # in-flight prefills' caches are preserved for 2P1D.
            total_num_scheduled_tokens = cache["total_num_scheduled_tokens"]
            num_tokens_padded = cache["num_tokens_padded"]
            num_tokens_across_dp = cache["num_tokens_across_dp"]
            attn_metadata = cache["attn_metadata"]
            logits_indices = cache["logits_indices"]
            spec_decode_metadata = cache["spec_decode_metadata"]
            spec_decode_common_attn_metadata = cache["spec_decode_common_attn_metadata"]
            cudagraph_mode = cache["cudagraph_mode"]
            batch_desc = cache["batch_desc"]
            cudagraph_stats = cache["cudagraph_stats"]
            # Re-sync num_computed_tokens from CPU: segment_a forward or
            # async state update may have modified the GPU buffer.
            # NOTE: In async speculative decoding, segment_a has already
            # corrected the GPU num_computed_tokens using the actual accepted
            # token count. Do not overwrite it with the scheduler's optimistic
            # CPU mirror, otherwise the next segment_a will use stale values
            # and positions_np can exceed max_model_len.
            num_reqs = self.input_batch.num_reqs
            if not self.use_async_spec_decode:
                self.num_computed_tokens[:num_reqs].copy_(
                    self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
                    non_blocking=True,
                )
            # Fast path skips _update_states, so no deferred corrections.
            deferred_state_corrections_fn = None
            # [方案③-fix Part 1] tail segment (PL/DL) reuses the head's cached
            # num_tokens_across_dp and skips _determine_batch_execution_and_padding,
            # so it does NOT enter the cross-DP all_reduce. But the idle DP's
            # dummy (_dummy_run) always calls _sync_metadata_across_dp. Without
            # this call, when this DP runs a tail step while the peer runs a
            # dummy, the peer's all_reduce deadlocks waiting for a tail step
            # that never calls it. Enter the collective here to keep the DP
            # all_reduce pairing 1:1; the result is ignored (the tail uses the
            # cached values above for the forward).
            self._sync_metadata_across_dp(
                num_tokens=num_tokens_padded,
                cudagraph_mode=cudagraph_mode,
            )
            logger.error(
                "[DPDBG] tail_sync_meta: cached_ntp=%s so_tnst=%s ntad=%s",
                num_tokens_padded,
                scheduler_output.total_num_scheduled_tokens,
                num_tokens_across_dp,
            )
        elif _cloud_fast_path:
            cache = self._cloud_prepare_cache
            self._cloud_prepare_cache = None  # consumed, clear for next iteration
            total_num_scheduled_tokens = cache["total_num_scheduled_tokens"]
            num_tokens_padded = cache["num_tokens_padded"]
            num_tokens_across_dp = cache["num_tokens_across_dp"]
            attn_metadata = cache["attn_metadata"]
            logits_indices = cache["logits_indices"]
            spec_decode_metadata = cache["spec_decode_metadata"]
            spec_decode_common_attn_metadata = cache["spec_decode_common_attn_metadata"]
            cudagraph_mode = cache["cudagraph_mode"]
            batch_desc = cache["batch_desc"]
            cudagraph_stats = cache["cudagraph_stats"]
            deferred_state_corrections_fn = None
        with record_function_or_nullcontext("prepare input"):
            with self.synchronize_input_prep():
                if not _fast_path and not _cloud_fast_path:
                    # Fix up prev_req_id_to_index for requests that were discarded
                    # in the previous sample_tokens step. If a request has
                    # prev_num_draft_len > 0 but is missing from
                    # prev_req_id_to_index, the parent _update_states would
                    # hit a KeyError. Reset prev_num_draft_len to 0 for such
                    # requests so they fall through safely.
                    if (
                        self.use_async_scheduling
                        and self.num_spec_tokens
                        and self.input_batch.prev_req_id_to_index is not None
                    ):
                        for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                            if (
                                req_id not in self.input_batch.prev_req_id_to_index
                                and (req_state := self.requests.get(req_id)) is not None
                                and req_state.prev_num_draft_len
                            ):
                                req_state.prev_num_draft_len = 0

                    # Edge-cloud tail segments (PL/DL) reuse the batch state
                    # established by their head segment (PF/DF).  Skipping
                    # _update_states prevents interleaving bugs when multiple
                    # prefills are in flight (2P1D) and avoids double-counting.
                    is_edge_tail_segment = (
                        self._edge_cloud_enabled
                        and is_edge_device()
                        and scheduler_output.batch_type in (
                            BatchType.PREFILL_LAST,
                            BatchType.DECODE_LAST,
                        )
                    )
                    scheduled_req_ids = tuple(scheduler_output.num_scheduled_tokens)
                    skip_update_states = (
                        is_edge_tail_segment
                        and self.input_batch.num_reqs > 0
                        and tuple(self.input_batch.req_ids) == scheduled_req_ids
                    )
                    if skip_update_states:
                        deferred_state_corrections_fn = None
                    else:
                        deferred_state_corrections_fn = self._update_states(
                            scheduler_output
                        )

                    num_reqs = self.input_batch.num_reqs
                    if num_reqs == 0:
                        # No active requests remaining (e.g. all were completed
                        # or removed by a prior _update_states call in
                        # cloud_prepare_early).  Return empty output consistent
                        # with the num_scheduled_tokens == 0 path.
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    req_ids = self.input_batch.req_ids
                    tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
                    num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
                    max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())

                    (
                        logits_indices,
                        spec_decode_metadata,
                        total_num_scheduled_tokens,
                        num_scheduled_tokens_compressed_list,
                    ) = self._prepare_inputs(
                        scheduler_output,
                        num_scheduled_tokens_np,
                    )

                    if not num_scheduled_tokens:
                        if (
                            self.parallel_config.distributed_executor_backend == "external_launcher"
                            and self.parallel_config.data_parallel_size > 1
                        ):
                            # this is a corner case when both external launcher
                            # and DP are enabled, num_scheduled_tokens could be
                            # 0, and has_unfinished_requests in the outer loop
                            # returns True. before returning early here we call
                            # dummy run to ensure coordinate_batch_across_dp
                            # is called into to avoid out of sync issues.
                            self._dummy_run(1)
                        if not has_kv_transfer_group():
                            # Return empty ModelRunnerOutput if no work to do.
                            return EMPTY_MODEL_RUNNER_OUTPUT
                        return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
                    if self.cache_config.kv_sharing_fast_prefill:
                        assert not self.num_prompt_logprobs, (
                            "--kv-sharing-fast-prefill produces incorrect "
                            "logprobs for prompt tokens, tokens, please disable "
                            "it when the requests need prompt logprobs"
                        )

                    # Apply deferred state corrections, then mamba preprocess
                    # (must run after _update_states, before input preparation).
                    if deferred_state_corrections_fn:
                        deferred_state_corrections_fn()
                        deferred_state_corrections_fn = None
                    if self.cache_config.mamba_cache_mode == "align":
                        if vllm_version_is("0.20.2"):
                            mamba_bufs = self._get_mamba_copy_bufs()
                            preprocess_bufs = mamba_bufs
                        else:
                            mamba_bufs = self._get_mamba_bufs()
                            preprocess_bufs = mamba_bufs.preprocess
                        mamba_utils.preprocess_mamba(
                            scheduler_output,
                            self.kv_cache_config,
                            self.cache_config,
                            self.mamba_state_idx,
                            self.input_batch,
                            self.requests,
                            self.compilation_config.static_forward_context,
                            self.model.get_mamba_state_copy_func(),
                            preprocess_bufs,
                        )
                        # preprocess_mamba resets num_accepted_tokens_cpu to 1
                        # for requests whose state was copied to a new block.
                        # Re-sync to GPU so the mamba kernel reads from the
                        # correct initial state slot (init_token_idx = 0).
                        self.num_accepted_tokens.np[:num_reqs] = (
                            self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                        )
                        self.num_accepted_tokens.copy_to_gpu(num_reqs)

                        if not vllm_version_is("0.20.2") and mamba_bufs.postprocess_align is not None:
                            mamba_utils.stage_postprocess_inputs_to_gpu(
                                mamba_bufs.postprocess_align,
                                scheduler_output,
                                self.input_batch.req_ids,
                                num_reqs,
                                self.requests,
                                self.mamba_state_idx,
                            )
                    if self.use_compress:
                        if deferred_state_corrections_fn:
                            deferred_state_corrections_fn()
                            deferred_state_corrections_fn = None
                        num_reqs = self.input_batch.num_reqs
                        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens_np)
                        dsa_positions_np = self._dsa_positions_np_buf[:total_num_scheduled_tokens]
                        np.add(
                            self.input_batch.num_computed_tokens_cpu[req_indices],
                            self.query_pos.np[:total_num_scheduled_tokens],
                            out=dsa_positions_np,
                        )

                    # Run core input preparation.
                    # NOTE: _prepare_inputs was already called inline above; it
                    # is NOT idempotent (rewrites num_accepted_tokens_cpu in
                    # place under async spec decode), so reuse its results here
                    # instead of letting _run_input_preparation call it again.
                    cache = self._run_input_preparation(
                        scheduler_output,
                        precomputed=(
                            logits_indices,
                            spec_decode_metadata,
                            total_num_scheduled_tokens,
                            num_scheduled_tokens_compressed_list,
                        ),
                    )
                    total_num_scheduled_tokens = cache["total_num_scheduled_tokens"]
                    num_tokens_padded = cache["num_tokens_padded"]
                    num_tokens_across_dp = cache["num_tokens_across_dp"]
                    attn_metadata = cache["attn_metadata"]
                    logits_indices = cache["logits_indices"]
                    spec_decode_metadata = cache["spec_decode_metadata"]
                    spec_decode_common_attn_metadata = cache["spec_decode_common_attn_metadata"]
                    cudagraph_mode = cache["cudagraph_mode"]
                    batch_desc = cache["batch_desc"]
                    cudagraph_stats = cache["cudagraph_stats"]

                    logger.debug(
                        "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                        "num_tokens_across_dp: %s",
                        cudagraph_mode,
                        batch_desc,
                        num_tokens_across_dp,
                    )

                # Save spec_decode_common_attn_metadata for cloud-side
                # MTP draft proposal.  On the cloud side,
                # execute_model_state is None (cloud is not the last PP
                # rank), so the metadata would otherwise be lost.
                num_reqs = self.input_batch.num_reqs
                if (
                    self._edge_cloud_enabled
                    and self.edge_cloud_cfg.role == "cloud"
                    and spec_decode_common_attn_metadata is not None
                ):
                    self._cloud_spec_decode_common_attn_metadata = spec_decode_common_attn_metadata
                    self._cloud_spec_decode_num_reqs = num_reqs

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output,
                num_tokens_padded
                if not (self.use_cp and self.pcp_manager.pcp_use_hybrid_attn)
                else total_num_scheduled_tokens,
                intermediate_tensors,
            )

            if not self.edge_cloud_cfg.role == "edge":
                # update global cos, sin
                update_cos_sin(positions)

        if self.dynamic_eplb:
            with record_function_or_nullcontext("EPLB weight D2D"):
                self.eplb_updator.forward_before()

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:  # type: ignore[has-type]
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False  # type: ignore[has-type]
        if self.ascend_config.enable_async_exponential:
            self.sampler.do_async_exponential(
                b_s=logits_indices.shape[0],
                head_dim=self.model_config.get_vocab_size(),
                generators=self.input_batch.sampling_metadata.generators,
            )

        # Cache segment_a prepare results so segment_e can skip redundant work.
        # Placed AFTER KV scales / async_exponential checks so cached cudagraph_mode
        # reflects the actual value used by the forward pass.
        # intermediate_tensors is None only for the first call (segment_a);
        # segment_e always receives cloud data via intermediate_tensors.
        if (self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "edge"
            and intermediate_tensors is None):
            _head_token = getattr(scheduler_output, "head_token", None)
            if _head_token is not None:
                self._edge_prepare_cache[_head_token] = {
                    "num_tokens_padded": num_tokens_padded,
                    "num_tokens_across_dp": num_tokens_across_dp,
                    "attn_metadata": attn_metadata,
                    "logits_indices": logits_indices,
                    "spec_decode_metadata": spec_decode_metadata,
                    "spec_decode_common_attn_metadata": spec_decode_common_attn_metadata,
                    "cudagraph_mode": cudagraph_mode,
                    "batch_desc": batch_desc,
                    "cudagraph_stats": cudagraph_stats,
                    "total_num_scheduled_tokens": total_num_scheduled_tokens,
                }
            logger.error(
                "[DPDBG] head_cache: tnst=%s ntad=%s ntp=%s",
                total_num_scheduled_tokens, num_tokens_across_dp, num_tokens_padded,
            )

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = self.model_config.is_encoder_decoder and num_encoder_reqs > 0

        # Run forward pass
        clear_kv_metadata = self.speculative_config is None

        # Save per-step state for layer slice continuation.
        if layer_slice_info is not None and not layer_slice_info.is_last_slice:
            self._layerwise_positions = positions
            # Deep-clone device tensors inside attn_metadata to prevent
            # corruption by subsequent decode batches.  The metadata's
            # device tensors (query_start_loc, state_indices, etc.) are
            # views into shared common_attn_metadata buffers that get
            # rebuilt on every batch.  If a decode batch runs between
            # two prefill slices, it overwrites these buffers in-place,
            # corrupting the saved metadata.  Cloning at save time
            # preserves the correct prefill values.
            if isinstance(attn_metadata, dict):
                self._layerwise_attn_metadata = {
                    k: _clone_gdn_attn_metadata(v) for k, v in attn_metadata.items()
                }
            elif attn_metadata is not None:
                self._layerwise_attn_metadata = _clone_gdn_attn_metadata(attn_metadata)
            else:
                self._layerwise_attn_metadata = attn_metadata
            self._layerwise_num_tokens_padded = num_tokens_padded
            self._layerwise_num_tokens_across_dp = num_tokens_across_dp
            self._layerwise_batch_desc = batch_desc
            self._layerwise_scheduler_output = scheduler_output
            self._layerwise_logits_indices = logits_indices
            self._layerwise_spec_decode_metadata = spec_decode_metadata
            self._layerwise_spec_decode_common_attn_metadata = spec_decode_common_attn_metadata
            self._layerwise_ec_connector_output = ec_connector_output
            self._layerwise_cudagraph_stats = cudagraph_stats

        with (
            record_function_or_nullcontext("forward"),
            set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                aclgraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
                model_instance=self.model,
                max_tokens_across_pcp=0 if self.pcp_size == 1 else self.pcp_manager.max_num_tokens_across_pcp,
                skip_compiled=has_encoder_input,
                has_sinks=self._has_sinks,
                input_ids=input_ids,
            ),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                **(
                    {"defer_finalize": not clear_kv_metadata}
                ),
            ) as kv_connector_output,
        ):
            hidden_states = self._model_forward(
                num_tokens_padded, input_ids, positions, intermediate_tensors,
                inputs_embeds, layer_slice_info=layer_slice_info,
                **model_kwargs
            )
        with record_function_or_nullcontext("post process"):
            aux_hidden_states = None
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = hidden_states
            if self.pcp_size > 1:
                # NOTE we must `slice` hidden_states because pcp_allgather_restore_idx
                # ignores the padding from CUDA Graph.
                hidden_states = self.pcp_manager.get_restore_hidden_states(hidden_states)
                if aux_hidden_states is not None:
                    aux_hidden_states = [
                        self.pcp_manager.get_restore_hidden_states(aux_hidden_states_pcp)
                        for aux_hidden_states_pcp in aux_hidden_states
                    ]
            
            # --- Layer slice: save intermediate state for non-last ---
            # Must happen BEFORE the edge-cloud early return below;
            # otherwise cloud non-last slices would return IntermediateTensors
            # to the worker immediately without saving continuation state,
            # causing all subsequent slices to see _layerwise_intermediate=None.
            if (
                layer_slice_info is not None
                and not layer_slice_info.is_last_slice
            ):
                # The model returns IntermediateTensors for non-last PP
                # ranks.  Detach the hidden/residual so they serve as
                # fresh inputs for the next slice's forward pass.
                assert isinstance(hidden_states, IntermediateTensors)
                self._layerwise_intermediate = hidden_states
                if self.debugger is not None:
                    self.debugger.stop()
                    self.debugger.step()
                return None
            # --- End layerwise chunk save ---

            # 边云场景：当 hidden_states 为 IntermediateTensors 时，说明当前段计算已完成，
            # 需要把结果返回给 NPUWorker 进行跨节点通信（isend_tensor_dict）。
            # 此处提前 return，跳过标准 PP 的 logits/sampling 流程。

            if is_edge_cloud_pp_mode() and isinstance(hidden_states, IntermediateTensors) and not is_edge_device():
                hidden_states.kv_connector_output = kv_connector_output
                self.kv_connector_output = kv_connector_output
                if self.debugger is not None:
                    self.debugger.stop()
                    self.debugger.step()
                return hidden_states

            if not self.broadcast_pp_output:
                # Common case.
                if self._edge_cloud_enabled and isinstance(
                    hidden_states, IntermediateTensors
                ):
                    # Edge-cloud head segment always returns IntermediateTensors,
                    # regardless of is_last_rank, so the worker can send them to
                    # the cloud side and receive results back for the tail segment.
                    # For embedding_only edge, the output tensors have actual
                    # batch size (no cudagraph padding on edge). The cloud-side
                    # sync_and_slice_intermediate_tensors copies the received
                    # prefix and zero-fills the padding locally, so no extra pad
                    # is required here.
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    self._finalize_dump_data()
                    self.suspend_head_state(scheduler_output)
                    return hidden_states
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    self._finalize_dump_data()
                    return hidden_states
                if self.is_pooling_model:
                    # Return the pooling output.
                    output = self._pool(
                        hidden_states, num_scheduled_tokens, num_scheduled_tokens_np, kv_connector_output
                    )
                    output.kv_connector_output = kv_connector_output
                    self._finalize_dump_data()
                    return output

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.
                assert not self.is_pooling_model

                if not get_pp_group().is_last_rank:
                    sample_hidden_states = hidden_states[logits_indices]
                    get_pp_group().send_tensor_dict(hidden_states.tensors, all_gather_group=get_tp_group())
                    logits = None
                else:
                    sample_hidden_states = hidden_states[logits_indices]
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()
                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

            # Apply structured output bitmasks if present
            self.execute_model_state = ExecuteModelState(
                scheduler_output,
                logits,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                attn_metadata,
                positions,
                ec_connector_output,
                cudagraph_stats,
                batch_desc,
            )
            self.kv_connector_output = kv_connector_output

        # Now the batch has been launched we can wait for corrections from the
        # previous model forward without breaking async scheduling.
        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()
        return None

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        if self._edge_cloud_enabled and not self.parallel_config.is_edge_node:
            # Cloud workers do not own segment_e / LM head / sampler in the
            # edge-cloud PD-separation topology. When the edge EngineCore
            # issues sample_tokens via collective_rpc, every worker dequeues
            # the request, but only the edge (rank 0) actually samples and
            # writes back to the executor's reply mq. Cloud workers must
            # return a no-op output immediately so the broadcast protocol
            # converges and no PP/HCCL primitive is touched here.
            self.execute_model_state = None
            self.kv_connector_output = None
            return EMPTY_MODEL_RUNNER_OUTPUT

        # [PD-FIX] execute_model discarded a stale tail segment (all reqs
        # already popped). Skip sampling and return EMPTY so update_from_output's
        # is_finished()/None guard skips those reqs (returning None here would
        # trigger "unexpected error" in _patched_step_with_batch_queue).
        if self._tail_segment_discarded:
            self._tail_segment_discarded = False
            self.execute_model_state = None
            self.kv_connector_output = None
            return EMPTY_MODEL_RUNNER_OUTPUT

        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            # receive sampled token ids from the last PP rank when using
            # async scheduling + pipeline parallelism so downstream code
            # (e.g., PCP input preparation) can access them.
            if self.use_async_scheduling and get_pp_group().world_size > 1:
                if not self._edge_cloud_enabled:
                    self._pp_receive_prev_sampled_token_ids_to_input_batch()
            if not kv_connector_output:
                if (
                    self._edge_cloud_enabled
                    and self.edge_cloud_cfg.role == "cloud"
                    and self.speculative_config
                    and self.speculative_config.method == "mtp"
                    and hasattr(self, "_edge_cloud_mtp_segments")
                    and "c" in self._edge_cloud_mtp_segments
                ):
                    self._run_mtp_cloud_segment()

                # Edge-cloud sync: receive num_accepted_tokens (and optionally
                # valid_sampled_token_count) from edge so that cloud can update
                # num_computed_tokens for the main model in embed_only MTP mode,
                # or run _update_states_after_model_execute() for hybrid models.
                if (
                    self._edge_cloud_enabled
                    and self.edge_cloud_cfg.role == "cloud"
                    and self.speculative_config
                    and (
                        self.model_config.is_hybrid
                        or (
                            self.edge_cloud_cfg.mode == "embedding_only"
                            and self.speculative_config.method == "mtp"
                        )
                    )
                    and self._last_scheduler_output is not None
                ):
                    pp_group = get_pp_group()
                    if pp_group.world_size == 2:
                        tensor_dict, recv_handles, recv_postprocess = (
                            pp_group.irecv_tensor_dict()
                        )
                        for handle in recv_handles:
                            handle.wait()
                        for postprocess in recv_postprocess:
                            postprocess()
                    else:
                        tensor_dict = None

                    tensor_dict = get_tp_group().broadcast_object(
                        tensor_dict, src=0
                    )
                    assert tensor_dict is not None
                    num_accepted = tensor_dict["num_accepted_tokens"].to(self.device)
                    num_reqs = num_accepted.size(0)
                    self.num_accepted_tokens.gpu[:num_reqs] = num_accepted

                    # For embed_only MTP, the cloud also needs the rejection-
                    # corrected valid_sampled_token_count and the prev-batch
                    # request mapping so that _prepare_inputs can run the async
                    # spec-decode correction kernel.
                    if (
                        self.edge_cloud_cfg.mode == "embedding_only"
                        and self.speculative_config.method == "mtp"
                        and "valid_sampled_token_count" in tensor_dict
                    ):
                        self.valid_sampled_token_count_gpu = tensor_dict[
                            "valid_sampled_token_count"
                        ].to(self.device)
                        # _bookkeeping_sync is not run on the cloud, so the
                        # prev-batch mapping must be reconstructed here.
                        self.input_batch.prev_req_id_to_index = {
                            req_id: i
                            for i, req_id in enumerate(self.input_batch.req_ids)
                        }

                    if self.model_config.is_hybrid:
                        if self.cache_config.mamba_cache_mode == "align":
                            for i, num_tokens in enumerate(
                                self.num_accepted_tokens.gpu[:num_reqs].cpu().numpy()
                            ):
                                self.input_batch.num_accepted_tokens_cpu[i] = num_tokens
                            mamba_utils.postprocess_mamba(
                                self._last_scheduler_output,
                                self.kv_cache_config,
                                self.cache_config,
                                self.input_batch,
                                self.requests,
                                self.mamba_state_idx,
                                self.compilation_config.static_forward_context,
                                self.model.get_mamba_state_copy_func(),
                                self._get_mamba_copy_bufs(),
                            )
                        else:
                            self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
                                self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True
                            )
                    else:
                        # For non-hybrid embed_only MTP, keep CPU mirror in sync
                        # so _prepare_inputs sees corrected num_accepted_tokens.
                        self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
                            self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True
                        )

                return None  # noqa
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            attn_metadata,
            positions,
            ec_connector_output,
            cudagraph_stats,
            batch_desc,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            # here we are different from gpu_model_runner,
            # the apply_grammar_bitmask uses torch.compile to optimize this,ascend does not support it now
            logits_dtype = logits.dtype
            logits = logits.to("cpu").float()
            apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)
            logits = logits.to(self.device).to(logits_dtype)

        with record_function_or_nullcontext("sample_token"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        if self.need_accepted_tokens:
            if self.sampling_done_event is None:
                self.sampling_done_event = torch.npu.Event()

            assert self.sampling_done_event is not None
            self.sampling_done_event.record()

        self.valid_sampled_token_count_gpu: torch.Tensor | None = None # type: ignore[no-redef]

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            self._draft_token_ids = self.propose_draft_token_ids(
                sampled_token_ids,
                self.input_batch.sampling_metadata,
                scheduler_output,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
                positions,
                scheduler_output.total_num_scheduled_tokens,
                hidden_states,
                aux_hidden_states,
                sample_hidden_states,
                batch_desc,
            )
            self._copy_draft_token_ids_to_cpu(scheduler_output)

        (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
            spec_decode_metadata,
        )

        with record_function_or_nullcontext("draft_token"):
            if self.speculative_config:
                use_padded_batch = (
                    self.speculative_config
                    and (
                        self.speculative_config.use_eagle()
                        or self.speculative_config.uses_draft_model()
                        or self.speculative_config.uses_extract_hidden_states()
                        or self.speculative_config.use_ngram_gpu()
                    )
                    and not self.speculative_config.disable_padded_drafter_batch
                )
                if use_padded_batch:
                    # EAGLE speculative decoding can use the GPU sampled tokens
                    # as inputs, and does not need to wait for bookkeeping to finish.
                    propose_draft_token_ids(sampler_output.sampled_token_ids)
                if self.speculative_config and not use_padded_batch:
                    # ngram and other speculative decoding methods use the sampled
                    # tokens on the CPU, so they are run after bookkeeping.
                    propose_draft_token_ids(valid_sampled_token_ids)

            # Edge-cloud sync: send num_accepted_tokens (and optionally
            # valid_sampled_token_count) to cloud so that cloud can run
            # _update_states_after_model_execute() for hybrid models, or correct
            # num_computed_tokens for embed_only MTP speculative decoding.
            if (
                self._edge_cloud_enabled
                and self.edge_cloud_cfg.role != "cloud"
                and self.speculative_config
                and (
                    self.model_config.is_hybrid
                    or (
                        self.edge_cloud_cfg.mode == "embedding_only"
                        and self.speculative_config.method == "mtp"
                    )
                )
            ):
                num_reqs = sampler_output.sampled_token_ids.size(0)
                num_accepted = (sampler_output.sampled_token_ids != -1).sum(dim=1).cpu()
                tensor_dict_to_send = {"num_accepted_tokens": num_accepted}
                if (
                    self.edge_cloud_cfg.mode == "embedding_only"
                    and self.speculative_config.method == "mtp"
                    and self.valid_sampled_token_count_gpu is not None
                ):
                    tensor_dict_to_send["valid_sampled_token_count"] = (
                        self.valid_sampled_token_count_gpu.cpu()
                    )
                if get_pp_group().world_size == 2:
                    send_work = get_pp_group().isend_tensor_dict(tensor_dict_to_send)
                    for handle in send_work:
                        handle.wait()

            # vLLM v0.18 defers KV connector finalization during target-model
            # forward when speculative decoding is enabled. Finalize here after
            # draft model runs so KV pool save/put can complete.
            if self.speculative_config is not None:
                self.finalize_kv_connector()

        routed_experts_lists = None
        if self.model_config.enable_return_routed_experts:
            if vllm_version_is("0.20.2"):
                capturer = RoutedExpertsCapturer.get_instance()
                if capturer is not None:
                    capturer.save_captured_experts(indices=self.cpu_slot_mapping)
            elif self.routed_experts_initialized:
                buf = self.routed_experts_capturer.get_device_buffer()
                total = scheduler_output.total_num_scheduled_tokens
                self.routed_experts_cpu[:total].copy_(buf[:total], non_blocking=True)
                self.routed_experts_slot_mapping_cpu[:total].copy_(
                    self.routed_experts_slot_mapping_device[:total],
                    non_blocking=True,
                )
                torch.npu.current_stream().synchronize()
                routed_experts_lists = RoutedExpertsLists(
                    routing_data=self.routed_experts_cpu[:total].numpy(),
                    slot_mapping=self.routed_experts_slot_mapping_cpu[:total].numpy(),
                )

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            kv_connector_output=kv_connector_output,
            pooler_output=[],
            ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
            cudagraph_stats=cudagraph_stats,
            **(
                {} if vllm_version_is("0.20.2") else {"routed_experts": routed_experts_lists}
            ),
        )
        if self.ascend_config.profiling_chunk_config.need_timing and hasattr(self, '_execution_start_time'):
            self._sync_device()
            model_runner_output.execution_time_ms = (time.perf_counter() - self._execution_start_time) * 1000.0

        if self.dynamic_eplb:
            with record_function_or_nullcontext("EPLB update"):
                self.eplb_updator.forward_end()

        self._finalize_dump_data()

        if self.need_accepted_tokens:
            assert self.sampling_done_event is not None
            with (
                record_function_or_nullcontext("async_state_update"),
                torch.npu.stream(global_stream()),
            ):
                global_stream().wait_event(self.sampling_done_event)
                self._update_states_after_model_execute(sampler_output.sampled_token_ids, scheduler_output)

        # In async scheduling + PP, broadcast sampled token ids from the
        # last PP rank so other PP ranks can receive them without going
        # through the scheduler/engine IPC path.
        if self.use_async_scheduling:
            pp = get_pp_group()
            if not self._edge_cloud_enabled and pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(sampler_output.sampled_token_ids)

        if not self.use_async_scheduling:
            return model_runner_output
        async_output = AsyncGPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
        )
        self.input_batch.set_async_sampled_token_ids(
            async_output.sampled_token_ids_cpu,
            async_output.async_copy_ready_event,
        )
        return async_output

    def _build_mtp_cloud_attn_metadata(
        self,
        positions: torch.Tensor,
        spec_step_idx: int,
    ) -> dict[str, Any] | None:
        """Build per-layer attention metadata for the MTP cloud decoder.

        On the cloud side, the MTP decoder layers are real (not
        PPMissingLayer) and need proper attention metadata to produce
        correct outputs.  Without it, the Ascend attention backend
        silently returns zeros, corrupting hidden states and causing
        low draft hit rates.

        Uses the spec_decode_common_attn_metadata saved during
        execute_model() and the drafter's draft_attn_groups to build
        per-layer metadata for each speculative step.
        """
        if (
            not hasattr(self, "_cloud_spec_decode_common_attn_metadata")
            or self._cloud_spec_decode_common_attn_metadata is None
        ):
            return None

        if (
            self.drafter is None
            or not hasattr(self.drafter, "draft_attn_groups")
            or not self.drafter.draft_attn_groups
        ):
            return None

        common_attn_metadata = self._cloud_spec_decode_common_attn_metadata
        num_reqs = getattr(self, "_cloud_spec_decode_num_reqs", 0)

        # Adapt common_attn_metadata for draft model positions.
        # The positions come from the edge side and reflect the draft
        # model's token positions for the current speculative step.
        common_attn_metadata = self.drafter.shallow_copy_metadata(
            common_attn_metadata
        )
        common_attn_metadata.positions = positions

        # Compute batch_size: each decode request contributes one
        # draft token per step.
        batch_size = num_reqs

        # Use the actual number of tokens carried by positions,
        # which already accounts for rejected tokens on the edge side.
        num_input_tokens = positions.shape[-1]
        num_actual_tokens = num_input_tokens
        common_attn_metadata.num_actual_tokens = num_actual_tokens
        common_attn_metadata.num_input_tokens = num_input_tokens

        if spec_step_idx > 0:
            # For steps after the first, each request has exactly one
            # query token and the sequence length has grown by
            # spec_step_idx compared to the target model.
            common_attn_metadata.max_query_len = 1
            common_attn_metadata.decode_token_per_req = 1

            # Increment seq_lens to account for previously accepted
            # draft tokens.  The target model's seq_lens already
            # includes one accepted token; each additional draft step
            # adds one more.
            common_attn_metadata.seq_lens = common_attn_metadata.seq_lens.clone()
            common_attn_metadata.seq_lens[:batch_size] += spec_step_idx
            if common_attn_metadata.seq_lens_cpu is not None:
                common_attn_metadata.seq_lens_cpu = (
                    common_attn_metadata.seq_lens_cpu.clone()
                )
                common_attn_metadata.seq_lens_cpu[:batch_size] += spec_step_idx
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu = (
                    common_attn_metadata._seq_lens_cpu.clone()
                )
                common_attn_metadata._seq_lens_cpu[:batch_size] += spec_step_idx
            if common_attn_metadata.num_computed_tokens_cpu is not None:
                common_attn_metadata.num_computed_tokens_cpu = (
                    common_attn_metadata.num_computed_tokens_cpu.clone()
                )
                common_attn_metadata.num_computed_tokens_cpu[:batch_size] += (
                    spec_step_idx
                )

            # Subsequent speculative steps are always decode-only.
            common_attn_metadata.attn_state = AscendAttentionState.SpecDecoding

            # Edge sends the SAME (first-pass) snapshot of
            # slot_mapping / query_start_loc / num_actual_tokens /
            # actual_seq_lengths_q for every speculative step.  From
            # the second step onward the cloud must rebuild them to
            # reflect a decode-only batch where each request owns
            # exactly one query token.  Without this rebuild the
            # attention backend reads stale offsets, picks the wrong
            # KV slot and num_decode_tokens (derived inside
            # split_decodes_and_prefills) ends up matching the
            # first-pass num_actual_tokens instead of batch_size.
            device = common_attn_metadata.seq_lens.device
            new_query_start_loc_cpu = torch.arange(
                batch_size + 1, dtype=torch.int32, device="cpu"
            )
            common_attn_metadata.query_start_loc_cpu = new_query_start_loc_cpu
            common_attn_metadata.query_start_loc = new_query_start_loc_cpu.to(
                device, non_blocking=True
            )
            common_attn_metadata.num_actual_tokens = batch_size
            common_attn_metadata.num_input_tokens = num_input_tokens
            common_attn_metadata.actual_seq_lengths_q = list(
                range(1, batch_size + 1)
            )

            # Recompute slot_mapping from the freshly received positions
            # and the (still valid) block_table.  Each decode token maps
            # to position // block_size -> slot offset within the block.
            block_table_tensor = common_attn_metadata.block_table_tensor
            if (
                block_table_tensor is not None
                and positions is not None
                and self.drafter is not None
                and hasattr(self.drafter, "kernel_block_size")
            ):
                block_size = self.drafter.kernel_block_size
                pos_flat = positions if positions.dim() == 1 else positions[0]
                pos_flat = pos_flat[:batch_size]
                exceeds = pos_flat >= self.model_config.max_model_len
                clamped = torch.where(exceeds, torch.zeros_like(pos_flat), pos_flat)
                block_numbers = clamped // block_size
                block_ids = block_table_tensor[:batch_size].gather(
                    dim=1, index=block_numbers.view(-1, 1).long()
                ).view(-1)
                new_slot_mapping = (
                    block_ids * block_size + clamped % block_size
                ).to(torch.int32)
                new_slot_mapping.masked_fill_(exceeds, PADDING_SLOT_ID)
                common_attn_metadata.slot_mapping = new_slot_mapping
        else:
            # For the first speculative step, preserve the original attn_state
            # from the target model's forward pass (e.g. PrefillNoCache during
            # the prefill phase, SpecDecoding during decode).  Overwriting it
            # with SpecDecoding unconditionally causes the cloud-side draft
            # model to read from an uninitialized KV cache, which corrupts
            # hidden states and leads to 100% draft-hit dead loops.
            if common_attn_metadata.attn_state is None:
                common_attn_metadata.attn_state = AscendAttentionState.SpecDecoding

        # Build per-layer attention metadata using draft_attn_groups.
        per_layer_attn_metadata: dict[str, Any] = {}
        for attn_group in self.drafter.draft_attn_groups:
            builder = attn_group.get_metadata_builder()
            if spec_step_idx == 0:
                attn_meta = builder.build(
                    0, common_attn_metadata
                )
            else:
                attn_meta = builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=spec_step_idx,
                )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_meta

        return per_layer_attn_metadata

    def _run_mtp_cloud_segment(self) -> None:
        from vllm_ascend.distributed.parallel_state import (
            edge_cloud_broadcast_recv_mtp,
        )

        # The edge side calls the MTP model for each speculative step
        # (including the first pass).  We loop the same number of times so
        # that every edge request has a matching cloud response.
        num_steps = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config else 1
        )

        for _ in range(num_steps):
            # Receive intermediate from edge (including positions and spec_step_idx)
            tensor_dict, comm_handles, comm_postprocess = (
                edge_cloud_broadcast_recv_mtp()
            )
            for handle in comm_handles:
                handle.wait()
            for postprocess in comm_postprocess:
                postprocess()
            intermediate = IntermediateTensors(tensor_dict)

            # Build kwargs for cloud segment
            positions = intermediate.tensors.get("positions", None)
            num_tokens = positions.shape[-1] if positions is not None else 0

            # Copy received tensors into persistent buffers so that the
            # ACLGraphWrapper-wrapped segment_c sees stable input addresses.
            intermediate = self._sync_edge_cloud_mtp_intermediate_tensors(
                num_tokens, intermediate
            )

            model_kwargs = {
                "intermediate_tensors": intermediate,
                "positions": positions,
            }
            spec_step_idx = 0
            if "spec_step_idx" in tensor_dict:
                spec_step_idx = tensor_dict["spec_step_idx"].item()
                model_kwargs["spec_step_idx"] = spec_step_idx

            # Build attention metadata for the MTP decoder layers.
            # Without this, the Ascend attention backend silently
            # returns zeros, corrupting hidden states.
            draft_attn_metadata = self._build_mtp_cloud_attn_metadata(
                positions, spec_step_idx
            )

            # Run cloud segment (all MTP decoder layers are on cloud)
            segment = self._edge_cloud_mtp_segments["c"]
            num_tokens = positions.shape[-1] if positions is not None else 0

            # Determine cudagraph runtime mode for the MTP cloud segment so
            # that ACLGraphWrapper can replay a captured graph during decode.
            cudagraph_runtime_mode = CUDAGraphMode.NONE
            batch_descriptor = BatchDescriptor(num_tokens)
            # if (
            #     self.edge_cloud_cfg.enable_decode_graph
            #     and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
            #     and num_tokens > 0
            # ):
            #     cudagraph_runtime_mode, batch_descriptor = (
            #         self.cudagraph_dispatcher.dispatch(
            #             num_tokens=num_tokens,
            #             uniform_decode=True,
            #             has_lora=False,
            #         )
            #     )

            with set_ascend_forward_context(
                attn_metadata=draft_attn_metadata,
                vllm_config=self.vllm_config,
                num_tokens=num_tokens,
                num_actual_tokens=num_tokens,
                batch_descriptor=batch_descriptor,
                aclgraph_runtime_mode=cudagraph_runtime_mode,
                is_draft_model=True,
            ):
                output = segment(**model_kwargs)
            assert isinstance(output, IntermediateTensors)

            # Send back to edge
            if get_pp_group().world_size == 2:
                send_work = get_pp_group().isend_tensor_dict(
                    {k: v.contiguous() if isinstance(v, torch.Tensor) else v
                     for k, v in output.items()}
                )
                for handle in send_work:
                    handle.wait()

    # overwrite _sample for lmhead_tp_enable and need_accepted_tokens
    def _sample(self, logits, spec_decode_metadata):
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        self.input_batch.update_async_output_token_ids()
        if spec_decode_metadata is None:
            if lmhead_tp_enable() and logits is not None:
                logits = logits[: self.input_batch.num_reqs]
            if self.input_batch.sampling_metadata.top_k is not None and get_ascend_config().enable_reduce_sample:
                max_topk = self.input_batch.top_k_cpu[self.input_batch.top_k_cpu < logits.shape[1]].max()
                self.sampler.prepare_sampling(max_topk)
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        if lmhead_tp_enable() and logits is not None:
            logits = logits[: len(spec_decode_metadata.logits_indices)]
        if self.input_batch.sampling_metadata.top_k is not None and get_ascend_config().enable_reduce_sample:
            max_topk = self.input_batch.top_k_cpu[self.input_batch.top_k_cpu < logits.shape[1]].max()
            self.rejection_sampler.prepare_sampling(max_topk)
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        return sampler_output

    # TODO: remove this func after eagle_proposer is refactored and
    #  _bookkeeping_sync is moved after propose_draft_token_ids
    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> tuple[
        LogprobsLists | None,
        list[list[int]],
        dict[str, LogprobsTensors | None],
        list[str],
        dict[str, int],
        list[int],
    ]:
        # TODO: implement PR 28597 from vllm
        discard_sampled_tokens_req_indices = self.discard_request_indices.np[: self.num_discarded_requests]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices = []
        logprobs_lists = None
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()
                if logprobs_tensors is not None:
                    logprobs_lists = logprobs_tensors.tolists()
            else:
                # Includes spec decode tokens.
                # parse_output returns (list[list[int]], LogprobsLists | None)
                valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    logprobs_tensors=logprobs_tensors,
                )
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            if self.num_spec_tokens <= 0:
                assert sampled_token_ids.shape[-1] == 1
                # Cache the sampled tokens on the NPU and avoid CPU sync.
                # These will be copied into input_ids in the next step
                # when preparing inputs.
                self.input_batch.prev_sampled_token_ids = sampled_token_ids

            self.input_batch.prev_req_id_to_index = {
                req_id: i for i, req_id in enumerate(self.input_batch.req_ids) if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # logprobs_lists is already set above:
        # - max_gen_len == 1: logprobs_tensors.tolists() (no cu_num_tokens)
        # - max_gen_len > 1: from RejectionSampler.parse_output() (filtered
        #   with cu_num_generated_tokens already set)

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        return (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    # all-gather one hidden-states in sp scene
    @staticmethod
    def _all_gather_hidden_states(hidden_states):
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        pad_size = get_forward_context().pad_size
        if pad_size > 0:
            hidden_states = hidden_states[:-pad_size, :]

        return hidden_states

    # all-gather a list of hidden-states in sp scene
    @staticmethod
    def _all_gather_hidden_states_list(hidden_states_list):
        return [NPUModelRunner._all_gather_hidden_states(hidden_states) for hidden_states in hidden_states_list]

    # all-gather hidden-states in last layer with aux-hidden-states in sp scene
    @staticmethod
    def _all_gather_hidden_states_and_aux(hidden_states):
        if isinstance(hidden_states, tuple):
            return (
                NPUModelRunner._all_gather_hidden_states(hidden_states[0]),
                NPUModelRunner._all_gather_hidden_states_list(hidden_states[1]),
            )
        return NPUModelRunner._all_gather_hidden_states(hidden_states)

    def _update_full_graph_params_if_needed(
        self,
        forward_context: ForwardContext,
        num_tokens_padded: int,
        positions: torch.Tensor | None,
        layer_indices: list[int] | None = None,
        graph_wrapper: ACLGraphWrapper | None = None,
    ) -> None:
        """更新 ACL 全图参数（仅在 FULL 图模式下生效）。

        边云模式下每个 segment wrapper 持有独立 GraphParams，避免不同
        segment 的 attention task handle / event / params 共用同一全局列表。
        """
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
        if hasattr(cudagraph_runtime_mode, "decode_mode"):
            cudagraph_runtime_mode = cudagraph_runtime_mode.decode_mode()
        if (
            cudagraph_runtime_mode == CUDAGraphMode.FULL
            and not forward_context.capturing
            and not _monitor.cudagraph_capturing_enabled
            and not self.use_sparse
        ):
            assert positions is not None
            if graph_wrapper is not None:
                assert graph_wrapper.graph_params is not None
            # Edge-cloud segments may contain mixed DSA+FIA layers.
            # DSA layers do not append entries to graph_params during capture,
            # so update_graph_params' zip over attn_keys vs attn_params would
            # misalign if DSA keys are present. Filter them out temporarily.
            # 使用显式字段 skip_graph_params_update 替代 hasattr 属性嗅探，
            # 避免后端字段命名变化导致误判
            original_attn_metadata = forward_context.attn_metadata
            if original_attn_metadata:
                filtered_metadata = {
                    k: v for k, v in original_attn_metadata.items()
                    if not getattr(v, 'skip_graph_params_update', False)
                }
                if len(filtered_metadata) != len(original_attn_metadata):
                    forward_context.attn_metadata = filtered_metadata
            try:
                update_full_graph_params(
                    self.attn_backend,
                    self.update_stream,
                    forward_context,
                    num_tokens_padded,
                    self.vllm_config,
                    self.speculative_config,
                    positions.shape[0],
                    layer_indices=layer_indices,
                    graph_params=graph_wrapper.graph_params if graph_wrapper is not None else None,
                    draft_graph_params=(
                        graph_wrapper.draft_graph_params if graph_wrapper is not None else None
                    ),
                    unfiltered_attn_metadata=original_attn_metadata,
                )
            finally:
                forward_context.attn_metadata = original_attn_metadata

    def _model_forward(
        self,
        num_tokens_padded: int,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ):
        """模型前向入口。标准路径与边云路径完全分离，职责单一。"""

        if self._edge_cloud_enabled:
            return self._edge_cloud_forward(
                num_tokens_padded, input_ids, positions, intermediate_tensors,
                inputs_embeds, **model_kwargs,
            )

        # ==================== 标准非边云路径（原逻辑完全保留，不做任何修改） ====================
        assert self.model is not None
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
        forward_context = get_forward_context()
        assert forward_context is not None
        if (
            forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and not forward_context.capturing
            and not _monitor.cudagraph_capturing_enabled
            and not self.use_sparse and not self.use_compress
        ):
            if self.enable_enpu:
                torch.npu.current_stream().synchronize()

            assert positions is not None
            update_full_graph_params(
                self.attn_backend,
                self.update_stream,
                forward_context,
                num_tokens_padded,
                self.vllm_config,
                self.speculative_config,
                positions.shape[0],
            )
        if get_forward_context().flash_comm_v1_enabled and not isinstance(hidden_states, IntermediateTensors):
            hidden_states = self._all_gather_hidden_states_and_aux(hidden_states)
        return hidden_states

    def _run_input_preparation(
        self,
        scheduler_output: "SchedulerOutput",
        precomputed: tuple | None = None,
    ) -> dict[str, Any]:
        """Run input preparation pipeline after _update_states.

        Executes _prepare_inputs, _determine_batch_execution_and_padding,
        and _build_attention_metadata. Returns all results as a dict that
        can be passed to the forward pass or cached for fast-path reuse.

        ``_prepare_inputs`` is NOT idempotent: under async spec decode it
        rewrites ``num_accepted_tokens_cpu`` in place by applying the
        previous-step index permutation, so calling it twice corrupts the
        per-request accepted-token counts (leading to wrong positions and a
        drop in MTP acceptance rate). When the caller (execute_model slow
        path) has already run ``_prepare_inputs`` inline, it passes the
        results via ``precomputed`` so we reuse them instead of re-running.
        ``cloud_prepare_early`` has no prior inline call and passes
        ``precomputed=None`` so we run it here exactly once.
        """
        num_reqs = self.input_batch.num_reqs
        # Guard against empty batch after _update_states
        # (e.g. cloud_prepare_early may have cleared all requests).
        if num_reqs == 0:
            return {
                "total_num_scheduled_tokens": 0,
                "num_tokens_padded": 0,
                "num_tokens_across_dp": None,
                "attn_metadata": None,
                "logits_indices": None,
                "spec_decode_metadata": None,
                "spec_decode_common_attn_metadata": None,
                "cudagraph_mode": CUDAGraphMode.NONE,
                "batch_desc": None,
                "cudagraph_stats": None,
                "num_scheduled_tokens_compressed_list": None,
            }
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())

        if precomputed is not None:
            (
                logits_indices,
                spec_decode_metadata,
                total_num_scheduled_tokens,
                num_scheduled_tokens_compressed_list,
            ) = precomputed
        else:
            (
                logits_indices,
                spec_decode_metadata,
                total_num_scheduled_tokens,
                num_scheduled_tokens_compressed_list,
            ) = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

        num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
        if self.pcp_size > 1:
            num_tokens_unpadded = self.pcp_manager.total_num_sampled_tokens_pcp

        cascade_attn_prefix_lens = None
        if self.cascade_attn_enabled and not self.parallel_config.enable_dbo:
            cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                num_scheduled_tokens_np,
                self.input_batch.num_computed_tokens_cpu[:num_reqs],
                scheduler_output.num_common_prefix_blocks,
            )

        (
            cudagraph_mode,
            batch_desc,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        ) = self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            use_cascade_attn=cascade_attn_prefix_lens is not None,
            force_eager=self.model_config.enforce_eager,
            num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
        )
        
        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )

        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens_np,
            num_tokens_padded,
            num_reqs_padded,
            self.parallel_config.num_ubatches,
        )

        pad_attn = cudagraph_mode == CUDAGraphMode.FULL
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

        if (
            cudagraph_mode == CUDAGraphMode.FULL
            or (enable_sp() and not self.model_config.use_mla)
            and self.pcp_size * self.dcp_size == 1
        ):
            num_reqs_padded = self._pad_query_start_loc_for_fia(
                num_tokens_padded,
                num_reqs_padded,
                num_reqs,
                cudagraph_mode,
                batch_desc.num_reqs,
            )

        (attn_metadata, spec_decode_common_attn_metadata) = (
            self._build_attention_metadata(
                num_tokens=(
                    num_tokens_unpadded
                    if not (self.use_cp and self.pcp_manager.pcp_use_hybrid_attn)
                    else total_num_scheduled_tokens
                ),
                num_tokens_padded=num_tokens_padded,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded,
                max_query_len=max_num_scheduled_tokens,
                ubatch_slices=ubatch_slices_attn,
                logits_indices=logits_indices,
                use_spec_decode=use_spec_decode,
                num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                num_scheduled_tokens_compressed_list=num_scheduled_tokens_compressed_list,
            )
        )

        return {
            "total_num_scheduled_tokens": total_num_scheduled_tokens,
            "num_tokens_padded": num_tokens_padded,
            "num_tokens_across_dp": num_tokens_across_dp,
            "attn_metadata": attn_metadata,
            "logits_indices": logits_indices,
            "spec_decode_metadata": spec_decode_metadata,
            "spec_decode_common_attn_metadata": spec_decode_common_attn_metadata,
            "cudagraph_mode": cudagraph_mode,
            "batch_desc": batch_desc,
            "cudagraph_stats": cudagraph_stats,
            "num_scheduled_tokens_compressed_list": num_scheduled_tokens_compressed_list,
        }

    def cloud_prepare_early(self, scheduler_output: "SchedulerOutput") -> None:
        """Pre-compute input preparation on cloud while edge runs segment_a.

        Caches results in self._cloud_prepare_cache so that when edge data
        arrives, execute_model can skip _update_states, _prepare_inputs,
        _determine_batch_execution_and_padding, and _build_attention_metadata,
        going directly to _preprocess (sync_and_gather) + _model_forward.
        """
        assert self._edge_cloud_enabled, (
            "cloud_prepare_early should only be called in edge-cloud mode"
        )
        assert self.edge_cloud_cfg.role == "cloud", (
            "cloud_prepare_early should only be called on cloud side"
        )

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if not num_scheduled_tokens:
            self._cloud_prepare_cache = None
            return

        # PD-separation: set _dp_batch_type_id BEFORE _run_input_preparation
        # (which calls _determine_batch_execution_and_padding -> sync_metadata
        # all_reduce), so the cloud real forward reports its real batch_type
        # (1=PF/2=PL/3=DF/4=DL) to the peer. Without this, sync_metadata runs
        # before execute_model sets _dp_batch_type_id (line ~2416), so it uses
        # a stale value (0 from the last DUMMY) -> the peer cloud sees
        # peer_bt=0, treats this as DUMMY and skips the all-toall, while this
        # cloud runs the real all-toall -> cross-DP all-toall mismatch ->
        # deadlock. (Mirror of execute_model's _bt_map at line ~2411.)
        _bt_map = {
            BatchType.EMPTY: 0, BatchType.PREFILL_FIRST: 1,
            BatchType.PREFILL_LAST: 2, BatchType.DECODE_FIRST: 3,
            BatchType.DECODE_LAST: 4,
        }
        self._dp_batch_type_id = _bt_map.get(scheduler_output.batch_type, 0)

        # Replicate scheduler_output handling from execute_model
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            num_scheduled_tokens_copy = (
                scheduler_output.num_scheduled_tokens.copy()
            )
            spec_decode_tokens_copy = (
                scheduler_output.scheduled_spec_decode_tokens.copy()
            )
            scheduler_output = replace(
                scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens_copy,
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,
            )

        if (
            (
                self.use_async_scheduling
                and self.num_spec_tokens
                and self._draft_token_ids is None
            )
            or (
                self.pcp_size > 1
                and self.supports_mm_inputs
                and get_pp_group().is_first_rank
                and not self.model_config.is_encoder_decoder
            )
        ):
            scheduler_output = deepcopy(scheduler_output)

        # Fix up prev_req_id_to_index (same as execute_model)
        if (
            self.use_async_scheduling
            and self.num_spec_tokens
            and self.input_batch.prev_req_id_to_index is not None
        ):
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if (
                    req_id not in self.input_batch.prev_req_id_to_index
                    and (req_state := self.requests.get(req_id)) is not None
                    and req_state.prev_num_draft_len
                ):
                    req_state.prev_num_draft_len = 0

        # --- _update_states ---
        self._update_states(scheduler_output)

        # --- Run core input preparation ---
        # cloud_prepare_early runs BEFORE the forward pass (outside
        # torch.inference_mode), but GDN attention builder does in-place
        # tensor copies that require inference mode.  Wrap the whole
        # preparation inside inference_mode to stay compatible with
        # PyTorch >= 2.0 inference tensor protection.
        with torch.inference_mode():
            cache = self._run_input_preparation(scheduler_output)

        # If the batch became empty after _update_states (num_reqs == 0),
        # _run_input_preparation returns a zeroed placeholder.  Don't cache
        # it — let execute_model fall through to the normal slow path which
        # will return EMPTY_MODEL_RUNNER_OUTPUT.
        if cache["total_num_scheduled_tokens"] == 0:
            self._cloud_prepare_cache = None
            return

        # --- update_cos_sin ---
        num_input_tokens = cache["num_tokens_padded"]
        if self.use_cp and self.pcp_manager.pcp_use_hybrid_attn:
            num_input_tokens = cache["total_num_scheduled_tokens"]
        update_cos_sin(self.positions[:num_input_tokens])

        # --- Cache all results ---
        self._cloud_prepare_cache = cache

    def _edge_cloud_forward(
        self,
        num_tokens_padded: int,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        layer_slice_info: Any = None,
        **model_kwargs: dict[str, Any],
    ):
        """边云场景的分段前向执行。

        核心设计：每个 segment 由标准 ACLGraphWrapper 包裹（runtime_mode=FULL），
        capture/replay 机制与标准流程完全一致，仅被包裹对象从完整 model 变为
        EdgeCloudSegment（局部层范围）。

        分段路由逻辑：
          - Edge 角色：
              • intermediate_tensors is None  → 执行 segment_a（首段，输出 IntermediateTensors）
              • intermediate_tensors is not None → 执行 segment_e（尾段，输出 hidden_states）
          - Cloud 角色：始终执行 segment_c（中段，输出 IntermediateTensors）
        """
        assert self.model is not None
        forward_context = get_forward_context()
        assert forward_context is not None
        
        # 判断当前是否应使用 ACL Graph（Decode 阶段且配置允许时）
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
        if hasattr(cudagraph_runtime_mode, "decode_mode"):
            cudagraph_runtime_mode = cudagraph_runtime_mode.decode_mode()
        use_graph = (
            cudagraph_runtime_mode == CUDAGraphMode.FULL
            and self.edge_cloud_cfg.enable_decode_graph
        )

        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
            **model_kwargs,
        }

        # Layer-sliced execution: inject the layer range for the current
        # slice so the model forward only runs those layers.
        if layer_slice_info is not None:
            slice_start = layer_slice_info.start_layer
            slice_end = layer_slice_info.end_layer
            # Edge-cloud cloud: the slice indices from PassiveScheduler are
            # relative to the cloud's local middle layers (0..num_local_layers).
            # forward_edge_cloud_segment operates on the *global* layer list
            # (0..num_hidden_layers) because the model keeps all layers
            # (non-local ones are PPMissingLayer).  Add head_k offset so the
            # slice covers the correct global range.
            if self._edge_cloud_enabled and self.edge_cloud_cfg.role == "cloud":
                slice_start += self.head_k
                slice_end += self.head_k
            model_inputs["layer_slice_start"] = slice_start
            model_inputs["layer_slice_end"] = slice_end
            # Non-last slices, and the last slice on a non-final PP rank,
            # must return IntermediateTensors.  The last slice on the last
            # PP rank should run norm + lm_head and return logits.
            if not layer_slice_info.is_last_slice or not get_pp_group().is_last_rank:
                model_inputs["layer_slice_return_intermediate"] = True

        if self._edge_cloud_enabled:
            if self.edge_cloud_cfg.role == "edge":
                segment = (
                    self.segment_a_wrapper
                    if intermediate_tensors is None
                    else self.segment_e_wrapper
                )
            else:
                segment = self.segment_c_wrapper
            run_model = partial(segment, **model_inputs)
        else:
            run_model = partial(self.model, **model_inputs)

        if self.edge_cloud_cfg.role == "edge":
            return self._edge_cloud_forward_edge(
                num_tokens_padded, input_ids, positions, intermediate_tensors,
                inputs_embeds, use_graph, forward_context, **model_kwargs,
            )
        else:
            return self._edge_cloud_forward_cloud(
                num_tokens_padded, positions, intermediate_tensors,
                use_graph, forward_context, layer_slice_info=layer_slice_info,
                **model_kwargs,
            )

    def _edge_cloud_forward_edge(
        self,
        num_tokens_padded: int,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
        use_graph: bool,
        forward_context,
        **model_kwargs: dict[str, Any],
    ):
        """Edge 侧分段执行：segment_a（首段）或 segment_e（尾段）。"""
        seg_a = self.segment_a_wrapper if use_graph else self.segment_a
        seg_e = self.segment_e_wrapper if use_graph else self.segment_e
        seg_a_graph = isinstance(seg_a, ACLGraphWrapper)
        seg_e_graph = isinstance(seg_e, ACLGraphWrapper)

        if intermediate_tensors is None:
            # Step 1：执行 Segment A（embedding + 首 head_k 层）
            # 此时 input_ids 有效，输出 IntermediateTensors 供跨节点传输
            from vllm_ascend.ascend_forward_context import _EXTRA_CTX
            old_layer_idx = _EXTRA_CTX.layer_idx
            if _EXTRA_CTX.layer_idx is not None:
                _EXTRA_CTX.layer_idx = 0
            try:
                if seg_a_graph and not forward_context.capturing:
                    self._update_full_graph_params_if_needed(
                        forward_context, num_tokens_padded, positions,
                        layer_indices=list(range(0, self.head_k)),
                        graph_wrapper=seg_a,
                    )
                hidden_states = seg_a(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )
            finally:
                if old_layer_idx is not None:
                    _EXTRA_CTX.layer_idx = old_layer_idx

            assert isinstance(hidden_states, IntermediateTensors)
            return hidden_states

        # Step 2：执行 Segment E（尾 tail_k 层 + norm）
        # intermediate_tensors 已由 NPUWorker 从 Cloud 侧接收
        #
        # 注意：segment_e 与 segment_a 共用同一个 scheduler_output，num_tokens
        # 保持不变（scheduler_output 在同一迭代内不变化）。若两者 num_tokens
        # 出现不一致，会导致 cudagraph shape 不匹配，引发图执行错误。
        # 关键：重置 layer_idx，使 weight_prefetch / EPLB 定位到尾段起始层，
        # 执行完毕后恢复原值，避免影响后续非边云路径
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        old_layer_idx = _EXTRA_CTX.layer_idx
        if _EXTRA_CTX.layer_idx is not None:
            _EXTRA_CTX.layer_idx = self.num_layers - self.tail_k

        try:
            tail_layer_indices = list(range(
                self.num_layers - self.tail_k,
                self.num_layers,
            ))
            if seg_e_graph and not forward_context.capturing:
                self._update_full_graph_params_if_needed(
                    forward_context, num_tokens_padded, positions,
                    layer_indices=tail_layer_indices,
                    graph_wrapper=seg_e,
                )
            hidden_states = seg_e(
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                **model_kwargs,
            )
        finally:
            # segment_e 执行完毕后恢复原始 layer_idx
            if old_layer_idx is not None:
                _EXTRA_CTX.layer_idx = old_layer_idx

    def _edge_cloud_forward_edge(
        self,
        num_tokens_padded: int,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
        use_graph: bool,
        forward_context,
        **model_kwargs: dict[str, Any],
    ):
        """Edge 侧分段执行：segment_a（首段）或 segment_e（尾段）。"""
        seg_a = self.segment_a_wrapper if use_graph else self.segment_a
        seg_e = self.segment_e_wrapper if use_graph else self.segment_e
        #seg_e = self.segment_e
        seg_a_graph = isinstance(seg_a, ACLGraphWrapper)
        seg_e_graph = isinstance(seg_e, ACLGraphWrapper)

        if intermediate_tensors is None:
            # Step 1：执行 Segment A（embedding + 首 head_k 层）
            # 此时 input_ids 有效，输出 IntermediateTensors 供跨节点传输
            from vllm_ascend.ascend_forward_context import _EXTRA_CTX
            old_layer_idx = _EXTRA_CTX.layer_idx
            if _EXTRA_CTX.layer_idx is not None:
                _EXTRA_CTX.layer_idx = 0
            try:
                hidden_states = seg_a(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )
                if seg_a_graph and not forward_context.capturing:
                    self._update_full_graph_params_if_needed(
                        forward_context, num_tokens_padded, positions,
                        layer_indices=list(range(0, self.head_k)),
                        graph_wrapper=seg_a,
                    )
            finally:
                if old_layer_idx is not None:
                    _EXTRA_CTX.layer_idx = old_layer_idx

            assert isinstance(hidden_states, IntermediateTensors)
            return hidden_states

        # Step 2：执行 Segment E（尾 tail_k 层 + norm）
        # intermediate_tensors 已由 NPUWorker 从 Cloud 侧接收
        #
        # 注意：segment_e 与 segment_a 共用同一个 scheduler_output，num_tokens
        # 保持不变（scheduler_output 在同一迭代内不变化）。若两者 num_tokens
        # 出现不一致，会导致 cudagraph shape 不匹配，引发图执行错误。
        # 关键：重置 layer_idx，使 weight_prefetch / EPLB 定位到尾段起始层，
        # 执行完毕后恢复原值，避免影响后续非边云路径
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        old_layer_idx = _EXTRA_CTX.layer_idx
        if _EXTRA_CTX.layer_idx is not None:
            _EXTRA_CTX.layer_idx = self.num_layers - self.tail_k

        try:
            tail_layer_indices = list(range(
                self.num_layers - self.tail_k,
                self.num_layers,
            ))
            hidden_states = seg_e(
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                **model_kwargs,
            )
            if seg_e_graph and not forward_context.capturing:
                self._update_full_graph_params_if_needed(
                    forward_context, num_tokens_padded, positions,
                    layer_indices=tail_layer_indices,
                    graph_wrapper=seg_e,
                )
        finally:
            # segment_e 执行完毕后恢复原始 layer_idx
            if old_layer_idx is not None:
                _EXTRA_CTX.layer_idx = old_layer_idx

        if forward_context.flash_comm_v1_enabled and not isinstance(hidden_states, IntermediateTensors):
            hidden_states = self._all_gather_hidden_states_and_aux(hidden_states)
        return hidden_states

    def _execute_layerwise_continuation(
        self,
        layer_slice_info: Any,
    ) -> IntermediateTensors | None:
        """Run model forward for a non-first layer slice.

        On the first slice, execute_model set up all the batch state
        (requests, attention metadata, positions, etc.) and saved the
        intermediate hidden_states/residual in
        ``self._layerwise_intermediate``.  Subsequent slices only need to
        feed that intermediate state back into the model for the current
        slice's layer range.
        """
        assert self._layerwise_intermediate is not None, (
            "Layer slice continuation requires saved intermediate state from "
            "the previous slice.  Was slice 0 executed first?"
        )

        intermediate_tensors = self._layerwise_intermediate
        self._layerwise_intermediate = None

        # Re-use the positions and attention metadata that slice 0 prepared.
        positions = self._layerwise_positions
        attn_metadata = self._layerwise_attn_metadata

        # Mark the attention metadata as a layer-slice continuation so that
        # GDN's causal_conv1d prefill path knows conv_state may have been
        # polluted by an interleaved decode batch.  See
        # vllm_ascend.ops.gdn._maybe_reset_initial_state_for_layer_slice.
        if isinstance(attn_metadata, dict):
            for per_layer_meta in attn_metadata.values():
                per_layer_meta._is_layer_slice_continuation = True
        elif attn_metadata is not None:
            attn_metadata._is_layer_slice_continuation = True

        num_tokens_padded = self._layerwise_num_tokens_padded
        num_tokens_across_dp = self._layerwise_num_tokens_across_dp
        batch_desc = self._layerwise_batch_desc

        # Non-first slices must participate in the cross-DP all_reduce
        # so that the peer DP's _dummy_run (which always calls
        # _sync_metadata_across_dp) does not deadlock.  The returned
        # values are discarded — continuation uses the slice-0 cached
        # results above, which are already correct.
        self._sync_metadata_across_dp(
            num_tokens=num_tokens_padded,
            cudagraph_mode=CUDAGraphMode.NONE,
        )

        has_encoder_input = False
        clear_kv_metadata = self.speculative_config is None

        with (
            record_function_or_nullcontext("layerwise forward"),
            set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=batch_desc,
                num_actual_tokens=num_tokens_padded,
                model_instance=self.model,
                max_tokens_across_pcp=0,
                skip_compiled=has_encoder_input,
            ),
            self.maybe_get_kv_connector_output(
                self._layerwise_scheduler_output,
                **({"defer_finalize": not clear_kv_metadata}),
            ) as kv_connector_output,
        ):
            hidden_states = self._model_forward(
                num_tokens_padded,
                None,               # input_ids
                positions,
                intermediate_tensors,
                None,               # inputs_embeds
                layer_slice_info=layer_slice_info,
            )

        with record_function_or_nullcontext("layerwise post process"):
            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = hidden_states
            if self.pcp_size > 1:
                hidden_states = self.pcp_manager.get_restore_hidden_states(
                    hidden_states
                )

            # Non-last slice: save intermediate and return None.
            if not layer_slice_info.is_last_slice:
                assert isinstance(hidden_states, IntermediateTensors)
                self._layerwise_intermediate = hidden_states
                return None

            # Edge-cloud cloud segment: always returns IntermediateTensors
            # regardless of PP rank, because logits are computed on the edge
            # side (segment_e) not on the cloud side (segment_c).
            if self._edge_cloud_enabled and self.edge_cloud_cfg.role == "cloud":
                assert isinstance(hidden_states, IntermediateTensors)
                hidden_states.kv_connector_output = kv_connector_output
                self.kv_connector_output = kv_connector_output
                return hidden_states

            # Last slice on a non-final PP rank: return IntermediateTensors
            # so NPUWorker performs the PP send.
            if not get_pp_group().is_last_rank:
                assert isinstance(hidden_states, IntermediateTensors)
                hidden_states.kv_connector_output = kv_connector_output
                self.kv_connector_output = kv_connector_output
                return hidden_states

            # Last slice on the last PP rank: the model returned logits
            # (after norm + lm_head).  Replicate the post-processing from
            # execute_model() so that sample_tokens() can produce the
            # final ModelRunnerOutput.
            assert not isinstance(hidden_states, IntermediateTensors)
            assert not self.broadcast_pp_output, (
                "Layerwise chunking with broadcast_pp_output is not yet "
                "supported on the last PP rank."
            )
            assert not self.is_pooling_model, (
                "Layerwise chunking with pooling models is not yet "
                "supported on the last PP rank."
            )
            logits_indices = self._layerwise_logits_indices
            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states)
            self.execute_model_state = ExecuteModelState(
                self._layerwise_scheduler_output,
                logits,
                self._layerwise_spec_decode_metadata,
                self._layerwise_spec_decode_common_attn_metadata,
                hidden_states,
                sample_hidden_states,
                None,   # aux_hidden_states
                self._layerwise_attn_metadata,
                self._layerwise_positions,
                self._layerwise_ec_connector_output,
                self._layerwise_cudagraph_stats,
                self._layerwise_batch_desc,
            )
            self.kv_connector_output = kv_connector_output
            return None

    def suspend_head_state(self, scheduler_output: SchedulerOutput) -> None:
        """Suspend the minimal head-segment context for later tail-segment pairing.

        Called just before the edge head-segment returns IntermediateTensors
        to the worker for cross-node transmission.
        """
        token = scheduler_output.head_token
        if not token:
            return
        if token in self._pending_head_states:
            raise RuntimeError(
                f"head_token={token} already suspended; "
                "previous tail did not consume it"
            )
        self._pending_head_states[token] = HeadState(
            head_token=token,
            scheduler_output=scheduler_output,
            req_ids=tuple(self.input_batch.req_ids),
        )

    def _resume_and_validate_head_state(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Resume and validate the suspended HeadState for a tail segment.

        The control-plane head_token identifies the HeadState saved by the
        matching edge head segment. Data-plane hidden tensors are aligned by
        scheduler-selected hidden channels, so no data-plane token tensor is
        required in IntermediateTensors.
        """
        token_ctrl = scheduler_output.head_token
        if not token_ctrl:
            raise RuntimeError(
                "PL/DL scheduler_output must carry head_token from cloud"
            )

        head_state = self._pending_head_states.pop(token_ctrl, None)
        if head_state is None:
            raise RuntimeError(
                f"No suspended HeadState for head_token={token_ctrl}; "
                f"tail dispatched without matching head or already consumed"
            )

        expected = self._expected_tail_batch_type(
            head_state.scheduler_output.batch_type
        )
        if scheduler_output.batch_type != expected:
            raise RuntimeError(
                f"HeadState batch_type mismatch: head was "
                f"{head_state.scheduler_output.batch_type}, "
                f"tail is {scheduler_output.batch_type}"
            )

        tail_req_ids = set(scheduler_output.num_scheduled_tokens.keys())
        head_req_ids = set(head_state.req_ids)
        if tail_req_ids != head_req_ids:
            raise RuntimeError(
                f"HeadState req_ids mismatch: head had "
                f"{head_req_ids}, tail scheduler_output has "
                f"{tail_req_ids}"
            )

    @staticmethod
    def _expected_tail_batch_type(head_bt: BatchType) -> BatchType:
        return {
            BatchType.PREFILL_FIRST: BatchType.PREFILL_LAST,
            BatchType.DECODE_FIRST: BatchType.DECODE_LAST,
        }[head_bt]

    def _edge_cloud_forward_cloud(
        self,
        num_tokens_padded: int,
        positions: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
        use_graph: bool,
        forward_context,
        layer_slice_info: Any = None,
        **model_kwargs: dict[str, Any],
    ):
        """Cloud 侧分段执行：segment_c（中段）。"""
        assert self.edge_cloud_cfg.role == "cloud", (
            "Cloud segment_c should only be executed when role == 'cloud'"
        )
        # Warmup（profile_run）期间，Cloud 通过 PP non-first rank 路径自行构造
        # minimal intermediate_tensors（仅含 shape 信息）用于图捕获，此时 intermediate_tensors
        # 不为 None 但也是 dummy。运行时 real inference 时，intermediate_tensors 由
        # Worker 层从 Edge 侧接收，为真实的 HiddenStates。区分方式：检查 in_profile_run。
        in_warmup = getattr(forward_context, "in_profile_run", False)
        assert intermediate_tensors is not None or in_warmup, (
            "Cloud segment_c requires intermediate_tensors from Edge side"
        )

        if layer_slice_info is not None:
            # Cloud prefill layer slicing changes layer_slice_start/end per
            # slice. EdgeCloudCompiledSegment/npugraph_ex may specialize those
            # Python int ranges from the first slice and reuse them for later
            # slices, causing wrong-layer execution. Use the raw segment for
            # sliced prefill while keeping graph/compile for decode and
            # non-sliced prefill.
            seg_c = self.segment_c_raw
            logger.debug(
                "[EdgeCloud] Cloud sliced prefill uses raw segment: "
                "slice=%d/%d local=[%d,%d) global=[%d,%d)",
                layer_slice_info.slice_index + 1,
                layer_slice_info.total_slices,
                layer_slice_info.start_layer,
                layer_slice_info.end_layer,
                layer_slice_info.start_layer + self.head_k,
                layer_slice_info.end_layer + self.head_k,
            )
        else:
            seg_c = self.segment_c_wrapper if use_graph else self.segment_c
        seg_c_graph = isinstance(seg_c, ACLGraphWrapper)

        if seg_c_graph:
            if layer_slice_info is not None:
                cloud_layer_indices = list(range(
                    layer_slice_info.start_layer + self.head_k,
                    layer_slice_info.end_layer + self.head_k,
                ))
            else:
                cloud_layer_indices = list(range(
                    self.head_k,
                    self.num_layers - self.tail_k,
                ))
        # intermediate_tensors 已由 NPUWorker 从 Edge 侧接收
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        old_layer_idx = _EXTRA_CTX.layer_idx
        if _EXTRA_CTX.layer_idx is not None:
            if layer_slice_info is not None:
                # Layer-sliced execution: each slice starts at a different
                # local layer.  Add head_k offset to get the global layer
                # index so that weight_prefetch / EPLB route to the correct
                # layer weights for this slice.
                _EXTRA_CTX.layer_idx = (
                    layer_slice_info.start_layer + self.head_k
                )
            else:
                _EXTRA_CTX.layer_idx = self.head_k

        if layer_slice_info is not None:
            model_kwargs = dict(model_kwargs)
            model_kwargs["layer_slice_start"] = (
                layer_slice_info.start_layer + self.head_k
            )
            model_kwargs["layer_slice_end"] = (
                layer_slice_info.end_layer + self.head_k
            )
            if not layer_slice_info.is_last_slice:
                model_kwargs["layer_slice_return_intermediate"] = True
            # [FIX] When ForwardContext is recreated for each slice,
            # moe_layer_index resets to 0, causing every slice to
            # reference all_moe_layers[0] (the first MoE layer).
            # Compute the correct starting index based on the slice's
            # global start layer.
            if (
                forward_context is not None
                and forward_context.all_moe_layers is not None
            ):
                from vllm.model_executor.models.utils import (
                    extract_layer_index,
                )
                global_start = layer_slice_info.start_layer + self.head_k
                moe_start = sum(
                    1
                    for name in forward_context.all_moe_layers
                    if extract_layer_index(name) < global_start
                )
                forward_context.moe_layer_index = moe_start

        hidden_states = seg_c(
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            **model_kwargs,
        )
        if seg_c_graph and not forward_context.capturing:
            self._update_full_graph_params_if_needed(
                forward_context, num_tokens_padded, positions,
                layer_indices=cloud_layer_indices,
                graph_wrapper=seg_c,
            )
        
        if old_layer_idx is not None:
            _EXTRA_CTX.layer_idx = old_layer_idx

        # Cloud 必须返回 IntermediateTensors，供 Worker 层发回 Edge 并最终由 Edge 计算 logits
        assert isinstance(hidden_states, IntermediateTensors)
        return hidden_states

    def _pad_for_sequence_parallelism(
        self, num_scheduled_tokens: int, for_cudagraph_capture: bool = False
    ) -> int:
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if enable_sp(self.vllm_config) or enable_sp_by_pass() or self.edge_cloud_cfg.cloud_enable_sp:
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    # These functions from upstream vllm handle PP+SP. Ascend's flashcomm1 SP
    # differs from vllm's native SP: flashcomm1 does NOT scatter the residual
    # before PP send, so the all_gather in sync_and_gather_intermediate_tensors
    # must be skipped. Both overrides use enable_sp() rather than
    # is_residual_scattered_for_sp() to reflect the actual Ascend SP state.
    def sync_and_slice_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        assert self.intermediate_tensors is not None
        tp = self.vllm_config.parallel_config.tensor_parallel_size

        if sync_self:
            assert intermediate_tensors is not None, (
                "sync_and_slice_intermediate_tensors received None; "
                "check PP/TP tensor delivery."
            )
            if (self._edge_cloud_enabled
                    and self.edge_cloud_cfg.role == "cloud"
                    and self.edge_cloud_cfg.mode == "embedding_only"
                    and self.supports_mm_inputs):
                # In edge-cloud embedding-only multimodal mode the edge sends
                # the full sequence (it does not SP-chunk, see worker.py).
                # The Cloud's first transformer layer expects full
                # hidden_states/residual and handles TP/SP internally (e.g. VL
                # first-layer special branch). Return all keys from the local
                # buffer so residual is always present.
                #
                # The edge side strips cudagraph/SP padding before transmission,
                # so the received tensors may be shorter than num_tokens. Copy
                # the received prefix and zero-fill the padding locally to avoid
                # a shape-mismatch copy_ error on NPUs (e.g. 60 vs 64).
                for k, v in intermediate_tensors.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    copy_len = num_tokens
                    dst = self.intermediate_tensors[k][:copy_len]
                    recv_len = min(v.shape[0], copy_len)
                    if recv_len:
                        dst[:recv_len].copy_(v[:recv_len], non_blocking=True)
                    if recv_len < copy_len:
                        dst[recv_len:].zero_()
                return IntermediateTensors(
                    {
                        k: v[:num_tokens]
                        for k, v in self.intermediate_tensors.items()
                    }
                )
            else:
                for k, v in intermediate_tensors.items():
                    copy_len = (num_tokens + tp - 1) // tp if enable_sp() else num_tokens
                    dst = self.intermediate_tensors[k][:copy_len]
                    # Senders may transmit only real tokens; fill graph padding locally.
                    recv_len = min(v.shape[0], copy_len)
                    if recv_len:
                        dst[:recv_len].copy_(v[:recv_len], non_blocking=True)
                    if recv_len < copy_len:
                        dst[recv_len:].zero_()

        return IntermediateTensors(
            {
                k: v[: (num_tokens + tp - 1) // tp]
                if enable_sp()
                else v[:num_tokens]
                for k, v in self.intermediate_tensors.items()
            }
        )

    def sync_and_gather_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        # vllm renamed sync_and_slice to sync_and_gather in v0.20.2.
        # The Ascend override logic is identical: skip the upstream all_gather
        # (flashcomm1 does not scatter residual before PP send).
        return self.sync_and_slice_intermediate_tensors(
            num_tokens, intermediate_tensors, sync_self
        )

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = False,
        force_eager: bool = False,
        # For cudagraph capture TODO(lucas): Refactor how we capture cudagraphs (will
        # be improved in model runner v2)
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
        for_cudagraph_capture: bool = False,
    ) -> tuple[CUDAGraphMode, BatchDescriptor, bool, torch.Tensor | None, CUDAGraphStat | None]:
        num_tokens_padded = self._pad_for_sequence_parallelism(
            num_tokens, for_cudagraph_capture=for_cudagraph_capture
        )
        is_all_decode = np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] > 0)
        uniform_decode = (
            (
                (is_all_decode if self.speculative_config else True)
                and (max_num_scheduled_tokens == self.uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )
        # Encoder-decoder models only support CG for decoder_step > 0 (no enc_output
        # is present). Also, chunked-prefill is disabled, so batch are uniform.
        has_encoder_output = self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        num_active_loras = (
            force_num_active_loras
            if force_num_active_loras is not None
            else len(self.input_batch.lora_id_to_lora_request)
        )
        has_lora = num_active_loras > 0 if force_has_lora is None else force_has_lora

        # ruff: noqa: E731
        def dispatch_cudagraph(num_tokens, disable_full=False, valid_modes=None):
            if force_eager:
                return (CUDAGraphMode.NONE, BatchDescriptor(num_tokens_padded))

            return self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                valid_modes=valid_modes,
                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
                num_active_loras=num_active_loras,
            )

        cudagraph_mode, batch_descriptor = dispatch_cudagraph(num_tokens_padded, use_cascade_attn or has_encoder_output)
        num_tokens_padded = batch_descriptor.num_tokens
        if enable_sp(self.vllm_config):
            assert batch_descriptor.num_tokens % self.vllm_config.parallel_config.tensor_parallel_size == 0, (
                "Sequence parallelism requires num_tokens to be a multiple of tensor parallel size"
            )
        # Extra coordination when running data-parallel since we need to coordinate
        # across ranks
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            _, num_tokens_across_dp, synced_cudagraph_mode = self._sync_metadata_across_dp(
                num_tokens=num_tokens_padded,
                cudagraph_mode=cudagraph_mode,
                allow_dp_padding=(cudagraph_mode != CUDAGraphMode.NONE) or enable_sp(self.vllm_config),
            )

            # Extract DP padding if there is any
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded,
                    valid_modes={synced_cudagraph_mode},
                )
                # Assert to make sure the agreed upon token count is correct otherwise
                # num_tokens_across_dp will no-longer be valid
                assert batch_descriptor.num_tokens == num_tokens_padded
        cudagraph_stats = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )

        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        num_scheduled_tokens_np: np.ndarray | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
        num_scheduled_tokens_compressed_list: list[np.ndarray] | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        """
        :return: tuple[attn_metadata, spec_decode_common_attn_metadata]
        """
        # Attention metadata is not needed for attention free models
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None
        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs
        attn_metadata: PerLayerAttnMetadata = {}
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]

        # Ensure the async GPU→CPU copy of corrected seq_lens (launched in
        # _prepare_inputs) has completed before we read optimistic_seq_lens_cpu.
        if self._seq_lens_cpu_event_pending and self._seq_lens_cpu_event is not None:
            self._seq_lens_cpu_event.synchronize()
            self._seq_lens_cpu_event_pending = False

        if for_cudagraph_capture:
            # For some attention backends (e.g. FA) with sliding window models we need
            # to make sure the backend see a max_seq_len that is larger to the sliding
            # window size when capturing to make sure the correct kernel is selected.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()

        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        def _get_pcp_metadata(block_table_tensor):
            if not self.use_cp:
                return None, block_table_tensor
            return self.pcp_manager.generate_pcp_metadata(
                num_tokens,
                self.query_lens,
                self.input_batch,
                num_scheduled_tokens_np,
                block_table_tensor,
                num_reqs_padded,
                num_reqs,
            )

        def _get_block_table_and_slot_mapping(kv_cache_gid: int, total_num_scheduled_tokens_compressed_list: list[int]):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if self.pcp_size > 1:
                total_num_pcp_pads = sum(self.pcp_manager.num_pcp_pads_cpu[:num_reqs])
                if self.pcp_manager.pcp_use_hybrid_attn:
                    num_scheduled_tokens_padded = self.pcp_manager.num_scheduled_tokens_padded
                    assert num_scheduled_tokens_padded is not None
                    maybe_pcp_full_tokens = sum(num_scheduled_tokens_padded) * self.pcp_size - total_num_pcp_pads
                else:
                    maybe_pcp_full_tokens = num_tokens * self.pcp_size - total_num_pcp_pads
            else:
                maybe_pcp_full_tokens = num_tokens_padded
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                blk_table_tensor = torch.zeros(
                    (num_reqs_padded, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (num_tokens_padded,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_gid]
                slot_mapping = blk_table.slot_mapping.gpu[:maybe_pcp_full_tokens]
                maybe_num_reqs_padded = num_reqs_padded * self.decode_token_per_req if self.use_cp else num_reqs_padded
                blk_table_tensor = blk_table.get_device_tensor()[:maybe_num_reqs_padded]

                # Fill unused with -1. Needed for reshape_and_cache in full cuda
                # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
                if self.pcp_size == 1:
                    if self.use_compress and total_num_scheduled_tokens_compressed_list is not None:
                        slot_mapping[
                            total_num_scheduled_tokens_compressed_list[
                                kv_cache_gid]:num_tokens_padded].fill_(-1)
                    elif self.use_compress:
                        # DSA dummy/graph-capture runs do not go through
                        # _prepare_inputs(), so no fresh compressed cache
                        # metadata is computed for them. Reusing values from
                        # the previous real request can feed stale block-table
                        # and [block, offset] scatter indices to DSA kernels.
                        slot_mapping[:num_tokens_padded].fill_(0)
                        blk_table_tensor[:maybe_num_reqs_padded].fill_(0)
                    else:
                        slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
                        blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)
            if self.pcp_size > 1:
                slot_mapping = self.pcp_manager.get_padded_slot_mapping(
                    num_tokens,
                    num_tokens_padded,
                    slot_mapping,
                    kv_cache_gid,
                )
            if self.model_config.enable_return_routed_experts and kv_cache_gid == 0:
                if vllm_version_is("0.20.2"):
                    self.cpu_slot_mapping = slot_mapping.cpu().numpy()
                elif self.routed_experts_initialized:
                    # snapshot slot_mapping into a private device
                    # buffer so the next ``_prepare_inputs`` does not
                    # overwrite it while D2H is still pending.
                    n = slot_mapping.shape[0]
                    self.routed_experts_slot_mapping_device[:n].copy_(
                        slot_mapping
                    )
            return blk_table_tensor, slot_mapping

        if self.use_compress and num_scheduled_tokens_compressed_list is not None:
            total_num_scheduled_tokens_compressed_list = [
                sum(num_scheduled_tokens_compressed)
                for num_scheduled_tokens_compressed in
                num_scheduled_tokens_compressed_list
            ]
            num_reqs_actual = num_reqs
        else:
            total_num_scheduled_tokens_compressed_list = None
            num_reqs_actual = num_reqs

        block_table_gid_0, slot_mapping_gid_0 = _get_block_table_and_slot_mapping(
            0, total_num_scheduled_tokens_compressed_list)  # type: ignore[arg-type]
        self.long_seq_metadata, block_table_gid_0 = _get_pcp_metadata(block_table_gid_0)
        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
            :num_reqs_padded
        ]
        num_prompt_tokens_cpu = self.input_batch.num_prompt_tokens_cpu_tensor[
            :num_reqs_padded
        ]
        is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
        is_prefilling[num_reqs:] = False
        seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs_padded]
        if self.use_async_spec_decode:
            # GPU tensors are authoritative in async mode.
            seq_lens_cpu = None
            num_computed_tokens_cpu = None

        cm_base = AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
            seq_lens=self.seq_lens[:num_reqs_padded],
            # Always pass optimistic_seq_lens_cpu via _seq_lens_cpu so NPU
            # attention backends can get CPU seq_lens without GPU->CPU sync.
            # This is separate from seq_lens_cpu (None in async) which eagle
            # proposer checks to distinguish async/non-async behavior.
            _seq_lens_cpu=self.optimistic_seq_lens_cpu[:num_reqs_padded],
            seq_lens_cpu_upper_bound=self.optimistic_seq_lens_cpu[:num_reqs_padded],
            # TODO
            seq_lens_cpu=seq_lens_cpu,
            # TODO
            # num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs_padded],
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_gid_0,
            slot_mapping=slot_mapping_gid_0,
            causal=True,
            is_prefilling=is_prefilling,
            num_input_tokens=num_tokens_padded,
            actual_seq_lengths_q=self.actual_seq_lengths_q,
            positions=self.positions,
            positions_cpu=self._dsa_positions_cpu_buf if self.use_compress else None,
            attn_state=self.attn_state,
            decode_token_per_req=self.decode_token_per_req,
            prefill_context_parallel_metadata=self.long_seq_metadata,
        )

        if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
            cm_base.num_logits_indices = logits_indices.size(0)
            cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(logits_indices)

        def _build_attn_group_metadata(
            kv_cache_gid: int,
            attn_gid: int,
            common_attn_metadata: CommonAttentionMetadata,
            num_reqs_actual: int,
            prefill_ratio_to_sas_metadata: dict,
            decode_ratio_to_sas_metadata: dict,
            common_ratio_to_sas_metadata: dict,
            ubid: int | None = None,
        ) -> None:
            attn_group = self.attn_groups[kv_cache_gid][attn_gid]
            builder = attn_group.get_metadata_builder(ubid or 0)
            cascade_attn_prefix_len = (
                cascade_attn_prefix_lens[kv_cache_gid][attn_gid] if cascade_attn_prefix_lens else 0
            )

            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):
                assert ubid is None, "UBatching not supported with GDN yet"
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[:num_reqs_padded],
                )

            if isinstance(builder, (AscendDSAMetadataBuilder, AscendDSACPMetadataBuilder)):
                compress_ratio = getattr(attn_group.kv_cache_spec, "compress_ratio", 1)
                if for_cudagraph_capture:
                    extra_attn_metadata_args = dict(
                        compress_ratio=compress_ratio,
                        prefill_ratio_to_sas_metadata=dict(),
                        decode_ratio_to_sas_metadata=dict(),
                        common_ratio_to_sas_metadata=dict(),
                        block_size=attn_group.kv_cache_spec.block_size,
                        )
                else:
                    extra_attn_metadata_args = dict(
                        compress_ratio=compress_ratio,
                        num_reqs_actual=num_reqs_actual,
                        prefill_ratio_to_sas_metadata=prefill_ratio_to_sas_metadata,
                        decode_ratio_to_sas_metadata=decode_ratio_to_sas_metadata,
                        common_ratio_to_sas_metadata=common_ratio_to_sas_metadata,
                        block_size=attn_group.kv_cache_spec.block_size,
                        )

            # add kvcomp_metadata into common_attn_metadata
            if (for_cudagraph_capture
                    and not isinstance(builder, (AscendDSAMetadataBuilder, AscendDSACPMetadataBuilder))):
                attn_metadata_i = builder.build_for_cudagraph_capture(common_attn_metadata)
            else:
                attn_metadata_i = builder.build(
                    common_prefix_len=cascade_attn_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args,
                )
                # NOTE(zxr): Due to the Triton operator does not deal with -1 padding in FullGraph mode,
                # the padding needs to be changed from -1 to 0 to avoid writing invalid mamba block.
                if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs() \
                    and isinstance(builder, GDNAttentionMetadataBuilder) and attn_metadata_i.num_prefills == 0:
                    if attn_metadata_i.num_decodes == 0 and attn_metadata_i.num_spec_decodes > 0:
                        attn_metadata_i.spec_state_indices_tensor[attn_metadata_i.num_spec_decodes:].fill_(0)
            if isinstance(builder, AscendDSAMetadataBuilder):
                prefill_ratio_to_sas_metadata = builder.prefill_ratio_to_sas_metadata  # type: ignore[assignment]
                decode_ratio_to_sas_metadata = builder.decode_ratio_to_sas_metadata  # type: ignore[assignment]
                common_ratio_to_sas_metadata = builder.common_ratio_to_sas_metadata  # type: ignore[assignment]

            if ubid is None:
                assert isinstance(attn_metadata, dict)
                attn_metadata_dict = attn_metadata
            else:
                assert isinstance(attn_metadata, list)
                attn_metadata_dict = attn_metadata[ubid]

            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = attn_metadata_i

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        prefill_ratio_to_sas_metadata: dict[Any, Any] = {}
        decode_ratio_to_sas_metadata: dict[Any, Any] = {}
        common_ratio_to_sas_metadata: dict[Any, Any] = {}
        spec_decode_common_attn_metadata = None
        for kv_cache_gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            cm = copy(cm_base)  # shallow copy
            # Basically only the encoder seq_lens, block_table and slot_mapping change
            # for each kv_cache_group.
            cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
                num_scheduled_tokens or {},
                kv_cache_group.kv_cache_spec,
                num_reqs_padded,
            )

            # Now, query_start_loc is padded.
            # But gdn needs an unpadded one.
            # gdn_query_start_loc is an unpadded version of query_start_loc.
            # TODO delete it if fia's check is removed.
            if self._has_gdn and self.attn_groups[kv_cache_gid]:
                attn_group = self.attn_groups[kv_cache_gid][0]
                builder = attn_group.get_metadata_builder(0)
                if isinstance(builder, GDNAttentionMetadataBuilder):
                    cm.query_start_loc_cpu = self.gdn_query_start_loc.cpu[: num_reqs_padded + 1]
                    cm.query_start_loc = self.gdn_query_start_loc.gpu[: num_reqs_padded + 1]

            if kv_cache_gid > 0:
                cm.block_table_tensor, cm.slot_mapping = _get_block_table_and_slot_mapping(
                    kv_cache_gid, total_num_scheduled_tokens_compressed_list)  # type: ignore[arg-type]
            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(self.drafter, AscendEagleProposer | AscendDraftModelProposer | AscendDflashProposer):
                    if self.drafter.attn_layer_names[0] in kv_cache_group.layer_names:
                        spec_decode_common_attn_metadata = cm
                else:
                    spec_decode_common_attn_metadata = cm
            if self.enable_hamming_sparse is True:
                from vllm_ascend.attention.kvcomp_attn.attention_utils import build_kvcomp_metadata
                build_kvcomp_metadata(self.kvcomp_meta_data, cm)
            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                _build_attn_group_metadata(
                    kv_cache_gid, attn_gid, cm, num_reqs_actual,
                    prefill_ratio_to_sas_metadata, decode_ratio_to_sas_metadata,
                    common_ratio_to_sas_metadata)
        if self.is_mm_prefix_lm:
            req_doc_ranges = {}
            for req_id in self.input_batch.req_ids:
                image_doc_ranges = []
                req_state = self.requests[req_id]
                for mm_feature in req_state.mm_features:
                    pos_info = mm_feature.mm_position
                    img_doc_range = pos_info.extract_embeds_range()
                    image_doc_ranges.extend(img_doc_range)
                req_idx = self.input_batch.req_id_to_index[req_id]
                req_doc_ranges[req_idx] = image_doc_ranges

            if isinstance(attn_metadata, list):
                for ub_metadata in attn_metadata:
                    for _metadata in ub_metadata.values():
                        _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]
            else:
                for _metadata in attn_metadata.values():
                    _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]

        if spec_decode_common_attn_metadata is not None and (
            num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
        ):
            # Currently the drafter still only uses piecewise cudagraphs (and modifies
            # the attention metadata in directly), and therefore does not want to use
            # padded attention metadata.
            spec_decode_common_attn_metadata = spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
        return attn_metadata, spec_decode_common_attn_metadata

    def _should_build_dummy_attn_metadata(
        self,
        force_attention: bool = False,
        is_profile: bool = False,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
    ) -> bool:
        """
        Determine whether attention metadata should be built during dummy_run.
        SubClass can override this to add custom conditions.
        """
        # If force_attention is True, we always capture attention, Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        return force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
        profile_cpp: bool = False,
        layer_slice_info: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # only support eager mode and piecewise graph now
        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode.valid_runtime_modes()

        # [SLICE-DIAG] Log whether _dummy_run receives layer_slice_info.
        _lsi = layer_slice_info
        logger.error(
            "[SLICE-DIAG] _dummy_run: layer_slice_info=%s "
            "is_first=%s is_last=%s start=%s end=%s total=%s "
            "num_tokens=%s uniform_decode=%s",
            type(_lsi).__name__ if _lsi is not None else "None",
            getattr(_lsi, "is_first_slice", None) if _lsi is not None else None,
            getattr(_lsi, "is_last_slice", None) if _lsi is not None else None,
            getattr(_lsi, "start_layer", None) if _lsi is not None else None,
            getattr(_lsi, "end_layer", None) if _lsi is not None else None,
            getattr(_lsi, "total_slices", None) if _lsi is not None else None,
            num_tokens,
            uniform_decode,
        )
        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens
        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            raise NotImplementedError("create_mixed_batch is used for warmup deepgemm, vllm-ascend does not need it")
        elif uniform_decode:
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        elif profile_cpp:
            num_reqs = 1
            num_scheduled_tokens_list = [num_tokens] * num_reqs
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs

        if not is_profile and self.dynamic_eplb:
            self.eplb_updator.forward_before()

        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        self.query_lens = torch.from_numpy(num_scheduled_tokens)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        self._dp_batch_type_id = 0  # DUMMY
        _cudagraph_mode, batch_desc, _, num_tokens_across_dp, _ = self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens,
            max_num_scheduled_tokens=max_query_len,
            use_cascade_attn=False,
            allow_microbatching=allow_microbatching,
            force_eager=is_profile or (cudagraph_runtime_mode == CUDAGraphMode.NONE) or profile_cpp,
            # `force_uniform_decode` is used for cudagraph capture; because for
            # capturing mixed prefill-decode batches, we sometimes use
            # num_tokens == num_reqs which looks like a uniform decode batch to the
            # dispatcher; but we actually want to capture a piecewise cudagraph
            force_uniform_decode=uniform_decode,
            # `force_has_lora` is used for cudagraph capture; because LoRA is
            # activated later in the context manager, but we need to know the
            # LoRA state when determining the batch descriptor for capture
            force_has_lora=num_active_loras > 0,
            force_num_active_loras=num_active_loras,
            for_cudagraph_capture=is_graph_capturing,
        )
        if self.use_cp:
            self.pcp_manager.init_batch_info(
                num_scheduled_tokens,
                num_reqs,
            )
            if self.speculative_config:
                self.pcp_manager.query_lens_pcp_full.cpu[:num_reqs] = torch.from_numpy(num_scheduled_tokens)
                self.pcp_manager.query_lens_pcp_full.copy_to_gpu()
        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )
        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        if num_tokens_across_dp is not None and num_tokens_padded != num_tokens:
            # pad is needed if the pad of `num_tokens` is triggered inside CudagraphDispatcher
            num_tokens_across_dp[:] = num_tokens_padded
            num_scheduled_tokens = num_scheduled_tokens.repeat(num_reqs_padded)

        # SP padding or cudagraph dispatcher may increase num_reqs_padded
        # beyond the length of num_scheduled_tokens. Ensure they match.
        if len(num_scheduled_tokens) < num_reqs_padded:
            if num_tokens_padded == num_reqs_padded * max_query_len:
                # Uniform decode: each request (including padded) has max_query_len tokens
                num_scheduled_tokens = np.full(num_reqs_padded, max_query_len, dtype=num_scheduled_tokens.dtype)
            else:
                # Mixed batch: padded requests have 0 scheduled tokens
                extended = np.zeros(num_reqs_padded, dtype=num_scheduled_tokens.dtype)
                extended[:len(num_scheduled_tokens)] = num_scheduled_tokens
                num_scheduled_tokens = extended
        # vllm-ascend does not support ubatch now
        ubatch_slices, ubatch_slices_padded = None, None
        attn_metadata: PerLayerAttnMetadata | None = None
        # Build attention metadata for dummy_run
        if self._should_build_dummy_attn_metadata(force_attention, is_profile, cudagraph_runtime_mode):
            if create_mixed_batch:
                raise NotImplementedError(
                    "create_mixed_batch is used for warmup deepgemm, vllm-ascend does not need it"
                )
            self.attn_state = AscendAttentionState.DecodeOnly
            if self.speculative_config and self.speculative_config.method == "mtp":
                # `AscendAttentionState.SpecDecoding` is only designed for mla
                if self.vllm_config.model_config.use_mla:
                    self.attn_state = AscendAttentionState.SpecDecoding
                else:
                    self.attn_state = AscendAttentionState.ChunkedPrefill
            # The reason why we use a fixed seq_len rather than max_query_len is that
            # _npu_paged_attention_get_workspace only returns max workspace with specific
            # seq_lens. We use this seq_len only when capturing graph, and still use max_query_len
            # in inference. This will be removed once npu_fused_infer_attention_score
            # outperforms _npu_paged_attention on all cases.
            if profile_seq_lens is not None:
                seq_lens = profile_seq_lens
            else:
                seq_lens = (
                    SEQ_LEN_WITH_MAX_PA_WORKSPACE
                    if is_graph_capturing and using_paged_attention(num_tokens, self.vllm_config)
                    else max_query_len
                )  # type: ignore[assignment]

            self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
            self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
            self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)

            cum_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np)
            self.query_start_loc.np[1 : num_reqs_padded + 1] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()
            if self._has_gdn:
                self.gdn_query_start_loc.np[1 : num_reqs_padded + 1] = cum_num_tokens
                self.gdn_query_start_loc.copy_to_gpu()

            if not profile_cpp:
                num_reqs_padded = self._pad_query_start_loc_for_fia(
                    num_tokens_padded, num_reqs_padded, num_reqs, cudagraph_runtime_mode, batch_desc.num_reqs
                )

            pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
            # check how to build dummy
            if self.use_compress:
                self.positions.fill_(127)
                self._dsa_positions_cpu_buf.fill_(127)
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=ubatch_slices_padded if pad_attn else ubatch_slices,
                for_cudagraph_capture=is_graph_capturing,
                num_scheduled_tokens_np=num_scheduled_tokens,
            )

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            remove_lora,
            # TODO: The next line is a temporary workaround
            # to fix the accuracy issue of test_llama32_lora.py,
            # which is introduced by vllm-project/vllm#32005
            num_active_loras=(self.lora_config.max_loras if self.lora_config is not None else num_active_loras),
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder or self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions[:num_tokens_padded]

            # update global cos, sin
            update_cos_sin(positions)

            # ========== 边云模式（Edge-Cloud Mode）中间张量处理 ==========
            # 边云模式下根据 role 判断是否需要 intermediate_tensors，
            # 替代标准 PP（Pipeline Parallelism）的 is_first_rank 判断（pp_size=1 时所有 rank 都是 first）。
            if self._edge_cloud_enabled:
                if self.edge_cloud_cfg.role == "edge":
                    # Edge 端：不需要中间张量（第一阶段）
                    intermediate_tensors = None
                else:
                    # Cloud 端：需要中间张量。
                    # Layer-sliced dummy: non-first slices reuse the
                    # intermediate state saved by the previous slice,
                    # just like _execute_layerwise_continuation.
                    if (
                        layer_slice_info is not None
                        and not layer_slice_info.is_first_slice
                        and self._layerwise_intermediate is not None
                    ):
                        intermediate_tensors = self._layerwise_intermediate
                        self._layerwise_intermediate = None
                    else:
                        intermediate_tokens = num_tokens_padded
                        # embedding-only 模式下 Cloud 从首层开始执行，输入来自 Edge 的
                        # embedding 输出，应为完整序列长度（运行时
                        # sync_and_slice_intermediate_tensors 亦使用完整 num_tokens）。
                        if enable_sp() and (self.edge_cloud_cfg.mode != "embedding_only"
                            or not self.supports_mm_inputs):
                            tp_size = get_tensor_model_parallel_world_size()
                            intermediate_tokens = (num_tokens_padded + tp_size - 1) // tp_size
                        if self.intermediate_tensors is None:
                            # 首次创建 intermediate_tensors，使用最大可能 token 数
                            max_actual_tokens = self.max_num_tokens
                            if enable_sp() and (self.edge_cloud_cfg.mode != "embedding_only"
                                or not self.supports_mm_inputs):
                                max_actual_tokens = (self.max_num_tokens + tp_size - 1) // tp_size
                            # 调用模型方法创建空的中间张量
                            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                                batch_size=max_actual_tokens, dtype=self.dtype, device=self.device
                            )
                            logger.info(
                                "[Cloud _dummy_run] Created intermediate_tensors "
                                "hidden_states shape=%s via make_empty_intermediate_tensors",
                                list(self.intermediate_tensors["hidden_states"].shape),
                            )
                        # 切片到实际需要的 token 数
                        intermediate_tensors = IntermediateTensors(
                            {k: v[:intermediate_tokens] for k, v in self.intermediate_tensors.items()}
                        )
                        # Zero-fill to avoid NaN from uninitialized memory
                        # (make_empty_intermediate_tensors may use torch.empty)
                        for _k, _v in intermediate_tensors.items():
                            _v.zero_()
                        _diag_hs = intermediate_tensors["hidden_states"]
                        logger.error(
                            "[PD-DIAG] D. cloud _dummy_run INPUT (intermediate_tensors): "
                            "shape=%s norm=%.6f mean=%.6f",
                            list(_diag_hs.shape),
                            float(_diag_hs.float().norm().item()),
                            float(_diag_hs.float().mean().item()),
                        )
            elif get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                # When PP and flashcomm1 are enabled, during dummy_run the estimated space should divide num_tokens by
                # tp_size; otherwise, on non-first PP ranks it would effectively perform an extra all-gather, leading
                # to incorrect memory estimation and potentially causing OOM.
                intermediate_tokens = num_tokens_padded
                if enable_sp():
                    tp_size = get_tensor_model_parallel_world_size()
                    intermediate_tokens = (num_tokens_padded + tp_size - 1) // tp_size
                if self.intermediate_tensors is None:
                    max_actual_tokens = self.max_num_tokens
                    if enable_sp():
                        max_actual_tokens = (self.max_num_tokens + tp_size - 1) // tp_size
                    self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                        batch_size=max_actual_tokens, dtype=self.dtype, device=self.device
                    )
                intermediate_tensors = IntermediateTensors(
                    {k: v[:intermediate_tokens] for k, v in self.intermediate_tensors.items()}
                )

            need_dummy_logits = not is_profile and lmhead_tp_enable()
            max_num_reqs_across_dp = max_num_reqs * self.uniform_decode_query_len
            dummy_indices = torch.zeros(max_num_reqs_across_dp, dtype=torch.int32)

            def dummy_compute_logits(hidden_states):
                if not need_dummy_logits:
                    return None
                return self.model.compute_logits(hidden_states[dummy_indices])

            def dummy_drafter_compute_logits(hidden_states):
                if not need_dummy_logits or self.drafter is None:
                    return
                if hasattr(self.drafter, "model") and hasattr(self.drafter.model, "compute_logits"):
                    return self.drafter.model.compute_logits(hidden_states[dummy_indices])

            # PD-separation: DUMMY matches real DP's segment.
            # - peer_bt == 0 (both DUMMY / idle): skip both segments for BOTH
            #   edge and cloud. No real data to all_gather, so no MoE forward
            #   and no cross-DP EP all-toall; only sync_metadata (already done)
            #   keeps the all_reduce pairing. This makes DUMMY instant.
            #   CRITICAL for cloud: without this, a cloud DUMMY always runs the
            #   full MoE forward (all-toall on the cross-DP EP group), so two
            #   idle clouds - each fed by its edge's self-fired dummies - end up
            #   with mismatched all-toall counts and deadlock (the dp=2
            #   PP-init-timeout symptom: cloud rank1 stuck in a dummy's
            #   all-toall, never reaching the real PREFILL_FIRST).
            # - Edge DUMMY with a real peer: run the matching segment only -
            #   peer head (PF/DF, bt 1/3) -> run head (segment_a); peer tail
            #   (PL/DL, bt 2/4) -> run tail (segment_e). Ensures 1:1
            #   all_gather pairing (1 per DUMMY, 1 per real).
            # - Cloud DUMMY with a real peer: do NOT skip - run the full middle
            #   forward so its all-toall sequence pairs 1:1 with the real
            #   cloud's middle segment.
            hidden_states = None
            _skip_head = False
            _skip_tail = False
            if not is_profile and not is_graph_capturing:
                _peer_bt = self._peer_batch_type_id
                if _peer_bt == 0:  # both DUMMY (idle or waiting)
                    # Skip both segments: no real data to all_gather.
                    # Only sync_metadata (already done) keeps the all_reduce
                    # pairing. This makes DUMMY instant (no MoE forward).
                    _skip_head = True
                    _skip_tail = True
                elif is_edge_device():
                    if _peer_bt in (2, 4):  # peer is tail (PL/DL)
                        _skip_head = True
                    elif _peer_bt in (1, 3):  # peer is head (PF/DF)
                        _skip_tail = True
                    # else: unmatched id - leave both False (run full forward)
                # else (cloud, peer real): no skip - full middle forward pairs
                # all-toall with the real cloud's middle.
                logger.error(
                    "[DPDBG] _dummy_run: dp_rank=%s role=%s peer_bt=%s "
                    "skip_head=%s skip_tail=%s",
                    self.dp_rank,
                    "edge" if is_edge_device() else "cloud",
                    _peer_bt, _skip_head, _skip_tail,
                )

            if not _skip_head:
                with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    in_profile_run=is_profile,
                    num_actual_tokens=num_tokens_padded,
                    aclgraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    model_instance=self.model,
                    has_sinks = self._has_sinks,
                    input_ids=input_ids,
                ):
                    # Build model kwargs for layer-sliced dummy execution.
                    _model_kwargs: dict[str, Any] = {}
                    if layer_slice_info is not None and self._edge_cloud_enabled:
                        _model_kwargs["layer_slice_start"] = (
                            layer_slice_info.start_layer + self.head_k
                        )
                        _model_kwargs["layer_slice_end"] = (
                            layer_slice_info.end_layer + self.head_k
                        )
                        if not layer_slice_info.is_last_slice:
                            _model_kwargs["layer_slice_return_intermediate"] = True
                    logger.error(
                        "[MODEL-FWD] _dummy_run _model_forward: "
                        "layer_slice_info=%s",
                        None if layer_slice_info is None else {
                            "start_layer": layer_slice_info.start_layer,
                            "end_layer": layer_slice_info.end_layer,
                            "total_slices": layer_slice_info.total_slices,
                            "slice_index": layer_slice_info.slice_index,
                            "is_first_slice": layer_slice_info.is_first_slice,
                            "is_last_slice": layer_slice_info.is_last_slice,
                        },
                    )
                    outputs = self._model_forward(
                        num_tokens_padded, input_ids, positions,
                        intermediate_tensors, inputs_embeds,
                        layer_slice_info=layer_slice_info,
                        **_model_kwargs,
                    )
                if self.use_aux_hidden_state_outputs:
                    hidden_states, _ = outputs
                elif isinstance(outputs, IntermediateTensors):
                    hidden_states = outputs["hidden_states"]
                else:
                    hidden_states = outputs
                # Layer-sliced dummy: save intermediate state for the
                # next slice (mirrors execute_model's layerwise state
                # saving for real prefill).
                if (
                    layer_slice_info is not None
                    and not layer_slice_info.is_last_slice
                    and isinstance(outputs, IntermediateTensors)
                ):
                    self._layerwise_intermediate = outputs
                    # Also save positions / attn_metadata for continuation.
                    self._layerwise_positions = positions
                    self._layerwise_attn_metadata = attn_metadata
                    self._layerwise_num_tokens_padded = num_tokens_padded
                    self._layerwise_num_tokens_across_dp = num_tokens_across_dp
                    self._layerwise_batch_desc = batch_desc
                # PD-separation diagnostic: log _dummy_run segment_a output
                if not is_profile and not is_graph_capturing:
                    _has_nan = bool(torch.isnan(hidden_states).any().item()) if hasattr(hidden_states, 'shape') and hidden_states.dim() > 0 else '?'
                    logger.error(
                        "[PD-DIAG] E. _dummy_run segment_a OUTPUT: "
                        "shape=%s has_nan=%s",
                        list(hidden_states.shape) if hasattr(hidden_states, 'shape') else '?',
                        _has_nan,
                    )
                dummy_compute_logits(hidden_states)

                if self.drafter:
                    self.drafter.dummy_run(
                        num_tokens=num_tokens_padded,
                        with_prefill=with_prefill,
                        num_reqs=num_reqs_padded,
                        num_tokens_across_dp=num_tokens_across_dp,
                        aclgraph_runtime_mode=cudagraph_runtime_mode,
                        batch_descriptor=batch_desc,
                        dummy_compute_logits=dummy_drafter_compute_logits,
                        in_graph_capturing=not force_attention,
                        is_profile=is_profile,
                    )
                if is_profile and self.dynamic_eplb:
                    target = self.model.language_model if hasattr(self.model, "language_model") else self.model
                    target.clear_all_moe_loads()
                if self.dynamic_eplb:
                    self.eplb_updator.forward_end()
                self._finalize_dump_data(dump=False)
                if self.use_compress and force_attention:
                    self.positions.fill_(0)
                    self._dsa_positions_cpu_buf.fill_(0)
            else:
                outputs = None

            # ========== Edge 设备特殊处理：Edge 首阶段需要执行最后一层 ==========
            if is_edge_device() and not _skip_tail:
                if outputs is not None:
                    assert isinstance(outputs, IntermediateTensors)

                # 重新准备 intermediate_tensors（与上文逻辑相同）
                intermediate_tokens = num_tokens_padded
                if enable_sp():
                    tp_size = get_tensor_model_parallel_world_size()
                    intermediate_tokens = (num_tokens_padded + tp_size - 1) // tp_size
                if self.intermediate_tensors is None:  # 增加deepseek-v4判断
                    max_actual_tokens = self.max_num_tokens
                    if enable_sp():
                        max_actual_tokens = (self.max_num_tokens + tp_size - 1) // tp_size
                        # 调用模型方法创建空的中间张量
                    self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                        batch_size=max_actual_tokens, dtype=self.dtype, device=self.device
                    )
                # 切片
                intermediate_tensors = IntermediateTensors(
                    {k: v[:intermediate_tokens] for k, v in self.intermediate_tensors.items()}
                )

                need_dummy_logits = not is_profile and lmhead_tp_enable()
                max_num_reqs_across_dp = max_num_reqs * self.uniform_decode_query_len
                dummy_indices = torch.zeros(max_num_reqs_across_dp, dtype=torch.int32)

                with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    in_profile_run=is_profile,
                    num_actual_tokens=num_tokens_padded,
                    aclgraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    model_instance=self.model,
                ):
                    outputs = self._model_forward(
                        num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds
                    )
                if self.use_aux_hidden_state_outputs:
                    hidden_states, _ = outputs
                elif isinstance(outputs, IntermediateTensors):
                    hidden_states = outputs["hidden_states"]
                else:
                    hidden_states = outputs
                # PD-separation diagnostic: log _dummy_run segment_ee output
                if not is_profile and not is_graph_capturing:
                    _has_nan = bool(torch.isnan(hidden_states).any().item()) if hasattr(hidden_states, 'shape') and hidden_states.dim() > 0 else '?'
                    logger.error(
                        "[PD-DIAG] E. _dummy_run segment_e OUTPUT: "
                        "shape=%s has_nan=%s",
                        list(hidden_states.shape) if hasattr(hidden_states, 'shape') else '?',
                        _has_nan,
                    )
                dummy_compute_logits(hidden_states)

                if is_profile and self.dynamic_eplb:
                    target = self.model.language_model if hasattr(self.model, "language_model") else self.model
                    target.clear_all_moe_loads()
                if self.dynamic_eplb:
                    self.eplb_updator.forward_end()

                self._finalize_dump_data(dump=False)
                if self.use_compress and force_attention:
                    self.positions.fill_(0)
                    self._dsa_positions_cpu_buf.fill_(0)
            return hidden_states, hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        output = None

        # For profile, have maximum num_reqs and that collectively have
        # maximum num_tokens.
        min_tokens_per_req = self.max_num_tokens // self.max_num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * self.max_num_reqs
        num_scheduled_tokens_list[-1] += self.max_num_tokens % self.max_num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        # TODO: need to rum a dummy sampler for generate task
        hidden_states = hidden_states[logit_indices]
        output = self.model.compute_logits(hidden_states)
        return output

    def profile_run(self) -> None:
        self.eplb_warmup()
        mc2_tokens_capacity = get_mc2_tokens_capacity()
        if self.max_num_tokens > mc2_tokens_capacity and select_moe_comm_method(
            mc2_tokens_capacity, self.vllm_config
        ) in {MoECommType.MC2, MoECommType.FUSED_MC2}:
            self._dummy_run(mc2_tokens_capacity, with_prefill=True, is_profile=True)
        origin_max_num_tokens = self.max_num_tokens
        # in the pcp scenario, the split sequence needs to be used for profile run
        # TODO: after the vllm pcp function is launched, this logic needs to be brought up to the community
        if self.pcp_size > 1:
            self.max_num_tokens = math.ceil(self.max_num_tokens / (self.pcp_size * 2)) * 2
        skip_mm_profile = (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "cloud"
            and self.supports_mm_inputs
        )
        original_supports_mm_inputs = self.supports_mm_inputs
        if skip_mm_profile:
            self.supports_mm_inputs = False
        try:
            super().profile_run()
        finally:
            self.supports_mm_inputs = original_supports_mm_inputs
            self.max_num_tokens = origin_max_num_tokens

    def eplb_warmup(self):
        if self.dynamic_eplb and not self.is_eplb_warmuped:
            self.is_eplb_warmuped = True
            self.eplb_adaptor = VllmEplbAdaptor(model=self.model)
            self.eplb_loader.set_adator(self.eplb_adaptor)
            self.eplb_updator.set_adaptor(self.eplb_adaptor)
            self.eplb_updator.warm_up_eplb()

    def load_model(self) -> None:
        # When the layer-slice runtime is enabled, install the qwen
        # forward-method patches before the model is constructed so the
        # rebound forward is what the loaded modules actually expose.
        # Loaded on demand to keep upstream vLLM unmodified for users that
        # don't enable this feature.
        self._layer_slice_enabled = os.path.exists(
            os.environ.get("VLLM_LAYER_SLICE_CONFIG", "layer_slice_config.yaml")
        )
        if self._layer_slice_enabled:
            import vllm_ascend.patch.models.qwen_layer_slice  # noqa: F401

        if self._edge_cloud_enabled:
            with DeviceMemoryProfiler() as m:
                self._load_model_edge_cloud()
            self.model_memory_usage = m.consumed_memory
            logger.info("Loading model weights took %.4f GB", m.consumed_memory / float(2**30))
            return

        logger.info("Starting to load model %s...", self.model_config.model)

        if self.ascend_config.mix_placement:
            # TODO: Enabling the mix placement in deepseek_v2.py
            # remove this part after the mix placement merged into vllm
            def mock_true():
                return True
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled = mock_true
            rocm_aiter_ops.is_fused_moe_enabled = mock_true

        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            if self.eplb_enable:
                def mock_pass(param1, param2):
                    return
                from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
                DefaultModelLoader._init_ep_weight_filter = mock_pass
            self.model: nn.Module = get_model(vllm_config=self.vllm_config)
            for name, _ in self.model.named_parameters():
                # sinks is a kind of parameter in attention
                # only set in weight name
                # TODO: remove it when fia merge in fiav2
                if "sink" in name:
                    self._has_sinks = True
                    break
            if self.dynamic_eplb:
                model_register(self.model)
            if self.drafter:
                logger.info("Loading drafter model...")
                if self.vllm_config.quant_config is not None:
                    patch_load_weights(self.vllm_config)
                with get_tp_context(self.drafter):
                    self.drafter.load_model(self.model)
                if self.use_aux_hidden_state_outputs:
                    from vllm.model_executor.models.interfaces import supports_eagle3
                    if not supports_eagle3(self.model):
                        raise RuntimeError(
                            "Model does not support EAGLE3 interface but "
                            "aux_hidden_state_outputs was requested"
                        )
                    aux_layers = self._get_eagle3_aux_layers_from_config()
                    if not aux_layers:
                        aux_layers = self.model.get_eagle3_default_aux_hidden_state_layers()
                    self.model.set_aux_hidden_state_layers(aux_layers)

            if self.lora_config:
                self.model = self.load_lora_model(self.model, self.vllm_config, self.device)
        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB", m.consumed_memory / float(2**30))

        # wrap the model with full graph wrapper if needed.
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.update_stream: torch.npu.Stream = torch.npu.Stream()
            self.model = ACLGraphWrapper(
                self.model,
                self.vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
                use_eagle=self.use_eagle,
                enable_enpu=self.enable_enpu,
            )

        if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            self._start_dump_data()

    def _start_dump_data(self) -> None:
        if self.debugger is None or self._debugger_started:
            return
        self.debugger.start(self.model)
        self._debugger_started = True

    def _finalize_dump_data(self, **kwargs) -> None:
        if self.debugger is None or not self._debugger_started:
            return
        if hasattr(self.debugger, "stop"):
            self.debugger.stop()
            self._debugger_started = False

        self.debugger.step(**kwargs)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self._mamba_bufs = None
        self._mamba_copy_bufs = None

        # For embedding_only edge, skip KV cache tensor allocation and
        # attention backend initialization. The edge does not execute any
        # attention layers; keeping a full kv_cache_config is only for the
        # scheduler to correctly schedule requests.
        if (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.mode == "embedding_only"
            and self.edge_cloud_cfg.role == "edge"
        ):
            # Edge does not execute any attention layers, but downstream code
            # (e.g. execute_model) still iterates over attn_groups using
            # kv_cache_groups as the outer loop.  Keep a list of empty lists
            # so that len(attn_groups) == len(kv_cache_groups) and inner
            # loops simply execute zero times.
            self.attn_groups = [
                [] for _ in range(len(kv_cache_config.kv_cache_groups))
            ]
            self.use_hybrid_blocks = False
            self.need_accepted_tokens = False
            self.may_reinitialize_input_batch(kv_cache_config)
            self.kv_cache = {}
            # Still initialize cudagraph dispatcher keys and ACL graph params,
            # otherwise edge segments (and edge-cloud MTP segments) have no
            # graph params and ACL graph capture/replay can hang.
            self._check_and_update_cudagraph_mode(
                [], kv_cache_config.kv_cache_groups
            )
            logger.info(
                "[EdgeCloud] embedding_only edge skipped KV cache tensor "
                "allocation and attention backend initialization."
            )
            return

        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        # NOTE(cmq): initialize_attn_backend must before using self.attn_groups
        self.initialize_attn_backend(kv_cache_config)
        self.use_hybrid_blocks = len(self.attn_groups) > 1
        # NOTE: Currently, we determine whether we need `num_accepted_tokens` through `MambaSpec`.
        self.need_accepted_tokens = any(
            [isinstance(attn_group[0].kv_cache_spec, MambaSpec) for attn_group in self.attn_groups]
        )

        self.may_reinitialize_input_batch(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)
        # TODO: refactor the logic of attention
        # Initialize drafter attention group initialization
        if self.speculative_config and (
            self.speculative_config.use_eagle() or self.speculative_config.uses_draft_model()
        ):
            assert isinstance(self.drafter, AscendEagleProposer | AscendDflashProposer | AscendDraftModelProposer)
            block_size = (self.kernel_block_sizes[0] if isinstance(
            self.kernel_block_sizes, list) else self.kernel_block_sizes)
            self.drafter.initialize_attn_backend(kv_cache_config, block_size)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    def _bind_routed_experts_capturer(self, capturer) -> None:
        # Upstream binds via ``module.router.set_capture_fn(...)`` on
        # FusedMoE layers whose router is a ``BaseRouter``. Ascend's
        # ``select_experts`` does not go through ``BaseRouter``, so the
        # upstream hook never fires. Instead, stash the capturer as a
        # plain attribute on every FusedMoE layer; ``apply()`` reads it
        # back on the hot path. Only used on vLLM main (PR #39568+);
        # the 0.20.2 path uses the ``RoutedExpertsCapturer.get_instance``
        # singleton and never calls this method.
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        for module in self.compilation_config.static_forward_context.values():
            if isinstance(module, FusedMoE):
                module._ascend_routed_experts_capturer = capturer

    def _align_memory(self, tensor: torch.Tensor, alignment: int) -> torch.Tensor:
        data_ptr = tensor.data_ptr()
        aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
        offset = (aligned_addr - data_ptr) // tensor.element_size()
        return tensor[int(offset) :]

    def initialize_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        # Change the memory buffer to the desired shape
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        if self.model_config.hf_text_config.model_type == "deepseek_v4":
            from vllm_ascend.utils import extract_dsv4_layer_index

            assert len(self.kv_caches) == 0
            for layer_name in sorted(
                    kv_caches,
                    key=lambda name: (extract_dsv4_layer_index(
                        self.model_config.hf_text_config, name), name)):
                self.kv_caches.append(kv_caches[layer_name])
            for layer_name, kv_cache in kv_caches.items():
                self.compilation_config.static_forward_context[
                    layer_name].kv_cache = [kv_cache]
        else:
            from vllm.v1.worker.utils import bind_kv_cache

            num_attn_module = 2 if self.model_config.hf_text_config.model_type == "longcat_flash" else 1
            bind_kv_cache(kv_caches, self.compilation_config.static_forward_context, self.kv_caches, num_attn_module)

        if self.enable_hamming_sparse is True:
            from vllm_ascend.worker.kvcomp_utils import init_and_bind_hashk_cache
            init_and_bind_hashk_cache(
                kv_caches=kv_caches,
                num_attn_module=num_attn_module,
                vllm_config=self.vllm_config,
                device=self.device,
                compilation_config=self.compilation_config,
                kvcomp_meta_data=self.kvcomp_meta_data
            )

        return kv_caches

    def _get_layer_kv_cache_specs(self, kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
        layer_kv_cache_spec: dict[str, KVCacheSpec] = {}
        for group_kv_cache_spec in kv_cache_config.kv_cache_groups:
            group_spec = group_kv_cache_spec.kv_cache_spec
            for layer_name in group_kv_cache_spec.layer_names:
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec[layer_name] = group_spec.kv_cache_specs[layer_name]
                else:
                    layer_kv_cache_spec[layer_name] = group_spec
        return layer_kv_cache_spec

    def _get_attention_kv_cache_dims(self, layer_name: str, kv_cache_spec: AttentionSpec) -> tuple[int, int]:
        if isinstance(kv_cache_spec, MLAAttentionSpec):
            attn_layers = get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,
                [layer_name],
            )
            attn_layer = attn_layers[layer_name]
            if isinstance(attn_layer, MLAAttention):
                # DeepSeek MLA: K=kv_lora_rank, V=qk_rope_head_dim
                return attn_layer.kv_lora_rank, attn_layer.qk_rope_head_dim
            # CacheOnlyAttentionLayer uses MLAAttentionSpec but isn't MLAAttention
            if isinstance(attn_layer, CacheOnlyAttentionLayer):
                return kv_cache_spec.head_size, kv_cache_spec.head_size
            # DeepseekV4IndexerCache (and its ascend subclass) is also cache-only
            # but does not inherit CacheOnlyAttentionLayer.
            if type(attn_layer).__name__ in (
                "DeepseekV4IndexerCache",
                "AscendDeepseekV4IndexerCache",
            ):
                return kv_cache_spec.head_size, kv_cache_spec.head_size
            # DSAAttention uses MLAAttentionSpec with unified head_size for K and V.
            if type(attn_layer).__name__ == "DSAAttention":
                return kv_cache_spec.head_size, kv_cache_spec.head_size
            raise TypeError(
                f"Expected MLAAttention layer for {layer_name}, got {type(attn_layer).__name__}."
            )

        head_size_v = kv_cache_spec.head_size_v if hasattr(kv_cache_spec, "head_size_v") else kv_cache_spec.head_size
        return kv_cache_spec.head_size, head_size_v

    def _allocate_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        NOTE: To support prefill disaggregation, we need to split kvcache tensor into
        k_cache and v cache, and the addr of both are aligned by 2M

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
            dict[str, tuple(torch.Tensor, torch.Tensor)] A map between layer names
            to their corresponding memory buffer for K cache and V cache.
        """
        # init kv cache tensors
        kv_cache_raw_tensors: dict[str, torch.Tensor | torch.Tensor | None | None] = {}
        # prefill disaggregation need the addr of cache tensor be aligned with 2M
        alignment = 2 * 1024 * 1024
        layer_kv_cache_spec = self._get_layer_kv_cache_specs(kv_cache_config)
        # If some tensors are shared by linear layers and attention layers,
        # the same tensor format must be maintained even if some layers
        # have only linear or attention layers, for example, the mtp layer.
        self.hybrid_with_attn_and_mamba = False
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            use_mamba, use_attn = False, False
            for layer_name in kv_cache_tensor.shared_by:
                if isinstance(layer_kv_cache_spec[layer_name], MambaSpec):
                    use_mamba = True
                if isinstance(layer_kv_cache_spec[layer_name], AttentionSpec):
                    use_attn = True
            self.hybrid_with_attn_and_mamba = self.hybrid_with_attn_and_mamba or (use_mamba and use_attn)
            for idx in range(len(kv_cache_tensor.shared_by)):
                layer_name = kv_cache_tensor.shared_by[idx]
                # Single tensor path for: mamba, hybrid attn-mamba, or cache_only_layers
                if (
                    "linear_attn" in layer_name
                    or self.hybrid_with_attn_and_mamba
                    or "cache_only_layers" in layer_name
                ) and layer_name not in kv_cache_raw_tensors:
                    # for mamba linear attention, attn-linear hybrid, or cache_only_layers (extract_hidden_states)
                    if self.vllm_config.kv_transfer_config is None:
                        tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
                    else:
                        cache_size_aligned = kv_cache_tensor.size + alignment
                        tensor = torch.zeros(cache_size_aligned, dtype=torch.int8, device=self.device)
                        tensor = self._align_memory(tensor, alignment)[: kv_cache_tensor.size]

                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache for all shared layers
                        kv_cache_raw_tensors[layer_name_inner] = tensor
                elif "attn" in layer_name and self.use_compress and layer_name not in kv_cache_raw_tensors.keys(
                ):
                    if self.vllm_config.kv_transfer_config is None:
                        tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
                    else:
                        cache_size_aligned = kv_cache_tensor.size + alignment
                        tensor = torch.zeros(cache_size_aligned, dtype=torch.int8, device=self.device)
                        tensor = self._align_memory(tensor, alignment)[: kv_cache_tensor.size]
                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        kv_cache_raw_tensors[layer_name_inner] = tensor
                elif "attn" in layer_name and layer_name not in kv_cache_raw_tensors and not use_mamba:
                    # NOTE: We need to init k cache tensor (nope cache tensor in mla) and
                    # v cache tensor (rope cache tensor in mla) separately to support prefill disaggregation,
                    # as it only support the 0-dim of kv_cache is `num_blocks`.
                    # For deepseek mla, we need to spilt cache tensor accrodding to the nope head dim
                    # and rope head dim.
                    current_kv_cache_spec = layer_kv_cache_spec[layer_name]
                    assert isinstance(current_kv_cache_spec, AttentionSpec)

                    if self.use_sparse:
                        # for deepseek v3.2, we split the kv cache according to the corresponding ratio
                        kv_cache_spec = layer_kv_cache_spec[layer_name]
                        current_sparse_c8 = kv_cache_spec_uses_sparse_c8(kv_cache_spec)
                        sparse_kv_cache_ratio = kv_cache_spec.sparse_kv_cache_ratio
                        k_tensor_split_factor = sparse_kv_cache_ratio[0]
                        v_tensor_split_factor = sparse_kv_cache_ratio[1]
                        dsa_k_tensor_split_factor = sparse_kv_cache_ratio[2]
                        dsa_k_scale_tensor_split_factor = sparse_kv_cache_ratio[3] if current_sparse_c8 else None
                    else:
                        k_dim, v_dim = self._get_attention_kv_cache_dims(layer_name, current_kv_cache_spec)
                        assert k_dim > 0 and v_dim > 0
                        kv_head_dim_list = [
                            k_dim,
                            v_dim,
                        ]
                        if enable_fa_quant(self.vllm_config):
                            k_tensor_split_factor, v_tensor_split_factor = (
                                self.vllm_config.quant_config.get_kv_quant_split_factor(layer_name, kv_head_dim_list)
                            )
                        else:
                            k_tensor_split_factor, v_tensor_split_factor = calc_split_factor(kv_head_dim_list)

                    k_tensor_size = int(kv_cache_tensor.size // k_tensor_split_factor)
                    v_tensor_size = int(kv_cache_tensor.size // v_tensor_split_factor)
                    dsa_k_tensor_size = None
                    dsa_k_scale_tensor_size = None
                    #### for deepseek sparse attention
                    if self.use_sparse:
                        dsa_k_tensor_size = int(kv_cache_tensor.size // dsa_k_tensor_split_factor)
                    if self.use_sparse and current_sparse_c8:
                        dsa_k_scale_tensor_size = int(kv_cache_tensor.size // dsa_k_scale_tensor_split_factor)

                    # for other attentions, e.g., self_attn, sliding window attn
                    if self.vllm_config.kv_transfer_config is None:
                        k_tensor = torch.zeros(k_tensor_size, dtype=torch.int8, device=self.device)
                        v_tensor = torch.zeros(v_tensor_size, dtype=torch.int8, device=self.device)
                        #### for deepseek sparse attention
                        if dsa_k_tensor_size is not None:
                            dsa_k_tensor = torch.zeros(dsa_k_tensor_size, dtype=torch.int8, device=self.device)
                        if dsa_k_scale_tensor_size is not None:
                            dsa_k_scale_tensor = torch.zeros(
                                dsa_k_scale_tensor_size, dtype=torch.int8, device=self.device
                            )
                    else:
                        k_tensor = torch.zeros(k_tensor_size + alignment, dtype=torch.int8, device=self.device)
                        v_tensor = torch.zeros(v_tensor_size + alignment, dtype=torch.int8, device=self.device)
                        k_tensor = self._align_memory(k_tensor, alignment)[:k_tensor_size]
                        v_tensor = self._align_memory(v_tensor, alignment)[:v_tensor_size]
                        #### for deepseek sparse attention
                        if dsa_k_tensor_size is not None:
                            dsa_k_tensor = torch.zeros(
                                dsa_k_tensor_size + alignment, dtype=torch.int8, device=self.device
                            )
                            dsa_k_tensor = self._align_memory(dsa_k_tensor, alignment)[:dsa_k_tensor_size]
                        if dsa_k_scale_tensor_size is not None:
                            dsa_k_scale_tensor = torch.zeros(
                                dsa_k_scale_tensor_size + alignment, dtype=torch.int8, device=self.device
                            )
                            dsa_k_scale_tensor = self._align_memory(
                                dsa_k_scale_tensor, alignment
                            )[:dsa_k_scale_tensor_size]

                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the attn kvcache for all shared layers
                        if "attn" in layer_name_inner and "linear_attn" not in layer_name_inner:
                            if self.use_sparse:
                                if current_sparse_c8:
                                    kv_cache_raw_tensors[layer_name_inner] = (
                                        k_tensor, v_tensor, dsa_k_tensor, dsa_k_scale_tensor
                                    )
                                else:
                                    kv_cache_raw_tensors[layer_name_inner] = (k_tensor, v_tensor, dsa_k_tensor)
                            else:
                                kv_cache_raw_tensors[layer_name_inner] = (k_tensor, v_tensor)
        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), "Some layers are not correctly initialized"

        return kv_cache_raw_tensors

    def _adjust_kv_layout(
        self,
        raw_tensor: torch.Tensor,
        kv_cache_shape_list: list[int],
        kv_cache_dtype_list: list[int],
        page_size_bytes: int,
    ):
        reshaped_kv_tensors = []
        storage_offset_bytes = raw_tensor.storage_offset()
        for shape, dtype in zip(kv_cache_shape_list, kv_cache_dtype_list):
            dtype_size = get_dtype_size(dtype)
            num_element_per_page = (
                page_size_bytes // dtype_size
            )

            stride = torch.empty(shape).stride()
            target_stride = (num_element_per_page, *stride[1:])
            assert storage_offset_bytes % dtype_size == 0
            tensor = torch.as_strided(
                raw_tensor.view(dtype),
                size=shape,
                stride=target_stride,
                storage_offset=storage_offset_bytes // dtype_size,
            )
            reshaped_kv_tensors.append(tensor)
            storage_offset_bytes += stride[0] * dtype_size
        return reshaped_kv_tensors


    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        layer_kv_cache_spec = self._get_layer_kv_cache_specs(kv_cache_config)
        for group in self._kv_cache_spec_attn_group_iterator():
            attn_backend = group.backend
            current_kv_cache_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue

                current_kv_cache_spec = layer_kv_cache_spec[layer_name]

                # TODO: remove this after the OOM issue is located and fixed, otherwise, some model may
                # encounter OOM issue
                if self.use_compress and isinstance(current_kv_cache_spec, (MLAAttentionSpec, SlidingWindowMLASpec)):
                    kv_tensor = kv_cache_raw_tensors[layer_name]
                    sum_page_size_bytes = kv_tensor.numel()
                    num_blocks = sum_page_size_bytes // current_kv_cache_spec.page_size_bytes
                    assert num_blocks == kv_cache_config.num_blocks, \
                        f"num_blocks: {num_blocks} should be equal to " \
                        f"kv_cache_config.num_blocks: {kv_cache_config.num_blocks}"
                    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                        num_blocks, current_kv_cache_spec.block_size,
                        current_kv_cache_spec.num_kv_heads,
                        current_kv_cache_spec.head_size)
                    kv_cache_shape_list = [kv_cache_shape]
                    kv_cache_dtype_list = [current_kv_cache_spec.dtype]

                    if hasattr(current_kv_cache_spec, "scale_dim") and current_kv_cache_spec.scale_dim != 0:
                        indexer_k_shape = kv_cache_shape
                        indexer_scale_shape = self.attn_backend.get_kv_cache_shape(
                                                num_blocks, current_kv_cache_spec.block_size,
                                                current_kv_cache_spec.num_kv_heads,
                                                current_kv_cache_spec.scale_dim
                                                )
                        kv_cache_shape_list = [indexer_k_shape, indexer_scale_shape]
                        kv_cache_dtype_list = [current_kv_cache_spec.dtype, current_kv_cache_spec.scale_dtype]

                    kv_cache = self._adjust_kv_layout(kv_tensor,
                                           kv_cache_shape_list,
                                           kv_cache_dtype_list,
                                           current_kv_cache_spec.page_size_bytes,
                                           )

                    kv_caches[layer_name] = kv_cache
                elif isinstance(current_kv_cache_spec, AttentionSpec):
                    # cache_only_layers (extract_hidden_states) are allocated
                    # as a single tensor by the branch at the top of
                    # _allocate_kv_cache_tensors; route them to the dedicated
                    # elif branch below before the sparse branch tries to
                    # unpack them as a (k, v, dsa_k[, scale]) tuple.
                    if self.use_sparse and "cache_only_layers" not in layer_name:
                        current_sparse_c8 = kv_cache_spec_uses_sparse_c8(current_kv_cache_spec)
                        if current_sparse_c8:
                            raw_k_tensor, raw_v_tensor, raw_dsa_k_tensor, raw_dsa_k_scale_tensor = kv_cache_raw_tensors[  # type: ignore
                                layer_name]
                            assert raw_dsa_k_tensor is not None
                            assert raw_dsa_k_scale_tensor is not None
                            sum_page_size_bytes = (
                                raw_k_tensor.numel()
                                + raw_v_tensor.numel()
                                + raw_dsa_k_tensor.numel()
                                + raw_dsa_k_scale_tensor.numel()
                            )
                        else:
                            raw_k_tensor, raw_v_tensor, raw_dsa_k_tensor = kv_cache_raw_tensors[  # type: ignore
                                layer_name]
                            assert raw_dsa_k_tensor is not None
                            sum_page_size_bytes = raw_k_tensor.numel() + raw_v_tensor.numel() + raw_dsa_k_tensor.numel()
                    elif self.use_hybrid_blocks and self.hybrid_with_attn_and_mamba:
                        # Currently, we ensure that the same kvcache format is used even if there
                        # is no shared layer, such as the full attention mtp layer of qwen3.5, etc.
                        raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[layer_name], kv_cache_raw_tensors[layer_name]
                        sum_page_size_bytes = raw_k_tensor.numel()
                    elif "cache_only_layers" in layer_name:
                        # Single tensor for extract_hidden_states (no K/V split)
                        raw_tensor = kv_cache_raw_tensors[layer_name]
                        assert raw_tensor is not None
                        assert raw_tensor.numel() % current_kv_cache_spec.page_size_bytes == 0
                        num_blocks = raw_tensor.numel() // current_kv_cache_spec.page_size_bytes
                        assert num_blocks >= kv_cache_config.num_blocks
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks,
                            current_kv_cache_spec.block_size,
                            current_kv_cache_spec.num_kv_heads,
                            current_kv_cache_spec.head_size,
                        )
                        k_cache = raw_tensor.view(current_kv_cache_spec.dtype).view(kv_cache_shape)
                        kv_caches[layer_name] = k_cache
                        continue  # Skip the rest of the AttentionSpec handling
                    else:
                        raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[  # type: ignore
                            layer_name
                        ]
                        sum_page_size_bytes = raw_k_tensor.numel() + raw_v_tensor.numel()
                    assert raw_k_tensor is not None
                    assert raw_v_tensor is not None
                    assert sum_page_size_bytes % current_kv_cache_spec.page_size_bytes == 0
                    num_blocks = sum_page_size_bytes // current_kv_cache_spec.page_size_bytes

                    # `num_blocks` is the number of blocks the model runner can use.
                    # `kv_cache_config.num_blocks` is the number of blocks that
                    # KVCacheManager may allocate.
                    # Since different GPUs may have different number of layers and
                    # different memory capacities, `num_blocks` can be different on
                    # different GPUs, and `kv_cache_config.num_blocks` is set to
                    # the min of all `num_blocks`. Verify it here.
                    assert num_blocks >= kv_cache_config.num_blocks

                    if hasattr(attn_backend, "get_supported_kernel_block_sizes") and self.use_hybrid_blocks:
                        block_size = attn_backend.get_supported_kernel_block_sizes()[0]

                        block_size_chunk = current_kv_cache_spec.block_size // block_size
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks * block_size_chunk,
                            block_size,
                            current_kv_cache_spec.num_kv_heads,
                            current_kv_cache_spec.head_size,
                        )
                        if self.hybrid_with_attn_and_mamba:
                            if not isinstance(current_kv_cache_spec, MLAAttentionSpec):
                                attn_tensor_page_size = int(np.prod(kv_cache_shape[1:])) * get_dtype_size(
                                    current_kv_cache_spec.dtype
                                )
                                conv_block_padding_size = raw_k_tensor.numel() - attn_tensor_page_size * 2
                                raw_kv_tensor = raw_k_tensor[conv_block_padding_size:]
                                raw_k_tensor = raw_kv_tensor[:attn_tensor_page_size]
                                raw_v_tensor = raw_kv_tensor[attn_tensor_page_size:]
                            else:
                                k_dim, v_dim = self._get_attention_kv_cache_dims(layer_name, current_kv_cache_spec)
                                nope_page_size = int(np.prod(kv_cache_shape[:-1])) * k_dim * get_dtype_size(
                                    current_kv_cache_spec.dtype
                                )
                                rope_page_size = int(np.prod(kv_cache_shape[:-1])) * v_dim * get_dtype_size(
                                    current_kv_cache_spec.dtype
                                )
                                conv_block_padding_size = raw_k_tensor.numel() - nope_page_size - rope_page_size
                                raw_kv_tensor = raw_k_tensor[conv_block_padding_size:]
                                raw_k_tensor = raw_kv_tensor[:nope_page_size]
                                raw_v_tensor = raw_kv_tensor[nope_page_size:]
                    else:
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks,
                            current_kv_cache_spec.block_size,
                            current_kv_cache_spec.num_kv_heads,
                            current_kv_cache_spec.head_size,
                        )
                    if not isinstance(current_kv_cache_spec, MLAAttentionSpec):
                        k_shape = kv_cache_shape[1:]
                        if hasattr(current_kv_cache_spec, "head_size_v"):
                            v_shape = (*kv_cache_shape[1:-1], current_kv_cache_spec.head_size_v)
                        else:
                            v_shape = k_shape
                    else:
                        # k_cache: nope_cache    v_cache: rope_cache
                        mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
                        k_dim, v_dim = self._get_attention_kv_cache_dims(layer_name, current_kv_cache_spec)
                        k_shape = (
                            mla_num_blocks,
                            mla_block_size,
                            num_kv_heads,
                            k_dim,
                        )
                        v_shape = (
                            mla_num_blocks,
                            mla_block_size,
                            num_kv_heads,
                            v_dim,
                        )
                    k_cache_dtype = v_cache_dtype = current_kv_cache_spec.dtype
                    if enable_fa_quant(self.vllm_config):
                        k_cache_dtype, v_cache_dtype = self.vllm_config.quant_config.get_kv_quant_dtype(
                            layer_name, current_kv_cache_spec.dtype, self.model_config
                        )
                    k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape)
                    v_cache = raw_v_tensor.view(v_cache_dtype).view(v_shape)

                    if self.use_sparse:
                        dsa_k_cache_shape = (
                            num_blocks,
                            current_kv_cache_spec.block_size,
                            current_kv_cache_spec.num_kv_heads,
                            self.model_config.hf_text_config.index_head_dim,
                        )
                        if current_sparse_c8:
                            # dsa_k
                            dsa_k_cache = raw_dsa_k_tensor.view(self.c8_k_cache_dtype).view(dsa_k_cache_shape)
                            # dsa_k_scale
                            dsa_k_scale_cache_shape = (
                                num_blocks,
                                current_kv_cache_spec.block_size,
                                current_kv_cache_spec.num_kv_heads,
                                1,
                            )
                            assert raw_dsa_k_scale_tensor is not None
                            dsa_k_scale_cache = (
                                raw_dsa_k_scale_tensor
                                .view(self.c8_k_scale_cache_dtype)
                                .view(dsa_k_scale_cache_shape)
                            )
                            kv_caches[layer_name] = (k_cache, v_cache, dsa_k_cache, dsa_k_scale_cache)
                        else:
                            # dsa_k
                            dsa_k_cache = raw_dsa_k_tensor.view(current_kv_cache_spec.dtype).view(dsa_k_cache_shape)
                            kv_caches[layer_name] = (k_cache, v_cache, dsa_k_cache)
                    else:
                        kv_caches[layer_name] = (k_cache, v_cache)
                elif isinstance(current_kv_cache_spec, MambaSpec):
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor is not None
                    assert raw_tensor.numel() % current_kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel() // current_kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks

                    # `num_blocks` is the number of blocks the model runner can use.
                    # `kv_cache_config.num_blocks` is the number of blocks that
                    # KVCacheManager may allocate.
                    # Since different GPUs may have different number of layers and
                    # different memory capacities, `num_blocks` can be different on
                    # different GPUs, and `kv_cache_config.num_blocks` is set to
                    # the min of all `num_blocks`. Verify it here.

                    state_tensors = []
                    target_idx = 0
                    start_idx = 0
                    # NOTE(zxr): in order to keep all tensor contiguous, we align ssm and kv block
                    # with same page size, so have to add extra padding block for kv, the overall
                    # layout of hybrid kv_cache on Ascend is:
                    # tensor1: [(kv_padding), conv           , ...]
                    # tensor2: [k           , ssm            , ...]
                    # tensor3: [v           , (mamba_padding), ...]
                    for shape, dtype in zip(current_kv_cache_spec.shapes, current_kv_cache_spec.dtypes):
                        # normally, there is conv state and ssm state in this loop. And there is only
                        # a conv state in some special models.
                        target_shape = (num_blocks, *shape)

                        target_idx += math.prod(target_shape) * get_dtype_size(dtype)
                        tensor = raw_tensor[start_idx:target_idx].view(dtype).view(target_shape)
                        start_idx = target_idx
                        state_tensors.append(tensor)
                    kv_caches[layer_name] = state_tensors
                else:
                    raise ValueError("Unknown KV cache spec type.")

        return kv_caches

    def may_reinitialize_input_batch(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
            if not isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec)
        ]

        # Generate kernel_block_sizes that matches each block_size
        # For attention backends that support virtual block splitting,
        # use the supported block sizes from the backend
        # For other backends (like Mamba), use [0] (no splitting)
        self.kernel_block_sizes = []
        for kv_cache_group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            if self.pcp_size > 1:
                self.pcp_manager.initialize_slot_mapping()
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                # All layers in the UniformTypeKVCacheSpecs have the same type,
                # Pick an arbitrary one to dispatch.
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            elif isinstance(kv_cache_spec, AttentionSpec):
                # This is an attention backend that supports virtual
                # block splitting. Get the supported block sizes from
                # the backend.
                attn_groups = self.attn_groups[kv_cache_group_id]
                backends = [attn_group.backend for attn_group in attn_groups]
                kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
                selected_kernel_size = select_common_block_size(
                    kv_manager_block_size, backends
                )
                self.kernel_block_sizes.append([selected_kernel_size])
            else:
                # This is likely Mamba or other non-attention cache,
                # no splitting.
                # NOTE: set kernel_block_sizes to 0 to disable slotmapping computation
                # of mamba block. In this case, BlockTable.block_size will never equal
                # to kernel_block_sizes[0]
                self.kernel_block_sizes.append([0])

        max_num_blocks = []
        max_model_len = max(self.max_model_len, self.max_encoder_len)
        for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            if isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            max_num_blocks_per_req = cdiv(max_model_len, block_sizes[i] * get_total_cp_world_size())
            if isinstance(kv_cache_group.kv_cache_spec, MambaSpec):
                if self.cache_config.enable_prefix_caching:
                    mamba_blocks_per_req = max_num_blocks_per_req
                else:
                    max_chunks = cdiv(
                        max_model_len,
                        self.cache_config.block_size * get_total_cp_world_size(),
                    )
                    mamba_blocks_per_req = max(max_num_blocks_per_req, max_chunks)
                mamba_blocks_per_req += kv_cache_group.kv_cache_spec.num_speculative_blocks
                max_num_blocks_per_req = max(max_num_blocks_per_req, mamba_blocks_per_req)
            max_num_blocks.append(max_num_blocks_per_req)

        if (block_sizes != [self.cache_config.block_size]
                or self.kernel_block_sizes != [[self.cache_config.block_size]]
                or len(kv_cache_config.kv_cache_groups) > 1):
            assert self.offload_config.uva.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details."
            )
            self.input_batch = NPUInputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                is_pooling_model=self.is_pooling_model,
                num_speculative_tokens=(
                    self.vllm_config.speculative_config.num_speculative_tokens
                    if self.vllm_config.speculative_config
                    else 0
                ),
                kernel_block_sizes=self.kernel_block_sizes,
                max_num_blocks_per_req=max_num_blocks,
                kv_cache_groups=kv_cache_config.kv_cache_groups,
            )

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]], set[type[AttentionBackend]]]:
            layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase, kv_cache_group_spec.layer_names)
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()
                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                key = (full_cls_name, layer_kv_cache_spec)
                attn_backends[key] = AttentionGroupKey(attn_backend, layer_kv_cache_spec)
                attn_backend_layers[key].append(layer_name)
            return (
                {attn_backends[k]: v for k, v in attn_backend_layers.items()},
                set(group_key.attn_backend for group_key in attn_backends.values()),
            )

        def create_attn_groups(
            attn_backends_map: dict[AttentionBackend, list[str]], kv_cache_group_id: int
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for (attn_backend, kv_cache_spec), layer_names in attn_backends_map.items():
                attn_metadata_builders = []
                attn_metadata_builders.append(
                    attn_backend.get_builder_cls()(
                        kv_cache_spec,
                        layer_names,
                        self.vllm_config,
                        self.device,
                    )
                )
                attn_group = AttentionGroup(
                    attn_backend, layer_names, kv_cache_spec, kv_cache_group_id, attn_metadata_builders
                )
                attn_groups.append(attn_group)
            return attn_groups

        attention_backend_maps = []
        attention_backend_list = []
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            attention_backend_maps.append(attn_backends[0])
            attention_backend_list.append(attn_backends[1])

        self._check_and_update_cudagraph_mode(attention_backend_list, kv_cache_config.kv_cache_groups)

        for i, attn_backend_map in enumerate(attention_backend_maps):
            self.attn_groups.append(create_attn_groups(attn_backend_map, i))

        # Calculate reorder batch threshold (if needed)
        self.calculate_reorder_batch_threshold()

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Check that if any backends reorder batches; that the reordering
        is compatible (e.g., decode threshold is the same)
        """
        for group in self._attn_group_iterator():
            attn_metadata_builder_i = group.get_metadata_builder()
            if hasattr(attn_metadata_builder_i, "reorder_batch_threshold"):  # noqa
                # check that if any backends reorder batches; that the reordering
                # is compatible (e.g., decode threshold is the same)
                reorder_batch_threshold_i = attn_metadata_builder_i.reorder_batch_threshold
                if reorder_batch_threshold_i is not None:  # noqa
                    if self.reorder_batch_threshold is not None:
                        if reorder_batch_threshold_i != self.reorder_batch_threshold:
                            raise ValueError(
                                f"Attention backend reorders decodes with "
                                f"threshold {reorder_batch_threshold_i} but other "
                                f"backend uses threshold "
                                f"{self.reorder_batch_threshold}"
                            )
                    else:
                        self.reorder_batch_threshold = reorder_batch_threshold_i  # noqa

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        # embedding_only: edge has no local attention layers, so its
        # static_forward_context is empty and the normal path returns {}.
        # Returning {} here lets the cloud's fresh spec (after weights
        # processing) drive the merged spec, avoiding stale dimensions
        # (e.g. head_size changed by quantization).
        if (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.mode == "embedding_only"
            and self.edge_cloud_cfg.role == "edge"
        ):
            return {}

        if (
            has_ec_transfer()
            and get_ec_transfer().is_producer
            and not self._edge_cloud_enabled
        ):
            return {}

        kv_cache_spec: dict[str, list[KVCacheSpec]] = defaultdict(list)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        # NOTE: Must process Attention/MLAAttention before MambaBase to maintain
        # ordering expected by graph parameter update logic in attention backends.
        mamba_layers: dict[str, MambaBase] = {}
        attn_layer_names = set()
        for layer_name, attn_module in attn_layers.items():
            if (isinstance(attn_module, Attention)
                    and (kv_tgt_layer := attn_module.kv_sharing_target_layer_name) is not None):
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue
            elif self.use_compress:
                # Skip modules that don't need KV cache (eg encoder-only attention)
                if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                    kv_cache_spec[layer_name] = spec
            elif isinstance(attn_module, Attention):
                if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                    kv_cache_spec[layer_name] = spec
                    attn_layer_names.add(layer_name)

            elif isinstance(attn_module, MLAAttention):
                if self.use_sparse:
                    # `MLAAttentionSpec` is temporarily patched to `AscendMLAAttentionSpec`.
                    # Re-importing it at runtime will therefore resolve to the patched class.
                    # Rename it here to make this behavior explicit.
                    from vllm.v1.kv_cache_interface import MLAAttentionSpec as AscendMLAAttentionSpec
                    # TODO(rjg-lyh): when kv_cache_spec's refactor is ready,
                    # implement it by creating a new kv_cache_spec class
                    kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
                        block_size=self.block_size,
                        num_kv_heads=1,
                        head_size=sum(self.sparse_head_dim),
                        sparse_head_dim=self.sparse_head_dim,
                        dtype=self.kv_cache_dtype,
                        cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
                        cache_sparse_c8=self.ascend_config.is_sparse_c8_layer(layer_name),
                    )
                elif spec := attn_module.get_kv_cache_spec(self.vllm_config):
                    from vllm.v1.kv_cache_interface import MLAAttentionSpec as AscendMLAAttentionSpec
                    if getattr(attn_module.impl, "fa_quant_layer", False):
                        head_size = attn_module.head_size + attn_module.qk_rope_head_dim
                        dtype, cache_dtype_str = attn_module.impl.dtype, None
                    else:
                        head_size, dtype, cache_dtype_str = spec.head_size, spec.dtype, spec.cache_dtype_str
                    kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
                        block_size=spec.block_size,
                        num_kv_heads=spec.num_kv_heads,
                        head_size=head_size,
                        dtype=dtype,
                        cache_dtype_str=cache_dtype_str,
                    )
                    attn_layer_names.add(layer_name)

            elif isinstance(attn_module, MambaBase):
                mamba_layers[layer_name] = attn_module

            elif isinstance(attn_module, CacheOnlyAttentionLayer):
                # Only CacheOnlyAttentionLayer (extract_hidden_states draft model)
                # is handled here. Other AttentionLayerBase subclasses such as
                # DeepseekV32IndexerCache are intentionally skipped: on Ascend,
                # the indexer's k_cache is replaced by IndexerWrapper, so its
                # KV cache is unused.
                if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                    # CacheOnlyAttentionLayer's module imports MLAAttentionSpec
                    # before the patch runs, so the spec it returns is an
                    # instance of the original (unpatched) class. Rebuilding
                    # with the patched AscendMLAAttentionSpec makes the spec
                    # picklable and keeps this branch consistent with the
                    # MLAAttention branch above.
                    from vllm.v1.kv_cache_interface import MLAAttentionSpec as AscendMLAAttentionSpec
                    kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
                        block_size=spec.block_size,
                        num_kv_heads=spec.num_kv_heads,
                        head_size=spec.head_size,
                        dtype=spec.dtype,
                        cache_dtype_str=spec.cache_dtype_str,
                    )
                    attn_layer_names.add(layer_name)

        if len(mamba_layers) > 0:
            mamba_page_size_padded = 0
            for layer_name, mamba_module in mamba_layers.items():
                if spec := mamba_module.get_kv_cache_spec(self.vllm_config):
                    kv_cache_spec[layer_name] = spec
                    mamba_page_size_padded = spec.page_size_bytes
            # align attn_page_size to mamba_page_size_padded
            for layer_name in attn_layer_names:
                if kv_cache_spec[layer_name].page_size_bytes < mamba_page_size_padded:  # type: ignore[attr-defined]
                    object.__setattr__(kv_cache_spec[layer_name], "page_size_padded", mamba_page_size_padded)

        # embedding_only edge has no local attention layers; the spec
        # is already empty (returned early above).  Cloud side keeps
        # With the PP-reuse init path make_layers directly creates
        # PPMissingLayer for non-local indices — no stale entries
        # ever register in static_forward_context, so no filtering
        # is needed.

        return kv_cache_spec

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> None:
        with update_pass_config(self):
            super()._check_and_update_cudagraph_mode(attention_backends, kv_cache_groups)

        capture_descs = self.cudagraph_dispatcher.get_capture_descs()
        capture_sizes = sorted({
            desc.num_tokens
            for _, descs in capture_descs
            for desc in descs
        })

        # NOTE: Since aclgraph_batch_sizes cannot be determined until here,
        # we set the graph params right before initializing the keys.
        if self.use_aclgraph:
            set_graph_params(capture_sizes)
            if self.speculative_config:
                set_draft_graph_params(capture_sizes)
            if self._edge_cloud_enabled:
                wrappers = []
                if self.edge_cloud_cfg.role == "edge":
                    wrappers = [self.segment_a_wrapper, self.segment_e_wrapper]
                elif self.edge_cloud_cfg.role == "cloud":
                    wrappers = [self.segment_c_wrapper]
                for wrapper in wrappers:
                    if isinstance(wrapper, ACLGraphWrapper):
                        wrapper.graph_params = make_graph_params(self.cudagraph_batch_sizes)
                        if self.speculative_config:
                            wrapper.draft_graph_params = make_graph_params(self.cudagraph_batch_sizes)

                # Also initialize graph params for edge-cloud MTP drafter segments.
                if (
                    self.speculative_config
                    and self.speculative_config.method == "mtp"
                    and hasattr(self, "_edge_cloud_mtp_segments")
                ):
                    for wrapper in self._edge_cloud_mtp_segments.values():
                        if isinstance(wrapper, ACLGraphWrapper):
                            wrapper.graph_params = make_graph_params(self.cudagraph_batch_sizes)
                            if self.speculative_config:
                                wrapper.draft_graph_params = make_graph_params(self.cudagraph_batch_sizes)

    def _get_aclgraph_wrappers(self) -> list[ACLGraphWrapper]:
        """返回所有可能残留 profile 阶段图捕获结果的 ACLGraphWrapper。"""
        wrappers: list[ACLGraphWrapper] = []
        if isinstance(self.model, ACLGraphWrapper):
            wrappers.append(self.model)
        for attr in ("segment_a_wrapper", "segment_e_wrapper", "segment_c_wrapper"):
            wrapper = getattr(self, attr, None)
            if isinstance(wrapper, ACLGraphWrapper):
                wrappers.append(wrapper)
        # Include edge-cloud MTP drafter segment wrappers so that
        # capture_model() can clear any stale entries from them.
        if hasattr(self, "_edge_cloud_mtp_segments"):
            for wrapper in self._edge_cloud_mtp_segments.values():
                if isinstance(wrapper, ACLGraphWrapper):
                    wrappers.append(wrapper)
        return wrappers

    def capture_model(self) -> int:
        # 边云模式的 ACL Graph 仍依赖父类 capture 循环触发 _dummy_run。
        # 实际捕获发生在 segment 级 ACLGraphWrapper 内，通信保持在图外。
        if self._edge_cloud_enabled and not self.edge_cloud_cfg.enable_decode_graph:
            return 0

        gpu_model_runner_cls = next((cls for cls in self.__class__.__mro__ if cls.__name__ == "GPUModelRunner"), None)
        if gpu_model_runner_cls is None:
            raise TypeError("Could not find GPUModelRunner in the MRO. The class hierarchy may have changed.")
        parent_module_name = gpu_model_runner_cls.__module__
        # profile_cudagraph_memory 阶段 ACLGraphWrapper 已捕获过图，
        # 但 CUDAGraphWrapper.clear_all_graphs() 不清除 ACLGraphWrapper
        # 的 concrete_aclgraph_entries。保留的 entry 会导致 capture_model
        # 中 _warmup_and_capture 走 REPLAY 而非 CAPTURE 路径，
        # REPLAY 时 forward_context.capturing 保持 False，
        # 使得 _update_full_graph_params_if_needed 错误执行 → 挂死。
        # 因此这里手动清空，强制重新 capture。
        for wrapper in self._get_aclgraph_wrappers():
            wrapper.concrete_aclgraph_entries.clear()
        with _torch_cuda_wrapper(), _replace_gpu_model_runner_function_wrapper(parent_module_name):
            result = GPUModelRunner.capture_model(self)
        return result

    def _prepare_multimodal_fields(self):
        """
        Ensures specific multimodal tensors are on CPU.
        This is necessary for fields like 'grid_thw' which are converted to numpy
        inside the model's forward pass.
        """
        if not self.multimodal_cpu_fields:
            return

        req_ids = self.input_batch.req_ids
        for req_id in req_ids:
            req = self.requests.get(req_id)
            if req is None:
                continue

            mm_data = getattr(req, "multimodal_data", None)
            if not mm_data:
                continue

            for field in self.multimodal_cpu_fields:
                if field in mm_data:
                    tensor = mm_data[field]
                    if isinstance(tensor, torch.Tensor) and tensor.device.type != "cpu":
                        mm_data[field] = tensor.cpu()

def _post_process_cudagraph_mode(tensor: torch.Tensor) -> int:
    """
    Synchronize cudagraph_mode across DP ranks by taking the minimum.
    If any rank has NONE (0), all ranks use NONE.
    This ensures all ranks send consistent values (all padded or all unpadded).
    """
    return int(tensor[1, :].min().item())

def _get_gpu_model_runner_module_name(model_runner) -> str:
    """Return the module name of GPUModelRunner found in the MRO."""
    gpu_model_runner_cls = next(
        (cls for cls in model_runner.__class__.__mro__ if cls.__name__ == "GPUModelRunner"),
        None,
    )
    if gpu_model_runner_cls is None:
        raise TypeError(
            "Could not find GPUModelRunner in the MRO. "
            "The class hierarchy may have changed."
        )
    return gpu_model_runner_cls.__module__

@contextmanager
def _torch_cuda_wrapper():
    class _EventPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            pass

    try:
        # replace cuda APIs with xpu APIs, this should work by default
        torch.Event = torch.npu.Event
        torch.cuda.Event = torch.npu.Event
        torch.cuda.Stream = torch.npu.Stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.stream = torch.npu.stream
        torch.cuda.synchronize = torch.npu.synchronize
        torch.cuda.mem_get_info = torch.npu.mem_get_info
        yield
    except Exception as e:
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        torch.cuda.default_stream = _StreamPlaceholder
        torch.cuda.current_stream = _StreamPlaceholder
        torch.cuda.stream = _StreamPlaceholder
        torch.cuda.synchronize = _StreamPlaceholder
        torch.cuda.mem_get_info = _StreamPlaceholder
        raise RuntimeError(f"NPUModelRunner init failed, error is {e}")
    finally:
        # if anything goes wrong, just patch it with a placeholder
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = torch.cuda.Stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.stream = torch.npu.stream
        torch.cuda.synchronize = torch.npu.synchronize
        torch.cuda.mem_get_info = torch.npu.mem_get_info

# TODO: This method will be removed subsequently and implemented in platform.
@contextmanager
def _replace_gpu_model_runner_function_wrapper(target_module_name):
    try:
        target_module = sys.modules[target_module_name]
        setattr(target_module, "graph_capture", graph_capture)  # noqa: B010
        yield
    except Exception as e:
        raise RuntimeError(f"NPUModelRunner failed, error is {e}")
    finally:
        setattr(target_module, "graph_capture", graph_capture)  # noqa: B010

# TODO: remove it when flash_comm1 is removed
@contextmanager
def update_pass_config(model_runner):
    try:
        original_pass_config_sp = model_runner.compilation_config.pass_config.enable_sp
        model_runner.compilation_config.pass_config.enable_sp = enable_sp(model_runner.vllm_config)
        yield
    finally:
        model_runner.compilation_config.pass_config.enable_sp = original_pass_config_sp
