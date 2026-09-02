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

import logging
import math
import os
import sys
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import partial
from multiprocessing import Manager
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias
from uuid import uuid4

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
from vllm.forward_context import (
    BatchDescriptor,
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
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
from vllm.utils.torch_utils import PIN_MEMORY, get_dtype_size
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    reorder_batch_to_split_decodes_and_prefills,
)
from vllm.v1.attention.selector import get_attn_backend  # type: ignore
from vllm.v1.core.sched.output import (
    BatchType,
    CachedRequestData,
    HiddenChannelType,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ECConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    RoutedExpertsLists,
    RoutedExpertsTensors,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID, RejectionSampler
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
from vllm.v1.worker.utils import AttentionGroup, select_common_block_size

# yapf: enable
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionState
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPMetadataBuilder
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    get_sfa_qsfa_packed_head_dim,
    using_paged_attention,
)

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
from vllm_ascend.edge_cloud_materialized import (
    supports_materialized_boundary_for_config,
)
from vllm_ascend.distributed.parallel_state import get_lmhead_tp_group
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_device_transfer_loader import D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_worker import EplbProcess
from vllm_ascend.eplb.eplb_updator import EplbUpdator
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
from vllm_ascend.spec_decode.step3p5 import AscendStep3p5MTPProposer
from vllm_ascend.spec_decode.suffix_proposer import AscendSuffixDecodingProposer
from vllm_ascend.spec_decode.utils import (
    correct_optimistic_seq_lens_cpu,
    update_num_computed_tokens_for_batch_change,
)
from vllm_ascend.utils import (
    AscendDeviceType,
    calc_split_factor,
    check_gdn_layer,
    embedding_tp_enable,
    enable_sfa_dcp_replicated_indexer,
    enable_sp,
    enable_sp_by_pass,
    get_ascend_device_type,
    get_c_env,
    global_stream,
    is_hidden_state_cache_spec,
    is_moe_model,
    kv_cache_spec_uses_sparse_c8,
    lmhead_tp_enable,
    oproj_tp_enable,
    set_potential_max_tokens,
    set_weight_prefetch_method,
    should_skip_allreduce_across_dp_group,
    sparse_kv_cache_has_indexer,
    vllm_version_is,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
from vllm_ascend.worker.pcp_utils import PCPAsyncSpecDecodeRebuildResult, PCPManager
from vllm_ascend.worker.utils import AscendKVBlockZeroer, copy_snapshot_to_gpu



from vllm_ascend.ascend_forward_context import (  # isort: skip
    MoECommType,
    get_mc2_tokens_capacity,
    select_moe_comm_method,
    set_ascend_forward_context,
    set_mc2_mask,
    set_mc2_tokens_capacity,
)

from vllm.model_executor.models.interfaces import supports_multimodal_pruning
from vllm.model_executor.models.utils import extract_layer_index

from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

from vllm.model_executor.layers.attention import Attention, MLAAttention

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec

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
    mtp_target_hidden_states: torch.Tensor | None
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


def _get_edge_cloud_moe_start_index(
    all_moe_layers: list[str] | None,
    start_layer: int,
) -> int:
    """Return the MoE cursor for a segment starting at ``start_layer``.

    Edge workers own two non-contiguous layer ranges (head and tail), while
    ``ForwardContext.moe_layer_index`` is reset to zero for every segment
    forward.  The tail segment must therefore advance the cursor past the
    locally registered head MoE layers before entering its first MoE.
    """
    if not all_moe_layers:
        return 0
    return sum(
        extract_layer_index(layer_name) < start_layer
        for layer_name in all_moe_layers
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


@dataclass
class CloudDraftPositionState:
    """Per-task target positions used to reconstruct cloud draft inputs."""

    target_positions: torch.Tensor
    num_scheduled_tokens: tuple[int, ...]
    is_prefill: bool
    base_positions: torch.Tensor | None = None
    # Request ids in the same row order as num_scheduled_tokens (the cloud
    # input_batch order at cache time).  Needed to pair the req_id-keyed
    # accepted counts with the position layout without positional
    # assumptions.
    req_ids: tuple[str, ...] = ()
    # Confirmed start position for each request.  SchedulerOutput may still
    # contain an optimistic value from an earlier async speculative step.
    actual_num_computed_tokens: tuple[int, ...] = ()


@dataclass(frozen=True)
class CloudPendingRequestCorrection:
    """Confirmed async-spec state for one completed cloud target task."""

    task_id: str
    generation: int
    num_draft_tokens: int
    optimistic_num_computed_tokens: int
    actual_num_computed_tokens: int
    num_accepted_tokens: int


def _freeze_scheduled_state(value: Any, memo: dict[int, Any] | None = None) -> Any:
    """Clone mutable state that has to survive a scheduler context switch.

    Input preparation deliberately reuses large CPU/NPU buffers.  A shallow
    copy of attention metadata therefore still points at storage that the next
    PREFILL or DECODE batch rewrites.  This helper preserves object structure
    (including shared aliases) while cloning tensors, numpy arrays, dataclasses
    and builtin containers.  Opaque configuration/backend objects are kept by
    reference because they are immutable for the lifetime of the runner.
    """
    if memo is None:
        memo = {}

    value_id = id(value)
    if value_id in memo:
        return memo[value_id]

    if isinstance(value, torch.Tensor):
        cloned = value.clone()
        memo[value_id] = cloned
        return cloned
    if isinstance(value, np.ndarray):
        cloned = value.copy()
        memo[value_id] = cloned
        return cloned
    if is_dataclass(value) and not isinstance(value, type):
        cloned = copy(value)
        memo[value_id] = cloned
        field_names: set[str] = set()
        for field in fields(value):
            field_names.add(field.name)
            if field.name == "_buffer_slot":
                # The frozen tensors no longer belong to the reusable pool.
                # Do not recursively clone the whole backing slot first.
                object.__setattr__(cloned, field.name, None)
                continue
            object.__setattr__(
                cloned,
                field.name,
                _freeze_scheduled_state(getattr(value, field.name), memo),
            )
        # Some attention metadata attaches pooled buffers dynamically rather
        # than declaring them as dataclass fields.
        for name, item in getattr(value, "__dict__", {}).items():
            if name not in field_names:
                if name == "_buffer_slot":
                    object.__setattr__(cloned, name, None)
                    continue
                object.__setattr__(
                    cloned, name, _freeze_scheduled_state(item, memo)
                )
        return cloned
    if isinstance(value, dict):
        cloned = copy(value)
        cloned.clear()
        memo[value_id] = cloned
        for key, item in value.items():
            cloned[key] = _freeze_scheduled_state(item, memo)
        return cloned
    if isinstance(value, list):
        cloned: list[Any] = []
        memo[value_id] = cloned
        cloned.extend(_freeze_scheduled_state(item, memo) for item in value)
        return cloned
    if isinstance(value, deque):
        cloned = deque(maxlen=value.maxlen)
        memo[value_id] = cloned
        cloned.extend(_freeze_scheduled_state(item, memo) for item in value)
        return cloned
    if isinstance(value, tuple):
        items = tuple(_freeze_scheduled_state(item, memo) for item in value)
        if hasattr(value, "_fields"):
            cloned = type(value)(*items)
        else:
            cloned = items
        memo[value_id] = cloned
        return cloned
    if isinstance(value, set):
        cloned = {_freeze_scheduled_state(item, memo) for item in value}
        memo[value_id] = cloned
        return cloned
    return value


def _restore_frozen_into_views(
    dst: Any,
    src: Any,
    *,
    memo: set[tuple[int, tuple]] | None = None,
    _path: str = "<root>",
) -> None:
    """Copy the contents of a frozen snapshot back into its view tensors.

    ``_freeze_scheduled_state`` clones every tensor, which preserves the
    contents across interleaved batches but breaks the addresses.  ACL
    graph replay, however, reads the exact addresses captured at capture
    time -- for attention metadata those are views into the runner /
    backend persistent buffers.  The edge segment_e fast path therefore
    keeps the original (view-based) objects next to the frozen snapshot
    and calls this to refresh the persistent buffers' contents before the
    forward pass, so a graph replay reads the captured addresses holding
    this step's data.  (Eager execution is unaffected: it reads the same
    tensors with the same, just-restored contents.)

    Non-tensor (scalar) fields need no restore: they belong to this
    step's view object and were never touched by intervening batches.

    ``memo`` deduplicates copies across a shared object graph: per-layer
    metadata entries (e.g. DSA's 61-layer dict) alias the same persistent
    buffers via distinct view objects, so the key is the destination
    region -- (data_ptr, shape) -- not the tensor object id.  All tensors
    involved are kept alive by the cache for the duration of the walk, so
    a data_ptr cannot be recycled mid-restore.

    Structurally or shape-mismatched entries are skipped WITH a warning:
    they would leave the graph reading stale buffer contents, and a
    silent skip hides exactly the failure this restore exists to prevent.
    """
    if dst is None or src is None:
        return
    if memo is None:
        memo = set()
    if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
        if (dst.shape == src.shape and dst.dtype == src.dtype
                and dst.device == src.device):
            key = (dst.data_ptr(), tuple(dst.shape))
            if key in memo:
                return
            memo.add(key)
            dst.copy_(src, non_blocking=True)
        else:
            logger.warning_once(
                "_restore_frozen_into_views: skipping %s (dst shape=%s "
                "dtype=%s device=%s, src shape=%s dtype=%s device=%s) -- "
                "graph replay may read stale buffer contents",
                _path,
                tuple(dst.shape), dst.dtype, dst.device,
                tuple(src.shape), src.dtype, src.device,
            )
        return
    if is_dataclass(dst) and not isinstance(dst, type):
        field_names = {f.name for f in fields(dst)}
        for name in field_names:
            _restore_frozen_into_views(
                getattr(dst, name),
                getattr(src, name, None),
                memo=memo,
                _path=f"{_path}.{name}",
            )
        for name, item in getattr(dst, "__dict__", {}).items():
            if name not in field_names:
                _restore_frozen_into_views(
                    item,
                    getattr(src, name, None),
                    memo=memo,
                    _path=f"{_path}.{name}",
                )
        return
    if isinstance(dst, dict) and isinstance(src, dict):
        for key, item in dst.items():
            if key in src:
                _restore_frozen_into_views(
                    item, src[key], memo=memo, _path=f"{_path}[{key!r}]"
                )
        return
    if isinstance(dst, (list, tuple)) and isinstance(src, (list, tuple)):
        for i, (d_item, s_item) in enumerate(zip(dst, src)):
            _restore_frozen_into_views(
                d_item, s_item, memo=memo, _path=f"{_path}[{i}]"
            )
        return


def _freeze_intermediate_tensors(
    intermediate_tensors: IntermediateTensors,
) -> IntermediateTensors:
    return IntermediateTensors(
        _freeze_scheduled_state(dict(intermediate_tensors.items()))
    )

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
        # Chunk-kernel prefill inputs (gdn.py recurrent path reads these:
        # ssm_state[prefill_state_indices] scatter read/write). If left as
        # shared-buffer views, an interleaved decode overwrites them and the
        # continuation's ssm_state scatter hits WRONG slots — corrupting the
        # decode request's GDN state as well.
        "prefill_query_start_loc",
        "prefill_state_indices",
        "prefill_has_initial_state",
        "batch_ptr",
        "token_chunk_offset_ptr",
        # Full-attention backend (AscendMetadata) fields: these are views
        # into the persistent per-batch buffers (block table / slot_mapping
        # / positions) that an interleaved decode batch rewrites in-place.
        # Without cloning, a sliced prefill continuation writes/reads KV
        # through the DECODE request's block table, corrupting both
        # requests' KV caches (garbled decode on the running request,
        # repeated tokens on the prefilling one).
        "block_tables",
        "slot_mapping",
        "seq_lens",
        "query_start_loc",
        "attn_mask",
        "actual_seq_lengths_q",
        "actual_seq_lengths_kv",
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

    # Nested per-phase metadata objects are attached BY REFERENCE from the
    # top-level fields (see _attach_non_spec_prefill_metadata in
    # gdn_attn_builder). Cloning only the top-level fields therefore leaves
    # the nested references pointing at the ORIGINAL shared buffers, which
    # an interleaved decode batch overwrites in-place — the GDN forward
    # reads exactly these nested tensors
    # (non_spec_prefill_metadata.causal_conv1d.query_start_loc etc.), so a
    # sliced prefill continuation would run its convolution with the decode
    # batch's sequence boundaries. Deep-clone the nested objects too.
    def _clone_dev_tensor(t):
        if isinstance(t, torch.Tensor) and t.device.type != "cpu":
            return t.clone()
        return t

    prefill_meta = getattr(cloned, "non_spec_prefill_metadata", None)
    if prefill_meta is not None:
        cloned_prefill = copy.copy(prefill_meta)
        causal = getattr(cloned_prefill, "causal_conv1d", None)
        if causal is not None:
            cloned_causal = copy.copy(causal)
            for attr in ("query_start_loc", "cache_indices",
                         "initial_state_mode"):
                t = getattr(cloned_causal, attr, None)
                if t is not None:
                    setattr(cloned_causal, attr, _clone_dev_tensor(t))
            cloned_prefill.causal_conv1d = cloned_causal
        chunk = getattr(cloned_prefill, "chunk", None)
        if chunk is not None:
            cloned_chunk = copy.copy(chunk)
            for attr in (
                "chunk_indices_chunk64",
                "chunk_offsets_chunk64",
                "update_chunk_offsets_chunk64",
                "final_chunk_indices_chunk64",
                "chunk_indices_large_block",
                "block_indices_cumsum",
                "keep_meta",
            ):
                t = getattr(cloned_chunk, attr, None)
                if t is not None:
                    setattr(cloned_chunk, attr, _clone_dev_tensor(t))
            cloned_prefill.chunk = cloned_chunk
        cloned.non_spec_prefill_metadata = cloned_prefill

    decode_meta = getattr(cloned, "non_spec_decode_metadata", None)
    if decode_meta is not None:
        cloned_decode = copy.copy(decode_meta)
        causal = getattr(cloned_decode, "causal_conv1d", None)
        if causal is not None:
            cloned_causal = copy.copy(causal)
            for attr in ("query_start_loc", "cache_indices",
                         "initial_state_mode"):
                t = getattr(cloned_causal, attr, None)
                if t is not None:
                    setattr(cloned_causal, attr, _clone_dev_tensor(t))
            cloned_decode.causal_conv1d = cloned_causal
        t = getattr(cloned_decode, "actual_seq_lengths", None)
        if t is not None:
            cloned_decode.actual_seq_lengths = _clone_dev_tensor(t)
        cloned.non_spec_decode_metadata = cloned_decode

    spec_meta = getattr(cloned, "spec_decode_metadata", None)
    if spec_meta is not None:
        cloned_spec = copy.copy(spec_meta)
        spec_causal = getattr(cloned_spec, "spec_causal_conv1d", None)
        if spec_causal is not None:
            cloned_sc = copy.copy(spec_causal)
            for attr in ("query_start_loc", "cache_indices",
                         "num_accepted_tokens"):
                t = getattr(cloned_sc, attr, None)
                if t is not None:
                    setattr(cloned_sc, attr, _clone_dev_tensor(t))
            cloned_spec.spec_causal_conv1d = cloned_sc
        t = getattr(cloned_spec, "actual_seq_lengths", None)
        if t is not None:
            cloned_spec.actual_seq_lengths = _clone_dev_tensor(t)
        cloned.spec_decode_metadata = cloned_spec

    return cloned

def _reorder_input_batch_to_so_order(input_batch, scheduler_output) -> bool:
    """Reorder ``input_batch`` so its request order matches the
    ``num_scheduled_tokens`` key order of ``scheduler_output``.

    Edge-cloud wire alignment: the e2c/c2e hidden/mrope transfer is laid
    out flat in the sender's ``input_batch`` order and consumed in the
    receiver's ``input_batch`` order.  The two sides' batch histories
    diverge under PD interleaving: the edge additionally executes PL/DL
    tail segments whose normal-path ``_update_states`` removes and
    re-adds requests (a request whose PL was interleaved with decode
    batches lands at index 0 on the edge but is appended at the end on
    the cloud).  Since the SchedulerOutput is the SAME object on both
    sides, its ``num_scheduled_tokens`` key order is a canonical order
    both runners can converge to.  ``swap_states`` keeps
    ``req_id_to_index`` and all per-request rows in sync.
    """
    target = list(scheduler_output.num_scheduled_tokens.keys())
    if len(target) != input_batch.num_reqs or list(
            input_batch.req_ids) == target:
        return False
    for dst, req_id in enumerate(target):
        src = input_batch.req_id_to_index[req_id]
        if src != dst:
            input_batch.swap_states(src, dst)
    return True

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

        if not vllm_version_is("0.23.0"):
            self.pin_memory = PIN_MEMORY

        # Replace the CUDA PrefetchOffloader set by parent __init__ with NPU version.
        offload_cfg = vllm_config.offload_config
        if (offload_cfg is not None
                and getattr(offload_cfg, "prefetch", None) is not None
                and getattr(offload_cfg.prefetch, "offload_group_size", 0) > 0):
            from vllm.model_executor.offloader.base import set_offloader

            from vllm_ascend.model_executor.offloader.prefetch import NPUPrefetchOffloader
            set_offloader(NPUPrefetchOffloader(
                group_size=offload_cfg.prefetch.offload_group_size,
                num_in_group=offload_cfg.prefetch.offload_num_in_group,
                prefetch_step=offload_cfg.prefetch.offload_prefetch_step,
                offload_params=offload_cfg.prefetch.offload_params,
            ))

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
        self._pending_edge_cloud_draft_contexts: dict[
            str, dict[str, Any]
        ] = {}
        # Request-keyed snapshot of num_computed_tokens taken at the top of
        # every _update_states (BEFORE the base update overwrites req_state
        # with the current SchedulerOutput's values).  The async spec-decode
        # correction in _prepare_inputs uses this as the base for
        # participating rows instead of the positional GPU buffer: PD
        # interleaving evicts/re-adds requests between batches, recycling
        # rows of the GPU buffer, so the positional gather can read another
        # request's value and poison positions/seq_lens (the EC NaN-row
        # crash).  Keyed by request identity, this survives the churn.
        self._prev_num_computed_by_req: dict[str, int] = {}
        # The snapshot above is rotated from this map at every
        # _update_states; this map only ever holds SchedulerOutput-recorded
        # num_computed values (never deferred-correction-rewound req_state
        # values), which is what makes the correction base exact.
        self._last_so_num_computed: dict[str, int] = {}
        # Whether the resident batch still matches the executing tail
        # SchedulerOutput BEFORE _update_states runs (see execute_model).
        # Only meaningful for edge tail segments; defaults to intact.
        self._edge_tail_membership_intact = True
        # Exact speculative token rows consumed by each in-flight target
        # verify, keyed by its head_token and then req_id.  Snapshotting the
        # repaired input buffer (rather than the global _draft_token_ids)
        # keeps the matching target tail correct when request groups
        # interleave or a member request finishes before the tail arrives.
        self._verified_draft_token_ids_by_head: dict[
            str, dict[str, torch.Tensor]
        ] = {}
        # Latest completed draft token IDs per request (one row of
        # _draft_token_ids each), recorded at DRAFT-OUT time.  This is the
        # authoritative source for the verify-time scatter in
        # _prepare_inputs: the global self._draft_token_ids can be
        # overwritten by another request's draft chain before this
        # request's verify executes, and the native _prepare_input_ids
        # scatter cannot cover requests that were absent from the previous
        # execute_model batch (prev_positions < 0, e.g. when another
        # request's PREFILL_FIRST ran between this request's PREFILL_LAST
        # and its first DECODE_FIRST).  Rows are views into the producing
        # chain's tensor, so later global overwrites do not corrupt them.
        self._worker_draft_token_ids_by_req: dict[str, torch.Tensor] = {}

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
        self._is_edge_cloud_tail_segment = False
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
            self._is_deepseek_v4 = model_type == "deepseek_v4"
            if self._is_deepseek_v4 and (
                self.edge_cloud_cfg.mode != "head_tail"
                or (self.head_k, self.tail_k) != (3, 1)
            ):
                raise ValueError(
                    "DeepSeek-V4 edge-cloud mode only supports the original "
                    "head-3/tail-1 layout because the first three Hash MoE "
                    "layers must stay on the edge. Set "
                    "edge_cloud_config.mode='head_tail' and "
                    "edge_cloud_config.edge_head_tail_layers=[3, 1]."
                )
            self._is_qwen3_5 = "qwen3_5" in model_type
            self._is_deepseek_v2 = (
                "deepseek" in model_type and not self._is_deepseek_v4
            )
            self._is_kimi_k25 = "kimi_k25" in outer_model_type or "kimi_k25" in model_type
            self._is_glm4_moe = "glm4_moe" in model_type or "glm_moe_dsa" in model_type
            self._is_minimax_m2 = "minimax_m2" in model_type
            self.num_layers = 0
            self.segment_a: Any = None
            self.segment_e: Any = None
            self.segment_c: Any = None
            self.segment_c_raw: Any = None
            self.segment_a_wrapper: Any = None
            self.segment_e_wrapper: Any = None
            self.segment_c_wrapper: Any = None
            # Cache segment_a prepare results for segment_e reuse.  Entries
            # are keyed because prefill/decode heads may be in flight at the
            # same time and must not overwrite one another.
            self._edge_prepare_cache_by_token: dict[str, dict[str, Any]] = {}
            self._edge_prepare_cache_max: int = 8
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
            if get_ascend_device_type() == AscendDeviceType.A5 and self.ascend_config.enable_sparse_c8:
                # A5 sparse C8 uses the same merged/packed KV layout as SFA QSFA.
                # qk_rope_head_dim = 0 signals the merged layout.
                packed_kv_head_dim = get_sfa_qsfa_packed_head_dim(
                    self.model_config.hf_text_config.kv_lora_rank,
                    self.model_config.hf_text_config.qk_rope_head_dim,
                )
                self.sparse_head_dim = (
                    packed_kv_head_dim,
                    0,
                    self.model_config.hf_text_config.index_head_dim,
                )
            else:
                self.sparse_head_dim = (
                    self.model_config.hf_text_config.kv_lora_rank,
                    self.model_config.hf_text_config.qk_rope_head_dim,
                    self.model_config.hf_text_config.index_head_dim,
                )
        # dsa c8
        self.use_sparse_c8 = self.ascend_config.enable_sparse_c8
        if self.use_sparse_c8:
            if get_ascend_device_type() == AscendDeviceType.A5:
                self.c8_k_cache_dtype = torch.float8_e4m3fn
                self.c8_k_scale_cache_dtype = torch.float32
            else:
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

        # reinit valid_sampled_token_count_cpu with torch.int64 dtype
        if self.use_async_scheduling and self.num_spec_tokens:
            self.valid_sampled_token_count_cpu = torch.empty(
                self.max_num_reqs,
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory,
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
            if not self.vllm_config.cache_config.enable_prefix_caching:
                self.vllm_config.cache_config.mamba_block_size = self.model_config.max_model_len
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

        self.sfa_dcp_replicated_indexer_size = 1
        if enable_sfa_dcp_replicated_indexer():
            self.sfa_dcp_replicated_indexer_size = self.dcp_size

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

        # Backends that consume CPU seq_lens (AscendAttentionBackend,
        # AscendMLABackend, and DSV4 compressed attention metadata) need
        # ``optimistic_seq_lens_cpu`` to match the corrected GPU seq_lens
        # in async spec decode mode; others (SFA, GDN, etc.) do not.
        self._needs_seq_lens_cpu_sync = self.use_compress or issubclass(
            self.attn_backend, (AscendAttentionBackend, AscendMLABackend)
        )

        # kv role
        self.is_kv_producer = False
        self.is_kv_consumer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
            self.is_kv_consumer = vllm_config.kv_transfer_config.is_kv_consumer

        set_cos_and_sin(vllm_config, self.max_num_reqs, self.uniform_decode_query_len, self.dtype, self.device)
        set_mc2_tokens_capacity(vllm_config, self.max_num_reqs, self.uniform_decode_query_len)
        set_mc2_mask(vllm_config, self.device)
        # Compute potential_max_tokens once here; it is reused by the skip-allreduce
        # decision and the o_proj static-exchange buffer sizing (see get_potential_max_tokens).
        set_potential_max_tokens(vllm_config)
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
            self.eplb_process = EplbProcess(
                shared_dict=self.shared_dict,
                policy_type=self.policy_type,
                enable_d2d=True,
                tp_size=self.parallel_config.tensor_parallel_size,
            )
            self.process = self.eplb_process._launch_process()
            self.eplb_updator = EplbUpdator(eplb_config, self.eplb_loader, self.eplb_process, self.process)
            # In pd colocation scenarios, we find that prefill/decode requests result in different
            # expert workloads. To reduce expert imbalance more effectively, we can coolect eplb
            # heat exclusively on a single stage rather than both prefill/decode.
            self.eplb_heat_collection_stage = eplb_config.eplb_heat_collection_stage
            # Currently, we set the maximum of tokens in decode stage as the threshold to distinguish
            # prefill with decode.
            self.eplb_pd_thresholds = self.max_num_reqs * self.uniform_decode_query_len
            self.eplb_heat_collection_status = True

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
        self.sampling_done_event: torch.npu.Event | None = None
        self.valid_sampled_token_count_gpu: torch.Tensor | None = None

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

        # Saved in execute_model() for legacy synchronous edge-cloud sampling
        # and auxiliary-hidden-state paths.
        self._last_scheduler_output: "SchedulerOutput | None" = None
        # True while the parent target graph-capture loop is running. Eager
        # edge-cloud draft segments must not communicate from inside it.
        self._edge_cloud_target_capture_in_progress = False

        # Latest cloud-side target metadata for draft paths that do not cross
        # an independent scheduling boundary.
        self._cloud_spec_decode_common_attn_metadata: AscendCommonAttentionMetadata | None = None
        self._cloud_spec_decode_num_reqs: int = 0
        # Independently scheduled drafts can run after unrelated prefill
        # or decode work.  Keep immutable target-step snapshots until the
        # matching draft task consumes them instead of relying on the latest
        # global metadata pointer.
        self._cloud_spec_decode_metadata_by_task: dict[
            str, tuple[AscendCommonAttentionMetadata, int]
        ] = {}
        # Bound sized for long-sequence (e.g. 64k) multi-request runs where the
        # draft chain can legitimately lag several verify steps behind.  Too
        # small a bound evicts an in-flight task's metadata; the matching
        # DRAFT_FIRST then raises in _reconstruct_cloud_draft_positions and the
        # cloud never sends the DRAFT_LAST response, deadlocking the edge's
        # matching recv on the shared DECODE channel.
        self._cloud_spec_decode_metadata_cache_max: int = 10240
        # Same per-task treatment for the verify step's scheduler_output. The
        # independently scheduled draft records request-keyed accepted-token
        # corrections after unrelated work may have replaced the latest
        # scheduler-output pointer.
        self._cloud_scheduler_output_by_task: dict[
            str, "SchedulerOutput"
        ] = {}
        self._cloud_draft_position_state_by_task: dict[
            str, CloudDraftPositionState
        ] = {}
        # Async target batches may be interleaved with unrelated prefill or
        # draft work before their accepted-token result reaches the cloud.
        # Keep the confirmed correction by request instead of writing it into
        # a positional global tensor whose row ownership may already differ.
        self._cloud_target_generation: int = 0
        self._cloud_target_generation_by_task: dict[str, int] = {}
        self._cloud_latest_target_generation_by_req: dict[str, int] = {}
        self._cloud_actual_num_computed_by_req: dict[
            str, tuple[int, int]
        ] = {}
        self._cloud_pending_request_corrections: dict[
            str, CloudPendingRequestCorrection
        ] = {}
        # Set only while preparing a cloud target whose optimistic CPU state
        # was corrected from the request-keyed records above.  This selects a
        # direct CPU->NPU copy and avoids the legacy positional gather.
        self._cloud_current_cpu_state_authoritative: bool = False
        self._eagle3_cloud_aux_hidden_states_by_task: dict[
            str, torch.Tensor
        ] = {}
        self.enable_hamming_sparse = (self.ascend_config.enable_hamming_sparse is True)
        self.enable_hamming_sparse = self.enable_hamming_sparse and not vllm_config.speculative_config
        if self.enable_hamming_sparse is True:
            from vllm_ascend.worker.kvcomp_utils import initialize_kvcomp_metadata
            self.kvcomp_meta_data = initialize_kvcomp_metadata(max_num_reqs=self.max_num_reqs,
                block_size=self.block_size, device=self.device, vllm_config=self.vllm_config,
                parallel_config=self.parallel_config, dtype=self.dtype)

        self.edge_cloud_cfg = self.ascend_config.edge_cloud_config
        self._edge_cloud_enabled = self.edge_cloud_cfg.enabled
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
            self._is_deepseek_v4 = model_type == "deepseek_v4"
            if self._is_deepseek_v4 and (
                self.edge_cloud_cfg.mode != "head_tail"
                or (self.head_k, self.tail_k) != (3, 1)
            ):
                raise ValueError(
                    "DeepSeek-V4 edge-cloud mode only supports the original "
                    "head-3/tail-1 layout because the first three Hash MoE "
                    "layers must stay on the edge. Set "
                    "edge_cloud_config.mode='head_tail' and "
                    "edge_cloud_config.edge_head_tail_layers=[3, 1]."
                )
            self._is_qwen3_5 = "qwen3_5" in model_type
            self._is_deepseek_v2 = (
                "deepseek" in model_type and not self._is_deepseek_v4
            )
            self._is_kimi_k25 = "kimi_k25" in outer_model_type or "kimi_k25" in model_type
            self._is_glm4_moe = "glm4_moe" in model_type or "glm_moe_dsa" in model_type
            self._is_minimax_m2 = "minimax_m2" in model_type
            self.num_layers = 0
            self.segment_a: Any = None
            self.segment_e: Any = None
            self.segment_c: Any = None
            self.segment_c_raw: Any = None
            self.segment_a_wrapper: Any = None
            self.segment_e_wrapper: Any = None
            self.segment_c_wrapper: Any = None
            # Cache segment_a prepare results for segment_e reuse (edge-cloud
            # only).  Keyed by head_token so that ahead-scheduled chunks
            # (chunk_prior with max_chunk_prefill_ahead >= 1) do not overwrite
            # an earlier chunk's cache before its segment_e consumes it: each
            # segment_a stores under its own head_token, and the matching
            # segment_e pops that exact entry.  Without this keying, chunk-1's
            # segment_a would clobber chunk-0's cache while chunk-0's segment_e
            # is still waiting for the cloud PL, and chunk-0's PL would run
            # segment_e with chunk-1's attn_metadata / num_tokens_padded.
            self._edge_prepare_cache_by_token: dict[str, dict] = {}
            # Bounded size guard: a segment_e that never arrives (e.g. request
            # aborted mid-prefill) would otherwise leak its entry.  2P1D keeps
            # at most 2 in-flight; the slack absorbs scheduling jitter.
            self._edge_prepare_cache_max: int = 8
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
            | AscendStep3p5MTPProposer
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
                and self.speculative_config.method in ("mtp", "eagle3")
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
        # Cloud-side cache for EAGLE3 aux hidden states produced by the target
        # model's cloud segment. These hidden states are consumed by the draft
        # model's cloud segment without crossing the edge-cloud boundary.
        self._eagle3_cloud_aux_hidden_states: torch.Tensor | None = None

    def _get_drafter(self):
        return get_spec_decode_method(self.speculative_config.method, self.vllm_config, self.device, self)

    def _eagle3_uses_aux_hidden_state(self) -> bool:
        if self.speculative_config is None or self.speculative_config.method != "eagle3":
            return False

        draft_model_config = self.speculative_config.draft_model_config
        if draft_model_config is None:
            return True

        eagle_config = getattr(draft_model_config.hf_config, "eagle_config", None)
        if eagle_config is None:
            return True
        return eagle_config.get("use_aux_hidden_state", True)

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

    def _use_materialized_residual_boundary(self) -> bool:
        return supports_materialized_boundary_for_config(self.model_config)

    def _make_empty_edge_cloud_intermediate_tensors(
        self,
        batch_size: int,
    ) -> IntermediateTensors:
        if self._use_materialized_residual_boundary():
            hidden_size = self.model_config.hf_text_config.hidden_size
            return IntermediateTensors({
                "hidden_states": torch.zeros(
                    (batch_size, hidden_size),
                    dtype=self.dtype,
                    device=self.device,
                )
            })
        assert self.model is not None
        return self.model.make_empty_intermediate_tensors(
            batch_size=batch_size,
            dtype=self.dtype,
            device=self.device,
        )

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
        edge_model = getattr(segment, "_edge_model", None)
        if getattr(edge_model, "edge_cloud_dynamic_step_segments", False):
            return segment

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
        is_draft: bool = False,
    ) -> Any:
        edge_model = getattr(segment, "_edge_model", None)
        if (
            is_draft
            and getattr(edge_model, "edge_cloud_dynamic_step_segments", False)
        ):
            return segment
        if not self.edge_cloud_cfg.enable_decode_graph:
            return segment
        if not self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            return segment
        if self._is_dummy_or_profile_run():
            return segment
        # 与标准（非边云）流程对齐：draft/proposer 仅在
        # `_use_aclgraph() and not speculative_config.enforce_eager` 时才用图
        # （见 llm_base_proposer.py:166 —— eagle3 enforce_eager=True 即代表 draft
        # 跑 eager）。注意：target 模型段的图开关由 model_config.enforce_eager /
        # cudagraph_mode 决定，与 speculative_config.enforce_eager 无关，因此该
        # 分支只对 draft 段（_edge_cloud_draft_segments）生效。
        if (
            is_draft
            and self.speculative_config is not None
            and self.speculative_config.enforce_eager
        ):
            return segment
        return EdgeCloudACLGraphWrapper(
            segment,
            self.vllm_config,
            runtime_mode=runtime_mode,
            cudagraph_options=None,
            # 与标准（非边云）流程的 ACLGraphWrapper 构造保持一致（model_runner_v1.py:5298），
            # 否则边云 wrapper 的 use_eagle 恒为 False：
            #  - target 验证路径（is_draft_model=False）不受影响（need_sync 恒为 True），
            #  - 但 draft 段（is_draft_model=True）回放前的 synchronize 屏障决策会与非边云分叉。
            use_eagle=self.use_eagle,
            enable_enpu=self.enable_enpu,
        )

    def _get_edge_cloud_segment_model(self, segment: Any) -> torch.nn.Module:
        """Unwrap ACLGraph / compiled wrappers to reach the EdgeCloudSegment.

        Edge-cloud draft segments may be wrapped by ``EdgeCloudCompiledSegment``
        (torch.compile) and/or ``EdgeCloudACLGraphWrapper`` (runtime graph
        capture).  Both wrappers hide the original ``EdgeCloudSegment`` and its
        ``_edge_model`` attribute.  This helper peels off the wrappers so that
        callers can access the underlying draft model for configuration lookups
        such as ``model.fc.input_size``.
        """
        while isinstance(segment, (EdgeCloudCompiledSegment, EdgeCloudACLGraphWrapper)):
            if isinstance(segment, EdgeCloudCompiledSegment):
                segment = segment._segment
            else:
                segment = segment.unwrap()
        return segment._edge_model

    def _prepare_eagle3_cloud_hidden_states(
        self,
        segment: Any,
        intermediate_tensors: IntermediateTensors,
        aux_hidden_states: torch.Tensor | None,
        num_tokens: int,
        is_first_step: bool,
    ) -> None:
        """Prepare the draft input outside the captured cloud segment.

        ``spec_step_idx`` selects this preparation at the caller.  The segment
        itself must have one static execution path so that ACL graph replay does
        not reuse the first-step fusion branch for later speculative steps.
        """
        if aux_hidden_states is None:
            if is_first_step:
                raise RuntimeError(
                    "EAGLE3 cloud segment received empty aux_hidden_states "
                    "on the first speculative step."
                )
            return

        draft_model = self._get_edge_cloud_segment_model(segment)
        if not draft_model.model.use_aux_hidden_state:
            return
        if aux_hidden_states.shape[0] != num_tokens:
            assert aux_hidden_states.shape[0] > num_tokens, (
                f"aux_hidden_states batch size {aux_hidden_states.shape[0]} "
                f"is smaller than draft num_tokens {num_tokens}"
            )
            aux_hidden_states = aux_hidden_states[:num_tokens]

        hidden_states = intermediate_tensors["hidden_states"]
        fused_hidden_states = draft_model.combine_hidden_states(aux_hidden_states)
        if hidden_states.numel() == 0:
            # Warmup or edge-cloud broadcast may send an empty placeholder
            # hidden_states tensor (shape [0] or [0, hidden_size]). Replace it
            # with the fused result so the cloud segment sees valid input.
            intermediate_tensors["hidden_states"] = fused_hidden_states
        else:
            assert hidden_states.shape == fused_hidden_states.shape, (
                "EAGLE3 cloud hidden_states buffer shape does not match the "
                f"fusion result: {hidden_states.shape} vs {fused_hidden_states.shape}"
            )
            hidden_states.copy_(fused_hidden_states)

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
        if not (
            self._is_qwen3_5
            or self._is_deepseek_v2
            or self._is_deepseek_v4
            or self._is_kimi_k25
            or self._is_glm4_moe
            or self._is_minimax_m2
        ):
            raise NotImplementedError(
                "edge-cloud mode currently supports Qwen3.5, "
                "DeepSeek-V2/V3/V4, Kimi-K2.5/K2.6, GLM-4/GLM-5 models, "
                "and MiniMax-M2 models."
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

            is_edge_cloud_draft_drafter = (
                self.speculative_config is not None
                and self.speculative_config.method in ("mtp", "eagle3")
            )
            if is_edge_cloud_draft_drafter:
                # Draft models (MTP/Eagle3) use the same edge-cloud layer range
                # mechanism as the main model. In both embedding_only and
                # head_tail modes all draft decoder layers run on the cloud, so
                # temporarily use head_k=tail_k=0 while loading the drafter.
                set_edge_cloud_layer_range(0, 0)
                if self.speculative_config.method == "eagle3":
                    import vllm_ascend.patch.models.eagle3_edge_cloud  # noqa: F401
                elif (
                    self.speculative_config.method == "mtp"
                    and self._is_deepseek_v4
                ):
                    import vllm_ascend.patch.models.deepseek_v4_mtp_edge_cloud  # noqa: F401

            with get_tp_context(self.drafter):
                self.drafter.load_model(self.model)

            if (
                is_edge_cloud_draft_drafter
                and hasattr(self.drafter, "model")
                and self.drafter.model is not None
            ):
                self._setup_edge_cloud_draft(
                    self.drafter.model, self.speculative_config.method
                )

            if is_edge_cloud_draft_drafter:
                # Do not leak the drafter's cloud-only range into later
                # main-model initialization or cache setup.
                set_edge_cloud_layer_range(self.head_k, self.tail_k)

            # In edge-cloud EAGLE3 mode the target model also needs its
            # auxiliary hidden-state layers configured, just like the standard
            # (non-edge-cloud) load path. The cloud segment reads these layers
            # in forward_edge_cloud_segment to produce aux_hidden_states for the
            # draft model. Without this the aux_hidden_states key is missing
            # from the cloud segment output, which causes KeyError or makes the
            # draft model fall back to dummy zero states.
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

    def _setup_edge_cloud_draft(
        self, draft_model: nn.Module, method: str
    ) -> None:
        """Shard the draft model into edge/cloud segments for edge-cloud mode.

        Supports both MTP (``Qwen3_5MTP``/``DeepSeekMTP`` style) and Eagle3
        (``Eagle3LlamaForCausalLM`` style) draft models.  The embedding/preprocessing
        and the output head live on the edge; all decoder layers + final norm live
        on the cloud.
        """
        if method == "mtp":
            predictor = self._get_mtp_predictor(draft_model)
            if predictor is None:
                logger.warning("[EdgeCloud] Cannot find MTP predictor for sharding")
                return
            # NOTE: "norm" is deliberately NOT stripped on the cloud.  The
            # cloud applies the MTP final norm right after its decoder layer
            # and ships only the normed hidden states back to the edge, which
            # eliminates the residual transfer (halving the cloud->edge
            # payload).  The edge keeps its own norm weights as a fallback for
            # payloads that still carry a pre-norm residual.
            edge_only_modules = (
                "embed_tokens",
                "fc",
                "pre_fc_norm_hidden",
                "pre_fc_norm_embedding",
            )
        elif method == "eagle3":
            # Eagle3 draft: draft_model.model is the LlamaModel (embed/layers/norm).
            if not hasattr(draft_model, "model"):
                logger.warning(
                    "[EdgeCloud] Eagle3 draft model has no .model attribute"
                )
                return
            predictor = draft_model.model
            # In cloud-fusion mode the fc projection (combine_hidden_states) and
            # its optional input_norm run on the cloud together with the target
            # model's aux hidden states. The edge side only embeds input_ids.
            edge_only_modules = (
                "embed_tokens",
            )
        else:
            logger.warning(
                "[EdgeCloud] Unsupported edge-cloud draft method: %s", method
            )
            return

        num_draft_layers = len(predictor.layers)
        if method == "mtp":
            num_spec_tokens = int(self.num_spec_tokens or 0)
            if num_spec_tokens <= 0:
                raise ValueError(
                    "MTP edge-cloud scheduling requires a positive "
                    "num_speculative_tokens"
                )
            if num_draft_layers <= 0:
                raise ValueError(
                    "MTP edge-cloud scheduling requires at least one "
                    "MTP layer"
                )
            logger.info(
                "[EdgeCloud] MTP scheduling: draft_kind=%s, draft_steps=%d, "
                "mtp_layers=%d",
                getattr(draft_model, "edge_cloud_draft_kind", "qwen_mtp"),
                num_spec_tokens,
                num_draft_layers,
            )

        # Capture module ids before sharding so we can clean stale
        # static_forward_context entries that point to removed layers.
        draft_module_ids = {id(module) for _, module in draft_model.named_modules()}

        custom_shard = getattr(draft_model, "shard_for_edge_cloud", None)
        uses_custom_shard = callable(custom_shard)
        if uses_custom_shard:
            # Model-specific adapters own only the internal partitioning.
            # Segment construction, communication, buffering, and scheduling
            # remain on the shared MTP/Eagle3 edge-cloud draft path.
            custom_shard(is_edge=is_edge_device())

        # Use the same edge-cloud layer range mechanism as the main model.
        # For draft models this was set to head_k=tail_k=0 before the drafter
        # was loaded, so all decoder layers run on the cloud.
        head_k, tail_k = get_edge_cloud_layer_range()

        local_layers: set[int] = set()
        if is_edge_device():
            if head_k > 0:
                local_layers.update(range(head_k))
            if tail_k > 0:
                local_layers.update(
                    range(num_draft_layers - tail_k, num_draft_layers)
                )
        else:
            local_layers.update(range(head_k, num_draft_layers - tail_k))

        layer_keys = (
            list(predictor.layers.keys())
            if isinstance(predictor.layers, nn.ModuleDict)
            else list(range(num_draft_layers))
        )
        for idx, key in enumerate(layer_keys):
            if (
                not uses_custom_shard
                and idx not in local_layers
                and not isinstance(predictor.layers[key], PPMissingLayer)
            ):
                predictor.layers[key] = PPMissingLayer()

        # Cloud side does not need embedding/preprocessing/output modules;
        # edge keeps them.
        if not uses_custom_shard and not is_edge_device():
            for module_name in edge_only_modules:
                module = getattr(predictor, module_name, None)
                if module is not None and not isinstance(module, PPMissingLayer):
                    setattr(predictor, module_name, PPMissingLayer())
            if (
                hasattr(draft_model, "lm_head")
                and not isinstance(draft_model.lm_head, PPMissingLayer)
            ):
                draft_model.lm_head = PPMissingLayer()

        # Re-collect MoE parameters now that some layers may be placeholders.
        if (
            not uses_custom_shard
            and hasattr(draft_model, "set_moe_parameters")
        ):
            draft_model.set_moe_parameters()

        self._clean_mtp_compilation_config(draft_model, draft_module_ids)

        if hasattr(self, "_edge_cloud_draft_segments"):
            delattr(self, "_edge_cloud_draft_segments")
        self._edge_cloud_draft_segments = {}

        # Pre-allocate persistent intermediate buffers for edge-cloud draft
        # segments. ACLGraphWrapper requires stable input tensor addresses
        # across graph replay, but edge_cloud_broadcast_recv_draft() allocates
        # fresh tensors every iteration. Copying received tensors into these
        # buffers before calling graph-wrapped segments avoids stale-address
        # crashes such as ACL error 507011.
        if hasattr(self, "_edge_cloud_draft_intermediate_buffers"):
            delattr(self, "_edge_cloud_draft_intermediate_buffers")
        # Eagle3LlamaForCausalLM patches make_empty_intermediate_tensors on the
        # model class, but predictor here is draft_model.model (LlamaModel).
        # Fall back to draft_model so the cloud side can allocate persistent
        # intermediate buffers for the draft segment.
        make_empty_fn = getattr(
            predictor, "make_empty_intermediate_tensors", None
        ) or getattr(draft_model, "make_empty_intermediate_tensors", None)
        if make_empty_fn is not None:
            max_draft_tokens = self.max_num_tokens
            # DeepSeek-V4 MTP has asymmetric draft boundaries under SP:
            # cloud segment C consumes a local sequence shard, while edge
            # segment E consumes the full sequence for hc_head / sampling.
            # Keep the old sizing for the other draft adapters until they
            # adopt the same full-on-wire contract.
            if (
                enable_sp()
                and (
                    not self._is_deepseek_v4_mtp_edge_cloud_draft()
                    or self.edge_cloud_cfg.role == "cloud"
                )
            ):
                tp_size = self.vllm_config.parallel_config.tensor_parallel_size
                max_draft_tokens = (self.max_num_tokens + tp_size - 1) // tp_size
            self._edge_cloud_draft_intermediate_buffers = make_empty_fn(
                batch_size=max_draft_tokens,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self._edge_cloud_draft_intermediate_buffers = None

        if self.edge_cloud_cfg.role == "edge":
            seg_a = self._create_segment_callable(
                draft_model, 0, 0, is_first_segment=True, is_last_segment=False
            )
            seg_e = self._create_segment_callable(
                draft_model, 0, 0, is_first_segment=False, is_last_segment=True
            )
            self._edge_cloud_draft_segments["a"] = self._wrap_segment_if_needed(
                seg_a, is_draft=True)
            self._edge_cloud_draft_segments["e"] = self._wrap_segment_if_needed(
                seg_e, is_draft=True)
        else:
            seg_c = self._create_segment_callable(
                draft_model, 0, 0, is_first_segment=False, is_last_segment=False
            )
            self._edge_cloud_draft_segments["c"] = self._wrap_segment_if_needed(
                seg_c, is_draft=True)

    def _is_deepseek_v4_mtp_edge_cloud_draft(self) -> bool:
        """Whether the active edge-cloud drafter is DeepSeek-V4 MTP."""
        draft_model = getattr(self.drafter, "model", None)
        return (
            getattr(draft_model, "edge_cloud_draft_kind", None)
            == "deepseek_v4_mtp"
        )

    def _gather_deepseek_v4_mtp_draft_tensors(
        self,
        tensor_dict: dict[str, Any],
        num_actual_tokens: int,
    ) -> dict[str, Any]:
        """Materialize full DeepSeek-V4 MTP draft tensors for the wire.

        The edge-cloud wire contract is independent of either peer's TP size:
        every sequence tensor is sent with ``num_actual_tokens`` rows. Inputs
        that are already full are left untouched; local SP shards are gathered
        and their padding tail is removed.
        """
        if not (
            self._is_deepseek_v4_mtp_edge_cloud_draft() and enable_sp()
        ):
            return tensor_dict

        tp_size = get_tensor_model_parallel_world_size()
        padded_tokens = round_up(num_actual_tokens, tp_size)
        local_tokens = padded_tokens // tp_size
        gathered: dict[str, Any] = {}
        for key, value in tensor_dict.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim == 0
                or value.numel() == 0
            ):
                gathered[key] = value
                continue
            if value.shape[0] == num_actual_tokens:
                gathered[key] = value
                continue
            if value.shape[0] != local_tokens:
                raise RuntimeError(
                    "DeepSeek-V4 MTP draft tensor has neither full nor local "
                    f"SP shape: key={key}, rows={value.shape[0]}, "
                    f"actual_tokens={num_actual_tokens}, "
                    f"local_tokens={local_tokens}, tp_size={tp_size}"
                )
            full = tensor_model_parallel_all_gather(
                value.contiguous(), dim=0
            )
            gathered[key] = full[:num_actual_tokens]
        return gathered

    def _chunk_deepseek_v4_mtp_draft_positions(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return the local SP position shard for a DSV4 MTP segment."""
        if not (
            self._is_deepseek_v4_mtp_edge_cloud_draft()
            and enable_sp()
        ):
            return positions
        if self.uses_mrope and positions.ndim > 1:
            chunked = self._chunk_deepseek_v4_mtp_draft_tensor(
                positions.movedim(-1, 0)
            )
            return chunked.movedim(0, -1)
        return self._chunk_deepseek_v4_mtp_draft_tensor(positions)

    def _localize_deepseek_v4_mtp_draft_tensor(
        self,
        tensor: torch.Tensor,
        num_actual_tokens: int,
        *,
        token_dim: int = 0,
    ) -> torch.Tensor:
        """Accept a full or already-local tensor and return the local shard."""
        tp_size = get_tp_group().world_size
        if tp_size <= 1:
            return tensor
        token_dim %= tensor.ndim
        rows = tensor.shape[token_dim]
        local_rows = (num_actual_tokens + tp_size - 1) // tp_size
        if rows == local_rows:
            return tensor
        if rows != num_actual_tokens:
            raise RuntimeError(
                "DeepSeek-V4 MTP tensor has neither full nor local SP "
                f"shape: rows={rows}, actual_tokens={num_actual_tokens}, "
                f"local_tokens={local_rows}, token_dim={token_dim}"
            )
        if token_dim:
            tensor = tensor.movedim(token_dim, 0)
        tensor = self._chunk_deepseek_v4_mtp_draft_tensor(tensor)
        if token_dim:
            tensor = tensor.movedim(0, token_dim)
        return tensor

    def _materialize_deepseek_v4_mtp_draft_tensor(
        self,
        tensor: torch.Tensor,
        num_actual_tokens: int,
        *,
        token_dim: int = 0,
    ) -> torch.Tensor:
        """Accept a full or local SP tensor and return the unpadded full one."""
        token_dim %= tensor.ndim
        if tensor.shape[token_dim] == num_actual_tokens:
            return tensor
        if token_dim:
            tensor = tensor.movedim(token_dim, 0)
        tensor = self._gather_deepseek_v4_mtp_draft_tensors(
            {"value": tensor}, num_actual_tokens
        )["value"]
        if token_dim:
            tensor = tensor.movedim(0, token_dim)
        return tensor

    @staticmethod
    def _chunk_deepseek_v4_mtp_draft_tensor(
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Pad along the token axis and return this TP rank's draft shard.

        ``sequence_parallel_chunk`` uses a 2-D ``F.pad`` tuple and therefore
        pads the hc_mult axis for DeepSeek-V4's 3-D HC tensors. Positions are
        1-D and are not compatible with that padding tuple either. Do the
        dim-0 padding explicitly so every draft tensor follows the same token
        layout.
        """
        tp_group = get_tp_group()
        tp_size = tp_group.world_size
        if tp_size <= 1:
            return tensor
        remainder = tensor.shape[0] % tp_size
        if remainder:
            pad_shape = (
                tp_size - remainder,
                *tensor.shape[1:],
            )
            padding = torch.zeros(
                pad_shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor = torch.cat((tensor, padding), dim=0)
        chunk_size = tensor.shape[0] // tp_size
        start = tp_group.rank_in_group * chunk_size
        return tensor.narrow(0, start, chunk_size).clone()

    @staticmethod
    def _pad_deepseek_v4_mtp_draft_input_ids(
        input_ids: torch.Tensor,
        num_actual_tokens: int,
    ) -> torch.Tensor:
        """Pad full Segment-A token ids for embedding reduce-scatter.

        The vocab-parallel embedding receives the same full token sequence on
        every TP rank and reduce-scatters its output along the token axis.  Its
        input length must therefore be divisible by the TP world size.  The
        companion positions and hidden states are padded by the SP chunking
        helper; pad input ids to the identical global length before either the
        in-segment or the multimodal pre-embedding path runs.
        """
        if input_ids.ndim != 1 or input_ids.shape[0] != num_actual_tokens:
            raise RuntimeError(
                "DeepSeek-V4 MTP Segment A expected full input ids: "
                f"shape={tuple(input_ids.shape)}, "
                f"actual_tokens={num_actual_tokens}"
            )
        tp_size = get_tp_group().world_size
        remainder = num_actual_tokens % tp_size
        if tp_size <= 1 or remainder == 0:
            return input_ids
        padding = torch.zeros(
            tp_size - remainder,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat((input_ids, padding), dim=0)

    def _sync_edge_cloud_draft_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors,
    ) -> IntermediateTensors:
        """Copy received draft intermediate tensors into persistent buffers.

        ACLGraphWrapper captures and replays graphs against fixed input
        addresses. edge_cloud_broadcast_recv_draft() returns freshly-allocated
        tensors each iteration, so we copy them into pre-allocated buffers
        (sized to max_num_tokens) and return sliced views with stable
        addresses for the current num_tokens.
        """
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        dsv4_mtp_sp = (
            self._is_deepseek_v4_mtp_edge_cloud_draft() and enable_sp()
        )
        chunk_on_cloud = (
            dsv4_mtp_sp and self.edge_cloud_cfg.role == "cloud"
        )
        copy_len = (
            (num_tokens + tp_size - 1) // tp_size
            if enable_sp() and (not dsv4_mtp_sp or chunk_on_cloud)
            else num_tokens
        )

        prepared: dict[str, torch.Tensor | Any] = {}
        for key, value in intermediate_tensors.items():
            if (
                chunk_on_cloud
                and isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.numel() > 0
            ):
                if value.shape[0] != num_tokens:
                    raise RuntimeError(
                        "DeepSeek-V4 MTP cloud expected a full draft tensor "
                        f"before SP chunking: key={key}, rows={value.shape[0]}, "
                        f"tokens={num_tokens}"
                    )
                value = self._chunk_deepseek_v4_mtp_draft_tensor(value)
            prepared[key] = value

        buffers = self._edge_cloud_draft_intermediate_buffers
        if buffers is None:
            return IntermediateTensors(prepared)

        synced: dict[str, torch.Tensor | Any] = {}
        for key, value in prepared.items():
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

        # On certain devices, CPU-side all_reduce may return dirty data. 
        # When dp_allreduce_on_npu is True, route DP metadata
        # synchronization through the NPU device group to avoid data corruption.
        device_str, group = (
            ("npu", get_dp_group().device_group)
            if self.ascend_config.dp_allreduce_on_npu
            else ("cpu", get_dp_group().cpu_group)
        )
        packed_tensor = torch.zeros(2, self.dp_size, device=device_str, dtype=torch.int32)
        packed_tensor[0][self.dp_rank] = num_tokens
        packed_tensor[1][self.dp_rank] = cudagraph_mode.value
        dist.all_reduce(packed_tensor, group=group)
        if device_str == "npu":
            packed_tensor = packed_tensor.cpu()

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

    def _consume_cloud_request_corrections(
        self,
        scheduler_output: "SchedulerOutput",
        previous_num_draft_tokens: dict[str, int],
    ) -> bool:
        """Apply complete request-keyed cloud corrections without device sync.

        Returns ``True`` only when every request that participated in the
        previous speculative step has a matching, current-generation result.
        In that case the CPU batch state is authoritative and input
        preparation can use its normal single CPU-to-device copy.  Incomplete
        or stale results are left pending instead of being applied by batch
        position.
        """
        if not (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "cloud"
            and self._uses_scheduled_edge_cloud_draft()
            and scheduler_output.batch_type
            in (BatchType.PREFILL_FIRST, BatchType.DECODE_FIRST)
        ):
            return False

        participating = {
            req_id: num_draft
            for req_id, num_draft in previous_num_draft_tokens.items()
            if num_draft > 0
            and req_id in self.input_batch.req_id_to_index
        }
        if not participating:
            return False

        resolved: list[
            tuple[str, int, CloudPendingRequestCorrection]
        ] = []
        for req_id, num_draft in participating.items():
            correction = self._cloud_pending_request_corrections.get(req_id)
            latest_generation = (
                self._cloud_latest_target_generation_by_req.get(req_id)
            )
            if (
                correction is None
                or correction.generation != latest_generation
                or correction.num_draft_tokens != num_draft
            ):
                return False

            req_index = self.input_batch.req_id_to_index[req_id]
            cpu_value = int(
                self.input_batch.num_computed_tokens_cpu[req_index]
            )
            if cpu_value not in (
                correction.optimistic_num_computed_tokens,
                correction.actual_num_computed_tokens,
            ):
                logger.warning(
                    "Cloud request correction does not match scheduler "
                    "state; leaving the request-keyed correction pending: "
                    "req=%s task=%s cpu=%d optimistic=%d actual=%d",
                    req_id,
                    correction.task_id,
                    cpu_value,
                    correction.optimistic_num_computed_tokens,
                    correction.actual_num_computed_tokens,
                )
                return False
            resolved.append((req_id, req_index, correction))

        # Apply only after the complete batch has been validated so a partial
        # mismatch cannot leave CPU and GPU correction paths mixed.
        for req_id, req_index, correction in resolved:
            actual = correction.actual_num_computed_tokens
            self.input_batch.num_computed_tokens_cpu[req_index] = actual
            req_state = self.requests.get(req_id)
            if req_state is not None:
                req_state.num_computed_tokens = actual
            self.input_batch.num_accepted_tokens_cpu[req_index] = (
                correction.num_accepted_tokens
            )
            self._cloud_pending_request_corrections.pop(req_id, None)

        return True
    def _strip_tail_new_block_ids(
        self, scheduler_output: "SchedulerOutput"
    ) -> "SchedulerOutput":
        """Make a tail segment's (PL/DL) SchedulerOutput safe to re-run
        through _update_states.

        A tail SO is a verbatim copy of its head segment's SO (only
        batch_type is rewritten; the KV metadata rides along).  The head
        segment's _update_states already applied this step's
        ``new_block_ids`` via ``block_ids.extend(...)`` -- which is NOT
        idempotent.  When PD interleaving evicted a request between the
        head and the tail, the tail re-runs _update_states and extends
        the same blocks a second time.  The duplicated entries displace
        every block appended afterwards, so subsequent prefill chunks /
        decode steps read stale, duplicated, or other requests' KV
        blocks (repetitive, cross-language garbage).

        The other _update_states effects are idempotent (num_computed is
        plain assignment; new_token_ids self-guards on num_tokens), so
        only new_block_ids needs stripping.  Requests resumed from
        preemption keep their entry: that path REPLACES block_ids (also
        idempotent) and asserts new_block_ids is not None.  Re-added
        requests rebuild their block_table row from req_state.block_ids,
        which the head segment already completed -- nothing is lost.
        """
        req_data = scheduler_output.scheduled_cached_reqs
        new_block_ids = getattr(req_data, "new_block_ids", None)
        if not new_block_ids or not any(new_block_ids):
            return scheduler_output
        resumed = req_data.resumed_req_ids or set()
        stripped = replace(
            req_data,
            new_block_ids=[
                nb if req_id in resumed else None
                for req_id, nb in zip(req_data.req_ids, new_block_ids)
            ],
        )
        return replace(scheduler_output, scheduled_cached_reqs=stripped)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> Callable | None:
        # Temporary rewind guard for KV-load-failure recompute.
        # This can be removed after the upstream fix is merged.
        req_data = scheduler_output.scheduled_cached_reqs

        self._cloud_current_cpu_state_authoritative = False
        if self._edge_cloud_enabled and self.edge_cloud_cfg.role == "cloud":
            for req_id in scheduler_output.finished_req_ids:
                self._cloud_pending_request_corrections.pop(req_id, None)
                self._cloud_latest_target_generation_by_req.pop(
                    req_id, None
                )
                self._cloud_actual_num_computed_by_req.pop(req_id, None)
        if self.use_async_scheduling:
            for i, req_id in enumerate(req_data.req_ids):
                req_state = self.requests.get(req_id)
                if req_state is None:
                    continue

                num_computed_tokens = req_data.num_computed_tokens[i]
                if num_computed_tokens < req_state.num_computed_tokens:
                    req_state.prev_num_draft_len = 0

        track_cloud_corrections = (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "cloud"
            and self._uses_scheduled_edge_cloud_draft()
            and scheduler_output.batch_type
            in (BatchType.PREFILL_FIRST, BatchType.DECODE_FIRST)
        )
        previous_num_draft_tokens = (
            {
                req_id: self.requests[req_id].prev_num_draft_len
                for req_id in req_data.req_ids
                if req_id in self.requests
            }
            if track_cloud_corrections
            else {}
        )

        # A PD-interleaved tail updates only a subset of the running requests.
        # The base update removes absent requests from input_batch together
        # with their prev_req_id_to_index entries, even though their sampled
        # token rows are still pending in prev_sampled_token_ids. Preserve the
        # mappings for live requests so the next tail can merge and consume
        # those rows instead of decoding from an unfilled placeholder.
        shelved_prev_map: dict[str, int] | None = None
        if (
            self._edge_cloud_enabled
            and self.use_async_scheduling
            and self.input_batch.prev_sampled_token_ids is not None
            and self.input_batch.prev_req_id_to_index
        ):
            shelved_prev_map = dict(
                self.input_batch.prev_req_id_to_index
            )

        # Request-keyed snapshot of num_computed taken at the top of
        # every _update_states.  This must come from a map that only ever
        # holds SchedulerOutput-recorded values (_last_so_num_computed):
        # req_state.num_computed_tokens is NOT usable here, because the
        # deferred spec-decode correction rewinds it after the forward
        # (correct_spec_decode_token_counts), while the PD scheduler has
        # ALREADY applied the same rejection rewind at settle time before
        # building the next SO -- reading req_state would see a
        # double-rewound value on any rejection step (observed: worker
        # base one behind truth after a valid=1 step, then
        # token/position mismatch -> NaN row -> missing-correction crash).
        # The map batch's SO records the start-of-step num_computed, which
        # is exactly the base the correction needs.
        self._prev_num_computed_by_req = self._last_so_num_computed
        _so_nc = dict(self._last_so_num_computed)
        _req_data = scheduler_output.scheduled_cached_reqs
        if _req_data is not None and _req_data.req_ids:
            _so_nc.update(
                {
                    req_id: int(nc)
                    for req_id, nc in zip(
                        _req_data.req_ids, _req_data.num_computed_tokens
                    )
                }
            )
        # Drop entries for requests the runner no longer tracks (finished
        # and popped by the base update in earlier steps).
        self._last_so_num_computed = {
            req_id: nc
            for req_id, nc in _so_nc.items()
            if req_id in self.requests
        }

        result = super()._update_states(scheduler_output)

        has_previous_cloud_spec = any(
            num_draft > 0 and req_id in self.input_batch.req_id_to_index
            for req_id, num_draft in previous_num_draft_tokens.items()
        )
        if has_previous_cloud_spec:
            if not self._consume_cloud_request_corrections(
                scheduler_output, previous_num_draft_tokens
            ):
                # Scheduled cloud draft results must arrive before the next
                # target. There is no safe positional fallback once batches
                # can be independently reordered on the edge and cloud.
                raise RuntimeError(
                    "Cloud target is missing request-keyed speculative "
                    "corrections for one or more active requests"
                )
            # The upstream callback would apply the same rejection correction
            # later using the positional prev_req_id_to_index map.  The CPU
            # state is already corrected by request identity, so suppress it.
            result = None
            self._cloud_current_cpu_state_authoritative = True

        if shelved_prev_map:
            prev_map = self.input_batch.prev_req_id_to_index
            if prev_map is not None:
                for req_id, prev_index in shelved_prev_map.items():
                    if req_id not in prev_map and req_id in self.requests:
                        prev_map[req_id] = prev_index

        self._purge_invalidated_cloud_draft_metadata(
            getattr(scheduler_output, "cloud_draft_invalidate_task_ids", None)
        )
        return result

    def _pad_query_start_loc_for_fia(
        self,
        query_start_loc: torch.Tensor,
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

        # avoid corner case of cudagraph config mode FULL to enter the first padding logic
        # e.g. 1 request with 1 token when num_spec > 1 (num_spec = 3 and cudagraph_batch_size = 4 for example)
        # will cause tokens are padded but requests are not
        if (
            num_tokens_padded == num_reqs_padded * self.uniform_decode_query_len
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.FULL
        ):
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            last_loc = query_start_loc.np[num_reqs]
            query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1] = (
                self.arange_np[1 : num_reqs_padded + 1 - num_reqs] * self.uniform_decode_query_len + last_loc
            )
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded

            # Do not insert if the last value already equals the num_tokens
            if query_start_loc.np[num_reqs_padded] < num_tokens_padded:
                # Insert a dummy request instead of change the last value directly
                query_start_loc.np[num_reqs_padded + 1] = num_tokens_padded
                num_reqs_padded = num_reqs_padded + 1

        copy_snapshot_to_gpu(query_start_loc)

        return num_reqs_padded

    def _update_discard_request_indices(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Recompute which requests must not have their tokens sampled.

        The mask compares the optimistic sequence length (num_computed +
        scheduled) against the request's known token count; a request
        that already has more tokens than this step would reach must not
        sample again.

        Factored out of _prepare_inputs so the edge segment_e fast path
        can refresh it as well: the fast path skips _prepare_inputs,
        which previously left num_discarded_requests /
        discard_request_indices stale from the segment_a prepare.  The
        indices are ROW numbers, and interleaved batches between
        segment_a and segment_e evict/re-add requests, churning
        input_batch rows -- stale indices then marked the WRONG request
        invalid: its sampled token was dropped from prev_sampled (no
        entry for the next DF -> [EC-INV-C1], stale decode input) and
        cleared from the returned output.
        """
        num_reqs = self.input_batch.num_reqs
        num_tokens_np = np.array(
            [self.requests[r].num_tokens for r in self.input_batch.req_ids],
            dtype=np.int32,
        )
        if self.pcp_size > 1:
            # while pcp > 1, we need the original num_scheduled_tokens
            # before split to calculate discard_requests_mask
            tokens_original = [
                scheduler_output.num_scheduled_tokens[i]
                for i in self.input_batch.req_ids
            ]
            original_seq_lens_np = (
                self.input_batch.num_computed_tokens_cpu[:num_reqs]
                + np.array(tokens_original, dtype=np.int32)
            )
            discard_requests_mask = original_seq_lens_np < num_tokens_np
        else:
            num_scheduled_np = np.array(
                [
                    scheduler_output.num_scheduled_tokens[i]
                    for i in self.input_batch.req_ids
                ],
                dtype=np.int32,
            )
            optimistic_seq_lens_np = (
                self.input_batch.num_computed_tokens_cpu[:num_reqs]
                + num_scheduled_np
            )
            discard_requests_mask = optimistic_seq_lens_np < num_tokens_np

        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[: self.num_discarded_requests] = (
            discard_request_indices
        )
        self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

        self.discard_request_mask.np[:num_reqs] = discard_requests_mask
        self.discard_request_mask.copy_to_gpu(num_reqs)

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        SpecDecodeMetadata | None,
        int,
    ]:
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
                num_reqs,
                pre_pcp_qsl,
                pre_pcp_positions,
            )

        if self.use_cp:
            self.pcp_manager.init_batch_info(
                num_scheduled_tokens,
                self.input_batch.num_reqs,
                self.input_batch.num_computed_tokens_cpu,
                self.input_batch.num_prompt_tokens,
            )

        # Build prev_positions before PCP prepares full-layout spec inputs so
        # PCP can repair async sampled/draft ids with device-side index math.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        self._compute_prev_positions(num_reqs)
        prev_positions_gpu = None
        if (
            self.use_async_scheduling
            and self.input_batch.prev_sampled_token_ids is not None
            and prev_req_id_to_index
        ):
            self.prev_positions.copy_to_gpu(num_reqs)
            prev_positions_gpu = self.prev_positions.gpu[:num_reqs]

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
                prev_positions=prev_positions_gpu,
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
        copy_snapshot_to_gpu(self.query_start_loc)

        # Now, query_start_loc is padded.
        # But gdn needs an unpadded one.
        # gdn_query_start_loc is an unpadded version of query_start_loc.
        # TODO delete it if fia's check is removed.
        if self._has_gdn:
            self.gdn_query_start_loc.np[0] = 0
            self.gdn_query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
            self.gdn_query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
            copy_snapshot_to_gpu(self.gdn_query_start_loc)

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

        # Fill unused with -1. Needed for reshape_and_cache in attention_cp
        self.query_start_loc.gpu[num_reqs + 1 :].fill_(-1)

        # Copy the tensors to the NPU.
        self._prepare_input_ids(scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens)
        # Repair placeholder spec rows the native scatter in
        # _prepare_input_ids could not cover (request absent from the
        # previous execute_model batch, or the global _draft_token_ids was
        # overwritten by another request's draft chain).  Runs before the
        # VERIFY-IN log so the log shows the actual verify inputs.
        self._scatter_worker_draft_tokens_for_verify(
            scheduler_output, num_reqs, cu_num_tokens
        )
        self._snapshot_verified_draft_tokens(
            scheduler_output, num_reqs, cu_num_tokens
        )
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
        self._update_discard_request_indices(scheduler_output)

        # Sync num_accepted_tokens from CPU (set by
        # _update_states_after_model_execute for hybrid models).
        if self._cloud_current_cpu_state_authoritative:
            # Request-keyed cloud correction already materialized counts in
            # the current input_batch row order.  Reusing prev_positions here
            # would reintroduce the stale-positive-index bug.
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            )
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()
        elif self.num_accepted_tokens_event is not None:
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
        # Edge-cloud tail segments reuse the GPU num_computed_tokens
        # corrected by their head segment, so the sync below is normally
        # skipped. That reuse is only safe when a re-sync would be wrong or
        # unneeded:
        #   * embedding_only: the edge has no attention layers, so stale
        #     positions / seq_lens / slot_mapping cannot corrupt attention.
        #   * async spec decode (MTP/eagle3): the head already wrote THIS
        #     step's corrected values into the GPU buffer; re-running the
        #     correction kernel on the tail reads those already-corrected
        #     values (buffer indices overlap) and double-counts the accepted
        #     tokens, and the CPU fallback is optimistic -> both wrong.
        # head_tail + non-spec-decode tails MUST sync on the slow path: when
        # PD interleaving churns input_batch between the head and tail
        # segments, the tail's _update_states refreshes only the CPU
        # num_computed_tokens while the GPU buffer keeps the head batch's
        # stale values -> positions / seq_lens / slot_mapping shift -> KV
        # cache writes land in wrong slots -> whole-batch corruption.
        # Spec-decode tails normally skip the sync and reuse the head
        # segment's corrected values -- but ONLY when the resident batch is
        # still the head's batch.  If an interleaved batch (e.g. another
        # request's PREFILL_FIRST) evicted and re-added the requests since
        # the head ran, the GPU rows hold the evictor's values and must be
        # resynced from the (exact, SO-derived) CPU state.
        _skip_tail_sync = (
            self._is_edge_cloud_tail_segment
            and (
                self.edge_cloud_cfg.mode == "embedding_only"
                or self.use_async_spec_decode
            )
            and self._edge_tail_membership_intact
        )
        # Locals consumed unconditionally below (pcp rebuild,
        # async_spec_decode_active check, mrope drift). Bind them to None
        # when the tail sync is skipped so those reads see "no correction
        # available" instead of raising UnboundLocalError.
        valid_sampled_token_count_gpu = None
        computed_token_tensor_cpu = None
        if not _skip_tail_sync:
            valid_sampled_token_count_gpu = self.valid_sampled_token_count_gpu
            if (
                self.use_async_spec_decode
                and not self._cloud_current_cpu_state_authoritative
            ):
                computed_token_tensor_cpu = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].to(
                    device=self.device, non_blocking=True
                )
            if self._cloud_current_cpu_state_authoritative:
                self.num_computed_tokens[:num_reqs].copy_(
                    self.input_batch.num_computed_tokens_cpu_tensor[
                        :num_reqs
                    ],
                    non_blocking=True,
                )
                # The authoritative path has no GPU-vs-CPU drift.  Reuse the
                # destination view below instead of issuing a second H2D copy
                # solely for M/XD-RoPE drift calculation.
                computed_token_tensor_cpu = self.num_computed_tokens[
                    :num_reqs
                ]
            elif (
                self.use_async_spec_decode
                and valid_sampled_token_count_gpu is not None
                and prev_req_id_to_index
                # Edge tail segments never take the correction kernel: their
                # SchedulerOutput carries the same (head-corrected)
                # num_computed as their head segment, and the prev map /
                # valid counts still belong to the PREVIOUS chain, so the
                # kernel would double-apply that chain's accepted counts.
                # Tails reaching this block (membership churned between
                # head and tail, see _skip_tail_sync) fall through to the
                # plain CPU copy, which is exact here.
                and not self._is_edge_cloud_tail_segment
            ):
                if prev_positions_gpu is None:
                    self.prev_positions.copy_to_gpu(num_reqs)
                self.prev_num_draft_tokens.copy_to_gpu()
                # Identity-keyed correction base (see _update_states): the
                # positional GPU buffer may hold another request's value
                # after PD-interleaved eviction/re-add churn.
                prev_computed_by_req = torch.tensor(
                    [
                        self._prev_num_computed_by_req.get(req_id, -1)
                        for req_id in self.input_batch.req_ids[:num_reqs]
                    ],
                    dtype=torch.int32,
                ).to(self.device, non_blocking=True)
                update_num_computed_tokens_for_batch_change(
                    self.num_computed_tokens,
                    self.num_accepted_tokens.gpu[:num_reqs],
                    self.prev_positions.gpu[:num_reqs],
                    valid_sampled_token_count_gpu,
                    self.prev_num_draft_tokens.gpu,
                    computed_token_tensor_cpu,
                    prev_computed_by_req=prev_computed_by_req,
                )
                # Edge-cloud cloud side: rows with no previous-batch mapping
                # (prev_positions == -1) are skipped by the correction
                # kernel.  A request that briefly left the cloud batch (e.g.
                # while its verify placeholder was in flight, or during the
                # prefill_last_pending -> running migration gap) returns to
                # a row recycled by condense, whose GPU num_computed still
                # holds the previous occupant's stale value; the skip then
                # freezes its attention seq len and the next reads hit
                # unwritten KV -> NaN (the frozen-request issue).  The
                # SO-derived CPU value is authoritative for such rows
                # (brand-new and returning requests alike), so resync them.
                if (
                    self._edge_cloud_enabled
                    and self.edge_cloud_cfg.role == "cloud"
                    and computed_token_tensor_cpu is not None
                ):
                    stale_rows = np.nonzero(
                        self.prev_positions.np[:num_reqs] < 0
                    )[0]
                    if len(stale_rows):
                        stale_idx = torch.from_numpy(stale_rows).to(
                            self.device, non_blocking=True
                        )
                        self.num_computed_tokens[stale_idx] = (
                            computed_token_tensor_cpu[stale_idx]
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

        pcp_manager = getattr(self, "pcp_manager", None)
        if pcp_manager is not None:
            cp_async_rebuild = pcp_manager.rebuild_async_spec_decode_inputs(
                use_async_spec_decode=self.use_async_spec_decode,
                valid_sampled_token_count_gpu=valid_sampled_token_count_gpu,
                prev_req_id_to_index=prev_req_id_to_index,
                prev_positions_gpu=prev_positions_gpu,
                with_prefill=with_prefill,
                enable_prompt_embeds=self.enable_prompt_embeds,
                has_req_prompt_embeds=bool(self.input_batch.req_prompt_embeds),
                supports_mm_inputs=self.supports_mm_inputs,
                num_reqs=num_reqs,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
                req_indices=req_indices,
                req_indices_gpu=req_indices_gpu,
                position_pcp=position_pcp if self.pcp_size > 1 else None,
                query_pos_gpu=self.query_pos.gpu,
                query_pos_np=self.query_pos.np,
                positions=self.positions,
                positions_np=positions_np,
                num_computed_tokens=self.num_computed_tokens,
                num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu,
                prev_positions_np=self.prev_positions.np,
                prev_num_draft_tokens_np=self.prev_num_draft_tokens.np,
                valid_sampled_token_count_event=self.valid_sampled_token_count_event,
                valid_sampled_token_count_cpu=self.valid_sampled_token_count_cpu,
                input_batch=self.input_batch,
                input_ids=self.input_ids,
                scheduler_output=scheduler_output,
                arange_np=self.arange_np,
                cu_num_tokens=cu_num_tokens,
                draft_token_ids=self._draft_token_ids,  # type: ignore[has-type]
                num_spec_tokens=self.num_spec_tokens,
                prepare_input_ids=self._prepare_input_ids,
            )
        else:
            cp_async_rebuild = PCPAsyncSpecDecodeRebuildResult(
                rebuilt=False,
                positions_ready_on_device=False,
            )

        if cp_async_rebuild.positions_ready_on_device:
            pass
        elif self.pcp_size > 1 or cp_async_rebuild.rebuilt:
            # PCP and async rebuild both compute the correct positions on CPU.
            # Copy positions_np to GPU so input_ids and positions stay aligned.

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

        # In async spec decode mode, optimistic_seq_lens_cpu assumes all
        # tokens from the previous speculative step were accepted. Correct it
        # on CPU using the valid-sampled-token counts that are already copied
        # asynchronously for scheduler bookkeeping. This avoids an extra
        # NPU->CPU seq_lens copy and the synchronize() in attention metadata.
        # Mirrors update_num_computed_tokens_for_batch_change on the GPU side.
        async_spec_decode_active = (
            self.use_async_spec_decode
            and valid_sampled_token_count_gpu is not None
            and prev_req_id_to_index
            and not self._cloud_current_cpu_state_authoritative
        )
        if self._needs_seq_lens_cpu_sync and async_spec_decode_active:
            self._correct_optimistic_seq_lens_cpu(num_reqs)

        # For non-PCP, compute slot_mapping on GPU. PCP slot_mapping was
        # already computed on GPU before PCP split the positions.
        if self.pcp_size <= 1:
            self.input_batch.block_table.compute_slot_mapping(
                num_reqs,
                self.query_start_loc.gpu[: num_reqs + 1],
                self.positions[:total_num_scheduled_tokens],
            )

        if (
            self.use_async_spec_decode
            and (self.uses_mrope or self.uses_xdrope_dim > 0)
            # None when the tail sync was skipped above: num_computed_tokens
            # already holds the head-corrected values, so there is no drift
            # to apply (and no CPU baseline to diff against).
            and computed_token_tensor_cpu is not None
        ):
            drift = self.num_computed_tokens[req_indices_gpu].to(
                torch.int64
            ) - computed_token_tensor_cpu[req_indices_gpu]
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

    def _sanitize_placeholder_input_ids_for_forward(
        self,
        scheduler_output: "SchedulerOutput",
        num_forward_tokens: int,
    ) -> None:
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not scheduled_spec_tokens:
            return
        if not any(
            PLACEHOLDER_TOKEN_ID in token_ids
            for token_ids in scheduled_spec_tokens.values()
        ):
            return

        input_ids = self.input_ids.gpu[:num_forward_tokens]
        input_ids.masked_fill_(input_ids == PLACEHOLDER_TOKEN_ID, 0)

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
            assert num_pcp_pads is not None
            if self.pcp_manager.pcp_use_hybrid_attn:
                if self.pcp_manager.num_prefill_reqs > 0:
                    cu_num_scheduled_tokens = (
                        self.pcp_manager.adjust_cu_num_scheduled_tokens_for_pcp(
                            cu_num_scheduled_tokens, num_pcp_pads
                        )
                    )
            else:
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

    def _correct_optimistic_seq_lens_cpu(self, num_reqs: int) -> None:
        """Correct ``optimistic_seq_lens_cpu`` for async spec-decode drift.

        The valid-sampled-token counts that drive the correction are copied
        device->host on a side stream at the end of the *previous* step (see
        :meth:`_copy_valid_sampled_token_count`). The host buffer must not be
        read until that copy has completed, otherwise the correction consumes
        stale counts and corrupts the CPU seq_lens. Callers that still build
        metadata from optimistic CPU seq_lens need this correction before
        attention metadata construction.

        Synchronizing on the event before the host read mirrors vLLM's own
        :meth:`_get_valid_sampled_token_count`. Because the copy was launched a
        full step earlier, the event is already signalled in steady state and
        the synchronize is effectively a no-op -- it does not reintroduce the
        seq_lens device->host copy + synchronize that this optimization removed.
        """
        assert self.valid_sampled_token_count_event is not None
        assert self.valid_sampled_token_count_cpu is not None
        self.valid_sampled_token_count_event.synchronize()
        correct_optimistic_seq_lens_cpu(
            self.optimistic_seq_lens_cpu.numpy(),
            self.prev_positions.np,
            self.prev_num_draft_tokens.np,
            self.valid_sampled_token_count_cpu.numpy(),
            num_reqs,
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
        elif isinstance(self.drafter, AscendNgramProposer):
            if vllm_version_is("0.23.0"):
                draft_token_ids = self.drafter.propose(valid_sampled_token_ids)
            else:
                draft_token_ids = self.drafter.propose(
                    scheduler_output.num_spec_tokens_to_schedule,
                    valid_sampled_token_ids,
                    self.input_batch.num_tokens_no_spec,
                    self.input_batch.token_ids_cpu,
                )
        elif isinstance(self.drafter, AscendSuffixDecodingProposer):
            if vllm_version_is("0.23.0"):
                draft_token_ids = self.drafter.propose(valid_sampled_token_ids)
            else:
                draft_token_ids = self.drafter.propose(
                    valid_sampled_token_ids,
                    num_speculative_tokens=scheduler_output.num_spec_tokens_to_schedule,
                )
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

            if vllm_version_is("0.23.0"):
                draft_token_ids = self.drafter.propose(
                    sampled_token_ids=valid_sampled_token_ids,
                    target_hidden_states=target_hidden_states,
                    common_attn_metadata=common_attn_metadata,
                )
            else:
                draft_token_ids = self.drafter.propose(
                    self.speculative_config.num_speculative_tokens,
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
                self._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)

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
            # In edge-cloud EAGLE3 mode the target model's aux hidden states are
            # fused on the cloud side; the edge proposer only needs a placeholder
            # tensor whose last dim matches the draft model's hidden size.
            is_edge_cloud_eagle3 = (
                self._edge_cloud_enabled
                and self.speculative_config is not None
                and self.speculative_config.method == "eagle3"
            )
            if spec_decode_metadata is None:
                # update pcp related params
                if self.pcp_size > 1:
                    token_indices_to_sample = query_start_loc_pcp_full[1 : num_reqs + 1] - 1
                    target_token_ids = input_ids_pcp_full[:num_scheduled_tokens]
                    target_positions = self._get_positions(num_scheduled_tokens)
                    target_hidden_states = hidden_states
                    if self.use_aux_hidden_state_outputs:
                        if is_edge_cloud_eagle3:
                            target_hidden_states = torch.zeros(
                                num_scheduled_tokens,
                                self.drafter.hidden_size,
                                dtype=hidden_states.dtype,
                                device=hidden_states.device,
                            )
                        else:
                            target_hidden_states = torch.cat([h for h in aux_hidden_states], dim=-1)
                else:
                    token_indices_to_sample = None
                    # input_ids can be None for multimodal models.
                    target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                    target_positions = self._get_positions(num_scheduled_tokens)
                    if self.use_aux_hidden_state_outputs:
                        if is_edge_cloud_eagle3:
                            target_hidden_states = torch.zeros(
                                num_scheduled_tokens,
                                self.drafter.hidden_size,
                                dtype=hidden_states.dtype,
                                device=hidden_states.device,
                            )
                        else:
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
                        if is_edge_cloud_eagle3:
                            target_hidden_states = torch.zeros(
                                token_indices.shape[0],
                                self.drafter.hidden_size,
                                dtype=hidden_states.dtype,
                                device=hidden_states.device,
                            )
                        else:
                            target_hidden_states = torch.cat([h for h in aux_hidden_states], dim=-1)
                else:
                    target_token_ids = self.input_ids.gpu[token_indices]
                    target_positions = self._get_positions(token_indices)
                    if self.use_aux_hidden_state_outputs:
                        if is_edge_cloud_eagle3:
                            target_hidden_states = torch.zeros(
                                token_indices.shape[0],
                                self.drafter.hidden_size,
                                dtype=hidden_states.dtype,
                                device=hidden_states.device,
                            )
                        else:
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
        else:
            raise ValueError(f"Unknown speculative decoding method: {self.speculative_config.method}")

        return draft_token_ids

    def _scatter_worker_draft_tokens_for_verify(
        self,
        scheduler_output: "SchedulerOutput",
        num_reqs: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Patch the spec region of input_ids.gpu with worker-local draft
        token IDs, keyed by req_id.

        Native vLLM repairs the placeholder spec tokens of an async verify
        batch on the GPU side in _prepare_input_ids, but only for requests
        that were present in the previous execute_model batch
        (prev_positions >= 0): it indexes self._draft_token_ids by the
        request's *previous* batch position.  Two edge-cloud scheduled-MTP
        situations defeat that scatter:

        1. Another request's batch (e.g. a PREFILL_FIRST) executed between
           this request's PREFILL_LAST and its first DECODE_FIRST, wiping
           prev_req_id_to_index, so the request re-enters as "new" and the
           native scatter silently skips it, leaving -1 placeholders in
           input_ids.gpu.  The verify forward then embeds zero-masked
           placeholder rows instead of the real draft tokens, every draft
           is rejected, and the acceptance rate collapses (while the
           output text stays correct via the bonus token).
        2. With several request groups in flight, another draft chain can
           overwrite the global self._draft_token_ids before this verify
           runs, so even a covered request gets a *different* request's
           drafts scattered into its spec rows.

        Both are fixed here by writing each request's spec rows from
        _worker_draft_token_ids_by_req, which is recorded per request when
        its draft chain completes and is immune to both effects.  Requests
        without an entry keep the native behavior (placeholders included),
        so non-edge-cloud async flows are unaffected.
        """
        if not self.use_async_scheduling:
            return
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not scheduled_spec_tokens:
            return
        drafts_by_req = self._worker_draft_token_ids_by_req
        if not drafts_by_req:
            return
        for cur_index in range(num_reqs):
            req_id = self.input_batch.req_ids[cur_index]
            draft_len = len(scheduled_spec_tokens.get(req_id, ()))
            if draft_len <= 0:
                continue
            entry = drafts_by_req.get(req_id)
            if entry is None:
                continue
            # Spec tokens occupy the last draft_len positions of the
            # request's scheduled tokens (mirrors the upstream scatter).
            end = int(cu_num_tokens[cur_index])
            take = min(draft_len, entry.shape[0])
            if take <= 0:
                continue
            self.input_ids.gpu[end - take:end] = entry[:take].to(
                dtype=torch.int32
            )

    def _uses_scheduled_edge_cloud_draft(self) -> bool:
        speculative_config = self.speculative_config
        if speculative_config is None:
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

    def _should_defer_edge_cloud_draft(
        self, scheduler_output: "SchedulerOutput"
    ) -> bool:
        return bool(
            self._uses_scheduled_edge_cloud_draft()
            and self._edge_cloud_enabled
            and is_edge_device()
            and self.drafter is not None
            and scheduler_output.batch_type in (
                BatchType.PREFILL_LAST,
                BatchType.DECODE_LAST,
            )
        )

    def _snapshot_mtp_target_hidden_states(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> torch.Tensor | None:
        """Freeze this target batch's pre-HC state before sampling."""
        if not self._should_defer_edge_cloud_draft(scheduler_output):
            return None
        mtp_hidden_states = getattr(
            self.get_model(),
            "get_mtp_target_hidden_states",
            lambda: None,
        )()
        if mtp_hidden_states is None:
            return None
        scheduled_token_count = sum(
            int(token_count)
            for token_count in scheduler_output.num_scheduled_tokens.values()
        )
        return mtp_hidden_states[:scheduled_token_count].clone()

    def _edge_cloud_drafter_uses_graph(self) -> bool:
        """Return whether the loaded edge-cloud draft segments use ACL graphs."""
        drafter = self.drafter
        if drafter is None or not drafter.use_cuda_graph:
            return False

        draft_model = getattr(drafter, "model", None)
        return not getattr(
            draft_model,
            "edge_cloud_dynamic_step_segments",
            False,
        )

    def _snapshot_verified_draft_tokens(
        self,
        scheduler_output: "SchedulerOutput",
        num_reqs: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Save the exact speculative rows consumed by a target verify."""
        if (
            not self.use_async_scheduling
            or not self._edge_cloud_enabled
            or self.edge_cloud_cfg.role != "edge"
            or not scheduler_output.head_token
            or not scheduler_output.scheduled_spec_decode_tokens
        ):
            return

        verified_rows: dict[str, torch.Tensor] = {}
        for req_idx in range(num_reqs):
            req_id = self.input_batch.req_ids[req_idx]
            draft_len = len(
                scheduler_output.scheduled_spec_decode_tokens.get(
                    req_id, ()
                )
            )
            if not draft_len:
                continue
            draft_row = self._worker_draft_token_ids_by_req.get(req_id)
            if draft_row is not None:
                # _scatter_worker_draft_tokens_for_verify just wrote this
                # exact row into input_ids.gpu. Keep its producing tensor
                # alive even if the per-request map is later cleaned up.
                verified_rows[req_id] = draft_row
            else:
                # Compatibility fallback for a row handled entirely by the
                # upstream native scatter.
                end = int(cu_num_tokens[req_idx])
                verified_rows[req_id] = self.input_ids.gpu[
                    end - draft_len : end
                ].clone()

        if verified_rows:
            self._verified_draft_token_ids_by_head[
                scheduler_output.head_token
            ] = verified_rows

    def _patch_deferred_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        req_ids: tuple[str, ...],
        num_scheduled: list[int],
        scheduled_token_ids: torch.Tensor,
    ) -> None:
        """Replace async spec placeholders with the verified draft IDs.

        ``self._draft_token_ids`` is a single global tensor and may already
        belong to another interleaved request group by the time the matching
        target tail stashes its deferred draft context.  Prefer the exact
        per-head rows captured from the repaired verify input, with the
        per-request DRAFT-OUT rows as a compatibility fallback.
        """
        scheduled_spec_tokens = (
            scheduler_output.scheduled_spec_decode_tokens
        )
        if not self.use_async_scheduling or not scheduled_spec_tokens:
            return

        verified_rows = self._verified_draft_token_ids_by_head.pop(
            scheduler_output.head_token, None
        )
        draft_rows: list[torch.Tensor] = []
        draft_req_ids: list[str] = []
        for req_id in req_ids:
            if not scheduled_spec_tokens.get(req_id):
                continue
            draft_row = (
                verified_rows.get(req_id)
                if verified_rows is not None
                else None
            )
            if draft_row is None:
                draft_row = self._worker_draft_token_ids_by_req.get(req_id)
                if draft_row is not None:
                    logger.warning(
                        "Verified draft snapshot missing request row; "
                        "using per-request fallback: task_id=%s, req_id=%s",
                        scheduler_output.head_token,
                        req_id,
                    )
            if draft_row is None:
                raise RuntimeError(
                    "Deferred draft context has no verified draft tokens: "
                    f"task_id={scheduler_output.head_token}, "
                    f"req_id={req_id}"
                )
            draft_rows.append(draft_row)
            draft_req_ids.append(req_id)

        if not draft_rows:
            return

        draft_token_ids_cpu = torch.stack(draft_rows).detach().cpu()
        draft_row_by_req = {
            req_id: row for row, req_id in enumerate(draft_req_ids)
        }
        start = 0
        for req_idx, scheduled in enumerate(num_scheduled):
            end = start + scheduled
            req_id = req_ids[req_idx]
            draft_len = len(scheduled_spec_tokens.get(req_id, ()))
            if draft_len:
                if draft_len > scheduled:
                    raise RuntimeError(
                        "Deferred draft token count exceeds scheduled tokens: "
                        f"task_id={scheduler_output.head_token}, "
                        f"req_id={req_id}, draft_len={draft_len}, "
                        f"scheduled={scheduled}"
                    )
                draft_row = draft_token_ids_cpu[
                    draft_row_by_req[req_id]
                ]
                if draft_row.shape[0] < draft_len:
                    raise RuntimeError(
                        "Verified draft row is shorter than the scheduled "
                        "draft tokens: "
                        f"task_id={scheduler_output.head_token}, "
                        f"req_id={req_id}, row_len={draft_row.shape[0]}, "
                        f"draft_len={draft_len}"
                    )
                scheduled_token_ids[end - draft_len : end] = (
                    draft_row[:draft_len].to(torch.long)
                )
            start = end

    def _stash_pending_edge_cloud_draft_context(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: torch.Tensor | list[list[int]],
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mtp_target_hidden_states: torch.Tensor | None,
    ) -> None:
        if scheduler_output.head_token is None:
            raise RuntimeError(
                "Cannot defer edge-cloud draft without the target head_token"
            )
        # The target head_token is already globally unique and is present on
        # both the cloud target step and the edge tail step.  Reusing it as
        # the draft-chain identity gives the cloud an exact metadata lookup
        # key even if several equal-shaped prefill/decode batches interleave.
        task_id = scheduler_output.head_token
        req_ids = tuple(self.input_batch.req_ids)
        num_reqs = len(req_ids)
        # The first draft pass (spec_step_idx == 0) must run over ALL tokens
        # the target model just processed (the full prompt after a prefill,
        # all verify tokens after a decode step), not just the last row of
        # each request. This is what populates the draft layer's own KV
        # cache with the prompt, and the cloud reuses the target step's
        # attention metadata verbatim for step 0, so the token counts must
        # match exactly.  Selecting only the last num_reqs rows here makes
        # the cloud run e.g. a 1-token query against a 57-token prefill
        # metadata, which fails the FusedInferAttention TND tiling check.
        num_scheduled = [
            int(scheduler_output.num_scheduled_tokens.get(req_id, 1))
            for req_id in req_ids
        ]
        # Strip acl_graph capture-bucket padding (e.g. 60 -> 64 tokens for a
        # 15-req verify step) so the stashed tensors match the *real*
        # scheduled_token_ids built below.  Without this, the deferred draft
        # concatenates a 60-row inputs_embeds with a 64-row hidden_states and
        # crashes in aclnnCat (error EZ1001) once the running batch drops below
        # a capture boundary (e.g. 16 -> 15 requests).
        scheduled_token_count = sum(num_scheduled)
        draft_positions = positions[:scheduled_token_count].clone()
        if mtp_target_hidden_states is not None:
            draft_hidden_states = mtp_target_hidden_states
        else:
            # The draft needs the target hidden states of every scheduled token
            # (sample_hidden_states only covers the logits rows).
            draft_hidden_states = hidden_states[:scheduled_token_count].clone()

        # Snapshot the scheduled token ids so the first draft step can build
        # the shifted input ids (target ids shifted left by one, closed by
        # the sampled next token), plus the per-request row whose hidden
        # state produces the proposed draft token.
        # The row that produces the proposed draft token is the last
        # scheduled row for a prefill; for a verify (decode) step it is the
        # row of the last ACCEPTED token, i.e. rejected draft tokens at the
        # tail of each request are skipped.  Using the last verify row
        # unconditionally would feed rejected-token hidden states/positions
        # into the next draft round and drift positions past the real
        # sequence.
        if scheduler_output.batch_type == BatchType.PREFILL_LAST:
            num_rejected = [0] * num_reqs
        elif torch.is_tensor(sampled_token_ids):
            valid_counts = (
                (sampled_token_ids[:num_reqs] != -1).sum(dim=1).tolist()
            )
            num_rejected = [
                max(n - max(int(v), 1), 0)
                for n, v in zip(num_scheduled, valid_counts)
            ]
        else:
            num_rejected = [
                max(n - max(len(s), 1), 0)
                for n, s in zip(num_scheduled, sampled_token_ids)
            ]
        pos_flat = positions[0] if positions.dim() == 2 else positions
        start_offsets = torch.zeros(num_reqs, dtype=torch.long)
        running = 0
        for req_idx, n in enumerate(num_scheduled):
            start_offsets[req_idx] = running
            running += n
        total_tokens = running
        start_pos = pos_flat[start_offsets.to(pos_flat.device)].cpu()
        scheduled_token_ids = torch.empty(total_tokens, dtype=torch.long)
        sample_row_indices = torch.empty(num_reqs, dtype=torch.long)
        start = 0
        for req_idx, n in enumerate(num_scheduled):
            end = start + n
            p0 = int(start_pos[req_idx])
            scheduled_token_ids[start:end] = torch.from_numpy(
                self.input_batch.token_ids_cpu[req_idx, p0 : p0 + n]
            ).to(torch.long)
            sample_row_indices[req_idx] = end - 1 - num_rejected[req_idx]
            start = end

        # Async scheduled MTP: the scheduler only sent fixed-length -1
        # placeholder spec tokens, so the spec region of token_ids_cpu read
        # above holds placeholders (native vLLM only ever repairs them on
        # the GPU side in _prepare_input_ids).  The real draft token ids are
        # worker-local -- the same tensor that was scattered into
        # input_ids.gpu for this verify forward.  Patch the spec rows here
        # so the first draft step embeds the true verified draft tokens
        # instead of placeholder embeddings; otherwise the draft hidden
        # states/KV diverge from the non-placeholder semantics and the
        # acceptance pattern changes even though the target verify (and
        # thus the output text) is unaffected.
        #
        self._patch_deferred_draft_token_ids(
            scheduler_output,
            req_ids,
            num_scheduled,
            scheduled_token_ids,
        )

        frozen_sampled_token_ids = _freeze_scheduled_state(
            sampled_token_ids
        )
        context: dict[str, Any] = {
            "positions": draft_positions,
            "hidden_states": draft_hidden_states,
            "num_scheduled_tokens": num_scheduled,
            "scheduled_token_ids": scheduled_token_ids,
            "sample_row_indices": sample_row_indices,
            "req_ids": req_ids,
            "draft_step_idx": 0,
            # Mid-prefill chunks run the same draft forward to populate MTP
            # KV, but their proposals must not be exposed to target decode.
            "is_last_prefill_chunk": getattr(
                scheduler_output, "is_last_prefill_chunk", True
            ),
            "draft_output_req_ids": tuple(
                getattr(
                    scheduler_output,
                    "draft_output_req_ids",
                    req_ids,
                )
            ),
        }
        self._pending_edge_cloud_draft_contexts[task_id] = context

        if torch.is_tensor(sampled_token_ids):
            assert self.drafter is not None
            next_token_ids, valid_sampled_tokens_count = (
                self.drafter.prepare_next_token_ids_padded(
                    frozen_sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_indices.gpu,
                    self.num_discarded_requests,
                )
            )
            self._copy_valid_sampled_token_count(
                next_token_ids, valid_sampled_tokens_count
            )
            context["next_token_ids"] = next_token_ids.clone()
        else:
            next_token_list = []
            for req_idx, req_id in enumerate(req_ids):
                sampled_tokens = sampled_token_ids[req_idx]
                next_token_list.append(
                    sampled_tokens[-1]
                    if sampled_tokens
                    else self.requests[req_id].get_token_id(
                        int(self.input_batch.num_tokens_no_spec[req_idx]) - 1
                    )
                )
            context["next_token_ids"] = torch.tensor(
                next_token_list, dtype=torch.long
            )
        logger.info(
            "[MTP-DEBUG] pending draft context stashed: task_id=%s, "
            "req_ids=%s, draft_step_idx=%s, pending_contexts=%d",
            task_id,
            req_ids,
            context["draft_step_idx"],
            len(self._pending_edge_cloud_draft_contexts),
        )
        # In async mode the global tensor may already belong to another
        # request group's draft chain.  Do not clear it here; per-request
        # rows are cleaned up when their requests finish.
        if not self.use_async_scheduling:
            self._draft_token_ids = None
            self._draft_token_ids_req_ids = None

    def clear_pending_edge_cloud_draft_for_req_ids(
        self,
        req_ids: set[str] | list[str],
        force_drop_task_ids: set[str] | list[str] = (),
    ) -> None:
        """Mark finished requests on pending deferred drafts.

        Aligned with the non-edge-cloud behavior, where the drafter still
        runs over the whole verify batch and finished requests' outputs
        are discarded afterwards: a pending draft context is dropped ONLY
        when every request of its parent batch has finished.  Partial
        finishes keep the draft alive — the cloud-side cached attention
        metadata is whole-batch, so dropping/filtering rows here would
        desync the token counts; the dead rows' draft tokens are instead
        discarded when the chain completes (see
        _run_edge_cloud_draft_last_segment).

        ``force_drop_task_ids`` carries chains the scheduler cut from its
        ready queues (all requests finished). A context with a suspended
        DRAFT_FIRST is retained even then: its cloud peer already received
        the control task and is waiting for the payload, so the preceding
        tail must resume that head before the chain can be drained safely.
        """
        req_id_set = set(req_ids)
        for req_id in req_id_set:
            self._worker_draft_token_ids_by_req.pop(req_id, None)
        force_dropped = set(force_drop_task_ids)
        for task_id, context in list(
            self._pending_edge_cloud_draft_contexts.items()
        ):
            ctx_req_ids = context.get("req_ids") or ()
            hit = req_id_set.intersection(ctx_req_ids)
            if hit:
                finished = context.setdefault("finished_req_ids", set())
                finished.update(hit)
                if not all(req_id in finished for req_id in ctx_req_ids):
                    # Partial finish: keep the draft alive.
                    continue
            elif task_id not in force_dropped:
                continue
            if context.get("deferred_draft_first_steps"):
                # The cloud has already entered recv for this head. Keep the
                # causal state until the prior tail resumes and sends it.
                continue
            self._pending_edge_cloud_draft_contexts.pop(task_id, None)

    def _get_pending_edge_cloud_draft_context(
        self, scheduler_output: "SchedulerOutput"
    ) -> dict[str, Any]:
        task_id = scheduler_output.draft_task_id
        if task_id is None:
            raise RuntimeError("DRAFT batch missing draft_task_id")
        context = self._pending_edge_cloud_draft_contexts.get(task_id)
        if context is None:
            raise RuntimeError(
                "DRAFT batch has no pending draft context: "
                f"task_id={task_id}"
            )
        return context

    def is_edge_cloud_draft_step_ready(
        self, scheduler_output: "SchedulerOutput"
    ) -> bool:
        """Whether this edge draft head's causal inputs are materialized."""
        step = int(scheduler_output.draft_step_idx or 0)
        if step == 0:
            return True
        task_id = scheduler_output.draft_task_id
        if task_id is None:
            return False
        context = self._pending_edge_cloud_draft_contexts.get(task_id)
        if context is None or int(context.get("draft_step_idx", 0)) < step:
            return False
        return all(
            key in context
            for key in (
                "last_draft_token_ids",
                "last_draft_positions",
                "last_draft_hidden_states",
            )
        )

    def mark_edge_cloud_draft_step_deferred(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Pin a context whose already-dispatched head awaits its prior tail."""
        context = self._get_pending_edge_cloud_draft_context(
            scheduler_output
        )
        context.setdefault("deferred_draft_first_steps", set()).add(
            int(scheduler_output.draft_step_idx or 0)
        )

    def unmark_edge_cloud_draft_step_deferred(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        task_id = scheduler_output.draft_task_id
        if task_id is None:
            return
        context = self._pending_edge_cloud_draft_contexts.get(task_id)
        if context is None:
            return
        deferred = context.get("deferred_draft_first_steps")
        if deferred is not None:
            deferred.discard(int(scheduler_output.draft_step_idx or 0))

    def _prepare_edge_cloud_draft_step_inputs(
        self, scheduler_output: "SchedulerOutput"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        context = self._get_pending_edge_cloud_draft_context(
            scheduler_output
        )
        draft_step_idx = int(scheduler_output.draft_step_idx or 0)
        # The chain's SchedulerOutputs are frozen copies of the parent
        # target batch and the context rows are laid out in that batch's
        # request order.  Both must agree on the request set: a diverged
        # set means rows would be consumed positionally against the wrong
        # requests (shifted draft rows), so fail loudly instead.
        context_req_ids = set(context["req_ids"])
        so_req_ids = set(scheduler_output.num_scheduled_tokens)
        if so_req_ids != context_req_ids:
            raise RuntimeError(
                "DRAFT step request set diverged from its stashed "
                f"context: task_id={scheduler_output.draft_task_id} "
                f"step={draft_step_idx} "
                f"missing_in_so={sorted(context_req_ids - so_req_ids)} "
                f"extra_in_so={sorted(so_req_ids - context_req_ids)}"
            )
        if draft_step_idx > 0:
            context["current_draft_full_positions"] = (
                context["last_draft_full_positions"] + 1
            )
            return (
                context["last_draft_token_ids"],
                context["last_draft_positions"] + 1,
                context["last_draft_hidden_states"],
                draft_step_idx,
            )

        # First speculative step: run the draft model over ALL tokens the
        # target model just processed.  The draft input ids are the target
        # token ids shifted left by one within each request, so the draft
        # layer's KV cache is populated for the whole sequence. The sampled
        # next token must close each request at the last ACCEPTED row
        # (sample_row_indices), exactly like the stock drafter's
        # input_ids[token_indices_to_sample] = next_token_ids -- NOT at the
        # last scheduled row.  When some verify draft tokens were rejected,
        # putting it at end-1 instead feeds a rejected token id (and its
        # embedding/KV) into the row that produces the next draft token,
        # which makes every decode-round draft miss.
        num_scheduled = context["num_scheduled_tokens"]
        scheduled_token_ids = context["scheduled_token_ids"]
        sample_row_indices = context["sample_row_indices"]
        next_token_ids = context["next_token_ids"].cpu()
        total_tokens = scheduled_token_ids.shape[0]
        input_ids = torch.empty(total_tokens, dtype=torch.long)
        start = 0
        for req_idx, n in enumerate(num_scheduled):
            end = start + n
            if n > 1:
                input_ids[start : end - 1] = scheduled_token_ids[
                    start + 1 : end
                ]
            # Tail rows past the last accepted row are unused (their KV is
            # overwritten next round); keep a deterministic shifted id there
            # like the stock drafter's buffer does.
            input_ids[end - 1] = scheduled_token_ids[end - 1]
            input_ids[int(sample_row_indices[req_idx])] = next_token_ids[
                req_idx
            ]
            start = end
        input_ids = input_ids.to(self.device, non_blocking=True)

        positions = context["positions"]
        hidden_states = context["hidden_states"]
        return input_ids, positions, hidden_states, draft_step_idx

    def _run_edge_cloud_draft_first_segment(
        self, scheduler_output: "SchedulerOutput"
    ) -> IntermediateTensors:
        context = self._get_pending_edge_cloud_draft_context(
            scheduler_output
        )
        input_ids, positions, hidden_states, draft_step_idx = (
            self._prepare_edge_cloud_draft_step_inputs(scheduler_output)
        )
        full_num_tokens = int(
            scheduler_output.total_num_scheduled_tokens
            if draft_step_idx == 0
            else len(scheduler_output.num_scheduled_tokens)
        )
        dsv4_mtp_sp = (
            self._is_deepseek_v4_mtp_edge_cloud_draft()
            and enable_sp()
        )
        if dsv4_mtp_sp:
            # The embedding lookup reduces/scatters full input_ids under SP.
            # Pad that full input first: the reduce-scatter collective itself
            # requires a dim-0 length divisible by TP size.
            input_ids = self._pad_deepseek_v4_mtp_draft_input_ids(
                input_ids,
                full_num_tokens,
            )
            # Segment A must receive positions and target hidden states with
            # the same rank-local token layout. Keep full positions for E,
            # which samples the full cloud reply before re-sharding it.
            position_token_dim = (
                positions.ndim - 1 if self.uses_mrope else 0
            )
            full_positions = context.get("current_draft_full_positions")
            if full_positions is None:
                full_positions = (
                    self._materialize_deepseek_v4_mtp_draft_tensor(
                        positions,
                        full_num_tokens,
                        token_dim=position_token_dim,
                    )
                )
            positions = self._localize_deepseek_v4_mtp_draft_tensor(
                full_positions,
                full_num_tokens,
                token_dim=position_token_dim,
            )
            hidden_states = self._localize_deepseek_v4_mtp_draft_tensor(
                hidden_states,
                full_num_tokens,
            )
            context["current_draft_full_positions"] = full_positions
        num_tokens = positions.shape[-1] if self.uses_mrope else positions.shape[0]
        # Match the main-model SP contract: the forward context describes the
        # padded full embedding input, while num_actual_tokens records the
        # unpadded full sequence.  Using the local SP row count here makes the
        # embedding custom op derive a second pad_size and append it to the
        # already-padded input before reduce-scatter.
        forward_num_tokens = input_ids.shape[0] if dsv4_mtp_sp else num_tokens
        forward_num_actual_tokens = (
            full_num_tokens if dsv4_mtp_sp else num_tokens
        )
        segment = self._edge_cloud_draft_segments["a"]
        # Independently scheduled draft batches do not enter
        # execute_model(), so they do not inherit its forward context. Keep
        # the draft segments eager until the scheduler path can also preserve
        # graph dispatch metadata and stable input buffers across A/C/E.
        with set_ascend_forward_context(
            attn_metadata=None,
            vllm_config=self.vllm_config,
            num_tokens=forward_num_tokens,
            num_actual_tokens=forward_num_actual_tokens,
            batch_descriptor=BatchDescriptor(forward_num_tokens),
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            is_draft_model=True,
        ):
            # The compiled segment is traced exactly once (during warmup,
            # via the proposer) and Dynamo guards are disabled
            # (skip_all_guards_unsafe), so the call signature here must
            # match the warmup trace exactly: warmup passes a real
            # inputs_embeds tensor whenever the drafter supports mm inputs
            # (the traced graph then consumes it instead of running
            # embed_input_ids), and never passes spec_step_idx.  Omitting
            # inputs_embeds feeds None into a tensor placeholder of the
            # cached graph and crashes with "tensor does not have a device".
            inputs_embeds = None
            if getattr(self.drafter, "supports_mm_inputs", False):
                inputs_embeds = self.drafter.model.embed_input_ids(
                    input_ids,
                    multimodal_embeddings=None,
                    is_multimodal=None,
                )
            segment_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "positions": positions,
                "inputs_embeds": inputs_embeds,
                "hidden_states": hidden_states,
            }
            draft_model = self.drafter.model
            if getattr(
                draft_model,
                "edge_cloud_dynamic_step_segments",
                False,
            ):
                segment_kwargs["spec_step_idx"] = draft_step_idx
            output = segment(**segment_kwargs)
        if not isinstance(output, IntermediateTensors):
            raise RuntimeError(
                "Edge-cloud draft first segment returned no intermediates"
            )
        if (
            self.speculative_config is not None
            and self.speculative_config.method == "eagle3"
            and draft_step_idx > 0
        ):
            # Eagle3 carries the previous draft layer's pre-norm residual to
            # the cloud. Its edge segment only embeds the proposed token.
            output["hidden_states"] = hidden_states
        context["current_draft_positions"] = positions
        return output

    def _compute_edge_cloud_draft_token_ids(
        self, hidden_states: torch.Tensor, draft_step_idx: int
    ) -> torch.Tensor:
        assert self.drafter is not None and self.drafter.model is not None
        if self.speculative_config.method == "eagle3":
            if get_ascend_config().enable_reduce_sample:
                return self.drafter.compute_draft_token_ids(hidden_states)
            logits = self.drafter.model.compute_logits(hidden_states)
            assert logits is not None
            if lmhead_tp_enable():
                logits = logits[: hidden_states.shape[0]]
            return logits.argmax(dim=-1)
        mtp_model = self.drafter.model
        if hasattr(mtp_model, "compute_logits"):
            try:
                logits = mtp_model.compute_logits(
                    hidden_states, draft_step_idx
                )
            except TypeError:
                logits = mtp_model.compute_logits(hidden_states)
        else:
            logits = mtp_model.logits_processor(
                mtp_model.lm_head, hidden_states
            )
        if get_ascend_config().enable_reduce_sample:
            if lmhead_tp_enable():
                logits = get_lmhead_tp_group().all_to_all(logits)
            else:
                inner_model = getattr(mtp_model, "model", None)
                logits_processor = getattr(
                    inner_model,
                    "logits_processor",
                    getattr(mtp_model, "logits_processor", None),
                )
                if logits_processor is None:
                    raise RuntimeError(
                        "MTP edge-cloud reduce-sample path has no logits "
                        "processor for TP gathering"
                    )
                logits = logits_processor._gather_logits(logits)
        if lmhead_tp_enable():
            logits = logits[: hidden_states.shape[0]]
        return logits.argmax(dim=-1)

    @staticmethod
    def _validate_edge_cloud_draft_payload_identity(
        scheduler_output: "SchedulerOutput", tensor_dict: dict[str, Any]
    ) -> None:
        expected_task_id = scheduler_output.draft_task_id
        actual_task_id = tensor_dict.get("draft_task_id")
        if expected_task_id is not None and actual_task_id != expected_task_id:
            raise RuntimeError(
                "DRAFT payload task mismatch: "
                f"expected={expected_task_id}, got={actual_task_id}"
            )
        expected_step = int(scheduler_output.draft_step_idx or 0)
        for field in ("draft_step_idx", "spec_step_idx"):
            actual_step = tensor_dict.get(field)
            if torch.is_tensor(actual_step):
                actual_step = int(actual_step.item())
            if actual_step is not None and int(actual_step) != expected_step:
                raise RuntimeError(
                    f"DRAFT payload {field} mismatch: "
                    f"expected={expected_step}, got={actual_step}"
                )
        actual_head_token = tensor_dict.get("head_token")
        if (
            scheduler_output.head_token is not None
            and actual_head_token != scheduler_output.head_token
        ):
            raise RuntimeError(
                "DRAFT payload head_token mismatch: "
                f"expected={scheduler_output.head_token}, "
                f"got={actual_head_token}"
            )

    def _run_edge_cloud_draft_last_segment(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors,
    ) -> ModelRunnerOutput:
        task_id = scheduler_output.draft_task_id
        context = (
            self._pending_edge_cloud_draft_contexts.get(task_id)
            if task_id else None
        )
        if context is None:
            # Drain: the owning request finished/aborted after its
            # DRAFT_FIRST was already dispatched to the cloud.  The cloud
            # does not track request lifecycle, so it still ran the draft
            # middle segment and isend the DRAFT_LAST response; the recv in
            # _execute_model_edge_draft_tail already consumed it to keep the
            # DECODE hidden channel paired.  With no draft context there is
            # no tail-segment compute to run (the result would be discarded
            # anyway), so return a token-less placeholder and let
            # update_from_output skip the gone request.
            logger.info(
                "[PD] drain stale DRAFT_LAST task_id=%s step=%s "
                "(request gone, draft context cleared)",
                task_id,
                scheduler_output.draft_step_idx,
            )
            req_ids = list(scheduler_output.num_scheduled_tokens)
            return ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            )
        draft_step_idx = int(scheduler_output.draft_step_idx or 0)
        positions = context.get("current_draft_full_positions")
        if positions is None:
            positions = context.get("current_draft_positions")
        if positions is None:
            positions = intermediate_tensors.tensors.get("positions")
        if positions is None:
            raise RuntimeError("DRAFT_LAST missing positions")
        num_tokens = int(
            scheduler_output.total_num_scheduled_tokens
            if draft_step_idx == 0
            else len(scheduler_output.num_scheduled_tokens)
        )
        intermediate_tensors = (
            self._sync_edge_cloud_draft_intermediate_tensors(
                num_tokens, intermediate_tensors
            )
        )
        segment = self._edge_cloud_draft_segments["e"]
        # DRAFT_LAST is dispatched independently as well, so it needs the
        # same explicit eager context as DRAFT_FIRST.
        with set_ascend_forward_context(
            attn_metadata=None,
            vllm_config=self.vllm_config,
            num_tokens=num_tokens,
            num_actual_tokens=num_tokens,
            batch_descriptor=BatchDescriptor(num_tokens),
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            is_draft_model=True,
        ):
            # Match the warmup trace signature exactly (guards are skipped,
            # see _run_mtp_edge_first_segment): warmup calls segment "e"
            # with only positions + intermediate_tensors, and the last
            # segment (final norm) does not consume spec_step_idx anyway.
            segment_output = segment(
                positions=positions,
                intermediate_tensors=intermediate_tensors,
            )
        if self.speculative_config.method == "eagle3":
            if not (
                isinstance(segment_output, tuple)
                and len(segment_output) == 2
                and all(torch.is_tensor(item) for item in segment_output)
            ):
                raise RuntimeError(
                    "Eagle3 last segment did not return its hidden-state pair"
                )
            last_hidden_states, hidden_states = segment_output
        else:
            if not torch.is_tensor(segment_output):
                raise RuntimeError(
                    "MTP last segment returned no hidden states"
                )
            last_hidden_states = hidden_states = segment_output
        num_reqs = len(context["req_ids"])
        if draft_step_idx == 0 and last_hidden_states.shape[0] != num_reqs:
            # The first draft pass ran over all scheduled tokens; only the
            # last row of each request produces the proposed draft token and
            # feeds the next speculative step.
            sample_rows = context["sample_row_indices"].to(
                hidden_states.device
            )
            logits_hidden_states = last_hidden_states[sample_rows]
            next_hidden_states = hidden_states[sample_rows]
            step_positions = (
                positions[:, sample_rows]
                if self.uses_mrope
                else positions[sample_rows]
            )
        else:
            logits_hidden_states = last_hidden_states
            next_hidden_states = hidden_states
            step_positions = positions
        if (
            self._is_deepseek_v4_mtp_edge_cloud_draft()
            and enable_sp()
        ):
            # Segment E validates the full cloud reply. Keep the sampled
            # logits input replicated across TP ranks: the vocab-parallel LM
            # head computes a different vocab shard on each rank and therefore
            # requires every rank to process the same request rows before its
            # logits all-gather. Only the recurrent state and positions for the
            # next DRAFT_FIRST are sequence-parallel shards.
            position_token_dim = (
                step_positions.ndim - 1 if self.uses_mrope else 0
            )
            full_step_positions = step_positions
            if logits_hidden_states.shape[0] != num_reqs:
                raise RuntimeError(
                    "DeepSeek-V4 MTP SP logits require full request rows: "
                    f"rows={logits_hidden_states.shape[0]}, reqs={num_reqs}"
                )
            next_hidden_states = (
                self._localize_deepseek_v4_mtp_draft_tensor(
                    next_hidden_states,
                    num_reqs,
                )
            )
            step_positions = self._localize_deepseek_v4_mtp_draft_tensor(
                step_positions,
                num_reqs,
                token_dim=position_token_dim,
            )
        else:
            full_step_positions = step_positions
        draft_token_ids = self._compute_edge_cloud_draft_token_ids(
            logits_hidden_states, draft_step_idx
        )
        draft_token_ids = self._gather_deepseek_v4_mtp_draft_tensors(
            {"draft_token_ids": draft_token_ids}, num_reqs
        )["draft_token_ids"]
        context["last_draft_hidden_states"] = next_hidden_states.clone()
        context["last_draft_positions"] = step_positions.clone()
        context["last_draft_full_positions"] = (
            full_step_positions.clone()
        )
        context["last_draft_token_ids"] = draft_token_ids.clone()
        draft_steps = context.setdefault("draft_token_id_steps", [])
        if len(draft_steps) != draft_step_idx:
            raise RuntimeError(
                "DRAFT step order mismatch: "
                f"expected={len(draft_steps)}, got={draft_step_idx}"
            )
        draft_steps.append(draft_token_ids.clone())
        next_step_idx = draft_step_idx + 1
        completed_draft_token_ids = None
        if next_step_idx < self.num_spec_tokens:
            context["draft_step_idx"] = next_step_idx
            # DRAFT_LAST completion is the readiness signal for the next
            # step. PDSeparatedScheduler derives the next DRAFT_FIRST locally
            # from this completed SchedulerOutput, so no worker-side pending
            # task or follow-up control RPC is needed.
        elif context.get("draft_output_req_ids"):
            self._draft_token_ids = torch.stack(draft_steps, dim=1)
            # Rows of _draft_token_ids follow this context's request order;
            # remember it so the deferred-draft context stash can map rows
            # back to requests even if the input batch was re-indexed since.
            self._draft_token_ids_req_ids = list(context["req_ids"])
            # Also record the rows per request: this is the authoritative
            # source for the verify-time spec scatter in _prepare_inputs,
            # which must survive both input-batch re-indexing and later
            # chains overwriting the global above.  Requests that finished
            # while the chain was in flight are skipped (non-edge-cloud
            # semantics: dead rows still go through the drafter, their
            # outputs are discarded here).
            finished_req_ids = context.get("finished_req_ids") or set()
            draft_output_req_ids = set(context["draft_output_req_ids"])
            draft_rows = self._draft_token_ids.tolist()
            vocab_size = self.model_config.get_vocab_size()
            for row, req_id in enumerate(self._draft_token_ids_req_ids):
                if (
                    req_id in finished_req_ids
                    or req_id not in draft_output_req_ids
                ):
                    continue
                bad_ids = [
                    token_id
                    for token_id in draft_rows[row]
                    if not 0 <= token_id < vocab_size
                ]
                if bad_ids:
                    # An out-of-range draft id means this row was computed
                    # from another request's hidden states/KV (a row
                    # shift somewhere in the chain).  Feeding it to the
                    # verify would put an illegal id into input_ids and
                    # into the rejection sampler, which indexes target
                    # probs by draft id -> device-side MTE fault (ACL
                    # 507035).  Drop the row so the verify falls back to
                    # the native scatter for this request, and report it
                    # loudly: this signals a chain/batch misalignment
                    # that must be fixed, not hidden.
                    logger.error(
                        "[DRAFT-OUT] task=%s req=%s produced out-of-vocab "
                        "draft ids %s (row=%s, vocab_size=%d); dropping "
                        "the row",
                        scheduler_output.draft_task_id,
                        req_id,
                        bad_ids,
                        draft_rows[row],
                        vocab_size,
                    )
                    continue
                self._worker_draft_token_ids_by_req[req_id] = (
                    self._draft_token_ids[row]
                )
            logger.info(
                "[DRAFT-OUT] task=%s drafts=%s",
                scheduler_output.draft_task_id,
                draft_rows,
            )
        else:
            logger.info(
                "[DRAFT-PREFILL] task=%s completed for mid-prefill chunk; "
                "draft KV populated and proposals discarded",
                scheduler_output.draft_task_id,
            )

        req_ids = list(context["req_ids"])
        if next_step_idx >= self.num_spec_tokens:
            # Native async spec-decode semantics: keep the real draft token
            # IDs in the worker.  The already queued next target batch carries
            # fixed-length placeholders and _prepare_input_ids scatters this
            # tensor into the actual verify inputs after this DRL executes.
            # Avoid the device->CPU->EngineCore->scheduler round trip entirely.
            if (
                context.get("draft_output_req_ids")
                and not self.use_async_scheduling
            ):
                output_rows = [
                    row
                    for row, req_id in enumerate(req_ids)
                    if req_id in context["draft_output_req_ids"]
                ]
                completed_draft_token_ids = DraftTokenIds(
                    [req_ids[row] for row in output_rows],
                    self._draft_token_ids[output_rows]
                    .detach()
                    .cpu()
                    .tolist(),
                )
            task_id = scheduler_output.draft_task_id
            assert task_id is not None
            self._pending_edge_cloud_draft_contexts.pop(task_id, None)

        output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        )
        if completed_draft_token_ids is not None:
            output.edge_cloud_draft_token_ids = completed_draft_token_ids
        return output

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
        if self.vllm_config.model_config.enable_return_routed_experts:
            if self.routed_experts_initialized:
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

        # [EDGE-SEGMENT-E LEAK FIX] Always consume this forward's segment_a
        # cache entry on segment_e entry, regardless of which path segment_e
        # takes below (fast-path, normal-path, or stale-tail discard).
        #
        # Before this, the entry was only popped on the fast path. When high
        # concurrency interleaving (2P1D) made the fast-path req_ids check
        # fail, segment_e fell through to the normal path WITHOUT popping ->
        # the entry orphaned. Orphans steadily filled the bounded cache,
        # forced eviction of LIVE entries, and their later segment_e
        # cache-missed onto the (incorrect-for-tail) normal path ->
        # _prepare_inputs recomputed against a modified input_batch ->
        # wrong draft token ids -> acceptance cliff. Aborted/finished reqs
        # (popped from self.requests during the head->tail window) hit the
        # stale-tail early-return below, which also bypassed the pop.
        #
        # Popping once here covers every segment_e path. It is always safe:
        # a head_token's segment_e runs at most once, and segment_a caches
        # under the same head_token it will later pop. If the fast path
        # cannot reuse the entry (req_ids mismatch / empty batch), the entry
        # is discarded rather than leaked -- the normal path recomputes, as
        # it did before, but the cache no longer fills.
        _edge_cache_entry: dict[str, Any] | None = None
        if (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "edge"
            and intermediate_tensors is not None
            and scheduler_output.head_token is not None
        ):
            _edge_cache_entry = self._edge_prepare_cache_by_token.pop(
                scheduler_output.head_token, None
            )

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
                # A req is stale for this tail batch if it was already
                # popped from self.requests (finished during the head->tail
                # window) OR if THIS batch's finished_req_ids will pop it in
                # _update_states before the cached-req loop reads
                # self.requests[req_id] (KeyError). The latter happens when
                # the scheduler finishes a request (max_tokens/stop/abort)
                # while another in-flight tail for it is still queued —
                # possible with batch_queue depth 4 and many concurrent reqs.
                finished = scheduler_output.finished_req_ids or set()
                stale = [
                    r
                    for r in tail_req_ids
                    if r not in self.requests or r in finished
                ]
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
                    if scheduler_output.head_token is not None:
                        self._verified_draft_token_ids_by_head.pop(
                            scheduler_output.head_token, None
                        )
                    self._tail_segment_discarded = True
                    return EMPTY_MODEL_RUNNER_OUTPUT
                if stale:
                    # Partial-stale: some reqs in this tail batch already
                    # finished on the edge during the head->tail window, but
                    # alive reqs still need this step. Filter the stale reqs
                    # out of scheduler_output and slice the received hidden
                    # tensors (per-req token ranges), then run segment_e for
                    # the alive reqs only.
                    # logger.warning(
                    #     "[EDGE-TAIL-PARTIAL-STALE-FILTER] batch_type=%s "
                    #     "head_token=%s stale=%s alive=%s; filtering stale "
                    #     "reqs from tail segment.",
                    #     scheduler_output.batch_type,
                    #     scheduler_output.head_token,
                    #     stale,
                    #     [r for r in tail_req_ids if r in self.requests],
                    # )
                    scheduler_output, intermediate_tensors = (
                        self._filter_stale_tail_batch(
                            scheduler_output, intermediate_tensors, stale
                        )
                    )
                    # The segment_a prepare cache was built for the FULL
                    # batch (attn_metadata / logits_indices / num_tokens all
                    # include the stale reqs' tokens). The segment_e fast
                    # path must not reuse it; force the normal prepare path.
                    # NOTE: this used to be `self._edge_prepare_cache = None`
                    # but the cache was later keyed by head_token
                    # (_edge_prepare_cache_by_token), which silently turned
                    # that assignment into a no-op and let the fast path
                    # reuse FULL-batch metadata for the filtered batch
                    # (garbage logits for the alive reqs).  Pop the entry
                    # for THIS head_token instead.
                    self._edge_prepare_cache_by_token.pop(
                        scheduler_output.head_token, None
                    )

        # Save scheduler_output for legacy synchronous edge-cloud sampling and
        # auxiliary-hidden-state paths.
        self._last_scheduler_output = scheduler_output

        # In edge-cloud mode, execute_model is called twice for the
        # same scheduler_output: head segment (intermediate_tensors is None) and
        # tail segment (intermediate_tensors is not None). The tail segment should
        # reuse the num_computed_tokens corrected by the head segment, instead of
        # re-running update_num_computed_tokens_for_batch_change or copying the
        # potentially optimistic CPU value back to GPU.
        self._is_edge_cloud_tail_segment = (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "edge"
            and intermediate_tensors is not None
        )
        # Whether the resident batch still matches this SchedulerOutput
        # BEFORE _update_states runs.  Only meaningful for edge tail
        # segments: if a PD-interleaved batch (e.g. another request's
        # PREFILL_FIRST) evicted the tail's requests between the head and
        # tail segments, the positional GPU num_computed buffer keeps the
        # evictor's values and the tail must NOT skip the resync
        # (_skip_tail_sync); it then repairs the buffer from the exact
        # SO-derived CPU state via the plain-copy branch.
        self._edge_tail_membership_intact = (
            not self._is_edge_cloud_tail_segment
            or (
                self.input_batch.num_reqs > 0
                and tuple(self.input_batch.req_ids)
                == tuple(scheduler_output.num_scheduled_tokens)
            )
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
            self.use_async_scheduling
            and self.num_spec_tokens
            and self._draft_token_ids is None  # type: ignore[has-type]
        ) or (
            # NOTE: This branch specifically triggers a deepcopy during the prefill phase 
            # only for PCP (Parallel Context Processing) + Multi-Modal (MM) scenarios. 
            # It does not affect other use cases. This is a temporary workaround and 
            # will be removed once upstream vLLM provides native support for PCP + MM.
            self.pcp_size > 1 and self.supports_mm_inputs and get_pp_group().is_first_rank
            and not self.model_config.is_encoder_decoder
        )):
            scheduler_output = deepcopy(scheduler_output)
        pp_group = get_pp_group()
        if pp_group.world_size > 1 and not pp_group.is_last_rank:
            new_token_ids = scheduler_output.scheduled_cached_reqs.new_token_ids
            if new_token_ids and all(not token_ids for token_ids in new_token_ids):
                scheduler_output = deepcopy(scheduler_output)
                scheduler_output.scheduled_cached_reqs.new_token_ids = []

        if has_kv_transfer_group():
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            assert kv_connector_metadata is not None
            # Preemption stores must run before _update_states() zeroes newly
            # allocated blocks that may reuse the same physical KV cache IDs.
            get_kv_transfer_group().handle_preemptions(kv_connector_metadata)

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        # Mamba align preprocessing only collects copy metadata on Ascend; the
        # actual state copy is deferred until the KV connector has loaded its
        # blocks below. The normal path prepares the metadata in this call,
        # while the cloud fast path restores metadata prepared by
        # cloud_prepare_early. An edge tail fast path must leave this as None:
        # segment_a already consumed its metadata, and the shared buffers may
        # have been reused by an interleaved batch before segment_e runs.
        mamba_preprocess_bufs = None

        # ---- segment_e fast path: reuse segment_a's cached prepare results ----
        # NOTE: if an intervening EMPTY batch cleared input_batch, we must
        # fall through to the normal path so _update_states can re-add the
        # requests before sampling.
        _fast_path = (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "edge"
            and intermediate_tensors is not None
            and _edge_cache_entry is not None
            and self.input_batch.num_reqs > 0
            and tuple(self.input_batch.req_ids)
            == tuple(scheduler_output.num_scheduled_tokens)
        )
        # ---- cloud fast path: reuse pre-computed prepare results ----
        _cloud_fast_path = (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "cloud"
            and intermediate_tensors is not None
            and self._cloud_prepare_cache is not None
            and self._cloud_prepare_cache.get("req_ids_key")
            == tuple(scheduler_output.num_scheduled_tokens)
        )
        if _fast_path:
            # [ascend fix] The edge tail fast path runs OUTSIDE the
            # synchronize_input_prep/prepare_inputs_event protection that
            # the slow path gets.  Its conditional _update_states writes
            # pinned CPU buffers (num_computed_tokens_cpu, block_table.np)
            # and the num_computed re-sync below does a non-blocking H2D
            # READ of a pinned buffer.  Under PD interleaving the GPU queue
            # backs up behind a slow prefill forward, so a previous batch's
            # pending H2D reads race with this batch's pinned writes, and
            # this batch's H2D read races with the next batch's pinned
            # writes -> corrupted num_computed/block_table -> wrong
            # positions/slot_mapping -> KV written to wrong slots -> whole
            # decode batch diverges.  Chain the fast path into the same
            # event protocol: wait for the previous prep's H2D before
            # touching pinned state, record after our H2D so the next
            # batch's prep waits for it.
            if self.prepare_inputs_event is not None:
                self.prepare_inputs_event.synchronize()

            # Pop this head_token's cache so a later segment_a (different
            # head_token) does not hand the wrong attn_metadata to this PL.
            cache = _edge_cache_entry

            # If intervening decode batches disrupted input_batch (req_ids no
            # longer match this tail's scheduled reqs), re-add the prefill req
            # via _update_states so the tail can sample and record the first
            # token. The cached layout (keyed by head_token) is still correct
            # and is reused below -- we do NOT recompute it via _prepareInputs.
            # When req_ids already match, the req is in input_batch, so skip
            # _update_states to avoid double-counting. deferred_state_corrections_fn
            # (None when MTP/spec-decode is off) is applied at the end of
            # execute_model.
            if (tuple(self.input_batch.req_ids)
                    != tuple(scheduler_output.num_scheduled_tokens)):
                # The head segment already applied this SO's
                # new_block_ids; strip them so the re-add below cannot
                # double-extend block_ids (see _strip_tail_new_block_ids).
                deferred_state_corrections_fn = self._update_states(
                    self._strip_tail_new_block_ids(scheduler_output)
                )
            else:
                # The fast path skips _update_states, but the cleanup riding
                # this tail SO must still be applied: finished_req_ids (and
                # free_encoder_mm_hashes) can arrive on a tail when their
                # EMPTY-batch delivery was deferred behind in-flight work.
                # Without this, finished requests linger in self.requests
                # (e.g. a drained mm request), corrupting registry scans such
                # as step_has_multimodal_req for later text-only batches.
                # Mirrors the removal in the base _update_states; the tail's
                # own reqs are guaranteed non-finished here (stale tails are
                # discarded earlier by the EDGE-TAIL-STALE-DISCARD guard).
                for req_id in scheduler_output.finished_req_ids or ():
                    self.requests.pop(req_id, None)
                    self.num_prompt_logprobs.pop(req_id, None)
                    self.input_batch.remove_request(req_id)
                for mm_hash in (scheduler_output.free_encoder_mm_hashes
                                or ()):
                    self.encoder_cache.pop(mm_hash, None)
                deferred_state_corrections_fn = None

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
            # Restore this batch's discard state as well.  The fast path
            # skips _prepare_inputs, and the shared discard buffers may have
            # been overwritten by an interleaved head segment executing
            # between this batch's head and tail (e.g. the last chunk's
            # PREFILL_FIRST, which discards nothing, running before a
            # mid-chunk PREFILL_LAST).  Without the restore, a mid-chunk PL
            # returns its (prompt-predicting) sampled token to the
            # scheduler, double-decrementing num_output_placeholders and
            # tripping the assert in AsyncScheduler._update_request_with_output.
            self.num_discarded_requests = cache["num_discarded_requests"]
            self.discard_request_indices.np[: self.num_discarded_requests] = (
                cache["discard_request_indices"]
            )
            self.discard_request_indices.copy_to_gpu(
                self.num_discarded_requests
            )
            # [ascend fix] Refresh the persistent buffers the ACL graph
            # captured and switch to the original view-based objects.
            #
            # Gate rationale (see _fast_path_view_restore_required):
            # * capability, not model family: only backends WITHOUT an
            #   explicit graph-params update channel (DSA/SFA, whose
            #   update_graph_params is a no-op) rely on metadata address
            #   identity and therefore need this restore.  Backends with
            #   a complete update channel (MLA/FIA, e.g. Qwen) read the
            #   frozen clones' content through their update flow and need
            #   nothing -- keeping their fast path free of restore cost.
            #   GDN models are excluded categorically (pool-slot views
            #   are unsafe to write back).
            # * cudagraph_mode == FULL: only FULL-graph replay hard-codes
            #   metadata addresses. PIECEWISE/NONE (e.g. prefill PL
            #   segments) consume the metadata objects eagerly, where the
            #   frozen clones are already correct; restoring there would
            #   waste large D2D copies (attn_mask/slot_mapping/positions).
            #
            # Cross-stream ordering: restore ops enqueue on the main
            # stream after the previous batch's replay, which is itself
            # event-ordered after its update_stream work, so ordering is
            # transitive. Moreover, under this gate the restore only runs
            # for backends without an update channel -- there is no
            # update_stream consumer of these buffers at all.
            use_views = (
                self._fast_path_view_restore_required()
                and cudagraph_mode == CUDAGraphMode.FULL
            )
            if use_views:
                restore_memo: set[tuple[int, tuple]] = set()
                attn_metadata_views = cache.get("attn_metadata_views")
                if attn_metadata_views is not None:
                    _restore_frozen_into_views(
                        attn_metadata_views, attn_metadata,
                        memo=restore_memo)
                    attn_metadata = attn_metadata_views
                spec_decode_metadata_views = cache.get(
                    "spec_decode_metadata_views"
                )
                if spec_decode_metadata_views is not None:
                    _restore_frozen_into_views(
                        spec_decode_metadata_views, spec_decode_metadata,
                        memo=restore_memo)
                    spec_decode_metadata = spec_decode_metadata_views
                spec_common_views = cache.get(
                    "spec_decode_common_attn_metadata_views"
                )
                if spec_common_views is not None:
                    _restore_frozen_into_views(
                        spec_common_views, spec_decode_common_attn_metadata,
                        memo=restore_memo)
                    spec_decode_common_attn_metadata = spec_common_views
                # positions buffer content is refreshed here as well (the
                # model reads the runner's reusable positions buffer); the
                # local `positions` variable is switched to the view where
                # the fast path consumes it below.
                positions_views = cache.get("positions_views")
                if positions_views is not None:
                    _restore_frozen_into_views(
                        positions_views, cache["positions"],
                        memo=restore_memo)
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
            # [ascend fix] The fast path skips _prepare_inputs, which also
            # recomputes the discard indices (requests that must not be
            # sampled).  Those indices are ROW numbers, and interleaved
            # batches between segment_a and segment_e churn input_batch
            # rows -- stale indices from the segment_a prepare mark the
            # WRONG request invalid: its sampled token is then dropped
            # from prev_sampled (next DF finds no entry -> [EC-INV-C1],
            # stale decode input) or cleared from the returned output.
            # Refresh them for the current layout before sampling.
            self._update_discard_request_indices(scheduler_output)
            # [ascend fix] see above: record so the next batch's
            # synchronize_input_prep waits for this H2D read of the pinned
            # buffer before overwriting it.
            if self.prepare_inputs_event is not None:
                self.prepare_inputs_event.record()
        elif _cloud_fast_path:
            cache = self._cloud_prepare_cache
            self._cloud_prepare_cache = None  # consumed, clear for next iteration
            mamba_preprocess_bufs = cache["mamba_preprocess_bufs"]
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
            # Apply leftover deferred corrections at the same post-launch
            # point as the slow path (see the end of execute_model).
            deferred_state_corrections_fn = cache.get(
                "deferred_state_corrections_fn"
            )
            # Mirror the slow path: keep the cloud-side spec decode common
            # attention metadata for MTP draft proposal.
            num_reqs = self.input_batch.num_reqs
            if spec_decode_common_attn_metadata is not None:
                self._cloud_spec_decode_common_attn_metadata = (
                    spec_decode_common_attn_metadata
                )
                self._cloud_spec_decode_num_reqs = num_reqs
        _pd_timing = (
            self._edge_cloud_enabled
            and scheduler_output.batch_type in (
                BatchType.DECODE_FIRST, BatchType.DECODE_LAST
            )
        )
        if _pd_timing:
            logger.info(
                "[PD-TIMING] runner prep enter batch_type=%s ts=%.4f",
                scheduler_output.batch_type,
                time.monotonic(),
            )
        with record_function_or_nullcontext("prepare input"):
            with self.synchronize_input_prep():
                if _pd_timing:
                    logger.info(
                        "[PD-TIMING] runner input-prep sync acquired "
                        "batch_type=%s ts=%.4f",
                        scheduler_output.batch_type,
                        time.monotonic(),
                    )
                if not _fast_path and not _cloud_fast_path:
                    # [EDGE-SEGMENT-E LEAK FIX] If we popped a segment_a entry
                    # but cannot take the fast path (req_ids mismatch from
                    # interleaving, or empty batch), the entry is discarded
                    # here instead of orphaning. Log so the residual leak
                    # rate can be confirmed on NPU (should be ~0 accumulation;
                    # non-zero count = the normal-path corruption cases that
                    # fix C would address).
                    if _edge_cache_entry is not None:
                        logger.debug(
                            "[EDGE-SEGMENT-E] head_token=%s segment_a cache "
                            "popped but fast-path skipped; entry discarded "
                            "(previously this orphaned and leaked).",
                            scheduler_output.head_token,
                        )
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
                    elif is_edge_tail_segment:
                        # Tail re-running _update_states after interleaving
                        # evicted a request: the head already applied this
                        # SO's new_block_ids; strip them so block_ids is
                        # not extended twice (chunked-prefill KV garbage).
                        deferred_state_corrections_fn = self._update_states(
                            self._strip_tail_new_block_ids(scheduler_output)
                        )
                    else:
                        deferred_state_corrections_fn = self._update_states(
                            scheduler_output
                        )
                    if _pd_timing:
                        logger.info(
                            "[PD-TIMING] runner update_states done "
                            "batch_type=%s ts=%.4f",
                            scheduler_output.batch_type,
                            time.monotonic(),
                        )

                    if has_ec_transfer() and get_ec_transfer().is_producer:
                        with self.maybe_get_ec_connector_output(
                            scheduler_output,
                            encoder_cache=self.encoder_cache,
                        ) as ec_connector_output:
                            self._execute_mm_encoder(scheduler_output)
                            self._finalize_dump_data()
                            return make_empty_encoder_model_runner_output(scheduler_output)

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
                    ) = self._prepare_inputs(
                        scheduler_output,
                        num_scheduled_tokens_np,
                    )
                    if _pd_timing:
                        logger.info(
                            "[PD-TIMING] runner update_states+prepare_inputs "
                            "done batch_type=%s ts=%.4f",
                            scheduler_output.batch_type,
                            time.monotonic(),
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

                    # NOTE(Angazenn): According to https://github.com/vllm-project/vllm/pull/30877,
                    # there should be a corresponding 'postprocess_mamba'. However, it is called inside
                    # '_update_states_after_model_execute', which is not overridden in vLLM-Ascend.
                    # We simply utilize the implementation in vLLM.
                    # need_accepted_tokens is false when this worker owns no
                    # local Mamba cache, notably on an embedding-only edge.
                    if (
                        self.cache_config.mamba_cache_mode == "align"
                        and self.need_accepted_tokens
                    ):
                        # preprocess_mamba reads req_state.num_computed_tokens (CPU)
                        # to decide copy operations, so we must apply deferred
                        # corrections before it runs.
                        if deferred_state_corrections_fn:
                            deferred_state_corrections_fn()
                            deferred_state_corrections_fn = None
                        mamba_bufs = self._get_mamba_bufs()
                        mamba_preprocess_bufs = mamba_bufs.preprocess
                        mamba_utils.preprocess_mamba(
                            scheduler_output,
                            self.kv_cache_config,
                            self.cache_config,
                            self.mamba_state_idx,
                            self.input_batch,
                            self.requests,
                            self.compilation_config.static_forward_context,
                            self.model.get_mamba_state_copy_func(),
                            mamba_preprocess_bufs,
                        )
                        # preprocess_mamba resets num_accepted_tokens_cpu to 1
                        # for requests whose state was copied to a new block.
                        # Re-sync to GPU so the mamba kernel reads from the
                        # correct initial state slot (init_token_idx = 0).
                        self.num_accepted_tokens.np[:num_reqs] = (
                            self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                        )
                        self.num_accepted_tokens.copy_to_gpu(num_reqs)

                        if mamba_bufs.postprocess_align is not None:
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
                    # skip_dsa_fill: already filled above between first
                    # _prepare_inputs and here, using the first call's values.
                    cache = self._run_input_preparation(
                        scheduler_output,
                        precomputed=(
                            logits_indices,
                            spec_decode_metadata,
                            total_num_scheduled_tokens,
                        ),
                        skip_dsa_fill=True
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

            # Edge-cloud cloud side: reuse the M-RoPE positions edge computed
            # and pushed via intermediate_tensors (cloud skipped
            # _init/_calc_mrope_positions). The wire tensor is [N, 3]
            # (dim-0 = sequence). Materialize it into
            # self.mrope_positions.gpu[:, :num_tokens_padded] ([3, N]) BEFORE
            # _preprocess, which reads `positions` as a view over this buffer
            # and runs update_cos_sin on it. Capture the received reference now:
            # _preprocess -> sync_and_slice_intermediate_tensors reassigns
            # `intermediate_tensors` to a local-buffer copy that omits
            # mrope_positions (the sync loop skips it). Gate on the tensor's
            # presence so this never KeyErrors regardless of whether the
            # edge/cloud include_mrope decision (early-recv hint vs sync)
            # agreed.
            recv_intermediate_tensors = intermediate_tensors
            if (self._edge_cloud_enabled
                    and self.edge_cloud_cfg.role == "cloud"
                    and self.uses_mrope
                    and recv_intermediate_tensors is not None
                    and "mrope_positions" in recv_intermediate_tensors.tensors):
                recv_intermediate_tensors.wait_for_comm()
                recv_mrope = recv_intermediate_tensors.tensors["mrope_positions"]
                # The edge strips cudagraph/SP padding before transmission,
                # so the wire tensor may be shorter than num_tokens_padded
                # (e.g. an MTP uniform-decode batch padded up to a FULL
                # cudagraph capture size). Copy only the rows actually
                # received -- mirrors the min() guard for hidden_states in
                # sync_and_slice_intermediate_tensors. The padding tail's
                # positions are never consumed (padding tokens are outside
                # the attention region and logits_indices).
                recv_len = min(recv_mrope.shape[0], num_tokens_padded)
                self.mrope_positions.gpu[:, :recv_len].copy_(
                    recv_mrope[:recv_len].t().contiguous()
                )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output,
                num_tokens_padded,
                intermediate_tensors,
            )
            if _fast_path:
                # _preprocess reads the runner's reusable positions buffer,
                # which may have been rewritten by an interleaved batch even
                # though the metadata cache itself is keyed by head_token.
                # When (and only when) the view restore above ran, its
                # contents were refreshed from the frozen snapshot; use the
                # persistent-buffer view so a FULL graph replay reads the
                # captured address holding this step's positions. Otherwise
                # the frozen clone is already the correct source.
                positions_views = cache.get("positions_views")
                if (
                    positions_views is not None
                    and self._fast_path_view_restore_required()
                    and cudagraph_mode == CUDAGraphMode.FULL
                ):
                    positions = positions_views
                else:
                    positions = cache["positions"]

            # Save the cloud target metadata and the exact positions passed
            # to the model. The scheduled draft can then reconstruct its
            # positions locally instead of receiving them from the edge.
            if (
                self._edge_cloud_enabled
                and self.edge_cloud_cfg.role == "cloud"
                and spec_decode_common_attn_metadata is not None
            ):
                self._cache_cloud_spec_decode_metadata(
                    scheduler_output,
                    spec_decode_common_attn_metadata,
                    self.input_batch.num_reqs,
                    positions,
                )

            if not self.edge_cloud_cfg.role == "edge":
                # update global cos, sin
                update_cos_sin(positions)

        if self.dynamic_eplb:
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
        if (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "edge"
            and intermediate_tensors is None
            and scheduler_output.head_token
        ):
            # Evict the oldest entry if the cache has grown beyond the bound.
            # Defensive cleanup: a segment_e that never arrives (e.g. request
            # aborted mid-prefill) would otherwise leak its entry forever.
            # Normal chunk_prior operation keeps at most prefill_inflight_limit
            # entries, so this branch only triggers on abnormal paths.
            if (
                len(self._edge_prepare_cache_by_token)
                >= self._edge_prepare_cache_max
            ):
                stale_token = next(iter(self._edge_prepare_cache_by_token))
                # logger.warning(
                #     "Edge segment_a cache exceeded bound (%d); evicting oldest "
                #     "head_token=%s (its segment_e likely never arrived, e.g. "
                #     "request abort).",
                #     self._edge_prepare_cache_max,
                #     stale_token,
                # )
                self._edge_prepare_cache_by_token.pop(stale_token, None)

            cache_entry = {
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
                "positions": positions,
                # Discard state is part of the prepare results too: the
                # segment_e fast path skips _prepare_inputs and must restore
                # this batch's own values instead of inheriting whatever an
                # interleaved head segment left in the shared buffers.
                "num_discarded_requests": self.num_discarded_requests,
                "discard_request_indices": self.discard_request_indices.np[
                    : self.num_discarded_requests
                ],
            }
            frozen_entry = _freeze_scheduled_state(cache_entry)
            # Keep the original (view-based) objects alive next to the
            # frozen snapshot. Their tensors reference the persistent
            # buffers the ACL graph captured (block_table / seq_lens /
            # query_start_loc / positions / ...); interleaved batches may
            # overwrite those buffers' contents before segment_e runs.
            # At segment_e the frozen contents are copied back into these
            # views and the views are handed to the forward pass, so a
            # graph replay reads the captured addresses holding this
            # step's data instead of freed clone memory (which previously
            # fed the tail Compressor garbage block tables -> AICORE
            # fault).
            frozen_entry["attn_metadata_views"] = attn_metadata
            frozen_entry["spec_decode_metadata_views"] = spec_decode_metadata
            frozen_entry["spec_decode_common_attn_metadata_views"] = (
                spec_decode_common_attn_metadata
            )
            frozen_entry["positions_views"] = positions
            self._edge_prepare_cache_by_token[scheduler_output.head_token] = (
                frozen_entry
            )

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = self.model_config.is_encoder_decoder and num_encoder_reqs > 0

        # Run forward pass
        clear_kv_metadata = self.speculative_config is None

        # Save per-step state for layer slice continuation.
        if layer_slice_info is not None and not layer_slice_info.is_last_slice:
            # All of these values can contain views into the reusable input
            # preparation buffers.  Freeze them as one object graph so shared
            # per-layer aliases remain shared but an interleaved batch cannot
            # mutate the continuation state.
            frozen_layerwise_state = _freeze_scheduled_state(
                {
                    "positions": positions,
                    "attn_metadata": attn_metadata,
                    "logits_indices": logits_indices,
                    "spec_decode_metadata": spec_decode_metadata,
                    "spec_decode_common_attn_metadata": (
                        spec_decode_common_attn_metadata
                    ),
                }
            )
            self._layerwise_positions = frozen_layerwise_state["positions"]
            self._layerwise_attn_metadata = frozen_layerwise_state[
                "attn_metadata"
            ]
            self._layerwise_num_tokens_padded = num_tokens_padded
            self._layerwise_num_tokens_across_dp = num_tokens_across_dp
            self._layerwise_batch_desc = batch_desc
            self._layerwise_scheduler_output = scheduler_output
            self._layerwise_logits_indices = frozen_layerwise_state[
                "logits_indices"
            ]
            self._layerwise_spec_decode_metadata = frozen_layerwise_state[
                "spec_decode_metadata"
            ]
            self._layerwise_spec_decode_common_attn_metadata = (
                frozen_layerwise_state[
                    "spec_decode_common_attn_metadata"
                ]
            )
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
                eplb_heat_collection_status=self.eplb_heat_collection_status if self.dynamic_eplb else False,
            ),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                **(
                    {"defer_finalize": not clear_kv_metadata}
                ),
            ) as kv_connector_output,
        ):
            if mamba_preprocess_bufs is not None:
                mamba_utils.do_mamba_copy_block(mamba_preprocess_bufs)
            if _pd_timing:
                logger.info(
                    "[PD-TIMING] runner prep done batch_type=%s ts=%.4f",
                    scheduler_output.batch_type,
                    time.monotonic(),
                )
            hidden_states = self._model_forward(
                num_tokens_padded, input_ids, positions, intermediate_tensors,
                inputs_embeds, layer_slice_info=layer_slice_info,
                **model_kwargs
            )
            if _pd_timing:
                logger.info(
                    "[PD-TIMING] runner forward done batch_type=%s ts=%.4f",
                    scheduler_output.batch_type,
                    time.monotonic(),
                )
        with record_function_or_nullcontext("post process"):
            aux_hidden_states = None
            if self.use_aux_hidden_state_outputs:
                if isinstance(hidden_states, IntermediateTensors):
                    # Edge-cloud non-last segments return IntermediateTensors
                    # instead of the (hidden_states, aux_hidden_states) tuple.
                    # The auxiliary hidden states for EAGLE3 are cached on the
                    # cloud side in _eagle3_cloud_aux_hidden_states; leave
                    # hidden_states as IntermediateTensors so the edge-cloud
                    # early-return paths below can handle it.
                    aux_hidden_states = None
                elif isinstance(hidden_states, (tuple, list)):
                    hidden_states, aux_hidden_states = hidden_states
                else:
                    # Edge-cloud last segment returns a plain hidden-states
                    # tensor (no auxiliary outputs). The EAGLE3 aux states are
                    # kept on the cloud side, so there is nothing to unpack.
                    aux_hidden_states = None
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
                # ranks. Snapshot hidden/residual because another scheduled
                # batch may reuse the graph output buffers before next slice.
                assert isinstance(hidden_states, IntermediateTensors)
                self._layerwise_intermediate = _freeze_intermediate_tensors(
                    hidden_states
                )
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
                    if self.dynamic_eplb:
                        self.eplb_updator.forward_end(self.eplb_heat_collection_status)
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
                self._snapshot_mtp_target_hidden_states(scheduler_output),
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
            # Sampling only runs on the edge, but the cloud must retain the
            # rejection-corrected state for its next target/draft forward.
            # Independently scheduled drafts are excluded: their num_accepted
            # payload rides the draft-step-0 execute_model payload and is
            # applied in _run_edge_cloud_draft_middle_segment. Receiving it
            # here as well would emit
            # a TP broadcast from the sample_tokens RPC stream, which races
            # draft broadcasts (execute_model stream) on the shared
            # mq_broadcaster and swaps payloads across ranks.
            if (
                self.speculative_config
                and not self._uses_scheduled_edge_cloud_draft()
                and (
                    self.model_config.is_hybrid
                    or self.speculative_config.method in ("mtp", "eagle3")
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
                num_accepted = tensor_dict["num_accepted_tokens"].to(
                    self.device
                )
                num_reqs = num_accepted.size(0)
                self.num_accepted_tokens.gpu[:num_reqs] = num_accepted

                if (
                    self.speculative_config.method in ("mtp", "eagle3")
                    and "valid_sampled_token_count" in tensor_dict
                ):
                    self.valid_sampled_token_count_gpu = tensor_dict[
                        "valid_sampled_token_count"
                    ].to(self.device)
                    self.input_batch.prev_req_id_to_index = {
                        req_id: i
                        for i, req_id in enumerate(self.input_batch.req_ids)
                    }

                if self.model_config.is_hybrid:
                    if self.cache_config.mamba_cache_mode == "align":
                        accepted_counts = (
                            self.num_accepted_tokens.gpu[:num_reqs]
                            .cpu()
                            .numpy()
                        )
                        for i, num_tokens in enumerate(accepted_counts):
                            self.input_batch.num_accepted_tokens_cpu[i] = (
                                num_tokens
                            )
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
                        self.input_batch.num_accepted_tokens_cpu_tensor[
                            :num_reqs
                        ].copy_(
                            self.num_accepted_tokens.gpu[:num_reqs],
                            non_blocking=True,
                        )
                else:
                    self.input_batch.num_accepted_tokens_cpu_tensor[
                        :num_reqs
                    ].copy_(
                        self.num_accepted_tokens.gpu[:num_reqs],
                        non_blocking=True,
                    )

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
        pp = get_pp_group()
        skip_pp_pd_broadcast = self.is_kv_producer and pp.world_size > 1

        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            # receive sampled token ids from the last PP rank when using
            # async scheduling + pipeline parallelism so downstream code
            # (e.g., PCP input preparation) can access them.
            if self.use_async_scheduling and pp.world_size > 1 and not skip_pp_pd_broadcast and not self._edge_cloud_enabled:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            if not kv_connector_output:
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
            mtp_target_hidden_states,
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

        is_mid_prefill_chunk = bool(
            self._edge_cloud_enabled
            and scheduler_output.batch_type == BatchType.PREFILL_LAST
            and not getattr(
                scheduler_output, "is_last_prefill_chunk", True
            )
        )
        can_skip_mid_prefill_sampling = bool(
            is_mid_prefill_chunk
            and self.speculative_config is None
            and not self.num_prompt_logprobs
            and not self.model_config.enable_return_routed_experts
            and not self.need_accepted_tokens
        )
        if can_skip_mid_prefill_sampling:
            # With no drafter and no sampling-dependent auxiliary output, a
            # middle chunk has no valid generated token.  Its target KV was
            # already written by execute_model(), so avoid the sampler and
            # bookkeeping while preserving the normal connector/profiling
            # output and end-of-forward hooks.
            req_ids = list(scheduler_output.num_scheduled_tokens)
            output = ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
                kv_connector_output=kv_connector_output,
                pooler_output=[],
                ec_connector_output=(
                    ec_connector_output if self.supports_mm_inputs else None
                ),
                cudagraph_stats=cudagraph_stats,
                **(
                    {}
                    if vllm_version_is("0.20.2")
                    else {"routed_experts": None}
                ),
            )
            if (
                self.ascend_config.profiling_chunk_config.need_timing
                and hasattr(self, "_execution_start_time")
            ):
                self._sync_device()
                output.execution_time_ms = (
                    time.perf_counter() - self._execution_start_time
                ) * 1000.0
            if self.dynamic_eplb:
                with record_function_or_nullcontext("EPLB update"):
                    self.eplb_updator.forward_end()
            self._finalize_dump_data()
            return output

        # With speculative decoding enabled, a mid-prefill chunk intentionally
        # continues through sampling and drafting. _prepare_inputs marked it
        # in discard_request_indices, so bookkeeping emits no target token and
        # prepare_next_token_ids_padded feeds the real next prompt token to the
        # drafter. This mirrors the native non-edge-cloud path and is required
        # to populate MTP KV for the entire prompt.

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

        self.valid_sampled_token_count_gpu = None

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

        edge_cloud_draft_state: dict[str, Any] | None = None
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
                defer_edge_cloud_draft = (
                    self._should_defer_edge_cloud_draft(scheduler_output)
                )
                hf_config = getattr(
                    self.vllm_config.model_config, "hf_config", None
                )
                logger.info(
                    "[MTP-DEBUG] draft defer decision: method=%s, "
                    "model_type=%s, batch_type=%s, scheduled_draft=%s, "
                    "edge_cloud_enabled=%s, is_edge_device=%s, "
                    "has_drafter=%s, defer=%s, head_token=%s",
                    getattr(self.speculative_config, "method", None),
                    getattr(hf_config, "model_type", None),
                    scheduler_output.batch_type,
                    self._uses_scheduled_edge_cloud_draft(),
                    self._edge_cloud_enabled,
                    is_edge_device(),
                    self.drafter is not None,
                    defer_edge_cloud_draft,
                    scheduler_output.head_token,
                )
                if defer_edge_cloud_draft:
                    sampled_token_ids = (
                        sampler_output.sampled_token_ids
                        if use_padded_batch
                        else valid_sampled_token_ids
                    )
                    self._stash_pending_edge_cloud_draft_context(
                        scheduler_output,
                        sampled_token_ids,
                        positions,
                        hidden_states,
                        mtp_target_hidden_states,
                    )
                elif use_padded_batch:
                    # EAGLE speculative decoding can use the GPU sampled tokens
                    # as inputs, and does not need to wait for bookkeeping to finish.
                    propose_draft_token_ids(sampler_output.sampled_token_ids)
                if not use_padded_batch and not defer_edge_cloud_draft:
                    # ngram and other speculative decoding methods use the sampled
                    # tokens on the CPU, so they are run after bookkeeping.
                    propose_draft_token_ids(valid_sampled_token_ids)

            # Edge-cloud sync: send num_accepted_tokens (and optionally
            # valid_sampled_token_count) to cloud so that it can update its
            # speculative-decoding state in both embedding_only and head_tail
            # modes.
            if (
                self._edge_cloud_enabled
                and self.edge_cloud_cfg.role != "cloud"
                and self.speculative_config
                and (
                    self.model_config.is_hybrid
                    or self._uses_scheduled_edge_cloud_draft()
                )
            ):
                if self._should_defer_edge_cloud_draft(scheduler_output):
                    # The async output already copies sampled_token_ids to the
                    # host. EngineCore derives both accepted-count fields from
                    # that existing result. Do not add another synchronous
                    # D2H here: the edge does not consume these cloud-only
                    # scalars, and the next local DRAFT_FIRST is already queued.
                    task_id = scheduler_output.head_token
                    context = (
                        self._pending_edge_cloud_draft_contexts.get(task_id)
                        if task_id is not None
                        else None
                    )
                    if context is None:
                        raise RuntimeError(
                            "Deferred edge-cloud draft context missing for "
                            f"head_token={task_id}"
                        )
                    edge_cloud_draft_state = {
                        "draft_task_id": task_id,
                        "draft_step_idx": 0,
                    }
                elif get_pp_group().world_size == 2:
                    num_accepted = (
                        (sampler_output.sampled_token_ids != -1)
                        .sum(dim=1)
                        .cpu()
                    )
                    tensor_dict_to_send = {
                        "num_accepted_tokens": num_accepted
                    }
                    send_work = get_pp_group().isend_tensor_dict(tensor_dict_to_send)
                    for handle in send_work:
                        handle.wait()

            # vLLM v0.18 defers KV connector finalization during target-model
            # forward when speculative decoding is enabled. Finalize here after
            # draft model runs so KV pool save/put can complete.
            if self.speculative_config is not None:
                self.finalize_kv_connector()

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
            routed_experts=None,
        )
        if edge_cloud_draft_state is not None:
            model_runner_output.edge_cloud_draft_state = (
                edge_cloud_draft_state
            )
        if self.ascend_config.profiling_chunk_config.need_timing and hasattr(self, '_execution_start_time'):
            self._sync_device()
            model_runner_output.execution_time_ms = (time.perf_counter() - self._execution_start_time) * 1000.0

        if self.dynamic_eplb:
            self.eplb_updator.forward_end(self.eplb_heat_collection_status)

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
            if not self._edge_cloud_enabled and pp.world_size > 1 and pp.is_last_rank and not skip_pp_pd_broadcast:
                self._pp_broadcast_prev_sampled_token_ids(sampler_output.sampled_token_ids)

        if not self.use_async_scheduling:
            if self.routed_experts_initialized:
                # Sync path: D2H was issued in ``_bookkeeping_sync`` and
                # synchronized by ``_to_list``'s event.synchronize(), so
                # the pinned buffers are ready to be wrapped as numpy.
                total = scheduler_output.total_num_scheduled_tokens
                model_runner_output.routed_experts = RoutedExpertsLists(
                    routing_data=self.routed_experts_cpu[:total].numpy(),
                    slot_mapping=self.routed_experts_slot_mapping_cpu[:total].numpy(),
                )
            return model_runner_output
        
        # Async path: produce a device-side snapshot that the async
        # copy stream can D2H later. Both tensors must be private
        # clones because:
        #   - ``routing_data`` source is the shared capturer buffer,
        #     which is ``clear_buffer()``-ed at the start of the
        #     next step on the default stream.
        #   - ``slot_mapping`` source is our own
        #     ``routed_experts_slot_mapping_device``, which the
        #     next ``_prepare_inputs`` overwrites on the default
        #     stream while the D2H is still pending on the copy
        #     stream.
        # Without clones, the copy stream would read torn data.
        routed_experts_snapshot = None
        if self.routed_experts_initialized:
            buf = self.routed_experts_capturer.get_device_buffer()
            total = scheduler_output.total_num_scheduled_tokens
            routed_experts_snapshot = RoutedExpertsTensors(
                routing_data=buf[:total].clone(),
                slot_mapping=self.routed_experts_slot_mapping_device[
                    :total
                ].clone(),
            )
        async_output = AsyncGPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
            routed_experts=routed_experts_snapshot,
        )
        self.input_batch.set_async_sampled_token_ids(
            async_output.sampled_token_ids_cpu,
            async_output.async_copy_ready_event,
        )
        return async_output

    def _cache_cloud_spec_decode_metadata(
        self,
        scheduler_output: "SchedulerOutput",
        common_attn_metadata: AscendCommonAttentionMetadata,
        num_reqs: int,
        positions: torch.Tensor,
    ) -> None:
        if not self._uses_scheduled_edge_cloud_draft():
            # Other draft implementations consume the metadata synchronously
            # and do not cross a scheduler boundary.
            self._cloud_spec_decode_common_attn_metadata = (
                common_attn_metadata
            )
            self._cloud_spec_decode_num_reqs = num_reqs
            return

        frozen_metadata = _freeze_scheduled_state(common_attn_metadata)

        # Keep the latest snapshot for the legacy synchronous draft path.
        self._cloud_spec_decode_common_attn_metadata = frozen_metadata
        self._cloud_spec_decode_num_reqs = num_reqs

        task_id = scheduler_output.head_token
        if task_id is None:
            raise RuntimeError(
                "Cannot cache cloud draft metadata without target head_token"
            )
        task_cache = self._cloud_spec_decode_metadata_by_task
        if (
            task_id not in task_cache
            and len(task_cache)
            >= self._cloud_spec_decode_metadata_cache_max
        ):
            stale_task_id = next(iter(task_cache))
            task_cache.pop(stale_task_id)
            self._cloud_scheduler_output_by_task.pop(stale_task_id, None)
            self._cloud_draft_position_state_by_task.pop(
                stale_task_id, None
            )
            self._cloud_target_generation_by_task.pop(stale_task_id, None)
            self._eagle3_cloud_aux_hidden_states_by_task.pop(
                stale_task_id, None
            )
            logger.warning(
                "Cloud draft metadata cache exceeded bound (%d); evicting "
                "unconsumed task_id=%s",
                self._cloud_spec_decode_metadata_cache_max,
                stale_task_id,
            )
        task_cache[task_id] = (frozen_metadata, num_reqs)
        # Freeze the verify step's scheduler_output alongside the metadata.
        # The draft task that consumes it is scheduled independently, so the
        # global _last_scheduler_output may already have moved on.
        self._cloud_scheduler_output_by_task[task_id] = replace(
            scheduler_output
        )
        num_scheduled_tokens = tuple(
            int(scheduler_output.num_scheduled_tokens[req_id])
            for req_id in self.input_batch.req_ids
        )
        num_tokens = sum(num_scheduled_tokens)
        if positions.shape[-1] < num_tokens:
            raise RuntimeError(
                "Cloud target positions are shorter than the scheduled "
                f"draft input: positions={positions.shape}, "
                f"num_tokens={num_tokens}, task_id={task_id}"
            )
        actual_num_computed_tokens: list[int] = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            scheduler_value = int(
                self.input_batch.num_computed_tokens_cpu[i]
            )
            confirmed = self._cloud_actual_num_computed_by_req.get(req_id)
            previous_generation = (
                self._cloud_latest_target_generation_by_req.get(req_id)
            )
            if (
                confirmed is not None
                and confirmed[0] == previous_generation
                and 0 <= scheduler_value - confirmed[1]
                <= self.num_spec_tokens
            ):
                actual_num_computed_tokens.append(confirmed[1])
            else:
                actual_num_computed_tokens.append(scheduler_value)
        position_state = CloudDraftPositionState(
            target_positions=positions[..., :num_tokens].clone(),
            num_scheduled_tokens=num_scheduled_tokens,
            is_prefill=(
                scheduler_output.batch_type == BatchType.PREFILL_FIRST
            ),
            req_ids=tuple(self.input_batch.req_ids),
            actual_num_computed_tokens=tuple(
                actual_num_computed_tokens
            ),
        )
        self._cloud_draft_position_state_by_task[task_id] = position_state
        generation = self._cloud_target_generation_by_task.get(task_id)
        if generation is None:
            self._cloud_target_generation += 1
            generation = self._cloud_target_generation
            self._cloud_target_generation_by_task[task_id] = generation
        for req_id in position_state.req_ids:
            self._cloud_latest_target_generation_by_req[req_id] = generation

    def _resolve_cloud_spec_decode_metadata(
        self,
        scheduler_output: "SchedulerOutput | None",
    ) -> tuple[AscendCommonAttentionMetadata | None, int]:
        if scheduler_output is None:
            return (
                self._cloud_spec_decode_common_attn_metadata,
                self._cloud_spec_decode_num_reqs,
            )

        task_id = scheduler_output.draft_task_id
        if task_id is None:
            raise RuntimeError("DRAFT batch missing draft_task_id")

        task_cache = self._cloud_spec_decode_metadata_by_task
        cached = task_cache.get(task_id)
        if cached is None:
            raise RuntimeError(
                "DRAFT has no matching target attention metadata: "
                f"task_id={task_id}"
            )
        return cached

    def _reconstruct_cloud_draft_positions(
        self,
        scheduler_output: "SchedulerOutput",
        num_tokens: int,
    ) -> torch.Tensor:
        """Reconstruct draft positions from the cached cloud target step."""
        task_id = scheduler_output.draft_task_id
        if task_id is None:
            raise RuntimeError("DRAFT batch missing draft_task_id")
        state = self._cloud_draft_position_state_by_task.get(task_id)
        if state is None:
            raise RuntimeError(
                "DRAFT has no matching target positions: "
                f"task_id={task_id}"
            )

        draft_step_idx = int(scheduler_output.draft_step_idx or 0)
        if draft_step_idx == 0:
            target_positions = state.target_positions
            if target_positions.shape[-1] != num_tokens:
                raise RuntimeError(
                    "DRAFT step-0 position/token mismatch: "
                    f"positions={target_positions.shape[-1]}, "
                    f"tokens={num_tokens}, task_id={task_id}"
                )

            accepted_counts = scheduler_output.num_accepted_tokens
            if not state.is_prefill and accepted_counts is None:
                raise RuntimeError(
                    "Decode DRAFT step 0 is missing num_accepted_tokens: "
                    f"task_id={task_id}"
                )
            if (
                accepted_counts is not None
                and len(accepted_counts)
                != len(state.num_scheduled_tokens)
            ):
                raise RuntimeError(
                    "DRAFT accepted-count/request mismatch: "
                    f"accepted={len(accepted_counts)}, "
                    f"requests={len(state.num_scheduled_tokens)}, "
                    f"task_id={task_id}"
                )

            sample_rows: list[int] = []
            start = 0
            for req_idx, scheduled in enumerate(state.num_scheduled_tokens):
                if scheduled <= 0:
                    raise RuntimeError(
                        "DRAFT request has no scheduled tokens: "
                        f"req_idx={req_idx}, task_id={task_id}"
                    )
                if state.is_prefill:
                    accepted = scheduled
                else:
                    assert accepted_counts is not None
                    if isinstance(accepted_counts, dict):
                        # Counts are keyed by req_id; pair them with the
                        # cached layout by identity, never by position.
                        req_id = state.req_ids[req_idx]
                        if req_id not in accepted_counts:
                            raise RuntimeError(
                                "DRAFT accepted counts missing request: "
                                f"req_id={req_id}, task_id={task_id}"
                            )
                        accepted_count = accepted_counts[req_id]
                    else:
                        accepted_count = accepted_counts[req_idx]
                    accepted = min(
                        max(int(accepted_count), 1),
                        scheduled,
                    )
                sample_rows.append(start + accepted - 1)
                start += scheduled

            row_indices = torch.tensor(
                sample_rows,
                dtype=torch.long,
                device=target_positions.device,
            )
            state.base_positions = target_positions.index_select(
                -1, row_indices
            )
            return target_positions

        base_positions = state.base_positions
        if base_positions is None:
            raise RuntimeError(
                "DRAFT follow-up step has no reconstructed base positions: "
                f"step={draft_step_idx}, task_id={task_id}"
            )
        if base_positions.shape[-1] != num_tokens:
            raise RuntimeError(
                "DRAFT follow-up position/token mismatch: "
                f"positions={base_positions.shape[-1]}, "
                f"tokens={num_tokens}, task_id={task_id}"
            )
        return base_positions + draft_step_idx

    def _purge_invalidated_cloud_draft_metadata(
        self, task_ids: list[str] | None
    ) -> None:
        """Cloud-side purge of draft metadata for edge-dropped tasks.

        The edge drops a deferred draft when every request of its parent
        verify/prefill batch finished (or was aborted), so the DRAFT
        batch never (fully) arrives and the normal pop at the last draft
        step never runs.  The edge stamps the affected task ids on a
        later SchedulerOutput (``cloud_draft_invalidate_task_ids``);
        purge the entries here instead of letting them occupy the
        bounded cache until eviction (which could otherwise evict a
        still-in-flight task and crash its DRAFT with "no matching
        target attention metadata").  The edge only invalidates tasks
        whose draft was never published/dispatched or already fully
        consumed, so purging cannot race an in-flight DRAFT batch.

        Only active on the cloud side with scheduled edge-cloud draft;
        a no-op everywhere else.
        """
        if not task_ids:
            return
        if not (
            self._edge_cloud_enabled
            and not is_edge_device()
            and self._uses_scheduled_edge_cloud_draft()
        ):
            return
        for task_id in task_ids:
            if (
                self._cloud_spec_decode_metadata_by_task.pop(task_id, None)
                is not None
            ):
                logger.info(
                    "Purged cloud draft metadata for invalidated "
                    "task_id=%s (draft dropped on the edge)",
                    task_id,
                )
            self._cloud_scheduler_output_by_task.pop(task_id, None)
            # NOTE: the original fix (518616040) forgot this dict; its
            # cloned positions tensors would otherwise linger until the
            # bounded metadata cache evicts the task.
            self._cloud_draft_position_state_by_task.pop(task_id, None)
            self._cloud_target_generation_by_task.pop(task_id, None)
            stale_req_ids = [
                req_id
                for req_id, correction in (
                    self._cloud_pending_request_corrections.items()
                )
                if correction.task_id == task_id
            ]
            for req_id in stale_req_ids:
                self._cloud_pending_request_corrections.pop(req_id, None)
            self._eagle3_cloud_aux_hidden_states_by_task.pop(task_id, None)

    def _build_edge_cloud_draft_attn_metadata(
        self,
        positions: torch.Tensor,
        spec_step_idx: int,
        scheduler_output: "SchedulerOutput | None" = None,
    ) -> dict[str, Any] | None:
        """Build per-layer attention metadata for the draft cloud decoder.

        On the cloud side, the draft decoder layers are real (not
        PPMissingLayer) and need proper attention metadata to produce
        correct outputs.  Without it, the Ascend attention backend
        silently returns zeros, corrupting hidden states and causing
        low draft hit rates.

        Uses the spec_decode_common_attn_metadata saved during
        execute_model() and the drafter's draft_attn_groups to build
        per-layer metadata for each speculative step.
        """
        common_attn_metadata, num_reqs = (
            self._resolve_cloud_spec_decode_metadata(scheduler_output)
        )
        if common_attn_metadata is None:
            return None

        if (
            self.drafter is None
            or not hasattr(self.drafter, "draft_attn_groups")
            or not self.drafter.draft_attn_groups
        ):
            return None

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
        common_attn_metadata.num_reqs = batch_size

        # Use the actual number of tokens carried by positions,
        # which already accounts for rejected tokens on the edge side.
        num_input_tokens = positions.shape[-1]
        num_actual_tokens = num_input_tokens
        common_attn_metadata.num_actual_tokens = num_actual_tokens
        common_attn_metadata.num_input_tokens = num_input_tokens

        if spec_step_idx > 0:
            if num_input_tokens != batch_size:
                raise RuntimeError(
                    "Edge-cloud draft follow-up step must carry exactly one "
                    "token per request: "
                    f"step={spec_step_idx}, tokens={num_input_tokens}, "
                    f"requests={batch_size}"
                )
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

            # Match the proposer path: MTP uses SpecDecoding metadata while
            # Eagle3 keeps the regular chunked-prefill attention state for
            # its single-token follow-up passes.
            common_attn_metadata.attn_state = (
                AscendAttentionState.ChunkedPrefill
                if self.speculative_config.method == "eagle3"
                else AscendAttentionState.SpecDecoding
            )

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
                common_attn_metadata.attn_state = (
                    AscendAttentionState.ChunkedPrefill
                    if self.speculative_config.method == "eagle3"
                    else AscendAttentionState.SpecDecoding
                )

        # Build per-layer attention metadata using draft_attn_groups.
        per_layer_attn_metadata: dict[str, Any] = {}
        draft_model = getattr(self.drafter, "model", None)
        uses_dsa_draft_metadata = bool(
            getattr(
                draft_model,
                "edge_cloud_uses_dsa_draft_metadata",
                False,
            )
        )
        for attn_group in self.drafter.draft_attn_groups:
            builder = attn_group.get_metadata_builder()
            extra_metadata_args: dict[str, Any] = {}
            if uses_dsa_draft_metadata:
                extra_metadata_args = {
                    "prefill_ratio_to_sas_metadata": {},
                    "decode_ratio_to_sas_metadata": {},
                    "common_ratio_to_sas_metadata": {},
                    "block_size": attn_group.kv_cache_spec.block_size,
                }
            if spec_step_idx == 0:
                attn_meta = builder.build(
                    0,
                    common_attn_metadata,
                    **extra_metadata_args,
                )
            else:
                attn_meta = builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=spec_step_idx,
                    **extra_metadata_args,
                )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_meta

        return per_layer_attn_metadata

    def _record_cloud_request_corrections(
        self,
        scheduler_output: "SchedulerOutput",
        num_accepted_values: dict[str, int] | list[int],
        valid_sampled_values: dict[str, int] | list[int] | None,
    ) -> int:
        """Record rejection corrections in request space, not batch space.

        The edge scheduler value for the next async target may still assume
        that every draft from this target was accepted.  The confirmed count
        lets the next target replace that optimistic value without gathering
        from a positional GPU buffer that unrelated work may have overwritten.
        """
        task_id = scheduler_output.draft_task_id
        if task_id is None or valid_sampled_values is None:
            return 0
        target_output = self._cloud_scheduler_output_by_task.get(task_id)
        position_state = self._cloud_draft_position_state_by_task.get(task_id)
        generation = self._cloud_target_generation_by_task.get(task_id)
        if (
            target_output is None
            or position_state is None
            or generation is None
        ):
            return 0

        req_ids = position_state.req_ids
        if isinstance(valid_sampled_values, dict):
            valid_by_req = valid_sampled_values
        else:
            valid_by_req = dict(zip(req_ids, valid_sampled_values))
        if isinstance(num_accepted_values, dict):
            accepted_by_req = num_accepted_values
        else:
            accepted_by_req = dict(zip(req_ids, num_accepted_values))

        start_by_req = {
            req_id: int(num_computed)
            for req_id, num_computed in zip(
                target_output.scheduled_cached_reqs.req_ids,
                target_output.scheduled_cached_reqs.num_computed_tokens,
            )
        }
        start_by_req.update({
            req.req_id: int(req.num_computed_tokens)
            for req in target_output.scheduled_new_reqs
        })
        actual_start_values = getattr(
            position_state, "actual_num_computed_tokens", ()
        )
        actual_start_by_req = dict(zip(req_ids, actual_start_values))

        recorded = 0
        spec_tokens = target_output.scheduled_spec_decode_tokens
        for req_id, valid_value in valid_by_req.items():
            num_draft = len(spec_tokens.get(req_id, ()))
            if num_draft <= 0 or req_id not in start_by_req:
                continue
            valid_count = int(valid_value)
            if not 1 <= valid_count <= num_draft + 1:
                logger.warning(
                    "Ignoring invalid cloud accepted count: task=%s req=%s "
                    "valid=%d num_draft=%d",
                    task_id,
                    req_id,
                    valid_count,
                    num_draft,
                )
                continue
            if (
                self._cloud_latest_target_generation_by_req.get(req_id)
                != generation
            ):
                logger.warning(
                    "Ignoring late cloud accepted state: task=%s req=%s "
                    "generation=%d latest=%s",
                    task_id,
                    req_id,
                    generation,
                    self._cloud_latest_target_generation_by_req.get(req_id),
                )
                continue

            optimistic = start_by_req[req_id] + int(
                target_output.num_scheduled_tokens[req_id]
            )
            actual_start = actual_start_by_req.get(
                req_id, start_by_req[req_id]
            )
            actual = actual_start + valid_count
            pending = CloudPendingRequestCorrection(
                task_id=task_id,
                generation=generation,
                num_draft_tokens=num_draft,
                optimistic_num_computed_tokens=optimistic,
                actual_num_computed_tokens=actual,
                num_accepted_tokens=int(
                    accepted_by_req.get(req_id, valid_count)
                ),
            )
            previous = self._cloud_pending_request_corrections.get(req_id)
            if previous is None or previous.generation <= generation:
                self._cloud_pending_request_corrections[req_id] = pending
                self._cloud_actual_num_computed_by_req[req_id] = (
                    generation,
                    actual,
                )
                recorded += 1
        return recorded

    def _run_edge_cloud_draft_middle_segment(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors,
    ) -> IntermediateTensors:
        """Run the cloud-side draft middle (target/C) segment forward.

        Pure compute: consumes the cloud-bound ``intermediate_tensors``
        (recv is done in the worker, mirroring
        ``_run_edge_cloud_draft_last_segment``) and returns the segment's
        output intermediates (the worker sends them, mirroring
        ``_run_edge_cloud_draft_first_segment``).  Cross-PP edge-cloud
        communication stays in the worker layer, consistent with the
        non-draft ``_execute_model_cloud`` path.

        The edge self-posts DRAFT_LAST together with DRAFT_FIRST, so the
        matching receive does not depend on a cloud worker ack or POST_OUT.
        The worker records (rather than waits for) the cloud->edge send,
        exactly like ``_execute_model_cloud``.
        """
        # DRAFT batches bypass execute_model/_update_states on the cloud, so
        # the purge hook there never runs for them.  Consume any piggybacked
        # draft-metadata invalidations here instead; the call is a guarded
        # no-op off the cloud path.
        self._purge_invalidated_cloud_draft_metadata(
            getattr(scheduler_output, "cloud_draft_invalidate_task_ids", None)
        )
        spec_step_idx = int(scheduler_output.draft_step_idx or 0)
        # The edge carries rejection-corrected sampling state on the step-0
        # SchedulerOutput. Keeping it on the control plane avoids extra CPU
        # tensors in the dynamic hidden-state payload.
        num_accepted_values = scheduler_output.num_accepted_tokens
        valid_sampled_values = scheduler_output.valid_sampled_token_count
        if num_accepted_values is not None:
            self._record_cloud_request_corrections(
                scheduler_output,
                num_accepted_values,
                valid_sampled_values,
            )
            if spec_step_idx != 0:
                logger.warning(
                    "num_accepted scheduler state arrived on draft step %d; "
                    "expected step 0",
                    spec_step_idx,
                )

        token_tensor_key = (
            "input_embeds"
            if self.speculative_config.method == "eagle3"
            else "hidden_states"
        )
        token_tensor = intermediate_tensors.tensors.get(token_tensor_key)
        if token_tensor is None:
            raise RuntimeError(
                "DRAFT cloud payload is missing the token tensor: "
                f"key={token_tensor_key}"
            )
        num_tokens = token_tensor.shape[0]
        positions = self._reconstruct_cloud_draft_positions(
            scheduler_output, num_tokens
        )
        full_positions = positions
        intermediate = self._sync_edge_cloud_draft_intermediate_tensors(
            num_tokens, intermediate_tensors
        )
        positions = self._chunk_deepseek_v4_mtp_draft_positions(
            full_positions
        )
        model_kwargs = {
            "intermediate_tensors": intermediate,
            "positions": positions,
            "spec_step_idx": spec_step_idx,
        }
        if self.speculative_config.method == "eagle3":
            task_id = scheduler_output.draft_task_id
            aux_hidden_states = (
                self._eagle3_cloud_aux_hidden_states_by_task.get(task_id)
                if spec_step_idx == 0 and task_id is not None
                else None
            )
            self._prepare_eagle3_cloud_hidden_states(
                self._edge_cloud_draft_segments["c"],
                intermediate,
                aux_hidden_states,
                num_tokens,
                is_first_step=spec_step_idx == 0,
            )
        draft_attn_metadata = self._build_edge_cloud_draft_attn_metadata(
            full_positions, spec_step_idx, scheduler_output
        )

        if is_forward_context_available():
            forward_context = get_forward_context()
            cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
            if hasattr(cudagraph_runtime_mode, "decode_mode"):
                cudagraph_runtime_mode = cudagraph_runtime_mode.decode_mode()
            batch_descriptor = forward_context.batch_descriptor
            num_actual_tokens = getattr(
                forward_context, "num_actual_tokens", num_tokens
            )
        else:
            cudagraph_runtime_mode = CUDAGraphMode.NONE
            batch_descriptor = BatchDescriptor(num_tokens)
            num_actual_tokens = num_tokens

        with set_ascend_forward_context(
            attn_metadata=draft_attn_metadata,
            vllm_config=self.vllm_config,
            num_tokens=num_tokens,
            num_actual_tokens=num_actual_tokens,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=cudagraph_runtime_mode,
            is_draft_model=True,
        ):
            output = self._edge_cloud_draft_segments["c"](**model_kwargs)
        if not isinstance(output, IntermediateTensors):
            raise RuntimeError(
                "Edge-cloud draft middle segment returned no intermediates"
            )

        if (
            scheduler_output.draft_task_id is not None
            and spec_step_idx + 1 >= self.num_spec_tokens
        ):
            self._cloud_spec_decode_metadata_by_task.pop(
                scheduler_output.draft_task_id, None
            )
            self._cloud_scheduler_output_by_task.pop(
                scheduler_output.draft_task_id, None
            )
            self._cloud_draft_position_state_by_task.pop(
                scheduler_output.draft_task_id, None
            )
            self._cloud_target_generation_by_task.pop(
                scheduler_output.draft_task_id, None
            )
            self._eagle3_cloud_aux_hidden_states_by_task.pop(
                scheduler_output.draft_task_id, None
            )
        return output

    # overwrite _sample for lmhead_tp_enable and need_accepted_tokens
    def _sample(self, logits, spec_decode_metadata):
        # Sample the next token and get logprobs if needed.
        self.input_batch.update_async_output_token_ids()
        sampling_metadata = self.input_batch.sampling_metadata
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
            # Sync scheduling: issue routed experts D2H into the pinned
            # CPU buffer BEFORE ``_to_list`` below. ``_to_list`` does
            # ``event.synchronize()`` on the async copy stream which
            # waits for every D2H queued on the default stream since
            # the last sync, so this enqueue is naturally covered
            # without requiring its own synchronize.
            if self.routed_experts_initialized:
                buf = self.routed_experts_capturer.get_device_buffer()
                total = scheduler_output.total_num_scheduled_tokens
                self.routed_experts_cpu[:total].copy_(buf[:total], non_blocking=True)
                self.routed_experts_slot_mapping_cpu[:total].copy_(
                    self.routed_experts_slot_mapping_device[:total],
                    non_blocking=True,
                )

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
                new_prev_map = {
                    req_id: i
                    for i, req_id in enumerate(self.input_batch.req_ids)
                    if i not in invalid_req_indices_set
                }
                sampled_token_ids, new_prev_map = (
                    self._merge_pending_prev_sampled(
                        sampled_token_ids, new_prev_map
                    )
                )
                self.input_batch.prev_sampled_token_ids = sampled_token_ids
                self.input_batch.prev_req_id_to_index = new_prev_map
            else:
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
        skip_dsa_fill: bool = False,
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

        Args:
            skip_dsa_fill: If True, skip filling _dsa_positions_cpu_buf
                (caller already filled it, e.g. slow path between first
                _prepare_inputs and _run_input_preparation).
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

            ) = precomputed
        else:
            (
                logits_indices,
                spec_decode_metadata,
                total_num_scheduled_tokens,

            ) = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

        # Fill _dsa_positions_cpu_buf for DSA compression.
        # cloud_prepare_early calls _run_input_preparation directly and
        # relies on this fill.  The slow path passes skip_dsa_fill=True
        # because it already filled above (between the first _prepare_inputs
        # and _run_input_preparation, to use the first call's values).
        if self.use_compress and not skip_dsa_fill:
            req_indices = np.repeat(
                self.arange_np[:num_reqs], num_scheduled_tokens_np
            )
            dsa_positions_np = self._dsa_positions_np_buf[
                :total_num_scheduled_tokens
            ]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                self.query_pos.np[:total_num_scheduled_tokens],
                out=dsa_positions_np,
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

        if self.dynamic_eplb:
            self.update_eplb_heat_collection_status(num_tokens_padded)

        pad_attn = cudagraph_mode == CUDAGraphMode.FULL
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

        if (
            cudagraph_mode == CUDAGraphMode.FULL
            or (enable_sp() and not self.model_config.use_mla)
            and self.pcp_size * self.dcp_size == 1
        ):
            num_reqs_padded = self._pad_query_start_loc_for_fia(
                self.query_start_loc,
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
            )
        )

        self._sanitize_placeholder_input_ids_for_forward(
            scheduler_output,
            num_tokens_padded
            if not (self.use_cp and self.pcp_manager.pcp_use_hybrid_attn)
            else total_num_scheduled_tokens,
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
        }

    def step_has_multimodal_req(self, scheduler_output) -> bool:
        """Whether the current step's batch contains any multimodal request.

        Used to decide whether mrope_positions must be transferred edge->cloud
        (only multimodal requests need it; text-only batches can be computed
        locally on the cloud because empty mm_features degrades M-RoPE to 1D
        without hitting the missing image_grid_thw). Must return the SAME value
        on edge and cloud (they share the scheduler_output and build req_state
        from the same NewRequestData.mm_features).
        """
        # cached/running reqs: covers decode of multimodal requests (whose
        # mm_features stay non-empty after prefill).
        if any(rs.mm_features for rs in self.requests.values()):
            return True
        # new reqs this step: cloud recv runs BEFORE cloud_prepare_early builds
        # req_state, so on the cloud side self.requests does not yet contain
        # this step's new reqs; check scheduler_output directly.
        for nr in scheduler_output.scheduled_new_reqs:
            if getattr(nr, "mm_features", None):
                return True
        return False

    def _init_mrope_positions(self, req_state) -> None:
        # In edge-cloud cloud mode: skip M-RoPE init only for multimodal
        # requests (their image_grid_thw / video_grid_thw did not cross the
        # edge->cloud mm_features boundary, so local init would KeyError).
        # Text-only requests (empty mm_features) init locally: _iter_mm_grid_hw
        # does not enter its loop, M-RoPE degrades to 1D, no crash. This lets
        # text-only batches skip the mrope transfer entirely.
        # profile_run / _dummy_run do not call this, so the role guard does not
        # affect profiling.
        if (self._edge_cloud_enabled
                and self.edge_cloud_cfg.role == "cloud"
                and req_state.mm_features):
            return
        super()._init_mrope_positions(req_state)

    def _calc_mrope_positions(self, scheduler_output) -> None:
        # In edge-cloud cloud mode: skip local calc only when the batch carries
        # wire mrope (edge transfers the whole-batch mrope buffer and
        # execute_model injects it). Text-only batches compute locally.
        # Use the edge scheduler's `has_mrope` stamp (authoritative) rather
        # than the cloud's own registry, which lags behind after an mm->text
        # transition (finished_req_ids flushed via EMPTY batches never reach
        # the cloud runner); a stale registry would skip local calc for a
        # text batch that has no wire mrope either, yielding garbage RoPE.
        if self._edge_cloud_enabled and self.edge_cloud_cfg.role == "cloud":
            has_mrope = getattr(scheduler_output, "has_mrope", None)
            if has_mrope is None:
                has_mrope = self.step_has_multimodal_req(scheduler_output)
            if has_mrope:
                return
        super()._calc_mrope_positions(scheduler_output)

    def cloud_prepare_early(self, scheduler_output: "SchedulerOutput") -> None:
        """Pre-compute input preparation on cloud while edge runs segment_a.

        Caches results in self._cloud_prepare_cache so that when edge data
        arrives, execute_model can skip _update_states, _prepare_inputs,
        _determine_batch_execution_and_padding, and _build_attention_metadata,
        going directly to _preprocess (sync_and_gather) + _model_forward.

        This method mirrors the normal execute_model slow path STEP BY STEP
        (same order, same side effects) so that the cloud fast path is
        observationally identical to not having run early preparation.
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
            # Still run _update_states: a zero-token slice may be the first
            # slice of a new request that just entered the cloud worker's
            # batch.  _update_states must see the request at least once to
            # populate req_data.all_token_ids; otherwise a later
            # DECODE_FIRST / DRAFT_FIRST will KeyError.
            self._update_states(scheduler_output)
            return

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

        # Passed to execute_model through _cloud_prepare_cache. Ascend's
        # preprocess_mamba only fills this buffer; execute_model performs the
        # copy after KV connector loading.
        mamba_preprocess_bufs = None

        # cloud_prepare_early runs BEFORE the forward pass (outside
        # torch.inference_mode), but GDN attention builder does in-place
        # tensor copies that require inference mode.  Wrap the whole
        # preparation inside inference_mode to match execute_model's
        # inference-mode context.
        with torch.inference_mode():
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

            # --- _update_states (same as slow path) ---
            deferred_state_corrections_fn = self._update_states(
                scheduler_output
            )

            # Same empty-batch guard as the slow path (returns EMPTY there).
            num_reqs = self.input_batch.num_reqs
            if num_reqs == 0:
                self._cloud_prepare_cache = None
                return

            # --- _prepare_inputs (first explicit call, same as slow path) ---
            req_ids = self.input_batch.req_ids
            tokens = [
                scheduler_output.num_scheduled_tokens[i] for i in req_ids
            ]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            (
                logits_indices,
                spec_decode_metadata,
                total_num_scheduled_tokens,
            ) = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            # --- mamba align preprocess (same as slow path) ---
            # Keep the local-cache ownership guard in sync with the slow path.
            if (
                self.cache_config.mamba_cache_mode == "align"
                and self.need_accepted_tokens
            ):
                # preprocess_mamba reads req_state.num_computed_tokens (CPU)
                # to decide copy operations, so we must apply deferred
                # corrections before it runs.
                if deferred_state_corrections_fn:
                    deferred_state_corrections_fn()
                    deferred_state_corrections_fn = None
                mamba_bufs = self._get_mamba_bufs()
                mamba_preprocess_bufs = mamba_bufs.preprocess
                mamba_utils.preprocess_mamba(
                    scheduler_output,
                    self.kv_cache_config,
                    self.cache_config,
                    self.mamba_state_idx,
                    self.input_batch,
                    self.requests,
                    self.compilation_config.static_forward_context,
                    self.model.get_mamba_state_copy_func(),
                    mamba_preprocess_bufs,
                )
                # preprocess_mamba resets num_accepted_tokens_cpu to 1
                # for requests whose state was copied to a new block.
                # Re-sync to GPU so the mamba kernel reads from the
                # correct initial state slot (init_token_idx = 0).
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
                self.num_accepted_tokens.copy_to_gpu(num_reqs)

                if mamba_bufs.postprocess_align is not None:
                    mamba_utils.stage_postprocess_inputs_to_gpu(
                        mamba_bufs.postprocess_align,
                        scheduler_output,
                        self.input_batch.req_ids,
                        num_reqs,
                        self.requests,
                        self.mamba_state_idx,
                    )

            # --- compress deferred corrections + DSA fill (slow path) ---
            if self.use_compress:
                if deferred_state_corrections_fn:
                    deferred_state_corrections_fn()
                    deferred_state_corrections_fn = None
                req_indices = np.repeat(
                    self.arange_np[:num_reqs], num_scheduled_tokens_np
                )
                dsa_positions_np = self._dsa_positions_np_buf[
                    :total_num_scheduled_tokens
                ]
                np.add(
                    self.input_batch.num_computed_tokens_cpu[req_indices],
                    self.query_pos.np[:total_num_scheduled_tokens],
                    out=dsa_positions_np,
                )

            # --- Core input preparation (skip _prepare_inputs and the DSA
            # fill already done above, exactly like the slow path) ---
            cache = self._run_input_preparation(
                scheduler_output,
                precomputed=(
                    logits_indices,
                    spec_decode_metadata,
                    total_num_scheduled_tokens,
                ),
                skip_dsa_fill=True,
            )

        # If the batch became empty after _update_states (num_reqs == 0),
        # _run_input_preparation returns a zeroed placeholder.  Don't cache
        # it — let execute_model fall through to the normal slow path which
        # will return EMPTY_MODEL_RUNNER_OUTPUT.
        if cache["total_num_scheduled_tokens"] == 0:
            self._cloud_prepare_cache = None
            return

        # Carry any leftover deferred corrections in the cache so the cloud
        # fast path can apply them at the same post-launch point as the
        # slow path (execute_model applies it after the batch is launched).
        cache["deferred_state_corrections_fn"] = deferred_state_corrections_fn
        cache["mamba_preprocess_bufs"] = mamba_preprocess_bufs

        # An early-returned batch can leave this cache alive until another
        # request reaches execute_model. Tag it with the exact request order
        # so the cloud fast path cannot consume another batch's attention
        # metadata, logits indices, or prepared token layout.
        cache["req_ids_key"] = tuple(scheduler_output.num_scheduled_tokens)

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

        # segment_e runs in a fresh ForwardContext. Since an edge worker owns
        # both the head and tail ranges, index zero refers to a head MoE, not
        # the first MoE executed by this tail segment.
        old_moe_layer_index = forward_context.moe_layer_index
        if forward_context.all_moe_layers:
            forward_context.moe_layer_index = (
                _get_edge_cloud_moe_start_index(
                    forward_context.all_moe_layers,
                    self.num_layers - self.tail_k,
                )
            )

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
            forward_context.moe_layer_index = old_moe_layer_index

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
                self._layerwise_intermediate = _freeze_intermediate_tensors(
                    hidden_states
                )
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
                self._snapshot_mtp_target_hidden_states(
                    self._layerwise_scheduler_output
                ),
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

    def _fast_path_view_restore_required(self) -> bool:
        """Whether any attention backend in use relies on address-identity
        of its attention metadata under FULL-graph replay.

        Backends split into two camps:

        - Content-channel backends (MLA / FIA / their CP variants) refresh
          graph-visible params through an explicit update flow before every
          replay.  The metadata object only carries content, so feeding the
          frozen clone works unchanged.  They declare this via
          ``updates_graph_params_before_replay = True`` on the impl class.
        - Address-identity backends (DSA, SFA: ``update_graph_params`` is a
          no-op) rely on the metadata tensors BEING views of persistent
          buffers that each step rewrites in place.  The segment_e frozen
          clones break that contract, so the fast path must restore the
          frozen contents back into the views before a FULL-graph replay.

        GDN models are unconditionally excluded: their metadata views
        alias reusable pool slots (``_buffer_slot``) that may have been
        reassigned to an in-flight batch, so writing back is unsafe; they
        keep the pre-existing frozen-clone behavior.

        New backends default to "no channel" (restore enabled) -- safe:
        for a content-channel backend the restore is merely a few
        redundant same-content copies, whereas a missed address-identity
        backend crashes under graph replay.
        """
        cached = getattr(self, "_view_restore_required_cache", None)
        if cached is not None:
            return cached
        required = False
        if not self._has_gdn:
            for groups in self.attn_groups:
                for group in groups:
                    impl_cls = group.backend.get_impl_cls()
                    if not getattr(
                        impl_cls, "updates_graph_params_before_replay", False
                    ):
                        required = True
                        break
                if required:
                    break
        self._view_restore_required_cache = required
        return required

    def _merge_pending_prev_sampled(
        self,
        sampled_token_ids: torch.Tensor,
        new_map: dict[str, int],
    ) -> tuple[torch.Tensor, dict[str, int]]:
        """Preserve pending sampled tokens from a previous, different batch.

        Upstream, every async sampling step covers the full running batch, so
        replacing ``prev_sampled_token_ids`` / ``prev_req_id_to_index``
        wholesale is safe. In edge-cloud PD separation, sampling only happens
        in tail segments (PL/DL) whose batch is a SUBSET of the running
        requests: a PL for request B can run between the DL that sampled
        request A's token and the DF that must consume it. A wholesale
        replace drops A's pending token; A's placeholder input (-1) is then
        never filled, and A decodes garbage from that step on. Merge instead:
        keep rows for requests that are absent from the current sampling
        batch but still alive.
        """
        prev_buf = self.input_batch.prev_sampled_token_ids
        prev_map = self.input_batch.prev_req_id_to_index
        if prev_buf is None or not prev_map:
            return sampled_token_ids, new_map
        stale = [
            (req_id, idx)
            for req_id, idx in prev_map.items()
            if req_id not in new_map and req_id in self.requests
        ]
        if not stale:
            return sampled_token_ids, new_map
        offset = sampled_token_ids.shape[0]
        combined = sampled_token_ids.new_empty(
            (offset + len(stale), *sampled_token_ids.shape[1:])
        )
        combined[:offset] = sampled_token_ids
        merged_map = dict(new_map)
        for j, (req_id, old_idx) in enumerate(stale):
            combined[offset + j] = prev_buf[old_idx]
            merged_map[req_id] = offset + j
        # logger.warning(
        #     "[EDGE-CLOUD-STASH] preserved %d pending sampled token(s) for "
        #     "reqs %s not in the current sampling batch (batch=%s)",
        #     len(stale),
        #     [r for r, _ in stale],
        #     list(new_map.keys()),
        # )
        return combined, merged_map

    @staticmethod
    def _expected_tail_batch_type(head_bt: BatchType) -> BatchType:
        return {
            BatchType.PREFILL_FIRST: BatchType.PREFILL_LAST,
            BatchType.DECODE_FIRST: BatchType.DECODE_LAST,
        }[head_bt]

    def _filter_stale_tail_batch(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: "IntermediateTensors | None",
        stale_req_ids: list[str],
    ) -> tuple[SchedulerOutput, "IntermediateTensors | None"]:
        """Remove stale (already-finished) requests from a PL/DL tail batch.

        The cloud shipped hidden tensors for the FULL head batch, but the
        stale reqs were popped from ``self.requests`` on the edge during the
        head->tail window. Rewrite ``scheduler_output`` to the alive subset
        and slice every token-major tensor in ``intermediate_tensors`` by the
        per-req token ranges, so segment_e only runs for alive reqs.

        Token layout assumption (verified against the data-plane contract):
        hidden tensors are [total_tokens, ...] with each req contributing
        ``num_scheduled_tokens[req_id]`` tokens in ``scheduler_output`` order.
        """
        stale = set(stale_req_ids)
        num_scheduled = scheduler_output.num_scheduled_tokens
        alive_req_ids = [r for r in num_scheduled if r not in stale]

        # ---- Slice token-major hidden tensors ----
        if intermediate_tensors is not None:
            total_tokens = sum(num_scheduled.values())
            keep_indices: list[int] = []
            offset = 0
            for req_id, n in num_scheduled.items():
                if req_id not in stale:
                    keep_indices.extend(range(offset, offset + n))
                offset += n
            new_tensors: dict[str, torch.Tensor] = {}
            for key, tensor in intermediate_tensors.tensors.items():
                if (isinstance(tensor, torch.Tensor)
                        and tensor.dim() > 0
                        and tensor.shape[0] == total_tokens):
                    index = torch.tensor(
                        keep_indices, dtype=torch.long, device=tensor.device
                    )
                    new_tensors[key] = tensor.index_select(0, index)
                else:
                    # Not token-major (e.g. scalar metadata): keep as-is.
                    new_tensors[key] = tensor
            new_intermediate = IntermediateTensors(new_tensors)
            # AsyncIntermediateTensors.__getattribute__ raises AttributeError
            # for attributes that were never set (kv_connector_output is only
            # attached on non-tail PP paths), so use getattr with a default.
            new_intermediate.kv_connector_output = getattr(
                intermediate_tensors, "kv_connector_output", None
            )
            intermediate_tensors = new_intermediate

        # ---- Rewrite per-req scheduling fields ----
        scheduler_output.num_scheduled_tokens = {
            r: num_scheduled[r] for r in alive_req_ids
        }
        scheduler_output.total_num_scheduled_tokens = sum(
            scheduler_output.num_scheduled_tokens.values()
        )
        if scheduler_output.scheduled_spec_decode_tokens:
            scheduler_output.scheduled_spec_decode_tokens = {
                r: t
                for r, t in scheduler_output.scheduled_spec_decode_tokens.items()
                if r not in stale
            }
        if scheduler_output.scheduled_new_reqs:
            scheduler_output.scheduled_new_reqs = [
                r for r in scheduler_output.scheduled_new_reqs
                if r.req_id not in stale
            ]
        cached = scheduler_output.scheduled_cached_reqs
        if cached is not None and cached.req_ids:
            keep = [
                i for i, r in enumerate(cached.req_ids) if r not in stale
            ]
            if len(keep) != len(cached.req_ids):
                # NOTE: new_token_ids is only populated in non-async PP mode
                # (it is empty under async scheduling), so it is NOT always
                # parallel to req_ids. Filter it by index only when the
                # lengths match; otherwise keep it as-is.
                if len(cached.new_token_ids) == len(cached.req_ids):
                    filtered_new_token_ids = [
                        cached.new_token_ids[i] for i in keep
                    ]
                else:
                    filtered_new_token_ids = cached.new_token_ids
                scheduler_output.scheduled_cached_reqs = CachedRequestData(
                    req_ids=[cached.req_ids[i] for i in keep],
                    resumed_req_ids={
                        r for r in cached.resumed_req_ids if r not in stale
                    },
                    new_token_ids=filtered_new_token_ids,
                    all_token_ids={
                        r: t
                        for r, t in cached.all_token_ids.items()
                        if r not in stale
                    },
                    new_block_ids=[cached.new_block_ids[i] for i in keep],
                    num_computed_tokens=[
                        cached.num_computed_tokens[i] for i in keep
                    ],
                    num_output_tokens=[
                        cached.num_output_tokens[i] for i in keep
                    ],
                )
        return scheduler_output, intermediate_tensors

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
                global_start = layer_slice_info.start_layer + self.head_k
                forward_context.moe_layer_index = (
                    _get_edge_cloud_moe_start_index(
                        forward_context.all_moe_layers,
                        global_start,
                    )
                )
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
        # In EAGLE3 edge-cloud mode the target model's cloud segment also returns
        # the auxiliary hidden states used by the draft model. Keep them on the
        # cloud side and remove them from the tensors sent back to the edge.
        # Build a new IntermediateTensors instead of mutating the returned one,
        # because the returned object may be reused by ACL graph replay.
        if "aux_hidden_states" in hidden_states.tensors:
            aux_hidden_states = hidden_states.tensors["aux_hidden_states"]
            self._eagle3_cloud_aux_hidden_states = aux_hidden_states
            scheduler_output = self._last_scheduler_output
            if (
                self._uses_scheduled_edge_cloud_draft()
                and self.speculative_config.method == "eagle3"
                and scheduler_output is not None
                and scheduler_output.head_token is not None
            ):
                self._eagle3_cloud_aux_hidden_states_by_task[
                    scheduler_output.head_token
                ] = _freeze_scheduled_state(aux_hidden_states)
            return IntermediateTensors(
                {
                    k: v
                    for k, v in hidden_states.tensors.items()
                    if k != "aux_hidden_states"
                }
            )
        self._eagle3_cloud_aux_hidden_states = None
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
                    if not isinstance(v, torch.Tensor) or k == "mrope_positions":
                        continue
                    copy_len = num_tokens
                    if k not in self.intermediate_tensors.tensors:
                        base_tensor = self.intermediate_tensors["hidden_states"]
                        self.intermediate_tensors[k] = v.new_empty(
                            (base_tensor.shape[0], *v.shape[1:])
                        )
                    dst = self.intermediate_tensors[k][:copy_len]
                    # Senders transmit only real tokens (edge may send fewer
                    # than the cloud's padded num_tokens, e.g. a small chunk
                    # under chunk_prefill_prior padded up to a cudagraph size).
                    # Copy only the rows actually received and zero-fill the
                    # graph padding tail -- mirrors the non-embedding_only
                    # branch below.  Without this min(), v[:copy_len] indexes
                    # past v's real rows (aclnnInplaceCopy error 161002).
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
                    # mrope_positions is an edge-cloud side-channel tensor that
                    # lives outside the model's layer-to-layer intermediate
                    # buffer (self.intermediate_tensors, declared by
                    # make_empty_intermediate_tensors as hidden/residual only).
                    # It is materialized into self.mrope_positions.gpu directly
                    # in execute_model before _preprocess; skip it here so the
                    # copy-into-local-buffer loop does not KeyError on it.
                    if k == "mrope_positions":
                        continue
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
        # vllm renamed sync_and_slice to sync_and_gather.
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
                allow_dp_padding=((cudagraph_mode != CUDAGraphMode.NONE)
                                  or enable_sp(self.vllm_config)
                                  or oproj_tp_enable()
                                  or embedding_tp_enable()),
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
    
    def _should_save_for_attn_metadata(self) -> bool:
        return False

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

            fixed_decode_seq_lens_cpu = None
            if self.use_async_spec_decode:
                fixed_decode_seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs].numpy()

            assert num_reqs_padded is not None
            return self.pcp_manager.generate_pcp_metadata(
                num_tokens,
                self.query_lens,
                self.input_batch,
                num_scheduled_tokens_np,
                block_table_tensor,
                num_reqs_padded,
                num_reqs,
                fixed_decode_seq_lens_cpu,
            )

        def _get_block_table_and_slot_mapping(
            kv_cache_gid: int,
        ):
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
                blk_table_tensor = blk_table.get_device_tensor()[:num_reqs_padded]
                # Fill unused with -1. Needed for reshape_and_cache in full cuda
                # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
                if self.pcp_size == 1:
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
                if self.routed_experts_initialized:
                    # snapshot slot_mapping into a private device
                    # buffer so the next ``_prepare_inputs`` does not
                    # overwrite it while D2H is still pending.
                    n = slot_mapping.shape[0]
                    self.routed_experts_slot_mapping_device[:n].copy_(
                        slot_mapping
                    )
            return blk_table_tensor, slot_mapping

        block_table_gid_0, slot_mapping_gid_0 = _get_block_table_and_slot_mapping(0)
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

        if self._should_save_for_attn_metadata():
            self.cm_base = cm_base
        
        def _build_attn_group_metadata(
            kv_cache_gid: int,
            attn_gid: int,
            common_attn_metadata: CommonAttentionMetadata,
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
                if for_cudagraph_capture:
                    prefill_ratio_to_sas_metadata = {}
                    decode_ratio_to_sas_metadata = {}
                    common_ratio_to_sas_metadata = {}
                extra_attn_metadata_args = dict(
                    num_reqs_actual=num_reqs,
                    prefill_ratio_to_sas_metadata=prefill_ratio_to_sas_metadata,
                    decode_ratio_to_sas_metadata=decode_ratio_to_sas_metadata,
                    common_ratio_to_sas_metadata=common_ratio_to_sas_metadata,
                    block_size=attn_group.kv_cache_spec.block_size,
                )
            if self._should_save_for_attn_metadata():
                self.per_gid_extra[(kv_cache_gid, attn_gid)] = (
                    cascade_attn_prefix_len, extra_attn_metadata_args)

            # add kvcomp_metadata into common_attn_metadata
            if (for_cudagraph_capture
                    and not isinstance(builder, (
                        AscendDSAMetadataBuilder,
                        AscendDSACPMetadataBuilder,
                        AscendSFADCPMetadataBuilder,
                    ))):
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
        
        if self._should_save_for_attn_metadata():
            self.per_gid_cm = [{} for _ in self.kv_cache_config.kv_cache_groups]
            self.per_gid_extra = {}

        def _save(name: str):
            if self._should_save_for_attn_metadata():
                self.per_gid_cm[kv_cache_gid][name] = getattr(cm, name)
        
        for kv_cache_gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            cm = copy(cm_base)  # shallow copy
            # Basically only the encoder seq_lens, block_table and slot_mapping change
            # for each kv_cache_group.
            cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
                num_scheduled_tokens or {},
                kv_cache_group.kv_cache_spec,
                num_reqs_padded,
            )

            _save('encoder_seq_lens')
            _save('encoder_seq_lens_cpu')

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
                    _save('query_start_loc_cpu')
                    _save('query_start_loc')

            if kv_cache_gid > 0:
                cm.block_table_tensor, cm.slot_mapping = _get_block_table_and_slot_mapping(
                    kv_cache_gid
                )
                _save('block_table_tensor')
                _save('slot_mapping')
            if self.speculative_config and isinstance(self.drafter, AscendStep3p5MTPProposer):
                # step3p5 MTP draft layers span multiple KV cache groups; capture
                # each group's block table / slot mapping so the proposer can
                # build per-step attention metadata for the active MTP layer.
                self.drafter.set_per_group_attn_metadata(
                    kv_cache_gid, cm.block_table_tensor, cm.slot_mapping)
            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(self.drafter, AscendEagleProposer | AscendDraftModelProposer | AscendDflashProposer):
                    if (
                        self.drafter.attn_layer_names
                        and self.drafter.attn_layer_names[0]
                        in kv_cache_group.layer_names
                    ):
                        spec_decode_common_attn_metadata = cm
                    elif (
                        self._edge_cloud_enabled
                        and self.edge_cloud_cfg.role == "edge"
                    ):
                        # In edge-cloud head_tail mode, the edge side does not
                        # host the draft model's attention layers (they run on
                        # the cloud), so the draft attention layer is not
                        # present in any local kv_cache_group.  The edge still
                        # needs a common attention metadata to prepare draft
                        # inputs (positions, seq_lens, block_table) for the
                        # edge-cloud draft round-trip.  Fall back to the target
                        # model's metadata.
                        spec_decode_common_attn_metadata = cm
                else:
                    spec_decode_common_attn_metadata = cm
            if self.enable_hamming_sparse is True:
                from vllm_ascend.attention.kvcomp_attn.attention_utils import build_kvcomp_metadata
                build_kvcomp_metadata(self.kvcomp_meta_data, cm)
            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                _build_attn_group_metadata(
                    kv_cache_gid,
                    attn_gid,
                    cm,
                    prefill_ratio_to_sas_metadata,
                    decode_ratio_to_sas_metadata,
                    common_ratio_to_sas_metadata,
                )
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # only support eager mode and piecewise graph now
        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode.valid_runtime_modes()
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
                self.input_batch.num_computed_tokens_cpu,
                self.input_batch.num_prompt_tokens,
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
        
        if self.dynamic_eplb:
            self.update_eplb_heat_collection_status(num_tokens_padded)
        
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
            copy_snapshot_to_gpu(self.query_start_loc)
            if self._has_gdn:
                self.gdn_query_start_loc.np[1 : num_reqs_padded + 1] = cum_num_tokens
                copy_snapshot_to_gpu(self.gdn_query_start_loc)

            if not profile_cpp:
                num_reqs_padded = self._pad_query_start_loc_for_fia(
                    self.query_start_loc,
                    num_tokens_padded,
                    num_reqs_padded,
                    num_reqs,
                    cudagraph_runtime_mode,
                    batch_desc.num_reqs,
                )

            # Dummy graph runs do not go through _prepare_inputs(), but GDN/Mamba
            # metadata reads block_table[:num_reqs_padded] below. Sync padded
            # rows as well so device-side metadata does not see stale block ids.
            self.input_batch.block_table.commit_block_table(num_reqs_padded)

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
            if not is_graph_capturing:
                for kv_cache_gid in range(len(self.kv_cache_config.kv_cache_groups)):
                    blk_table = self.input_batch.block_table[kv_cache_gid]
                    blk_table.slot_mapping.gpu.fill_(-1)

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
                    # Cloud 端：需要中间张量
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
                        self.intermediate_tensors = (
                            self._make_empty_edge_cloud_intermediate_tensors(
                                batch_size=max_actual_tokens,
                            )
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
                    self.intermediate_tensors = (
                        self._make_empty_edge_cloud_intermediate_tensors(
                            batch_size=max_actual_tokens,
                        )
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
                eplb_heat_collection_status=self.eplb_heat_collection_status if self.dynamic_eplb else False,
            ):
                outputs = self._model_forward(
                    num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds
                )
            if self.use_aux_hidden_state_outputs:
                if isinstance(outputs, IntermediateTensors):
                    hidden_states = outputs["hidden_states"]
                else:
                    hidden_states, _ = outputs
            elif isinstance(outputs, IntermediateTensors):
                hidden_states = outputs["hidden_states"]
            else:
                hidden_states = outputs
            dummy_compute_logits(hidden_states)

            is_scheduled_edge_cloud_draft = (
                self._edge_cloud_enabled
                and self.speculative_config is not None
                and self.speculative_config.method in ("mtp", "eagle3")
            )
            skip_eager_edge_cloud_drafter = (
                self._edge_cloud_target_capture_in_progress
                and is_scheduled_edge_cloud_draft
                and self.drafter is not None
                and not self._edge_cloud_drafter_uses_graph()
            )
            if (
                self.drafter
                and not profile_cpp
                and not skip_eager_edge_cloud_drafter
            ):
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

            # ========== Edge 设备特殊处理：Edge 首阶段需要执行最后一层 ==========
            if is_edge_device():
                # 断言：边设备输出必须是 IntermediateTensors 类型
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
                if isinstance(outputs, IntermediateTensors):
                    hidden_states = outputs["hidden_states"]
                elif isinstance(outputs, (tuple, list)):
                    hidden_states, _ = outputs
                else:
                    hidden_states = outputs
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
        uses_local_mc2_warmup = select_moe_comm_method(
            mc2_tokens_capacity, self.vllm_config
        ) in {MoECommType.MC2, MoECommType.FUSED_MC2}
        needs_aligned_edge_cloud_draft_warmup = (
            self._edge_cloud_enabled
            and self._uses_scheduled_edge_cloud_draft()
            and is_moe_model(self.vllm_config)
            and self.vllm_config.parallel_config.enable_expert_parallel
        )
        if self.max_num_tokens > mc2_tokens_capacity and (
            uses_local_mc2_warmup
            or needs_aligned_edge_cloud_draft_warmup
        ):
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

    def update_eplb_heat_collection_status(self, num_tokens_padded: int):
        if self.eplb_heat_collection_stage == "prefill":
            # collect eplb heat for prefill requests.
            self.eplb_heat_collection_status = num_tokens_padded > self.eplb_pd_thresholds
        elif self.eplb_heat_collection_stage == "decode":
            # collect eplb heat for decode requests.
            self.eplb_heat_collection_status = num_tokens_padded <= self.eplb_pd_thresholds
        else:
            # collect eplb heat for all requests.
            self.eplb_heat_collection_status =  True

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

        load_model_start_time = time.perf_counter()
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
            if self.drafter:
                logger.info("Loading drafter model...")
                if self.vllm_config.quant_config is not None:
                    patch_load_weights(self.vllm_config)
                with get_tp_context(self.drafter):
                    self.drafter.load_model(self.model)

            pp_group = get_pp_group()
            should_configure_aux_hidden_states = (
                self.use_aux_hidden_state_outputs
                if pp_group.world_size == 1
                else self._eagle3_uses_aux_hidden_state()
            )
            if should_configure_aux_hidden_states:
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

                if pp_group.world_size > 1:
                    inner_model = self.model
                    if hasattr(inner_model, "get_language_model"):
                        inner_model = inner_model.get_language_model()
                    elif hasattr(inner_model, "language_model"):
                        language_model = inner_model.language_model
                        inner_model = (
                            language_model()
                            if callable(language_model)
                            else language_model
                        )
                    if hasattr(inner_model, "model"):
                        inner_model = inner_model.model
                    from vllm_ascend.patch.worker.patch_eagle3_pp_aux import (
                        patch_eagle3_pp_aux_propagation,
                    )

                    if patch_eagle3_pp_aux_propagation(inner_model):
                        self.model.make_empty_intermediate_tensors = (
                            inner_model.make_empty_intermediate_tensors
                        )

            if self.lora_config:
                self.model = self.load_lora_model(self.model, self.vllm_config, self.device)
        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB", m.consumed_memory / float(2**30))

        from vllm.model_executor.offloader.base import get_offloader
        get_offloader().post_init()

        mm_config = self.model_config.multimodal_config
        self.is_multimodal_pruning_enabled = (
            supports_multimodal_pruning(self.get_model())
            and mm_config is not None
            and mm_config.is_multimodal_pruning_enabled()
        ) # type: bool
        
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

        load_model_total_time = time.perf_counter() - load_model_start_time
        logger.info(
            "Model runner load_model total time: %.2f seconds",
            load_model_total_time,
        )

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
            # Initialize cudagraph dispatcher keys + ACL graph params ONLY for the
            # MTP edge-cloud path. The MTP drafter segments (_edge_cloud_mtp_segments)
            # rely on graph_params being set here; without it ACL graph capture/replay
            # hangs. (Mirrors the `method == "mtp"` guard in _check_and_update_cudagraph_mode.)
            #
            # DO NOT run this for the non-MTP embedding_only edge. Passing empty
            # attention backends leaves min_cg_support at ALWAYS, which initializes
            # the dispatcher keys (keys_initialized=True) and makes dispatch() return
            # FULL instead of NONE. That flips the edge decode tail (segment_e) from
            # eager into ACL-graph capture/replay and adds a per-step
            # update_full_graph_params sync, costing ~2% throughput (94 -> 92 token/s)
            # and hurting the edge/cloud overlap under --async-scheduling. The edge
            # tail has no attention layers, so eager is both correct and faster here
            # -- this is exactly the pre-MTP behavior.
            if (
                self.speculative_config is not None
                and self.speculative_config.method == "mtp"
            ):
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
        # In edge-cloud head_tail mode (首一尾一), a kv cache group whose
        # layers all live on the cloud produces an EMPTY attn_group on the
        # edge (and vice versa); skip empty groups instead of indexing [0].
        self.need_accepted_tokens = any(
            [
                isinstance(attn_group[0].kv_cache_spec, MambaSpec)
                for attn_group in self.attn_groups
                if attn_group
            ]
        )

        self.may_reinitialize_input_batch(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)
        # TODO: refactor the logic of attention
        if (
            self.speculative_config
            and self.drafter is not None
            and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
            )
        ):
            assert isinstance(self.drafter, AscendEagleProposer | AscendDflashProposer | AscendDraftModelProposer)
            skip_edge_drafter_attn_init = (
                self._edge_cloud_enabled
                and self.edge_cloud_cfg.role == "edge"
                and self.speculative_config.method in ("mtp", "eagle3")
            )
            if skip_edge_drafter_attn_init:
                # All draft decoder layers run on the cloud. Their stale
                # static-forward-context entries have already been removed on
                # the edge, so the edge KV cache config intentionally contains
                # no draft attention layers.
                self.drafter.draft_attn_groups = []
                logger.info(
                    "[EdgeCloud] Edge skipped %s drafter attention backend "
                    "initialization.",
                    self.speculative_config.method,
                )
            else:
                block_size = (
                    self.kernel_block_sizes[0]
                    if isinstance(self.kernel_block_sizes, list)
                    else self.kernel_block_sizes
                )
                self.drafter.initialize_attn_backend(kv_cache_config, block_size)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    def _bind_routed_experts_capturer(self, capturer=None) -> None:
        if vllm_version_is("0.23.0"):
            # Upstream binds via ``module.router.set_capture_fn(...)`` on
            # FusedMoE layers whose router is a ``BaseRouter``. Ascend's
            # ``select_experts`` does not go through ``BaseRouter``, so the
            # upstream hook never fires. Instead, stash the capturer as a
            # plain attribute on every FusedMoE layer; ``apply()`` reads it
            # back on the hot path.
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE

            for module in self.compilation_config.static_forward_context.values():
                if isinstance(module, FusedMoE):
                    module._ascend_routed_experts_capturer = capturer
        else:
            # test_qwen3_moe_routing_replay
            from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

            for module in self.compilation_config.static_forward_context.values():
                if isinstance(module, AscendMoERunner):
                    module._ascend_routed_experts_capturer = capturer
                    module.routed_experts._ascend_routed_experts_capturer = capturer

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
            self.kv_cache_names: list[str] = []
            for layer_name in sorted(
                    kv_caches,
                    key=lambda name: (extract_dsv4_layer_index(
                        self.model_config.hf_text_config, name), name)):
                self.kv_caches.append(kv_caches[layer_name])
                self.kv_cache_names.append(layer_name)
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
        if isinstance(kv_cache_spec, AscendMLAAttentionSpec):
            attn_layers = get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,
                [layer_name],
            )
            attn_layer = attn_layers[layer_name]
            if isinstance(attn_layer, MLAAttention):
                # DeepSeek MLA: K=kv_lora_rank, V=qk_rope_head_dim
                return attn_layer.kv_lora_rank, attn_layer.qk_rope_head_dim
            # CacheOnlyAttentionLayer uses AscendMLAAttentionSpec but isn't MLAAttention
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

    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    def _allocate_int8_cache_tensor(
        self,
        numel: int,
        alignment: int,
    ) -> torch.Tensor:
        """Allocate an int8 raw cache tensor.

        When KV transfer is enabled, the returned tensor's data_ptr is aligned
        to `alignment`. This keeps the original Mooncake/ADXL alignment behavior.
        """
        if numel <= 0:
            raise ValueError(f"Invalid cache tensor size: {numel}")

        if self.vllm_config.kv_transfer_config is None:
            return torch.zeros(numel, dtype=torch.int8, device=self.device)

        raw_tensor = torch.zeros(
            numel + alignment,
            dtype=torch.int8,
            device=self.device,
        )
        return self._align_memory(raw_tensor, alignment)[:numel]

    def _allocate_sparse_c8_indexer_tensors(
        self,
        dsa_k_tensor_size: int,
        dsa_k_scale_tensor_size: int,
        alignment: int,
        scale_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate dsa_k and dsa_k_scale from one aligned int8 raw allocation.

        Both returned tensors are logical views into the same underlying storage:

            sparse_c8_raw
              ├── dsa_k_tensor        int8 raw bytes
              └── dsa_k_scale_tensor  scale dtype raw bytes stored as int8 view

        `dsa_k_scale_tensor` is still returned as int8 raw storage. Later reshape
        code should continue to use:

            raw_dsa_k_scale_tensor.view(scale_dtype).view(scale_shape)

        This reduces HCCL/Mooncake registration count because register_buffer
        can merge these two views into one registered memory range.
        """
        if dsa_k_tensor_size <= 0:
            raise ValueError(
                f"Invalid dsa_k_tensor_size: {dsa_k_tensor_size}"
            )
        if dsa_k_scale_tensor_size <= 0:
            raise ValueError(
                f"Invalid dsa_k_scale_tensor_size: {dsa_k_scale_tensor_size}"
            )

        scale_dtype_size = torch.empty((), dtype=scale_dtype).element_size()

        # Ensure the scale view starts at an address aligned for scale_dtype.
        scale_offset = self._align_up(dsa_k_tensor_size, scale_dtype_size)
        total_raw_size = scale_offset + dsa_k_scale_tensor_size

        sparse_c8_raw_tensor = self._allocate_int8_cache_tensor(
            total_raw_size,
            alignment,
        )

        dsa_k_tensor = sparse_c8_raw_tensor[:dsa_k_tensor_size]
        dsa_k_scale_tensor = sparse_c8_raw_tensor[
            scale_offset : scale_offset + dsa_k_scale_tensor_size
        ]

        assert dsa_k_tensor.is_contiguous()
        assert dsa_k_scale_tensor.is_contiguous()
        assert dsa_k_scale_tensor.data_ptr() % scale_dtype_size == 0
        assert dsa_k_scale_tensor.numel() % scale_dtype_size == 0

        return dsa_k_tensor, dsa_k_scale_tensor

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
                    or is_hidden_state_cache_spec(layer_kv_cache_spec.get(layer_name))
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

                    dsa_k_tensor_split_factor = None
                    if self.use_sparse:
                        # for deepseek v3.2, we split the kv cache according to the corresponding ratio
                        kv_cache_spec = layer_kv_cache_spec[layer_name]
                        current_sparse_c8 = kv_cache_spec_uses_sparse_c8(kv_cache_spec)
                        assert isinstance(kv_cache_spec, AscendMLAAttentionSpec)
                        assert kv_cache_spec.sparse_head_dim is not None
                        has_indexer_cache = sparse_kv_cache_has_indexer(kv_cache_spec)

                        if current_sparse_c8:
                            assert kv_cache_tensor.size % kv_cache_spec.page_size_bytes == 0
                            num_blocks = kv_cache_tensor.size // kv_cache_spec.page_size_bytes
                            num_heads = kv_cache_spec.block_size * kv_cache_spec.num_kv_heads
                            packed_kv_head_dim, _, index_head_dim = kv_cache_spec.sparse_head_dim
                            k_tensor_split_factor = 1.0
                            k_tensor_size = (
                                num_blocks
                                * num_heads
                                * packed_kv_head_dim
                                * get_dtype_size(kv_cache_spec.c8_k_cache_dtype)
                            )
                            v_tensor_size = None
                            if has_indexer_cache:
                                dsa_k_tensor_size = (
                                    num_blocks
                                    * num_heads
                                    * index_head_dim
                                    * kv_cache_spec.sfa_dcp_replicated_indexer_size
                                    * get_dtype_size(kv_cache_spec.c8_k_cache_dtype)
                                )
                                dsa_k_scale_tensor_size = (
                                    num_blocks
                                    * num_heads
                                    * kv_cache_spec.sfa_dcp_replicated_indexer_size
                                    * get_dtype_size(kv_cache_spec.c8_k_scale_cache_dtype)
                                )
                            else:
                                dsa_k_tensor_size = None
                                dsa_k_scale_tensor_size = None
                            dsa_k_tensor_split_factor = None
                        elif has_indexer_cache:
                            sparse_kv_cache_ratio = kv_cache_spec.sparse_kv_cache_ratio
                            k_tensor_split_factor = sparse_kv_cache_ratio[0]
                            v_tensor_split_factor = sparse_kv_cache_ratio[1]
                            dsa_k_tensor_split_factor = sparse_kv_cache_ratio[2]
                        else:
                            k_dim, v_dim, _ = kv_cache_spec.sparse_head_dim
                            k_tensor_split_factor, v_tensor_split_factor = calc_split_factor([k_dim, v_dim])
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

                    if not (self.use_sparse and current_sparse_c8):
                        k_tensor_size = int(kv_cache_tensor.size // k_tensor_split_factor)
                        v_tensor_size = (
                            int(kv_cache_tensor.size // v_tensor_split_factor)
                            if v_tensor_split_factor is not None
                            else None
                        )
                        dsa_k_tensor_size = None
                        dsa_k_scale_tensor_size = None
                    #### for deepseek sparse attention
                    if self.use_sparse and has_indexer_cache and not current_sparse_c8:
                        assert dsa_k_tensor_split_factor is not None
                        dsa_k_tensor_size = int(kv_cache_tensor.size // dsa_k_tensor_split_factor)
                    # Allocate raw int8 tensors. Even bf16/fp16 KV cache entries
                    # are allocated as int8 raw bytes first and then viewed as
                    # the target dtype in _reshape_kv_cache_tensors.
                    dsa_k_tensor = None
                    dsa_k_scale_tensor = None
                    v_tensor = None
                    k_tensor = self._allocate_int8_cache_tensor(
                        k_tensor_size,
                        alignment,
                    )
                    if v_tensor_size is not None:
                        v_tensor = self._allocate_int8_cache_tensor(
                            v_tensor_size,
                            alignment,
                        )

                    if self.use_sparse and dsa_k_tensor_size is not None:
                        if current_sparse_c8:
                            assert dsa_k_scale_tensor_size is not None

                            (
                                dsa_k_tensor,
                                dsa_k_scale_tensor,
                            ) = self._allocate_sparse_c8_indexer_tensors(
                                dsa_k_tensor_size=dsa_k_tensor_size,
                                dsa_k_scale_tensor_size=dsa_k_scale_tensor_size,
                                alignment=alignment,
                                scale_dtype=current_kv_cache_spec.c8_k_scale_cache_dtype,
                            )
                        else:
                            dsa_k_tensor = self._allocate_int8_cache_tensor(
                                dsa_k_tensor_size,
                                alignment,
                            )

                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the attn kvcache for all shared layers
                        if "attn" in layer_name_inner and "linear_attn" not in layer_name_inner:
                            if self.use_sparse:
                                if current_sparse_c8:
                                    if has_indexer_cache:
                                        # Sparse C8 with indexer: packed KV, indexer K, and indexer K scale.
                                        kv_cache_raw_tensors[layer_name_inner] = (
                                            k_tensor, dsa_k_tensor, dsa_k_scale_tensor
                                        )
                                    else:
                                        # Sparse C8 without indexer: packed KV only.
                                        kv_cache_raw_tensors[layer_name_inner] = (k_tensor,)
                                else:
                                    if has_indexer_cache:
                                        # Sparse non-C8 with indexer: regular K/V plus indexer K.
                                        kv_cache_raw_tensors[layer_name_inner] = (
                                            k_tensor, v_tensor, dsa_k_tensor
                                        )
                                    else:
                                        # Sparse non-C8 without indexer: regular K/V only.
                                        kv_cache_raw_tensors[layer_name_inner] = (k_tensor, v_tensor)
                            else:
                                # Dense attention: regular K/V only.
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
        overlap_full_kv_cache: bool = False,
    ):
        reshaped_kv_tensors = []
        base_storage_offset_bytes = raw_tensor.storage_offset()
        storage_offset_bytes = base_storage_offset_bytes
        for idx, (shape, dtype) in enumerate(zip(kv_cache_shape_list, kv_cache_dtype_list)):
            if overlap_full_kv_cache and idx == 2:
                storage_offset_bytes = base_storage_offset_bytes
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
                if self.use_compress and isinstance(current_kv_cache_spec,
                                                    (AscendMLAAttentionSpec, AscendSlidingWindowMLASpec)):
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
                    overlap_full_kv_cache = False

                    if hasattr(current_kv_cache_spec, "scale_dim") and current_kv_cache_spec.scale_dim != 0:
                        indexer_k_shape = kv_cache_shape
                        indexer_scale_shape = self.attn_backend.get_kv_cache_shape(
                                                num_blocks, current_kv_cache_spec.block_size,
                                                current_kv_cache_spec.num_kv_heads,
                                                current_kv_cache_spec.scale_dim
                                                )
                        if get_ascend_device_type() in {AscendDeviceType.A5}:
                            indexer_full_shape = self.attn_backend.get_kv_cache_shape(
                                num_blocks, current_kv_cache_spec.block_size,
                                current_kv_cache_spec.num_kv_heads,
                                current_kv_cache_spec.head_size
                                + current_kv_cache_spec.scale_dim
                                * get_dtype_size(current_kv_cache_spec.scale_dtype))
                            kv_cache_shape_list = [
                                indexer_k_shape, indexer_scale_shape, indexer_full_shape
                            ]
                            kv_cache_dtype_list = [
                                current_kv_cache_spec.dtype,
                                current_kv_cache_spec.scale_dtype,
                                current_kv_cache_spec.dtype,
                            ]
                            overlap_full_kv_cache = True
                        else:
                            kv_cache_shape_list = [indexer_k_shape, indexer_scale_shape]
                            kv_cache_dtype_list = [
                                current_kv_cache_spec.dtype, current_kv_cache_spec.scale_dtype
                            ]
                            overlap_full_kv_cache = False

                    kv_cache = self._adjust_kv_layout(kv_tensor,
                                           kv_cache_shape_list,
                                           kv_cache_dtype_list,
                                           current_kv_cache_spec.page_size_bytes,
                                           overlap_full_kv_cache=overlap_full_kv_cache,
                                           )

                    kv_caches[layer_name] = kv_cache
                elif isinstance(current_kv_cache_spec, AttentionSpec):
                    # cache_only_layers (extract_hidden_states) are allocated
                    # as a single tensor by the branch at the top of
                    # _allocate_kv_cache_tensors; route them to the dedicated
                    current_sparse_c8 = kv_cache_spec_uses_sparse_c8(current_kv_cache_spec)
                    has_indexer_cache = sparse_kv_cache_has_indexer(current_kv_cache_spec)
                    raw_dsa_k_tensor = None
                    raw_dsa_k_scale_tensor = None
                    if self.use_sparse and has_indexer_cache and "cache_only_layers" not in layer_name:
                        assert isinstance(current_kv_cache_spec, AscendMLAAttentionSpec)
                        assert current_kv_cache_spec.sparse_head_dim is not None
                        if current_sparse_c8:
                            raw_k_tensor, raw_dsa_k_tensor, raw_dsa_k_scale_tensor = (
                                kv_cache_raw_tensors[layer_name]  # type: ignore
                            )
                            assert raw_dsa_k_tensor is not None
                            assert raw_dsa_k_scale_tensor is not None
                            sum_page_size_bytes = (
                                raw_k_tensor.numel()
                                + raw_dsa_k_tensor.numel()
                                + raw_dsa_k_scale_tensor.numel()
                            )
                        else:
                            raw_k_tensor, raw_v_tensor, raw_dsa_k_tensor = kv_cache_raw_tensors[  # type: ignore
                                layer_name]
                            assert raw_dsa_k_tensor is not None
                            sum_page_size_bytes = raw_k_tensor.numel() + raw_v_tensor.numel() + raw_dsa_k_tensor.numel()
                    elif self.use_sparse and "cache_only_layers" not in layer_name:
                        assert isinstance(current_kv_cache_spec, AscendMLAAttentionSpec)
                        assert current_kv_cache_spec.sparse_head_dim is not None
                        if current_sparse_c8:
                            (raw_k_tensor,) = kv_cache_raw_tensors[layer_name]  # type: ignore
                            sum_page_size_bytes = raw_k_tensor.numel()
                        else:
                            raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[layer_name]  # type: ignore
                            sum_page_size_bytes = raw_k_tensor.numel() + raw_v_tensor.numel()
                    elif (
                        self.use_hybrid_blocks
                        and self.hybrid_with_attn_and_mamba
                        and "cache_only_layers" not in layer_name
                        and not is_hidden_state_cache_spec(current_kv_cache_spec)
                    ):
                        # Currently, we ensure that the same kvcache format is used even if there
                        # is no shared layer, such as the full attention mtp layer of qwen3.5, etc.
                        raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[layer_name], kv_cache_raw_tensors[layer_name]
                        sum_page_size_bytes = raw_k_tensor.numel()
                    elif (
                        "cache_only_layers" in layer_name
                        or is_hidden_state_cache_spec(current_kv_cache_spec)
                    ):
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
                        raw_tensor = raw_tensor.view(current_kv_cache_spec.dtype)
                        page_size_padded = getattr(
                            current_kv_cache_spec, "page_size_padded", None
                        )
                        if page_size_padded is not None:
                            # The cache-only page is aligned to the hybrid common
                            # page, so each block has trailing padding. Stride the
                            # block dim (dim 0) by the full padded page to skip it
                            # (cf. upstream GPUModelRunner page_size_padded view).
                            dtype_size = get_dtype_size(current_kv_cache_spec.dtype)
                            page_stride = page_size_padded // dtype_size
                            strides = [1] * len(kv_cache_shape)
                            for dim_idx in range(len(kv_cache_shape) - 2, -1, -1):
                                strides[dim_idx] = strides[dim_idx + 1] * kv_cache_shape[dim_idx + 1]
                            strides[0] = page_stride
                            k_cache = torch.as_strided(
                                raw_tensor, size=kv_cache_shape, stride=tuple(strides)
                            )
                        else:
                            k_cache = raw_tensor.view(kv_cache_shape)
                        kv_caches[layer_name] = k_cache
                        continue  # Skip the rest of the AttentionSpec handling
                    else:
                        raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[  # type: ignore
                            layer_name
                        ]
                        sum_page_size_bytes = raw_k_tensor.numel() + raw_v_tensor.numel()
                    assert raw_k_tensor is not None
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
                            if not isinstance(current_kv_cache_spec, AscendMLAAttentionSpec):
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
                    if not isinstance(current_kv_cache_spec, AscendMLAAttentionSpec):
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
                        if self.use_sparse and current_sparse_c8:
                            assert current_kv_cache_spec.sparse_head_dim is not None
                            k_shape = (
                                mla_num_blocks,
                                mla_block_size,
                                num_kv_heads,
                                current_kv_cache_spec.sparse_head_dim[0],
                            )
                            v_dim = 0
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

                    if self.use_sparse and current_sparse_c8:
                        k_cache_dtype = self.c8_k_cache_dtype

                    k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape)
                    if self.use_sparse and current_sparse_c8:
                        v_cache = None
                    else:
                        v_cache = raw_v_tensor.view(v_cache_dtype).view(v_shape)

                    if self.use_sparse and has_indexer_cache:
                        assert raw_dsa_k_tensor is not None
                        dsa_k_cache_shape = (
                            num_blocks * current_kv_cache_spec.sfa_dcp_replicated_indexer_size,
                            current_kv_cache_spec.block_size,
                            current_kv_cache_spec.num_kv_heads,
                            self.model_config.hf_text_config.index_head_dim,
                        )
                        if current_sparse_c8:
                            # dsa_k
                            dsa_k_cache = raw_dsa_k_tensor.view(self.c8_k_cache_dtype).view(dsa_k_cache_shape)
                            # dsa_k_scale
                            dsa_k_scale_cache_shape = (
                                num_blocks * current_kv_cache_spec.sfa_dcp_replicated_indexer_size,
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
                            if get_ascend_device_type() == AscendDeviceType.A5:
                                kv_caches[layer_name] = (k_cache, dsa_k_cache, dsa_k_scale_cache)
                            elif v_cache is not None:
                                kv_caches[layer_name] = (k_cache, v_cache, dsa_k_cache, dsa_k_scale_cache)
                            else:
                                kv_caches[layer_name] = (k_cache, dsa_k_cache, dsa_k_scale_cache)
                        else:
                            # dsa_k
                            dsa_k_cache = raw_dsa_k_tensor.view(current_kv_cache_spec.dtype).view(dsa_k_cache_shape)
                            kv_caches[layer_name] = (k_cache, v_cache, dsa_k_cache)
                    elif self.use_sparse and current_sparse_c8:
                        kv_caches[layer_name] = (k_cache,)
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
                    start_idx = 0
                    # NOTE(zxr): in order to keep all tensor contiguous, we align ssm and kv block
                    # with same page size, so have to add extra padding block for kv, the overall
                    # layout of hybrid kv_cache on Ascend is:
                    # tensor1: [(kv_padding), conv           , ...]
                    # tensor2: [k           , ssm            , ...]
                    # tensor3: [v           , (mamba_padding), ...]
                    #
                    # [FIX] When THIS tensor is also shared by a full-attention
                    # layer, the ssm region aliases the K region block-for-block
                    # by design: the block allocator assigns each block id to
                    # exactly one purpose (KV or mamba state), which keeps the
                    # aliasing consistent.  That requires ssm block N to map
                    # onto K block N, i.e. the ssm region must start where the
                    # K region starts and the ssm block stride must equal the
                    # attention K block size.  A contiguous ssm view breaks
                    # the mapping whenever the per-block ssm size differs
                    # from the K block size (edge-cloud with cloud TP8: 6
                    # per-rank ssm heads -> 393216B ssm block vs 786432B K
                    # block, so ssm block N lands on K block N/2 and mamba
                    # state writes clobber live KV, producing NaN logits).
                    #
                    # Pure-mamba tensors (not shared with attention) keep the
                    # legacy contiguous carve, so non-sharing configurations
                    # (e.g. non-edge-cloud) are byte-identical to before.
                    # Tensor sizes and num_blocks are untouched — only the
                    # view offsets/strides change, and the padded ssm span is
                    # asserted to fit inside the K region (no KV block is
                    # lost or shrunk).
                    k_block_bytes = 0
                    if self.hybrid_with_attn_and_mamba:
                        for _kct in kv_cache_config.kv_cache_tensors:
                            _names = list(_kct.shared_by)
                            if layer_name not in _names:
                                continue
                            for _name in _names:
                                _spec = layer_kv_cache_spec.get(_name)
                                if (isinstance(_spec, AttentionSpec)
                                        and not isinstance(_spec, AscendMLAAttentionSpec)
                                        and not is_hidden_state_cache_spec(_spec)):
                                    k_block_bytes = (_spec.block_size
                                                     * _spec.num_kv_heads
                                                     * _spec.head_size
                                                     * get_dtype_size(_spec.dtype))
                                    break
                            if k_block_bytes:
                                break
                    # Same offset the attention carve computes for the K
                    # region (raw = conv | ssm≡K | V, with K == V here):
                    # conv_block_padding_size = numel - 2 * K_span.
                    ssm_offset = (raw_tensor.numel()
                                  - 2 * num_blocks * k_block_bytes
                                  ) if k_block_bytes else None
                    for shape, dtype in zip(current_kv_cache_spec.shapes, current_kv_cache_spec.dtypes):
                        # normally, there is conv state and ssm state in this loop. And there is only
                        # a conv state in some special models.
                        target_shape = (num_blocks, *shape)
                        dtype_size = get_dtype_size(dtype)
                        block_bytes = math.prod(shape) * dtype_size
                        if ssm_offset is not None and len(shape) == 3:
                            # ssm state: must alias the K region 1:1 by block.
                            if block_bytes > k_block_bytes:
                                raise RuntimeError(
                                    f"[hybrid kv layout] {layer_name}: ssm "
                                    f"block ({block_bytes}B) is larger than "
                                    f"the attention K block ({k_block_bytes}B); "
                                    f"the shared attn/mamba tensor cannot "
                                    f"alias ssm onto the K region 1:1. "
                                    f"Disable hybrid block sharing for this "
                                    f"configuration.")
                            span = num_blocks * k_block_bytes
                            tensor = torch.as_strided(
                                raw_tensor[ssm_offset:ssm_offset + span].view(dtype),
                                size=target_shape,
                                stride=(k_block_bytes // dtype_size,
                                        *torch.empty(shape).stride()))
                        else:
                            tensor = raw_tensor[start_idx:start_idx + num_blocks * block_bytes].view(dtype).view(target_shape)
                            start_idx += num_blocks * block_bytes
                        state_tensors.append(tensor)
                    if ssm_offset is not None and any(
                            len(s) == 3 for s in current_kv_cache_spec.shapes):
                        # The head states (conv) must not spill into the K
                        # region the ssm view is anchored to.
                        assert 0 <= start_idx <= ssm_offset, (
                            f"[hybrid kv layout] {layer_name}: head states "
                            f"span {start_idx}B but the K region starts at "
                            f"{ssm_offset}B (raw {raw_tensor.numel()}B, K "
                            f"block {k_block_bytes}B); page accounting "
                            f"mismatch.")
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
                if not attn_groups:
                    # Edge-cloud head_tail (首一尾一): this kv cache group's
                    # layers all live on the peer side, so there is no local
                    # backend to query. The value is unused for an empty
                    # group; keep list alignment with the group index.
                    self.kernel_block_sizes.append(
                        [kv_cache_group.kv_cache_spec.block_size]
                    )
                    continue
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
                    mamba_blocks_per_req = (
                        max_num_blocks_per_req if self.cache_config.enable_prefix_caching else 1
                        )
                else:
                    max_chunks = cdiv(
                        max_model_len,
                        self.cache_config.block_size * get_total_cp_world_size(),
                    )
                    mamba_blocks_per_req = max(max_num_blocks_per_req, max_chunks)
                mamba_blocks_per_req += kv_cache_group.kv_cache_spec.num_speculative_blocks
                max_num_blocks_per_req = max(max_num_blocks_per_req, mamba_blocks_per_req)
                max_num_blocks_per_req += kv_cache_group.kv_cache_spec.num_speculative_blocks
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
                cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
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

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """Edge-cloud embedding_only: keep the edge's ``input_batch`` order in
        sync with the cloud's so the metadata-free e2c tensor transfer stays
        layout-aligned.

        In embedding_only the edge runs NO attention layers (head_k=tail_k=0),
        so ``kv_cache_groups`` is empty and the base ``_may_reorder_batch``
        returns early WITHOUT reordering. The cloud, however, runs the full
        transformer and its attention backend sets ``reorder_batch_threshold``
        (=1, or 1 + num_speculative_tokens), so the cloud DOES reorder
        (``reorder_batch_to_split_decodes_and_prefills`` moves prefills to the
        end of ``input_batch``). The e2c transfer sends tensors in the edge's
        order and the cloud reads them in the cloud's order; if only the cloud
        reorders, the two orders diverge and the cloud reads each request's
        mrope/hidden from the wrong buffer offset (dec_off mismatch -> wrong
        RoPE position -> token divergence). Apply the SAME reorder on the
        edge, using the cloud's threshold, so both sides share an identical
        ``input_batch`` order.
        """
        if self._edge_cloud_enabled:
            # Under PD interleaving the edge's input_batch history diverges
            # from the cloud's (PL/DL tails that miss the fast path run a
            # full _update_states, removing/re-adding requests at different
            # positions). Re-anchor BOTH sides to the SO's
            # num_scheduled_tokens key order at every batch so the flat
            # e2c/c2e wire layout always matches the consumer's batch
            # layout. Without this, after a prefill joins the decode batch
            # via an interleaved PL, the edge sends hidden/mrope ordered
            # [A, Z1, Z2] while the cloud reads [Z1, Z2, A] and every
            # request decodes with another request's embeds/mrope.
            _reorder_input_batch_to_so_order(self.input_batch,
                                             scheduler_output)
        if (self._edge_cloud_enabled
                and self.edge_cloud_cfg.mode == "embedding_only"
                and self.edge_cloud_cfg.role == "edge"):
            if self.reorder_batch_threshold is not None:
                threshold = self.reorder_batch_threshold
            else:
                # Match the ascend attention backend's decode_threshold
                # (=1, or 1 + num_speculative_tokens under spec decode).
                threshold = 1
                if self.speculative_config is not None:
                    threshold = 1 + self.speculative_config.num_speculative_tokens
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=threshold,
            )
            return
        super()._may_reorder_batch(scheduler_output)

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
                    impl = attn_module.impl
                    has_indexer = bool(getattr(impl, "has_indexer", False))
                    use_sparse_c8_sfa_for_layer = bool(getattr(impl, "use_sparse_c8_sfa", False))
                    use_sparse_c8_indexer_for_layer = bool(getattr(impl, "use_sparse_c8_indexer", False))

                    if use_sparse_c8_sfa_for_layer:
                        packed_kv_head_dim = get_sfa_qsfa_packed_head_dim(
                            self.model_config.hf_text_config.kv_lora_rank,
                            self.model_config.hf_text_config.qk_rope_head_dim,
                        )
                        sparse_head_dim = (
                            packed_kv_head_dim,
                            0,
                            (
                                self.model_config.hf_text_config.index_head_dim
                                if use_sparse_c8_indexer_for_layer
                                else 0
                            ),
                        )
                    elif has_indexer:
                        sparse_head_dim = self.sparse_head_dim
                    else:
                        # Layers that reuse another layer's top-k indices only
                        # need the MLA latent and RoPE caches.
                        sparse_head_dim = (
                            self.model_config.hf_text_config.kv_lora_rank,
                            self.model_config.hf_text_config.qk_rope_head_dim,
                            0,
                        )

                    kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
                        block_size=self.block_size,
                        num_kv_heads=1,
                        head_size=sum(sparse_head_dim),
                        sparse_head_dim=sparse_head_dim,
                        dtype=self.kv_cache_dtype,
                        cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
                        cache_sparse_c8=use_sparse_c8_sfa_for_layer,
                        sfa_dcp_replicated_indexer_size=self.sfa_dcp_replicated_indexer_size,
                    )
                elif spec := attn_module.get_kv_cache_spec(self.vllm_config):
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
                    # Rebuild to a fresh, picklable spec (the returned one
                    # references a stale MLAAttentionSpec class shadowed by
                    # patch_kv_cache_interface.py). Keep the HiddenStateCacheSpec
                    # type so get_kv_cache_groups isolates this cache-only layer
                    # into its own group; downgrading to MLAAttentionSpec would
                    # break page-size unification on hybrid models (e.g. Qwen3.5).
                    kv_cache_spec[layer_name] = HiddenStateCacheSpec(
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
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_attn_backend = None

        for attn_backend_set, kv_cache_group in zip(
            attention_backends, kv_cache_groups
        ):
            for attn_backend in attn_backend_set:
                builder_cls = attn_backend.get_builder_cls()
                cg_support = builder_cls.get_cudagraph_support(
                    self.vllm_config, kv_cache_group.kv_cache_spec
                )
                if cg_support.value < min_cg_support.value:
                    min_cg_support = cg_support
                    min_cg_attn_backend = attn_backend.__name__

        with update_pass_config(self):
            cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
                min_cg_support,
                min_cg_attn_backend,
                self.uniform_decode_query_len,
                self.parallel_config.tensor_parallel_size,
                self.kv_cache_config,
                self.max_num_reqs,
            )
            self.cudagraph_dispatcher.initialize_cudagraph_keys(
                cudagraph_mode, self.uniform_decode_query_len
            )

        if (
            self.speculative_config
            and self.drafter is not None
            and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_extract_hidden_states()
            )
        ):
            assert isinstance(
                self.drafter,
                AscendEagleProposer | AscendDflashProposer | AscendExtractHiddenStatesProposer,
            )
            self.drafter.initialize_cudagraph_keys(cudagraph_mode)

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

                # Also initialize graph params for edge-cloud draft drafter segments.
                if (
                    self.speculative_config
                    and self.speculative_config.method in ("mtp", "eagle3")
                    and hasattr(self, "_edge_cloud_draft_segments")
                ):
                    for wrapper in self._edge_cloud_draft_segments.values():
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
        # Include edge-cloud draft drafter segment wrappers so that
        # capture_model() can clear any stale entries from them.
        if hasattr(self, "_edge_cloud_draft_segments"):
            for wrapper in self._edge_cloud_draft_segments.values():
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
        self._edge_cloud_target_capture_in_progress = True
        try:
            with _torch_cuda_wrapper(), _replace_gpu_model_runner_function_wrapper(
                parent_module_name
            ):
                cuda_graph_size = GPUModelRunner.capture_model(self)
        finally:
            self._edge_cloud_target_capture_in_progress = False

        mgr = self.encoder_cudagraph_manager
        if mgr is not None and hasattr(self, "update_stream"):
            mgr.update_stream = self.update_stream

        self._zero_dsa_state_block0()
        return cuda_graph_size

    def _zero_dsa_state_block0(self) -> None:
        """Zero physical block 0 (null/dummy block) of every DSA compressor
        state cache. Always-on; called once at the end of capture_model.

        The compressor kernel indexes state_cache in raw-position space
        (block_table[req][pos // 8]), but the decode-time state block table
        carries only one valid block; positions >= 8 hit zero-padding
        entries. Kernel writes to entry 0 are silently dropped, while reads
        are NOT guarded and land on physical block 0, whose content is
        capture-order-dependent residue (NaN when the size-1 capture runs
        last -> accuracy corruption at the first decode compression
        boundary). Zeroing block 0 makes the phantom read deterministic and
        restores eager parity.

        Once after capture is sufficient: real inference never writes state
        block 0 (kernel writes to entry 0 are dropped; decode padding slots
        are -1, not 0), so the zeroed content persists for the process
        lifetime. Verified: b0nan stays constant across prefill+decode.
        """
        try:
            caches = getattr(self, "_dsa_state_caches_for_zero", None)
            if not caches:
                # （重）收集。profile/dummy 阶段 KV cache 尚未初始化，此时
                # 收集到 0 个属正常——不能缓存空列表，否则后续永远不再重试。
                caches = []
                names = getattr(self, "kv_cache_names", None) or []
                runner_caches = getattr(self, "kv_caches", None) or []
                for i, name in enumerate(names):
                    if "compressor.state_cache" not in name or i >= len(runner_caches):
                        continue
                    entry = runner_caches[i]
                    cache = entry[0] if isinstance(entry, (list, tuple)) else entry
                    if isinstance(cache, torch.Tensor) and cache.numel() > 0:
                        caches.append(cache)
                if not caches:
                    return
                # 收集成功才缓存
                self._dsa_state_caches_for_zero = caches
            for cache in caches:
                cache[0].zero_()
        except Exception:
            pass

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

    def _init_kv_zero_meta(self) -> None:
        """One-time precomputation for _zero_block_ids.

        Delegates to KVBlockZeroer.init_meta with the runner's state.
        Called from gpu_worker.py outside the CuMem pool context.
        """
        self._kv_block_zeroer = AscendKVBlockZeroer(self.device, self.pin_memory)
        self._kv_block_zeroer.init_meta(
            attn_groups_iter=self._kv_cache_spec_attn_group_iterator(),
            kernel_block_sizes=self.kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            runner_only_attn_layers=self.runner_only_attn_layers,
            static_forward_context=(self.compilation_config.static_forward_context),
        )


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
            self.record = lambda *a, **kw: None
            self.synchronize = lambda *a, **kw: None
            self.wait = lambda *a, **kw: None
            self.query = lambda *a, **kw: True

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
    import vllm.v1.worker.encoder_cudagraph as _vllm_encoder_cudagraph

    from vllm_ascend.worker.encoder_acl_graph import EncoderAclGraphManager

    _encoder_mgr_orig = _vllm_encoder_cudagraph.EncoderCudaGraphManager
    _vllm_encoder_cudagraph.EncoderCudaGraphManager = EncoderAclGraphManager
    target_module = None
    try:
        target_module = sys.modules[target_module_name]
        setattr(target_module, "graph_capture", graph_capture)  # noqa: B010
        yield
    except Exception as e:
        raise RuntimeError(f"NPUModelRunner failed, error is {e}")
    finally:
        _vllm_encoder_cudagraph.EncoderCudaGraphManager = _encoder_mgr_orig
        if target_module is not None:
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
