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

from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu
from vllm.distributed.parallel_state import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.distributed.utils import fc3_all_gather_and_maybe_unpad_impl
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEPrepareOutput
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import enable_sp, enable_sp_by_pass, npu_stream_switch, prefill_context_parallel_enable

# [DPDBG] MoE collective instrumentation (dummy+real share this path).
# (1) Non-blocking torch.npu.Event markers e0..e3 around all_gather/a2a/
#   reduce_scatter + a watchdog thread that query()s them. record/query add NO
#   stream sync -> do NOT perturb the deadlock. CAVEAT: HCCL collectives run on
#   hcclStreams_[key] (per-group), not the current stream, so an event recorded
#   on the current stream may read done before the collective completes. If all
#   events read done while .item()@5699 still hangs, mode (1) is inconclusive.
# (2) Precise per-collective synchronize()+log, env-gated VLLM_DPDBG_MOE_SYNC=1
#   (PERTURBS the deadlock - debug only). The blocking sync localizes:
#     ag_done hangs -> all_gather; pre_rs hangs (ag_done passed) -> a2a;
#     rs_done hangs (pre_rs passed) -> reduce_scatter.
import os as _os
import logging as _logging
import threading as _threading
from collections import deque as _deque

_dpdbg_moe_logger = _logging.getLogger("vllm")
_DPDBG_MOE_HIST = _deque(maxlen=64)
_DPDBG_MOE_CTR = [0]
_DPDBG_MOE_LOCK = _threading.Lock()
_DPDBG_MOE_WD = [False]
_DPDBG_MOE_SYNC = _os.environ.get("VLLM_DPDBG_MOE_SYNC") == "1"


def _dpdbg_moe_rank():
    try:
        return dist.get_rank() if dist.is_initialized() else -1
    except Exception:
        return -1


def _dpdbg_moe_compiling():
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


def _dpdbg_moe_rec():
    """Record a non-blocking event on the current stream; None if compiling."""
    if _dpdbg_moe_compiling():
        return None
    try:
        e = torch.npu.Event()
        e.record()
        return e
    except Exception:
        return None


def _dpdbg_moe_q(e):
    try:
        return 1 if (e is not None and e.query()) else (0 if e is not None else -1)
    except Exception:
        return -1


def _dpdbg_moe_sync_log(label):
    if not _DPDBG_MOE_SYNC or _dpdbg_moe_compiling():
        return
    try:
        torch.npu.synchronize()
        _dpdbg_moe_logger.error("[DPDBG] moe_sync %s done r=%s", label, _dpdbg_moe_rank())
    except Exception:
        pass


def _dpdbg_moe_wd():
    import time
    while True:
        time.sleep(2.0)
        try:
            with _DPDBG_MOE_LOCK:
                items = list(_DPDBG_MOE_HIST)
            for fid, rank, bt, lyr, e0, e1, e2, e3 in items[-8:]:
                q0 = _dpdbg_moe_q(e0); q1 = _dpdbg_moe_q(e1)
                q2 = _dpdbg_moe_q(e2); q3 = _dpdbg_moe_q(e3)
                if q0 == 1 and q1 == 1 and q2 == 1 and q3 == 1:
                    continue  # fully complete - skip to reduce idle noise
                _dpdbg_moe_logger.error(
                    "[DPDBG] moe_evt fwd=%s r=%s bt=%s lyr=%s e0(pre-ag)=%s "
                    "e1(ag_done)=%s e2(a2a_done)=%s e3(rs_done)=%s",
                    fid, rank, bt, lyr, q0, q1, q2, q3,
                )
        except Exception:
            pass


def _dpdbg_moe_ensure_wd():
    if not _DPDBG_MOE_WD[0]:
        with _DPDBG_MOE_LOCK:
            if not _DPDBG_MOE_WD[0]:
                t = _threading.Thread(target=_dpdbg_moe_wd, daemon=True)
                t.start()
                _DPDBG_MOE_WD[0] = True


class PrepareAndFinalize(ABC):
    """
    Abstract base class for MoE (Mixture-of-Experts) tensor preparation and finalization
    in distributed environments. Subclasses implement specific communication strategies
    (e.g., AllGather, All2All, MC2) to handle tensor padding, slicing,
    broadcasting, and reduction across TP/DP/EP groups.

    Attributes:
        moe_config (FusedMoEConfig): Configuration object containing TP/DP/EP group info,
                                     sizes, ranks, and communication settings.
    """

    quant_stream: torch.npu.Stream | None = None

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config
        ascend_config = get_ascend_config()
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
        if self.multistream_overlap_gate and PrepareAndFinalize.quant_stream is None:
            PrepareAndFinalize.quant_stream = torch.npu.Stream()

    @abstractmethod
    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Prepare tensors before MoE computation. May involve:
          - Padding to align communication boundaries
          - Slicing across tensor-parallel ranks
          - Broadcasting across data-parallel ranks

        Args:
            hidden_states (torch.Tensor): Input features, shape [num_tokens, hidden_size]
            router_logits (torch.Tensor): Router outputs, shape [num_tokens, num_experts]
            enable_shared_expert_dp (bool): Skip DP communication for shared experts
            replace_allreduce (bool): Bypass default all-reduce behavior
            quant_type: none, w8a8, w4a8, mxfp8, or mxfp4

        Returns:
            MoEPrepareOutput:
                - processed hidden_states (may be padded/sliced/broadcasted)
                - processed router_logits (may be recomputed or broadcasted)
                - optional communication mask (e.g., mc2_mask for sparse ops)
                - optional padded hidden state shape for finalization
                - optional per-token scale for quantized path
        """
        raise NotImplementedError("Prepare not implemented.")

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        Finalize MoE output. May involve:
          - Gathering sliced tensors across TP ranks
          - Reducing or scattering across DP ranks
          - Unpadding to original token count
          - Applying all-reduce across TP/EP if requested

        Args:
            hidden_states (torch.Tensor): MoE layer output, possibly padded or sliced
            reduce_results (bool): Whether to apply all-reduce across TP/EP groups

        Returns:
            torch.Tensor: Final output with shape [original_num_tokens, hidden_size]
        """
        raise NotImplementedError("Finalize function not implemented.")


class PrepareAndFinalizeWithAll2All(PrepareAndFinalize):
    """
    MoE communication strategy using All-to-All style slicing.
    Similar to MC2 but does not use mc2_mask; instead pads to TP size for uniform slicing.
    Will be used when num_tokens exceed mc2's limitation (512 tokens/rank).
    """

    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self._restore_tp_across_dp()

    def _restore_tp_across_dp(self):
        """Restore original TP configuration (same as MC2)."""
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Preparation steps:
          1. Pad hidden_states and router_logits to next multiple of TP size.
          2. If TP > 1, split along token dim and select current TP rank's slice.
          3. Save splits for later all-gather in finalize.

        Skips if `enable_shared_expert_dp` or `replace_allreduce` is True.

        Returns:
            MoEPrepareOutput where `mc2_mask` is None for All2All path.
        """
        self.replace_allreduce = replace_allreduce
        self.enable_shared_expert_dp = enable_shared_expert_dp

        padded_hidden_states_shape = hidden_states.shape
        if not (self.replace_allreduce or self.enable_shared_expert_dp):
            self.num_tokens, _ = hidden_states.shape
            pad_size = self.tp_size - self.num_tokens  # Pad to TP size (cyclic)

            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
                padded_hidden_states_shape = hidden_states.shape

            if self.tp_size > 1:
                split_hidden_states = torch.tensor_split(hidden_states, self.tp_size, dim=0)
                split_router_logits = torch.tensor_split(router_logits, self.tp_size, dim=0)

                hidden_states = split_hidden_states[self.tp_rank]
                router_logits = split_router_logits[self.tp_rank]

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=None,
            padded_hidden_states_shape=padded_hidden_states_shape,
            pertoken_scale=None,
        )

    def pad_and_split_input_ids(
        self,
        input_ids,
    ):
        if not (self.replace_allreduce or self.enable_shared_expert_dp):
            pad_size = self.tp_size - self.num_tokens
            if pad_size > 0:
                input_ids = nn.functional.pad(input_ids, (0, pad_size))

            if self.tp_size > 1:
                input_ids = torch.tensor_split(input_ids, self.tp_size, dim=0)
                input_ids = input_ids[self.tp_rank]
        return input_ids

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        Finalization steps:
          1. If TP > 1, all-gather slices to reconstruct full tensor.
          2. Unpad to original token count.
          3. Return [original_num_tokens, hidden_size] tensor.

        Skips if `enable_shared_expert_dp` or `replace_allreduce` is True.
        """

        if not (self.enable_shared_expert_dp or self.replace_allreduce):
            if self.tp_size > 1:
                assert padded_hidden_states_shape is not None
                # Cannot reuse `split_hidden_states` from prepare phase as it
                # may share memory with original hidden_states. Since shared
                # experts may use the original tensor, reusing it would cause
                # in-place modification during all_gather, corrupting the data.
                gathered_hidden_states = torch.empty(
                    padded_hidden_states_shape, device=hidden_states.device, dtype=hidden_states.dtype
                )
                split_hidden_states = torch.tensor_split(gathered_hidden_states, self.tp_size, dim=0)
                dist.all_gather(list(split_hidden_states), hidden_states, self.moe_config.tp_group.device_group)
                hidden_states = gathered_hidden_states

            if self.num_tokens < hidden_states.shape[0]:
                hidden_states = hidden_states[: self.num_tokens]

        return hidden_states


class PrepareAndFinalizeWithMC2(PrepareAndFinalizeWithAll2All):
    """
    MoE communication strategy using MC2, which is based on All2All. Hence, it inherits
    All2All and share the same finalize method.
    Designed for Ascend or environments requiring explicit padding and slicing control.
    Relies on `mc2_mask` and `padded_num_tokens` from forward_context for alignment.
    """

    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self._restore_tp_across_dp()

    def _restore_tp_across_dp(self):
        """
        Restore original TP configuration.
        vLLM flattens TP and DP into a single dimension; this method recovers
        the true TP world size and rank for correct tensor slicing.
        """
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Preparation steps:
          1. Fetch `mc2_mask` and target padding length from forward context.
          2. Pad `hidden_states` and `router_logits` to target length if needed.
          3. If TP > 1, split tensors along token dimension and select current TP rank's slice.
          4. Split and return corresponding `mc2_mask`.

        Skips padding/slicing if `enable_shared_expert_dp` or `replace_allreduce` is True.

        Returns:
            MoEPrepareOutput, possibly sliced/padded.
        """
        self.replace_allreduce = replace_allreduce
        self.enable_shared_expert_dp = enable_shared_expert_dp
        mc2_mask = _EXTRA_CTX.mc2_mask
        if self.tp_size > 1:
            # Also slice mc2_mask
            split_mc2_mask = torch.tensor_split(mc2_mask, self.tp_size, dim=0)
            mc2_mask = split_mc2_mask[self.tp_rank]

        padded_hidden_states_shape = hidden_states.shape
        if not self.replace_allreduce:
            self.num_tokens, _ = hidden_states.shape
            target_pad_length = _EXTRA_CTX.padded_num_tokens
            pad_size = target_pad_length - self.num_tokens

            # Pad if necessary (unless shared expert DP is enabled)
            if pad_size > 0 and not self.enable_shared_expert_dp:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
                padded_hidden_states_shape = hidden_states.shape

            # Slice across TP ranks
            if self.tp_size > 1 and not self.enable_shared_expert_dp:
                split_hidden_states = torch.tensor_split(hidden_states, self.tp_size, dim=0)
                split_router_logits = torch.tensor_split(router_logits, self.tp_size, dim=0)
                hidden_states = split_hidden_states[self.tp_rank]
                router_logits = split_router_logits[self.tp_rank]

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=mc2_mask,
            padded_hidden_states_shape=padded_hidden_states_shape,
            pertoken_scale=None,
        )

    def pad_and_split_input_ids(
        self,
        input_ids,
    ):
        if not self.replace_allreduce:
            forward_context = get_forward_context()
            target_pad_length = forward_context.padded_num_tokens
            pad_size = target_pad_length - self.num_tokens
            if pad_size > 0 and not self.enable_shared_expert_dp:
                input_ids = nn.functional.pad(input_ids, (0, pad_size))

            if self.tp_size > 1 and not self.enable_shared_expert_dp:
                input_ids = torch.tensor_split(input_ids, self.tp_size, dim=0)
                input_ids = input_ids[self.tp_rank]
        return input_ids


class PrepareAndFinalizeWithAllGather(PrepareAndFinalize):
    """
    MoE communication strategy using All-Gather + Reduce-Scatter on EP group.
    There are two sets of prepare and finalize:
    1. _prepare_with_dp_group/_finalize_with_dp_group: When sequence parallelism is not enabled,
    we gather inputs across DP ranks before MoE, scatter outputs after.
    The communication and calculation process is as follows (AG, AR and RS
    are abbreviations for All-Gather, All-Reduce and Reduce-Scatter, respectively):

    Attn → TP AR → DP AG → MoE → DP RS → TP AR

    2. _prepare_with_ep_group/_finalize_with_ep_group: When sequence parallelism is enabled,
    the above process becomes:

    TP AG → Attn → TP RS → TP AG → DP AG → MoE → DP RS → TP RS

    This strategy further combines TP AG + DP AG into EP All-Gather and TP RS + DP RS
    into EP Reduce-Scatter to improve communication performance. The optimized process is as follows:

    TP AG → Attn → TP RS → EP AG → MoE → EP RS
    """

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Preparation steps:
          AllGather hidden_states and router_logits to form global tensors.

        Returns:
            MoEPrepareOutput with global tensors.
        """
        if enable_sp() or enable_sp_by_pass():
            return self._prepare_with_ep_group(hidden_states, router_logits, quant_type)

        return self._prepare_with_dp_group(hidden_states, router_logits, enable_shared_expert_dp, replace_allreduce)

    def _prepare_with_ep_group(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor, quant_type=QuantType.NONE
    ) -> MoEPrepareOutput:
        pertoken_scale = None
        if quant_type == QuantType.W8A8:
            hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
        elif quant_type == QuantType.MXFP8:
            hidden_states, pertoken_scale = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=torch.float8_e4m3fn)
        elif quant_type == QuantType.MXFP4:
            # MXFP4 with AllGather+EP currently does not pre-quantize
            # per-token activations in prepare. Keep quantization in the MoE MLP path.
            pass

        if self.multistream_overlap_gate:
            assert PrepareAndFinalize.quant_stream is not None
            PrepareAndFinalize.quant_stream.wait_stream(torch.npu.current_stream())
            with npu_stream_switch(PrepareAndFinalize.quant_stream, enabled=self.multistream_overlap_gate):
                hidden_states = fc3_all_gather_and_maybe_unpad_impl(hidden_states)
        else:
            hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states, True, True)
            router_logits = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(router_logits, True, True)

        # TODO(fuzhihong): To adapt to self.num_token in the all_gather_input_id_with_dp_group method,
        #  when flashcomm1 is used and dp = N(N >=2).
        self.num_tokens = hidden_states.shape[0]

        if pertoken_scale is not None:
            pertoken_scale = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(pertoken_scale, True, True)

        if self.multistream_overlap_gate:
            torch.npu.current_stream().wait_stream(PrepareAndFinalize.quant_stream)

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=None,
            padded_hidden_states_shape=None,
            pertoken_scale=pertoken_scale,
        )

    def _prepare_with_dp_group(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Preparation steps:
          1. Fetch max token count across DP group from forward context.
          2. Pad local tensors to that size.
          3. All-gather across DP group to form global input tensor.

        Returns:
            MoEPrepareOutput with global tensors.
        """
        self.enable_shared_expert_dp = enable_shared_expert_dp
        if self.moe_config.dp_size > 1:
            max_tokens_across_dp = _EXTRA_CTX.max_tokens_across_dp

            self.num_tokens = hidden_states.shape[0]
            pad_size = max_tokens_across_dp - self.num_tokens
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))

            # [DPDBG] e0: before all_gather
            if not _dpdbg_moe_compiling():
                with _DPDBG_MOE_LOCK:
                    _DPDBG_MOE_CTR[0] += 1
                    self._dpdbg_moe_fid = _DPDBG_MOE_CTR[0]
                try:
                    self._dpdbg_moe_bt = _EXTRA_CTX.dp_batch_type_id
                    self._dpdbg_moe_lyr = _EXTRA_CTX.layer_idx
                except Exception:
                    self._dpdbg_moe_bt = -1
                    self._dpdbg_moe_lyr = -1
                self._dpdbg_moe_e0 = _dpdbg_moe_rec()
                _dpdbg_moe_ensure_wd()
            # All-gather across DP group
            hidden_states = self.moe_config.dp_group.all_gather(hidden_states, 0)
            router_logits = self.moe_config.dp_group.all_gather(router_logits, 0)
            # [DPDBG] e1: after all_gather; precise sync if VLLM_DPDBG_MOE_SYNC=1
            if not _dpdbg_moe_compiling():
                self._dpdbg_moe_e1 = _dpdbg_moe_rec()
            _dpdbg_moe_sync_log("ag_done")

        if prefill_context_parallel_enable() and self.moe_config.pcp_size > 1:
            max_tokens_across_pcp = _EXTRA_CTX.max_tokens_across_pcp

            self.num_tokens_pcp = hidden_states.shape[0]
            pad_size = max_tokens_across_pcp - self.num_tokens_pcp
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))

            hidden_states = get_pcp_group().all_gather(
                hidden_states,
                dim=0,
            )
            router_logits = get_pcp_group().all_gather(
                router_logits,
                dim=0,
            )

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=None,
            padded_hidden_states_shape=None,
            pertoken_scale=None,
        )

    def all_gather_input_id_with_dp_group(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.moe_config.dp_size > 1:
            max_tokens_across_dp = _EXTRA_CTX.max_tokens_across_dp
            pad_size = max_tokens_across_dp - self.num_tokens
            if pad_size > 0:
                input_ids = nn.functional.pad(input_ids, (0, pad_size))

            input_ids = self.moe_config.dp_group.all_gather(input_ids, 0)
        return input_ids

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        Finalization steps:
          Reduce Scatter hidden states.

        Returns:
            Tensor with shape [local_num_tokens, hidden_size]
        """
        if enable_sp() or enable_sp_by_pass():
            return self._finalize_with_ep_group(hidden_states)

        return self._finalize_with_dp_group(hidden_states, reduce_results)

    def _finalize_with_ep_group(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Argument `reduce_results` is not needed in this func. Given sequence parallelism is enabled:
        1. Reduce_results is False usually happens when models have shared experts and need to
        allreduce hidden states after results of shared experts and routed experts are added in FusedMoe.
        We do reduce scatter for hidden states here, then skip allreudce in FusedMoe and add it to the
        result of shared experts.
        2 Reduce_results is True usually happens when model has no shared experts. We still do reduce scatter
        here, then skip allreudce in FusedMoe.
        """
        hidden_states = torch.ops.vllm.maybe_pad_and_reduce(hidden_states, True)

        return hidden_states

    def _finalize_with_dp_group(self, hidden_states: torch.Tensor, reduce_results: bool) -> torch.Tensor:
        """
        Finalization steps:
          1. If DP > 1 and not shared expert, reduce-scatter output across DP group.
          2. Slice to original local token count.
          3. If `reduce_results=True` and TP/EP > 1, apply tensor_model_parallel_all_reduce.

        Returns:
            Tensor with shape [original_local_num_tokens, hidden_size]
        """
        if self.moe_config.dp_size > 1 and not self.enable_shared_expert_dp:
            # [DPDBG] e2: before reduce_scatter (a2a done boundary);
            # precise sync here (VLLM_DPDBG_MOE_SYNC=1) localizes a2a vs rs.
            _dpdbg_e2 = _dpdbg_moe_rec() if not _dpdbg_moe_compiling() else None
            _dpdbg_moe_sync_log("pre_rs")
            hidden_states = get_dp_group().reduce_scatter(hidden_states, 0)
            hidden_states = hidden_states[: self.num_tokens]
            # [DPDBG] e3: after reduce_scatter; append full entry to history.
            if not _dpdbg_moe_compiling():
                _dpdbg_e3 = _dpdbg_moe_rec()
                with _DPDBG_MOE_LOCK:
                    _DPDBG_MOE_HIST.append((
                        getattr(self, "_dpdbg_moe_fid", -1),
                        _dpdbg_moe_rank(),
                        getattr(self, "_dpdbg_moe_bt", -1),
                        getattr(self, "_dpdbg_moe_lyr", -1),
                        getattr(self, "_dpdbg_moe_e0", None),
                        getattr(self, "_dpdbg_moe_e1", None),
                        _dpdbg_e2,
                        _dpdbg_e3,
                    ))
            _dpdbg_moe_sync_log("rs_done")

        if prefill_context_parallel_enable() and self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().reduce_scatter(hidden_states, dim=0)
        return hidden_states
