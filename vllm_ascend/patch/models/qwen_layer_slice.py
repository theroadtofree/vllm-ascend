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
# ----------------------------------------------------------------------------
# Layer-slice patch for Qwen2 / Qwen3 / Qwen3.5 / Qwen3-Next models.
#
# Why:
#   The PD-mix layer-slice runtime in vllm-ascend (passive_scheduler +
#   model_runner_v1._edge_cloud_forward) needs to drive a model.forward over
#   a sub-range of decoder layers and (optionally) force IntermediateTensors
#   to be returned even on the last PP rank, so the runtime can stitch
#   slices/segments together.  Upstream vLLM model.forward signatures do
#   not expose layer slicing, so the dest fork carries small kwargs
#   additions on `Qwen2Model.forward`, `Qwen3NextModel.forward` and the
#   surrounding `*ForCausalLM` / `*ForConditionalGeneration` wrappers.
#
# How:
#   This patch keeps upstream vLLM untouched and re-binds those forward
#   methods at runtime.  It is loaded on demand from
#   `vllm_ascend/worker/model_runner_v1.py` only when the layer-slice
#   feature is actually enabled (layer-slice YAML config is present).
#
#   - `Qwen3Model` extends `Qwen2Model`, so patching `Qwen2Model.forward`
#     transparently covers Qwen3 dense models as well.
#   - `Qwen3_5Model` extends `Qwen3NextModel`, so patching
#     `Qwen3NextModel.forward` transparently covers Qwen3.5.
#   - `*ForCausalLM` / `*ForCausalLMBase` wrappers only need to forward the
#     three new kwargs to their inner `self.model(...)` call.
#   - `Qwen3_5ForConditionalGeneration.forward` keeps the upstream
#     "intermediate_tensors is not None ⇒ inputs_embeds = None" guard.
# ----------------------------------------------------------------------------

from itertools import islice

import torch
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM, Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLMBase,
    Qwen3_5ForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextForCausalLM,
    Qwen3NextModel,
)
from vllm.sequence import IntermediateTensors


def _qwen2_model_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    layer_slice_start: int | None = None,
    layer_slice_end: int | None = None,
    layer_slice_return_intermediate: bool = False,
) -> torch.Tensor | IntermediateTensors:
    """Patched ``Qwen2Model.forward`` adding layer-slice support.

    Behavior is identical to upstream when all ``layer_slice_*`` kwargs are
    left at their defaults.  When ``layer_slice_start``/``layer_slice_end``
    are provided, only that 0-based sub-range (relative to ``self.start_layer``)
    of decoder layers is executed; when ``layer_slice_return_intermediate`` is
    True, ``IntermediateTensors`` are returned even on the last PP rank.
    """
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    # Determine the layer range to execute.  When layer slicing is
    # active, layer_slice_start/end (0-based within the local PP
    # rank's layers) restrict the iteration; otherwise the full
    # [start_layer, end_layer) range is used.
    exec_start = (
        self.start_layer + layer_slice_start
        if layer_slice_start is not None
        else self.start_layer
    )
    exec_end = (
        self.start_layer + layer_slice_end
        if layer_slice_end is not None
        else self.end_layer
    )

    aux_hidden_states = self._maybe_add_hidden_state(
        [], 0, hidden_states, residual
    )
    for idx, layer in enumerate(islice(self.layers, exec_start, exec_end)):
        hidden_states, residual = layer(positions, hidden_states, residual)
        self._maybe_add_hidden_state(
            aux_hidden_states, idx + 1, hidden_states, residual
        )

    if not get_pp_group().is_last_rank or layer_slice_return_intermediate:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )

    hidden_states, _ = self.norm(hidden_states, residual)

    if len(aux_hidden_states) > 0:
        return hidden_states, aux_hidden_states

    return hidden_states


def _qwen3_next_model_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    layer_slice_start: int | None = None,
    layer_slice_end: int | None = None,
    layer_slice_return_intermediate: bool = False,
) -> (
    torch.Tensor
    | IntermediateTensors
    | tuple[torch.Tensor, list[torch.Tensor]]
):
    """Patched ``Qwen3NextModel.forward`` (also covers ``Qwen3_5Model``)."""
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    exec_start = (
        self.start_layer + layer_slice_start
        if layer_slice_start is not None
        else self.start_layer
    )
    exec_end = (
        self.start_layer + layer_slice_end
        if layer_slice_end is not None
        else self.end_layer
    )

    aux_hidden_states = self._maybe_add_hidden_state(
        [], 0, hidden_states, residual
    )
    for layer_idx, layer in enumerate(
        islice(self.layers, exec_start, exec_end),
        start=exec_start,
    ):
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )
        self._maybe_add_hidden_state(
            aux_hidden_states, layer_idx + 1, hidden_states, residual
        )

    if not get_pp_group().is_last_rank or layer_slice_return_intermediate:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )
    hidden_states, _ = self.norm(hidden_states, residual)
    if aux_hidden_states:
        return hidden_states, aux_hidden_states
    return hidden_states


def _make_lm_forward(inner_attr: str = "model"):
    """Build a ``forward`` that pipes layer_slice kwargs to ``self.<inner_attr>``.

    Used for ``Qwen2ForCausalLM`` / ``Qwen3ForCausalLM`` /
    ``Qwen3NextForCausalLM`` / ``Qwen3_5ForCausalLMBase``.  Their original
    ``forward`` is a thin wrapper that calls ``self.model(...)``; we keep that
    contract and only forward the three extra kwargs.
    """

    def _forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        layer_slice_start: int | None = None,
        layer_slice_end: int | None = None,
        layer_slice_return_intermediate: bool = False,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        inner = getattr(self, inner_attr)
        return inner(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            layer_slice_start=layer_slice_start,
            layer_slice_end=layer_slice_end,
            layer_slice_return_intermediate=layer_slice_return_intermediate,
        )

    return _forward


def _qwen3_5_cond_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    layer_slice_start: int | None = None,
    layer_slice_end: int | None = None,
    layer_slice_return_intermediate: bool = False,
    **kwargs: object,
) -> torch.Tensor | IntermediateTensors:
    """Patched ``Qwen3_5ForConditionalGeneration.forward`` with layer slicing.

    Preserves the upstream rule that pipeline-stage tensors override
    pre-computed input embeddings.
    """
    if intermediate_tensors is not None:
        inputs_embeds = None

    return self.language_model.model(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        layer_slice_start=layer_slice_start,
        layer_slice_end=layer_slice_end,
        layer_slice_return_intermediate=layer_slice_return_intermediate,
    )


# ---------------------------------------------------------------------------
# Apply the patches.  We patch *base* model classes so subclasses
# (Qwen3Model -> Qwen2Model, Qwen3_5Model -> Qwen3NextModel) inherit the
# new forward automatically.
# ---------------------------------------------------------------------------
Qwen2Model.forward = _qwen2_model_forward
Qwen3NextModel.forward = _qwen3_next_model_forward

Qwen2ForCausalLM.forward = _make_lm_forward("model")
Qwen3ForCausalLM.forward = _make_lm_forward("model")
Qwen3NextForCausalLM.forward = _make_lm_forward("model")
Qwen3_5ForCausalLMBase.forward = _make_lm_forward("model")

Qwen3_5ForConditionalGeneration.forward = _qwen3_5_cond_forward
