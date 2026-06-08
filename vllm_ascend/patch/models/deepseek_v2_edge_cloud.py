#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

from itertools import islice
from typing import Any

import torch
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2ForCausalLM,
    DeepseekV2Model,
)
from vllm.sequence import IntermediateTensors


def _forward_edge_cloud_segment_deepseek_v2(
    self: DeepseekV2Model,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    is_first_segment: bool | None = None,
    is_last_segment: bool | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    num_layers = len(self.layers)
    assert 0 <= start_layer <= end_layer <= num_layers, (
        f"Invalid segment range [{start_layer}, {end_layer}) for {num_layers} layers"
    )

    if is_first_segment is None:
        is_first_segment = start_layer == 0 and get_pp_group().is_first_rank
    if is_last_segment is None:
        is_last_segment = end_layer == num_layers and get_pp_group().is_last_rank

    if is_first_segment:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None, (
            "intermediate_tensors is None in edge-cloud segment; "
            "check that all TP ranks receive tensors correctly."
        )
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    # DeepseekV2DecoderLayer.forward does not accept **kwargs; do not forward
    # extra_layer_kwargs here or unrelated model_kwargs will raise TypeError.
    for layer in islice(self.layers, start_layer, end_layer):
        hidden_states, residual = layer(positions, hidden_states, residual)

    if not is_last_segment:
        if residual is None:
            residual = torch.zeros_like(hidden_states)
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )

    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


def _deepseek_v2_lm_forward_edge_cloud_segment(
    self: DeepseekV2ForCausalLM,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    return self.model.forward_edge_cloud_segment(
        start_layer,
        end_layer,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
        **extra_layer_kwargs,
    )


DeepseekV2Model.forward_edge_cloud_segment = _forward_edge_cloud_segment_deepseek_v2
DeepseekV2ForCausalLM.forward_edge_cloud_segment = (
    _deepseek_v2_lm_forward_edge_cloud_segment
)
