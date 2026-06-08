#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

from typing import Any

import torch
from vllm.model_executor.models.kimi_k25 import (
    KimiK25ForConditionalGeneration,
)
from vllm.sequence import IntermediateTensors

import vllm_ascend.patch.models.deepseek_v2_edge_cloud  # noqa: F401


def _kimi_k25_forward_edge_cloud_segment(
    self: KimiK25ForConditionalGeneration,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    return self.language_model.forward_edge_cloud_segment(
        start_layer,
        end_layer,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
        **extra_layer_kwargs,
    )


KimiK25ForConditionalGeneration.forward_edge_cloud_segment = (
    _kimi_k25_forward_edge_cloud_segment
)
