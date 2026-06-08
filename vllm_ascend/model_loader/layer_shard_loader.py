#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import logger
from vllm.model_executor.models.utils import PPMissingLayer


@dataclass
class EdgeCloudLayerPlan:
    role: str
    total_layers: int
    k: int | list[int] | tuple[int, int] = 1
    mode: str = "head_tail"

    def __post_init__(self):
        if isinstance(self.k, (list, tuple)) and len(self.k) == 2:
            self.head_k = int(self.k[0])
            self.tail_k = int(self.k[1])
        else:
            self.head_k = self.tail_k = int(self.k)

        if self.mode not in ("head_tail", "embedding_only"):
            raise ValueError(f"mode must be 'head_tail' or 'embedding_only', got {self.mode}")
        if self.mode == "embedding_only":
            self.head_k = 0
            self.tail_k = 0

        if self.role not in ("edge", "cloud"):
            raise ValueError(f"role must be edge or cloud, got {self.role}")
        if self.mode == "head_tail":
            if self.head_k <= 0 or self.tail_k <= 0:
                raise ValueError("head/tail layer counts must be positive in 'head_tail' mode")
            if self.head_k + self.tail_k >= self.total_layers:
                raise ValueError(
                    "edge head/tail layers must leave at least one cloud layer: "
                    f"head_k={self.head_k}, tail_k={self.tail_k}, "
                    f"total_layers={self.total_layers}"
                )

    def get_local_layers(self) -> set[int]:
        if self.mode == "embedding_only":
            if self.role == "edge":
                return set()
            return set(range(self.total_layers))
        if self.role == "edge":
            return set(range(self.head_k)) | set(
                range(self.total_layers - self.tail_k, self.total_layers)
            )
        return set(range(self.head_k, self.total_layers - self.tail_k))

    def get_released_layers(self) -> set[int]:
        return set(range(self.total_layers)) - self.get_local_layers()


class LayerShardLoader:
    """Replace non-local layers with PPMissingLayer before loading weights."""

    @staticmethod
    def _get_language_model(model: nn.Module) -> nn.Module:
        if (
            hasattr(model, "language_model")
            and hasattr(model.language_model, "model")
            and hasattr(model.language_model.model, "layers")
        ):
            return model.language_model
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model
        raise ValueError(
            "Model must expose transformer layers via model.layers or "
            f"language_model.model.layers, got {type(model).__name__}"
        )

    @staticmethod
    def _get_transformer_model(model: nn.Module) -> nn.Module:
        return LayerShardLoader._get_language_model(model).model

    @classmethod
    def apply_sharding(
        cls,
        model: nn.Module,
        layer_plan: EdgeCloudLayerPlan,
        compilation_config: Any = None,
    ) -> None:
        language_model = cls._get_language_model(model)
        transformer_model = cls._get_transformer_model(model)
        layers = transformer_model.layers
        local_layers = layer_plan.get_local_layers()

        converted = 0
        for i in range(len(layers)):
            if i not in local_layers and not isinstance(layers[i], PPMissingLayer):
                old_layer = layers[i]
                layers[i] = PPMissingLayer()
                del old_layer
                converted += 1

        if layer_plan.role == "cloud":
            for module_name in ("embed_tokens", "norm"):
                module = getattr(transformer_model, module_name, None)
                if module is not None and not isinstance(module, PPMissingLayer):
                    setattr(transformer_model, module_name, PPMissingLayer())
            lm_head = getattr(language_model, "lm_head", None)
            if lm_head is not None and not isinstance(lm_head, PPMissingLayer):
                language_model.lm_head = PPMissingLayer()
            # Cloud side does not need vision components for VL models.
            for attr in ("vision_tower", "multi_modal_projector", "mm_projector"):
                module = getattr(model, attr, None)
                if module is not None and not isinstance(module, PPMissingLayer):
                    setattr(model, attr, PPMissingLayer())
                    logger.info(
                        "[LayerShardLoader] Replaced %s with PPMissingLayer on cloud",
                        attr,
                    )

        if compilation_config is not None:
            cls._clean_compilation_config(model, compilation_config)

        logger.info(
            "[LayerShardLoader] role=%s head_k=%d tail_k=%d kept_layers=%s "
            "skipped_layers=%s converted=%d",
            layer_plan.role,
            layer_plan.head_k,
            layer_plan.tail_k,
            sorted(local_layers),
            sorted(layer_plan.get_released_layers()),
            converted,
        )
        cls.validate_sharding(model, layer_plan)

    @classmethod
    def _clean_compilation_config(cls, model: nn.Module, compilation_config: Any) -> None:
        model_module_ids = {id(module) for _, module in model.named_modules()}
        removed_prefixes: list[str] = []
        for prefix in list(compilation_config.static_forward_context.keys()):
            module = compilation_config.static_forward_context[prefix]
            if id(module) not in model_module_ids:
                del compilation_config.static_forward_context[prefix]
                removed_prefixes.append(prefix)

        if removed_prefixes:
            logger.info(
                "[LayerShardLoader] Removed %d stale static_forward_context "
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

    @classmethod
    def validate_sharding(cls, model: nn.Module, layer_plan: EdgeCloudLayerPlan) -> None:
        layers = cls._get_transformer_model(model).layers
        local_layers = layer_plan.get_local_layers()
        conflict = False
        for idx, layer in enumerate(layers):
            is_missing = isinstance(layer, PPMissingLayer)
            should_be_local = idx in local_layers
            if is_missing == should_be_local:
                logger.error(
                    "[LayerShardLoader] unexpected layer state: layer=%d "
                    "is_missing=%s should_be_local=%s role=%s",
                    idx,
                    is_missing,
                    should_be_local,
                    layer_plan.role,
                )
                conflict = True
        if conflict:
            raise RuntimeError(
                "Edge-cloud layer sharding validation failed. Some local layers "
                "are missing or some remote layers are still real."
            )
        if not conflict:
            logger.info(
                "[LayerShardLoader] Sharding validation passed for %s side.",
                layer_plan.role,
            )

    @staticmethod
    def release_layer_weights(layer: nn.Module) -> None:
        for param in layer.parameters(recurse=False):
            if param is not None and param.data is not None:
                param.data = torch.empty(0, device=param.device, dtype=param.dtype)
        for buf in layer.buffers(recurse=False):
            if buf is not None and buf.data is not None:
                buf.data = torch.empty(0, device=buf.device, dtype=buf.dtype)
