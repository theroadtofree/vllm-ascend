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

from typing import Any


def _vllm_pd_scheduler_schema_available() -> bool:
    try:
        from vllm.v1.core.sched.output import BatchType, HiddenChannelType, SchedulerOutput
    except ImportError:
        return False

    required_batch_types = (
        "PD_MIX",
        "PURE_PREFILL",
        "PURE_DECODE",
        "EMPTY",
        "PREFILL_FIRST",
        "PREFILL_LAST",
        "DECODE_FIRST",
        "DECODE_LAST",
        "PREFILL_DRAFT_FIRST",
        "PREFILL_DRAFT_LAST",
        "DECODE_DRAFT_FIRST",
        "DECODE_DRAFT_LAST",
    )
    if any(not hasattr(BatchType, name) for name in required_batch_types):
        return False

    required_channels = HiddenChannelType.SIX_DIRECTIONAL_CHANNELS
    if any(not hasattr(HiddenChannelType, name) for name in required_channels):
        return False

    fields = getattr(SchedulerOutput, "__dataclass_fields__", {})
    return all(
        name in fields
        for name in (
            "batch_type",
            "head_token",
            "hidden_channel",
            "parent_req_id",
            "draft_task_id",
            "draft_step_idx",
            "num_accepted_tokens",
            "valid_sampled_token_count",
        )
    )


def validate_pd_separation_scheduler_conflicts(vllm_config: Any, ascend_config: Any) -> None:
    edge_cloud = getattr(ascend_config, "edge_cloud_config", None)
    if edge_cloud is None or not getattr(edge_cloud, "enabled", False):
        return
    pd = getattr(edge_cloud, "pd_separation", None)
    if pd is None or not getattr(pd, "enabled", False):
        return

    if not _vllm_pd_scheduler_schema_available():
        raise ValueError(
            "edge_cloud_config.pd_separation.enabled requires vLLM PD scheduler "
            "schema: BatchType, HiddenChannelType, and "
            "SchedulerOutput batch/head/channel/scheduled draft fields."
        )

    if getattr(ascend_config, "recompute_scheduler_enable", False):
        raise ValueError(
            "edge_cloud_config.pd_separation.enabled is incompatible with "
            "additional_config.recompute_scheduler_enable. Disable one of them."
        )

    if getattr(ascend_config, "SLO_limits_for_dynamic_batch", -1) != -1:
        raise ValueError(
            "edge_cloud_config.pd_separation.enabled is incompatible with "
            "additional_config.SLO_limits_for_dynamic_batch. Disable one of them."
        )

    profiling_chunk_config = getattr(ascend_config, "profiling_chunk_config", None)
    if getattr(profiling_chunk_config, "enabled", False):
        raise ValueError(
            "edge_cloud_config.pd_separation.enabled is incompatible with "
            "additional_config.profiling_chunk_config.enabled. Disable one of them."
        )
