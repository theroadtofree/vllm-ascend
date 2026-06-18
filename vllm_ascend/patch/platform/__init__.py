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

import os

import vllm_ascend.patch.platform.patch_distributed  # noqa
import vllm_ascend.patch.platform.patch_kv_cache_interface  # noqa
import vllm_ascend.patch.platform.patch_kv_cache_utils  # noqa
import vllm_ascend.patch.platform.patch_mla_prefill_backend  # noqa
import vllm_ascend.patch.platform.patch_pd_scheduler_shim  # noqa
import vllm_ascend.patch.platform.patch_serve_headless  # noqa
from vllm_ascend import envs
from vllm_ascend.utils import is_310p

if not is_310p():
    import vllm_ascend.patch.platform.patch_mamba_config  # noqa
else:
    import vllm_ascend.patch.platform.patch_mamba_config_310  # noqa
import vllm_ascend.patch.platform.patch_minimax_m2_config  # noqa
import vllm_ascend.patch.platform.patch_minimax_usage_accounting  # noqa
import vllm_ascend.patch.platform.patch_glm_tool_call_parser  # noqa
import vllm_ascend.patch.platform.patch_qwen3_5_config  # noqa
import vllm_ascend.patch.platform.patch_torch_accelerator  # noqa
import vllm_ascend.patch.platform.patch_tool_choice_none_content  # noqa

if (
    os.getenv("DYNAMIC_EPLB", "false").lower() in ("true", "1")
    or os.getenv("EXPERT_MAP_RECORD", "false") == "true"
    or os.getenv("VLLM_PP_NON_LEADER_ENGINE_CORE", "0") in ("1", "true", "True")
):
    import vllm_ascend.patch.platform.patch_multiproc_executor  # noqa

# EngineCore PD-separation / edge-cloud / passive-PP hooks. Unconditionally
# loaded — every behavior change inside the patch is gated at runtime by the
# ``ascend_config.edge_cloud_config.pd_separation.enabled`` /
# ``parallel_config.is_edge_node`` / ``envs.VLLM_PP_SCHEDULER_ZMQ_ADDR`` checks,
# so when none of those are on the patched code paths are byte-equivalent to
# upstream vLLM. Loading must be unconditional because the leader (edge)
# process has no env-level signal at platform-init time that PD/edge-cloud is
# requested — the flag is set on
# the VllmConfig only and reaches us via ``EngineCore.__init__``.
import vllm_ascend.patch.platform.patch_engine_core  # noqa

if envs.VLLM_ASCEND_BALANCE_SCHEDULING:
    import vllm_ascend.patch.platform.patch_balance_schedule  # noqa
