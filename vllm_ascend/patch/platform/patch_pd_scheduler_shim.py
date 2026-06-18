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

import sys


def install_ascend_pd_scheduler_shims() -> None:
    """Route upstream PD scheduler imports to vllm-ascend implementations.

    vLLM keeps the shared SchedulerOutput schema, while vllm-ascend owns the
    scheduler implementations. Some upstream glue imports the historical vLLM
    module paths lazily, so install aliases before those paths are resolved.
    """
    try:
        import vllm_ascend.core.passive_scheduler as passive_scheduler
        import vllm_ascend.core.pd_separated_scheduler as pd_separated_scheduler
    except ImportError:
        return

    sys.modules["vllm.v1.core.sched.passive_scheduler"] = passive_scheduler
    sys.modules["vllm.v1.core.sched.pd_separated_scheduler"] = pd_separated_scheduler


def install_ascend_passive_engine_core_shims() -> None:
    """Re-expose vllm-ascend PassiveEngineCore + ZMQ channel classes on the
    legacy ``vllm.v1.engine.core`` module path.

    vllm-pdmix's downstream fork used to ship four classes inside
    ``vllm/v1/engine/core.py``:
    ``PassiveEngineCoreProc`` / ``PPSchedulerZmqPublisher`` /
    ``PPSchedulerZmqSubscriber`` / ``PPSchedulerZmqChannel``. The migration
    moves them to :mod:`vllm_ascend.v1.engine.passive_core`. To keep any
    callsite that still does ``from vllm.v1.engine.core import
    PassiveEngineCoreProc`` working, attach them as attributes on the
    upstream module. This is purely additive — upstream attributes are
    untouched.
    """
    try:
        import vllm.v1.engine.core as upstream_core
        from vllm_ascend.v1.engine.passive_core import (
            PassiveEngineCoreProc,
            PPSchedulerZmqChannel,
            PPSchedulerZmqPublisher,
            PPSchedulerZmqSubscriber,
        )
    except ImportError:
        return

    for name, obj in (
        ("PassiveEngineCoreProc", PassiveEngineCoreProc),
        ("PPSchedulerZmqPublisher", PPSchedulerZmqPublisher),
        ("PPSchedulerZmqSubscriber", PPSchedulerZmqSubscriber),
        ("PPSchedulerZmqChannel", PPSchedulerZmqChannel),
    ):
        # Only set if upstream does not already export the symbol — a
        # later upstream that lands these natively must take priority.
        if not hasattr(upstream_core, name):
            setattr(upstream_core, name, obj)


install_ascend_pd_scheduler_shims()
install_ascend_passive_engine_core_shims()
