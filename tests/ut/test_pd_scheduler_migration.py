# SPDX-License-Identifier: Apache-2.0

import importlib
from types import SimpleNamespace


def _make_vllm_config(async_scheduling: bool = False):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            async_scheduling=async_scheduling,
            scheduler_cls=None,
            pd_prefill_inflight_limit=1,
        )
    )


def _make_ascend_config(
    *,
    edge_cloud_enabled: bool = True,
    pd_enabled: bool = True,
    next_prefill_prior_enable: bool = True,
):
    pd = SimpleNamespace(
        enabled=pd_enabled,
        next_prefill_prior_enable=next_prefill_prior_enable,
        prefill_inflight_limit=2 if next_prefill_prior_enable else 1,
    )
    edge_cloud = SimpleNamespace(enabled=edge_cloud_enabled, pd_separation=pd)
    return SimpleNamespace(edge_cloud_config=edge_cloud)


def test_pd_separated_scheduler_module_is_owned_by_vllm_ascend():
    module = importlib.import_module("vllm_ascend.core.pd_separated_scheduler")

    assert module.PDSeparatedScheduler.__module__ == "vllm_ascend.core.pd_separated_scheduler"
    assert module.AsyncPDSeparatedScheduler.__module__ == "vllm_ascend.core.pd_separated_scheduler"


def test_passive_scheduler_module_is_owned_by_vllm_ascend():
    module = importlib.import_module("vllm_ascend.core.passive_scheduler")

    assert module.PassiveScheduler.__module__ == "vllm_ascend.core.passive_scheduler"
    assert module.DispatchPolicy.EXPECT_ALTERNATION.value == "expect_alternation"


def test_pd_scheduler_cls_is_set_to_ascend_path():
    from vllm_ascend.platform import NPUPlatform

    vllm_config = _make_vllm_config(async_scheduling=False)
    ascend_config = _make_ascend_config()

    NPUPlatform._configure_pd_separation_scheduler(vllm_config, ascend_config)

    assert vllm_config.scheduler_config.scheduler_cls == (
        "vllm_ascend.core.pd_separated_scheduler.PDSeparatedScheduler"
    )
    # next_prefill_prior_enable=True → limit back-filled to 2.
    assert vllm_config.scheduler_config.pd_prefill_inflight_limit == 2


def test_async_pd_scheduler_cls_is_set_to_ascend_path():
    from vllm_ascend.platform import NPUPlatform

    vllm_config = _make_vllm_config(async_scheduling=True)
    ascend_config = _make_ascend_config()

    NPUPlatform._configure_pd_separation_scheduler(vllm_config, ascend_config)

    assert vllm_config.scheduler_config.scheduler_cls == (
        "vllm_ascend.core.pd_separated_scheduler.AsyncPDSeparatedScheduler"
    )
    assert vllm_config.scheduler_config.pd_prefill_inflight_limit == 2


def test_pd_scheduler_back_fills_default_inflight_limit_when_prior_disabled():
    from vllm_ascend.platform import NPUPlatform

    vllm_config = _make_vllm_config(async_scheduling=False)
    ascend_config = _make_ascend_config(next_prefill_prior_enable=False)

    NPUPlatform._configure_pd_separation_scheduler(vllm_config, ascend_config)

    # next_prefill_prior_enable=False → 1P1D.
    assert vllm_config.scheduler_config.pd_prefill_inflight_limit == 1


def test_pd_scheduler_is_noop_when_edge_cloud_disabled():
    from vllm_ascend.platform import NPUPlatform

    vllm_config = _make_vllm_config(async_scheduling=False)
    ascend_config = _make_ascend_config(edge_cloud_enabled=False)

    NPUPlatform._configure_pd_separation_scheduler(vllm_config, ascend_config)

    assert vllm_config.scheduler_config.scheduler_cls is None
    assert vllm_config.scheduler_config.pd_prefill_inflight_limit == 1


def test_pd_scheduler_is_noop_when_pd_separation_disabled():
    from vllm_ascend.platform import NPUPlatform

    vllm_config = _make_vllm_config(async_scheduling=False)
    ascend_config = _make_ascend_config(pd_enabled=False)

    NPUPlatform._configure_pd_separation_scheduler(vllm_config, ascend_config)

    assert vllm_config.scheduler_config.scheduler_cls is None
    assert vllm_config.scheduler_config.pd_prefill_inflight_limit == 1


def test_install_passive_scheduler_shim_aliases_upstream_import():
    import sys
    import vllm_ascend.patch.platform.patch_serve_headless as patch_serve_headless

    patch_serve_headless._install_ascend_passive_scheduler_shim()

    assert sys.modules["vllm.v1.core.sched.passive_scheduler"] is sys.modules[
        "vllm_ascend.core.passive_scheduler"
    ]
