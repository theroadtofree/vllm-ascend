# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend.scheduler_conflicts import validate_pd_separation_scheduler_conflicts


def _vllm_config():
    # scheduler_config no longer carries enable_pd_separation; the flag now
    # lives under additional_config.edge_cloud_config.pd_separation.
    return SimpleNamespace(scheduler_config=SimpleNamespace())


def _ascend_config(
    *,
    pd_separation_enabled: bool = True,
    edge_cloud_enabled: bool = True,
    recompute_scheduler_enable: bool = False,
    slo_limits_for_dynamic_batch: int = -1,
    profiling_chunk_enabled: bool = False,
):
    edge_cloud = SimpleNamespace(
        enabled=edge_cloud_enabled,
        pd_separation=SimpleNamespace(enabled=pd_separation_enabled),
    )
    return SimpleNamespace(
        edge_cloud_config=edge_cloud,
        recompute_scheduler_enable=recompute_scheduler_enable,
        SLO_limits_for_dynamic_batch=slo_limits_for_dynamic_batch,
        profiling_chunk_config=SimpleNamespace(enabled=profiling_chunk_enabled),
    )


@pytest.mark.parametrize(
    ("ascend_config", "expected"),
    [
        pytest.param(
            _ascend_config(recompute_scheduler_enable=True),
            "recompute_scheduler_enable",
            id="recompute-scheduler",
        ),
        pytest.param(
            _ascend_config(slo_limits_for_dynamic_batch=100),
            "SLO_limits_for_dynamic_batch",
            id="dynamic-batch",
        ),
        pytest.param(
            _ascend_config(profiling_chunk_enabled=True),
            "profiling_chunk_config",
            id="profiling-chunk",
        ),
    ],
)
def test_pd_separation_rejects_ascend_scheduler_overrides(ascend_config, expected):
    with pytest.raises(ValueError, match=expected):
        validate_pd_separation_scheduler_conflicts(_vllm_config(), ascend_config)


def test_pd_separation_allows_default_ascend_scheduler_config():
    validate_pd_separation_scheduler_conflicts(_vllm_config(), _ascend_config())


def test_scheduler_conflict_check_is_noop_when_pd_separation_disabled():
    validate_pd_separation_scheduler_conflicts(
        _vllm_config(),
        _ascend_config(
            pd_separation_enabled=False,
            recompute_scheduler_enable=True,
            slo_limits_for_dynamic_batch=100,
            profiling_chunk_enabled=True,
        ),
    )


def test_scheduler_conflict_check_is_noop_when_edge_cloud_disabled():
    validate_pd_separation_scheduler_conflicts(
        _vllm_config(),
        _ascend_config(
            edge_cloud_enabled=False,
            recompute_scheduler_enable=True,
            slo_limits_for_dynamic_batch=100,
            profiling_chunk_enabled=True,
        ),
    )


def test_pd_separation_requires_vllm_pd_scheduler_schema(monkeypatch):
    import vllm_ascend.scheduler_conflicts as conflicts

    monkeypatch.setattr(conflicts, "_vllm_pd_scheduler_schema_available", lambda: False)

    with pytest.raises(ValueError, match="BatchType.*HiddenChannelType.*SchedulerOutput"):
        conflicts.validate_pd_separation_scheduler_conflicts(_vllm_config(), _ascend_config())
