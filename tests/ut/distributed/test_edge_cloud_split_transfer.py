#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for the edge-cloud split metadata/tensor transport helpers.

The helpers under test split a single ``tensor_dict`` into two cross-WAN
trips: a small CPU-pickle metadata pre-publish (issued BEFORE
``_model_forward``) and a payload-only ``isend`` (issued AFTER it).

Rank-selection invariant under test: in the vllm edge-cloud layout, only
the cross-domain pair (edge NPU0, cloud NPU0) sits in a PP group of size
2; every other rank gets a singleton PP group. So ``pp_group.world_size
== 2`` is the unambiguous "I am the PP boundary" selector. Using
``tp_group.rank_in_group == 0`` instead happens to alias the boundary on
the standard layout but makes non-boundary TP ranks skip the TP-internal
broadcast leg of recv_metadata, which deadlocks the collective.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.parallel_state import TensorMetadata

from vllm_ascend.distributed.parallel_state import (
    edge_cloud_irecv_tensors,
    edge_cloud_isend_tensors,
    edge_cloud_recv_metadata,
    edge_cloud_send_metadata,
    split_tensor_dict_metadata,
)


def _make_groups(*, pp_world=2, tp_world=1, tp_rank=0, pp_rank=0):
    pp = MagicMock()
    pp.world_size = pp_world
    pp.rank_in_group = pp_rank
    pp.ranks = list(range(pp_world)) if pp_world > 1 else [0]
    pp.device_group = MagicMock(name="pp_device_group")
    pp.cpu_group = MagicMock(name="pp_cpu_group")

    tp = MagicMock()
    tp.world_size = tp_world
    tp.rank_in_group = tp_rank
    tp.ranks = list(range(tp_world))
    tp.device_group = MagicMock(name="tp_device_group")
    tp.cpu_group = MagicMock(name="tp_cpu_group")
    return pp, tp


def test_split_tensor_dict_metadata_drops_real_tensors():
    """Sanity-check the public re-export: a tensor key must produce a
    ``TensorMetadata`` entry and scalar keys must pass through verbatim."""
    payload = split_tensor_dict_metadata({
        "hidden_states": torch.zeros(4, 8, dtype=torch.float32),
        "num_tokens": 4,
    })
    keys = {k: v for k, v in payload}
    assert isinstance(keys["hidden_states"], TensorMetadata)
    assert keys["hidden_states"].size == torch.Size([4, 8])
    assert keys["num_tokens"] == 4


def test_send_metadata_fires_only_on_pp_boundary():
    # PP boundary = the only rank whose pp_group has world_size == 2.
    pp, tp = _make_groups(pp_world=2, tp_world=2, tp_rank=0, pp_rank=0)
    payload = [("hidden_states", TensorMetadata("npu", torch.float16, (8, 16)))]

    with (
        patch("vllm_ascend.distributed.parallel_state.get_pp_group", return_value=pp),
        patch("vllm_ascend.distributed.parallel_state.get_tp_group", return_value=tp),
    ):
        edge_cloud_send_metadata(payload)

    pp.send_object.assert_called_once_with(payload, dst=1)


def test_send_metadata_noop_on_non_pp_boundary():
    # A non-boundary cloud TP rank lives in a singleton PP group
    # (pp_group.world_size == 1) — sending would be undefined.
    pp, tp = _make_groups(pp_world=1, tp_world=4, tp_rank=2, pp_rank=0)
    with (
        patch("vllm_ascend.distributed.parallel_state.get_pp_group", return_value=pp),
        patch("vllm_ascend.distributed.parallel_state.get_tp_group", return_value=tp),
    ):
        edge_cloud_send_metadata([("k", 1)])

    pp.send_object.assert_not_called()


def test_recv_metadata_pp_boundary_recvs_then_broadcasts():
    # Cloud PP boundary (pp_world=2): pull from peer, fan out to local TP.
    pp, tp = _make_groups(pp_world=2, tp_world=4, tp_rank=0, pp_rank=1)
    expected = [("hidden_states", TensorMetadata("npu", torch.float16, (8, 16)))]
    pp.recv_object.return_value = expected

    with (
        patch("vllm_ascend.distributed.parallel_state.get_pp_group", return_value=pp),
        patch("vllm_ascend.distributed.parallel_state.get_tp_group", return_value=tp),
    ):
        got = edge_cloud_recv_metadata()

    pp.recv_object.assert_called_once_with(src=0)
    tp.broadcast_object.assert_called_once_with(expected, src=0)
    assert got == expected


def test_recv_metadata_non_boundary_only_takes_broadcast():
    # Cloud non-boundary TP rank: pp_group is a singleton; the data must
    # come exclusively from the TP-internal broadcast. This is the
    # regression case that deadlocked with the previous
    # ``tp_group.rank_in_group == 0`` selector.
    pp, tp = _make_groups(pp_world=1, tp_world=4, tp_rank=3, pp_rank=0)
    expected = [("hidden_states", TensorMetadata("npu", torch.float16, (8, 16)))]
    tp.broadcast_object.return_value = expected

    with (
        patch("vllm_ascend.distributed.parallel_state.get_pp_group", return_value=pp),
        patch("vllm_ascend.distributed.parallel_state.get_tp_group", return_value=tp),
    ):
        got = edge_cloud_recv_metadata()

    pp.recv_object.assert_not_called()
    tp.broadcast_object.assert_called_once_with(None, src=0)
    assert got == expected


def test_recv_metadata_no_tp_no_pp_returns_empty():
    # Single-NPU edge or cloud (TP=1, PP=1): no peer, no broadcast needed.
    pp, tp = _make_groups(pp_world=1, tp_world=1, tp_rank=0, pp_rank=0)
    with (
        patch("vllm_ascend.distributed.parallel_state.get_pp_group", return_value=pp),
        patch("vllm_ascend.distributed.parallel_state.get_tp_group", return_value=tp),
    ):
        got = edge_cloud_recv_metadata()

    pp.recv_object.assert_not_called()
    tp.broadcast_object.assert_not_called()
    assert got == []


def test_isend_tensors_skips_empty_and_emits_no_metadata():
    pp, tp = _make_groups(pp_world=2, tp_world=1, tp_rank=0, pp_rank=0)
    real = torch.zeros(4, 8, dtype=torch.float16)
    empty = torch.empty(0, dtype=torch.float16)
    tensor_dict = {"hidden_states": real, "residual": empty, "step": 7}

    fake_handle = MagicMock(name="isend_handle")
    with (
        patch("vllm_ascend.distributed.parallel_state.get_pp_group", return_value=pp),
        patch("vllm_ascend.distributed.parallel_state.get_tp_group", return_value=tp),
        patch("torch.distributed.isend", return_value=fake_handle) as mock_isend,
    ):
        handles = edge_cloud_isend_tensors(tensor_dict)

    # Exactly one isend (only the non-empty CPU tensor "hidden_states";
    # "residual" is empty, "step" is a plain int).
    assert handles == [fake_handle]
    assert mock_isend.call_count == 1
    pp.send_object.assert_not_called()


def test_isend_tensors_noop_on_non_pp_boundary():
    pp, tp = _make_groups(pp_world=1, tp_world=4, tp_rank=1, pp_rank=0)
    real = torch.zeros(4, 8, dtype=torch.float16)
    with (
        patch("vllm_ascend.distributed.parallel_state.get_pp_group", return_value=pp),
        patch("vllm_ascend.distributed.parallel_state.get_tp_group", return_value=tp),
        patch("torch.distributed.isend") as mock_isend,
    ):
        handles = edge_cloud_isend_tensors({"hidden_states": real})

    assert handles == []
    mock_isend.assert_not_called()


def test_irecv_tensors_calls_buffer_provider_and_returns_handles():
    pp, tp = _make_groups(pp_world=2, tp_world=1, tp_rank=0, pp_rank=1)
    meta = TensorMetadata("cpu", torch.float16, (4, 8))
    payload = [("hidden_states", meta), ("step", 12)]

    pre_alloc = torch.zeros(4, 8, dtype=torch.float16)
    received_keys: list[str] = []

    def buffer_provider(key, m):
        received_keys.append(key)
        assert m is meta
        return pre_alloc

    fake_handle = MagicMock(name="irecv_handle")
    with (
        patch("vllm_ascend.distributed.parallel_state.get_pp_group", return_value=pp),
        patch("vllm_ascend.distributed.parallel_state.get_tp_group", return_value=tp),
        patch("torch.distributed.irecv", return_value=fake_handle) as mock_irecv,
    ):
        tensor_dict, handles, postprocess = edge_cloud_irecv_tensors(
            payload, buffer_provider=buffer_provider
        )

    assert received_keys == ["hidden_states"]
    assert tensor_dict["hidden_states"] is pre_alloc
    assert tensor_dict["step"] == 12
    assert handles == [fake_handle]
    assert mock_irecv.call_count == 1
    assert callable(postprocess[0])
    postprocess[0]()  # TP=1 → no-op branch must not raise


def test_irecv_tensors_non_boundary_skips_irecv_but_keeps_buffer():
    # Non-boundary TP rank: still allocates a buffer to participate in
    # the upcoming TP broadcast, but does NOT issue a cross-domain irecv.
    pp, tp = _make_groups(pp_world=1, tp_world=4, tp_rank=2, pp_rank=0)
    meta = TensorMetadata("cpu", torch.float16, (4, 8))
    payload = [("hidden_states", meta)]
    pre_alloc = torch.zeros(4, 8, dtype=torch.float16)

    with (
        patch("vllm_ascend.distributed.parallel_state.get_pp_group", return_value=pp),
        patch("vllm_ascend.distributed.parallel_state.get_tp_group", return_value=tp),
        patch("torch.distributed.irecv") as mock_irecv,
    ):
        tensor_dict, handles, _ = edge_cloud_irecv_tensors(
            payload, buffer_provider=lambda k, m: pre_alloc
        )

    assert tensor_dict["hidden_states"] is pre_alloc
    assert handles == []
    mock_irecv.assert_not_called()


def test_irecv_tensors_empty_metadata_skips_irecv_but_keeps_buffer():
    pp, tp = _make_groups(pp_world=2, tp_world=1, tp_rank=0, pp_rank=1)
    meta = TensorMetadata("cpu", torch.float16, (0, 8))
    payload = [("hidden_states", meta)]
    empty_buf = torch.empty(0, 8, dtype=torch.float16)

    with (
        patch("vllm_ascend.distributed.parallel_state.get_pp_group", return_value=pp),
        patch("vllm_ascend.distributed.parallel_state.get_tp_group", return_value=tp),
        patch("torch.distributed.irecv") as mock_irecv,
    ):
        tensor_dict, handles, _ = edge_cloud_irecv_tensors(
            payload, buffer_provider=lambda k, m: empty_buf
        )

    assert tensor_dict["hidden_states"] is empty_buf
    assert handles == []
    mock_irecv.assert_not_called()


def test_send_metadata_then_isend_tensors_does_not_call_send_object():
    """Core property: the tensor send after pre-publish must NOT trigger any
    CPU pickle round-trip."""
    pp, tp = _make_groups(pp_world=2, tp_world=1, tp_rank=0, pp_rank=0)
    tensors = {"hidden_states": torch.zeros(2, 4, dtype=torch.float16)}
    payload = split_tensor_dict_metadata(tensors)

    with (
        patch("vllm_ascend.distributed.parallel_state.get_pp_group", return_value=pp),
        patch("vllm_ascend.distributed.parallel_state.get_tp_group", return_value=tp),
        patch("torch.distributed.isend", return_value=MagicMock()),
    ):
        edge_cloud_send_metadata(payload)
        pp.send_object.reset_mock()  # only count post-publish behaviour
        edge_cloud_isend_tensors(tensors)

    pp.send_object.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
