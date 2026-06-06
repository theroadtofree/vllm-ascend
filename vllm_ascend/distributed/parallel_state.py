from typing import Any, Callable

import torch
from vllm.config import ParallelConfig, get_current_vllm_config
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    Handle,
    TensorMetadata,
    _split_tensor_dict,
    get_pp_group,
    get_tp_group,
    get_world_group,
    init_model_parallel_group,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import enable_dsa_cp_with_layer_shard, flashcomm2_enable

# Currently, mc2 op need their own group coordinator.
_MC2: GroupCoordinator | None = None

# Module specific tensor parallel groups
_MLP_TP: GroupCoordinator | None = None
_OTP: GroupCoordinator | None = None
_LMTP: GroupCoordinator | None = None
_EMBED_TP: GroupCoordinator | None = None

# flashcomm specific groups
_FLASHCOMM2_OTP: GroupCoordinator | None = None
_FLASHCOMM2_ODP: GroupCoordinator | None = None
_FC3_QUANT_X: GroupCoordinator | None = None

# shard_weight across rank groups
_SHARD_WEIGHT: GroupCoordinator | None = None

_P_TP: GroupCoordinator | None = None

_DYNAMIC_EPLB: GroupCoordinator | None = None


def init_ascend_model_parallel(
    parallel_config: ParallelConfig,
):
    if model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    if parallel_config.enable_edge_cloud:
        # Edge-cloud mode does not use the standard uniform rank layout
        # (DP * PP * PCP * TP). Skip MC2 / P_TP / DYNAMIC_EPLB init.
        return
    world_size = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)
    global_tp_size = parallel_config.tensor_parallel_size
    global_dp_size = parallel_config.data_parallel_size
    global_pp_size = parallel_config.pipeline_parallel_size
    global_pcp_size = parallel_config.prefill_context_parallel_size

    # The layout of all ranks: ExternalDP * EP
    # ExternalDP is the data parallel group that is not part of the model,
    # every dp rank can generate independently (in verl integration).
    all_ranks = torch.arange(world_size).reshape(
        -1,
        global_dp_size,
        global_pp_size,
        global_pcp_size,
        global_tp_size,
    )

    pd_tp_ratio = get_ascend_config().pd_tp_ratio
    pd_head_ratio = get_ascend_config().pd_head_ratio
    global _P_TP
    assert _P_TP is None, "distributed prefill tensor parallel group is already initialized"
    prefill_tensor_model_parallel_size = pd_tp_ratio
    # divide alltoall groups
    if pd_head_ratio > 1 and get_current_vllm_config().kv_transfer_config.is_kv_producer:
        num_head_replica = get_ascend_config().num_head_replica
        remote_tp_size = global_tp_size // pd_tp_ratio
        if num_head_replica <= 1:
            group_ranks = all_ranks.view(-1, prefill_tensor_model_parallel_size).unbind(0)
        else:
            group_ranks = all_ranks.clone().view(
                global_dp_size * global_pp_size * global_pcp_size, -1, num_head_replica
            )  # [DP_size, num_head, num_head_replica]
            group_ranks = group_ranks.permute(0, 2, 1)
            group_ranks = group_ranks.reshape(-1, group_ranks.size(-1))  # [DP_size * num_head_replica, num_head]
            alltoall_group_size = group_ranks.size(-1) // remote_tp_size
            group_ranks = group_ranks.unsqueeze(-1).view(
                global_dp_size * global_pp_size * global_pcp_size,
                num_head_replica,
                -1,
                alltoall_group_size,
            )  # [DP_size, num_head_replica, num_alltoall_group, alltoall_group_size]
            group_ranks = group_ranks.reshape(-1, alltoall_group_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        local_rank = get_world_group().local_rank
        num = next((i for i, ranks in enumerate(group_ranks) if local_rank in ranks), None)
        _P_TP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name=f"p_tp_{num}")

    # EP like group ranks
    group_ranks = (
        all_ranks.transpose(1, 2)
        .reshape(
            -1,
            global_dp_size * global_pcp_size * global_tp_size,
        )
        .unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]

    global _MC2
    _MC2 = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name="mc2")

    if get_ascend_config().eplb_config.dynamic_eplb:
        global _DYNAMIC_EPLB
        _DYNAMIC_EPLB = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="dynamic_eplb"
        )

    if get_ascend_config().multistream_overlap_gate:
        global _FC3_QUANT_X
        _FC3_QUANT_X = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="fc3_quant_x"
        )

    if parallel_config.enable_edge_cloud:
        return

    # Initialize fine-grained TP process groups on Ascend for four components:
    # 1. LM Head: output logits projection (`lmhead_tensor_parallel_size`)
    # 2. O Proj: attention output projection (`oproj_tensor_parallel_size`)
    # 3. Embedding: The token embedding table at the input of the model (`embedding_tensor_parallel_size`)
    # 4. MLP: feed-forward network in transformer blocks (`mlp_tensor_parallel_size`)
    _group_cache = {}

    def _create_or_get_group(group_size: int, group_name: str) -> GroupCoordinator:
        if group_size is None:
            return None
        if group_size not in _group_cache:
            rank_grid = torch.arange(world_size).reshape(global_pp_size, global_dp_size, global_tp_size)
            num_chunks = global_dp_size // group_size
            group_ranks = []
            for pp_idx in range(global_pp_size):
                stage_ranks = rank_grid[pp_idx]  # (dp, tp)
                for chunk in range(num_chunks):
                    for tp_idx in range(global_tp_size):
                        group = stage_ranks[chunk * group_size : (chunk + 1) * group_size, tp_idx].tolist()
                        group_ranks.append(group)
            pg = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name=group_name)
            _group_cache[group_size] = pg

        return _group_cache[group_size]

    otp_size = get_ascend_config().finegrained_tp_config.oproj_tensor_parallel_size
    lmhead_tp_size = get_ascend_config().finegrained_tp_config.lmhead_tensor_parallel_size
    embedding_tp_size = get_ascend_config().finegrained_tp_config.embedding_tensor_parallel_size
    mlp_tp_size = get_ascend_config().finegrained_tp_config.mlp_tensor_parallel_size

    global _OTP, _LMTP, _EMBED_TP, _MLP_TP

    if otp_size > 0:
        _OTP = _create_or_get_group(otp_size, "otp")
    if lmhead_tp_size > 0:
        _LMTP = _create_or_get_group(lmhead_tp_size, "lmheadtp")
    if embedding_tp_size > 0:
        _EMBED_TP = _create_or_get_group(embedding_tp_size, "emtp")
    if mlp_tp_size > 0:
        _MLP_TP = _create_or_get_group(mlp_tp_size, "mlptp")

    # TODO: Extract and unify the logic across different communication group.
    flashcomm2_otp_group_ranks = []
    if flashcomm2_enable():
        flashcomm2_otp_size = get_ascend_config().flashcomm2_oproj_tensor_parallel_size
        num_fc2_oproj_tensor_parallel_groups: int = global_tp_size // flashcomm2_otp_size
        global _FLASHCOMM2_OTP
        global _FLASHCOMM2_ODP

        _FLASHCOMM2_OTP = None
        _FLASHCOMM2_ODP = get_tp_group()

        if flashcomm2_otp_size > 1:
            odp_group_ranks: list[list[int]] = [
                [] for _ in range(flashcomm2_otp_size * global_dp_size * global_pp_size)
            ]
            for dp_group_index in range(global_dp_size):
                for pp_group_index in range(global_pp_size):
                    dp_pp_serial_index = dp_group_index * global_pp_size + pp_group_index
                    tp_base_rank = dp_pp_serial_index * global_tp_size
                    odp_base_index = dp_pp_serial_index * flashcomm2_otp_size

                    for i in range(num_fc2_oproj_tensor_parallel_groups):
                        ranks = []
                        for j in range(flashcomm2_otp_size):
                            tp_local_rank = i + j * num_fc2_oproj_tensor_parallel_groups
                            assert tp_local_rank < global_tp_size
                            global_rank = tp_base_rank + tp_local_rank
                            ranks.append(global_rank)

                            odp_group_index = odp_base_index + j
                            odp_group_ranks[odp_group_index].append(global_rank)
                        flashcomm2_otp_group_ranks.append(ranks)

            _FLASHCOMM2_OTP = init_model_parallel_group(
                flashcomm2_otp_group_ranks, get_world_group().local_rank, backend, group_name="flashcomm2_otp"
            )
            _FLASHCOMM2_ODP = init_model_parallel_group(
                odp_group_ranks, get_world_group().local_rank, backend, group_name="flashcomm2_odp"
            )

    def create_shard_weight_group(module_tp_group_ranks: None) -> GroupCoordinator:
        # Argument module_tp_group_ranks: The module specific tensor parallel group.
        # There are three situations.
        # 1. If it is None, then the TP_size of the specific module is 1 and is replicated linear layer.
        # 2. If it is not None, and the module tp_group is same as the global tp_group.
        # 3. If it is not None, and the module tp_group is different from the global tp_group.(eg. flashcomm2_otp)
        group_ranks = []
        pp_group_ranks = all_ranks.transpose(2, 4).reshape(-1, global_pp_size)
        if module_tp_group_ranks is None:
            # If it is None, then the TP_size of this shard weight is 1.
            shard_weight_group_ranks = pp_group_ranks.transpose(0, 1).unbind(0)
            group_ranks = [x.tolist() for x in shard_weight_group_ranks]
        else:
            # combine standard tp group and non-standard tp group to build  shard_weight comm_group
            module_tp_tanspose_ranks = module_tp_group_ranks.transpose(0, 1)
            G = world_size // (global_pp_size * module_tp_group_ranks.size(1))
            shard_weight_group_ranks = torch.stack([t.view(global_pp_size, G) for t in module_tp_tanspose_ranks], dim=1)
            group_ranks = shard_weight_group_ranks.view(-1, G).tolist()
        return init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name="shard_weight")

    # Create shard weight group if enabled
    if get_ascend_config().layer_sharding is not None:
        global _SHARD_WEIGHT
        if flashcomm2_enable():
            if len(flashcomm2_otp_group_ranks) == 0:
                FC2_group_ranks = None
            else:
                FC2_group_ranks = torch.tensor(flashcomm2_otp_group_ranks).squeeze(0)
            _SHARD_WEIGHT = create_shard_weight_group(FC2_group_ranks)
        elif enable_dsa_cp_with_layer_shard():
            # For dsa_cp, all shard layers are replicated.
            _SHARD_WEIGHT = create_shard_weight_group(None)
        else:
            # For standard tp, use global tp group_ranks
            tp_group_ranks = all_ranks.view(-1, global_tp_size)
            _SHARD_WEIGHT = create_shard_weight_group(tp_group_ranks)


def model_parallel_initialized():
    return _MC2 is not None


def get_mc2_group() -> GroupCoordinator:
    assert _MC2 is not None, "mc2 group is not initialized"
    return _MC2


def get_mlp_tp_group() -> GroupCoordinator:
    assert _MLP_TP is not None, "mlp group is not initialized"
    return _MLP_TP


def get_otp_group() -> GroupCoordinator:
    assert _OTP is not None, "output tensor parallel group is not initialized"
    return _OTP


def get_lmhead_tp_group() -> GroupCoordinator:
    assert _LMTP is not None, "lm head tensor parallel group is not initialized"
    return _LMTP


def get_embed_tp_group() -> GroupCoordinator:
    assert _EMBED_TP is not None, "emtp group is not initialized"
    return _EMBED_TP


def get_flashcomm2_otp_group() -> GroupCoordinator:
    return _FLASHCOMM2_OTP


def get_flashcomm2_odp_group() -> GroupCoordinator:
    assert _FLASHCOMM2_ODP is not None, "output data parallel group for flashcomm2 is not initialized"
    return _FLASHCOMM2_ODP


def get_shard_weight_group() -> GroupCoordinator:
    assert _SHARD_WEIGHT is not None, "output shard weight parallel group for flashcomm2 is not initialized"
    return _SHARD_WEIGHT


def get_p_tp_group() -> GroupCoordinator:
    assert _P_TP is not None, "distributed prefill tensor parallel group is not initialized"
    return _P_TP


def get_fc3_quant_x_group() -> GroupCoordinator:
    assert _FC3_QUANT_X is not None, "fc3 quant x group is not initialized"
    return _FC3_QUANT_X


def get_dynamic_eplb_group() -> GroupCoordinator:
    assert _DYNAMIC_EPLB is not None, "Dynamic eplb group is not initialized"
    return _DYNAMIC_EPLB


def destroy_ascend_model_parallel():
    global _MC2
    if _MC2:
        _MC2.destroy()
    _MC2 = None

    global _MLP_TP
    if _MLP_TP:
        _MLP_TP.destroy()
    _MLP_TP = None

    global _LMTP
    if _LMTP:
        _LMTP.destroy()
    _LMTP = None

    global _EMBED_TP
    if _EMBED_TP:
        _EMBED_TP.destroy()
    _EMBED_TP = None

    global _OTP
    if _OTP:
        _OTP.destroy()
    _OTP = None

    global _P_TP
    if _P_TP:
        _P_TP.destroy()
    _P_TP = None

    global _FLASHCOMM2_OTP
    if _FLASHCOMM2_OTP and get_ascend_config().flashcomm2_oproj_tensor_parallel_size != 1:
        _FLASHCOMM2_OTP.destroy()
        _FLASHCOMM2_OTP = None

    global _FLASHCOMM2_ODP
    if _FLASHCOMM2_ODP and get_ascend_config().flashcomm2_oproj_tensor_parallel_size != 1:
        _FLASHCOMM2_ODP.destroy()
        _FLASHCOMM2_ODP = None

    global _SHARD_WEIGHT
    if _SHARD_WEIGHT:
        _SHARD_WEIGHT.destroy()
    _SHARD_WEIGHT = None

    global _FC3_QUANT_X
    if _FC3_QUANT_X:
        _FC3_QUANT_X.destroy()
    _FC3_QUANT_X = None

    global _DYNAMIC_EPLB
    if _DYNAMIC_EPLB:
        _DYNAMIC_EPLB.destroy()
    _DYNAMIC_EPLB = None


def edge_cloud_broadcast_recv() -> tuple[
    dict[str, torch.Tensor | Any] | None,
    list[Handle],
    list[Callable[[], None]],
]:
    """Receive PP tensors and broadcast them within the local edge/cloud TP group."""
    pp_group = get_pp_group()
    tp_group = get_tp_group()
    is_pp_npu0 = pp_group.world_size == 2

    if is_pp_npu0:
        tensor_dict, comm_handles, comm_postprocess = pp_group.irecv_tensor_dict()
        assert tensor_dict is not None, (
            "edge_cloud_broadcast_recv: PP tensor_dict is None, "
            "sender may have failed."
        )

        metadata_list, _ = _split_tensor_dict(tensor_dict)
        tp_group.broadcast_object(metadata_list, src=0)

        def broadcast_postprocess():
            _, tensor_list = _split_tensor_dict(tensor_dict) if tensor_dict else (None, [])
            handles = []
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    continue
                group = tp_group.cpu_group if tensor.is_cpu else tp_group.device_group
                handles.append(
                    torch.distributed.broadcast(
                        tensor, src=tp_group.ranks[0], group=group, async_op=True
                    )
                )
            for handle in handles:
                handle.wait()

        comm_postprocess.append(broadcast_postprocess)
        return tensor_dict, comm_handles, comm_postprocess

    metadata_list = tp_group.broadcast_object(None, src=0)
    if metadata_list is None:
        metadata_list = []
    recv_tensor_dict: dict[str, torch.Tensor | Any] = {}

    for key, value in metadata_list:
        if isinstance(value, TensorMetadata):
            tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
            recv_tensor_dict[key] = tensor
        else:
            recv_tensor_dict[key] = value

    def broadcast_postprocess():
        handles = []
        for tensor in recv_tensor_dict.values():
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                continue
            group = tp_group.cpu_group if tensor.is_cpu else tp_group.device_group
            handles.append(
                torch.distributed.broadcast(
                    tensor, src=tp_group.ranks[0], group=group, async_op=True
                )
            )
        for handle in handles:
            handle.wait()

    return recv_tensor_dict, [], [broadcast_postprocess]


# ---------------------------------------------------------------------------
# Edge-cloud split metadata / tensor transfer (WAN-latency optimization)
#
# Motivation: ``isend_tensor_dict`` couples metadata (TensorMetadata describing
# dtype / shape / device) and the tensor payload on the same critical path: it
# first ``send_object(metadata_list)`` (a synchronous CPU pickle round-trip)
# and only then ``isend`` each tensor. On a cross-WAN edge-cloud link, the
# extra synchronous metadata round-trip happens right after ``_model_forward``
# completes — exactly the moment we want hidden_states to start flowing.
#
# Because every per-step PP tensor's dtype/shape is fully determined before
# ``_model_forward`` runs (it only depends on ``num_tokens_padded`` and the
# model's hidden_size), we can pre-publish the metadata while forward is still
# computing, and once forward finishes, send only the raw tensor data.
#
# The helpers below intentionally mirror the structure of
# ``edge_cloud_broadcast_recv`` / ``isend_tensor_dict`` so callers can switch
# in/out of the split-channel path with minimal change.
# ---------------------------------------------------------------------------


def edge_cloud_send_metadata(
    metadata_payload: list[tuple[str, Any]],
    dst: int | None = None,
) -> None:
    """Send a tensor_dict's metadata-only description on the PP channel.

    ``metadata_payload`` is the list produced by ``_split_tensor_dict`` (i.e.
    each entry is ``(key, TensorMetadata)`` for tensors and ``(key, value)``
    for plain Python scalars). The receiver pre-allocates buffers from this
    list and starts ``irecv`` immediately, so ``edge_cloud_isend_tensors``
    afterwards carries no metadata overhead.

    Only the PP boundary rank (the one whose ``pp_group.world_size == 2``,
    i.e. global rank 0 on edge or rank ``edge_npu_count`` on cloud — the
    same rank that ``isend_tensor_dict`` would have used) actually puts the
    payload on the wire; every other rank is a no-op. The TP-side fan-out
    is the receiver's responsibility (see ``edge_cloud_recv_metadata``).
    """
    pp_group = get_pp_group()
    # Mirror the original ``edge_cloud_broadcast_recv`` selector: in the
    # edge-cloud vLLM layout, only the cross-domain pair (edge NPU0,
    # cloud NPU0) sit in a PP group of size 2. Every other rank gets a
    # singleton PP group (see vllm/distributed/parallel_state.py:1579-1589),
    # so ``world_size != 2`` cleanly identifies "no peer to send to".
    if pp_group.world_size != 2:
        return

    if dst is None:
        dst = (pp_group.rank_in_group + 1) % pp_group.world_size
    pp_group.send_object(metadata_payload, dst=dst)


def edge_cloud_recv_metadata(
    src: int | None = None,
) -> list[tuple[str, Any]]:
    """Receive the metadata published by ``edge_cloud_send_metadata``.

    Returns the metadata_list ``[(key, TensorMetadata | value), ...]`` from
    which the caller is expected to allocate or reuse receive buffers and then
    invoke ``edge_cloud_irecv_tensors`` to start the actual tensor transfer.

    Uses the same rank selector as the original
    ``edge_cloud_broadcast_recv``: only the PP boundary rank actually pulls
    the cross-domain payload off the wire and then fans it out via the
    TP-internal ``broadcast_object``. Every other rank on the same side
    just takes that broadcast.

    This split is critical for correctness — TP is a collective, so all
    TP ranks MUST enter the broadcast together. Using
    ``tp_group.rank_in_group == 0`` as the selector would happen to match
    the PP boundary in the standard layout, but it also makes non-boundary
    TP ranks skip the broadcast leg entirely, which deadlocks the
    collective. ``pp_group.world_size == 2`` is the unambiguous signal.
    """
    pp_group = get_pp_group()
    tp_group = get_tp_group()

    if pp_group.world_size == 2:
        # PP boundary: pull the metadata across the WAN, then publish it
        # locally so non-boundary TP ranks on this side can size their
        # receive buffers identically.
        if src is None:
            src = (pp_group.rank_in_group - 1) % pp_group.world_size
        metadata_payload = pp_group.recv_object(src=src)
        if tp_group.world_size > 1:
            tp_group.broadcast_object(metadata_payload, src=0)
    else:
        # Non-boundary TP rank: just take the broadcast.
        if tp_group.world_size <= 1:
            return []
        metadata_payload = tp_group.broadcast_object(None, src=0)

    if metadata_payload is None:
        metadata_payload = []
    return metadata_payload


def edge_cloud_isend_tensors(
    tensor_dict: dict[str, torch.Tensor | Any],
    dst: int | None = None,
) -> list[Handle]:
    """Send the tensor payload of a tensor_dict WITHOUT any metadata exchange.

    Assumes ``edge_cloud_send_metadata`` was already called earlier in the
    step with a payload describing exactly the same tensor keys / dtypes /
    shapes that appear in ``tensor_dict``. Only the PP boundary rank
    (``pp_group.world_size == 2``) puts the tensor on the wire; every other
    rank is a no-op (the receiver fans the tensors out via TP-internal
    broadcast in ``edge_cloud_irecv_tensors``).
    """
    pp_group = get_pp_group()
    if pp_group.world_size != 2:
        return []

    if dst is None:
        dst = (pp_group.rank_in_group + 1) % pp_group.world_size

    handles: list[Handle] = []
    group = pp_group.device_group
    metadata_group = pp_group.cpu_group
    for _, tensor in tensor_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.numel() == 0:
            continue
        comm_group = metadata_group if tensor.is_cpu else group
        handle = torch.distributed.isend(
            tensor, dst=pp_group.ranks[dst], group=comm_group
        )
        if tensor.is_cuda:
            tensor.record_stream(torch.cuda.current_stream(tensor.device))
        handles.append(handle)
    return handles


def edge_cloud_irecv_tensors(
    metadata_payload: list[tuple[str, Any]],
    buffer_provider: Callable[[str, TensorMetadata], torch.Tensor],
    src: int | None = None,
) -> tuple[
    dict[str, torch.Tensor | Any],
    list[Handle],
    list[Callable[[], None]],
]:
    """Start an async cross-domain tensor receive using pre-known metadata.

    ``buffer_provider(key, TensorMetadata)`` is called for every tensor key
    and must return a contiguous device tensor of matching dtype/shape (the
    intent is to hand back a slice of the runner's pre-allocated
    ``self.intermediate_tensors`` buffer to avoid an extra ``torch.empty``).
    Non-tensor metadata entries are forwarded verbatim.

    Returns ``(tensor_dict, comm_handles, comm_postprocess)``. The TP-internal
    broadcast is folded into ``comm_postprocess`` so the receiver behaves the
    same way as the existing ``edge_cloud_broadcast_recv`` (use with
    ``AsyncIntermediateTensors`` for lazy wait).

    ``is_pp_boundary`` is selected via ``pp_group.world_size == 2`` (the
    same selector ``edge_cloud_broadcast_recv`` uses). Non-boundary TP
    ranks on the same side allocate matching buffers but skip the
    cross-domain ``irecv``; they later receive the data through the
    TP-internal broadcast inside the postprocess callable.
    """
    pp_group = get_pp_group()
    tp_group = get_tp_group()
    is_pp_boundary = pp_group.world_size == 2

    if src is None and is_pp_boundary:
        src = (pp_group.rank_in_group - 1) % pp_group.world_size

    tensor_dict: dict[str, torch.Tensor | Any] = {}
    handles: list[Handle] = []

    group = pp_group.device_group if is_pp_boundary else None
    metadata_group = pp_group.cpu_group if is_pp_boundary else None

    for key, value in metadata_payload:
        if isinstance(value, TensorMetadata):
            tensor = buffer_provider(key, value)
            tensor_dict[key] = tensor
            if tensor.numel() == 0:
                continue
            if is_pp_boundary:
                comm_group = metadata_group if tensor.is_cpu else group
                handles.append(
                    torch.distributed.irecv(
                        tensor, src=pp_group.ranks[src], group=comm_group
                    )
                )
        else:
            tensor_dict[key] = value

    def tp_broadcast_postprocess() -> None:
        # After the cross-domain irecv completes on rank 0, fan the tensors
        # out across the local TP group so every rank sees the same data.
        if tp_group.world_size <= 1:
            return
        bcast_handles = []
        for tensor in tensor_dict.values():
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                continue
            bcast_group = (
                tp_group.cpu_group if tensor.is_cpu else tp_group.device_group
            )
            bcast_handles.append(
                torch.distributed.broadcast(
                    tensor,
                    src=tp_group.ranks[0],
                    group=bcast_group,
                    async_op=True,
                )
            )
        for handle in bcast_handles:
            handle.wait()

    return tensor_dict, handles, [tp_broadcast_postprocess]


def split_tensor_dict_metadata(
    tensor_dict: dict[str, torch.Tensor | Any],
) -> list[tuple[str, Any]]:
    """Public re-export of ``_split_tensor_dict`` returning only the metadata
    list. Convenience wrapper so callers can construct the payload from a real
    tensor_dict (e.g. when the sender already has the future-shape buffers
    around) without importing vllm-internal symbols.
    """
    metadata_list, _ = _split_tensor_dict(tensor_dict)
    return metadata_list
