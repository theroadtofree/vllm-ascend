from dataclasses import dataclass
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
from vllm.logger import logger

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

# Edge-cloud pre-computed tensor metadata cache
_EDGE_CLOUD_TENSOR_META: "EdgeCloudTensorMeta | None" = None


@dataclass
class EdgeCloudTensorMeta:
    """Pre-computed tensor metadata for edge-cloud hidden state transfer.

    Avoids sending metadata via send_object/recv_object (pickle+Gloo) by
    computing it locally on both sides from config + SchedulerOutput.
    """
    # List of (key, TensorMetadata) pairs, matching the output of _split_tensor_dict
    metadata_list: list[tuple[str, Any]]
    # Ordered list of tensor keys (same order as metadata_list, only tensor entries)
    tensor_keys: list[str]


def init_edge_cloud_tensor_meta(
    hidden_size: int,
    hidden_dtype: torch.dtype = torch.bfloat16,
    has_residual: bool = True,
):
    """Initialize the pre-computed tensor metadata for edge-cloud transfers.

    Called once during worker initialization. The metadata is used by
    edge_cloud_irecv_tensor_dict to allocate receive buffers without
    requiring an inter-node metadata exchange.

    Args:
        hidden_size: model hidden dimension (from hf_text_config.hidden_size)
        hidden_dtype: torch.dtype derived directly from model_config.dtype
            (equivalent to MindIE's config.torch_dtype from config.json),
            eliminating the need for a separate user-configured dtype string.
        has_residual: dynamically detected — True if the model produces a
            residual tensor in IntermediateTensors (most decoder models do).
    """
    global _EDGE_CLOUD_TENSOR_META

    dtype = hidden_dtype
    device = "npu"

    metadata_list: list[tuple[str, Any]] = [
        ("hidden_states", TensorMetadata(device, dtype, (0, hidden_size))),
    ]
    tensor_keys = ["hidden_states"]

    if has_residual:
        metadata_list.append(
            ("residual", TensorMetadata(device, dtype, (0, hidden_size)))
        )
        tensor_keys.append("residual")

    _EDGE_CLOUD_TENSOR_META = EdgeCloudTensorMeta(
        metadata_list=metadata_list,
        tensor_keys=tensor_keys,
    )
    logger.info(
        "[EdgeCloud] Initialized tensor meta: keys=%s, dtype=%s, hidden_size=%d",
        tensor_keys,
        dtype,
        hidden_size,
    )


def get_edge_cloud_tensor_meta() -> EdgeCloudTensorMeta:
    """Get the pre-computed edge-cloud tensor metadata."""
    assert _EDGE_CLOUD_TENSOR_META is not None, (
        "Edge-cloud tensor meta not initialized. "
        "Call init_edge_cloud_tensor_meta() first."
    )
    return _EDGE_CLOUD_TENSOR_META


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


def edge_cloud_isend_tensor_dict(
    tensor_dict: dict[str, torch.Tensor | Any],
    dst: int | None = None,
    num_tokens: int | None = None,
) -> list[Handle]:
    """Send tensor dict without metadata sync (edge-cloud optimized).

    Skips send_object(metadata_list) — the receiver already knows the
    tensor structure from init_edge_cloud_tensor_meta() and computes
    shapes locally from SchedulerOutput.total_num_scheduled_tokens.

    This eliminates the inter-node pickle+Gloo metadata exchange that
    the standard GroupCoordinator.isend_tensor_dict performs.

    Args:
        tensor_dict: tensors to send. Non-tensor / zero-numel entries are
            skipped, matching the receiver's allocation logic.
        dst: destination rank in PP group (default: next rank).
        num_tokens: when provided, each tensor is sliced to
            ``tensor[:num_tokens]`` along dim-0 before send. This pushes the
            sender/receiver shape-alignment responsibility to the sender so
            that the receiver can safely allocate buffers based on
            ``SchedulerOutput.total_num_scheduled_tokens`` alone (no
            metadata wire transfer needed). The slice is a zero-copy view
            and adds negligible overhead. Must be set whenever the model
            forward may produce padded outputs (cudagraph / SP / DP padding,
            etc.). When ``None``, tensors are sent as-is preserving the
            previous behavior for callers that already guarantee unpadded
            output.
    """
    pp_group = get_pp_group()
    if pp_group.world_size <= 1:
        return []

    if dst is None:
        dst = (pp_group.rank_in_group + 1) % pp_group.world_size

    group = pp_group.device_group

    # Guard against silent key/order drift between sender and receiver.
    # The receiver iterates ec_meta.metadata_list in a fixed order; if the
    # sender's tensor_dict adds, drops, or reorders keys (e.g. a future
    # model patch starts returning an extra entry in IntermediateTensors),
    # the wire payload no longer matches the receiver's pre-allocated
    # buffers �� and because there is no metadata exchange anymore, the
    # mismatch would corrupt data silently or only surface as an HCCL
    # crash. Fail fast here with a precise message instead.
    ec_meta = get_edge_cloud_tensor_meta()
    sender_tensor_keys = [
        k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)
    ]
    assert sender_tensor_keys == ec_meta.tensor_keys, (
        "edge_cloud_isend_tensor_dict: tensor key set/order does not match "
        f"the pre-computed EdgeCloudTensorMeta. sender={sender_tensor_keys}, "
        f"expected={ec_meta.tensor_keys}. If this is a new model, extend "
        "init_edge_cloud_tensor_meta() so both sides agree."
    )

    handles: list[Handle] = []
    for _, value in tensor_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.numel() == 0:
            continue
        # Slice to the receiver-expected length so the wire shape exactly
        # matches the buffer the receiver pre-allocates from num_tokens.
        # This is a view (no copy) and is the cheapest way to defeat
        # cudagraph / SP / DP padding mismatches without a metadata
        # exchange.
        if num_tokens is not None and value.shape[0] > num_tokens:
            value = value[:num_tokens]
        if not value.is_contiguous():
            # isend requires contiguous storage; a non-contiguous slice
            # only happens when upstream code returned a non-standard
            # layout, in which case we materialize once.
            value = value.contiguous()
        handle = torch.distributed.isend(
            value, dst=pp_group.ranks[dst], group=group
        )
        if value.is_cuda:
            value.record_stream(torch.cuda.current_stream(value.device))
        handles.append(handle)

    return handles


def edge_cloud_irecv_tensor_dict(
    num_tokens: int,
    src: int | None = None,
) -> tuple[dict[str, torch.Tensor | Any], list[Handle], list[Callable[[], None]]]:
    """Receive tensor dict without metadata sync (edge-cloud optimized).

    Computes metadata locally from num_tokens + the pre-computed
    EdgeCloudTensorMeta, pre-allocates tensors, then issues irecv
    for each. This eliminates the inter-node pickle+Gloo metadata
    exchange that the standard GroupCoordinator.irecv_tensor_dict
    performs.

    Args:
        num_tokens: total_num_scheduled_tokens from SchedulerOutput
        src: source rank in PP group (default: previous rank)
    """
    pp_group = get_pp_group()
    if not torch.distributed.is_initialized() or pp_group.world_size == 1:
        return {}, [], []

    if src is None:
        src = (pp_group.rank_in_group - 1) % pp_group.world_size

    ec_meta = get_edge_cloud_tensor_meta()
    group = pp_group.device_group

    tensor_dict: dict[str, Any] = {}
    handles: list[Handle] = []
    postprocess: list[Callable[[], None]] = []

    for key, value in ec_meta.metadata_list:
        if isinstance(value, TensorMetadata):
            # Replace the placeholder dim-0 with actual num_tokens
            actual_size = (num_tokens,) + value.size[1:]
            full_tensor = torch.empty(
                actual_size, dtype=value.dtype, device=value.device
            )

            if full_tensor.numel() == 0:
                tensor_dict[key] = full_tensor
                continue

            handle = torch.distributed.irecv(
                full_tensor, src=pp_group.ranks[src], group=group
            )
            handles.append(handle)
            tensor_dict[key] = full_tensor
        else:
            tensor_dict[key] = value

    return tensor_dict, handles, postprocess


def edge_cloud_broadcast_recv(
    num_tokens: int,
) -> tuple[
    dict[str, torch.Tensor | Any] | None,
    list[Handle],
    list[Callable[[], None]],
]:
    """Receive PP tensors and broadcast within the local edge/cloud TP group.

    Uses locally computed metadata instead of receiving it from the sender.
    This eliminates the inter-node pickle+Gloo metadata exchange, while
    still broadcasting metadata within the local TP group (intra-node)
    so that non-NPU0 TP ranks can allocate tensors.
    """
    pp_group = get_pp_group()
    tp_group = get_tp_group()
    is_pp_npu0 = pp_group.world_size == 2
    ec_meta = get_edge_cloud_tensor_meta()

    if is_pp_npu0:
        # PP rank 0: receive tensor data from the other side (edge/cloud)
        # without metadata sync — shapes are computed locally
        tensor_dict, comm_handles, comm_postprocess = edge_cloud_irecv_tensor_dict(
            num_tokens=num_tokens,
        )
        assert tensor_dict is not None, (
            "edge_cloud_broadcast_recv: tensor_dict is None, sender may have failed."
        )

        # Broadcast locally-computed metadata + num_tokens to other TP ranks
        # so they can allocate tensors (this is intra-node, fast)
        ###metadata_list = ec_meta.metadata_list
        ###tp_group.broadcast_object([num_tokens, metadata_list], src=0)

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

    # Non-PP-NPU0 ranks: receive metadata from NPU 0 via TP broadcast,
    # allocate tensors, then broadcast-recv actual data
    ###broadcast_data = tp_group.broadcast_object(None, src=0)
    #recv_num_tokens = broadcast_data[0]
    #metadata_list = broadcast_data[1]
    metadata_list = ec_meta.metadata_list
    recv_num_tokens = num_tokens
    if metadata_list is None:
        metadata_list = []

    recv_tensor_dict: dict[str, torch.Tensor | Any] = {}

    for key, value in metadata_list:
        if isinstance(value, TensorMetadata):
            # Replace placeholder dim-0 with actual num_tokens
            actual_size = (recv_num_tokens,) + value.size[1:]
            tensor = torch.empty(actual_size, dtype=value.dtype, device=value.device)
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
