# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""batch_type -> six-channel mapping and channel -> physical wire resolution.

This module absorbs the channel-selection logic that used to live in the
worker (``_hidden_channel_for``) and the scheduler's data-plane channel
management.  The scheduler only needs to label a batch with its
``BatchType``; the mapping to channels lives here.

Direction convention: ``*_FIRST`` batches travel edge->cloud (UP),
``*_LAST`` batches travel cloud->edge (DOWN).  Both peers map the same
wire to the same channel — e.g. the edge's PREFILL_FIRST send and the
cloud's matching recv are both PREFILL_UP.

Physical resolution is the identity: each CommChannelType owns a
dedicated HCCL communicator + stream whose HiddenChannelType has the
same value ("prefill_up" -> HiddenChannelType.PREFILL_UP).  One
communicator therefore carries exactly one task type in one direction.
"""

from __future__ import annotations

from vllm.v1.core.sched.output import BatchType, HiddenChannelType

from vllm_ascend.distributed.edge_cloud_comm.types import (
    BatchKind,
    CommChannelType,
)

_FIRST_BATCHES = frozenset(
    {
        BatchType.PREFILL_FIRST,
        BatchType.DECODE_FIRST,
        BatchType.PREFILL_DRAFT_FIRST,
        BatchType.DECODE_DRAFT_FIRST,
    }
)
_LAST_BATCHES = frozenset(
    {
        BatchType.PREFILL_LAST,
        BatchType.DECODE_LAST,
        BatchType.PREFILL_DRAFT_LAST,
        BatchType.DECODE_DRAFT_LAST,
    }
)
_DRAFT_BATCHES = frozenset(
    {
        BatchType.PREFILL_DRAFT_FIRST,
        BatchType.PREFILL_DRAFT_LAST,
        BatchType.DECODE_DRAFT_FIRST,
        BatchType.DECODE_DRAFT_LAST,
    }
)
_PREFILL_DRAFT_BATCHES = frozenset(
    {BatchType.PREFILL_DRAFT_FIRST, BatchType.PREFILL_DRAFT_LAST}
)


def is_draft_batch_type(batch_type: BatchType) -> bool:
    """True for any scheduled draft batch (prefill- or decode-phase)."""
    return batch_type in _DRAFT_BATCHES


def is_prefill_phase_draft(batch_type: BatchType) -> bool:
    """True when a scheduled draft batch belongs to a prefill-phase chain."""
    return batch_type in _PREFILL_DRAFT_BATCHES


def kind_for_batch_type(batch_type: BatchType) -> BatchKind:
    """Business kind of a batch.

    The phase of a scheduled draft is encoded in the batch type itself
    (PREFILL_DRAFT_* -> PREFILL_DRAFT pair, DECODE_DRAFT_* -> DECODE pair
    shared with plain decode — the two never co-exist in flight).
    """
    if batch_type in (BatchType.PREFILL_FIRST, BatchType.PREFILL_LAST):
        return BatchKind.PREFILL
    if batch_type in (BatchType.DECODE_FIRST, BatchType.DECODE_LAST):
        return BatchKind.DECODE
    if batch_type in _DRAFT_BATCHES:
        return (
            BatchKind.PREFILL_DRAFT
            if batch_type in _PREFILL_DRAFT_BATCHES
            else BatchKind.DECODE_DRAFT
        )
    raise RuntimeError(f"No BatchKind for batch_type={batch_type}")


def channel_for_direction(kind: BatchKind, up: bool) -> CommChannelType:
    """Logical channel for one wire direction of a business kind.

    ``up=True`` is the edge->cloud request direction, ``up=False`` the
    cloud->edge response.  DECODE and DECODE_DRAFT share the DECODE pair
    (they never co-exist in flight).
    """
    if kind is BatchKind.PREFILL:
        return CommChannelType.PREFILL_UP if up else CommChannelType.PREFILL_DOWN
    if kind is BatchKind.PREFILL_DRAFT:
        return (
            CommChannelType.PREFILL_DRAFT_UP
            if up
            else CommChannelType.PREFILL_DRAFT_DOWN
        )
    return CommChannelType.DECODE_UP if up else CommChannelType.DECODE_DOWN


def channel_for(
    batch_type: BatchType,
    kind: BatchKind | None = None,
) -> CommChannelType:
    """Logical six-channel mapping for a batch.

    Same wire -> same channel on both peers; direction is encoded in the
    batch type (FIRST = up, LAST = down), not in the device role.
    """
    if kind is None:
        kind = kind_for_batch_type(batch_type)
    if batch_type in _FIRST_BATCHES:
        up = True
    elif batch_type in _LAST_BATCHES:
        up = False
    else:
        raise RuntimeError(f"No CommChannelType for batch_type={batch_type}")
    return channel_for_direction(kind, up)


def transport_for(channel: CommChannelType) -> HiddenChannelType:
    """Physical wire of a channel — the identity mapping.

    Each channel owns a dedicated communicator whose HiddenChannelType
    has the same value, so resolution is just a value lookup.  Kept as
    a function so every resolution site shares one implementation (and
    one place to validate).
    """
    transport = getattr(HiddenChannelType, channel.name, None)
    if transport is None:
        raise RuntimeError(
            f"No physical channel group for {channel!r}: the six "
            "directional channels require create_six_channel_groups()"
        )
    return transport
