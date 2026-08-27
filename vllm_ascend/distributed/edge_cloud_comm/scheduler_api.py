# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Scheduler ⇄ comm-layer bidirectional interface (design doc section 8.3).

① scheduler -> comm (recv-request injection): the scheduler hands a recv
   request to the comm layer as early as schedule time, pushing the irecv
   lead time from "the worker reaches the consume point" to "SchedulerOutput
   issued".  Two realizations share the one schema defined here:

   * cross-process (multiproc executor): the scheduler emits a recv-hint
     (plain dict, pickle-friendly) on the sideband MQ; the worker-side
     comm thread turns it into a ``submit_recv`` call.  The live producer
     is the arrival-time pre-posting path (``PassiveEngineCore`` / edge
     ``EngineCore`` patch -> hint MQ -> ``NPUWorker.start_early_irecv``;
     schema in :mod:`.scheduler_link`); the CHER dispatch-time producer
     has been retired.
   * same process (uni-proc / in-process executor): the scheduler may call
     ``EdgeCloudCommService.submit_recv`` directly with a request built by
     :func:`recv_request_from_hint` / :func:`build_recv_request`, and hand
     the CommFuture to the worker alongside the SchedulerOutput.

② comm -> scheduler (completion feedback): the ``SchedulerCommSink``
   protocol (service.py).  ``poll_completions`` at the worker loop head
   drives it.  This period ships :class:`LoggingSchedulerCommSink`, a
   no-op beyond debug logging, so the chain end exists; the cross-process
   transport that would let the *scheduler process* consume completions
   (for event-driven tail scheduling, design doc section 6) is future
   work.

This module must stay import-light (scheduler processes import it): no
torch, no wire-layer imports.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import logger
from vllm.v1.core.sched.output import HiddenChannelType

from vllm_ascend.distributed.edge_cloud_comm.mapping import (
    channel_for,
    kind_for_batch_type,
)
from vllm_ascend.distributed.edge_cloud_comm.types import (
    BatchKind,
    CommChannelType,
    CommRequest,
    CommResult,
)

# ------------------------------------------------------------------ #
# ① scheduler -> comm: recv-request injection                         #
# ------------------------------------------------------------------ #


def build_recv_request(
    *,
    batch_type: Any,
    num_tokens: int,
    seqno: int | None = None,
    sp_chunk: bool = False,
    include_mrope: bool = True,
    src_dst: int | None = None,
) -> CommRequest:
    """Canonical recv-request builder (same-process schedulers and the
    worker's own consume points share this so field assembly never drifts).
    """
    kind = kind_for_batch_type(batch_type)
    return CommRequest(
        channel=channel_for(batch_type, kind),
        op="recv",
        kind=kind,
        num_tokens=num_tokens,
        seqno=seqno,
        sp_chunk=sp_chunk,
        include_mrope=include_mrope,
        src_dst=src_dst,
    )


def make_recv_hint(
    *,
    head_token: str,
    hidden_channel: HiddenChannelType | None,
    num_tokens: int,
    has_mrope: bool = True,
) -> dict[str, Any]:
    """Canonical recv-hint payload for the scheduler -> worker sideband MQ.

    Plain pickle-friendly dict; the schema lives here so the producer
    (scheduler side) and the consumer (worker guard thread) can never
    drift apart.  ``hidden_channel`` is a legacy pool label from the
    old transport-pinning scheme — still emitted for schema stability,
    ignored by the consumer (the physical wire is derived from the
    channel alone now).
    """
    return {
        "head_token": head_token,
        "hidden_channel": (
            hidden_channel.value if hidden_channel is not None else None
        ),
        "num_tokens": num_tokens,
        "has_mrope": has_mrope,
    }


def recv_request_from_hint(
    hint: dict[str, Any], *, sp_chunk: bool
) -> tuple[CommRequest | None, str | None]:
    """Parse and validate a recv-hint into a CommRequest.

    Hints always describe the prefill hidden recv (PREFILL_UP); the
    ``sp_chunk`` flag is caller-computed because it depends on worker-side
    model state.  The legacy ``hidden_channel`` hint field (a pool label
    from the old transport-pinning scheme) is accepted but ignored: the
    physical wire is now derived from the channel alone.  Returns
    ``(request, head_token)``; ``request`` is None (with a warning
    logged) for incomplete/malformed hints.
    """
    head_token = hint.get("head_token")
    if not head_token:
        return None, None
    num_tokens = hint.get("num_tokens")
    if num_tokens is None:
        logger.warning(
            "[edge-cloud-comm] recv-hint incomplete %s, skipping.",
            {
                k: hint.get(k)
                for k in ("head_token", "hidden_channel", "num_tokens")
            },
        )
        return None, head_token
    return (
        CommRequest(
            channel=CommChannelType.PREFILL_UP,
            op="recv",
            kind=BatchKind.PREFILL,
            num_tokens=num_tokens,
            sp_chunk=sp_chunk,
            include_mrope=hint.get("has_mrope", True),
        ),
        head_token,
    )


# ------------------------------------------------------------------ #
# ② comm -> scheduler: completion feedback                            #
# ------------------------------------------------------------------ #


class LoggingSchedulerCommSink:
    """No-op ``SchedulerCommSink``: debug-logs each completion.

    Registered once per worker process so the feedback chain has a live
    end-to-end path (submit -> reap -> sink) from day one; swapping in a
    real cross-process reporter later touches only this class.
    """

    def on_comm_complete(
        self,
        channel: CommChannelType,
        kind: BatchKind,
        result: CommResult,
    ) -> None:
        logger.debug(
            "[edge-cloud-comm] comm complete: channel=%s kind=%s status=%s",
            channel.value,
            kind.value,
            result.status.value,
        )
