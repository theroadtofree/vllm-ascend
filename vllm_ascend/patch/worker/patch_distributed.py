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
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

import logging
import pickle
from functools import wraps
from typing import Any, Callable, cast

import torch
import vllm
from torch.distributed import Backend
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    TensorMetadata,
    _get_unique_name,
    _register_group,
    _split_tensor_dict,
)

from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator
from vllm_ascend.patch.worker._hccl_pg_registry import HcclPgRegistry, make_hccl_pg_key
from vllm_ascend.utils import create_hccl_pg_options

_HCCL_PG_REGISTRY = HcclPgRegistry()
logger = logging.getLogger(__name__)


def _normalize_backend(backend: str | Backend) -> str:
    return str(backend)


def _resolve_reuse_domain(group_name: str) -> str:
    group_base_name = group_name.split(":")[0]
    if "eplb" in group_base_name or group_base_name == "mc2":
        return group_base_name
    return "shared"


def _create_device_group(
    ranks: list[int],
    backend: str,
    hccl_pg_options: object,
):
    return torch.distributed.new_group(
        ranks,
        backend=backend,
        pg_options=hccl_pg_options,
    )


def _acquire_hccl_group(
    *,
    ranks: list[int],
    backend: str,
    hccl_pg_options: object,
    reuse_domain: str,
):
    # Coordinator construction must remain process-serial and globally ordered:
    # new_group is collective, and the registry only deduplicates equivalent
    # HCCL groups within that ordering contract. It is not a concurrent PG factory.
    hccl_key = make_hccl_pg_key(ranks, backend, hccl_pg_options, reuse_domain)
    device_group = _HCCL_PG_REGISTRY.acquire(
        ranks=ranks,
        backend=backend,
        pg_options=hccl_pg_options,
        reuse_domain=reuse_domain,
        create_fn=lambda: _create_device_group(ranks, backend, hccl_pg_options),
    )
    return device_group, hccl_key


def _wrap_destroy_distributed_environment(destroy_fn):
    if getattr(cast(Any, destroy_fn), "_hccl_registry_clearing_wrapped", False) is True:
        return destroy_fn

    @wraps(destroy_fn)
    def wrapped(*args, **kwargs):
        try:
            return destroy_fn(*args, **kwargs)
        finally:
            _HCCL_PG_REGISTRY.clear()

    cast(Any, wrapped)._hccl_registry_clearing_wrapped = True
    return wrapped


def _patch_destroy_distributed_environment():
    destroy_fn = _wrap_destroy_distributed_environment(vllm.distributed.parallel_state.destroy_distributed_environment)
    vllm.distributed.parallel_state.destroy_distributed_environment = destroy_fn
    vllm.distributed.destroy_distributed_environment = destroy_fn


class GroupCoordinatorPatch(GroupCoordinator):
    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str | Backend,
        use_device_communicator: bool,  # whether to use device communicator
        use_message_queue_broadcaster: bool = False,
        group_name: str | None = None,
    ):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        # Store all group_ranks so that create_alternate_groups can
        # iterate over every subgroup — torch.distributed.new_group
        # is a collective on the default group and must be called by
        # every rank, even for subgroups it does not belong to.
        self._all_group_ranks = group_ranks

        self.backend = _normalize_backend(torch_distributed_backend)
        self._acquired_hccl_keys = []
        self._unshared_hccl_groups = []
        self.use_device_communicator = use_device_communicator
        self.device_communicator = None
        self.mq_broadcaster = None
        self.cpu_group = None
        self.device_group = None
        self.device = None
        self.use_custom_op_call = True
        self.use_cpu_custom_send_recv = False

        reuse_domain = _resolve_reuse_domain(group_name)

        try:
            for ranks in group_ranks:
                hccl_pg_options = create_hccl_pg_options(group_name)
                device_group, hccl_key = _acquire_hccl_group(
                    ranks=ranks,
                    backend=self.backend,
                    hccl_pg_options=hccl_pg_options,
                    reuse_domain=reuse_domain,
                )
                if hccl_key is not None:
                    self._acquired_hccl_keys.append(hccl_key)
                elif self.backend == "hccl" and self.rank in ranks:
                    self._unshared_hccl_groups.append(device_group)

                # a group with `gloo` backend, to allow direct coordination between
                # processes through the CPU.
                cpu_group = torch.distributed.new_group(ranks, backend="gloo")
                if self.rank in ranks:
                    self.ranks = ranks
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    self.device_group = device_group
                    self.cpu_group = cpu_group

            assert self.cpu_group is not None
            assert self.device_group is not None

            # Alternate device/cpu groups for dual-channel PP communication.
            # When set, these provide a second independent communication channel
            # over the same ranks. Used in PP to separate decode from
            # non-decode traffic.
            self.alt_device_group: torch.distributed.ProcessGroup | None = None
            self.alt_cpu_group: torch.distributed.ProcessGroup | None = None
            # Phase6 hidden data-plane channels. The default device/cpu groups
            # are PREFILL_1, the legacy alt groups are DECODE, and the extra
            # hidden groups below are PREFILL_2.
            self.prefill2_device_group: torch.distributed.ProcessGroup | None = None
            self.prefill2_cpu_group: torch.distributed.ProcessGroup | None = None

            self.device = torch.npu.current_device()
            if use_device_communicator and self.world_size > 1:
                self.device_communicator = NPUCommunicator(
                    cpu_group=self.cpu_group,
                    device=self.device,
                    device_group=self.device_group,
                    unique_name=self.unique_name,
                )

            from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

            if use_message_queue_broadcaster and self.world_size > 1:
                self.mq_broadcaster = MessageQueue.create_from_process_group(
                    self.cpu_group,
                    1 << 22,
                    6,
                )
        except Exception:
            try:
                self.destroy()
            except Exception:
                logger.exception("Failed to clean up partially initialized GroupCoordinatorPatch")
            raise

    def destroy(self):
        cpu_group = getattr(self, "cpu_group", None)
        if cpu_group is not None:
            torch.distributed.destroy_process_group(cpu_group)
        if hasattr(self, "cpu_group"):
            del self.cpu_group

        if hasattr(self, "_acquired_hccl_keys"):
            for hccl_key in reversed(self._acquired_hccl_keys):
                _HCCL_PG_REGISTRY.release(hccl_key)
            self._acquired_hccl_keys = []

        if hasattr(self, "_unshared_hccl_groups"):
            for device_group in reversed(self._unshared_hccl_groups):
                torch.distributed.destroy_process_group(device_group)
            self._unshared_hccl_groups = []

        device_group = getattr(self, "device_group", None)
        if device_group is not None and self.backend != "hccl":
            torch.distributed.destroy_process_group(device_group)
        if hasattr(self, "device_group"):
            del self.device_group

        device_communicator = getattr(self, "device_communicator", None)
        if device_communicator is not None:
            device_communicator.destroy()
            self.device_communicator = None

        alt_cpu_group = getattr(self, "alt_cpu_group", None)
        if alt_cpu_group is not None:
            torch.distributed.destroy_process_group(alt_cpu_group)
            self.alt_cpu_group = None

        alt_device_group = getattr(self, "alt_device_group", None)
        if alt_device_group is not None:
            torch.distributed.destroy_process_group(alt_device_group)
            self.alt_device_group = None

        prefill2_cpu_group = getattr(self, "prefill2_cpu_group", None)
        if prefill2_cpu_group is not None:
            torch.distributed.destroy_process_group(prefill2_cpu_group)
            self.prefill2_cpu_group = None

        prefill2_device_group = getattr(self, "prefill2_device_group", None)
        if prefill2_device_group is not None:
            torch.distributed.destroy_process_group(prefill2_device_group)
            self.prefill2_device_group = None

        if getattr(self, "mq_broadcaster", None) is not None:
            self.mq_broadcaster = None

    def create_alternate_groups(
        self,
        torch_distributed_backend: str | Backend,
    ) -> None:
        """Create alternate device and cpu groups over the same ranks.

        Must be called collectively by all ranks in the **default** group
        (i.e. every rank that participates in ``torch.distributed``), because
        ``torch.distributed.new_group`` is a collective operation on the
        default group. After calling this, communication methods can use
        ``use_alt_group=True`` to route through the alternate
        communication channel.
        """
        assert self.alt_device_group is None, (
            "Alternate groups already created"
        )
        hccl_pg_options = create_hccl_pg_options("pp_alt")
        self_alt_device_group = None
        self_alt_cpu_group = None
        # Iterate over ALL subgroups so that every rank participates in
        # every new_group call (required because new_group is collective
        # on the default group).  Only save the group this rank belongs to.
        for ranks in self._all_group_ranks:
            alt_device_group = torch.distributed.new_group(
                ranks,
                backend=torch_distributed_backend,
                pg_options=hccl_pg_options,
            )
            alt_cpu_group = torch.distributed.new_group(
                ranks, backend="gloo"
            )
            if self.rank in ranks:
                self_alt_device_group = alt_device_group
                self_alt_cpu_group = alt_cpu_group
        assert self_alt_device_group is not None
        assert self_alt_cpu_group is not None
        self.alt_device_group = self_alt_device_group
        self.alt_cpu_group = self_alt_cpu_group

    def create_hidden_channel_groups(
        self,
        torch_distributed_backend: str | Backend,
    ) -> None:
        """Create the extra Phase6 PREFILL_2 group.

        The default pp group is PREFILL_1 and the existing alternate group is
        DECODE.  This method adds PREFILL_2 as the third independent data-plane
        channel over the same ranks.
        """
        assert self.prefill2_device_group is None, (
            "PREFILL_2 hidden channel group already created"
        )
        hccl_pg_options = create_hccl_pg_options("pp_prefill2")
        prefill2_device_group = None
        prefill2_cpu_group = None
        for ranks in self._all_group_ranks:
            device_group = torch.distributed.new_group(
                ranks,
                backend=torch_distributed_backend,
                pg_options=hccl_pg_options,
            )
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                prefill2_device_group = device_group
                prefill2_cpu_group = cpu_group
        assert prefill2_device_group is not None
        assert prefill2_cpu_group is not None
        self.prefill2_device_group = prefill2_device_group
        self.prefill2_cpu_group = prefill2_cpu_group

    def _hidden_channel_groups(self, channel: Any):
        value = getattr(channel, "value", channel)
        if value == "prefill_1":
            return self.device_group, self.cpu_group
        if value == "decode":
            assert self.alt_device_group is not None
            assert self.alt_cpu_group is not None
            return self.alt_device_group, self.alt_cpu_group
        if value == "prefill_2":
            assert self.prefill2_device_group is not None
            assert self.prefill2_cpu_group is not None
            return self.prefill2_device_group, self.prefill2_cpu_group
        raise ValueError(f"Unknown hidden channel: {channel}")

    def send_object_on_hidden_channel(
        self, obj: Any, dst: int, channel: Any
    ) -> None:
        """Synchronous send of a pickled object (used by tests/fallback)."""
        _, cpu_group = self._hidden_channel_groups(channel)
        object_tensor = torch.frombuffer(
            bytearray(pickle.dumps(obj)), dtype=torch.uint8
        )
        size_tensor = torch.tensor(
            [object_tensor.numel()], dtype=torch.long, device="cpu"
        )
        torch.distributed.send(size_tensor, dst=self.ranks[dst], group=cpu_group)
        torch.distributed.send(object_tensor, dst=self.ranks[dst], group=cpu_group)

    def send_object_on_hidden_channel_async(
        self, obj: Any, dst: int, channel: Any
    ) -> list[Any]:
        """Asynchronous send of a pickled object; returns isend handles."""
        _, cpu_group = self._hidden_channel_groups(channel)
        object_tensor = torch.frombuffer(
            bytearray(pickle.dumps(obj)), dtype=torch.uint8
        )
        size_tensor = torch.tensor(
            [object_tensor.numel()], dtype=torch.long, device="cpu"
        )
        h1 = torch.distributed.isend(
            size_tensor, dst=self.ranks[dst], group=cpu_group
        )
        h2 = torch.distributed.isend(
            object_tensor, dst=self.ranks[dst], group=cpu_group
        )
        return [h1, h2]

    def recv_object_on_hidden_channel(self, src: int, channel: Any) -> Any:
        _, cpu_group = self._hidden_channel_groups(channel)
        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")
        torch.distributed.recv(size_tensor, src=self.ranks[src], group=cpu_group)
        object_tensor = torch.empty(
            size_tensor.item(), dtype=torch.uint8, device="cpu"
        )
        torch.distributed.recv(object_tensor, src=self.ranks[src], group=cpu_group)
        return pickle.loads(object_tensor.numpy().tobytes())

    def isend_tensor_dict_on_hidden_channel(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        channel: Any,
        dst: int | None = None,
    ) -> list[Any]:
        if self.world_size <= 1:
            return []
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size, f"Invalid dst rank ({dst})"
        device_group, cpu_group = self._hidden_channel_groups(channel)
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        # Use async send for metadata so the edge head segment can return
        # immediately even when the cloud worker has not yet reached the
        # matching recv (e.g. it is still executing earlier prefill slices).
        handles = self.send_object_on_hidden_channel_async(
            metadata_list, dst, channel
        )

        tensor_keys = [k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)]
        assert len(tensor_keys) == len(tensor_list)
        for tensor in tensor_list:
            if tensor.numel() == 0:
                continue
            group = cpu_group if tensor.is_cpu else device_group
            if tensor.device.type == "npu":
                tensor.record_stream(torch.npu.current_stream(tensor.device))
            handles.append(torch.distributed.isend(
                tensor, dst=self.ranks[dst], group=group
            ))
        return handles

    def irecv_tensor_dict_on_hidden_channel(
        self,
        channel: Any,
        src: int | None = None,
    ) -> tuple[dict[str, torch.Tensor | Any] | None,
               list[Any], list[Callable[[], None]]]:
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None, [], []
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert src < self.world_size, f"Invalid src rank ({src})"
        device_group, cpu_group = self._hidden_channel_groups(channel)
        recv_metadata_list = self.recv_object_on_hidden_channel(src, channel)
        tensor_dict: dict[str, Any] = {}
        handles = []
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
                tensor_dict[key] = tensor
                if tensor.numel() == 0:
                    continue
                group = cpu_group if tensor.is_cpu else device_group
                handles.append(torch.distributed.irecv(
                    tensor, src=self.ranks[src], group=group
                ))
            else:
                tensor_dict[key] = value
        return tensor_dict, handles, []

    # ------------------------------------------------------------------
    # Dual-channel (alternate group) overrides
    #
    # Upstream vllm/distributed/parallel_state.py is kept clean. The
    # following methods replicate the upstream behavior bit-for-bit when
    # use_alt_group=False, and route through self.alt_*_group when
    # use_alt_group=True. Required by vllm_ascend/worker/worker.py to
    # separate ALL_DECODE traffic from the rest of PP traffic.
    #
    # Additionally, isend_tensor_dict here adds the NPU `record_stream`
    # branch so the upstream parallel_state.py does not need a device
    # check for "npu".
    # ------------------------------------------------------------------

    def send_object(self, obj: Any, dst: int, use_alt_group: bool = False) -> None:
        """Send the input object list to the destination rank.

        NOTE: ``dst`` is the local rank of the destination rank.
        """
        assert dst < self.world_size, f"Invalid dst rank ({dst})"
        assert dst != self.rank_in_group, (
            "Invalid destination rank. Destination rank is the same as the current rank."
        )

        cpu_group = self.alt_cpu_group if use_alt_group else self.cpu_group

        # Serialize object to tensor and get the size as well
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)
        size_tensor = torch.tensor(
            [object_tensor.numel()], dtype=torch.long, device="cpu"
        )

        # Send object size
        torch.distributed.send(size_tensor, dst=self.ranks[dst], group=cpu_group)
        # Send object
        torch.distributed.send(object_tensor, dst=self.ranks[dst], group=cpu_group)
        return None

    def recv_object(self, src: int, use_alt_group: bool = False) -> Any:
        """Receive the input object list from the source rank.

        NOTE: ``src`` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"
        assert src != self.rank_in_group, (
            "Invalid source rank. Source rank is the same as the current rank."
        )

        cpu_group = self.alt_cpu_group if use_alt_group else self.cpu_group

        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")
        # Receive object size
        rank_size = torch.distributed.recv(
            size_tensor, src=self.ranks[src], group=cpu_group
        )

        # Tensor to receive serialized objects into.
        object_tensor = torch.empty(  # type: ignore[call-overload]
            size_tensor.item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device="cpu",
        )
        rank_object = torch.distributed.recv(
            object_tensor, src=self.ranks[src], group=cpu_group
        )

        assert rank_object == rank_size, (
            "Received object sender rank does not match the size sender rank."
        )
        return pickle.loads(object_tensor.numpy().tobytes())

    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
        all_gather_group: "GroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
        use_alt_group: bool = False,
    ) -> dict[str, torch.Tensor | Any] | None:
        """Synchronous tensor-dict send. See upstream docstring for semantics.

        ``use_alt_group``: If True, use the alternate device/cpu groups for
        communication. Requires ``create_alternate_groups`` to have been
        called first.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict
        handles = self.isend_tensor_dict(
            tensor_dict,
            dst=dst,
            all_gather_group=all_gather_group,
            all_gather_tensors=all_gather_tensors,
            use_alt_group=use_alt_group,
        )
        for handle in handles:
            handle.wait()
        return None

    def isend_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
        all_gather_group: "GroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
        use_alt_group: bool = False,
    ):
        """Async tensor-dict send. Returns the list of distributed handles."""
        if self.world_size <= 1:
            return []

        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        if self.use_cpu_custom_send_recv:
            if self.device_communicator is None:
                raise ValueError("No device communicator found")
            # custom device communicator path is synchronous
            self.device_communicator.send_tensor_dict(  # type: ignore
                tensor_dict, dst
            )
            return []

        all_gather_size = (
            1 if all_gather_group is None else all_gather_group.world_size
        )
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        if use_alt_group:
            assert self.alt_device_group is not None, (
                "Alternate groups not created. "
                "Call create_alternate_groups() first."
            )
            group = self.alt_device_group
            metadata_group = self.alt_cpu_group
        else:
            group = self.device_group
            metadata_group = self.cpu_group

        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        self.send_object(metadata_list, dst=dst, use_alt_group=use_alt_group)

        tensor_keys = [k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)]
        assert len(tensor_keys) == len(tensor_list)

        handles = []
        for key, tensor in zip(tensor_keys, tensor_list):
            if tensor.numel() == 0:
                continue

            if self._should_use_all_gather(
                key, tensor.numel(), all_gather_group, all_gather_tensors
            ):
                tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

            comm_group = metadata_group if tensor.is_cpu else group
            handle = torch.distributed.isend(
                tensor, dst=self.ranks[dst], group=comm_group
            )
            # NPU record_stream branch — moved here from upstream parallel_state.py.
            if tensor.is_cuda:
                tensor.record_stream(torch.cuda.current_stream(tensor.device))
            elif tensor.device.type == "npu":
                tensor.record_stream(torch.npu.current_stream(tensor.device))
            handles.append(handle)
        return handles

    def recv_tensor_dict(
        self,
        src: int | None = None,
        all_gather_group: "GroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
        use_alt_group: bool = False,
    ) -> dict[str, torch.Tensor | Any] | None:
        """Synchronous tensor-dict recv. See upstream docstring for semantics."""
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None
        tensor_dict, handles, postprocess = self.irecv_tensor_dict(
            src=src,
            all_gather_group=all_gather_group,
            all_gather_tensors=all_gather_tensors,
            use_alt_group=use_alt_group,
        )
        for handle in handles:
            handle.wait()
        for fn in postprocess:
            fn()
        return tensor_dict

    def irecv_tensor_dict(
        self,
        src: int | None = None,
        all_gather_group: "GroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
        use_alt_group: bool = False,
    ):
        """Async tensor-dict recv. Returns ``(tensor_dict, handles, postprocess)``."""
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None, [], []

        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert src < self.world_size, f"Invalid src rank ({src})"

        if self.use_cpu_custom_send_recv:
            if self.device_communicator is None:
                raise ValueError("No device communicator found")
            # custom device communicator path is synchronous
            sync_tensor_dict = self.device_communicator.recv_tensor_dict(  # type: ignore
                src
            )
            return sync_tensor_dict, [], []

        all_gather_size = (
            1 if all_gather_group is None else all_gather_group.world_size
        )
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        if use_alt_group:
            assert self.alt_device_group is not None, (
                "Alternate groups not created. "
                "Call create_alternate_groups() first."
            )
            group = self.alt_device_group
            metadata_group = self.alt_cpu_group
        else:
            group = self.device_group
            metadata_group = self.cpu_group

        recv_metadata_list = self.recv_object(src=src, use_alt_group=use_alt_group)
        tensor_dict: dict[str, Any] = {}
        handles: list[Any] = []
        postprocess: list[Callable[[], None]] = []

        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                full_tensor = torch.empty(
                    value.size, dtype=value.dtype, device=value.device
                )
                if full_tensor.numel() == 0:
                    tensor_dict[key] = full_tensor
                    continue

                if self._should_use_all_gather(
                    key, full_tensor.numel(), all_gather_group, all_gather_tensors
                ):
                    orig_shape = full_tensor.shape
                    slice_tensor = full_tensor.reshape(all_gather_size, -1)[
                        all_gather_rank
                    ]
                    comm_group = metadata_group if slice_tensor.is_cpu else group
                    handle = torch.distributed.irecv(
                        slice_tensor, src=self.ranks[src], group=comm_group
                    )
                    handles.append(handle)

                    def _postprocess(
                        key: str = key,
                        slice_tensor: torch.Tensor = slice_tensor,
                        orig_shape: tuple[int, ...] = tuple(orig_shape),
                        all_gather_group=all_gather_group,
                    ) -> None:
                        assert all_gather_group is not None
                        tensor_dict[key] = all_gather_group.all_gather(
                            slice_tensor, dim=0
                        ).reshape(orig_shape)

                    postprocess.append(_postprocess)
                    tensor_dict[key] = slice_tensor
                else:
                    comm_group = metadata_group if full_tensor.is_cpu else group
                    handle = torch.distributed.irecv(
                        full_tensor, src=self.ranks[src], group=comm_group
                    )
                    handles.append(handle)
                    tensor_dict[key] = full_tensor
            else:
                tensor_dict[key] = value
        return tensor_dict, handles, postprocess

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        assert -input_.dim() <= scatter_dim < input_.dim(), (
            f"Invalid scatter dim ({scatter_dim}) for input tensor with shape {input_.size()}"
        )
        assert -input_.dim() <= gather_dim < input_.dim(), (
            f"Invalid gather dim ({gather_dim}) for input tensor with shape {input_.size()}"
        )
        assert self.device_communicator is not None, "device_communicator should be initialized when world_size > 1"
        return self.device_communicator.all_to_all(input_, scatter_dim, gather_dim, scatter_sizes, gather_sizes)


vllm.distributed.parallel_state.GroupCoordinator = GroupCoordinatorPatch
_patch_destroy_distributed_environment()
