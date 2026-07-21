# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-model edge head_tail batched :class:`ModelRunner`.

This module is the edge-cloud batched-compute counterpart of
``NPUModelRunner``. It subclasses
``vllm_ascend.worker.model_runner_v1.NPUModelRunner`` and overrides
``initialize_kv_cache`` plus a handful of new methods
(``execute_model_pre`` / ``execute_model_batched_head`` /
``execute_model_batched_tail`` / ``execute_model_post_batched`` /
``initialize_kv_cache_post``) that drive the shared-model edge
``head_tail`` batched forward (a single ``_model_forward`` +
``compute_logits`` call replaces ``dp_size`` per-dp_rank calls).

The original ``NPUModelRunner`` (``model_runner_v1.py``) is **not
modified**. All batched-compute extensions live in this module so the
base runner keeps its single-DP semantics.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode
from vllm.config.vllm import VllmConfig
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, set_edge_cloud_layer_range
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    ModelRunnerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.utils import (
    vllm_version_is,
)
from vllm_ascend.worker.edge_cloud.execute_model_bundle import (
    _ExecuteModelBundle,
    _MergedAttnContext,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner, get_tp_context
from vllm_ascend.eplb.utils import model_register
from vllm_ascend.patch.worker.patch_draft_quarot import patch_load_weights
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)


class BatchedModelRunner(NPUModelRunner):
    """ModelRunner for the shared-model edge ``head_tail`` batched
    compute path.

    Subclasses :class:`NPUModelRunner` and:

    - Overrides :meth:`initialize_kv_cache` with an edge-cloud
      head_tail coordination branch that registers the current
      worker's ``KVCacheConfig`` and, on the LAST caller, allocates
      ONE global KV buffer spanning all dp_ranks' blocks (the per-
      dp_rank offset / num_blocks bookkeeping lives in
      ``self._per_dp_offsets`` / ``self._per_dp_num_blocks`` /
      ``self._global_num_blocks``).
    - Adds :meth:`initialize_kv_cache_post` — runs the downstream
      hooks (drafter, KV transfer, routed-experts capturer) that
      ``initialize_kv_cache`` would normally call but that the
      head_tail branch defers (the global buffer is built before
      them so they can mirror the bound state).
    - Adds the batched-compute entry points
      :meth:`execute_model_pre` (per-dp_rank preprocess),
      :meth:`execute_model_batched_head` / ``_tail`` (single-shot
      merged forward + ``compute_logits``),
      :meth:`execute_model_post_batched` (per-dp_rank state write).
    - Adds the merged-attn helper :meth:`_get_or_build_merged_attn_ctx`
      that builds a merged ``AscendCommonAttentionMetadata`` from
      the per-dp_rank ``_ExecuteModelBundle``s and feeds it back
      into each per-(kv_cache_gid, attn_gid) builder (the per-layer
      ``AscendMetadata`` / ``GDNAttentionMetadata`` is produced by
      the builder itself — no per-layer manual merge logic). Also
      adds the per-layer routing helpers
      :meth:`_get_bpb_for_layer` / :meth:`_is_c8_attn_layer` and
      the global-buffer allocator
      :meth:`_allocate_global_kv_cache_tensors`.

    The original :class:`NPUModelRunner` is unchanged. Use this class
    in place of :class:`NPUModelRunner` when constructing the shared-
    model edge ``head_tail`` worker.
    """

    # ------------------------------------------------------------------
    # Class-level sync state for shared-model-edge KV cache global remap
    # ------------------------------------------------------------------
    # When the shared-model edge runs in ``head_tail`` mode, every
    # dp_rank worker's ``BatchedModelRunner.initialize_kv_cache`` is
    # called independently and produces a per-dp_rank
    # ``KVCacheConfig``. To allow a single batched forward to address
    # the union of all dp_rank KV blocks as one global buffer, the
    # runners coordinate here:
    #
    # - ``_KV_CACHE_CONFIGS_PER_DP_RANK``: each runner registers its
    #   config under its own ``data_parallel_rank``. Last writer wins
    #   the construction.
    # - ``_KV_CACHE_CONSTRUCTED``: set by the LAST caller after the
    #   global buffer is built and bound; subsequent entries (e.g.
    #   unit tests calling ``initialize_kv_cache`` twice) short-
    #   circuit.
    #
    # These are class-level (not module-level) because the state
    # conceptually belongs to the runner class — the worker module
    # doesn't need to import ``KVCacheConfig`` or know about the
    # construction protocol. The original ``NPUModelRunner`` does not
    # have these attributes; the subclass adds them.
    _KV_CACHE_CONFIGS_PER_DP_RANK: "dict[int, KVCacheConfig]" = {}
    _KV_CACHE_CONSTRUCTED: bool = False

    def __init__(self, vllm_config: VllmConfig, device: torch.device, local_rank: int = 0):
        self.local_rank = local_rank
        super().__init__(vllm_config, device)

    # ------------------------------------------------------------------
    # KV cache init (overridden): edge-cloud head_tail coordination
    # ------------------------------------------------------------------
    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize KV cache based on ``kv_cache_config``.

        Edge-cloud non-embedding_only extension (single-NPU multi-DP
        only)
        ------------------------------------------------------------
        When the edge runs in ``head_tail`` mode on a SINGLE NPU that
        hosts ``data_parallel_size`` virtual DP workers, every
        dp_rank worker receives a per-dp_rank ``KVCacheConfig`` whose
        ``num_blocks`` reflects only that dp_rank's share. To allow a
        single batched forward to address the union of all dp_rank KV
        blocks as one global buffer, this method:

        1. Registers the current worker's ``KVCacheConfig`` in the
           class-level
           :attr:`BatchedModelRunner._KV_CACHE_CONFIGS_PER_DP_RANK`
           (keyed by ``data_parallel_rank``).
        2. Checks — BEFORE calling ``_allocate_kv_cache_tensors`` —
           whether every dp_rank has registered. If not, returns
           immediately; per-dp_rank allocation is **deferred** in
           favour of the global buffer the last caller will build.
        3. If this worker is the last caller, flips
           :attr:`BatchedModelRunner._KV_CACHE_CONSTRUCTED = True`,
           computes per-dp_rank block offsets and
           ``global_num_blocks = sum(...)``, and delegates to
           :meth:`_allocate_global_kv_cache_tensors` on THIS runner
           to build the shared buffer.
        4. Propagates the shared ``self.kv_caches`` reference to
           every registered worker so all dp_ranks point at the same
           tensor.

        The "last caller" can be ANY dp_rank worker — not restricted
        to the leader. The leader's model_runner is used as the
        executor (it owns the device), but detection happens here
        without any leader-specific branch.

        This branch is only taken when ALL of the following hold:
        - ``edge_cloud_enabled``
        - ``role == "edge"``
        - ``mode != "embedding_only"`` (i.e. ``head_tail``)
        - ``is_shared_model_edge`` (single-NPU multi-DP edge layout)
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self._mamba_bufs = None
        self._mamba_copy_bufs = None

        # For embedding_only edge, skip KV cache tensor allocation and
        # attention backend initialization. The edge does not execute
        # any attention layers; keeping a full kv_cache_config is only
        # for the scheduler to correctly schedule requests.
        if (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.mode == "embedding_only"
            and self.edge_cloud_cfg.role == "edge"
        ):
            self.attn_groups = [
                [] for _ in range(len(kv_cache_config.kv_cache_groups))
            ]
            self.use_hybrid_blocks = False
            self.need_accepted_tokens = False
            self.may_reinitialize_input_batch(kv_cache_config)
            self.kv_cache = {}
            logger.info(
                "[EdgeCloud] embedding_only edge skipped KV cache "
                "tensor allocation and attention backend initialization."
            )
            return

        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(
            kv_cache_config)
        # NOTE(cmq): initialize_attn_backend must before using
        # self.attn_groups
        self.initialize_attn_backend(kv_cache_config)
        self.use_hybrid_blocks = len(self.attn_groups) > 1
        # NOTE: Currently, we determine whether we need
        # ``num_accepted_tokens`` through ``MambaSpec``.
        from vllm.v1.kv_cache_interface import MambaSpec as _MambaSpec
        self.need_accepted_tokens = any(
            [isinstance(attn_group[0].kv_cache_spec, _MambaSpec)
             for attn_group in self.attn_groups])

        self.may_reinitialize_input_batch(kv_cache_config)
        # === Edge-cloud single-NPU multi-DP, head_tail mode ===
        # Only the shared-model edge topology (single NPU, multiple
        # DP ranks) needs the global KV buffer; other edge layouts
        # (multi-NPU, single-DP) fall through to the standard
        # per-dp_rank allocation below.
        if (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "edge"
            and self.edge_cloud_cfg.mode != "embedding_only"
            and self.parallel_config.is_shared_model_edge
        ):
            # ``_KV_CACHE_CONFIGS_PER_DP_RANK`` is pre-registered by
            # :meth:`SharedModelEdgeWorker.initialize_from_config`
            # (which knows ``self.local_rank`` — the unique virtual
            # worker identifier in the shared-model edge topology —
            # whereas ``self.parallel_config.data_parallel_rank`` is
            # the same for every virtual worker on the edge since
            # they all share the same NPU). The model_runner only
            # READS the registry here.
            dp_size = self.parallel_config.data_parallel_size
            if (len(
                    BatchedModelRunner._KV_CACHE_CONFIGS_PER_DP_RANK)
                    < dp_size):
                logger.info(
                    "[EdgeCloud] head_tail shared-model edge: "
                    "deferred allocation (%d/%d dp_ranks registered); "
                    "waiting for the last caller.",
                    len(BatchedModelRunner
                        ._KV_CACHE_CONFIGS_PER_DP_RANK),
                    dp_size,
                )
                return
            if BatchedModelRunner._KV_CACHE_CONSTRUCTED:
                return

            # Claim the construction slot.
            BatchedModelRunner._KV_CACHE_CONSTRUCTED = True

            # Per-dp_rank block layout: dp_k's local physical block
            # N maps to global physical block ``N * dp_size + k``
            # (interleaved by dp_rank).
            sorted_configs = sorted(
                BatchedModelRunner._KV_CACHE_CONFIGS_PER_DP_RANK.items(),
                key=lambda kv: kv[0])
            per_dp_offsets: dict[int, int] = {}
            per_dp_num_blocks: dict[int, int] = {}
            for rank, cfg in sorted_configs:
                per_dp_offsets[rank] = rank
                per_dp_num_blocks[rank] = cfg.num_blocks
            template_num_blocks = min(
                cfg.num_blocks for cfg in
                BatchedModelRunner._KV_CACHE_CONFIGS_PER_DP_RANK.values())
            global_num_blocks = template_num_blocks * dp_size

            # Persist offsets on this runner.
            self._per_dp_offsets = per_dp_offsets
            self._per_dp_num_blocks = per_dp_num_blocks
            self._global_num_blocks = global_num_blocks

            # Allocate ONE global KV buffer on THIS runner (the last
            # caller). ``bind_kv_cache`` mutates
            # ``compilation_config.static_forward_context`` which is
            # shared across all dp_rank runners.
            kv_caches = self._allocate_global_kv_cache_tensors(
                BatchedModelRunner._KV_CACHE_CONFIGS_PER_DP_RANK)
            return
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)
        # TODO: refactor the logic of attention
        # Initialize drafter attention group initialization
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
        ):
            assert isinstance(
                self.drafter,
                AscendEagleProposer | AscendDflashProposer
                | AscendDraftModelProposer)
            block_size = (
                self.kernel_block_sizes[0]
                if isinstance(self.kernel_block_sizes, list)
                else self.kernel_block_sizes)
            self.drafter.initialize_attn_backend(
                self.kv_cache_config, block_size)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    def initialize_kv_cache_post(self):
        """Run the post-allocation KV-cache hooks.

        Does NOT call ``initialize_kv_cache_tensors`` (already done
        by :meth:`_allocate_global_kv_cache_tensors` for the
        head_tail branch, or by :meth:`initialize_kv_cache` for the
        standard path). Only runs the downstream hooks (drafter,
        routed-experts capturer) — the KV transfer registration is
        handled by the head_tail branch's
        :meth:`_allocate_global_kv_cache_tensors` directly.
        """
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
        ):
            assert isinstance(
                self.drafter,
                AscendEagleProposer | AscendDflashProposer
                | AscendDraftModelProposer)
            skip_edge_drafter_attn_init = (
                self._edge_cloud_enabled
                and self.edge_cloud_cfg.role == "edge"
                and self.speculative_config.method in ("mtp", "eagle3")
            )
            if skip_edge_drafter_attn_init:
                # All draft decoder layers run on the cloud. Their stale
                # static-forward-context entries have already been removed on
                # the edge, so the edge KV cache config intentionally contains
                # no draft attention layers.
                self.drafter.draft_attn_groups = []
                logger.info(
                    "[EdgeCloud] Edge skipped %s drafter attention backend "
                    "initialization.",
                    self.speculative_config.method,
                )
            else:
                block_size = (
                    self.kernel_block_sizes[0]
                    if isinstance(self.kernel_block_sizes, list)
                    else self.kernel_block_sizes
                )
                self.drafter.initialize_attn_backend(self.kv_cache_config, block_size)

        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    # ------------------------------------------------------------------
    # Global KV buffer allocator (head_tail last caller)
    # ------------------------------------------------------------------
    def _allocate_global_kv_cache_tensors(
        self,
        kv_cache_config_per_dp: dict[int, KVCacheConfig],
    ) -> dict[str, torch.Tensor]:
        """Allocate ONE global KV buffer spanning every dp_rank's
        blocks.

        Called once on the LAST caller's model_runner by
        :meth:`initialize_kv_cache` after every dp_rank has registered
        its ``KVCacheConfig``.

        Implementation
        --------------
        1. Pick the smallest ``num_blocks`` across dp_ranks as the
           "template" config — it has the smallest per-tensor
           ``size`` for each layer (since ``size`` is linear in
           ``num_blocks``).
        2. Build a synthetic ``KVCacheConfig`` that mirrors the
           template but with ``num_blocks = global_num_blocks`` and
           each ``kv_cache_tensor.size`` scaled by
           ``global_num_blocks / template.num_blocks``.
        3. Delegate to the existing
           :meth:`initialize_kv_cache_tensors` to allocate + bind
           the global buffer. The result populates the leader's
           ``self.kv_caches`` and binds the forward context exactly
           once.

        Returns the global ``kv_caches`` dict (``{layer_name: tensor}``)
        for callers that want to introspect it; ``self.kv_caches`` is
        also populated as a side effect.
        """
        if not kv_cache_config_per_dp:
            raise RuntimeError(
                "_allocate_global_kv_cache_tensors called with empty "
                "kv_cache_config_per_dp")

        # 1. Pick the template config (smallest num_blocks). All
        #    configs share the same layer spec / block_size /
        #    dtype / head layout, only ``num_blocks`` and the
        #    resulting ``size`` differ.
        template_dp_rank = min(
            kv_cache_config_per_dp,
            key=lambda k: kv_cache_config_per_dp[k].num_blocks)
        template_cfg = kv_cache_config_per_dp[template_dp_rank]
        global_num_blocks = sum(
            cfg.num_blocks for cfg in kv_cache_config_per_dp.values())
        if global_num_blocks <= template_cfg.num_blocks:
            raise RuntimeError(
                "Global num_blocks (%d) must exceed template "
                "num_blocks (%d, dp_rank=%d). This indicates a "
                "dp_rank did not register a positive num_blocks."
                % (global_num_blocks, template_cfg.num_blocks,
                   template_dp_rank))

        # 2. Build the synthetic config with scaled sizes.
        scale = global_num_blocks / template_cfg.num_blocks
        scaled_tensors: list[KVCacheTensor] = []
        for tensor in template_cfg.kv_cache_tensors:
            scaled_tensors.append(
                KVCacheTensor(
                    size=int(tensor.size * global_num_blocks
                             // template_cfg.num_blocks),
                    shared_by=list(tensor.shared_by),
                ))
        synthetic = replace(
            deepcopy(template_cfg),
            num_blocks=global_num_blocks,
            kv_cache_tensors=scaled_tensors,
        )
        # Run the SAME init chain as the standard
        # ``initialize_kv_cache`` path.
        kv_caches = self.initialize_kv_cache_tensors(synthetic)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

        logger.info(
            "[EdgeCloud] head_tail shared-model edge: allocated "
            "global KV buffer of %d blocks (template dp_rank=%d, "
            "%d blocks; scale=%.2fx across %d dp_ranks).",
            global_num_blocks, template_dp_rank,
            template_cfg.num_blocks, scale,
            len(kv_cache_config_per_dp))
        return kv_caches

    def bind_to_shared_model(self, model: nn.Module) -> None:
        """Bind this runner to a model object loaded by another runner.

        Used by ``SharedModelEdgeWorker`` follower workers to share a single
        ``nn.Module`` instance across multiple model runners in the same
        process. Replaces ``self.model`` with ``model`` and mirrors the
        post-model-creation side effects of :meth:`load_model`:

        - edge-cloud: re-derives ``num_layers`` and re-creates the segment
          callables from the shared (already sharded) model.
        - dynamic EPLB: registers the shared model.
        - drafter: runs the drafter ``load_model`` hook against the shared
          model and sets the eagle3 aux hidden state layers.
        - ACLGraphWrapper: wraps the shared model if cudagraph mode is on.
        - profiler: starts the data dump if cudagraph mode is on.

        The caller is responsible for:

        - ensuring ``model`` has already been fully loaded by another
          runner in the same process (i.e. the leader
          ``SharedModelEdgeWorker``);
        - assigning ``self.model_memory_usage`` after binding, because only
          the leader's profile run actually measures it.
        """
        self.model = model

        # Edge-cloud specific state, derived from the (already sharded) model.
        if self._edge_cloud_enabled:
            # Locate the transformer layers — the model may be wrapped in a
            # multimodal ConditionalGeneration (language_model.model.layers)
            # or be a plain CausalLM (model.layers).
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                transformer_layers = self.model.model.layers
            else:
                transformer_layers = self.model.language_model.model.layers
            self.num_layers = len(transformer_layers)
            if hasattr(self.model, "set_moe_parameters"):
                self.model.set_moe_parameters()
            if self.edge_cloud_cfg.role == "edge":
                self.segment_a = self._create_segment_callable(
                    self.model,
                    0,
                    self.head_k,
                    is_first_segment=True,
                    is_last_segment=False,
                )
                self.segment_e = self._create_segment_callable(
                    self.model,
                    self.num_layers - self.tail_k,
                    self.num_layers,
                    is_first_segment=False,
                    is_last_segment=True,
                )
                self.segment_a_wrapper = self._wrap_segment_if_needed(
                    self.segment_a)
                self.segment_e_wrapper = self._wrap_segment_if_needed(
                    self.segment_e)
            else:
                self.segment_c = self._create_segment_callable(
                    self.model,
                    self.head_k,
                    self.num_layers - self.tail_k,
                    is_first_segment=False,
                    is_last_segment=False,
                )
                self.segment_c_wrapper = self._wrap_segment_if_needed(
                    self.segment_c)

        # Standard post-model-creation side effects.
        if self.dynamic_eplb:
            model_register(self.model)
        if self.drafter:
            logger.info("Loading drafter model for shared model binding...")
            if self.vllm_config.quant_config is not None:
                patch_load_weights(self.vllm_config)

            is_edge_cloud_draft_drafter = (
                self.speculative_config is not None
                and self.speculative_config.method in ("mtp", "eagle3")
            )
            if is_edge_cloud_draft_drafter:
                # Draft models (MTP/Eagle3) use the same edge-cloud layer range
                # mechanism as the main model. In both embedding_only and
                # head_tail modes all draft decoder layers run on the cloud, so
                # temporarily use head_k=tail_k=0 while loading the drafter.
                set_edge_cloud_layer_range(0, 0)
                if self.speculative_config.method == "eagle3":
                    import vllm_ascend.patch.models.eagle3_edge_cloud  # noqa: F401
            
            with get_tp_context(self.drafter):
                self.drafter.load_model(self.model)

            if (
                is_edge_cloud_draft_drafter
                and hasattr(self.drafter, "model")
                and self.drafter.model is not None
            ):
                self._setup_edge_cloud_draft(
                    self.drafter.model, self.speculative_config.method
                )

            if is_edge_cloud_draft_drafter:
                # Do not leak the drafter's cloud-only range into later
                # main-model initialization or cache setup.
                set_edge_cloud_layer_range(self.head_k, self.tail_k)
            
            if self.use_aux_hidden_state_outputs:
                from vllm.model_executor.models.interfaces import supports_eagle3
                if not supports_eagle3(self.model):
                    raise RuntimeError(
                        "Model does not support EAGLE3 interface but "
                        "aux_hidden_state_outputs was requested"
                    )
                aux_layers = self._get_eagle3_aux_layers_from_config()
                if not aux_layers:
                    aux_layers = self.model.get_eagle3_default_aux_hidden_state_layers()
                self.model.set_aux_hidden_state_layers(aux_layers)

        # wrap the model with full graph wrapper if needed.
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            if not isinstance(self.model, ACLGraphWrapper):
                self.update_stream: torch.npu.Stream = torch.npu.Stream()
                self.model = ACLGraphWrapper(
                    self.model,
                    self.vllm_config,
                    runtime_mode=CUDAGraphMode.FULL,
                    use_eagle=self.use_eagle,
                    enable_enpu=self.enable_enpu,
                )

        if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            self._start_dump_data()

    def _should_save_for_attn_metadata(self) -> bool:
        return True

    def _sync_metadata_across_dp(
        self,
        num_tokens: int,
        is_draft_model: bool = False,
        cudagraph_mode: "CUDAGraphMode" = CUDAGraphMode.NONE,
        allow_dp_padding: bool = False,
    ) -> tuple[int, None, "CUDAGraphMode"]:
        """No-op for the shared-model edge path.

        The base implementation calls ``dist.all_reduce`` to
        coordinate ``num_tokens`` / ``cudagraph_mode`` across
        the DP group. On the shared-model edge every
        ``dp_rank`` lives in the SAME process / SAME NPU; there
        is no distributed DP group, and the cross-dp
        coordination that the upstream code expects (uniform
        ``num_tokens`` / ``cudagraph_mode``) is performed
        instead by the leader runner's
        ``_get_or_build_merged_attn_ctx`` /
        ``cudagraph_dispatcher.dispatch`` over the per-dp_rank
        bundles. Return the caller's own values so the
        downstream flow proceeds unchanged.
        """
        return num_tokens, None, cudagraph_mode
    # ------------------------------------------------------------------
    # Batched compute entry points
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def execute_model_pre(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> "_ExecuteModelBundle | ModelRunnerOutput | None":
        """Per-dp_rank preprocess for the batched compute path.

        Mirrors the prep portion of ``NPUModelRunner.execute_model``
        (everything BEFORE ``set_ascend_forward_context``). Updates
        ``self.input_batch``. Always runs the standard path
        (segment_e / cloud fast paths are reserved for the
        second call into ``execute_model`` when the cloud has
        already produced ``intermediate_tensors``; here
        ``intermediate_tensors`` is always ``None``).

        For empty / no-work cases, returns the same early
        ``ModelRunnerOutput`` / ``None`` values as the original
        ``execute_model``; the busy_loop detects these via
        ``handle_output`` and skips the batched forward. Otherwise
        returns a :class:`_ExecuteModelBundle` consumed by
        ``execute_model_batched_head`` / ``_tail`` / ``_post_batched``.

        Does NOT call ``set_ascend_forward_context``, ``_model_forward``
        or ``compute_logits``. Does NOT write
        ``self.execute_model_state`` (that is the job of
        ``execute_model_post_batched``). Does NOT populate
        ``self._edge_prepare_cache`` (the head-tail busy_loop
        doesn't reuse the segment_e fast path — every round runs
        a fresh ``execute_model_pre``).
        """
        if self.vllm_config.model_config.enable_return_routed_experts:
            if vllm_version_is("0.20.2"):
                capturer = RoutedExpertsCapturer.get_instance()
                if capturer is not None:
                    capturer.clear_buffer()
            elif self.routed_experts_initialized:
                self.routed_experts_capturer.clear_buffer()

        if self.ascend_config.profiling_chunk_config.need_timing:
            if getattr(scheduler_output, "disable_profiling_timing",
                       False):
                self.ascend_config.profiling_chunk_config.need_timing = (
                    False)
            else:
                self._sync_device()
                self._execution_start_time = time.perf_counter()
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called after "
                "execute_model() returns None.")

        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            num_scheduled_tokens_copy = (
                scheduler_output.num_scheduled_tokens.copy())
            spec_decode_tokens_copy = (
                scheduler_output.scheduled_spec_decode_tokens.copy())
            scheduler_output = replace(
                scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens_copy,
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,
            )

        self._start_dump_data()
        if ((
            self.use_async_scheduling and self.num_spec_tokens
                and self._draft_token_ids is None
        ) or (
            self.pcp_size > 1 and self.supports_mm_inputs
            and get_pp_group().is_first_rank
            and not self.model_config.is_encoder_decoder
        )):
            scheduler_output = deepcopy(scheduler_output)
        num_scheduled_tokens = (
            scheduler_output.total_num_scheduled_tokens)
        with record_function_or_nullcontext("prepare input"):
            with self.synchronize_input_prep():
                if (
                    self.use_async_scheduling
                    and self.num_spec_tokens
                    and self.input_batch.prev_req_id_to_index
                    is not None
                ):
                    for req_id in (
                            scheduler_output.scheduled_cached_reqs
                            .req_ids):
                        if (
                            req_id
                            not in self.input_batch.prev_req_id_to_index
                            and (req_state := self.requests.get(req_id))
                                is not None
                            and req_state.prev_num_draft_len
                        ):
                            req_state.prev_num_draft_len = 0

                deferred_state_corrections_fn = self._update_states(
                    scheduler_output)

                if has_ec_transfer() and get_ec_transfer().is_producer:
                    with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                    ) as ec_connector_output:
                        self._execute_mm_encoder(scheduler_output)
                        self._finalize_dump_data()
                    return make_empty_encoder_model_runner_output(
                        scheduler_output)

                num_reqs = self.input_batch.num_reqs
                req_ids = self.input_batch.req_ids
                tokens = [
                    scheduler_output.num_scheduled_tokens[i]
                    for i in req_ids]
                num_scheduled_tokens_np = np.array(
                    tokens, dtype=np.int32)

                if not num_scheduled_tokens:
                    if (
                        self.parallel_config.distributed_executor_backend
                        == "external_launcher"
                        and self.parallel_config.data_parallel_size > 1
                    ):
                        self._dummy_run(1)
                    if not has_kv_transfer_group():
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(
                        scheduler_output, self.vllm_config)
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please "
                        "disable it when the requests need prompt "
                        "logprobs")

                # Apply deferred state corrections, then mamba
                # preprocess (must run after _update_states, before
                # input preparation).
                if deferred_state_corrections_fn:
                    deferred_state_corrections_fn()
                    deferred_state_corrections_fn = None
                if self.cache_config.mamba_cache_mode == "align":
                    if vllm_version_is("0.20.2"):
                        mamba_bufs = self._get_mamba_copy_bufs()
                        preprocess_bufs = mamba_bufs
                    else:
                        mamba_bufs = self._get_mamba_bufs()
                        preprocess_bufs = mamba_bufs.preprocess
                    mamba_utils.preprocess_mamba(
                        scheduler_output,
                        self.kv_cache_config,
                        self.cache_config,
                        self.mamba_state_idx,
                        self.input_batch,
                        self.requests,
                        self.compilation_config.static_forward_context,
                        self.model.get_mamba_state_copy_func(),
                        preprocess_bufs,
                    )
                    self.num_accepted_tokens.np[:num_reqs] = (
                        self.input_batch.num_accepted_tokens_cpu[
                            :num_reqs])
                    self.num_accepted_tokens.copy_to_gpu(num_reqs)
                    if (not vllm_version_is("0.20.2")
                            and mamba_bufs.postprocess_align is not None):
                        mamba_utils.stage_postprocess_inputs_to_gpu(
                            mamba_bufs.postprocess_align,
                            scheduler_output,
                            self.input_batch.req_ids,
                            num_reqs,
                            self.requests,
                            self.mamba_state_idx,
                        )
                if self.use_compress:
                    if deferred_state_corrections_fn:
                        deferred_state_corrections_fn()
                        deferred_state_corrections_fn = None
                    num_reqs = self.input_batch.num_reqs
                    req_indices = np.repeat(
                        self.arange_np[:num_reqs],
                        num_scheduled_tokens_np)
                    dsa_positions_np = (
                        self._dsa_positions_np_buf[
                            :total_num_scheduled_tokens])
                    np.add(
                        self.input_batch.num_computed_tokens_cpu[
                            req_indices],
                        self.query_pos.np[:total_num_scheduled_tokens],
                        out=dsa_positions_np,
                    )

                # Run core input preparation.
                cache = self._run_input_preparation(scheduler_output)
                total_num_scheduled_tokens = (
                    cache["total_num_scheduled_tokens"])
                num_tokens_padded = cache["num_tokens_padded"]
                num_tokens_across_dp = cache["num_tokens_across_dp"]
                attn_metadata = cache["attn_metadata"]
                logits_indices = cache["logits_indices"]
                spec_decode_metadata = cache["spec_decode_metadata"]
                spec_decode_common_attn_metadata = (
                    cache["spec_decode_common_attn_metadata"])
                cudagraph_mode = cache["cudagraph_mode"]
                batch_desc = cache["batch_desc"]
                cudagraph_stats = cache["cudagraph_stats"]

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output,
                num_tokens_padded
                if not (self.use_cp
                        and self.pcp_manager.pcp_use_hybrid_attn)
                else total_num_scheduled_tokens,
                None,
            )

            if not self.edge_cloud_cfg.role == "edge":
                # update global cos, sin
                update_cos_sin(positions)

        if self.dynamic_eplb:
            with record_function_or_nullcontext("EPLB weight D2D"):
                self.eplb_updator.forward_before()

        # Set cudagraph mode to none if calc_kv_scales is true.
        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            self.calculate_kv_scales = False
        if self.ascend_config.enable_async_exponential:
            self.sampler.do_async_exponential(
                b_s=logits_indices.shape[0],
                head_dim=self.model_config.get_vocab_size(),
                generators=self.input_batch.sampling_metadata.generators,
            )

        # Re-bundle the per-dp_rank attention state that
        # ``_build_attention_metadata`` already saved to ``self.*``
        # (via ``_should_save_for_attn_metadata()=True``): the
        # ``self.cm_base`` ``AscendCommonAttentionMetadata``
        # directly, plus a list of per-kv_cache_gid override
        # dicts (``self.per_gid_cm[kv_cache_gid]`` — exactly what
        # ``_save`` wrote for each gid: ``encoder_seq_lens`` /
        # ``encoder_seq_lens_cpu`` / GDN's ``query_start_loc(_cpu)``
        # / gid>0's ``block_table_tensor`` / ``slot_mapping``),
        # plus the per-(kv_cache_gid, attn_gid) ``(cascade_attn_prefix_len,
        # extra_attn_metadata_args)`` tuple saved by
        # ``_build_attn_group_metadata``. The batched merge step
        # re-uses these structures verbatim — it never re-extracts
        # the raw fields.
        # ---- guard the async GPU->CPU copy (matches
        # ``_build_attention_metadata``) ----
        cm_base = getattr(self, "cm_base", None)
        if cm_base is None:
            raise RuntimeError(
                "BatchedModelRunner.execute_model_pre expected "
                "self.cm_base to be saved by _build_attention_metadata "
                "(via _should_save_for_attn_metadata=True).")
        per_gid_cm = getattr(self, "per_gid_cm", None)
        if per_gid_cm is None:
            raise RuntimeError(
                "BatchedModelRunner.execute_model_pre expected "
                "self.per_gid_cm to be saved by _build_attention_metadata.")
        per_gid_extra = getattr(self, "per_gid_extra", None)
        if per_gid_extra is None:
            raise RuntimeError(
                "BatchedModelRunner.execute_model_pre expected "
                "self.per_gid_extra to be saved by "
                "_build_attn_group_metadata (via "
                "_should_save_for_attn_metadata=True).")
        common_attn_metadata = cm_base

        return _ExecuteModelBundle(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=None,
            hidden_states=None,
            logits_indices=logits_indices,
            spec_decode_metadata=spec_decode_metadata,
            spec_decode_common_attn_metadata=(
                spec_decode_common_attn_metadata),
            scheduler_output=scheduler_output,
            num_tokens_padded=num_tokens_padded,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_mode=cudagraph_mode,
            batch_desc=batch_desc,
            attn_metadata=attn_metadata,
            num_reqs_actual=self.input_batch.num_reqs,
            common_attn_metadata=common_attn_metadata,
            per_gid_cm=per_gid_cm,
            per_gid_extra=per_gid_extra,
            ec_connector_output=ec_connector_output,
            cudagraph_stats=cudagraph_stats,
            deferred_state_corrections_fn=(
                deferred_state_corrections_fn),
        )

    @torch.inference_mode()
    def execute_model_batched_head(
        self,
        bundles: list[_ExecuteModelBundle],
        batched_dp_ranks: list[int] | None = None,
        pp_send_work_by_channel: dict | None = None,
    ) -> list[IntermediateTensors]:
        """1 batched head model_forward = 1 ``model.embed_tokens``.

        Called ONCE per round from
        :class:`SharedModelWorkerProc.worker_busy_loop`. The forward
        context is set up for the merged batch; ``_model_forward``
        runs once on the shared ``nn.Module``; the resulting
        ``hidden_states`` is sliced back to per-dp_rank.

        Returns per-dp_rank ``IntermediateTensors`` slices, padded to
        ``self.max_num_tokens`` in the ``embedding_only`` mode so
        that the cloud's pre-allocated buffer is large enough.
        """
        # Per-bundle actual (non-padded) token counts.
        n_actuals = [
            (b.attn_metadata[0][
                next(iter(b.attn_metadata[0]))].num_actual_tokens
             if isinstance(b.attn_metadata, list) and b.attn_metadata
             else next(iter(b.attn_metadata.values()))
             .num_actual_tokens)
            for b in bundles
        ]

        num_tokens_merged = sum(n_actuals)
        num_tokens_across_dp_merged = None

        any_bundle = bundles[0]
        is_non_embedding_only_edge = (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "edge"
            and self.edge_cloud_cfg.mode != "embedding_only")
        if is_non_embedding_only_edge:
            assert batched_dp_ranks is not None, (
                "execute_model_batched_head: batched_dp_ranks "
                "required for non-embedding_only edge mode.")
            ctx = self._get_or_build_merged_attn_ctx(
                bundles, batched_dp_ranks,
                pp_send_work_by_channel=pp_send_work_by_channel)
            head_attn_metadata: Any = ctx.merged_attn_metadata
            batch_descriptor = ctx.merged_batch_descriptor
            cudagraph_mode = ctx.merged_cudagraph_mode
            num_tokens_padded_merged = ctx.num_tokens_padded_merged
            merged_input_ids = ctx.merged_input_ids
            merged_positions = ctx.merged_positions
            merged_inputs_embeds = ctx.merged_inputs_embeds
            # Capture the inverse token perm so we can un-permute
            # the head ``_model_forward`` output (which is in
            # reordered decode-first layout) back to cat-order
            # before slicing per-dp_rank. The cloud middle forward
            # consumes the head's hidden_states with its own
            # per-dp_rank ``query_lens`` (in cat-order); reordered
            # hidden_states would mis-align with those query_lens.
            inv_merged_token_perm = ctx.inv_merged_token_perm
        else:
            head_attn_metadata = any_bundle.attn_metadata
            batch_descriptor = any_bundle.batch_desc
            cudagraph_mode = any_bundle.cudagraph_mode
            num_tokens_padded_merged = sum(
                b.num_tokens_padded for b in bundles)
            if all(b.input_ids is not None for b in bundles):
                merged_input_ids = torch.cat(
                    [b.input_ids[:n]
                     for b, n in zip(bundles, n_actuals)])
            else:
                merged_input_ids = None
            if all(b.positions is not None for b in bundles):
                cat_dim = bundles[0].positions.dim() - 1
                merged_positions = torch.cat(
                    [b.positions[..., :n]
                     for b, n in zip(bundles, n_actuals)],
                    dim=cat_dim)
            else:
                merged_positions = None
            if all(b.inputs_embeds is not None for b in bundles):
                merged_inputs_embeds = torch.cat(
                    [b.inputs_embeds[:n]
                     for b, n in zip(bundles, n_actuals)])
            else:
                merged_inputs_embeds = None
            inv_merged_token_perm = None
        with (
            record_function_or_nullcontext("forward"),
            set_ascend_forward_context(
                attn_metadata=head_attn_metadata,
                vllm_config=self.vllm_config,
                num_tokens=num_tokens_padded_merged,
                num_tokens_across_dp=num_tokens_across_dp_merged,
                aclgraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_descriptor,
                num_actual_tokens=num_tokens_merged,
                model_instance=self.model,
                max_tokens_across_pcp=(
                    0 if self.pcp_size == 1
                    else self.pcp_manager
                    .max_num_tokens_across_pcp),
                skip_compiled=False,
            ),
        ):
            hidden_states = self._model_forward(
                num_tokens_padded_merged,
                merged_input_ids,
                merged_positions,
                None,
                merged_inputs_embeds,
            )

        head_hidden = hidden_states["hidden_states"]
        head_residual = hidden_states["residual"]
        # Undo the decode-first token reorder so per-dp_rank
        # slicing produces hidden_states in cat-order
        # (``scheduler_output`` order) — the cloud middle
        # forward uses its own ``query_lens`` derived from the
        # same ``scheduler_output`` and expects this layout.
        if (inv_merged_token_perm is not None
                and inv_merged_token_perm.numel() > 0):
            head_hidden = torch.cat([
                head_hidden[:inv_merged_token_perm.shape[0]][
                    inv_merged_token_perm],
                head_hidden[inv_merged_token_perm.shape[0]:],
            ])
            head_residual = torch.cat([
                head_residual[:inv_merged_token_perm.shape[0]][
                    inv_merged_token_perm],
                head_residual[inv_merged_token_perm.shape[0]:],
            ])
        token_offsets = [0]
        for n in n_actuals:
            token_offsets.append(token_offsets[-1] + n)
        results: list[IntermediateTensors] = []
        for i, b in enumerate(bundles):
            slice_hs = head_hidden[
                token_offsets[i]:token_offsets[i + 1]]
            slice_res = head_residual[
                token_offsets[i]:token_offsets[i + 1]]
            if (self.edge_cloud_cfg.mode == "embedding_only"
                    and slice_hs.shape[0] < self.max_num_tokens):
                pad = torch.zeros(
                    self.max_num_tokens - slice_hs.shape[0],
                    *slice_hs.shape[1:],
                    dtype=slice_hs.dtype,
                    device=slice_hs.device,
                )
                slice_hs = torch.cat([slice_hs, pad], dim=0)
                slice_res = torch.cat([slice_res, pad], dim=0)
            results.append(
                IntermediateTensors({
                    "hidden_states": slice_hs,
                    "residual": slice_res,
                }))
        return results

    @torch.inference_mode()
    def execute_model_batched_tail(
        self,
        bundles: list[_ExecuteModelBundle],
        intermediates: list[IntermediateTensors],
        batched_dp_ranks: list[int] | None = None,
        pp_send_work_by_channel: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """1 batched tail model_forward + 1 batched ``compute_logits``.

        Called ONCE per round on the leader runner
        (``self.worker[0].model_runner``) to avoid duplicated forward
        across dp_ranks. Returns a 4-tuple of:

        * ``merged_hidden_states`` — the full [merged_num_tokens]
          hidden_states from the batched tail ``_model_forward``.
        * ``merged_sample_hidden_states`` — ``hidden_states`` indexed
          by ``logits_indices`` (per-sample rows).
        * ``merged_logits`` — the per-sample logits from
          ``compute_logits``.
        * ``kv_connector_output`` — the captured kv-connector output.
        """
        # Per-bundle actual token counts (defensive trim of the
        # cloud-returned intermediates).
        n_actuals_tail = [
            (b.attn_metadata[0][
                next(iter(b.attn_metadata[0]))].num_actual_tokens
             if isinstance(b.attn_metadata, list) and b.attn_metadata
             else next(iter(b.attn_metadata.values()))
             .num_actual_tokens)
            for b in bundles
        ]
        if all(it["hidden_states"] is not None
               for it in intermediates):
            merged_hidden = torch.cat(
                [it["hidden_states"][:n]
                 for it, n in zip(intermediates, n_actuals_tail)])
        else:
            merged_hidden = None
        if all(it["residual"] is not None for it in intermediates):
            merged_residual = torch.cat(
                [it["residual"][:n]
                 for it, n in zip(intermediates, n_actuals_tail)])
        else:
            merged_residual = None

        any_bundle = bundles[0]
        is_non_embedding_only_edge = (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "edge"
            and self.edge_cloud_cfg.mode != "embedding_only")
        # When non-embedding_only edge, the merged attn ctx
        # applies a decode-first reorder so the attention
        # kernel's ``split_decodes_and_prefills`` works on the
        # merged layout. We need to permute the per-token
        # inputs (cloud-returned intermediate tensors) into
        # that same reordered layout BEFORE the tail forward
        # reads them via ``merged_query_start_loc``.
        if is_non_embedding_only_edge:
            assert batched_dp_ranks is not None, (
                "execute_model_batched_tail: batched_dp_ranks "
                "required for non-embedding_only edge mode.")
            logger.info(
                "[PD] execute_model_batched_tail: "
                "calling _get_or_build_merged_attn_ctx "
                "dp_ranks=%s", batched_dp_ranks)
            ctx = self._get_or_build_merged_attn_ctx(
                bundles, batched_dp_ranks,
                pp_send_work_by_channel=pp_send_work_by_channel)
            logger.info(
                "[PD] execute_model_batched_tail: "
                "_get_or_build_merged_attn_ctx done "
                "dp_ranks=%s", batched_dp_ranks)
            tail_attn_metadata: Any = ctx.merged_attn_metadata
            batch_descriptor = ctx.merged_batch_descriptor
            cudagraph_mode = ctx.merged_cudagraph_mode
            num_tokens_padded_merged = ctx.num_tokens_padded_merged
            merged_token_perm = ctx.merged_token_perm
            inv_merged_token_perm = ctx.inv_merged_token_perm
            # ``merged_token_perm is None`` when the merged
            # batch is decode-only and the reorder is a no-op
            # (cat-order IS decode-first order); skip the
            # permute work in that case.
            if merged_token_perm is not None:
                if merged_hidden is not None:
                    merged_hidden = torch.cat([
                        merged_hidden[:merged_token_perm.shape[0]][
                            merged_token_perm],
                        merged_hidden[merged_token_perm.shape[0]:],
                    ])
                if merged_residual is not None:
                    merged_residual = torch.cat([
                        merged_residual[:merged_token_perm.shape[0]][
                            merged_token_perm],
                        merged_residual[merged_token_perm.shape[0]:],
                    ])
        else:
            tail_attn_metadata = any_bundle.attn_metadata
            batch_descriptor = any_bundle.batch_desc
            cudagraph_mode = any_bundle.cudagraph_mode
            num_tokens_padded_merged = sum(
                b.num_tokens_padded for b in bundles)
            merged_token_perm = None
            inv_merged_token_perm = None

        merged_intermediate = IntermediateTensors({
            "hidden_states": merged_hidden,
            "residual": merged_residual,
        })

        token_offsets = [0]
        for n in n_actuals_tail:
            token_offsets.append(token_offsets[-1] + n)
        merged_logits_indices = torch.cat(
            [bundles[i].logits_indices + token_offsets[i]
             for i in range(len(bundles))])
        # Map cat-order ``merged_logits_indices`` into the
        # reordered (decode-first) merged token layout that
        # ``_model_forward`` produces. After
        # ``merged_sample_hidden_states = hidden_states[indices]``,
        # each row corresponds to the same physical token the
        # bundle's ``logits_indices`` originally pointed at.
        if (inv_merged_token_perm is not None
                and inv_merged_token_perm.numel() > 0
                and merged_logits_indices.numel() > 0):
            merged_logits_indices = inv_merged_token_perm.to(
                self.device)[merged_logits_indices]
        # FULL mode: copy ``intermediate_tensors`` to leader
        # runner's pre-allocated buffer so ``seg_e`` (which is
        # ``ACLGraphWrapper``-wrapped) reads from a stable
        # device pointer across cudagraph captures. The
        # pre-allocated buffer is lazily created here (same
        # factory as NPUModelRunner's ``execute_model`` path)
        # so we don't depend on ``execute_model`` having run
        # first. The buffer is sized to ``self.max_num_tokens``
        # (the per-dp_rank max); cudagraph dispatch clamps the
        # merged padded size to within this bound, so the
        # buffer is always large enough.
        if cudagraph_mode == CUDAGraphMode.FULL:
            if self.intermediate_tensors is None:
                self.intermediate_tensors = (
                    self.model.make_empty_intermediate_tensors(
                        batch_size=self.max_num_tokens,
                        dtype=self.dtype,
                        device=self.device,
                    ))
            for k, v in merged_intermediate.items():
                if not isinstance(v, torch.Tensor):
                    continue
                # Pad the tail (beyond ``merged_num_actual``)
                # up to ``merged_num_tokens_padded`` with 0 so
                # the pre-allocated buffer's padding carries a
                # deterministic value (instead of whatever stale
                # bytes were left from a previous round).
                if v.shape[0] < num_tokens_padded_merged:
                    pad_shape = list(v.shape)
                    pad_shape[0] = (
                        num_tokens_padded_merged - v.shape[0])
                    pad = torch.zeros(
                        pad_shape, dtype=v.dtype, device=v.device)
                    v = torch.cat([v, pad], dim=0)
                elif v.shape[0] > num_tokens_padded_merged:
                    v = v[:num_tokens_padded_merged]
                dst = self.intermediate_tensors[k][
                    :num_tokens_padded_merged]
                dst.copy_(v, non_blocking=True)
                merged_intermediate[k] = dst
        kv_connector_output = None
        num_tokens_merged = sum(n_actuals_tail)
        num_tokens_across_dp_merged = None
        logger.info(
            "[PD] execute_model_batched_tail: "
            "calling _model_forward num_tokens=%d cudagraph=%s",
            num_tokens_padded_merged, cudagraph_mode)
        with (
            record_function_or_nullcontext("forward"),
            set_ascend_forward_context(
                attn_metadata=tail_attn_metadata,
                vllm_config=self.vllm_config,
                num_tokens=num_tokens_padded_merged,
                num_tokens_across_dp=num_tokens_across_dp_merged,
                aclgraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_descriptor,
                num_actual_tokens=num_tokens_merged,
                model_instance=self.model,
                max_tokens_across_pcp=(
                    0 if self.pcp_size == 1
                    else self.pcp_manager
                    .max_num_tokens_across_pcp),
                skip_compiled=False,
            ),
            self.maybe_get_kv_connector_output(
                any_bundle.scheduler_output,
                **{"defer_finalize": True},
            ) as kv_connector_output,
        ):
            hidden_states = self._model_forward(
                num_tokens_padded_merged,
                (ctx.merged_input_ids
                 if (is_non_embedding_only_edge
                     and ctx.merged_input_ids is not None)
                 else any_bundle.input_ids),
                (ctx.merged_positions
                 if (is_non_embedding_only_edge
                     and ctx.merged_positions is not None)
                 else any_bundle.positions),
                merged_intermediate,
                (ctx.merged_inputs_embeds
                 if (is_non_embedding_only_edge
                     and ctx.merged_inputs_embeds is not None)
                 else any_bundle.inputs_embeds),
            )
        logger.info(
            "[PD] execute_model_batched_tail: "
            "_model_forward returned")

        merged_sample_hidden_states = (
            hidden_states[merged_logits_indices])
        merged_logits = self.model.compute_logits(
            merged_sample_hidden_states)
        # Undo the decode-first token reorder so the worker's
        # per-dp_rank slicing
        # ``merged_hidden[token_offsets[i]:token_offsets[i+1]]``
        # (in cat order) returns each dp_rank's tokens in the
        # same layout as ``bundle[i].logits_indices`` expects.
        if (inv_merged_token_perm is not None
                and inv_merged_token_perm.numel() > 0):
            hidden_states = torch.cat([
                hidden_states[:inv_merged_token_perm.shape[0]][
                    inv_merged_token_perm],
                hidden_states[inv_merged_token_perm.shape[0]:],
            ])

        return (hidden_states, merged_sample_hidden_states,
                merged_logits, kv_connector_output)

    @torch.inference_mode()
    def execute_model_post_batched(
        self,
        bundle: _ExecuteModelBundle,
        sample_hidden_states_k: torch.Tensor,
        logits_k: torch.Tensor,
        hidden_states_k: torch.Tensor,
        kv_connector_output: Any,
    ) -> None:
        """Per-dp_rank state writing after batched ``compute_logits``.

        Mirrors the tail portion of ``NPUModelRunner.execute_model``:
        writes ``self.execute_model_state`` and
        ``self.kv_connector_output`` and triggers the deferred state
        corrections from ``_update_states``.

        Strictly does NOT include any ``sample_tokens`` logic:
        no ``apply_grammar_bitmask``, no ``_sample``, no
        ``_bookkeeping_sync``, no ``propose_draft_token_ids``, no
        ``ModelRunnerOutput`` return. It also does NOT call
        ``set_ascend_forward_context`` / ``_model_forward`` /
        ``compute_logits`` (those live in
        ``execute_model_batched_tail``).
        """
        # ``ExecuteModelState`` is a NamedTuple defined in the base
        # model_runner_v1 module — use the same constructor.
        from vllm_ascend.worker.model_runner_v1 import (
            ExecuteModelState,
        )
        self.execute_model_state = ExecuteModelState(
            scheduler_output=bundle.scheduler_output,
            logits=logits_k,
            spec_decode_metadata=bundle.spec_decode_metadata,
            spec_decode_common_attn_metadata=(
                bundle.spec_decode_common_attn_metadata),
            hidden_states=hidden_states_k,
            sample_hidden_states=sample_hidden_states_k,
            aux_hidden_states=None,
            attn_metadata=bundle.attn_metadata,
            positions=bundle.positions,
            ec_connector_output=bundle.ec_connector_output,
            cudagraph_stats=bundle.cudagraph_stats,
            batch_desc=bundle.batch_desc,
        )
        self.kv_connector_output = kv_connector_output
        if bundle.deferred_state_corrections_fn:
            bundle.deferred_state_corrections_fn()

    # ------------------------------------------------------------------
    # Merged attn ctx (head_tail shared-model edge batched forward)
    # ------------------------------------------------------------------
    def _get_bpb_for_layer(self, layer_name: str) -> int:
        """Look up ``blocks_per_phys_block`` for the kv_cache_group
        this layer belongs to.

        Each kv_cache_group has its own ``BlockTable`` with its own
        ``blocks_per_phys_block`` (see
        vllm_ascend/worker/block_table.py:47,67,72). Attention
        groups in hybrid mode have ``bpb > 1``; MambaSpec groups
        always have ``bpb == 1`` (``kernel_sizes=[0]``).
        """
        for group_idx, group in enumerate(
                self.kv_cache_config.kv_cache_groups):
            if layer_name in group.layer_names:
                return (
                    self.input_batch.block_table
                    .block_tables[group_idx].blocks_per_phys_block
                )
        raise KeyError(
            f"Layer {layer_name!r} not found in any kv_cache_group")

    def _is_c8_attn_layer(self, layer_name: str) -> bool:
        """Return whether ``layer_name`` uses the C8 (INT8 KV cache)
        backend.

        The C8 backend
        (``vllm_ascend.attention.attention_v1.AscendC8AttentionBackendImpl``)
        explicitly splits the merged batch into decode / prefill
        sub-batches in Python
        (``_forward_c8_chunked_prefill``). That split assumes a
        contiguous ``[decodes…][prefills…]`` layout, which the
        simple block_table / slot_mapping / cu_seqlen concat in
        ``_get_or_build_merged_attn_ctx`` does NOT guarantee when
        different dp_ranks have different per-dp_rank
        ``attn_state``. We therefore refuse C8 layers in the
        batched path; per-dp_rank execution handles them.
        """
        # The edge cloud shared model never runs through a
        # C8-quantized attention layer today, so returning False is
        # correct in practice.
        del layer_name
        return False

    def _apply_decode_first_reorder(
        self,
        need_reorder: bool,
        merged_query_start_loc_cpu: torch.Tensor,
        merged_num_reqs: int,
        merged_num_reqs_padded: int,
        merged_num_tokens_padded: int,
        merged_cudagraph_mode: "CUDAGraphMode",
        merged_seq_lens: torch.Tensor | None,
        merged_seq_lens_cpu: torch.Tensor | None,
        merged_seq_lens_cpu_upper: torch.Tensor | None,
        merged__seq_lens_cpu: torch.Tensor | None,
        merged_num_computed_tokens_cpu: torch.Tensor | None,
        merged_is_prefilling: torch.Tensor | None,
    ):
        """Apply the decode-first reorder to the cat-order
        ``merged_*`` tensors.

        When ``need_reorder`` is False (e.g.
        ``merged_attn_state == AscendAttentionState.DecodeOnly``,
        every req is a decode), this method still produces
        well-formed identity perm tensors
        (``merged_perm = arange(...)``, ``merged_token_perm =
        arange(...)``, etc.) and a ``merged_query_start_loc_cpu``
        copy of the input — so downstream permute / un-permute
        code in ``_get_or_build_merged_attn_ctx`` /
        ``execute_model_batched_head`` /
        ``execute_model_batched_tail`` stays uniform regardless
        of whether the reorder was actually applied.

        Returns a tuple of
        ``(merged_query_start_loc_cpu, merged_query_start_loc,
        merged_perm, merged_token_perm, inv_merged_token_perm,
        merged_seq_lens, merged_seq_lens_cpu,
        merged_seq_lens_cpu_upper, merged__seq_lens_cpu,
        merged_num_computed_tokens_cpu,
        merged_is_prefilling)``. ``merged_perm`` is the
        per-req perm used by the caller's Step 5 per-gid
        permute (block_tables / encoder_seq_lens /
        extra_args); when ``need_reorder`` is False it's
        ``None`` and Step 5 skips the permute.
        """
        if not need_reorder:
            # Identity perm: cat-order == decode-first order
            # (every actual row is a decode). Signal no-reorder
            # downstream by setting perm tensors to ``None``
            # — the head/tail stages ``is None`` checks skip
            # every permute / un-permute operation. ``merged_perm``
            # is also ``None``; the caller's Step 5 per-gid
            # permute gates on ``need_reorder`` so it skips.
            merged_perm = None
            merged_token_perm = None
            inv_merged_token_perm = None
            # The pre-allocated ``self.query_start_loc.gpu``
            # buffer is filled in the caller (FULL-mode copy
            # happens there too).
            merged_query_start_loc = (
                merged_query_start_loc_cpu.to(self.device))
            return (
                merged_query_start_loc_cpu,
                merged_query_start_loc,
                merged_perm,
                merged_token_perm,
                inv_merged_token_perm,
                merged_seq_lens,
                merged_seq_lens_cpu,
                merged_seq_lens_cpu_upper,
                merged__seq_lens_cpu,
                merged_num_computed_tokens_cpu,
                merged_is_prefilling,
            )

        # ---- Decode-first perm computation.
        # A req is classified as **prefill** iff
        # ``merged_is_prefilling[i]`` is set (sourced from
        # ``num_computed_tokens_cpu < num_prompt_tokens_cpu`` per
        # [model_runner_v1.py:4378](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L4378)
        # at per-dp_rank ``_build_attention_metadata`` time). The
        # ``(query_lens > 1)`` operand that previously OR'd here
        # was redundant in the only spec-decoding mixed path that
        # reaches this helper: for every
        # ``SpecDecoding ⊔ {ChunkedPrefill, PrefillNoCache}`` the
        # merged attn-state collapse at lines 1626-1636 routes the
        # merge to ``ChunkedPrefill`` (forcing reorder), and every
        # SpecDecoding req has ``num_valid_tokens == 1`` per
        # [model_runner_v1.py:1840-1844](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1840-L1844)
        # hence ``num_scheduled == 1 + num_drafts >= 2``, so the
        # query-length operand would already be True for the
        # whole SpecDecoding segment; the OR added no
        # decode-vs-prefill information. GDN builder's
        # spec-path num_decodes / num_prefills is computed
        # independently from ``spec_sequence_masks`` and
        # ``query_lens`` at
        # [gdn_attn.py:213-222](vllm/vllm/v1/attention/backends/gdn_attn.py#L213-L222)
        # (and only the ``spec_sequence_masks is None`` branch
        # at line 199 calls into ``split_decodes_and_prefills``,
        # which would otherwise be the vLLM-side ``is_prefill |=
        # is_prefilling`` analogue at
        # [utils.py:551-555](vllm/vllm/v1/attention/backends/utils.py#L551-L555)).
        # ``merged_query_lens_cpu_actual`` is still materialised
        # here because the reorder perm ``merged_perm`` needs it
        # at line 1521 to recompute ``merged_query_start_loc_cpu``
        # in decode-first order; only the prefill-classification
        # vector drops the ``query_lens`` operand.
        # Padded rows (idx >= merged_num_reqs) are appended
        # AFTER prefill rows so the kernel's ``first_prefill``
        # correctly identifies the decode/prefill boundary.
        merged_query_lens_cpu_actual = (
            merged_query_start_loc_cpu[1:merged_num_reqs + 1]
            - merged_query_start_loc_cpu[:merged_num_reqs])
        if merged_is_prefilling is not None:
            is_prefill_merged_actual = (
                merged_is_prefilling[:merged_num_reqs])
        else:
            is_prefill_merged_actual = (
                merged_query_lens_cpu_actual > 1)
        decode_indices = torch.where(
            ~is_prefill_merged_actual)[0]
        prefill_indices = torch.where(
            is_prefill_merged_actual)[0]
        padded_indices = torch.arange(
            merged_num_reqs, merged_num_reqs_padded,
            dtype=torch.int64)
        merged_perm = torch.cat([
            decode_indices.to(torch.int64),
            prefill_indices.to(torch.int64),
            padded_indices,
        ])

        # ``merged_token_perm`` maps cat-order token index →
        # reordered token index. Each req ``i_old`` in cat-order
        # has token range ``[cat_qsl[i_old]:cat_qsl[i_old+1]]``;
        # in the reordered tensor it occupies position
        # ``sum(query_lens[merged_perm[:k]])`` for ``k = new
        # position of i_old``. We build it by walking the
        # reorder in new-position order and emitting the
        # cat-order token indices for that req.
        cat_qsl_actual = merged_query_start_loc_cpu[
            :merged_num_reqs + 1]
        merged_token_perm_parts: list[torch.Tensor] = []
        for k in range(merged_num_reqs):
            i_old = merged_perm[k].item()
            s = cat_qsl_actual[i_old].item()
            e = cat_qsl_actual[i_old + 1].item()
            merged_token_perm_parts.append(
                torch.arange(s, e, dtype=torch.int64))
        if merged_token_perm_parts:
            merged_token_perm = torch.cat(
                merged_token_perm_parts)
        else:
            merged_token_perm = torch.empty(
                0, dtype=torch.int64)
        # ``inv_merged_token_perm``: reordered → cat-order.
        # Only valid for the actual-merged range
        # (``merged_num_actual_tokens``); padded token
        # positions are absent.
        inv_merged_token_perm = torch.empty_like(
            merged_token_perm)
        if merged_token_perm.numel() > 0:
            inv_merged_token_perm[merged_token_perm] = (
                torch.arange(
                    merged_token_perm.shape[0],
                    dtype=torch.int64))

        # ---- Apply ``merged_perm`` to per-req fields. After
        # this point, every per-req tensor is in decode-first
        # order.
        # ``merged_seq_lens_cpu_upper`` is never ``None``
        # (always present in the standard path). Permute it
        # and re-establish aliases for ``merged__seq_lens_cpu``
        # and ``merged_seq_lens_cpu`` (``merged__seq_lens_cpu``
        # is always non-None, ``merged_seq_lens_cpu`` may be
        # ``None`` in async spec decode — preserve that).
        merged_seq_lens_cpu_upper = (
            merged_seq_lens_cpu_upper[merged_perm].contiguous())
        merged__seq_lens_cpu = merged_seq_lens_cpu_upper
        if merged_seq_lens_cpu is not None:
            merged_seq_lens_cpu = merged_seq_lens_cpu_upper
        if merged_num_computed_tokens_cpu is not None:
            merged_num_computed_tokens_cpu = (
                merged_num_computed_tokens_cpu[merged_perm]
                .contiguous())
        if merged_is_prefilling is not None:
            merged_is_prefilling = (
                merged_is_prefilling[merged_perm].contiguous())
        # NPU ``merged_seq_lens`` is consumed by some kernels
        # too; permute it as well.
        if merged_seq_lens is not None:
            merged_seq_lens = (
                merged_seq_lens[merged_perm].contiguous())

        # ---- Re-cumsum the actual portion of
        # ``merged_query_start_loc_cpu`` from the permuted
        # per-req scheduled token count
        # (``merged_query_lens_cpu_actual`` already computed
        # above for the perm construction), and reuse the
        # already-built padded tail from Step 3. Padded rows
        # are dummy (uniform/mixed constant or arange);
        # reusing the cat-order padded tail is safe because
        # their values don't depend on req order. NOTE:
        # ``seq_lens`` is NOT the right input — it is the
        # total context length, not the scheduled count, and
        # these differ for prefill reqs.
        qsl_actual = torch.zeros(
            merged_num_reqs + 1, dtype=torch.int32)
        qsl_actual[1:] = torch.cumsum(
            merged_query_lens_cpu_actual[merged_perm], dim=0)
        merged_query_start_loc_cpu = torch.cat([
            qsl_actual,
            merged_query_start_loc_cpu[merged_num_reqs + 1:],
        ])
        # The pre-allocated ``self.query_start_loc.gpu``
        # buffer is filled in the caller (FULL-mode copy
        # happens there too).
        merged_query_start_loc = (
            merged_query_start_loc_cpu.to(self.device))

        return (
            merged_query_start_loc_cpu,
            merged_query_start_loc,
            merged_perm,
            merged_token_perm,
            inv_merged_token_perm,
            merged_seq_lens,
            merged_seq_lens_cpu,
            merged_seq_lens_cpu_upper,
            merged__seq_lens_cpu,
            merged_num_computed_tokens_cpu,
            merged_is_prefilling,
        )

    def _get_or_build_merged_attn_ctx(
        self,
        bundles: list["_ExecuteModelBundle"],
        batched_dp_ranks: list[int],
        pp_send_work_by_channel: dict | None = None,
    ) -> "_MergedAttnContext":
        """Build (or reuse cached) merged per-layer attn state.

        ``bundle.attn_metadata`` is a per-layer dict
        (``AttnMetadataDict = dict[layer_name, AscendMetadata]``);
        the attention kernel reads it as
        ``forward_context.attn_metadata[layer]`` per layer. We merge
        **per layer** across dp_ranks so that the result is itself a
        per-layer dict that the kernel can consume unchanged.

        Called once per round on the LEADER runner by
        ``_BatchedExecuteMarker.run_batched_head``;
        ``drain_batched_round`` reuses the cache for the tail
        segment.
        """
        cached = getattr(self, "_merged_attn_ctx_cache", None)
        if cached is not None:
            logger.info(
                "[PD] _get_or_build_merged_attn_ctx: "
                "REUSING cached ctx")
            return cached

        logger.info(
            "[PD] _get_or_build_merged_attn_ctx: "
            "BUILDING new ctx dp_ranks=%s", batched_dp_ranks)

        per_dp_offsets = getattr(self, "_per_dp_offsets", None)
        if per_dp_offsets is None:
            raise RuntimeError(
                "BatchedModelRunner._get_or_build_merged_attn_ctx "
                "called but self._per_dp_offsets is unset. Did "
                "initialize_kv_cache run for all dp_ranks?")

        from vllm_ascend.attention.attention_v1 import (
            AscendAttentionState,
        )

        dp_size = self.parallel_config.data_parallel_size
        bs = self.block_size
        num_kv_cache_gids = len(self.kv_cache_config.kv_cache_groups)

        # ---- Step 0: per-dp_rank unpad (slide per-req fields down
        # to actual sizes via ``cm.unpadded``).
        cms_unpadded: list[AscendCommonAttentionMetadata] = []
        for b in bundles:
            cm = b.common_attn_metadata
            cms_unpadded.append(
                cm.unpadded(cm.num_actual_tokens, b.num_reqs_actual))
        logger.info("[PD] _get_or_build_merged_attn_ctx: Step 0 unpad done n_unpadded=%d", len(cms_unpadded))

        # ---- Step 1: merged scalars + ``attn_state`` unification.
        merged_num_actual_tokens = sum(
            cm.num_actual_tokens for cm in cms_unpadded)
        merged_max_query_len = max(cm.max_query_len for cm in cms_unpadded)
        merged_max_seq_len = max(cm.max_seq_len for cm in cms_unpadded)
        # ``merged_num_reqs`` is the sum of per-dp_rank actual
        # request counts (NOT the cudagraph-padded total). The
        # merged tensor is initially this size, and per-req fields
        # are then padded out to ``merged_num_reqs_padded`` (see
        # ``_cat_per_req_pad``).
        merged_num_reqs = sum(b.num_reqs_actual for b in bundles)

        states = [cm.attn_state for cm in cms_unpadded]
        unique_states = set(states)
        if len(unique_states) == 1:
            merged_attn_state = next(iter(unique_states))
        elif unique_states <= {AscendAttentionState.DecodeOnly,
                               AscendAttentionState.SpecDecoding}:
            # DecodeOnly ⊔ SpecDecoding are treated identically by
            # AscendSFAImpl / AscendMLAImpl / AscendDSAImpl /
            # AscendDSACPImpl build paths (they key on
            # ``attn_state in {DecodeOnly, SpecDecoding}`` — see
            # sfa_v1.py:355 / mla_v1.py:667 / dsa_v1.py:1332 /
            # dsa_cp.py:852). Picking SpecDecoding here keeps
            # spec-aware forward and avoids the prefill-like
            # ``else`` branch which assumes >= 2-token queries.
            # The GDN builder additionally still receives
            # spec-decoding fields through
            # ``merged_extra_args`` below (line 2213-2252),
            # derived from ``num_decode_draft_tokens_cpu`` (which
            # has no attn_state dependency — see
            # gdn_attn.py:178-313).
            merged_attn_state = AscendAttentionState.SpecDecoding
        else:
            # Any mixed set that includes a prefill state
            # (``ChunkedPrefill`` / ``PrefillNoCache``) collapses
            # to ``ChunkedPrefill``. The per-dp_rank
            # ``_build_attn_state`` already guarantees the merged
            # ``np.all(num_computed_tokens_cpu == 0)`` cannot hold
            # (any decode-bearing dp_rank has ``num_computed > 0``)
            # when the merged batch includes decode-class dp_ranks,
            # so ``PrefillNoCache`` is unreachable here — keep it
            # exclusively for the single-state branch above.
            merged_attn_state = AscendAttentionState.ChunkedPrefill
        logger.info("[PD] _get_or_build_merged_attn_ctx: Step 1 scalars done merged_num_actual_tokens=%d merged_num_reqs=%d", merged_num_actual_tokens, merged_num_reqs)

        # ---- Step 2: dispatch the merged batch to get the merged
        # padded sizes and the merged ``CUDAGraphMode``. The merged
        # mode is what the batched head/tail forward should run
        # under (NOT a single dp_rank's mode).
        any_batch_desc = bundles[0].batch_desc
        if any_batch_desc is None:
            merged_batch_descriptor = None
            merged_cudagraph_mode = CUDAGraphMode.NONE
            merged_num_tokens_padded = merged_num_actual_tokens
            merged_num_reqs_padded = merged_num_reqs
        else:
            has_lora_any = any(
                b.batch_desc is not None and b.batch_desc.has_lora
                for b in bundles)
            uniform_decode_merged = all(
                b.batch_desc is not None and b.batch_desc.uniform
                for b in bundles)
            num_active_loras_merged = sum(
                (b.batch_desc.num_active_loras
                 if b.batch_desc is not None else 0)
                for b in bundles)
            merged_cudagraph_mode, merged_batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(
                    num_tokens=merged_num_actual_tokens,
                    has_lora=has_lora_any,
                    uniform_decode=uniform_decode_merged,
                    valid_modes=None,
                    invalid_modes=None,
                    num_active_loras=num_active_loras_merged,
                ))
            merged_num_reqs_padded = (
                merged_batch_descriptor.num_reqs
                if merged_batch_descriptor.num_reqs is not None
                else merged_num_reqs)
            merged_num_tokens_padded = (
                merged_batch_descriptor.num_tokens)
        logger.info("[PD] _get_or_build_merged_attn_ctx: Step 2 dispatch done mode=%s num_tokens_padded=%d", merged_cudagraph_mode, merged_num_tokens_padded)
        logger.info("[PD] _get_or_build_merged_attn_ctx: _pp_send_work_by_channel=%s", pp_send_work_by_channel)

        # ---- Step 3: merged cu_seqlen (``query_start_loc`` /
        # ``query_start_loc_cpu``). Unpad → cumsum-merge → pad to
        # ``merged_num_reqs_padded + 1`` by repeating the last
        # value (padded rows have query length 0).
        merged_qsl_cpu_list = [0]
        for cm in cms_unpadded:
            n = cm.num_reqs  # == num_reqs_actual (unpadded)
            if n > 0:
                local = cm.query_start_loc_cpu[1:n + 1]
                cumulative = local + merged_qsl_cpu_list[-1]
                merged_qsl_cpu_list.extend(cumulative.tolist())
            else:
                merged_qsl_cpu_list.append(merged_qsl_cpu_list[-1])
        merged_query_start_loc_cpu = torch.tensor(
            merged_qsl_cpu_list, dtype=torch.int32, device="cpu")

        if (merged_query_start_loc_cpu.shape[0] - 1
                < merged_num_reqs_padded):
            last = merged_query_start_loc_cpu[-1].item()
            if merged_num_tokens_padded == merged_num_reqs_padded * self.uniform_decode_query_len:
                pad = torch.arange(1, merged_num_reqs_padded + 1 - merged_num_reqs, dtype=torch.int32, device="cpu") * self.uniform_decode_query_len + last
            else:
                pad = torch.full(
                    (merged_num_reqs_padded + 1
                     - merged_query_start_loc_cpu.shape[0],),
                    last, dtype=torch.int32, device="cpu")
            merged_query_start_loc_cpu = torch.cat(
                [merged_query_start_loc_cpu, pad])
        elif (merged_query_start_loc_cpu.shape[0] - 1
              > merged_num_reqs_padded):
            merged_query_start_loc_cpu = merged_query_start_loc_cpu[
                :merged_num_reqs_padded + 1]

        merged_query_start_loc = merged_query_start_loc_cpu.to(
            self.device)
        logger.info("[PD] _get_or_build_merged_attn_ctx: Step 3 cu_seqlen done query_start_length=%d", len(merged_qsl_cpu_list))
        logger.info("[PD] _get_or_build_merged_attn_ctx: _pp_send_work_by_channel=%s", pp_send_work_by_channel)
        # The FULL mode buffer copy happens AFTER the
        # decode-first reorder (further down) so the buffer
        # reflects the reordered (decode-first) layout the
        # attention kernel consumes.

        # ---- Step 4: merged per-req fields via unpad-cat-pad.
        def _cat_per_req_pad(get_field, pad_value, dtype,
                             device) -> torch.Tensor | None:
            parts = [get_field(cm) for cm in cms_unpadded]
            parts = [p for p in parts if p is not None]
            if not parts:
                return None
            merged = torch.cat(parts, dim=0)
            if merged.shape[0] < merged_num_reqs_padded:
                pad_shape = list(merged.shape)
                pad_shape[0] = (
                    merged_num_reqs_padded - merged.shape[0])
                pad = torch.full(
                    pad_shape, pad_value, dtype=dtype, device=device)
                merged = torch.cat([merged, pad], dim=0)
            elif merged.shape[0] > merged_num_reqs_padded:
                merged = merged[:merged_num_reqs_padded]
            return merged

        any_cm = cms_unpadded[0] if cms_unpadded else bundles[
            0].common_attn_metadata
        merged_seq_lens = _cat_per_req_pad(
            lambda cm: cm.seq_lens, 0, any_cm.seq_lens.dtype,
            self.device)
        merged_seq_lens_cpu_upper = _cat_per_req_pad(
            lambda cm: cm.seq_lens_cpu_upper_bound, 0,
            any_cm.seq_lens_cpu_upper_bound.dtype
            if any_cm.seq_lens_cpu_upper_bound is not None
            else torch.int32,
            "cpu") if any(cm.seq_lens_cpu_upper_bound is not None
                          for cm in cms_unpadded) else None
        # Per ``model_runner_v1.py:4393-4396`` invariant:
        # ``_seq_lens_cpu`` is always
        # ``self.optimistic_seq_lens_cpu[:num_reqs_padded]`` (same
        # buffer as ``seq_lens_cpu_upper_bound``); ``seq_lens_cpu``
        # is either that same buffer or ``None`` (async spec
        # decode). Reuse the upper buffer as an alias here so the
        # three fields share one cat output and one FULL-mode
        # ``copy_``.
        merged__seq_lens_cpu = merged_seq_lens_cpu_upper
        if any(cm.seq_lens_cpu is not None for cm in cms_unpadded):
            merged_seq_lens_cpu = merged_seq_lens_cpu_upper
        else:
            merged_seq_lens_cpu = None
        merged_num_computed_tokens_cpu = _cat_per_req_pad(
            lambda cm: cm.num_computed_tokens_cpu, 0,
            any_cm.num_computed_tokens_cpu.dtype
            if any_cm.num_computed_tokens_cpu is not None
            else torch.int32,
            "cpu") if any(cm.num_computed_tokens_cpu is not None
                          for cm in cms_unpadded) else None
        # ``is_prefilling``: pad value is False (matches
        # ``_build_attention_metadata`` line 4378). The tensor
        # itself is on CPU (built by comparing the CPU
        # ``num_computed_tokens_cpu_tensor`` /
        # ``num_prompt_tokens_cpu_tensor``), so the merged tensor
        # also lives on CPU.
        merged_is_prefilling = _cat_per_req_pad(
            lambda cm: cm.is_prefilling, False,
            torch.bool, "cpu")
        if merged_is_prefilling is not None:
            merged_is_prefilling[merged_num_reqs:] = False

        # ---- Decode-first reorder (2-way: decode → prefill).
        # Many attention builders (notably
        # ``GDNAttentionMetadataBuilder``) call
        # ``split_decodes_and_prefills`` which assumes the batch
        # is already in decode-first order
        # (``first_prefill = argmax(is_prefill)`` gives
        # ``num_decodes`` directly). Per-dp_rank ``_may_reorder_
        # batch`` only sorts INSIDE a single dp_rank; the merged
        # batch still needs a global decode-first reorder.
        #
        # The reorder is a no-op when ``merged_attn_state ==
        # AscendAttentionState.DecodeOnly`` (every req is a
        # decode) — ``decode_indices`` already covers all
        # actual rows in cat order, ``prefill_indices`` is
        # empty, padded rows go to the tail unchanged.
        # ``_apply_decode_first_reorder`` short-circuits in
        # that case and produces identity perm tensors so the
        # downstream permute calls below stay uniform.
        need_reorder = (
            merged_attn_state
            not in (
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding,
            ))
        logger.info(
            "[PD] _get_or_build_merged_attn_ctx: "
            "calling _apply_decode_first_reorder "
            "need_reorder=%s merged_num_reqs=%d merged_num_reqs_padded=%d",
            need_reorder, merged_num_reqs, merged_num_reqs_padded)
        (
            merged_query_start_loc_cpu,
            merged_query_start_loc,
            merged_perm,
            merged_token_perm,
            inv_merged_token_perm,
            merged_seq_lens,
            merged_seq_lens_cpu,
            merged_seq_lens_cpu_upper,
            merged__seq_lens_cpu,
            merged_num_computed_tokens_cpu,
            merged_is_prefilling,
        ) = self._apply_decode_first_reorder(
            need_reorder=need_reorder,
            merged_query_start_loc_cpu=(
                merged_query_start_loc_cpu),
            merged_num_reqs=merged_num_reqs,
            merged_num_reqs_padded=merged_num_reqs_padded,
            merged_num_tokens_padded=merged_num_tokens_padded,
            merged_cudagraph_mode=merged_cudagraph_mode,
            merged_seq_lens=merged_seq_lens,
            merged_seq_lens_cpu=merged_seq_lens_cpu,
            merged_seq_lens_cpu_upper=(
                merged_seq_lens_cpu_upper),
            merged__seq_lens_cpu=merged__seq_lens_cpu,
            merged_num_computed_tokens_cpu=(
                merged_num_computed_tokens_cpu),
            merged_is_prefilling=merged_is_prefilling,
        )
        # FULL mode: copy the permuted
        # ``merged_seq_lens_cpu_upper`` to the leader runner's
        # pre-allocated ``self.optimistic_seq_lens_cpu`` buffer
        # so the attention kernel reads from a stable CPU
        # pointer across cudagraph captures. Re-establish the
        # aliases for ``merged_seq_lens_cpu`` /
        # ``merged__seq_lens_cpu`` so they share the same
        # stable address (matching the standard path's
        # invariant at ``model_runner_v1.py:4393-4396``).
        if merged_cudagraph_mode == CUDAGraphMode.FULL:
            self.optimistic_seq_lens_cpu[
                :merged_num_reqs_padded].copy_(
                    merged_seq_lens_cpu_upper)
            seq_lens_cpu_buf = (
                self.optimistic_seq_lens_cpu[
                    :merged_num_reqs_padded])
            merged_seq_lens_cpu_upper = seq_lens_cpu_buf
            merged__seq_lens_cpu = seq_lens_cpu_buf
            if merged_seq_lens_cpu is not None:
                merged_seq_lens_cpu = seq_lens_cpu_buf
            # NPU ``merged_seq_lens``: copy to leader runner's
            # ``self.seq_lens`` buffer so the attention kernel
            # reads from a stable device pointer across
            # cudagraph captures.
            if merged_seq_lens is not None:
                self.seq_lens[:merged_num_reqs_padded].copy_(
                    merged_seq_lens)
                merged_seq_lens = (
                    self.seq_lens[:merged_num_reqs_padded])
            # ``merged_query_start_loc``: copy to leader runner's
            # pre-allocated ``self.query_start_loc.gpu`` buffer
            # so the kernel reads from a stable device pointer
            # across cudagraph captures.
            self.query_start_loc.gpu[
                :merged_num_reqs_padded + 1].copy_(
                    merged_query_start_loc)
            merged_query_start_loc = (
                self.query_start_loc.gpu[
                    :merged_num_reqs_padded + 1])
        logger.info(
            "[PD] _get_or_build_merged_attn_ctx: "
            "decode_reorder+FULL_buffers done, "
            "cudagraph_mode=%s",
            merged_cudagraph_mode)

        # ``positions`` / ``positions_cpu`` (DSA): ``unpadded``
        # keeps them full-length; slice to
        # ``[:num_actual_tokens]`` and cat.
        if all(cm.positions is not None for cm in cms_unpadded):
            merged_positions = torch.cat(
                [cm.positions[:cm.num_actual_tokens]
                 for cm in cms_unpadded],
                dim=-1)
            if merged_positions.shape[-1] < merged_num_tokens_padded:
                pad_shape = list(merged_positions.shape)
                pad_shape[-1] = (
                    merged_num_tokens_padded
                    - merged_positions.shape[-1])
                pad = torch.zeros(
                    pad_shape, dtype=merged_positions.dtype,
                    device=merged_positions.device)
                merged_positions = torch.cat(
                    [merged_positions, pad], dim=-1)
            elif (merged_positions.shape[-1]
                  > merged_num_tokens_padded):
                merged_positions = merged_positions[
                    ..., :merged_num_tokens_padded]
            # Decode-first reorder: ``positions`` is per-token,
            # apply ``merged_token_perm`` over the actual range.
            # Padded positions (>= merged_num_actual_tokens)
            # keep their zero fill.
            if (merged_num_actual_tokens > 0
                    and need_reorder):
                merged_positions_actual = (
                    merged_positions[
                        ..., :merged_num_actual_tokens][
                        ..., merged_token_perm])
                merged_positions = torch.cat([
                    merged_positions_actual,
                    merged_positions[
                        ..., merged_num_actual_tokens:],
                ], dim=-1)
            # FULL mode: copy to leader runner's pre-allocated
            # ``self.positions`` buffer so ``cm_merged.positions``
            # (a slice view of this buffer) and
            # ``_model_forward(positions=...)`` share a stable
            # device pointer across cudagraph captures.
            if merged_cudagraph_mode == CUDAGraphMode.FULL:
                self.positions[:merged_num_tokens_padded].copy_(
                    merged_positions)
                merged_positions = (
                    self.positions[:merged_num_tokens_padded])
        else:
            merged_positions = None
        if all(cm.positions_cpu is not None for cm in cms_unpadded):
            merged_positions_cpu = torch.cat(
                [cm.positions_cpu[:cm.num_actual_tokens]
                 for cm in cms_unpadded],
                dim=-1)
            if (merged_positions_cpu.shape[-1]
                    < merged_num_tokens_padded):
                pad_shape = list(merged_positions_cpu.shape)
                pad_shape[-1] = (
                    merged_num_tokens_padded
                    - merged_positions_cpu.shape[-1])
                pad = torch.zeros(
                    pad_shape,
                    dtype=merged_positions_cpu.dtype,
                    device=merged_positions_cpu.device)
                merged_positions_cpu = torch.cat(
                    [merged_positions_cpu, pad], dim=-1)
            elif (merged_positions_cpu.shape[-1]
                  > merged_num_tokens_padded):
                merged_positions_cpu = merged_positions_cpu[
                    ..., :merged_num_tokens_padded]
            if (merged_num_actual_tokens > 0
                    and need_reorder):
                merged_positions_cpu_actual = (
                    merged_positions_cpu[
                        ..., :merged_num_actual_tokens][
                        ..., merged_token_perm])
                merged_positions_cpu = torch.cat([
                    merged_positions_cpu_actual,
                    merged_positions_cpu[
                        ..., merged_num_actual_tokens:],
                ], dim=-1)
        else:
            merged_positions_cpu = None

        # ``actual_seq_lengths_q`` is the per-req prefix-sum of
        # scheduled tokens (``query_start_loc_cpu[1:]``), built
        # by ``AscendMetadataBuilder.build`` as
        # ``query_start_loc_cpu[1:].tolist()``. Each element is
        # the cumulative token count at that req — NOT a
        # per-token expansion. So the merged value is just the
        # reordered merged qsl minus its leading ``0``
        # (``qsl[0]`` is always 0).
        if all(cm.actual_seq_lengths_q is not None
               and len(cm.actual_seq_lengths_q) > 0
               for cm in cms_unpadded):
            merged_actual_seq_lengths_q = (
                merged_query_start_loc_cpu[1:].tolist())
        else:
            merged_actual_seq_lengths_q = None
        logger.info(
            "[PD] _get_or_build_merged_attn_ctx: "
            "positions+actual_seq_lengths merged, "
            "entering Step 5")

        # ---- Step 5: per-(kv_cache_gid, attn_gid) build.
        merged_attn_metadata: dict = {}
        from vllm.v1.attention.backends.gdn_attn import (
            GDNAttentionMetadataBuilder,
        )
        logger.info(
            "[PD] _get_or_build_merged_attn_ctx: "
            "entering Step 5 per-layer build "
            "kv_gids=%d merged_num_reqs=%d merged_num_tokens=%d "
            "merged_num_reqs_padded=%d merged_num_tokens_padded=%d "
            "cudagraph_mode=%s",
            num_kv_cache_gids, merged_num_reqs,
            merged_num_actual_tokens, merged_num_reqs_padded,
            merged_num_tokens_padded, merged_cudagraph_mode)
        for kv_cache_gid in range(num_kv_cache_gids):
            for attn_gid in range(
                    len(self.attn_groups[kv_cache_gid])):
                attn_group = self.attn_groups[kv_cache_gid][attn_gid]
                builder = attn_group.get_metadata_builder(0)
                bpb = self._get_bpb_for_layer(
                    attn_group.layer_names[0])

                # Per-gid block_table_tensor. gid=0 is on
                # ``cm_base.block_table_tensor`` (kept full-length
                # by ``unpadded``); gid>0 lives on
                # ``b.per_gid_cm[kv_cache_gid]["block_table_tensor"]``
                # (saved by ``_save('block_table_tensor')`` in
                # ``_build_attention_metadata`` only for
                # ``kv_cache_gid > 0``). Unpad to actual
                # num_reqs_actual, remap+cat, pad to
                # ``merged_num_reqs_padded``.
                logger.info(
                    "[PD] _get_or_build_merged_attn_ctx: "
                    "Step 5 block_table remap kv_gid=%d attn_gid=%d "
                    "dp_ranks=%s",
                    kv_cache_gid, attn_gid, batched_dp_ranks)
                parts_bt = []
                for cm_unpadded, b, k in zip(
                        cms_unpadded, bundles, batched_dp_ranks):
                    if kv_cache_gid == 0:
                        gid_bt = cm_unpadded.block_table_tensor
                    else:
                        gid_bt = b.per_gid_cm[kv_cache_gid][
                            "block_table_tensor"]
                    local = gid_bt[:b.num_reqs_actual]
                    remapped = torch.where(
                        local == 0, local,
                        ((local // bpb) * dp_size
                         + per_dp_offsets[k]) * bpb
                        + (local % bpb))
                    parts_bt.append(remapped)
                merged_block_tables = torch.cat(parts_bt, dim=0)
                if (merged_block_tables.shape[0]
                        < merged_num_reqs_padded):
                    pad_shape = list(merged_block_tables.shape)
                    pad_shape[0] = (
                        merged_num_reqs_padded
                        - merged_block_tables.shape[0])
                    pad = torch.zeros(
                        pad_shape,
                        dtype=merged_block_tables.dtype,
                        device=merged_block_tables.device)
                    merged_block_tables = torch.cat(
                        [merged_block_tables, pad], dim=0)
                elif (merged_block_tables.shape[0]
                      > merged_num_reqs_padded):
                    merged_block_tables = merged_block_tables[
                        :merged_num_reqs_padded]
                # Decode-first reorder: ``block_table_tensor`` is
                # per-req, so apply ``merged_perm``. Padded rows
                # (``merged_perm[merged_num_reqs:]``) map to
                # themselves so the row identity is preserved.
                # In DecodeOnly mode (``need_reorder=False``) the
                # perm is the identity map so the slice is a
                # no-op; ``.contiguous()`` is still safe.
                if need_reorder:
                    merged_block_tables = (
                        merged_block_tables[merged_perm]
                        .contiguous())
                # FULL mode: copy to leader runner's pre-allocated
                # per-gid block_table buffer so ``cm_merged``
                # (a slice view of this buffer) has a stable
                # device pointer across cudagraph captures.
                if merged_cudagraph_mode == CUDAGraphMode.FULL:
                    self.input_batch.block_table[
                        kv_cache_gid].block_table.gpu[
                            :merged_num_reqs_padded].copy_(
                                merged_block_tables)
                    merged_block_tables = (
                        self.input_batch.block_table[
                            kv_cache_gid].block_table.gpu[
                                :merged_num_reqs_padded])

                # Per-gid slot_mapping.
                parts_sm = []
                for cm_unpadded, b, k in zip(
                        cms_unpadded, bundles, batched_dp_ranks):
                    if kv_cache_gid == 0:
                        gid_sm = cm_unpadded.slot_mapping
                    else:
                        gid_sm = b.per_gid_cm[kv_cache_gid][
                            "slot_mapping"]
                    n = b.common_attn_metadata.num_actual_tokens
                    local = gid_sm[:n]
                    remapped = torch.where(
                        local == PAD_SLOT_ID, local,
                        ((local // bs) * dp_size
                         + per_dp_offsets[k]) * bs
                        + (local % bs))
                    parts_sm.append(remapped)
                merged_slot_mapping = torch.cat(parts_sm, dim=0)
                if (merged_slot_mapping.shape[0]
                        < merged_num_tokens_padded):
                    pad = torch.full(
                        (merged_num_tokens_padded
                         - merged_slot_mapping.shape[0],),
                        PAD_SLOT_ID,
                        dtype=merged_slot_mapping.dtype,
                        device=merged_slot_mapping.device)
                    merged_slot_mapping = torch.cat(
                        [merged_slot_mapping, pad], dim=0)
                elif (merged_slot_mapping.shape[0]
                      > merged_num_tokens_padded):
                    merged_slot_mapping = merged_slot_mapping[
                        :merged_num_tokens_padded]
                # Decode-first reorder: ``slot_mapping`` is
                # per-token, so apply ``merged_token_perm`` over
                # the actual-merged range. Padded positions
                # (``>= merged_num_actual_tokens``) keep their
                # ``PAD_SLOT_ID`` filler untouched. In
                # DecodeOnly mode the perm is the identity map
                # so this is skipped.
                if (merged_num_actual_tokens > 0
                        and need_reorder):
                    merged_slot_mapping_actual = (
                        merged_slot_mapping[
                            :merged_num_actual_tokens][
                            merged_token_perm])
                    merged_slot_mapping = torch.cat([
                        merged_slot_mapping_actual,
                        merged_slot_mapping[
                            merged_num_actual_tokens:],
                    ])
                # FULL mode: copy to leader runner's pre-allocated
                # per-gid slot_mapping buffer so ``cm_merged``
                # (a slice view of this buffer) has a stable
                # device pointer across cudagraph captures.
                if merged_cudagraph_mode == CUDAGraphMode.FULL:
                    self.input_batch.block_table[
                        kv_cache_gid].slot_mapping.gpu[
                            :merged_num_tokens_padded].copy_(
                                merged_slot_mapping)
                    merged_slot_mapping = (
                        self.input_batch.block_table[
                            kv_cache_gid].slot_mapping.gpu[
                            :merged_num_tokens_padded])

                # Per-gid encoder_seq_lens / encoder_seq_lens_cpu.
                def _cat_per_gid_per_req(field_name, pad_value):
                    tensors = []
                    for b in bundles:
                        t = b.per_gid_cm[kv_cache_gid].get(field_name)
                        if t is not None:
                            tensors.append((b, t))
                    if not tensors:
                        return None
                    merged = torch.cat(
                        [t[:b.num_reqs_actual] for b, t in tensors],
                        dim=0)
                    if merged.shape[0] < merged_num_reqs_padded:
                        pad_shape = list(merged.shape)
                        pad_shape[0] = (
                            merged_num_reqs_padded - merged.shape[0])
                        pad = torch.full(
                            pad_shape, pad_value,
                            dtype=merged.dtype,
                            device=merged.device)
                        merged = torch.cat([merged, pad], dim=0)
                    elif merged.shape[0] > merged_num_reqs_padded:
                        merged = merged[:merged_num_reqs_padded]
                    return merged

                merged_encoder_seq_lens = _cat_per_gid_per_req(
                    "encoder_seq_lens", 0)
                merged_encoder_seq_lens_cpu = _cat_per_gid_per_req(
                    "encoder_seq_lens_cpu", 0)
                # Decode-first reorder: encoder_seq_lens is
                # per-req, apply ``merged_perm`` (padded rows
                # carry 0 and reorder among themselves, harmless).
                # In DecodeOnly mode the perm is the identity
                # map so the slice is skipped.
                if need_reorder:
                    if merged_encoder_seq_lens is not None:
                        merged_encoder_seq_lens = (
                            merged_encoder_seq_lens[merged_perm]
                            .contiguous())
                    if merged_encoder_seq_lens_cpu is not None:
                        merged_encoder_seq_lens_cpu = (
                            merged_encoder_seq_lens_cpu[merged_perm]
                            .contiguous())

                # GDN per-gid override of ``query_start_loc`` /
                # ``query_start_loc_cpu``. Each dp_rank's
                # ``gdn_query_start_loc`` is the unpadded version of
                # its cm_base ``query_start_loc`` (set by
                # ``_build_attention_metadata`` line 4538-4539 when
                # the builder is ``GDNAttentionMetadataBuilder``).
                # NOTE: the standard ``query_start_loc`` and
                # ``gdn_query_start_loc`` are filled with the
                # same cumsum in the per-dp_rank path (see
                # model_runner_v1.py:1395-1407 and 4756-4762);
                # GDN gid uses ``query_start_loc`` / ``query_start_loc_cpu``
                # to derive ``query_lens_cpu = qsl[1:] - qsl[:-1]`` for
                # ``split_decodes_and_prefills`` and reads
                # ``qsl[-1]`` for ``spec_token_size``. The per-dp_rank
                # ``gdn_query_start_loc`` (``model_runner_v1.py:1403-1407``)
                # pads the tail with a CONSTANT
                # (``cu_num_tokens[-1]``), NOT with the
                # uniform-decode-style arange that the standard
                # ``query_start_loc`` may use
                # (``model_runner_v1.py:1172-1212``). After our
                # decode-first reorder, the per-req cumsum is the
                # same as ``merged_query_start_loc_cpu`` (same
                # ``cu_num_tokens``); we just overwrite the padded
                # tail (entries >= ``merged_num_reqs + 1``) with the
                # constant ``merged_num_actual_tokens`` so padded
                # rows have ``qsl[k+1] - qsl[k] == 0`` (matching
                # per-dp_rank GDN semantics). Without this, on a
                # uniform-decode merged batch the GDN gid would see
                # padded rows with ``qsl_diff = uniform_decode_query_len``
                # and ``split_decodes_and_prefills`` would
                # misclassify them.
                gdn_merged_qsl_cpu = None
                gdn_merged_qsl = None
                if isinstance(builder, GDNAttentionMetadataBuilder):
                    gdn_merged_qsl_cpu = merged_query_start_loc_cpu.clone()
                    gdn_merged_qsl_cpu[
                        merged_num_reqs + 1:] = merged_num_actual_tokens
                    gdn_merged_qsl = gdn_merged_qsl_cpu.to(self.device)

                # Assemble the per-gid merged cm.
                cm_merged = AscendCommonAttentionMetadata(
                    query_start_loc=(
                        gdn_merged_qsl
                        if gdn_merged_qsl is not None
                        else merged_query_start_loc),
                    query_start_loc_cpu=(
                        gdn_merged_qsl_cpu
                        if gdn_merged_qsl_cpu is not None
                        else merged_query_start_loc_cpu),
                    seq_lens=merged_seq_lens,
                    seq_lens_cpu=merged_seq_lens_cpu,
                    num_reqs=merged_num_reqs_padded,
                    num_actual_tokens=merged_num_actual_tokens,
                    max_query_len=merged_max_query_len,
                    max_seq_len=merged_max_seq_len,
                    block_table_tensor=merged_block_tables,
                    slot_mapping=merged_slot_mapping,
                    encoder_seq_lens=merged_encoder_seq_lens,
                    encoder_seq_lens_cpu=merged_encoder_seq_lens_cpu,
                    causal=any_cm.causal,
                    is_prefilling=merged_is_prefilling,
                    num_input_tokens=merged_num_tokens_padded,
                    actual_seq_lengths_q=merged_actual_seq_lengths_q,
                    positions=merged_positions,
                    positions_cpu=merged_positions_cpu,
                    attn_state=merged_attn_state,
                    num_computed_tokens_cpu=(
                        merged_num_computed_tokens_cpu),
                    decode_token_per_req=any_cm.decode_token_per_req,
                    seq_lens_cpu_upper_bound=(
                        merged_seq_lens_cpu_upper),
                    _seq_lens_cpu=merged__seq_lens_cpu,
                )

                # Per-(kv_cache_gid, attn_gid) extras. Read from
                # ``bundle.per_gid_extra[(k, a)]`` (the
                # ``(cascade_attn_prefix_len,
                # extra_attn_metadata_args)`` tuple saved by
                # ``_build_attn_group_metadata``).
                #
                # Per-dp_rank pipelines can each populate a
                # different subset of ``extra_attn_metadata_args``
                # keys (e.g. only the GDN+spec-bearing dp_rank
                # carries ``num_accepted_tokens`` /
                # ``num_decode_draft_tokens_cpu``; a dp_rank with
                # DSA backends carries DSA-specific keys). The
                # union of those keys is what the merged builder
                # must receive, so we first union the keys over
                # all bundles and then compute ``merged_extra_args``
                # key-by-key (instead of iterating the first
                # bundle's key order — which silently drops keys
                # present only in later bundles).
                # Materialise the per-bundle ``(cascade_attn_prefix_len,
                # extra_attn_metadata_args)`` tuple once so the
                # key-by-key loop only pays for ``NB`` indexing
                # lookups total (instead of ``NB`` per key).
                per_bundle_extra_args = [
                    b.per_gid_extra[(kv_cache_gid, attn_gid)]
                    for b in bundles]
                merged_cascade_attn_prefix_len = 0
                for cascade_attn_prefix_len, _ in (
                        per_bundle_extra_args):
                    merged_cascade_attn_prefix_len += (
                        cascade_attn_prefix_len)
                # Union of keys across all bundles.
                extra_args_keys: set[str] = set()
                for _, extra_args in per_bundle_extra_args:
                    extra_args_keys.update(extra_args.keys())
                merged_extra_args: dict[str, Any] = {}
                for k_key in extra_args_keys:
                    if k_key in ("num_accepted_tokens",
                                 "num_decode_draft_tokens_cpu"):
                        # Per-req tensor merge: every bundle
                        # contributes its ``b.num_reqs_actual``
                        # slice — either its real ``v[k_key]``
                        # value or, for bundles that did not
                        # populate the key this round, a default
                        # tensor of length ``b.num_reqs_actual``
                        # filled with the per-dp_rank padding
                        # value (``num_accepted_tokens`` →
                        # ``1``,
                        # [model_runner_v1.py:1501](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1501);
                        # ``num_decode_draft_tokens_cpu`` → ``-1``,
                        # [model_runner_v1.py:1695](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1695)).
                        # Skipping non-populating bundles would
                        # under-count ``merged_num_reqs`` and
                        # shift later dp_ranks into the padding
                        # tail — explicitly forbidden by the
                        # parallel-FULL cudagraph pinned tail.
                        # Device convention per key matches the
                        # single-dp_rank call site at
                        # [model_runner_v1.py:4445-4446](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L4445-L4446):
                        # ``num_accepted_tokens`` on the leader
                        # runner's NPU, ``num_decode_draft_tokens_cpu``
                        # on CPU (matches the GDN builder's
                        # ``num_decode_draft_tokens_cpu`` argument
                        # name at
                        # [gdn_attn.py:161](vllm/vllm/v1/attention/backends/gdn_attn.py#L161)).
                        if k_key == "num_decode_draft_tokens_cpu":
                            pad_value = -1
                            tensor_device = "cpu"
                        else:
                            pad_value = 1
                            tensor_device = self.device
                        tensors_with_bundles = []
                        for b, (_, v) in zip(
                                bundles, per_bundle_extra_args):
                            t = v.get(k_key)
                            if t is None:
                                # Fill the bundle's per-req range
                                # with the per-dp_rank padding
                                # value so this bundle still
                                # contributes ``b.num_reqs_actual``
                                # rows to the merged tensor at the
                                # correct position (NOT appended
                                # to the tail — otherwise later
                                # dp_ranks would shift into the
                                # padding tail and break per-req
                                # alignment with the rest of the
                                # merged batch).
                                t = torch.full(
                                    (b.num_reqs_actual,),
                                    pad_value,
                                    dtype=torch.int32,
                                    device=tensor_device)
                            else:
                                t = t[:b.num_reqs_actual]
                            tensors_with_bundles.append((b, t))
                        t_cat = torch.cat(
                            [t for _, t in tensors_with_bundles],
                            dim=0)
                        if (t_cat.shape[0]
                                < merged_num_reqs_padded):
                            pad = torch.full(
                                (merged_num_reqs_padded
                                 - t_cat.shape[0],),
                                pad_value,
                                dtype=t_cat.dtype,
                                device=t_cat.device)
                            t_cat = torch.cat([t_cat, pad])
                        elif t_cat.shape[0] > merged_num_reqs_padded:
                            t_cat = t_cat[:merged_num_reqs_padded]
                        # Decode-first reorder: per-req extra
                        # args must follow the merged batch
                        # layout. In DecodeOnly mode the perm is
                        # the identity map so the slice is
                        # skipped.
                        if need_reorder:
                            t_cat = t_cat[merged_perm]
                        merged_extra_args[k_key] = t_cat.contiguous()
                    elif k_key == "num_reqs_actual":
                        # DSA scalar: sum across dp_ranks.
                        merged_extra_args[k_key] = sum(
                            v.get(k_key, 0) for _, v in
                            per_bundle_extra_args)
                    elif k_key in ("compress_ratio", "block_size"):
                        # DSA scalar: take-first (uniform).
                        for _, v in per_bundle_extra_args:
                            if k_key in v:
                                merged_extra_args.setdefault(
                                    k_key, v[k_key])
                                break
                    elif k_key.endswith("_ratio_to_sas_metadata"):
                        # DSA per-builder state: not supported in
                        # the batched path.
                        raise NotImplementedError(
                            f"Layer {attn_group.layer_names[0]}: "
                            f"DSA {k_key!r} is set on dp_rank "
                            f"bundle; the batched forward "
                            f"does not yet support merging "
                            f"per-builder ratio metadata.")
                    else:
                        # Unknown extra arg — pass through the
                        # first non-None value (defensive).
                        for _, v in per_bundle_extra_args:
                            if k_key in v:
                                merged_extra_args.setdefault(
                                    k_key, v[k_key])
                                break

                logger.info(
                    "[PD] _get_or_build_merged_attn_ctx: "
                    "Step 5 builder.build kv_gid=%d attn_gid=%d",
                    kv_cache_gid, attn_gid)
                attn_metadata_i = builder.build(
                    common_prefix_len=merged_cascade_attn_prefix_len,
                    common_attn_metadata=cm_merged,
                    **merged_extra_args)
                for ln in attn_group.layer_names:
                    merged_attn_metadata[ln] = attn_metadata_i

        # ---- Step 6: ``merged_input_ids`` /
        # ``merged_positions`` / ``merged_inputs_embeds`` for
        # downstream ``_model_forward`` consumption.
        n_actuals = [
            b.common_attn_metadata.num_actual_tokens for b in bundles]
        if all(b.input_ids is not None for b in bundles):
            merged_input_ids_ctx = torch.cat(
                [b.input_ids[:n]
                 for b, n in zip(bundles, n_actuals)])
        else:
            merged_input_ids_ctx = None
        if all(b.inputs_embeds is not None for b in bundles):
            merged_inputs_embeds_ctx = torch.cat(
                [b.inputs_embeds[:n]
                 for b, n in zip(bundles, n_actuals)])
        else:
            merged_inputs_embeds_ctx = None
        # ``merged_positions`` is built from per-dp_rank
        # ``bundle.positions`` (the same tensor
        # ``_preprocess`` returned; matches what
        # ``_model_forward(positions=...)`` reads). Do NOT
        # source from ``cm.positions`` (AscendCommonAttentionMetadata
        # may carry a different positions tensor — e.g.
        # ``mrope_positions`` / ``xdrope_positions`` path) since
        # ``forward`` consumes the buffer we write here, not the
        # metadata field.
        if all(b.positions is not None for b in bundles):
            cat_dim = bundles[0].positions.dim() - 1
            merged_positions_ctx = torch.cat(
                [b.positions[..., :n]
                 for b, n in zip(bundles, n_actuals)],
                dim=cat_dim)
        else:
            merged_positions_ctx = None
        # Decode-first reorder: ``input_ids`` / ``inputs_embeds``
        # are per-token. Apply ``merged_token_perm`` over the
        # actual range so the per-token order matches the
        # reordered ``query_start_loc`` / ``positions`` /
        # ``slot_mapping``. In DecodeOnly mode the perm is the
        # identity map so the slice is skipped.
        if (merged_input_ids_ctx is not None
                and merged_num_actual_tokens > 0
                and need_reorder):
            merged_input_ids_ctx_actual = (
                merged_input_ids_ctx[:merged_num_actual_tokens][
                    merged_token_perm])
            merged_input_ids_ctx = torch.cat([
                merged_input_ids_ctx_actual,
                merged_input_ids_ctx[merged_num_actual_tokens:],
            ])
        if (merged_inputs_embeds_ctx is not None
                and merged_num_actual_tokens > 0
                and need_reorder):
            merged_inputs_embeds_ctx_actual = (
                merged_inputs_embeds_ctx[:merged_num_actual_tokens][
                    merged_token_perm])
            merged_inputs_embeds_ctx = torch.cat([
                merged_inputs_embeds_ctx_actual,
                merged_inputs_embeds_ctx[merged_num_actual_tokens:],
            ])
        if (merged_positions_ctx is not None
                and merged_num_actual_tokens > 0
                and need_reorder):
            merged_positions_ctx_actual = (
                merged_positions_ctx[..., :merged_num_actual_tokens][
                    ..., merged_token_perm])
            merged_positions_ctx = torch.cat([
                merged_positions_ctx_actual,
                merged_positions_ctx[..., merged_num_actual_tokens:],
            ], dim=-1)
        # FULL mode: copy ``input_ids`` / ``inputs_embeds`` to
        # leader runner's pre-allocated buffers so
        # ``_model_forward(input_ids=..., inputs_embeds=...)``
        # share stable device pointers across cudagraph
        # captures. ``positions`` was already copied in Step 4
        # (FULL mode path) — reuse ``self.positions[:n]`` here
        # instead of re-cating.
        if merged_cudagraph_mode == CUDAGraphMode.FULL:
            if merged_input_ids_ctx is not None:
                # Pad the tail (beyond ``merged_num_actual_tokens``)
                # with 0 so the pre-allocated buffer's padding
                # carries a deterministic value instead of
                # whatever stale bytes were left from a previous
                # round. The leading actual rows are then
                # overwritten with the merged cat output.
                self.input_ids.gpu[
                    merged_num_actual_tokens:merged_num_tokens_padded
                ].fill_(0)
                self.input_ids.gpu[:merged_num_actual_tokens].copy_(
                    merged_input_ids_ctx)
                merged_input_ids_ctx = (
                    self.input_ids.gpu[:merged_num_tokens_padded])
            if merged_inputs_embeds_ctx is not None:
                self.inputs_embeds.gpu[
                    merged_num_actual_tokens:merged_num_tokens_padded
                ].fill_(0)
                self.inputs_embeds.gpu[:merged_num_actual_tokens].copy_(
                    merged_inputs_embeds_ctx)
                merged_inputs_embeds_ctx = (
                    self.inputs_embeds.gpu[:merged_num_tokens_padded])
            if merged_positions_ctx is not None:
                # Match the buffer that ``_preprocess`` /
                # ``_build_attention_metadata`` uses to source
                # positions — plain / mrope / xdrope — so that
                # ``_model_forward(positions=...)`` reads back the
                # same buffer it would in the standard path.
                if self.uses_mrope:
                    pos_buf = self.mrope_positions.gpu[
                        :, :merged_num_tokens_padded]
                    pos_buf[:, merged_num_actual_tokens:
                            merged_num_tokens_padded].fill_(0)
                    pos_buf[:, :merged_num_actual_tokens].copy_(
                        merged_positions_ctx)
                elif self.uses_xdrope_dim > 0:
                    pos_buf = self.xdrope_positions.gpu[
                        :, :merged_num_tokens_padded]
                    pos_buf[:, merged_num_actual_tokens:
                            merged_num_tokens_padded].fill_(0)
                    pos_buf[:, :merged_num_actual_tokens].copy_(
                        merged_positions_ctx)
                else:
                    pos_buf = self.positions[
                        :merged_num_tokens_padded]
                    pos_buf[merged_num_actual_tokens:
                            merged_num_tokens_padded].fill_(0)
                    pos_buf[:merged_num_actual_tokens].copy_(
                        merged_positions_ctx)
                merged_positions_ctx = pos_buf

        ctx = _MergedAttnContext(
            merged_attn_metadata=merged_attn_metadata,
            merged_batch_descriptor=merged_batch_descriptor,
            merged_cudagraph_mode=merged_cudagraph_mode,
            num_tokens_padded_merged=merged_num_tokens_padded,
            merged_input_ids=merged_input_ids_ctx,
            merged_positions=merged_positions_ctx,
            merged_inputs_embeds=merged_inputs_embeds_ctx,
            merged_token_perm=merged_token_perm,
            inv_merged_token_perm=inv_merged_token_perm,
        )
        logger.info(
            "[PD] _get_or_build_merged_attn_ctx: "
            "DONE, caching ctx")
        self._merged_attn_ctx_cache = ctx
        return ctx

