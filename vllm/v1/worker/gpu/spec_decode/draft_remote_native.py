# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Remote draft worker for draft_propose_v1.

By default uses the same greedy full-context replay as :class:`VLLMGreedyDraftFn`,
because a stateless RPC worker cannot preserve draft KV across target steps while
still feeding EAGLE metadata that assumes a full prefix in cache.

Set ``VLLM_REMOTE_DRAFT_USE_EAGLE_PARITY=1`` to force in-process
:class:`DraftModelProposer` (experimental; expects coherent KV or full replay).
"""

from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest import mock

import torch
import torch.distributed as dist

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import init_distributed_environment
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory, initialize_model_parallel
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups, get_kv_cache_groups
from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec, init_attn_backend, init_kv_cache
from vllm.v1.worker.gpu.spec_decode.draft_remote_inference import (
    VLLMGreedyDraftFn,
    _parse_dtype,
    remote_draft_tensor_parallel_size,
)
from vllm.v1.worker.gpu.spec_decode.draft_rpc_payload import (
    DRAFT_PROPOSE_V1,
    deserialize_draft_propose_v1,
    tensor_chunk_from_payload,
)
from vllm.v1.worker.utils import prepare_kernel_block_sizes
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.v1.spec_decode.draft_model import DraftModelProposer

logger = init_logger(__name__)


def build_vllm_config_for_native_remote_draft(
    target_model: str,
    draft_model: str,
    *,
    num_speculative_tokens: int,
    max_model_len: int,
    dtype: str = "auto",
    tensor_parallel_size: int | None = None,
) -> VllmConfig:
    """Construct VllmConfig matching speculative draft_model engine layout."""
    tp = remote_draft_tensor_parallel_size(tensor_parallel_size)
    parallel_config = ParallelConfig(tensor_parallel_size=tp)
    # Remote draft runs a new graph per RPC; torch.compile + piecewise CUDA graphs
    # can mis-match dynamic shapes and cause illegal memory access. Default eager.
    # Set VLLM_REMOTE_DRAFT_ENFORCE_EAGER=0 to try compilation (expert / perf only).
    enforce_eager = os.environ.get("VLLM_REMOTE_DRAFT_ENFORCE_EAGER", "1") != "0"
    model_config = ModelConfig(
        model=target_model,
        runner="generate",
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype=dtype,
        enforce_eager=enforce_eager,
    )
    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=parallel_config,
        model=draft_model,
        method="draft_model",
        num_speculative_tokens=num_speculative_tokens,
        draft_tensor_parallel_size=tp,
    )
    device = DeviceConfig(device=current_platform.device_type)
    scheduler_config = SchedulerConfig(
        max_num_seqs=int(os.environ.get("VLLM_REMOTE_DRAFT_MAX_NUM_SEQS", "256")),
        max_num_batched_tokens=int(
            os.environ.get("VLLM_REMOTE_DRAFT_MAX_NUM_BATCHED_TOKENS", "8192")
        ),
        max_model_len=max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=int(os.environ.get("VLLM_REMOTE_DRAFT_CACHE_BLOCK_SIZE", "16")),
        gpu_memory_utilization=float(
            os.environ.get("VLLM_REMOTE_DRAFT_GPU_MEMORY_UTILIZATION", "0.6")
        ),
        cache_dtype="auto",
    )
    backend = os.environ.get("VLLM_REMOTE_DRAFT_ATTENTION_BACKEND") or None
    attention_config = AttentionConfig(backend=backend) if backend else AttentionConfig()

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        speculative_config=speculative_config,
        device_config=device,
        parallel_config=parallel_config,
        load_config=LoadConfig(),
        scheduler_config=scheduler_config,
        attention_config=attention_config,
    )


class DraftModelNativeParityFn:
    """draft_propose_v1 handler: greedy replay by default, optional EAGLE parity."""

    accepts_rpc_dict = True

    def __init__(
        self,
        target_model: str,
        draft_model: str,
        *,
        num_speculative_tokens: int,
        max_model_len: int | None = None,
        dtype: str = "auto",
        tensor_parallel_size: int | None = None,
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.num_speculative_tokens = num_speculative_tokens
        self.max_model_len = max_model_len or int(
            os.environ.get("VLLM_REMOTE_DRAFT_MAX_SEQ_LEN", "8192")
        )
        self._tensor_parallel_size = remote_draft_tensor_parallel_size(
            tensor_parallel_size
        )
        self._use_eagle_parity = (
            os.environ.get("VLLM_REMOTE_DRAFT_USE_EAGLE_PARITY", "0") == "1"
        )
        if self._use_eagle_parity and self._tensor_parallel_size > 1:
            raise ValueError(
                "Native EAGLE parity (VLLM_REMOTE_DRAFT_USE_EAGLE_PARITY=1) runs in a "
                "single process and only supports tensor_parallel_size=1. For "
                "multi-GPU draft TP, leave EAGLE parity off (default) or use "
                "--backend vllm (greedy LLM replay)."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("DraftModelNativeParityFn requires CUDA.")

        self.device = torch.device("cuda:0")
        init_workspace_manager(self.device)

        self.proposer: DraftModelProposer | None = None
        self._greedy_fn: VLLMGreedyDraftFn | None = None
        self.vllm_config: VllmConfig | None = None

        if not self._use_eagle_parity:
            resolved = _parse_dtype(dtype) if isinstance(dtype, str) else dtype
            self._greedy_fn = VLLMGreedyDraftFn(
                draft_model,
                max_seq_len=self.max_model_len,
                dtype=resolved,
                tensor_parallel_size=self._tensor_parallel_size,
            )
            logger.info(
                "DraftModelNativeParityFn greedy-replay backend draft_model=%s "
                "tp=%d (set VLLM_REMOTE_DRAFT_USE_EAGLE_PARITY=1 for DraftModelProposer)",
                draft_model,
                self._tensor_parallel_size,
            )
            return

        self.vllm_config = build_vllm_config_for_native_remote_draft(
            target_model,
            draft_model,
            num_speculative_tokens=num_speculative_tokens,
            max_model_len=self.max_model_len,
            dtype=dtype,
            tensor_parallel_size=self._tensor_parallel_size,
        )

        # initialize_model_parallel() reads get_current_vllm_config(); set config first.
        with set_current_vllm_config(self.vllm_config):
            self._maybe_init_distributed()
            self._setup_proposer_and_kv()

        logger.info(
            "DraftModelNativeParityFn EAGLE parity draft_model=%s target_model=%s",
            draft_model,
            target_model,
        )

    def _maybe_init_distributed(self) -> None:
        if dist.is_initialized():
            return
        tmp = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{tmp}",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)

    def _setup_proposer_and_kv(self) -> None:
        proposer = DraftModelProposer(
            vllm_config=self.vllm_config,
            device=self.device,
            runner=None,
        )
        proposer.load_model(torch.nn.Identity())

        merged_specs = get_kv_cache_spec(self.vllm_config)
        if not merged_specs:
            raise RuntimeError(
                "No KV cache specs found after loading draft model; "
                "cannot initialize native remote draft."
            )
        global_groups = get_kv_cache_groups(self.vllm_config, merged_specs)
        avail = int(
            float(os.environ.get("VLLM_REMOTE_DRAFT_KV_MEMORY_BYTES", str(6 * GiB_bytes)))
        )
        kv_cache_config = get_kv_cache_config_from_groups(
            self.vllm_config, global_groups, avail
        )
        self._kv_num_blocks = kv_cache_config.num_blocks

        attn_backends, attn_groups_nested = init_attn_backend(
            kv_cache_config,
            self.vllm_config,
            self.device,
            active_layer_names=None,
        )
        kernel_block_sizes = prepare_kernel_block_sizes(
            kv_cache_config, attn_groups_nested
        )

        proposer.initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        proposer.initialize_cudagraph_keys(CUDAGraphMode.NONE)

        self._kv_caches = init_kv_cache(
            runner_kv_caches=[],
            forward_context=self.vllm_config.compilation_config.static_forward_context,
            kv_cache_config=kv_cache_config,
            attn_backends=attn_backends,
            device=self.device,
            cache_dtype=str(self.vllm_config.cache_config.cache_dtype),
        )

        self.proposer = proposer

    def _zero_kv_caches_before_rpc(self) -> None:
        """Reset KV so each RPC does not reuse stale slots from prior mappings.

        We rebuild block_table/slot_mapping per request; without clearing KV,
        later RPCs read/write inconsistent physical pages and can hit index
        errors or CUBLAS failures. Target-side draft reuses one continuous
        block_table; remote parity without shipping block ids requires a fresh
        KV view each call unless you disable this (expert only).
        """
        if os.environ.get("VLLM_REMOTE_DRAFT_ZERO_KV_EACH_RPC", "1") == "0":
            return
        for t in self._kv_caches.values():
            t.zero_()

    def __call__(self, req: dict[str, Any]) -> list[list[int]]:
        if req.get("rpc_schema") != DRAFT_PROPOSE_V1:
            raise ValueError(
                f"DraftModelNativeParityFn expects rpc_schema={DRAFT_PROPOSE_V1!r}, "
                f"got {req.get('rpc_schema')!r}"
            )

        if not self._use_eagle_parity:
            assert self._greedy_fn is not None
            ctx = req.get("context_token_ids")
            if ctx is None:
                raise ValueError(
                    "draft_propose_v1 requires context_token_ids for greedy replay "
                    "(upgrade target vLLM worker)."
                )
            next_t = tensor_chunk_from_payload(
                req["next_token_ids"], torch.device("cpu")
            ).to(torch.int32)
            next_list = [int(x) for x in next_t.view(-1).tolist()]
            k = int(req["num_speculative_tokens"])
            return self._greedy_fn(
                next_list,
                k,
                context_token_ids=[[int(x) for x in row] for row in ctx],
            )

        assert self.proposer is not None and self.vllm_config is not None
        block_size = self.proposer.block_size
        self._zero_kv_caches_before_rpc()

        deser = deserialize_draft_propose_v1(
            req,
            self.device,
            block_size=block_size,
            num_kv_blocks=self._kv_num_blocks,
            omit_target_hs_fill_hidden_size=self.proposer.hidden_size,
            omit_target_hs_dtype=self.proposer.dtype,
        )

        target_hs = deser.target_hidden_states.to(
            device=self.device, dtype=self.proposer.dtype
        )

        sampling_metadata = mock.MagicMock()

        with set_current_vllm_config(self.vllm_config):
            out = self.proposer.propose(
                target_token_ids=deser.target_token_ids,
                target_positions=deser.target_positions,
                target_hidden_states=target_hs,
                next_token_ids=deser.next_token_ids,
                token_indices_to_sample=deser.token_indices_to_sample,
                common_attn_metadata=deser.common_attn_metadata,
                sampling_metadata=sampling_metadata,
                mm_embed_inputs=None,
                num_rejected_tokens_gpu=deser.num_rejected_tokens_gpu,
                slot_mappings=None,
            )

        return out.detach().cpu().long().tolist()


def cleanup_native_draft_dist() -> None:
    """Best-effort cleanup for tests."""
    try:
        cleanup_dist_env_and_memory()
    except Exception:
        pass
