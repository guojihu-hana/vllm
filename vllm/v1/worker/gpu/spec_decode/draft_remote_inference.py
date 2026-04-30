# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Remote draft proposal backends for dedicated-GPU speculative decoding."""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def remote_draft_tensor_parallel_size(explicit: int | None = None) -> int:
    """Tensor parallel size for remote draft ``LLM`` / native parity worker."""
    tp = 1 if explicit is None else explicit
    if tp < 1:
        raise ValueError(f"tensor_parallel_size must be >= 1, got {tp}")
    return tp


def _parse_dtype(name: str) -> torch.dtype | str:
    n = name.lower().strip()
    if n in ("auto", ""):
        return "auto"
    if n in ("float16", "fp16"):
        return torch.float16
    if n in ("bfloat16", "bf16"):
        return torch.bfloat16
    if n in ("float32", "fp32"):
        return torch.float32
    return name


class VLLMGreedyDraftFn:
    """Draft proposals from a dedicated local vLLM engine.

    This backend uses vLLM's own decoding path for remote draft generation, which is
    generally closer to in-process draft behavior than raw HF replay.
    """

    def __init__(
        self,
        model: str,
        max_seq_len: int | None = None,
        dtype: torch.dtype | str | None = None,
        tensor_parallel_size: int | None = None,
    ) -> None:
        from vllm import LLM, SamplingParams

        self.max_seq_len = max_seq_len or int(
            os.environ.get("VLLM_REMOTE_DRAFT_MAX_SEQ_LEN", "8192")
        )
        resolved_dtype: torch.dtype | str | None = dtype
        if resolved_dtype is None:
            resolved_dtype = _parse_dtype(
                os.environ.get("VLLM_REMOTE_DRAFT_DTYPE", "auto")
            )
        elif isinstance(resolved_dtype, str):
            resolved_dtype = _parse_dtype(resolved_dtype)
        tp = remote_draft_tensor_parallel_size(tensor_parallel_size)
        logger.info(
            "Loading remote draft vLLM engine %s (max_seq_len=%s, dtype=%s, tp=%d)",
            model,
            self.max_seq_len,
            resolved_dtype,
            tp,
        )
        # Match draft_remote_native.build_vllm_config_for_native_remote_draft:
        # VLLM_REMOTE_DRAFT_ENFORCE_EAGER=0 disables eager (allows compile/CUDAGraph).
        _enforce_eager = os.environ.get("VLLM_REMOTE_DRAFT_ENFORCE_EAGER", "1") != "0"
        self._llm = LLM(
            model=model,
            tokenizer=model,
            trust_remote_code=True,
            max_model_len=self.max_seq_len,
            dtype=resolved_dtype,
            tensor_parallel_size=tp,
            gpu_memory_utilization=float(
                os.environ.get("VLLM_REMOTE_DRAFT_GPU_MEMORY_UTILIZATION", "0.85")
            ),
            enforce_eager=_enforce_eager,
        )
        self._sampling_params_cls = SamplingParams
        self._warned_hidden_ignored = False

    def __call__(
        self,
        next_token_ids: list[int],
        num_speculative_tokens: int,
        context_token_ids: list[list[int]] | None = None,
        target_hidden_states: torch.Tensor | None = None,
    ) -> list[list[int]]:
        if target_hidden_states is not None and not self._warned_hidden_ignored:
            logger.warning(
                "VLLMGreedyDraftFn currently ignores target_hidden_states."
            )
            self._warned_hidden_ignored = True
        if context_token_ids is None:
            raise ValueError(
                "VLLMGreedyDraftFn requires context_token_ids in the RPC payload."
            )
        if len(context_token_ids) != len(next_token_ids):
            raise ValueError(
                "context_token_ids length must match next_token_ids "
                f"({len(context_token_ids)} vs {len(next_token_ids)})."
            )
        prompts: list[dict[str, list[int]]] = []
        for seq, nt in zip(context_token_ids, next_token_ids, strict=True):
            seq_i = [int(t) for t in seq]
            nt_i = int(nt)
            if not seq_i or seq_i[-1] != nt_i:
                seq_i = seq_i + [nt_i]
            if len(seq_i) > self.max_seq_len:
                seq_i = seq_i[-self.max_seq_len :]
            prompts.append({"prompt_token_ids": seq_i})

        sp = self._sampling_params_cls(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=int(num_speculative_tokens),
            detokenize=False,
            skip_special_tokens=False,
        )
        outputs = self._llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"Unexpected number of draft outputs: got {len(outputs)} "
                f"for {len(prompts)} prompts."
            )
        draft_token_ids: list[list[int]] = []
        for out in outputs:
            draft = list(out.outputs[0].token_ids)
            if len(draft) != num_speculative_tokens:
                if len(draft) < num_speculative_tokens:
                    draft = draft + [draft[-1] if draft else 0] * (
                        num_speculative_tokens - len(draft)
                    )
                else:
                    draft = draft[:num_speculative_tokens]
            draft_token_ids.append(draft)
        return draft_token_ids
