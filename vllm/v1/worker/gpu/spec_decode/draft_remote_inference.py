# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Remote draft proposal backends for dedicated-GPU speculative decoding."""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


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


def _load_model(model: str, device: torch.device, dtype: torch.dtype | str | None):
    from transformers import AutoModelForCausalLM

    kwargs: dict = {"trust_remote_code": True}
    if dtype is not None and dtype != "auto":
        kwargs["torch_dtype"] = dtype
    m = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    if m.config.pad_token_id is None and m.config.eos_token_id is not None:
        m.config.pad_token_id = m.config.eos_token_id
    m.eval()
    m.to(device)
    return m


class HFTransformersDraftFn:
    """Callable used by :class:`DraftRemoteServer` for GPU draft proposals."""

    def __init__(
        self,
        model: str,
        device: torch.device,
        max_seq_len: int | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        self.device = device
        self.max_seq_len = max_seq_len or int(
            os.environ.get("VLLM_REMOTE_DRAFT_MAX_SEQ_LEN", "8192")
        )
        logger.info(
            "Loading remote draft HF model %s on %s (max_seq_len=%s)",
            model,
            device,
            self.max_seq_len,
        )
        resolved_dtype: torch.dtype | str | None = dtype
        if resolved_dtype is None:
            resolved_dtype = _parse_dtype(
                os.environ.get("VLLM_REMOTE_DRAFT_DTYPE", "auto")
            )
        elif isinstance(resolved_dtype, str):
            resolved_dtype = _parse_dtype(resolved_dtype)
        if resolved_dtype == "auto" and device.type == "cuda":
            resolved_dtype = torch.float16
        self._model = _load_model(model, device, resolved_dtype)

    @torch.inference_mode()
    def _greedy_extend(self, input_ids: list[int], k: int) -> list[int]:
        if k <= 0:
            return []
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len :]

        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        outputs = self._model(ids, use_cache=True)
        past = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        first = int(logits.argmax(dim=-1).item())
        draft: list[int] = [first]

        cur = torch.tensor([[first]], dtype=torch.long, device=self.device)
        for _ in range(1, k):
            outputs = self._model(cur, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            nxt = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
            draft.append(nxt)
            cur = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
        return draft

    def __call__(
        self,
        next_token_ids: list[int],
        num_speculative_tokens: int,
        context_token_ids: list[list[int]] | None = None,
        target_hidden_states: torch.Tensor | None = None,
    ) -> list[list[int]]:
        del target_hidden_states
        if context_token_ids is None:
            raise ValueError(
                "HFTransformersDraftFn requires context_token_ids in the RPC payload."
            )
        if len(context_token_ids) != len(next_token_ids):
            raise ValueError(
                "context_token_ids length must match next_token_ids "
                f"({len(context_token_ids)} vs {len(next_token_ids)})."
            )
        out: list[list[int]] = []
        for seq, nt in zip(context_token_ids, next_token_ids, strict=True):
            seq_i = [int(t) for t in seq]
            nt_i = int(nt)
            if seq_i and seq_i[-1] == nt_i:
                full = seq_i
            else:
                full = seq_i + [nt_i]
            out.append(self._greedy_extend(full, num_speculative_tokens))
        return out


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
        logger.info(
            "Loading remote draft vLLM engine %s (max_seq_len=%s, dtype=%s)",
            model,
            self.max_seq_len,
            resolved_dtype,
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
            tensor_parallel_size=1,
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
