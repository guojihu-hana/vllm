# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Remote draft proposal backends for dedicated-GPU speculative decoding."""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.draft_rpc_payload import (
    DRAFT_PROPOSE_V2,
    parse_draft_propose_v2_payload,
)

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
            os.environ.get("MAX_SEQ_LEN", "8192")
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
        _enforce_eager = os.environ.get("VLLM_REMOTE_DRAFT_ENFORCE_EAGER", "0") != "0"
        # Prefix caching is what makes the v2 session protocol cheap: each
        # follow-up step's prompt = previous prompt + a few accepted tokens,
        # so the unchanged prefix's KV is reused. Default on; can be disabled
        # for parity testing against the v1 stateless path.
        _enable_prefix_caching = (
            os.environ.get("VLLM_REMOTE_DRAFT_ENABLE_PREFIX_CACHING", "1") != "0"
        )
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
            enable_prefix_caching=_enable_prefix_caching,
            # vllm.entrypoints.llm.LLM defaults this to True, which disables
            # get_metrics() (it raises "Stat logging disabled"). Force-enable
            # so we can read vllm:prefix_cache_{queries,hits} counters below.
            disable_log_stats=False,
        )
        self._sampling_params_cls = SamplingParams
        self._warned_hidden_ignored = False

        # Prefix-cache hit-rate logging. Only meaningful when prefix caching
        # is enabled on the underlying LLM; otherwise the counters never
        # appear in get_metrics() and we silently skip.
        self._prefix_cache_log_interval = int(
            os.environ.get("VLLM_REMOTE_DRAFT_PREFIX_CACHE_LOG_INTERVAL", "50")
        )
        self._prefix_cache_enabled = _enable_prefix_caching
        self._call_count = 0
        self._last_pc_queries = 0
        self._last_pc_hits = 0
        self._pc_metrics_warned = False

    def _read_prefix_cache_counters(self) -> tuple[int, int] | None:
        """Return cumulative ``(queries, hits)`` from the LLM's Prometheus counters.

        Sums across engines / label sets so multi-engine setups still
        produce one number. Returns ``None`` when the counters are absent
        (prefix caching off or metrics not yet emitted).
        """
        try:
            metrics = self._llm.get_metrics()
        except Exception as e:  # noqa: BLE001
            if not self._pc_metrics_warned:
                logger.warning(
                    "Could not read prefix-cache metrics from LLM: %s", e
                )
                self._pc_metrics_warned = True
            return None
        queries = 0
        hits = 0
        seen = False
        for m in metrics:
            if m.name == "vllm:prefix_cache_queries":
                queries += int(getattr(m, "value", 0))
                seen = True
            elif m.name == "vllm:prefix_cache_hits":
                hits += int(getattr(m, "value", 0))
                seen = True
        if not seen:
            return None
        return queries, hits

    def _maybe_log_prefix_cache(self) -> None:
        if not self._prefix_cache_enabled:
            return
        if self._prefix_cache_log_interval <= 0:
            return
        if self._call_count % self._prefix_cache_log_interval != 0:
            return
        snapshot = self._read_prefix_cache_counters()
        if snapshot is None:
            if not self._pc_metrics_warned:
                logger.warning(
                    "Prefix cache enabled but no vllm:prefix_cache_* counters "
                    "found yet (call_count=%d).",
                    self._call_count,
                )
                self._pc_metrics_warned = True
            return
        cum_queries, cum_hits = snapshot
        delta_queries = cum_queries - self._last_pc_queries
        delta_hits = cum_hits - self._last_pc_hits
        cum_rate = cum_hits / cum_queries if cum_queries else 0.0
        win_rate = delta_hits / delta_queries if delta_queries else 0.0
        logger.info(
            "[remote draft] prefix cache (over %d generate calls): "
            "window hits=%d queries=%d rate=%.3f | "
            "cumulative hits=%d queries=%d rate=%.3f",
            self._prefix_cache_log_interval,
            delta_hits,
            delta_queries,
            win_rate,
            cum_hits,
            cum_queries,
            cum_rate,
        )
        self._last_pc_queries = cum_queries
        self._last_pc_hits = cum_hits

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
        self._call_count += 1
        self._maybe_log_prefix_cache()
        return draft_token_ids


class SessionGreedyDraftFn:
    """Session-aware greedy draft backend for ``draft_propose_v2``.

    Maintains per-session token history on the server. Each RPC carries only
    the delta accepted by the target since the last step, plus an explicit
    ``evict`` list for finished requests. Wraps ``VLLMGreedyDraftFn`` for the
    actual generation, relying on vLLM prefix caching to make repeated calls
    on the same session do incremental prefill (~O(K_accepted+1)) instead of
    re-prefilling the whole prompt every step.
    """

    accepts_rpc_dict = True

    def __init__(
        self,
        model: str,
        *,
        max_seq_len: int | None = None,
        dtype: torch.dtype | str | None = None,
        tensor_parallel_size: int | None = None,
    ) -> None:
        self._greedy = VLLMGreedyDraftFn(
            model,
            max_seq_len=max_seq_len,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
        )
        self._sessions: dict[str, list[int]] = {}
        # Soft cap so a buggy target does not OOM the server map.
        self._max_sessions = int(
            os.environ.get("VLLM_REMOTE_DRAFT_MAX_SESSIONS", "4096")
        )

    def __call__(self, req: dict[str, Any]) -> list[list[int]]:
        num_spec, sessions, evict_ids = parse_draft_propose_v2_payload(req)

        for sid in evict_ids:
            self._sessions.pop(sid, None)

        prompts: list[list[int]] = []
        order: list[str] = []
        for entry in sessions:
            if entry.is_first or entry.id not in self._sessions:
                if entry.is_first and entry.id in self._sessions:
                    logger.debug(
                        "Session %s reset by target (is_first=True)", entry.id
                    )
                self._sessions[entry.id] = list(entry.tokens)
            else:
                self._sessions[entry.id].extend(entry.tokens)

            history = self._sessions[entry.id]
            if not history:
                raise ValueError(
                    f"draft_propose_v2 session {entry.id!r} has empty token "
                    "history (first call must include at least one token)."
                )
            prompts.append(history)
            order.append(entry.id)

        if len(self._sessions) > self._max_sessions:
            live = set(order)
            stale = [sid for sid in self._sessions if sid not in live]
            for sid in stale[
                : len(self._sessions) - self._max_sessions
            ]:
                self._sessions.pop(sid, None)
            logger.warning(
                "draft session cache exceeded cap (%d); evicted %d stale entries.",
                self._max_sessions,
                len(stale),
            )

        if not prompts:
            return []

        # Reuse VLLMGreedyDraftFn's LLM.generate path. ``next_token_ids`` is
        # required by that signature but only used for legacy dedupe; pass
        # each session's last token (already the desired starting point).
        next_token_ids = [seq[-1] for seq in prompts]
        return self._greedy(
            next_token_ids,
            num_spec,
            context_token_ids=prompts,
        )

    def session_count(self) -> int:
        return len(self._sessions)
