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



class PersistentEngineDraftFn:
    """Persistent-engine draft backend for ``draft_propose_v2``.

    Each target-side session is mapped to a single long-lived in-engine
    ``Request``. Per RPC we:

    1. Append target's committed delta via ``request.append_output_token_ids``
       (or ``add_request`` for fresh sessions).
    2. Drive ``self._engine.step()`` exactly ``K`` times — each step processes
       all live sessions in one batched forward and samples one token per
       running request. After ``K`` steps each session has ``K`` newly-sampled
       drafts at positions ``[pre_len .. pre_len+K-1]``.
    3. Snapshot pre-step state and **fully** roll back: shrink
       ``_all_token_ids`` / ``_output_token_ids`` by the number of new
       samples; reset ``num_computed_tokens``; evict any prefix-cache entry
       that the K-step-induced ``update_block_hashes`` polluted; reset each
       single-type manager's ``num_cached_block[req_id]`` to the snapshotted
       value. The KV slots that held draft K/V are reused on the next call's
       prefill (overwritten in-place).

    The state-rollback's coverage of ``num_cached_block`` /
    ``cached_block_hash_to_block`` is what distinguishes this from the naive
    rollback path. The naive path corrupted those structures: blocks that
    became full during the K-step loop were cached with hashes computed over
    ``prompt + delta + draft_samples``; rollback truncated only the request's
    own ``block_hashes`` list, leaving the cache map and ``num_cached_block``
    out of sync. The downstream effects (further calls' ``cache_blocks`` no-
    opping because ``num_cached_block`` was inflated past valid_blocks; future
    requests' prefix lookups racing against polluted hashes) destabilized
    drafts enough for target acceptance to collapse to ~0%.

    This implementation skips the per-RPC ``add_request`` lifecycle (Python
    bookkeeping, scheduler enqueue, ``output_processor`` registration,
    ``_handle_stopped_request`` cleanup) — the savings vs.
    ``SessionGreedyDraftFn`` and ``LLM.generate``-per-call are real but
    bounded: ~1 ms saved per session per RPC at typical configs. Most of the
    gain comes from KV cache being persistent at the slot level, which both
    backends already get via prefix caching.
    """

    accepts_rpc_dict = True

    def __init__(
        self,
        model: str,
        *,
        num_speculative_tokens: int,
        max_seq_len: int | None = None,
        dtype: torch.dtype | str | None = None,
        tensor_parallel_size: int | None = None,
    ) -> None:
        # InprocClient (synchronous in-process EngineCore) is what gives us
        # direct access to scheduler internals for the rollback. The default
        # LLM(...) path uses SyncMPClient (engine in a subprocess), which we
        # cannot reach across processes.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

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

        _enforce_eager = (
            os.environ.get("VLLM_REMOTE_DRAFT_ENFORCE_EAGER", "0") != "0"
        )
        # Prefix caching keeps inactive sessions' KV alive in the block pool.
        _enable_prefix_caching = (
            os.environ.get("VLLM_REMOTE_DRAFT_ENABLE_PREFIX_CACHING", "1") != "0"
        )

        self._K = int(num_speculative_tokens)
        logger.info(
            "Loading persistent-engine draft model %s (max_seq_len=%s, "
            "dtype=%s, tp=%d, K=%d, prefix_caching=%s)",
            model, self.max_seq_len, resolved_dtype, tp,
            self._K, _enable_prefix_caching,
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
            disable_log_stats=False,
            # Async scheduling pipelines schedule + execute one step ahead of
            # output processing (``step_with_batch_queue`` in
            # vllm/v1/engine/core.py). For our K-step driver this means call
            # N's K calls to ``engine.step()`` only process K-1 sampled
            # outputs (the last batch is still in flight); call N+1's step 1
            # then pops the leftover batch and appends its draft sample to
            # ``request._output_token_ids`` — silently corrupting the
            # session's committed state and tanking target acceptance.
            # Disable async scheduling here so each ``step()`` is fully
            # synchronous: schedule → forward → sample → bookkeeping all
            # complete before returning.
            # Persistent draft engine: async_scheduling forced to False (K-step driver requires synchronous step()).
            async_scheduling=False,
        )

        self._engine = self._llm.llm_engine
        engine_core_client = getattr(self._engine, "engine_core", None)
        actual_engine_core = getattr(engine_core_client, "engine_core", None)
        scheduler = getattr(actual_engine_core, "scheduler", None)
        if scheduler is None:
            raise RuntimeError(
                "PersistentEngineDraftFn requires InprocClient (synchronous "
                "in-process engine). Set "
                "VLLM_ENABLE_V1_MULTIPROCESSING=0 before constructing the LLM."
            )
        self._scheduler = scheduler
        self._kv_cache_manager = scheduler.kv_cache_manager
        self._block_pool = self._kv_cache_manager.block_pool
        self._coordinator = self._kv_cache_manager.coordinator
        self._single_type_managers = list(
            self._coordinator.single_type_managers
        )
        self._block_size = int(
            actual_engine_core.vllm_config.cache_config.block_size
        )
        # Reach through ExecutorBase to the model_runner's input_batch.
        # We need to keep ``input_batch.token_ids_cpu`` and
        # ``num_tokens_no_spec`` in sync with target's incremental delta:
        # ``request.append_output_token_ids`` only mutates the Request, not
        # the input_batch — but the model's forward reads input_ids by
        # ``index_select`` on ``token_ids_cpu_tensor``
        # (gpu_model_runner.py:1862). Without this sync, target's delta
        # tokens never reach the model and the K-step decode runs against
        # the previous call's stale draft samples — drafts diverge from
        # target's ctx and target rejects ~100% of them.
        executor = actual_engine_core.model_executor
        driver_worker = getattr(executor, "driver_worker", None)
        if driver_worker is None:
            raise RuntimeError(
                "PersistentEngineDraftFn could not find executor.driver_worker; "
                "TP>1 / multi-process executors are not supported by this "
                "backend (in-batch token_ids sync requires direct access)."
            )
        worker = getattr(driver_worker, "worker", driver_worker)
        model_runner = getattr(worker, "model_runner", None)
        if model_runner is None or not hasattr(model_runner, "input_batch"):
            raise RuntimeError(
                "PersistentEngineDraftFn could not reach "
                "model_runner.input_batch via "
                "executor.driver_worker.worker.model_runner."
            )
        self._model_runner = model_runner
        self._input_batch = model_runner.input_batch

        # Greedy, no early stop. ``max_tokens=max_seq_len`` so the request
        # never finishes on its own — we manage its lifecycle.
        self._draft_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=self.max_seq_len,
            ignore_eos=True,
            detokenize=False,
            stop=[],
            stop_token_ids=[],
        )

        self._live_sessions: set[str] = set()
        # session_id (target's view) -> internal engine request id (after
        # input_processor.assign_request_id appends a random suffix).
        self._session_to_engine_id: dict[str, str] = {}
        self._max_sessions = int(
            os.environ.get("VLLM_REMOTE_DRAFT_MAX_SESSIONS", "4096")
        )
        self._call_count = 0
        self._debug_log_first_n = int(
            os.environ.get("VLLM_REMOTE_DRAFT_PERSISTENT_DEBUG_FIRST_N", "3")
        )
        self._debug_log_interval = int(
            os.environ.get("VLLM_REMOTE_DRAFT_PERSISTENT_LOG_INTERVAL", "100")
        )
        self._short_draft_count = 0
        self._oversize_delta_count = 0

    def session_count(self) -> int:
        return len(self._live_sessions)

    def __call__(self, req: dict[str, Any]) -> list[list[int]]:
        K, sessions, evict_ids = parse_draft_propose_v2_payload(req)
        if K != self._K:
            raise ValueError(
                f"PersistentEngineDraftFn K mismatch: payload={K} "
                f"engine={self._K}. Set "
                "VLLM_REMOTE_DRAFT_NUM_SPECULATIVE_TOKENS to match the target."
            )

        for sid in evict_ids:
            self._abort(sid)

        order: list[str] = []
        internal_ids: list[str] = []
        snapshots: list[dict[str, Any]] = []
        for entry in sessions:
            if entry.is_first or entry.id not in self._live_sessions:
                if entry.id in self._live_sessions:
                    self._abort(entry.id)
                if not entry.tokens:
                    raise ValueError(
                        f"draft_propose_v2 session {entry.id!r} marked "
                        "is_first but has empty tokens (need at least the "
                        "prompt)."
                    )
                internal_id = self._engine.add_request(
                    request_id=entry.id,
                    prompt={"prompt_token_ids": list(entry.tokens)},
                    params=self._draft_params,
                )
                if internal_id is None:
                    internal_id = entry.id
                self._live_sessions.add(entry.id)
                self._session_to_engine_id[entry.id] = internal_id
                request = self._scheduler.requests.get(internal_id)
                if request is None:
                    raise RuntimeError(
                        f"add_request did not register session {entry.id!r} "
                        f"(internal id {internal_id!r}) into "
                        "scheduler.requests."
                    )
            else:
                internal_id = self._session_to_engine_id.get(
                    entry.id, entry.id
                )
                request = self._scheduler.requests.get(internal_id)
                if request is None:
                    raise RuntimeError(
                        f"Session {entry.id!r} (internal {internal_id!r}) "
                        "expected alive but missing from scheduler. Target "
                        "should resend with is_first=True."
                    )
                if entry.tokens:
                    delta_n = len(entry.tokens)
                    if delta_n > K + 1:
                        self._oversize_delta_count += 1
                        if self._oversize_delta_count <= 8:
                            logger.warning(
                                "[persistent draft] sid=%s incremental "
                                "delta=%d exceeds K+1=%d; possible "
                                "target/draft session desync.",
                                entry.id, delta_n, K + 1,
                            )
                    pre_append_len = len(request._all_token_ids)
                    request.append_output_token_ids(list(entry.tokens))
                    # Sync the delta into input_batch so the next forward
                    # actually reads target's tokens (not the previous K-step
                    # loop's stale draft samples that are still in
                    # token_ids_cpu at these positions).
                    self._sync_delta_into_input_batch(
                        internal_id, pre_append_len, list(entry.tokens)
                    )

            order.append(entry.id)
            internal_ids.append(internal_id)
            snapshots.append(self._snapshot(request, internal_id))

        # Drive K engine steps. Each step processes all live sessions in one
        # batched forward; the first step covers prefill of any new tokens
        # (initial prompt or appended delta), subsequent steps are decode.
        for _ in range(K):
            self._engine.step()

        drafts: list[list[int]] = []
        diag_rows: list[tuple[str, int, int, list[int]]] = []
        for sid, internal_id, snapshot in zip(
            order, internal_ids, snapshots, strict=True
        ):
            request = self._scheduler.requests.get(internal_id)
            if request is None:
                logger.warning(
                    "Session %s vanished mid-step; returning zero drafts.",
                    sid,
                )
                drafts.append([0] * K)
                self._live_sessions.discard(sid)
                self._session_to_engine_id.pop(sid, None)
                diag_rows.append(
                    (sid, snapshot["all_len"], -1, [0] * K)
                )
                continue

            pre_len = snapshot["all_len"]
            cur_all = request._all_token_ids
            n = len(cur_all) - pre_len
            new_tokens = list(cur_all[pre_len:pre_len + K]) if n > 0 else []
            if len(new_tokens) < K:
                self._short_draft_count += 1
                pad = new_tokens[-1] if new_tokens else 0
                new_tokens = new_tokens + [pad] * (K - len(new_tokens))
            drafts.append(new_tokens)
            diag_rows.append(
                (sid, pre_len, len(cur_all), list(new_tokens))
            )
            self._rollback(request, internal_id, snapshot, n)

        if len(self._live_sessions) > self._max_sessions:
            in_batch = set(order)
            stale = [s for s in self._live_sessions if s not in in_batch]
            over = len(self._live_sessions) - self._max_sessions
            for s in stale[:over]:
                self._abort(s)
            logger.warning(
                "Persistent-engine session cache exceeded cap (%d); "
                "evicted %d stale entries.",
                self._max_sessions, min(over, len(stale)),
            )

        self._call_count += 1
        self._maybe_log_persistent_debug(diag_rows, evict_ids)
        return drafts

    def _snapshot(self, request, internal_id: str) -> dict[str, Any]:
        """Capture the state we'll restore after the K decode steps.

        ``num_cached_block`` is the per-request cached-block watermark in each
        single-type manager. ``cache_blocks`` is monotonic on this value, so
        if the K-step loop pushes it past the snapshot, the rollback must
        revert it (otherwise ``cache_blocks`` for committed-only blocks on
        future calls would no-op and prefix caching silently degrades).
        """
        return {
            "all_len": len(request._all_token_ids),
            "block_hashes_len": len(request.block_hashes),
            "num_cached_block": [
                stm.num_cached_block.get(internal_id, 0)
                for stm in self._single_type_managers
            ],
        }

    def _rollback(
        self,
        request,
        internal_id: str,
        snapshot: dict[str, Any],
        n: int,
    ) -> None:
        """Restore request + KV-cache-manager state to the pre-K-step snapshot.

        Mirrors the in-process spec decoder's ``seq_lens -=
        num_rejected_tokens_gpu`` plus the cache-management rollback that
        the in-engine spec path normally does via ``spec_token_ids`` (which
        we cannot use here since we feed drafts through the sampler, not the
        scheduler's spec slot).
        """
        if n <= 0:
            # Even with n==0 we may need to undo polluted cache entries if
            # the K-step loop somehow grew block_hashes without growing
            # _all_token_ids — defensive but cheap.
            self._restore_cache_manager_state(
                request, internal_id, snapshot
            )
            return

        del request._output_token_ids[-n:]
        del request._all_token_ids[-n:]
        request.num_computed_tokens = max(
            request.num_prompt_tokens,
            len(request._all_token_ids),
        )
        self._restore_cache_manager_state(request, internal_id, snapshot)

    def _restore_cache_manager_state(
        self,
        request,
        internal_id: str,
        snapshot: dict[str, Any],
    ) -> None:
        """Evict K-step-polluted cache entries and clamp ``num_cached_block``
        at the post-rollback valid block count.

        During the K decode steps, if positions cross a block boundary,
        ``request.update_block_hashes()`` computes a hash for that block over
        ``prompt + delta + draft_samples`` and ``allocate_slots`` calls
        ``cache_blocks`` which writes that hash → block mapping into the
        block pool's ``cached_block_hash_to_block``. Truncating
        ``request.block_hashes`` only undoes the request-side bookkeeping;
        the block-pool side keeps the polluted mapping.

        Important: a block-boundary crossing during the K-step loop is *not*
        the only way ``cache_blocks`` runs. The very first step's
        ``allocate_slots`` typically caches all of the request's already-full
        prompt blocks (the legitimate ones). Those cachings must be kept —
        the per-block ``block_hash`` is set after the first cache and the
        block pool asserts ``blk.block_hash is None`` on the next caching
        attempt. So the rollback must NOT pull ``num_cached_block`` back
        below the legitimate post-rollback count, or the next call's
        ``cache_blocks`` will walk over already-cached blocks and trip that
        assertion (we hit this in production: persistent draft #1 ran fine,
        the next call crashed inside ``cache_full_blocks``).
        """
        valid_blocks = len(request._all_token_ids) // self._block_size
        prev_hashes_len = snapshot["block_hashes_len"]
        if len(request.block_hashes) > prev_hashes_len:
            # Block hashes added during the K-step loop. Any of these whose
            # corresponding block has ``block_hash`` set was cached by
            # ``cache_blocks`` and is now polluted (covers ≥ 1 draft-sample
            # position). Evict each from the pool's cache map and reset the
            # block's hash. ``_maybe_evict_cached_block`` is a no-op for
            # blocks that were never cached, so it's safe to call on all of
            # them.
            for stm in self._single_type_managers:
                req_blocks = stm.req_to_blocks.get(internal_id, [])
                for i in range(prev_hashes_len, len(request.block_hashes)):
                    if i < len(req_blocks):
                        blk = req_blocks[i]
                        if blk.block_hash is not None:
                            self._block_pool._maybe_evict_cached_block(blk)
            del request.block_hashes[prev_hashes_len:]
        elif valid_blocks < len(request.block_hashes):
            # Defensive: rollback shrank past the block-hashes list. Trim
            # tail. Should not happen with the K-step semantics above but
            # keeps the structures consistent.
            del request.block_hashes[valid_blocks:]

        # Clamp ``num_cached_block`` at ``valid_blocks`` so future
        # ``cache_blocks`` calls don't try to re-cache the polluted blocks
        # we just evicted. Do NOT pull below the current value otherwise:
        # legitimate cachings of already-committed prompt blocks (done in
        # the K-step loop's first step) must remain reflected in the
        # watermark.
        for stm in self._single_type_managers:
            cur = stm.num_cached_block.get(internal_id)
            if cur is None:
                continue
            if cur > valid_blocks:
                stm.num_cached_block[internal_id] = valid_blocks

        # Roll back ``input_batch.num_tokens_no_spec`` too: ``_bookkeeping_sync``
        # advanced it by 1 per draft sample, so post-K-step it sits at
        # ``pre_len + K``. Leaving it there means the next call's
        # ``_sync_delta_into_input_batch`` would write the delta past the end
        # marker, and the model's index_select for the scheduled
        # ``[pre_len .. pre_len + delta]`` range would read from the inflated
        # span — i.e., still see stale draft samples.
        req_idx = self._input_batch.req_id_to_index.get(internal_id)
        if req_idx is not None:
            target_n = len(request._all_token_ids)
            if int(self._input_batch.num_tokens_no_spec[req_idx]) > target_n:
                self._input_batch.num_tokens_no_spec[req_idx] = target_n

    def _maybe_log_persistent_debug(
        self,
        diag_rows: list[tuple[str, int, int, list[int]]],
        evict_ids: list[str],
    ) -> None:
        verbose = self._call_count <= self._debug_log_first_n
        summary = (
            self._debug_log_interval > 0
            and self._call_count % self._debug_log_interval == 0
        )
        if not (verbose or summary):
            return
        if verbose:
            for sid, pre_len, post_len, drafts in diag_rows:
                logger.info(
                    "[persistent draft #%d] sid=%s pre_len=%d post_len=%d "
                    "drafts=%s",
                    self._call_count, sid, pre_len, post_len, drafts,
                )
            if evict_ids:
                logger.info(
                    "[persistent draft #%d] evicted=%s",
                    self._call_count, evict_ids,
                )
        if summary:
            logger.info(
                "[persistent draft] calls=%d live_sessions=%d "
                "short_draft_count=%d oversize_delta_count=%d",
                self._call_count, len(self._live_sessions),
                self._short_draft_count, self._oversize_delta_count,
            )

    def _abort(self, sid: str) -> None:
        internal_id = self._session_to_engine_id.get(sid, sid)
        try:
            self._engine.abort_request([internal_id])
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "abort_request(%s -> %s) failed: %s", sid, internal_id, e
            )
        finally:
            self._live_sessions.discard(sid)
            self._session_to_engine_id.pop(sid, None)

    def _sync_delta_into_input_batch(
        self,
        internal_id: str,
        pre_append_len: int,
        delta: list[int],
    ) -> None:
        """Mirror target's just-appended delta into input_batch.

        ``request.append_output_token_ids`` updates only the Request's
        ``_all_token_ids`` / ``_output_token_ids``. The model forward,
        however, reads input ids via ``torch.index_select`` on
        ``input_batch.token_ids_cpu_tensor`` keyed by the scheduled
        positions (gpu_model_runner.py:1862). If we don't mirror the delta
        into ``token_ids_cpu`` AND advance ``num_tokens_no_spec``, step 1's
        prefill of the unprocessed range reads whatever is in those slots
        — typically the previous K-step loop's draft samples that
        ``_bookkeeping_sync`` wrote there at the end of the prior call. The
        draft model then conditions on stale tokens, drafts diverge from
        target's actual context, and target rejects ~100% of them.

        For ``is_first`` sessions we don't need this: the request is still
        in the scheduler's waiting queue, so it isn't in
        ``input_batch.req_id_to_index`` yet, and step 1's schedule path
        (waiting → running) populates ``token_ids_cpu`` from the prompt
        directly. Only incremental sessions hit this helper.
        """
        req_idx = self._input_batch.req_id_to_index.get(internal_id)
        if req_idx is None:
            # Defensive: session is incremental (already running) but not
            # in the input_batch index. Skip silently — the next step will
            # rebuild from prompt if needed.
            return
        delta_n = len(delta)
        if delta_n == 0:
            return
        end = pre_append_len + delta_n
        self._input_batch.token_ids_cpu[req_idx, pre_append_len:end] = delta
        self._input_batch.is_token_ids[req_idx, pre_append_len:end] = True
        self._input_batch.num_tokens_no_spec[req_idx] = end
