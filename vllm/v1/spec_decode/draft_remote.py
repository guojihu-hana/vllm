# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import msgspec
import torch
import zmq

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.worker.gpu.spec_decode.draft_rpc_payload import (
    DraftSessionEntry,
    build_draft_propose_v1_payload,
    build_draft_propose_v2_payload,
)

logger = init_logger(__name__)


def _use_session_protocol() -> bool:
    """v2 (session-incremental) opt-in.

    Default off so existing deployments keep using v1 until both target and
    server are upgraded. Enable on both sides to drop per-step bandwidth from
    O(seq_len × batch) to O(K_accepted+1 × batch) and let prefix caching reuse
    KV across steps.
    """
    return os.environ.get("VLLM_REMOTE_DRAFT_USE_SESSION_PROTOCOL", "0") == "1"


@dataclass
class DraftRpcResponse:
    draft_token_ids: list[list[int]]
    payload_bytes: int = 0


class DraftRpcClient:

    def __init__(self, endpoint: str, timeout_ms: int, max_retries: int) -> None:
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(endpoint)

    def close(self) -> None:
        self._socket.close(linger=0)

    def _request_once(self, payload: dict[str, Any]) -> DraftRpcResponse:
        encoded_payload = msgspec.msgpack.encode(payload)
        self._socket.send(encoded_payload)
        raw = self._socket.recv()
        resp = msgspec.msgpack.decode(raw, type=DraftRpcResponse)
        resp.payload_bytes = len(encoded_payload)
        return resp

    def propose(self, payload: dict[str, Any]) -> DraftRpcResponse:
        attempts = self.max_retries + 1
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                return self._request_once(payload)
            except Exception as e:  # noqa: BLE001
                last_error = e
                logger.warning("Draft RPC call failed: %s", e)
                # REQ socket requires reconnect after timeout/error.
                self._socket.close(linger=0)
                self._socket = self._ctx.socket(zmq.REQ)
                self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
                self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
                self._socket.connect(self.endpoint)
        raise RuntimeError(
            f"Draft RPC failed after {attempts} attempts."
        ) from last_error


class DraftRemoteProposer(DraftModelProposer):
    """Draft proposer that offloads token proposal to a remote draft worker."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        if runner is None:
            raise ValueError("DraftRemoteProposer requires runner=GPUModelRunner.")
        self._runner = runner
        spec_cfg = self.speculative_config
        assert spec_cfg.remote_draft_endpoint is not None
        self.rpc_client = DraftRpcClient(
            endpoint=spec_cfg.remote_draft_endpoint,
            timeout_ms=spec_cfg.remote_draft_rpc_timeout_ms,
            max_retries=spec_cfg.remote_draft_max_retries,
        )
        # Session protocol state: req_id -> length of context already sent.
        # Only populated when v2 protocol is active.
        self._session_lengths: dict[str, int] = {}
        self._use_session_protocol: bool = _use_session_protocol()
        logger.info(
            "Initialized remote draft proposer endpoint=%s timeout_ms=%d retries=%d "
            "session_protocol=%s",
            spec_cfg.remote_draft_endpoint,
            spec_cfg.remote_draft_rpc_timeout_ms,
            spec_cfg.remote_draft_max_retries,
            self._use_session_protocol,
        )
        if spec_cfg.method == "draft_model":
            logger.info(
                "Remote draft_model: omitting target_hidden_states from draft_propose_v1 "
                "(DraftModelProposer uses token ids + attn metadata only)."
            )

    def load_model(self, target_model: torch.nn.Module) -> None:  # noqa: ARG002
        # Remote mode does not load draft weights in target worker.
        return

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes) -> None:  # noqa: ANN001, ARG002
        # Remote mode does not initialize local draft attn backend.
        return

    def initialize_cudagraph_keys(self, cudagraph_mode) -> None:  # noqa: ANN001, ARG002
        return

    def validate_same_kv_cache_group(self, kv_cache_config) -> None:  # noqa: ANN001, ARG002
        return

    def dummy_run(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return

    def propose(  # type: ignore[override]
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata,
        sampling_metadata,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu=None,
        slot_mappings=None,
    ) -> torch.Tensor:
        del sampling_metadata, mm_embed_inputs, slot_mappings
        expected_shape = (next_token_ids.shape[0], self.num_speculative_tokens)
        tp_group = get_tp_group()
        is_tp_leader = tp_group.rank_in_group == 0

        if is_tp_leader:
            context_token_ids = self._runner.gather_remote_draft_context_token_ids()
            is_draft_model_mode = (
                self.speculative_config.method == "draft_model"
            )
            if is_draft_model_mode:
                # Server-side greedy backend will treat context[i][-1] as the
                # request's "next token" (the token to start drafting from).
                # ``gather_remote_draft_context_token_ids`` already does this in
                # the common path, but for partial-prefill / all-PLACEHOLDER
                # samples it falls back to ``backup_next_token_ids`` via the
                # propose-arg ``next_token_ids``; patch those rows here so we
                # can drop ``next_token_ids`` from the wire entirely.
                nt_cpu = next_token_ids.detach().cpu().tolist()
                for i, nt in enumerate(nt_cpu):
                    nt_int = int(nt)
                    seq = context_token_ids[i]
                    if not seq or seq[-1] != nt_int:
                        seq.append(nt_int)

            if is_draft_model_mode and self._use_session_protocol:
                payload = self._build_v2_payload(context_token_ids)
            else:
                payload = build_draft_propose_v1_payload(
                    target_token_ids=target_token_ids,
                    target_positions=target_positions,
                    target_hidden_states=target_hidden_states,
                    next_token_ids=next_token_ids,
                    token_indices_to_sample=token_indices_to_sample,
                    common_attn_metadata=common_attn_metadata,
                    num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                    num_speculative_tokens=self.num_speculative_tokens,
                    context_token_ids=context_token_ids,
                    include_target_hidden_states=not is_draft_model_mode,
                    omit_unused_for_greedy=is_draft_model_mode,
                )
            resp = self.rpc_client.propose(payload)
            tokens = torch.tensor(resp.draft_token_ids,
                                  dtype=torch.int64,
                                  device=self.device)
        else:
            tokens = torch.empty(expected_shape, dtype=torch.int64, device=self.device)

        tp_group.broadcast(tokens, src=0)
        if tuple(tokens.shape) != expected_shape:
            raise RuntimeError(
                f"Invalid draft RPC response shape {tuple(tokens.shape)} "
                f"expected {expected_shape}."
            )
        return tokens

    def _build_v2_payload(
        self, context_token_ids: list[list[int]]
    ) -> dict[str, Any]:
        """Build a session-incremental ``draft_propose_v2`` payload.

        For each request: send the full context on the first step (or after a
        gap), otherwise send only the suffix appended since the previous step.
        Sessions for req_ids that left the batch are listed under ``evict``
        so the server can free per-session state.
        """
        ib = self._runner.input_batch
        num_reqs = ib.num_reqs
        req_ids = list(ib.req_ids[:num_reqs])
        if len(req_ids) != len(context_token_ids):
            raise RuntimeError(
                f"v2 session payload mismatch: req_ids={len(req_ids)} "
                f"context_rows={len(context_token_ids)}"
            )

        sessions: list[DraftSessionEntry] = []
        for sid, ctx in zip(req_ids, context_token_ids, strict=True):
            prev_len = self._session_lengths.get(sid, 0)
            cur_len = len(ctx)
            if prev_len == 0 or prev_len > cur_len or ctx[:prev_len] is None:
                # Fresh session, or our cached prefix length is stale (e.g.,
                # server lost state across reconnect). Resend full context.
                tokens = list(ctx)
                is_first = True
            else:
                tokens = list(ctx[prev_len:])
                is_first = False
                if not tokens:
                    # No new tokens since last step (rare: discard, no sample).
                    # Still emit an entry so server preserves order; mark as
                    # incremental with empty delta — server should re-issue
                    # K drafts from cached state.
                    pass
            sessions.append(
                DraftSessionEntry(id=sid, is_first=is_first, tokens=tokens)
            )
            self._session_lengths[sid] = cur_len

        live = set(req_ids)
        evict = [sid for sid in self._session_lengths if sid not in live]
        for sid in evict:
            del self._session_lengths[sid]

        return build_draft_propose_v2_payload(
            num_speculative_tokens=self.num_speculative_tokens,
            sessions=sessions,
            evict=evict or None,
        )
