# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import msgspec
import torch
import zmq

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.worker.gpu.spec_decode.draft_rpc_payload import build_draft_propose_v1_payload

logger = init_logger(__name__)


@dataclass
class DraftRpcResponse:
    draft_token_ids: list[list[int]]


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
        self._socket.send(msgspec.msgpack.encode(payload))
        raw = self._socket.recv()
        return msgspec.msgpack.decode(raw, type=DraftRpcResponse)

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
        logger.info(
            "Initialized remote draft proposer endpoint=%s timeout_ms=%d retries=%d",
            spec_cfg.remote_draft_endpoint,
            spec_cfg.remote_draft_rpc_timeout_ms,
            spec_cfg.remote_draft_max_retries,
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

        context_token_ids = self._runner.gather_remote_draft_context_token_ids()
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
        )
        resp = self.rpc_client.propose(payload)
        tokens = torch.tensor(resp.draft_token_ids, dtype=torch.int64, device=self.device)
        expected_shape = (next_token_ids.shape[0], self.num_speculative_tokens)
        if tuple(tokens.shape) != expected_shape:
            raise RuntimeError(
                f"Invalid draft RPC response shape {tuple(tokens.shape)} "
                f"expected {expected_shape}."
            )
        return tokens
