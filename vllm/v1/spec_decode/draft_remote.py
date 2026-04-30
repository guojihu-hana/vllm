# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import msgspec
import torch
import zmq

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.worker.gpu.spec_decode.draft_rpc_payload import build_draft_propose_v1_payload

logger = init_logger(__name__)


@dataclass
class DraftRpcResponse:
    draft_token_ids: list[list[int]]
    server_elapsed_ms: float = 0.0
    rpc_round_trip_ms: float = 0.0
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
        t0 = time.perf_counter()
        encoded_payload = msgspec.msgpack.encode(payload)
        self._socket.send(encoded_payload)
        raw = self._socket.recv()
        resp = msgspec.msgpack.decode(raw, type=DraftRpcResponse)
        # Attach round-trip time for communication/decode-side attribution.
        resp.rpc_round_trip_ms = (time.perf_counter() - t0) * 1000.0
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
        self._timing_log_interval = int(
            os.environ.get("VLLM_REMOTE_DRAFT_TIMING_LOG_INTERVAL", "50")
        )
        self._timing_calls = 0
        self._sum_target_prepare_ms = 0.0
        self._sum_rpc_round_trip_ms = 0.0
        self._sum_server_elapsed_ms = 0.0
        self._sum_comm_ms = 0.0
        self._sum_payload_bytes = 0
        logger.info(
            "Initialized remote draft proposer endpoint=%s timeout_ms=%d retries=%d",
            spec_cfg.remote_draft_endpoint,
            spec_cfg.remote_draft_rpc_timeout_ms,
            spec_cfg.remote_draft_max_retries,
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
            omit_target_hs_payload = (
                self.speculative_config.method == "draft_model"
            )
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
                include_target_hidden_states=not omit_target_hs_payload,
            )
            resp = self.rpc_client.propose(payload)
            rpc_round_trip_ms = float(getattr(resp, "rpc_round_trip_ms", 0.0))
            server_elapsed_ms = float(getattr(resp, "server_elapsed_ms", 0.0))
            comm_ms = max(0.0, rpc_round_trip_ms - server_elapsed_ms)
            payload_bytes = int(getattr(resp, "payload_bytes", 0))
            target_prepare_ms = float(
                getattr(self._runner, "_remote_draft_target_prepare_ms", 0.0)
            )
            self._timing_calls += 1
            self._sum_target_prepare_ms += target_prepare_ms
            self._sum_rpc_round_trip_ms += rpc_round_trip_ms
            self._sum_server_elapsed_ms += server_elapsed_ms
            self._sum_comm_ms += comm_ms
            self._sum_payload_bytes += payload_bytes
            if self._timing_calls % self._timing_log_interval == 0:
                n = float(self._timing_calls)
                logger.info(
                    "Remote draft timing avg over %d calls: "
                    "target_prepare=%.3f ms, draft_generate=%.3f ms, communication=%.3f ms "
                    "(rpc_round_trip=%.3f ms, payload_bytes=%.1f)",
                    self._timing_calls,
                    self._sum_target_prepare_ms / n,
                    self._sum_server_elapsed_ms / n,
                    self._sum_comm_ms / n,
                    self._sum_rpc_round_trip_ms / n,
                    self._sum_payload_bytes / n,
                )
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
