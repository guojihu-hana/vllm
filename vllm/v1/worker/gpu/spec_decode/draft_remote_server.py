# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from typing import Any

import msgspec
import torch
import zmq

from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.draft_rpc_payload import DRAFT_PROPOSE_V1

logger = init_logger(__name__)

DraftProposalFn = Callable[..., list[list[int]]]


class RepeatTokenDraftFn:
    """Default fallback draft proposer used by remote server.

    This keeps server wiring functional for deployments that bring their own
    proposal function and avoids hard-coding model/runtime dependencies here.
    """

    def __call__(
        self,
        next_token_ids: list[int],
        num_speculative_tokens: int,
        context_token_ids: list[list[int]] | None = None,
        target_hidden_states: torch.Tensor | None = None,
    ) -> list[list[int]]:
        del context_token_ids, target_hidden_states
        return [[tid] * num_speculative_tokens for tid in next_token_ids]


class DraftRemoteServer:

    def __init__(
        self,
        endpoint: str,
        propose_fn: DraftProposalFn | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.propose_fn = propose_fn or RepeatTokenDraftFn()
        self.ctx = zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.bind(endpoint)
        logger.info("Draft remote server bound on %s", endpoint)

    def close(self) -> None:
        self.socket.close(linger=0)

    def run_forever(self) -> None:
        while True:
            self.serve_once()

    def serve_once(self) -> None:
        raw = self.socket.recv()
        t0 = time.perf_counter()
        req = msgspec.msgpack.decode(raw, type=dict[str, Any])
        if req.get("rpc_schema") == DRAFT_PROPOSE_V1:
            if not getattr(self.propose_fn, "accepts_rpc_dict", False):
                raise ValueError(
                    "Received draft_propose_v1 RPC but this worker backend does not "
                    "implement native parity decoding. Start draft_remote_server.py "
                    "with --backend native --target-model ... --model <draft>."
                )
            draft_token_ids = self.propose_fn(req)
        else:
            next_token_ids = list(req.get("next_token_ids", []))
            num_speculative_tokens = int(req.get("num_speculative_tokens", 1))
            ctx = req.get("context_token_ids")
            if ctx is not None:
                ctx = [[int(x) for x in row] for row in ctx]
            target_hidden_states = None
            hs_bytes = req.get("target_hidden_states_bytes")
            hs_shape = req.get("target_hidden_states_shape")
            hs_dtype = req.get("target_hidden_states_dtype")
            if hs_bytes is not None and hs_shape is not None and hs_dtype is not None:
                if hs_dtype != "float16":
                    raise ValueError(
                        f"Unsupported target_hidden_states_dtype: {hs_dtype}"
                    )
                target_hidden_states = torch.frombuffer(
                    bytes(hs_bytes), dtype=torch.float16
                ).reshape(hs_shape).clone()
            draft_token_ids = self.propose_fn(
                next_token_ids,
                num_speculative_tokens,
                context_token_ids=ctx,
                target_hidden_states=target_hidden_states,
            )
        resp = {
            "draft_token_ids": draft_token_ids,
            "server_elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }
        self.socket.send(msgspec.msgpack.encode(resp))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remote draft worker for vLLM speculative decoding (ZMQ REP)."
    )
    p.add_argument(
        "--endpoint",
        type=str,
        default="tcp://0.0.0.0:18861",
        help="ZMQ bind address (e.g. tcp://0.0.0.0:18861)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Draft model id or local path.",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="native",
        choices=("native", "vllm", "hf"),
        help=(
            "Draft inference backend: native uses DraftModelProposer parity RPC "
            "(draft_propose_v1); vllm uses LLM.generate greedy replay; hf uses "
            "transformers."
        ),
    )
    p.add_argument(
        "--target-model",
        type=str,
        default=None,
        help="Target model id/path (required for --backend native).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for draft weights (use cuda with CUDA_VISIBLE_DEVICES=1).",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Max prompt tokens sent per request (truncate from the left).",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=("auto", "float16", "bfloat16", "float32"),
        help="Weight dtype for HF load (default: auto / env VLLM_REMOTE_DRAFT_DTYPE).",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Draft worker tensor parallel size (default 1). Requires enough visible "
            "GPUs. Not supported with --backend native when "
            "VLLM_REMOTE_DRAFT_USE_EAGLE_PARITY=1."
        ),
    )
    return p.parse_args()


def _dtype_str_for_native(dtype: torch.dtype | str | None) -> str:
    if dtype is None or dtype == "auto":
        return "auto"
    if isinstance(dtype, torch.dtype):
        return str(dtype).replace("torch.", "")
    return dtype


def main() -> None:
    args = _parse_args()
    dtype: torch.dtype | str | None
    if args.dtype is None:
        dtype = None
    elif args.dtype == "auto":
        dtype = "auto"
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if args.model:
        from vllm.v1.worker.gpu.spec_decode.draft_remote_inference import (
            HFTransformersDraftFn,
            VLLMGreedyDraftFn,
        )
        from vllm.v1.worker.gpu.spec_decode.draft_remote_native import (
            DraftModelNativeParityFn,
        )

        if args.backend == "native":
            if args.target_model is None:
                raise ValueError("--target-model is required when --backend native.")
            num_spec = int(
                os.environ.get(
                    "VLLM_REMOTE_DRAFT_NUM_SPECULATIVE_TOKENS",
                    os.environ.get("VLLM_REMOTE_NUM_SPECULATIVE_TOKENS", "1"),
                )
            )
            propose_fn = DraftModelNativeParityFn(
                target_model=args.target_model,
                draft_model=args.model,
                num_speculative_tokens=num_spec,
                max_model_len=args.max_seq_len,
                dtype=_dtype_str_for_native(dtype),
                tensor_parallel_size=args.tensor_parallel_size,
            )
        elif args.backend == "vllm":
            propose_fn = VLLMGreedyDraftFn(
                args.model,
                max_seq_len=args.max_seq_len,
                dtype=dtype,
                tensor_parallel_size=args.tensor_parallel_size,
            )
        else:
            propose_fn = HFTransformersDraftFn(
                args.model,
                device=torch.device(args.device),
                max_seq_len=args.max_seq_len,
                dtype=dtype,
            )
    else:
        propose_fn = RepeatTokenDraftFn()
        logger.warning(
            "No --model: using RepeatTokenDraftFn (not real inference). "
            "Pass --model <hf_id_or_path> for GPU draft."
        )

    server = DraftRemoteServer(args.endpoint, propose_fn=propose_fn)
    try:
        server.run_forever()
    finally:
        server.close()


if __name__ == "__main__":
    main()
