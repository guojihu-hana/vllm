# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
import threading

import torch

from vllm.v1.spec_decode.draft_remote import DraftRpcClient
from vllm.v1.worker.gpu.spec_decode.draft_remote_server import DraftRemoteServer


def test_draft_remote_rpc_roundtrip():
    port = random.randint(20000, 50000)
    endpoint = f"tcp://127.0.0.1:{port}"
    server = DraftRemoteServer(endpoint)

    t = threading.Thread(target=server.serve_once, daemon=True)
    t.start()

    client = DraftRpcClient(endpoint=endpoint, timeout_ms=5000, max_retries=0)
    resp = client.propose(
        {
            "next_token_ids": [11, 12],
            "num_speculative_tokens": 3,
        }
    )
    assert resp.draft_token_ids == [[11, 11, 11], [12, 12, 12]]
    t.join(timeout=5)
    client.close()
    server.close()


def test_draft_remote_rpc_with_context_token_ids_ignored_by_stub():
    port = random.randint(30000, 40000)
    endpoint = f"tcp://127.0.0.1:{port}"
    server = DraftRemoteServer(endpoint)

    t = threading.Thread(target=server.serve_once, daemon=True)
    t.start()

    client = DraftRpcClient(endpoint=endpoint, timeout_ms=5000, max_retries=0)
    resp = client.propose(
        {
            "next_token_ids": [7],
            "num_speculative_tokens": 2,
            "context_token_ids": [[1, 2, 7]],
        }
    )
    assert resp.draft_token_ids == [[7, 7]]
    t.join(timeout=5)
    client.close()
    server.close()


def test_draft_remote_rpc_retries_exhausted():
    port = random.randint(50001, 59999)
    endpoint = f"tcp://127.0.0.1:{port}"
    client = DraftRpcClient(endpoint=endpoint, timeout_ms=10, max_retries=1)
    try:
        try:
            client.propose(
                {
                    "next_token_ids": [1],
                    "num_speculative_tokens": 2,
                }
            )
            raise AssertionError("Expected RuntimeError when remote endpoint is absent")
        except RuntimeError as e:
            assert "Draft RPC failed after" in str(e)
    finally:
        client.close()


def test_draft_remote_rpc_transports_hidden_states():
    port = random.randint(60000, 65000)
    endpoint = f"tcp://127.0.0.1:{port}"
    captured: dict[str, torch.Tensor | None] = {"hs": None}

    def _fn(next_token_ids, num_speculative_tokens, context_token_ids=None, target_hidden_states=None):  # noqa: ANN001
        del context_token_ids
        captured["hs"] = target_hidden_states
        return [[tid] * num_speculative_tokens for tid in next_token_ids]

    server = DraftRemoteServer(endpoint, propose_fn=_fn)
    t = threading.Thread(target=server.serve_once, daemon=True)
    t.start()
    client = DraftRpcClient(endpoint=endpoint, timeout_ms=5000, max_retries=0)
    hs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
    resp = client.propose(
        {
            "next_token_ids": [9, 10],
            "num_speculative_tokens": 2,
            "target_hidden_states_shape": [2, 2],
            "target_hidden_states_dtype": "float16",
            "target_hidden_states_bytes": hs.numpy().tobytes(),
        }
    )
    assert resp.draft_token_ids == [[9, 9], [10, 10]]
    t.join(timeout=5)
    client.close()
    server.close()
    assert captured["hs"] is not None
    assert torch.equal(captured["hs"], hs)
