# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import threading

import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.v1.spec_decode.draft_remote import DraftRpcClient
from vllm.v1.worker.gpu.spec_decode.draft_remote_server import DraftRemoteServer
from vllm.v1.worker.gpu.spec_decode.draft_rpc_payload import (
    DRAFT_PROPOSE_V1,
    build_draft_propose_v1_payload,
    deserialize_draft_propose_v1,
    portable_common_attn_to_payload,
    rebuild_common_attn_metadata,
    tensor_chunk_from_payload,
    tensor_chunk_to_payload,
)

BLOCK_SIZE = 16


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_tensor_chunk_roundtrip():
    device = _device()
    x = torch.randn(3, 5, dtype=torch.float16, device=device)
    p = tensor_chunk_to_payload(x)
    y = tensor_chunk_from_payload(p, device)
    assert torch.allclose(x.cpu(), y.cpu())


def test_tensor_chunk_roundtrip_bfloat16():
    device = _device()
    x = torch.randn(3, 5, dtype=torch.bfloat16, device=device)
    p = tensor_chunk_to_payload(x)
    y = tensor_chunk_from_payload(p, device)
    assert torch.allclose(x.cpu(), y.cpu())


def test_rebuild_block_table_row_covers_speculative_length():
    """When max_seq_len == seq_len, inner EAGLE steps still need extra KV blocks."""
    device = _device()
    portable = {
        "query_start_loc": tensor_chunk_to_payload(
            torch.tensor([0, 1], dtype=torch.int32).cpu()
        ),
        "query_start_loc_cpu": tensor_chunk_to_payload(
            torch.tensor([0, 1], dtype=torch.int32).cpu()
        ),
        "seq_lens": tensor_chunk_to_payload(torch.tensor([30], dtype=torch.int32).cpu()),
        "num_reqs": 1,
        "num_actual_tokens": 1,
        "max_query_len": 1,
        "max_seq_len": 30,
        "causal": True,
    }
    positions = torch.tensor([[29]], device=device, dtype=torch.int64)
    cad = rebuild_common_attn_metadata(
        portable,
        device,
        block_size=BLOCK_SIZE,
        positions=positions,
        num_kv_blocks=256,
        num_speculative_tokens=4,
    )
    expected_width = (30 + 4 + BLOCK_SIZE - 1) // BLOCK_SIZE
    assert cad.block_table_tensor.shape[1] == expected_width
    assert expected_width == 3
    assert int(cad.block_table_tensor[0, 2].item()) != 0


def test_common_attn_metadata_portable_roundtrip_logical():
    device = _device()
    batch_spec = BatchSpec(seq_lens=[4, 5], query_lens=[3, 2])
    cad = create_common_attn_metadata(
        batch_spec, block_size=BLOCK_SIZE, device=device, arange_block_indices=True
    )
    portable = portable_common_attn_to_payload(cad)
    rebuilt = rebuild_common_attn_metadata(portable, device, block_size=BLOCK_SIZE)
    assert torch.equal(cad.seq_lens, rebuilt.seq_lens)
    assert torch.equal(cad.query_start_loc, rebuilt.query_start_loc)
    assert cad.num_actual_tokens == rebuilt.num_actual_tokens
    assert cad.num_reqs == rebuilt.num_reqs


def test_build_draft_propose_v1_payload_has_schema():
    device = _device()
    batch_spec = BatchSpec(seq_lens=[3, 2], query_lens=[3, 2])
    cad = create_common_attn_metadata(
        batch_spec, block_size=BLOCK_SIZE, device=device, arange_block_indices=True
    )
    tt = torch.randint(0, 100, (5,), device=device, dtype=torch.int32)
    tp = torch.arange(5, device=device, dtype=torch.int64)
    th = torch.randn(5, 16, dtype=torch.float16, device=device)
    nt = torch.tensor([10, 20], dtype=torch.int32, device=device)
    payload = build_draft_propose_v1_payload(
        target_token_ids=tt,
        target_positions=tp,
        target_hidden_states=th,
        next_token_ids=nt,
        token_indices_to_sample=None,
        common_attn_metadata=cad,
        num_rejected_tokens_gpu=None,
        num_speculative_tokens=2,
    )
    assert payload["rpc_schema"] == DRAFT_PROPOSE_V1


def test_build_draft_propose_v1_payload_omits_hidden_for_draft_model_mode():
    device = _device()
    batch_spec = BatchSpec(seq_lens=[3, 2], query_lens=[3, 2])
    cad = create_common_attn_metadata(
        batch_spec, block_size=BLOCK_SIZE, device=device, arange_block_indices=True
    )
    tt = torch.randint(0, 100, (5,), device=device, dtype=torch.int32)
    tp = torch.arange(5, device=device, dtype=torch.int64)
    th = torch.randn(5, 64, dtype=torch.float16, device=device)
    nt = torch.tensor([10, 20], dtype=torch.int32, device=device)
    payload = build_draft_propose_v1_payload(
        target_token_ids=tt,
        target_positions=tp,
        target_hidden_states=th,
        next_token_ids=nt,
        token_indices_to_sample=None,
        common_attn_metadata=cad,
        num_rejected_tokens_gpu=None,
        num_speculative_tokens=2,
        include_target_hidden_states=False,
    )
    assert "target_hidden_states" not in payload
    ds = deserialize_draft_propose_v1(
        payload,
        device=device,
        block_size=BLOCK_SIZE,
        omit_target_hs_fill_hidden_size=64,
        omit_target_hs_dtype=torch.float16,
    )
    assert ds.target_hidden_states.shape == (5, 64)
    assert torch.all(ds.target_hidden_states == 0)


def test_build_draft_propose_v1_payload_omits_unused_for_greedy():
    device = _device()
    batch_spec = BatchSpec(seq_lens=[3, 2], query_lens=[3, 2])
    cad = create_common_attn_metadata(
        batch_spec, block_size=BLOCK_SIZE, device=device, arange_block_indices=True
    )
    tt = torch.randint(0, 100, (5,), device=device, dtype=torch.int32)
    tp = torch.arange(5, device=device, dtype=torch.int64)
    th = torch.randn(5, 64, dtype=torch.float16, device=device)
    nt = torch.tensor([10, 20], dtype=torch.int32, device=device)
    tis = torch.tensor([2, 4], dtype=torch.int32, device=device)
    nrt = torch.tensor([0, 1], dtype=torch.int32, device=device)
    payload = build_draft_propose_v1_payload(
        target_token_ids=tt,
        target_positions=tp,
        target_hidden_states=th,
        next_token_ids=nt,
        token_indices_to_sample=tis,
        common_attn_metadata=cad,
        num_rejected_tokens_gpu=nrt,
        num_speculative_tokens=2,
        context_token_ids=[[1, 2, 3], [4, 5]],
        include_target_hidden_states=False,
        omit_unused_for_greedy=True,
    )
    for key in (
        "target_token_ids",
        "target_positions",
        "common_attn_metadata",
        "token_indices_to_sample",
        "num_rejected_tokens_gpu",
        "target_hidden_states",
    ):
        assert key not in payload, f"{key} should be stripped for greedy mode"
    assert payload["next_token_ids"] is not None
    assert payload["context_token_ids"] == [[1, 2, 3], [4, 5]]
    assert payload["num_speculative_tokens"] == 2

    with pytest.raises(ValueError, match="omit_unused_for_greedy=True"):
        deserialize_draft_propose_v1(
            payload,
            device=device,
            block_size=BLOCK_SIZE,
            omit_target_hs_fill_hidden_size=64,
        )


def test_deserialize_missing_hs_requires_fill_dimensions():
    device = _device()
    batch_spec = BatchSpec(seq_lens=[3, 2], query_lens=[3, 2])
    cad = create_common_attn_metadata(
        batch_spec, block_size=BLOCK_SIZE, device=device, arange_block_indices=True
    )
    tt = torch.randint(0, 100, (5,), device=device, dtype=torch.int32)
    tp = torch.arange(5, device=device, dtype=torch.int64)
    th = torch.randn(5, 16, dtype=torch.float16, device=device)
    nt = torch.tensor([10, 20], dtype=torch.int32, device=device)
    payload = build_draft_propose_v1_payload(
        target_token_ids=tt,
        target_positions=tp,
        target_hidden_states=th,
        next_token_ids=nt,
        token_indices_to_sample=None,
        common_attn_metadata=cad,
        num_rejected_tokens_gpu=None,
        num_speculative_tokens=2,
        include_target_hidden_states=False,
    )
    with pytest.raises(ValueError, match="omit_target_hs_fill_hidden_size"):
        deserialize_draft_propose_v1(
            payload, device=device, block_size=BLOCK_SIZE
        )


class DictDraftFn:
    accepts_rpc_dict = True

    def __call__(self, req: dict) -> list[list[int]]:
        assert req["rpc_schema"] == DRAFT_PROPOSE_V1
        n = int(req["num_speculative_tokens"])
        batch = tensor_chunk_from_payload(req["next_token_ids"], torch.device("cpu")).numel()
        return [[(i % 5000)] * n for i in range(batch)]


def test_draft_remote_rpc_v1_dict_dispatch():
    import random

    port = random.randint(20000, 50000)
    endpoint = f"tcp://127.0.0.1:{port}"
    server = DraftRemoteServer(endpoint, propose_fn=DictDraftFn())

    t = threading.Thread(target=server.serve_once, daemon=True)
    t.start()

    device = _device()
    batch_spec = BatchSpec(seq_lens=[3, 2], query_lens=[3, 2])
    cad = create_common_attn_metadata(
        batch_spec, block_size=BLOCK_SIZE, device=device, arange_block_indices=True
    )
    tt = torch.randint(0, 100, (5,), device=device, dtype=torch.int32)
    tp = torch.arange(5, device=device, dtype=torch.int64)
    th = torch.randn(5, 16, dtype=torch.float16, device=device)
    nt = torch.tensor([10, 20], dtype=torch.int32, device=device)
    payload = build_draft_propose_v1_payload(
        target_token_ids=tt,
        target_positions=tp,
        target_hidden_states=th,
        next_token_ids=nt,
        token_indices_to_sample=None,
        common_attn_metadata=cad,
        num_rejected_tokens_gpu=None,
        num_speculative_tokens=3,
    )

    client = DraftRpcClient(endpoint=endpoint, timeout_ms=5000, max_retries=0)
    resp = client.propose(payload)
    assert len(resp.draft_token_ids) == 2
    assert all(len(row) == 3 for row in resp.draft_token_ids)
    t.join(timeout=5)
    client.close()
    server.close()
