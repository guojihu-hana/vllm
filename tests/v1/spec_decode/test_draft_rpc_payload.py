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
    DRAFT_PROPOSE_V2,
    DraftSessionEntry,
    build_draft_propose_v1_payload,
    build_draft_propose_v2_payload,
    deserialize_draft_propose_v1,
    parse_draft_propose_v2_payload,
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
        "next_token_ids",
    ):
        assert key not in payload, f"{key} should be stripped for greedy mode"
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


def test_build_parse_draft_propose_v2_roundtrip():
    sessions = [
        DraftSessionEntry(id="r0", is_first=True, tokens=[1, 2, 3, 4]),
        DraftSessionEntry(id="r1", is_first=False, tokens=[7]),
    ]
    payload = build_draft_propose_v2_payload(
        num_speculative_tokens=3,
        sessions=sessions,
        evict=["r9"],
    )
    assert payload["rpc_schema"] == DRAFT_PROPOSE_V2
    assert payload["num_speculative_tokens"] == 3
    assert payload["evict"] == ["r9"]
    assert payload["sessions"][0] == {
        "id": "r0",
        "is_first": True,
        "tokens": [1, 2, 3, 4],
    }

    k, parsed_sessions, evict = parse_draft_propose_v2_payload(payload)
    assert k == 3
    assert evict == ["r9"]
    assert [s.id for s in parsed_sessions] == ["r0", "r1"]
    assert parsed_sessions[0].is_first is True
    assert parsed_sessions[0].tokens == [1, 2, 3, 4]
    assert parsed_sessions[1].is_first is False
    assert parsed_sessions[1].tokens == [7]


def test_parse_draft_propose_v2_rejects_wrong_schema():
    bad = {
        "rpc_schema": DRAFT_PROPOSE_V1,
        "num_speculative_tokens": 1,
        "sessions": [],
    }
    with pytest.raises(ValueError, match=DRAFT_PROPOSE_V2):
        parse_draft_propose_v2_payload(bad)


class _StubSessionFn:
    """Test-only session backend: no LLM, just records and replays."""

    accepts_rpc_dict = True

    def __init__(self) -> None:
        self.sessions: dict[str, list[int]] = {}
        self.calls: list[tuple[list[str], list[str]]] = []

    def __call__(self, req):
        k, sessions, evict = parse_draft_propose_v2_payload(req)
        for sid in evict:
            self.sessions.pop(sid, None)
        order = []
        for s in sessions:
            if s.is_first:
                self.sessions[s.id] = list(s.tokens)
            else:
                self.sessions[s.id].extend(s.tokens)
            order.append(s.id)
        self.calls.append((order, list(evict)))
        # Echo session id length as drafts (deterministic + observable).
        return [
            [self.sessions[sid][-1]] * k for sid in order
        ]


def test_session_lifecycle_via_rpc():
    import random

    port = random.randint(40000, 49999)
    endpoint = f"tcp://127.0.0.1:{port}"
    fn = _StubSessionFn()
    server = DraftRemoteServer(endpoint, propose_fn=fn)
    serve_thread = threading.Thread(
        target=lambda: [server.serve_once() for _ in range(3)], daemon=True
    )
    serve_thread.start()

    client = DraftRpcClient(endpoint=endpoint, timeout_ms=5000, max_retries=0)
    # Step 1: two new sessions.
    p1 = build_draft_propose_v2_payload(
        num_speculative_tokens=2,
        sessions=[
            DraftSessionEntry(id="a", is_first=True, tokens=[10, 11, 12]),
            DraftSessionEntry(id="b", is_first=True, tokens=[20, 21]),
        ],
    )
    r1 = client.propose(p1)
    assert r1.draft_token_ids == [[12, 12], [21, 21]]
    assert fn.sessions["a"] == [10, 11, 12]
    assert fn.sessions["b"] == [20, 21]

    # Step 2: increments only.
    p2 = build_draft_propose_v2_payload(
        num_speculative_tokens=2,
        sessions=[
            DraftSessionEntry(id="a", is_first=False, tokens=[13]),
            DraftSessionEntry(id="b", is_first=False, tokens=[22, 23]),
        ],
    )
    r2 = client.propose(p2)
    assert r2.draft_token_ids == [[13, 13], [23, 23]]
    assert fn.sessions["a"] == [10, 11, 12, 13]
    assert fn.sessions["b"] == [20, 21, 22, 23]

    # Step 3: evict 'a', keep 'b'.
    p3 = build_draft_propose_v2_payload(
        num_speculative_tokens=2,
        sessions=[DraftSessionEntry(id="b", is_first=False, tokens=[24])],
        evict=["a"],
    )
    r3 = client.propose(p3)
    assert r3.draft_token_ids == [[24, 24]]
    assert "a" not in fn.sessions
    assert fn.sessions["b"] == [20, 21, 22, 23, 24]

    serve_thread.join(timeout=5)
    client.close()
    server.close()




# ---------------------------------------------------------------------------
# PersistentEngineDraftFn lifecycle tests.
#
# The implementation drives ``engine.step()`` over a long-lived in-engine
# Request per session. Each RPC: append target's delta, run K steps, sample
# the K drafts, and roll back request + cache-manager state. These tests use
# fakes for the scheduler / engine / kv-cache-manager and verify the wiring
# (add vs append, K steps, draft truncation, snapshot-based rollback,
# polluted-block eviction).
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Stand-in for vllm.v1.request.Request."""

    def __init__(self, prompt_token_ids):
        self._all_token_ids = list(prompt_token_ids)
        self._output_token_ids = []
        self.num_prompt_tokens = len(prompt_token_ids)
        self.num_computed_tokens = 0
        self.block_hashes = []
        self.request_id = None  # set by engine.add_request

    def append_output_token_ids(self, tokens):
        if isinstance(tokens, int):
            tokens = [tokens]
        self._output_token_ids.extend(tokens)
        self._all_token_ids.extend(tokens)


class _FakeBlock:
    def __init__(self, block_id):
        self.block_id = block_id
        self.block_hash = None


class _FakeBlockPool:
    def __init__(self):
        # Maps block_hash -> KVCacheBlock; we just track which entries got
        # evicted so tests can assert the rollback called eviction.
        self.evicted_blocks: list[_FakeBlock] = []

    def _maybe_evict_cached_block(self, block: _FakeBlock) -> bool:
        if block.block_hash is None:
            return False
        self.evicted_blocks.append(block)
        block.block_hash = None
        return True


class _FakeSingleTypeManager:
    def __init__(self):
        self.num_cached_block: dict[str, int] = {}
        self.req_to_blocks: dict[str, list[_FakeBlock]] = {}


class _FakeKVCacheCoordinator:
    def __init__(self, single_type_managers):
        self.single_type_managers = single_type_managers


class _FakeKVCacheManager:
    def __init__(self):
        self.block_pool = _FakeBlockPool()
        self.coordinator = _FakeKVCacheCoordinator(
            [_FakeSingleTypeManager()]
        )


class _FakeScheduler:
    def __init__(self):
        self.requests: dict[str, _FakeRequest] = {}
        self.kv_cache_manager = _FakeKVCacheManager()


class _Fake2DArray:
    """Supports ``arr[req_idx, slice] = list_or_bool`` like numpy 2D access."""

    def __init__(self, max_seq_len: int, fill):
        self._rows: dict[int, list] = {}
        self._max = max_seq_len
        self._fill = fill

    def _row(self, idx: int) -> list:
        if idx not in self._rows:
            self._rows[idx] = [self._fill] * self._max
        return self._rows[idx]

    def __setitem__(self, key, value):
        req_idx, sl = key
        row = self._row(req_idx)
        if isinstance(sl, slice):
            length = sl.stop - sl.start
            if isinstance(value, (list, tuple)):
                row[sl] = list(value)
            else:  # scalar (e.g. True)
                row[sl] = [value] * length
        else:
            row[sl] = value

    def __getitem__(self, key):
        if isinstance(key, tuple):
            req_idx, sl = key
            return self._row(req_idx)[sl]
        return self._row(key)


class _FakeInputBatch:
    """Minimal stand-in for GPUModelRunner.input_batch used by the
    persistent backend's ``_sync_delta_into_input_batch`` and rollback
    resets."""

    def __init__(self, max_seq_len: int = 8192):
        self.req_id_to_index: dict[str, int] = {}
        self.token_ids_cpu = _Fake2DArray(max_seq_len, fill=0)
        self.is_token_ids = _Fake2DArray(max_seq_len, fill=False)
        self.num_tokens_no_spec: dict[int, int] = {}


class _FakeEngine:
    """Records the sequence of (op, args) calls and, on each step, appends
    one ``draft_token`` per alive request to mimic greedy single-token
    sampling."""

    def __init__(self, scheduler: _FakeScheduler, draft_token: int = 99):
        self._scheduler = scheduler
        self._draft_token = draft_token
        self.calls: list[tuple] = []

    def add_request(self, request_id, prompt, params):
        self.calls.append(("add", request_id, len(prompt["prompt_token_ids"])))
        req = _FakeRequest(prompt["prompt_token_ids"])
        req.request_id = request_id
        self._scheduler.requests[request_id] = req
        return request_id

    def abort_request(self, request_ids):
        self.calls.append(("abort", list(request_ids)))
        for rid in request_ids:
            self._scheduler.requests.pop(rid, None)

    def step(self):
        self.calls.append(("step",))
        for req in self._scheduler.requests.values():
            req.num_computed_tokens = len(req._all_token_ids)
            req.append_output_token_ids(self._draft_token)


def _build_persistent_fn(K: int, draft_token: int = 99, block_size: int = 16):
    """Bypass __init__ to construct a PersistentEngineDraftFn with fakes."""
    from vllm.v1.worker.gpu.spec_decode.draft_remote_inference import (
        PersistentEngineDraftFn,
    )

    fn = PersistentEngineDraftFn.__new__(PersistentEngineDraftFn)
    scheduler = _FakeScheduler()
    engine = _FakeEngine(scheduler, draft_token=draft_token)
    fn._engine = engine
    fn._scheduler = scheduler
    fn._kv_cache_manager = scheduler.kv_cache_manager
    fn._block_pool = scheduler.kv_cache_manager.block_pool
    fn._coordinator = scheduler.kv_cache_manager.coordinator
    fn._single_type_managers = list(
        fn._coordinator.single_type_managers
    )
    fn._input_batch = _FakeInputBatch()
    fn._model_runner = None  # not exercised by the fake-engine flow
    fn._block_size = block_size
    fn._K = K
    fn._draft_params = None
    fn._live_sessions = set()
    fn._session_to_engine_id = {}
    fn._max_sessions = 4096
    fn._call_count = 0
    fn._debug_log_first_n = 0
    fn._debug_log_interval = 0
    fn._short_draft_count = 0
    fn._oversize_delta_count = 0
    return fn, engine, scheduler


def test_persistent_engine_first_call_adds_and_steps_K_times():
    fn, engine, scheduler = _build_persistent_fn(K=3, draft_token=77)
    payload = build_draft_propose_v2_payload(
        num_speculative_tokens=3,
        sessions=[DraftSessionEntry(id="r0", is_first=True, tokens=[1, 2, 3])],
    )
    drafts = fn(payload)
    assert drafts == [[77, 77, 77]]
    ops = [c[0] for c in engine.calls]
    assert ops == ["add", "step", "step", "step"]
    # After rollback the request still owns the prompt; drafts removed.
    req = scheduler.requests["r0"]
    assert req._all_token_ids == [1, 2, 3]
    assert req._output_token_ids == []
    assert req.num_computed_tokens == 3


def test_persistent_engine_incremental_call_appends_no_readd():
    fn, engine, scheduler = _build_persistent_fn(K=2, draft_token=55)
    fn(build_draft_propose_v2_payload(
        num_speculative_tokens=2,
        sessions=[DraftSessionEntry(id="a", is_first=True, tokens=[10, 11])],
    ))
    engine.calls.clear()
    drafts = fn(build_draft_propose_v2_payload(
        num_speculative_tokens=2,
        sessions=[DraftSessionEntry(id="a", is_first=False, tokens=[12])],
    ))
    assert drafts == [[55, 55]]
    ops = [c[0] for c in engine.calls]
    # No re-add; just K steps.
    assert ops == ["step", "step"]
    req = scheduler.requests["a"]
    # Prompt + first delta + second delta; drafts rolled back.
    assert req._all_token_ids == [10, 11, 12]


def test_persistent_engine_multi_session_batched_in_step():
    fn, engine, scheduler = _build_persistent_fn(K=2, draft_token=42)
    drafts = fn(build_draft_propose_v2_payload(
        num_speculative_tokens=2,
        sessions=[
            DraftSessionEntry(id="a", is_first=True, tokens=[1, 2]),
            DraftSessionEntry(id="b", is_first=True, tokens=[5, 6, 7]),
        ],
    ))
    assert drafts == [[42, 42], [42, 42]]
    ops = [c[0] for c in engine.calls]
    # Both adds first, then K steps shared by the batch.
    assert ops == ["add", "add", "step", "step"]


def test_persistent_engine_evict_aborts_request():
    fn, engine, scheduler = _build_persistent_fn(K=1)
    fn(build_draft_propose_v2_payload(
        num_speculative_tokens=1,
        sessions=[DraftSessionEntry(id="a", is_first=True, tokens=[1])],
    ))
    assert "a" in fn._live_sessions
    fn(build_draft_propose_v2_payload(
        num_speculative_tokens=1,
        sessions=[DraftSessionEntry(id="b", is_first=True, tokens=[2])],
        evict=["a"],
    ))
    assert "a" not in fn._live_sessions
    assert "a" not in scheduler.requests
    assert "b" in fn._live_sessions


def test_persistent_engine_is_first_resets_live_session():
    fn, engine, scheduler = _build_persistent_fn(K=1)
    fn(build_draft_propose_v2_payload(
        num_speculative_tokens=1,
        sessions=[DraftSessionEntry(id="a", is_first=True, tokens=[1, 2, 3])],
    ))
    fn(build_draft_propose_v2_payload(
        num_speculative_tokens=1,
        sessions=[DraftSessionEntry(id="a", is_first=True, tokens=[99, 100])],
    ))
    # is_first=True on an existing session aborts and re-adds.
    req = scheduler.requests["a"]
    assert req._all_token_ids[:2] == [99, 100]


def test_persistent_engine_K_mismatch_raises():
    fn, _, _ = _build_persistent_fn(K=4)
    payload = build_draft_propose_v2_payload(
        num_speculative_tokens=2,
        sessions=[DraftSessionEntry(id="a", is_first=True, tokens=[1, 2])],
    )
    with pytest.raises(ValueError, match="K mismatch"):
        fn(payload)


def test_persistent_engine_oversize_delta_is_counted_but_accepted():
    fn, _, scheduler = _build_persistent_fn(K=2)
    fn(build_draft_propose_v2_payload(
        num_speculative_tokens=2,
        sessions=[DraftSessionEntry(id="a", is_first=True, tokens=[1, 2])],
    ))
    # Delta of 5 > K+1=3.
    fn(build_draft_propose_v2_payload(
        num_speculative_tokens=2,
        sessions=[DraftSessionEntry(id="a", is_first=False,
                                     tokens=[3, 4, 5, 6, 7])],
    ))
    assert fn._oversize_delta_count == 1
    req = scheduler.requests["a"]
    assert req._all_token_ids == [1, 2, 3, 4, 5, 6, 7]


def test_persistent_engine_rollback_resets_num_cached_block():
    fn, _, scheduler = _build_persistent_fn(K=4, block_size=4)
    stm = fn._single_type_managers[0]
    payload = build_draft_propose_v2_payload(
        num_speculative_tokens=4,
        sessions=[DraftSessionEntry(id="a", is_first=True,
                                     tokens=[1, 2, 3, 4])],
    )
    # Pretend the K-step loop pushed num_cached_block past the snapshot.
    # We simulate this by patching engine.step to bump it.
    real_step = fn._engine.step

    def step_with_pollution():
        real_step()
        stm.num_cached_block[fn._session_to_engine_id["a"]] = 99
    fn._engine.step = step_with_pollution

    fn(payload)
    req = scheduler.requests[fn._session_to_engine_id["a"]]
    valid_blocks = len(req._all_token_ids) // 4
    # Inflated to 99 by the K-step pollution; rollback clamps it at
    # valid_blocks. We must not pull below ``valid_blocks`` (legitimate
    # prompt-block cachings done by the K-step's first allocate_slots
    # must stay reflected in the watermark, else the pool's
    # ``assert blk.block_hash is None`` fires on the next call).
    assert stm.num_cached_block[req.request_id] == valid_blocks


def test_persistent_engine_rollback_evicts_polluted_block_hashes():
    fn, _, scheduler = _build_persistent_fn(K=4, block_size=4)
    stm = fn._single_type_managers[0]
    pool = fn._block_pool

    # Seed the session: real Request.__init__ runs update_block_hashes for
    # the prompt, so pre-snapshot block_hashes already reflects committed
    # full blocks. Our fake skips that, so we set it up manually after the
    # first call (which is the is_first/add path).
    blk0 = _FakeBlock(0)
    fn(build_draft_propose_v2_payload(
        num_speculative_tokens=4,
        sessions=[DraftSessionEntry(id="a", is_first=True,
                                     tokens=[1, 2, 3, 4])],
    ))
    rid = fn._session_to_engine_id["a"]
    # Inject the pre-snapshot state that real-engine update_block_hashes
    # would have set up during add_request.
    scheduler.requests[rid].block_hashes = ["h0"]
    blk0.block_hash = "h0"
    stm.req_to_blocks[rid] = [blk0]

    # Now the second call's K-step loop pushes 2 polluted block hashes.
    blk1 = _FakeBlock(1)
    blk2 = _FakeBlock(2)

    _orig_step = fn._engine.step

    def step_with_block_growth():
        _orig_step()
        stm.req_to_blocks[rid] = [blk0, blk1, blk2]
        scheduler.requests[rid].block_hashes = ["h0", "h1_polluted",
                                                  "h2_polluted"]
        blk1.block_hash = "h1_polluted"
        blk2.block_hash = "h2_polluted"
    fn._engine.step = step_with_block_growth

    pool.evicted_blocks.clear()
    fn(build_draft_propose_v2_payload(
        num_speculative_tokens=4,
        sessions=[DraftSessionEntry(id="a", is_first=False, tokens=[5])],
    ))
    evicted_ids = {b.block_id for b in pool.evicted_blocks}
    assert evicted_ids == {1, 2}
    assert blk1.block_hash is None
    assert blk2.block_hash is None
    assert blk0.block_hash == "h0", "pre-snapshot block must not be evicted"


def test_persistent_engine_delta_is_synced_into_input_batch():
    """Regression: ``request.append_output_token_ids`` only updates the
    Request, but the model forward reads input ids from
    ``input_batch.token_ids_cpu``. Without explicit sync, the model conditions
    on stale tokens and target acceptance collapses to ~0%."""
    fn, _, scheduler = _build_persistent_fn(K=2, draft_token=42)
    # Seed session.
    fn(build_draft_propose_v2_payload(
        num_speculative_tokens=2,
        sessions=[DraftSessionEntry(id="a", is_first=True, tokens=[10, 11])],
    ))
    # Pretend the engine registered the request in the input_batch (real
    # engine does this when the request transitions waiting → running).
    rid = fn._session_to_engine_id["a"]
    fn._input_batch.req_id_to_index[rid] = 0
    # Simulate that the K-step bookkeeping wrote draft samples at positions
    # 2..3 (the two tokens after the prompt).
    fn._input_batch.token_ids_cpu[0, 2:4] = [777, 888]
    fn._input_batch.num_tokens_no_spec[0] = 4

    # Now target sends an incremental delta. After the call, input_batch's
    # token_ids_cpu must reflect the delta at positions 2..(2+len(delta)),
    # not the stale draft samples.
    fn(build_draft_propose_v2_payload(
        num_speculative_tokens=2,
        sessions=[DraftSessionEntry(id="a", is_first=False,
                                     tokens=[20, 21])],
    ))
    row = fn._input_batch.token_ids_cpu[0]
    assert row[2] == 20, (
        f"expected target delta at position 2, got {row[2]} (stale draft sample)"
    )
    assert row[3] == 21, (
        f"expected target delta at position 3, got {row[3]} (stale draft sample)"
    )
    # After K=2 steps + rollback, num_tokens_no_spec should be back at the
    # post-delta length (4), not the inflated post-K-step value.
    assert fn._input_batch.num_tokens_no_spec[0] == 4
