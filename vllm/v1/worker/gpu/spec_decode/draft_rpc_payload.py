# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Msgpack-friendly serialization for DraftModel parity RPC.

Two protocol versions coexist:

- ``draft_propose_v1``: stateless. Each RPC carries the full per-request token
  history in ``context_token_ids`` (or full target tensors for EAGLE parity).
  Bandwidth is O(seq_len × batch) per step.
- ``draft_propose_v2``: session-aware. Target maintains a per-request session
  id; first call sends the full prompt, subsequent calls send only the
  ``tokens`` accepted/sampled since the previous step. The remote server
  reconstructs the full sequence and relies on vLLM prefix caching for KV
  reuse, so per-step bandwidth and prefill compute drop to O(K_accepted+1).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata

DRAFT_PROPOSE_V1 = "draft_propose_v1"
DRAFT_PROPOSE_V2 = "draft_propose_v2"


@dataclass
class DraftSessionEntry:
    """One per-request entry in a ``draft_propose_v2`` payload.

    ``is_first`` marks a fresh session (or one the server may have evicted):
    ``tokens`` is then the complete prompt + first sampled token. Otherwise
    ``tokens`` is the delta accepted/sampled by the target since the previous
    step (length == accepted_count + 1 in the common case).
    """

    id: str
    is_first: bool
    tokens: list[int]


def build_draft_propose_v2_payload(
    *,
    num_speculative_tokens: int,
    sessions: list[DraftSessionEntry],
    evict: list[str] | None = None,
) -> dict[str, Any]:
    """Build a ``draft_propose_v2`` RPC payload."""
    payload: dict[str, Any] = {
        "rpc_schema": DRAFT_PROPOSE_V2,
        "num_speculative_tokens": int(num_speculative_tokens),
        "sessions": [
            {"id": s.id, "is_first": bool(s.is_first), "tokens": list(s.tokens)}
            for s in sessions
        ],
    }
    if evict:
        payload["evict"] = list(evict)
    return payload


def parse_draft_propose_v2_payload(
    req: dict[str, Any],
) -> tuple[int, list[DraftSessionEntry], list[str]]:
    """Parse a ``draft_propose_v2`` RPC payload.

    Returns ``(num_speculative_tokens, sessions, evict_ids)``.
    """
    if req.get("rpc_schema") != DRAFT_PROPOSE_V2:
        raise ValueError(
            f"Expected rpc_schema={DRAFT_PROPOSE_V2!r}, got "
            f"{req.get('rpc_schema')!r}"
        )
    raw_sessions = req.get("sessions") or []
    sessions: list[DraftSessionEntry] = []
    for s in raw_sessions:
        sessions.append(
            DraftSessionEntry(
                id=str(s["id"]),
                is_first=bool(s.get("is_first", False)),
                tokens=[int(t) for t in (s.get("tokens") or [])],
            )
        )
    evict = [str(x) for x in (req.get("evict") or [])]
    return int(req["num_speculative_tokens"]), sessions, evict


def _dtype_to_str(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def _str_to_dtype(name: str) -> torch.dtype:
    key = name.replace("torch.", "")
    if key == "bfloat16":
        return torch.bfloat16
    return getattr(torch, key)


def tensor_chunk_to_payload(tensor: torch.Tensor) -> dict[str, Any]:
    """Serialize a CPU tensor to msgpack-friendly dict."""
    t = tensor.detach().contiguous().cpu()
    dt = t.dtype
    # NumPy / torch.numpy() do not support bfloat16; match tensor_chunk_from_payload
    # (uint16 bit pattern round-trip).
    if dt == torch.bfloat16:
        raw = t.view(torch.uint16).numpy().tobytes()
    else:
        raw = t.numpy().tobytes()
    return {
        "shape": list(t.shape),
        "dtype": _dtype_to_str(dt),
        "data": raw,
    }


def tensor_chunk_from_payload(payload: dict[str, Any], device: torch.device) -> torch.Tensor:
    dt = _str_to_dtype(payload["dtype"])
    raw = bytes(payload["data"])
    shape = tuple(payload["shape"])
    numel = prod(shape) if shape else 0
    if dt == torch.bfloat16:
        u16 = torch.frombuffer(bytearray(raw), dtype=torch.uint16, count=numel)
        t = u16.view(torch.bfloat16).reshape(shape).clone()
    else:
        t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(shape).clone()
    return t.to(device=device)


def portable_common_attn_to_payload(cad: CommonAttentionMetadata) -> dict[str, Any]:
    """Portable fields only (no block_table / slot_mapping)."""
    out: dict[str, Any] = {
        "query_start_loc": tensor_chunk_to_payload(cad.query_start_loc.cpu()),
        "query_start_loc_cpu": tensor_chunk_to_payload(cad.query_start_loc_cpu.cpu()),
        "seq_lens": tensor_chunk_to_payload(cad.seq_lens.cpu()),
        "num_reqs": cad.num_reqs,
        "num_actual_tokens": cad.num_actual_tokens,
        "max_query_len": cad.max_query_len,
        "max_seq_len": cad.max_seq_len,
        "causal": cad.causal,
    }
    if cad.dcp_local_seq_lens is not None:
        out["dcp_local_seq_lens"] = tensor_chunk_to_payload(cad.dcp_local_seq_lens.cpu())
    if cad.encoder_seq_lens is not None:
        out["encoder_seq_lens"] = tensor_chunk_to_payload(cad.encoder_seq_lens.cpu())
    if cad._seq_lens_cpu is not None:
        out["_seq_lens_cpu"] = tensor_chunk_to_payload(cad._seq_lens_cpu.cpu())
    if cad._num_computed_tokens_cpu is not None:
        out["_num_computed_tokens_cpu"] = tensor_chunk_to_payload(
            cad._num_computed_tokens_cpu.cpu()
        )
    return out


def _allocate_block_tables_for_remote(
    seq_lens: torch.Tensor,
    max_blocks_row: int,
    num_kv_blocks: int,
    block_size: int,
    device: torch.device,
    *,
    num_speculative_tokens: int = 0,
) -> torch.Tensor:
    """Give each request a disjoint range of physical KV block ids in [0, num_kv_blocks)."""
    batch_size = seq_lens.shape[0]
    block_table_tensor = torch.zeros(
        batch_size, max_blocks_row, dtype=torch.int32, device=device
    )
    next_free = 0
    for i in range(batch_size):
        sl = int(seq_lens[i].item())
        # SpecDecodeBasePropose.propose() inner loop advances positions up to K-1 extra
        # decode steps; block indices must exist for pos // block_size throughout.
        span = sl + max(0, int(num_speculative_tokens))
        need = (span + block_size - 1) // block_size
        need = min(need, max_blocks_row)
        if need == 0:
            continue
        if next_free + need > num_kv_blocks:
            raise RuntimeError(
                "Remote draft worker KV pool is too small for this batch "
                f"(need physical blocks up to {next_free + need}, "
                f"num_kv_blocks={num_kv_blocks}). "
                "Raise VLLM_REMOTE_DRAFT_KV_MEMORY_BYTES or reduce load."
            )
        block_table_tensor[i, :need] = torch.arange(
            next_free, next_free + need, dtype=torch.int32, device=device
        )
        next_free += need
    return block_table_tensor


def _slot_mapping_from_block_table(
    query_start_loc: torch.Tensor,
    positions_1d: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Match BlockTable._compute_slot_mapping_kernel for decode_context_parallel_size=1."""
    num_tokens = positions_1d.shape[0]
    device = positions_1d.device
    q_hi = query_start_loc[1:].to(torch.int64)
    t_ix = torch.arange(num_tokens, device=device, dtype=torch.int64)
    req = torch.searchsorted(q_hi, t_ix, right=True)
    pos = positions_1d.to(torch.int64)
    max_blk_ix = block_table.shape[1] - 1
    blk_ix = (pos // block_size).clamp(min=0, max=max_blk_ix)
    phys = block_table[req, blk_ix].to(torch.int64)
    return phys * block_size + (pos % block_size)


def rebuild_common_attn_metadata(
    portable: dict[str, Any],
    device: torch.device,
    block_size: int,
    max_block_idx: int = 10000,
    *,
    positions: torch.Tensor | None = None,
    num_kv_blocks: int | None = None,
    num_speculative_tokens: int = 0,
) -> CommonAttentionMetadata:
    """Rebuild CommonAttentionMetadata with fresh block_table and slot_mapping.

    When ``positions`` and ``num_kv_blocks`` are set (native remote draft worker),
    block_table and slot_mapping are chosen to stay within the allocated KV pool.

    Otherwise falls back to random tensors (unit tests / legacy).
    """
    query_start_loc = tensor_chunk_from_payload(portable["query_start_loc"], device).to(
        torch.int32
    )
    query_start_loc_cpu = tensor_chunk_from_payload(
        portable["query_start_loc_cpu"], torch.device("cpu")
    ).to(torch.int32)
    seq_lens = tensor_chunk_from_payload(portable["seq_lens"], device).to(torch.int32)

    batch_size = int(seq_lens.shape[0])
    query_lens = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.int32)
    num_tokens = int(query_lens.sum().item())

    max_seq_len = int(portable["max_seq_len"])
    max_query_len = int(portable["max_query_len"])

    context_lens = [
        int(seq_lens[i].item()) - int(query_lens[i].item()) for i in range(batch_size)
    ]
    num_computed_tokens_cpu = torch.tensor(context_lens, dtype=torch.int32)

    # block_table row width must cover EAGLE inner steps: positions grow by up to
    # num_speculative_tokens after the first pass. portable max_seq_len often
    # equals current max(seq_lens) only; without this, need is clamped down and
    # tail block_table columns stay zero → bad slot_mapping → CUDA/CUBLAS errors.
    sl_max = int(seq_lens.max().item()) if seq_lens.numel() else max_seq_len
    planned_max_seq = max(max_seq_len, sl_max) + max(0, int(num_speculative_tokens))
    max_blocks = (planned_max_seq + block_size - 1) // block_size
    if positions is not None and num_kv_blocks is not None:
        pos_1d = positions[0] if positions.ndim > 1 else positions
        if int(pos_1d.shape[0]) != num_tokens:
            raise ValueError(
                f"positions length {pos_1d.shape[0]} != num_tokens {num_tokens} "
                "from common_attn_metadata"
            )
        block_table_tensor = _allocate_block_tables_for_remote(
            seq_lens,
            max_blocks,
            num_kv_blocks,
            block_size,
            device,
            num_speculative_tokens=num_speculative_tokens,
        )
        slot_mapping = _slot_mapping_from_block_table(
            query_start_loc, pos_1d, block_table_tensor, block_size
        )
    else:
        block_table_tensor = torch.randint(
            0,
            max_block_idx,
            (batch_size, max_blocks),
            dtype=torch.int32,
            device=device,
        )
        slot_mapping = torch.randint(
            0, max_block_idx, (num_tokens,), dtype=torch.int64, device=device
        )

    dcp = None
    if portable.get("dcp_local_seq_lens") is not None:
        dcp = tensor_chunk_from_payload(portable["dcp_local_seq_lens"], device).to(
            torch.int32
        )

    enc_gpu = None
    enc_cpu = None
    if portable.get("encoder_seq_lens") is not None:
        enc_gpu = tensor_chunk_from_payload(portable["encoder_seq_lens"], device).to(
            torch.int32
        )

    _seq_cpu = None
    if portable.get("_seq_lens_cpu") is not None:
        _seq_cpu = tensor_chunk_from_payload(portable["_seq_lens_cpu"], torch.device("cpu"))

    _nct_cpu = None
    if portable.get("_num_computed_tokens_cpu") is not None:
        _nct_cpu = tensor_chunk_from_payload(
            portable["_num_computed_tokens_cpu"], torch.device("cpu")
        )

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        _seq_lens_cpu=_seq_cpu,
        _num_computed_tokens_cpu=_nct_cpu,
        num_reqs=int(portable["num_reqs"]),
        num_actual_tokens=int(portable["num_actual_tokens"]),
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=bool(portable.get("causal", True)),
        dcp_local_seq_lens=dcp,
        encoder_seq_lens=enc_gpu,
        encoder_seq_lens_cpu=None,
    )


def build_draft_propose_v1_payload(
    *,
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    next_token_ids: torch.Tensor,
    token_indices_to_sample: torch.Tensor | None,
    common_attn_metadata: CommonAttentionMetadata,
    num_rejected_tokens_gpu: torch.Tensor | None,
    num_speculative_tokens: int,
    context_token_ids: list[list[int]] | None = None,
    include_target_hidden_states: bool = True,
    omit_unused_for_greedy: bool = False,
) -> dict[str, Any]:
    """Build ``draft_propose_v1`` RPC payload.

    When ``include_target_hidden_states`` is False, the target LM hidden-state
    tensor is not serialized (saves bandwidth and avoids D2H on the caller).
    Remote workers must deserialize with ``omit_target_hs_fill_hidden_size``.
    Typical use case: ``SpeculativeConfig.method == \"draft_model\"`` —
    ``DraftModelProposer`` does not consume target hidden states in the draft
    forward (``pass_hidden_states_to_model=False``); EAGLE-style methods keep
    the default ``True``.

    When ``omit_unused_for_greedy`` is True, also drops fields that the greedy
    replay backends (``VLLMGreedyDraftFn`` and ``DraftModelNativeParityFn``
    without ``VLLM_REMOTE_DRAFT_USE_EAGLE_PARITY=1``) never read:
    ``target_token_ids``, ``target_positions``, ``common_attn_metadata``,
    ``token_indices_to_sample``, ``num_rejected_tokens_gpu``, and
    ``next_token_ids``. Each one of those forces a D2H sync + msgpack encode on
    the target's critical path, so skipping them removes both bandwidth and
    latency overhead. The caller is responsible for ensuring each
    ``context_token_ids[i]`` already ends with the desired starting token —
    server-side greedy backends derive ``next_token_id`` from
    ``context_token_ids[i][-1]``. The native EAGLE parity backend rejects
    payloads built with this flag.
    """
    payload: dict[str, Any] = {
        "rpc_schema": DRAFT_PROPOSE_V1,
        "num_speculative_tokens": num_speculative_tokens,
    }
    if not omit_unused_for_greedy:
        payload["next_token_ids"] = tensor_chunk_to_payload(next_token_ids.cpu())
        payload["target_token_ids"] = tensor_chunk_to_payload(target_token_ids)
        payload["target_positions"] = tensor_chunk_to_payload(target_positions)
        payload["common_attn_metadata"] = portable_common_attn_to_payload(
            common_attn_metadata
        )
        if token_indices_to_sample is not None:
            payload["token_indices_to_sample"] = tensor_chunk_to_payload(
                token_indices_to_sample.cpu()
            )
        if num_rejected_tokens_gpu is not None:
            payload["num_rejected_tokens_gpu"] = tensor_chunk_to_payload(
                num_rejected_tokens_gpu.cpu()
            )
    # ``DraftModelProposer`` does not pass hidden states into the draft forward
    # (pass_hidden_states_to_model=False). For remote draft_model mode, omitting this
    # tensor avoids enormous CPU/sync + Msgpack payloads (target LM hidden dim × tokens).
    if include_target_hidden_states:
        payload["target_hidden_states"] = tensor_chunk_to_payload(target_hidden_states.cpu())
    if context_token_ids is not None:
        payload["context_token_ids"] = context_token_ids
    return payload


@dataclass
class DraftProposeV1Deserialized:
    target_token_ids: torch.Tensor
    target_positions: torch.Tensor
    target_hidden_states: torch.Tensor
    next_token_ids: torch.Tensor
    token_indices_to_sample: torch.Tensor | None
    common_attn_metadata: CommonAttentionMetadata
    num_rejected_tokens_gpu: torch.Tensor | None
    num_speculative_tokens: int
    context_token_ids: list[list[int]] | None = None


def deserialize_draft_propose_v1(
    req: dict[str, Any],
    device: torch.device,
    block_size: int,
    *,
    num_kv_blocks: int | None = None,
    omit_target_hs_fill_hidden_size: int | None = None,
    omit_target_hs_dtype: torch.dtype | None = None,
) -> DraftProposeV1Deserialized:
    if "target_token_ids" not in req or "common_attn_metadata" not in req:
        raise ValueError(
            "draft_propose_v1 payload was built with omit_unused_for_greedy=True "
            "(target_token_ids / common_attn_metadata stripped). EAGLE parity "
            "deserialization needs them — either run the server with the greedy "
            "backend (default) or have the caller pass omit_unused_for_greedy=False."
        )

    token_indices = None
    if req.get("token_indices_to_sample") is not None:
        token_indices = tensor_chunk_from_payload(
            req["token_indices_to_sample"], device
        ).to(torch.int32)

    num_rejected = None
    if req.get("num_rejected_tokens_gpu") is not None:
        num_rejected = tensor_chunk_from_payload(
            req["num_rejected_tokens_gpu"], device
        ).to(torch.int32)

    target_token_ids = tensor_chunk_from_payload(req["target_token_ids"], device).to(
        torch.int32
    )

    target_positions = tensor_chunk_from_payload(req["target_positions"], device).to(
        torch.int64
    )
    num_spec = int(req["num_speculative_tokens"])
    ctx_raw = req.get("context_token_ids")
    context_token_ids: list[list[int]] | None = None
    if ctx_raw is not None:
        context_token_ids = [[int(x) for x in row] for row in ctx_raw]

    cad = rebuild_common_attn_metadata(
        req["common_attn_metadata"],
        device,
        block_size=block_size,
        positions=target_positions,
        num_kv_blocks=num_kv_blocks,
        num_speculative_tokens=num_spec,
    )

    th_payload = req.get("target_hidden_states")
    num_tokens_rows = int(target_token_ids.shape[0])
    if th_payload is not None:
        target_hidden_states = tensor_chunk_from_payload(th_payload, device)
    else:
        if omit_target_hs_fill_hidden_size is None:
            raise ValueError(
                "draft_propose_v1 payload omitted target_hidden_states; pass "
                "omit_target_hs_fill_hidden_size (+ optional omit_target_hs_dtype) "
                "to deserialize."
            )
        dt_fill = omit_target_hs_dtype if omit_target_hs_dtype is not None else torch.float16
        target_hidden_states = torch.zeros(
            (num_tokens_rows, omit_target_hs_fill_hidden_size),
            dtype=dt_fill,
            device=device,
        )

    return DraftProposeV1Deserialized(
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=tensor_chunk_from_payload(req["next_token_ids"], device).to(
            torch.int32
        ),
        token_indices_to_sample=token_indices,
        common_attn_metadata=cad,
        num_rejected_tokens_gpu=num_rejected,
        num_speculative_tokens=num_spec,
        context_token_ids=context_token_ids,
    )
