#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-request speculative acceptance measurement via vLLM /metrics delta.

Usage:
  python tools/spec_request_acceptance.py -p "tell a little story"
  python tools/spec_request_acceptance.py --no-memory-profile -p "hi"
  python tools/spec_request_acceptance.py --kv-peak-poll-interval 0.05 -p "long story"  # KV peak

Memory section (default ON, ``--no-memory-profile`` to disable) uses only HTTP:
``/metrics`` gauges ``vllm:kv_cache_usage_perc`` and ``vllm:cache_config_info``,
optional GET ``/server_info?config_format=json`` (vLLM exposes it only when
``VLLM_SERVER_DEV_MODE=1``). If ``kv_cache_memory_bytes`` is
missing, pool size is inferred from ``kv_cache_tensors[].size`` when present, else
``num_gpu_blocks`` × a page-size guess from attention fields in the JSON dump.

Model weight bytes are usually not exported; see printed notes.
``--metrics-sleep`` adjusts the post-response /metrics scrape; ``--kv-peak-poll-interval``
polls /metrics in a background thread during the blocking chat call to capture peak
``vllm:kv_cache_usage_perc`` (best-effort for KV occupancy during that request).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any


@dataclass
class CounterPair:
    accepted_key: str
    drafted_key: str
    accepted_value: float
    drafted_value: float


@dataclass
class GpuMemoryHints:
    """Best-effort memory snapshot from reachable HTTP endpoints (not NVIDIA NVML).

    KV usage is ``vllm:kv_cache_usage_perc`` (0–1 fraction of allocated blocks). When
    ``kv_cache_usage_frac_peak_request`` is set, a background poller sampled /metrics
    during the blocking chat request.

    KV used bytes are estimated when both pool size and usage fraction exist.
    Model parameter memory is rarely exported over HTTP — see notes.
    """

    kv_cache_usage_frac_before: float | None
    kv_cache_usage_frac_after: float | None
    kv_cache_usage_frac_peak_request: float | None
    kv_cache_pool_bytes: int | None
    kv_cache_pool_bytes_source: str | None
    kv_cache_used_bytes_est_before: int | None
    kv_cache_used_bytes_est_after: int | None
    kv_cache_used_bytes_est_peak_request: int | None
    cache_config_num_gpu_blocks: int | None
    cache_config_block_size: int | None
    model_weights_bytes_hint: int | None
    notes: list[str]


def _get_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_prometheus_counters(metrics_text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        # metric_name{labels} value
        # metric_name value
        parts = line.split()
        if len(parts) != 2:
            continue
        key, value = parts
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


def _normalize_metric_key(full_key: str) -> str:
    # strip labels so same metric with different labels can be grouped
    return full_key.split("{", 1)[0]


def _pick_spec_counter_pair(counters: dict[str, float]) -> CounterPair:
    grouped: dict[str, float] = {}
    for k, v in counters.items():
        grouped[_normalize_metric_key(k)] = grouped.get(_normalize_metric_key(k), 0.0) + v

    keys = list(grouped.keys())
    spec_keys = [k for k in keys if "spec" in k.lower() or "draft" in k.lower()]
    accepted_candidates = [
        k
        for k in spec_keys
        if "accept" in k.lower() and not k.lower().endswith(("rate", "ratio"))
    ]
    drafted_candidates = [
        k
        for k in spec_keys
        if "draft" in k.lower() and not k.lower().endswith(("rate", "ratio"))
    ]

    def _prefer_counter_metric(cands: list[str]) -> list[str]:
        # prioritize *_total style counters
        return sorted(
            cands,
            key=lambda x: (
                0 if x.endswith("_total") else 1,
                0 if "token" in x.lower() else 1,
                len(x),
            ),
        )

    accepted_sorted = _prefer_counter_metric(accepted_candidates)
    drafted_sorted = _prefer_counter_metric(drafted_candidates)
    if not accepted_sorted or not drafted_sorted:
        raise RuntimeError(
            "Cannot find speculative accepted/drafted counters in /metrics. "
            "Try opening /metrics manually and search for keywords: spec, draft, accepted."
        )

    accepted_key = accepted_sorted[0]
    drafted_key = drafted_sorted[0]
    return CounterPair(
        accepted_key=accepted_key,
        drafted_key=drafted_key,
        accepted_value=grouped[accepted_key],
        drafted_value=grouped[drafted_key],
    )


def _extract_text(resp: dict[str, Any]) -> str:
    try:
        return str(resp["choices"][0]["message"]["content"])
    except Exception:  # noqa: BLE001
        return json.dumps(resp, ensure_ascii=False)


def _format_bytes_simple(n: int | None) -> str:
    if n is None:
        return "N/A"
    x = float(n)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    if i == 0:
        return f"{int(n)} B"
    return f"{x:.4g} {units[i]}"


def _safe_int_maybe(s: str | Any) -> int | None:
    if s is None or s == "" or str(s).lower() in ("none", "null"):
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _deep_scan_kv_weights_bytes(root: Any) -> tuple[int | None, int | None]:
    """Return (maybe kv_pool_bytes_from_config, maybe model_weights_hint)."""
    kv_pool = None
    weights = None

    stack: list[Any] = [root]
    seen: set[int] = set()

    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if lk == "kv_cache_memory_bytes":
                    maybe = _safe_int_maybe(v)
                    if maybe is not None:
                        kv_pool = maybe
                elif "model_memory" in lk and "peak" not in lk and "activation" not in lk:
                    maybe_w = _safe_int_maybe(v)
                    if maybe_w is not None:
                        weights = maybe_w if weights is None else weights
                elif isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(obj, list):
            for it in obj:
                if isinstance(it, (dict, list)):
                    stack.append(it)
    return kv_pool, weights


def _deep_sum_kv_cache_tensor_sizes(root: Any) -> int | None:
    """Sum ``KVCacheTensor.size`` (bytes) if present in a JSON dump (runtime config)."""
    total = 0
    found = False
    stack: list[Any] = [root]
    seen: set[int] = set()
    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(obj, dict):
            sb = obj.get("shared_by")
            sz = obj.get("size")
            if isinstance(sb, list) and isinstance(sz, int) and sz >= 0:
                total += sz
                found = True
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(obj, list):
            for it in obj:
                if isinstance(it, (dict, list)):
                    stack.append(it)
    return total if found else None


def _torch_dtype_nbytes(obj: Any) -> int | None:
    """Best-effort element size for serialized torch dtype / string."""
    s = str(obj).lower()
    if any(x in s for x in ("bfloat16", "float16", "fp16")):
        return 2
    if "float32" in s or "fp32" in s:
        return 4
    if "float64" in s or "fp64" in s:
        return 8
    if "fp8" in s or "float8" in s or "e4m3" in s or "e5m2" in s:
        return 1
    if "int8" in s:
        return 1
    return None


def _try_attention_page_size_bytes(d: dict[str, Any]) -> int | None:
    """
    Match vLLM ``AttentionSpec.real_page_size_bytes`` (no per-token quant extras).

    Serialized ``dtype`` may be a string or nested; ``head_size_v`` may be absent.
    """
    try:
        nk = d.get("num_kv_heads")
        bk = d.get("block_size")
        hs = d.get("head_size_v")
        if hs is None:
            hs = d.get("head_size")
        dt = d.get("dtype")
        if nk is None or bk is None or hs is None or dt is None:
            return None
        nkh = _safe_int_maybe(nk)
        bki = _safe_int_maybe(bk)
        hsi = _safe_int_maybe(hs)
        if nkh is None or bki is None or hsi is None or nkh <= 0 or bki <= 0 or hsi <= 0:
            return None
        el = _torch_dtype_nbytes(dt)
        if el is None:
            return None
        return 2 * bki * nkh * hsi * el
    except (TypeError, KeyError):
        return None


def _collect_attention_page_size_bytes_candidates(root: Any) -> list[int]:
    out: list[int] = []
    stack: list[Any] = [root]
    seen: set[int] = set()
    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(obj, dict):
            psz = _try_attention_page_size_bytes(obj)
            if psz is not None:
                out.append(psz)
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(obj, list):
            for it in obj:
                if isinstance(it, (dict, list)):
                    stack.append(it)
    return out


def _infer_kv_pool_bytes(
    server_info_json: dict[str, Any] | None,
    n_blocks: int | None,
) -> tuple[int | None, str | None, list[str]]:
    """
    When Prometheus / cache_config does not expose ``kv_cache_memory_bytes``,
    try (1) sum of KVCacheTensor sizes, (2) num_gpu_blocks × derived page bytes.
    """
    notes: list[str] = []
    if server_info_json is None:
        return None, None, []

    tensor_sum = _deep_sum_kv_cache_tensor_sizes(server_info_json)
    if tensor_sum is not None and tensor_sum > 0:
        return (
            tensor_sum,
            "server_info: sum(kv_cache_tensors[].size)",
            notes,
        )

    if n_blocks is None or n_blocks <= 0:
        return None, None, notes

    pages = _collect_attention_page_size_bytes_candidates(server_info_json)
    uniq = sorted(set(pages))
    if not uniq:
        return None, None, notes

    if len(uniq) == 1:
        est = n_blocks * uniq[0]
        notes.append(
            "kv_cache_pool_bytes estimated as num_gpu_blocks × attention page_size_bytes "
            "(from server_info attention spec fields; ignores quant padding / multi-tensor layout)."
        )
        return est, "estimated: num_gpu_blocks × attention page_size_bytes", notes

    est = n_blocks * max(uniq)
    notes.append(
        "kv_cache_pool_bytes is a rough upper bound: multiple attention page_size "
        f"candidates {uniq} in server_info; using max page size × num_gpu_blocks."
    )
    return est, "estimated: num_gpu_blocks × max(page_size guesses)", notes


def _hints_from_prometheus_metrics(metrics_after: str) -> tuple[list[float], list[dict[str, str]]]:
    kv_fracs: list[float] = []
    cfg_labels_list: list[dict[str, str]] = []

    try:
        from prometheus_client.parser import text_string_to_metric_families
    except ImportError:
        for line in metrics_after.splitlines():
            if line.startswith("vllm:kv_cache_usage_perc"):
                parts = line.rsplit(maxsplit=1)
                if len(parts) != 2:
                    continue
                try:
                    kv_fracs.append(float(parts[1]))
                except ValueError:
                    continue
        return kv_fracs, cfg_labels_list

    for mf in text_string_to_metric_families(metrics_after):
        for s in mf.samples:
            nm = getattr(s, "name", "")
            lbl = dict(getattr(s, "labels", {}) or {})
            val = float(getattr(s, "value", 0))
            if nm == "vllm:kv_cache_usage_perc":
                kv_fracs.append(val)
            elif nm == "vllm:cache_config_info":
                cfg_labels_list.append(lbl)

    return kv_fracs, cfg_labels_list


def _choose_cache_labels(cfg_samples: list[dict[str, str]]) -> dict[str, str] | None:
    """Prefer cache_config labels for engine=0."""
    if not cfg_samples:
        return None
    for lbl in cfg_samples:
        if lbl.get("engine") == "0":
            return lbl
    return cfg_samples[0]


def gpu_memory_hints(
    metrics_before: str | None,
    metrics_after: str,
    server_info_json: dict[str, Any] | None,
    kv_usage_frac_peak_request: float | None,
) -> GpuMemoryHints:
    notes: list[str] = []

    kv_before_fracs: list[float] = []
    if metrics_before is not None:
        kv_before_fracs, _ = _hints_from_prometheus_metrics(metrics_before)
    kv_after_fracs, cfg_lbl_samples = _hints_from_prometheus_metrics(metrics_after)
    kv_before = max(kv_before_fracs) if kv_before_fracs else None
    kv_after = max(kv_after_fracs) if kv_after_fracs else None

    cfg_labels = _choose_cache_labels(cfg_lbl_samples)
    pool_from_prom = None
    n_blocks = None
    blk_sz = None
    if cfg_labels:
        pool_from_prom = _safe_int_maybe(cfg_labels.get("kv_cache_memory_bytes"))
        n_blocks = _safe_int_maybe(cfg_labels.get("num_gpu_blocks"))
        blk_sz = _safe_int_maybe(cfg_labels.get("block_size"))

    si_kv = None
    si_weights = None
    if server_info_json is not None:
        si_kv, si_weights = _deep_scan_kv_weights_bytes(server_info_json)

    pool: int | None = pool_from_prom if pool_from_prom is not None else si_kv
    pool_source: str | None = None
    if pool_from_prom is not None:
        pool_source = "prometheus: vllm:cache_config_info kv_cache_memory_bytes"
    elif si_kv is not None:
        pool_source = "server_info: cache_config.kv_cache_memory_bytes"

    if pool is None:
        inferred, src, infer_notes = _infer_kv_pool_bytes(server_info_json, n_blocks)
        notes.extend(infer_notes)
        if inferred is not None:
            pool = inferred
            pool_source = src

    if pool is None and cfg_labels is not None:
        notes.append(
            "kv_cache_pool_bytes unavailable (kv_cache_memory_bytes not in Prometheus "
            "cache_config labels; not in server_info; and could not infer from "
            "kv_cache_tensors or attention page fields)."
        )

    def _used(frac: float | None) -> int | None:
        if frac is not None and pool is not None:
            return max(0, int(round(frac * float(pool))))
        return None

    used_before = _used(kv_before)
    used_after = _used(kv_after)
    used_peak = _used(kv_usage_frac_peak_request)

    if kv_usage_frac_peak_request is None:
        notes.append(
            "Peak KV during inference: use --kv-peak-poll-interval (e.g. 0.05) to sample "
            "`vllm:kv_cache_usage_perc` while the chat request runs."
        )

    weights_hint = si_weights
    if weights_hint is None:
        notes.append(
            "model_weights_bytes_hint: not exported on /metrics. Use vLLM worker log "
            'line containing "Model loading took" or NVIDIA tools on the server host.'
        )

    return GpuMemoryHints(
        kv_cache_usage_frac_before=kv_before,
        kv_cache_usage_frac_after=kv_after,
        kv_cache_usage_frac_peak_request=kv_usage_frac_peak_request,
        kv_cache_pool_bytes=pool,
        kv_cache_pool_bytes_source=pool_source,
        kv_cache_used_bytes_est_before=used_before,
        kv_cache_used_bytes_est_after=used_after,
        kv_cache_used_bytes_est_peak_request=used_peak,
        cache_config_num_gpu_blocks=n_blocks,
        cache_config_block_size=blk_sz,
        model_weights_bytes_hint=weights_hint,
        notes=[n for n in notes if n],
    )


def _poll_kv_peak_during_chat(
    metrics_url: str, interval_sec: float, stop_event: threading.Event, peak_holder: list[float | None]
) -> None:
    """Background: scrape ``vllm:kv_cache_usage_perc`` until ``stop_event`` is set."""
    while not stop_event.is_set():
        try:
            txt = _get_text(metrics_url, timeout=min(10.0, 30.0))
            fracs, _ = _hints_from_prometheus_metrics(txt)
            if fracs:
                m = max(fracs)
                peak_holder[0] = m if peak_holder[0] is None else max(peak_holder[0], m)
        except (OSError, urllib.error.HTTPError, urllib.error.URLError, ValueError):
            pass
        if stop_event.wait(timeout=interval_sec):
            break


def _fetch_server_info_optional(
    scheme: str, host: str, port: int, timeout: float
) -> tuple[dict[str, Any] | None, str | None]:
    """Return (parsed JSON or None, optional user-facing note on failure)."""
    url = f"{scheme}://{host}:{port}/server_info?config_format=json"
    try:
        txt = _get_text(url, timeout)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, (
                "/server_info returned 404 — vLLM only mounts this route when "
                "VLLM_SERVER_DEV_MODE=1 in the server process (see "
                "entrypoints/serve/instrumentator/__init__.py). "
                "Without it, KV pool is inferred only from /metrics labels."
            )
        return None, f"/server_info HTTP {e.code}; skipping config JSON."
    except urllib.error.URLError as e:
        return None, f"/server_info unreachable ({e}); skipping config JSON."
    try:
        return json.loads(txt), None
    except json.JSONDecodeError:
        return None, "/server_info returned non-JSON; skipping config parse."


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure per-request speculative accepted/drafted token deltas."
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--https", action="store_true")
    ap.add_argument("-p", "--prompt", required=True)
    ap.add_argument("--model", default=None, help="Optional model field in request.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--show-output", action="store_true")
    ap.add_argument(
        "--memory-profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After the request, fetch /metrics (and optional /server_info) "
        "and print best-effort KV pool / used estimate and model-weight hints.",
    )
    ap.add_argument(
        "--metrics-sleep",
        type=float,
        default=0.3,
        help="Seconds to wait after the chat response before scraping /metrics (KV gauges).",
    )
    ap.add_argument(
        "--kv-peak-poll-interval",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="If > 0 with --memory-profile, periodically GET /metrics in a background "
        "thread while the chat request runs, and report max "
        "`vllm:kv_cache_usage_perc` (best-effort KV occupancy during inference). "
        "Example: 0.05 for 50ms polling.",
    )
    args = ap.parse_args()

    scheme = "https" if args.https else "http"
    chat_url = f"{scheme}://{args.host}:{args.port}/v1/chat/completions"
    metrics_url = f"{scheme}://{args.host}:{args.port}/metrics"

    try:
        before_text = _get_text(metrics_url, timeout=10)
    except urllib.error.URLError as e:
        raise SystemExit(f"Failed to read metrics endpoint: {e}") from e
    before_counters = _parse_prometheus_counters(before_text)
    before = _pick_spec_counter_pair(before_counters)

    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.model:
        payload["model"] = args.model

    peak_holder: list[float | None] = [None]
    stop_peak = threading.Event()
    poll_thr: threading.Thread | None = None
    poll_interval = args.kv_peak_poll_interval if args.memory_profile else 0.0
    if poll_interval > 0:
        poll_thr = threading.Thread(
            target=_poll_kv_peak_during_chat,
            args=(metrics_url, poll_interval, stop_peak, peak_holder),
            daemon=True,
            name="kv-metrics-poll",
        )
        poll_thr.start()

    t0 = time.time()
    try:
        try:
            resp = _get_json(chat_url, payload, timeout=args.timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise SystemExit(f"HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise SystemExit(f"Request failed: {e}") from e
    finally:
        stop_peak.set()
        if poll_thr is not None:
            poll_thr.join(timeout=5.0)
    peak_frac = peak_holder[0]
    elapsed = time.time() - t0

    # Give metrics exporter a brief moment to flush gauges (e.g. kv_cache_usage_perc).
    time.sleep(max(0.0, args.metrics_sleep))
    after_text = _get_text(metrics_url, timeout=10)
    after_counters = _parse_prometheus_counters(after_text)
    after = _pick_spec_counter_pair(after_counters)

    # Allow metric key renaming edge case across snapshot.
    if before.accepted_key != after.accepted_key or before.drafted_key != after.drafted_key:
        print(
            "Warning: detected different metric keys between snapshots.",
            file=sys.stderr,
        )
        print(
            f"before accepted={before.accepted_key}, drafted={before.drafted_key}",
            file=sys.stderr,
        )
        print(
            f"after  accepted={after.accepted_key}, drafted={after.drafted_key}",
            file=sys.stderr,
        )

    accepted_delta = max(0.0, after.accepted_value - before.accepted_value)
    drafted_delta = max(0.0, after.drafted_value - before.drafted_value)
    acceptance = (accepted_delta / drafted_delta * 100.0) if drafted_delta > 0 else 0.0

    print(f"request_latency_s: {elapsed:.3f}")
    print(f"accepted_tokens_delta: {int(round(accepted_delta))}")
    print(f"drafted_tokens_delta: {int(round(drafted_delta))}")
    print(f"acceptance_rate_pct: {acceptance:.2f}")
    print(f"accepted_metric: {after.accepted_key}")
    print(f"drafted_metric: {after.drafted_key}")

    if args.memory_profile:
        si_json, si_note = _fetch_server_info_optional(
            scheme, args.host, args.port, min(10.0, args.timeout)
        )
        h = gpu_memory_hints(before_text, after_text, si_json, peak_frac)
        if si_note:
            h = replace(h, notes=[si_note, *h.notes])
        print("\n=== gpu_memory_profile (HTTP endpoints, best-effort) ===")
        print(
            "kv_cache_usage_perc (vllm:kv_cache_usage_perc, 0–100; engine max if multi):"
        )
        if h.kv_cache_usage_frac_before is not None:
            print(
                "  before_request:",
                f"{h.kv_cache_usage_frac_before * 100:.4f}%",
            )
        else:
            print("  before_request: N/A")
        if h.kv_cache_usage_frac_after is not None:
            print(
                "  after_request (after --metrics-sleep):",
                f"{h.kv_cache_usage_frac_after * 100:.4f}%",
            )
        else:
            print("  after_request (after --metrics-sleep): N/A")
        if h.kv_cache_usage_frac_peak_request is not None:
            print(
                "  peak_during_chat_request (--kv-peak-poll-interval):",
                f"{h.kv_cache_usage_frac_peak_request * 100:.4f}%",
            )
        else:
            print(
                "  peak_during_chat_request: N/A (set --kv-peak-poll-interval e.g. 0.05)"
            )
        print(
            "kv_cache_pool_bytes:",
            h.kv_cache_pool_bytes if h.kv_cache_pool_bytes is not None else "N/A",
            f"({_format_bytes_simple(h.kv_cache_pool_bytes)})",
        )
        print(
            "kv_cache_pool_bytes_source:",
            h.kv_cache_pool_bytes_source if h.kv_cache_pool_bytes_source else "N/A",
        )
        print(
            "kv_cache_used_bytes_est (usage_frac × pool, when pool known):"
        )
        print(
            "  before_request:",
            h.kv_cache_used_bytes_est_before
            if h.kv_cache_used_bytes_est_before is not None
            else "N/A",
            f"({_format_bytes_simple(h.kv_cache_used_bytes_est_before)})",
        )
        print(
            "  after_request:",
            h.kv_cache_used_bytes_est_after
            if h.kv_cache_used_bytes_est_after is not None
            else "N/A",
            f"({_format_bytes_simple(h.kv_cache_used_bytes_est_after)})",
        )
        print(
            "  peak_during_chat_request:",
            h.kv_cache_used_bytes_est_peak_request
            if h.kv_cache_used_bytes_est_peak_request is not None
            else "N/A",
            f"({_format_bytes_simple(h.kv_cache_used_bytes_est_peak_request)})",
        )
        print(
            "cache_config num_gpu_blocks, block_size:",
            h.cache_config_num_gpu_blocks,
            h.cache_config_block_size,
        )
        print(
            "model_weights_bytes_hint:",
            h.model_weights_bytes_hint if h.model_weights_bytes_hint is not None else "N/A",
            f"({_format_bytes_simple(h.model_weights_bytes_hint)})",
        )
        if h.notes:
            print("notes:")
            for n in h.notes:
                print(f"  - {n}")

    if args.show_output:
        print("\n=== model_output ===")
        print(_extract_text(resp))


if __name__ == "__main__":
    main()
