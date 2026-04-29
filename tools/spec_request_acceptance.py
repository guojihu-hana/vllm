#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-request speculative acceptance measurement via vLLM /metrics delta.

Usage:
  python tools/spec_request_acceptance.py -p "tell a little story"

This script:
1) snapshots Prometheus metrics before request
2) sends one chat completion request
3) snapshots metrics after request
4) reports per-request drafted/accepted token deltas
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class CounterPair:
    accepted_key: str
    drafted_key: str
    accepted_value: float
    drafted_value: float


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

    t0 = time.time()
    try:
        resp = _get_json(chat_url, payload, timeout=args.timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"Request failed: {e}") from e
    elapsed = time.time() - t0

    # Give metrics exporter a brief moment to flush counters.
    time.sleep(0.3)
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

    if args.show_output:
        print("\n=== model_output ===")
        print(_extract_text(resp))


if __name__ == "__main__":
    main()
