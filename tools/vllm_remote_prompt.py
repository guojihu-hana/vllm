#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
向远端 vLLM OpenAI 兼容 HTTP 服务发 chat 请求。

用法示例:
  # 改脚本里 DEFAULT_MESSAGES / DEFAULT_HOST 后直接运行
  python tools/vllm_remote_prompt.py

  # 命令行覆盖
  python tools/vllm_remote_prompt.py -p "用三句话解释 speculative decoding"

  # 从文件读用户消息（整文件作为一条 user 内容）
  python tools/vllm_remote_prompt.py --prompt-file my_prompt.txt

  # 指定模型与采样
  python tools/vllm_remote_prompt.py --model Qwen/Qwen3-4B -p "1+1=?"

环境变量（可选）:
  VLLM_REMOTE_HOST, VLLM_REMOTE_PORT, VLLM_REMOTE_MODEL
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 在这里改默认配置最省事（命令行参数会覆盖）
# ---------------------------------------------------------------------------
DEFAULT_HOST = os.environ.get("VLLM_REMOTE_HOST", "100.101.93.30")
DEFAULT_PORT = int(os.environ.get("VLLM_REMOTE_PORT", "8000"))
DEFAULT_MODEL = os.environ.get("VLLM_REMOTE_MODEL") or ""

DEFAULT_MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "你好，用一句话介绍你自己。"},
]

DEFAULT_MAX_TOKENS = 3072
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = -1


def _chat_url(host: str, port: int, use_https: bool) -> str:
    scheme = "https" if use_https else "http"
    return f"{scheme}://{host}:{port}/v1/chat/completions"


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {e.code}: {err_body}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"请求失败: {e}") from e
    return json.loads(body)


def _build_messages(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.prompt_file is not None:
        text = Path(args.prompt_file).read_text(encoding="utf-8")
        return [{"role": "user", "content": text}]
    if args.prompt is not None:
        return [{"role": "user", "content": args.prompt}]
    return list(DEFAULT_MESSAGES)


def main() -> None:
    p = argparse.ArgumentParser(description="向远端 vLLM 发送 chat 请求（OpenAI 兼容 API）")
    p.add_argument("--host", default=DEFAULT_HOST, help="vLLM 服务 IP 或主机名")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="端口，默认 8000")
    p.add_argument("--https", action="store_true", help="使用 https")
    p.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL or None,
        help="模型名；省略则不在 JSON 里带 model（单模型服务通常可用）",
    )
    p.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="单条 user 消息；不设则用脚本里的 DEFAULT_MESSAGES",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="从文件读入整段文本作为 user 消息",
    )
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="HTTP 超时（秒）",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="打印完整 JSON 响应",
    )
    args = p.parse_args()

    url = _chat_url(args.host, args.port, args.https)
    messages = _build_messages(args)

    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.model:
        payload["model"] = args.model

    result = _post_json(url, payload, args.timeout)

    if args.raw:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    try:
        choice = result["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "")
    except (KeyError, IndexError, TypeError) as e:
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(f"无法解析响应: {e}") from e

    print(content, end="" if content.endswith("\n") else "\n")


if __name__ == "__main__":
    main()
