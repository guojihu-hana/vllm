# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def cuda_timer(
    *,
    enabled: bool,
    start_event: torch.cuda.Event | None,
    end_event: torch.cuda.Event | None,
) -> Iterator[list[float]]:
    """CUDA event timer context manager.

    Usage:
        with cuda_timer(enabled=..., start_event=..., end_event=...) as elapsed_ms:
            ...
        measured = elapsed_ms[0]

    When ``enabled`` is False, it still yields and returns ``0.0`` without timing.
    """
    elapsed_ms: list[float] = [0.0]
    if enabled:
        assert start_event is not None
        assert end_event is not None
        start_event.record()
    try:
        yield elapsed_ms
    finally:
        if enabled:
            assert start_event is not None
            assert end_event is not None
            end_event.record()
            end_event.synchronize()
            elapsed_ms[0] = float(start_event.elapsed_time(end_event))


class SpecDecodeTimingTracker:
    """Stateful CUDA-event timing tracker for spec decode target/draft paths."""

    def __init__(
        self,
        *,
        enabled: bool,
        log_interval: int,
        log_fn,
    ) -> None:
        self.enabled = enabled
        self.log_interval = log_interval
        self._log_fn = log_fn

        self._calls = 0
        self._window_size = 50
        self._target_recent: deque[float] = deque(maxlen=self._window_size)
        self._draft_recent: deque[float] = deque(maxlen=self._window_size)
        self._last_target_ms = 0.0

        self._target_start_event = (
            torch.cuda.Event(enable_timing=True) if enabled else None
        )
        self._target_end_event = (
            torch.cuda.Event(enable_timing=True) if enabled else None
        )
        self._draft_start_event = (
            torch.cuda.Event(enable_timing=True) if enabled else None
        )
        self._draft_end_event = (
            torch.cuda.Event(enable_timing=True) if enabled else None
        )

    @contextmanager
    def target_timer(self, *, enabled: bool) -> Iterator[list[float]]:
        with cuda_timer(
            enabled=self.enabled and enabled,
            start_event=self._target_start_event,
            end_event=self._target_end_event,
        ) as elapsed_ms:
            yield elapsed_ms
        if self.enabled and enabled:
            self._last_target_ms = elapsed_ms[0]

    @contextmanager
    def draft_timer(self) -> Iterator[list[float]]:
        with cuda_timer(
            enabled=self.enabled,
            start_event=self._draft_start_event,
            end_event=self._draft_end_event,
        ) as elapsed_ms:
            yield elapsed_ms
        self.observe_draft(elapsed_ms[0])

    def observe_draft(self, draft_ms: float) -> None:
        if not self.enabled:
            return
        self._calls += 1
        self._draft_recent.append(draft_ms)
        self._target_recent.append(self._last_target_ms)
        if self._calls % self.log_interval != 0:
            return
        n = len(self._target_recent)
        if n == 0:
            return
        target_avg = sum(self._target_recent) / n
        draft_avg = sum(self._draft_recent) / n
        self._log_fn(
            "Target avg: %.3f ms (recent %d calls); "
            "Draft avg: %.3f ms (recent %d calls)",
            target_avg,
            n,
            draft_avg,
            n,
        )
