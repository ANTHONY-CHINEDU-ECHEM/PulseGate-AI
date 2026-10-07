"""Blink detection from the eye aspect ratio."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlinkEvent:
    start: float
    end: float
    depth: float          # lowest eye opening during the blink, relative to the open eye baseline

    @property
    def duration(self) -> float:
        return self.end - self.start


class BlinkDetector:
    """Adaptive threshold blink detector.

    Eye aspect ratios differ between people, glasses and camera angles, so the
    detector tracks each user's own open eye level (a high percentile of the
    recent past) and looks for short dips below a fraction of it. Dips that last
    too long are treated as squinting, looking down or occlusion, not as blinks.
    """

    def __init__(self, close_ratio: float = 0.72, open_ratio: float = 0.85, min_closed_s: float = 0.04,
                 max_closed_s: float = 0.6, baseline_window_s: float = 4.0, max_abs_yaw: float = 30.0, max_abs_pitch: float = 25.0):
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.min_closed_s = min_closed_s
        self.max_closed_s = max_closed_s
        self.baseline_window_s = baseline_window_s
        self.max_abs_yaw = max_abs_yaw
        self.max_abs_pitch = max_abs_pitch
        self.reset()

    @classmethod
    def from_config(cls, cfg) -> "BlinkDetector":
        b = cfg.blink
        return cls(b.close_ratio, b.open_ratio, b.min_closed_s, b.max_closed_s, b.baseline_window_s)

    def reset(self) -> None:
        self._history: deque[tuple[float, float]] = deque()
        self._closed_since: float | None = None
        self._last_open_t: float | None = None
        self._min_ratio = 1.0
        self.events: list[BlinkEvent] = []
        self.baseline: float | None = None
        self.ratio: float = 1.0

    @property
    def count(self) -> int:
        return len(self.events)

    def count_since(self, t: float) -> int:
        return sum(1 for e in self.events if e.start >= t)

    def update(self, t: float, ear: float, yaw: float = 0.0, pitch: float = 0.0) -> BlinkEvent | None:
        """Feed one frame. Returns a ``BlinkEvent`` on the frame where a blink completes."""
        if abs(yaw) > self.max_abs_yaw or abs(pitch) > self.max_abs_pitch or not np.isfinite(ear):
            self._closed_since = None
            return None
        self._history.append((t, ear))
        while self._history and t - self._history[0][0] > self.baseline_window_s:
            self._history.popleft()
        if len(self._history) < 5:
            self._last_open_t = t
            return None
        self.baseline = float(np.percentile([e for _, e in self._history], 85))
        if self.baseline < 1e-3:
            return None
        self.ratio = ear / self.baseline
        event = None
        if self._closed_since is None:
            if self.ratio < self.close_ratio:
                # the eye started closing somewhere between the last open frame and this one
                self._closed_since = self._last_open_t if self._last_open_t is not None else t
                self._min_ratio = self.ratio
            else:
                self._last_open_t = t
        else:
            self._min_ratio = min(self._min_ratio, self.ratio)
            if self.ratio > self.open_ratio:
                duration = t - self._closed_since
                if self.min_closed_s <= duration <= self.max_closed_s:
                    event = BlinkEvent(self._closed_since, t, float(self._min_ratio))
                    self.events.append(event)
                self._closed_since = None
                self._last_open_t = t
            elif t - self._closed_since > 2.0 * self.max_closed_s:
                self._closed_since = None       # long closure, start over once the eye reopens
        return event
