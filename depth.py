"""Depth from motion: is the face a rigid 3D object or a flat surface?

When a real head turns, the nose moves differently from the cheeks and ears,
so the landmark motion between two poses cannot be explained by a single plane
to plane mapping (a homography). When a photograph or a screen is tilted, it
can. The residual of the best homography between two landmark sets, measured
in units of the eye distance, is therefore a direct test for planarity.

The cue only speaks about static media. A replayed video of a real head turn
shows genuine 3D motion and passes it, which is why it is one signal of several.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..vision.geometry import RIGID_POINTS, interocular_distance


@dataclass(frozen=True)
class DepthResult:
    valid: bool                 # enough head rotation was observed to judge
    residual: float             # median non planar residual over the compared pose pairs
    pairs: int
    max_rotation_deg: float
    score: float                # 0 flat, 1 clearly three dimensional

    def as_dict(self) -> dict:
        return {
            "valid": self.valid, "residual": round(self.residual, 4), "pairs": self.pairs,
            "max_rotation_deg": round(self.max_rotation_deg, 1), "score": round(self.score, 3),
        }


def planar_residual(landmarks_a: np.ndarray, landmarks_b: np.ndarray, points: tuple[int, ...] = RIGID_POINTS) -> float:
    """RMS error of the best homography from pose A to pose B, in eye distances of B."""
    idx = list(points)
    a = np.asarray(landmarks_a)[idx, :2].astype(np.float32)
    b = np.asarray(landmarks_b)[idx, :2].astype(np.float32)
    homography, _ = cv2.findHomography(a, b, 0)
    if homography is None:
        return 0.0
    projected = cv2.perspectiveTransform(a[None], homography)[0]
    rms = float(np.sqrt(((projected - b) ** 2).sum(axis=1).mean()))
    return rms / max(interocular_distance(landmarks_b), 1e-6)


class DepthCue:
    """Collects poses during a session and compares well separated pairs."""

    def __init__(self, min_rotation_deg: float = 12.0, min_pairs: int = 3, residual_threshold: float = 0.028, max_keep: int = 60):
        self.min_rotation_deg = min_rotation_deg
        self.min_pairs = min_pairs
        self.residual_threshold = residual_threshold
        self.max_keep = max_keep
        self.reset()

    @classmethod
    def from_config(cls, cfg) -> "DepthCue":
        d = cfg.depth
        return cls(d.min_yaw_delta_deg, d.min_pairs, d.residual_threshold)

    def reset(self) -> None:
        self._poses: list[tuple[float, float, np.ndarray]] = []

    def add(self, yaw: float, pitch: float, landmarks: np.ndarray) -> None:
        """Keep a pose if it differs enough from those already stored."""
        for y, p, _ in self._poses:
            if abs(y - yaw) < 2.0 and abs(p - pitch) < 2.0:
                return
        if len(self._poses) < self.max_keep:
            self._poses.append((float(yaw), float(pitch), np.asarray(landmarks, dtype=np.float32).copy()))

    def evaluate(self) -> DepthResult:
        residuals, rotations = [], []
        n = len(self._poses)
        for i in range(n):
            for j in range(i + 1, n):
                yi, pi, li = self._poses[i]
                yj, pj, lj = self._poses[j]
                rotation = float(np.hypot(yi - yj, pi - pj))
                if rotation >= self.min_rotation_deg:
                    residuals.append(planar_residual(li, lj))
                    rotations.append(rotation)
        if len(residuals) < self.min_pairs:
            return DepthResult(False, 0.0, len(residuals), max(rotations, default=0.0), 0.0)
        residual = float(np.median(residuals))
        # soft score: 0 at half the threshold, 1 at one and a half times the threshold
        score = float(np.clip((residual - 0.5 * self.residual_threshold) / self.residual_threshold, 0.0, 1.0))
        return DepthResult(True, residual, len(residuals), max(rotations), score)
