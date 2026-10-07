"""Face detection with YuNet through OpenCV."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected face in full frame pixel coordinates."""

    box: np.ndarray          # x, y, w, h
    points: np.ndarray       # 5 x 2: right eye, left eye, nose tip, right mouth corner, left mouth corner
    score: float

    @property
    def center(self) -> np.ndarray:
        return np.array([self.box[0] + self.box[2] / 2.0, self.box[1] + self.box[3] / 2.0])

    @property
    def size(self) -> float:
        return float(max(self.box[2], self.box[3]))


class FaceDetector:
    """Thin wrapper around ``cv2.FaceDetectorYN``.

    Frames are downscaled so that the long side equals ``max_side`` before the
    detector runs, which keeps latency flat across camera resolutions. Results
    are mapped back to the original frame.
    """

    def __init__(self, model_path: str | Path, score_threshold: float = 0.6, nms_threshold: float = 0.3, max_side: int = 480):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"face detector model not found: {model_path}")
        self._net = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, nms_threshold, 200)
        self._max_side = int(max_side)
        self._input_size: tuple[int, int] | None = None

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self._max_side / float(max(h, w)))
        if scale < 1.0:
            small = cv2.resize(frame_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr
        size = (small.shape[1], small.shape[0])
        if size != self._input_size:
            self._net.setInputSize(size)
            self._input_size = size
        _, faces = self._net.detect(small)
        if faces is None:
            return []
        out = []
        for row in np.asarray(faces, dtype=np.float32):
            box = row[:4] / scale
            pts = row[4:14].reshape(5, 2) / scale
            out.append(Detection(box=box, points=pts, score=float(row[14])))
        out.sort(key=lambda d: d.size, reverse=True)
        return out
