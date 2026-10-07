"""Dense face landmarks from the MediaPipe Face Mesh network, run with ONNX Runtime."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MESH_INPUT = 192


@dataclass(frozen=True)
class FaceRoi:
    """Rotated square region that is fed to the mesh network."""

    cx: float
    cy: float
    size: float
    angle: float      # radians, in plane rotation of the face

    def matrix(self, out_size: int = MESH_INPUT) -> np.ndarray:
        """Affine transform from frame pixels to network input pixels."""
        s = out_size / self.size
        cos, sin = np.cos(self.angle), np.sin(self.angle)
        rot = np.array([[cos, sin], [-sin, cos]], dtype=np.float64) * s
        shift = np.array([out_size / 2.0, out_size / 2.0]) - rot @ np.array([self.cx, self.cy])
        return np.hstack([rot, shift[:, None]])


def roi_from_detection(box: np.ndarray, points: np.ndarray, scale: float = 1.5) -> FaceRoi:
    """Initial region from a detector box and its eye points."""
    right_eye, left_eye = points[0], points[1]
    angle = float(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]))
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    return FaceRoi(float(cx), float(cy), float(max(box[2], box[3]) * scale), angle)


def roi_from_landmarks(landmarks: np.ndarray, scale: float = 1.5) -> FaceRoi:
    """Tracking region from the previous mesh, following the MediaPipe graph."""
    right_eye, left_eye = landmarks[33, :2], landmarks[263, :2]
    angle = float(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]))
    cos, sin = np.cos(angle), np.sin(angle)
    rot = np.array([[cos, sin], [-sin, cos]])
    aligned = landmarks[:, :2] @ rot.T
    lo, hi = aligned.min(axis=0), aligned.max(axis=0)
    centre = rot.T @ ((lo + hi) / 2.0)
    return FaceRoi(float(centre[0]), float(centre[1]), float(max(hi - lo) * scale), angle)


class FaceMesh:
    """468 point 3D face mesh.

    ``predict`` returns landmarks in frame pixels (z shares the x scale and is
    negative towards the camera) together with a face presence probability.
    """

    def __init__(self, model_path: str | Path, threads: int = 1):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"face mesh model not found: {model_path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self._session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def predict(self, frame_bgr: np.ndarray, roi: FaceRoi) -> tuple[np.ndarray, float]:
        matrix = roi.matrix()
        crop = cv2.warpAffine(frame_bgr, matrix, (MESH_INPUT, MESH_INPUT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        blob = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
        raw, flag = self._session.run(None, {self._input: blob})
        pts = raw.reshape(468, 3).astype(np.float64)
        inverse = cv2.invertAffineTransform(matrix)
        xy = pts[:, :2] @ inverse[:, :2].T + inverse[:, 2]
        z = pts[:, 2] * (roi.size / MESH_INPUT)
        presence = 1.0 / (1.0 + np.exp(-float(flag.ravel()[0])))
        return np.column_stack([xy, z]), float(presence)
