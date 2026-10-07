"""Capture quality checks that decide whether a frame is fit for scoring."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .tracker import FaceObservation

GUIDANCE = {
    "no_face": "Look at the camera",
    "multiple_faces": "Only one person in view please",
    "too_far": "Move closer",
    "too_close": "Move back a little",
    "off_center": "Centre your face in the oval",
    "too_dark": "Find more light",
    "too_bright": "Too much light on your face",
    "blurry": "Hold still",
    "not_frontal": "Face the camera",
}


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    issues: tuple[str, ...]
    brightness: float
    sharpness: float

    @property
    def guidance(self) -> str:
        return GUIDANCE.get(self.issues[0], "") if self.issues else ""


def assess_quality(frame_bgr: np.ndarray, obs: FaceObservation | None, cfg, check_pose: bool = True) -> QualityReport:
    """Rule based quality gate. The order of the issues is the order in which a user should fix them."""
    q = cfg.quality
    if obs is None:
        return QualityReport(False, ("no_face",), 0.0, 0.0)
    h, w = frame_bgr.shape[:2]
    issues: list[str] = []
    if obs.num_faces > 1 and not cfg.session.allow_multiple_faces:
        issues.append("multiple_faces")
    size = obs.size
    if size < q.min_face_px:
        issues.append("too_far")
    elif size > q.max_face_frac * min(h, w):
        issues.append("too_close")
    offset = np.abs(obs.center - np.array([w / 2.0, h / 2.0])) / np.array([w, h])
    if offset.max() > q.center_tolerance:
        issues.append("off_center")
    x0, y0 = max(0, int(obs.box[0])), max(0, int(obs.box[1]))
    x1, y1 = min(w, int(obs.box[0] + obs.box[2])), min(h, int(obs.box[1] + obs.box[3]))
    brightness, sharpness = 0.0, 0.0
    if x1 - x0 > 8 and y1 - y0 > 8:
        gray = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        # sharpness is measured at a fixed face size so that it does not depend on camera resolution
        norm = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        sharpness = float(cv2.Laplacian(norm, cv2.CV_64F).var())
        if brightness < q.min_brightness:
            issues.append("too_dark")
        elif brightness > q.max_brightness:
            issues.append("too_bright")
        if sharpness < q.min_sharpness:
            issues.append("blurry")
    if check_pose and (abs(obs.pose.yaw) > q.max_abs_yaw_deg or abs(obs.pose.pitch) > q.max_abs_pitch_deg):
        issues.append("not_frontal")
    return QualityReport(not issues, tuple(issues), brightness, sharpness)
