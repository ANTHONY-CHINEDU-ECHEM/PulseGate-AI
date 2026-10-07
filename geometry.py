"""Geometric measurements on the 468 point face mesh.

Conventions used across the project
* image coordinates: x to the right, y down, z away from the camera
* yaw is positive when the nose points towards image right
  (the subject turns to their own left when the image is not mirrored)
* pitch is positive when the subject looks up
* roll is positive for a clockwise in plane rotation as seen in the image
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Six point eye contours in the order p1..p6 of the eye aspect ratio definition.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)     # image left for an unmirrored frontal face
LEFT_EYE = (362, 385, 387, 263, 373, 380)
EYE_OUTER = (33, 263)
MOUTH_INNER_VERTICAL = (13, 14)
MOUTH_INNER_HORIZONTAL = (78, 308)
MOUTH_OUTER_HORIZONTAL = (61, 291)
NOSE_TIP = 1

# Points that barely move with expression, used for head pose and the depth cue.
RIGID_POINTS = (
    1, 4, 5, 195, 197, 6, 168,                 # nose ridge
    9, 8, 10, 151, 109, 338, 67, 297,          # forehead
    33, 133, 362, 263,                         # eye corners
    127, 356, 234, 454, 116, 345, 50, 280,     # temples and cheek bones
    98, 327, 64, 294,                          # nose wings
)

# Skin regions for remote photoplethysmography.
FOREHEAD = (109, 10, 338, 297, 299, 9, 69, 67)
RIGHT_CHEEK = (116, 117, 118, 119, 100, 142, 203, 206, 216, 192, 213, 147, 123)
LEFT_CHEEK = (345, 346, 347, 348, 329, 371, 423, 426, 436, 416, 433, 376, 352)

FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(landmarks: np.ndarray, eye: tuple[int, ...]) -> float:
    """Eye aspect ratio of Soukupova and Cech (2016) on the mesh eye contour."""
    p = landmarks[list(eye), :2]
    horizontal = _dist(p[0], p[3])
    if horizontal < 1e-6:
        return 0.0
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * horizontal)


def mouth_open_ratio(landmarks: np.ndarray) -> float:
    """Inner lip opening divided by inner mouth width."""
    width = _dist(landmarks[MOUTH_INNER_HORIZONTAL[0], :2], landmarks[MOUTH_INNER_HORIZONTAL[1], :2])
    if width < 1e-6:
        return 0.0
    return _dist(landmarks[MOUTH_INNER_VERTICAL[0], :2], landmarks[MOUTH_INNER_VERTICAL[1], :2]) / width


def interocular_distance(landmarks: np.ndarray) -> float:
    return _dist(landmarks[EYE_OUTER[0], :2], landmarks[EYE_OUTER[1], :2])


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float
    roll: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.yaw, self.pitch, self.roll)


class HeadPoseEstimator:
    """Head rotation from a rigid fit of the canonical face to the predicted mesh.

    The face mesh network predicts a weak perspective 3D shape. Aligning the
    canonical model to it with the Kabsch algorithm gives a rotation without
    needing camera intrinsics.
    """

    def __init__(self, canonical: np.ndarray, points: tuple[int, ...] = RIGID_POINTS):
        canon = np.asarray(canonical, dtype=np.float64).copy()
        canon[:, 1] *= -1.0      # canonical y is up, image y is down
        canon[:, 2] *= -1.0      # canonical z points to the camera, image z points away
        self._idx = list(points)
        ref = canon[self._idx]
        self._ref = ref - ref.mean(axis=0)

    def rotation(self, landmarks: np.ndarray) -> np.ndarray:
        pts = np.asarray(landmarks, dtype=np.float64)[self._idx]
        pts = pts - pts.mean(axis=0)
        h = self._ref.T @ pts
        u, _, vt = np.linalg.svd(h)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        return vt.T @ np.diag([1.0, 1.0, d]) @ u.T

    def estimate(self, landmarks: np.ndarray) -> HeadPose:
        r = self.rotation(landmarks)
        forward = r @ np.array([0.0, 0.0, -1.0])     # direction the nose points
        right = r @ np.array([1.0, 0.0, 0.0])
        yaw = np.degrees(np.arctan2(forward[0], -forward[2]))
        pitch = np.degrees(np.arctan2(-forward[1], np.hypot(forward[0], forward[2])))
        roll = np.degrees(np.arctan2(right[1], right[0]))
        return HeadPose(float(yaw), float(pitch), float(roll))


def square_box(box_xywh: np.ndarray | tuple[float, float, float, float], scale: float = 1.0) -> tuple[float, float, float]:
    """Centre and side of the square that covers a detector box, optionally enlarged."""
    x, y, w, h = [float(v) for v in box_xywh]
    return x + w / 2.0, y + h / 2.0, max(w, h) * scale


def crop_square(image: np.ndarray, cx: float, cy: float, side: float, out_size: int | None = None) -> np.ndarray:
    """Crop a square around a centre. Parts outside the frame are mirrored.

    With ``out_size=None`` the crop keeps the native pixel pitch, which the
    texture stream relies on. Otherwise the crop is resized with area averaging.
    """
    side_px = max(int(round(side)), 2)
    x0 = int(round(cx - side_px / 2.0))
    y0 = int(round(cy - side_px / 2.0))
    h, w = image.shape[:2]
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x0 + side_px - w), max(0, y0 + side_px - h)
    if pad_l or pad_t or pad_r or pad_b:
        limit_x, limit_y = max(w - 1, 0), max(h - 1, 0)
        image = cv2.copyMakeBorder(
            image, min(pad_t, limit_y), min(pad_b, limit_y), min(pad_l, limit_x), min(pad_r, limit_x),
            cv2.BORDER_REFLECT_101,
        )
        x0 += min(pad_l, limit_x)
        y0 += min(pad_t, limit_y)
        if image.shape[0] < y0 + side_px or image.shape[1] < x0 + side_px or x0 < 0 or y0 < 0:
            extra_b = max(0, y0 + side_px - image.shape[0])
            extra_r = max(0, x0 + side_px - image.shape[1])
            extra_t, extra_l = max(0, -y0), max(0, -x0)
            image = cv2.copyMakeBorder(image, extra_t, extra_b, extra_l, extra_r, cv2.BORDER_REPLICATE)
            x0 += extra_l
            y0 += extra_t
    crop = image[y0:y0 + side_px, x0:x0 + side_px]
    if out_size is not None and crop.shape[0] != out_size:
        interp = cv2.INTER_AREA if crop.shape[0] > out_size else cv2.INTER_LINEAR
        crop = cv2.resize(crop, (out_size, out_size), interpolation=interp)
    return crop


def polygon_mask(shape_hw: tuple[int, int], points: np.ndarray) -> np.ndarray:
    """Boolean mask of the convex hull of a set of image points."""
    mask = np.zeros(shape_hw, dtype=np.uint8)
    hull = cv2.convexHull(np.round(points).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)
