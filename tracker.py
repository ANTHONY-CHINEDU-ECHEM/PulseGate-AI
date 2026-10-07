"""Face tracking: detection, dense landmarks and per frame measurements."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import Config
from .detector import Detection, FaceDetector
from .geometry import (
    LEFT_EYE, RIGHT_EYE, HeadPose, HeadPoseEstimator, eye_aspect_ratio, interocular_distance,
    mouth_open_ratio, square_box,
)
from .landmarks import FaceMesh, FaceRoi, roi_from_detection, roi_from_landmarks


@dataclass
class FaceObservation:
    """Everything measured on one frame for the tracked face."""

    timestamp: float
    box: np.ndarray                 # detector style box x, y, w, h
    landmarks: np.ndarray           # 468 x 3 in frame pixels
    pose: HeadPose
    ear_left: float
    ear_right: float
    mouth_open: float
    presence: float
    num_faces: int
    interocular: float
    detector_points: np.ndarray | None = None
    extras: dict = field(default_factory=dict)

    @property
    def ear(self) -> float:
        return 0.5 * (self.ear_left + self.ear_right)

    @property
    def center(self) -> np.ndarray:
        return np.array([self.box[0] + self.box[2] / 2.0, self.box[1] + self.box[3] / 2.0])

    @property
    def size(self) -> float:
        return float(max(self.box[2], self.box[3]))


def box_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Detector like box derived from the mesh.

    YuNet boxes span roughly from the hairline to the chin and from ear to ear.
    The mesh covers the same extent, so its bounding box is a stable substitute
    between detector runs.
    """
    lo = landmarks[:, :2].min(axis=0)
    hi = landmarks[:, :2].max(axis=0)
    return np.array([lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]], dtype=np.float32)


class FaceTracker:
    """Tracks the most prominent face across frames.

    The detector runs on the first frame, after a lost track and every
    ``redetect_every`` frames. In between, the mesh region follows the previous
    landmarks, which is both faster and steadier than detecting every frame.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        v = cfg.vision
        self.detector = FaceDetector(cfg.path("assets.detector"), v.detector_score, v.detector_nms, v.detector_max_side)
        self.mesh = FaceMesh(cfg.path("assets.facemesh"))
        self.pose = HeadPoseEstimator(np.load(cfg.path("assets.canonical_face")))
        self._roi: FaceRoi | None = None
        self._since_detect = 0
        self._num_faces = 0
        self._last_det: Detection | None = None

    def fork(self) -> "FaceTracker":
        """A tracker with its own state that shares the loaded networks.

        Loading the models takes far longer than a session lasts, so concurrent
        sessions share them and only keep separate tracking state.
        """
        import copy

        other = copy.copy(self)
        other.reset()
        return other

    def reset(self) -> None:
        self._roi = None
        self._since_detect = 0
        self._num_faces = 0
        self._last_det = None

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        return self.detector.detect(frame_bgr)

    def process(self, frame_bgr: np.ndarray, timestamp: float = 0.0) -> FaceObservation | None:
        v = self.cfg.vision
        need_detection = self._roi is None or self._since_detect >= int(v.redetect_every)
        if need_detection:
            detections = self.detector.detect(frame_bgr)
            self._num_faces = len(detections)
            self._since_detect = 0
            if detections:
                self._last_det = detections[0]
                if self._roi is None:
                    self._roi = roi_from_detection(detections[0].box, detections[0].points, v.roi_scale)
            elif self._roi is None:
                return None
        self._since_detect += 1

        landmarks, presence = self.mesh.predict(frame_bgr, self._roi)
        if need_detection and presence >= v.min_presence:
            # One refinement pass: the region from the detector box is looser
            # than the region derived from the mesh itself.
            self._roi = roi_from_landmarks(landmarks, v.roi_scale)
            landmarks, presence = self.mesh.predict(frame_bgr, self._roi)
        if presence < v.min_presence:
            self._roi = None
            return None
        self._roi = roi_from_landmarks(landmarks, v.roi_scale)
        return self.observe(landmarks, presence, timestamp)

    def observe(self, landmarks: np.ndarray, presence: float, timestamp: float) -> FaceObservation:
        return FaceObservation(
            timestamp=float(timestamp),
            box=box_from_landmarks(landmarks),
            landmarks=landmarks,
            pose=self.pose.estimate(landmarks),
            ear_left=eye_aspect_ratio(landmarks, LEFT_EYE),
            ear_right=eye_aspect_ratio(landmarks, RIGHT_EYE),
            mouth_open=mouth_open_ratio(landmarks),
            presence=presence,
            num_faces=self._num_faces,
            interocular=interocular_distance(landmarks),
            detector_points=None if self._last_det is None else self._last_det.points,
        )

    def process_image(self, image_bgr: np.ndarray, near: np.ndarray | None = None) -> FaceObservation | None:
        """Stateless variant for still images.

        ``near`` selects the detection closest to a known position instead of
        the largest face, which matters when a second face is in the scene.
        """
        self.reset()
        detections = self.detector.detect(image_bgr)
        if not detections:
            return None
        chosen = detections[0]
        if near is not None:
            chosen = min(detections, key=lambda d: float(np.linalg.norm(d.center - near)))
            if np.linalg.norm(chosen.center - near) > 0.6 * chosen.size:
                return None
        self._num_faces = len(detections)
        self._last_det = chosen
        v = self.cfg.vision
        roi = roi_from_detection(chosen.box, chosen.points, v.roi_scale)
        landmarks, presence = self.mesh.predict(image_bgr, roi)
        if presence >= v.min_presence:
            landmarks, presence = self.mesh.predict(image_bgr, roi_from_landmarks(landmarks, v.roi_scale))
        obs = None if presence < v.min_presence else self.observe(landmarks, presence, 0.0)
        self.reset()
        return obs


def context_crop(frame_bgr: np.ndarray, box: np.ndarray, scale: float) -> np.ndarray:
    """Native resolution square crop around a face box with surrounding context."""
    from .geometry import crop_square

    cx, cy, side = square_box(box, scale)
    return crop_square(frame_bgr, cx, cy, side)


def draw_observation(frame_bgr: np.ndarray, obs: FaceObservation, color: tuple[int, int, int] = (90, 220, 120)) -> np.ndarray:
    """Debug overlay with the mesh points and the face box."""
    out = frame_bgr.copy()
    for x, y in obs.landmarks[::3, :2]:
        cv2.circle(out, (int(x), int(y)), 1, color, -1, cv2.LINE_AA)
    x, y, w, h = [int(round(float(t))) for t in obs.box]
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
    return out
