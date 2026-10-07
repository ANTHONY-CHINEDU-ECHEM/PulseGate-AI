"""Shared fixtures. Heavy objects are created once per test run."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pulsegate.config import load_config  # noqa: E402

SAMPLES = ROOT / "assets" / "samples"


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def face_image() -> np.ndarray:
    image = cv2.imread(str(SAMPLES / "face.jpg"))
    assert image is not None, "sample image is missing"
    return image


@pytest.fixture(scope="session")
def clip_path() -> Path:
    path = SAMPLES / "live_clip.mp4"
    assert path.exists(), "sample clip is missing"
    return path


@pytest.fixture(scope="session")
def clip_frames(clip_path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, float(fps)


@pytest.fixture(scope="session")
def tracker(cfg):
    from pulsegate.vision.tracker import FaceTracker

    return FaceTracker(cfg)


@pytest.fixture(scope="session")
def engine(cfg):
    if not cfg.path("assets.passive_model").exists():
        pytest.skip("no trained passive model in models/, run 'python manage.py train'")
    from pulsegate.engine.session import LivenessEngine

    return LivenessEngine(cfg)


@pytest.fixture(scope="session")
def canonical(cfg) -> np.ndarray:
    return np.load(cfg.path("assets.canonical_face"))


def rotate_canonical(canonical: np.ndarray, yaw_deg: float, pitch_deg: float = 0.0, scale: float = 12.0, centre=(320.0, 240.0)) -> np.ndarray:
    """Synthetic landmarks: the canonical face in image coordinates, turned by known angles.

    Positive yaw points the nose to image right, positive pitch looks up.
    """
    pts = np.asarray(canonical, dtype=np.float64).copy()
    pts[:, 1] *= -1.0
    pts[:, 2] *= -1.0
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    r_yaw = np.array([[np.cos(yaw), 0, -np.sin(yaw)], [0, 1, 0], [np.sin(yaw), 0, np.cos(yaw)]])
    r_pitch = np.array([[1, 0, 0], [0, np.cos(pitch), np.sin(pitch)], [0, -np.sin(pitch), np.cos(pitch)]])
    out = pts @ (r_yaw @ r_pitch).T * scale
    out[:, 0] += centre[0]
    out[:, 1] += centre[1]
    return out
