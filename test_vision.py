import numpy as np
import pytest

from pulsegate.vision.landmarks import FaceRoi, roi_from_landmarks
from pulsegate.vision.quality import assess_quality


def test_detector_finds_the_sample_face(tracker, face_image):
    detections = tracker.detect(face_image)
    assert len(detections) == 1
    assert detections[0].score > 0.8
    h, w = face_image.shape[:2]
    assert 0 < detections[0].center[0] < w and 0 < detections[0].center[1] < h


def test_detector_returns_nothing_on_an_empty_scene(tracker):
    assert tracker.detect(np.full((240, 320, 3), 128, np.uint8)) == []
    assert tracker.process_image(np.zeros((240, 320, 3), np.uint8)) is None


def test_mesh_measurements_are_plausible(tracker, face_image):
    obs = tracker.process_image(face_image)
    assert obs is not None and obs.landmarks.shape == (468, 3)
    assert obs.presence > 0.9
    assert abs(obs.pose.yaw) < 15 and abs(obs.pose.pitch) < 20 and abs(obs.pose.roll) < 10      # roughly frontal
    assert 0.15 < obs.ear < 0.45                                                                 # eyes open
    assert obs.mouth_open < 0.25                                                                 # mouth closed
    assert 30 < obs.interocular < obs.size
    # the eyes are above the mouth and the right eye is on the image left
    assert obs.landmarks[33, 1] < obs.landmarks[13, 1]
    assert obs.landmarks[33, 0] < obs.landmarks[263, 0]


def test_mirrored_image_flips_the_yaw_sign(tracker, face_image):
    obs = tracker.process_image(face_image)
    mirrored = tracker.process_image(np.ascontiguousarray(face_image[:, ::-1]))
    assert mirrored is not None
    assert mirrored.pose.yaw == pytest.approx(-obs.pose.yaw, abs=4.0)
    assert mirrored.pose.pitch == pytest.approx(obs.pose.pitch, abs=4.0)


def test_tracking_is_consistent_with_detection(tracker, clip_frames):
    frames, fps = clip_frames
    tracker.reset()
    sizes = []
    for i, frame in enumerate(frames[:40]):
        obs = tracker.process(frame, i / fps)
        assert obs is not None
        sizes.append(obs.size)
    tracker.reset()
    assert np.std(sizes) / np.mean(sizes) < 0.08            # the tracked box does not wobble


def test_forked_trackers_do_not_share_state(tracker, face_image):
    a, b = tracker.fork(), tracker.fork()
    assert a.process(face_image, 0.0) is not None
    assert b._roi is None and a._roi is not None
    assert a.mesh is b.mesh                                 # the networks are shared


def test_near_selects_the_requested_face(tracker, face_image):
    obs = tracker.process_image(face_image)
    assert tracker.process_image(face_image, near=obs.center) is not None
    assert tracker.process_image(face_image, near=np.array([5.0, 5.0])) is None


def test_roi_matrix_centres_the_face():
    roi = FaceRoi(cx=200, cy=150, size=100, angle=0.0)
    matrix = roi.matrix(192)
    assert matrix @ np.array([200, 150, 1.0]) == pytest.approx([96, 96])
    assert matrix @ np.array([250, 150, 1.0]) == pytest.approx([192, 96])


def test_roi_from_landmarks_is_square_and_covers_the_mesh(tracker, face_image):
    obs = tracker.process_image(face_image)
    roi = roi_from_landmarks(obs.landmarks, 1.5)
    extent = obs.landmarks[:, :2].max(0) - obs.landmarks[:, :2].min(0)
    assert roi.size >= 1.4 * extent.max()


def test_quality_gate(cfg, tracker, face_image):
    obs = tracker.process_image(face_image)
    assert assess_quality(face_image, obs, cfg).ok
    dark = (face_image * 0.15).astype(np.uint8)
    assert "too_dark" in assess_quality(dark, obs, cfg).issues
    assert assess_quality(face_image, None, cfg).issues == ("no_face",)
    small = tracker.process_image(np.ascontiguousarray(face_image[::2, ::2]))
    if small is not None and small.size < cfg.quality.min_face_px:
        report = assess_quality(face_image[::2, ::2], small, cfg)
        assert "too_far" in report.issues and report.guidance == "Move closer"
