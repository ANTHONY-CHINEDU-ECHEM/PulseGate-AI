import numpy as np
import pytest

from pulsegate.vision import geometry as g

from conftest import rotate_canonical


def test_eye_aspect_ratio_on_a_known_shape():
    pts = np.zeros((468, 3))
    eye = g.LEFT_EYE
    for index, xy in zip(eye, [(0, 0), (1, -1), (3, -1), (4, 0), (3, 1), (1, 1)]):
        pts[index, :2] = xy
    assert g.eye_aspect_ratio(pts, eye) == pytest.approx(0.5)      # (2 + 2) / (2 * 4)
    for index in (eye[1], eye[2], eye[4], eye[5]):
        pts[index, 1] = 0.0
    assert g.eye_aspect_ratio(pts, eye) == pytest.approx(0.0)


@pytest.mark.parametrize("yaw,pitch", [(0, 0), (20, 0), (-25, 0), (0, 15), (0, -12), (18, 10)])
def test_head_pose_recovers_known_rotation(canonical, yaw, pitch):
    estimator = g.HeadPoseEstimator(canonical)
    pose = estimator.estimate(rotate_canonical(canonical, yaw, pitch))
    assert pose.yaw == pytest.approx(yaw, abs=0.5)
    assert pose.pitch == pytest.approx(pitch, abs=0.5)
    assert abs(pose.roll) < 0.5


def test_head_pose_ignores_scale_and_position(canonical):
    estimator = g.HeadPoseEstimator(canonical)
    a = estimator.estimate(rotate_canonical(canonical, 14, 6, scale=5.0, centre=(100, 80)))
    b = estimator.estimate(rotate_canonical(canonical, 14, 6, scale=30.0, centre=(900, 500)))
    assert a.yaw == pytest.approx(b.yaw, abs=1e-6)
    assert a.pitch == pytest.approx(b.pitch, abs=1e-6)


def test_crop_square_inside_and_outside_the_frame():
    image = np.arange(100 * 120 * 3, dtype=np.uint8).reshape(100, 120, 3)
    inside = g.crop_square(image, 60, 50, 40)
    assert inside.shape == (40, 40, 3)
    assert np.array_equal(inside, image[30:70, 40:80])
    outside = g.crop_square(image, 5, 5, 60)
    assert outside.shape == (60, 60, 3)
    huge = g.crop_square(image, 60, 50, 400)
    assert huge.shape == (400, 400, 3)
    resized = g.crop_square(image, 60, 50, 80, out_size=32)
    assert resized.shape == (32, 32, 3)


def test_square_box_uses_the_longer_side():
    cx, cy, side = g.square_box((10, 20, 30, 50), scale=2.0)
    assert (cx, cy, side) == (25.0, 45.0, 100.0)


def test_mouth_ratio_grows_with_opening(canonical):
    pts = rotate_canonical(canonical, 0)
    closed = g.mouth_open_ratio(pts)
    pts[g.MOUTH_INNER_VERTICAL[1], 1] += 30
    assert g.mouth_open_ratio(pts) > closed + 0.2
