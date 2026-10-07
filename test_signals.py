import numpy as np
import pytest

from pulsegate.signals.blink import BlinkDetector
from pulsegate.signals.depth import DepthCue, planar_residual
from pulsegate.signals.rppg import PulseEstimator, pos_pulse

from conftest import rotate_canonical


def run_blinks(ear: np.ndarray, fps: float = 30.0, **kwargs) -> BlinkDetector:
    detector = BlinkDetector(**kwargs)
    for i, value in enumerate(ear):
        detector.update(i / fps, float(value))
    return detector


def ear_trace(seconds: float, fps: float, blinks: list[tuple[float, float]], level: float = 0.3, noise: float = 0.004, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fps)
    ear = np.full_like(t, level) + rng.normal(0, noise, len(t))
    for start, duration in blinks:
        ear[(t >= start) & (t < start + duration)] = 0.3 * level
    return ear


def test_two_blinks_are_counted():
    detector = run_blinks(ear_trace(8, 30, [(3.0, 0.15), (5.0, 0.2)]))
    assert detector.count == 2
    assert detector.events[0].start == pytest.approx(3.0, abs=0.08)
    assert 0.1 < detector.events[0].duration < 0.3


def test_long_closure_is_not_a_blink():
    assert run_blinks(ear_trace(10, 30, [(3.0, 2.0)])).count == 0


def test_steady_eyes_produce_no_blinks():
    assert run_blinks(ear_trace(10, 30, [], noise=0.01)).count == 0


def test_blinks_survive_a_low_frame_rate():
    assert run_blinks(ear_trace(8, 12, [(3.0, 0.17), (5.5, 0.25)]), fps=12).count == 2


def test_baseline_adapts_to_narrow_eyes():
    assert run_blinks(ear_trace(8, 30, [(4.0, 0.15)], level=0.17, noise=0.002)).count == 1


def test_blinks_are_ignored_while_the_head_is_turned():
    detector = BlinkDetector()
    for i, value in enumerate(ear_trace(6, 30, [(3.0, 0.15)])):
        detector.update(i / 30, float(value), yaw=45.0)
    assert detector.count == 0


def skin_series(bpm: float, amplitude: float, seconds: float = 12.0, fps: float = 30.0, noise: float = 0.002, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fps)
    wave = np.sin(2 * np.pi * bpm / 60 * t) + 0.3 * np.sin(4 * np.pi * bpm / 60 * t)
    drift = 1 + 0.03 * np.sin(2 * np.pi * 0.2 * t)
    base = np.array([90.0, 110.0, 150.0])                              # B, G, R
    strength = np.array([0.35, 1.0, 0.5])
    skin = base * (1 + amplitude * wave[:, None] * strength) * drift[:, None] * (1 + noise * rng.standard_normal((len(t), 3)))
    return t, skin, drift


@pytest.mark.parametrize("bpm", [54, 72, 96, 132])
def test_pulse_rate_is_recovered(bpm):
    t, skin, drift = skin_series(bpm, 0.008)
    estimator = PulseEstimator()
    for ti, colour, d in zip(t, skin, drift):
        estimator.add_sample(ti, colour, np.full(3, 100.0) * d)
    result = estimator.estimate()
    assert result.valid
    assert result.bpm == pytest.approx(bpm, abs=2.5)
    assert result.snr_db > 3 and result.score > 0.6 and not result.ambient_flicker


def test_no_pulse_gives_no_evidence():
    t, skin, drift = skin_series(72, 0.0, noise=0.003)
    estimator = PulseEstimator()
    for ti, colour, d in zip(t, skin, drift):
        estimator.add_sample(ti, colour, np.full(3, 100.0) * d)
    result = estimator.estimate()
    assert result.valid and result.score < 0.3


def test_too_little_video_is_reported():
    estimator = PulseEstimator(min_seconds=8)
    t, skin, _ = skin_series(72, 0.008, seconds=4)
    for ti, colour in zip(t, skin):
        estimator.add_sample(ti, colour)
    result = estimator.estimate()
    assert not result.valid and "enough" in result.reason


def test_brightness_flicker_is_cancelled_by_the_projection():
    t, skin, _ = skin_series(0, 0.0)
    flicker = 1 + 0.02 * np.sin(2 * np.pi * 1.3 * t)                   # lamp or display beat at 78 per minute
    estimator = PulseEstimator()
    for ti, colour, f in zip(t, skin, flicker):
        estimator.add_sample(ti, colour * f, np.full(3, 100.0) * f)
    assert estimator.estimate().score == 0.0


def test_coloured_ambient_flicker_is_not_mistaken_for_a_pulse():
    t, skin, _ = skin_series(0, 0.0)
    flicker = 1 + 0.02 * np.sin(2 * np.pi * 1.3 * t)[:, None] * np.array([0.3, 1.0, 0.4])    # mostly green
    estimator = PulseEstimator()
    for ti, colour, f in zip(t, skin, flicker):
        estimator.add_sample(ti, colour * f, np.full(3, 100.0) * f)
    result = estimator.estimate()
    assert result.bpm == pytest.approx(78, abs=3)                      # the rhythm is found in the skin
    assert result.ambient_flicker and result.score == 0.0              # and rejected because the room shares it


def test_irregular_frame_timing_is_resampled():
    t, skin, drift = skin_series(78, 0.008)
    rng = np.random.default_rng(1)
    t = t + rng.uniform(-0.012, 0.012, len(t))
    estimator = PulseEstimator()
    for ti, colour, d in zip(np.sort(t), skin, drift):
        estimator.add_sample(ti, colour, np.full(3, 100.0) * d)
    assert estimator.estimate().bpm == pytest.approx(78, abs=3)


def test_pos_projection_cancels_brightness_changes():
    rng = np.random.default_rng(0)
    n = 300
    brightness = 1 + 0.05 * np.sin(np.linspace(0, 20, n))
    rgb = np.outer(brightness, [150.0, 110.0, 90.0]) + rng.normal(0, 0.01, (n, 3))
    assert np.std(pos_pulse(rgb, 30.0)) < 1e-2


def test_flat_surface_has_no_residual(canonical):
    rng = np.random.default_rng(0)
    flat = rotate_canonical(canonical, 0)
    flat[:, 2] = 0
    homography = np.array([[0.9, 0.05, 12.0], [-0.03, 1.05, -8.0], [2e-4, -1e-4, 1.0]])
    homog = np.column_stack([flat[:, :2], np.ones(len(flat))]) @ homography.T
    tilted = flat.copy()
    tilted[:, :2] = homog[:, :2] / homog[:, 2:3] + rng.normal(0, 0.05, (len(flat), 2))
    assert planar_residual(flat, tilted) < 0.005


def test_real_rotation_has_a_clear_residual(canonical):
    assert planar_residual(rotate_canonical(canonical, 0), rotate_canonical(canonical, 20)) > 0.03


def test_depth_cue_needs_rotation_before_it_judges(canonical):
    cue = DepthCue(min_rotation_deg=12, min_pairs=3)
    for yaw in (0, 3, 6):
        cue.add(yaw, 0, rotate_canonical(canonical, yaw))
    assert not cue.evaluate().valid
    for yaw in (14, 18, 22):
        cue.add(yaw, 0, rotate_canonical(canonical, yaw))
    result = cue.evaluate()
    assert result.valid and result.score == 1.0 and result.pairs >= 3


def test_depth_cue_scores_a_tilted_photo_as_flat(canonical):
    cue = DepthCue(min_rotation_deg=12, min_pairs=3)
    flat = rotate_canonical(canonical, 0)
    for step, yaw in enumerate((0, 14, 18, 22, 26)):
        # the reported angle changes but the landmarks only move by a plane to plane mapping
        homography = np.array([[1 - 0.01 * step, 0.0, 3.0 * step], [0.0, 1.0, 0.0], [1.5e-4 * step, 0.0, 1.0]])
        homog = np.column_stack([flat[:, :2], np.ones(len(flat))]) @ homography.T
        moved = flat.copy()
        moved[:, :2] = homog[:, :2] / homog[:, 2:3]
        cue.add(yaw, 0, moved)
    result = cue.evaluate()
    assert result.valid and result.score == 0.0
