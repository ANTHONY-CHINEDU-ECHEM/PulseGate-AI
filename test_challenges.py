import numpy as np
import pytest

from pulsegate.engine.challenges import Baseline, Challenge, ChallengeEvaluator, sample_challenges
from pulsegate.signals.blink import BlinkEvent
from pulsegate.vision.geometry import HeadPose
from pulsegate.vision.tracker import FaceObservation


def observation(yaw=0.0, pitch=0.0, mouth=0.03, t=0.0) -> FaceObservation:
    return FaceObservation(
        timestamp=t, box=np.array([100, 100, 150, 150], dtype=np.float32), landmarks=np.zeros((468, 3)),
        pose=HeadPose(yaw, pitch, 0.0), ear_left=0.3, ear_right=0.3, mouth_open=mouth, presence=1.0, num_faces=1, interocular=60.0,
    )


def run(cfg, challenge, script, baseline=None, fps=20.0):
    """``script`` maps time to keyword arguments of :func:`observation` or to a blink event."""
    evaluator = ChallengeEvaluator(cfg, baseline or Baseline())
    record = evaluator.start(challenge, 0.0)
    for step in range(int(10 * fps)):
        t = step / fps
        state = script(t)
        blink = state.pop("blink", None)
        evaluator.update(observation(t=t, **state), t, blink)
        if record.passed is not None:
            break
    return record


def test_turn_left_passes_on_positive_yaw(cfg):
    record = run(cfg, Challenge.TURN_LEFT, lambda t: {"yaw": 25.0 if t > 1.5 else 0.0})
    assert record.passed is True
    assert 1.5 < record.response_time < 2.0


def test_turn_right_passes_on_negative_yaw(cfg):
    assert run(cfg, Challenge.TURN_RIGHT, lambda t: {"yaw": -25.0 if t > 1.5 else 0.0}).passed is True


def test_wrong_direction_fails(cfg):
    record = run(cfg, Challenge.TURN_LEFT, lambda t: {"yaw": -25.0 if t > 1.5 else 0.0})
    assert record.passed is False and "wrong" in record.detail


def test_no_response_times_out(cfg):
    record = run(cfg, Challenge.OPEN_MOUTH, lambda t: {})
    assert record.passed is False and record.detail == "timeout"
    assert record.finished_at > cfg.challenge.timeout_s


def test_response_before_reaction_time_does_not_count(cfg):
    # the head is already turned when the prompt appears and returns before a human could have reacted
    record = run(cfg, Challenge.TURN_LEFT, lambda t: {"yaw": 25.0 if t < 0.4 else 0.0})
    assert record.passed is False and record.detail == "timeout"


def test_turn_is_measured_against_the_resting_pose(cfg):
    baseline = Baseline(yaw=10.0)
    assert run(cfg, Challenge.TURN_LEFT, lambda t: {"yaw": 20.0}, baseline).passed is False
    assert run(cfg, Challenge.TURN_LEFT, lambda t: {"yaw": 30.0 if t > 1 else 10.0}, baseline).passed is True


def test_single_noisy_frame_is_not_enough(cfg):
    record = run(cfg, Challenge.LOOK_UP, lambda t: {"pitch": 20.0 if abs(t - 2.0) < 1e-6 else 0.0})
    assert record.passed is False


def test_look_up_and_open_mouth(cfg):
    assert run(cfg, Challenge.LOOK_UP, lambda t: {"pitch": 15.0 if t > 1 else 0.0}).passed is True
    assert run(cfg, Challenge.OPEN_MOUTH, lambda t: {"mouth": 0.6 if t > 1 else 0.03}).passed is True


def test_blink_twice_needs_two_blinks_after_the_prompt(cfg):
    def one(t):
        return {"blink": BlinkEvent(1.9, 2.0, 0.3) if abs(t - 2.0) < 1e-6 else None}

    def two(t):
        if abs(t - 2.0) < 1e-6:
            return {"blink": BlinkEvent(1.9, 2.0, 0.3)}
        if abs(t - 3.0) < 1e-6:
            return {"blink": BlinkEvent(2.9, 3.0, 0.3)}
        return {}

    def early(t):
        if abs(t - 0.1) < 1e-6:
            return {"blink": BlinkEvent(0.0, 0.1, 0.3)}
        if abs(t - 3.0) < 1e-6:
            return {"blink": BlinkEvent(2.9, 3.0, 0.3)}
        return {}

    assert run(cfg, Challenge.BLINK_TWICE, one).passed is False
    assert run(cfg, Challenge.BLINK_TWICE, two).passed is True
    assert run(cfg, Challenge.BLINK_TWICE, early).passed is False


def test_neutral_pose_check(cfg):
    evaluator = ChallengeEvaluator(cfg, Baseline(yaw=2.0, pitch=1.0, mouth=0.03))
    assert evaluator.is_neutral(observation(yaw=4.0, pitch=0.0))
    assert not evaluator.is_neutral(observation(yaw=25.0))
    assert not evaluator.is_neutral(observation(mouth=0.5))


def test_sampled_sequences_are_random_unique_and_never_chain_turns(cfg):
    rng = np.random.default_rng(0)
    pool = list(cfg.challenge.pool)
    seen = set()
    for _ in range(300):
        order = sample_challenges(pool, 3, rng)
        assert len(order) == 3 and len(set(order)) == 3
        for a, b in zip(order, order[1:]):
            assert not ({a, b} == {Challenge.TURN_LEFT, Challenge.TURN_RIGHT})
        seen.add(tuple(order))
    assert len(seen) > 30                         # many distinct sequences, so a recording cannot anticipate them


def test_update_without_start_is_an_error(cfg):
    with pytest.raises(RuntimeError):
        ChallengeEvaluator(cfg, Baseline()).update(observation(), 0.0, None)
