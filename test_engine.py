"""Integration tests on real footage. They need the trained passive model in ``models``."""
import json

import numpy as np
import pytest

from pulsegate.data.attacks.common import face_ref_from_landmarks
from pulsegate.data.attacks.simulator import Simulator
from pulsegate.engine.session import Stage


def feed(session, frames, fps, start=0, stop=None):
    status = None
    for i, frame in enumerate(frames[start:stop]):
        status = session.update(frame, i / fps)
        if status.stage == Stage.DONE:
            break
    return status


def test_single_image_check_reports_a_decision(engine, face_image):
    result = engine.check_image(face_image)
    assert result["decision"] in ("live", "not_live")
    assert 0.0 <= result["score"] <= 1.0
    assert set(result["passive"]) >= {"live", "context", "texture", "species"}
    assert json.dumps(result)                                   # serialisable


def test_image_without_a_face_is_inconclusive(engine):
    result = engine.check_image(np.full((360, 480, 3), 90, np.uint8))
    assert result["decision"] == "inconclusive" and result["reasons"] == ["no_face"]


def test_scoring_is_deterministic(engine, face_image):
    assert engine.check_image(face_image)["score"] == engine.check_image(face_image)["score"]


def test_passive_session_on_real_footage_is_accepted(engine, clip_frames):
    frames, fps = clip_frames
    session = engine.new_session("passive")
    feed(session, frames, fps, stop=int(12 * fps))
    result = session.finalize()
    assert result.mode == "passive" and result.challenges == []
    assert result.passive["frames"] >= engine.cfg.passive.min_frames
    assert result.decision == "live", result.as_dict()
    assert result.passive["score"] > engine.passive.threshold


def test_scripted_interactive_session_passes_on_real_footage(engine, clip_frames):
    """The person in the clip raises her chin, turns right and turns left, in that order."""
    frames, fps = clip_frames
    session = engine.new_session("interactive", seed=3, plan=["look_up", "turn_right", "turn_left"])
    status = feed(session, frames, fps)
    result = session.finalize()
    assert status.stage == Stage.DONE
    assert [c["challenge"] for c in result.challenges] == ["look_up", "turn_right", "turn_left"]
    assert all(c["passed"] for c in result.challenges), result.challenges
    assert result.depth["valid"] and result.depth["score"] > 0.5        # a real head has depth
    assert result.decision == "live", result.as_dict()


def test_the_same_footage_fails_when_the_prompts_do_not_match(engine, clip_frames):
    frames, fps = clip_frames
    session = engine.new_session("interactive", seed=3, plan=["turn_left", "open_mouth", "blink_twice"])
    feed(session, frames, fps)
    result = session.finalize()
    assert result.decision == "not_live"
    assert result.reasons[0].startswith("challenge_failed")


def test_replayed_footage_is_rejected(engine, tracker, clip_frames):
    """The same clip shown on a simulated monitor in front of the camera."""
    frames, fps = clip_frames
    h, w = frames[0].shape[:2]
    sim = Simulator((w, h))
    rng = np.random.default_rng(21)
    params = sim.sample("replay_monitor", rng)
    params.medium.face_width = 170.0
    params.medium.pitch = 0.8
    background = frames[-1]
    source_tracker = tracker.fork()
    bg = source_tracker.process_image(background)
    bg_face = face_ref_from_landmarks(bg.landmarks, bg.box)
    session = engine.new_session("passive")
    for i, frame in enumerate(frames[: int(8 * fps)]):
        obs = source_tracker.process(frame, i / fps)
        face = face_ref_from_landmarks(obs.landmarks, obs.box)
        attack, _ = sim.render(frame, face, params, rng, background, bg_face)
        session.update(attack, i / fps)
    result = session.finalize()
    assert result.decision == "not_live", result.as_dict()
    assert result.passive["score"] < engine.passive.threshold


def test_session_without_a_face_is_inconclusive(engine):
    session = engine.new_session("passive")
    blank = np.full((432, 768, 3), 100, np.uint8)
    for i in range(30):
        session.update(blank, i / 12)
    result = session.finalize()
    assert result.decision == "inconclusive"
    assert result.reasons[0].startswith("insufficient_quality")


def test_losing_the_face_mid_session_ends_it(engine, clip_frames):
    frames, fps = clip_frames
    session = engine.new_session("interactive", seed=1)
    blank = np.full_like(frames[0], 100)
    t = 0.0
    for frame in frames[: int(4 * fps)]:
        session.update(frame, t)
        t += 1 / fps
    assert session.stage != Stage.POSITIONING
    for _ in range(int(3 * fps)):
        status = session.update(blank, t)
        t += 1 / fps
    assert status.stage == Stage.DONE
    assert status.result.decision == "not_live" and "integrity:face_lost" in status.result.reasons


def test_swapping_the_face_position_is_flagged(engine, clip_frames):
    frames, fps = clip_frames
    session = engine.new_session("interactive", seed=1)
    t = 0.0
    for frame in frames[: int(4 * fps)]:
        session.update(frame, t)
        t += 1 / fps
    shifted = np.roll(frames[int(4 * fps)], 220, axis=1)          # the face jumps sideways between two frames
    for _ in range(3):                                            # the tracker needs a frame to find the face again
        status = session.update(shifted, t)
        t += 1 / fps
        if status.stage == Stage.DONE:
            break
    assert status.stage == Stage.DONE and "integrity:track_discontinuity" in status.result.reasons


def test_sessions_draw_different_challenges(engine):
    plans = {tuple(c.value for c in engine.new_session("interactive").plan) for _ in range(40)}
    assert len(plans) > 10
    assert engine.new_session("passive").plan == []


def test_result_is_complete_and_serialisable(engine, clip_frames):
    frames, fps = clip_frames
    session = engine.new_session("passive")
    feed(session, frames, fps, stop=int(11 * fps))
    payload = session.finalize().as_dict()
    assert set(payload) == {"session_id", "mode", "decision", "score", "reasons", "contributions", "signals", "duration_s", "frames", "model"}
    assert set(payload["signals"]) == {"passive", "challenges", "depth", "pulse", "blinks"}
    assert payload["model"]["threshold"] == pytest.approx(engine.threshold_for("passive"))
    json.dumps(payload)


def test_interactive_sessions_use_a_more_forgiving_passive_threshold(engine):
    assert engine.threshold_for("interactive") < engine.threshold_for("passive")
    assert engine.new_session("interactive").passive_threshold == engine.threshold_for("interactive")
    assert engine.new_session("passive").passive_threshold == engine.passive.threshold


def test_invalid_mode_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.new_session("telepathic")


def test_video_file_analysis(engine, clip_path):
    result = engine.analyze_video(str(clip_path), max_seconds=11)
    assert result.decision == "live"
    assert result.frames > 100
    with pytest.raises(FileNotFoundError):
        engine.analyze_video("does_not_exist.mp4")
