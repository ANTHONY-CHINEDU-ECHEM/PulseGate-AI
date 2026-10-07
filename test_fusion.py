import pytest

from pulsegate.engine.fusion import INCONCLUSIVE, LIVE, NOT_LIVE, Evidence, fuse, passive_evidence


def genuine(**changes) -> Evidence:
    base = dict(passive=0.97, passive_frames=40, passive_threshold=0.6, challenges_passed=3, challenges_total=3,
                depth_score=1.0, depth_residual=0.06, pulse_score=0.4)
    base.update(changes)
    return Evidence(**base)


def test_genuine_session_is_accepted(cfg):
    verdict = fuse(genuine(), cfg)
    assert verdict.decision == LIVE and verdict.score > cfg.fusion.live_threshold
    assert abs(sum(verdict.contributions.values()) - verdict.score) < 1e-9


def test_failed_challenge_is_disqualifying_even_with_a_perfect_passive_score(cfg):
    verdict = fuse(genuine(passive=1.0, challenges_passed=2, failed_challenge="turn_left:timeout"), cfg)
    assert verdict.decision == NOT_LIVE
    assert verdict.reasons == ["challenge_failed:turn_left:timeout"]
    assert verdict.score < cfg.fusion.live_threshold


def test_passive_floor_is_disqualifying_even_when_challenges_pass(cfg):
    verdict = fuse(genuine(passive=0.05), cfg)
    assert verdict.decision == NOT_LIVE and "passive_model_rejected" in verdict.reasons


def test_flat_surface_is_disqualifying(cfg):
    verdict = fuse(genuine(depth_score=0.0, depth_residual=0.008), cfg)
    assert verdict.decision == NOT_LIVE and "flat_surface" in verdict.reasons


def test_integrity_problem_ends_the_session(cfg):
    verdict = fuse(genuine(integrity_issues=["face_lost"]), cfg)
    assert verdict.decision == NOT_LIVE and verdict.reasons == ["integrity:face_lost"]


def test_too_few_frames_is_inconclusive_not_a_rejection(cfg):
    verdict = fuse(genuine(passive_frames=2, quality_issue="too_dark"), cfg)
    assert verdict.decision == INCONCLUSIVE and verdict.reasons == ["insufficient_quality:too_dark"]


def test_missing_signals_are_left_out_not_counted_as_zero(cfg):
    with_all = fuse(genuine(pulse_score=1.0, depth_score=1.0), cfg)
    without = fuse(genuine(pulse_score=None, depth_score=None), cfg)
    assert without.decision == LIVE
    assert "pulse" not in without.contributions and "depth" not in without.contributions
    assert without.score == pytest.approx(with_all.score, abs=0.05)


def test_weak_pulse_alone_does_not_reject_a_real_user(cfg):
    assert fuse(genuine(pulse_score=0.0), cfg).decision == LIVE


def test_borderline_passive_score_needs_the_other_signals(cfg):
    borderline = dict(passive=0.5, passive_threshold=0.6)
    assert fuse(genuine(**borderline), cfg).decision == LIVE
    weak = fuse(genuine(**borderline, depth_score=0.2, pulse_score=0.0), cfg)
    assert weak.decision == NOT_LIVE and weak.reasons[0].startswith("low_confidence")


def test_passive_mode_uses_spontaneous_blinking(cfg):
    still = Evidence(passive=0.7, passive_frames=30, passive_threshold=0.6, blink_seen=False)
    blinking = Evidence(passive=0.7, passive_frames=30, passive_threshold=0.6, blink_seen=True)
    assert fuse(blinking, cfg).score > fuse(still, cfg).score
    assert "blink" in fuse(blinking, cfg).contributions


def test_passive_evidence_maps_the_threshold_to_one_half():
    assert passive_evidence(0.8, 0.8) == pytest.approx(0.5)
    assert passive_evidence(1.0, 0.8) == pytest.approx(1.0)
    assert passive_evidence(0.0, 0.8) == pytest.approx(0.0)
    assert passive_evidence(0.4, 0.8) == pytest.approx(0.25)
    assert passive_evidence(0.9, 0.8) == pytest.approx(0.75)
