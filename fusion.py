"""Decision fusion: several independent signals, one auditable verdict."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LIVE, NOT_LIVE, INCONCLUSIVE = "live", "not_live", "inconclusive"


@dataclass
class Evidence:
    """Signals gathered during one session. ``None`` means the signal is not available."""

    passive: float | None = None            # aggregated probability from the passive model
    passive_frames: int = 0
    passive_threshold: float = 0.5
    challenges_passed: int = 0
    challenges_total: int = 0
    failed_challenge: str = ""
    depth_score: float | None = None
    depth_residual: float | None = None
    pulse_score: float | None = None
    blink_seen: bool | None = None          # spontaneous blinking, used when no challenges ran
    integrity_issues: list[str] = field(default_factory=list)
    quality_issue: str = ""


@dataclass
class Verdict:
    decision: str
    score: float
    reasons: list[str]
    contributions: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "decision": self.decision, "score": round(self.score, 4), "reasons": self.reasons,
            "contributions": {k: round(v, 4) for k, v in self.contributions.items()},
        }


def passive_evidence(prob: float, threshold: float) -> float:
    """Rescale a probability so that the operating threshold maps to one half."""
    threshold = float(np.clip(threshold, 1e-3, 1 - 1e-3))
    if prob >= threshold:
        return 0.5 + 0.5 * (prob - threshold) / (1.0 - threshold)
    return 0.5 * prob / threshold


def fuse(evidence: Evidence, cfg) -> Verdict:
    """Combine the evidence.

    Hard gates come first, because some findings are disqualifying on their own
    (a failed challenge, a flat face, a passive score far below its threshold).
    What is left is a weighted average over the signals that are available.

    Each signal covers a different attack. Challenges stop photographs and
    recordings, the depth cue stops flat media that are tilted to fake a turn,
    and the passive model is the only check that sees a screen relaying a live
    accomplice. With the default weights a passive score just under its
    threshold is accepted only when every other check is passed, and a score
    further below needs a detected pulse on top.
    """
    f = cfg.fusion
    weights = dict(f.weights)
    reasons: list[str] = []

    if evidence.integrity_issues:
        return Verdict(NOT_LIVE, 0.0, [f"integrity:{issue}" for issue in evidence.integrity_issues], {})
    if evidence.passive is None or evidence.passive_frames < int(cfg.passive.min_frames):
        why = evidence.quality_issue or "too_few_usable_frames"
        return Verdict(INCONCLUSIVE, 0.0, [f"insufficient_quality:{why}"], {})

    terms: dict[str, tuple[float, float]] = {}       # name -> (weight, evidence in 0..1)
    terms["passive"] = (float(weights["passive"]), passive_evidence(evidence.passive, evidence.passive_threshold))
    if evidence.challenges_total > 0:
        terms["challenge"] = (float(weights["challenge"]), evidence.challenges_passed / evidence.challenges_total)
    elif evidence.blink_seen is not None:
        terms["blink"] = (0.6 * float(weights["challenge"]), 1.0 if evidence.blink_seen else 0.0)
    if evidence.depth_score is not None:
        terms["depth"] = (float(weights["depth"]), float(evidence.depth_score))
    if evidence.pulse_score is not None:
        terms["pulse"] = (float(weights["pulse"]), float(evidence.pulse_score))
    total = sum(w for w, _ in terms.values())
    score = sum(w * e for w, e in terms.values()) / total
    contributions = {name: w * e / total for name, (w, e) in terms.items()}

    if evidence.passive < float(f.passive_floor_ratio) * evidence.passive_threshold:
        reasons.append("passive_model_rejected")
    if bool(f.require_challenges) and evidence.challenges_total > 0 and evidence.challenges_passed < evidence.challenges_total:
        reasons.append(f"challenge_failed:{evidence.failed_challenge or 'unknown'}")
    if evidence.depth_score is not None and evidence.depth_score <= 0.0:
        reasons.append("flat_surface")
    if reasons:
        return Verdict(NOT_LIVE, min(score, float(f.live_threshold) - 1e-3), reasons, contributions)
    if score >= float(f.live_threshold):
        notes = ["all_checks_passed"]
        if evidence.pulse_score is not None and evidence.pulse_score >= 0.5:
            notes.append("pulse_detected")
        return Verdict(LIVE, score, notes, contributions)
    weakest = min(terms.items(), key=lambda kv: kv[1][1])[0]
    return Verdict(NOT_LIVE, score, [f"low_confidence:{weakest}"], contributions)
