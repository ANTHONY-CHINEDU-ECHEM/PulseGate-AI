"""Active challenge and response.

The system asks for a short random sequence of actions and checks that each one
happens after its prompt and inside its time window. A photograph cannot act,
and a recorded video cannot know which action will be requested next or when.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..signals.blink import BlinkEvent
from ..vision.tracker import FaceObservation


class Challenge(str, Enum):
    BLINK_TWICE = "blink_twice"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    OPEN_MOUTH = "open_mouth"
    LOOK_UP = "look_up"


PROMPTS = {
    Challenge.BLINK_TWICE: "Blink twice",
    Challenge.TURN_LEFT: "Turn your head to your left",
    Challenge.TURN_RIGHT: "Turn your head to your right",
    Challenge.OPEN_MOUTH: "Open your mouth",
    Challenge.LOOK_UP: "Raise your chin",
}


@dataclass
class Baseline:
    """Resting pose measured while the user positions their face."""

    yaw: float = 0.0
    pitch: float = 0.0
    mouth: float = 0.0


@dataclass
class ChallengeRecord:
    challenge: Challenge
    issued_at: float
    deadline: float
    passed: bool | None = None
    finished_at: float | None = None
    detail: str = ""
    progress: float = 0.0
    _hits: int = field(default=0, repr=False)
    _blinks: int = field(default=0, repr=False)

    @property
    def prompt(self) -> str:
        return PROMPTS[self.challenge]

    @property
    def response_time(self) -> float | None:
        return None if self.finished_at is None else self.finished_at - self.issued_at

    def as_dict(self) -> dict:
        return {
            "challenge": self.challenge.value, "passed": bool(self.passed), "detail": self.detail,
            "response_time_s": None if self.response_time is None else round(self.response_time, 2),
        }


def sample_challenges(pool: list[str], count: int, rng: np.random.Generator) -> list[Challenge]:
    """Random sequence without repeats.

    Left and right turns never follow each other directly, so returning from
    one turn cannot be mistaken for the other.
    """
    choices = [Challenge(name) for name in pool]
    count = min(count, len(choices))
    turns = {Challenge.TURN_LEFT, Challenge.TURN_RIGHT}
    for _ in range(50):
        order = [choices[i] for i in rng.permutation(len(choices))[:count]]
        if all(not (a in turns and b in turns) for a, b in zip(order, order[1:])):
            return order
    return order


class ChallengeEvaluator:
    """Judges one challenge at a time from the stream of face observations."""

    def __init__(self, cfg, baseline: Baseline):
        self.cfg = cfg.challenge
        self.baseline = baseline
        self.current: ChallengeRecord | None = None

    def start(self, challenge: Challenge, t: float) -> ChallengeRecord:
        self.current = ChallengeRecord(challenge, issued_at=t, deadline=t + float(self.cfg.timeout_s))
        return self.current

    def is_roughly_frontal(self, obs: FaceObservation) -> bool:
        """Facing the camera with the mouth closed, wherever the head happens to rest."""
        c = self.cfg
        limit = 1.6 * float(c.neutral_yaw_deg)
        return (
            abs(obs.pose.yaw) <= limit and abs(obs.pose.pitch) <= limit
            and obs.mouth_open < max(0.5 * c.open_mouth_ratio, self.baseline.mouth + 0.08)
        )

    def is_neutral(self, obs: FaceObservation) -> bool:
        """True when the head is back at rest, required before the next prompt appears."""
        c = self.cfg
        return (
            abs(obs.pose.yaw - self.baseline.yaw) <= c.neutral_yaw_deg
            and abs(obs.pose.pitch - self.baseline.pitch) <= c.neutral_yaw_deg
            and obs.mouth_open < max(0.5 * c.open_mouth_ratio, self.baseline.mouth + 0.08)
        )

    def update(self, obs: FaceObservation, t: float, blink: BlinkEvent | None) -> ChallengeRecord:
        rec = self.current
        if rec is None:
            raise RuntimeError("no challenge is running")
        if rec.passed is not None:
            return rec
        c = self.cfg
        if t > rec.deadline:
            rec.passed, rec.finished_at, rec.detail = False, t, "timeout"
            return rec
        # A response that is quicker than human reaction time was not caused by the prompt.
        reacting = t >= rec.issued_at + float(c.settle_s)
        d_yaw = obs.pose.yaw - self.baseline.yaw
        d_pitch = obs.pose.pitch - self.baseline.pitch
        hit = False
        kind = rec.challenge
        if kind in (Challenge.TURN_LEFT, Challenge.TURN_RIGHT):
            sign = 1.0 if kind == Challenge.TURN_LEFT else -1.0      # positive yaw is the subject's left
            rec.progress = float(np.clip(sign * d_yaw / c.turn_yaw_deg, 0.0, 1.0))
            if reacting and sign * d_yaw <= -float(c.turn_yaw_deg):
                rec.passed, rec.finished_at, rec.detail = False, t, "turned the wrong way"
                return rec
            hit = sign * d_yaw >= c.turn_yaw_deg
        elif kind == Challenge.LOOK_UP:
            rec.progress = float(np.clip(d_pitch / c.look_up_pitch_deg, 0.0, 1.0))
            hit = d_pitch >= c.look_up_pitch_deg
        elif kind == Challenge.OPEN_MOUTH:
            rec.progress = float(np.clip(obs.mouth_open / c.open_mouth_ratio, 0.0, 1.0))
            hit = obs.mouth_open >= c.open_mouth_ratio
        elif kind == Challenge.BLINK_TWICE:
            if blink is not None and blink.start >= rec.issued_at + 0.5 * float(c.settle_s):
                rec._blinks += 1
            rec.progress = min(1.0, rec._blinks / 2.0)
            if rec._blinks >= 2:
                rec.passed, rec.finished_at, rec.detail = True, t, "two blinks"
            return rec
        if hit and reacting:
            rec._hits += 1
        elif not hit:
            rec._hits = 0
        if rec._hits >= 2:      # two consecutive frames guard against a single noisy estimate
            rec.passed, rec.finished_at, rec.detail = True, t, "action observed"
        return rec
