"""Liveness session: the state machine that turns a stream of frames into a verdict."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from ..config import Config
from ..models.inference import PassiveLiveness, PassiveScore
from ..signals.blink import BlinkDetector
from ..signals.depth import DepthCue
from ..signals.rppg import PulseEstimator, PulseResult
from ..vision.quality import QualityReport, assess_quality
from ..vision.tracker import FaceObservation, FaceTracker
from .challenges import Baseline, Challenge, ChallengeEvaluator, ChallengeRecord, sample_challenges
from .fusion import INCONCLUSIVE, Evidence, Verdict, fuse


class Stage(str, Enum):
    POSITIONING = "positioning"
    RECENTER = "recenter"
    CHALLENGE = "challenge"
    HOLD = "hold"
    DONE = "done"


@dataclass
class LivenessResult:
    """Final, serialisable outcome of a session."""

    session_id: str
    mode: str
    decision: str
    score: float
    reasons: list[str]
    contributions: dict
    passive: dict
    challenges: list[dict]
    depth: dict
    pulse: dict
    blinks: int
    duration_s: float
    frames: int
    model: dict
    pulse_detail: PulseResult | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id, "mode": self.mode, "decision": self.decision, "score": round(self.score, 4),
            "reasons": self.reasons, "contributions": {k: round(v, 4) for k, v in self.contributions.items()},
            "signals": {"passive": self.passive, "challenges": self.challenges, "depth": self.depth, "pulse": self.pulse, "blinks": self.blinks},
            "duration_s": round(self.duration_s, 2), "frames": self.frames, "model": self.model,
        }


@dataclass
class SessionStatus:
    """What a user interface needs to draw after each frame."""

    stage: Stage
    prompt: str
    guidance: str = ""
    step: int = 0
    steps: int = 0
    challenge_progress: float = 0.0
    time_left: float = 0.0
    observation: FaceObservation | None = None
    quality: QualityReport | None = None
    passive: PassiveScore | None = None
    blinks: int = 0
    result: LivenessResult | None = None

    def as_dict(self) -> dict:
        return {
            "stage": self.stage.value, "prompt": self.prompt, "guidance": self.guidance, "step": self.step, "steps": self.steps,
            "challenge_progress": round(self.challenge_progress, 3), "time_left_s": round(self.time_left, 2),
            "face_detected": self.observation is not None, "blinks": self.blinks,
            "result": None if self.result is None else self.result.as_dict(),
        }


class LivenessEngine:
    """Owns the loaded models and creates sessions."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracker = FaceTracker(cfg)
        self.passive = PassiveLiveness(cfg)

    def new_session(self, mode: str = "interactive", seed: int | None = None, plan: list | None = None,
                    tracker=None, score_passive: bool = True) -> "LivenessSession":
        """Create a session.

        ``plan`` fixes the challenge sequence and ``tracker`` replaces the face
        tracker. Both exist for tests and experiments, production code leaves
        them unset so that challenges are random.
        """
        return LivenessSession(self, mode=mode, seed=seed, plan=plan, tracker=tracker, score_passive=score_passive)

    def threshold_for(self, mode: str) -> float:
        """Passive threshold of a session mode.

        A session with challenges has three more checks behind the passive
        model, so its passive threshold sits at a more forgiving operating
        point than the one used when the passive model decides almost alone.
        """
        point = self.cfg.passive.interactive_operating_point if mode == "interactive" else self.cfg.passive.operating_point
        return self.passive.threshold_at(str(point))

    def model_info(self, mode: str = "passive") -> dict:
        meta = self.passive.meta
        point = self.cfg.passive.interactive_operating_point if mode == "interactive" else self.cfg.passive.operating_point
        return {
            "name": meta.get("name", "PulseGateNet"), "run": meta.get("run", ""), "threshold": self.threshold_for(mode),
            "operating_point": str(point), "parameters": meta.get("parameters"),
        }

    def check_image(self, image_bgr: np.ndarray) -> dict:
        """Single image check. Uses the passive model only, the weakest mode of operation."""
        obs = self.tracker.fork().process_image(image_bgr)
        if obs is None:
            return {"decision": INCONCLUSIVE, "reasons": ["no_face"], "score": 0.0}
        quality = assess_quality(image_bgr, obs, self.cfg, check_pose=True)
        score = self.passive.score_frame(image_bgr, obs.box)
        if score is None:
            return {"decision": INCONCLUSIVE, "reasons": ["face_too_small"], "score": 0.0, "face_px": round(obs.size, 1)}
        decision = "live" if score.live >= self.passive.threshold else "not_live"
        return {
            "decision": decision, "score": round(score.live, 4), "threshold": round(self.passive.threshold, 4),
            "reasons": ["passive_model_accepted" if decision == "live" else f"passive_model_rejected:{score.species}"],
            "passive": score.as_dict(), "quality_issues": list(quality.issues),
            "face": {"box": [round(float(v), 1) for v in obs.box], "yaw": round(obs.pose.yaw, 1), "pitch": round(obs.pose.pitch, 1)},
        }

    def analyze_video(self, path: str, max_seconds: float | None = None, stride: int = 1, on_frame=None) -> LivenessResult:
        """Passive analysis of a recorded clip: texture, pulse, spontaneous blinking."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not np.isfinite(fps) or fps <= 1:
            fps = 30.0
        session = self.new_session(mode="passive")
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = index / fps
            index += 1
            if max_seconds is not None and t > max_seconds:
                break
            if (index - 1) % stride:
                continue
            status = session.update(frame, t)
            if on_frame is not None:
                on_frame(frame, t, status)
            if status.stage == Stage.DONE:
                break
        cap.release()
        return session.finalize()


class LivenessSession:
    """One liveness check.

    ``mode="interactive"`` runs random challenges and is meant for a live
    camera. ``mode="passive"`` skips the challenges and is meant for recorded
    clips or flows where the user cannot be prompted.
    """

    def __init__(self, engine: LivenessEngine, mode: str = "interactive", seed: int | None = None,
                 plan: list | None = None, tracker=None, score_passive: bool = True):
        if mode not in ("interactive", "passive"):
            raise ValueError("mode must be 'interactive' or 'passive'")
        self.engine = engine
        self.cfg = engine.cfg
        self.mode = mode
        self.session_id = uuid.uuid4().hex[:12]
        self.created = time.time()
        self._rng = np.random.default_rng(seed)
        self.passive_threshold = engine.threshold_for(mode)
        self.tracker = tracker if tracker is not None else engine.tracker.fork()
        self._score_passive = bool(score_passive)
        self.blinks = BlinkDetector.from_config(self.cfg)
        self.pulse = PulseEstimator.from_config(self.cfg)
        self.depth = DepthCue.from_config(self.cfg)
        self.stage = Stage.POSITIONING
        self.result: LivenessResult | None = None
        self.records: list[ChallengeRecord] = []
        count = int(self.cfg.challenge.count) if mode == "interactive" else 0
        if plan is not None:
            self.plan: list[Challenge] = [Challenge(c) for c in plan] if mode == "interactive" else []
        else:
            self.plan = sample_challenges(list(self.cfg.challenge.pool), count, self._rng) if count else []
        self._evaluator: ChallengeEvaluator | None = None
        self._baseline_samples: list[tuple[float, float, float]] = []
        self._ok_since: float | None = None
        self._t0: float | None = None
        self._t_last: float = 0.0
        self._last_seen: float | None = None
        self._prev: FaceObservation | None = None
        self._multi_face_s = 0.0
        self._neutral_since: float | None = None
        self._recent: list[tuple[float, float, float, float]] = []
        self._next_prompt_at = 0.0
        self._hold_since: float | None = None
        self._scores: list[tuple[float, PassiveScore]] = []
        self._integrity: list[str] = []
        self._last_quality_issue = ""
        self._frames = 0

    # ------------------------------------------------------------------ public
    @property
    def done(self) -> bool:
        return self.stage == Stage.DONE

    def update(self, frame_bgr: np.ndarray, t: float | None = None) -> SessionStatus:
        """Feed the next frame. ``t`` is the capture time in seconds (any origin)."""
        if self.done:
            return self._status(None, None, None)
        t = time.monotonic() if t is None else float(t)
        if self._t0 is None:
            self._t0 = t
        dt = max(0.0, t - self._t_last) if self._frames else 0.0
        self._t_last = t
        self._frames += 1
        obs = self.tracker.process(frame_bgr, t)
        quality = assess_quality(frame_bgr, obs, self.cfg, check_pose=self.stage in (Stage.POSITIONING, Stage.HOLD))
        if not quality.ok:
            self._last_quality_issue = quality.issues[0]
        score = None
        blink = None
        if obs is not None:
            self._check_integrity(obs, t, dt)
            blink = self.blinks.update(t, obs.ear, obs.pose.yaw, obs.pose.pitch)
            self.pulse.add(t, frame_bgr, obs.landmarks)
            if self.stage != Stage.POSITIONING:
                self.depth.add(obs.pose.yaw, obs.pose.pitch, obs.landmarks)
            if self._score_passive and self._scorable(obs, quality):
                score = self.engine.passive.score_frame(frame_bgr, obs.box)
                if score is not None:
                    self._scores.append((t, score))
            self._last_seen = t
            self._prev = obs
        elif self._last_seen is not None and self.stage != Stage.POSITIONING:
            if t - self._last_seen > float(self.cfg.session.max_face_lost_s) and self.mode == "interactive":
                self._integrity.append("face_lost")

        if self._integrity:
            self._finish(t)
        elif t - self._t0 > float(self.cfg.session.max_duration_s):
            self._finish(t, timed_out=True)
        else:
            self._advance(obs, quality, blink, t)
        return self._status(obs, quality, score)

    def finalize(self) -> LivenessResult:
        """Close the session with whatever evidence exists (end of a clip, user abort)."""
        if not self.done:
            pending = self.mode == "interactive" and (len(self.records) < len(self.plan) or self.records[-1].passed is None)
            self._finish(self._t_last, incomplete=pending)
        assert self.result is not None
        return self.result

    # ------------------------------------------------------------------ internals
    def _scorable(self, obs: FaceObservation, quality: QualityReport) -> bool:
        q = self.cfg.quality
        blocking = {"too_far", "too_dark", "too_bright", "blurry"}
        if blocking & set(quality.issues):
            return False
        return abs(obs.pose.yaw) <= q.max_abs_yaw_deg and abs(obs.pose.pitch) <= q.max_abs_pitch_deg

    def _check_integrity(self, obs: FaceObservation, t: float, dt: float) -> None:
        if self.mode != "interactive" or self.stage == Stage.POSITIONING:
            return
        if self._prev is not None and 0 < dt < 0.5:
            jump = float(np.linalg.norm(obs.center - self._prev.center)) / max(self._prev.size, 1.0)
            resize = obs.size / max(self._prev.size, 1.0)
            if jump > 0.7 or resize > 1.6 or resize < 0.6:
                self._integrity.append("track_discontinuity")
        if obs.num_faces > 1 and not self.cfg.session.allow_multiple_faces:
            self._multi_face_s += dt
            if self._multi_face_s > 1.0:
                self._integrity.append("multiple_faces")

    def _advance(self, obs: FaceObservation | None, quality: QualityReport, blink, t: float) -> None:
        c = self.cfg.challenge
        if self.stage == Stage.POSITIONING:
            if obs is not None and quality.ok:
                if self._ok_since is None:
                    self._ok_since = t
                    self._baseline_samples = []
                self._baseline_samples.append((obs.pose.yaw, obs.pose.pitch, obs.mouth_open))
                if t - self._ok_since >= float(c.positioning_s) and len(self._baseline_samples) >= 5:
                    base = np.median(np.asarray(self._baseline_samples), axis=0)
                    self._evaluator = ChallengeEvaluator(self.cfg, Baseline(float(base[0]), float(base[1]), float(base[2])))
                    self._enter_recenter_or_hold(t)
            else:
                self._ok_since = None
        elif self.stage == Stage.RECENTER:
            # The next prompt waits until the head is frontal and at rest. The resting pose is
            # measured again each time, because people rarely return to exactly where they started.
            if obs is not None:
                self._recent.append((t, obs.pose.yaw, obs.pose.pitch, obs.mouth_open))
            self._recent = [r for r in self._recent if t - r[0] <= 0.45]
            neutral = False
            if obs is not None and len(self._recent) >= 3 and self._evaluator.is_roughly_frontal(obs):
                recent = np.asarray(self._recent)
                neutral = float(np.ptp(recent[:, 1])) <= 5.0 and float(np.ptp(recent[:, 2])) <= 5.0
            if neutral:
                self._neutral_since = t if self._neutral_since is None else self._neutral_since
            else:
                self._neutral_since = None
            if self._neutral_since is not None and t - self._neutral_since >= 0.3 and t >= self._next_prompt_at:
                rest = np.median(np.asarray(self._recent)[:, 1:], axis=0)
                self._evaluator.baseline = Baseline(float(rest[0]), float(rest[1]), float(rest[2]))
                challenge = self.plan[len(self.records)]
                self.records.append(self._evaluator.start(challenge, t))
                self.stage = Stage.CHALLENGE
        elif self.stage == Stage.CHALLENGE:
            record = self.records[-1]
            if obs is not None:
                self._evaluator.update(obs, t, blink)
            elif t > record.deadline:
                record.passed, record.finished_at, record.detail = False, t, "timeout"
            if record.passed is True:
                self._enter_recenter_or_hold(t)
            elif record.passed is False:
                self._finish(t)
        elif self.stage == Stage.HOLD:
            if self._hold_since is None:
                self._hold_since = t
            enough_frames = len(self._scores) >= int(self.cfg.passive.min_frames)
            if self.mode == "interactive":
                held = t - self._hold_since
                pulse_ready = self.pulse.seconds >= float(self.cfg.rppg.min_seconds)
                if enough_frames and (pulse_ready or held >= 3.0) and held >= 0.8:
                    self._finish(t)
            elif t - self._hold_since >= float(self.cfg.session.passive_window_s) and enough_frames:
                self._finish(t)

    def _enter_recenter_or_hold(self, t: float) -> None:
        if len(self.records) < len(self.plan):
            self.stage = Stage.RECENTER
            self._neutral_since = None
            self._recent = []
            # an unpredictable pause, so that the timing of prompts cannot be anticipated
            self._next_prompt_at = t + float(self._rng.uniform(0.4, 1.2))
        else:
            self.stage = Stage.HOLD
            self._hold_since = None

    def _aggregate_passive(self) -> tuple[float | None, dict]:
        if not self._scores:
            return None, {"frames": 0}
        probs = np.sort(np.array([s.live for _, s in self._scores]))
        trim = int(len(probs) * float(self.cfg.passive.trim_fraction))
        core = probs[trim: len(probs) - trim] if len(probs) - 2 * trim >= 1 else probs
        value = float(core.mean()) if str(self.cfg.passive.aggregate) == "trimmed_mean" else float(np.median(probs))
        species: dict[str, int] = {}
        for _, s in self._scores:
            species[s.species] = species.get(s.species, 0) + 1
        threshold = self.passive_threshold
        return value, {
            "score": round(value, 4), "threshold": round(threshold, 4), "frames": int(len(probs)),
            "min": round(float(probs[0]), 4), "max": round(float(probs[-1]), 4),
            "share_below_threshold": round(float((probs < threshold).mean()), 3),
            "context": round(float(np.mean([s.context for _, s in self._scores])), 4),
            "texture": round(float(np.mean([s.texture for _, s in self._scores])), 4),
            "predicted_class": max(species, key=species.get),
        }

    def _finish(self, t: float, timed_out: bool = False, incomplete: bool = False) -> None:
        duration = 0.0 if self._t0 is None else t - self._t0
        passive_value, passive_info = self._aggregate_passive()
        depth = self.depth.evaluate()
        pulse = self.pulse.estimate()
        passed = sum(1 for r in self.records if r.passed)
        failed = next((r for r in self.records if r.passed is False), None)
        total = len(self.plan)
        evidence = Evidence(
            passive=passive_value, passive_frames=passive_info.get("frames", 0), passive_threshold=self.passive_threshold,
            challenges_passed=passed, challenges_total=total,
            failed_challenge="" if failed is None else f"{failed.challenge.value}:{failed.detail}",
            depth_score=depth.score if depth.valid else None, depth_residual=depth.residual if depth.valid else None,
            pulse_score=pulse.score if pulse.valid else None,
            blink_seen=(self.blinks.count > 0) if (self.mode == "passive" and duration >= 5.0) else None,
            integrity_issues=list(dict.fromkeys(self._integrity)), quality_issue=self._last_quality_issue,
        )
        verdict: Verdict = fuse(evidence, self.cfg)
        unfinished = total > 0 and failed is None and passed < total
        if unfinished and not evidence.integrity_issues and (timed_out or incomplete):
            # Nothing was failed, the session simply did not get through its prompts: not a rejection.
            why = "session_timeout" if timed_out else "session_incomplete"
            verdict = Verdict(INCONCLUSIVE, min(verdict.score, float(self.cfg.fusion.live_threshold) - 1e-3), [why], verdict.contributions)
        self.result = LivenessResult(
            session_id=self.session_id, mode=self.mode, decision=verdict.decision, score=verdict.score, reasons=verdict.reasons,
            contributions=verdict.contributions, passive=passive_info, challenges=[r.as_dict() for r in self.records],
            depth=depth.as_dict(), pulse=pulse.as_dict(), blinks=self.blinks.count, duration_s=duration, frames=self._frames,
            model=self.engine.model_info(self.mode), pulse_detail=pulse,
        )
        self.stage = Stage.DONE

    def _status(self, obs, quality, score) -> SessionStatus:
        steps = len(self.plan)
        step = min(len(self.records), steps)
        prompt, guidance, progress, left = "", "", 0.0, 0.0
        if self.stage == Stage.POSITIONING:
            prompt = "Position your face in the oval"
            guidance = quality.guidance if quality is not None and not quality.ok else "Hold still"
            if self._ok_since is not None:
                progress = float(np.clip((self._t_last - self._ok_since) / float(self.cfg.challenge.positioning_s), 0, 1))
        elif self.stage == Stage.RECENTER:
            prompt = "Look straight at the camera"
            guidance = "Keep your head still for a moment"
        elif self.stage == Stage.CHALLENGE:
            record = self.records[-1]
            prompt, progress, left = record.prompt, record.progress, max(0.0, record.deadline - self._t_last)
        elif self.stage == Stage.HOLD:
            if self.mode != "interactive":
                prompt = "Analysing"
            elif obs is not None and self._evaluator is not None and not self._evaluator.is_roughly_frontal(obs):
                prompt = "Look straight at the camera"
            else:
                prompt = "Hold still"
        elif self.result is not None:
            prompt = {"live": "Verified: live person", "not_live": "Not verified", "inconclusive": "Could not decide, please retry"}[self.result.decision]
        return SessionStatus(
            stage=self.stage, prompt=prompt, guidance=guidance, step=step, steps=steps, challenge_progress=progress, time_left=left,
            observation=obs, quality=quality, passive=score, blinks=self.blinks.count, result=self.result,
        )
