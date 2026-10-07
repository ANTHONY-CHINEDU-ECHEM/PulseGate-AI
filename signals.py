"""Validation of the non learned signals on real data.

* Depth cue and head pose: MUCT photographed every subject with five cameras at
  once, which gives real 3D view changes with known geometry. Tilting the
  frontal photograph in software gives the matching flat attack.
* Challenge protocol: recorded footage of people moving their heads is played
  against random challenge sequences to measure how often a recording passes.
* Pulse: a synthetic heartbeat of known rate is injected into the skin pixels of
  real footage and has to be recovered through the whole pipeline.
* Blinks: detections on real footage, exported for inspection.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..data.attacks.common import plane_homography
from ..data.muct import index_muct
from ..engine.session import LivenessEngine, Stage
from ..signals.blink import BlinkDetector
from ..signals.depth import planar_residual
from ..signals.rppg import PulseEstimator
from ..vision.geometry import FACE_OVAL
from ..vision.tracker import FaceObservation, FaceTracker
from .metrics import roc_auc


def depth_validation(cfg: Config, subjects: int = 276, seed: int = 0, log=print) -> dict:
    """Real multi view pairs against software tilted photographs."""
    jpg_dir = cfg.path("paths.muct_dir") / "jpg"
    index = index_muct(jpg_dir)
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(index["subject"].unique())[:subjects]
    tracker = FaceTracker(cfg)
    pose_rows: list[tuple] = []
    real: list[dict] = []
    flat: list[dict] = []
    for n, subject in enumerate(chosen, 1):
        rows = index[index["subject"] == subject]
        lighting = rows["lighting"].iloc[0]
        views: dict[str, tuple[FaceObservation, np.ndarray]] = {}
        for camera in "abcde":
            match = rows[(rows["camera"] == camera) & (rows["lighting"] == lighting)]
            if match.empty:
                continue
            image = cv2.imread(str(jpg_dir / match["file"].iloc[0]))
            obs = tracker.process_image(image)
            if obs is not None:
                views[camera] = (obs, image)
                pose_rows.append((camera, obs.pose.yaw, obs.pose.pitch, obs.pose.roll))
        if "a" not in views:
            continue
        front, image = views["a"]
        for camera in "bcde":
            if camera in views:
                other = views[camera][0]
                real.append({
                    "pair": f"a{camera}", "residual": planar_residual(front.landmarks, other.landmarks),
                    "rotation": float(np.hypot(other.pose.yaw - front.pose.yaw, other.pose.pitch - front.pose.pitch)),
                })
        h, w = image.shape[:2]
        for _ in range(2):
            tilt = np.radians(rng.choice([-1, 1]) * rng.uniform(15, 30))
            nod = np.radians(rng.normal(0, 6))
            homography = plane_homography((front.center[0], front.center[1]), 1.0, tilt, nod, 0.0, (w / 2, h / 2), focal=1.25 * max(w, h))
            warped = cv2.warpPerspective(image, homography, (w, h), borderMode=cv2.BORDER_REPLICATE)
            obs = tracker.process_image(warped)
            if obs is not None:
                flat.append({
                    "tilt_deg": float(abs(np.degrees(tilt))), "residual": planar_residual(front.landmarks, obs.landmarks),
                    "yaw_shift": float(abs(obs.pose.yaw - front.pose.yaw)), "pitch_shift": float(abs(obs.pose.pitch - front.pose.pitch)),
                })
        if n % 50 == 0:
            log(f"  depth validation: {n} of {len(chosen)} subjects")

    threshold = float(cfg.depth.residual_threshold)
    min_rot = float(cfg.depth.min_yaw_delta_deg)
    strong = [r for r in real if r["rotation"] >= min_rot]
    flat_res = np.array([f["residual"] for f in flat])
    real_res = np.array([r["residual"] for r in strong])
    labels = np.concatenate([np.ones(len(real_res)), np.zeros(len(flat_res))])
    poses = {}
    for camera in "abcde":
        rows = np.array([(y, p, r) for c, y, p, r in pose_rows if c == camera])
        if len(rows):
            poses[camera] = {"yaw_mean": float(rows[:, 0].mean()), "yaw_std": float(rows[:, 0].std()),
                             "pitch_mean": float(rows[:, 1].mean()), "pitch_std": float(rows[:, 1].std()), "images": int(len(rows))}
    by_pair = {}
    for pair in ("ab", "ac", "ad", "ae"):
        values = np.array([r["residual"] for r in real if r["pair"] == pair])
        rot = np.array([r["rotation"] for r in real if r["pair"] == pair])
        if len(values):
            by_pair[pair] = {"median_residual": float(np.median(values)), "median_rotation_deg": float(np.median(rot)), "pairs": int(len(values))}
    turn = float(cfg.challenge.turn_yaw_deg)
    return {
        "subjects": int(len(chosen)), "threshold": threshold, "min_rotation_deg": min_rot,
        "pose_by_camera": poses, "real_by_pair": by_pair,
        "real_pairs": int(len(real_res)), "flat_pairs": int(len(flat_res)),
        "auc": roc_auc(labels, np.concatenate([real_res, flat_res])),
        "real_median_residual": float(np.median(real_res)), "flat_median_residual": float(np.median(flat_res)),
        "real_above_threshold": float((real_res >= threshold).mean()), "flat_above_threshold": float((flat_res >= threshold).mean()),
        "real_scored_flat": float((real_res <= 0.5 * threshold).mean()), "flat_scored_flat": float((flat_res <= 0.5 * threshold).mean()),
        "flat_median_yaw_shift_deg": float(np.median([f["yaw_shift"] for f in flat])),
        "flat_median_tilt_deg": float(np.median([f["tilt_deg"] for f in flat])),
        "flat_reaching_turn_threshold": float(np.mean([f["yaw_shift"] >= turn for f in flat])),
        "samples": {"real": [round(float(v), 5) for v in real_res], "flat": [round(float(v), 5) for v in flat_res],
                    "real_rotation": [round(r["rotation"], 2) for r in strong]},
    }


class CachedTracker:
    """Replays stored observations, so thousands of sessions can run over one clip quickly."""

    def __init__(self, observations: list[FaceObservation | None], offset: int = 0):
        self.observations = observations
        self.index = offset

    def process(self, frame, timestamp: float = 0.0):
        obs = self.observations[self.index] if self.index < len(self.observations) else None
        self.index += 1
        if obs is None:
            return None
        clone = FaceObservation(**{**obs.__dict__})
        clone.timestamp = timestamp
        return clone


def track_video(cfg: Config, path: Path) -> tuple[list[FaceObservation | None], float]:
    tracker = FaceTracker(cfg)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(tracker.process(frame, len(out) / fps))
    cap.release()
    return out, float(fps)


def challenge_replay_experiment(cfg: Config, engine: LivenessEngine, videos: list[Path], sessions: int = 600, seed: int = 0, log=print) -> dict:
    """How often does recorded footage satisfy a random challenge sequence?

    The clips show people turning, nodding, blinking and opening their mouths,
    which is the most favourable material an attacker with a recording could
    have. Each simulated session starts at a random point of a clip and draws
    its own random challenges.
    """
    rng = np.random.default_rng(seed)
    tracks = [(p.stem, *track_video(cfg, p)) for p in videos]
    # The stored observations carry the face. The picture only has to pass the brightness and sharpness checks.
    frame = np.random.default_rng(1).normal(128, 25, (432, 768, 3)).clip(0, 255).astype(np.uint8)
    passed_all, passed_counts, started = 0, [], 0
    failures: dict[str, int] = {}
    per_challenge: dict[str, list[int]] = {}
    for i in range(sessions):
        name, observations, fps = tracks[i % len(tracks)]
        horizon = int(float(cfg.session.max_duration_s) * fps)
        offset = int(rng.integers(0, max(1, len(observations) - horizon)))
        session = engine.new_session("interactive", seed=int(rng.integers(0, 2 ** 31)), tracker=CachedTracker(observations, offset), score_passive=False)
        for k in range(horizon):
            status = session.update(frame, k / fps)
            if status.stage == Stage.DONE:
                break
        result = session.finalize()
        if not session.records:
            continue
        started += 1
        n_pass = sum(1 for r in session.records if r.passed)
        passed_counts.append(n_pass)
        for r in session.records:
            per_challenge.setdefault(r.challenge.value, []).append(int(bool(r.passed)))
        if n_pass == len(session.plan):
            passed_all += 1
        else:
            failed = next((r for r in session.records if r.passed is False), None)
            key = result.reasons[0] if failed is None else f"{failed.challenge.value}: {failed.detail}"
            failures[key] = failures.get(key, 0) + 1
    return {
        "sessions": int(started), "clips": [t[0] for t in tracks], "challenges_per_session": int(cfg.challenge.count),
        "passed_all": int(passed_all), "pass_rate": float(passed_all / max(1, started)),
        "mean_challenges_passed": float(np.mean(passed_counts)) if passed_counts else 0.0,
        "per_challenge_pass_rate": {k: float(np.mean(v)) for k, v in sorted(per_challenge.items())},
        "per_challenge_trials": {k: int(len(v)) for k, v in sorted(per_challenge.items())},
        "failure_reasons": dict(sorted(failures.items(), key=lambda kv: -kv[1])),
    }


def scripted_session(cfg: Config, engine: LivenessEngine, video: Path, start_s: float, plan: list[str], seed: int = 3,
                     on_frame=None) -> dict:
    """Run a full interactive session over real footage with prompts that match the recorded movements.

    This exercises every stage of the engine on real images. It says nothing
    about security: the prompts were chosen to fit the clip.
    """
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    session = engine.new_session("interactive", seed=seed, plan=plan)
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        status = session.update(frame, index / fps)
        if on_frame is not None:
            on_frame(frame, index / fps, status, session)
        index += 1
        if status.stage == Stage.DONE:
            break
    cap.release()
    return session.finalize().as_dict()


def inject_pulse(frame: np.ndarray, landmarks: np.ndarray, value: float, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Modulate the skin of one frame the way a heartbeat does. ``value`` is the pulse wave in minus one to one."""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(landmarks[list(FACE_OVAL), :2]).astype(np.int32)), 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 3)
    gains = 1.0 + amplitude * value * np.array([0.35, 1.0, 0.5], dtype=np.float32)       # B, G, R relative strength
    out = frame.astype(np.float32) * (1.0 + mask[:, :, None] * (gains - 1.0))
    # stochastic rounding keeps changes smaller than one grey level alive on average
    return np.clip(np.floor(out + rng.random(out.shape, dtype=np.float32)), 0, 255).astype(np.uint8)


def pulse_validation(cfg: Config, video: Path, starts_s: list[float], seconds: float = 12.0, seed: int = 0, log=print) -> dict:
    """Recover an injected heartbeat from real footage at several strengths."""
    tracker = FaceTracker(cfg)
    rng = np.random.default_rng(seed)
    rows = []
    for start in starts_s:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        frames = []
        for _ in range(int(seconds * fps)):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        tracker.reset()
        observations = [tracker.process(f, i / fps) for i, f in enumerate(frames)]
        for amplitude in (0.0, 0.003, 0.006, 0.012):
            for bpm in (66.0, 84.0, 102.0):
                estimator = PulseEstimator.from_config(cfg)
                for i, (frame, obs) in enumerate(zip(frames, observations)):
                    if obs is None:
                        continue
                    t = i / fps
                    wave = np.sin(2 * np.pi * bpm / 60.0 * t) + 0.3 * np.sin(4 * np.pi * bpm / 60.0 * t + 0.8)
                    shown = inject_pulse(frame, obs.landmarks, float(wave), amplitude, rng) if amplitude > 0 else frame
                    estimator.add(t, shown, obs.landmarks)
                result = estimator.estimate()
                rows.append({
                    "start_s": start, "amplitude": amplitude, "true_bpm": bpm if amplitude > 0 else None, "valid": result.valid,
                    "bpm": result.bpm, "snr_db": result.snr_db, "score": result.score,
                    "error_bpm": abs(result.bpm - bpm) if amplitude > 0 and result.valid else None,
                })
                if amplitude == 0.0:
                    break
        log(f"  pulse validation: window at {start:.0f} s done")
    summary = {}
    for amplitude in sorted({r["amplitude"] for r in rows}):
        part = [r for r in rows if r["amplitude"] == amplitude and r["valid"]]
        entry = {"runs": len(part), "median_snr_db": float(np.median([r["snr_db"] for r in part])) if part else None,
                 "mean_score": float(np.mean([r["score"] for r in part])) if part else None}
        if amplitude > 0 and part:
            errors = np.array([r["error_bpm"] for r in part])
            entry.update({"mean_abs_error_bpm": float(errors.mean()), "within_3_bpm": float((errors <= 3.0).mean())})
        summary[f"{amplitude:.3f}"] = entry
    return {"video": video.stem, "seconds": seconds, "fps": fps, "windows": starts_s, "by_amplitude": summary, "runs": rows}


def blink_trace(cfg: Config, video: Path, start_s: float = 0.0, seconds: float | None = None) -> dict:
    """Eye opening over time with detected blinks, for the figure and for manual review."""
    tracker = FaceTracker(cfg)
    detector = BlinkDetector.from_config(cfg)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    times, ears, ratios = [], [], []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok or (seconds is not None and index / fps > seconds):
            break
        t = start_s + index / fps
        index += 1
        obs = tracker.process(frame, t)
        if obs is None:
            continue
        detector.update(t, obs.ear, obs.pose.yaw, obs.pose.pitch)
        times.append(t)
        ears.append(obs.ear)
        ratios.append(detector.ratio)
    cap.release()
    return {
        "video": video.stem, "fps": fps, "times": times, "ear": ears, "ratio": ratios,
        "blinks": [{"start": e.start, "end": e.end, "depth": e.depth} for e in detector.events],
    }


def validate_signals(cfg: Config, log=print) -> dict:
    """Run every signal validation and write ``reports/signals.json``."""
    reports = cfg.path("paths.reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    video_dir = cfg.path("paths.video_dir")
    female = video_dir / "head_pose_face_detection_female.mp4"
    male = video_dir / "head_pose_face_detection_male.mp4"
    target = reports / "signals.json"
    out: dict = json.loads(target.read_text()) if target.exists() else {}      # finished parts are kept
    if "depth" not in out:
        log("depth cue and head pose on MUCT multi camera views")
        out["depth"] = depth_validation(cfg, log=log)
        target.write_text(json.dumps(out, indent=2))
    engine = LivenessEngine(cfg)
    if female.exists() and male.exists():
        if "challenge_replay" not in out:
            log("challenge protocol against recorded footage")
            out["challenge_replay"] = challenge_replay_experiment(cfg, engine, [female, male], log=log)
            target.write_text(json.dumps(out, indent=2))
        log("scripted end to end session on real footage")
        out["scripted_session"] = scripted_session(cfg, engine, female, 103.0, ["look_up", "turn_right", "turn_left"])
        log("pulse recovery with injected heartbeat")
        out["pulse"] = pulse_validation(cfg, female, [12.0, 26.0, 46.0, 74.0], log=log)
        trace = blink_trace(cfg, female)
        out["blinks"] = {"video": trace["video"], "detected": len(trace["blinks"]), "seconds": trace["times"][-1] if trace["times"] else 0.0,
                         "events": trace["blinks"]}
        (reports / "blink_trace.json").write_text(json.dumps(trace))
    (reports / "signals.json").write_text(json.dumps(out, indent=2))
    return out
