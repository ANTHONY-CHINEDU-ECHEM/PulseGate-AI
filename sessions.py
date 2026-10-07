"""Session level evaluation on real video.

Genuine sessions are windows of real footage from a camera, people and rooms
the model never saw in training. Attack sessions take the same footage and
present it through the simulator with one fixed medium per session and a
little hand tremor, which gives temporally consistent replay and print videos.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..config import Config
from ..data.attacks.common import face_ref_from_landmarks
from ..data.attacks.simulator import LIVE, SPECIES, Simulator
from ..engine.session import LivenessEngine
from ..vision.tracker import FaceTracker

_STATE: dict = {}


def read_window(path: Path, start: int, count: int) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, float(fps)


def find_windows(cfg: Config, path: Path, window_s: float, max_windows: int, min_face_px: float) -> list[int]:
    """Start frames of non overlapping windows in which one face is tracked almost throughout."""
    tracker = FaceTracker(cfg)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    usable = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        obs = tracker.process(frame, len(usable) / fps)
        usable.append(obs is not None and obs.size >= min_face_px and abs(obs.pose.yaw) < 60)
    cap.release()
    size = int(round(window_s * fps))
    starts = [s for s in range(0, len(usable) - size + 1, size) if np.mean(usable[s:s + size]) >= 0.85]
    if len(starts) > max_windows:
        pick = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in pick]
    return starts


def _init(cfg_dict: dict) -> None:
    cv2.setNumThreads(1)
    cfg = Config(cfg_dict)
    _STATE["cfg"] = cfg
    _STATE["engine"] = LivenessEngine(cfg)
    _STATE["source_tracker"] = FaceTracker(cfg)


def _tremor(n: int, rng: np.random.Generator) -> np.ndarray:
    """Slow random hand movement: pixels for the shift, radians for the tilt."""
    walk = np.cumsum(rng.standard_normal((n, 5)), axis=0)
    kernel = np.hanning(9) / np.hanning(9).sum()
    smooth = np.column_stack([np.convolve(walk[:, i], kernel, mode="same") for i in range(5)])
    smooth -= smooth.mean(axis=0)
    return smooth * np.array([0.9, 0.9, 0.004, 0.004, 0.003])


def _run(task: dict) -> dict:
    engine = _STATE["engine"]
    frames, fps = read_window(Path(task["video"]), task["start"], task["count"])
    rng = np.random.default_rng(task["seed"])
    species = task["species"]
    if species != LIVE:
        h, w = frames[0].shape[:2]
        sim = Simulator((w, h))
        tracker: FaceTracker = _STATE["source_tracker"]
        tracker.reset()
        background, _ = read_window(Path(task["background_video"]), task["background_frame"], 1)
        bg_obs = tracker.process_image(background[0])
        tracker.reset()
        first = tracker.process(frames[0], 0.0)
        if bg_obs is None or first is None:
            return {**task, "skipped": True}
        bg_face = face_ref_from_landmarks(bg_obs.landmarks, bg_obs.box)
        params = sim.sample(species, rng)
        medium = params.medium.sheet if species == "cutout_mask" else params.medium
        medium.face_width = float(np.clip(medium.face_width, 100, 1.3 * first.size))
        if species != "cutout_mask":
            medium.offset = (medium.offset[0], medium.offset[1] * 0.5)
        else:
            params.background.face_width = float(np.clip(params.background.face_width, 100, 1.2 * bg_obs.size))
        tremor = _tremor(len(frames), rng)
        tracker.reset()
        rendered, last_face = [], face_ref_from_landmarks(first.landmarks, first.box)
        for i, frame in enumerate(frames):
            obs = tracker.process(frame, i / fps)
            if obs is not None:
                last_face = face_ref_from_landmarks(obs.landmarks, obs.box)
            out, _ = sim.render(frame, last_face, params, rng, background[0], bg_face, tuple(tremor[i]))
            rendered.append(out)
        frames = rendered
    session = engine.new_session("passive")
    for i, frame in enumerate(frames):
        session.update(frame, i / fps)
    result = session.finalize()
    scores = [s.live for _, s in session._scores]
    return {
        "video": Path(task["video"]).stem, "start": task["start"], "species": species, "skipped": False,
        "decision": result.decision, "score": result.score, "reasons": "|".join(result.reasons),
        "passive": result.passive.get("score"), "passive_frames": result.passive.get("frames", 0),
        "frame_scores": scores, "blinks": result.blinks,
        "pulse_valid": result.pulse["valid"], "pulse_bpm": result.pulse["bpm"], "pulse_snr_db": result.pulse["snr_db"],
        "pulse_score": result.pulse["score"], "predicted_class": result.passive.get("predicted_class", ""),
    }


def evaluate_sessions(cfg: Config, videos: list[str] | None = None, window_s: float = 10.0, live_windows: int = 13,
                      attack_windows: int = 5, workers: int = 2, log=print) -> dict:
    """Run genuine and simulated attack sessions on the sample videos. Writes ``reports/sessions.json``."""
    video_dir = cfg.path("paths.video_dir")
    # Only footage of a person who never appears in training. Clips used as training domains are left out.
    names = videos or ["head_pose_face_detection_female.mp4"]
    paths = [video_dir / n for n in names if (video_dir / n).exists()]
    if not paths:
        raise FileNotFoundError(f"no sample videos in {video_dir}. Run 'python manage.py download_data' first.")
    tasks = []
    seed = int(cfg.dataset.seed)
    for vi, path in enumerate(paths):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        count = int(round(window_s * fps))
        starts = find_windows(cfg, path, window_s, live_windows, float(cfg.crop.min_face_px))
        log(f"{path.name}: {len(starts)} usable windows of {window_s:.0f} s")
        for start in starts:
            tasks.append({"video": str(path), "start": start, "count": count, "species": LIVE, "seed": seed + len(tasks)})
        if starts:
            pick = np.linspace(0, len(starts) - 1, min(attack_windows, len(starts))).round().astype(int)
            other = paths[(vi + 1) % len(paths)]
            for k in pick:
                for species in SPECIES:
                    tasks.append({
                        "video": str(path), "start": starts[k], "count": count, "species": species, "seed": seed + 7919 + len(tasks),
                        "background_video": str(other), "background_frame": 40 + 30 * int(k),
                    })
    log(f"running {len(tasks)} sessions")
    start_time = time.time()
    if workers > 1:
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg.to_dict(),)) as pool:
            rows = []
            for i, row in enumerate(pool.imap_unordered(_run, tasks), 1):
                rows.append(row)
                if i % 20 == 0:
                    log(f"  {i} of {len(tasks)} sessions done")
    else:
        _init(cfg.to_dict())
        rows = [_run(t) for t in tasks]
    rows = [r for r in rows if not r.get("skipped")]
    table = pd.DataFrame(rows)
    threshold = LivenessEngine(cfg).passive.threshold if workers > 1 else _STATE["engine"].passive.threshold

    def frame_rate(mask, accept: bool) -> tuple[float | None, int]:
        scores = np.concatenate([np.asarray(s) for s in table.loc[mask, "frame_scores"]]) if mask.any() else np.zeros(0)
        if len(scores) == 0:
            return None, 0
        return float((scores >= threshold).mean() if accept else (scores < threshold).mean()), int(len(scores))

    live = table["species"] == LIVE
    bpcer, live_frames = frame_rate(live, accept=False)
    per_video = {}
    for name in sorted(table["video"].unique()):
        rate, n = frame_rate(live & (table["video"] == name), accept=False)
        per_video[name] = {"frame_bpcer": rate, "frames": n, "sessions": int((live & (table["video"] == name)).sum())}
    per_species = {}
    for species in SPECIES:
        mask = table["species"] == species
        rate, n = frame_rate(mask, accept=True)
        decided = table.loc[mask & (table["decision"] != "inconclusive")]
        per_species[species] = {
            "frame_apcer": rate, "frames": n, "sessions": int(mask.sum()),
            "sessions_accepted": int((table.loc[mask, "decision"] == "live").sum()),
            "sessions_inconclusive": int((table.loc[mask, "decision"] == "inconclusive").sum()),
            "mean_passive": float(decided["passive"].mean()) if len(decided) else None,
        }
    all_live = np.concatenate([np.asarray(s) for s in table.loc[live, "frame_scores"]]) if live.any() else np.zeros(0)
    all_attack = np.concatenate([np.asarray(s) for s in table.loc[~live, "frame_scores"]]) if (~live).any() else np.zeros(0)
    from .metrics import equal_error_rate, roc_auc
    labels = np.concatenate([np.ones(len(all_live)), np.zeros(len(all_attack))])
    both = np.concatenate([all_live, all_attack])
    summary = {
        "threshold": threshold, "window_s": window_s, "videos": [p.name for p in paths],
        "live": {
            "sessions": int(live.sum()), "frames": live_frames, "frame_bpcer": bpcer,
            "sessions_accepted": int((table.loc[live, "decision"] == "live").sum()),
            "sessions_rejected": int((table.loc[live, "decision"] == "not_live").sum()),
            "sessions_inconclusive": int((table.loc[live, "decision"] == "inconclusive").sum()),
            "median_pulse_snr_db": float(table.loc[live & table["pulse_valid"], "pulse_snr_db"].median()) if (live & table["pulse_valid"]).any() else None,
            "sessions_with_blink": int((table.loc[live, "blinks"] > 0).sum()),
            "per_video": per_video,
        },
        "attacks": {
            "sessions": int((~live).sum()), "frames": int(len(all_attack)),
            "frame_apcer": float((all_attack >= threshold).mean()) if len(all_attack) else None,
            "sessions_accepted": int((table.loc[~live, "decision"] == "live").sum()),
            "median_pulse_snr_db": float(table.loc[~live & table["pulse_valid"], "pulse_snr_db"].median()) if (~live & table["pulse_valid"]).any() else None,
            "per_species": per_species,
        },
        "frame_level": {
            "auc": roc_auc(labels, both) if len(all_live) and len(all_attack) else None,
            "eer": equal_error_rate(labels, both)[0] if len(all_live) and len(all_attack) else None,
        },
        "minutes": round((time.time() - start_time) / 60.0, 1),
    }
    reports = cfg.path("paths.reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "sessions.json").write_text(json.dumps(summary, indent=2))
    table.assign(frame_scores=table["frame_scores"].map(lambda v: json.dumps([round(float(x), 4) for x in v]))).to_csv(reports / "session_results.csv", index=False)
    log(json.dumps({k: summary[k] for k in ("live", "frame_level")}, indent=2)[:1500])
    return summary
