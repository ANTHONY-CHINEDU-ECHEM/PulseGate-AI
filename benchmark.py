"""Latency of each stage on this machine, measured on the bundled sample clip."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import PROJECT_ROOT, Config
from ..engine.session import LivenessEngine


def _stats(values: list[float]) -> dict:
    arr = np.asarray(values) * 1000.0
    return {"median_ms": round(float(np.median(arr)), 2), "p95_ms": round(float(np.percentile(arr, 95)), 2), "runs": int(len(arr))}


def benchmark(cfg: Config, frames_to_use: int = 150, log=print) -> dict:
    clip = PROJECT_ROOT / "assets" / "samples" / "live_clip.mp4"
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    frames = []
    while len(frames) < frames_to_use:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise FileNotFoundError(f"sample clip not found at {clip}")
    engine = LivenessEngine(cfg)
    tracker = engine.tracker.fork()
    detect, track, passive, full = [], [], [], []
    for frame in frames[:40]:
        t0 = time.perf_counter()
        tracker.detect(frame)
        detect.append(time.perf_counter() - t0)
    tracker.reset()
    observations = []
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        observations.append(tracker.process(frame, i / fps))
        track.append(time.perf_counter() - t0)
    for frame, obs in zip(frames, observations):
        if obs is None:
            continue
        t0 = time.perf_counter()
        engine.passive.score_frame(frame, obs.box)
        passive.append(time.perf_counter() - t0)
    session = engine.new_session("passive")
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        session.update(frame, i / fps)
        full.append(time.perf_counter() - t0)
    result = {
        "machine": {"processor": platform.processor() or platform.machine(), "system": platform.system(), "cpu_threads_used": 1},
        "frame_size": [int(frames[0].shape[1]), int(frames[0].shape[0])],
        "face_detection": _stats(detect), "tracking_with_landmarks": _stats(track[5:]), "passive_model_three_patches": _stats(passive[5:]),
        "full_session_update": _stats(full[5:]),
    }
    result["frames_per_second_possible"] = round(1000.0 / result["full_session_update"]["median_ms"], 1)
    reports = cfg.path("paths.reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    Path(reports / "latency.json").write_text(json.dumps(result, indent=2))
    log(json.dumps(result, indent=2))
    return result
