"""How do genuine people filmed by an unseen camera score?

This is the measurement that exposed the main weakness of the first model and
that every later design change was judged by. Frames come from clips whose
people never appear in training. Each model in ``models/experiments`` that is
named ``ablation_*`` is scored next to the shipped model, each at its own
validation thresholds.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..models.inference import PassiveLiveness
from ..vision.tracker import FaceTracker

HELD_OUT = {
    "same room, unseen person": "head_pose_face_detection_female.mp4",
    "unseen room, unseen people": "face_demographics_walking_and_pause.mp4",
}


def collect_crops(cfg: Config, video: Path, stride: int = 4) -> list[tuple[np.ndarray, float]]:
    """Context crops of near frontal faces from a clip, with the face size in pixels."""
    tracker = FaceTracker(cfg)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    q = cfg.quality
    crops, index = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        obs = tracker.process(frame, index / fps)
        index += 1
        if index % stride or obs is None or obs.size < cfg.crop.min_face_px:
            continue
        if abs(obs.pose.yaw) > q.max_abs_yaw_deg or abs(obs.pose.pitch) > q.max_abs_pitch_deg:
            continue
        from ..vision.geometry import crop_square, square_box

        cx, cy, side = square_box(obs.box, cfg.crop.context_scale)
        crops.append((crop_square(frame, cx, cy, side), float(obs.size)))
    cap.release()
    return crops


def evaluate_domain_gap(cfg: Config, log=print) -> dict:
    video_dir = cfg.path("paths.video_dir")
    clips = {label: collect_crops(cfg, video_dir / name) for label, name in HELD_OUT.items() if (video_dir / name).exists()}
    if not clips:
        raise FileNotFoundError(f"held out clips not found in {video_dir}")
    experiments = cfg.path("paths.models_dir") / "experiments"
    candidates = [(p.stem, p.with_suffix(".onnx"), p) for p in sorted(experiments.glob("ablation_*.json"))]
    candidates.append(("shipped", cfg.path("assets.passive_model"), cfg.path("assets.passive_meta")))
    summary_path = cfg.path("paths.dataset_dir") / "summary.json"
    trained_on = set()
    if summary_path.exists():
        trained_on = {d.get("file") for d in json.loads(summary_path.read_text()).get("video_domains", {}).values()}
    out: dict = {"clips": {label: {"file": HELD_OUT[label], "frames": len(c)} for label, c in clips.items()}, "models": {}}
    for name, model_path, meta_path in candidates:
        if not model_path.exists():
            continue
        model = PassiveLiveness(cfg, model_path, meta_path)
        thresholds = model.meta.get("thresholds", {})
        entry = {"note": model.meta.get("note", ""), "validation_eer": model.meta.get("validation", {}).get("eer"), "clips": {}}
        for label, crops in clips.items():
            scores = np.array([model.score_crop(crop).live for crop, _ in crops])
            sizes = np.array([size for _, size in crops])
            entry["clips"][label] = {
                "median_score": float(np.median(scores)),
                "rejected_at_eer_threshold": float((scores < thresholds.get("eer", 0.5)).mean()),
                "rejected_at_apcer_5pct_threshold": float((scores < thresholds.get("bpcer_at_apcer_5pct", 0.5)).mean()),
                "rejected_at_apcer_1pct_threshold": float((scores < thresholds.get("bpcer_at_apcer_1pct", 0.5)).mean()),
                "rejected_at_forgiving_threshold": float((scores < thresholds.get("apcer_at_bpcer_1pct", 0.5)).mean()),
                "rejected_at_shipped_operating_point": float((scores < model.threshold).mean()),
                "median_face_px": float(np.median(sizes)),
                # the shipped model was trained on every domain in the dataset summary, earlier stages state theirs in the note
                "clip_used_in_training": bool(name == "shipped" and HELD_OUT[label] in trained_on),
            }
        out["models"][name] = entry
        log(f"{name}: " + ", ".join(f"{label}: median {c['median_score']:.2f}, rejected {100 * c['rejected_at_eer_threshold']:.0f}%" for label, c in entry["clips"].items()))
    reports = cfg.path("paths.reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "domain_gap.json").write_text(json.dumps(out, indent=2))
    return out
