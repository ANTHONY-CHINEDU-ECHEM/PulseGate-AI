"""Add real video as an extra training domain.

A model trained on one image source learns the look of that source. Genuine
frames from a second camera, used unprocessed, are the cheapest way to show it
that "live" is not a camera style. The same frames also go through the attack
simulator, so the person and the room appear in every class and cannot become
a shortcut. Frames are split by time: the last part of the clip is validation.
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
from ..vision.geometry import crop_square, square_box
from ..vision.tracker import FaceTracker
from .attacks.common import FaceRef, face_ref_from_landmarks
from .attacks.simulator import FAMILY, LIVE, SPECIES, Simulator, describe

_STATE: dict = {}


def select_frames(cfg: Config, video: Path, max_frames: int, log=print) -> tuple[list[dict], tuple[int, int]]:
    """Track the clip once and keep near frontal frames with a face of usable size."""
    tracker = FaceTracker(cfg)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    kept, index, size = [], 0, (0, 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        size = (frame.shape[1], frame.shape[0])
        obs = tracker.process(frame, index / fps)
        if obs is not None and obs.size >= cfg.crop.min_face_px and abs(obs.pose.yaw) <= 30 and abs(obs.pose.pitch) <= 25:
            face = face_ref_from_landmarks(obs.landmarks, obs.box)
            kept.append({"frame": index, "box": [float(v) for v in face.box], "points": face.points.astype(float).tolist()})
        index += 1
    cap.release()
    log(f"{video.name}: {len(kept)} usable frames of {index}")
    if len(kept) > max_frames:
        pick = np.linspace(0, len(kept) - 1, max_frames).round().astype(int)
        kept = [kept[i] for i in pick]
    return kept, size


def _read(video: str, index: int) -> np.ndarray:
    cap = _STATE.get("cap")
    if cap is None:
        cap = _STATE["cap"] = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read frame {index}")
    return frame


def _init(cfg_dict: dict, out_dir: str, size: tuple[int, int]) -> None:
    cv2.setNumThreads(1)
    cfg = Config(cfg_dict)
    _STATE.update(cfg=cfg, tracker=FaceTracker(cfg), sim=Simulator(size), out_dir=Path(out_dir), cap=None)


def _face(item: dict) -> FaceRef:
    return FaceRef(box=np.asarray(item["box"], dtype=np.float32), points=np.asarray(item["points"], dtype=np.float32))


def _task(task: dict) -> dict | None:
    cfg, tracker, sim = _STATE["cfg"], _STATE["tracker"], _STATE["sim"]
    rng = np.random.default_rng(task["seed"])
    frame = _read(task["video"], task["source"]["frame"])
    face = _face(task["source"])
    kind = task["kind"]
    params_text = "{}"
    if kind == "raw":
        out, centre = frame, face.center          # the camera frame exactly as recorded
    else:
        params = sim.sample(LIVE if kind == "live" else kind, rng)
        limit = 1.25 * face.size
        if kind == LIVE:
            params.live.face_width = float(np.clip(params.live.face_width, 95, limit))
            out, centre = sim.render(frame, face, params, rng)
        else:
            medium = params.medium.sheet if kind == "cutout_mask" else params.medium
            medium.face_width = float(np.clip(medium.face_width, 100, limit))
            params.background.face_width = float(np.clip(params.background.face_width, 100, limit))
            background = _read(task["video"], task["background"]["frame"])
            out, centre = sim.render(frame, face, params, rng, background, _face(task["background"]))
        params_text = json.dumps(describe(params))
    obs = tracker.process_image(out, near=centre)
    if obs is None or obs.size < cfg.crop.min_face_px:
        return None
    cx, cy, side = square_box(obs.box, cfg.crop.context_scale)
    crop = crop_square(out, cx, cy, side)
    species = LIVE if kind in ("raw", LIVE) else kind
    rel = Path(task["split"]) / species / f"{task['sample_id']}.jpg"
    path = _STATE["out_dir"] / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, int(cfg.dataset.jpeg_quality)])
    return {
        "sample_id": task["sample_id"], "path": rel.as_posix(), "split": task["split"], "label": int(species == LIVE),
        "species": species, "family": FAMILY[species], "subject": task["subject"], "lighting": "video", "camera": "video",
        "gender": "unknown", "glasses": "unknown", "source": f"{Path(task['video']).name}#{task['source']['frame']}",
        "background": "" if species == LIVE else f"#{task['background']['frame']}",
        "face_px": round(float(obs.size), 1), "crop_px": int(crop.shape[0]), "yaw": round(obs.pose.yaw, 1), "pitch": round(obs.pose.pitch, 1),
        "params": params_text, "domain": "video_raw" if kind == "raw" else "video",
    }


def add_video_domain(cfg: Config, video: str | Path, name: str | None = None, max_frames: int = 900, attack_every: int = 2,
                     val_fraction: float = 0.2, log=print) -> pd.DataFrame:
    """Render one clip into the dataset and extend ``manifest.csv``.

    Every selected frame contributes an unprocessed genuine crop. Every
    ``attack_every`` th frame also contributes a simulated camera variant and
    one presentation per attack species.
    """
    video = Path(video)
    name = name or video.stem.replace("-", "_")
    out_dir = cfg.path("paths.dataset_dir")
    manifest_path = out_dir / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"background": str})
    if "domain" not in manifest.columns:
        manifest["domain"] = "muct"
    subject = f"video_{name}"
    manifest = manifest[manifest["subject"] != subject]          # rebuilding a clip replaces its old samples
    frames, size = select_frames(cfg, video, max_frames, log)
    if len(frames) < 20:
        raise SystemExit("too few usable frames in this clip")
    cut = int(round(len(frames) * (1.0 - val_fraction)))
    rng = np.random.default_rng(int(cfg.dataset.seed) + 17)
    tasks = []
    prefix = f"vd_{name}_"
    for i, item in enumerate(frames):
        split = "train" if i < cut else "val"
        pool = range(0, cut) if split == "train" else range(cut, len(frames))
        kinds = ["raw"] + ([LIVE] + list(SPECIES) if i % attack_every == 0 else [])
        for kind in kinds:
            background = frames[int(rng.choice(pool))]
            tasks.append({
                "sample_id": f"{prefix}{len(tasks):06d}", "kind": kind, "split": split, "subject": subject, "video": str(video),
                "source": item, "background": background, "seed": int(cfg.dataset.seed * 7919 + len(tasks)),
            })
    log(f"rendering {len(tasks)} presentations from {len(frames)} frames")
    start = time.time()
    workers = max(1, int(cfg.dataset.workers))
    rows = []
    if workers == 1:
        _init(cfg.to_dict(), str(out_dir), size)
        rows = [_task(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg.to_dict(), str(out_dir), size)) as pool:
            for done, row in enumerate(pool.imap_unordered(_task, tasks, chunksize=8), 1):
                rows.append(row)
                if done % 500 == 0:
                    log(f"  {done} of {len(tasks)} rendered")
    rows = [r for r in rows if r is not None]
    added = pd.DataFrame(rows).sort_values("sample_id")
    manifest = pd.concat([manifest, added], ignore_index=True)
    manifest.to_csv(manifest_path, index=False)
    log(f"added {len(added)} samples from {video.name} in {time.time() - start:.0f} s: " + ", ".join(f"{k} {v}" for k, v in added["species"].value_counts().items()))
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.setdefault("video_domains", {})[name] = {
        "file": video.name, "frames_used": len(frames), "samples": int(len(added)), "per_split": added.groupby("split").size().to_dict(),
        "per_species": added.groupby("species").size().to_dict(), "unprocessed_live": int((added["domain"] == "video_raw").sum()),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return added
