"""Build the PulseGate presentation attack dataset from MUCT.

Each source photograph yields several genuine presentations (the photograph
seen through a randomised camera) and one presentation per attack species,
rendered by the simulator. Subjects are split into train, validation and test
so that no person appears in two partitions, and attack backgrounds are drawn
from the same partition as the source.
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
from .attacks.common import FaceRef
from .attacks.simulator import FAMILY, LIVE, Simulator, describe
from .muct import index_muct

_STATE: dict = {}


def split_subjects(subjects: list[str], fractions: dict, seed: int) -> dict[str, str]:
    """Assign every subject to exactly one partition."""
    rng = np.random.default_rng(seed)
    order = list(subjects)
    rng.shuffle(order)
    n = len(order)
    n_train = int(round(fractions["train"] * n))
    n_val = int(round(fractions["val"] * n))
    out = {}
    for i, subject in enumerate(order):
        out[subject] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return out


def detect_sources(cfg: Config, index: pd.DataFrame, jpg_dir: Path, cache: Path, log=print) -> pd.DataFrame:
    """Run the face detector once on every source photograph and cache the result."""
    if cache.exists():
        cached = pd.read_json(cache, orient="records", dtype={"file": str})
        if len(cached) and set(index["file"]) >= set(cached["file"]):
            return index.merge(cached, on="file", how="inner")
    tracker = FaceTracker(cfg)
    rows = []
    for i, name in enumerate(index["file"]):
        image = cv2.imread(str(jpg_dir / name))
        detections = tracker.detect(image) if image is not None else []
        if detections:
            best = detections[0]
            rows.append({"file": name, "box": [float(v) for v in best.box], "points": best.points.astype(float).tolist(), "det_score": best.score})
        if (i + 1) % 500 == 0:
            log(f"  detected faces in {i + 1} of {len(index)} source images")
    cached = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cached.to_json(cache, orient="records")
    return index.merge(cached, on="file", how="inner")


def _init_worker(cfg_dict: dict, jpg_dir: str, out_dir: str) -> None:
    cv2.setNumThreads(1)
    cfg = Config(cfg_dict)
    _STATE["cfg"] = cfg
    _STATE["tracker"] = FaceTracker(cfg)
    _STATE["sim"] = Simulator(tuple(cfg.dataset.frame_size))
    _STATE["jpg_dir"] = Path(jpg_dir)
    _STATE["out_dir"] = Path(out_dir)


def _face(row: dict) -> FaceRef:
    return FaceRef(box=np.asarray(row["box"], dtype=np.float32), points=np.asarray(row["points"], dtype=np.float32))


def _render_task(task: dict) -> dict | None:
    cfg, tracker, sim = _STATE["cfg"], _STATE["tracker"], _STATE["sim"]
    rng = np.random.default_rng(task["seed"])
    source = cv2.imread(str(_STATE["jpg_dir"] / task["source"]["file"]))
    face = _face(task["source"])
    params = sim.sample(task["species"], rng)
    if task["species"] == LIVE:
        frame, centre = sim.render(source, face, params, rng)
    else:
        background = cv2.imread(str(_STATE["jpg_dir"] / task["background"]["file"]))
        frame, centre = sim.render(source, face, params, rng, background, _face(task["background"]))
    obs = tracker.process_image(frame, near=centre)
    if obs is None or obs.size < cfg.crop.min_face_px:
        return None
    cx, cy, side = square_box(obs.box, cfg.crop.context_scale)
    crop = crop_square(frame, cx, cy, side)
    rel = Path(task["split"]) / task["species"] / f"{task['sample_id']}.jpg"
    path = _STATE["out_dir"] / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, int(cfg.dataset.jpeg_quality)])
    src = task["source"]
    return {
        "sample_id": task["sample_id"], "path": rel.as_posix(), "split": task["split"],
        "label": int(task["species"] == LIVE), "species": task["species"], "family": FAMILY[task["species"]],
        "subject": src["subject"], "lighting": src["lighting"], "camera": src["camera"], "gender": src["gender"],
        "glasses": src["glasses"], "source": src["file"],
        "background": "" if task["species"] == LIVE else task["background"]["file"],
        "face_px": round(float(obs.size), 1), "crop_px": int(crop.shape[0]),
        "yaw": round(obs.pose.yaw, 1), "pitch": round(obs.pose.pitch, 1),
        "params": json.dumps(describe(params)),
    }


def plan_tasks(cfg: Config, sources: pd.DataFrame) -> list[dict]:
    """Deterministic list of render jobs."""
    ds = cfg.dataset
    rng = np.random.default_rng(ds.seed)
    split_of = split_subjects(sorted(sources["subject"].unique()), dict(ds.split), ds.seed)
    sources = sources.assign(split=sources["subject"].map(split_of)).reset_index(drop=True)
    records = sources.to_dict("records")
    by_split: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        by_split.setdefault(rec["split"], []).append(i)
    tasks = []
    counter = 0
    for i, rec in enumerate(records):
        variants = [LIVE] * int(ds.live_variants) + list(ds.species)
        for species in variants:
            background = None
            if species != LIVE:
                for _ in range(20):
                    j = int(rng.choice(by_split[rec["split"]]))
                    if records[j]["subject"] != rec["subject"]:
                        break
                background = records[j]
            tasks.append({
                "sample_id": f"pg{counter:06d}", "species": species, "split": rec["split"], "source": rec,
                "background": background, "seed": int(ds.seed * 1_000_003 + counter),
            })
            counter += 1
    return tasks


def build_dataset(cfg: Config, limit: int = 0, log=print) -> pd.DataFrame:
    """Render the dataset and write ``manifest.csv``. ``limit`` caps the number of source images."""
    jpg_dir = cfg.path("paths.muct_dir") / "jpg"
    out_dir = cfg.path("paths.dataset_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    index = index_muct(jpg_dir)
    log(f"MUCT index: {len(index)} photographs of {index['subject'].nunique()} subjects")
    sources = detect_sources(cfg, index, jpg_dir, out_dir / "source_faces.json", log)
    log(f"faces found in {len(sources)} source photographs")
    if limit:
        keep = sources.groupby("subject").head(max(1, limit // sources["subject"].nunique() + 1)).head(limit)
        sources = keep
    tasks = plan_tasks(cfg, sources)
    log(f"rendering {len(tasks)} presentations with {cfg.dataset.workers} workers")
    start = time.time()
    rows: list[dict] = []
    dropped = 0
    workers = max(1, int(cfg.dataset.workers))
    args = (cfg.to_dict(), str(jpg_dir), str(out_dir))
    if workers == 1:
        _init_worker(*args)
        iterator = map(_render_task, tasks)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=args)
        iterator = pool.imap_unordered(_render_task, tasks, chunksize=8)
    for done, row in enumerate(iterator, 1):
        if row is None:
            dropped += 1
        else:
            rows.append(row)
        if done % 1000 == 0:
            rate = done / (time.time() - start)
            log(f"  {done} of {len(tasks)} rendered, {dropped} dropped, {rate:.1f} per second, about {(len(tasks) - done) / rate / 60:.0f} min left")
    if pool is not None:
        pool.close()
        pool.join()
    manifest = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    summary = {
        "samples": int(len(manifest)), "dropped_no_face": int(dropped), "sources": int(len(sources)),
        "subjects": int(sources["subject"].nunique()),
        "per_split": manifest.groupby("split").size().to_dict(),
        "per_species": manifest.groupby("species").size().to_dict(),
        "subjects_per_split": manifest.groupby("split")["subject"].nunique().to_dict(),
        "seconds": round(time.time() - start, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    return manifest
