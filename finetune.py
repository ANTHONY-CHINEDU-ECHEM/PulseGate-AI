"""Fine tuning on your own camera captures.

A model trained on simulated attacks benefits a great deal from a few minutes
of real examples from the deployment camera. ``python manage.py collect``
records them, this module mixes them with the original training data (so the
network does not forget what it knew) and continues training at a low learning
rate. Thresholds and calibration are refitted with a held out part of the
captures included.
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd

from ..config import Config
from ..data.captures import load_captures
from ..data.dataset import load_manifest
from .train import train_model


def split_captures(captures: pd.DataFrame, val_fraction: float = 0.2, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole recording sessions where possible, since neighbouring frames are near duplicates."""
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for _, group in captures.groupby("species"):
        sessions = sorted(group["session"].unique())
        if len(sessions) >= 3:
            held = set(rng.choice(sessions, size=max(1, int(round(val_fraction * len(sessions)))), replace=False))
            val_parts.append(group[group["session"].isin(held)])
            train_parts.append(group[~group["session"].isin(held)])
        else:
            cut = int(round(len(group) * (1.0 - val_fraction)))      # frames are in time order: the tail is held out
            train_parts.append(group.iloc[:cut])
            val_parts.append(group.iloc[cut:])
    return pd.concat(train_parts), pd.concat(val_parts)


def finetune_model(cfg: Config, epochs: int = 4, replay_per_capture: float = 3.0, log=print) -> dict:
    captures = load_captures(cfg.path("paths.capture_dir"))
    if captures.empty or captures["label"].nunique() < 2:
        raise SystemExit(
            "fine tuning needs genuine and attack captures. Record them with\n"
            "  python manage.py collect label=live\n  python manage.py collect label=replay_phone\n  python manage.py collect label=print_matte"
        )
    log("captures per class: " + ", ".join(f"{k} {v}" for k, v in captures["species"].value_counts().items()))
    capture_train, capture_val = split_captures(captures, seed=int(cfg.train.seed))
    manifest = load_manifest(cfg.path("paths.dataset_dir"))
    root = cfg.path("paths.dataset_dir")
    manifest = manifest.assign(path=manifest["path"].map(lambda p: str(root / p)))
    base_train = manifest[manifest["split"] == "train"]
    keep = int(min(len(base_train), max(2000, replay_per_capture * len(capture_train))))
    base_train = base_train.sample(n=keep, random_state=int(cfg.train.seed))
    base_val = manifest[manifest["split"] == "val"].sample(n=min(1500, int((manifest["split"] == "val").sum())), random_state=0)
    columns = ["sample_id", "path", "label", "species", "subject"]
    # captures are few, so they are repeated to carry weight against the replayed training data
    repeat = int(np.clip(round(0.5 * len(base_train) / max(1, len(capture_train))), 1, 8))
    train_frame = pd.concat([base_train[columns]] + [capture_train[columns]] * repeat, ignore_index=True)
    val_frame = pd.concat([base_val[columns], capture_val[columns]], ignore_index=True)
    log(f"fine tuning on {len(capture_train)} captures (repeated {repeat} times) plus {len(base_train)} original samples")

    tuned = copy.deepcopy(cfg)
    tuned.train.epochs = int(epochs)
    tuned.train.lr = float(cfg.train.lr) * 0.1
    tuned.train.warmup_epochs = 0.3
    tuned.train.run_name = f"{cfg.train.run_name}_finetuned"
    checkpoint = cfg.path("paths.models_dir") / f"{cfg.train.run_name}.pt"
    meta = train_model(tuned, log=log, frames=(train_frame, val_frame), init_checkpoint=checkpoint)
    run = tuned.train.run_name
    log("to use the fine tuned model, add these two options to any command:")
    log(f"  assets.passive_model=models/{run}.onnx assets.passive_meta=models/{run}.json")
    return meta
