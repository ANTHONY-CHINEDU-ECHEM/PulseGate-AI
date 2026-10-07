"""Unseen attack protocol.

A detector that has seen every attack type in training tells little about the
attack nobody thought of. Here one attack family at a time is removed from
training and validation, and the resulting model is tested on exactly that
family. Thresholds come from validation data without the held out family.

All runs, including a reference run that sees every family, use the same
reduced training budget, so the comparison is between like and like.
"""
from __future__ import annotations

import copy
import json

from ..config import Config
from ..data.attacks.simulator import FAMILY, SPECIES
from ..data.dataset import load_manifest
from ..models.inference import PassiveLiveness
from ..training.train import train_model
from .evaluate import score_frame_table
from .metrics import equal_error_rate, roc_auc

FAMILIES = ("print", "replay", "mask")


def _run(cfg: Config, name: str, held_out: list[str], epochs: int, max_train: int, log) -> tuple[PassiveLiveness, dict]:
    experiments = cfg.path("paths.models_dir") / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    run_cfg = copy.deepcopy(cfg)
    run_cfg.train.exclude_species = held_out
    run_cfg.train.epochs = int(epochs)
    run_cfg.train.max_train_samples = int(max_train)
    run_cfg.train.run_name = name
    run_cfg.paths.models_dir = str(experiments)
    meta_path = experiments / f"{name}.json"
    if meta_path.exists() and (experiments / f"{name}.onnx").exists():
        log(f"reusing the finished run {name}")
        meta = json.loads(meta_path.read_text())
    else:
        log(f"training {name} without {held_out}")
        meta = train_model(run_cfg, log=log)
    return PassiveLiveness(run_cfg, experiments / f"{name}.onnx", meta_path), meta


def evaluate_unseen(cfg: Config, epochs: int = 5, max_train: int = 14000, families: tuple[str, ...] = FAMILIES, log=print) -> dict:
    root = cfg.path("paths.dataset_dir")
    manifest = load_manifest(root)
    test = manifest[manifest["split"] == "test"]
    reports = cfg.path("paths.reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    point = str(cfg.passive.operating_point)
    out: dict = {"epochs": epochs, "max_train_samples": max_train, "operating_point": point, "families": {}}

    model, meta = _run(cfg, "unseen_reference", [], epochs, max_train, log)
    threshold = float(meta["thresholds"][point])
    scored = score_frame_table(model, test, root)
    out["reference"] = {
        "threshold": threshold, "eer": equal_error_rate(scored["label"], scored["score"])[0],
        "bpcer": float((scored[scored["label"] == 1]["score"] < threshold).mean()),
        "apcer": {s: float((scored[scored["species"] == s]["score"] >= threshold).mean()) for s in SPECIES},
        "auc": {s: roc_auc(*_pair(scored, s)) for s in SPECIES},
    }
    (reports / "unseen.json").write_text(json.dumps(out, indent=2))

    for family in families:
        held_out = [s for s, f in FAMILY.items() if f == family]
        model, meta = _run(cfg, f"unseen_{family}", held_out, epochs, max_train, log)
        threshold = float(meta["thresholds"][point])
        part = test[(test["label"] == 1) | (test["species"].isin(held_out))]
        seen = test[(test["label"] == 0) & (~test["species"].isin(held_out))]
        scored = score_frame_table(model, part, root)
        scored_seen = score_frame_table(model, seen, root)
        entry = {
            "held_out_species": held_out, "threshold": threshold,
            "auc_vs_unseen": roc_auc(scored["label"], scored["score"]),
            "eer_vs_unseen": equal_error_rate(scored["label"], scored["score"])[0],
            "bpcer": float((scored[scored["label"] == 1]["score"] < threshold).mean()),
            "apcer_unseen": {s: float((scored[scored["species"] == s]["score"] >= threshold).mean()) for s in held_out},
            "auc_unseen": {s: roc_auc(*_pair(scored, s)) for s in held_out},
            "apcer_seen_pooled": float((scored_seen["score"] >= threshold).mean()),
            "validation_eer_seen": meta["validation"]["eer"],
        }
        out["families"][family] = entry
        (reports / "unseen.json").write_text(json.dumps(out, indent=2))
        log(f"  unseen {family}: AUC {entry['auc_vs_unseen']:.4f}, EER {100 * entry['eer_vs_unseen']:.2f}%, APCER {entry['apcer_unseen']}")
    return out


def _pair(scored, species: str):
    part = scored[(scored["label"] == 1) | (scored["species"] == species)]
    return part["label"], part["score"]
