"""Measure and calibrate the passive model on your own camera.

The shipped model was trained on simulated attacks. This command answers the
question every deployment has to answer for itself: how does it behave on this
camera, in this room, against real photographs and real screens? It scores the
crops recorded with ``python manage.py collect`` and reports error rates at the
shipped threshold. With ``apply=true`` it stores a threshold fitted to the
captures in ``configs/local.yaml``, which is loaded on top of the defaults.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import yaml

from ..config import PROJECT_ROOT, Config
from ..models.inference import PassiveLiveness
from ..data.captures import load_captures
from .metrics import equal_error_rate, pad_report, roc_auc, threshold_at_apcer

LOCAL_CONFIG = PROJECT_ROOT / "configs" / "local.yaml"


def calibrate(cfg: Config, apply: bool = False, target_apcer: float = 0.01, log=print) -> dict:
    captures = load_captures(cfg.path("paths.capture_dir"))
    if captures.empty:
        raise SystemExit("no captures found. Record some with 'python manage.py collect label=live' first.")
    model = PassiveLiveness(cfg)
    scores = []
    for path in captures["path"]:
        crop = cv2.imread(path, cv2.IMREAD_COLOR)
        scores.append(model.score_crop(crop).live if crop is not None else np.nan)
    captures = captures.assign(score=scores).dropna(subset=["score"])
    live = captures[captures["label"] == 1]["score"].to_numpy()
    attack = captures[captures["label"] == 0]["score"].to_numpy()
    report: dict = {
        "shipped_threshold": model.threshold,
        "captures": {name: int(count) for name, count in captures["species"].value_counts().items()},
    }
    if len(live):
        report["live"] = {
            "count": int(len(live)), "median_score": float(np.median(live)), "lowest_score": float(live.min()),
            "rejected_at_shipped_threshold": float((live < model.threshold).mean()),
        }
    if len(attack):
        report["attacks"] = pad_report(captures["label"], captures["score"], captures["species"], model.threshold)["per_species"]
        report["attacks_accepted_at_shipped_threshold"] = float((attack >= model.threshold).mean())
    suggestion = None
    if len(live) >= 20 and len(attack) >= 20:
        report["auc"] = roc_auc(captures["label"], captures["score"])
        eer, _ = equal_error_rate(captures["label"], captures["score"])
        report["eer"] = eer
        if attack.max() < live.min():
            # the captures separate cleanly: sit in the middle of the gap, not on the edge of either class
            suggestion = float((attack.max() + live.min()) / 2.0)
        else:
            suggestion = threshold_at_apcer(captures["label"], captures["score"], target_apcer)
        report["suggested_threshold"] = suggestion
        report["live_rejected_at_suggested_threshold"] = float((live < suggestion).mean())
    else:
        report["note"] = "record at least 20 genuine and 20 attack crops to get a suggested threshold"
    log(json.dumps(report, indent=2))
    if apply:
        if suggestion is None:
            raise SystemExit("cannot apply: genuine and attack captures are both needed")
        local = yaml.safe_load(LOCAL_CONFIG.read_text()) if LOCAL_CONFIG.exists() else {}
        local = local or {}
        local.setdefault("passive", {})["threshold"] = round(float(suggestion), 4)
        LOCAL_CONFIG.write_text(yaml.safe_dump(local, sort_keys=False))
        log(f"stored passive.threshold={suggestion:.4f} in {LOCAL_CONFIG}")
    return report
