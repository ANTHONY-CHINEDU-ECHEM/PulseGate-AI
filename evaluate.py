"""Image level evaluation of the passive model on the held out test subjects.

Everything here runs through :class:`PassiveLiveness`, the same code path as
the live application, so the numbers describe what is actually shipped. The
decision threshold comes from the validation split and is never tuned on test.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..config import Config
from ..data.dataset import load_manifest
from ..models.inference import PassiveLiveness
from ..models.network import CLASS_NAMES
from ..models.texture_baseline import texture_features
from .metrics import (
    bootstrap_ci, bpcer_at, equal_error_rate, error_curves, expected_calibration_error, pad_report, roc_auc, summarize,
)


def score_frame_table(model: PassiveLiveness, frame: pd.DataFrame, root: Path, transform=None, log=None) -> pd.DataFrame:
    """Score every crop in ``frame``. ``transform`` optionally perturbs the crop first."""
    rows = []
    for i, row in enumerate(frame.itertuples(index=False)):
        crop = cv2.imread(str(root / row.path), cv2.IMREAD_COLOR)
        if transform is not None:
            crop = transform(crop)
        s = model.score_crop(crop)
        rows.append((s.live, s.context, s.texture, s.species))
        if log is not None and (i + 1) % 1000 == 0:
            log(f"  scored {i + 1} of {len(frame)}")
    out = frame.reset_index(drop=True).copy()
    out[["score", "score_context", "score_texture", "predicted"]] = pd.DataFrame(rows, index=out.index)
    return out


def _ci(scored: pd.DataFrame, metric, seed: int = 0) -> list[float]:
    lo, hi = bootstrap_ci(scored["label"], scored["score"], scored["subject"], metric, n_boot=400, seed=seed)
    return [lo, hi]


def confusion_matrix(scored: pd.DataFrame) -> dict:
    names = list(CLASS_NAMES)
    matrix = np.zeros((len(names), len(names)), dtype=int)
    for true, pred in zip(scored["species"], scored["predicted"]):
        if pred in names:
            matrix[names.index(true), names.index(pred)] += 1
    return {"classes": names, "matrix": matrix.tolist()}


def slice_report(scored: pd.DataFrame, threshold: float) -> dict:
    """Error rates per capture condition, to expose uneven performance across groups."""
    out: dict[str, dict] = {}
    live = scored[scored["label"] == 1]
    attack = scored[scored["label"] == 0]
    for column in ("gender", "glasses", "lighting", "camera"):
        groups = {}
        for value in sorted(scored[column].astype(str).unique()):
            l = live[live[column].astype(str) == value]
            a = attack[attack[column].astype(str) == value]
            groups[value] = {
                "bpcer": float((l["score"] < threshold).mean()) if len(l) else None,
                "apcer": float((a["score"] >= threshold).mean()) if len(a) else None,
                "bona_fide": int(len(l)), "attacks": int(len(a)),
            }
        out[column] = groups
    bins = [0, 120, 150, 180, 1000]
    labels = ["under 120 px", "120 to 150 px", "150 to 180 px", "over 180 px"]
    size = pd.cut(scored["face_px"], bins=bins, labels=labels)
    groups = {}
    for value in labels:
        l = live[size[live.index] == value]
        a = attack[size[attack.index] == value]
        groups[value] = {
            "bpcer": float((l["score"] < threshold).mean()) if len(l) else None,
            "apcer": float((a["score"] >= threshold).mean()) if len(a) else None,
            "bona_fide": int(len(l)), "attacks": int(len(a)),
        }
    out["face_size"] = groups
    return out


# ----------------------------------------------------------------------------
# Robustness: perturb the test crops and hold the threshold fixed
# ----------------------------------------------------------------------------

def _jpeg(q: int):
    def apply(img):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img
    return apply


def _blur(sigma: float):
    return lambda img: cv2.GaussianBlur(img, (0, 0), sigma)


def _gain(g: float):
    return lambda img: np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)


def _noise(sigma: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    return lambda img: np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def _lowres(factor: float):
    def apply(img):
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(8, int(w * factor)), max(8, int(h * factor))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return apply


ROBUSTNESS = {
    "jpeg_quality": [(90, _jpeg(90)), (60, _jpeg(60)), (40, _jpeg(40)), (25, _jpeg(25)), (15, _jpeg(15))],
    "gaussian_blur_sigma": [(0.5, _blur(0.5)), (1.0, _blur(1.0)), (1.5, _blur(1.5)), (2.5, _blur(2.5)), (3.5, _blur(3.5))],
    "brightness_gain": [(0.4, _gain(0.4)), (0.6, _gain(0.6)), (0.8, _gain(0.8)), (1.3, _gain(1.3)), (1.7, _gain(1.7))],
    "sensor_noise_sigma": [(3, _noise(3)), (6, _noise(6)), (10, _noise(10)), (16, _noise(16)), (24, _noise(24))],
    "resolution_factor": [(0.8, _lowres(0.8)), (0.6, _lowres(0.6)), (0.45, _lowres(0.45)), (0.3, _lowres(0.3))],
}


def robustness_report(model: PassiveLiveness, frame: pd.DataFrame, root: Path, threshold: float, samples: int, seed: int, log=print) -> dict:
    subset = pd.concat([g.sample(n=min(len(g), max(1, samples // 6)), random_state=seed) for _, g in frame.groupby("species")])
    out: dict[str, list] = {}
    base = score_frame_table(model, subset, root)
    reference = pad_report(base["label"], base["score"], base["species"], threshold)
    out["reference"] = {"bpcer": reference["bpcer"], "apcer": reference["apcer_pooled"], "samples": int(len(subset))}
    for name, levels in ROBUSTNESS.items():
        rows = []
        for level, transform in levels:
            scored = score_frame_table(model, subset, root, transform)
            rep = pad_report(scored["label"], scored["score"], scored["species"], threshold)
            rows.append({"level": level, "bpcer": rep["bpcer"], "apcer": rep["apcer_pooled"], "auc": roc_auc(scored["label"], scored["score"])})
        out[name] = rows
        log(f"  robustness {name}: " + ", ".join(f"{r['level']}: BPCER {100 * r['bpcer']:.1f} APCER {100 * r['apcer']:.1f}" for r in rows))
    return out


# ----------------------------------------------------------------------------
# Baseline
# ----------------------------------------------------------------------------

def baseline_report(cfg: Config, manifest: pd.DataFrame, root: Path, max_train: int, log=print) -> tuple[dict, pd.DataFrame]:
    """Train and test the classical colour texture baseline with the same protocol."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    crop = cfg.crop

    def features(frame: pd.DataFrame) -> np.ndarray:
        return np.stack([
            texture_features(cv2.imread(str(root / p), cv2.IMREAD_COLOR), crop.context_scale, crop.context_size, crop.patch_size)
            for p in frame["path"]
        ])

    train = manifest[manifest["split"] == "train"]
    if max_train and len(train) > max_train:
        train = pd.concat([g.sample(n=min(len(g), max_train // 6), random_state=0) for _, g in train.groupby("species")])
    val = manifest[manifest["split"] == "val"]
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)
    log(f"  baseline features for {len(train)} train, {len(val)} validation and {len(test)} test crops")
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))
    clf.fit(features(train), train["label"].to_numpy())
    val_scores = clf.predict_proba(features(val))[:, 1]
    from .metrics import threshold_at_apcer
    threshold = threshold_at_apcer(val["label"], val_scores, 0.01)
    test = test.copy()
    test["score"] = clf.predict_proba(features(test))[:, 1]
    report = summarize(test["label"], test["score"], test["species"], threshold)
    report["feature_dim"] = int(clf.named_steps["standardscaler"].n_features_in_)
    return report, test[["sample_id", "label", "species", "score"]]


def operating_point_table(scored: pd.DataFrame, thresholds: dict) -> dict:
    """Error rates on the test set at every threshold that was fixed on validation data."""
    table = {}
    for name, threshold in thresholds.items():
        rep = pad_report(scored["label"], scored["score"], scored["species"], float(threshold))
        table[name] = {
            "threshold": float(threshold), "bpcer": rep["bpcer"], "apcer_pooled": rep["apcer_pooled"],
            "apcer_worst_species": rep["apcer_worst_species"], "apcer": {k: v["apcer"] for k, v in rep["per_species"].items()},
        }
    return table


def evaluate_model(cfg: Config, robustness_samples: int = 1200, baseline_train: int = 9000, log=print) -> dict:
    """Run the full image level evaluation and write ``reports/metrics.json`` and the score table."""
    root = cfg.path("paths.dataset_dir")
    reports = cfg.path("paths.reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(root)
    model = PassiveLiveness(cfg)
    threshold = model.threshold
    test = manifest[manifest["split"] == "test"]
    log(f"scoring {len(test)} test crops from {test['subject'].nunique()} unseen subjects at threshold {threshold:.4f}")
    scored = score_frame_table(model, test, root, log=log)
    scored.drop(columns=["params"], errors="ignore").to_csv(reports / "test_scores.csv", index=False)

    main = summarize(scored["label"], scored["score"], scored["species"], threshold)
    main["confidence_intervals_95"] = {
        "eer": _ci(scored, lambda y, s: equal_error_rate(y, s)[0]),
        "auc": _ci(scored, roc_auc),
        "bpcer": _ci(scored, lambda y, s: float((s[y == 1] < threshold).mean())),
        "apcer_pooled": _ci(scored, lambda y, s: float((s[y == 0] >= threshold).mean())),
        "bpcer_at_apcer_1pct": _ci(scored, lambda y, s: bpcer_at(y, s, 0.01)),
    }
    main["ece"] = expected_calibration_error(scored["label"], scored["score"])
    streams = {}
    val_frame = manifest[manifest["split"] == "val"]
    val_scored = score_frame_table(model, val_frame, root)
    from .metrics import threshold_at_apcer
    for name, column in (("fused", "score"), ("context_only", "score_context"), ("texture_only", "score_texture")):
        stream_threshold = threshold_at_apcer(val_scored["label"], val_scored[column], 0.01)
        rep = summarize(scored["label"], scored[column], scored["species"], stream_threshold)
        streams[name] = {k: rep[k] for k in ("auc", "eer", "bpcer", "apcer_pooled", "apcer_worst_species", "acer", "bpcer_at_apcer_1pct", "threshold")}
    species_accuracy = float((scored["species"] == scored["predicted"]).mean())

    log("training the classical baseline")
    baseline, baseline_scores = baseline_report(cfg, manifest, root, baseline_train, log)
    baseline_scores.to_csv(reports / "baseline_scores.csv", index=False)
    log("running robustness sweeps")
    robustness = robustness_report(model, test, root, threshold, robustness_samples, int(cfg.dataset.seed), log)

    thresholds, apcer, bpcer = error_curves(scored["label"], scored["score"])
    keep = np.unique(np.linspace(0, len(thresholds) - 1, 400).astype(int))
    result = {
        "model": model.meta.get("run", ""), "threshold": threshold, "operating_point": str(cfg.passive.operating_point),
        "test": {"samples": int(len(scored)), "subjects": int(scored["subject"].nunique()), "bona_fide": int((scored["label"] == 1).sum()), "attacks": int((scored["label"] == 0).sum())},
        "metrics": main, "operating_points": operating_point_table(scored, model.meta.get("thresholds", {})),
        "streams": streams, "species_accuracy": species_accuracy,
        "confusion": confusion_matrix(scored), "slices": slice_report(scored, threshold),
        "baseline": baseline, "robustness": robustness,
        "curve": {"threshold": thresholds[keep].tolist(), "apcer": apcer[keep].tolist(), "bpcer": bpcer[keep].tolist()},
    }
    (reports / "metrics.json").write_text(json.dumps(result, indent=2))
    m = main
    log(
        f"test AUC {m['auc']:.5f}  EER {100 * m['eer']:.2f}%  BPCER {100 * m['bpcer']:.2f}%  APCER pooled {100 * m['apcer_pooled']:.2f}%  "
        f"worst species {100 * m['apcer_worst_species']:.2f}%  ACER {100 * m['acer']:.2f}%"
    )
    log(f"baseline AUC {baseline['auc']:.4f}  EER {100 * baseline['eer']:.2f}%  ACER {100 * baseline['acer']:.2f}%")
    return result
