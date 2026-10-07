"""Presentation attack detection metrics following ISO/IEC 30107 part 3.

Score convention: the score is the estimated probability that a presentation is
bona fide (a live person). A presentation is accepted when ``score >= threshold``.

* APCER, attack presentation classification error rate: share of attacks accepted.
  The standard reports it per attack instrument species and takes the worst one.
* BPCER, bona fide presentation classification error rate: share of genuine
  presentations rejected.
* ACER: mean of the worst species APCER and BPCER.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_arrays(labels: Iterable, scores: Iterable) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels).astype(int).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError("labels and scores must have the same length")
    return y, s


def error_curves(labels: Iterable, scores: Iterable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thresholds with the pooled APCER and BPCER at each of them.

    The curves are exact: every distinct score is a candidate threshold, plus
    one threshold above the maximum (nothing accepted).
    """
    y, s = _as_arrays(labels, scores)
    live, attack = np.sort(s[y == 1]), np.sort(s[y == 0])
    if len(live) == 0 or len(attack) == 0:
        raise ValueError("both classes are needed to compute error curves")
    thresholds = np.unique(np.concatenate([s, [np.nextafter(s.max(), np.inf)]]))
    apcer = 1.0 - np.searchsorted(attack, thresholds, side="left") / len(attack)
    bpcer = np.searchsorted(live, thresholds, side="left") / len(live)
    return thresholds, apcer, bpcer


def equal_error_rate(labels: Iterable, scores: Iterable) -> tuple[float, float]:
    """Equal error rate and the threshold where APCER and BPCER cross."""
    thresholds, apcer, bpcer = error_curves(labels, scores)
    i = int(np.argmin(np.abs(apcer - bpcer)))
    return float((apcer[i] + bpcer[i]) / 2.0), float(thresholds[i])


def roc_auc(labels: Iterable, scores: Iterable) -> float:
    """Area under the ROC curve from the rank statistic, ties counted as one half."""
    y, s = _as_arrays(labels, scores)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def threshold_at_apcer(labels: Iterable, scores: Iterable, target: float) -> float:
    """Lowest threshold whose pooled APCER does not exceed ``target``."""
    thresholds, apcer, _ = error_curves(labels, scores)
    ok = np.where(apcer <= target)[0]
    return float(thresholds[ok[0]]) if len(ok) else float(thresholds[-1])


def threshold_at_bpcer(labels: Iterable, scores: Iterable, target: float) -> float:
    """Highest threshold whose BPCER does not exceed ``target``."""
    thresholds, _, bpcer = error_curves(labels, scores)
    ok = np.where(bpcer <= target)[0]
    return float(thresholds[ok[-1]]) if len(ok) else float(thresholds[0])


def pad_report(labels: Iterable, scores: Iterable, species: Iterable[str], threshold: float) -> dict:
    """APCER per species, worst case APCER, BPCER and ACER at a fixed threshold."""
    y, s = _as_arrays(labels, scores)
    sp = np.asarray(list(species))
    live = s[y == 1]
    bpcer = float((live < threshold).mean()) if len(live) else float("nan")
    per_species = {}
    for name in sorted(set(sp[y == 0])):
        mask = (sp == name) & (y == 0)
        per_species[str(name)] = {"apcer": float((s[mask] >= threshold).mean()), "count": int(mask.sum())}
    attacks = s[y == 0]
    apcer_pooled = float((attacks >= threshold).mean()) if len(attacks) else float("nan")
    apcer_max = max((v["apcer"] for v in per_species.values()), default=float("nan"))
    return {
        "threshold": float(threshold),
        "bpcer": bpcer,
        "apcer_pooled": apcer_pooled,
        "apcer_worst_species": float(apcer_max),
        "acer": float((apcer_max + bpcer) / 2.0),
        "accuracy": float(((s >= threshold).astype(int) == y).mean()),
        "per_species": per_species,
        "bona_fide_count": int(len(live)),
        "attack_count": int(len(attacks)),
    }


def summarize(labels: Iterable, scores: Iterable, species: Iterable[str], threshold: float) -> dict:
    """Threshold free metrics plus the fixed threshold report."""
    y, s = _as_arrays(labels, scores)
    eer, eer_threshold = equal_error_rate(y, s)
    report = pad_report(y, s, species, threshold)
    report.update({
        "auc": roc_auc(y, s),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "bpcer_at_apcer_5pct": bpcer_at(y, s, 0.05),
        "bpcer_at_apcer_1pct": bpcer_at(y, s, 0.01),
        "bpcer_at_apcer_0p1pct": bpcer_at(y, s, 0.001),
    })
    return report


def bpcer_at(labels: Iterable, scores: Iterable, apcer_target: float) -> float:
    y, s = _as_arrays(labels, scores)
    threshold = threshold_at_apcer(y, s, apcer_target)
    return float((s[y == 1] < threshold).mean())


def bootstrap_ci(labels: Iterable, scores: Iterable, groups: Iterable, metric, n_boot: int = 500, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Confidence interval from resampling whole groups (subjects), not single images.

    Images of one person are correlated, so resampling images would understate
    the uncertainty.
    """
    y, s = _as_arrays(labels, scores)
    g = np.asarray(list(groups))
    unique = np.unique(g)
    index = {name: np.where(g == name)[0] for name in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index[name] for name in chosen])
        if len(set(y[idx])) < 2:
            continue
        values.append(metric(y[idx], s[idx]))
    if not values:
        return float("nan"), float("nan")
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def expected_calibration_error(labels: Iterable, scores: Iterable, bins: int = 15) -> float:
    y, s = _as_arrays(labels, scores)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (s >= lo) & (s < hi) if hi < 1.0 else (s >= lo) & (s <= hi)
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - s[mask].mean())
    return float(total)
