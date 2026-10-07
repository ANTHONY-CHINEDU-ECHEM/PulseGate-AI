import numpy as np
import pytest

from pulsegate.evaluation import metrics as m


def test_perfect_separation():
    labels = np.array([1] * 50 + [0] * 50)
    scores = np.concatenate([np.linspace(0.8, 1.0, 50), np.linspace(0.0, 0.2, 50)])
    eer, threshold = m.equal_error_rate(labels, scores)
    assert eer == 0.0
    assert 0.2 < threshold <= 0.8
    assert m.roc_auc(labels, scores) == 1.0


def test_random_scores_give_chance_level():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 6000)
    scores = rng.random(6000)
    assert abs(m.roc_auc(labels, scores) - 0.5) < 0.03
    assert abs(m.equal_error_rate(labels, scores)[0] - 0.5) < 0.03


def test_auc_matches_scikit_learn_including_ties():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 500)
    scores = np.round(rng.random(500) * 0.6 + labels * 0.25, 2)       # rounding creates ties
    assert m.roc_auc(labels, scores) == pytest.approx(sklearn_metrics.roc_auc_score(labels, scores), abs=1e-12)


def test_error_rate_definitions():
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.2, 0.1, 0.1, 0.95, 0.05])
    species = ["live"] * 4 + ["print", "print", "print", "replay", "replay", "replay"]
    report = m.pad_report(labels, scores, species, threshold=0.5)
    assert report["bpcer"] == pytest.approx(1 / 4)                    # one genuine sample rejected
    assert report["apcer_pooled"] == pytest.approx(2 / 6)             # two attacks accepted
    assert report["per_species"]["print"]["apcer"] == pytest.approx(1 / 3)
    assert report["per_species"]["replay"]["apcer"] == pytest.approx(1 / 3)
    assert report["apcer_worst_species"] == pytest.approx(1 / 3)
    assert report["acer"] == pytest.approx((1 / 3 + 1 / 4) / 2)


def test_worst_species_drives_acer():
    labels = np.array([1] * 10 + [0] * 20)
    scores = np.array([0.9] * 10 + [0.1] * 10 + [0.9] * 5 + [0.1] * 5)
    species = ["live"] * 10 + ["easy"] * 10 + ["hard"] * 10
    report = m.pad_report(labels, scores, species, 0.5)
    assert report["apcer_pooled"] == pytest.approx(0.25)
    assert report["apcer_worst_species"] == pytest.approx(0.5)


def test_threshold_at_apcer_respects_the_target():
    rng = np.random.default_rng(2)
    labels = np.array([1] * 2000 + [0] * 2000)
    scores = np.concatenate([rng.normal(0.7, 0.15, 2000), rng.normal(0.3, 0.15, 2000)])
    for target in (0.05, 0.01):
        threshold = m.threshold_at_apcer(labels, scores, target)
        assert (scores[labels == 0] >= threshold).mean() <= target
        lower = np.nextafter(threshold, -np.inf)
        assert (scores[labels == 0] >= lower).mean() >= (scores[labels == 0] >= threshold).mean()
    assert m.bpcer_at(labels, scores, 0.01) > m.bpcer_at(labels, scores, 0.05)


def test_threshold_at_bpcer_respects_the_target():
    rng = np.random.default_rng(3)
    labels = np.array([1] * 1000 + [0] * 1000)
    scores = np.concatenate([rng.normal(0.7, 0.15, 1000), rng.normal(0.3, 0.15, 1000)])
    threshold = m.threshold_at_bpcer(labels, scores, 0.01)
    assert (scores[labels == 1] < threshold).mean() <= 0.01


def test_bootstrap_resamples_groups():
    rng = np.random.default_rng(4)
    labels = np.array([1] * 300 + [0] * 300)
    scores = np.concatenate([rng.normal(0.7, 0.2, 300), rng.normal(0.3, 0.2, 300)])
    groups = np.tile(np.arange(30), 20)
    lo, hi = m.bootstrap_ci(labels, scores, groups, m.roc_auc, n_boot=100)
    assert lo <= m.roc_auc(labels, scores) <= hi


def test_single_class_is_an_error():
    with pytest.raises(ValueError):
        m.error_curves([1, 1, 1], [0.1, 0.2, 0.3])


def test_calibration_error_of_calibrated_scores_is_small():
    rng = np.random.default_rng(5)
    scores = rng.random(20000)
    labels = (rng.random(20000) < scores).astype(int)
    assert m.expected_calibration_error(labels, scores) < 0.02
    assert m.expected_calibration_error(labels, np.clip(scores * 0.5, 0, 1)) > 0.15
