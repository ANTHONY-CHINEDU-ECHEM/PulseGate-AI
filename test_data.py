import numpy as np
import pandas as pd
import pytest

from pulsegate.data.builder import plan_tasks, split_subjects
from pulsegate.data.muct import NAME_PATTERN, index_muct


def test_subject_split_is_disjoint_complete_and_reproducible():
    subjects = [f"s{i:03d}" for i in range(276)]
    fractions = {"train": 0.7, "val": 0.15, "test": 0.15}
    a = split_subjects(subjects, fractions, seed=1)
    assert a == split_subjects(subjects, fractions, seed=1)
    assert a != split_subjects(subjects, fractions, seed=2)
    assert set(a) == set(subjects)
    counts = pd.Series(a).value_counts()
    assert counts["train"] == 193 and counts["val"] == 41 and counts["test"] == 42


def fake_sources(n_subjects=12, per_subject=3) -> pd.DataFrame:
    rows = []
    for s in range(n_subjects):
        for k in range(per_subject):
            rows.append({
                "file": f"i{s:03d}q{'abc'[k]}-fn.jpg", "subject": f"muct{s:03d}", "lighting": "q", "camera": "abc"[k],
                "gender": "female", "glasses": "no", "box": [100, 100, 200, 200], "points": [[0, 0]] * 5, "det_score": 0.9,
            })
    return pd.DataFrame(rows)


def test_task_plan_never_mixes_subjects_across_partitions(cfg):
    tasks = plan_tasks(cfg, fake_sources())
    per_source = cfg.dataset.live_variants + len(cfg.dataset.species)
    assert len(tasks) == 36 * per_source
    assert len({t["sample_id"] for t in tasks}) == len(tasks)
    assert len({t["seed"] for t in tasks}) == len(tasks)
    split_of_subject = {}
    for t in tasks:
        assert split_of_subject.setdefault(t["source"]["subject"], t["split"]) == t["split"]
    for t in tasks:
        if t["background"] is not None:
            assert split_of_subject[t["background"]["subject"]] == t["split"]      # no leakage through backgrounds
            assert t["background"]["subject"] != t["source"]["subject"]
        else:
            assert t["species"] == "live"


def test_task_plan_is_deterministic(cfg):
    a, b = plan_tasks(cfg, fake_sources()), plan_tasks(cfg, fake_sources())
    assert [(t["sample_id"], t["species"], t["seed"]) for t in a] == [(t["sample_id"], t["species"], t["seed"]) for t in b]


def test_muct_file_names_are_parsed(tmp_path):
    for name in ("i000qa-fn.jpg", "i123ze-mg.jpg", "notes.txt", "i12qa-fn.jpg"):
        (tmp_path / name).write_bytes(b"x")
    frame = index_muct(tmp_path)
    assert list(frame["file"]) == ["i000qa-fn.jpg", "i123ze-mg.jpg"]
    row = frame.iloc[1]
    assert (row["subject"], row["lighting"], row["camera"], row["gender"], row["glasses"]) == ("muct123", "z", "e", "male", "yes")
    assert NAME_PATTERN.match("../evil.jpg") is None


def test_empty_directory_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        index_muct(tmp_path)


def test_dataset_applies_identical_preprocessing_to_both_views(cfg, tmp_path):
    torch = pytest.importorskip("torch")
    import cv2

    from pulsegate.data.dataset import PresentationDataset

    rng = np.random.default_rng(0)
    rows = []
    for i, species in enumerate(["live", "replay_phone"]):
        cv2.imwrite(str(tmp_path / f"{i}.jpg"), rng.integers(0, 255, (260 + 40 * i, 260 + 40 * i, 3)).astype(np.uint8))
        rows.append({"path": f"{i}.jpg", "label": int(species == "live"), "species": species})
    frame = pd.DataFrame(rows)
    eval_set = PresentationDataset(frame, tmp_path, cfg.crop, train=False)
    context, texture, label, species = eval_set[1]
    assert context.shape == (3, 112, 112) and texture.shape == (3, 96, 96)
    assert label.item() == 0.0 and species.item() == 3
    assert torch.equal(eval_set[1][0], eval_set[1][0])            # evaluation is deterministic
    train_set = PresentationDataset(frame, tmp_path, cfg.crop, train=True, seed=1)
    first = train_set[0][0]
    assert torch.equal(first, train_set[0][0])                     # the same epoch repeats exactly
    train_set.set_epoch(1)
    assert not torch.equal(first, train_set[0][0])                 # a new epoch draws new augmentation


def test_clutter_stays_out_of_the_face_area(cfg):
    pytest.importorskip("torch")
    from pulsegate.data.dataset import PresentationDataset

    rng = np.random.default_rng(0)
    crop = rng.integers(60, 200, (300, 300, 3)).astype(np.uint8)
    changed_outside = 0
    for seed in range(30):
        out = PresentationDataset._clutter(crop, np.random.default_rng(seed))
        assert out.shape == crop.shape and out.dtype == np.uint8
        centre = (slice(48, 270), slice(96, 204))                 # the face and what is below it
        assert np.abs(out[centre].astype(int) - crop[centre].astype(int)).max() <= 1
        changed_outside += int(np.abs(out.astype(int) - crop.astype(int)).max() > 10)
    assert changed_outside >= 28                                   # something is drawn almost every time


def test_camera_degradation_keeps_shape_and_changes_pixels(cfg):
    pytest.importorskip("torch")
    from pulsegate.data.dataset import PresentationDataset

    dataset = PresentationDataset(pd.DataFrame({"path": [], "label": [], "species": []}), ".", cfg.crop, train=True)
    rng = np.random.default_rng(1)
    crop = rng.integers(0, 255, (240, 240, 3)).astype(np.uint8)
    for seed in range(20):
        out = dataset._degrade(crop, np.random.default_rng(seed))
        assert out.shape == crop.shape and out.dtype == np.uint8
        assert not np.array_equal(out, crop)
