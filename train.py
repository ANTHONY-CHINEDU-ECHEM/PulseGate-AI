"""Training loop for PulseGateNet."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import PresentationDataset, load_manifest
from ..evaluation.metrics import equal_error_rate, pad_report, roc_auc, threshold_at_apcer, threshold_at_bpcer
from ..models.export import export_onnx
from ..models.network import CLASS_NAMES, PulseGateNet, count_parameters


class WeightAverage:
    """Exponential moving average of the weights.

    The averaged network is the one that is validated, saved and exported. It
    moves more smoothly than the raw weights and usually generalises better,
    which matters here because the model can memorise the training people.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))      # short warm up
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                self.shadow[key].copy_(value)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(dataset: PresentationDataset, batch_size: int, shuffle: bool, workers: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=shuffle,
        persistent_workers=workers > 0, generator=generator,
    )


@torch.no_grad()
def predict(model: PulseGateNet, loader: DataLoader) -> dict[str, np.ndarray]:
    """Raw logits of all three heads and the species prediction for a whole loader."""
    model.eval()
    out: dict[str, list] = {"logit": [], "logit_context": [], "logit_texture": [], "species": [], "label": []}
    for context, texture, label, _ in loader:
        pred = model(context.contiguous(memory_format=torch.channels_last), texture.contiguous(memory_format=torch.channels_last))
        for key in ("logit", "logit_context", "logit_texture"):
            out[key].append(pred[key].numpy())
        out["species"].append(pred["species"].argmax(dim=1).numpy())
        out["label"].append(label.numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar that minimises validation log loss."""
    z = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_t.exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(np.clip(log_t.exp().item(), 0.25, 10.0))


def operating_points(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Decision thresholds fixed on validation data and reused untouched on the test set."""
    _, eer_threshold = equal_error_rate(labels, scores)
    return {
        "eer": eer_threshold,
        "bpcer_at_apcer_5pct": threshold_at_apcer(labels, scores, 0.05),
        "bpcer_at_apcer_1pct": threshold_at_apcer(labels, scores, 0.01),
        "apcer_at_bpcer_1pct": threshold_at_bpcer(labels, scores, 0.01),
    }


def select_frames(cfg: Config, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = manifest[manifest["split"] == "train"]
    val = manifest[manifest["split"] == "val"]
    excluded = list(cfg.train.exclude_species or [])
    if excluded:
        train = train[~train["species"].isin(excluded)]
        val = val[~val["species"].isin(excluded)]
    cap = int(cfg.train.max_train_samples or 0)
    if cap and len(train) > cap:
        train = train.sample(n=cap, random_state=int(cfg.train.seed))
    return train, val


def train_model(cfg: Config, log=print, frames: tuple[pd.DataFrame, pd.DataFrame] | None = None,
                init_checkpoint: str | Path | None = None) -> dict:
    """Train, calibrate, export and write the model card data. Returns the metadata dictionary.

    ``frames`` replaces the default train and validation tables (paths may be
    absolute), ``init_checkpoint`` starts from existing weights. Both are used
    for fine tuning on a user's own captures.
    """
    t = cfg.train
    set_seed(int(t.seed))
    torch.set_num_threads(max(1, torch.get_num_threads()))
    dataset_dir = cfg.path("paths.dataset_dir")
    models_dir = cfg.path("paths.models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)
    if frames is None:
        train_frame, val_frame = select_frames(cfg, load_manifest(dataset_dir))
    else:
        train_frame, val_frame = frames
    log(f"train {len(train_frame)} samples, validation {len(val_frame)} samples, excluded species {list(t.exclude_species or [])}")

    train_set = PresentationDataset(train_frame, dataset_dir, cfg.crop, train=True, seed=int(t.seed),
                                    degrade_probability=float(t.get("degrade_probability", 0.0)),
                                    clutter_probability=float(t.get("clutter_probability", 0.0)))
    val_set = PresentationDataset(val_frame, dataset_dir, cfg.crop, train=False)
    train_loader = make_loader(train_set, int(t.batch_size), True, int(t.num_workers), int(t.seed))
    val_loader = make_loader(val_set, 256, False, int(t.num_workers), int(t.seed))

    dropout = float(t.get("dropout", 0.2))
    model = PulseGateNet(width=float(t.width), dropout=dropout, stream_dropout=float(t.stream_dropout)).to(memory_format=torch.channels_last)
    averaged = PulseGateNet(width=float(t.width), dropout=dropout).to(memory_format=torch.channels_last)
    if init_checkpoint is None and t.get("init_checkpoint"):
        from ..config import resolve_path

        init_checkpoint = resolve_path(str(t.init_checkpoint))
    if init_checkpoint is not None:
        model.load_state_dict(torch.load(init_checkpoint, map_location="cpu", weights_only=False)["state_dict"])
        log(f"starting from {Path(init_checkpoint).name}")
    log(f"PulseGateNet with {count_parameters(model):,} parameters")
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim <= 1 else decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(t.weight_decay)}, {"params": no_decay, "weight_decay": 0.0}], lr=float(t.lr),
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * int(t.epochs)
    warmup = max(1, int(float(t.warmup_epochs) * steps_per_epoch))

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    ema_decay = float(t.get("ema_decay", 0.0))
    ema = WeightAverage(model, ema_decay) if ema_decay > 0 else None

    n_live = int((train_frame["label"] == 1).sum())
    n_attack = int((train_frame["label"] == 0).sum())
    pos_weight = torch.tensor(n_attack / max(1, n_live))       # balances the two classes
    smooth = float(t.label_smoothing)

    def live_loss(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        target = label * (1.0 - smooth) + (1.0 - label) * smooth
        return F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight)

    counts = train_frame["species"].value_counts()
    species_weights = torch.zeros(len(CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        if counts.get(name, 0) > 0:
            species_weights[i] = len(train_frame) / (len(counts) * counts[name])     # inverse frequency
    species_loss = nn.CrossEntropyLoss(weight=species_weights)

    history: list[dict] = []
    best = {"eer": 1.0, "epoch": -1}
    run = str(t.run_name)
    checkpoint = models_dir / f"{run}.pt"
    resume_path = models_dir / f"{run}.resume.pt"
    first_epoch, elapsed_before = 0, 0.0
    if bool(t.get("resume", True)) and resume_path.exists():
        # An interrupted run continues from the last finished epoch instead of starting over.
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("epochs") == int(t.epochs) and saved.get("train_samples") == len(train_frame):
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            history, best, first_epoch = saved["history"], saved["best"], saved["epoch"]
            if ema is not None and saved.get("ema") is not None:
                ema.shadow, ema.updates = saved["ema"], saved["ema_updates"]
            elapsed_before = history[-1]["minutes"] * 60.0 if history else 0.0
            log(f"resuming after epoch {first_epoch}")
    start = time.time() - elapsed_before
    step = first_epoch * steps_per_epoch
    for epoch in range(first_epoch, int(t.epochs)):
        model.train()
        train_set.set_epoch(epoch)
        running, seen, correct = 0.0, 0, 0
        for context, texture, label, species in train_loader:
            context = context.contiguous(memory_format=torch.channels_last)
            texture = texture.contiguous(memory_format=torch.channels_last)
            out = model(context, texture)
            loss = (
                live_loss(out["logit"], label)
                + float(t.aux_stream_weight) * (live_loss(out["logit_context"], label) + live_loss(out["logit_texture"], label))
                + float(t.aux_species_weight) * species_loss(out["species"], species)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            step += 1
            running += loss.item() * len(label)
            seen += len(label)
            correct += int(((out["logit"] > 0).float() == label).sum())
            if step % 50 == 0:
                log(f"  epoch {epoch + 1} step {step % steps_per_epoch or steps_per_epoch}/{steps_per_epoch}  loss {running / seen:.4f}  {(time.time() - start) / 60.0:.1f} min")
        if ema is not None:
            ema.copy_to(averaged)
        else:
            averaged.load_state_dict(model.state_dict())
        val = predict(averaged, val_loader)
        scores = sigmoid(val["logit"])
        eer, _ = equal_error_rate(val["label"], scores)
        row = {
            "epoch": epoch + 1, "train_loss": running / seen, "train_accuracy": correct / seen,
            "val_eer": eer, "val_auc": roc_auc(val["label"], scores),
            "val_eer_context": equal_error_rate(val["label"], sigmoid(val["logit_context"]))[0],
            "val_eer_texture": equal_error_rate(val["label"], sigmoid(val["logit_texture"]))[0],
            "lr": scheduler.get_last_lr()[0], "minutes": (time.time() - start) / 60.0,
        }
        history.append(row)
        log(
            f"epoch {row['epoch']:2d}  loss {row['train_loss']:.4f}  acc {row['train_accuracy']:.4f}  "
            f"val EER {100 * eer:.2f}%  context {100 * row['val_eer_context']:.2f}%  texture {100 * row['val_eer_texture']:.2f}%  "
            f"AUC {row['val_auc']:.5f}  {row['minutes']:.1f} min"
        )
        # keep the best epoch; later epochs win ties because they are better converged
        if eer <= best["eer"] + 1e-9:
            best = {"eer": eer, "epoch": epoch + 1}
            torch.save({"state_dict": averaged.state_dict(), "width": float(t.width), "classes": CLASS_NAMES}, checkpoint)
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "history": history, "best": best, "epoch": epoch + 1, "epochs": int(t.epochs), "train_samples": len(train_frame),
            "ema": None if ema is None else ema.shadow, "ema_updates": 0 if ema is None else ema.updates,
        }, resume_path)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = averaged
    model.load_state_dict(state["state_dict"])
    val = predict(model, val_loader)
    temperature = fit_temperature(val["logit"], val["label"])
    scores = sigmoid(val["logit"] / temperature)
    thresholds = operating_points(val["label"], scores)
    point = str(cfg.passive.operating_point)
    report = pad_report(val["label"], scores, val_frame["species"].tolist(), thresholds[point])

    onnx_path = models_dir / f"{run}.onnx"
    export_onnx(model, onnx_path, int(cfg.crop.context_size), int(cfg.crop.patch_size))
    meta = {
        "name": "PulseGateNet",
        "run": run,
        "parameters": count_parameters(model),
        "classes": list(CLASS_NAMES),
        "crop": {"context_scale": float(cfg.crop.context_scale), "context_size": int(cfg.crop.context_size), "patch_size": int(cfg.crop.patch_size),
                 "input_norm": str(cfg.crop.get("input_norm", "standardise"))},
        "temperature": temperature,
        "thresholds": thresholds,
        "operating_point": point,
        "threshold": thresholds[point],
        "best_epoch": best["epoch"],
        "validation": {"eer": best["eer"], "auc": roc_auc(val["label"], scores), **{k: report[k] for k in ("bpcer", "apcer_pooled", "apcer_worst_species", "acer")}},
        "train_samples": int(len(train_frame)), "val_samples": int(len(val_frame)),
        "excluded_species": list(t.exclude_species or []),
        "train_config": {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in t.items()},
        "train_minutes": round((time.time() - start) / 60.0, 1),
        "history": history,
    }
    if init_checkpoint is not None:
        # keep the record of the stage this run continued from
        previous = Path(init_checkpoint).with_suffix(".json")
        if previous.exists():
            earlier = json.loads(previous.read_text())
            meta["earlier_stages"] = earlier.get("earlier_stages", []) + [{
                "run": earlier.get("run"), "history": earlier.get("history", []), "best_epoch": earlier.get("best_epoch"),
                "validation": earlier.get("validation"), "train_config": earlier.get("train_config"), "train_minutes": earlier.get("train_minutes"),
            }]
    meta_path = models_dir / f"{run}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    resume_path.unlink(missing_ok=True)
    log(f"best epoch {best['epoch']} with validation EER {100 * best['eer']:.2f}%, temperature {temperature:.3f}")
    log(f"saved {checkpoint.name}, {onnx_path.name} and {meta_path.name} in {models_dir}")
    return meta


def recalibrate_model(cfg: Config, run: str | None = None, log=print) -> dict:
    """Refit temperature and thresholds of a trained checkpoint and export it again.

    Useful after changing ``passive.operating_point`` or the validation data.
    Training history and settings stored with the model are kept.
    """
    from ..models.gradcam import load_checkpoint

    run = run or str(cfg.train.run_name)
    models_dir = cfg.path("paths.models_dir")
    meta_path = models_dir / f"{run}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"name": "PulseGateNet", "run": run}
    dataset_dir = cfg.path("paths.dataset_dir")
    _, val_frame = select_frames(cfg, load_manifest(dataset_dir))
    model = load_checkpoint(models_dir / f"{run}.pt").to(memory_format=torch.channels_last)
    val = predict(model, make_loader(PresentationDataset(val_frame, dataset_dir, cfg.crop, train=False), 256, False, int(cfg.train.num_workers), 0))
    temperature = fit_temperature(val["logit"], val["label"])
    scores = sigmoid(val["logit"] / temperature)
    thresholds = operating_points(val["label"], scores)
    point = str(cfg.passive.operating_point)
    report = pad_report(val["label"], scores, val_frame["species"].tolist(), thresholds[point])
    export_onnx(model.to(memory_format=torch.contiguous_format), models_dir / f"{run}.onnx", int(cfg.crop.context_size), int(cfg.crop.patch_size))
    meta.update({
        "temperature": temperature, "thresholds": thresholds, "operating_point": point, "threshold": thresholds[point],
        "validation": {"eer": equal_error_rate(val["label"], scores)[0], "auc": roc_auc(val["label"], scores),
                       **{k: report[k] for k in ("bpcer", "apcer_pooled", "apcer_worst_species", "acer")}},
        "val_samples": int(len(val_frame)),
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"{run}: operating point {point} at threshold {thresholds[point]:.4f}, validation BPCER {100 * report['bpcer']:.2f}%, pooled APCER {100 * report['apcer_pooled']:.2f}%")
    return meta
