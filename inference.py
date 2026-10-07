"""Runtime wrapper around the exported network. Needs only ONNX Runtime, not PyTorch."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..config import Config
from ..vision.geometry import crop_square, square_box
from .preprocess import build_inputs


@dataclass(frozen=True)
class PassiveScore:
    """Single frame verdict of the passive model."""

    live: float                 # calibrated probability that the presentation is bona fide
    context: float              # the same from the context stream alone
    texture: float              # the same from the texture stream alone
    species: str                # most likely class name
    species_probs: dict

    def as_dict(self) -> dict:
        return {
            "live": round(self.live, 4), "context": round(self.context, 4), "texture": round(self.texture, 4),
            "species": self.species, "species_probs": {k: round(v, 4) for k, v in self.species_probs.items()},
        }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


class PassiveLiveness:
    """Scores a face crop with PulseGateNet."""

    def __init__(self, cfg: Config, model_path: str | Path | None = None, meta_path: str | Path | None = None, threads: int = 1):
        model_path = Path(model_path) if model_path else cfg.path("assets.passive_model")
        meta_path = Path(meta_path) if meta_path else cfg.path("assets.passive_meta")
        if not model_path.exists():
            raise FileNotFoundError(f"passive model not found: {model_path}. Train one with 'python manage.py train'.")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        crop = self.meta.get("crop", {})
        self.context_scale = float(crop.get("context_scale", cfg.crop.context_scale))
        self.context_size = int(crop.get("context_size", cfg.crop.context_size))
        self.patch_size = int(crop.get("patch_size", cfg.crop.patch_size))
        self.input_norm = str(crop.get("input_norm", "standardise"))
        self.temperature = float(self.meta.get("temperature", 1.0))
        self.classes = list(self.meta.get("classes", ["live", "attack"]))
        self._fixed_threshold = cfg.passive.threshold
        self.threshold = self.threshold_at(str(cfg.passive.operating_point))
        self.min_face_px = float(cfg.crop.min_face_px)
        self.patches = int(cfg.passive.tta_patches)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self._session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])

    def threshold_at(self, operating_point: str) -> float:
        """Decision threshold for a named operating point. An explicit ``passive.threshold`` overrides all of them."""
        if self._fixed_threshold is not None:
            return float(self._fixed_threshold)
        return float(self.meta.get("thresholds", {}).get(operating_point, self.meta.get("threshold", 0.5)))

    def score_crop(self, crop_bgr: np.ndarray, patches: int | None = None) -> PassiveScore:
        """Score a context crop produced by :func:`crop_face`."""
        context, texture = build_inputs(crop_bgr, self.context_scale, self.context_size, self.patch_size, patches or self.patches, self.input_norm)
        logits, species = self._session.run(None, {"context": context, "texture": texture})
        mean_logits = logits.mean(axis=0) / self.temperature
        probs = _sigmoid(mean_logits)
        sp = species.mean(axis=0)
        sp = np.exp(sp - sp.max())
        sp /= sp.sum()
        names = self.classes if len(self.classes) == len(sp) else [str(i) for i in range(len(sp))]
        return PassiveScore(
            live=float(probs[0]), context=float(probs[1]), texture=float(probs[2]),
            species=names[int(sp.argmax())], species_probs={n: float(v) for n, v in zip(names, sp)},
        )

    def crop_face(self, frame_bgr: np.ndarray, box: np.ndarray) -> np.ndarray:
        cx, cy, side = square_box(box, self.context_scale)
        return crop_square(frame_bgr, cx, cy, side)

    def score_frame(self, frame_bgr: np.ndarray, box: np.ndarray) -> PassiveScore | None:
        """Score the face at ``box``. Returns ``None`` when the face is too small to judge."""
        if max(float(box[2]), float(box[3])) < self.min_face_px:
            return None
        return self.score_crop(self.crop_face(frame_bgr, box))
