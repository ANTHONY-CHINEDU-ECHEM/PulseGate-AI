"""PyTorch dataset over the rendered presentation crops."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..models.network import CLASS_NAMES
from ..models.preprocess import context_view, normalize_scale, patch_view, to_tensor

CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def load_manifest(dataset_dir: str | Path) -> pd.DataFrame:
    path = Path(dataset_dir) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run 'python manage.py build_dataset' first.")
    return pd.read_csv(path, dtype={"background": str}).fillna({"background": ""})


class PresentationDataset(Dataset):
    """Returns ``(context, texture, label, species)`` tensors for one crop.

    Training applies light geometric and photometric jitter. The heavy lifting
    (camera, medium and scene variation) already happened in the simulator.
    """

    def __init__(self, frame: pd.DataFrame, root: str | Path, crop_cfg, train: bool = False, seed: int = 0,
                 degrade_probability: float = 0.0, clutter_probability: float = 0.0):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(root)
        self.context_scale = float(crop_cfg.context_scale)
        self.context_size = int(crop_cfg.context_size)
        self.patch_size = int(crop_cfg.patch_size)
        self.input_norm = str(crop_cfg.get("input_norm", "standardise"))
        self.train = bool(train)
        self.degrade_probability = float(degrade_probability)
        self.clutter_probability = float(clutter_probability)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.frame)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _degrade(self, crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Camera variation applied to genuine and attack crops alike.

        It never changes the label: a washed out, blurred or recompressed
        attack is still an attack, and so is a genuine face. Its purpose is to
        take away every cue that describes the camera instead of the medium.
        Without it the network learns that the look of the training database
        means "live", and rejects real people filmed by any other camera.
        """
        img = crop.astype(np.float32) / 255.0
        # tone and colour: what auto exposure, white balance and picture styles change
        if rng.random() < 0.8:
            luma = img @ np.array([0.114, 0.587, 0.299], dtype=np.float32)
            img = luma[:, :, None] + float(rng.uniform(0.25, 1.3)) * (img - luma[:, :, None])
            img = img * rng.uniform(0.92, 1.08, size=3).astype(np.float32)
            img = np.power(np.clip(img, 0.0, 1.0), float(rng.uniform(0.7, 1.45)))
            mean = img.mean()
            img = (img - mean) * float(rng.uniform(0.45, 1.25)) + mean + float(rng.uniform(-0.12, 0.12))
        crop = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
        # optics and codec: one or two of blur, resolution loss, noise, compression
        for _ in range(int(rng.integers(1, 3))):
            choice = rng.integers(0, 4)
            if choice == 0:
                crop = cv2.GaussianBlur(crop, (0, 0), float(rng.uniform(0.4, 1.6)))
            elif choice == 1:
                side = crop.shape[0]
                small = max(48, int(side * rng.uniform(0.5, 0.92)))
                crop = cv2.resize(cv2.resize(crop, (small, small), interpolation=cv2.INTER_AREA), (side, side),
                                  interpolation=cv2.INTER_LINEAR if rng.random() < 0.5 else cv2.INTER_CUBIC)
            elif choice == 2:
                noise = rng.normal(0.0, float(rng.uniform(1.0, 6.0)), crop.shape)
                crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            else:
                ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(18, 90))])
                crop = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else crop
        return crop

    @staticmethod
    def _clutter(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw door frames, wall corners, pictures and lamps beside the head.

        The training photographs have plain backdrops, so the only straight
        edges a network ever sees near a face are paper borders and device
        bezels. Without this step it flags every real room with a door frame or
        a window as an attack. Clutter is added to genuine and attack crops in
        the same way, only in the outer parts of the crop, so it carries no label.
        """
        side = crop.shape[0]
        img = crop.astype(np.float32)
        ys, xs = np.mgrid[0:side, 0:side].astype(np.float32) / side
        for _ in range(int(rng.integers(1, 4))):
            kind = rng.integers(0, 4)
            left = rng.random() < 0.5
            tilt = rng.normal(0.0, 0.04) * (ys - 0.5)
            if kind == 0:                                   # wall corner or door: one side of a line is lit differently
                if rng.random() < 0.8:
                    edge = rng.uniform(0.03, 0.21)
                    layer = (xs + tilt < edge) if left else (xs + tilt > 1.0 - edge)
                else:
                    layer = ys < rng.uniform(0.02, 0.1)
            elif kind == 1:                                 # frame, window, poster or shelf in a corner
                w, h = rng.uniform(0.06, 0.2), rng.uniform(0.1, 0.6)
                x0 = rng.uniform(-0.05, 0.21 - w) if left else rng.uniform(0.79, 1.05 - w)
                y0 = rng.uniform(-0.1, 0.75)
                layer = (xs > x0) & (xs < x0 + w) & (ys > y0) & (ys < y0 + h)
                if rng.random() < 0.4:                      # only the outline
                    t = rng.uniform(0.012, 0.03)
                    layer = layer & ~((xs > x0 + t) & (xs < x0 + w - t) & (ys > y0 + t) & (ys < y0 + h - t))
            elif kind == 2:                                 # pole, cable or frame edge
                width = rng.uniform(0.006, 0.03)
                centre = rng.uniform(0.02, 0.2) if left else rng.uniform(0.8, 0.98)
                layer = np.abs(xs + tilt - centre) < width
            else:                                           # lamp or bright window patch in an upper corner
                cx = rng.uniform(0.0, 0.16) if left else rng.uniform(0.84, 1.0)
                cy, rx, ry = rng.uniform(0.0, 0.3), rng.uniform(0.03, 0.1), rng.uniform(0.02, 0.12)
                layer = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 < 1.0
            layer = cv2.GaussianBlur(layer.astype(np.float32), (0, 0), float(rng.uniform(0.4, 2.2)))
            if kind == 3 or rng.random() < 0.45:            # an object with its own colour
                colour = rng.uniform(15, 250) + rng.normal(0, 14, size=3)
                alpha = layer[:, :, None] * float(rng.uniform(0.55, 1.0))
                img = img * (1 - alpha) + colour.astype(np.float32) * alpha
            else:                                           # the same wall under different light
                gain = float(rng.uniform(0.45, 0.85) if rng.random() < 0.5 else rng.uniform(1.2, 1.8))
                img = img * (1.0 + (gain - 1.0) * layer[:, :, None])
        return np.clip(img, 0, 255).astype(np.uint8)

    def _augment(self, crop: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if rng.random() < 0.5:
            crop = np.ascontiguousarray(crop[:, ::-1])
        if rng.random() < self.clutter_probability:
            crop = self._clutter(crop, rng)
        if rng.random() < self.degrade_probability:
            crop = self._degrade(crop, rng)
        gain = rng.uniform(0.9, 1.1)
        if abs(gain - 1.0) > 0.01:
            crop = np.clip(crop.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        side = crop.shape[0]
        sub = int(round(side * rng.uniform(0.86, 1.0)))
        x0 = int(rng.integers(0, side - sub + 1))
        y0 = int(rng.integers(0, side - sub + 1))
        context = context_view(crop[y0:y0 + sub, x0:x0 + sub], self.context_size)
        offset = (float(rng.uniform(-0.25, 0.25)), float(rng.uniform(-0.3, 0.32)))
        patch = patch_view(crop, self.patch_size, offset)
        return context, patch

    def __getitem__(self, index: int):
        cv2.setNumThreads(0)
        row = self.frame.iloc[index]
        crop = cv2.imread(str(self.root / row["path"]), cv2.IMREAD_COLOR)
        if crop is None:
            raise FileNotFoundError(self.root / row["path"])
        crop = normalize_scale(crop, self.context_scale)
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, index))
            context, patch = self._augment(crop, rng)
        else:
            context = context_view(crop, self.context_size)
            patch = patch_view(crop, self.patch_size)
        return (
            torch.from_numpy(to_tensor(context, self.input_norm)), torch.from_numpy(to_tensor(patch, self.input_norm)),
            torch.tensor(float(row["label"])), torch.tensor(CLASS_INDEX[row["species"]]),
        )
