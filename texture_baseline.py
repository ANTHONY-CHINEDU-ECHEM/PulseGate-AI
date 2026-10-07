"""Classical baseline: colour texture descriptors with logistic regression.

Before deep networks, the strongest single image presentation attack detectors
described micro texture with local binary patterns computed in luminance and
chrominance spaces (Maatta et al., 2011; Boulkenafet et al., 2016). The baseline
keeps the neural network honest: if hand made features do as well, the network
is not earning its complexity.
"""
from __future__ import annotations

import cv2
import numpy as np

from .preprocess import context_view, normalize_scale, patch_view

_UNIFORM_LUT: np.ndarray | None = None


def _uniform_lut() -> np.ndarray:
    """Map the 256 LBP codes to 59 bins: 58 uniform patterns plus one bin for the rest."""
    global _UNIFORM_LUT
    if _UNIFORM_LUT is None:
        lut = np.zeros(256, dtype=np.uint8)
        next_bin = 0
        for code in range(256):
            bits = [(code >> i) & 1 for i in range(8)]
            transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
            if transitions <= 2:
                lut[code] = next_bin
                next_bin += 1
            else:
                lut[code] = 58
        _UNIFORM_LUT = lut
    return _UNIFORM_LUT


def lbp_histogram(channel: np.ndarray) -> np.ndarray:
    """Normalised histogram of uniform LBP codes in an 8 neighbourhood of radius 1."""
    c = channel.astype(np.int16)
    centre = c[1:-1, 1:-1]
    code = np.zeros(centre.shape, dtype=np.uint8)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(shifts):
        neighbour = c[1 + dy: c.shape[0] - 1 + dy, 1 + dx: c.shape[1] - 1 + dx]
        code |= ((neighbour >= centre).astype(np.uint8) << bit)
    hist = np.bincount(_uniform_lut()[code].ravel(), minlength=59).astype(np.float32)
    return hist / max(hist.sum(), 1.0)


def spectral_profile(gray: np.ndarray, bins: int = 12) -> np.ndarray:
    """Radially averaged log power spectrum, a compact summary of periodic structure."""
    g = gray.astype(np.float32)
    g = (g - g.mean()) * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    power = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2)
    h, w = g.shape
    ys, xs = np.mgrid[0:h, 0:w]
    radius = np.hypot(ys - h / 2, xs - w / 2) / (min(h, w) / 2)
    idx = np.clip((radius * bins).astype(int), 0, bins)
    sums = np.bincount(idx.ravel(), weights=power.ravel(), minlength=bins + 1)[:bins]
    counts = np.bincount(idx.ravel(), minlength=bins + 1)[:bins]
    return (sums / np.maximum(counts, 1)).astype(np.float32)


def texture_features(crop_bgr: np.ndarray, context_scale: float = 1.8, context_size: int = 112, patch_size: int = 96) -> np.ndarray:
    """Feature vector from the same two views the network sees."""
    crop = normalize_scale(crop_bgr, context_scale)
    parts = []
    for view in (context_view(crop, context_size), patch_view(crop, patch_size, (0.0, 0.0), context_scale)):
        ycrcb = cv2.cvtColor(view, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)
        for channel in (ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2], hsv[:, :, 0], hsv[:, :, 1]):
            parts.append(lbp_histogram(channel))
        parts.append(spectral_profile(ycrcb[:, :, 0]) / 20.0)
        stats = np.concatenate([ycrcb.reshape(-1, 3).mean(0), ycrcb.reshape(-1, 3).std(0), hsv.reshape(-1, 3)[:, 1:].mean(0)]) / 255.0
        parts.append(stats.astype(np.float32))
    return np.concatenate(parts)
