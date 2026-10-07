"""Input preparation shared by training, evaluation and live inference.

The network looks at a face twice:

* the *context view* is the whole crop (face plus surroundings) resized to a
  small square. Bezels, paper edges, hands and global colour live here.
* the *texture view* is a patch cut from the middle of the face at the native
  pixel pitch of the camera, never resized. Moire, halftone dots, pixel grids
  and the noise statistics of a recaptured image live here.
"""
from __future__ import annotations

import cv2
import numpy as np

MAX_FACE_PX = 256      # larger faces are reduced so the texture view matches training conditions


def face_px_of(crop: np.ndarray, context_scale: float) -> float:
    return crop.shape[0] / float(context_scale)


def normalize_scale(crop: np.ndarray, context_scale: float, max_face: int = MAX_FACE_PX) -> np.ndarray:
    """Shrink very large faces so that the face side is at most ``max_face`` pixels."""
    face = face_px_of(crop, context_scale)
    if face <= max_face:
        return crop
    side = int(round(crop.shape[0] * max_face / face))
    return cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)


def context_view(crop: np.ndarray, size: int) -> np.ndarray:
    if crop.shape[0] == size:
        return crop
    interp = cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interp)


def patch_view(crop: np.ndarray, size: int, offset: tuple[float, float] = (0.0, 0.0), context_scale: float = 1.8) -> np.ndarray:
    """Native resolution patch. ``offset`` is in face widths relative to the crop centre."""
    side = crop.shape[0]
    if side < size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        side = size
    face = side / float(context_scale)
    cx = side / 2.0 + offset[0] * face
    cy = side / 2.0 + offset[1] * face
    x0 = int(np.clip(round(cx - size / 2.0), 0, side - size))
    y0 = int(np.clip(round(cy - size / 2.0), 0, side - size))
    return crop[y0:y0 + size, x0:x0 + size]


def to_tensor(image_bgr: np.ndarray, norm: str = "standardise") -> np.ndarray:
    """uint8 BGR image to float32 RGB, channels first, standardised per channel.

    Every view is shifted and scaled to zero mean and unit spread in each
    colour channel. Exposure, contrast and colour cast of the camera therefore
    never reach the network. Those properties differ far more between two
    webcams than between a face and a photograph of it, so a network that sees
    them learns the camera instead of the attack.

    The alternative ``norm="fixed"`` maps 0..255 to minus one..one without looking
    at the image. It exists for the ablation that shows why it is not the default.
    """
    rgb = image_bgr[:, :, ::-1].astype(np.float32)
    if norm == "fixed":
        return np.ascontiguousarray((rgb / 127.5 - 1.0).transpose(2, 0, 1))
    mean = rgb.mean(axis=(0, 1), keepdims=True)
    std = rgb.std(axis=(0, 1), keepdims=True)
    out = np.clip((rgb - mean) / (std + 4.0), -4.0, 4.0)
    return np.ascontiguousarray(out.transpose(2, 0, 1))


# Fixed patch positions for test time averaging: centre, right cheek, left cheek, forehead, chin.
TTA_OFFSETS = ((0.0, 0.0), (-0.22, 0.08), (0.22, 0.08), (0.0, -0.28), (0.0, 0.3))


def build_inputs(crop: np.ndarray, context_scale: float, context_size: int, patch_size: int, patches: int = 1,
                 norm: str = "standardise") -> tuple[np.ndarray, np.ndarray]:
    """Return ``(context, texture)`` batches for one face crop.

    The context view is repeated so that both arrays have ``patches`` rows.
    """
    crop = normalize_scale(crop, context_scale)
    ctx = to_tensor(context_view(crop, context_size), norm)
    offsets = TTA_OFFSETS[: max(1, min(patches, len(TTA_OFFSETS)))]
    tex = np.stack([to_tensor(patch_view(crop, patch_size, off, context_scale), norm) for off in offsets])
    return np.repeat(ctx[None], len(offsets), axis=0), tex
