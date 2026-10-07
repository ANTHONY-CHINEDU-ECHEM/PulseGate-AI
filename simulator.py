"""Scene level orchestration of the presentation attack simulator.

``Simulator`` produces camera frames for genuine presentations and for five
attack instrument species. Parameters are sampled once per sample (or once per
video session) and can be reused across frames, which is what makes temporally
consistent attack videos possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .common import (
    CameraParams, FaceRef, apply_camera, gradient_field, sample_camera, similarity_matrix, srgb_to_linear,
)
from .paper import CutoutParams, PrintParams, render_cutout, render_print, sample_cutout, sample_print
from .screen import ReplayParams, render_replay, sample_replay

SPECIES = ("print_matte", "print_glossy", "replay_phone", "replay_monitor", "cutout_mask")
LIVE = "live"
ALL_CLASSES = (LIVE,) + SPECIES
FAMILY = {
    "live": "live", "print_matte": "print", "print_glossy": "print",
    "replay_phone": "replay", "replay_monitor": "replay", "cutout_mask": "mask",
}


@dataclass
class LiveParams:
    face_width: float
    offset: tuple[float, float]
    roll: float
    gain: float
    gamma: float
    light: tuple[float, float]       # direction and strength of an illumination gradient
    flip: bool = False


@dataclass
class SceneParams:
    """Everything needed to re render one presentation."""

    species: str
    camera: CameraParams
    live: LiveParams | None = None
    medium: Any = None
    background: LiveParams | None = None
    background_blur: float = 0.0
    meta: dict = field(default_factory=dict)


def sample_live(rng: np.random.Generator, face_width: tuple[float, float] = (105, 250)) -> LiveParams:
    return LiveParams(
        face_width=float(rng.uniform(*face_width)),
        offset=(float(rng.normal(0, 22)), float(rng.normal(0, 26))),
        roll=float(rng.normal(0, 0.07)),
        gain=float(np.exp(rng.normal(0, 0.15))),
        gamma=float(rng.uniform(0.85, 1.2)),
        light=(float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0, 0.4)) if rng.random() < 0.4 else 0.0),
        flip=bool(rng.random() < 0.5),
    )


class Simulator:
    """Renders genuine and attack presentations into camera frames."""

    def __init__(self, frame_size: tuple[int, int] = (480, 640)):
        self.frame_w, self.frame_h = int(frame_size[0]), int(frame_size[1])

    # ------------------------------------------------------------------ sampling
    def sample(self, species: str, rng: np.random.Generator) -> SceneParams:
        camera = sample_camera(rng)
        if species == LIVE:
            return SceneParams(species=species, camera=camera, live=sample_live(rng))
        background = sample_live(rng, face_width=(140, 300))
        blur = float(abs(rng.normal(0, 1.2)))
        if species == "print_matte":
            medium: Any = sample_print(rng, "matte")
        elif species == "print_glossy":
            medium = sample_print(rng, "glossy")
        elif species == "replay_phone":
            medium = sample_replay(rng, str(rng.choice(["phone", "tablet"], p=[0.75, 0.25])))
        elif species == "replay_monitor":
            medium = sample_replay(rng, "monitor")
        elif species == "cutout_mask":
            medium = sample_cutout(rng)
            background = sample_live(rng, face_width=(112, 225))
            blur = 0.0
        else:
            raise ValueError(f"unknown species '{species}'")
        return SceneParams(species=species, camera=camera, medium=medium, background=background, background_blur=blur)

    # ------------------------------------------------------------------ genuine
    def _place(self, image_bgr: np.ndarray, face: FaceRef, p: LiveParams, jitter=(0, 0, 0)) -> tuple[np.ndarray, FaceRef]:
        """Warp a source photo into the camera frame. Returns linear light and face geometry."""
        if p.flip:
            image_bgr = image_bgr[:, ::-1]
            w = image_bgr.shape[1]
            pts = face.points.copy()
            pts[:, 0] = w - 1 - pts[:, 0]
            pts = pts[[1, 0, 2, 4, 3]]
            box = face.box.copy()
            box[0] = w - 1 - (face.box[0] + face.box[2])
            face = FaceRef(box=box, points=pts)
        scale = p.face_width / face.size
        target = np.array([self.frame_w / 2 + p.offset[0] + jitter[0], self.frame_h / 2 + p.offset[1] + jitter[1]])
        matrix = similarity_matrix(scale, p.roll + jitter[2], face.center, target)
        src = image_bgr
        if scale < 0.7:
            k = int(round(1.0 / scale))
            src = cv2.blur(src, (k, k)) if k >= 2 else src
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_LINEAR
        warped = cv2.warpAffine(np.ascontiguousarray(src), matrix, (self.frame_w, self.frame_h), flags=interp, borderMode=cv2.BORDER_REPLICATE)
        linear = srgb_to_linear(warped)
        linear = np.power(linear, p.gamma) * p.gain
        if p.light[1] > 0.01:
            linear = linear * (1.0 - p.light[1] * gradient_field((self.frame_h, self.frame_w), p.light[0]))[:, :, None]
        return linear.astype(np.float32), face.transformed(matrix)

    def render_live(self, image_bgr: np.ndarray, face: FaceRef, params: SceneParams, rng: np.random.Generator,
                    jitter=(0, 0, 0)) -> tuple[np.ndarray, np.ndarray]:
        linear, placed = self._place(image_bgr, face, params.live, jitter)
        return apply_camera(linear, params.camera, rng), placed.center

    # ------------------------------------------------------------------ attacks
    def render_attack(
        self, image_bgr: np.ndarray, face: FaceRef, background_bgr: np.ndarray, background_face: FaceRef,
        params: SceneParams, rng: np.random.Generator, jitter: tuple[float, float, float, float, float] = (0, 0, 0, 0, 0),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render one attack frame. Returns the 8 bit frame and the expected face centre."""
        background, placed = self._place(background_bgr, background_face, params.background)
        if params.background_blur > 0.3:
            background = cv2.GaussianBlur(background, (0, 0), params.background_blur)
        medium = params.medium
        if isinstance(medium, ReplayParams):
            frame, _, centre = render_replay(image_bgr, face, background, medium, rng, jitter)
        elif isinstance(medium, PrintParams):
            frame, _, centre = render_print(image_bgr, face, background, medium, rng, jitter)
        elif isinstance(medium, CutoutParams):
            frame, _, centre = render_cutout(image_bgr, face, background, placed, medium, rng, jitter)
        else:
            raise TypeError(f"unsupported medium {type(medium).__name__}")
        return apply_camera(frame, params.camera, rng), centre

    def render(self, image_bgr: np.ndarray, face: FaceRef, params: SceneParams, rng: np.random.Generator,
               background_bgr: np.ndarray | None = None, background_face: FaceRef | None = None,
               jitter: tuple = (0, 0, 0, 0, 0)) -> tuple[np.ndarray, np.ndarray]:
        if params.species == LIVE:
            return self.render_live(image_bgr, face, params, rng, jitter[:3])
        if background_bgr is None or background_face is None:
            raise ValueError("attack rendering needs a background image and its face geometry")
        return self.render_attack(image_bgr, face, background_bgr, background_face, params, rng, jitter)


def describe(params: SceneParams) -> dict:
    """Compact, JSON friendly summary of the most informative scene parameters."""
    out: dict[str, Any] = {"species": params.species, "jpeg": params.camera.jpeg_quality, "blur": round(params.camera.blur_sigma, 2)}
    m = params.medium
    if isinstance(m, ReplayParams):
        out.update(device=m.device, pitch=round(m.pitch, 3), banding=m.banding is not None, glare=len(m.glare))
    elif isinstance(m, PrintParams):
        out.update(process=m.process, cell=round(m.cell, 2), border=round(m.border, 3))
    elif isinstance(m, CutoutParams):
        out.update(process=m.sheet.process, eye_holes=m.eye_holes > 0, mask_scale=round(m.mask_scale, 2))
    return out
