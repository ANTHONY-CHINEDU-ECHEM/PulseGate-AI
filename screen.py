"""Replay attack simulation: a face shown on a phone, tablet or monitor.

The display is rendered at sub pixel resolution, viewed through a tilted plane
and integrated by the sensor aperture. Moire, colour fringing and the loss of
contrast are consequences of that geometry instead of painted on effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .common import (
    FaceRef, add_glare, plane_homography, rounded_rect_alpha, sample_glare_blobs, srgb_to_linear,
    warp_with_aperture,
)

MAX_CANVAS = 2700      # longest side of the high resolution display canvas

DEVICES = {
    # pitch: frame pixels covered by one display pixel, bezel as a fraction of the screen width
    "phone": {"pitch": (0.18, 0.50), "aspect": (0.46, 0.60), "bezel": (0.015, 0.07), "radius": (0.04, 0.12)},
    "tablet": {"pitch": (0.30, 0.75), "aspect": (0.68, 0.80), "bezel": (0.03, 0.09), "radius": (0.02, 0.06)},
    "monitor": {"pitch": (0.45, 1.35), "aspect": (1.30, 1.80), "bezel": (0.01, 0.05), "radius": (0.0, 0.01)},
}


@dataclass
class ReplayParams:
    device: str
    pitch: float
    aspect: float                  # screen width divided by height
    content_width: float           # source pixels shown across the screen, in face widths
    content_shift: tuple[float, float]
    bezel: float
    bezel_color: tuple[float, float, float]
    corner_radius: float
    subpixel: str                  # rgb, bgr or none
    brightness: float
    black_level: float
    gamma_shift: float
    tint: tuple[float, float, float]
    veil: float
    glare: list = field(default_factory=list)
    banding: tuple[float, float, float, float] | None = None     # period px, amplitude, phase, angle
    fringe: tuple[float, float, float, float] | None = None      # period px, amplitude, angle, phase
    face_width: float = 180.0      # width of the replayed face in the camera frame, pixels
    offset: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0
    pitch_angle: float = 0.0
    roll: float = 0.0
    fill: float = 0.85
    letterbox: tuple[float, float, float] = (0.0, 0.0, 0.0)


def sample_replay(rng: np.random.Generator, device: str | None = None) -> ReplayParams:
    if device is None:
        device = str(rng.choice(["phone", "tablet", "monitor"], p=[0.5, 0.2, 0.3]))
    spec = DEVICES[device]
    tint_strength = rng.uniform(0.0, 0.14)
    warm = rng.random() < 0.25
    tint = (1.0 + (0 if warm else tint_strength), 1.0, 1.0 - (0 if warm else 0.6 * tint_strength))
    if warm:
        tint = (1.0 - tint_strength, 1.0, 1.0 + 0.5 * tint_strength)
    bezel_tone = float(rng.choice([0.004, 0.01, 0.03, 0.35, 0.7]))
    return ReplayParams(
        device=device,
        pitch=float(np.exp(rng.uniform(np.log(spec["pitch"][0]), np.log(spec["pitch"][1])))),
        aspect=float(rng.uniform(*spec["aspect"])),
        content_width=float(rng.uniform(1.4, 3.2)),
        content_shift=(float(rng.normal(0, 0.12)), float(rng.normal(0.05, 0.15))),
        bezel=float(rng.uniform(*spec["bezel"])),
        bezel_color=(bezel_tone, bezel_tone, bezel_tone),
        corner_radius=float(rng.uniform(*spec["radius"])),
        subpixel=str(rng.choice(["rgb", "bgr"], p=[0.8, 0.2])),
        brightness=float(rng.uniform(0.75, 1.3)),
        # emissive panels: OLED blacks are close to zero, LCD backlights leak a little
        black_level=float(rng.uniform(0.001, 0.006)) if rng.random() < 0.5 else float(rng.uniform(0.006, 0.04)),
        gamma_shift=float(rng.uniform(0.85, 1.15)),
        tint=tint,
        veil=float(min(0.05, rng.exponential(0.008))),
        glare=sample_glare_blobs(rng, strength=float(np.clip(rng.exponential(0.07), 0.01, 0.4))) if rng.random() < 0.55 else [],
        banding=(float(rng.uniform(25, 220)), float(rng.uniform(0.02, 0.14)), float(rng.uniform(0, 2 * np.pi)), float(rng.normal(0, 0.05)))
        if rng.random() < 0.35 else None,
        fringe=(float(rng.uniform(5, 40)), float(rng.uniform(0.008, 0.05)), float(rng.uniform(0, np.pi)), float(rng.uniform(0, 2 * np.pi)))
        if rng.random() < 0.3 else None,
        face_width=float(rng.uniform(105, 250)),
        offset=(float(rng.normal(0, 22)), float(rng.normal(0, 26))),
        yaw=float(rng.normal(0, 0.17)),
        pitch_angle=float(rng.normal(0, 0.14)),
        roll=float(rng.normal(0, 0.07)),
        fill=float(rng.uniform(0.6, 1.0)),
        letterbox=tuple(float(v) for v in ([0.0] * 3 if rng.random() < 0.7 else [rng.uniform(0.02, 0.9)] * 3)),
    )


def _screen_content(source_linear: np.ndarray, face: FaceRef, p: ReplayParams, screen_wh: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Cut the part of the source that is on screen. Returns content and face centre on screen."""
    ws, hs = screen_wh
    width_src = p.content_width * face.size
    cx = face.center[0] + p.content_shift[0] * face.size
    cy = face.center[1] + p.content_shift[1] * face.size
    scale = ws / width_src
    matrix = np.array([[scale, 0, ws / 2 - scale * cx], [0, scale, hs / 2 - scale * cy]], dtype=np.float64)
    if scale < 0.7:
        # area averaging before the warp avoids aliasing the source itself
        k = int(round(1.0 / scale))
        source_linear = cv2.blur(source_linear, (k, k)) if k >= 2 else source_linear
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_LINEAR
    content = cv2.warpAffine(source_linear, matrix, (ws, hs), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=p.letterbox)
    face_on_screen = matrix[:, :2] @ face.center + matrix[:, 2]
    return np.clip(content, 0.0, 1.0), face_on_screen


def render_replay(
    source_bgr: np.ndarray, face: FaceRef, background_linear: np.ndarray, p: ReplayParams,
    rng: np.random.Generator, jitter: tuple[float, float, float, float, float] = (0, 0, 0, 0, 0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render the replay scene in linear light.

    Returns the frame, the coverage mask of the device and the expected face
    centre in the frame. ``jitter`` adds per frame hand movement
    (dx, dy, yaw, pitch, roll) on top of the session parameters.
    """
    fh, fw = background_linear.shape[:2]
    source = srgb_to_linear(source_bgr)

    # Screen geometry: how many display pixels are needed for the wanted face width.
    content_frame_width = p.face_width * p.content_width
    ws = max(48, int(round(content_frame_width / p.pitch)))
    hs = max(48, int(round(ws / p.aspect)))
    k = 3 if (p.subpixel != "none" and 3 * max(ws, hs) <= MAX_CANVAS) else 1
    if k == 1 and max(ws, hs) > MAX_CANVAS:
        shrink = MAX_CANVAS / max(ws, hs)
        ws, hs = int(ws * shrink), int(hs * shrink)
    pitch = content_frame_width / ws

    content, face_on_screen = _screen_content(source, face, p, (ws, hs))

    # Emission model of the panel.
    emitted = np.power(content, p.gamma_shift)
    emitted = p.black_level + (1.0 - p.black_level) * emitted
    emitted = emitted * (np.asarray(p.tint, dtype=np.float32) * p.brightness)

    if k == 3:
        tile = np.zeros((3, 3, 3), dtype=np.float32)
        order = (2, 1, 0) if p.subpixel == "rgb" else (0, 1, 2)       # BGR channel index per stripe
        for column, channel in enumerate(order):
            tile[:, column, channel] = 3.0
        tile[2, :, :] *= 0.55                                           # dark gap between pixel rows
        tile /= tile.mean(axis=(0, 1), keepdims=True) + 1e-6
        hi = np.repeat(np.repeat(emitted, 3, axis=0), 3, axis=1)
        hi *= np.tile(tile, (hs, ws, 1))
    else:
        hi = emitted

    # Device body around the screen, with the cover glass reflecting the room.
    bezel_px = int(round(p.bezel * ws * k))
    canvas = cv2.copyMakeBorder(hi, bezel_px, bezel_px, bezel_px, bezel_px, cv2.BORDER_CONSTANT, value=p.bezel_color)
    ch, cw = canvas.shape[:2]
    if p.glare or p.veil > 0:
        reflect = np.full((ch, cw), p.veil, dtype=np.float32)
        if p.glare:
            reflect += add_glare((ch, cw), rng, p.glare)
        canvas = canvas + reflect[:, :, None]
    alpha = rounded_rect_alpha((ch, cw), int(p.corner_radius * cw))

    anchor = (face_on_screen[0] * k + bezel_px, face_on_screen[1] * k + bezel_px)
    target = (fw / 2 + p.offset[0] + jitter[0], fh / 2 + p.offset[1] + jitter[1])
    homography = plane_homography(
        anchor, pitch / k, p.yaw + jitter[2], p.pitch_angle + jitter[3], p.roll + jitter[4], target, focal=1.25 * max(fw, fh),
    )
    warped, coverage = warp_with_aperture(canvas, alpha, homography, (fw, fh), samples_per_pixel=k / pitch, fill=p.fill)

    if p.banding is not None:
        period, amplitude, phase, angle = p.banding
        ys, xs = np.mgrid[0:fh, 0:fw].astype(np.float32)
        wave = np.sin(2 * np.pi * (ys * np.cos(angle) + xs * np.sin(angle)) / period + phase)
        warped = warped * (1.0 + amplitude * np.tanh(2.5 * wave))[:, :, None]
    if p.fringe is not None:
        period, amplitude, angle, phase = p.fringe
        ys, xs = np.mgrid[0:fh, 0:fw].astype(np.float32)
        t = (xs * np.cos(angle) + ys * np.sin(angle)) / period
        bend = 0.6 * np.sin(2 * np.pi * (xs * np.sin(angle) - ys * np.cos(angle)) / (9.0 * period))
        for channel, shift in enumerate((0.0, 2.1, 4.2)):
            warped[:, :, channel] *= 1.0 + amplitude * np.cos(2 * np.pi * (t + bend) + phase + shift)

    frame = background_linear * (1.0 - coverage[:, :, None]) + warped
    return frame.astype(np.float32), coverage, np.array(target, dtype=np.float64)
