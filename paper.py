"""Print attack simulation: a face on paper, flat or cut out as a mask.

The sheet is rendered as a reflectance map at several samples per camera pixel.
Halftone screening, ink impurity, paper fibre and gloss are modelled on the
sheet, then the sheet is viewed through the same aperture model as a display.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .common import (
    FaceRef, add_glare, apply_transform, gradient_field, plane_homography, sample_glare_blobs,
    smooth_noise, srgb_to_linear, warp_with_aperture,
)

MAX_CANVAS = 2400
INK_ANGLES = {"c": 15.0, "m": 75.0, "y": 0.0, "k": 45.0}


@dataclass
class PrintParams:
    finish: str                    # matte or glossy
    process: str                   # halftone, stochastic or continuous
    cell: float                    # camera pixels per halftone cell
    content_width: float           # source pixels across the sheet, in face widths
    content_shift: tuple[float, float]
    aspect: float                  # sheet width divided by height
    border: float                  # white margin as a fraction of the sheet width
    paper_white: tuple[float, float, float]
    black_level: float
    saturation: float
    tone_gamma: float
    ink_density: float
    ink_crosstalk: float
    texture: float
    lines: tuple[float, float] | None
    illumination: tuple[float, float]       # direction, strength
    gloss: list = field(default_factory=list)
    shadow: float = 0.2
    face_width: float = 180.0
    offset: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0
    pitch_angle: float = 0.0
    roll: float = 0.0
    fill: float = 0.85


def sample_print(rng: np.random.Generator, finish: str | None = None) -> PrintParams:
    if finish is None:
        finish = str(rng.choice(["matte", "glossy"]))
    if finish == "matte":
        process = str(rng.choice(["halftone", "stochastic"], p=[0.55, 0.45]))
        white = rng.uniform(0.72, 0.92)
        black = rng.uniform(0.02, 0.08)
        saturation = rng.uniform(0.65, 1.0)
        gloss = sample_glare_blobs(rng, strength=float(rng.uniform(0.01, 0.06))) if rng.random() < 0.2 else []
    else:
        process = str(rng.choice(["continuous", "stochastic"], p=[0.7, 0.3]))
        white = rng.uniform(0.82, 0.96)
        black = rng.uniform(0.008, 0.03)
        saturation = rng.uniform(0.85, 1.1)
        gloss = sample_glare_blobs(rng, strength=float(np.clip(rng.exponential(0.1), 0.02, 0.5))) if rng.random() < 0.8 else []
    warmth = rng.normal(0.0, 0.025)
    return PrintParams(
        finish=finish,
        process=process,
        cell=float(rng.uniform(0.6, 1.8)),
        content_width=float(rng.uniform(1.4, 2.9)),
        content_shift=(float(rng.normal(0, 0.1)), float(rng.normal(0.05, 0.14))),
        aspect=float(rng.uniform(0.66, 0.82)) if rng.random() < 0.75 else float(rng.uniform(1.2, 1.5)),
        border=float(rng.uniform(0.02, 0.09)) if rng.random() < 0.5 else 0.0,
        paper_white=(float(white * (1 - warmth - 0.02)), float(white), float(white * (1 + warmth))),
        black_level=float(black),
        saturation=float(saturation),
        tone_gamma=float(rng.uniform(0.85, 1.25)),
        ink_density=float(rng.uniform(0.78, 0.95)),
        ink_crosstalk=float(rng.uniform(0.04, 0.16)),
        texture=float(rng.uniform(0.005, 0.03)),
        lines=(float(rng.uniform(6, 30)), float(rng.uniform(0.01, 0.05))) if rng.random() < 0.25 else None,
        illumination=(float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0.0, 0.35))),
        gloss=gloss,
        shadow=float(rng.uniform(0.0, 0.4)),
        face_width=float(rng.uniform(105, 250)),
        offset=(float(rng.normal(0, 22)), float(rng.normal(0, 26))),
        yaw=float(rng.normal(0, 0.17)),
        pitch_angle=float(rng.normal(0, 0.14)),
        roll=float(rng.normal(0, 0.07)),
        fill=float(rng.uniform(0.6, 1.0)),
    )


def _screen_distribution() -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    values = np.sort((0.5 + 0.25 * (np.cos(grid)[:, None] + np.cos(grid)[None, :])).ravel())
    return values[::16].astype(np.float32), np.linspace(0.0, 1.0, len(values))[::16].astype(np.float32)


_SCREEN_VALUES, _SCREEN_RANKS = _screen_distribution()


def _halftone(tone: np.ndarray, p: PrintParams, period: float, rng: np.random.Generator) -> np.ndarray:
    """Amplitude modulated CMYK screening. ``tone`` is linear BGR in 0..1."""
    h, w = tone.shape[:2]
    c, m, y = 1.0 - tone[:, :, 2], 1.0 - tone[:, :, 1], 1.0 - tone[:, :, 0]
    k = np.minimum(np.minimum(c, m), y) * float(rng.uniform(0.3, 0.8))
    # under colour removal: the coloured inks only cover what black leaves open
    rest = 1.0 - k + 1e-4
    inks = {"c": (c - k) / rest, "m": (m - k) / rest, "y": (y - k) / rest, "k": k}
    d, x = p.ink_density, p.ink_crosstalk
    # absorption of each ink in the B, G, R channels (real inks are not ideal block dyes)
    absorb = {
        "c": np.array([x, 2.2 * x, d], dtype=np.float32),
        "m": np.array([2.0 * x, d, x], dtype=np.float32),
        "y": np.array([d, x, 0.5 * x], dtype=np.float32),
        "k": np.array([d, d, d], dtype=np.float32),
    }
    out = np.ones_like(tone)
    jitter = float(rng.uniform(-8, 8))
    col = np.arange(w, dtype=np.float32) * (2 * np.pi / period)
    row = np.arange(h, dtype=np.float32) * (2 * np.pi / period)

    def plane_wave(a: float, b: float) -> np.ndarray:
        # cos(a x + b y) from separable terms, far cheaper than a full grid of angles
        return np.outer(np.cos(b * row), np.cos(a * col)) - np.outer(np.sin(b * row), np.sin(a * col))

    for name, coverage in inks.items():
        angle = np.radians(INK_ANGLES[name] + jitter)
        ca, sa = float(np.cos(angle)), float(np.sin(angle))
        screen = 0.5 + 0.25 * (plane_wave(ca, sa) + plane_wave(-sa, ca))
        # Equalise the threshold function so that the inked area equals the requested coverage.
        screen = np.interp(screen, _SCREEN_VALUES, _SCREEN_RANKS).astype(np.float32)
        dots = np.clip((np.clip(coverage, 0, 1) - screen) * 8.0, 0.0, 1.0)
        out *= 1.0 - dots[:, :, None] * absorb[name]
    return out


def render_sheet(source_linear: np.ndarray, face: FaceRef, p: PrintParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """Render the printed sheet.

    Returns the reflectance canvas, the affine map from source pixels to canvas
    pixels and the number of canvas samples per camera pixel.
    """
    content_frame_width = p.face_width * p.content_width
    process = p.process
    samples = 4.0 / p.cell if process == "halftone" else 2.0
    if samples * content_frame_width > MAX_CANVAS or samples * content_frame_width / p.aspect > MAX_CANVAS:
        process = "stochastic" if process == "halftone" else process
        samples = min(2.0, MAX_CANVAS / max(content_frame_width, content_frame_width / p.aspect))
    wp = max(48, int(round(content_frame_width * samples)))
    hp = max(48, int(round(wp / p.aspect)))

    scale = wp / (p.content_width * face.size)
    cx = face.center[0] + p.content_shift[0] * face.size
    cy = face.center[1] + p.content_shift[1] * face.size
    matrix = np.array([[scale, 0, wp / 2 - scale * cx], [0, scale, hp / 2 - scale * cy]], dtype=np.float64)
    src = source_linear
    if scale < 0.7:
        kk = int(round(1.0 / scale))
        src = cv2.blur(src, (kk, kk)) if kk >= 2 else src
    content = cv2.warpAffine(src, matrix, (wp, hp), flags=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    content = np.clip(content, 0.0, 1.0)

    luma = content @ np.array([0.114, 0.587, 0.299], dtype=np.float32)
    tone = luma[:, :, None] + p.saturation * (content - luma[:, :, None])
    tone = np.power(np.clip(tone, 0.0, 1.0), p.tone_gamma)

    if process == "halftone":
        reflect = _halftone(tone, p, period=4.0, rng=rng)
        reflect = cv2.GaussianBlur(reflect, (0, 0), 0.6)          # dot gain
    elif process == "stochastic":
        grain = rng.standard_normal(tone.shape[:2]).astype(np.float32)
        strength = float(rng.uniform(0.04, 0.13))
        reflect = tone * (1.0 + strength * grain[:, :, None] * (1.0 - 0.6 * tone))
    else:
        reflect = tone
    reflect = p.black_level + (1.0 - p.black_level) * np.clip(reflect, 0.0, 1.2)
    reflect = reflect * np.asarray(p.paper_white, dtype=np.float32)

    if p.texture > 0:
        fibre = smooth_noise((hp, wp), float(rng.uniform(1.5, 5.0)) * max(1.0, samples / 2), rng)
        reflect *= 1.0 + p.texture * fibre[:, :, None]
    if p.lines is not None:
        period, amplitude = p.lines
        rows = np.sin(2 * np.pi * np.arange(hp, dtype=np.float32) / (period * samples / 2))
        reflect *= 1.0 + amplitude * np.sign(rows)[:, None, None] * (np.abs(rows)[:, None, None] > 0.92)

    border = int(round(p.border * wp))
    if border > 0:
        reflect = cv2.copyMakeBorder(reflect, border, border, border, border, cv2.BORDER_CONSTANT, value=p.paper_white)
        matrix = matrix.copy()
        matrix[:, 2] += border

    ch, cw = reflect.shape[:2]
    direction, strength = p.illumination
    if strength > 0.01:
        reflect *= (1.0 - strength * gradient_field((ch, cw), direction))[:, :, None]
    if p.gloss:
        reflect += add_glare((ch, cw), rng, p.gloss)[:, :, None]
    return reflect.astype(np.float32), matrix, samples


def cast_shadow(background: np.ndarray, coverage: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Soft shadow of the medium on whatever is behind it."""
    if strength <= 0.01:
        return background
    shift = np.array([[1, 0, rng.uniform(-10, 10)], [0, 1, rng.uniform(2, 14)]], dtype=np.float32)
    h, w = coverage.shape
    shadow = cv2.warpAffine(coverage, shift, (w, h))
    shadow = cv2.GaussianBlur(shadow, (0, 0), float(rng.uniform(3, 10)))
    return background * (1.0 - strength * shadow * (1.0 - coverage))[:, :, None]


def render_print(
    source_bgr: np.ndarray, face: FaceRef, background_linear: np.ndarray, p: PrintParams,
    rng: np.random.Generator, jitter: tuple[float, float, float, float, float] = (0, 0, 0, 0, 0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a hand held print in front of the background. Linear light output."""
    fh, fw = background_linear.shape[:2]
    sheet, to_canvas, samples = render_sheet(srgb_to_linear(source_bgr), face, p, rng)
    anchor = apply_transform(to_canvas, face.center[None])[0]
    target = (fw / 2 + p.offset[0] + jitter[0], fh / 2 + p.offset[1] + jitter[1])
    homography = plane_homography(
        (anchor[0], anchor[1]), 1.0 / samples, p.yaw + jitter[2], p.pitch_angle + jitter[3], p.roll + jitter[4],
        target, focal=1.25 * max(fw, fh),
    )
    warped, coverage = warp_with_aperture(sheet, None, homography, (fw, fh), samples_per_pixel=samples, fill=p.fill)
    background = cast_shadow(background_linear, coverage, p.shadow, rng)
    frame = background * (1.0 - coverage[:, :, None]) + warped
    return frame.astype(np.float32), coverage, np.array(target, dtype=np.float64)


@dataclass
class CutoutParams:
    sheet: PrintParams
    mask_scale: float              # size of the paper face relative to the wearer's face
    oval: tuple[float, float]      # half axes of the cut line in eye distances
    wobble: float                  # irregularity of the scissor line
    eye_holes: float               # hole radius in eye distances, 0 for none
    mouth_hole: bool
    rim: float                     # visible white paper edge, canvas pixels
    curve: float                   # shading from bending the mask around the face


def sample_cutout(rng: np.random.Generator) -> CutoutParams:
    sheet = sample_print(rng)
    sheet.border = 0.0
    sheet.content_width = float(rng.uniform(2.0, 2.5))
    sheet.content_shift = (0.0, 0.0)
    sheet.aspect = 0.8
    return CutoutParams(
        sheet=sheet,
        mask_scale=float(rng.uniform(0.98, 1.18)),
        oval=(float(rng.uniform(1.05, 1.35)), float(rng.uniform(1.5, 1.95))),
        wobble=float(rng.uniform(0.0, 0.05)),
        eye_holes=float(rng.uniform(0.2, 0.3)) if rng.random() < 0.55 else 0.0,
        mouth_hole=bool(rng.random() < 0.15),
        rim=float(rng.uniform(0.0, 2.5)),
        curve=float(rng.uniform(0.0, 0.3)),
    )


def render_cutout(
    source_bgr: np.ndarray, face: FaceRef, wearer_linear: np.ndarray, wearer_face: FaceRef, p: CutoutParams,
    rng: np.random.Generator, jitter: tuple[float, float, float, float, float] = (0, 0, 0, 0, 0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a printed face mask worn in front of another person's face."""
    fh, fw = wearer_linear.shape[:2]
    sheet_params = p.sheet
    sheet_params.face_width = wearer_face.size * p.mask_scale
    sheet, to_canvas, samples = render_sheet(srgb_to_linear(source_bgr), face, sheet_params, rng)
    ch, cw = sheet.shape[:2]
    pts = apply_transform(to_canvas, face.points)
    eye_mid = 0.5 * (pts[0] + pts[1])
    eye_dist = float(np.linalg.norm(pts[1] - pts[0]))
    axis = (pts[1] - pts[0]) / (eye_dist + 1e-6)
    down = np.array([-axis[1], axis[0]])
    centre = eye_mid + 0.42 * eye_dist * down

    # Cut line: an oval with a slightly irregular radius.
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    wobble = 1.0 + p.wobble * np.sin(3 * angles + rng.uniform(0, 6.28)) + 0.5 * p.wobble * np.sin(7 * angles + rng.uniform(0, 6.28))
    outline = centre + (np.cos(angles) * p.oval[0] * eye_dist * wobble)[:, None] * axis + (np.sin(angles) * p.oval[1] * eye_dist * wobble)[:, None] * down
    alpha = np.zeros((ch, cw), dtype=np.float32)
    cv2.fillPoly(alpha, [np.round(outline).astype(np.int32)], 1.0, cv2.LINE_AA)
    if p.rim > 0.3:
        k = max(1, int(round(p.rim * samples)))
        inner = cv2.erode(alpha, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        rim = (alpha - inner)[:, :, None]
        sheet = sheet * (1 - rim) + np.asarray(sheet_params.paper_white, dtype=np.float32) * rim
    if p.eye_holes > 0:
        for eye in (pts[0], pts[1]):
            cv2.ellipse(alpha, (int(eye[0]), int(eye[1])), (int(p.eye_holes * eye_dist * 1.35), int(p.eye_holes * eye_dist)), 0, 0, 360, 0.0, -1, cv2.LINE_AA)
    if p.mouth_hole:
        mouth = 0.5 * (pts[3] + pts[4])
        cv2.ellipse(alpha, (int(mouth[0]), int(mouth[1])), (int(0.42 * eye_dist), int(0.18 * eye_dist)), 0, 0, 360, 0.0, -1, cv2.LINE_AA)
    if p.curve > 0:
        xs = (np.arange(cw, dtype=np.float32) - centre[0]) / (p.oval[0] * eye_dist + 1e-6)
        sheet = sheet * (1.0 - p.curve * np.clip(xs, -1.2, 1.2) ** 2)[None, :, None]

    # Align the paper face with the wearer's face.
    target_pts = wearer_face.points.astype(np.float64) + np.array([jitter[0], jitter[1]])
    fit, _ = cv2.estimateAffinePartial2D(pts.astype(np.float32), target_pts.astype(np.float32), method=cv2.LMEDS)
    if fit is None:
        fit = np.array([[1.0 / samples, 0, 0], [0, 1.0 / samples, 0]])
    wearer_centre = target_pts.mean(axis=0)
    grow = np.array([[p.mask_scale, 0, (1 - p.mask_scale) * wearer_centre[0]], [0, p.mask_scale, (1 - p.mask_scale) * wearer_centre[1]], [0, 0, 1]])
    homography = grow @ np.vstack([fit, [0, 0, 1]])
    frame_per_canvas = float(np.sqrt(abs(np.linalg.det(homography[:2, :2]))))
    warped, coverage = warp_with_aperture(
        sheet, alpha, homography, (fw, fh), samples_per_pixel=1.0 / max(frame_per_canvas, 1e-3), fill=sheet_params.fill,
    )
    background = cast_shadow(wearer_linear, coverage, min(0.5, sheet_params.shadow + 0.1), rng)
    frame = background * (1.0 - coverage[:, :, None]) + warped
    return frame.astype(np.float32), coverage, wearer_face.center.copy()
