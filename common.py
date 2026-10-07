"""Shared building blocks of the presentation attack simulator.

All images inside the simulator are float32 BGR arrays in linear light with a
nominal range of 0 to 1. Linear light matters because glare, illumination and
sensor noise add up physically only before the display transfer curve.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceRef:
    """Face geometry of a source image: detector style box and five points."""

    box: np.ndarray        # x, y, w, h
    points: np.ndarray     # 5 x 2: right eye, left eye, nose, right mouth corner, left mouth corner

    @property
    def center(self) -> np.ndarray:
        return np.array([self.box[0] + self.box[2] / 2.0, self.box[1] + self.box[3] / 2.0], dtype=np.float64)

    @property
    def size(self) -> float:
        return float(max(self.box[2], self.box[3]))

    @property
    def eye_distance(self) -> float:
        return float(np.linalg.norm(self.points[1] - self.points[0]))

    def transformed(self, matrix: np.ndarray) -> "FaceRef":
        """Apply a 2x3 affine or 3x3 projective transform."""
        pts = apply_transform(matrix, self.points)
        corners = np.array([
            [self.box[0], self.box[1]], [self.box[0] + self.box[2], self.box[1]],
            [self.box[0] + self.box[2], self.box[1] + self.box[3]], [self.box[0], self.box[1] + self.box[3]],
        ])
        c = apply_transform(matrix, corners)
        centre = c.mean(axis=0)
        w = 0.5 * (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3]))
        h = 0.5 * (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1]))
        box = np.array([centre[0] - w / 2, centre[1] - h / 2, w, h], dtype=np.float32)
        return FaceRef(box=box, points=pts.astype(np.float32))


def face_ref_from_detection(detection) -> FaceRef:
    """Face geometry from a ``Detection`` of the YuNet detector."""
    return FaceRef(box=np.asarray(detection.box, dtype=np.float32), points=np.asarray(detection.points, dtype=np.float32))


def face_ref_from_landmarks(landmarks: np.ndarray, box: np.ndarray) -> FaceRef:
    """Face geometry from a 468 point mesh, using the same five point layout."""
    pts = np.array([
        landmarks[[33, 133], :2].mean(axis=0), landmarks[[362, 263], :2].mean(axis=0),
        landmarks[1, :2], landmarks[61, :2], landmarks[291, :2],
    ], dtype=np.float32)
    return FaceRef(box=np.asarray(box, dtype=np.float32), points=pts)


def apply_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape == (2, 3):
        return pts @ m[:, :2].T + m[:, 2]
    homog = np.column_stack([pts, np.ones(len(pts))]) @ m.T
    return homog[:, :2] / homog[:, 2:3]


def srgb_to_linear(img_u8: np.ndarray) -> np.ndarray:
    return np.power(img_u8.astype(np.float32) / 255.0, 2.2)


def linear_to_srgb(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    return np.power(np.clip(img, 0.0, 1.0), 1.0 / gamma)


def to_uint8(img_srgb: np.ndarray) -> np.ndarray:
    return np.clip(img_srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def smooth_noise(shape_hw: tuple[int, int], sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Zero mean, unit variance noise with a chosen correlation length."""
    h, w = shape_hw
    step = max(1, int(sigma / 2))
    small = rng.standard_normal((h // step + 2, w // step + 2)).astype(np.float32)
    if sigma / step > 0.3:
        small = cv2.GaussianBlur(small, (0, 0), sigma / step)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    out -= out.mean()
    return out / (out.std() + 1e-6)


def gradient_field(shape_hw: tuple[int, int], angle: float) -> np.ndarray:
    """Linear ramp from 0 to 1 across the image in a given direction."""
    h, w = shape_hw
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = (xs - w / 2) * np.cos(angle) + (ys - h / 2) * np.sin(angle)
    ramp -= ramp.min()
    return ramp / (ramp.max() + 1e-6)


def plane_homography(
    anchor_xy: tuple[float, float], scale: float, yaw: float, pitch: float, roll: float,
    target_xy: tuple[float, float], focal: float,
) -> np.ndarray:
    """Homography that places a flat medium in front of a pinhole camera.

    ``anchor_xy`` on the medium lands on ``target_xy`` in the frame. ``scale`` is
    the number of frame pixels per medium pixel at the anchor. Angles in radians
    tilt the plane around the vertical, horizontal and optical axes.
    """
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    r_yaw = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
    r_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    r_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    rot = r_roll @ r_pitch @ r_yaw
    # medium pixel (u, v, 1) -> plane point -> rotated -> camera at distance ``focal``
    to_plane = np.array([[scale, 0, -scale * anchor_xy[0]], [0, scale, -scale * anchor_xy[1]], [0, 0, 0]], dtype=np.float64)
    cam = rot @ to_plane
    cam[2, 2] += focal
    k = np.array([[focal, 0, target_xy[0]], [0, focal, target_xy[1]], [0, 0, 1]], dtype=np.float64)
    h = k @ cam
    return h / h[2, 2]


def similarity_matrix(scale: float, angle: float, src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    """2x3 similarity that maps ``src_xy`` onto ``dst_xy`` with scale and rotation."""
    c, s = np.cos(angle) * scale, np.sin(angle) * scale
    m = np.array([[c, -s, 0.0], [s, c, 0.0]])
    m[:, 2] = np.asarray(dst_xy, dtype=np.float64) - m[:, :2] @ np.asarray(src_xy, dtype=np.float64)
    return m


def warp_with_aperture(
    canvas: np.ndarray, alpha: np.ndarray | None, homography: np.ndarray, frame_wh: tuple[int, int],
    samples_per_pixel: float, fill: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a high resolution medium the way a sensor does.

    Each sensor pixel integrates light over a small aperture and is then point
    sampled. The box filter models the aperture. Whatever periodic structure is
    left above the sampling limit folds back as moire, exactly as on a camera.
    Returns the premultiplied colour and the coverage of the medium in the frame.
    """
    kernel = int(round(samples_per_pixel * fill))
    if alpha is None:
        alpha = np.ones(canvas.shape[:2], dtype=np.float32)
    else:
        canvas = canvas * alpha[:, :, None]
    if kernel >= 2:
        canvas = cv2.blur(canvas, (kernel, kernel))
        alpha = cv2.blur(alpha, (kernel, kernel))
    warped = cv2.warpPerspective(canvas, homography, frame_wh, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    coverage = cv2.warpPerspective(alpha, homography, frame_wh, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, coverage


def rounded_rect_alpha(shape_hw: tuple[int, int], radius: int) -> np.ndarray:
    h, w = shape_hw
    alpha = np.ones((h, w), dtype=np.float32)
    r = int(max(0, min(radius, h // 2 - 1, w // 2 - 1)))
    if r < 2:
        return alpha
    corner = np.zeros((r, r), dtype=np.float32)
    cv2.circle(corner, (r, r), r, 1.0, -1, cv2.LINE_AA)
    alpha[:r, :r] = corner
    alpha[:r, w - r:] = corner[:, ::-1]
    alpha[h - r:, :r] = corner[::-1, :]
    alpha[h - r:, w - r:] = corner[::-1, ::-1]
    return alpha


# ----------------------------------------------------------------------------
# Camera model shared by genuine and attack frames
# ----------------------------------------------------------------------------

@dataclass
class CameraParams:
    gain: float = 1.0
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)     # B, G, R
    blur_sigma: float = 0.0
    motion_length: int = 0
    motion_angle: float = 0.0
    downscale: float = 1.0
    shot_noise: float = 0.0
    read_noise: float = 0.0
    chroma_noise: float = 0.0
    gamma: float = 2.2
    contrast: float = 1.0
    sharpen: float = 0.0
    saturation: float = 1.0
    flare: float = 0.0             # veiling glare of the lens, added in linear light
    exposure_target: float = 0.0   # mean scene level the auto exposure aims for, 0 disables it
    jpeg_quality: int = 0          # 0 disables compression
    extras: dict = field(default_factory=dict)


def sample_camera(rng: np.random.Generator) -> CameraParams:
    """Draw a random consumer camera.

    The same distribution serves every class, so that blur, noise or
    compression on their own never reveal the label.
    """
    wb = 1.0 + rng.normal(0.0, 0.04, size=3)
    return CameraParams(
        gain=float(np.exp(rng.normal(0.0, 0.12))),
        white_balance=(float(wb[0]), float(wb[1]), float(wb[2])),
        blur_sigma=float(abs(rng.normal(0.0, 0.7))) if rng.random() < 0.8 else float(rng.uniform(1.0, 2.0)),
        motion_length=int(rng.integers(3, 9)) if rng.random() < 0.12 else 0,
        motion_angle=float(rng.uniform(0, np.pi)),
        downscale=float(rng.uniform(0.5, 0.95)) if rng.random() < 0.35 else 1.0,
        shot_noise=float(rng.uniform(0.0004, 0.006)),
        read_noise=float(rng.uniform(0.001, 0.012)),
        chroma_noise=float(rng.uniform(0.0, 0.5)),
        gamma=float(rng.uniform(1.9, 2.5)),
        contrast=float(rng.uniform(0.9, 1.15)),
        sharpen=float(rng.uniform(0.2, 1.0)) if rng.random() < 0.45 else 0.0,
        saturation=float(rng.uniform(0.6, 1.3)),
        flare=float(min(0.06, rng.exponential(0.012))) if rng.random() < 0.5 else 0.0,
        exposure_target=float(rng.uniform(0.09, 0.30)),
        jpeg_quality=int(rng.integers(32, 96)) if rng.random() < 0.9 else 0,
    )


def apply_camera(frame_linear: np.ndarray, cam: CameraParams, rng: np.random.Generator) -> np.ndarray:
    """Turn scene radiance into the 8 bit frame a webcam would deliver."""
    gain = cam.gain
    if cam.exposure_target > 0:
        # Auto exposure meters the scene, so a bright display or a dull print
        # does not end up brighter or darker than a real face on average.
        h0, w0 = frame_linear.shape[:2]
        centre = frame_linear[h0 // 5: 4 * h0 // 5, w0 // 5: 4 * w0 // 5]
        level = float(centre.mean()) + 1e-4
        gain *= float(np.clip((cam.exposure_target / level) ** 0.85, 0.35, 3.5))
    img = frame_linear * (gain * np.asarray(cam.white_balance, dtype=np.float32))
    if cam.flare > 0:
        img = img + cam.flare
    h, w = img.shape[:2]
    if cam.blur_sigma > 0.15:
        img = cv2.GaussianBlur(img, (0, 0), cam.blur_sigma)
    if cam.motion_length >= 3:
        k = np.zeros((cam.motion_length, cam.motion_length), dtype=np.float32)
        c = (cam.motion_length - 1) / 2.0
        dx, dy = np.cos(cam.motion_angle) * c, np.sin(cam.motion_angle) * c
        cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
        img = cv2.filter2D(img, -1, k / k.sum())
    if cam.downscale < 0.999:
        small = cv2.resize(img, (max(8, int(w * cam.downscale)), max(8, int(h * cam.downscale))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR if rng.random() < 0.5 else cv2.INTER_CUBIC)
    img = np.clip(img, 0.0, None)
    sigma = np.sqrt(cam.shot_noise * img + cam.read_noise ** 2)
    noise = rng.standard_normal(img.shape).astype(np.float32)
    if cam.chroma_noise > 0:
        # demosaicing correlates noise between neighbouring pixels
        soft = cv2.GaussianBlur(noise, (0, 0), 0.7) * 1.9
        noise = (1 - cam.chroma_noise) * noise + cam.chroma_noise * soft
    img = img + sigma * noise
    out = linear_to_srgb(img, cam.gamma)
    if abs(cam.contrast - 1.0) > 1e-3:
        out = np.clip((out - 0.5) * cam.contrast + 0.5, 0.0, 1.0)
    if abs(cam.saturation - 1.0) > 1e-3:
        luma = out @ np.array([0.114, 0.587, 0.299], dtype=np.float32)
        out = np.clip(luma[:, :, None] + cam.saturation * (out - luma[:, :, None]), 0.0, 1.0)
    if cam.sharpen > 0:
        soft = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = np.clip(out + cam.sharpen * (out - soft), 0.0, 1.0)
    frame = to_uint8(out)
    if cam.jpeg_quality > 0:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(cam.jpeg_quality)])
        if ok:
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return frame


def add_glare(shape_hw: tuple[int, int], rng: np.random.Generator, blobs: list[dict]) -> np.ndarray:
    """Additive specular reflection map built from soft blobs and window shapes."""
    h, w = shape_hw
    scale = max(1, int(max(h, w) / 320))
    hs, ws = max(8, h // scale), max(8, w // scale)
    glare = np.zeros((hs, ws), dtype=np.float32)
    for blob in blobs:
        layer = np.zeros((hs, ws), dtype=np.float32)
        cx, cy = int(blob["u"] * ws), int(blob["v"] * hs)
        ax, ay = max(2, int(blob["ru"] * ws)), max(2, int(blob["rv"] * hs))
        if blob.get("shape") == "window":
            box = cv2.boxPoints(((cx, cy), (2 * ax, 2 * ay), float(np.degrees(blob["angle"]))))
            cv2.fillConvexPoly(layer, box.astype(np.int32), 1.0)
            layer = cv2.GaussianBlur(layer, (0, 0), max(1.0, blob["soft"] * min(ax, ay)))
        else:
            cv2.ellipse(layer, (cx, cy), (ax, ay), float(np.degrees(blob["angle"])), 0, 360, 1.0, -1)
            layer = cv2.GaussianBlur(layer, (0, 0), max(1.5, blob["soft"] * max(ax, ay)))
        glare += blob["intensity"] * layer / (layer.max() + 1e-6)
    return cv2.resize(glare, (w, h), interpolation=cv2.INTER_LINEAR)


def sample_glare_blobs(rng: np.random.Generator, strength: float) -> list[dict]:
    blobs = []
    for _ in range(int(rng.integers(0, 3))):
        blobs.append({
            "u": float(rng.uniform(0.05, 0.95)), "v": float(rng.uniform(0.05, 0.95)),
            "ru": float(rng.uniform(0.05, 0.35)), "rv": float(rng.uniform(0.05, 0.35)),
            "angle": float(rng.uniform(0, np.pi)), "soft": float(rng.uniform(0.15, 0.8)),
            "intensity": float(strength * rng.uniform(0.3, 1.0)),
            "shape": "window" if rng.random() < 0.4 else "blob",
        })
    return blobs
