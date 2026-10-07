"""Heads up display drawn on camera frames.

Drawing is separate from capture so the same overlay serves the webcam app,
recorded clips and the figures in the documentation.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..engine.session import SessionStatus, Stage
from ..vision.geometry import FACE_OVAL, LEFT_EYE, RIGHT_EYE

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (245, 245, 245)
MUTED = (170, 170, 170)
GREEN = (110, 200, 60)
RED = (70, 70, 230)
AMBER = (40, 180, 250)
TEAL = (190, 170, 40)
PANEL = (28, 24, 22)


def _text(img, text, org, scale=0.6, color=WHITE, thickness=1):
    cv2.putText(img, text, (int(org[0]), int(org[1])), FONT, scale, color, thickness, cv2.LINE_AA)


def _panel(img, x0, y0, x1, y1, alpha=0.62):
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(img.shape[1], int(x1)), min(img.shape[0], int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    region = img[y0:y1, x0:x1]
    region[:] = (region * (1 - alpha) + np.array(PANEL) * alpha).astype(np.uint8)


def _bar(img, x, y, w, h, value, color, threshold=None):
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (70, 70, 70), -1)
    cv2.rectangle(img, (int(x), int(y)), (int(x + w * float(np.clip(value, 0, 1))), int(y + h)), color, -1)
    if threshold is not None:
        tx = int(x + w * float(np.clip(threshold, 0, 1)))
        cv2.line(img, (tx, int(y - 3)), (tx, int(y + h + 3)), WHITE, 1, cv2.LINE_AA)


def draw_hud(frame_bgr: np.ndarray, status: SessionStatus, threshold: float = 0.5, mirror: bool = False,
             pulse_text: str = "", fps: float | None = None, passive_running: float | None = None) -> np.ndarray:
    """Return a copy of the frame with guidance, signal read outs and the verdict.

    With ``mirror=True`` the picture is flipped like a selfie preview. Face
    geometry is flipped with it, text is drawn afterwards so it stays readable.
    """
    img = frame_bgr[:, ::-1].copy() if mirror else frame_bgr.copy()
    h, w = img.shape[:2]
    s = max(0.6, min(w, h) / 720.0)                    # scale drawing with resolution
    obs = status.observation
    result = status.result

    accent = AMBER
    if status.stage in (Stage.CHALLENGE, Stage.RECENTER, Stage.HOLD):
        accent = TEAL
    if result is not None:
        accent = {"live": GREEN, "not_live": RED}.get(result.decision, AMBER)

    # guide oval
    centre = (w // 2, int(h * 0.47))
    axes = (int(min(w, h) * 0.24), int(min(w, h) * 0.33))
    cv2.ellipse(img, centre, axes, 0, 0, 360, accent, max(1, int(2 * s)), cv2.LINE_AA)

    if obs is not None:
        pts = obs.landmarks[:, :2].copy()
        if mirror:
            pts[:, 0] = w - 1 - pts[:, 0]
        hull = np.round(pts[list(FACE_OVAL)]).astype(np.int32)
        cv2.polylines(img, [hull], True, accent, 1, cv2.LINE_AA)
        for eye in (LEFT_EYE, RIGHT_EYE):
            cv2.polylines(img, [np.round(pts[list(eye)]).astype(np.int32)], True, accent, 1, cv2.LINE_AA)

    # top bar
    bar_h = int(46 * s)
    _panel(img, 0, 0, w, bar_h)
    _text(img, "PulseGate AI", (14 * s, 30 * s), 0.8 * s, WHITE, max(1, int(2 * s)))
    stage_label = {
        Stage.POSITIONING: "POSITIONING", Stage.RECENTER: "CHALLENGE", Stage.CHALLENGE: "CHALLENGE",
        Stage.HOLD: "ANALYSING", Stage.DONE: "RESULT",
    }[status.stage]
    if status.steps and status.stage in (Stage.RECENTER, Stage.CHALLENGE):
        stage_label += f" {min(status.step + (1 if status.stage == Stage.RECENTER else 0), status.steps)}/{status.steps}"
    size = cv2.getTextSize(stage_label, FONT, 0.6 * s, 1)[0]
    _text(img, stage_label, (w - size[0] - 14 * s, 29 * s), 0.6 * s, accent, max(1, int(1.5 * s)))

    # signal panel
    px, py, pw = int(12 * s), bar_h + int(12 * s), int(250 * s)
    rows = 5
    _panel(img, px, py, px + pw, py + int((28 * rows + 16) * s))
    line = py + int(26 * s)
    step_y = int(28 * s)
    passive = passive_running if passive_running is not None else (status.passive.live if status.passive is not None else None)
    _text(img, "Passive", (px + 10 * s, line), 0.5 * s, MUTED)
    if passive is not None:
        _bar(img, px + 86 * s, line - 11 * s, 100 * s, 10 * s, passive, GREEN if passive >= threshold else RED, threshold)
        _text(img, f"{passive:.2f}", (px + 196 * s, line), 0.5 * s, WHITE)
    else:
        _text(img, "waiting", (px + 86 * s, line), 0.5 * s, MUTED)
    line += step_y
    _text(img, "Head", (px + 10 * s, line), 0.5 * s, MUTED)
    if obs is not None:
        _text(img, f"yaw {obs.pose.yaw:+.0f}  pitch {obs.pose.pitch:+.0f}", (px + 86 * s, line), 0.5 * s, WHITE)
    line += step_y
    _text(img, "Eyes", (px + 10 * s, line), 0.5 * s, MUTED)
    if obs is not None:
        _text(img, f"open {obs.ear:.2f}  blinks {status.blinks}", (px + 86 * s, line), 0.5 * s, WHITE)
    line += step_y
    _text(img, "Pulse", (px + 10 * s, line), 0.5 * s, MUTED)
    _text(img, pulse_text or "measuring", (px + 86 * s, line), 0.5 * s, WHITE if pulse_text else MUTED)
    line += step_y
    _text(img, "Camera", (px + 10 * s, line), 0.5 * s, MUTED)
    quality = status.quality
    q_text = "good" if quality is not None and quality.ok else (quality.issues[0].replace("_", " ") if quality is not None and quality.issues else "")
    if fps is not None:
        q_text = f"{q_text}  {fps:.0f} fps".strip()
    _text(img, q_text, (px + 86 * s, line), 0.5 * s, WHITE)

    # prompt area
    bottom_h = int(96 * s)
    _panel(img, 0, h - bottom_h, w, h)
    prompt = status.prompt
    scale = 0.95 * s
    size = cv2.getTextSize(prompt, FONT, scale, 2)[0]
    _text(img, prompt, ((w - size[0]) / 2, h - bottom_h + 40 * s), scale, accent if result is not None else WHITE, max(1, int(2 * s)))
    if result is not None:
        detail = f"score {result.score:.2f}   " + ", ".join(r.replace("_", " ") for r in result.reasons[:2])
        size = cv2.getTextSize(detail, FONT, 0.55 * s, 1)[0]
        _text(img, detail, ((w - size[0]) / 2, h - bottom_h + 70 * s), 0.55 * s, WHITE)
    elif status.stage == Stage.CHALLENGE:
        bw = int(w * 0.4)
        _bar(img, (w - bw) / 2, h - bottom_h + 58 * s, bw, 8 * s, status.challenge_progress, TEAL)
        _text(img, f"{status.time_left:.1f}s", ((w + bw) / 2 + 12 * s, h - bottom_h + 67 * s), 0.5 * s, MUTED)
    elif status.guidance:
        size = cv2.getTextSize(status.guidance, FONT, 0.6 * s, 1)[0]
        _text(img, status.guidance, ((w - size[0]) / 2, h - bottom_h + 70 * s), 0.6 * s, AMBER)
        if status.stage == Stage.POSITIONING and status.challenge_progress > 0:
            bw = int(w * 0.25)
            _bar(img, (w - bw) / 2, h - bottom_h + 80 * s, bw, 5 * s, status.challenge_progress, AMBER)
    return img
