"""Real time liveness check on a webcam with an on screen guide."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from ..config import Config
from ..engine.session import LivenessEngine, Stage
from .hud import draw_hud

HELP = "keys:  r restart   p passive mode   i interactive mode   s save report   q quit"


def open_camera(cfg: Config) -> cv2.VideoCapture:
    w = cfg.webcam
    cap = cv2.VideoCapture(int(w.camera_index))
    if not cap.isOpened():
        raise RuntimeError(
            f"camera {w.camera_index} could not be opened. Close other apps that use the camera, "
            "or pick another one with webcam.camera_index=1"
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(w.height))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def run_webcam(cfg: Config, mode: str = "interactive") -> None:
    """Open the camera and run liveness sessions until the user quits."""
    engine = LivenessEngine(cfg)
    cap = open_camera(cfg)
    window = str(cfg.webcam.window_name)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    out_dir = cfg.path("paths.reports_dir") / "sessions"
    session = engine.new_session(mode)
    print(HELP)
    last_frame_t = time.monotonic()
    fps = 0.0
    pulse_text, pulse_checked = "", 0.0
    running: list[float] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera frame could not be read")
                break
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last_frame_t, 1e-3) if fps else 1.0 / max(now - last_frame_t, 1e-3)
            last_frame_t = now
            status = session.update(frame, now)
            if status.passive is not None:
                running = (running + [status.passive.live])[-15:]
            if now - pulse_checked > 1.0 and not session.done:
                pulse_checked = now
                estimate = session.pulse.estimate()
                pulse_text = f"{estimate.bpm:.0f} bpm  {estimate.snr_db:+.1f} dB" if estimate.valid else ""
            if status.result is not None and status.result.pulse.get("valid"):
                pulse_text = f"{status.result.pulse['bpm']:.0f} bpm  {status.result.pulse['snr_db']:+.1f} dB"
            view = draw_hud(
                frame, status, session.passive_threshold, mirror=bool(cfg.webcam.mirror), pulse_text=pulse_text, fps=fps,
                passive_running=sum(running) / len(running) if running else None,
            )
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("i"), ord("p")):
                mode = {"i": "interactive", "p": "passive"}.get(chr(key), mode)
                session = engine.new_session(mode)
                running, pulse_text = [], ""
            if key == ord("s") and status.result is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = out_dir / f"session_{status.result.session_id}"
                stem.with_suffix(".json").write_text(json.dumps(status.result.as_dict(), indent=2))
                cv2.imwrite(str(stem.with_suffix(".jpg")), view)
                print(f"saved {stem}.json")
            if status.stage == Stage.DONE and status.result is not None and not getattr(session, "_printed", False):
                session._printed = True
                print(json.dumps(status.result.as_dict(), indent=2))
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_collect(cfg: Config, label: str, seconds: float = 20.0, every: int = 3) -> Path:
    """Record your own genuine or attack captures for calibration and fine tuning.

    ``label`` is ``live`` or one of the attack species. Hold the photograph or
    the phone in front of the camera for attack labels. Crops are written in
    the same layout as the main dataset, under ``data/captures``.
    """
    from ..data.captures import CLASS_NAMES
    from ..vision.geometry import crop_square, square_box
    from ..vision.tracker import FaceTracker

    if label not in CLASS_NAMES:
        raise ValueError(f"label must be one of {', '.join(CLASS_NAMES)}")
    tracker = FaceTracker(cfg)
    cap = open_camera(cfg)
    out_dir = cfg.path("paths.capture_dir") / label
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    saved, index = 0, 0
    start = time.monotonic()
    window = f"{cfg.webcam.window_name} capture: {label}"
    try:
        while time.monotonic() - start < seconds:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            obs = tracker.process(frame, time.monotonic())
            view = frame[:, ::-1].copy() if cfg.webcam.mirror else frame.copy()
            if obs is not None and obs.size >= cfg.crop.min_face_px and index % every == 0:
                cx, cy, side = square_box(obs.box, cfg.crop.context_scale)
                cv2.imwrite(str(out_dir / f"{stamp}_{saved:04d}.jpg"), crop_square(frame, cx, cy, side), [cv2.IMWRITE_JPEG_QUALITY, 97])
                saved += 1
            left = seconds - (time.monotonic() - start)
            cv2.putText(view, f"{label}: {saved} crops saved, {left:.0f}s left, q stops", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 180, 250), 2, cv2.LINE_AA)
            cv2.imshow(window, view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"saved {saved} crops to {out_dir}")
    return out_dir
