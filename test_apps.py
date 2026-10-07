"""The desktop apps, driven by the sample clip instead of a camera and with the window calls stubbed."""
import copy
import json

import cv2
import numpy as np
import pytest

from pulsegate.apps import hud, webcam
from pulsegate.engine.session import Stage


class FakeWindow:
    """Stands in for the OpenCV window functions and feeds key presses."""

    def __init__(self, keys_after_frames: dict[int, str]):
        self.keys = keys_after_frames
        self.frames = 0
        self.last = None

    def imshow(self, name, image):
        self.frames += 1
        self.last = image

    def wait_key(self, delay):
        key = self.keys.get(self.frames)
        return ord(key) if key else 255


@pytest.fixture
def headless(monkeypatch, clip_path):
    def install(keys):
        window = FakeWindow(keys)
        monkeypatch.setattr(webcam, "open_camera", lambda cfg: cv2.VideoCapture(str(clip_path)))
        monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
        monkeypatch.setattr(cv2, "imshow", window.imshow)
        monkeypatch.setattr(cv2, "waitKey", window.wait_key)
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
        return window
    return install


def test_webcam_loop_runs_a_passive_session_and_saves_the_record(cfg, engine, headless, tmp_path, monkeypatch, capsys):
    # passive mode finishes on its own; clock time is replaced by clip time so the session sees 12 frames per second
    ticks = iter(np.arange(0, 1000, 1 / 12.0))
    monkeypatch.setattr(webcam.time, "monotonic", lambda: float(next(ticks)))
    window = headless({200: "s", 210: "r", 230: "q"})
    local = copy.deepcopy(cfg)
    local.paths.reports_dir = str(tmp_path)
    webcam.run_webcam(local, mode="passive")
    assert window.frames == 230                                   # stopped by the quit key
    assert window.last.shape[2] == 3
    saved = list((tmp_path / "sessions").glob("session_*.json"))
    assert len(saved) == 1
    record = json.loads(saved[0].read_text())
    assert record["mode"] == "passive" and record["decision"] == "live"
    assert saved[0].with_suffix(".jpg").exists()
    assert '"decision": "live"' in capsys.readouterr().out          # the verdict is also printed


def test_webcam_loop_survives_the_end_of_the_stream(cfg, engine, headless, capsys):
    headless({})
    webcam.run_webcam(cfg, mode="interactive")                     # reads until the clip is exhausted
    assert "could not be read" in capsys.readouterr().out


def test_collect_writes_labelled_crops(cfg, headless, tmp_path):
    headless({})
    local = copy.deepcopy(cfg)
    local.paths.capture_dir = str(tmp_path)
    out = webcam.run_collect(local, label="live", seconds=1.5, every=2)
    crops = sorted(out.glob("*.jpg"))
    assert out == tmp_path / "live" and len(crops) >= 5
    image = cv2.imread(str(crops[0]))
    assert image.shape[0] == image.shape[1] and image.shape[0] > 150


def test_collect_rejects_unknown_labels(cfg):
    with pytest.raises(ValueError):
        webcam.run_collect(cfg, label="hologram", seconds=0.1)


def test_hud_draws_every_stage_without_changing_the_input(engine, clip_frames):
    frames, fps = clip_frames
    session = engine.new_session("interactive", seed=3, plan=["look_up", "turn_right", "turn_left"])
    seen = set()
    for i, frame in enumerate(frames):
        before = frame.copy()
        status = session.update(frame, i / fps)
        for mirror in (False, True):
            view = hud.draw_hud(frame, status, session.passive_threshold, mirror=mirror, pulse_text="60 bpm", fps=fps)
            assert view.shape == frame.shape and view.dtype == np.uint8
        assert np.array_equal(frame, before)
        seen.add(status.stage)
        if status.stage == Stage.DONE:
            break
    assert seen == {Stage.POSITIONING, Stage.RECENTER, Stage.CHALLENGE, Stage.HOLD, Stage.DONE}


def test_hud_handles_a_frame_without_a_face(engine):
    session = engine.new_session("interactive")
    blank = np.full((480, 640, 3), 90, np.uint8)
    status = session.update(blank, 0.0)
    view = hud.draw_hud(blank, status, 0.5)
    assert view.shape == blank.shape and status.guidance == "Look at the camera"
