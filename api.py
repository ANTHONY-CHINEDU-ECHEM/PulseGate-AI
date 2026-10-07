"""HTTP service for liveness checks.

Three ways to use it
* ``POST /v1/liveness/image``: one still image, passive model only
* ``POST /v1/liveness/video``: a short clip, passive model plus pulse and blinking
* ``/v1/sessions``: an interactive session with random challenges, fed frame by frame

A small browser client for the interactive flow is served at ``/``.
"""
from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .. import __version__
from ..config import Config, load_config
from ..engine.session import LivenessEngine, LivenessSession

STATIC_DIR = Path(__file__).parent / "static"


class SessionStore:
    """In memory sessions with a time to live. One process, one store."""

    def __init__(self, ttl_s: float, max_sessions: int = 64):
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._items: dict[str, tuple[float, LivenessSession]] = {}

    def purge(self) -> None:
        now = time.time()
        for key in [k for k, (touched, _) in self._items.items() if now - touched > self.ttl_s]:
            del self._items[key]

    def add(self, session: LivenessSession) -> None:
        self.purge()
        if len(self._items) >= self.max_sessions:
            raise HTTPException(status_code=429, detail="too many open sessions, try again shortly")
        self._items[session.session_id] = (time.time(), session)

    def get(self, session_id: str) -> LivenessSession:
        self.purge()
        item = self._items.get(session_id)
        if item is None:
            raise HTTPException(status_code=404, detail="unknown or expired session")
        self._items[session_id] = (time.time(), item[1])
        return item[1]

    def remove(self, session_id: str) -> bool:
        return self._items.pop(session_id, None) is not None

    def __len__(self) -> int:
        return len(self._items)


async def _read_limited(upload: UploadFile, limit_bytes: int) -> bytes:
    data = await upload.read(limit_bytes + 1)
    if len(data) > limit_bytes:
        raise HTTPException(status_code=413, detail="upload is too large")
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    return data


def _decode_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="the upload is not a readable image")
    if max(image.shape[:2]) > 1920:
        scale = 1920.0 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def create_app(cfg: Config | None = None, engine: LivenessEngine | None = None) -> FastAPI:
    cfg = cfg or load_config()
    engine = engine or LivenessEngine(cfg)
    store = SessionStore(float(cfg.api.session_ttl_s))
    lock = threading.Lock()          # the networks are shared, so inference is serialised
    limit = int(float(cfg.api.max_upload_mb) * 1024 * 1024)

    app = FastAPI(
        title="PulseGate AI",
        version=__version__,
        description="Face liveness detection: passive texture analysis, random challenges, depth from motion and remote pulse.",
    )
    app.state.engine = engine
    app.state.store = store

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf8")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "open_sessions": len(store)}

    @app.get("/v1/model")
    def model() -> dict:
        info = engine.model_info()
        meta = engine.passive.meta
        info.update({"validation": meta.get("validation", {}), "thresholds": meta.get("thresholds", {}), "classes": meta.get("classes", [])})
        return info

    @app.post("/v1/liveness/image")
    async def liveness_image(file: UploadFile = File(...)) -> dict:
        image = _decode_image(await _read_limited(file, limit))
        with lock:
            return engine.check_image(image)

    @app.post("/v1/liveness/video")
    async def liveness_video(file: UploadFile = File(...)) -> dict:
        data = await _read_limited(file, limit)
        suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
            handle.write(data)
            handle.flush()
            try:
                with lock:
                    result = engine.analyze_video(handle.name, max_seconds=float(cfg.api.max_video_seconds))
            except FileNotFoundError as exc:
                raise HTTPException(status_code=400, detail="the upload is not a readable video") from exc
        return result.as_dict()

    @app.post("/v1/sessions")
    def create_session(mode: str = "interactive") -> dict:
        if mode not in ("interactive", "passive"):
            raise HTTPException(status_code=422, detail="mode must be interactive or passive")
        session = engine.new_session(mode)
        store.add(session)
        return {"session_id": session.session_id, "mode": mode, "steps": len(session.plan), "ttl_s": store.ttl_s}

    @app.post("/v1/sessions/{session_id}/frames")
    async def push_frame(session_id: str, file: UploadFile = File(...), timestamp: float | None = Form(default=None)) -> dict:
        session = store.get(session_id)
        image = _decode_image(await _read_limited(file, limit))
        # Server time is authoritative. A client supplied clock could be used to fake response timing.
        del timestamp
        with lock:
            status = session.update(image, time.monotonic())
        return status.as_dict()

    @app.post("/v1/sessions/{session_id}/finish")
    def finish_session(session_id: str) -> dict:
        session = store.get(session_id)
        with lock:
            return session.finalize().as_dict()

    @app.get("/v1/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        session = store.get(session_id)
        return {
            "session_id": session.session_id, "mode": session.mode, "stage": session.stage.value,
            "result": None if session.result is None else session.result.as_dict(),
        }

    @app.delete("/v1/sessions/{session_id}")
    def delete_session(session_id: str) -> JSONResponse:
        if not store.remove(session_id):
            raise HTTPException(status_code=404, detail="unknown or expired session")
        return JSONResponse({"deleted": session_id})

    return app
