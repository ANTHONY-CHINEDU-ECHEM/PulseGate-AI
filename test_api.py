import cv2
import numpy as np
import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from pulsegate.apps.api import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client(cfg, engine):
    return fastapi_testclient.TestClient(create_app(cfg, engine))


def jpeg(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


def test_health_and_model(client):
    assert client.get("/health").json()["status"] == "ok"
    model = client.get("/v1/model").json()
    assert model["name"] == "PulseGateNet" and 0 < model["threshold"] < 1
    assert "live" in model["classes"]


def test_browser_client_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "PulseGate AI" in response.text


def test_image_endpoint(client, face_image):
    response = client.post("/v1/liveness/image", files={"file": ("face.jpg", jpeg(face_image), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] in ("live", "not_live") and "passive" in body


def test_image_endpoint_rejects_garbage(client):
    assert client.post("/v1/liveness/image", files={"file": ("x.jpg", b"not an image", "image/jpeg")}).status_code == 400
    assert client.post("/v1/liveness/image", files={"file": ("x.jpg", b"", "image/jpeg")}).status_code == 400
    assert client.post("/v1/liveness/image").status_code == 422


def test_video_endpoint(client, clip_path):
    with open(clip_path, "rb") as handle:
        response = client.post("/v1/liveness/video", files={"file": ("clip.mp4", handle.read(), "video/mp4")})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "passive" and body["decision"] == "live"


def test_video_endpoint_rejects_garbage(client):
    assert client.post("/v1/liveness/video", files={"file": ("clip.mp4", b"0000", "video/mp4")}).status_code in (200, 400)


def test_interactive_session_flow(client, clip_frames):
    frames, _ = clip_frames
    created = client.post("/v1/sessions").json()
    assert created["mode"] == "interactive" and created["steps"] == 3
    sid = created["session_id"]
    status = None
    for frame in frames[:30]:
        response = client.post(f"/v1/sessions/{sid}/frames", files={"file": ("f.jpg", jpeg(frame), "image/jpeg")})
        assert response.status_code == 200
        status = response.json()
    assert status["face_detected"] and status["stage"] in ("positioning", "recenter", "challenge", "done")
    assert client.get(f"/v1/sessions/{sid}").json()["session_id"] == sid
    final = client.post(f"/v1/sessions/{sid}/finish").json()
    assert final["decision"] in ("live", "not_live", "inconclusive")
    assert client.delete(f"/v1/sessions/{sid}").status_code == 200
    assert client.get(f"/v1/sessions/{sid}").status_code == 404


def test_unknown_session_and_bad_mode(client, face_image):
    assert client.get("/v1/sessions/nope").status_code == 404
    assert client.post("/v1/sessions/nope/frames", files={"file": ("f.jpg", jpeg(face_image), "image/jpeg")}).status_code == 404
    assert client.post("/v1/sessions?mode=psychic").status_code == 422


def test_upload_limit(cfg, engine, face_image):
    import copy

    small = copy.deepcopy(cfg)
    small.api.max_upload_mb = 0.001
    tight = fastapi_testclient.TestClient(create_app(small, engine))
    response = tight.post("/v1/liveness/image", files={"file": ("face.jpg", jpeg(face_image), "image/jpeg")})
    assert response.status_code == 413


def test_sessions_expire(cfg, engine):
    import copy
    import time

    short = copy.deepcopy(cfg)
    short.api.session_ttl_s = 0.05
    app_client = fastapi_testclient.TestClient(create_app(short, engine))
    sid = app_client.post("/v1/sessions").json()["session_id"]
    time.sleep(0.1)
    assert app_client.get(f"/v1/sessions/{sid}").status_code == 404
