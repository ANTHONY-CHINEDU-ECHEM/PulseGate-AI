import numpy as np
import pytest

from pulsegate.data.attacks import paper
from pulsegate.data.attacks.common import (
    FaceRef, apply_camera, apply_transform, face_ref_from_detection, plane_homography, sample_camera, similarity_matrix,
)
from pulsegate.data.attacks.simulator import ALL_CLASSES, FAMILY, LIVE, SPECIES, Simulator, describe


@pytest.fixture(scope="module")
def source(tracker, face_image):
    detection = tracker.detect(face_image)[0]
    return face_image, face_ref_from_detection(detection)


@pytest.mark.parametrize("species", ALL_CLASSES)
def test_every_class_renders_a_valid_frame(source, species):
    image, face = source
    sim = Simulator((480, 640))
    rng = np.random.default_rng(11)
    params = sim.sample(species, rng)
    frame, centre = sim.render(image, face, params, rng, image[:, ::-1].copy(), face)
    assert frame.shape == (640, 480, 3) and frame.dtype == np.uint8
    assert 0 <= centre[0] < 480 and 0 <= centre[1] < 640
    assert 10 < frame.mean() < 245 and frame.std() > 5          # not black, white or flat
    assert describe(params)["species"] == species


def test_rendering_is_deterministic_for_a_seed(source):
    image, face = source
    sim = Simulator((480, 640))
    frames = []
    for _ in range(2):
        rng = np.random.default_rng(5)
        params = sim.sample("replay_monitor", rng)
        frames.append(sim.render(image, face, params, rng, image, face)[0])
    assert np.array_equal(frames[0], frames[1])


def test_different_seeds_give_different_scenes(source):
    image, face = source
    sim = Simulator((480, 640))
    out = []
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        out.append(sim.render(image, face, sim.sample("print_matte", rng), rng, image, face)[0])
    assert np.abs(out[0].astype(int) - out[1].astype(int)).mean() > 3


def test_attacks_need_a_background(source):
    image, face = source
    sim = Simulator()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sim.render(image, face, sim.sample("replay_phone", rng), rng)


def test_unknown_species_is_rejected():
    with pytest.raises(ValueError):
        Simulator().sample("hologram", np.random.default_rng(0))


def test_every_species_has_a_family():
    assert set(FAMILY) == set(ALL_CLASSES)
    assert {FAMILY[s] for s in SPECIES} == {"print", "replay", "mask"}
    assert FAMILY[LIVE] == "live"


def test_session_parameters_can_be_reused_with_jitter(source):
    image, face = source
    sim = Simulator((480, 640))
    rng = np.random.default_rng(3)
    params = sim.sample("replay_phone", rng)
    a, ca = sim.render(image, face, params, rng, image, face, jitter=(0, 0, 0, 0, 0))
    b, cb = sim.render(image, face, params, rng, image, face, jitter=(6, -4, 0.01, 0, 0))
    assert cb[0] - ca[0] == pytest.approx(6) and cb[1] - ca[1] == pytest.approx(-4)
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 0.5


def test_halftone_preserves_tone_and_cannot_print_darker_than_ink():
    rng = np.random.default_rng(0)
    p = paper.sample_print(rng, "matte")
    p.ink_density, p.ink_crosstalk = 0.9, 0.05
    means = []
    for value in (0.05, 0.3, 0.6, 0.9):
        tone = np.full((64, 64, 3), value, dtype=np.float32)
        means.append(float(paper._halftone(tone, p, 4.0, np.random.default_rng(1)).mean()))
    assert means == sorted(means)                                 # brighter in, brighter out
    assert means[0] > 0.05                                        # ink is never perfectly black
    assert abs(means[2] - 0.6) < 0.15
    white = paper._halftone(np.ones((32, 32, 3), np.float32), p, 4.0, np.random.default_rng(1))
    assert white.min() > 0.99                                     # no ink on white paper


def test_plane_homography_keeps_the_anchor_and_the_scale():
    h = plane_homography((100, 50), 0.5, 0.0, 0.0, 0.0, (320, 240), focal=800)
    pts = apply_transform(h, np.array([[100.0, 50.0], [102.0, 50.0]]))
    assert pts[0] == pytest.approx([320, 240])
    assert pts[1][0] - pts[0][0] == pytest.approx(1.0, abs=1e-6)
    tilted = plane_homography((100, 50), 0.5, 0.5, 0.0, 0.0, (320, 240), focal=800)
    assert apply_transform(tilted, np.array([[100.0, 50.0]]))[0] == pytest.approx([320, 240])


def test_similarity_matrix_maps_source_to_destination():
    m = similarity_matrix(2.0, 0.3, np.array([10.0, 20.0]), np.array([100.0, 200.0]))
    assert apply_transform(m, np.array([[10.0, 20.0]]))[0] == pytest.approx([100, 200])
    far = apply_transform(m, np.array([[11.0, 20.0]]))[0]
    assert np.linalg.norm(far - [100, 200]) == pytest.approx(2.0)


def test_face_ref_follows_a_transform():
    face = FaceRef(box=np.array([10, 20, 40, 60], np.float32), points=np.array([[20, 40], [40, 40], [30, 50], [22, 65], [38, 65]], np.float32))
    moved = face.transformed(similarity_matrix(2.0, 0.0, np.zeros(2), np.array([5.0, 7.0])))
    assert moved.size == pytest.approx(120)
    assert moved.eye_distance == pytest.approx(40)
    assert moved.center == pytest.approx([65, 107])


def test_camera_model_outputs_valid_images_for_any_draw():
    rng = np.random.default_rng(0)
    scene = rng.random((120, 160, 3)).astype(np.float32) * 0.6
    for _ in range(25):
        frame = apply_camera(scene, sample_camera(rng), rng)
        assert frame.shape == (120, 160, 3) and frame.dtype == np.uint8
