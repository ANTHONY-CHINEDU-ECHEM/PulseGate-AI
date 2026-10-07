import pytest

from pulsegate.config import PROJECT_ROOT, Config, load_config, parse_overrides


def test_defaults_load_and_expose_attributes():
    cfg = load_config()
    assert cfg.crop.context_size == 112
    assert cfg.fusion.weights.passive == pytest.approx(0.5)
    assert abs(sum(cfg.fusion.weights.values()) - 1.0) < 1e-9


def test_overrides_are_typed():
    cfg = load_config(overrides=["train.epochs=3", "webcam.mirror=false", "train.exclude_species=[replay_phone]", "new.section.value=1.5"])
    assert cfg.train.epochs == 3 and isinstance(cfg.train.epochs, int)
    assert cfg.webcam.mirror is False
    assert cfg.train.exclude_species == ["replay_phone"]
    assert cfg.new.section.value == 1.5


def test_bad_override_is_rejected():
    with pytest.raises(ValueError):
        parse_overrides(["no_equals_sign"])


def test_paths_resolve_against_the_repository_root():
    cfg = load_config()
    assert cfg.path("assets.detector") == PROJECT_ROOT / "models/third_party/face_detection_yunet_2023mar.onnx"
    assert cfg.path("assets.detector").exists()
    assert cfg.path("assets.facemesh").exists()


def test_round_trip_to_plain_dict():
    cfg = load_config()
    plain = cfg.to_dict()
    assert isinstance(plain["train"], dict) and not isinstance(plain["train"], Config)
    assert Config(plain).train.batch_size == cfg.train.batch_size


def test_missing_key_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = load_config().does_not_exist
