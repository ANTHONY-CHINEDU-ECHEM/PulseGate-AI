import numpy as np
import pytest

from pulsegate.models import preprocess as pp
from pulsegate.models.texture_baseline import lbp_histogram, texture_features

torch = pytest.importorskip("torch")

from pulsegate.models.export import export_onnx  # noqa: E402
from pulsegate.models.network import CLASS_NAMES, PulseGateNet, count_parameters  # noqa: E402


def test_forward_shapes_and_head_count():
    model = PulseGateNet().eval()
    out = model(torch.randn(3, 3, 112, 112), torch.randn(3, 3, 96, 96))
    assert out["logit"].shape == (3,) and out["logit_context"].shape == (3,) and out["logit_texture"].shape == (3,)
    assert out["species"].shape == (3, len(CLASS_NAMES))
    assert 0.5e6 < count_parameters(model) < 2e6


def test_stream_dropout_only_acts_in_training():
    torch.manual_seed(0)
    model = PulseGateNet(stream_dropout=0.5)
    context, texture = torch.randn(8, 3, 112, 112), torch.randn(8, 3, 96, 96)
    model.eval()
    with torch.no_grad():
        a, b = model(context, texture)["logit"], model(context, texture)["logit"]
    assert torch.equal(a, b)


def test_each_stream_has_its_own_gradient_path():
    model = PulseGateNet()
    out = model(torch.randn(2, 3, 112, 112), torch.randn(2, 3, 96, 96))
    out["logit_texture"].sum().backward()
    assert model.texture.features[0][0].weight.grad.abs().sum() > 0
    assert model.context.features[0][0].weight.grad is None


def test_high_pass_residual_keeps_texture_and_drops_brightness():
    from pulsegate.models.network import TextureStream

    flat = torch.full((1, 3, 32, 32), 0.3)
    grid = flat.clone()
    grid[:, :, ::2, ::2] += 0.2
    grid[:, :, 1::2, 1::2] += 0.2
    assert TextureStream.high_pass(flat).abs().max() < 1e-6            # a flat patch has no residual
    assert TextureStream.high_pass(grid).abs().mean() > 0.2            # a pixel grid has a strong one
    assert torch.allclose(TextureStream.high_pass(grid), TextureStream.high_pass(grid + 0.4), atol=1e-6)


def test_onnx_export_matches_pytorch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    model = PulseGateNet().eval()
    path = export_onnx(model, tmp_path / "net.onnx")          # raises when outputs differ
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch in (1, 5):                                      # the batch axis is dynamic
        context = np.random.rand(batch, 3, 112, 112).astype(np.float32)
        texture = np.random.rand(batch, 3, 96, 96).astype(np.float32)
        logits, species = session.run(None, {"context": context, "texture": texture})
        assert logits.shape == (batch, 3) and species.shape == (batch, len(CLASS_NAMES))


def test_preprocessing_shapes_and_range():
    crop = np.random.randint(0, 255, (300, 300, 3), np.uint8)
    context, texture = pp.build_inputs(crop, 1.8, 112, 96, patches=3)
    assert context.shape == (3, 3, 112, 112) and texture.shape == (3, 3, 96, 96)
    assert context.dtype == np.float32 and -4.0 <= context.min() and context.max() <= 4.0
    assert not np.array_equal(texture[0], texture[1])         # different patch positions


def test_texture_patch_keeps_native_pixels():
    crop = np.random.randint(0, 255, (300, 300, 3), np.uint8)
    patch = pp.patch_view(crop, 96)
    assert np.array_equal(patch, crop[102:198, 102:198])      # a plain slice, never resampled


def test_large_faces_are_reduced_and_small_crops_still_work():
    big = np.zeros((900, 900, 3), np.uint8)                   # face of 500 pixels
    assert pp.normalize_scale(big, 1.8).shape[0] == round(900 * 256 / 500)
    small = np.zeros((200, 200, 3), np.uint8)
    assert pp.normalize_scale(small, 1.8) is small
    tiny = np.zeros((80, 80, 3), np.uint8)
    assert pp.patch_view(tiny, 96).shape == (96, 96, 3)


def test_patch_offsets_stay_inside_the_crop():
    crop = np.zeros((170, 170, 3), np.uint8)
    for offset in pp.TTA_OFFSETS + ((2.0, 2.0), (-2.0, -2.0)):
        assert pp.patch_view(crop, 96, offset).shape == (96, 96, 3)


def test_tensor_conversion_swaps_channels_and_standardises():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (32, 32, 3)).astype(np.uint8)
    image[:, :, 2] = np.clip(image[:, :, 2].astype(int) + 60, 0, 255)      # stronger red in BGR
    tensor = pp.to_tensor(image)
    assert tensor.shape == (3, 32, 32) and tensor.dtype == np.float32
    assert np.allclose(tensor.mean(axis=(1, 2)), 0.0, atol=1e-4)           # every channel is centred
    assert np.all(tensor.std(axis=(1, 2)) > 0.8)
    red = image[:, :, 2].astype(np.float32)
    assert np.corrcoef(tensor[0].ravel(), red.ravel())[0, 1] > 0.999       # channel 0 is red


def test_inputs_do_not_depend_on_exposure_contrast_or_colour_cast():
    rng = np.random.default_rng(1)
    image = rng.integers(60, 180, (48, 48, 3)).astype(np.float32)
    reference = pp.to_tensor(image.astype(np.uint8))
    brighter = pp.to_tensor(np.clip(image + 40, 0, 255).astype(np.uint8))
    flatter = pp.to_tensor(np.clip((image - 120) * 0.6 + 120, 0, 255).astype(np.uint8))
    tinted = pp.to_tensor(np.clip(image * np.array([1.15, 1.0, 0.9]), 0, 255).astype(np.uint8))
    assert np.abs(brighter - reference).max() < 0.02
    assert np.abs(flatter - reference).mean() < 0.06
    assert np.abs(tinted - reference).mean() < 0.06


def test_flat_images_do_not_blow_up():
    tensor = pp.to_tensor(np.full((16, 16, 3), 128, np.uint8))
    assert np.isfinite(tensor).all() and np.abs(tensor).max() < 1e-3


def test_lbp_histogram_is_a_distribution_and_sees_texture():
    rng = np.random.default_rng(0)
    flat = lbp_histogram(np.full((64, 64), 120, np.uint8))
    noisy = lbp_histogram(rng.integers(0, 255, (64, 64)).astype(np.uint8))
    assert flat.shape == (59,) and flat.sum() == pytest.approx(1.0)
    assert flat.max() > 0.99                                  # one pattern only
    assert noisy.max() < 0.5


def test_baseline_features_have_a_fixed_length():
    rng = np.random.default_rng(0)
    a = texture_features(rng.integers(0, 255, (250, 250, 3)).astype(np.uint8))
    b = texture_features(rng.integers(0, 255, (410, 410, 3)).astype(np.uint8))
    assert a.shape == b.shape and np.isfinite(a).all()
