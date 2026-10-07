"""Grad CAM explanations for the context stream.

The map shows which parts of the face crop pushed the decision towards
"attack" (or towards "live"). It answers the audit question of what the model
reacted to: a bezel, a paper edge, a glare patch, or nothing in particular.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .network import PulseGateNet
from .preprocess import context_view, normalize_scale, patch_view, to_tensor


def load_checkpoint(path: str | Path) -> PulseGateNet:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = PulseGateNet(width=float(state.get("width", 1.0)))
    model.load_state_dict(state["state_dict"])
    return model.eval()


class GradCam:
    def __init__(self, model: PulseGateNet, layer_index: int = 4):
        self.model = model.eval()
        self._activation: torch.Tensor | None = None
        self.model.context.features[layer_index].register_forward_hook(self._keep)

    def _keep(self, module, inputs, output) -> None:
        output.retain_grad()
        self._activation = output

    def explain(self, crop_bgr: np.ndarray, context_scale: float = 1.8, context_size: int = 112, patch_size: int = 96,
                towards: str = "attack", norm: str = "standardise", normalise: bool = True) -> tuple[np.ndarray, float]:
        """Return a heat map at crop resolution and the live probability.

        With ``normalise=True`` the map is scaled to 0..1 on its own. Pass
        ``False`` to compare the strength of evidence between images.
        """
        crop = normalize_scale(crop_bgr, context_scale)
        context = torch.from_numpy(to_tensor(context_view(crop, context_size), norm))[None]
        texture = torch.from_numpy(to_tensor(patch_view(crop, patch_size, (0.0, 0.0), context_scale), norm))[None]
        self.model.zero_grad(set_to_none=True)
        out = self.model(context, texture)
        logit = out["logit_context"][0]
        (-logit if towards == "attack" else logit).backward()
        activation = self._activation
        weights = activation.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1))[0].detach().numpy()
        if normalise:
            cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (crop_bgr.shape[1], crop_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        cam = np.clip(cam, 0.0, 1.0 if normalise else None)
        return cam, float(torch.sigmoid(out["logit"][0].detach()))


def overlay(crop_bgr: np.ndarray, cam: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """Blend a heat map over the image. Cold regions stay untouched."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = (strength * cam)[:, :, None]
    return (crop_bgr * (1 - alpha) + heat * alpha).astype(np.uint8)
