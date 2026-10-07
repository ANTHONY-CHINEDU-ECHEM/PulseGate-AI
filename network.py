"""PulseGateNet: a compact two stream network for single frame presentation attack detection.

Two views of the same face go through separate streams.

* The context stream reads the face with its surroundings at low resolution and
  learns global evidence such as bezels, paper edges, glare and colour.
* The texture stream reads a patch at native camera resolution, preceded by a
  fixed high pass residual, and learns local evidence such as moire, halftone
  dots and pixel grids.

Each stream has its own classifier (deep supervision), so either one can be
evaluated alone, and a fusion head combines both. An auxiliary head names the
attack species, which regularises the features and makes decisions explainable.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

CLASS_NAMES = ("live", "print_matte", "print_glossy", "replay_phone", "replay_monitor", "cutout_mask")


def _conv(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with an identity skip.

    Plain convolutions are used on purpose: on a CPU they train several times
    faster than depthwise separable blocks of similar accuracy at this scale.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = _conv(channels, channels)
        self.conv2 = _conv(channels, channels, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.conv2(self.conv1(x)))


class ContextStream(nn.Module):
    def __init__(self, width: float = 1.0, out_dim: int = 192):
        super().__init__()
        c = [max(8, int(round(v * width))) for v in (16, 32, 64, 128)]
        self.features = nn.Sequential(
            _conv(3, c[0], 3, 2),                      # 112 -> 56
            _conv(c[0], c[1], 3, 2),                   # 28
            ResidualBlock(c[1]),
            _conv(c[1], c[2], 3, 2),                   # 14
            ResidualBlock(c[2]),
            _conv(c[2], c[3], 3, 2),                   # 7
            ResidualBlock(c[3]),
            _conv(c[3], out_dim, 1),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).mean(dim=(2, 3))


class TextureStream(nn.Module):
    def __init__(self, width: float = 1.0, out_dim: int = 160):
        super().__init__()
        c = [max(8, int(round(v * width))) for v in (16, 32, 48, 96, 128)]
        self.features = nn.Sequential(
            _conv(6, c[0], 3, 1),                      # 96, full resolution keeps fine structure
            _conv(c[0], c[1], 3, 2),                   # 48
            _conv(c[1], c[2], 3, 2),                   # 24
            ResidualBlock(c[2]),
            _conv(c[2], c[3], 3, 2),                   # 12
            ResidualBlock(c[3]),
            _conv(c[3], c[4], 3, 2),                   # 6
            _conv(c[4], out_dim, 1),
        )
        self.out_dim = out_dim

    @staticmethod
    def high_pass(x: torch.Tensor) -> torch.Tensor:
        """Fixed high pass residual: what remains after removing the local mean."""
        return (x - F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)) * 4.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(torch.cat([x, self.high_pass(x)], dim=1)).mean(dim=(2, 3))


class PulseGateNet(nn.Module):
    def __init__(self, width: float = 1.0, num_classes: int = len(CLASS_NAMES), dropout: float = 0.2, stream_dropout: float = 0.0):
        super().__init__()
        self.context = ContextStream(width)
        self.texture = TextureStream(width)
        self.context_head = nn.Linear(self.context.out_dim, 1)
        self.texture_head = nn.Linear(self.texture.out_dim, 1)
        fused = self.context.out_dim + self.texture.out_dim
        self.fusion = nn.Sequential(nn.Dropout(dropout), nn.Linear(fused, 128), nn.ReLU(inplace=True))
        self.live_head = nn.Linear(128, 1)
        self.species_head = nn.Linear(128, num_classes)
        self.stream_dropout = float(stream_dropout)

    def forward(self, context: torch.Tensor, texture: torch.Tensor) -> dict[str, torch.Tensor]:
        fc = self.context(context)
        ft = self.texture(texture)
        logit_c = self.context_head(fc).squeeze(1)
        logit_t = self.texture_head(ft).squeeze(1)
        if self.training and self.stream_dropout > 0:
            # Occasionally hide one stream so the fusion head cannot lean on a single view.
            draw = torch.rand(fc.shape[0], device=fc.device)
            keep_c = (draw > self.stream_dropout).float().unsqueeze(1)
            keep_t = ((draw < 1.0 - self.stream_dropout)).float().unsqueeze(1)
            fc, ft = fc * keep_c, ft * keep_t
        hidden = self.fusion(torch.cat([fc, ft], dim=1))
        return {
            "logit": self.live_head(hidden).squeeze(1),
            "logit_context": logit_c,
            "logit_texture": logit_t,
            "species": self.species_head(hidden),
        }


class ExportWrapper(nn.Module):
    """Flat outputs for ONNX export: three liveness logits and the species logits."""

    def __init__(self, model: PulseGateNet):
        super().__init__()
        self.model = model

    def forward(self, context: torch.Tensor, texture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(context, texture)
        logits = torch.stack([out["logit"], out["logit_context"], out["logit_texture"]], dim=1)
        return logits, out["species"]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
