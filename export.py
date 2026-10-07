"""ONNX export of the trained network."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .network import ExportWrapper, PulseGateNet


def export_onnx(model: PulseGateNet, path: str | Path, context_size: int = 112, patch_size: int = 96, check: bool = True) -> Path:
    """Write the model as ONNX with a dynamic batch axis and verify it against PyTorch."""
    path = Path(path)
    wrapper = ExportWrapper(model).eval().to(memory_format=torch.contiguous_format)
    context = torch.randn(2, 3, context_size, context_size)
    texture = torch.randn(2, 3, patch_size, patch_size)
    torch.onnx.export(
        wrapper, (context, texture), str(path), input_names=["context", "texture"],
        output_names=["liveness_logits", "species_logits"], opset_version=17, dynamo=False,
        dynamic_axes={"context": {0: "batch"}, "texture": {0: "batch"}, "liveness_logits": {0: "batch"}, "species_logits": {0: "batch"}},
    )
    if check:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            expected = wrapper(context, texture)
        got = session.run(None, {"context": context.numpy(), "texture": texture.numpy()})
        gap = max(float(np.abs(expected[0].numpy() - got[0]).max()), float(np.abs(expected[1].numpy() - got[1]).max()))
        if gap > 1e-3:
            raise RuntimeError(f"ONNX output differs from PyTorch by {gap:.5f}")
    return path
