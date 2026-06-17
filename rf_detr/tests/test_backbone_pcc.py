# SPDX-License-Identifier: Apache-2.0
"""Per-stage PCC test for the windowed-DINOv2 backbone (tt vs torch reference).

Runs the reference backbone to get golden per-layer hidden states (the windowed
[16, 101, 384] tensors after the out-stage layers 1/4/7/10) and the 4 reshaped
feature maps [1, 384, 40, 40], then runs ``TtDinoBackbone`` and asserts each
stage matches the reference within ``PCC_TARGET``.

Uses the real preprocessed image when the validation oracle is present, else a
fixed random input (PCC is relative, so either works).
"""

from __future__ import annotations

import os

import pytest
import torch

from rf_detr.reference.weights import load_rf_detr_base

PCC_TARGET = 0.99
_OUT_STAGES = (1, 4, 7, 10)
_ORACLE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "reference_outputs.pt"
)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item() + 1e-12
    return float((a @ b).item() / denom)


def test_backbone_pcc(device, capsys):
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    ref, _cfg = load_rf_detr_base()
    ref = ref.eval()
    wb = ref.backbone[0].encoder.encoder

    if os.path.isfile(_ORACLE):
        pixel_values = torch.load(_ORACLE, map_location="cpu")["pixel_values"].float()
    else:
        pixel_values = torch.randn(1, 3, 560, 560)

    # ---- golden: reference feature maps + per-layer hidden states ----
    with torch.no_grad():
        golden_feats = wb(pixel_values)  # 4 x [1,384,40,40]
        embed = wb.embeddings(pixel_values)
        golden_hidden = {}
        h = embed
        for i, layer in enumerate(wb.encoder.layer):
            h = layer(h)
            if i in _OUT_STAGES:
                golden_hidden[i] = h.clone()

    from rf_detr.tt.ttnn_backbone import TtDinoBackbone

    tt = TtDinoBackbone(ref, device)
    tt_hidden = tt.run_layers(embed)  # dict idx -> torch [16,101,384]
    tt_feats = tt.feature_maps(pixel_values)

    failures = []
    with capsys.disabled():
        print("\n=== per-layer hidden-state PCC (windowed [16,101,384]) ===")
        for i in _OUT_STAGES:
            p = _pcc(golden_hidden[i], tt_hidden[i])
            print(f"  after layer {i:2d}: PCC={p:.5f}  {'OK' if p >= PCC_TARGET else 'FAIL'}")
            if p < PCC_TARGET:
                failures.append(f"layer {i}: {p:.5f} < {PCC_TARGET}")
        print("=== feature-map PCC ([1,384,40,40]) ===")
        for j, (g, t) in enumerate(zip(golden_feats, tt_feats)):
            p = _pcc(g, t)
            stage = (2, 5, 8, 11)[j]
            print(f"  feature map {j} (stage {stage}): PCC={p:.5f}  {'OK' if p >= PCC_TARGET else 'FAIL'}")
            if p < PCC_TARGET:
                failures.append(f"feature map {j}: {p:.5f} < {PCC_TARGET}")

    assert not failures, "Backbone PCC failures:\n  " + "\n  ".join(failures)
