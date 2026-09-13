# SPDX-License-Identifier: Apache-2.0
"""Host-only unit test for the device feature-shaping permutation matrix.

``build_shaping_perm`` builds ``P[1600, 1616]`` by pushing an index tensor through the
reference ``window_unpartition``. This test checks, in torch on the host (no device),
that ``P @ X`` is *exactly* the reference shaping chain (drop the per-window cls row ->
window_unpartition -> raster [1, 40, 40, C] -> channels-first -> flatten back to
channels-last [1, 1600, C]) for a random ``X``, and that ``P`` is a genuine row selection
(one 1.0 per row, cls columns never selected, every patch row selected exactly once).
"""

from __future__ import annotations

import torch

from rf_detr.reference.configuration_rf_detr import RfDetrBackboneConfig
from rf_detr.reference.modeling_rf_detr import WindowedDinoBackbone
from rf_detr.tt.ttnn_backbone import build_shaping_perm

H = W = 560


def _reference_shaping(wb, hs, height, width):
    """The reference WindowedDinoBackbone.forward tail without the layernorm, returned channels-last."""
    hs = hs[:, 1:]  # drop CLS token (per window)
    hs = wb.window_unpartition(hs, height, width)
    hs = hs.reshape(1, height // wb.cfg.patch_size, width // wb.cfg.patch_size, -1)
    hs = hs.permute(0, 3, 1, 2).contiguous()  # [1, C, 40, 40]
    return hs.flatten(2).transpose(1, 2).contiguous()  # [1, 1600, C] (what TtProjector consumes)


def test_shaping_perm_matches_reference_exactly():
    torch.manual_seed(0)
    cfg = RfDetrBackboneConfig()
    wb = WindowedDinoBackbone(cfg)  # only cfg + window_unpartition are used (no weights needed)
    nw2 = cfg.num_windows ** 2
    seq = (H // cfg.patch_size // cfg.num_windows) * (W // cfg.patch_size // cfg.num_windows) + 1
    C = cfg.hidden_size

    perm, rows = build_shaping_perm(wb, H, W)
    assert perm.shape == (nw2 * (seq - 1), nw2 * seq) == (1600, 1616)
    assert rows.shape == (1600,)

    # P is a row selection: one 1.0 per row, all other entries 0.
    assert torch.equal(perm.sum(dim=1), torch.ones(1600))
    assert set(perm.unique().tolist()) == {0.0, 1.0}
    # cls rows (w * seq) are never selected; every patch row is selected exactly once.
    cls_cols = torch.arange(nw2) * seq
    assert torch.equal(perm[:, cls_cols], torch.zeros(1600, nw2))
    patch_cols = torch.tensor([w * seq + 1 + p for w in range(nw2) for p in range(seq - 1)])
    assert torch.equal(perm[:, patch_cols].sum(dim=0), torch.ones(1600))
    assert torch.equal(perm.argmax(dim=1), rows)

    # P @ X == reference shaping, bit-exact (each output row is exactly one input row).
    hs = torch.randn(nw2, seq, C)
    ref = _reference_shaping(wb, hs, H, W)[0]  # [1600, C]
    got = perm @ hs.reshape(nw2 * seq, C)
    assert torch.equal(got, ref)
    assert torch.equal(hs.reshape(nw2 * seq, C)[rows], ref)

    # Spot-check the raster mapping: raster (i, j) -> window (i//10, j//10), patch (i%10, j%10).
    grid = H // cfg.patch_size
    per_side = grid // cfg.num_windows
    for i, j in ((0, 0), (0, 39), (12, 27), (39, 0), (39, 39)):
        win = (i // per_side) * cfg.num_windows + (j // per_side)
        patch = (i % per_side) * per_side + (j % per_side)
        assert int(rows[i * grid + j]) == win * seq + 1 + patch
