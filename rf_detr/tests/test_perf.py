# SPDX-License-Identifier: Apache-2.0
"""Tracy-profileable perf harness for the RF-DETR TT-NN forward.

Tracy captures every ttnn op issued inside ``model(pixel_values)``. Run with::

    python3 -m tracy --no-runtime-analysis --collect-noc-traces \
        --profiler-capture-perf-counters=all --op-support-count=10000 \
        -v -r -o ./tracy_out -m pytest rf_detr/tests/test_perf.py

After one dispatch (which captures the projector+transformer metal-trace and
populates the program cache) we run a steady-state captured pass.
"""

from __future__ import annotations

import os

import torch

from rf_detr.reference.weights import get_preprocessor, load_rf_detr_base

_IMAGE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cats_000000039769.jpg")


def test_rf_detr_forward_perf(device):
    import ttnn

    from rf_detr.tt.ttnn_rf_detr import TtRfDetr

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    ref, cfg = load_rf_detr_base()
    ref = ref.eval()
    pre = get_preprocessor(cfg)
    if os.path.isfile(_IMAGE):
        from PIL import Image

        pixel_values = pre(Image.open(_IMAGE).convert("RGB")).float()
    else:
        pixel_values = torch.randn(1, 3, cfg.image_resolution, cfg.image_resolution)

    model = TtRfDetr(ref, device)

    # Warm-up pass: capture the trace + populate program cache.
    _ = model(pixel_values)

    if hasattr(ttnn, "synchronize_device"):
        ttnn.synchronize_device(device)
    out = model(pixel_values)
    if hasattr(ttnn, "synchronize_device"):
        ttnn.synchronize_device(device)

    assert out.logits.shape == (1, cfg.num_queries, cfg.num_labels)
    assert out.pred_boxes.shape == (1, cfg.num_queries, 4)
