# SPDX-License-Identifier: Apache-2.0
"""Pretrained-weight end-to-end evaluation: torch reference vs TT-NN on a real image.

Loads the published ``Roboflow/rf-detr-base`` checkpoint into the reference model,
builds ``TtRfDetr`` from it, and runs both on the canonical COCO image (two cats +
two remotes on a pink couch). Asserts:

  * the on-device model's detections agree with the fp32 reference at
    detection-IoU >= ``ACC_GATE`` (order-invariant, see ``detection_accuracy``);
  * the on-device model itself detects >= 2 cats and >= 1 remote above conf 0.5
    (i.e. the port reproduces the real objects, not just matches the reference).
"""

from __future__ import annotations

import os

import pytest
import torch

from rf_detr.benchmark import detection_accuracy
from rf_detr.reference.weights import get_preprocessor, load_rf_detr_base

ACC_GATE = 98.5  # bf16 caps small-box IoU just under 99; see README "Known caveats"
CONF = 0.5
_IMAGE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cats_000000039769.jpg")


def _detections(logits, boxes, id2label, conf=CONF):
    prob = logits.sigmoid()[0]
    scores, labels = prob.max(-1)
    keep = scores > conf
    return [(id2label.get(int(labels[i]), str(int(labels[i]))), float(scores[i]))
            for i in range(scores.shape[0]) if keep[i]]


def test_pretrained_torch_vs_tt(device, capsys):
    from PIL import Image

    from rf_detr.tt.ttnn_rf_detr import TtRfDetr

    if not os.path.isfile(_IMAGE):
        pytest.skip(f"test image missing: {_IMAGE} (run scripts/download_weights.sh)")

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    ref, cfg = load_rf_detr_base()
    ref = ref.eval()
    pre = get_preprocessor(cfg)
    pixel_values = pre(Image.open(_IMAGE).convert("RGB")).float()

    with torch.no_grad():
        golden = ref(pixel_values, collect_intermediates=False)

    model = TtRfDetr(ref, device)
    _ = model(pixel_values)  # warm up (capture trace / program cache)
    out = model(pixel_values)

    acc = detection_accuracy(golden.logits, golden.pred_boxes, out.logits, out.pred_boxes)
    tt_dets = _detections(out.logits, out.pred_boxes, cfg.id2label or {})
    n_cat = sum(1 for name, _ in tt_dets if name == "cat")
    n_remote = sum(1 for name, _ in tt_dets if name == "remote")

    with capsys.disabled():
        print(f"\n=== {os.path.basename(_IMAGE)} ===")
        print(f"  detection-IoU vs reference : {acc:.3f}  (gate >= {ACC_GATE})")
        print(f"  tt detections (>{CONF}): cat={n_cat}, remote={n_remote}")
        for name, score in sorted(tt_dets, key=lambda x: -x[1]):
            print(f"    {name:12s} {score:.3f}")

    assert acc >= ACC_GATE, f"detection-IoU {acc:.3f} < gate {ACC_GATE}"
    assert n_cat >= 2, f"expected >= 2 cats from tt model, got {n_cat}"
    assert n_remote >= 1, f"expected >= 1 remote from tt model, got {n_remote}"
