# SPDX-License-Identifier: Apache-2.0
"""Generate media/demo_source.png + media/demo_detections.png demo artifacts.

Runs the on-device ``TtRfDetr`` on an image and draws every detection above a
confidence threshold (box + COCO label + score) over the input. The source image
is also saved verbatim so the README can show an input/output pair.

Run from the repo root with::

    PYTHONPATH=$PWD:$TT_METAL_HOME:$TT_METAL_HOME/ttnn \
    RF_DETR_DEVICE=<n> \
    python -m scripts.make_demo [--image path/to/image.jpg]

Defaults to the canonical COCO image in ./data. Requires matplotlib.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MEDIA = _REPO_ROOT / "media"
_DEFAULT_IMAGE = _REPO_ROOT / "data" / "cats_000000039769.jpg"

# A readable, repeatable palette cycled across detections.
_COLORS = ["lime", "cyan", "magenta", "orange", "yellow", "red", "deepskyblue", "springgreen"]


def _cxcywh_to_xyxy_pixels(box, W, H):
    cx, cy, w, h = box
    return [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]


def visualize(img_pil, detections, out_path: Path) -> None:
    """detections: list of (label_name, score, [cx,cy,w,h] normalized)."""
    W, H = img_pil.size
    fig, ax = plt.subplots(1, 1, figsize=(12, 12 * H / W))
    ax.imshow(img_pil)
    for i, (name, score, box) in enumerate(detections):
        color = _COLORS[i % len(_COLORS)]
        x0, y0, x1, y1 = _cxcywh_to_xyxy_pixels(box, W, H)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, edgecolor=color, linewidth=3))
        ax.text(x0, max(0, y0 - 6), f"{name} {score:.2f}", color="black", fontsize=12,
                bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor="none"))
    ax.set_title(f"RF-DETR on Blackhole p150a — {len(detections)} detections", fontsize=14)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def main() -> None:
    import ttnn

    from rf_detr.reference.weights import get_preprocessor, load_rf_detr_base
    from rf_detr.tt.ttnn_rf_detr import TtRfDetr

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(_DEFAULT_IMAGE))
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--device-id", type=int, default=int(os.environ.get("RF_DETR_DEVICE", "0")))
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    ref, cfg = load_rf_detr_base()
    ref = ref.eval()
    pre = get_preprocessor(cfg)
    id2label = cfg.id2label or {}

    img_pil = Image.open(args.image).convert("RGB")
    pixel_values = pre(img_pil).float()

    _MEDIA.mkdir(parents=True, exist_ok=True)
    img_pil.save(_MEDIA / "demo_source.png")

    device_params = dict(l1_small_size=32768, trace_region_size=90_000_000, num_command_queues=1)
    device = ttnn.open_device(device_id=args.device_id, **device_params)
    try:
        model = TtRfDetr(ref, device)
        _ = model(pixel_values)  # warm up (capture trace)
        out = model(pixel_values)
    finally:
        ttnn.close_device(device)

    prob = out.logits.sigmoid()[0]
    scores, labels = prob.max(-1)
    keep = scores > args.conf
    detections = [
        (id2label.get(int(labels[i]), str(int(labels[i]))), float(scores[i]), out.pred_boxes[0][i].tolist())
        for i in range(scores.shape[0]) if keep[i]
    ]
    detections.sort(key=lambda d: -d[1])

    out_path = _MEDIA / "demo_detections.png"
    visualize(img_pil, detections, out_path)
    print(f"  {Path(args.image).name}  ({img_pil.size[0]}x{img_pil.size[1]})  "
          f"{len(detections)} detections (>{args.conf})  ->  {out_path.name}")
    for name, score, _ in detections:
        print(f"    {name:12s} {score:.3f}")


if __name__ == "__main__":
    main()
