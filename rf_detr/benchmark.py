# SPDX-License-Identifier: Apache-2.0
"""RF-DETR inference benchmark harness.

Emits results in a grep-parseable format so an experiment loop can pick them up:

    inference_speed: <frames/sec>   (median over N warm iterations, device-synchronized)
    accuracy:        <percent>      (order-invariant detection-IoU vs the fp32 reference)
    peak_dram:       <MiB>          (best-effort device DRAM peak)

``--impl ttnn`` (default) runs the on-device ``TtRfDetr`` and scores it against the
torch reference. ``--impl torch`` times the pure CPU reference (the optimization
baseline; accuracy is 100 by definition and peak_dram is 0).

The accuracy number is a *detection-level* agreement, not a raw per-tensor PCC:
RF-DETR's two-stage top-k permutes the 300 queries, so element-wise PCC of the
logits/boxes is meaningless. We instead match each confident reference detection
to the best same-label IoU among the tt detections (see ``detection_accuracy``).

Usage:
    python -m rf_detr.benchmark --impl ttnn  --device-id 0 --iters 20
    python -m rf_detr.benchmark --impl torch --iters 10
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

from rf_detr.reference.weights import get_preprocessor, load_rf_detr_base

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_IMAGE = os.path.join(_REPO_ROOT, "data", "cats_000000039769.jpg")


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two tensors (flattened)."""
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item() + 1e-12
    return float((a @ b).item() / denom)


def _cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def _iou(a, b):
    """a: [N,4] xyxy, b: [M,4] xyxy -> [N,M] IoU."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def detection_accuracy(ref_logits, ref_boxes, tt_logits, tt_boxes, score_thresh=0.25, min_k=5):
    """Order-invariant detection agreement vs the fp32 reference (the ground truth).

    For each *confident* reference detection (score > thresh), find the best same-label
    IoU among the ttnn detections; accuracy = 100 * mean(best IoU). 100 == identical
    detections; a label miss or box drift lowers it. Robust to the discrete query
    permutation. Low-confidence (~noise) predictions are excluded — they are not
    reported detections and must not gate the optimization loop.
    """
    ref_s, ref_l = ref_logits.sigmoid()[0].max(-1)
    keep = ref_s > score_thresh
    if int(keep.sum()) < min_k:  # only pad when almost nothing is confident
        keep = torch.zeros_like(ref_s, dtype=torch.bool)
        keep[ref_s.topk(min(min_k, ref_s.numel())).indices] = True
    ref_b = _cxcywh_to_xyxy(ref_boxes[0][keep])
    ref_lab = ref_l[keep]
    tt_s, tt_l = tt_logits.sigmoid()[0].max(-1)
    tt_b = _cxcywh_to_xyxy(tt_boxes[0])
    ious = []
    for i in range(ref_b.shape[0]):
        same = tt_l == ref_lab[i]
        if bool(same.any()):
            ious.append(float(_iou(ref_b[i : i + 1], tt_b[same]).max()))
        else:
            ious.append(0.0)
    return 100.0 * (sum(ious) / max(len(ious), 1))


def _peak_dram_mib(device) -> float:
    import ttnn

    try:
        mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
        return mv.total_bytes_allocated_per_bank * mv.num_banks / (1024 * 1024)
    except Exception:
        return -1.0


def _load_pixel_values(image_path: str, pre) -> torch.Tensor:
    try:
        from PIL import Image

        pixel_values = pre(Image.open(image_path).convert("RGB"))
    except Exception:
        pixel_values = torch.randn(1, 3, 560, 560)
    if pixel_values.dim() == 3:
        pixel_values = pixel_values.unsqueeze(0)
    return pixel_values.float()


def run_torch(args) -> dict:
    torch.manual_seed(0)
    ref, cfg = load_rf_detr_base()
    ref = ref.eval()
    pre = get_preprocessor(cfg)
    pixel_values = _load_pixel_values(args.image, pre)

    with torch.no_grad():
        for _ in range(max(1, args.warmup)):
            ref(pixel_values, collect_intermediates=False)
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            ref(pixel_values, collect_intermediates=False)
            times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return {"inference_speed": 1.0 / med if med > 0 else 0.0, "accuracy": 100.0,
            "peak_dram": 0.0, "median_latency_ms": med * 1000.0}


def run_ttnn(args) -> dict:
    import ttnn

    from rf_detr.tt.ttnn_rf_detr import TtRfDetr

    torch.manual_seed(0)
    ref, cfg = load_rf_detr_base()
    ref = ref.eval()
    pre = get_preprocessor(cfg)
    pixel_values = _load_pixel_values(args.image, pre)

    with torch.no_grad():
        golden = ref(pixel_values, collect_intermediates=False)

    device_params = dict(l1_small_size=32768, trace_region_size=90_000_000, num_command_queues=1)
    device = ttnn.open_device(device_id=args.device_id, **device_params)
    try:
        model = TtRfDetr(ref, device)

        # ---- accuracy (detection-level, order-invariant vs fp32 reference) ----
        out = model(pixel_values)
        pcc_logits = pcc(golden.logits, out.logits)
        pcc_boxes = pcc(golden.pred_boxes, out.pred_boxes)
        accuracy = detection_accuracy(golden.logits, golden.pred_boxes, out.logits, out.pred_boxes)

        # ---- speed (warm, device-synchronized, median) ----
        for _ in range(max(1, args.warmup)):
            model(pixel_values)
            ttnn.synchronize_device(device)
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            model(pixel_values)
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        peak_dram = _peak_dram_mib(device)
    finally:
        ttnn.close_device(device)

    print(f"pcc_logits (raw, query-permuted): {pcc_logits:.6f}")
    print(f"pcc_pred_boxes (raw, query-permuted): {pcc_boxes:.6f}")
    return {"inference_speed": 1.0 / med if med > 0 else 0.0, "accuracy": accuracy,
            "peak_dram": peak_dram, "median_latency_ms": med * 1000.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["torch", "ttnn"], default="ttnn")
    ap.add_argument("--device-id", type=int, default=int(os.environ.get("RF_DETR_DEVICE", "0")))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--image", default=_DEFAULT_IMAGE)
    args = ap.parse_args(argv)

    torch.set_grad_enabled(False)
    res = run_torch(args) if args.impl == "torch" else run_ttnn(args)

    print(f"impl: {args.impl}")
    print(f"median_latency_ms: {res['median_latency_ms']:.3f}")
    print(f"inference_speed: {res['inference_speed']:.4f}")
    print(f"accuracy: {res['accuracy']:.4f}")
    print(f"peak_dram: {res['peak_dram']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
