# rf-detr-p150 — Blackhole p150a vs RTX 5090 (same host, same weights, same input)

Date 2026-09-14. Facts only; every number below is measured in this pass on the GPU or copied
(with its source line) from the p150a validation reports. The p150a was NOT touched.

## What was run

| | |
|---|---|
| Model | RF-DETR-base, the port's own torch reference `models/rf-detr-p150/code/rf_detr/reference/modeling_rf_detr.py::RfDetrForObjectDetection` (strict load, 487 tensors) — the network the p150a port was PCC-gated against |
| Weights | `Roboflow/rf-detr-base` @ `7b95b089788e6c7db56d5ea9b0a07ca08ea6ac0a` (tt-model.yaml `weights.revision` = `serve.env.TT_WEIGHTS_REVISION`), from the HF cache `~/.cache/huggingface/hub/models--Roboflow--rf-detr-base/snapshots/7b95b089…/{model.safetensors,config.json}`, `HF_HUB_OFFLINE=1` |
| Input | `media/demo_source.png` (640x480 cats) -> `rf_detr.reference.weights.get_preprocessor(cfg)`: bilinear antialias squash to 560x560, x1/255, ImageNet mean/std -> `pixel_values [1,3,560,560]` fp32, batch 1. Identical to `server/app.py` `STATE["pre"]` |
| GPU | NVIDIA GeForce RTX 5090 (sm_120), driver 580.126.18, power limit 600 W, 32607 MiB; idle 29.9 W |
| venv | `/home/deepgadget/experiments/tt-models/.venv-gpu/main` — Python 3.12.13, torch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19, torchvision 0.26.0, numpy 1.26.4, safetensors 0.8.0, huggingface_hub 1.31.0, pillow 12.3.0, triton 3.6.0 |
| Script | `logs/gpu-vs-p150/rf-detr/bench_rf_detr_gpu.py` (uses `logs/gpu-vs-p150/bench_common.py`); log `logs/gpu-vs-p150/rf-detr/full_run.log`; raw JSON `reports/gpu-vs-p150/rf-detr.json` (= `logs/gpu-vs-p150/rf-detr/result.json`); CPU reference tensors `logs/gpu-vs-p150/rf-detr/cpu_fp32_reference.pt` |
| Command | `HF_HUB_OFFLINE=1 /home/deepgadget/experiments/tt-models/.venv-gpu/main/bin/python bench_rf_detr_gpu.py --iters 50 --warmup 10` |
| Loop | per precision: 10 warm-ups + 50 timed iterations, `torch.cuda.synchronize()` before/after each; wall-clock (perf_counter) is the primary number, CUDA-event time recorded alongside (within 0.02 ms of wall) |
| p150a source | `reports/gpu-vs-p150/p150_numbers.json` -> `reports/megakernel/STATUS.md:3` (Hub `tt serve`, 8.24 ms) and `logs/opt-rf-detr/FINAL_REPORT2.md:20,28-31` (stage split, served total). `DEVICE_VALIDATION.md` does not exist in `models/rf-detr-p150` (tt-model-package branch); the FINAL_REPORT2 split is the per-stage source |

Timing definitions (they match the p150a `timing_ms` keys):

- **incl_h2d** = `pixel_values.to("cuda")` + forward + `logits.cpu()` + `pred_boxes.cpu()`, pageable host tensors (what the server preprocess produces). Compare with p150a `timing_ms.inference` = host im2col + from_torch + upload + trace replay + 2 readbacks (**8.24 ms**, STATUS.md:3; trace-only device time 7.44 ms, FR2:28).
- **excl_h2d** = forward only, input already resident, outputs left on the device.
- **served-like** = base64 decode + PNG decode (PIL) + reference preprocess + incl_h2d forward + the server's `_postprocess` (copied verbatim; fastapi is not in the venv), threshold 0.5. Compare with p150a `timing_ms.total` (**15.43 ms**, FR2:31 dev-image served run; the Hub run only recorded the inference median).

## Correctness check (GPU vs CPU fp32 reference)

CPU fp32 single forward: 271 ms. GPU fp32 strict (no TF32) on the same `pixel_values`:

| metric | value |
|---|---:|
| raw PCC logits `[1,300,91]` / pred_boxes `[1,300,4]` (query order) | **1.000000 / 1.000000** |
| max abs diff logits / boxes | 5.0e-4 / 2.0e-4 |
| same top-300 proposals in the same order | yes (300/300) |
| detection-IoU agreement (`rf_detr.benchmark.detection_accuracy`, the p150a gate >= 98.5) | **100.00** |
| detections @0.5 | identical to CPU: cat 0.9603 [7.35,54.64,318.45,472.14], cat 0.9335, remote 0.8976, remote 0.7280, couch 0.6728 |

PCC > 0.999 holds for fp32; the GPU runs the right model.

Per-precision accuracy vs the CPU fp32 reference (same image). RF-DETR's two-stage top-k permutes the
300 queries, so the raw element-wise PCC collapses as soon as the proposal ranking changes; the
order-invariant detection-IoU agreement (the metric the p150a was gated on, 98.70 on the same image) is the
one to read. "query-matched" pairs queries by their `init_reference_points` (< 1e-3 apart) and gives the PCC
over those pairs; for bf16 the reference points themselves move by more than 1e-3, so only 1 query pairs.

| GPU precision | raw PCC logits / boxes | queries matched | matched PCC logits / boxes | fg-matched PCC logits / boxes | detection-IoU agreement | detections @0.5 |
|---|---:|---:|---:|---:|---:|---|
| fp32 strict | 1.000000 / 1.000000 | 300/300, same order | 1.0 / 1.0 | 1.0 / 1.0 (5 fg) | **100.00** | same 5 |
| tf32 | 0.897770 / 0.700856 | 203/300 | 0.992702 / 0.970612 | 0.999972 / 1.000000 (5 fg) | **99.98** | same 5 labels; scores within 0.008 |
| bf16 autocast | 0.661301 / 0.193554 | 1/300 (bf16 ref-points drift > 1e-3) | n/a | n/a | **99.78** | same 5 labels; scores within 0.015 |
| fp16 autocast | 0.796526 / 0.468419 | 94/300 | 0.987154 / 0.942414 | 0.999983 / 1.000000 (4 fg) | **99.95** | same 5 labels |
| p150a (bf16 device, FR2:21-24) | tail PCC fg-matched boxes 0.999969 / logits 0.998028 | | | | **98.70** | same 5 labels, conf_dev <= 0.0076 |

All GPU precisions pass the port's 98.5 gate; fp16 autocast is numerically fine here (no overflow, same 5 detections).

## GPU latency (batch 1, 560x560, median / min / p90 of 50 iterations, wall-clock ms)

Eager PyTorch:

| precision | incl_h2d median / min / p90 | excl_h2d median / min / p90 | CUDA-event excl | first call ms | power mean W (excl loop) | power mean W (incl loop) | peak mem alloc / reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 strict (`allow_tf32=False`, `'highest'`) | **7.985** / 7.935 / 8.027 | **7.730** / 7.678 / 7.763 | 7.717 | 7.7 (321.6 incl. cuDNN init on the very first call) | 363.8 | 245.5 | 300 / 318 |
| tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | **5.261** / 5.228 / 5.306 | **4.999** / 4.971 / 5.028 | 4.986 | 5.1 | 323.4 | 339.8 | 300 / 318 |
| bf16 autocast (+TF32 remainder) | **5.928** / 5.889 / 6.009 | **5.709** / 5.660 / 5.794 | 5.695 | 6.6 | 262.5 | 287.1 | 327 / 378 |
| fp16 autocast (+TF32 remainder) | **5.908** / 5.870 / 5.945 | **5.660** / 5.629 / 5.700 | 5.648 | 6.2 | 259.9 | 256.1 | 268 / 378 |
| tf32, pinned host input (informational) | 5.205 / 5.178 / 5.252 | | | | | | |
| bf16 autocast, pinned host input (informational) | 5.889 / 5.845 / 5.990 | | | | | | |

`torch.compile` (inductor, `dynamic=False`; compile time well under the 5-min budget), same accuracy class as the eager precision:

| variant | compile s | incl_h2d median / min / p90 | excl_h2d median / min / p90 | power W (excl loop) | peak mem MiB | detection-IoU |
|---|---:|---:|---:|---:|---:|---:|
| tf32 + compile default | 14.4 | 3.117 / 3.105 / 3.138 | 2.865 / 2.849 / 2.872 | 361.5 | 174 | 99.98 |
| tf32 + compile reduce-overhead (CUDA graphs) | 10.1 | **2.741** / 2.633 / 2.836 | **2.385** / 2.375 / 2.402 | 393.2 | 131 | 99.98 |
| fp16 autocast + compile default | 15.1 | 3.159 / 3.114 / 3.223 | 2.936 / 2.887 / 2.974 | 277.7 | 164 | 99.95 |
| fp16 autocast + compile reduce-overhead (CUDA graphs) | 11.6 | **2.223** / 2.203 / 2.238 | **1.789** / 1.778 / 1.795 | 355.5 | 131 | 99.95 |

Other facts: model load (safetensors -> cuda) 0.14 s; first fp32 call 322 ms (includes CUDA/cuDNN
initialisation); idle GPU power 29.9 W; GPU utilisation during the eager excl loops 60-84 % — at batch 1 the
eager graph is launch-bound (bf16 autocast is *slower* than tf32 because of the extra cast kernels), which is
why CUDA graphs (`reduce-overhead`) give the largest gain.

Served-like loop (same host work as `server/app.py::predict`, 50 iterations, medians ms):

| GPU precision | decode (base64 + PNG) | preprocess | inference (incl_h2d) | postprocess | **total** | p90 total | detections |
|---|---:|---:|---:|---:|---:|---:|---|
| fp32 strict | 5.27 | 1.01 | 8.06 | 0.33 | **14.72** | 16.51 | same 5 as CPU |
| tf32 | 5.30 | 0.71 | 5.35 | 0.12 | **11.58** | 12.04 | same 5 labels |
| bf16 autocast | 5.63 | 0.98 | 6.50 | 0.31 | **13.30** | 14.08 | same 5 labels |
| p150a (FR2:31) | 5.75 | ~0.9 | 8.61 | ~0.2 | **15.43** | | same 5 labels |

The host stages are the same code on both sides (PIL PNG decode ~5.3-5.8 ms dominates), so the served
totals differ almost only by the inference term.

## Comparison with the p150a (matching definitions)

Ratio = p150a ms / GPU ms (> 1 means the GPU is faster). p150a precision: bf16 activations on device
(backbone HiFi2/HiFi4 fused ops, decoder HiFi4 + fp32 accumulate), one metal trace.

| row | p150a (definition) | GPU precision | GPU ms | ratio p150a/GPU |
|---|---:|---|---:|---:|
| device forward (p150a `timing_ms.inference` incl. upload + readback vs GPU incl_h2d) | 8.24 (STATUS.md:3, Hub tt serve, 50 warm) | fp32 strict | 7.985 | **1.03** |
| | | tf32 | 5.261 | **1.57** |
| | | bf16 autocast | 5.928 | **1.39** |
| | | fp16 autocast | 5.908 | **1.39** |
| | | tf32 + compile reduce-overhead | 2.741 | **3.01** |
| | | fp16 autocast + compile reduce-overhead | 2.223 | **3.71** |
| GPU forward only (excl_h2d) vs p150a `timing_ms.inference` 8.24 | 8.24 | fp32 strict / tf32 / bf16 / fp16 | 7.730 / 4.999 / 5.709 / 5.660 | 1.07 / 1.65 / 1.44 / 1.46 |
| GPU forward only (excl_h2d) vs p150a trace-only device time 7.44 (FR2:28) | 7.44 | fp32 strict / tf32 / bf16 / fp16 | 7.730 / 4.999 / 5.709 / 5.660 | 0.96 / 1.49 / 1.30 / 1.31 |
| | | tf32 + compile reduce-overhead / fp16 + compile reduce-overhead | 2.385 / 1.789 | 3.12 / 4.16 |
| served e2e (p150a `timing_ms.total` vs GPU served-like total) | 15.43 (FR2:31, dev-image run; Hub-run total not recorded, 15.43/15.59 bracket it) | fp32 strict | 14.72 | **1.05** |
| | | tf32 | 11.58 | **1.33** |
| | | bf16 autocast | 13.30 | **1.16** |

Reading: with the p150a's own precision class (bf16 activations) the eager RTX 5090 forward is 1.4x faster
than the p150a's fused trace (5.9 vs 8.24 ms incl. transfers); in strict fp32 the two are within 3 %
(7.99 vs 8.24 ms), and the GPU's eager fp32 forward-only time (7.73 ms) is slightly slower than the p150a's
trace-only device time (7.44 ms). The best GPU configuration measured (fp16 autocast + inductor + CUDA graphs)
is 3.7x faster than the p150a device forward (2.22 vs 8.24 ms) with detection-IoU 99.95 vs 98.70 for the p150a.
End-to-end, the shared ~6-7 ms of host PNG decode / pre / post work compresses the gap to 1.05-1.33x.

Not measured / not claimed: p150a power (not measured in any pass -> no power or efficiency comparison;
the GPU drew 260-394 W mean during the dense loops, 30 W idle). p150a numbers were not re-measured.
The GPU numbers exclude HTTP/JSON framing, as do the p150a `timing_ms` keys.

## Reproduce

```bash
cd /home/deepgadget/experiments/tt-models/logs/gpu-vs-p150/rf-detr
HF_HUB_OFFLINE=1 /home/deepgadget/experiments/tt-models/.venv-gpu/main/bin/python bench_rf_detr_gpu.py --iters 50 --warmup 10 | tee full_run.log
# outputs: /home/deepgadget/experiments/tt-models/reports/gpu-vs-p150/rf-detr.json, ./result.json, ./cpu_fp32_reference.pt
```

GPU released after the run: `nvidia-smi --query-compute-apps=pid --format=csv,noheader` -> empty.
