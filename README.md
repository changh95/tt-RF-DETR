# tt-RF-DETR

End-to-end port of [RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow) to
Tenstorrent **tt-metal** (tt-nn + tt-metallium), running on a single Blackhole
p150a chip.

This repository contains a TT-NN implementation of the `Roboflow/rf-detr-base`
checkpoint, a faithful torch reference used as a numerical shadow, pytest suites
for per-stage PCC and real-image detection correctness, a benchmark that reports
inference speed / accuracy / peak DRAM, a Tracy-profileable perf test, and a
script to pull the published weights.

The TT-NN forward runs the **whole detector on the chip** — the windowed
DINOv2-S/14 backbone, the C2f projector, the two-stage deformable transformer,
and the detection heads. The only host work left is the backbone *embeddings*
(patch conv + window partition) and *feature-map shaping* (LayerNorm + window
unpartition), which are reshape-heavy and not tile-friendly; everything with FLOPs
runs on device.

---

## Demo

RF-DETR-base run on a single Blackhole p150a. Each colored box is a detection
above the 0.5 confidence threshold (label + score); boxes are the model's
`cxcywh` outputs scaled back to image space. Reproduce with
`python -m scripts.make_demo` (defaults to the shipped COCO image; pass
`--image <path>` for your own).

| Input (`media/demo_source.png`) | Detections on p150a (`media/demo_detections.png`) |
|:---:|:---:|
| ![](media/demo_source.png) | ![](media/demo_detections.png) |

---

## Contents

```
rf_detr/
├── reference/
│   ├── configuration_rf_detr.py  # self-contained config (no transformers dep)
│   ├── modeling_rf_detr.py       # faithful torch reference; keys match the checkpoint 1:1
│   ├── weights.py                # Roboflow/rf-detr-base (HF Hub) → strict 487-key load
│   └── validate_reference.py     # acceptance gate + oracle-tensor dump
├── tt/
│   ├── ttnn_backbone.py          # windowed DINOv2-S/14 (12 layers) on device
│   ├── ttnn_projector.py         # C2f projector (1x1 as linear, 3x3 via ttnn.conv2d)
│   ├── ttnn_transformer.py       # two-stage select + 3 deformable decoder layers + heads
│   └── ttnn_rf_detr.py           # end-to-end forward (+ metal-trace of the tail)
├── benchmark.py                  # FPS + detection-IoU accuracy + peak DRAM (grep-parseable)
└── tests/
    ├── test_backbone_pcc.py      # per-stage backbone PCC vs torch reference
    ├── test_transformer_pcc.py   # Hungarian-matched transformer-tail PCC
    ├── test_pretrained_eval.py   # real-image end-to-end detection (torch ↔ tt)
    └── test_perf.py              # single-iter forward, Tracy-profileable
scripts/
├── download_weights.sh           # pulls the checkpoint + the canonical COCO image
└── make_demo.py                  # draws detections over an image
conftest.py                       # minimal pytest `device` fixture (l1 + trace region)
```

The code imports `ttnn` and uses the tt-metal / tt-nn build you point it at —
this repo does **not** contain the tt-metal monorepo. See **Environment setup**.

---

## Environment setup

1. Build tt-metal with its Python bindings (and Tracy support if you plan to run
   `test_perf.py`). Instructions: https://github.com/tenstorrent/tt-metal.
2. Install the Python deps for this repo. A working set:

   ```bash
   pip install torch torchvision safetensors huggingface_hub pillow numpy scipy matplotlib pytest
   ```

   `torch` can be CPU-only; the reference is only used as a numerical shadow.
   `scipy` is used for the Hungarian match in the transformer PCC test;
   `matplotlib` only for `scripts/make_demo.py`.
3. Point Python at tt-metal's `ttnn` bindings and set the env the runtime expects:

   ```bash
   export TT_METAL_HOME=/path/to/tt-metal
   export PYTHONPATH=$PWD:$TT_METAL_HOME:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools
   # Restrict visibility to the one Blackhole chip you want to use (optional):
   export TT_VISIBLE_DEVICES=0
   # Which chip the tests/benchmark open (default 0):
   export RF_DETR_DEVICE=0
   ```

4. Download the weights + sample image:

   ```bash
   bash scripts/download_weights.sh
   ```

   Populates `./weights/` (`model.safetensors` + `config.json` from
   `Roboflow/rf-detr-base`) and `./data/` (the COCO `000000039769` image). The
   loader also resolves the checkpoint straight from the HF Hub cache if you skip
   this step; override the local snapshot dir with `TT_RF_DETR_WEIGHTS`.

---

## Running the tests

```bash
# 0) Validate the torch reference (strict 487-key load + cat/remote detections)
python -m rf_detr.reference.validate_reference

# 1) Backbone per-stage PCC (tt vs torch reference)
pytest rf_detr/tests/test_backbone_pcc.py -v -s --device-id 0

# 2) Transformer-tail PCC (Hungarian-matched; two-stage top-k permutes queries)
pytest rf_detr/tests/test_transformer_pcc.py -v -s --device-id 0

# 3) Real-image end-to-end detection (torch reference ↔ TT-NN)
pytest rf_detr/tests/test_pretrained_eval.py -v -s --device-id 0

# 4) Wall-clock FPS / accuracy / peak-DRAM benchmark
python -m rf_detr.benchmark --impl ttnn --device-id 0 --iters 20

# 5) Tracy-profiled single-iter forward (requires Tracy-enabled tt-metal build)
python -m tracy --no-runtime-analysis --collect-noc-traces \
    --profiler-capture-perf-counters=all --op-support-count=10000 \
    -v -r -o ./tracy_out -m pytest rf_detr/tests/test_perf.py
```

---

## Results

### Per-stage PCC (tt vs torch reference)

Each stage of the TT-NN forward matches the torch reference within tight bounds.
The backbone and projector are compared element-wise. The transformer tail needs
care: RF-DETR's two-stage top-k selects 300 of 1600 proposals, and the
rank-299/300 selection gap (~3e-4) is well below the bf16 ULP, so a couple of
boundary queries reshuffle — a *raw* element-wise PCC of the 300-query
logits/boxes is meaningless (~0.16 boxes / ~0.65 logits). We first realign the
queries with a **Hungarian match** (box-L1 + class-logit-L1). Even after the
match, the *all-300* PCC (~0.96 boxes / ~0.94 logits on this image) is dominated
by the ~295 low-confidence background queries, whose pairing is arbitrary on a
sparse scene. The number that actually measures port fidelity is the matched PCC
over the **confident (foreground) queries** — the real detections — which is
essentially exact (this is what `test_transformer_pcc.py` gates on).

| Stage | PCC | Shape |
|---|---:|---|
| backbone feature maps (stages 2/5/8/11)         | 0.998–0.9998 | (1, 384, 40, 40) ×4 |
| projector output (C2f)                          | 0.9999       | (1, 256, 40, 40) |
| transformer — foreground-matched `pred_boxes`   | ~1.000       | (n_fg, 4) |
| transformer — foreground-matched `logits`       | ~0.997       | (n_fg, 91) |
| transformer — all-300 matched (context only)    | 0.96 / 0.94  | (300, 4) / (300, 91) |

### Real-data evaluation

Accuracy is reported as an **order-invariant detection-IoU agreement** against
the fp32 torch reference (the ground truth for the port): for each confident
reference detection, the best same-label IoU among the on-device detections is
taken, and the score is the mean over reference detections (100 = identical
detections). This is the gate the optimization loop ran against — raw per-tensor
PCC is not, for the query-permutation reason above.

| Impl | detection-IoU vs reference | objects detected |
|---|---:|---|
| Torch reference (CPU, fp32) | 100.00 (by definition) | 2 cat + 2 remote |
| **TT-NN on p150a**          | **98.67** | 2 cat + 2 remote, per-object IoU 0.96–0.99 |

All four canonical objects are detected on device; the sub-100 score is bf16
capping the small-box IoU just under 1.0, not a missed or mislabeled object.

### Performance

Measured with `rf_detr/benchmark.py` in wall-clock mode (warm, device-synchronized
median) at batch 1 on a single p150a. Board-state variance is ~15%, so treat
the FPS as a band; Tracy-instrumented runs are slower and should be read as a
per-op breakdown, not a headline latency.

| Metric | Value |
|---|---:|
| Throughput (best, fully on-device + traced tail) | **~21.3 FPS** (~47 ms) |
| Throughput (typical, board variance)             | 19.6–21.3 FPS |
| Backbone device time                             | ~28–30 ms (execution-bound) |
| Backbone op dispatches (pre-trace)               | ~216 |
| peak DRAM (full on-device, bf16)                 | ~66–71 MiB |
| peak DRAM (bf8 backbone weights)                 | 30 MiB |

### Optimization trajectory

Each kept row improved throughput without dropping below the accuracy gate.
Numbers are detection-IoU vs the fp32 reference.

| # | Change | FPS | acc | status |
|---:|---|---:|---:|:--|
| 0 | Torch CPU reference (numerical shadow)                                         |  —   |  —    | baseline |
| 1 | TT-NN windowed-DINOv2 backbone on device; projector + decoder on host          | 17.42 | 99.67 | keep |
| 2 | metal-trace the device backbone                                                | 17.98 | 99.67 | **discard** — backbone is execution-bound, not dispatch-bound |
| 3 | `bfloat8_b` backbone matmul weights                                            | 18.23 | 99.70 | keep — peak DRAM 49 → 30 MiB |
| 4 | Full on-device chain (backbone + C2f projector + 2-stage deformable transformer + heads, bf16) | 18.78 | 98.67 | keep |
| 5 | metal-trace the projector + transformer tail (dispatch-bound)                  | **21.31** | 98.67 | keep — **best** (+22% over baseline) |

Several attempted improvements regressed or fell below the accuracy gate and were
rejected:

- **LoFi backbone matmul fidelity** — 20.80 FPS / acc 98.19. The backbone matmuls
  are *utilization*-bound, not fidelity-bound; dropping fidelity lost accuracy
  without a real speedup.
- **Fused SDPA backbone attention** — 22.04 FPS / acc 98.08. SDPA mishandles the
  non-tile-aligned sequence lengths (101 per window / 1616 merged) softmax;
  explicit masked `ttnn.softmax` is correct, SDPA isn't here.
- **Exact weight-folds** (attn-scale → Q, layer_scale1/2 → proj/fc2) — 20.69 FPS /
  acc 98.40. No speedup (the folded elementwise ops weren't the cost) and bf16
  rounding of the folded weights dropped accuracy below the gate.
- **Backbone fp32 accumulation** (HiFi4/HiFi2 + `fp32_dest_acc`) — hangs /
  times out on these matmul shapes; bf16 is the precision optimum.
- **L1-interleaved placement** on backbone ops — 19.08 FPS / acc 98.59. No win
  (−3%, within noise); a real L1 win needs *sharded* matmuls (kernel work),
  not interleaved-L1 placement, which still reshuffles per op.
- **2 command queues** — regressed (~19 FPS). The per-call input upload is tiny
  relative to the host-bound backbone, and the benchmark fully synchronizes each
  inference, so the event sync only adds overhead.

The env knobs `RFDETR_BB_FIDELITY` / `RFDETR_BB_FP32ACC` / `RFDETR_BB_L1` (default
off, model behavior unchanged) are left in `TtRfDetr` so these sweeps are
reproducible. Exact numbers for every discarded attempt live in the `tt-metal`
branch commit history that produced this repo.

---

## Known caveats

- **One image per forward.** `TtRfDetr.__call__` runs `B == 1`. True image
  batching would mean staging the backbone matmuls as batched shapes — not wired
  up.
- **Accuracy is detection-IoU, not raw PCC.** The two-stage top-k permutes the
  300 queries, so the *raw* logits/boxes PCC (~0.65 / ~0.16) is meaningless. Even
  the *all-300* Hungarian-matched PCC (~0.96 / ~0.94) understates the port, since
  it is dominated by the ~295 arbitrary background queries; the meaningful numbers
  are the foreground (confident-query) matched PCC (~1.000 / ~0.997) and the
  order-invariant detection-IoU (98.67).
- **bf16 is the precision optimum, and the headroom is thin.** `bfloat8_b`
  activations drop accuracy below the gate; full fp32 is not cleanly supported by
  ttnn (`topk` / `grid_sample` / `embedding` are bf16-only) and fp32 accumulation
  hangs on these shapes. The on-device bf16 tail lowered accuracy from the torch
  baseline 99.67 to 98.67, leaving little room for further numeric-changing
  optimization (the gate is 98.5).
- **Backbone is execution-bound.** The `[1616, 384]` qkv/fc1/fc2/proj matmuls run
  at ~2–3 TFLOPS (small K=384 contraction, default program config, unsharded
  activations). metal-trace of the backbone alone is ~6% (within noise) because
  the cost is real kernel time, not dispatch. The next real lever is L1-sharded
  matmul program configs / C++ kernels — deferred, and gated by the thin accuracy
  headroom.
- **Host glue remains for layout.** Backbone embeddings (patch conv + window
  partition) and feature-map shaping (LayerNorm + window unpartition) run on host
  — they are reshape/transpose-heavy and not tile-friendly. Moving them on-device
  is what would let the *whole* model (not just the tail) be metal-traced.
- **Reference is a faithful re-implementation.** The official `transformers`
  RF-DETR modeling needs `transformers` v5 APIs unavailable here, so the
  reference was validated by component bit-match against the HF Dinov2 /
  Deformable-DETR oracles plus docstring box agreement, then by a strict 487-key
  load of the published checkpoint — not by importing the full official model.
- **Only RF-DETR-base is exercised** (DINOv2-S/14 backbone, 3-layer decoder, 300
  queries, 91 classes at 560×560).
- **Shared multi-tenant board.** On a shared p300/p150a board, a neighbor's
  `tt-smi -r` / snapshot can wedge PCIe and hang a run; run one benchmark at a
  time and reset only your own chip.

---

## License

Apache 2.0 (matches the upstream RF-DETR, DINOv2, and tt-metal licenses).

---

## Acknowledgements

- Original model: Roboflow RF-DETR — https://github.com/roboflow/rf-detr
- Backbone: Meta AI's DINOv2 — https://github.com/facebookresearch/dinov2
- Deformable attention: Deformable-DETR — https://github.com/fundamentalvision/Deformable-DETR
- Reference math cross-checked against HuggingFace `transformers` (Dinov2 / Deformable-DETR / rf_detr)
- Runtime: Tenstorrent tt-metal / tt-nn — https://github.com/tenstorrent/tt-metal
- Demo/test image: COCO `val2017/000000039769` — https://cocodataset.org
