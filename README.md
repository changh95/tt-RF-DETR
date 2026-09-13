# tt-RF-DETR

End-to-end port of [RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow) to
Tenstorrent **tt-metal** (tt-nn + tt-metallium), running on a single Blackhole
p150a chip.

This repository contains a TT-NN implementation of the `Roboflow/rf-detr-base`
checkpoint, a faithful torch reference used as a numerical shadow, pytest suites
for per-stage PCC and real-image detection correctness, a benchmark that reports
inference speed / accuracy / peak DRAM, a Tracy-profileable perf test, and a
script to pull the published weights.

The TT-NN forward runs the **whole detector on the chip** — the DINOv2 patch
embedding, the windowed DINOv2-S/14 backbone with its feature-map shaping, the
C2f projector, the two-stage deformable transformer, and the detection heads —
as **one metal-trace** replayed per image. The only per-image host work left is
an im2col of the 560×560 image into a persistent buffer (one strided copy), the
upload of that buffer, and the two readbacks (logits, boxes). The transformer
decoder runs on fused kernels (`RFDETR_DEC=fused`, the default); the previous
op-by-op decoder is kept behind `RFDETR_DEC=legacy` for A/B.

Measured on a single p150a (batch 1, `rf_detr/benchmark.py`, warm,
device-synchronized median): **8.1 ms per image (123 FPS)** at detection-IoU
**98.70** vs the fp32 reference, down from 12.0 ms (83 FPS) with the legacy
decoder and 39.6 ms (25 FPS) for the pre-campaign pipeline. See **Results**.

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
│   ├── ttnn_backbone.py          # patch embedding + windowed DINOv2-S/14 (layers 0-10) + feature shaping on device
│   ├── ttnn_projector.py         # C2f projector (1x1 as linear, 3x3 via ttnn.conv2d)
│   ├── ttnn_transformer.py       # two-stage select + 3 deformable decoder layers + heads (fused / legacy decoder)
│   └── ttnn_rf_detr.py           # end-to-end forward: the whole device graph as ONE metal-trace
├── benchmark.py                  # FPS + detection-IoU accuracy + peak DRAM (grep-parseable)
└── tests/
    ├── test_backbone_pcc.py      # per-stage backbone PCC vs torch reference
    ├── test_transformer_pcc.py   # Hungarian-matched transformer-tail PCC
    ├── test_pretrained_eval.py   # real-image end-to-end detection (torch ↔ tt)
    ├── test_perf.py              # single-iter forward, Tracy-profileable
    └── test_shaping_perm.py      # host-only (torch): device feature-shaping permutation == reference, bit-exact
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
   loader also resolves the two files straight from the HF Hub (or its cache,
   with an offline fallback) if you skip this step; override the local snapshot
   dir with `TT_RF_DETR_WEIGHTS`. `HF_MODEL` / `TT_WEIGHTS_REVISION` override
   the Hub repo id / revision (defaults: `Roboflow/rf-detr-base`, default branch).

---

## Running the tests

```bash
# 0) Validate the torch reference (strict 487-key load + cat/remote detections)
python -m rf_detr.reference.validate_reference

# 0b) Host-only unit test (torch, no device): the device feature-shaping
#     permutation matrix equals the reference shaping chain bit-exactly
pytest rf_detr/tests/test_shaping_perm.py -v

# 1) Backbone per-stage PCC (tt vs torch reference)
pytest rf_detr/tests/test_backbone_pcc.py -v -s --device-id 0

# 2) Transformer-tail PCC (Hungarian-matched; two-stage top-k permutes queries)
pytest rf_detr/tests/test_transformer_pcc.py -v -s --device-id 0

# 3) Real-image end-to-end detection (torch reference ↔ TT-NN)
pytest rf_detr/tests/test_pretrained_eval.py -v -s --device-id 0

# 4) Wall-clock FPS / accuracy / peak-DRAM benchmark (fused decoder, the default)
python -m rf_detr.benchmark --impl ttnn --device-id 0 --iters 20 --warmup 5

# 4b) The same with the legacy (pre-fusion) decoder, for A/B
RFDETR_DEC=legacy python -m rf_detr.benchmark --impl ttnn --device-id 0 --iters 20 --warmup 5

# 4c) Eager (no metal-trace) run of the same device graph, for debugging
RFDETR_EAGER=1 python -m rf_detr.benchmark --impl ttnn --device-id 0 --iters 5

# 5) Tracy-profiled single-iter forward (requires Tracy-enabled tt-metal build)
python -m tracy --no-runtime-analysis --collect-noc-traces \
    --profiler-capture-perf-counters=all --op-support-count=10000 \
    -v -r -o ./tracy_out -m pytest rf_detr/tests/test_perf.py
```

Every device test and the benchmark read the same knobs as `TtRfDetr` (see
**Knobs** below), so `RFDETR_DEC=legacy pytest rf_detr/tests/...` gates the
legacy decoder and `RFDETR_BB_INPUT=rowmajor ...` the bit-exact input path.

### Knobs

All knobs are environment variables read once when `TtRfDetr` is built; the
defaults are the fast, validated path. Constructor arguments
(`TtDinoBackbone(..., attn=, matmul=, input_path=, upload=)`,
`TtTransformer(..., dec=, features=)`, `TtRfDetr(..., use_trace=)`) override
the environment.

| Knob | Values (default first) | What it selects |
|---|---|---|
| `RFDETR_DEC` | `fused` \| `legacy` | Decoder implementation. `fused`: `multi_scale_deformable_attn` cross-attention with the sampling grid from one fused matmul+addcmul, fused-qkv + SDPA self-attention, `minimal_matmul(relu)` FFN, table-based sine embedding and box refinement (~33 launches per decoder layer). `legacy`: the pre-fusion op chain (~55 launches per layer; `grid_sample` with the 16→32 channel pad, explicit q/k/v linears + matmul/softmax attention, per-coordinate sine embedding). Same weights, math and gates; `legacy` reproduces the previous decoder's output apart from `decoder.norm` now being applied once (an exact simplification on both paths). |
| `RFDETR_DEC_FEATURES` | `msda,sa,ffn,sine,refine` | Comma list enabling the fused-decoder fusions individually (A/B). `sine2`, `lin_out` are measured alternatives, not in the default set. |
| `RFDETR_BB_INPUT` | `patch` \| `rowmajor` \| `tile` | Backbone input path. `patch`: fp32 im2col upload + tilize + ONE fused patch-embedding matmul on device (not bit-exact vs the torch embed: bf16 kernel weights; feature-map PCCs unchanged to 1e-5). `rowmajor`: host embed uploaded bf16 ROW_MAJOR, tilize on device — the bit-exact fallback. `tile`: host tilize (the oldest path). |
| `RFDETR_BB_UPLOAD` | `merged` \| `windowed` | `tile` path only: upload the embed as `[1, 1616, 384]` + one device reshape, or as `[16, 101, 384]`. |
| `RFDETR_BB_ATTN` | `sdpa` \| `matmul` | Backbone attention: fused `scaled_dot_product_attention` (no explicit mask — the kernel masks the padded keys) or the explicit q·kᵀ → scale → softmax → ·v chain. |
| `RFDETR_BB_MATMUL` | `minimal` \| `linear` | Backbone matmuls: `minimal_matmul` for qkv/fc1 and the fused matmul+layerscale+residual op for proj/fc2, or `ttnn.linear` + multiply + add. |
| `RFDETR_BB_L1` | `1` \| `0` | Keep the backbone working set in L1 (default on) or in DRAM. |
| `RFDETR_BB_FIDELITY` / `RFDETR_BB_FP32ACC` | unset \| `LoFi`/`HiFi2`/`HiFi4`, `0` \| `1` | Backbone matmul math fidelity / fp32 accumulation (default: ttnn default, off). Left in for sweeps; both non-default settings were measured and rejected (see the trajectory). |
| `RFDETR_EAGER` | `0` \| `1` | `1` runs the same device graph eagerly instead of capturing/replaying the metal-trace (debugging). |

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
| **TT-NN on p150a, fused decoder** (`RFDETR_DEC=fused`, default) | **98.70** | 2 cat + 2 remote (+ couch), scores within 0.008 of the reference |
| TT-NN on p150a, legacy decoder (`RFDETR_DEC=legacy`) | 98.74 | 2 cat + 2 remote (+ couch), scores within 0.016 of the reference |
| TT-NN on p150a, pre-campaign pipeline (eager backbone + traced tail) | 98.62–98.67 | 2 cat + 2 remote, per-object IoU 0.96–0.99 |

All four canonical objects are detected on device; the sub-100 score is bf16
capping the small-box IoU just under 1.0, not a missed or mislabeled object.
The accuracy gate is 98.5. On a 7-image robustness set (demo image + hflip,
vflip, crop, dark, bright, rot90) the fused and legacy decoders have the same
mean detection-IoU (98.63); per-image values move by up to ±0.3 for ULP-level
changes, so the demo-image score alone is a coarse proxy.

Per-stage PCC of the current state (fused decoder; the backbone is untouched by
the decoder fusion): hidden state after layers 1/4/7/10 0.99975 / 0.99876 /
0.99907 / 0.99921; feature maps 0.99985 / 0.99939 / 0.99909 / 0.99925;
foreground-matched transformer tail boxes / logits 0.999969 / 0.998028
(legacy decoder: 0.999972 / 0.996969).

### Performance

Measured with `rf_detr/benchmark.py` in wall-clock mode (warm, device-synchronized
median, `--iters 20 --warmup 5`) at batch 1 on a single p150a, 2026-09-13.
Run-to-run noise of that harness on this host is about ±0.3 ms; Tracy-instrumented
runs are slower and should be read as a per-op breakdown, not a headline latency.
"Legacy" below is `RFDETR_DEC=legacy` on the same code (the pre-fusion decoder).

| Metric | Legacy decoder | **Fused decoder (default)** |
|---|---:|---:|
| Median latency / throughput (`benchmark.py`) | 12.0 ms / 83 FPS (repeats 11.8–12.0 ms, 83–85 FPS) | **8.1 ms / 123 FPS** (repeats 8.1–8.2 ms, 122–123 FPS) |
| Served over HTTP (uvicorn, warm, `timing_ms.inference`, 50 requests) | 11.70 ms median | **8.61 ms** median (min 8.18); server total 15.4 ms incl. 5.8 ms PNG decode; 19.6 ms client wall on localhost |
| Device time of the one trace | 10.93 ms = ingest 0.13 + backbone + shaping 4.18 + projector 0.55 + transformer 6.07 | **7.44 ms** = ingest 0.13 + backbone + shaping 4.16 + projector 0.56 + transformer 2.59 |
| Decoder layer (self-attn + deformable cross-attn + FFN + 3 LN), device | 1.79 ms × 3 (~55 launches each) | **0.77 ms × 3** (~33 launches each) |
| Host work per image | im2col + `from_torch` + upload + readbacks | ~0.8 ms (im2col 0.16 + from_torch 0.20 + upload 0.30 + readbacks 0.16) |
| Backbone device ops | same (the backbone is untouched by the decoder fusion) | ~125 (2 ingest + 11 layers × 10 fused ops + 4 shapings × 3 + global-layer reshapes), all inside the trace |
| peak DRAM (bf16, incl. trace buffers) | 73.8 MiB | 74.7 MiB |
| Pre-campaign pipeline (eager backbone with host shaping + traced tail) | 39.6 ms / 25.3 FPS | — |

Note: the pre-campaign 39.6 ms figure is the campaign's baseline re-measurement
of that code with the current harness on this host; the 21.3 FPS (~47 ms) row
in the trajectory below is the earlier measurement of the same state (board and
harness differed; that measurement quoted ~15% board-state variance).

### Optimization trajectory

Each kept row improved throughput without dropping below the accuracy gate.
Numbers are detection-IoU vs the fp32 reference. Rows 0–5 are the original
history of the port; the steps S1–S5 described after the rejected-attempts list
took it from 25.3 to 83–85 FPS (one metal-trace for the whole graph, fused
backbone ops, patch embedding on device), and step M1 (the fused decoder)
took it to 122–123 FPS.

| # | Change | FPS | acc | status |
|---:|---|---:|---:|:--|
| 0 | Torch CPU reference (numerical shadow)                                         |  —   |  —    | baseline |
| 1 | TT-NN windowed-DINOv2 backbone on device; projector + decoder on host          | 17.42 | 99.67 | keep |
| 2 | metal-trace the device backbone                                                | 17.98 | 99.67 | **discard** — backbone is execution-bound, not dispatch-bound |
| 3 | `bfloat8_b` backbone matmul weights                                            | 18.23 | 99.70 | keep — peak DRAM 49 → 30 MiB |
| 4 | Full on-device chain (backbone + C2f projector + 2-stage deformable transformer + heads, bf16) | 18.78 | 98.67 | keep |
| 5 | metal-trace the projector + transformer tail (dispatch-bound)                  | 21.31 | 98.67 | keep (+22% over row 1; superseded by S1–S5 and M1 below) |

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

The env knobs `RFDETR_BB_FIDELITY` / `RFDETR_BB_FP32ACC` / `RFDETR_BB_L1` are left
in `TtRfDetr` so these sweeps are reproducible. Fidelity / fp32-acc default off.

**Steps S1–S5 (whole graph in one trace, fused backbone ops, patch embedding on
device): 39.6 → 11.8 ms.** Median `benchmark.py` latency on p150a after each step,
detection-IoU in brackets:

| Step | Change | ms / image |
|---|---|---:|
| — | pre-campaign: eager backbone, 4 host readbacks + host shaping + host tilize, traced tail | 39.6 (98.67) |
| S1 | feature-map shaping on device (row-permutation matmul + final LN, `build_shaping_perm`, exact) → the whole backbone → projector → transformer graph is ONE metal-trace; `RFDETR_BB_L1` default **on** (31.0 → 26.0 ms on the traced graph — it was a wash on the eager pipeline only because the host syncs hid it) | 25.5 |
| S2 | backbone layer fused to 10 ops: `RFDETR_BB_ATTN=sdpa` (fused `scaled_dot_product_attention` — correct here once NO explicit mask is passed; the kernel masks the padded keys itself, which is what the "SDPA is wrong" row above got wrong) and `RFDETR_BB_MATMUL=minimal` (`minimal_matmul` kernels, layerscale + residual fused into proj/fc2); unused layer 11 skipped; global layers run in the merged layout | 14.4 (98.71) |
| S3 | `RFDETR_BB_UPLOAD=merged`: the embed is uploaded as the merged `[1, 1616, 384]` view (21% less host tilize) and reshaped once on device; bit-identical outputs | 13.9 |
| S4a | `RFDETR_BB_INPUT=rowmajor`: bf16 ROW_MAJOR upload, tilize on device inside the trace (no host tilize); bit-identical outputs | 12.6 |
| S4b | `RFDETR_BB_INPUT=patch` (default): patch embedding on device — host does one im2col into a persistent fp32 `[16, 101, 608]` buffer; tilize + one fused matmul `X @ W_patch + pos` (bf16 kernel, HiFi4, fp32 accumulation, cls/pos/bias folded into a constant) run in the trace. Not bit-exact (embed mean\|d\| 3e-4 vs 1e-4), feature-map PCCs unchanged to 1e-5, deterministic | 11.8 (98.74) |

Rejected in S1–S5 with numbers: running every layer in the merged layout with
SDPA's on-device block-diagonal mask (dense-mask kernel does 16× the attention
FLOPs, traced backbone slower, and its on-device mask generation was
non-deterministic run to run: detection-IoU 98.15–98.69); bf16 pixels for the
on-device patch embedding (embed error 1e-3, detection-IoU 98.21 < 98.5);
`HiFi4`/`HiFi2` + fp32 accumulation for the fused backbone matmuls (higher
per-stage PCC but detection-IoU 98.39 / 98.41 < 98.5).

**Step M1 (fused decoder, `RFDETR_DEC=fused`): 11.8 → 8.1 ms.** After S1–S5 the
transformer was 6.07 of the 10.93 ms device time: three decoder layers of ~55
launches each on `[1, 300, 256]` tensors, purely launch-bound (~16 µs in-trace
launch floor on this p150a). M1 rewrote the decoder with the fused kernels this
tt-metal tree already has, one fusion at a time (each measured, `RFDETR_DEC_FEATURES`
switches them individually):

| Fusion | Change | `benchmark.py` ms (det-IoU) |
|---|---|---:|
| — | legacy decoder on the final tree (`RFDETR_DEC=legacy`) | 11.75–12.0 (98.74) |
| `msda` | deformable cross-attention sampling core = ONE `ttnn.experimental.multi_scale_deformable_attn` (no 16→32 channel pad, no `grid_sample` layout round-trips, no slice/multiply/sum); the sampling grid is ONE fused matmul+addcmul with per-image broadcast tables shared by the 3 layers; output_proj + residual fused (~30 → 15 ops per layer) | 9.18 (98.68) |
| `sa` | self-attention = pos-term linear + fused qkv matmul (query-pos term as the fused residual) + `scaled_dot_product_attention` + fused out-projection (19 → 6 ops) | 8.79 (98.72) |
| `ffn` | `minimal_matmul(fused_activation=relu)` + fused linear2+residual (4 → 2 ops; −0.11 ms in the fenced trace, benchmark delta within noise) | 8.82 (98.72) |
| `sine`, `refine` | sine embedding = one table matmul + sin + cos + `where` (~30 → 4 ops); box refinement with precomputed `[w,h,w,h]` / `[x,y,0,0]` tables instead of slices + concat (9 → 4 ops, bit-identical) | 8.12 (98.70) |
| — | `decoder.norm` applied once after the layer loop (only the last intermediate feeds the heads at inference; exact, both paths) | **8.1 (98.70)** |

Rejected in M1 with numbers: the Sequential/Parallel op-fusion framework (not
trace-capturable as shipped and no device-time win on Blackhole), a hand-written
unified decoder kernel via `ttnn.generic_op` (dispatchable, but thousands of
lines of kernel code), `ttnn.embedding(layout=TILE)` for the top-k gather
(already tiled), fused-relu MLPs for the heads (no measurable gain, code
removed), `ttnn.linear` + `add` for the out-projections (+0.10 ms), a 2-op sine
embedding with cos folded into sin (`sine2`, 1-ULP shift), and a three-layer
value projection through the head-split ops (head_dim 16 is not tile-aligned).
Per-fusion detection-IoU when applied *alone* dips below the gate on the demo
image (`sa` 98.41, `ffn` 98.19) while every cumulative state and the 7-image
robustness mean equal legacy — the proxy is discontinuous at the 1-ULP level, so
only cumulative, gated states were kept.

A further ~16–20% of device time is the estimated ceiling of a hand-written
unified decoder kernel (one launch per layer); it was scoped, not built.

---

## Known caveats

- **One image per forward.** `TtRfDetr.__call__` runs `B == 1`. True image
  batching would mean staging the backbone matmuls as batched shapes — not wired
  up.
- **Accuracy is detection-IoU, not raw PCC.** The two-stage top-k permutes the
  300 queries, so the *raw* logits/boxes PCC (~0.65 / ~0.16) is meaningless. Even
  the *all-300* Hungarian-matched PCC (~0.96 / ~0.94) understates the port, since
  it is dominated by the ~295 arbitrary background queries; the meaningful numbers
  are the foreground (confident-query) matched PCC (~1.000 / ~0.998) and the
  order-invariant detection-IoU (98.70).
- **bf16 is the precision optimum, and the headroom is thin.** `bfloat8_b`
  activations drop accuracy below the gate; full fp32 is not cleanly supported by
  ttnn (`topk` / `grid_sample` / `embedding` are bf16-only) and fp32 accumulation
  hangs on these shapes. The on-device bf16 tail lowered accuracy from the torch
  baseline 99.67 to ~98.7, leaving little room for further numeric-changing
  optimization (the gate is 98.5). Several fusions that raised per-stage PCC
  lowered the detection-IoU (see the rejected lists) — gate on the model's own
  metric, not on PCC.
- **Backbone is now the largest device segment.** With the fused ops the backbone
  + shaping is 4.16 ms of the 7.44 ms trace (11 layers × 10 ops); the qkv/fc1/
  fc2/proj matmuls use the `minimal_matmul` kernels (2–5× faster than
  `ttnn.linear`'s default program on these shapes) but are still small-K
  (384) contractions on unsharded activations. The next lever there is
  L1-sharded program configs / custom kernels — deferred, and gated by the thin
  accuracy headroom.
- **Decoder is launch-bound.** After M1 a decoder layer is ~33 launches / 0.77 ms
  on `[1, 300, 256]` tensors; the remaining structural lever is a unified
  per-layer kernel (one launch per layer, estimated −16–20% of device time),
  which was scoped but not built.
- **Host glue is only the im2col.** The per-image host work is one strided copy
  of the 560×560 image into a persistent fp32 `[16, 101, 608]` buffer, its
  upload, and the two readbacks (~0.8 ms). The device graph is shape-locked to
  the model's 560×560 input (40×40 patch grid, 16 windows of 101 tokens).
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
