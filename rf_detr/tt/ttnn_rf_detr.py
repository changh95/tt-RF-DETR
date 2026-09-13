# SPDX-License-Identifier: Apache-2.0
"""End-to-end RF-DETR on Tenstorrent.

On-device chain: patch embedding (S4) -> windowed DINOv2 backbone (12 layers + device
feature shaping) -> C2f projector -> two-stage deformable transformer + heads. The only
host glue left is an im2col of the 560x560 image into a persistent fp32 ``[16, 101, 608]``
buffer (one strided copy) that is uploaded ROW_MAJOR; tilize + the patch-embed matmul
(+ cls/pos) run inside the trace (see ``TtDinoBackbone``, knob ``RFDETR_BB_INPUT``;
``rowmajor`` keeps the S1-S3 host embeddings but skips the host tilize, bit-identical to
``tile``).

The WHOLE device graph is captured into ONE metal-trace and replayed per inference:
``__call__`` builds the host input (``backbone.host_input``), ``copy_host_to_device_tensor``
into the persistent device input buffer the trace reads from, ``execute_trace``, then two
``to_torch`` readbacks (logits, boxes). There is no
mid-graph host work any more: the pre-S1 pipeline read the hidden state back after
layers 1/4/7/10, did LN + drop-cls + window_unpartition on host and re-uploaded 4 maps
(~8 ms of syncs, host shaping and host tilize per image); that shaping now runs on
device (``TtDinoBackbone.shape_features``), so the backbone's ops no longer pay host
dispatch either. The backbone layer itself is 10 fused ops (SDPA attention, matmul +
layerscale + residual fused, minimal_matmul kernels; layer 11 skipped as unused), see
``ttnn_backbone.py``: 39.6 ms (eager, host shaping) -> 25.5 (S1, one trace) -> 14.4 ms
(S2) -> 13.9 ms (S3, merged embed upload) -> 12.6 ms (S4a, no host tilize) -> 11.8 ms
(S4b, patch embedding on device) per image on p150a at detection-IoU 98.71 / 98.74.
M1 (branch ``opt/rf-detr-v3-megakernel``, knob ``RFDETR_DEC=fused|legacy``) rewrote the
launch-bound decoder with the fused kernels that exist in this tree (see
``ttnn_transformer.py``: ``multi_scale_deformable_attn`` cross-attention with the sampling
grid from one fused matmul+addcmul, SDPA self-attention on a fused qkv, ``minimal_matmul``
FFN, table-based sine embedding and box refinement): device trace 10.9 -> 7.4 ms,
11.8 -> 8.1 ms per image (123 FPS) at detection-IoU 98.70.

Debug paths: ``TtRfDetr(..., use_trace=False)`` or env ``RFDETR_EAGER=1`` runs the same
device graph eagerly (no trace); ``TtDinoBackbone.feature_maps_host`` keeps the old
host-shaping pipeline for A/B comparisons.

2-CQ overlap (upload on cq_id=1 + event so compute waits) was implemented and measured
on the old pipeline: it *regressed* FPS because the benchmark runs each inference fully
synchronized (no cross-inference pipelining to overlap). The device stays on a single CQ.
"""

import os

import ttnn
from rf_detr.reference.modeling_rf_detr import RfDetrOutput
from rf_detr.tt.ttnn_backbone import TtDinoBackbone
from rf_detr.tt.ttnn_projector import TtProjector
from rf_detr.tt.ttnn_transformer import TtTransformer

N_QUERIES = 300
NUM_CLASSES = 91


class TtRfDetr:
    def __init__(self, ref_model, device, use_trace=None):
        self.ref = ref_model.eval()
        self.device = device
        # Backbone knobs, env-tunable:
        #   RFDETR_BB_FIDELITY = LoFi|HiFi2|HiFi4 (default: ttnn default) ; RFDETR_BB_FP32ACC = 1 (default 0)
        #   RFDETR_BB_L1 = 0|1 (default 1: backbone working set in L1; 31.0 -> 26.0 ms on the traced graph)
        #   RFDETR_BB_ATTN = sdpa|matmul (default sdpa: fused scaled_dot_product_attention kernel)
        #   RFDETR_BB_MATMUL = minimal|linear (default minimal: minimal_matmul kernel for qkv/fc1 and the
        #                      fused matmul+layerscale+residual op for proj/fc2; linear = ttnn.linear chain)
        #   RFDETR_BB_INPUT = patch|rowmajor|tile (default patch: fp32 im2col upload + patch-embed matmul on
        #                      device; rowmajor = host embed uploaded bf16 ROW_MAJOR + device tilize, bit-exact
        #                      vs tile; tile = the S1-S3 host tilize). Read by TtDinoBackbone (tests follow it).
        #   RFDETR_BB_UPLOAD = merged|windowed (tile path only, default merged: embed uploaded as [1,1616,384]
        #                      + one device reshape)
        # Decoder knobs (M1), read by TtTransformer:
        #   RFDETR_DEC = fused|legacy (default fused: the fused-kernel decoder -- multi_scale_deformable_attn for the
        #                deformable cross-attention with the sampling grid from one fused matmul+addcmul, see
        #                ttnn_transformer.py; legacy = the pre-M1 op chain, kept for A/B)
        #   RFDETR_DEC_FEATURES = comma list of the fusions to enable individually (default: all adopted ones)
        _fid = os.environ.get("RFDETR_BB_FIDELITY")
        _mf = getattr(ttnn.MathFidelity, _fid) if _fid else None
        _fp32 = os.environ.get("RFDETR_BB_FP32ACC", "0") == "1"
        _l1 = os.environ.get("RFDETR_BB_L1", "1") == "1"
        _attn = os.environ.get("RFDETR_BB_ATTN", "sdpa")
        _mm = os.environ.get("RFDETR_BB_MATMUL", "minimal")
        self.backbone = TtDinoBackbone(
            ref_model, device, math_fidelity=_mf, fp32_acc=_fp32, l1=_l1, attn=_attn, matmul=_mm
        )
        self.projector = TtProjector(ref_model, device)
        self.transformer = TtTransformer(ref_model, device)

        device.enable_program_cache()

        # Trace is the production path; RFDETR_EAGER=1 (or use_trace=False) runs the graph eagerly.
        self.use_trace = (os.environ.get("RFDETR_EAGER", "0") != "1") if use_trace is None else bool(use_trace)
        self._trace_id = None
        self._persistent_in = None   # persistent device input (spec = backbone.host_input's tensor)
        self._logits_out = None      # device tensor (trace output)
        self._boxes_out = None       # device tensor (trace output)

    # ------------------------------------------------------------------ pieces
    def _embed_host(self, pixel_values):
        """Per-image host work -> host ttnn tensor (not yet uploaded): the fp32 ROW_MAJOR im2col
        [16, 101, 608] (input path "patch", default), or the bf16 embed (ROW_MAJOR [16, 101, 384] / TILE)."""
        return self.backbone.host_input(pixel_values)

    def _device_graph(self, embed_dev):
        """The whole device graph: backbone layers + shaping -> projector -> transformer. Device in/out."""
        feats = self.backbone.forward_device(embed_dev)  # 4 x [1,1600,384]
        source = self.projector(feats)  # [1,1600,256]
        return self.transformer.forward_device(source)  # device (logits, pred_boxes)

    @staticmethod
    def _read_outputs(logits_dev, boxes_dev):
        logits_t = ttnn.to_torch(logits_dev).float().reshape(1, N_QUERIES, NUM_CLASSES)
        boxes_t = ttnn.to_torch(boxes_dev).float().reshape(1, N_QUERIES, 4)
        return logits_t, boxes_t

    # ------------------------------------------------------------------- trace
    def _capture_trace(self, embed_host):
        device = self.device
        # Persistent device input buffer the trace reads from.
        self._persistent_in = ttnn.to_device(embed_host, device)

        # Warm run (eager) so conv2d prepared-weights are cached and the program cache
        # is populated; mutating ops (p["w"] = prepared) must run OUTSIDE the capture.
        logits, boxes = self._device_graph(self._persistent_in)
        ttnn.synchronize_device(device)
        ttnn.deallocate(logits)
        ttnn.deallocate(boxes)

        # Capture the whole backbone + projector + transformer device graph.
        self._trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._logits_out, self._boxes_out = self._device_graph(self._persistent_in)
        ttnn.end_trace_capture(device, self._trace_id, cq_id=0)
        ttnn.synchronize_device(device)

    def _run_trace(self, embed_host):
        ttnn.copy_host_to_device_tensor(embed_host, self._persistent_in, cq_id=0)
        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)
        return self._read_outputs(self._logits_out, self._boxes_out)

    def _run_eager(self, embed_host):
        embed_dev = ttnn.to_device(embed_host, self.device)
        logits, boxes = self._device_graph(embed_dev)
        return self._read_outputs(logits, boxes)

    # -------------------------------------------------------------------- call
    def __call__(self, pixel_values):
        embed_host = self._embed_host(pixel_values)
        if not self.use_trace:
            logits, pred_boxes = self._run_eager(embed_host)
        else:
            if self._trace_id is None:
                self._capture_trace(embed_host)
            logits, pred_boxes = self._run_trace(embed_host)
        return RfDetrOutput(logits=logits, pred_boxes=pred_boxes)
