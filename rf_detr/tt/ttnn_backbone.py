# SPDX-License-Identifier: Apache-2.0
"""TTNN port of RF-DETR's windowed DINOv2-S/14 backbone.

Embeddings (patch conv + cls + interpolated pos-embed + window partition) run on host;
the transformer layers AND the feature-map shaping of the four out-index layers
(1/4/7/10) run on device, so the whole backbone -> projector -> transformer graph is
one device graph with no mid-graph host readback (see ``TtRfDetr``).

Only layers 0..10 are built and run: RF-DETR consumes the hidden states after layers
1/4/7/10 (reference ``out_indices`` 2/5/8/11 count the embedding output as index 0) and
nothing reads the output of layer 11, so the reference computes it for nothing; skipping
it is exact (a global layer, ~0.95 ms of the traced graph).

Per-layer op fusion (S2): attention is one ``scaled_dot_product_attention`` call (scale
fused, padded keys masked by the kernel), and each ``linear -> * layerscale -> + residual``
pair (proj, fc2) is one ``dit_minimal_matmul_addcmul_fused`` call, so a block is
LN, qkv matmul, split heads, SDPA, concat heads, proj+ls1+res, LN, fc1, gelu, fc2+ls2+res
(10 ops instead of 19). The qkv and fc1 matmuls use the same ``minimal_matmul`` kernel
(2-3x faster than ``ttnn.linear``'s default program on these shapes); the exact gelu stays a
separate op because fusing it into the matmul is slower. Global layers stay in the merged
layout for the whole block.

Windowing: 560/14 = 40 patch grid, num_windows=4 -> 16 windows of 10x10=100 patches.
Windowed layers operate on [16, 101, 384] (1 cls + 100 patches per window).
Global layers (2,5,8,11) attend over the merged [1, 1616, 384] sequence.

Input path (S4, knob ``RFDETR_BB_INPUT``): the per-image host work is only what the device
cannot do, everything else is traced device ops (``_ingest``):

* ``patch`` (default): the patch embedding runs ON DEVICE. Host: im2col of the 560x560 image
  into a persistent fp32 ``[16, 101, 608]`` buffer (one strided copy, 0.03 ms; row 0 of every
  window is a zero row for the cls token, cols 588..607 zero pad) -> ``from_torch`` ROW_MAJOR
  (0.14 ms) -> ``copy_host_to_device`` (3.9 MB, 0.18 ms). Device, inside the trace:
  ``tilize_with_zero_padding`` (0.10 ms) then ONE ``dit_minimal_matmul_addcmul_fused`` call
  ``pos_full + 1.0 * (X @ W_patch) * ones`` (0.08 ms) = the conv as a matmul with the flattened
  ``[588, 384]`` kernel (bf16, HiFi4, fp32 accumulation) plus a constant ``[16, 101, 384]``
  tensor holding ``cls + pos_cls`` on the cls rows and ``pos_patch + conv_bias`` on the patch
  rows (built once from the reference modules, so it cannot drift). This is exact algebra
  (verified fp32 on host: max|d| 0 vs ``wb.embeddings``) but NOT bit-exact on device: the
  kernel is bf16 and the accumulation order differs, embed mean|d| 3e-4 vs the fp32 reference
  (the S1-S3 bf16 embed had 1e-4); the four feature-map PCCs are unchanged to 1e-5 and the
  outputs are deterministic run to run. Uploading the pixels as bf16 instead is NOT enough
  (embed error 1e-3, detection-IoU 98.21 < gate); fp32 W as well needs an fp32 pos tensor +
  a typecast and buys nothing downstream.
* ``rowmajor``: the S1-S3 host embeddings (torch conv + cls/pos + window partition, 0.9-1.5 ms)
  are cast to bf16 and uploaded ROW_MAJOR ``[16, 101, 384]`` (0.11 ms of host work instead of the
  0.9-1.5 ms host tilize); ``tilize_with_zero_padding`` on device (0.07 ms, in L1) yields the
  padded ``[16, 128, 384]`` the layers run on. Bit-identical to ``tile`` (logits bit-equal).
* ``tile``: the S1-S3 path -- host tilize via ``from_torch(TILE)`` in the ``RFDETR_BB_UPLOAD``
  layout (merged ``[1, 1616, 384]`` + one device reshape, or windowed).

Tried and rejected (S3, lever C): running ALL layers in the merged [1, 1616, 384] layout with
windowed attention expressed as global attention plus SDPA's on-device block-diagonal mask
(``cu_window_seqlens=[0, 101, ..., 1616]``). Numerically equivalent (PCC 1.0 vs the batched
SDPA) but the dense-mask kernel visits every K chunk for every Q chunk (16x the FLOPs:
0.17 vs 0.05 ms per layer), so the traced backbone got slower (4.51 vs 4.20 ms) while the
other ops gained only launch-floor crumbs from the 21% fewer rows; and it was
NON-deterministic run to run (feature maps differ by up to 1.1, detection-IoU 98.15-98.69),
the on-device partial-tile mask generation being the only op that changed. See
logs/opt-rf-detr/RESULTS.md (S3) and s3_merged_layout.patch.

Device-side feature shaping (``shape_features``): the reference does
``layernorm(hs)[:, 1:]`` -> ``window_unpartition`` -> raster ``[1, 384, 40, 40]``.
Dropping the 16 per-window cls rows and re-ordering the windows into raster order is a
pure row permutation, so it is expressed as ONE matmul with a constant 0/1 matrix
``P[1600, 1616]`` (``P @ X`` selects rows; exact in bf16 at HiFi4 -- every output row is
1.0 * one input row + 0 * the rest), followed by the final ``ttnn.layer_norm``. ``P`` is
built once on host by pushing an index tensor through the reference
``window_unpartition`` (``build_shaping_perm``), so it cannot drift from the reference.
The result is channels-last ``[1, 1600, 384]`` in raster order -- exactly what the
projector consumes.
"""

import os

import torch
import ttnn

# Backbone matmul weights: bf16 for accuracy. (bf8 saved DRAM but, once the projector +
# transformer also run on-device in bf16, the accumulated error pushed detection accuracy
# below the 99% gate — so the deepest stage keeps bf16.)
WEIGHT_DTYPE = ttnn.bfloat16

# Layers whose hidden state becomes a feature map (reference out_indices 2/5/8/11 count the
# embedding output as hidden_states[0], i.e. the outputs of layers 1/4/7/10). Layers after the
# last one feed nothing, so the device graph stops there.
OUT_LAYERS = (1, 4, 7, 10)


def _lin(linear, device, dtype=WEIGHT_DTYPE):
    """torch nn.Linear -> (ttnn weight [in,out], ttnn bias [1,out] or None)."""
    w = ttnn.from_torch(
        linear.weight.detach().t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    b = None
    if linear.bias is not None:
        b = ttnn.from_torch(
            linear.bias.detach().reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
    return w, b


def _vec(t, device):
    return ttnn.from_torch(t.detach().reshape(1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def build_shaping_perm(wb, height, width):
    """Row-permutation matrix for the feature-map shaping of one out-index layer.

    Returns ``(P, rows)``: ``P`` is float32 ``[n_patches, nw2 * seq]`` (here [1600, 1616]) with
    exactly one 1.0 per row, and ``rows`` is the LongTensor of source rows such that, for the
    merged windowed hidden state ``X = hs.reshape(nw2 * seq, C)``,
    ``P @ X == X[rows] == shaping(hs).flatten(2).transpose(1, 2)[0]`` where ``shaping`` is the
    reference ``hs[:, 1:]`` -> ``window_unpartition`` -> ``reshape(1, H/p, W/p, C)`` ->
    ``permute(0, 3, 1, 2)`` chain (without the layernorm, which is row-wise and commutes).
    Built by pushing the flat row index through the reference ``window_unpartition``.
    """
    cfg = wb.cfg
    nw2 = cfg.num_windows ** 2
    n_h, n_w = height // cfg.patch_size, width // cfg.patch_size
    per_window = (n_h // cfg.num_windows) * (n_w // cfg.num_windows)
    seq = per_window + 1  # + cls
    rows = torch.arange(nw2 * seq, dtype=torch.float32).reshape(nw2, seq, 1)[:, 1:]  # drop cls
    rows = wb.window_unpartition(rows, height, width).reshape(-1).long()  # raster order
    assert rows.numel() == n_h * n_w == nw2 * per_window
    perm = torch.zeros(rows.numel(), nw2 * seq, dtype=torch.float32)
    perm[torch.arange(rows.numel()), rows] = 1.0
    return perm, rows


class TtDinoBackbone:
    def __init__(
        self,
        ref_model,
        device,
        weight_dtype=WEIGHT_DTYPE,
        math_fidelity=None,
        fp32_acc=False,
        l1=True,
        image_size=None,
        attn="sdpa",
        matmul="minimal",
        upload=None,
        input_path=None,
    ):
        self.device = device
        # Input path (S4): what the host uploads per image and which traced device ops turn it into the
        # windowed [16, 101, 384] TILE tensor the layers consume (module docstring). None => env
        # RFDETR_BB_INPUT (default "patch": patch embedding on device, 11.8 ms; "rowmajor" is the bit-exact
        # fallback, 12.6 ms; "tile" the S1-S3 host tilize, 13.9-14.2 ms).
        self.input_path = input_path or os.environ.get("RFDETR_BB_INPUT", "patch")
        if self.input_path not in ("patch", "rowmajor", "tile"):
            raise ValueError(f"RFDETR_BB_INPUT must be patch|rowmajor|tile, got {self.input_path!r}")
        # Upload layout of the "tile" input path (S3). "merged": the host embed is uploaded as [1, 1616, 384] (tile-padded to
        # 1632 rows: 1.25 MB, host tilize 0.87 ms) and ONE on-device reshape (0.08 ms) turns it into the
        # windowed [16, 101, 384] (padded [16, 128, 384]: 1.5 MB, host tilize 1.12 ms) the layers run on;
        # the reshape also lands the input in L1. Exact (pure data movement). "windowed": upload as
        # [16, 101, 384] directly (the S1/S2 path). None => env RFDETR_BB_UPLOAD (default merged).
        self.upload = upload or os.environ.get("RFDETR_BB_UPLOAD", "merged")
        # L1 placement: keep the layer's working set (op outputs) in on-chip L1 instead of DRAM
        # round-trips. Default ON: on the traced graph it is 31.0 -> 26.0 ms (S1 measurement; it
        # was a wash on the old eager pipeline, whose host syncs hid it). `l1=False` => ttnn
        # default (DRAM interleaved). Not bit-identical (different matmul blocking/rounding),
        # but within the gates (accuracy 98.617 vs 98.513 with DRAM).
        self.mem = ttnn.L1_MEMORY_CONFIG if l1 else None
        wb = ref_model.backbone[0].encoder.encoder  # WindowedDinoBackbone
        self.wb = wb  # kept for host-side embeddings (+ the host shaping debug path)
        self.cfg = wb.cfg
        self.hidden = self.cfg.hidden_size
        self.num_heads = self.cfg.num_attention_heads
        self.head_dim = self.cfg.hidden_size // self.num_heads
        self.eps = self.cfg.layer_norm_eps
        self.num_windows = self.cfg.num_windows
        self.nw2 = self.num_windows ** 2
        self.out_layers = OUT_LAYERS
        # The device graph is shape-locked to the model's input resolution (560 -> 40x40 patches).
        self.image_size = int(image_size or getattr(ref_model.cfg, "image_resolution", 560))
        self.grid = self.image_size // self.cfg.patch_size  # 40
        self.seq_per_window = (self.grid // self.num_windows) ** 2 + 1  # 101
        self.merged_len = self.nw2 * self.seq_per_window  # 1616
        self.n_patches = self.grid * self.grid  # 1600
        # Precision knob: compute_kernel_config controls matmul math fidelity + fp32 accumulation.
        # None => ttnn default. Passed to every backbone matmul.
        self.compute_config = (
            ttnn.init_device_compute_kernel_config(
                device.arch(), math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_acc, packer_l1_acc=True
            )
            if math_fidelity is not None or fp32_acc
            else None
        )

        # Attention kernel. "sdpa": ttnn.transformer.scaled_dot_product_attention (FlashAttention-2
        # kernel; the 1/sqrt(d) scale is fused via scale=, and the kernel masks the tile-padded key
        # columns 101->128 / 1616->1632 itself -- do NOT pass an attn_mask: the provided-mask path
        # mishandles the padded columns, which was the README's old "SDPA is wrong here" finding).
        # "matmul": the explicit q@kT -> scale -> softmax -> probs@v chain (kept for A/B debugging).
        # Chunk sizes: windowed 101 -> one 128 chunk per window (single-pass softmax, 96 work items);
        # global 1616 -> q 96 (17 chunks x 6 heads = 102 items on the 110-core grid), k 256.
        # exp_approx_mode=False (no measurable cost). Default 32/32 chunks were slower (0.30 vs 0.13
        # ms per global layer) AND dropped the detection-IoU gate (98.36): more softmax rescale steps.
        self.attn = attn
        grid = device.compute_with_storage_grid_size()
        self.sdpa_pc_window = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=128, k_chunk_size=128, exp_approx_mode=False
        )
        self.sdpa_pc_global = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=96, k_chunk_size=256, exp_approx_mode=False
        )

        # Matmul kernel. "minimal": ttnn.experimental.minimal_matmul for qkv / fc1 and its fused
        # sibling dit_minimal_matmul_addcmul_fused for proj / fc2 (out = residual + (h @ W + b) * ls,
        # i.e. layerscale-multiply + residual-add folded into the matmul). The minimal_matmul kernel
        # is 2-5x faster than ttnn.linear's default program on these shapes (batched fc2
        # [16,128,1536] x [1536,384]: 0.51 -> 0.10 ms; qkv 0.19 -> 0.06). "linear": ttnn.linear +
        # multiply + add (kept for A/B debugging). Compute config HiFi2 WITHOUT fp32 accumulation
        # unless the fidelity knobs say otherwise: the fused op's own default (HiFi2+fp32acc) and
        # HiFi4+fp32acc give slightly higher per-stage PCC but drop the detection-IoU gate
        # (98.39 / 98.41 < 98.5). Explicit small tile blocks: the default 8x8x8 blocks need ~1.1 MB
        # of L1 circular buffers per core and clash with the L1-resident working set.
        # The fused kernel compiles ONE TensorAccessor type for both addcmul inputs, so the
        # layerscale vector must live in the same buffer type (L1 / DRAM) as the residual: ``_ls``.
        self.matmul = matmul
        self.mm_compute_config = self.compute_config or ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
        )
        self.mm_config = ttnn.MinimalMatmulConfig(  # qkv [.,384]x[384,1152], fc1 [.,384]x[384,1536]
            M_block_size=8, K_block_size=4, N_block_size=4, subblock_h=2, subblock_w=2,
            compute_with_storage_grid_size=grid,
        )
        self.dit_config = ttnn.MinimalMatmulConfig(  # proj [.,384]x[384,384], fc2 [.,1536]x[1536,384]
            M_block_size=4, K_block_size=4, N_block_size=4, subblock_h=2, subblock_w=2,
            compute_with_storage_grid_size=grid,
        )

        self.layers = []
        for layer in wb.encoder.layer[: max(self.out_layers) + 1]:  # layer 11's output is never consumed
            att = layer.attention.attention
            qkv_w = torch.cat([att.query.weight, att.key.weight, att.value.weight], dim=0)  # [3*384,384]
            qkv_b = torch.cat([att.query.bias, att.key.bias, att.value.bias], dim=0)
            qkv_w_tt = ttnn.from_torch(
                qkv_w.detach().t().contiguous(), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
            )
            qkv_b_tt = ttnn.from_torch(
                qkv_b.detach().reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            proj_w, proj_b = _lin(layer.attention.output.dense, device, dtype=weight_dtype)
            fc1_w, fc1_b = _lin(layer.mlp.fc1, device, dtype=weight_dtype)
            fc2_w, fc2_b = _lin(layer.mlp.fc2, device, dtype=weight_dtype)
            ls1 = _vec(layer.layer_scale1.lambda1, device)
            ls2 = _vec(layer.layer_scale2.lambda1, device)
            self.layers.append(
                {
                    "global": layer.global_attention,
                    "norm1_w": _vec(layer.norm1.weight, device),
                    "norm1_b": _vec(layer.norm1.bias, device),
                    "qkv_w": qkv_w_tt,
                    "qkv_b": qkv_b_tt,
                    "proj_w": proj_w,
                    "proj_b": proj_b,
                    "ls1": ls1,
                    "ls1_l1": ttnn.to_memory_config(ls1, ttnn.L1_MEMORY_CONFIG) if l1 else ls1,
                    "norm2_w": _vec(layer.norm2.weight, device),
                    "norm2_b": _vec(layer.norm2.bias, device),
                    "fc1_w": fc1_w,
                    "fc1_b": fc1_b,
                    "fc2_w": fc2_w,
                    "fc2_b": fc2_b,
                    "ls2": ls2,
                    "ls2_l1": ttnn.to_memory_config(ls2, ttnn.L1_MEMORY_CONFIG) if l1 else ls2,
                }
            )

        # ---- device-side feature shaping: final layernorm + row-permutation matmul ----
        self.final_norm_w = _vec(wb.layernorm.weight, device)
        self.final_norm_b = _vec(wb.layernorm.bias, device)
        self.final_eps = float(wb.layernorm.eps)
        perm, self.perm_rows = build_shaping_perm(wb, self.image_size, self.image_size)
        # 0/1 matrix, exact in bf16. HiFi4 keeps the full bf16 mantissa of X, so P @ X is a
        # bit-exact row gather (one nonzero term per output; verified bit-exact on device, and
        # fp32 accumulation adds nothing but +50% op time, so it stays off). LoFi is NOT exact.
        self.perm = ttnn.from_torch(perm, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.perm_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, packer_l1_acc=False
        )

        # ---- on-device patch embedding (input path "patch") ----
        # X[16, 101, 608] (fp32 im2col, zero cls row, zero pad cols) @ W_patch[608, 384] (bf16, rows 588.. zero)
        # + pos_full[16, 101, 384] (bf16): cls rows = cls_token + pos[cls], patch rows = pos[patch] + conv bias,
        # window-partitioned by the reference module. The fused op needs its two addcmul inputs (pos_full,
        # ones) in the same buffer type; both live in DRAM. fp32 pixels + HiFi4 + fp32 accumulation: bf16
        # pixels lose the detection-IoU gate (98.21), see the module docstring.
        self.patch_dim = self.cfg.num_channels * self.cfg.patch_size ** 2  # 588
        self.patch_dim_padded = -(-self.patch_dim // 32) * 32  # 608
        self.patch_w = self.patch_pos = self.patch_ones = None
        self._im2col_buf = None
        if self.input_path == "patch":
            self._build_patch_embed()

    # ------------------------------------------------------------------- input
    @property
    def input_shape(self):
        """"tile" path: shape the host embed [16, 101, 384] is reshaped to (a free torch view) before
        ``from_torch``; the other paths upload the windowed shape."""
        if self.input_path == "tile" and self.upload == "merged":
            return (1, self.merged_len, self.hidden)
        return (self.nw2, self.seq_per_window, self.hidden)

    def _build_patch_embed(self):
        emb = self.wb.embeddings
        conv = emb.patch_embeddings.projection
        w = torch.zeros(self.patch_dim_padded, self.hidden)
        w[: self.patch_dim] = conv.weight.detach().reshape(self.hidden, self.patch_dim).t()  # (c, kh, kw) order
        self.patch_w = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        pos = emb.interpolate_pos_encoding(torch.zeros(1, self.n_patches + 1, self.hidden), self.image_size, self.image_size)
        cls_row = emb.cls_token.detach() + pos[:, :1]
        patch_rows = pos[:, 1:] + conv.bias.detach().reshape(1, 1, -1)
        pos_full = emb.window_partition(torch.cat([cls_row, patch_rows], dim=1), self.image_size, self.image_size)
        assert tuple(pos_full.shape) == (self.nw2, self.seq_per_window, self.hidden)
        self.patch_pos = ttnn.from_torch(
            pos_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.patch_ones = ttnn.from_torch(
            torch.ones(1, 1, self.hidden), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.patch_compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self._im2col_buf = torch.zeros(self.nw2, self.seq_per_window, self.patch_dim_padded, dtype=torch.float32)

    def _im2col_host(self, pixel_values):
        """[1, 3, 560, 560] -> the persistent fp32 im2col buffer [16, 101, 608] in window order: window
        (wh, ww) raster, token (h_pw, w_pw) raster after the zero cls row, features (c, kh, kw) like the
        flattened conv kernel. One strided copy (casts if pixel_values is not fp32)."""
        if pixel_values.shape[0] != 1:
            raise ValueError("the device graph is built for batch 1")
        c, p, nw = self.cfg.num_channels, self.cfg.patch_size, self.num_windows
        w = self.grid // nw  # patches per window side
        src = pixel_values.reshape(c, nw, w, p, nw, w, p).permute(1, 4, 2, 5, 0, 3, 6)  # wh ww hpw wpw c kh kw
        self._im2col_buf[:, 1:, : self.patch_dim].view(nw, nw, w, w, c, p, p).copy_(src)
        return self._im2col_buf

    def host_input(self, pixel_values):
        """Per-image host work -> host ttnn tensor to upload (the trace's persistent input has this spec)."""
        if self.input_path == "patch":
            return ttnn.from_torch(self._im2col_host(pixel_values), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        return self.host_input_from_embed(self.wb.embeddings(pixel_values))

    def host_input_from_embed(self, embed):
        """Host embed [16, 101, 384] (torch) -> host ttnn tensor, "rowmajor" or "tile" style ("patch" has no
        embed input; it uses the exact rowmajor upload, which is what ``run_layers`` wants)."""
        if self.input_path == "tile":
            return ttnn.from_torch(embed.reshape(self.input_shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return ttnn.from_torch(
            embed.reshape(self.nw2, self.seq_per_window, self.hidden).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def _ingest(self, x):
        """Device input tensor (any input path, recognised by layout/shape) -> windowed [16, 101, 384] TILE
        (padded [16, 128, 384]) in ``self.mem``; traced with the rest of the graph."""
        if x.layout == ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.tilize_with_zero_padding(x, memory_config=self.mem, use_multicore=True)
            if x.shape[2] == self.patch_dim_padded:  # im2col -> patch embedding
                return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                    x, self.patch_w, 1.0, self.patch_pos, self.patch_ones,
                    bias_tensor=None, config=self.dit_config, memory_config=self.mem, dtype=ttnn.bfloat16,
                    compute_kernel_config=self.patch_compute_config,
                )
            return x
        if x.shape[0] == 1:  # merged TILE upload
            return ttnn.reshape(x, (self.nw2, self.seq_per_window, self.hidden), memory_config=self.mem)
        return x

    # ------------------------------------------------------------------ layers
    def _merge_windows(self, x):
        """[16, 101, 384] -> [1, 1616, 384] (the global-attention / shaping layout)."""
        b, s, c = x.shape
        return ttnn.reshape(x, (b // self.nw2, self.nw2 * s, c))

    def _attention(self, normed, p, is_global):
        """norm1 output [b, s, 384] -> multi-head self-attention context [b, s, 384] (heads concatenated)."""
        cc = self.compute_config
        mc = self.mem
        qkv = self._matmul(normed, p["qkv_w"], p["qkv_b"])
        if self.attn == "sdpa":
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                qkv, num_heads=self.num_heads, transpose_key=False
            )  # 3 x [b,h,s,d]
            ctx = ttnn.transformer.scaled_dot_product_attention(
                q, k, v,
                is_causal=False,
                scale=self.head_dim ** -0.5,
                program_config=self.sdpa_pc_global if is_global else self.sdpa_pc_window,
                compute_kernel_config=cc,
                memory_config=mc,
            )
        else:
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                qkv, num_heads=self.num_heads, transpose_key=True
            )
            scores = ttnn.matmul(q, k, compute_kernel_config=cc, memory_config=mc)
            scores = ttnn.multiply(scores, self.head_dim ** -0.5, memory_config=mc)
            probs = ttnn.softmax(scores, dim=-1, memory_config=mc)
            ctx = ttnn.matmul(probs, v, compute_kernel_config=cc, memory_config=mc)  # [b,h,s,d]
        ctx = ttnn.transformer.concatenate_heads(ctx)  # [b,s,384]
        ttnn.deallocate(qkv)
        return ctx

    @staticmethod
    def _ls(p, key, residual):
        """Layerscale vector in the same buffer type as ``residual`` (layer 0's residual is the DRAM
        input buffer, later residuals live in L1 when l1=True)."""
        if residual.memory_config().buffer_type == ttnn.BufferType.L1:
            return p[key + "_l1"]
        return p[key]

    def _matmul(self, h, w, b):
        """h @ w + b (qkv, fc1)."""
        if self.matmul == "minimal":
            return ttnn.experimental.minimal_matmul(
                h, w, bias_tensor=b, config=self.mm_config, memory_config=self.mem,
                compute_kernel_config=self.mm_compute_config,
            )
        return ttnn.linear(h, w, bias=b, compute_kernel_config=self.compute_config, memory_config=self.mem)

    def _matmul_ls_residual(self, h, w, b, ls_key, p, residual):
        """residual + layerscale * (h @ w + b) (proj, fc2): one fused op, or linear -> multiply -> add."""
        mc = self.mem
        if self.matmul == "minimal":
            return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                h, w, 1.0, residual, self._ls(p, ls_key, residual),
                bias_tensor=b,
                config=self.dit_config,
                memory_config=mc,
                compute_kernel_config=self.mm_compute_config,
            )
        y = ttnn.linear(h, w, bias=b, compute_kernel_config=self.compute_config, memory_config=mc)
        y = ttnn.multiply(y, p[ls_key], memory_config=mc)
        return ttnn.add(y, residual, memory_config=mc)

    def _layer(self, x, p, x_merged=None):
        """One DINO block. ``x``: [16, 101, 384]. ``x_merged`` may pass the already-merged view of ``x``
        for a global layer.

        Global layers run the WHOLE block (attention + MLP + both residuals) in the merged
        [1, 1616, 384] layout and reshape back to the windowed layout once at the end: the MLP is
        row-wise, so this is the same math, but a [1632, 1536] x [1536, 384] matmul is ~3x faster
        than the batched [16, 128, 1536] x [1536, 384] one and the 21% window padding is not
        computed (windowed layers cannot avoid it: their attention needs the [16, ...] layout).
        """
        windowed_shape = (x.shape[0], x.shape[1], x.shape[2])
        if p["global"]:
            x = x_merged if x_merged is not None else self._merge_windows(x)
        mc = self.mem  # None => DRAM default; L1_MEMORY_CONFIG keeps the working set on-chip

        # ---- attention (norm1 -> MHA -> proj -> layerscale1 -> residual) ----
        normed = ttnn.layer_norm(x, weight=p["norm1_w"], bias=p["norm1_b"], epsilon=self.eps, memory_config=mc)
        ctx = self._attention(normed, p, p["global"])  # [b,s,384]
        x = self._matmul_ls_residual(ctx, p["proj_w"], p["proj_b"], "ls1", p, x)

        # ---- mlp (norm2 -> fc1 -> gelu -> fc2 -> layerscale2 -> residual) ----
        h = ttnn.layer_norm(x, weight=p["norm2_w"], bias=p["norm2_b"], epsilon=self.eps, memory_config=mc)
        h = self._matmul(h, p["fc1_w"], p["fc1_b"])
        h = ttnn.gelu(h, fast_and_approximate_mode=False, memory_config=mc)  # exact; fusing it is slower
        x = self._matmul_ls_residual(h, p["fc2_w"], p["fc2_b"], "ls2", p, x)
        if p["global"]:
            x = ttnn.reshape(x, windowed_shape)  # back to [16, 101, 384] for the next windowed layer
        return x

    # --------------------------------------------------------- device shaping
    def shape_features(self, x_merged, apply_norm=True):
        """[1, 1616, 384] merged hidden state -> feature map [1, 1600, 384], channels-last, raster order.

        ``P @ X`` drops the per-window cls rows and re-orders windows (exact row gather), then the
        backbone's final layernorm. ``apply_norm=False`` returns the bare gather (for exactness tests).
        """
        y = ttnn.matmul(
            self.perm, x_merged, compute_kernel_config=self.perm_compute_config, memory_config=self.mem
        )  # [1, 1600, 384]
        if not apply_norm:
            return y
        return ttnn.layer_norm(
            y, weight=self.final_norm_w, bias=self.final_norm_b, epsilon=self.final_eps, memory_config=self.mem
        )

    def forward_device(self, x):
        """Whole device backbone. ``x``: the device input tensor of the configured input path (the fp32
        ROW_MAJOR im2col [16, 101, 608], the bf16 ROW_MAJOR embed [16, 101, 384], or the TILE embed in
        ``input_shape``), see ``_ingest``.

        Returns the 4 shaped feature maps as device tensors [1, 1600, 384] (channels-last, raster
        order, final LN applied) -- no host round trip, so it is metal-trace-able end to end.
        The merged view built for the shaping is reused by the global layer that follows; the
        loop ends with the last out layer (10), see the module docstring.
        """
        feats = []
        merged = None
        x = self._ingest(x)
        for i, p in enumerate(self.layers):
            x = self._layer(x, p, x_merged=merged)
            merged = None
            if i in self.out_layers:
                merged = self._merge_windows(x)
                feats.append(self.shape_features(merged))
        return feats

    def feature_maps(self, pixel_values):
        """Full backbone through the configured input path: host_input -> device ingest + layers + shaping
        -> host. Returns list of 4 torch tensors [1,384,40,40] (reference layout)."""
        x = ttnn.to_device(self.host_input(pixel_values), self.device)
        feats = self.forward_device(x)
        n = self.grid
        return [
            ttnn.to_torch(f).float().reshape(pixel_values.shape[0], n, n, -1).permute(0, 3, 1, 2).contiguous()
            for f in feats
        ]

    # ------------------------------------------------ eager / host debug path
    def run_layers(self, embed_windowed_torch):
        """Debug path: embed [16, 101, 384] -> dict idx->torch hidden after layers 1,4,7,10 (readbacks).
        Starts from the given (exact) embed, so it isolates the layers from the input path."""
        x = ttnn.to_device(self.host_input_from_embed(embed_windowed_torch), self.device)
        x = self._ingest(x)
        out = {}
        for i, p in enumerate(self.layers):
            x = self._layer(x, p)
            if i in self.out_layers:
                out[i] = ttnn.to_torch(x).float()
        return out

    def feature_maps_host(self, pixel_values):
        """Debug path (the pre-S1 pipeline): device layers with 4 readbacks + host shaping
        (torch LN + drop cls + window_unpartition). Returns list of 4 torch tensors [1,384,40,40]."""
        embed = self.wb.embeddings(pixel_values)  # [16,101,384] host
        hidden = self.run_layers(embed)
        _, _, H, W = pixel_values.shape
        feats = []
        for i in self.out_layers:
            hs = hidden[i]
            hs = self.wb.layernorm(hs)
            hs = hs[:, 1:]  # drop cls per window
            hs = self.wb.window_unpartition(hs, H, W)
            hs = hs.reshape(pixel_values.shape[0], H // self.cfg.patch_size, W // self.cfg.patch_size, -1)
            hs = hs.permute(0, 3, 1, 2).contiguous()
            feats.append(hs)
        return feats
