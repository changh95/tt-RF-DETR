# SPDX-License-Identifier: Apache-2.0
"""TTNN port of RF-DETR's transformer tail (on device).

Mirrors ``RfDetrForObjectDetection.forward`` after the backbone+projector:
two-stage query selection (group 0, inference) -> 3-layer deformable decoder
(post-LN) -> detection heads.

Input : projector output ``source`` as a channels-last ttnn tensor [1, 1600, 256]
        ( == reference ``source.flatten(2).transpose(1, 2)`` ).
Output: ``logits`` torch [1, 300, 91] and ``pred_boxes`` torch [1, 300, 4].

Config (B=1, single level, INFERENCE => group_detr=1, use only group [0] modules
and the first 300 of query_feat/refpoint_embed):
  d_model=256, num_queries=300, num_classes=91, decoder_layers=3,
  self-attn heads=8 (head_dim 32), cross-attn(deformable) heads=16 (head_dim 16),
  n_levels=1, n_points=2, decoder_ffn_dim=2048, spatial_shape=(40, 40),
  valid_ratios=1 (no padding => invalid_mask all-False => masking skipped),
  num_pos_feats(sine)=d_model//2=128.

Decoder implementation knob ``RFDETR_DEC`` (M1, "megakernel-style" op fusion; the whole
tail runs on [1, 300, 256] tensors and is launch-bound, so op COUNT is the lever):
  * ``fused`` (default): the fused-kernel decoder, ~55 -> ~33 launches per decoder layer:
      - deformable cross-attention sampling core = ONE ``ttnn.experimental.multi_scale_deformable_attn``
        (bilinear grid_sample + attention-weighted sum over the 2 points, num_levels==1 path) on
        per-head ROW_MAJOR value (16,40,40,16) / grid (16,600,1,2) / attn (16,300,2); the sampling grid
        ``2*loc-1 = (2*ref_xy-1) + offsets * (0.5*ref_wh)`` is computed in the flat [1,300,64] layout
        by ONE ``dit_minimal_matmul_addcmul_fused`` call (the offsets linear with the per-image
        broadcast tables ``ref_b`` / ``ref_s`` as its residual / scale, tables = two 0/1-matmuls of
        ``reference_points`` shared by the 3 layers), then a ROW_MAJOR-first permute; output_proj +
        residual is one fused call;
      - self-attention: q/k use hidden+query_pos and v uses hidden, so qkv = ``pos_term + hidden @ [Wq|Wk|Wv]
        + b`` with ``pos_term = query_pos @ [Wq|Wk|0]`` (one linear + one fused matmul+addcmul),
        ``split_query_key_value_and_split_heads`` -> ``scaled_dot_product_attention`` (scale fused, no
        mask: the kernel masks its own 300->320 key padding) -> ``concatenate_heads`` -> out-proj +
        residual fused;
      - FFN: ``minimal_matmul(fused_activation=relu)`` + ``dit_minimal_matmul_addcmul_fused`` (linear2 +
        residual);
      - sine embedding: ``pos[1,300,4] @ T[4,512]`` (block-diagonal inv_dim_t table with the coord 0/1 swap
        folded in) -> one sin, one cos, one masked ``where`` (~30 ops -> 4);
      - ``_refine_bboxes``: ``where(mask_xy, d, exp(d)) * ref_wh_dup + ref_xy0`` with the ``[ref_w, ref_h,
        ref_w, ref_h]`` / ``[ref_x, ref_y, 0, 0]`` tables (constants for the proposals, 0/1-matmuls of the
        reference points otherwise) instead of 4 slices + concat.
    Post-LN structure, weights, math and gates are unchanged; every fusion is numerically at or above
    the legacy chain against the fp32 reference (see logs/opt-rf-detr/RESULTS.md "## M1").
  * ``legacy``: the pre-M1 chain (grid_sample with the 16->32 channel pad, explicit q/k/v linears and
    matmul/softmax attention, per-coordinate sine embedding) -- kept for A/B.
  ``RFDETR_DEC_FEATURES`` (comma list, default = all of ``msda,sa,ffn,sine,refine`` when fused)
  enables the fusions individually for measurement; the constructor arguments ``dec`` / ``features``
  override the environment.

Key device idioms / facts (validated on Blackhole p150):
  * ``ttnn.topk`` returns (values, UINT16 indices); matches torch top-300 exactly.
  * ``ttnn.embedding(idx, table)`` does row-gather of a [N, C] table (used for the
    two-stage topk coordinate/query gather).
  * ``ttnn.grid_sample`` (legacy) takes channel-LAST input (N, H, W, C) + grid (N, Hg, Wg, 2)
    and returns (N, Hg, Wg, C); it requires C % 32 == 0, so the 16-wide head_dim is
    zero-padded to 32 and sliced back. ``multi_scale_deformable_attn`` (fused) takes D % 16 == 0
    ROW_MAJOR bf16 DRAM-interleaved tensors and needs no pad.
  * ROW_MAJOR reshapes that change the last dim are real data-movement ops; permutes in ROW_MAJOR
    on these small tensors are cheaper than in TILE, so the glue converts to ROW_MAJOR first.
"""

import math
import os

import torch
import ttnn

H = W = 40
HW = H * W
N_QUERIES = 300
D_MODEL = 256
NUM_CLASSES = 91
N_HEADS_SELF = 8
HEAD_DIM_SELF = D_MODEL // N_HEADS_SELF  # 32
N_HEADS_CROSS = 16
HEAD_DIM_CROSS = D_MODEL // N_HEADS_CROSS  # 16
HEAD_DIM_CROSS_PAD = 32  # grid_sample (legacy path) needs channels % 32 == 0
N_POINTS = 2
NUM_POS_FEATS = D_MODEL // 2  # 128
FFN_DIM = 2048

WEIGHT_DTYPE = ttnn.bfloat16  # bf16 weights preserve box-head precision (bf8 dropped detection IoU); full fp32 unsupported by ttnn topk/embedding
ACT_DTYPE = ttnn.bfloat16

DEC_MODES = ("fused", "legacy")
# Fusions of the "fused" decoder (each independently switchable for A/B; see the module docstring).
DEC_FEATURES = ("msda", "sa", "ffn", "sine", "refine")
DEC_FEATURES_OPTIONAL = ("sine2", "lin_out")  # measured alternatives, not part of the default set


def _lin(linear, device, weight_dtype=WEIGHT_DTYPE):
    """torch nn.Linear -> (ttnn weight [in,out], ttnn bias [1,out] or None)."""
    w = ttnn.from_torch(
        linear.weight.detach().t().contiguous(), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    b = None
    if linear.bias is not None:
        b = ttnn.from_torch(
            linear.bias.detach().reshape(1, -1), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device
        )
    return {"w": w, "b": b}


def _ln(layernorm, device):
    return {
        "w": ttnn.from_torch(layernorm.weight.detach().reshape(1, 1, -1), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device),
        "b": ttnn.from_torch(layernorm.bias.detach().reshape(1, 1, -1), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device),
        "eps": float(layernorm.eps),
    }


def _mlp(mlp, device, weight_dtype=WEIGHT_DTYPE):
    """Reference MLP (relu between layers, no act on last)."""
    return [_lin(layer, device, weight_dtype=weight_dtype) for layer in mlp.layers]


def _const(t, device, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t.float().contiguous(), dtype=ACT_DTYPE, layout=layout, device=device)


def resolve_dec_features(dec=None, features=None):
    """(dec mode, frozenset of enabled fusions) from the arguments or the RFDETR_DEC / RFDETR_DEC_FEATURES env."""
    dec = dec or os.environ.get("RFDETR_DEC", "fused")
    if dec not in DEC_MODES:
        raise ValueError(f"RFDETR_DEC must be one of {DEC_MODES}, got {dec!r}")
    if dec == "legacy":
        return dec, frozenset()
    if features is None:
        env = os.environ.get("RFDETR_DEC_FEATURES")
        features = [f for f in env.split(",") if f] if env is not None else list(DEC_FEATURES)
    features = frozenset(features)
    unknown = features - set(DEC_FEATURES) - set(DEC_FEATURES_OPTIONAL)
    if unknown:
        raise ValueError(f"unknown RFDETR_DEC_FEATURES {sorted(unknown)}; known: {DEC_FEATURES + DEC_FEATURES_OPTIONAL}")
    return dec, features


class TtTransformer:
    def __init__(self, ref_model, device, dec=None, features=None):
        self.device = device
        self.dec, self.features = resolve_dec_features(dec, features)
        ref = ref_model
        tf = ref.transformer
        cfg = ref.cfg

        # ---- two-stage selection heads (group 0 only) ----
        self.enc_output = _lin(tf.enc_output[0], device)
        self.enc_output_norm = _ln(tf.enc_output_norm[0], device)
        self.enc_out_class_embed = _lin(tf.enc_out_class_embed[0], device)
        self.enc_out_bbox_embed = _mlp(tf.enc_out_bbox_embed[0], device)

        # ---- decoder shared pieces ----
        dec_ = tf.decoder
        self.ref_point_head = _mlp(dec_.ref_point_head, device)
        self.dec_norm = _ln(dec_.norm, device)

        # ---- final heads ----
        self.class_embed = _lin(ref.class_embed, device)
        self.bbox_embed = _mlp(ref.bbox_embed, device)

        # ---- input-independent constants ----
        # output_proposals [1,1600,4] (invalid_mask is all-False at 40x40 => no masking).
        _, output_proposals, invalid_mask = ref._gen_proposals(
            torch.zeros(1, HW, D_MODEL), torch.zeros(1, HW, dtype=torch.bool), [(H, W)]
        )
        assert not bool(invalid_mask.any()), "invalid_mask must be all-False at 40x40"
        self.output_proposals = ttnn.from_torch(
            output_proposals, dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device
        )

        # refpoint_embed[:300] [300,4], query_feat[:300] [300,256]
        refpoint = ref.refpoint_embed.weight[:N_QUERIES].detach()  # [300,4]
        self.refpoint_embed = ttnn.from_torch(
            refpoint.reshape(1, N_QUERIES, 4), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device
        )
        query_feat = ref.query_feat.weight[:N_QUERIES].detach()  # [300,256]
        self.target = ttnn.from_torch(
            query_feat.reshape(1, N_QUERIES, D_MODEL), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device
        )

        # sine embedding: inv_dim_t [1,1,128] and even/odd selection masks.
        dim_t = torch.arange(NUM_POS_FEATS, dtype=torch.float32)
        dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / NUM_POS_FEATS)
        inv_dim_t = (2 * math.pi) / dim_t  # [128]
        self.inv_dim_t = ttnn.from_torch(
            inv_dim_t.reshape(1, 1, NUM_POS_FEATS), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device
        )
        even = torch.zeros(NUM_POS_FEATS)
        even[0::2] = 1.0
        odd = 1.0 - even
        self.sine_even = ttnn.from_torch(
            even.reshape(1, 1, NUM_POS_FEATS), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.sine_odd = ttnn.from_torch(
            odd.reshape(1, 1, NUM_POS_FEATS), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )

        # ---- fused-decoder constants (M1) ----
        grid = device.compute_with_storage_grid_size()
        # minimal_matmul / dit fused blocking for the [300 (10 tiles), K] x [K, N] decoder matmuls: 1 row tile
        # per core, K streamed in 8-tile blocks (the default 8x8x8 blocks are 1.6x slower here).
        self.mm_cfg = ttnn.MinimalMatmulConfig(
            M_block_size=1, K_block_size=8, N_block_size=2, subblock_h=1, subblock_w=2, compute_with_storage_grid_size=grid
        )
        self.relu = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        self.ones = {n: _const(torch.ones(1, n), device) for n in (D_MODEL, 3 * D_MODEL)}
        # SDPA over the 300 queries: one 320-wide key chunk (single-pass softmax, no rescale steps), 32-row
        # query chunks (10 chunks x 8 heads = 80 work items on the 110-core grid).
        self.sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=320, exp_approx_mode=False
        )
        if self.has("sine") or self.has("sine2"):
            # T[4,512]: column block b (128 wide) carries coordinate order[b] * inv_dim_t (the reference swaps
            # coords 0/1 before the concat); even columns -> sin, odd -> cos.
            T = torch.zeros(4, 4 * NUM_POS_FEATS)
            for blk, c in enumerate((1, 0, 2, 3)):
                T[c, blk * NUM_POS_FEATS:(blk + 1) * NUM_POS_FEATS] = inv_dim_t
            self.sine_T = _const(T, device)
            even512 = torch.zeros(4 * NUM_POS_FEATS)
            even512[0::2] = 1.0
            self.sine_even512 = _const(even512.reshape(1, 1, -1), device)
            bias = torch.zeros(4 * NUM_POS_FEATS)
            bias[1::2] = math.pi / 2  # cos(x) = sin(x + pi/2): the 2-op variant folds cos into the sin
            self.sine_bias_pi2 = _const(bias.reshape(1, -1), device)
        if self.has("refine"):
            self.mask_xy = _const(torch.tensor([1.0, 1.0, 0.0, 0.0]).reshape(1, 1, 4), device)
            self.T_xy0 = _const(torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0])), device)  # ref -> [x, y, 0, 0]
            self.T_wh4 = _const(torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32), device)  # -> [w, h, w, h]
            # proposals are constants: their tables are too
            self.prop_xy0 = _const(torch.cat([output_proposals[..., :2], torch.zeros(1, HW, 2)], -1), device)
            self.prop_whd = _const(output_proposals[..., 2:].repeat(1, 1, 2), device)
            # reference_points = refine(topk_coords, refpoint_embed): the deltas are the constant refpoint_embed ->
            # t = [rp_x, rp_y, exp(rp_w), exp(rp_h)] once (device exp on the same bf16 values as the legacy path).
            self.refpoint_t = ttnn.where(self.mask_xy, self.refpoint_embed, ttnn.exp(self.refpoint_embed))
        if self.has("msda"):
            # grid tables: ref_b = ref @ T_xy2 - 1 = 2*ref_xy - 1 and ref_s = ref @ T_wh_half = 0.5*ref_wh, both
            # repeated over the 32 (head, point) pairs of the flat [1,300,64] offsets layout (column j = h*4 + p*2 + c).
            T_xy2 = torch.zeros(4, 2 * N_HEADS_CROSS * N_POINTS)
            T_wh_half = torch.zeros(4, 2 * N_HEADS_CROSS * N_POINTS)
            for j in range(2 * N_HEADS_CROSS * N_POINTS):
                T_xy2[j % 2, j] = 2.0
                T_wh_half[2 + j % 2, j] = 0.5  # (offsets / n_points * ref_wh * 0.5) * 2 with n_points == 2
            self.grid_T_xy2 = _const(T_xy2, device)
            self.grid_T_wh_half = _const(T_wh_half, device)
            self.grid_neg1 = _const(-torch.ones(1, 2 * N_HEADS_CROSS * N_POINTS), device)

        self.layers = [self._extract_layer(l, device) for l in dec_.layers]

    def has(self, feature):
        return feature in self.features

    def _extract_layer(self, layer, device):
        sa = layer.self_attn
        # nn.MultiheadAttention in_proj_weight [768,256] -> q/k/v [256,256] each.
        qw, kw, vw = sa.in_proj_weight.detach().chunk(3, dim=0)
        qb, kb, vb = sa.in_proj_bias.detach().chunk(3, dim=0)

        def _mk(w, b):
            return {
                "w": ttnn.from_torch(w.t().contiguous(), dtype=WEIGHT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device),
                "b": ttnn.from_torch(b.reshape(1, -1), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device),
            }

        ca = layer.cross_attn
        p = {
            "sa_q": _mk(qw, qb),
            "sa_k": _mk(kw, kb),
            "sa_v": _mk(vw, vb),
            "sa_out": _lin(sa.out_proj, device),
            "norm1": _ln(layer.norm1, device),
            "ca_sampling_offsets": _lin(ca.sampling_offsets, device),
            "ca_attention_weights": _lin(ca.attention_weights, device),
            "ca_value_proj": _lin(ca.value_proj, device),
            "ca_output_proj": _lin(ca.output_proj, device),
            "norm2": _ln(layer.norm2, device),
            "linear1": _lin(layer.linear1, device),
            "linear2": _lin(layer.linear2, device),
            "norm3": _ln(layer.norm3, device),
        }
        if self.has("sa"):
            # fused qkv = pos_term + hidden @ [Wq|Wk|Wv] + [bq|bk|bv], pos_term = query_pos @ [Wq|Wk|0]
            p["sa_qkv"] = _mk(sa.in_proj_weight.detach(), sa.in_proj_bias.detach())  # [256,768] / [1,768]
            p["sa_pos_w"] = ttnn.from_torch(
                torch.cat([qw, kw, torch.zeros_like(vw)], 0).t().contiguous(), dtype=WEIGHT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device
            )
        return p

    # ---------------- ops ----------------
    def _layer_norm(self, x, ln):
        return ttnn.layer_norm(x, weight=ln["w"], bias=ln["b"], epsilon=ln["eps"])

    def _linear(self, x, p, activation=None):
        return ttnn.linear(x, p["w"], bias=p["b"], activation=activation, compute_kernel_config=self.compute_config)

    def _linear_residual(self, x, p, residual):
        """residual + (x @ w + b): one fused matmul+addcmul call (scale = ones), or linear + add."""
        if self.has("lin_out"):
            return ttnn.add(residual, self._linear(x, p))
        return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            x, p["w"], 1.0, residual, self.ones[p["w"].shape[-1]], bias_tensor=p["b"],
            config=self.mm_cfg, compute_kernel_config=self.compute_config,
        )

    def _mlp_fwd(self, x, layers):
        """Reference MLP: relu between the layers, none after the last (fusing the relu into minimal_matmul was
        measured at no gain: probe_m1_smoke3.py / m1_probe2.log)."""
        for i, p in enumerate(layers):
            x = self._linear(x, p)
            if i < len(layers) - 1:
                x = ttnn.relu(x)
        return x

    # ---------------- box refinement ----------------
    def _refine_bboxes(self, reference_points, deltas):
        """cxcy = delta_xy*ref_wh + ref_xy; wh = exp(delta_wh)*ref_wh.  shapes [...,4] (legacy: slices + concat)."""
        if self.has("refine"):
            xy0 = ttnn.linear(reference_points, self.T_xy0, compute_kernel_config=self.compute_config)  # [x, y, 0, 0]
            whd = ttnn.linear(reference_points, self.T_wh4, compute_kernel_config=self.compute_config)  # [w, h, w, h]
            return self._refine_tables(xy0, whd, deltas)
        ref_xy = reference_points[..., :2]
        ref_wh = reference_points[..., 2:]
        d_xy = deltas[..., :2]
        d_wh = deltas[..., 2:]
        new_cxcy = ttnn.add(ttnn.multiply(d_xy, ref_wh), ref_xy)
        new_wh = ttnn.multiply(ttnn.exp(d_wh), ref_wh)
        return ttnn.concat([new_cxcy, new_wh], dim=-1)

    def _refine_tables(self, xy0, whd, deltas):
        """refine with precomputed tables: where(mask_xy, d, exp(d)) * [w,h,w,h] + [x,y,0,0] (4 ops)."""
        t = ttnn.where(self.mask_xy, deltas, ttnn.exp(deltas))
        return ttnn.add(ttnn.multiply(t, whd), xy0)

    # ---------------- sine embedding ----------------
    def _sine_embed(self, pos):
        """encode_sinusoidal_position_embedding(pos[1,300,4], 128) -> [1,300,512].

        Per coord: full = coord * inv_dim_t [1,300,128];
        out = sin(full)*even_mask + cos(full)*odd_mask. Then swap coords 0/1, concat.
        Fused: full512 = pos @ T (block-diagonal table, swap folded in), out = where(even, sin, cos);
        "sine2": out = sin(pos @ T + pi/2 * odd) (cos folded into sin; bf16 pi/2 = 4.8e-4 rad off, below the
        output's bf16 ULP).
        """
        if self.has("sine2"):
            return ttnn.sin(ttnn.linear(pos, self.sine_T, bias=self.sine_bias_pi2, compute_kernel_config=self.compute_config))
        if self.has("sine"):
            full = ttnn.linear(pos, self.sine_T, compute_kernel_config=self.compute_config)  # [1,300,512]
            return ttnn.where(self.sine_even512, ttnn.sin(full), ttnn.cos(full))
        embs = []
        for c in range(4):
            coord = pos[:, :, c : c + 1]  # [1,300,1]
            full = ttnn.multiply(coord, self.inv_dim_t)  # broadcast -> [1,300,128]
            s = ttnn.sin(full)
            cs = ttnn.cos(full)
            e = ttnn.add(ttnn.multiply(s, self.sine_even), ttnn.multiply(cs, self.sine_odd))
            embs.append(e)
        embs[0], embs[1] = embs[1], embs[0]
        return ttnn.concat(embs, dim=-1)  # [1,300,512]

    # ---------------- self-attention ----------------
    def _self_attention(self, hidden, query_pos, p):
        """hidden + MHA(q=k=hidden+query_pos, v=hidden) (the residual add is included)."""
        if self.has("sa"):
            pos_term = ttnn.linear(query_pos, p["sa_pos_w"], compute_kernel_config=self.compute_config)  # [1,300,768]
            qkv = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                hidden, p["sa_qkv"]["w"], 1.0, pos_term, self.ones[3 * D_MODEL], bias_tensor=p["sa_qkv"]["b"],
                config=self.mm_cfg, compute_kernel_config=self.compute_config,
            )
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=N_HEADS_SELF, transpose_key=False)
            ctx = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=HEAD_DIM_SELF ** -0.5,
                program_config=self.sdpa_pc, compute_kernel_config=self.compute_config,
            )  # [1,8,300,32]
            ctx = ttnn.transformer.concatenate_heads(ctx)  # [1,300,256]
            return self._linear_residual(ctx, p["sa_out"], hidden)
        # q = k = hidden + query_pos; v = hidden
        qk_in = ttnn.add(hidden, query_pos)
        q = self._linear(qk_in, p["sa_q"])
        k = self._linear(qk_in, p["sa_k"])
        v = self._linear(hidden, p["sa_v"])
        # reshape to [1, heads, 300, head_dim]
        q = ttnn.transpose(ttnn.reshape(q, (1, N_QUERIES, N_HEADS_SELF, HEAD_DIM_SELF)), 1, 2)
        k = ttnn.transpose(ttnn.reshape(k, (1, N_QUERIES, N_HEADS_SELF, HEAD_DIM_SELF)), 1, 2)
        v = ttnn.transpose(ttnn.reshape(v, (1, N_QUERIES, N_HEADS_SELF, HEAD_DIM_SELF)), 1, 2)
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=self.compute_config)
        scores = ttnn.multiply(scores, HEAD_DIM_SELF ** -0.5)
        probs = ttnn.softmax(scores, dim=-1)
        ctx = ttnn.matmul(probs, v, compute_kernel_config=self.compute_config)  # [1,heads,300,head_dim]
        ctx = ttnn.reshape(ttnn.transpose(ctx, 1, 2), (1, N_QUERIES, D_MODEL))
        return ttnn.add(hidden, self._linear(ctx, p["sa_out"]))

    # ---------------- deformable cross-attention ----------------
    def _value_for_layer(self, source, p):
        """Per-layer value projection in the layout the sampling core wants: [1,1600,256] TILE (legacy) or
        per-head channels-last (16,40,40,16) ROW_MAJOR (MSDA)."""
        if not self.has("msda"):
            return self._linear(source, p["ca_value_proj"])
        v = self._linear(source, p["ca_value_proj"])  # [1,1600,256]
        v = ttnn.to_layout(v, ttnn.ROW_MAJOR_LAYOUT)  # RM first: the permute is cheaper in ROW_MAJOR
        v = ttnn.reshape(v, (1, HW, N_HEADS_CROSS, HEAD_DIM_CROSS))
        v = ttnn.permute(v, (0, 2, 1, 3))  # [1,16,1600,16]
        return ttnn.reshape(v, (N_HEADS_CROSS, H, W, HEAD_DIM_CROSS))

    def _grid_tables(self, reference_points):
        """Per-image broadcast tables for the fused sampling grid (shared by the 3 layers):
        ref_b = 2*ref_xy - 1 and ref_s = 0.5*ref_wh, each tiled over the 32 (head, point) pairs -> [1,300,64]."""
        ref_b = ttnn.linear(reference_points, self.grid_T_xy2, bias=self.grid_neg1, compute_kernel_config=self.compute_config)
        ref_s = ttnn.linear(reference_points, self.grid_T_wh_half, compute_kernel_config=self.compute_config)
        return ref_b, ref_s

    def _deformable_attention(self, hidden, query_pos, reference_points, value, p, grid_tables=None):
        """hidden + MSDeformAttn (16 heads, 1 level, 2 points); reference_points [1,300,4]; ``value`` from
        ``_value_for_layer``. The residual add is included."""
        query = ttnn.add(hidden, query_pos)
        if self.has("msda"):
            ref_b, ref_s = grid_tables
            # attention weights: softmax over the 2 points -> (16,300,2) ROW_MAJOR
            attn_w = self._linear(query, p["ca_attention_weights"])  # [1,300,32]
            attn_w = ttnn.reshape(attn_w, (1, N_QUERIES, N_HEADS_CROSS, N_POINTS))
            attn_w = ttnn.softmax(attn_w, dim=-1)
            attn_w = ttnn.permute(attn_w, (0, 2, 1, 3))  # [1,16,300,2]
            attn_w = ttnn.reshape(attn_w, (N_HEADS_CROSS, N_QUERIES, N_POINTS))
            attn_w = ttnn.to_layout(attn_w, ttnn.ROW_MAJOR_LAYOUT)
            # sampling grid 2*loc-1 = ref_b + (query @ W_off + b_off) * ref_s in the flat [1,300,64] layout
            grid = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                query, p["ca_sampling_offsets"]["w"], 1.0, ref_b, ref_s, bias_tensor=p["ca_sampling_offsets"]["b"],
                config=self.mm_cfg, compute_kernel_config=self.compute_config,
            )
            grid = ttnn.to_layout(grid, ttnn.ROW_MAJOR_LAYOUT)
            grid = ttnn.reshape(grid, (N_QUERIES, N_HEADS_CROSS, N_POINTS * 2))  # (300, 16, [p, xy])
            grid = ttnn.permute(grid, (1, 0, 2))  # (16, 300, 4)
            grid = ttnn.reshape(grid, (N_HEADS_CROSS, N_QUERIES * N_POINTS, 1, 2))
            out = ttnn.experimental.multi_scale_deformable_attn(value, grid, attn_w, align_corners=False)  # (16,300,16) RM
            out = ttnn.permute(out, (1, 0, 2))  # (300,16,16)
            out = ttnn.reshape(out, (1, N_QUERIES, D_MODEL))
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
            return self._linear_residual(out, p["ca_output_proj"], hidden)

        # ---- legacy chain: grid_sample on the 32-padded per-head value ----
        value_proj = value
        # sampling_offsets -> [1,300,16,1,2,2]; attention_weights -> softmax over 2 points
        offsets = self._linear(query, p["ca_sampling_offsets"])  # [1,300,64]
        offsets = ttnn.reshape(offsets, (1, N_QUERIES, N_HEADS_CROSS, N_POINTS, 2))  # n_levels=1 squeezed
        attn_w = self._linear(query, p["ca_attention_weights"])  # [1,300,32]
        attn_w = ttnn.reshape(attn_w, (1, N_QUERIES, N_HEADS_CROSS, N_POINTS))
        attn_w = ttnn.softmax(attn_w, dim=-1)  # over n_levels*n_points = 2

        # 4-d reference points: loc = ref_xy + offsets / n_points * ref_wh * 0.5
        # ref slices: [1,300,1,1,2] broadcastable to [1,300,16,2,2]
        ref_xy = ttnn.reshape(reference_points[:, :, :2], (1, N_QUERIES, 1, 1, 2))
        ref_wh = ttnn.reshape(reference_points[:, :, 2:], (1, N_QUERIES, 1, 1, 2))
        loc = ttnn.add(ref_xy, ttnn.multiply(ttnn.multiply(offsets, (0.5 / N_POINTS)), ref_wh))  # [1,300,16,2,2]

        # grid_sample core (single level).
        # value [1,1600,256] -> [1,1600,16,16] -> per-head NHWC [16,40,40,16] (pad C->32)
        value = ttnn.reshape(value_proj, (1, HW, N_HEADS_CROSS, HEAD_DIM_CROSS))
        value = ttnn.permute(value, (0, 2, 1, 3))  # [1,16,1600,16]
        value = ttnn.reshape(value, (N_HEADS_CROSS, H, W, HEAD_DIM_CROSS))  # [16,40,40,16]
        value = ttnn.pad(value, [(0, 0), (0, 0), (0, 0), (0, HEAD_DIM_CROSS_PAD - HEAD_DIM_CROSS)], value=0.0)

        # sampling_grids = 2*loc - 1 ; grid [16, 300, 2(points), 2(xy)]
        grids = ttnn.subtract(ttnn.multiply(loc, 2.0), 1.0)  # [1,300,16,2,2]
        grids = ttnn.permute(grids, (0, 2, 1, 3, 4))  # [1,16,300,2,2]
        grids = ttnn.reshape(grids, (N_HEADS_CROSS, N_QUERIES, N_POINTS, 2))  # [16,300,2,2]

        # grid_sample requires ROW_MAJOR layout for both value and grid.
        value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
        grids = ttnn.to_layout(grids, ttnn.ROW_MAJOR_LAYOUT)
        sampled = ttnn.grid_sample(
            value, grids, mode="bilinear", padding_mode="zeros", align_corners=False
        )  # [16,300,2,32]
        sampled = ttnn.to_layout(sampled, ttnn.TILE_LAYOUT)
        sampled = sampled[:, :, :, :HEAD_DIM_CROSS]  # [16,300,2,16]

        # weight by attention_weights and sum over points.
        # attn_w [1,300,16,2] -> [16,300,2,1]
        aw = ttnn.permute(attn_w, (0, 2, 1, 3))  # [1,16,300,2]
        aw = ttnn.reshape(aw, (N_HEADS_CROSS, N_QUERIES, N_POINTS, 1))
        weighted = ttnn.multiply(sampled, aw)  # [16,300,2,16]
        out = ttnn.sum(weighted, dim=2)  # [16,300,16]
        # -> [1,300,256]: out is per-head [16,300,16]; reshape to [1,16,300,16]->[1,300,16,16]->[1,300,256]
        out = ttnn.reshape(out, (1, N_HEADS_CROSS, N_QUERIES, HEAD_DIM_CROSS))
        out = ttnn.permute(out, (0, 2, 1, 3))  # [1,300,16,16]
        out = ttnn.reshape(out, (1, N_QUERIES, D_MODEL))
        return ttnn.add(hidden, self._linear(out, p["ca_output_proj"]))

    # ---------------- FFN ----------------
    def _ffn(self, hidden, p):
        """hidden + linear2(relu(linear1(hidden))) (the residual add is included)."""
        if self.has("ffn"):
            h = ttnn.experimental.minimal_matmul(
                hidden, p["linear1"]["w"], bias_tensor=p["linear1"]["b"], fused_activation=self.relu,
                config=self.mm_cfg, compute_kernel_config=self.compute_config,
            )
            return self._linear_residual(h, p["linear2"], hidden)
        return ttnn.add(hidden, self._linear(ttnn.relu(self._linear(hidden, p["linear1"])), p["linear2"]))

    def _decoder_layer(self, hidden, query_pos, reference_points, value, p, grid_tables=None):
        # self-attention (post-LN)
        hidden = self._layer_norm(self._self_attention(hidden, query_pos, p), p["norm1"])
        # deformable cross-attention
        hidden = self._layer_norm(self._deformable_attention(hidden, query_pos, reference_points, value, p, grid_tables), p["norm2"])
        # FFN
        hidden = self._layer_norm(self._ffn(hidden, p), p["norm3"])
        return hidden

    def forward_device(self, source):
        """source: ttnn channels-last [1,1600,256] (TILE, ACT_DTYPE). Returns device (logits, pred_boxes).

        Pure-device forward (no host ops) so the whole region is metal-trace-able.
        """
        # ---- two-stage proposal heads ----
        object_query = self._layer_norm(self._linear(source, self.enc_output), self.enc_output_norm)
        enc_class = self._linear(object_query, self.enc_out_class_embed)  # [1,1600,91]
        delta_bbox = self._mlp_fwd(object_query, self.enc_out_bbox_embed)  # [1,1600,4]
        if self.has("refine"):
            enc_coord = self._refine_tables(self.prop_xy0, self.prop_whd, delta_bbox)  # [1,1600,4], constant tables
        else:
            enc_coord = self._refine_bboxes(self.output_proposals, delta_bbox)  # [1,1600,4]

        # scores = enc_class.max(-1) -> [1,1600]; topk 300
        scores = ttnn.max(enc_class, dim=-1)  # [1,1600]
        scores = ttnn.reshape(scores, (1, HW))
        if scores.dtype != ttnn.bfloat16:  # topk requires bf16/bf8 input (ranking only)
            scores = ttnn.typecast(scores, ttnn.bfloat16)
        _, topk_idx = ttnn.topk(scores, N_QUERIES, dim=-1)  # idx [1,300] uint16
        topk_idx = ttnn.typecast(topk_idx, ttnn.uint32)

        # gather rows of enc_coord [1,1600,4] by topk_idx -> [1,300,4] (embedding row-gather)
        enc_coord_tbl = ttnn.to_layout(ttnn.reshape(enc_coord, (HW, 4)), ttnn.TILE_LAYOUT)
        idx_rm = ttnn.to_layout(topk_idx, ttnn.ROW_MAJOR_LAYOUT)
        topk_coords = ttnn.embedding(idx_rm, enc_coord_tbl)  # [1,300,4]
        topk_coords = ttnn.to_layout(topk_coords, ttnn.TILE_LAYOUT)

        # reference_points = refine_bboxes(topk_coords, refpoint_embed[:300]) -> [1,300,4]
        if self.has("refine"):
            xy0 = ttnn.linear(topk_coords, self.T_xy0, compute_kernel_config=self.compute_config)
            whd = ttnn.linear(topk_coords, self.T_wh4, compute_kernel_config=self.compute_config)
            reference_points = ttnn.add(ttnn.multiply(self.refpoint_t, whd), xy0)
        else:
            reference_points = self._refine_bboxes(topk_coords, self.refpoint_embed)
        init_reference_points = reference_points

        # ---- decoder query positional embedding (lite refine: ref fixed across layers) ----
        # valid_ratios=1 => ref_inputs == reference_points (single level).
        query_sine = self._sine_embed(reference_points)  # [1,300,512]
        query_pos = self._mlp_fwd(query_sine, self.ref_point_head)  # [1,300,256]
        grid_tables = self._grid_tables(reference_points) if self.has("msda") else None

        hidden = self.target
        for p in self.layers:
            value = self._value_for_layer(source, p)  # cross-attn value projection (per-layer weights)
            hidden = self._decoder_layer(hidden, query_pos, reference_points, value, p, grid_tables)
        # decoder.norm is applied to every layer's output in the reference, but only the last intermediate feeds the
        # heads at inference, so it is applied once here (identical result, 2 launches fewer).
        last_intermediate = self._layer_norm(hidden, self.dec_norm)

        # ---- heads ----
        logits = self._linear(last_intermediate, self.class_embed)  # [1,300,91]
        boxes_delta = self._mlp_fwd(last_intermediate, self.bbox_embed)  # [1,300,4]
        pred_boxes = self._refine_bboxes(init_reference_points, boxes_delta)  # [1,300,4]
        return logits, pred_boxes

    def __call__(self, source):
        """source: ttnn channels-last [1,1600,256]. Returns (logits, pred_boxes) as torch."""
        device = self.device
        if not isinstance(source, ttnn.Tensor):
            source = ttnn.from_torch(source, dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
        source = ttnn.to_layout(source, ttnn.TILE_LAYOUT)
        if source.dtype != ACT_DTYPE:
            source = ttnn.typecast(source, ACT_DTYPE)
        logits, pred_boxes = self.forward_device(source)
        logits_t = ttnn.to_torch(logits).float().reshape(1, N_QUERIES, NUM_CLASSES)
        pred_boxes_t = ttnn.to_torch(pred_boxes).float().reshape(1, N_QUERIES, 4)
        return logits_t, pred_boxes_t
