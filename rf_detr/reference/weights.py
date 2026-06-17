# SPDX-License-Identifier: Apache-2.0
# ------------------------------------------------------------------------
# RF-DETR-base reference weight loading + preprocessing.
# ------------------------------------------------------------------------
"""Build the reference model, load the published checkpoint (strict), and
provide the matching image preprocessing.

The published weights live at ``Roboflow/rf-detr-base`` on the HuggingFace Hub
(``model.safetensors`` — 487 tensors — plus ``config.json`` for the COCO
``id2label`` map). The snapshot directory is resolved, in order:

  1. ``$TT_RF_DETR_WEIGHTS`` — a local directory containing ``model.safetensors``
     and ``config.json`` (what ``scripts/download_weights.sh`` populates).
  2. ``huggingface_hub.snapshot_download("Roboflow/rf-detr-base")`` — downloads
     to (and reuses) the standard HF cache.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_file

from .configuration_rf_detr import RfDetrConfig
from .modeling_rf_detr import RfDetrForObjectDetection

HF_REPO_ID = "Roboflow/rf-detr-base"


def resolve_snapshot_dir() -> str:
    """Return a directory holding ``model.safetensors`` + ``config.json``.

    Honors ``$TT_RF_DETR_WEIGHTS`` first; otherwise pulls (or reuses) the
    ``Roboflow/rf-detr-base`` snapshot from the HuggingFace Hub cache.
    """
    local = os.environ.get("TT_RF_DETR_WEIGHTS")
    if local:
        if not os.path.isfile(os.path.join(local, "model.safetensors")):
            raise FileNotFoundError(
                f"$TT_RF_DETR_WEIGHTS={local} has no model.safetensors "
                f"(run scripts/download_weights.sh or unset the env var)"
            )
        return local

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=["model.safetensors", "config.json"],
    )


def _weights_path(snapshot_dir: str) -> str:
    return os.path.join(snapshot_dir, "model.safetensors")


def _config_path(snapshot_dir: str) -> str:
    return os.path.join(snapshot_dir, "config.json")


def _load_id2label(config_path: str) -> dict[int, str]:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return {int(k): v for k, v in cfg["id2label"].items()}


def load_rf_detr_base(
    weights_path: str | None = None,
    config_path: str | None = None,
    device: str = "cpu",
) -> tuple[RfDetrForObjectDetection, RfDetrConfig]:
    """Build RfDetrForObjectDetection, load weights strictly, return (model, config).

    Asserts zero missing and zero unexpected keys. ``weights_path`` /
    ``config_path`` default to the resolved ``Roboflow/rf-detr-base`` snapshot.
    """
    if weights_path is None or config_path is None:
        snapshot = resolve_snapshot_dir()
        weights_path = weights_path or _weights_path(snapshot)
        config_path = config_path or _config_path(snapshot)

    cfg = RfDetrConfig()
    cfg.id2label = _load_id2label(config_path)

    model = RfDetrForObjectDetection(cfg)
    model.eval()

    state_dict = load_file(weights_path)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)

    print(f"[load_rf_detr_base] checkpoint tensors: {len(ckpt_keys)}")
    print(f"[load_rf_detr_base] model tensors:      {len(model_keys)}")
    print(f"[load_rf_detr_base] missing keys:       {len(missing)}")
    print(f"[load_rf_detr_base] unexpected keys:    {len(unexpected)}")
    if missing:
        print("  MISSING (first 20):")
        for k in missing[:20]:
            print("   ", k)
    if unexpected:
        print("  UNEXPECTED (first 20):")
        for k in unexpected[:20]:
            print("   ", k)

    result = model.load_state_dict(state_dict, strict=True)
    assert len(result.missing_keys) == 0, f"missing keys: {result.missing_keys[:20]}"
    assert len(result.unexpected_keys) == 0, f"unexpected keys: {result.unexpected_keys[:20]}"
    print("[load_rf_detr_base] strict load OK (0 missing, 0 unexpected)")

    model.to(device)
    model.eval()
    return model, cfg


# ----------------------------------------------------------------------------
# Preprocessing: resize 560x560 (bilinear), rescale 1/255, ImageNet normalize.
# Matches RfDetrImageProcessor (do_resize, do_rescale, do_normalize).
# ----------------------------------------------------------------------------
class RfDetrPreprocessor:
    def __init__(self, cfg: RfDetrConfig):
        self.size = cfg.image_resolution
        self.mean = torch.tensor(cfg.image_mean).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.image_std).view(1, 3, 1, 1)
        self.rescale_factor = cfg.rescale_factor

    def __call__(self, image) -> torch.Tensor:
        return self.preprocess(image)

    def preprocess(self, image) -> torch.Tensor:
        """PIL.Image (RGB) -> pixel_values [1,3,size,size] float32."""
        import numpy as np
        import torch.nn.functional as F

        if image.mode != "RGB":
            image = image.convert("RGB")
        # HWC uint8 -> CHW float
        arr = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0).float()
        # resize to (size, size) bilinear, align_corners=False (PIL BILINEAR equivalent
        # for the torchvision "use_fast" path the RfDetrImageProcessor uses).
        arr = F.interpolate(
            arr, size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=True
        )
        arr = arr * self.rescale_factor
        arr = (arr - self.mean) / self.std
        return arr


def get_preprocessor(cfg: RfDetrConfig) -> RfDetrPreprocessor:
    return RfDetrPreprocessor(cfg)
