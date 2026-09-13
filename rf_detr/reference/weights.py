# SPDX-License-Identifier: Apache-2.0
# ------------------------------------------------------------------------
# RF-DETR-base reference weight loading + preprocessing.
# ------------------------------------------------------------------------
"""Build the reference model, load the published checkpoint (strict), and
provide the matching image preprocessing.

The published weights live at ``Roboflow/rf-detr-base`` on the HuggingFace Hub
(``model.safetensors`` — 487 tensors — plus ``config.json`` for the COCO
``id2label`` map). The two files are resolved, in order:

  1. ``$TT_RF_DETR_WEIGHTS`` — a local directory containing ``model.safetensors``
     and ``config.json`` (what ``scripts/download_weights.sh`` populates).
  2. ``huggingface_hub.hf_hub_download(repo_id, filename, revision=...)`` for each
     file — a cache hit when the snapshot was pre-downloaded (``tt-model serve``
     does this for the pinned revision), a download otherwise. ``repo_id``
     defaults to ``$HF_MODEL`` (set by the tt-model launcher) or ``HF_REPO_ID``;
     ``revision`` defaults to ``$TT_WEIGHTS_REVISION`` (a commit sha) or the
     repo's default branch. When the Hub is unreachable the lookup is retried
     with ``local_files_only=True`` so a boot from a warm cache never needs the
     network.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_file

from .configuration_rf_detr import RfDetrConfig
from .modeling_rf_detr import RfDetrForObjectDetection

HF_REPO_ID = "Roboflow/rf-detr-base"
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
WEIGHT_FILES = (WEIGHTS_FILE, CONFIG_FILE)


def _hf_download(repo_id: str, filename: str, revision: str | None) -> str:
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    except Exception as first:  # network down / offline: fall back to the cache
        try:
            return hf_hub_download(
                repo_id=repo_id, filename=filename, revision=revision, local_files_only=True
            )
        except Exception:
            raise first


def resolve_weight_files(
    repo_id: str | None = None, revision: str | None = None
) -> tuple[str, str]:
    """Return ``(model.safetensors path, config.json path)``.

    Honors ``$TT_RF_DETR_WEIGHTS`` (a local directory) first; otherwise resolves
    both files from the Hub cache / Hub for ``repo_id`` at ``revision``.
    """
    local = os.environ.get("TT_RF_DETR_WEIGHTS")
    if local:
        paths = tuple(os.path.join(local, fn) for fn in WEIGHT_FILES)
        for p in paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"$TT_RF_DETR_WEIGHTS={local} has no {os.path.basename(p)} "
                    f"(run scripts/download_weights.sh or unset the env var)"
                )
        return paths

    repo_id = repo_id or os.environ.get("HF_MODEL") or HF_REPO_ID
    revision = revision or os.environ.get("TT_WEIGHTS_REVISION") or None
    return tuple(_hf_download(repo_id, fn, revision) for fn in WEIGHT_FILES)


def resolve_snapshot_dir(repo_id: str | None = None, revision: str | None = None) -> str:
    """Return a directory holding ``model.safetensors`` + ``config.json``.

    Kept for the tests/scripts that expect a directory; the files come from
    :func:`resolve_weight_files`, which lands them side by side in one snapshot.
    """
    weights_path, _ = resolve_weight_files(repo_id, revision)
    return os.path.dirname(weights_path)


def _weights_path(snapshot_dir: str) -> str:
    return os.path.join(snapshot_dir, WEIGHTS_FILE)


def _config_path(snapshot_dir: str) -> str:
    return os.path.join(snapshot_dir, CONFIG_FILE)


def _load_id2label(config_path: str) -> dict[int, str]:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return {int(k): v for k, v in cfg["id2label"].items()}


def load_rf_detr_base(
    weights_path: str | None = None,
    config_path: str | None = None,
    device: str = "cpu",
    repo_id: str | None = None,
    revision: str | None = None,
) -> tuple[RfDetrForObjectDetection, RfDetrConfig]:
    """Build RfDetrForObjectDetection, load weights strictly, return (model, config).

    Asserts zero missing and zero unexpected keys. ``weights_path`` /
    ``config_path`` default to the files resolved by :func:`resolve_weight_files`
    for ``repo_id`` (default ``$HF_MODEL`` / ``Roboflow/rf-detr-base``) at
    ``revision`` (default ``$TT_WEIGHTS_REVISION``).
    """
    if weights_path is None or config_path is None:
        w, c = resolve_weight_files(repo_id, revision)
        weights_path = weights_path or w
        config_path = config_path or c

    cfg = RfDetrConfig()
    cfg.id2label = _load_id2label(config_path)

    model = RfDetrForObjectDetection(cfg)
    model.eval()

    print(f"[load_rf_detr_base] Loading weights from {weights_path}")
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
