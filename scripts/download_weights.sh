#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Fetch everything tt-RF-DETR's tests / benchmark / demo need:
#   1. RF-DETR-base checkpoint (Roboflow/rf-detr-base on the HuggingFace Hub:
#      model.safetensors — 487 tensors — + config.json for the COCO id2label map)
#   2. The canonical COCO val image (two cats + two remotes, id 000000039769)
#
# Overridable via env:
#   TT_RF_DETR_WEIGHTS  default ./weights   (snapshot dir: model.safetensors + config.json)
#   TT_RF_DETR_DATA     default ./data      (sample image lives here)

set -euo pipefail

WEIGHTS_DIR="${TT_RF_DETR_WEIGHTS:-./weights}"
DATA_DIR="${TT_RF_DETR_DATA:-./data}"
mkdir -p "$WEIGHTS_DIR" "$DATA_DIR"

have() { [[ -s "$1" ]]; }

echo "Weights dir: $WEIGHTS_DIR"
echo "Data dir:    $DATA_DIR"

# --- RF-DETR-base checkpoint from the HuggingFace Hub ---
if have "$WEIGHTS_DIR/model.safetensors" && have "$WEIGHTS_DIR/config.json"; then
  echo "ok  Roboflow/rf-detr-base already present"
else
  echo "... Roboflow/rf-detr-base (model.safetensors + config.json)"
  python - "$WEIGHTS_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download

dst = sys.argv[1]
for fn in ("model.safetensors", "config.json"):
    p = hf_hub_download(repo_id="Roboflow/rf-detr-base", filename=fn, local_dir=dst)
    print("   ", p)
PY
fi

# --- Canonical COCO image (two cats + two remotes) ---
if have "$DATA_DIR/cats_000000039769.jpg"; then
  echo "ok  data/cats_000000039769.jpg already present"
else
  echo "... data/cats_000000039769.jpg"
  curl -fL -o "$DATA_DIR/cats_000000039769.jpg" \
    "http://images.cocodataset.org/val2017/000000039769.jpg"
fi

echo
echo "Done."
ls -lh "$WEIGHTS_DIR" "$DATA_DIR" 2>/dev/null || true
