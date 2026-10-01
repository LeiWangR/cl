#!/usr/bin/env bash
set -euo pipefail
python -m compileall -q amcl train.py eval_linear.py scripts
pytest -q
python scripts/smoke_test.py
python scripts/training_smoke.py
python - <<'PY'
from PIL import Image
import torch
from amcl.augmentations import build_transform
im = Image.new("RGB", (32, 32), (128, 64, 32))
for name in ["default", "color", "a", "ab", "abc", "abcd", "abcde"]:
    x = build_transform(32, name)(im)
    assert x.shape == (3, 32, 32)
    assert torch.isfinite(x).all()
print("augmentation smoke test passed")
PY
