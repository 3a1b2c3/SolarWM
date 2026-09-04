#!/bin/bash
# Build SolarWM's MiniMax-H3 backend venv (.venv-h3), per
# environments/README.md's H3-specific instructions -- adapted for this
# GB300 box (aarch64, CUDA 13.2) the same way as every other project set up
# tonight:
#   1. No hardcoded python3.10 -- uses whatever python3 is on PATH.
#   2. torch/torchvision unpinned from cu132 (not the doc's pinned
#      torch==2.6.0 from cu124) -- a pinned version can silently not exist
#      on a different CUDA index and pip falls back to something wrong
#      instead of erroring (confirmed the hard way on H3-World tonight).
#   3. flash-attn unpinned (not the doc's pinned ==2.8.3) -- that exact
#      pinned version already failed to build from source on THIS box
#      tonight for a different project (zing-world-model), so pinning to
#      it again here would very likely just repeat the same failure.
#   4. Explicitly NOT using uv, even though SolarWM's own pyproject.toml
#      has uv-specific build config ([tool.uv.extra-build-dependencies])
#      that looks like the intended tool -- staying consistent with every
#      other script written tonight, per explicit instruction.
#
# SolarWM's docs also warn: do not install Wan/LTX/H3 into the same venv.
# This script only builds the H3 one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found on PATH." >&2
  exit 1
fi

VENV="$HERE/.venv-h3"
PY="$VENV/bin/python"

echo "[solarwm-h3-setup] creating venv ($(python3 --version 2>&1))..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"
"$PY" -m pip install --upgrade pip setuptools wheel

echo "[solarwm-h3-setup] torch + torchvision (cu132, unpinned -- see script header)..."
"$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

echo "[solarwm-h3-setup] H3-specific deps (diffusers, transformers, peft, imageio)..."
"$PY" -m pip install \
  diffusers==0.40.0 \
  transformers==5.12.1 peft==0.20.0 \
  imageio==2.37.4 imageio-ffmpeg==0.6.0

echo "[solarwm-h3-setup] installing SolarWM itself (train extras)..."
"$PY" -m pip install -e ".[train]"

echo "[solarwm-h3-setup] flash-attn (--no-build-isolation, unpinned, source build -- see script header)..."
"$PY" -m pip install --no-build-isolation flash-attn

echo
echo "[solarwm-h3-setup] probing environment..."
"$PY" -m solarwm environment probe

echo
echo "Done. Activate with: source .venv-h3/bin/activate"
echo "Next: bash download_h3_models.sh"
