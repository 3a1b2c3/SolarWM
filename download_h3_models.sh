#!/bin/bash
# Download MiniMax-H3 weights for SolarWM.
#
# NOTE, real and worth reading before running: SolarWM does NOT download
# from MiniMaxAI/MiniMax-H3 (the original HF repo) directly -- it uses its
# OWN repackaged checkpoints at junchaoh-cs/SolarWM (per
# docs/backends/minimax-h3.md). This is a different repo from the sibling
# H3-World project's download, which does pull straight from
# MiniMaxAI/MiniMax-H3.
#
# ALSO REAL: docs/backends/minimax-h3.md's own availability table says the
# training/inference latent data (minimax-h3-158f-768p-nomind-v1) has
# "Code and weights available; latent upload coming soon" -- i.e. the model
# weights this script downloads should work, but the DATA needed to
# actually run inference/training might not be published yet. Verify
# against docs/latent-wds.md before assuming a full example run will work
# end-to-end.
#
#   bash download_h3_models.sh                    # download to ./SolarWM-models
#   SOLAR_MODEL_ROOT=/path/to/dest bash download_h3_models.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

SOLAR_MODEL_ROOT="${SOLAR_MODEL_ROOT:-$HERE/../SolarWM-models}"

VENV="$HERE/.venv-h3"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found -- run 'bash setup_env_h3.sh' first." >&2
  exit 1
fi
source "$VENV/bin/activate"

echo "[solarwm-h3-download] upgrading huggingface_hub..."
"$PY" -m pip install --upgrade huggingface_hub

echo "[solarwm-h3-download] downloading junchaoh-cs/SolarWM (SolarWM-h3-33B-*) -> $SOLAR_MODEL_ROOT ..."
mkdir -p "$SOLAR_MODEL_ROOT"
hf download junchaoh-cs/SolarWM \
  --include "SolarWM-h3-33B-*/**" \
  --local-dir "$SOLAR_MODEL_ROOT"

echo
echo "Done. SOLAR_MODEL_ROOT=$SOLAR_MODEL_ROOT"
echo "Expect subfolders like SolarWM-h3-33B-base and SolarWM-h3-33B-bid-stage0p5-158f"
echo "(the latter is what --set checkpoint.resume_from points at for inference)."
echo
echo "Still needed before running inference: the minimax-h3-158f-768p-nomind-v1"
echo "latent data under SOLAR_DATA_ROOT -- see docs/latent-wds.md. That data was"
echo "flagged in SolarWM's own docs as 'upload coming soon' as of this writing;"
echo "verify it's actually published before expecting run_h3_example.sh to work."
