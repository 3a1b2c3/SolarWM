#!/bin/bash
# Generate straight/left/right racer examples with SolarWM's h3_infer.py
# (one model load via --mind-batch), then burn a WASDIJKL key overlay onto
# each via ../overlay_keys.py. See build_direction_actions.py's docstring:
# the action matrix fed to the overlay is a constant per-clip direction
# LABEL (one key held the whole clip), not real per-frame action data --
# SolarWM's base pipeline has no action-conditioning input at all, so
# direction only comes from the prompt text (prompt.txt / prompt_left.txt /
# prompt_right.txt).
#
#   bash examples/racer/run_examples.sh [--num-frames N] [--steps N]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv-h3/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found -- run 'bash setup_env_h3.sh' first." >&2
  exit 1
fi

NUM_FRAMES=158
STEPS=50
while [ $# -gt 0 ]; do
  case "$1" in
    --num-frames) NUM_FRAMES="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

OUT_DIR="$ROOT/outputs/racer"
mkdir -p "$OUT_DIR"
IMAGE="$HERE/Screenshot.png"

MANIFEST="$OUT_DIR/manifest.json"
"$PY" - "$MANIFEST" "$IMAGE" "$HERE" "$OUT_DIR" "$NUM_FRAMES" "$STEPS" <<'PYEOF'
import json, sys
manifest_path, image, racer_dir, out_dir, num_frames, steps = sys.argv[1:]
from pathlib import Path
racer_dir = Path(racer_dir)
entries = []
for name, prompt_file in (("straight", "prompt.txt"), ("left", "prompt_left.txt"), ("right", "prompt_right.txt")):
    entries.append({
        "prompt": (racer_dir / prompt_file).read_text(encoding="utf-8").strip(),
        "image": image,
        "num_frames": int(num_frames),
        "steps": int(steps),
        "out_dir": out_dir,
        "name": name,
    })
Path(manifest_path).write_text(json.dumps(entries))
print(f"wrote {manifest_path}: {len(entries)} entries")
PYEOF

echo "[racer] generating straight/left/right (one model load)..."
"$PY" "$ROOT/h3_infer.py" --mind-batch "$MANIFEST"

echo "[racer] overlaying WASDIJKL key on each..."
for d in straight left right; do
  "$PY" "$HERE/build_direction_actions.py" \
    --direction "$d" --num-frames "$NUM_FRAMES" \
    --out "$OUT_DIR/${d}_actions.npy"
  "$PY" "$ROOT/examples/overlay_keys.py" \
    --video "$OUT_DIR/$d.mp4" --actions "$OUT_DIR/${d}_actions.npy" \
    --out "$OUT_DIR/${d}_overlay.mp4"
done

echo
echo "Done. Outputs:"
echo "  $OUT_DIR/straight_overlay.mp4"
echo "  $OUT_DIR/left_overlay.mp4"
echo "  $OUT_DIR/right_overlay.mp4"
