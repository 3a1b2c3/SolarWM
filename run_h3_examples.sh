#!/bin/bash
# Run the bundled MiniMax-H3 racer examples end to end -- no dataset, no arguments needed.
# Each generated clip gets a WASDIJKL key overlay burned in afterward (see NOTE below).
#
# Every case here is i2v (the fl2va workflow: first frame + prompt), each at the source
# image's native resolution (832x480 for racer, 640x352 for left/right):
#
#   racer       examples/first_frame.png        832x480  straight
#   left/right  examples/racer/Screenshot.png   640x352  steers toward wall/fence
#
# Share examples/racer/prompt*.txt.
#
# NOTE on the overlay: SolarWM's h3_infer.py has no per-frame action-conditioning
# input at all (prompt + first frame only) -- there is no real per-frame steering
# signal to overlay. So each case's overlay is a constant key (W/A/D) held for the
# whole clip, labeling which prompt variant produced it, built by
# examples/racer/build_direction_actions.py and drawn by examples/overlay_keys.py.
#
# Usage:
#   bash run_h3_examples.sh                    default: racer + left + right
#   bash run_h3_examples.sh racer              just the straight racer case
#   bash run_h3_examples.sh left               just the left case
#   bash run_h3_examples.sh right              just the right case
#   bash run_h3_examples.sh racer --steps 50   extra flags pass through to h3_infer.py
#     (NOTE: overriding --num-frames here will NOT change the overlay -- it's still
#     built against NUM_FRAMES below. Edit NUM_FRAMES if you need them to match.)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

WHICH="steer"
case "${1:-}" in
  racer|left|right|steer) WHICH="$1"; shift ;;
esac

# 124 = 17*7+5, the smallest value h3_infer.py's frame-count constraint (17n+5 in
# [120,360]) actually accepts -- the old default of 61 wasn't valid and was getting
# silently rounded up to 120 by h3_infer.py, which would have mismatched whatever
# the overlay was built against. Keep this and the overlay in sync.
STEPS=30
NUM_FRAMES=124
SMALL=(--steps "$STEPS" --num-frames "$NUM_FRAMES")
PROMPT="$HERE/examples/racer/prompt.txt"
IMG_RACER="$HERE/examples/first_frame.png"
IMG_SHOT="$HERE/examples/racer/Screenshot.png"

for path in "$PROMPT" "$IMG_RACER" "$IMG_SHOT"; do
  if [ ! -e "$path" ]; then
    echo "ERROR: example asset missing: $path" >&2
    exit 1
  fi
done

VENV="$HERE/.venv-h3"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found -- run 'bash setup_env_h3.sh' first." >&2
  exit 1
fi

# $1 name (matches h3_infer.py's --name, so {name}.mp4 in outputs/h3/)
# $2 direction (straight/left/right, for build_direction_actions.py)
imprint_keys() {
  local name="$1" direction="$2"
  local out_dir="$HERE/outputs/h3"
  echo "[overlay] imprinting keys on $name ($direction)..."
  "$PY" "$HERE/examples/racer/build_direction_actions.py" \
    --direction "$direction" --num-frames "$NUM_FRAMES" \
    --out "$out_dir/${name}_actions.npy"
  "$PY" "$HERE/examples/overlay_keys.py" \
    --video "$out_dir/${name}.mp4" \
    --actions "$out_dir/${name}_actions.npy" \
    --out "$out_dir/${name}_overlay.mp4"
}

run_racer() {
  echo "============================================================"
  echo "Example: racer (fl2va i2v -- 832x480, straight)"
  echo "============================================================"
  bash "$HERE/run_h3_infer.sh" \
    --image "$IMG_RACER" \
    --prompt-file "$PROMPT" \
    --width 832 --height 480 \
    --name racer \
    "${SMALL[@]}" \
    "$@"
  imprint_keys racer straight
}

run_left() {
  echo "============================================================"
  echo "Example: left (fl2va i2v -- 640x352, buggy steers left)"
  echo "============================================================"
  # Uses Screenshot.png, the image that actually matches the racer prompt.
  bash "$HERE/run_h3_infer.sh" \
    --image "$IMG_SHOT" \
    --prompt-file "$HERE/examples/racer/prompt_left.txt" \
    --width 640 --height 352 \
    --name racer_left \
    "${SMALL[@]}" \
    "$@"
  imprint_keys racer_left left
}

run_right() {
  echo "============================================================"
  echo "Example: right (fl2va i2v -- 640x352, buggy steers right)"
  echo "============================================================"
  # Uses Screenshot.png, the image that actually matches the racer prompt.
  bash "$HERE/run_h3_infer.sh" \
    --image "$IMG_SHOT" \
    --prompt-file "$HERE/examples/racer/prompt_right.txt" \
    --width 640 --height 352 \
    --name racer_right \
    "${SMALL[@]}" \
    "$@"
  imprint_keys racer_right right
}

case "$WHICH" in
  racer) run_racer "$@" ;;
  left)  run_left "$@" ;;
  right) run_right "$@" ;;
  steer)
    run_racer "$@"
    echo
    run_left "$@"
    echo
    run_right "$@"
    ;;
esac

echo
echo "Outputs under: $HERE/outputs/h3/"
ls -lh "$HERE/outputs/h3/" 2>/dev/null || true
