#!/bin/bash
# Run the bundled MiniMax-H3 racer examples end to end -- no dataset, no arguments needed.
#
# Every case here is i2v (the fl2va workflow: first frame + prompt), each at the source
# image's native resolution (832x480 for racer, 640x352 for left/right):
#
#   racer       examples/first_frame.png        832x480  straight
#   left/right  examples/racer/Screenshot.png   640x352  steers toward wall/fence
#
# Share examples/racer/prompt*.txt.
#
# Usage:
#   bash run_h3_examples.sh                    default: racer + left + right
#   bash run_h3_examples.sh racer              just the straight racer case
#   bash run_h3_examples.sh left               just the left case
#   bash run_h3_examples.sh right              just the right case
#   bash run_h3_examples.sh racer --steps 50   extra flags pass through to h3_infer.py
#
# Defaults are deliberately small (30 steps / 61 frames) so a first run fails fast
# instead of after a long generation. Override with --steps / --num-frames.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

WHICH="steer"
case "${1:-}" in
  racer|left|right|steer) WHICH="$1"; shift ;;
esac

SMALL=(--steps 30 --num-frames 61)
PROMPT="$HERE/examples/racer/prompt.txt"
IMG_RACER="$HERE/examples/first_frame.png"
IMG_SHOT="$HERE/examples/racer/Screenshot.png"

for path in "$PROMPT" "$IMG_RACER" "$IMG_SHOT"; do
  if [ ! -e "$path" ]; then
    echo "ERROR: example asset missing: $path" >&2
    exit 1
  fi
done

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
