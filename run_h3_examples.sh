#!/bin/bash
# Run the bundled MiniMax-H3 examples end to end -- no dataset, no arguments needed.
#
# Wraps run_h3_infer.sh with the examples/ assets already wired up:
#
#   racer   fl2va  examples/first_frame.png + examples/racer/prompt.txt
#                  (first_frame.png is 832x480, matching the default resolution)
#   t2v     t2va   examples/racer/prompt.txt, prompt only, no conditioning frame
#   both    run racer then t2v
#
# Usage:
#   bash run_h3_examples.sh                 racer (image + prompt)
#   bash run_h3_examples.sh t2v             text only
#   bash run_h3_examples.sh both            both, in sequence
#   bash run_h3_examples.sh racer --steps 50 --num-frames 97
#     ^-- extra flags pass straight through to h3_infer.py
#
# Defaults are deliberately small (30 steps / 61 frames) so a first run fails fast
# instead of after a long generation. Override with --steps / --num-frames.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

WHICH="racer"
case "${1:-}" in
  racer|t2v|both) WHICH="$1"; shift ;;
esac

STEPS_DEFAULT=(--steps 30 --num-frames 61)
IMAGE="$HERE/examples/first_frame.png"
PROMPT="$HERE/examples/racer/prompt.txt"

for path in "$IMAGE" "$PROMPT"; do
  if [ ! -e "$path" ]; then
    echo "ERROR: example asset missing: $path" >&2
    exit 1
  fi
done

run_racer() {
  echo "============================================================"
  echo "Example: racer (fl2va -- first frame + prompt)"
  echo "============================================================"
  bash "$HERE/run_h3_infer.sh" \
    --image "$IMAGE" \
    --prompt-file "$PROMPT" \
    --name racer \
    "${STEPS_DEFAULT[@]}" \
    "$@"
}

run_t2v() {
  echo "============================================================"
  echo "Example: t2v (t2va -- prompt only)"
  echo "============================================================"
  bash "$HERE/run_h3_infer.sh" \
    --prompt-file "$PROMPT" \
    --name racer_t2v \
    "${STEPS_DEFAULT[@]}" \
    "$@"
}

case "$WHICH" in
  racer) run_racer "$@" ;;
  t2v)   run_t2v "$@" ;;
  both)  run_racer "$@"; echo; run_t2v "$@" ;;
esac

echo
echo "Outputs under: $HERE/outputs/h3/"
ls -lh "$HERE/outputs/h3/" 2>/dev/null || true
