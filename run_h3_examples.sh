#!/bin/bash
# Run the bundled MiniMax-H3 examples end to end -- no dataset, no arguments needed.
#
# Every case here is i2v (the fl2va workflow: first frame + prompt). Each image is
# run at its own native resolution, so nothing is rescaled:
#
#   racer       examples/first_frame.png        832x480
#   screenshot  examples/racer/Screenshot.png   640x352  (RGBA, converted to RGB)
#   all         both, in sequence
#
# Both share examples/racer/prompt.txt.
#
# Usage:
#   bash run_h3_examples.sh                    racer
#   bash run_h3_examples.sh screenshot         the 640x352 case
#   bash run_h3_examples.sh all                both
#   bash run_h3_examples.sh racer --steps 50   extra flags pass through to h3_infer.py
#
# Defaults are deliberately small (30 steps / 61 frames) so a first run fails fast
# instead of after a long generation. Override with --steps / --num-frames.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

WHICH="racer"
case "${1:-}" in
  racer|screenshot|man|all) WHICH="$1"; shift ;;
esac

SMALL=(--steps 30 --num-frames 61)
PROMPT="$HERE/examples/racer/prompt.txt"
IMG_RACER="$HERE/examples/first_frame.png"
IMG_SHOT="$HERE/examples/racer/Screenshot.png"
# The man case has no source image, so it runs t2va (prompt only) rather than fl2va.
PROMPT_MAN="$HERE/examples/man/prompt.txt"

for path in "$PROMPT" "$IMG_RACER" "$IMG_SHOT" "$PROMPT_MAN"; do
  if [ ! -e "$path" ]; then
    echo "ERROR: example asset missing: $path" >&2
    exit 1
  fi
done

# $1 name, $2 image, $3 width, $4 height, rest -> passthrough
run_case() {
  local name="$1" image="$2" width="$3" height="$4"
  shift 4
  echo "============================================================"
  echo "Example: $name (fl2va i2v -- ${width}x${height})"
  echo "============================================================"
  bash "$HERE/run_h3_infer.sh" \
    --image "$image" \
    --prompt-file "$PROMPT" \
    --width "$width" \
    --height "$height" \
    --name "$name" \
    "${SMALL[@]}" \
    "$@"
}

run_man() {
  echo "============================================================"
  echo "Example: man (t2va -- prompt only, no source image)"
  echo "============================================================"
  bash "$HERE/run_h3_infer.sh" \
    --prompt-file "$PROMPT_MAN" \
    --width 832 --height 480 \
    --name man \
    "${SMALL[@]}" \
    "$@"
}

case "$WHICH" in
  racer)      run_case racer "$IMG_RACER" 832 480 "$@" ;;
  screenshot) run_case screenshot "$IMG_SHOT" 640 352 "$@" ;;
  man)        run_man "$@" ;;
  all)
    run_case racer "$IMG_RACER" 832 480 "$@"
    echo
    run_case screenshot "$IMG_SHOT" 640 352 "$@"
    echo
    run_man "$@"
    ;;
esac

echo
echo "Outputs under: $HERE/outputs/h3/"
ls -lh "$HERE/outputs/h3/" 2>/dev/null || true
