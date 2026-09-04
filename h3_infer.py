"""Standalone MiniMax-H3 inference: prompt (+ optional first frame) -> video WITH audio.

This deliberately bypasses SolarWM's own `solarwm infer`, which replays samples from a
WDS validation index and therefore needs the (currently gated) SolarWM-Data dataset. The
H3 base checkpoint is a complete diffusers modular pipeline, so it can be driven directly
with nothing but a prompt and an image.

API notes, taken from introspecting the installed diffusers 0.40.0 rather than from docs:
  * MiniMaxH3Blocks exposes workflows t2va / fl2va / ref2va.
      t2va   -> prompt only
      fl2va  -> prompt + image and/or last_image
      ref2va -> prompt + references
  * Passing workflow= to from_pretrained matters for more than tidiness: without it the
    loader pulls BOTH 61.7 GB transformer partitions.
  * fl2va accepts: image, last_image, height, width, prompt, num_frames, generator,
    latents, audio_latents, num_inference_steps, attention_kwargs, output_type
  * and returns, among others: videos, audio, sampling_rate.

Audio is genuinely generated here -- these are "VA" (video+audio) workflows and the
pipeline owns an audio_vae. SolarWM's own H3 path only ever encodes silence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import MiniMaxH3ModularPipeline

DEFAULT_MODEL_PATH = "/localhome/kschmid/SolarWM-models/SolarWM-h3-33B-base"

# The H3 video VAE only encodes frame counts of the form 17*n+5, and the modular
# pipeline additionally requires the rounded value to land in [120, 360] (5-15s @ 24fps).
H3_FRAME_MIN = 120
H3_FRAME_MAX = 360
H3_FRAME_STEP = 17
H3_FRAME_OFFSET = 5


def round_to_h3_frames(num_frames: int) -> int:
    """Round up to the next 17*n+5 the VAE can encode, clamped into [120, 360]."""
    n = max(0, -(-(num_frames - H3_FRAME_OFFSET) // H3_FRAME_STEP))
    rounded = H3_FRAME_STEP * n + H3_FRAME_OFFSET
    if rounded < H3_FRAME_MIN:
        rounded = H3_FRAME_STEP * ((H3_FRAME_MIN - H3_FRAME_OFFSET + H3_FRAME_STEP - 1) // H3_FRAME_STEP) + H3_FRAME_OFFSET
    if rounded > H3_FRAME_MAX:
        rounded = H3_FRAME_STEP * ((H3_FRAME_MAX - H3_FRAME_OFFSET) // H3_FRAME_STEP) + H3_FRAME_OFFSET
    return rounded


def to_uint8_frames(videos) -> np.ndarray:
    """Normalise whatever the pipeline returns into (F, H, W, 3) uint8."""
    array = videos.float().cpu().numpy() if torch.is_tensor(videos) else np.asarray(videos)
    # Drop a leading batch dim if present.
    while array.ndim > 4:
        array = array[0]
    if array.ndim != 4:
        raise RuntimeError(f"unexpected video array shape: {array.shape}")
    # Channels-first (F, 3, H, W) -> channels-last.
    if array.shape[1] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (0, 2, 3, 1))
    if array.dtype != np.uint8:
        # Models emit either [0,1] or [-1,1]; detect and rescale.
        low = float(array.min())
        array = (array + 1.0) / 2.0 if low < -0.01 else array
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255.0).round().astype(np.uint8)
    return array


def to_int16_audio(audio) -> np.ndarray:
    """Normalise audio into (samples, channels) int16."""
    array = audio.float().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio)
    while array.ndim > 2:
        array = array[0]
    if array.ndim == 1:
        array = array[:, None]
    # (channels, samples) -> (samples, channels); channels is the small axis.
    if array.shape[0] < array.shape[1]:
        array = array.T
    if array.dtype != np.int16:
        array = np.clip(array, -1.0, 1.0)
        array = (array * 32767.0).round().astype(np.int16)
    return array


def write_wav(path: Path, audio: np.ndarray, sampling_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(audio.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(int(sampling_rate))
        handle.writeframes(audio.tobytes())


def mux(video_path: Path, wav_path: Path, out_path: Path) -> bool:
    """Combine silent video + wav. Uses imageio-ffmpeg's bundled binary so this works
    even when ffmpeg isn't on PATH."""
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"
    cmd = [
        ffmpeg, "-y", "-i", str(video_path), "-i", str(wav_path),
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"WARNING: mux failed ({result.returncode}); silent video and wav kept separately.")
        print(result.stderr.decode(errors="replace")[-600:], file=sys.stderr)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="MiniMax-H3 video+audio inference.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", type=Path, help="read the prompt from a file, e.g. examples/racer/prompt.txt")
    parser.add_argument("--image", type=Path, help="first frame; omit for the t2va workflow")
    parser.add_argument("--last-image", type=Path)
    parser.add_argument("--workflow", choices=("t2va", "fl2va", "ref2va"))
    parser.add_argument("--num-frames", type=int, default=158)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/h3"))
    parser.add_argument("--name", default="h3_sample")
    args = parser.parse_args()

    if args.prompt_file:
        args.prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if not args.prompt:
        print("ERROR: pass --prompt or --prompt-file", file=sys.stderr)
        return 2

    workflow = args.workflow or ("fl2va" if args.image or args.last_image else "t2va")
    if workflow == "fl2va" and not (args.image or args.last_image):
        print("ERROR: fl2va needs --image and/or --last-image", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device visible.", file=sys.stderr)
        return 2

    rounded_frames = round_to_h3_frames(args.num_frames)
    if rounded_frames != args.num_frames:
        print(
            f"NOTE: --num-frames {args.num_frames} isn't encodable by the H3 VAE "
            f"(needs 17*n+5 in [120,360]); using {rounded_frames} instead.",
            flush=True,
        )
        args.num_frames = rounded_frames

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_path} (workflow={workflow})...", flush=True)
    # workflow= keeps the loader from pulling both transformer partitions.
    pipe = MiniMaxH3ModularPipeline.from_pretrained(args.model_path, workflow=workflow)
    # from_pretrained only builds the pipeline from its config -- every component stays
    # None until load_components() runs. Skipping this surfaces later as a confusing
    # "'NoneType' object has no attribute 'image_processor'" inside the encoder block.
    print("Loading components (this pulls the transformer partition; expect it to be slow)...", flush=True)
    # No workflow= here: from_pretrained already selected it, collapsing the blocks into
    # SequentialPipelineBlocks, which has no _workflow_map. Passing it again raises
    # "workflows is not supported because _workflow_map is not set".
    pipe.load_components(torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    call_kwargs = {
        "prompt": args.prompt,
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "output_type": "np",
    }
    if args.image:
        call_kwargs["image"] = Image.open(args.image).convert("RGB")
    if args.last_image:
        call_kwargs["last_image"] = Image.open(args.last_image).convert("RGB")

    print(f"Generating {args.num_frames} frames at {args.width}x{args.height}, {args.steps} steps...", flush=True)
    output = pipe(**call_kwargs)

    videos = getattr(output, "videos", None)
    audio = getattr(output, "audio", None)
    sampling_rate = getattr(output, "sampling_rate", None)
    if videos is None:
        raise RuntimeError(f"pipeline returned no 'videos'; got {type(output).__name__}")

    frames = to_uint8_frames(videos)
    silent_path = args.output_dir / f"{args.name}_silent.mp4"
    final_path = args.output_dir / f"{args.name}.mp4"

    import imageio.v2 as imageio

    imageio.mimwrite(str(silent_path), frames, fps=args.fps, quality=8)
    print(f"video: {silent_path}  ({frames.shape[0]} frames)")

    if audio is None or sampling_rate is None:
        print("NOTE: pipeline returned no audio; keeping the silent mp4 only.")
        silent_path.replace(final_path)
        print(f"done: {final_path}")
        return 0

    samples = to_int16_audio(audio)
    wav_path = args.output_dir / f"{args.name}.wav"
    write_wav(wav_path, samples, sampling_rate)
    print(f"audio: {wav_path}  ({samples.shape[0]} samples @ {sampling_rate} Hz, {samples.shape[1]}ch)")

    if mux(silent_path, wav_path, final_path):
        silent_path.unlink()
        print(f"done: {final_path}  (video + audio)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
