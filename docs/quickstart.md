# Quickstart

This guide runs the Wan2.2-5B Stage0.5 153f recipe with released preencoded
latents. This route does not require raw-WDS. For another model or training
stage, finish the common setup below and continue with its
[backend guide](#next-steps).

## 1. Set up the Wan environment

Wan, LTX, and MiniMax-H3 use separate environments. Follow the Wan setup in
[Runtime environments](../environments/README.md), activate it, and install
SolarWM:

```bash
export SOLAR_REPO=/path/to/SolarWM
cd "$SOLAR_REPO"
python -m pip install -e .
solarwm environment probe
```

## 2. Download weights and data

Choose local directories for the model, data, and outputs:

```bash
export SOLAR_MODEL_ROOT=/path/to/SolarWM-models
export SOLAR_DATA_HOME=/path/to/SolarWM-Data
export SOLAR_DATA_ROOT="$SOLAR_DATA_HOME/releases-v1"
export SOLAR_OUTPUT_ROOT=/path/to/outputs
mkdir -p "$SOLAR_MODEL_ROOT" "$SOLAR_DATA_HOME" "$SOLAR_OUTPUT_ROOT"
```

Download the Wan2.2-5B base model, its 81f initialization checkpoint, and the
public data repository:

```bash
python -m pip install --upgrade huggingface_hub

hf download junchaoh-cs/SolarWM \
  --include "SolarWM-5B-base/**" \
  --include "SolarWM-5B-bid-stage0p5-81f/**" \
  --local-dir "$SOLAR_MODEL_ROOT"

hf download junchaoh-cs/SolarWM-Data \
  --repo-type dataset \
  --exclude "SolarWM-Data-Annotation/**" \
  --local-dir "$SOLAR_DATA_HOME"
```

Download
[`wan22-ti2v5b-153f-480p-v1`](https://modelscope.ai/datasets/Junchao-cs/SolarWM-Data_Latent-WDS_wan22-ti2v5b-153f-480p-v1)
and place the downloaded generation at:

```text
$SOLAR_DATA_ROOT/latent-wds/wan22-ti2v5b-153f-480p-v1/
```

The main data repository supplies the matching recipe indexes. The latent
generation supplies the training payload, so raw-WDS is not needed for this
quickstart.

## 3. Check the configuration

Resolve the example with your local paths before starting training:

```bash
solarwm config resolve \
  --config configs/examples/wan22_ti2v_5b/train_stage0p5_fm_153f.yaml \
  --set model.base_path="$SOLAR_MODEL_ROOT/SolarWM-5B-base" \
  --set checkpoint.path="$SOLAR_MODEL_ROOT/SolarWM-5B-bid-stage0p5-81f/model.pt" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set runtime.validate_every=0 \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/wan5-stage0p5-153f"
```

## 4. Launch training

The following command runs the example on one eight-GPU node:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m solarwm train \
  --config configs/examples/wan22_ti2v_5b/train_stage0p5_fm_153f.yaml \
  --set distributed.world_size=8 \
  --set train.global_batch_size=8 \
  --set model.base_path="$SOLAR_MODEL_ROOT/SolarWM-5B-base" \
  --set checkpoint.path="$SOLAR_MODEL_ROOT/SolarWM-5B-bid-stage0p5-81f/model.pt" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set runtime.validate_every=0 \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/wan5-stage0p5-153f"
```

Periodic video validation is disabled in this latent-only quickstart because
the Wan validation recipe reads raw video. The output directory contains the
resolved configuration, launch manifest, and checkpoints. Prepare raw-WDS only
if you later need the full raw corpus, an online-encoding workflow, or raw-video
validation and inference.

## Next steps

- [Wan2.2 TI2V-5B](backends/wan22-ti2v-5b.md): complete Stage0.5, Stage1,
  Stage2, and inference commands.
- [Wan2.2 I2V-A14B](backends/wan22-i2v-a14b.md)
- [LTX-2.5](backends/ltx25.md)
- [MiniMax-H3](backends/minimax-h3.md)
- [Download and access](data-access.md): raw-WDS and preencoded latent options.
