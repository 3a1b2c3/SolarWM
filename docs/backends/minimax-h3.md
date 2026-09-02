# MiniMax-H3

The MiniMax-H3 backend supports Stage0.5 flow-matching LoRA training,
inference, and preencoding for 158-frame video.

## Availability and data

| Route | Data | Status |
|---|---|---|
| Stage0.5 | `minimax-h3-158f-768p-nomind-v1` latent | Code and weights available; latent upload coming soon |
| Inference | Same latent generation | Code and weights available; latent upload coming soon |
| Preencoding | raw-WDS input | Available |

H3 training and inference use preencoded data only. Track the payload in the
[latent-WDS release list](../latent-wds.md). Use
[Download and access](../data-access.md) only when preparing a new latent
generation from raw-WDS.

## Setup

Activate the H3 environment from
[Runtime environments](../../environments/README.md), then set:

```bash
export SOLAR_REPO=/path/to/SolarWM
export SOLAR_MODEL_ROOT=/path/to/SolarWM-models
export SOLAR_DATA_ROOT=/path/to/SolarWM-Data/releases-v1
export SOLAR_OUTPUT_ROOT=/path/to/outputs
cd "$SOLAR_REPO"
```

Download the H3 base model and released adapter:

```bash
python -m pip install --upgrade huggingface_hub
hf download junchaoh-cs/SolarWM \
  --include "SolarWM-h3-33B-*/**" \
  --local-dir "$SOLAR_MODEL_ROOT"
```

Set the paths used below:

```bash
export H3_BASE="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-base"
export H3_SUPPORT="$SOLAR_DATA_ROOT/latent-wds/minimax-h3-158f-768p-nomind-v1/support"
```

The H3 latent download includes `h3_silence_153_158_170.safetensors` and
`encoder_contract.json` inside its `support/` directory. They are part of the
same latent package and do not require a separate download.

## Stage0.5 training

The following command uses two eight-GPU nodes. Set `NODE_RANK=0` on the first
node and `NODE_RANK=1` on the second.

```bash
export NODE_RANK=0
export MASTER_ADDR=hostname-or-ip-of-node-0
export MASTER_PORT=29500

torchrun --nnodes=2 --node-rank="$NODE_RANK" --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m solarwm train \
  --config configs/examples/minimax_h3/stage0p5-158f-lora384-sp2.yaml \
  --set distributed.world_size=16 \
  --set train.global_batch_size=8 \
  --set validation.sample_count=16 \
  --set model.checkpoint_path="$H3_BASE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage0p5-158f"
```

Run `solarwm config resolve` with the same config and `--set` arguments before
training if you want to inspect the resolved configuration.

## Inference

Inference uses the same latent generation and support files:

```bash
torchrun --standalone --nproc-per-node=8 -m solarwm infer \
  --config configs/examples/minimax_h3/infer-158f-lora384-sp2.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set checkpoint.resume_from="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-bid-stage0p5-158f" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage0p5-158f-infer"
```

## Optional preencoding

To create a new H3 latent generation from raw-WDS:

```bash
torchrun --standalone --nproc-per-node=8 -m solarwm preencode \
  --config configs/examples/minimax_h3/preencode-158f.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set preencode.output_root="$SOLAR_OUTPUT_ROOT/preencoded/minimax-h3-158f-768p-nomind-v1" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-preencode-158f"
```

## Model and camera conventions

The H3 latent format stores absolute C2W cameras and normalized intrinsics. The
reader converts them to first-frame-relative W2C before model conditioning.
Released H3 configs apply the `logd4` translation transform.

## License

The MiniMax-H3 base model and released adapter remain subject to the MiniMax-H3
Community License included with the model packages. Review that license before
download, use, or redistribution; the SolarWM code license does not replace it.
