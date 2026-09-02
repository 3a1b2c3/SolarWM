"""Strict validation for the supported MiniMax-H3 Stage0.5 profile."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from solarwm.data import resolve_index_path
from solarwm.errors import ConfigurationError

from .camera import h3_fused_prope_contract
from .codec import H3_PREENCODE_VERSION
from .geometry import validate_stage0p5_geometry


@dataclass(frozen=True)
class H3RunContract:
    """Resolved fields that identify the supported H3 configuration."""

    action: str
    stage: str
    pixel_frames: int
    encoded_latents: int
    sequence_parallel_size: int
    adapter_rank: int
    camera_translation_transform: str
    data_input_mode: str


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"{path}.{key} is required")
    return mapping[key]


def _equal(mapping: Mapping[str, Any], key: str, expected: Any, path: str) -> None:
    observed = _required(mapping, key, path)
    if isinstance(expected, bool):
        matches = observed is expected
    elif isinstance(expected, int):
        matches = (
            isinstance(observed, int) and not isinstance(observed, bool) and observed == expected
        )
    elif isinstance(expected, float):
        matches = (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and float(observed) == expected
        )
    else:
        matches = observed == expected
    if not matches:
        raise ConfigurationError(f"{path}.{key} must be {expected!r}, got {observed!r}")


def _positive_int(mapping: Mapping[str, Any], key: str, path: str) -> int:
    value = _required(mapping, key, path)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{path}.{key} must be a positive integer")
    return value


def _number(mapping: Mapping[str, Any], key: str, path: str) -> float:
    value = _required(mapping, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path}.{key} must be numeric")
    return float(value)


def _close(mapping: Mapping[str, Any], key: str, expected: float, path: str) -> None:
    observed = _number(mapping, key, path)
    if abs(observed - expected) > 1e-12:
        raise ConfigurationError(f"{path}.{key} must be {expected}, got {observed}")


def _nonempty_path(mapping: Mapping[str, Any], key: str, path: str) -> str:
    value = _required(mapping, key, path)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{path}.{key} must be a non-empty path")
    return value


def _validate_model(model: Mapping[str, Any], *, action: str, input_mode: str) -> None:
    if "checkpoint_digest" in model:
        raise ConfigurationError(
            "model does not support removed content-digest fields: ['checkpoint_digest']"
        )
    family = str(_required(model, "family", "model")).strip().lower()
    if family not in {"minimax_h3", "minimax-h3"}:
        raise ConfigurationError(f"model.family must select MiniMax-H3, got {family!r}")
    _nonempty_path(model, "checkpoint_path", "model")
    if action == "preencode":
        codec_identity = str(_required(model, "codec_identity", "model")).strip()
        if not codec_identity:
            raise ConfigurationError("model.codec_identity must be non-empty")
    for key, expected in (
        ("architecture", "minimax-h3-33b"),
        ("torch_dtype", "bfloat16"),
        ("camera_attention_mode", "fused_prope"),
        ("camera_translation_transform", "logd4"),
        ("camera_intrinsics_mode", "wan_fixed"),
    ):
        _equal(model, key, expected, "model")
    for key, expected in (
        ("attention_head_dim", 128),
        ("camera_prope_head_dim_start", 96),
        ("camera_prope_head_dim_end", 128),
        ("latent_channels", 24),
        ("latent_height", 48),
        ("latent_width", 84),
        ("rows_per_latent", 1008),
        ("num_frames_per_block", 5),
        ("max_prior_clean_chunks", 5),
    ):
        _equal(model, key, expected, "model")
    if action == "preencode":
        return
    for key, expected in (
        ("training_mode", "lora"),
        ("transformer_subfolder", "transformer"),
        ("transformer_device_map", None),
        ("attention_backend", "flash"),
        ("load_conditioners", False),
    ):
        _equal(model, key, expected, "model")
    adapter = _mapping(model, "adapter")
    for key, expected in (
        ("type", "lora"),
        ("target", "block_qkvo_ffn"),
        ("rank", 384),
        ("alpha", 384),
        ("dropout", 0.0),
        ("bias", "none"),
        ("dtype", "bfloat16"),
        ("expected_target_linear_modules", 312),
        ("expected_trainable_parameters", 2_075_394_048),
    ):
        _equal(adapter, key, expected, "model.adapter")


def _validate_data(data: Mapping[str, Any], *, action: str) -> str:
    input_mode = str(_required(data, "input_mode", "data")).strip().lower()
    allowed = {"raw"} if action == "preencode" else {"preencoded"}
    if input_mode not in allowed:
        raise ConfigurationError(
            f"MiniMax-H3 {action} data.input_mode must be one of {sorted(allowed)!r}; "
            f"got {input_mode!r}"
        )
    transport = _mapping(data, "transport")
    kind = str(_required(transport, "kind", "data.transport")).strip().lower()
    if kind not in {"local", "gcs"}:
        raise ConfigurationError("data.transport.kind must be local or gcs")
    root = _nonempty_path(transport, "root", "data.transport")
    if kind == "local":
        if not root.startswith("/"):
            raise ConfigurationError("local data.transport.root must be absolute")
        if data.get("index_root") is not None:
            index_root = _nonempty_path(data, "index_root", "data")
            if not index_root.startswith("/"):
                raise ConfigurationError("data.index_root must be absolute")
    else:
        if not root.startswith("gs://"):
            raise ConfigurationError("gcs data.transport.root must be a gs:// URI")
        cache_dir = _nonempty_path(transport, "cache_dir", "data.transport")
        if not cache_dir.startswith("/"):
            raise ConfigurationError("data.transport.cache_dir must be absolute")
        if _number(transport, "cache_max_gib", "data.transport") <= 0:
            raise ConfigurationError("data.transport.cache_max_gib must be positive")
        index_root = _nonempty_path(data, "index_root", "data")
        if not index_root.startswith("/"):
            raise ConfigurationError(
                "gcs transport requires absolute data.index_root for staged controls"
            )
    index_fields = ("index",) if action == "preencode" else ("train_index", "test_index")
    for field in index_fields:
        _nonempty_path(data, field, "data")
        try:
            resolve_index_path(data, field)
        except Exception as exc:
            raise ConfigurationError(f"cannot resolve data.{field}: {exc}") from exc
    for key, expected in (
        ("pixel_frames", 158),
        ("encoded_latents", 47),
        ("train_target_latents", 47),
        ("height", 768),
        ("width", 1344),
        ("latent_channels", 24),
        ("latent_height", 48),
        ("latent_width", 84),
        ("frame_sampling", "contiguous"),
        ("source_fps_policy", "audit_only"),
        ("random_start", False),
        ("resolution_label", "default"),
    ):
        _equal(data, key, expected, "data")
    _close(data, "max_relative_translation", 20.0, "data")
    _close(data, "max_camera_absolute_value", 20.0, "data")
    validate_stage0p5_geometry(
        pixel_frames=int(data["pixel_frames"]),
        encoded_latents=int(data["encoded_latents"]),
        height=int(data["height"]),
        width=int(data["width"]),
        latent_channels=int(data["latent_channels"]),
        latent_height=int(data["latent_height"]),
        latent_width=int(data["latent_width"]),
    )
    if action == "train":
        workers = data.get("num_workers", 1)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ConfigurationError("data.num_workers must be a positive integer")
        if data.get("prefetch_factor") is not None:
            prefetch = data["prefetch_factor"]
            if isinstance(prefetch, bool) or not isinstance(prefetch, int) or prefetch < 1:
                raise ConfigurationError("data.prefetch_factor must be a positive integer")
        shard_prefetch = data.get("gcs_prefetch_shards", 0)
        if (
            isinstance(shard_prefetch, bool)
            or not isinstance(shard_prefetch, int)
            or shard_prefetch < 0
        ):
            raise ConfigurationError("data.gcs_prefetch_shards must be a non-negative integer")
    if input_mode == "preencoded":
        _equal(data, "preencode_version", H3_PREENCODE_VERSION, "data")
        _equal(data, "dataset_name", "h3_preencoded_wds", "data")
        _nonempty_path(data, "silence_latents_path", "data")
        _nonempty_path(data, "encoder_contract_path", "data")
    else:
        _equal(data, "fixed_start_from_index", True, "data")
    return input_mode


def _validate_route(train: Mapping[str, Any]) -> None:
    for key, expected in (
        ("stage", "stage0p5"),
        ("causal_mode", "bidirectional"),
        ("objective", "flow_matching"),
    ):
        _equal(train, key, expected, "train")


def _validate_training(
    train: Mapping[str, Any],
    distributed: Mapping[str, Any],
) -> None:
    for key, expected in (
        ("precision", "bfloat16"),
        ("micro_batch_size", 1),
        ("gradient_accumulation_steps", 1),
        ("video_timestep_shift", 12.0),
        ("audio_timestep_shift", 3.0),
        ("keyframe_noise_augmentation", 0.999),
        ("audio_loss_weight", 0.0),
    ):
        _equal(train, key, expected, "train")
    optimizer = _mapping(train, "optimizer")
    for key, expected in (
        ("name", "fp32_master_adamw"),
        ("learning_rate", 1.0e-4),
        ("warmup_steps", 500),
    ):
        _equal(optimizer, key, expected, "train.optimizer")
    fsdp = _mapping(train, "fsdp")
    for key, expected in (
        ("sharding_strategy", "FULL_SHARD"),
        ("activation_checkpointing", True),
        ("preserve_checkpoint_dtype", True),
        ("param_dtype", None),
        ("reduce_dtype", "float32"),
        ("buffer_dtype", None),
    ):
        _equal(fsdp, key, expected, "train.fsdp")

    world_size = _positive_int(distributed, "world_size", "distributed")
    sequence_parallel = _positive_int(distributed, "sequence_parallel_size", "distributed")
    _equal(distributed, "rank_partition", "node_shard", "distributed")
    _equal(distributed, "context_parallel_size", 1, "distributed")
    _equal(distributed, "sp_peers_share_sample", True, "distributed")
    _equal(distributed, "sp_peers_share_rng", True, "distributed")
    if sequence_parallel != 2:
        raise ConfigurationError("MiniMax-H3 Stage0.5 requires sequence_parallel_size=2")
    if world_size % sequence_parallel:
        raise ConfigurationError("distributed.world_size must be divisible by SP size")
    # The world size is NOT pinned. SP=2, micro batch and accumulation above are
    # architectural; the number of ranks is a deployment choice, and the global batch
    # identity below is what actually has to hold. The checked example retains the
    # 256-rank release profile, while another valid scale reproduces the recipe rather
    # than the released optimisation trajectory.
    computed_global_batch = (
        world_size
        // sequence_parallel
        * int(train["micro_batch_size"])
        * int(train["gradient_accumulation_steps"])
    )
    if computed_global_batch != int(train["global_batch_size"]):
        raise ConfigurationError(
            "global batch mismatch: (world_size / SP) * micro_batch * grad_accum "
            f"is {computed_global_batch}, configured {train['global_batch_size']}"
        )


def _validate_validation(validation: Mapping[str, Any], *, action: str) -> None:
    del action
    _positive_int(validation, "sample_count", "validation")
    for name in ("selection_seed", "noise_seed"):
        value = _required(validation, name, "validation")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigurationError(f"validation.{name} must be a non-negative integer")
    for key, expected in (
        ("pixel_frames", 158),
        ("latent_frames", 47),
        ("fps", 24),
        ("num_inference_steps", 30),
        ("passes", ["live", "ema"]),
    ):
        _equal(validation, key, expected, "validation")
    smoke_step = _required(validation, "smoke_step", "validation")
    if isinstance(smoke_step, bool) or not isinstance(smoke_step, int) or smoke_step < 0:
        raise ConfigurationError("validation.smoke_step must be a non-negative integer")
    _close(validation, "max_relative_translation", 20.0, "validation")
    _close(validation, "max_camera_absolute_value", 20.0, "validation")


def _validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    _equal(checkpoint, "save_optimizer", True, "checkpoint")
    ema = _mapping(checkpoint, "ema")
    for key, expected in (
        ("enabled", True),
        ("dtype", "float32"),
        ("sharded", True),
        ("decay", 0.9999),
        ("start_step", 0),
        ("update_every_steps", 1),
    ):
        _equal(ema, key, expected, "checkpoint.ema")


def _validate_inference_distributed(distributed: Mapping[str, Any]) -> None:
    world_size = _positive_int(distributed, "world_size", "distributed")
    sequence_parallel = _positive_int(
        distributed,
        "sequence_parallel_size",
        "distributed",
    )
    if sequence_parallel != 2 or world_size % sequence_parallel:
        raise ConfigurationError("H3 inference requires a world divisible by SP2")
    for key, expected in (
        ("context_parallel_size", 1),
        ("sp_peers_share_sample", True),
        ("sp_peers_share_rng", True),
    ):
        _equal(distributed, key, expected, "distributed")


def validate_h3_config(config: Mapping[str, Any]) -> H3RunContract:
    """Validate one config against the supported H3 profile."""

    if not isinstance(config, Mapping):
        raise ConfigurationError("config must be a mapping")
    action = str(config.get("action", "")).strip().lower()
    if action not in {"train", "infer", "preencode"}:
        raise ConfigurationError("action must be train, infer, or preencode")
    model = _mapping(config, "model")
    data = _mapping(config, "data")
    metadata = config.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ConfigurationError("metadata must be a mapping")
    declared_input_mode = str(data.get("input_mode", "")).strip().lower()
    _validate_model(model, action=action, input_mode=declared_input_mode)
    input_mode = _validate_data(data, action=action)

    if action != "preencode":
        train = _mapping(config, "train")
        _validate_route(train)
        _validate_validation(_mapping(config, "validation"), action=action)
        if action == "train":
            _validate_training(train, _mapping(config, "distributed"))
            _validate_checkpoint(_mapping(config, "checkpoint"))
        else:
            _validate_inference_distributed(_mapping(config, "distributed"))
            _nonempty_path(_mapping(config, "checkpoint"), "resume_from", "checkpoint")
    else:
        preencode = _mapping(config, "preencode")
        _equal(preencode, "codec_protocol", "solarwm.minimax_h3.codec.v1", "preencode")
        _nonempty_path(preencode, "output_root", "preencode")

    runtime = _mapping(config, "runtime")
    _nonempty_path(runtime, "output_dir", "runtime")
    contract = h3_fused_prope_contract()
    if contract["camera_prope_head_slice"] != [96, 128]:
        raise ConfigurationError("internal H3 camera suffix contract is inconsistent")
    return H3RunContract(
        action=action,
        stage="preencode" if action == "preencode" else "stage0p5",
        pixel_frames=158,
        encoded_latents=47,
        sequence_parallel_size=(
            0
            if action == "preencode"
            else int(_mapping(config, "distributed")["sequence_parallel_size"])
        ),
        adapter_rank=0 if action == "preencode" else 384,
        camera_translation_transform="logd4",
        data_input_mode=input_mode,
    )


__all__ = ["H3RunContract", "validate_h3_config"]
