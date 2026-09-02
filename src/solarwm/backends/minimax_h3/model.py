"""Official Diffusers MiniMax-H3 adapter for Stage0.5 + fused camera PRoPE.

This module is intentionally heavy and is imported only by runtime actions.
The subclass adds no modules or parameters, so official checkpoint keys remain
unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MINIMAX_H3_MODALITY_NUM,
        MiniMaxH3AttnProcessor,
        MiniMaxH3Transformer3DModel,
        MiniMaxH3TransformerBlock,
        MiniMaxH3TransformerOutput,
        _apply_rotary_emb,
    )
    from diffusers.utils import apply_lora_scale
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - optional runtime
    raise ImportError(
        "MiniMax-H3 requires an H3-compatible Diffusers build and PyTorch runtime"
    ) from exc

from .camera import WAN_FIXED_CX, WAN_FIXED_CY, WAN_FIXED_FX, WAN_FIXED_FY
from .diffusers_compat import patch_minimax_h3_parameter_dtype
from .distributed import (
    get_sp_size,
    is_sequence_parallel_enabled,
    sequence_all_gather,
    sequence_all_to_all,
    shard_stage0p5_packed_sequence,
)

get_parameter_dtype = patch_minimax_h3_parameter_dtype()
FusedPrope = Callable[
    ...,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
]


@dataclass(frozen=True)
class H3AttentionControl:
    attention_mask: Any
    fused_prope: bool | FusedPrope | None
    prope_token_indices: torch.Tensor | None
    prope_frame_ids: torch.Tensor | None
    cam_viewmats: torch.Tensor | None
    cam_K: torch.Tensor | None
    prope_kwargs: dict[str, Any] | None


def _default_prope() -> FusedPrope:
    from .torch_prope import prope_qkv

    return prope_qkv


def _camera_rows(
    tensor: torch.Tensor,
    *,
    name: str,
    batch: int,
    selected_rows: int,
    frame_ids: torch.Tensor | None,
    matrix_size: int,
) -> torch.Tensor:
    if tensor.ndim != 4 or tuple(tensor.shape[-2:]) != (matrix_size, matrix_size):
        raise ValueError(f"{name} must be [B,N,{matrix_size},{matrix_size}]")
    if tensor.shape[0] not in (1, batch):
        raise ValueError(f"{name} batch cannot serve attention batch")
    if tensor.shape[0] == 1 and batch != 1:
        tensor = tensor.expand(batch, -1, -1, -1)
    if tensor.shape[1] == selected_rows:
        return tensor
    if frame_ids is None:
        raise ValueError(f"{name} requires frame IDs for frame-aligned cameras")
    frame_ids = frame_ids.to(device=tensor.device, dtype=torch.long)
    if frame_ids.ndim != 1 or frame_ids.numel() != selected_rows:
        raise ValueError("camera frame IDs must align with selected token rows")
    if frame_ids.numel() and (int(frame_ids.min()) < 0 or int(frame_ids.max()) >= tensor.shape[1]):
        raise ValueError(f"camera frame IDs address outside {name}")
    return tensor.index_select(1, frame_ids)


class SolarMiniMaxH3AttnProcessor(MiniMaxH3AttnProcessor):
    """Unwrap the SolarWM attention control at each official H3 block."""

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: Any = None,
    ) -> torch.Tensor:
        if not isinstance(attention_mask, H3AttentionControl):
            return super().__call__(attn, hidden_states, rotary_emb, attention_mask)
        if rotary_emb is None:
            raise ValueError("H3 attention requires native MM-RoPE")
        control = attention_mask
        return SolarMiniMaxH3Transformer3DModel._attention_forward(
            attn,
            hidden_states,
            rotary_emb,
            control.attention_mask,
            fused_prope=control.fused_prope,
            prope_token_indices=control.prope_token_indices,
            prope_frame_ids=control.prope_frame_ids,
            cam_viewmats=control.cam_viewmats,
            cam_K=control.cam_K,
            prope_kwargs=control.prope_kwargs,
        )


class SolarMiniMaxH3Transformer3DModel(MiniMaxH3Transformer3DModel):
    """Official H3 weights with packed SP2 and camera-PRoPE plumbing."""

    def _install_solar_processors(self) -> None:
        for block in self.transformer_blocks:
            block.attn.set_processor(SolarMiniMaxH3AttnProcessor())

    @classmethod
    def from_config(
        cls, config: Any = None, return_unused_kwargs: bool = False, **kwargs: Any
    ) -> Any:
        result = super().from_config(config, return_unused_kwargs=return_unused_kwargs, **kwargs)
        if return_unused_kwargs:
            model, unused = result
            model._install_solar_processors()
            return model, unused
        result._install_solar_processors()
        return result

    @classmethod
    def strict_from_pretrained(cls, path: str, **kwargs: Any) -> Any:
        kwargs.pop("output_loading_info", None)
        model, info = cls.from_pretrained(path, output_loading_info=True, **kwargs)
        problems = {
            key: info.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if info.get(key)
        }
        if problems:
            raise RuntimeError(f"MiniMax-H3 checkpoint did not strict-load: {problems}")
        return model

    @staticmethod
    def _prepare_prope(
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        fused_prope: bool | FusedPrope | None,
        prope_token_indices: torch.Tensor | None,
        prope_frame_ids: torch.Tensor | None,
        cam_viewmats: torch.Tensor | None,
        cam_K: torch.Tensor | None,
        prope_kwargs: dict[str, Any] | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Callable[[torch.Tensor], torch.Tensor] | None,
        torch.Tensor | None,
    ]:
        cameras_present = cam_viewmats is not None or cam_K is not None
        if fused_prope is False:
            return query, key, value, None, None
        if not cameras_present and fused_prope is not True and not callable(fused_prope):
            return query, key, value, None, None
        if cam_viewmats is None or cam_K is None or prope_token_indices is None:
            raise ValueError("fused camera PRoPE requires viewmats, K, and explicit row indices")
        indices = prope_token_indices.to(device=query.device, dtype=torch.long)
        if indices.ndim != 1:
            raise ValueError("camera PRoPE indices must be one-dimensional")
        if indices.numel() == 0:
            return query, key, value, None, indices
        if int(indices.min()) < 0 or int(indices.max()) >= query.shape[1]:
            raise ValueError("camera PRoPE index is outside the local packed sequence")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("camera PRoPE indices contain duplicates")
        views = _camera_rows(
            cam_viewmats,
            name="cam_viewmats",
            batch=query.shape[0],
            selected_rows=indices.numel(),
            frame_ids=prope_frame_ids,
            matrix_size=4,
        ).to(device=query.device)
        intrinsics = _camera_rows(
            cam_K,
            name="cam_K",
            batch=query.shape[0],
            selected_rows=indices.numel(),
            frame_ids=prope_frame_ids,
            matrix_size=3,
        ).to(device=query.device)
        fixed_intrinsics = torch.zeros_like(intrinsics)
        fixed_intrinsics[..., 0, 0] = WAN_FIXED_FX
        fixed_intrinsics[..., 1, 1] = WAN_FIXED_FY
        fixed_intrinsics[..., 0, 2] = WAN_FIXED_CX
        fixed_intrinsics[..., 1, 2] = WAN_FIXED_CY
        fixed_intrinsics[..., 2, 2] = 1.0
        if callable(fused_prope):
            selected_q = query.index_select(1, indices).transpose(1, 2)
            selected_k = key.index_select(1, indices).transpose(1, 2)
            selected_v = value.index_select(1, indices).transpose(1, 2)
            selected_q, selected_k, selected_v, apply_output = fused_prope(
                selected_q,
                selected_k,
                selected_v,
                viewmats=views,
                Ks=fixed_intrinsics,
                **(prope_kwargs or {}),
            )
            return (
                query.index_copy(1, indices, selected_q.transpose(1, 2)),
                key.index_copy(1, indices, selected_k.transpose(1, 2)),
                value.index_copy(1, indices, selected_v.transpose(1, 2)),
                apply_output,
                indices,
            )

        # Run the default projective transform over the complete packed
        # document. Identity poses outside camera rows make
        # the paired Q/K and V/output transforms algebraically neutral, but
        # preserving this BF16 execution order keeps loss computation stable.
        dtype = query.dtype
        batch, sequence = query.shape[:2]
        full_views = (
            torch.eye(4, device=query.device, dtype=dtype)
            .view(1, 1, 4, 4)
            .expand(batch, sequence, 4, 4)
            .clone()
        )
        full_intrinsics = (
            torch.eye(3, device=query.device, dtype=dtype)
            .view(1, 1, 3, 3)
            .expand(batch, sequence, 3, 3)
            .clone()
        )
        full_views = full_views.index_copy(1, indices, views.to(dtype))
        full_intrinsics = full_intrinsics.index_copy(1, indices, fixed_intrinsics.to(dtype))
        query, key, value, apply_output = _default_prope()(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            viewmats=full_views,
            Ks=full_intrinsics,
            **(prope_kwargs or {}),
        )
        return (
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            apply_output,
            None,
        )

    @classmethod
    def _attention_forward(
        cls,
        attn: Any,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Any,
        *,
        fused_prope: bool | FusedPrope | None,
        prope_token_indices: torch.Tensor | None,
        prope_frame_ids: torch.Tensor | None,
        cam_viewmats: torch.Tensor | None,
        cam_K: torch.Tensor | None,
        prope_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise RuntimeError("the H3 runtime supports unmasked Stage0.5 only")
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        # Native 128-d MM-RoPE first; camera transform only touches [96:128).
        query = _apply_rotary_emb(query, *rotary_emb)
        key = _apply_rotary_emb(key, *rotary_emb)
        query, key, value, apply_output, camera_indices = cls._prepare_prope(
            query=query,
            key=key,
            value=value,
            fused_prope=fused_prope,
            prope_token_indices=prope_token_indices,
            prope_frame_ids=prope_frame_ids,
            cam_viewmats=cam_viewmats,
            cam_K=cam_K,
            prope_kwargs=prope_kwargs,
        )
        if is_sequence_parallel_enabled():
            if attn.heads % get_sp_size():
                raise RuntimeError("H3 attention heads must be divisible by SP size")
            query = sequence_all_to_all(query, 2, 1)
            key = sequence_all_to_all(key, 2, 1)
            value = sequence_all_to_all(value, 2, 1)
        attended = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=getattr(attn.processor, "_attention_backend", None),
            parallel_config=getattr(attn.processor, "_parallel_config", None),
        )
        if is_sequence_parallel_enabled():
            attended = sequence_all_to_all(attended, 1, 2)
        if apply_output is not None:
            if camera_indices is None:
                attended = apply_output(attended.transpose(1, 2)).transpose(1, 2)
            elif camera_indices.numel():
                selected = attended.index_select(1, camera_indices).transpose(1, 2)
                attended = attended.index_copy(
                    1, camera_indices, apply_output(selected).transpose(1, 2)
                )
        attended = attended.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](attended))

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        *,
        attention_mask: Any = None,
        fused_prope: bool | FusedPrope | None = None,
        prope_token_indices: torch.Tensor | None = None,
        prope_frame_ids: torch.Tensor | None = None,
        cam_viewmats: torch.Tensor | None = None,
        cam_K: torch.Tensor | None = None,
        prope_kwargs: dict[str, Any] | None = None,
        stage0p5_sequence_parallel: bool = False,
    ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
        del attention_kwargs
        if attention_mask is not None:
            raise RuntimeError("MiniMax-H3 Stage0.5 requires attention_mask=None")
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError("position_ids must be [sequence,3]")
        sequence_length = int(position_ids.shape[0])
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError("token tags/timestep indices must match packed rows")
        if is_sequence_parallel_enabled() != bool(stage0p5_sequence_parallel):
            raise RuntimeError("H3 forward and initialized SP topology disagree")

        video = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
        audio = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
        text = self.context_embedder(
            encoder_hidden_states.to(get_parameter_dtype(self.context_embedder))
        )
        text = self.token_refiner(text)
        packed = text.new_zeros((text.shape[0], sequence_length, text.shape[-1]))
        packed = packed.index_copy(1, text_indices, text)
        packed = packed.index_copy(1, video_indices, video.to(text.dtype))
        packed = packed.index_copy(1, audio_indices, audio.to(text.dtype))
        temb = self.time_embedder(
            self.time_proj(timestep).to(get_parameter_dtype(self.time_embedder))
        )
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
        if is_sequence_parallel_enabled():
            shard = shard_stage0p5_packed_sequence(
                hidden_states=packed,
                position_ids=position_ids,
                token_tags=token_tags,
                timestep_indices=timestep_indices,
                prope_token_indices=prope_token_indices,
                prope_frame_ids=prope_frame_ids,
                camera_viewmats=cam_viewmats,
                camera_K=cam_K,
            )
            packed = shard.hidden_states
            position_ids = shard.position_ids
            token_tags = shard.token_tags
            timestep_indices = shard.timestep_indices
            adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
            prope_token_indices = shard.prope_token_indices
            prope_frame_ids = shard.prope_frame_ids
            cam_viewmats = shard.camera_viewmats
            cam_K = shard.camera_K
        rotary_emb = self.rope(position_ids)
        control = H3AttentionControl(
            attention_mask=None,
            fused_prope=fused_prope,
            prope_token_indices=prope_token_indices,
            prope_frame_ids=prope_frame_ids,
            cam_viewmats=cam_viewmats,
            cam_K=cam_K,
            prope_kwargs=prope_kwargs,
        )
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def block_forward(states: torch.Tensor, current_block: Any = block) -> torch.Tensor:
                    return current_block(states, temb, adaln_indices, rotary_emb, control)

                packed = self._gradient_checkpointing_func(block_forward, packed)
            else:
                packed = block(packed, temb, adaln_indices, rotary_emb, control)
        packed = self.norm_out(packed, temb, timestep_indices).to(
            get_parameter_dtype(self.proj_out)
        )
        video_output = self.proj_out(packed)
        audio_output = self.audio_proj_out(packed)
        if is_sequence_parallel_enabled():
            video_output = sequence_all_gather(video_output, dim=1)
            audio_output = sequence_all_gather(audio_output, dim=1)
            if video_output.shape[1] != sequence_length:
                raise RuntimeError("H3 SP output gather did not restore the packed sequence")
        video_output = video_output.index_select(1, video_indices)
        audio_output = audio_output.index_select(1, audio_indices)
        if not return_dict:
            return video_output, audio_output
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)


__all__ = [
    "H3AttentionControl",
    "MiniMaxH3TransformerBlock",
    "SolarMiniMaxH3AttnProcessor",
    "SolarMiniMaxH3Transformer3DModel",
]
