# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-step conditional video generation with a drifting-field objective."""

from __future__ import annotations

from typing import Any, Callable, Dict, TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from fastgen.methods.model import FastGenModel
from fastgen.drifting_loss import drifting_loss
from fastgen.features import DinoV3FeatureExtractor

if TYPE_CHECKING:
    from fastgen.configs.methods.config_driftworld import ModelConfig


def _repeat_condition(condition: Any, repeats: int) -> Any:
    """Repeat batch-aligned tensors in a nested conditioning structure."""
    if isinstance(condition, torch.Tensor):
        return condition.repeat_interleave(repeats, dim=0)
    if isinstance(condition, dict):
        return {key: _repeat_condition(value, repeats) for key, value in condition.items()}
    if isinstance(condition, tuple):
        return tuple(_repeat_condition(value, repeats) for value in condition)
    if isinstance(condition, list):
        return [_repeat_condition(value, repeats) for value in condition]
    return condition


def _video_tokens(samples: torch.Tensor, *, drop_first_frame: bool) -> torch.Tensor:
    """Map ``[B, N, C, T, H, W]`` videos to ``[B*T*H*W, N, C]``."""
    if samples.ndim != 6:
        raise ValueError("video samples must have shape [B, N, C, T, H, W]")
    if drop_first_frame:
        if samples.shape[3] <= 1:
            raise ValueError("cannot drop the only latent frame")
        samples = samples[:, :, :, 1:]
    batch, candidates, channels, frames, height, width = samples.shape
    return samples.permute(0, 3, 4, 5, 1, 2).reshape(batch * frames * height * width, candidates, channels)


def _video_block_tokens(
    samples: torch.Tensor,
    *,
    block_shape: tuple[int, int, int],
    drop_first_frame: bool,
) -> torch.Tensor:
    """Group non-overlapping video latent blocks into drifting-field tokens."""
    if samples.ndim != 6:
        raise ValueError("video samples must have shape [B, N, C, T, H, W]")
    if drop_first_frame:
        if samples.shape[3] <= 1:
            raise ValueError("cannot drop the only latent frame")
        samples = samples[:, :, :, 1:]

    temporal, spatial_h, spatial_w = block_shape
    if min(temporal, spatial_h, spatial_w) <= 0:
        raise ValueError("trajectory_drift_block entries must be positive")

    batch, candidates, channels, frames, height, width = samples.shape
    if frames % temporal or height % spatial_h or width % spatial_w:
        raise ValueError(
            f"video latent shape {(frames, height, width)} must be divisible by "
            f"trajectory block {block_shape}"
        )

    samples = samples.reshape(
        batch,
        candidates,
        channels,
        frames // temporal,
        temporal,
        height // spatial_h,
        spatial_h,
        width // spatial_w,
        spatial_w,
    )
    return (
        samples.permute(0, 3, 5, 7, 1, 2, 4, 6, 8)
        .reshape(
            batch * (frames // temporal) * (height // spatial_h) * (width // spatial_w),
            candidates,
            channels * temporal * spatial_h * spatial_w,
        )
    )


def _dinov3_video_tokens(
    features: torch.Tensor, *, batch: int, candidates: int, frames: int
) -> torch.Tensor:
    """Map frame patch features to independent batch/time/patch drift groups."""
    if features.ndim != 3 or features.shape[0] != batch * candidates * frames:
        raise ValueError(
            "DINOv3 features must have shape [batch*candidates*frames, patches, dim]"
        )
    patches, feature_dim = features.shape[1:]
    return (
        features.reshape(batch, candidates, frames, patches, feature_dim)
        .permute(0, 2, 3, 1, 4)
        .reshape(batch * frames * patches, candidates, feature_dim)
    )


def _compare_temporal_sample_force(
    generated: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor | None,
    sample_force: torch.Tensor,
    sample_metrics: dict[str, torch.Tensor],
    *,
    batch: int,
    candidates: int,
    frames: int,
    negative_weight: float,
    radii: list[float],
) -> dict[str, torch.Tensor]:
    """Compare DriftWorld forces after swapping candidate and temporal axes.

    The regular field treats candidates as samples independently at every frame.
    The temporal field treats frames as samples independently for every candidate.
    Both returned forces are reshaped to ``[B, N, F, S, D]`` before comparison.
    """
    groups, actual_candidates, feature_dim = generated.shape
    if actual_candidates != candidates or groups % (batch * frames):
        raise ValueError("generated groups do not match batch/candidate/frame layout")
    spatial = groups // (batch * frames)

    generated_layout = generated.reshape(
        batch, frames, spatial, candidates, feature_dim
    )
    temporal_generated = (
        generated_layout.permute(0, 3, 2, 1, 4)
        .reshape(batch * candidates * spatial, frames, feature_dim)
    )

    def temporal_targets(tokens: torch.Tensor | None) -> torch.Tensor | None:
        if tokens is None:
            return None
        samples = tokens.shape[1]
        target_layout = tokens.reshape(
            batch, frames, spatial, samples, feature_dim
        )
        return (
            target_layout.permute(0, 2, 1, 3, 4)
            .reshape(batch, 1, spatial, frames * samples, feature_dim)
            .expand(-1, candidates, -1, -1, -1)
            .reshape(batch * candidates * spatial, frames * samples, feature_dim)
        )

    _, temporal_metrics, temporal_force = drifting_loss(
        temporal_generated,
        temporal_targets(positive),
        negative=temporal_targets(negative),
        negative_weight=negative_weight,
        radii=radii,
        return_force=True,
    )

    sample_aligned = (
        sample_force.reshape(batch, frames, spatial, candidates, feature_dim)
        .permute(0, 3, 1, 2, 4)
    )
    temporal_aligned = (
        temporal_force.reshape(batch, candidates, spatial, frames, feature_dim)
        .permute(0, 1, 3, 2, 4)
    )
    sample_norm = sample_aligned.square().sum(dim=-1).sqrt()
    temporal_norm = temporal_aligned.square().sum(dim=-1).sqrt()
    cosine = torch.nn.functional.cosine_similarity(
        sample_aligned, temporal_aligned, dim=-1, eps=1e-8
    )
    ratio = temporal_norm / sample_norm.clamp_min(1e-8)
    metrics = {
        "sample_force_norm": sample_norm.mean(),
        "temporal_force_norm": temporal_norm.mean(),
        "temporal_over_sample_force": ratio.mean(),
        "force_cosine": cosine.mean(),
        "sample_radius_force_rms": sample_metrics["radius_force_rms"],
        "temporal_radius_force_rms": temporal_metrics["radius_force_rms"],
        "temporal_over_sample_radius_force_rms": (
            temporal_metrics["radius_force_rms"]
            / sample_metrics["radius_force_rms"].clamp_min(1e-8)
        ),
    }
    for frame in range(frames):
        metrics[f"frame_{frame + 1}_force_cosine"] = cosine[:, :, frame].mean()
        metrics[f"frame_{frame + 1}_temporal_over_sample_force"] = (
            ratio[:, :, frame].mean()
        )
    return {key: value.detach() for key, value in metrics.items()}


class DriftWorldModel(FastGenModel):
    """Train a one-step TI2V generator with a latent-space drifting field."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def build_model(self):
        if self.config.student_sample_steps != 1:
            raise ValueError("DriftWorldModel currently supports one-step generation only")
        if self.config.generated_samples_per_condition < 2:
            raise ValueError("generated_samples_per_condition must be at least two")
        if not self.config.drift_radii or any(radius <= 0 for radius in self.config.drift_radii):
            raise ValueError("drift_radii must contain positive values")
        if self.config.static_negative_weight < 0:
            raise ValueError("static_negative_weight must be non-negative")
        if (
            self.config.local_drift_weight < 0
            or self.config.trajectory_drift_weight < 0
            or self.config.dinov3_drift_weight < 0
        ):
            raise ValueError("drifting-field weights must be non-negative")
        if (
            self.config.local_drift_weight
            + self.config.trajectory_drift_weight
            + self.config.dinov3_drift_weight
            <= 0
        ):
            raise ValueError("at least one drifting-field weight must be positive")
        super().build_model()
        self.load_student_weights_and_ema()
        if self.config.framewise_vae:
            if not hasattr(self.net, "vae") or not hasattr(self.net.vae, "set_framewise"):
                raise ValueError("framewise_vae requires a video encoder with set_framewise()")
            self.net.vae.set_framewise(True)
        if self.config.dinov3_drift_weight > 0:
            self.dinov3 = DinoV3FeatureExtractor(
                repo_dir=self.config.dinov3_repo_dir,
                weights_path=self.config.dinov3_weights_path,
                model_name=self.config.dinov3_model_name,
                input_size=self.config.dinov3_input_size,
                block_indices=self.config.dinov3_block_indices,
                motion_alpha=self.config.dinov3_motion_alpha,
                motion_lambda=self.config.dinov3_motion_lambda,
                motion_quantile=self.config.dinov3_motion_quantile,
                motion_threshold=self.config.dinov3_motion_threshold,
                frame_chunk_size=self.config.dinov3_frame_chunk_size,
            )

    def on_train_begin(self, is_fsdp=False):
        super().on_train_begin(is_fsdp=is_fsdp)
        if hasattr(self, "dinov3"):
            # Keep the reference DINOv3 backbone in fp32; only its input gradient
            # is needed and its frozen weights are not part of FSDP or the optimizer.
            self.dinov3.to(device=self.device).eval()

    def _decode_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Differentiably decode a latent batch in bounded candidate chunks."""
        chunk_size = self.config.dinov3_vae_chunk_size
        if chunk_size <= 0:
            raise ValueError("dinov3_vae_chunk_size must be positive")
        outputs = []
        for start in range(0, latents.shape[0], chunk_size):
            chunk = latents[start : start + chunk_size]
            outputs.append(
                checkpoint(self.net.vae.decode, chunk, use_reentrant=False)
                if chunk.requires_grad
                else self.net.vae.decode(chunk)
            )
        return torch.cat(outputs, dim=0)

    def _dinov3_drifting_fields(
        self, generated: torch.Tensor, positive: torch.Tensor
    ) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Decode videos and construct Bridge-compatible DINOv3 drift fields."""
        batch, candidates = generated.shape[:2]
        if positive.shape[1] != 1:
            raise ValueError("DINOv3 drifting currently requires one positive video per condition")
        generated_flat = generated.flatten(0, 1)
        generated_pixels = self._decode_video_latents(generated_flat)
        with torch.no_grad():
            positive_pixels = self._decode_video_latents(positive[:, 0])

        # The first pixel frame is the TI2V condition. Every future frame uses
        # the preceding real frame as both the motion reference and static negative.
        generated_frames = generated_pixels[:, :, 1:].permute(0, 2, 1, 3, 4).flatten(0, 1)
        positive_frames = positive_pixels[:, :, 1:].permute(0, 2, 1, 3, 4).flatten(0, 1)
        previous_frames = positive_pixels[:, :, :-1].permute(0, 2, 1, 3, 4).flatten(0, 1)
        frames = generated_pixels.shape[2] - 1

        generated_features = self.dinov3(generated_frames)
        with torch.no_grad():
            positive_features = self.dinov3(positive_frames)
            previous_features = self.dinov3(previous_frames)

        fields = []
        for layer_name in generated_features:
            generated_tokens = _dinov3_video_tokens(
                generated_features[layer_name],
                batch=batch,
                candidates=candidates,
                frames=frames,
            )
            positive_tokens = _dinov3_video_tokens(
                positive_features[layer_name], batch=batch, candidates=1, frames=frames
            )
            previous_tokens = _dinov3_video_tokens(
                previous_features[layer_name], batch=batch, candidates=1, frames=frames
            )
            motion_weight = self.dinov3.motion_weights(
                positive_features[layer_name], previous_features[layer_name]
            ).reshape(-1)
            fields.append(
                (layer_name, generated_tokens, positive_tokens, previous_tokens, motion_weight)
            )
        return fields

    def _prepare_data(self, data: Dict[str, Any]) -> tuple[torch.Tensor, Any]:
        if "neg_condition" not in data:
            data = {**data, "neg_condition": data["condition"]}
        real, condition, _ = self._prepare_training_data(data)
        return real, condition

    def _positives(self, data: Dict[str, Any], real: torch.Tensor) -> torch.Tensor:
        positive = data.get("positive")
        if positive is None:
            return real.unsqueeze(1)
        if positive.ndim == real.ndim:
            positive = positive.unsqueeze(1)
        if positive.ndim != real.ndim + 1 or positive.shape[0] != real.shape[0]:
            raise ValueError("positive must have shape [B, P, C, T, H, W]")
        return positive

    @staticmethod
    def _static_negative(real: torch.Tensor) -> torch.Tensor:
        return real[:, :, :1].expand(-1, -1, real.shape[2], -1, -1).unsqueeze(1)

    def _get_outputs(
        self, gen_data: torch.Tensor, input_student: torch.Tensor, condition: Any = None
    ) -> Dict[str, torch.Tensor | Callable]:
        count = self.config.generated_samples_per_condition
        batch = gen_data.shape[0] // count
        generated = gen_data.reshape(batch, count, *gen_data.shape[1:])[:, 0]
        inputs = input_student.reshape(batch, count, *input_student.shape[1:])[:, 0]
        noise_scale = self.net.noise_scheduler.max_sigma
        return {
            "gen_rand": generated,
            "input_rand": inputs / noise_scale if noise_scale > 0 else inputs,
        }

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        del iteration
        real, condition = self._prepare_data(data)
        positive = self._positives(data, real)
        batch = real.shape[0]
        count = self.config.generated_samples_per_condition
        repeated_condition = _repeat_condition(condition, count)

        noise = torch.randn(batch * count, *self.input_shape, device=self.device, dtype=real.dtype)
        input_student = self.net.noise_scheduler.latents(noise=noise)
        timestep = torch.full(
            (batch * count,),
            self.net.noise_scheduler.max_t,
            device=self.device,
            dtype=self.net.noise_scheduler.t_precision,
        )
        generated_flat = self.gen_data_from_net(input_student, timestep, condition=repeated_condition)
        generated = generated_flat.reshape(batch, count, *generated_flat.shape[1:])

        static_negative = self._static_negative(real) if self.config.static_negative_weight > 0 else None
        fields = []
        if self.config.local_drift_weight > 0:
            fields.append(
                (
                    "local",
                    self.config.local_drift_weight,
                    _video_tokens(generated, drop_first_frame=self.config.mask_conditioning_latent_slot),
                    _video_tokens(positive, drop_first_frame=self.config.mask_conditioning_latent_slot),
                    None
                    if static_negative is None
                    else _video_tokens(
                        static_negative, drop_first_frame=self.config.mask_conditioning_latent_slot
                    ),
                )
            )
        if self.config.trajectory_drift_weight > 0:
            block_shape = tuple(self.config.trajectory_drift_block)
            fields.append(
                (
                    "trajectory",
                    self.config.trajectory_drift_weight,
                    _video_block_tokens(
                        generated,
                        block_shape=block_shape,
                        drop_first_frame=self.config.mask_conditioning_latent_slot,
                    ),
                    _video_block_tokens(
                        positive,
                        block_shape=block_shape,
                        drop_first_frame=self.config.mask_conditioning_latent_slot,
                    ),
                    None
                    if static_negative is None
                    else _video_block_tokens(
                        static_negative,
                        block_shape=block_shape,
                        drop_first_frame=self.config.mask_conditioning_latent_slot,
                    ),
                )
            )

        weighted_loss = generated.new_zeros((), dtype=torch.float32)
        total_weight = 0.0
        metrics = {}
        weighted_components = {}
        for field_name, field_weight, generated_tokens, positive_tokens, negative_tokens in fields:
            field_result = drifting_loss(
                generated_tokens,
                positive_tokens,
                negative=negative_tokens,
                negative_weight=self.config.static_negative_weight,
                radii=self.config.drift_radii,
                return_force=(
                    self.config.compare_temporal_sample_force
                    and field_name == "local"
                ),
            )
            field_loss, field_metrics = field_result[:2]
            weighted_loss = weighted_loss + field_weight * field_loss
            total_weight += field_weight
            weighted_components[field_name] = field_weight * field_loss
            metrics[f"{field_name}_drifting_loss"] = field_loss.detach()
            metrics.update({f"{field_name}_{key}": value for key, value in field_metrics.items()})
            if self.config.compare_temporal_sample_force and field_name == "local":
                future_frames = generated.shape[3] - int(
                    self.config.mask_conditioning_latent_slot
                )
                comparison = _compare_temporal_sample_force(
                    generated_tokens,
                    positive_tokens,
                    negative_tokens,
                    field_result[2],
                    field_metrics,
                    batch=batch,
                    candidates=count,
                    frames=future_frames,
                    negative_weight=self.config.static_negative_weight,
                    radii=self.config.drift_radii,
                )
                metrics.update(
                    {f"force_compare_local_{key}": value for key, value in comparison.items()}
                )

        if self.config.dinov3_drift_weight > 0:
            dinov3_losses = []
            for layer_name, generated_tokens, positive_tokens, negative_tokens, motion_weight in (
                self._dinov3_drifting_fields(generated, positive)
            ):
                layer_result = drifting_loss(
                    generated_tokens,
                    positive_tokens,
                    negative=negative_tokens,
                    negative_weight=self.config.static_negative_weight,
                    group_weight=motion_weight,
                    radii=self.config.drift_radii,
                    return_force=self.config.compare_temporal_sample_force,
                )
                layer_loss, layer_metrics = layer_result[:2]
                dinov3_losses.append(layer_loss)
                metrics[f"dinov3_{layer_name}_drifting_loss"] = layer_loss.detach()
                metrics.update(
                    {
                        f"dinov3_{layer_name}_{key}": value
                        for key, value in layer_metrics.items()
                    }
                )
                if self.config.compare_temporal_sample_force:
                    comparison = _compare_temporal_sample_force(
                        generated_tokens,
                        positive_tokens,
                        negative_tokens,
                        layer_result[2],
                        layer_metrics,
                        batch=batch,
                        candidates=count,
                        frames=generated.shape[3] - 1,
                        negative_weight=self.config.static_negative_weight,
                        radii=self.config.drift_radii,
                    )
                    metrics.update(
                        {
                            f"force_compare_dinov3_{layer_name}_{key}": value
                            for key, value in comparison.items()
                        }
                    )
            dinov3_loss = torch.stack(dinov3_losses).mean()
            weighted_loss = weighted_loss + self.config.dinov3_drift_weight * dinov3_loss
            total_weight += self.config.dinov3_drift_weight
            weighted_components["dinov3"] = self.config.dinov3_drift_weight * dinov3_loss
            metrics["dinov3_drifting_loss"] = dinov3_loss.detach()
        loss = weighted_loss / total_weight
        for component_name, component_loss in weighted_components.items():
            metrics[f"{component_name}_weighted_loss"] = (
                component_loss / total_weight
            ).detach()

        # Log unambiguous VAE/DINO names, configured weights, weighted
        # contributions, and their balance in the actual optimized objective.
        if "local" in weighted_components:
            vae_raw = metrics["local_drifting_loss"]
            vae_weighted = weighted_components["local"] / total_weight
            metrics["vae_drifting_loss"] = vae_raw
            metrics["vae_drift_weight"] = vae_raw.new_tensor(
                self.config.local_drift_weight
            )
            metrics["vae_weighted_loss"] = vae_weighted.detach()
            # Preserve the existing latent aliases for dashboard compatibility.
            metrics["latent_drifting_loss"] = vae_raw
            metrics["latent_weighted_loss"] = vae_weighted.detach()
        if "dinov3" in weighted_components:
            dino_raw = metrics["dinov3_drifting_loss"]
            dino_weighted = weighted_components["dinov3"] / total_weight
            metrics["dino_drifting_loss"] = dino_raw
            metrics["dino_drift_weight"] = dino_raw.new_tensor(
                self.config.dinov3_drift_weight
            )
            metrics["dino_weighted_loss"] = dino_weighted.detach()
        if "local" in weighted_components and "dinov3" in weighted_components:
            pair_total = (vae_weighted + dino_weighted).clamp_min(1e-12)
            ratio = dino_weighted / vae_weighted.clamp_min(1e-12)
            metrics["dino_to_vae_loss_ratio"] = ratio.detach()
            metrics["vae_loss_fraction"] = (vae_weighted / pair_total).detach()
            metrics["dino_loss_fraction"] = (dino_weighted / pair_total).detach()
            metrics["dinov3_to_latent_loss_ratio"] = ratio.detach()
            metrics["latent_loss_fraction"] = (vae_weighted / pair_total).detach()
            metrics["dinov3_loss_fraction"] = (dino_weighted / pair_total).detach()
        loss_map = {"total_loss": loss, "drifting_loss": loss, **metrics}
        return loss_map, self._get_outputs(generated_flat, input_student, condition)
