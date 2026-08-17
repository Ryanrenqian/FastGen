"""DINOv3 patch features and motion-token weighting for drifting loss."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class DinoV3FeatureExtractor(nn.Module):
    """Frozen DINOv3 ViT-B/16 whose input gradient remains enabled."""

    def __init__(
        self,
        *,
        repo_dir: str,
        weights_path: str,
        model_name: str = "dinov3_vitb16",
        input_size: int = 256,
        block_indices: Sequence[int] = (2, 5, 8),
        motion_alpha: float = 2.5,
        motion_lambda: float = 12.0,
        motion_quantile: float = 0.98,
        motion_threshold: float = 0.35,
        frame_chunk_size: int = 8,
    ):
        super().__init__()
        if not os.path.isdir(repo_dir):
            raise FileNotFoundError(f"DINOv3 repository is missing: {repo_dir}")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"DINOv3 weights are missing: {weights_path}")
        if input_size <= 0 or input_size % 16:
            raise ValueError("DINOv3 input_size must be a positive multiple of 16")
        if frame_chunk_size <= 0:
            raise ValueError("DINOv3 frame_chunk_size must be positive")

        if model_name != "dinov3_vitb16":
            raise ValueError(f"unsupported DINOv3 model: {model_name}")
        # Import the official backbone directly. Loading the repository's full
        # hubconf also imports optional segmentation dependencies unrelated to
        # feature extraction.
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from dinov3.hub.backbones import dinov3_vitb16

        if weights_path.endswith(".safetensors"):
            self.model = dinov3_vitb16(pretrained=False)
            self._load_modelscope_weights(weights_path)
        else:
            self.model = dinov3_vitb16(weights=weights_path)
        self.model.eval().requires_grad_(False)
        self.input_size = int(input_size)
        self.block_indices = tuple(int(index) for index in block_indices)
        self.motion_alpha = float(motion_alpha)
        self.motion_lambda = float(motion_lambda)
        self.motion_quantile = float(motion_quantile)
        self.motion_threshold = float(motion_threshold)
        self.frame_chunk_size = int(frame_chunk_size)
        if not self.block_indices:
            raise ValueError("DINOv3 block_indices cannot be empty")
        if not 0 < self.motion_quantile <= 1:
            raise ValueError("DINOv3 motion_quantile must be in (0, 1]")
        if not 0 <= self.motion_threshold < 1:
            raise ValueError("DINOv3 motion_threshold must be in [0, 1)")
        if self.motion_lambda < 0:
            raise ValueError("DINOv3 motion_lambda must be non-negative")

        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def _load_modelscope_weights(self, weights_path: str) -> None:
        """Strictly map ModelScope's Hugging Face DINOv3 checkpoint."""
        from safetensors.torch import load_file

        source = load_file(weights_path, device="cpu")
        mapped = {
            "cls_token": source["embeddings.cls_token"],
            "storage_tokens": source["embeddings.register_tokens"],
            "mask_token": source["embeddings.mask_token"].squeeze(1),
            "patch_embed.proj.weight": source["embeddings.patch_embeddings.weight"],
            "patch_embed.proj.bias": source["embeddings.patch_embeddings.bias"],
            "norm.weight": source["norm.weight"],
            "norm.bias": source["norm.bias"],
        }
        consumed = {
            "embeddings.cls_token", "embeddings.register_tokens",
            "embeddings.mask_token", "embeddings.patch_embeddings.weight",
            "embeddings.patch_embeddings.bias", "norm.weight", "norm.bias",
        }
        for layer in range(len(self.model.blocks)):
            src, dst = f"layer.{layer}", f"blocks.{layer}"
            q = source[f"{src}.attention.q_proj.weight"]
            k = source[f"{src}.attention.k_proj.weight"]
            v = source[f"{src}.attention.v_proj.weight"]
            q_bias = source[f"{src}.attention.q_proj.bias"]
            v_bias = source[f"{src}.attention.v_proj.bias"]
            mapped[f"{dst}.attn.qkv.weight"] = torch.cat((q, k, v), dim=0)
            mapped[f"{dst}.attn.qkv.bias"] = torch.cat(
                (q_bias, torch.zeros_like(q_bias), v_bias), dim=0
            )
            consumed.update({
                f"{src}.attention.q_proj.weight", f"{src}.attention.k_proj.weight",
                f"{src}.attention.v_proj.weight", f"{src}.attention.q_proj.bias",
                f"{src}.attention.v_proj.bias",
            })
            translations = {
                "attention.o_proj.weight": "attn.proj.weight",
                "attention.o_proj.bias": "attn.proj.bias",
                "layer_scale1.lambda1": "ls1.gamma",
                "layer_scale2.lambda1": "ls2.gamma",
                "mlp.up_proj.weight": "mlp.fc1.weight",
                "mlp.up_proj.bias": "mlp.fc1.bias",
                "mlp.down_proj.weight": "mlp.fc2.weight",
                "mlp.down_proj.bias": "mlp.fc2.bias",
                "norm1.weight": "norm1.weight", "norm1.bias": "norm1.bias",
                "norm2.weight": "norm2.weight", "norm2.bias": "norm2.bias",
            }
            for source_suffix, target_suffix in translations.items():
                source_key = f"{src}.{source_suffix}"
                mapped[f"{dst}.{target_suffix}"] = source[source_key]
                consumed.add(source_key)

        unexpected_source = set(source) - consumed
        if unexpected_source:
            raise RuntimeError(f"unmapped ModelScope DINOv3 weights: {sorted(unexpected_source)}")
        incompatible = self.model.load_state_dict(mapped, strict=False)
        expected_buffers = {"rope_embed.periods"} | {
            f"blocks.{layer}.attn.qkv.bias_mask" for layer in range(len(self.model.blocks))
        }
        if set(incompatible.missing_keys) != expected_buffers or incompatible.unexpected_keys:
            raise RuntimeError(
                "ModelScope DINOv3 conversion did not strictly match the official backbone: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        frames = (frames.float() + 1.0).mul(0.5).clamp(0.0, 1.0)
        if frames.shape[-2:] != (self.input_size, self.input_size):
            frames = F.interpolate(
                frames,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        return (frames - self.mean) / self.std

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return normalized patch tokens for ``[F, 3, H, W]`` frames."""
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("DINOv3 frames must have shape [F, 3, H, W]")
        outputs: dict[str, list[torch.Tensor]] = {
            f"block_{index}": [] for index in self.block_indices
        }
        for start in range(0, frames.shape[0], self.frame_chunk_size):
            chunk = self._preprocess(frames[start : start + self.frame_chunk_size])
            def extract(input_frames):
                return tuple(self.model.get_intermediate_layers(
                    input_frames, n=list(self.block_indices), reshape=False,
                    return_class_token=False, norm=True,
                ))

            features = checkpoint(extract, chunk, use_reentrant=False) if chunk.requires_grad else extract(chunk)
            for index, feature in zip(self.block_indices, features):
                outputs[f"block_{index}"].append(feature)
        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}

    def motion_weights(
        self, positive: torch.Tensor, previous: torch.Tensor
    ) -> torch.Tensor:
        """Create Bridge-style outlier-robust motion weights for patch tokens."""
        delta = torch.linalg.vector_norm(
            positive.detach().float() - previous.detach().float(), dim=-1
        )
        scale = torch.quantile(
            delta, self.motion_quantile, dim=1, keepdim=True
        )
        normalized = (delta / (scale + 1e-6)).clamp(0.0, 1.0)
        median = torch.median(delta, dim=1, keepdim=True).values
        gate = (1.0 - median / (scale + 1e-6)).clamp(0.0, 1.0)
        excess = (normalized - self.motion_threshold).clamp_min(0.0)
        excess = excess / max(1.0 - self.motion_threshold, 1e-6)
        return 1.0 + self.motion_lambda * torch.tanh(self.motion_alpha * excess) * gate
