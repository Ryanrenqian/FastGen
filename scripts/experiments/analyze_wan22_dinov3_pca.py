#!/usr/bin/env python3
"""Compare pretrained Wan2.2 TI2V generations with GT in DINOv3 PCA space.

This is an inference-only probe. It never loads a trained FastGen checkpoint and
does not alter the training objective. Generated and GT patch tokens are fitted
with one shared PCA basis per sample/layer so pseudo-colors are comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from fastgen.configs.net import Wan22_I2V_5B_Config
from fastgen.datasets.decoders import decode_video_segment
from fastgen.datasets.wds_dataloaders import transform_video
from fastgen.features import DinoV3FeatureExtractor
from fastgen.utils import instantiate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(
            "/mnt/dataset/demo5_dataset/manifests/"
            "demo5_clean_10k_f49_seed10.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--frame-start", type=int, default=30)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument(
        "--dinov3-repo",
        type=Path,
        default=Path("/mnt/home/renqian/imgGen/runtime/MODEL/dinov3/repo"),
    )
    parser.add_argument(
        "--dinov3-weights",
        type=Path,
        default=Path("/mnt/home/renqian/imgGen/runtime/MODEL/dinov3/model.safetensors"),
    )
    parser.add_argument("--dinov3-blocks", type=int, nargs="+", default=(2, 5, 8))
    parser.add_argument("--fps", type=int, default=4)
    return parser.parse_args()


def read_rows(path: Path, count: int, seed: int, required_frames: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(float(row.get("n_frames") or 0)) >= required_frames
            and Path(row.get("video_path", "").strip()).is_file()
        ]
    random.Random(seed).shuffle(rows)
    return rows[:count]


def caption(row: dict[str, str]) -> str:
    return next(
        (
            row.get(key, "").strip()
            for key in ("caption_l3", "caption_l2", "caption_l1")
            if row.get(key, "").strip()
        ),
        "",
    )


def joint_pca_rgb(
    generated: torch.Tensor, gt: torch.Tensor, output_size: tuple[int, int]
) -> tuple[torch.Tensor, list[float]]:
    """Project ``[F,P,D]`` token pairs with a shared robustly scaled PCA."""
    if generated.shape != gt.shape or generated.ndim != 3:
        raise ValueError("generated and GT tokens must have identical [F,P,D] shapes")
    frames, patches, feature_dim = generated.shape
    patch_side = int(round(patches**0.5))
    if patch_side * patch_side != patches:
        raise ValueError(f"expected a square DINO patch grid, got {patches} patches")

    pair = torch.cat((generated.float(), gt.float()), dim=0)
    flat = pair.reshape(-1, feature_dim)
    centered = flat - flat.mean(0, keepdim=True)
    _, singular, basis = torch.pca_lowrank(centered, q=3, center=False, niter=4)
    projected = centered @ basis
    lower = torch.quantile(projected, 0.01, dim=0, keepdim=True)
    upper = torch.quantile(projected, 0.99, dim=0, keepdim=True)
    projected = ((projected - lower) / (upper - lower).clamp_min(1e-6)).clamp(0, 1)
    projected = projected.reshape(2, frames, patch_side, patch_side, 3)
    projected = projected.permute(0, 1, 4, 2, 3).reshape(2 * frames, 3, patch_side, patch_side)
    projected = F.interpolate(projected, size=output_size, mode="nearest")
    projected = projected.reshape(2, frames, 3, *output_size)
    variance = singular.square()
    explained = (variance / variance.sum().clamp_min(1e-12)).cpu().tolist()
    return projected, explained


def uint8_frames(video: torch.Tensor) -> np.ndarray:
    """Convert ``[F,3,H,W]`` in [0,1] to RGB uint8."""
    return (
        video.detach().float().clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
    )


def label_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    pair = np.concatenate((left, right), axis=1)
    cv2.putText(pair, "Generated", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(
        pair, "GT", (left.shape[1] + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    return pair


def write_video(path: Path, frames_rgb: list[np.ndarray], fps: int) -> None:
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_contact_sheet(path: Path, frames_rgb: list[np.ndarray]) -> None:
    sheet = np.concatenate(frames_rgb, axis=0)
    if not cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write contact sheet: {path}")


def main() -> None:
    args = parse_args()
    if args.frames < 5 or (args.frames - 1) % 4:
        raise ValueError("frames must satisfy frames>=5 and (frames-1)%4==0")
    if args.num_samples < 1 or args.num_steps < 1:
        raise ValueError("num-samples and num-steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    config = Wan22_I2V_5B_Config.copy()
    config.disable_grad_ckpt = True
    net = instantiate(config)
    net.init_preprocessors()
    net.eval().requires_grad_(False)
    net.to(device=device, dtype=torch.bfloat16)
    dino = DinoV3FeatureExtractor(
        repo_dir=str(args.dinov3_repo),
        weights_path=str(args.dinov3_weights),
        model_name="dinov3_vitb16",
        input_size=256,
        block_indices=tuple(args.dinov3_blocks),
        frame_chunk_size=8,
    ).to(device).eval()

    required_frames = args.frame_start + args.frames
    rows = read_rows(args.index, args.num_samples, args.seed, required_frames)
    if len(rows) != args.num_samples:
        raise RuntimeError(f"requested {args.num_samples} samples, found {len(rows)} usable rows")

    summaries = []
    latent_frames = (args.frames - 1) // 4 + 1
    for sample_index, row in enumerate(rows):
        raw = decode_video_segment(
            row["video_path"],
            row["video_path"],
            args.frames,
            output_format="torch",
            start_frame=args.frame_start,
        )
        if raw is None or raw.shape[0] < args.frames:
            raise RuntimeError(f"failed to decode {row['video_path']}")
        gt = transform_video(raw, args.frames, (args.width, args.height))["real"].unsqueeze(0)
        gt = gt.to(device=device, dtype=torch.bfloat16)

        prompt = caption(row)
        with torch.inference_mode():
            positive_text = net.text_encoder.encode([prompt], precision=torch.bfloat16)
            negative_text = net.text_encoder.encode([""], precision=torch.bfloat16)
            first_frame = net.vae.encode(gt[:, :, :1], mode="argmax")
            condition = {"text_embeds": positive_text, "first_frame_cond": first_frame}
            negative = {"text_embeds": negative_text, "first_frame_cond": first_frame}
            noise = torch.randn(
                1, 48, latent_frames, args.height // 16, args.width // 16,
                device=device, dtype=torch.bfloat16,
            )
            generated_latent = net.sample(
                noise,
                condition=condition,
                neg_condition=negative,
                guidance_scale=args.guidance_scale,
                num_steps=args.num_steps,
                shift=args.shift,
                show_progress=True,
            )
            generated = net.vae.decode(generated_latent).float()
            gt_float = gt.float()

            # Exclude the copied TI2V condition frame from feature comparison.
            generated_future = generated[:, :, 1:].permute(0, 2, 1, 3, 4).flatten(0, 1)
            gt_future = gt_float[:, :, 1:].permute(0, 2, 1, 3, 4).flatten(0, 1)
            generated_features = dino(generated_future)
            gt_features = dino(gt_future)

        sample_dir = args.output_dir / f"sample_{sample_index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        rgb_gen = uint8_frames((generated[0].permute(1, 0, 2, 3) + 1) * 0.5)
        rgb_gt = uint8_frames((gt_float[0].permute(1, 0, 2, 3) + 1) * 0.5)
        rgb_pairs = [label_pair(left, right) for left, right in zip(rgb_gen, rgb_gt)]
        write_video(sample_dir / "rgb_generated_vs_gt.mp4", rgb_pairs, args.fps)
        write_contact_sheet(sample_dir / "rgb_generated_vs_gt.png", rgb_pairs)

        layer_metrics = {}
        for layer_name in generated_features:
            generated_tokens = generated_features[layer_name]
            gt_tokens = gt_features[layer_name]
            pca, explained = joint_pca_rgb(
                generated_tokens, gt_tokens, (args.height, args.width)
            )
            pca_gen = uint8_frames(pca[0])
            pca_gt = uint8_frames(pca[1])
            pca_pairs = [label_pair(left, right) for left, right in zip(pca_gen, pca_gt)]
            write_video(sample_dir / f"{layer_name}_pca_generated_vs_gt.mp4", pca_pairs, args.fps)
            write_contact_sheet(sample_dir / f"{layer_name}_pca_generated_vs_gt.png", pca_pairs)
            cosine = F.cosine_similarity(generated_tokens.float(), gt_tokens.float(), dim=-1)
            layer_metrics[layer_name] = {
                "aligned_patch_cosine_mean": float(cosine.mean()),
                "aligned_patch_cosine_per_frame": cosine.mean(dim=1).cpu().tolist(),
                "pca_3_component_relative_variance": explained,
            }

        summary = {
            "sample_index": sample_index,
            "video_path": row["video_path"],
            "caption": prompt,
            "rgb_frame_indices": list(range(args.frame_start, args.frame_start + args.frames)),
            "compared_future_frame_indices": list(
                range(args.frame_start + 1, args.frame_start + args.frames)
            ),
            "pretrained_model": config.model_id_or_local_path,
            "num_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "shift": args.shift,
            "layers": layer_metrics,
        }
        (sample_dir / "metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
