#!/usr/bin/env python3
"""Visualize Wan temporal features and decompose no-pool TFD force by slot.

The probe mirrors the balanced 4:4/no-pool recipe: one stride-1 trajectory,
three random-step trajectories, shared sigma-0.1 feature noise, a repeated-first
frame fixed negative, and the mixed motion/semantic feature taps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from fastgen.configs.net import Wan22_I2V_5B_Config
from fastgen.datasets.decoders import decode_video_segment
from fastgen.datasets.wds_dataloaders import transform_video
from fastgen.utils import instantiate


LAYERS = (8, 10, 12, 20, 24)
TAPS = ("self_attn_delta",) * 3 + ("block_output",) * 2
LAYER_WEIGHTS = (1.0, 1.0, 1.0, 0.225, 0.225)
VARIANT_NAMES = ("stride1", "random1", "random2", "random3", "static")


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
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--random-step-min", type=int, default=1)
    parser.add_argument("--random-step-max", type=int, default=3)
    parser.add_argument("--radii", type=float, nargs="+", default=(0.02, 0.05, 0.2))
    parser.add_argument("--anchor-margin", type=float, default=0.5)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    return parser.parse_args()


def init_distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    return rank, world_size, torch.device("cuda", local_rank)


def read_rows(path: Path, count: int, seed: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(float(row.get("n_frames") or 0)) >= 49
            and row.get("video_path", "").strip()
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


def random_walk_indices(
    frames: int, step_min: int, step_max: int, rng: random.Random
) -> list[int]:
    indices = [0]
    for _ in range(frames - 1):
        indices.append(indices[-1] + rng.randint(step_min, step_max))
    return indices


def drifting_loss(
    generated: torch.Tensor,
    positive: torch.Tensor,
    fixed_negative: torch.Tensor,
    radii: tuple[float, ...],
    negative_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Official-aligned TFD drift with extra affinity diagnostics."""
    generated = generated.float()
    positive = positive.float()
    frozen_generated = generated.detach()
    fixed_negative = fixed_negative.detach().float()
    fixed_weights = generated.new_full(fixed_negative.shape[:2], negative_weight)
    targets = torch.cat([frozen_generated, fixed_negative, positive], dim=1)
    target_weights = torch.cat(
        [
            generated.new_ones(frozen_generated.shape[:2]),
            fixed_weights,
            generated.new_ones(positive.shape[:2]),
        ],
        dim=1,
    )
    num_generated = generated.shape[1]
    negative_end = num_generated + fixed_negative.shape[1]
    feature_dim = generated.shape[2]
    with torch.no_grad():
        distance = torch.sqrt(
            torch.clamp(torch.cdist(frozen_generated, targets).pow(2), min=1e-8)
        )
        scale = (distance * target_weights[:, None]).mean() / target_weights.mean()
        input_scale = torch.clamp(scale / math.sqrt(float(feature_dim)), min=1e-3)
        generated_scaled = frozen_generated / input_scale
        targets_scaled = targets / input_scale
        normalized_distance = distance / torch.clamp(scale, min=1e-3)
        diagonal = F.pad(
            torch.eye(num_generated, device=generated.device).unsqueeze(0),
            (0, targets.shape[1] - num_generated),
        )
        normalized_distance = normalized_distance + diagonal * 1e6
        force = torch.zeros_like(generated_scaled)
        positive_mass = generated.new_zeros(positive.shape[1])
        positive_ess = generated.new_zeros(())
        for radius in radii:
            logits = -normalized_distance / float(radius)
            affinity = torch.sqrt(
                torch.clamp(
                    torch.softmax(logits, -1) * torch.softmax(logits, -2),
                    min=1e-6,
                )
            )
            affinity = affinity * target_weights[:, None]
            negative_affinity = affinity[:, :, :negative_end]
            positive_affinity = affinity[:, :, negative_end:]
            coefficients = torch.cat(
                [
                    -negative_affinity * positive_affinity.sum(-1, keepdim=True),
                    positive_affinity * negative_affinity.sum(-1, keepdim=True),
                ],
                dim=2,
            )
            radius_force = torch.einsum("biy,byd->bid", coefficients, targets_scaled)
            radius_force -= coefficients.sum(-1)[..., None] * generated_scaled
            norm = radius_force.pow(2).mean()
            force += radius_force / torch.sqrt(torch.clamp(norm, min=1e-8))
            normalized_positive = positive_affinity / positive_affinity.sum(
                -1, keepdim=True
            ).clamp_min(1e-12)
            positive_mass += normalized_positive.mean((0, 1))
            positive_ess += normalized_positive.square().sum(-1).reciprocal().mean()
        positive_mass /= len(radii)
        positive_ess /= len(radii)
        target = generated_scaled + force
    loss = (generated / input_scale.detach() - target.detach()).pow(2).mean((-1, -2))
    return loss, {
        "positive_mass": positive_mass,
        "positive_ess": positive_ess,
        "scale": scale,
    }


def anchor_loss(
    generated: torch.Tensor, anchors: torch.Tensor, alpha: float
) -> torch.Tensor:
    distance = torch.cdist(anchors.float(), generated.float())
    bandwidth = torch.clamp(torch.median(distance.detach()), min=1e-6)
    generated_support = torch.exp(-distance / bandwidth).mean(dim=2)
    anchor_distance = torch.cdist(anchors.float(), anchors.float())
    diagonal = torch.eye(anchors.shape[1], device=anchors.device, dtype=torch.bool)[None]
    anchor_support = torch.exp(-anchor_distance / (2 * bandwidth)).masked_fill(
        diagonal, 0
    ).sum(2) / (anchors.shape[1] - 1)
    return torch.relu(alpha * anchor_support - generated_support).mean(dim=1)


def slot_norm(gradient: torch.Tensor, feature_shape: tuple[int, ...]) -> torch.Tensor:
    shaped = gradient.reshape(feature_shape)
    return shaped.square().mean(dim=(0, 1, 3, 4)).sqrt()


def force_diagnostics(
    feature: torch.Tensor,
    radii: tuple[float, ...],
    anchor_margin: float,
    anchor_weight: float,
    layer_weight: float,
) -> tuple[dict[str, np.ndarray | float], torch.Tensor]:
    """Use static queries and four real trajectories, matching TFD geometry."""
    positive_tensor = feature[:4]
    static_tensor = feature[4:5]
    positive = positive_tensor.flatten(1)[None]
    generated_shape = (4, *static_tensor.shape[1:])
    generated = static_tensor.expand(generated_shape).reshape(1, 4, -1).clone()
    generated.requires_grad_(True)
    fixed_negative = static_tensor.flatten(1)[None]
    drift, info = drifting_loss(
        generated, positive, fixed_negative, radii, negative_weight=1.0
    )
    anchor = anchor_loss(generated, positive, anchor_margin)
    drift_scalar = layer_weight * drift.mean()
    total_scalar = drift_scalar + layer_weight * anchor_weight * anchor.mean()
    drift_grad = torch.autograd.grad(drift_scalar, generated, retain_graph=True)[0]
    total_grad = torch.autograd.grad(total_scalar, generated)[0]
    anchor_grad = total_grad - drift_grad
    drift_slots = slot_norm(drift_grad, generated_shape)
    total_slots = slot_norm(total_grad, generated_shape)
    anchor_slots = slot_norm(anchor_grad, generated_shape)
    result = {
        "drift_slot_norm": drift_slots.detach().cpu().numpy(),
        "total_slot_norm": total_slots.detach().cpu().numpy(),
        "anchor_slot_norm": anchor_slots.detach().cpu().numpy(),
        "positive_mass": info["positive_mass"].detach().cpu().numpy(),
        "positive_ess": float(info["positive_ess"].detach().cpu()),
        "kernel_scale": float(info["scale"].detach().cpu()),
    }
    return result, total_grad.detach().reshape(generated_shape)


def temporal_cosines(gradients: list[torch.Tensor]) -> np.ndarray:
    layers = len(gradients)
    slots = gradients[0].shape[2]
    output = torch.empty(layers, layers, slots, dtype=torch.float32)
    for row in range(layers):
        for column in range(layers):
            for slot in range(slots):
                left = gradients[row][:, :, slot].flatten().float()
                right = gradients[column][:, :, slot].flatten().float()
                output[row, column, slot] = F.cosine_similarity(
                    left, right, dim=0, eps=1e-12
                )
    return output.cpu().numpy()


def pca_scores(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = values.astype(np.float64)
    values -= values.mean(axis=0, keepdims=True)
    gram = values @ values.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    scores = eigenvectors[:, order[:2]] * np.sqrt(eigenvalues[:2])[None]
    explained = eigenvalues[:2] / max(eigenvalues.sum(), 1e-12)
    return scores.astype(np.float32), explained.astype(np.float32)


def scatter_plot(
    scores: np.ndarray, variants: np.ndarray, slots: np.ndarray, title: str, path: Path
) -> None:
    width, height = 1000, 820
    canvas = np.full((height, width, 3), 255, np.uint8)
    left, top, right, bottom = 90, 75, width - 35, height - 100
    low = np.percentile(scores, 1, axis=0)
    high = np.percentile(scores, 99, axis=0)
    span = np.maximum(high - low, 1e-8)
    colors = ((220, 80, 40), (40, 150, 50), (40, 90, 220), (180, 60, 180))
    markers = (cv2.MARKER_CROSS, cv2.MARKER_TILTED_CROSS, cv2.MARKER_TRIANGLE_UP, cv2.MARKER_DIAMOND, cv2.MARKER_SQUARE)
    for point, variant, slot in zip(scores, variants, slots):
        x = left + int(np.clip((point[0] - low[0]) / span[0], 0, 1) * (right - left))
        y = bottom - int(np.clip((point[1] - low[1]) / span[1], 0, 1) * (bottom - top))
        cv2.drawMarker(canvas, (x, y), colors[int(slot)], markers[int(variant)], 6, 1)
    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 0, 0), 1)
    cv2.putText(canvas, title, (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.putText(canvas, "color: temporal slot 1..4; marker: trajectory/static", (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)
    for index, name in enumerate(VARIANT_NAMES):
        x = 40 + index * 185
        cv2.drawMarker(canvas, (x, height - 48), (0, 0, 0), markers[index], 10, 1)
        cv2.putText(canvas, name, (x + 10, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    cv2.imwrite(str(path), canvas)


def heatmap(values: np.ndarray, title: str, rows: list[str], columns: list[str], path: Path, signed: bool = False) -> None:
    values = np.asarray(values, np.float32)
    if signed:
        bound = max(float(np.max(np.abs(values))), 1e-8)
        scaled = (np.clip(values / bound, -1, 1) + 1) * 127.5
        color = cv2.COLORMAP_COOL
    else:
        low, high = float(np.min(values)), float(np.max(values))
        scaled = (values - low) / max(high - low, 1e-8) * 255
        color = cv2.COLORMAP_VIRIDIS
    image = cv2.applyColorMap(scaled.astype(np.uint8), color)
    cell_w, cell_h = 150, 70
    image = cv2.resize(image, (values.shape[1] * cell_w, values.shape[0] * cell_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((image.shape[0] + 130, image.shape[1] + 190, 3), 255, np.uint8)
    canvas[65 : 65 + image.shape[0], 170 : 170 + image.shape[1]] = image
    cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    for row, label in enumerate(rows):
        cv2.putText(canvas, label, (8, 108 + row * cell_h), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)
    for column, label in enumerate(columns):
        cv2.putText(canvas, label, (175 + column * cell_w, canvas.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            cv2.putText(canvas, f"{values[row, column]:.3f}", (180 + column * cell_w, 105 + row * cell_h), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
    cv2.imwrite(str(path), canvas)


def aggregate(output_dir: Path, world_size: int, args: argparse.Namespace) -> None:
    parts = [np.load(output_dir / f"rank_{rank}.npz", allow_pickle=True) for rank in range(world_size)]
    concatenate = lambda key: np.concatenate([part[key] for part in parts], axis=0)
    representations = concatenate("representations")
    drift_force = concatenate("drift_force")
    total_force = concatenate("total_force")
    anchor_force = concatenate("anchor_force")
    positive_mass = concatenate("positive_mass")
    positive_ess = concatenate("positive_ess")
    kernel_scale = concatenate("kernel_scale")
    force_cosine = concatenate("force_cosine")
    paths = concatenate("paths")
    captions = concatenate("captions")
    samples, layers, variants, slots, channels = representations.shape
    pca_explained = []
    variant_labels = np.tile(np.repeat(np.arange(variants), slots), samples)
    slot_labels = np.tile(np.arange(slots), samples * variants)
    for layer_position, (layer, tap) in enumerate(zip(LAYERS, TAPS)):
        flattened = representations[:, layer_position].reshape(-1, channels)
        scores, explained = pca_scores(flattened)
        pca_explained.append(explained)
        scatter_plot(
            scores,
            variant_labels,
            slot_labels,
            f"PCA: block {layer} {tap} (PC1 {explained[0]:.1%}, PC2 {explained[1]:.1%})",
            output_dir / f"pca_block_{layer}_{tap}.png",
        )
    layer_labels = [f"b{layer} {tap}" for layer, tap in zip(LAYERS, TAPS)]
    slot_labels_text = ["slot1/f1-4", "slot2/f5-8", "slot3/f9-12", "slot4/f13-16"]
    median_total = np.median(total_force, axis=0)
    median_drift = np.median(drift_force, axis=0)
    median_anchor = np.median(anchor_force, axis=0)
    total_fraction = median_total / np.maximum(median_total.sum(axis=1, keepdims=True), 1e-12)
    heatmap(total_fraction, "Median total gradient share by temporal slot", layer_labels, slot_labels_text, output_dir / "temporal_total_force_share.png")
    heatmap(median_drift, "Median TFD drift gradient norm", layer_labels, slot_labels_text, output_dir / "temporal_drift_force.png")
    heatmap(median_anchor, "Median anchor gradient norm", layer_labels, slot_labels_text, output_dir / "temporal_anchor_force.png")
    heatmap(np.median(positive_mass, axis=0), "Mean positive affinity mass", layer_labels, list(VARIANT_NAMES[:4]), output_dir / "positive_trajectory_affinity.png")
    median_cosine = np.median(force_cosine, axis=0)
    for slot in range(slots):
        heatmap(median_cosine[:, :, slot], f"Layer-force cosine: temporal slot {slot + 1}", layer_labels, layer_labels, output_dir / f"force_cosine_slot_{slot + 1}.png", signed=True)
    summary = {
        "num_samples": int(samples),
        "sigma": args.sigma,
        "layers": list(LAYERS),
        "taps": list(TAPS),
        "layer_weights": list(LAYER_WEIGHTS),
        "variants": list(VARIANT_NAMES),
        "temporal_slots": slot_labels_text,
        "query_for_force": "four identical repeated-first-frame static videos",
        "positive_trajectories": "one stride-1 plus three random walks with steps in [1,3]",
        "median_total_force_share_by_layer_and_slot": total_fraction.tolist(),
        "median_positive_affinity_mass_by_layer": np.median(positive_mass, axis=0).tolist(),
        "median_positive_affinity_ess_by_layer": np.median(positive_ess, axis=0).tolist(),
        "median_kernel_scale_by_layer": np.median(kernel_scale, axis=0).tolist(),
        "pca_explained_variance_pc1_pc2": np.asarray(pca_explained).tolist(),
        "median_layer_force_cosine_by_slot": median_cosine.tolist(),
        "paths": paths.tolist(),
        "captions": captions.tolist(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "aggregate_metrics.npz",
        representations=representations,
        drift_force=drift_force,
        total_force=total_force,
        anchor_force=anchor_force,
        positive_mass=positive_mass,
        positive_ess=positive_ess,
        kernel_scale=kernel_scale,
        force_cosine=force_cosine,
        paths=paths,
        captions=captions,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.num_samples < 2 or args.sigma <= 0:
        raise ValueError("num-samples must be >=2 and sigma must be positive")
    rank, world_size, device = init_distributed()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = read_rows(args.index, args.num_samples, args.seed)
    indexed_rows = list(enumerate(selected_rows))[rank::world_size]
    torch.manual_seed(args.seed + rank)

    config = Wan22_I2V_5B_Config.copy()
    config.disable_grad_ckpt = True
    net = instantiate(config)
    net.init_preprocessors()
    net.eval().requires_grad_(False)
    net.to(device=device, dtype=torch.bfloat16)

    all_repr, all_drift, all_total, all_anchor = [], [], [], []
    all_mass, all_ess, all_scale, all_cosine = [], [], [], []
    used_paths, used_captions = [], []
    decode_length = 1 + (args.frames - 1) * args.random_step_max
    for global_index, row in indexed_rows:
        path = row["video_path"]
        frames = decode_video_segment(
            path, path, decode_length, output_format="torch"
        )
        if frames is None or frames.shape[0] < decode_length:
            continue
        full = transform_video(frames, decode_length, (args.width, args.height))["real"]
        trajectory_rng = random.Random(args.seed + global_index * 9_973)
        indices = [list(range(args.frames))]
        indices.extend(
            random_walk_indices(
                args.frames, args.random_step_min, args.random_step_max, trajectory_rng
            )
            for _ in range(3)
        )
        positives = torch.stack([full[:, item] for item in indices])
        static = positives[:1, :, :1].expand(-1, -1, args.frames, -1, -1).contiguous()
        variants = torch.cat([positives, static], dim=0).to(
            device=device, dtype=torch.bfloat16
        )
        with torch.no_grad():
            latents = net.vae.encode(variants, mode="argmax")
            first_frame = net.vae.encode(variants[:1, :, :1], mode="argmax").expand(
                variants.shape[0], -1, -1, -1, -1
            )
            text = net.text_encoder.encode(
                [caption(row)], precision=torch.bfloat16
            ).expand(variants.shape[0], -1, -1)
            epsilon = torch.randn_like(latents[:1]).expand_as(latents)
            sigmas = torch.full(
                (variants.shape[0],), args.sigma, device=device, dtype=torch.float32
            )
            noisy = net.noise_scheduler.forward_process(latents, epsilon, sigmas)
            condition = {"text_embeds": text, "first_frame_cond": first_frame}
            motion_features = net(
                noisy,
                sigmas,
                condition=condition,
                return_features_early=True,
                feature_indices=set(LAYERS[:3]),
                feature_tap="self_attn_delta",
            )
            semantic_features = net(
                noisy,
                sigmas,
                condition=condition,
                return_features_early=True,
                feature_indices=set(LAYERS[3:]),
                feature_tap="block_output",
            )
        features = motion_features + semantic_features
        sample_repr, sample_drift, sample_total, sample_anchor = [], [], [], []
        sample_mass, sample_ess, sample_scale, gradients = [], [], [], []
        for feature, weight in zip(features, LAYER_WEIGHTS):
            feature = feature.float()[:, :, 1:]
            sample_repr.append(
                feature.mean(dim=(-1, -2)).permute(0, 2, 1).cpu().numpy()
            )
            diagnostics, gradient = force_diagnostics(
                feature,
                tuple(args.radii),
                args.anchor_margin,
                args.anchor_weight,
                weight,
            )
            sample_drift.append(diagnostics["drift_slot_norm"])
            sample_total.append(diagnostics["total_slot_norm"])
            sample_anchor.append(diagnostics["anchor_slot_norm"])
            sample_mass.append(diagnostics["positive_mass"])
            sample_ess.append(diagnostics["positive_ess"])
            sample_scale.append(diagnostics["kernel_scale"])
            gradients.append(gradient.cpu())
        all_repr.append(np.stack(sample_repr))
        all_drift.append(np.stack(sample_drift))
        all_total.append(np.stack(sample_total))
        all_anchor.append(np.stack(sample_anchor))
        all_mass.append(np.stack(sample_mass))
        all_ess.append(np.asarray(sample_ess, np.float32))
        all_scale.append(np.asarray(sample_scale, np.float32))
        all_cosine.append(temporal_cosines(gradients))
        used_paths.append(path)
        used_captions.append(caption(row))
        print(f"[rank {rank}] {len(used_paths)}/{len(indexed_rows)} {path}", flush=True)

    if not all_repr:
        raise RuntimeError(f"Rank {rank} decoded no usable videos")
    np.savez_compressed(
        args.output_dir / f"rank_{rank}.npz",
        representations=np.stack(all_repr),
        drift_force=np.stack(all_drift),
        total_force=np.stack(all_total),
        anchor_force=np.stack(all_anchor),
        positive_mass=np.stack(all_mass),
        positive_ess=np.stack(all_ess),
        kernel_scale=np.stack(all_scale),
        force_cosine=np.stack(all_cosine),
        paths=np.asarray(used_paths),
        captions=np.asarray(used_captions),
    )
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        aggregate(args.output_dir, world_size, args)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
