#!/usr/bin/env python3
"""Analyze the temporal geometry of Wan-VAE latents with PCA and k-means.

The script intentionally loads only the VAE, rather than the full Wan denoiser.
It can analyze both absolute latent features (mostly appearance/content) and
first-frame-relative deltas (mostly temporal change).
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_INDEX = Path(
    "/mnt/dataset/demo5_dataset/manifests/demo5_clean_10k_f49_seed10.csv"
)
DEFAULT_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
CLUSTER_COLORS = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCA and temporal clustering analysis for Wan-VAE latents."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--index", type=Path, default=DEFAULT_INDEX, help="CSV video manifest"
    )
    source.add_argument(
        "--video", type=Path, action="append", help="Video path; repeatable"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--path-column", default="video_path")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--encode-mode",
        choices=("framewise", "temporal"),
        default="framewise",
        help="Framewise matches the current FastGen Wan2.2 experiment contract.",
    )
    parser.add_argument(
        "--feature-pool",
        choices=("spatial-grid", "channel-moments", "flatten"),
        default="spatial-grid",
    )
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument(
        "--analysis-space",
        choices=("absolute", "delta"),
        action="append",
        help="May be repeated; defaults to both absolute and delta.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=0,
        help="Number of clusters; 0 selects k by silhouette score.",
    )
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--cluster-components", type=int, default=8)
    parser.add_argument("--kmeans-restarts", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="bfloat16"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples < 1 or args.frames < 2:
        raise ValueError("num-samples must be positive and frames must be >= 2")
    if args.start_frame < 0 or min(args.width, args.height, args.grid_size) < 1:
        raise ValueError("start-frame must be non-negative and sizes must be positive")
    if args.clusters == 1 or args.clusters < 0:
        raise ValueError("clusters must be 0 (auto) or >= 2")
    if args.max_clusters < 2 or args.cluster_components < 1:
        raise ValueError("max-clusters must be >= 2 and cluster-components positive")


def read_video_paths(
    index: Path, path_column: str, count: int, seed: int
) -> list[Path]:
    with index.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or path_column not in rows[0]:
        raise ValueError(f"Manifest must contain a {path_column!r} column: {index}")
    paths = [Path(row[path_column].strip()) for row in rows if row[path_column].strip()]
    random.Random(seed).shuffle(paths)
    return paths[:count]


def decode_selected_frames(path: Path, start_frame: int, count: int) -> torch.Tensor | None:
    """Decode exact zero-based frames without branch-local decoder changes."""
    import av

    selected = []
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            return None
        for index, frame in enumerate(container.decode(video=0)):
            if index < start_frame:
                continue
            if index >= start_frame + count:
                break
            selected.append(frame.to_ndarray(format="rgb24"))
    if len(selected) != count:
        return None
    return torch.from_numpy(np.stack(selected))


def latent_features(latent: torch.Tensor, pool: str, grid_size: int) -> np.ndarray:
    """Convert [C,T,H,W] latent maps to one feature vector per latent time."""
    if latent.ndim != 4:
        raise ValueError(f"Expected [C,T,H,W], got {tuple(latent.shape)}")
    temporal = latent.float().permute(1, 0, 2, 3)
    if pool == "spatial-grid":
        feature = F.adaptive_avg_pool2d(temporal, (grid_size, grid_size)).flatten(1)
    elif pool == "channel-moments":
        feature = torch.cat(
            [temporal.mean((-1, -2)), temporal.std((-1, -2), unbiased=False)], dim=1
        )
    elif pool == "flatten":
        feature = temporal.flatten(1)
    else:
        raise ValueError(f"Unknown feature pool: {pool}")
    return feature.cpu().numpy().astype(np.float32, copy=False)


def standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return (values - mean) / scale, mean[0], scale[0]


def pca(
    values: np.ndarray, components: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scores, components, and explained-variance ratios using SVD."""
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("PCA requires a 2-D array with at least two observations")
    centered = values - values.mean(axis=0, keepdims=True)
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    available = min(values.shape[0] - 1, values.shape[1])
    keep = available if components is None else min(components, available)
    variance = np.square(singular) / max(values.shape[0] - 1, 1)
    total = variance.sum()
    explained = variance[:keep] / total if total > 0 else np.zeros(keep, np.float32)
    return u[:, :keep] * singular[:keep], vt[:keep], explained


def squared_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.maximum(
        (values * values).sum(1, keepdims=True)
        + (centers * centers).sum(1)[None]
        - 2 * values @ centers.T,
        0.0,
    )


def kmeans_once(
    values: np.ndarray, clusters: int, rng: np.random.Generator, max_iter: int = 200
) -> tuple[np.ndarray, np.ndarray, float]:
    """Small dependency-free k-means++ implementation."""
    observations = values.shape[0]
    centers = [values[int(rng.integers(observations))]]
    for _ in range(1, clusters):
        nearest = squared_distances(values, np.stack(centers)).min(axis=1)
        total = nearest.sum()
        if total <= 1e-12:
            centers.append(values[int(rng.integers(observations))])
        else:
            centers.append(values[int(rng.choice(observations, p=nearest / total))])
    centers_array = np.stack(centers)
    labels = np.zeros(observations, dtype=np.int32)
    for _ in range(max_iter):
        distances = squared_distances(values, centers_array)
        updated_labels = distances.argmin(axis=1).astype(np.int32)
        if np.array_equal(updated_labels, labels) and _ > 0:
            break
        labels = updated_labels
        updated_centers = centers_array.copy()
        for cluster in range(clusters):
            members = values[labels == cluster]
            if len(members):
                updated_centers[cluster] = members.mean(axis=0)
            else:
                updated_centers[cluster] = values[int(rng.integers(observations))]
        if np.allclose(updated_centers, centers_array, rtol=1e-6, atol=1e-7):
            centers_array = updated_centers
            break
        centers_array = updated_centers
    final_distances = squared_distances(values, centers_array)
    labels = final_distances.argmin(axis=1).astype(np.int32)
    inertia = float(final_distances[np.arange(observations), labels].sum())
    return labels, centers_array, inertia


def kmeans(
    values: np.ndarray, clusters: int, seed: int, restarts: int
) -> tuple[np.ndarray, np.ndarray, float]:
    if not 2 <= clusters <= values.shape[0]:
        raise ValueError("clusters must be between 2 and the number of observations")
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for restart in range(max(restarts, 1)):
        candidate = kmeans_once(values, clusters, np.random.default_rng(seed + restart))
        if best is None or candidate[2] < best[2]:
            best = candidate
    assert best is not None
    return best


def silhouette(values: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(values):
        return -1.0
    distances = np.sqrt(squared_distances(values, values))
    scores = []
    for index, label in enumerate(labels):
        same = labels == label
        same[index] = False
        a = float(distances[index, same].mean()) if same.any() else 0.0
        b = min(
            float(distances[index, labels == other].mean())
            for other in unique
            if other != label
        )
        scores.append((b - a) / max(a, b, 1e-12) if same.any() else 0.0)
    return float(np.mean(scores))


def select_clusters(
    values: np.ndarray, requested: int, maximum: int, seed: int, restarts: int
) -> tuple[np.ndarray, np.ndarray, int, float, float, dict[int, float]]:
    if requested:
        labels, centers, inertia = kmeans(values, requested, seed, restarts)
        score = silhouette(values, labels)
        return labels, centers, requested, inertia, score, {requested: score}
    candidates = range(2, min(maximum, len(values) - 1) + 1)
    results = []
    for clusters in candidates:
        labels, centers, inertia = kmeans(values, clusters, seed, restarts)
        results.append((silhouette(values, labels), -inertia, clusters, labels, centers))
    if not results:
        raise ValueError("At least three feature observations are needed for auto clustering")
    best = max(results, key=lambda item: (item[0], item[1]))
    score, negative_inertia, clusters, labels, centers = best
    return (
        labels,
        centers,
        clusters,
        -negative_inertia,
        score,
        {item[2]: item[0] for item in results},
    )


def temporal_distance(features: np.ndarray) -> np.ndarray:
    """Mean normalized feature distance for every pair of time positions."""
    times = features.shape[1]
    result = np.zeros((times, times), np.float32)
    for left in range(times):
        for right in range(times):
            result[left, right] = np.linalg.norm(
                features[:, left] - features[:, right], axis=1
            ).mean() / math.sqrt(features.shape[2])
    return result


def transition_matrix(labels: np.ndarray, clusters: int) -> np.ndarray:
    counts = np.zeros((clusters, clusters), np.float32)
    for trajectory in labels:
        for source, target in zip(trajectory[:-1], trajectory[1:]):
            counts[source, target] += 1
    row_sum = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)


def bgr_time_colors(times: int) -> list[tuple[int, int, int]]:
    ramp = np.linspace(0, 255, times, dtype=np.uint8)[:, None]
    mapped = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)[:, 0]
    return [tuple(int(channel) for channel in color) for color in mapped]


def scatter_trajectories(
    scores: np.ndarray,
    labels: np.ndarray,
    frame_positions: np.ndarray,
    explained: np.ndarray,
    title: str,
    path: Path,
) -> None:
    samples, times, _ = scores.shape
    canvas = np.full((900, 1200, 3), 255, np.uint8)
    left, top, width, height = 110, 90, 950, 690
    x, y = scores[..., 0], scores[..., 1]
    x_pad = max(float(np.ptp(x)) * 0.08, 1e-3)
    y_pad = max(float(np.ptp(y)) * 0.08, 1e-3)
    x_min, x_max = float(x.min() - x_pad), float(x.max() + x_pad)
    y_min, y_max = float(y.min() - y_pad), float(y.max() + y_pad)

    def point(value: np.ndarray) -> tuple[int, int]:
        px = left + int((float(value[0]) - x_min) / max(x_max - x_min, 1e-8) * width)
        py = top + height - int((float(value[1]) - y_min) / max(y_max - y_min, 1e-8) * height)
        return px, py

    cv2.rectangle(canvas, (left, top), (left + width, top + height), (30, 30, 30), 2)
    colors = bgr_time_colors(times)
    for sample in range(samples):
        trajectory = [point(scores[sample, time]) for time in range(times)]
        cv2.polylines(canvas, [np.asarray(trajectory)], False, (205, 205, 205), 1, cv2.LINE_AA)
        for time, center in enumerate(trajectory):
            cluster_color = CLUSTER_COLORS[int(labels[sample, time]) % len(CLUSTER_COLORS)]
            cv2.circle(canvas, center, 6, cluster_color, 2, cv2.LINE_AA)
            cv2.circle(canvas, center, 3, colors[time], -1, cv2.LINE_AA)
    centroids = scores.mean(axis=0)
    centroid_points = np.asarray([point(value) for value in centroids])
    cv2.polylines(canvas, [centroid_points], False, (0, 0, 0), 3, cv2.LINE_AA)
    for time, center in enumerate(centroid_points):
        cv2.circle(canvas, tuple(center), 9, colors[time], -1, cv2.LINE_AA)
        cv2.putText(
            canvas, str(int(frame_positions[time])), tuple(center + (8, -8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
    cv2.putText(canvas, title, (35, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"PC1 ({explained[0]:.1%})", (500, 850), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(canvas, f"PC2 ({explained[1]:.1%})", (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(
        canvas, "fill=time, ring=cluster, black=mean trajectory", (705, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1
    )
    cv2.imwrite(str(path), canvas)


def heatmap(
    values: np.ndarray, row_labels: Iterable[str], column_labels: Iterable[str], title: str, path: Path
) -> None:
    rows, columns = values.shape
    cell = max(34, min(90, 720 // max(rows, columns, 1)))
    left, top = 145, 75
    canvas = np.full((top + rows * cell + 90, left + columns * cell + 45, 3), 255, np.uint8)
    low, high = float(values.min()), float(values.max())
    scaled = ((values - low) / max(high - low, 1e-8) * 255).astype(np.uint8)
    colors = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)
    colors = cv2.resize(colors, (columns * cell, rows * cell), interpolation=cv2.INTER_NEAREST)
    canvas[top : top + rows * cell, left : left + columns * cell] = colors
    cv2.putText(canvas, title, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 0, 0), 2, cv2.LINE_AA)
    for row, label in enumerate(row_labels):
        cv2.putText(
            canvas, str(label), (8, top + row * cell + cell // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1
        )
    for column, label in enumerate(column_labels):
        cv2.putText(
            canvas, str(label), (left + column * cell + 3, top + rows * cell + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1
        )
    cv2.imwrite(str(path), canvas)


def cluster_timeline(labels: np.ndarray, frame_positions: np.ndarray, path: Path) -> None:
    samples, times = labels.shape
    cell_w = max(24, min(90, 900 // times))
    cell_h = max(8, min(28, 620 // samples))
    left, top = 95, 70
    canvas = np.full((top + samples * cell_h + 75, left + times * cell_w + 35, 3), 255, np.uint8)
    cv2.putText(canvas, "Cluster assignment over time", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2)
    for sample in range(samples):
        cv2.putText(
            canvas, str(sample), (25, top + sample * cell_h + cell_h - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1
        )
        for time in range(times):
            color = CLUSTER_COLORS[int(labels[sample, time]) % len(CLUSTER_COLORS)]
            start = (left + time * cell_w, top + sample * cell_h)
            end = (start[0] + cell_w, start[1] + cell_h)
            cv2.rectangle(canvas, start, end, color, -1)
            cv2.rectangle(canvas, start, end, (255, 255, 255), 1)
    for time, frame in enumerate(frame_positions):
        cv2.putText(
            canvas, str(int(frame)),
            (left + time * cell_w + 2, top + samples * cell_h + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1
        )
    cv2.imwrite(str(path), canvas)


def save_preview(video: torch.Tensor, path: Path) -> None:
    """Write a labeled RGB-frame contact sheet from normalized [C,T,H,W]."""
    frames = (video.permute(1, 2, 3, 0).float().clamp(-1, 1) + 1) * 127.5
    frames = frames.byte().cpu().numpy()
    bordered = []
    for index, frame in enumerate(frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        canvas = cv2.copyMakeBorder(bgr, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(
            canvas, f"t={index}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, (255, 255, 255), 1, cv2.LINE_AA
        )
        bordered.append(canvas)
    cv2.imwrite(str(path), np.concatenate(bordered, axis=1))


def sample_trajectory_plot(
    scores: np.ndarray,
    labels: np.ndarray,
    frame_positions: np.ndarray,
    explained: np.ndarray,
    title: str,
    path: Path,
) -> None:
    canvas = np.full((520, 680, 3), 255, np.uint8)
    left, top, width, height = 75, 62, 530, 380
    x_pad = max(float(np.ptp(scores[:, 0])) * 0.18, 0.1)
    y_pad = max(float(np.ptp(scores[:, 1])) * 0.18, 0.1)
    x_min, x_max = float(scores[:, 0].min() - x_pad), float(scores[:, 0].max() + x_pad)
    y_min, y_max = float(scores[:, 1].min() - y_pad), float(scores[:, 1].max() + y_pad)

    def point(value: np.ndarray) -> tuple[int, int]:
        x = left + int((float(value[0]) - x_min) / max(x_max - x_min, 1e-8) * width)
        y = top + height - int((float(value[1]) - y_min) / max(y_max - y_min, 1e-8) * height)
        return x, y

    points = np.asarray([point(value) for value in scores])
    colors = bgr_time_colors(len(scores))
    cv2.rectangle(canvas, (left, top), (left + width, top + height), (35, 35, 35), 1)
    cv2.polylines(canvas, [points], False, (45, 45, 45), 3, cv2.LINE_AA)
    for time, center in enumerate(points):
        cluster_color = CLUSTER_COLORS[int(labels[time]) % len(CLUSTER_COLORS)]
        cv2.circle(canvas, tuple(center), 10, cluster_color, 3, cv2.LINE_AA)
        cv2.circle(canvas, tuple(center), 5, colors[time], -1, cv2.LINE_AA)
        cv2.putText(
            canvas, str(int(frame_positions[time])), tuple(center + (8, -9)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA
        )
    cv2.putText(canvas, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 0, 0), 2)
    cv2.putText(
        canvas, f"PC1 {explained[0]:.1%} / PC2 {explained[1]:.1%}",
        (400, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (70, 70, 70), 1
    )
    cv2.imwrite(str(path), canvas)


def write_rows(
    path: Path, video_paths: list[str], frame_positions: np.ndarray, scores: np.ndarray, labels: np.ndarray
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "video_path", "latent_time", "source_frame", "pc1", "pc2", "cluster"])
        for sample, video_path in enumerate(video_paths):
            for time, frame in enumerate(frame_positions):
                writer.writerow(
                    [sample, video_path, time, int(frame),
                     float(scores[sample, time, 0]), float(scores[sample, time, 1]),
                     int(labels[sample, time])]
                )


def analyze_space(
    raw_features: np.ndarray,
    space: str,
    frame_positions: np.ndarray,
    video_paths: list[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    features = raw_features.copy()
    if space == "delta":
        features -= features[:, :1]
    flattened = features.reshape(-1, features.shape[-1])
    normalized, _, _ = standardize(flattened)
    needed = max(2, args.cluster_components)
    scores, _, explained = pca(normalized, needed)
    cluster_values = scores[:, : min(args.cluster_components, scores.shape[1])]
    labels, _, clusters, inertia, score, candidates = select_clusters(
        cluster_values, args.clusters, args.max_clusters, args.seed, args.kmeans_restarts
    )
    shaped_scores = scores[:, :2].reshape(features.shape[0], features.shape[1], 2)
    shaped_labels = labels.reshape(features.shape[:2])
    normalized_shaped = normalized.reshape(features.shape)
    distances = temporal_distance(normalized_shaped)
    transitions = transition_matrix(shaped_labels, clusters)
    distribution = np.stack(
        [(shaped_labels == cluster).mean(axis=0) for cluster in range(clusters)]
    )
    adjacent_speed = np.linalg.norm(
        normalized_shaped[:, 1:] - normalized_shaped[:, :-1], axis=2
    ).mean(axis=0) / math.sqrt(normalized_shaped.shape[2])
    temporal_centroids = normalized_shaped.mean(axis=0)
    centroid_speed = np.linalg.norm(
        temporal_centroids[1:] - temporal_centroids[:-1], axis=1
    ) / math.sqrt(normalized_shaped.shape[2])
    per_sample_path_length = np.linalg.norm(
        normalized_shaped[:, 1:] - normalized_shaped[:, :-1], axis=2
    ).sum(axis=1) / math.sqrt(normalized_shaped.shape[2])

    scatter_trajectories(
        shaped_scores, shaped_labels, frame_positions, explained,
        f"Wan-VAE {space} latent trajectories", output_dir / "pca_trajectory.png"
    )
    cluster_timeline(shaped_labels, frame_positions, output_dir / "cluster_timeline.png")
    heatmap(
        distances, frame_positions, frame_positions,
        "Mean pairwise temporal feature distance", output_dir / "temporal_distance.png"
    )
    heatmap(
        distribution, [f"cluster {i}" for i in range(clusters)], frame_positions,
        "Cluster proportion by source frame", output_dir / "cluster_distribution.png"
    )
    heatmap(
        transitions, [f"from {i}" for i in range(clusters)],
        [f"to {i}" for i in range(clusters)],
        "Temporal cluster transition probability", output_dir / "cluster_transition.png"
    )
    write_rows(output_dir / "points.csv", video_paths, frame_positions, shaped_scores, shaped_labels)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    for sample in range(features.shape[0]):
        sample_trajectory_plot(
            shaped_scores[sample], shaped_labels[sample], frame_positions, explained,
            f"Sample {sample:03d}: {space} PCA trajectory",
            sample_dir / f"sample_{sample:03d}.png",
        )

    summary: dict[str, object] = {
        "space": space,
        "num_samples": int(features.shape[0]),
        "latent_times": int(features.shape[1]),
        "feature_dimensions": int(features.shape[2]),
        "source_frame_positions": frame_positions.astype(int).tolist(),
        "pca_explained_variance": explained.tolist(),
        "pca_explained_variance_pc1_pc2": float(explained[:2].sum()),
        "clusters": clusters,
        "kmeans_inertia": inertia,
        "silhouette_score": score,
        "auto_k_silhouette_scores": {str(k): value for k, value in candidates.items()},
        "cluster_distribution_by_time": distribution.tolist(),
        "cluster_transition_probability": transitions.tolist(),
        "mean_adjacent_feature_speed": adjacent_speed.tolist(),
        "temporal_centroid_speed": centroid_speed.tolist(),
        "per_sample_path_length": per_sample_path_length.tolist(),
        "per_sample_cluster_sequence": shaped_labels.tolist(),
        "mean_pairwise_temporal_distance": distances.tolist(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def extract_features(
    paths: list[Path], args: argparse.Namespace, preview_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[int, ...]]:
    from diffusers import AutoencoderKLWan

    from fastgen.datasets.wds_dataloaders import transform_video

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu --dtype float32")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    if device.type == "cpu" and dtype == torch.bfloat16:
        raise ValueError("Use --dtype float32 with --device cpu")
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id,
        cache_dir=os.environ.get("HF_HOME"),
        subfolder="vae",
        local_files_only=os.environ.get("LOCAL_FILES_ONLY", "false").lower()
        in {"1", "true", "yes"},
        torch_dtype=torch.float32,
    ).eval().requires_grad_(False)
    vae.to(device=device, dtype=dtype)

    def encode(video: torch.Tensor) -> torch.Tensor:
        video = video.to(device=device, dtype=dtype)
        if args.encode_mode == "framewise":
            raw = torch.cat(
                [
                    vae.encode(video[:, :, index : index + 1], return_dict=False)[0].mode()
                    for index in range(video.shape[2])
                ],
                dim=2,
            )
        else:
            raw = vae.encode(video, return_dict=False)[0].mode()
        mean = torch.as_tensor(
            vae.config.latents_mean, device=device, dtype=dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        inverse_std = torch.as_tensor(
            vae.config.latents_std, device=device, dtype=dtype
        ).reciprocal().view(1, vae.config.z_dim, 1, 1, 1)
        return (raw - mean) * inverse_std

    collected: list[np.ndarray] = []
    used_paths: list[str] = []
    latent_shape: tuple[int, ...] | None = None
    for path in paths:
        frames = decode_selected_frames(path, args.start_frame, args.frames)
        if frames is None or len(frames) < args.frames:
            print(f"skip undecodable/short video: {path}", flush=True)
            continue
        video = transform_video(frames, args.frames, (args.width, args.height))["real"]
        with torch.inference_mode():
            latent = encode(video[None])[0]
        features = latent_features(latent, args.feature_pool, args.grid_size)
        if collected and features.shape != collected[0].shape:
            print(f"skip inconsistent latent shape {features.shape}: {path}", flush=True)
            continue
        collected.append(features)
        used_paths.append(str(path))
        latent_shape = tuple(int(value) for value in latent.shape)
        if preview_dir is not None:
            preview_dir.mkdir(parents=True, exist_ok=True)
            save_preview(video, preview_dir / f"sample_{len(collected) - 1:03d}.jpg")
        print(f"encoded {len(collected)}/{len(paths)}: {path}", flush=True)
    if not collected or latent_shape is None:
        raise RuntimeError("No usable videos were encoded")
    features = np.stack(collected)
    if args.encode_mode == "framewise":
        frame_positions = np.arange(args.start_frame, args.start_frame + features.shape[1])
    else:
        frame_positions = np.rint(
            np.linspace(args.start_frame, args.start_frame + args.frames - 1, features.shape[1])
        ).astype(int)
    return features, frame_positions, used_paths, latent_shape


def build_web_report(
    output_dir: Path,
    used_paths: list[str],
    frame_positions: np.ndarray,
    summaries: dict[str, dict[str, object]],
    args: argparse.Namespace,
) -> None:
    absolute = summaries.get("absolute")
    delta = summaries.get("delta")
    cards = []
    for sample, video_path in enumerate(used_paths):
        absolute_path = (absolute or {}).get("per_sample_path_length", [None] * len(used_paths))[sample]
        delta_path = (delta or {}).get("per_sample_path_length", [None] * len(used_paths))[sample]
        absolute_clusters = (absolute or {}).get("per_sample_cluster_sequence", [[]] * len(used_paths))[sample]
        delta_clusters = (delta or {}).get("per_sample_cluster_sequence", [[]] * len(used_paths))[sample]
        cards.append(
            {
                "sample": sample,
                "video_path": video_path,
                "absolute_path_length": absolute_path,
                "delta_path_length": delta_path,
                "absolute_clusters": absolute_clusters,
                "delta_clusters": delta_clusters,
            }
        )
    payload = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    overview_spaces = [space for space in ("absolute", "delta") if space in summaries]
    overview = "".join(
        f"""<section class=\"space\"><h2>{space.title()} feature space</h2>
        <p>PC1+PC2: {float(summaries[space]['pca_explained_variance_pc1_pc2']):.1%};
        selected K={summaries[space]['clusters']}; silhouette={float(summaries[space]['silhouette_score']):.3f}</p>
        <div class=\"overview-grid\">
          <figure><img src=\"{space}/pca_trajectory.png\"><figcaption>all trajectories</figcaption></figure>
          <figure><img src=\"{space}/cluster_timeline.png\"><figcaption>cluster timeline</figcaption></figure>
          <figure><img src=\"{space}/temporal_distance.png\"><figcaption>temporal distance</figcaption></figure>
          <figure><img src=\"{space}/cluster_transition.png\"><figcaption>cluster transition</figcaption></figure>
        </div></section>"""
        for space in overview_spaces
    )
    document = f"""<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Wan-VAE 时序 PCA 与聚类分析</title>
<style>
:root {{--bg:#f3f6fa;--panel:#fff;--ink:#172033;--muted:#64748b;--line:#dbe3ef;--accent:#2563eb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
header,main{{max-width:1480px;margin:auto;padding:24px}} header h1{{margin:0 0 6px;font-size:29px}}
.meta{{color:var(--muted)}} .space,.controls,.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:0 5px 18px #1e293b0b}}
.space{{padding:20px;margin:18px 0}} .overview-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}
figure{{margin:0}} img{{display:block;width:100%;height:auto;border-radius:8px;background:#eef2f7}} figcaption{{padding:5px;color:var(--muted);text-align:center}}
.controls{{position:sticky;top:8px;z-index:5;padding:12px 16px;margin:20px 0;display:flex;gap:12px;align-items:center}}
select{{padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:white}}
#examples{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}} .card{{overflow:hidden}}
.card-head{{display:flex;justify-content:space-between;gap:12px;padding:14px 16px}} .id{{font-weight:700;color:var(--accent)}}
.path{{color:var(--muted);font:12px/1.35 ui-monospace,monospace;overflow-wrap:anywhere}} .preview{{width:100%;border-radius:0}}
.plots{{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:10px}} .metrics{{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:0 14px 16px}}
.metric{{background:#f8fafc;border-radius:8px;padding:9px}} .metric b{{display:block}} .clusters{{font:12px ui-monospace,monospace;color:var(--muted)}}
@media(max-width:1000px){{.overview-grid,#examples{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Wan-VAE 时序 PCA 与聚类分析</h1>
<div class=\"meta\">模型：{html.escape(args.model_id)} · 编码：{args.encode_mode} · 特征：{args.feature_pool} ·
样例：{len(used_paths)} · 输入帧：{','.join(map(str, frame_positions.tolist()))}</div></header><main>
{overview}
<div class=\"controls\"><b>多案例浏览</b><label>排序 <select id=\"sort\"><option value=\"sample\">样例编号</option>
<option value=\"delta\">Delta 变化强度</option><option value=\"absolute\">Absolute 变化强度</option></select></label></div>
<div id=\"examples\"></div></main><script>
const samples={payload}; const root=document.querySelector('#examples');
const f=v=>v==null?'n/a':Number(v).toFixed(3);
function render(){{let rows=[...samples],key=document.querySelector('#sort').value;
 if(key==='delta')rows.sort((a,b)=>(b.delta_path_length??-1)-(a.delta_path_length??-1));
 if(key==='absolute')rows.sort((a,b)=>(b.absolute_path_length??-1)-(a.absolute_path_length??-1));
 if(key==='sample')rows.sort((a,b)=>a.sample-b.sample);
 root.innerHTML=rows.map(s=>`<article class=\"card\"><div class=\"card-head\"><div><div class=\"id\">Sample ${{String(s.sample).padStart(3,'0')}}</div><div class=\"path\">${{s.video_path}}</div></div></div>
 <img class=\"preview\" loading=\"lazy\" src=\"previews/sample_${{String(s.sample).padStart(3,'0')}}.jpg\">
 <div class=\"plots\"><figure><img loading=\"lazy\" src=\"absolute/samples/sample_${{String(s.sample).padStart(3,'0')}}.png\"><figcaption>absolute PCA</figcaption></figure>
 <figure><img loading=\"lazy\" src=\"delta/samples/sample_${{String(s.sample).padStart(3,'0')}}.png\"><figcaption>delta PCA</figcaption></figure></div>
 <div class=\"metrics\"><div class=\"metric\"><b>Absolute path ${{f(s.absolute_path_length)}}</b><span class=\"clusters\">clusters: ${{s.absolute_clusters.join(' → ')}}</span></div>
 <div class=\"metric\"><b>Delta path ${{f(s.delta_path_length)}}</b><span class=\"clusters\">clusters: ${{s.delta_clusters.join(' → ')}}</span></div></div></article>`).join('')}}
document.querySelector('#sort').addEventListener('change',render);render();
</script></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    spaces = args.analysis_space or ["absolute", "delta"]
    paths = args.video or read_video_paths(
        args.index, args.path_column, args.num_samples, args.seed
    )
    paths = paths[: args.num_samples]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, frame_positions, used_paths, latent_shape = extract_features(
        paths, args, args.output_dir / "previews"
    )
    np.savez_compressed(
        args.output_dir / "latent_features.npz",
        features=features,
        source_frame_positions=frame_positions,
        video_paths=np.asarray(used_paths),
    )
    summaries = {
        space: analyze_space(
            features, space, frame_positions, used_paths, args.output_dir / space, args
        )
        for space in dict.fromkeys(spaces)
    }
    run_summary = {
        "model_id": args.model_id,
        "encode_mode": args.encode_mode,
        "feature_pool": args.feature_pool,
        "grid_size": args.grid_size if args.feature_pool == "spatial-grid" else None,
        "input_frames": args.frames,
        "input_resolution": [args.width, args.height],
        "latent_shape_chw_per_video": list(latent_shape),
        "video_paths": used_paths,
        "analyses": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    build_web_report(args.output_dir, used_paths, frame_positions, summaries, args)
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
