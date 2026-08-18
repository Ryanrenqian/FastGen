# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "experiments"
    / "analyze_wan_vae_temporal.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_wan_vae_temporal", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_latent_features_preserves_temporal_axis():
    latent = torch.arange(3 * 5 * 8 * 8, dtype=torch.float32).reshape(3, 5, 8, 8)

    grid = MODULE.latent_features(latent, "spatial-grid", grid_size=2)
    moments = MODULE.latent_features(latent, "channel-moments", grid_size=2)

    assert grid.shape == (5, 3 * 2 * 2)
    assert moments.shape == (5, 3 * 2)
    assert not np.allclose(grid[0], grid[-1])


def test_pca_recovers_one_dimensional_trajectory():
    value = np.linspace(-2, 2, 9, dtype=np.float32)
    features = np.stack([value, value * 3, value * -0.5], axis=1)

    scores, components, explained = MODULE.pca(features, components=2)

    assert scores.shape == (9, 2)
    assert components.shape == (2, 3)
    assert explained[0] > 0.999


def test_kmeans_and_temporal_transitions():
    rng = np.random.default_rng(7)
    values = np.concatenate(
        [rng.normal(-3, 0.05, (20, 2)), rng.normal(3, 0.05, (20, 2))]
    ).astype(np.float32)

    labels, _, inertia = MODULE.kmeans(values, clusters=2, seed=3, restarts=5)
    assert inertia < 1.0
    assert len(np.unique(labels)) == 2
    assert MODULE.silhouette(values, labels) > 0.95

    transitions = MODULE.transition_matrix(np.asarray([[0, 0, 1], [0, 1, 1]]), 2)
    np.testing.assert_allclose(transitions, [[1 / 3, 2 / 3], [0, 1]])


def test_select_clusters_finds_two_separated_groups():
    rng = np.random.default_rng(11)
    values = np.concatenate(
        [rng.normal(-4, 0.1, (24, 3)), rng.normal(4, 0.1, (24, 3))]
    ).astype(np.float32)

    _, _, clusters, _, score, candidates = MODULE.select_clusters(
        values, requested=0, maximum=5, seed=10, restarts=5
    )

    assert clusters == 2
    assert score > 0.95
    assert set(candidates) == {2, 3, 4, 5}


def test_analyze_space_writes_temporal_artifacts(tmp_path):
    rng = np.random.default_rng(19)
    features = rng.normal(0, 0.05, (4, 5, 6)).astype(np.float32)
    features += np.linspace(0, 3, 5, dtype=np.float32)[None, :, None]
    args = Namespace(
        cluster_components=3,
        clusters=2,
        max_clusters=4,
        seed=10,
        kmeans_restarts=3,
    )

    summary = MODULE.analyze_space(
        features,
        "delta",
        np.arange(30, 35),
        [f"video_{index}.mp4" for index in range(4)],
        tmp_path,
        args,
    )

    assert summary["latent_times"] == 5
    assert len(summary["mean_adjacent_feature_speed"]) == 4
    for name in (
        "pca_trajectory.png",
        "cluster_timeline.png",
        "temporal_distance.png",
        "cluster_distribution.png",
        "cluster_transition.png",
        "points.csv",
        "summary.json",
    ):
        assert (tmp_path / name).stat().st_size > 0


def test_build_web_report_contains_all_samples(tmp_path):
    args = Namespace(
        model_id="test/wan-vae", encode_mode="framewise", feature_pool="spatial-grid"
    )
    summaries = {
        space: {
            "pca_explained_variance_pc1_pc2": 0.75,
            "clusters": 2,
            "silhouette_score": 0.5,
            "per_sample_path_length": [1.25, 2.5],
            "per_sample_cluster_sequence": [[0, 1], [1, 1]],
        }
        for space in ("absolute", "delta")
    }

    MODULE.build_web_report(
        tmp_path, ["one.mp4", "two.mp4"], np.asarray([30, 31]), summaries, args
    )

    report = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Wan-VAE 时序 PCA 与聚类分析" in report
    assert "one.mp4" in report and "two.mp4" in report
    assert "delta_path_length" in report


def test_save_preview_writes_all_frames(tmp_path):
    video = torch.linspace(-1, 1, 3 * 4 * 8 * 10).reshape(3, 4, 8, 10)
    path = tmp_path / "preview.jpg"

    MODULE.save_preview(video, path)

    width, height = Image.open(path).size
    assert width == 4 * 10
    assert height == 8 + 28
