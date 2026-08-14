#!/usr/bin/env python3
"""Paired motion/blur metrics for HYP-21 generated-video A/B outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--treatment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--motion-quantile", type=float, default=0.8)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def load_video(path: Path) -> np.ndarray:
    frames = iio.imread(path, plugin="pyav")
    if frames.ndim != 4 or frames.shape[-1] < 3 or len(frames) < 2:
        raise ValueError(f"invalid video tensor from {path}: {frames.shape}")
    return frames[..., :3].astype(np.float32) / 255.0


def luminance(video: np.ndarray) -> np.ndarray:
    return (
        0.2126 * video[..., 0]
        + 0.7152 * video[..., 1]
        + 0.0722 * video[..., 2]
    )


def laplacian(video_y: np.ndarray) -> np.ndarray:
    padded = np.pad(video_y, ((0, 0), (1, 1), (1, 1)), mode="reflect")
    return (
        -4.0 * padded[:, 1:-1, 1:-1]
        + padded[:, :-2, 1:-1]
        + padded[:, 2:, 1:-1]
        + padded[:, 1:-1, :-2]
        + padded[:, 1:-1, 2:]
    )


def symmetric_motion_mask(
    control_y: np.ndarray, treatment_y: np.ndarray, quantile: float
) -> np.ndarray:
    control_activity = np.abs(np.diff(control_y, axis=0)).mean(axis=0)
    treatment_activity = np.abs(np.diff(treatment_y, axis=0)).mean(axis=0)
    activity = np.maximum(control_activity, treatment_activity)
    threshold = np.quantile(activity, quantile)
    mask = activity >= threshold
    if not mask.any():
        mask[...] = True
    return mask


def metrics(video: np.ndarray, motion_mask: np.ndarray) -> dict[str, float]:
    video_y = luminance(video)
    temporal_delta = np.abs(np.diff(video_y, axis=0))
    lap = laplacian(video_y)
    motion_lap = np.square(lap[:, motion_mask])
    # High-frequency temporal change is reported separately from spatial motion.
    # It is a flicker/ghosting proxy, not a perceptual quality score.
    hf_delta = np.abs(np.diff(lap, axis=0))
    return {
        "temporal_l1": float(temporal_delta.mean()),
        "motion_temporal_l1": float(temporal_delta[:, motion_mask].mean()),
        "motion_laplacian_energy": float(motion_lap.mean()),
        "motion_laplacian_p90": float(np.quantile(np.abs(lap[:, motion_mask]), 0.9)),
        "motion_hf_temporal_delta": float(hf_delta[:, motion_mask].mean()),
    }


def paired_files(control_dir: Path, treatment_dir: Path) -> list[tuple[Path, Path]]:
    control = {path.name: path for path in control_dir.glob("*.mp4")}
    treatment = {path.name: path for path in treatment_dir.glob("*.mp4")}
    names = sorted(control.keys() & treatment.keys())
    if not names:
        raise FileNotFoundError("no same-name MP4 pairs found")
    missing_control = sorted(treatment.keys() - control.keys())
    missing_treatment = sorted(control.keys() - treatment.keys())
    if missing_control or missing_treatment:
        raise ValueError(
            f"unpaired videos: control_missing={missing_control}, "
            f"treatment_missing={missing_treatment}"
        )
    return [(control[name], treatment[name]) for name in names]


def bootstrap_interval(values: np.ndarray, samples: int, rng: np.random.Generator):
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    means = values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    args = parse_args()
    if not 0.0 < args.motion_quantile < 1.0:
        raise ValueError("motion_quantile must be between zero and one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for control_path, treatment_path in paired_files(
        args.control_dir, args.treatment_dir
    ):
        control = load_video(control_path)
        treatment = load_video(treatment_path)
        if control.shape != treatment.shape:
            raise ValueError(
                f"shape mismatch for {control_path.name}: "
                f"{control.shape} vs {treatment.shape}"
            )
        mask = symmetric_motion_mask(
            luminance(control), luminance(treatment), args.motion_quantile
        )
        rows.append(
            {
                "name": control_path.name,
                "motion_mask_fraction": float(mask.mean()),
                "control": metrics(control, mask),
                "treatment": metrics(treatment, mask),
            }
        )

    rng = np.random.default_rng(args.seed)
    summary = {}
    for key in rows[0]["control"]:
        control_values = np.array([row["control"][key] for row in rows])
        treatment_values = np.array([row["treatment"][key] for row in rows])
        difference = treatment_values - control_values
        relative = difference / np.maximum(np.abs(control_values), 1e-12)
        summary[key] = {
            "control_mean": float(control_values.mean()),
            "treatment_mean": float(treatment_values.mean()),
            "mean_difference": float(difference.mean()),
            "difference_ci95": bootstrap_interval(
                difference, args.bootstrap_samples, rng
            ),
            "mean_relative_change": float(relative.mean()),
            "treatment_win_fraction": float((difference > 0).mean()),
        }

    result = {
        "num_pairs": len(rows),
        "motion_mask_definition": (
            "top temporal-activity pixels from the symmetric max of control "
            "and treatment; identical mask for both arms"
        ),
        "motion_quantile": args.motion_quantile,
        "summary": summary,
        "pairs": rows,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
