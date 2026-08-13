#!/usr/bin/env python3
"""Evaluate single-step Wan2.2 TI2V reconstruction with and without CFG.

The experiment uses paired inputs: every guidance scale sees the same clean
latent, Gaussian noise, text prompt, first-frame condition, and RF timestep.
Pixel metrics are reported against both the source video and its VAE
reconstruction so that VAE error is separated from denoiser error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from fastgen.configs.net import Wan22_I2V_5B_Config
from fastgen.datasets.decoders import decode_video_segment
from fastgen.datasets.wds_dataloaders import transform_video
from fastgen.utils import instantiate
from fastgen.utils.basic_utils import save_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(
            "/mnt/dataset/cosmos3-dataset/data_index/webdata/"
            "all_final_selected_sixth_10k_f17_seed10.csv"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Episode-tree dataset with paired <stem>.mp4 and <stem>_prompt.txt files.",
    )
    parser.add_argument(
        "--exclude-dir-patterns",
        nargs="*",
        default=["badcase", "失败case", "from_realrobot", "test", ".claude"],
        help="Case-insensitive path-component substrings excluded in dataset-dir mode.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--timesteps", type=float, nargs="+", default=[0.08, 0.10, 0.12])
    parser.add_argument("--cfg-scales", type=float, nargs="+", default=[1.0, 3.0, 6.0])
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def init_distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    return rank, world_size, torch.device("cuda", local_rank)


def read_rows(path: Path, count: int, seed: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:count]


def scan_episode_rows(root: Path, exclude_patterns: list[str]) -> list[dict[str, str]]:
    """Find simulation RGB videos and their sidecar prompts."""
    patterns = [pattern.casefold() for pattern in exclude_patterns]
    rows = []
    for base, directories, files in os.walk(root):
        relative = Path(base).relative_to(root)
        excluded = any(
            pattern in component.casefold()
            for component in relative.parts
            for pattern in patterns
        )
        if excluded:
            directories[:] = []
            continue
        names = set(files)
        for name in files:
            if not name.endswith(".mp4") or name.endswith("_depth.mp4"):
                continue
            prompt_name = f"{name[:-4]}_prompt.txt"
            if prompt_name not in names:
                continue
            video_path = Path(base) / name
            prompt_path = Path(base) / prompt_name
            prompt = prompt_path.read_text(encoding="utf-8", errors="replace").strip()
            if prompt:
                rows.append({"video_path": str(video_path), "caption": prompt})
    return sorted(rows, key=lambda row: row["video_path"])


def dataset_rows(
    args: argparse.Namespace, rank: int, world_size: int
) -> list[dict[str, str]]:
    if args.dataset_dir is None:
        return read_rows(args.index, args.num_samples, args.seed)

    manifest = args.output_dir / "dataset_manifest.json"
    if rank == 0:
        rows = scan_episode_rows(args.dataset_dir, args.exclude_dir_patterns)
        manifest.write_text(json.dumps(rows), encoding="utf-8")
        print(f"Discovered {len(rows)} eligible episode/prompt pairs", flush=True)
    if world_size > 1:
        dist.barrier()
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    random.Random(args.seed).shuffle(rows)
    return rows[: args.num_samples]


def caption(row: dict[str, str]) -> str:
    return next(
        (
            row.get(key, "").strip()
            for key in ("caption_l3", "caption_l2", "caption_l1")
            if row.get(key, "").strip()
        ),
        "",
    )


def _unit_range(video: torch.Tensor) -> torch.Tensor:
    return ((video.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def reconstruction_metrics(
    prediction: torch.Tensor, target: torch.Tensor, skip_first_frame: bool = True
) -> dict[str, float]:
    """Return pixel MSE, PSNR, SSIM, and temporal-difference MSE."""
    prediction = _unit_range(prediction)
    target = _unit_range(target)
    if skip_first_frame and prediction.shape[2] > 1:
        spatial_prediction = prediction[:, :, 1:]
        spatial_target = target[:, :, 1:]
    else:
        spatial_prediction = prediction
        spatial_target = target

    mse = F.mse_loss(spatial_prediction, spatial_target).item()
    psnr = -10.0 * math.log10(max(mse, 1e-12))

    # Windowed SSIM averaged over channels, frames, and spatial positions.
    pred_2d = spatial_prediction.permute(0, 2, 1, 3, 4).flatten(0, 1)
    target_2d = spatial_target.permute(0, 2, 1, 3, 4).flatten(0, 1)
    mu_pred = F.avg_pool2d(pred_2d, 11, stride=1, padding=5)
    mu_target = F.avg_pool2d(target_2d, 11, stride=1, padding=5)
    var_pred = F.avg_pool2d(pred_2d.square(), 11, stride=1, padding=5) - mu_pred.square()
    var_target = F.avg_pool2d(target_2d.square(), 11, stride=1, padding=5) - mu_target.square()
    covariance = (
        F.avg_pool2d(pred_2d * target_2d, 11, stride=1, padding=5)
        - mu_pred * mu_target
    )
    c1, c2 = 0.01**2, 0.03**2
    ssim = (
        (2 * mu_pred * mu_target + c1)
        * (2 * covariance + c2)
        / ((mu_pred.square() + mu_target.square() + c1) * (var_pred + var_target + c2))
    ).mean().item()

    pred_delta = prediction[:, :, 1:] - prediction[:, :, :-1]
    target_delta = target[:, :, 1:] - target[:, :, :-1]
    temporal_mse = F.mse_loss(pred_delta, target_delta).item()
    return {"mse": mse, "psnr": psnr, "ssim": ssim, "temporal_mse": temporal_mse}


def decode(net, latents: torch.Tensor) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return net.vae.decode(latents).float()


def preserve_first_latent(output: torch.Tensor, first_frame: torch.Tensor) -> torch.Tensor:
    output = output.clone()
    output[:, :, 0] = first_frame[:, :, 0]
    return output


def aggregate(output_dir: Path, world_size: int, args: argparse.Namespace) -> None:
    records = []
    for rank in range(world_size):
        records.extend(json.loads((output_dir / f"rank_{rank}.json").read_text()))

    grouped: dict[tuple[float, float], list[dict[str, float]]] = defaultdict(list)
    for record in records:
        grouped[(record["timestep"], record["cfg_scale"])].append(record)

    summary_rows = []
    metric_names = [
        "latent_mse",
        "source_mse",
        "source_psnr",
        "source_ssim",
        "source_temporal_mse",
        "vae_mse",
        "vae_psnr",
        "vae_ssim",
        "vae_temporal_mse",
    ]
    for (timestep, cfg_scale), items in sorted(grouped.items()):
        row = {"timestep": timestep, "cfg_scale": cfg_scale, "num_samples": len(items)}
        for name in metric_names:
            values = np.asarray([item[name] for item in items], dtype=np.float64)
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_median"] = float(np.median(values))
            row[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)

    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    paired = []
    baseline = {(r["sample"], r["timestep"]): r for r in records if r["cfg_scale"] == 1.0}
    for scale in args.cfg_scales:
        if scale == 1.0:
            continue
        for timestep in args.timesteps:
            treatment = [r for r in records if r["cfg_scale"] == scale and r["timestep"] == timestep]
            deltas = defaultdict(list)
            for item in treatment:
                control = baseline[(item["sample"], item["timestep"])]
                for name in metric_names:
                    deltas[name].append(item[name] - control[name])
            pair_row = {
                "timestep": timestep,
                "cfg_scale": scale,
                "comparison": "cfg_minus_conditional_only",
                "num_samples": len(treatment),
            }
            for name, values in deltas.items():
                pair_row[f"{name}_mean_delta"] = float(np.mean(values))
                pair_row[f"{name}_median_delta"] = float(np.median(values))
                lower_is_better = "psnr" not in name and "ssim" not in name
                pair_row[f"{name}_cfg_win_rate"] = float(
                    np.mean(np.asarray(values) < 0 if lower_is_better else np.asarray(values) > 0)
                )
            paired.append(pair_row)

    result = {
        "experiment": "wan22_ti2v_single_step_cfg_reconstruction",
        "paired_design": True,
        "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "num_samples": len({record["sample"] for record in records}),
        "timesteps": args.timesteps,
        "cfg_scales": args.cfg_scales,
        "frames": args.frames,
        "resolution": [args.width, args.height],
        "spatial_metrics_skip_pixel_frame_zero": True,
        "negative_prompt": "empty string",
        "summary": summary_rows,
        "paired_deltas": paired,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "records.json").open("w") as handle:
        json.dump(records, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if 1.0 not in args.cfg_scales:
        raise ValueError("cfg-scales must include 1.0 as the conditional-only baseline")
    if any(not 0 < timestep < 1 for timestep in args.timesteps):
        raise ValueError("timesteps must be in (0, 1)")

    rank, world_size, device = init_distributed()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = dataset_rows(args, rank, world_size)
    indexed_rows = list(enumerate(rows))[rank::world_size]
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    config = Wan22_I2V_5B_Config.copy()
    config.disable_grad_ckpt = True
    net = instantiate(config)
    net.init_preprocessors()
    net.eval().requires_grad_(False)
    net.to(device=device, dtype=torch.bfloat16)

    records = []
    for sample_index, row in indexed_rows:
        path = row["video_path"]
        # Read bytes for compatibility with both the legacy WebDataset decoder
        # and the newer decoder that also accepts local paths.
        frames = decode_video_segment(path, Path(path).read_bytes(), args.frames, output_format="torch")
        if frames is None or frames.shape[0] < args.frames:
            print(f"[rank {rank}] skipped undecodable sample {sample_index}: {path}", flush=True)
            continue
        source = transform_video(frames, args.frames, (args.width, args.height))["real"]
        source = source.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        prompt = caption(row)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            clean_latent = net.vae.encode(source, mode="argmax")
            first_frame = net.vae.encode(source[:, :, :1], mode="argmax")
            cond_text = net.text_encoder.encode([prompt], precision=torch.bfloat16)
            uncond_text = net.text_encoder.encode([""], precision=torch.bfloat16)
        vae_reconstruction = decode(net, clean_latent)

        generator = torch.Generator(device=device).manual_seed(args.seed * 100000 + sample_index)
        epsilon = torch.randn(clean_latent.shape, generator=generator, device=device, dtype=clean_latent.dtype)
        sample_dir = args.output_dir / f"sample_{sample_index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "metadata.json").write_text(
            json.dumps({"video_path": path, "caption": prompt}, indent=2), encoding="utf-8"
        )
        if args.save_videos:
            save_video(source[0], sample_dir / "source.mp4", save_as_gif=False, fps=args.fps, quality=18)
            save_video(vae_reconstruction[0], sample_dir / "vae_reconstruction.mp4", save_as_gif=False, fps=args.fps, quality=18)

        for timestep in args.timesteps:
            t = torch.full((1,), timestep, device=device, dtype=torch.float32)
            noisy = net.noise_scheduler.forward_process(clean_latent, epsilon, t)
            condition = {"text_embeds": cond_text, "first_frame_cond": first_frame}
            uncondition = {"text_embeds": uncond_text, "first_frame_cond": first_frame}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                cond_flow = net(noisy, t, condition=condition, fwd_pred_type="flow")
                uncond_flow = net(noisy, t, condition=uncondition, fwd_pred_type="flow")

            for scale in args.cfg_scales:
                # Keep the conditional-only baseline bit-identical to the model
                # output instead of reconstructing it through BF16 arithmetic.
                guided_flow = cond_flow if scale == 1.0 else uncond_flow + scale * (cond_flow - uncond_flow)
                restored_latent = net.noise_scheduler.flow_to_x0(noisy, guided_flow, t)
                restored_latent = preserve_first_latent(restored_latent, first_frame)
                restored = decode(net, restored_latent)
                source_metrics = reconstruction_metrics(restored, source)
                vae_metrics = reconstruction_metrics(restored, vae_reconstruction)
                record = {
                    "sample": sample_index,
                    "video_path": path,
                    "timestep": timestep,
                    "cfg_scale": scale,
                    "latent_mse": F.mse_loss(
                        restored_latent[:, :, 1:].float(), clean_latent[:, :, 1:].float()
                    ).item(),
                    **{f"source_{key}": value for key, value in source_metrics.items()},
                    **{f"vae_{key}": value for key, value in vae_metrics.items()},
                }
                records.append(record)
                if args.save_videos:
                    scale_name = "conditional_only" if scale == 1.0 else f"cfg_{scale:g}"
                    save_video(
                        restored[0],
                        sample_dir / f"t_{timestep:.2f}_{scale_name}.mp4",
                        save_as_gif=False,
                        fps=args.fps,
                        quality=18,
                    )
                print(
                    f"[rank {rank}] sample={sample_index} t={timestep:.2f} cfg={scale:g} "
                    f"PSNR={source_metrics['psnr']:.3f} SSIM={source_metrics['ssim']:.4f}",
                    flush=True,
                )

    (args.output_dir / f"rank_{rank}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        aggregate(args.output_dir, world_size, args)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
