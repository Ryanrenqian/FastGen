# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute NVIDIA EDM-compatible FID reference statistics."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from fastgen.networks.inception import InceptionV3
from fastgen.utils.distributed import clean_up, ddp, is_rank0, synchronize
from scripts.fid.fid import calculate_inception_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Image directory or EDM-format zip")
    parser.add_argument("--dest", required=True, help="Destination .npz file")
    parser.add_argument("--batch", type=int, default=64, help="Maximum batch size per GPU")
    parser.add_argument("--workers", type=int, default=3, help="DataLoader workers per rank")
    parser.add_argument("--num", type=int, default=None, help="Optional deterministic dataset subset size")
    parser.add_argument("--seed", type=int, default=0, help="Subset selection seed")
    args = parser.parse_args()

    if not args.dest.endswith(".npz"):
        parser.error("--dest must end in .npz")

    ddp.init()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    detector = InceptionV3().to(device).eval()
    mu, sigma = calculate_inception_stats(
        detector,
        feature_dim=2048,
        image_path=args.data,
        num_expected=args.num,
        seed=args.seed,
        max_batch_size=args.batch,
        num_workers=args.workers,
        device=device,
    )

    if is_rank0():
        dest = Path(args.dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dest, mu=mu, sigma=sigma)
        print(f"Saved FID reference statistics to {dest}")
    synchronize()
    clean_up()


if __name__ == "__main__":
    main()
