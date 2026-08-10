# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute NVIDIA EDM FID for an existing image directory."""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from fastgen.networks.inception import InceptionV3
from fastgen.utils.distributed import clean_up, ddp, is_rank0, synchronize
from scripts.fid.fid import calculate_fid_from_inception_stats, calculate_inception_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--num", type=int, default=50_000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--detector-pkl", default=None)
    parser.add_argument("--detector-code-root", default=None)
    args = parser.parse_args()

    ddp.init()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    if args.detector_pkl is None:
        detector = InceptionV3().to(device).eval()
    else:
        if args.detector_code_root is None:
            parser.error("--detector-code-root is required with --detector-pkl")
        sys.path.insert(0, args.detector_code_root)
        with open(args.detector_pkl, "rb") as file:
            base_detector = pickle.load(file).to(device).eval()

        class PklDetector(torch.nn.Module):
            def forward(self, images):
                images = ((images + 1) * 127.5).clip(0, 255).to(torch.uint8)
                return base_detector(images, return_features=True).view(images.shape[0], 2048)

        detector = PklDetector()
    mu, sigma = calculate_inception_stats(
        detector,
        feature_dim=2048,
        image_path=args.images,
        num_expected=args.num,
        seed=args.seed,
        max_batch_size=args.batch,
        device=device,
    )

    if is_rank0():
        ref = np.load(args.ref)
        fid = calculate_fid_from_inception_stats(mu, sigma, ref["mu"], ref["sigma"])
        result = {
            "fid": fid,
            "num_images": args.num,
            "images": os.path.abspath(args.images),
            "reference": os.path.abspath(args.ref),
            "detector": os.path.abspath(args.detector_pkl) if args.detector_pkl else "FastGen TorchScript",
        }
        dest = Path(args.dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    synchronize()
    clean_up()


if __name__ == "__main__":
    main()
