#!/usr/bin/env python3
"""Build a deterministic clean demo5 CSV manifest for Wan TFD training."""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import av


DEFAULT_EXCLUDES = ("badcase", "失败case", "from_realrobot", "test", ".claude")
FIELDS = (
    "video_path",
    "n_frames",
    "fps",
    "height",
    "width",
    "caption_l3",
    "caption_l1",
    "caption_l2",
    "kw_class",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/dataset/demo5_dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--min-frames", type=int, default=49)
    parser.add_argument("--min-width", type=int, default=640)
    parser.add_argument("--min-height", type=int, default=384)
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDES))
    return parser.parse_args()


def is_excluded(path: Path, root: Path, patterns: list[str]) -> bool:
    relative = path.relative_to(root)
    folded = [pattern.casefold() for pattern in patterns]
    return any(pattern in component.casefold() for component in relative.parts for pattern in folded)


def discover_pairs(root: Path, patterns: list[str]) -> list[tuple[Path, Path]]:
    pairs = []
    for base, directories, files in os.walk(root):
        base_path = Path(base)
        if is_excluded(base_path, root, patterns):
            directories[:] = []
            continue
        names = set(files)
        for name in files:
            if not name.endswith(".mp4") or name.endswith("_depth.mp4"):
                continue
            prompt_name = f"{name[:-4]}_prompt.txt"
            if prompt_name in names:
                pairs.append((base_path / name, base_path / prompt_name))
    return sorted(pairs)


def inspect_video(video_path: Path) -> tuple[int, float, int, int] | None:
    try:
        with av.open(str(video_path), mode="r") as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            fps = float(stream.average_rate or stream.guessed_rate or 0)
            frames = int(stream.frames or 0)
            if frames <= 0 and stream.duration and stream.time_base and fps > 0:
                frames = int(float(stream.duration * stream.time_base) * fps)
            return frames, fps, int(stream.height), int(stream.width)
    except (av.error.FFmpegError, OSError, ValueError):
        return None


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], int]:
    pairs = discover_pairs(args.root, args.exclude)
    random.Random(args.seed).shuffle(pairs)
    rows = []
    for video_path, prompt_path in pairs:
        prompt = " ".join(
            prompt_path.read_text(encoding="utf-8", errors="replace").split()
        )
        if not prompt:
            continue
        metadata = inspect_video(video_path)
        if metadata is None:
            continue
        frames, fps, height, width = metadata
        if frames < args.min_frames or width < args.min_width or height < args.min_height:
            continue
        rows.append(
            {
                "video_path": str(video_path),
                "n_frames": frames,
                "fps": fps,
                "height": height,
                "width": width,
                "caption_l3": prompt,
                "caption_l1": "",
                "caption_l2": "",
                "kw_class": video_path.parent.parent.name,
            }
        )
        if len(rows) == args.size:
            break
    return rows, len(pairs)


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.min_frames <= 0:
        raise ValueError("size and min-frames must be positive")
    if not args.root.is_dir():
        raise FileNotFoundError(f"demo5 root does not exist: {args.root}")

    rows, discovered = build_rows(args)
    if len(rows) != args.size:
        raise RuntimeError(f"Only found {len(rows)} eligible videos; requested {args.size}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, args.output)
    print(f"discovered_pairs={discovered}")
    print(f"selected_rows={len(rows)}")
    print(f"min_frames={args.min_frames}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
