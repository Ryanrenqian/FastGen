#!/usr/bin/env python3
"""Convert Hugging Face ImageNet-1k 64x64 Parquet shards to an EDM zip."""

import argparse
import io
import json
import zipfile
from pathlib import Path

from PIL import Image
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=1_281_167)
    parser.add_argument("--expected-classes", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    return parser.parse_args()


def normalize_image(data: bytes) -> tuple[bytes, str]:
    with Image.open(io.BytesIO(data)) as image:
        if image.size == (64, 64) and image.mode == "RGB" and image.format in {"JPEG", "PNG"}:
            return data, "jpg" if image.format == "JPEG" else "png"
        if image.size != (64, 64):
            raise ValueError(f"Expected a 64x64 image, got {image.size}")
        converted = io.BytesIO()
        image.convert("RGB").save(converted, format="PNG", optimize=False)
        return converted.getvalue(), "png"


def main() -> None:
    args = parse_args()
    shards = sorted(args.input_dir.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No train-*.parquet shards found in {args.input_dir}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    total_rows = sum(pq.ParquetFile(shard).metadata.num_rows for shard in shards)
    if total_rows != args.expected_images:
        raise ValueError(f"Expected {args.expected_images} rows, found {total_rows}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels: list[list[object]] = []
    class_ids: set[int] = set()
    index = 0
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(batch_size=args.batch_size, columns=["image", "label"]):
                images = batch.column(0).field("bytes").to_pylist()
                batch_labels = batch.column(1).to_pylist()
                for data, label in zip(images, batch_labels):
                    if data is None:
                        raise ValueError(f"Missing image bytes at row {index}")
                    image_bytes, extension = normalize_image(data)
                    name = f"{index // 10_000:05d}/img{index:08d}.{extension}"
                    archive.writestr(name, image_bytes)
                    label = int(label)
                    labels.append([name, label])
                    class_ids.add(label)
                    index += 1
                if index % 100_000 < len(images):
                    print(f"Converted {index}/{total_rows} images", flush=True)

        expected_ids = set(range(args.expected_classes))
        if class_ids != expected_ids:
            missing = sorted(expected_ids - class_ids)
            extra = sorted(class_ids - expected_ids)
            raise ValueError(f"Class IDs differ from [0, {args.expected_classes}): missing={missing}, extra={extra}")
        archive.writestr("dataset.json", json.dumps({"labels": labels}, separators=(",", ":")))

    print(f"Wrote {index} images across {len(class_ids)} classes to {args.output}")


if __name__ == "__main__":
    main()
