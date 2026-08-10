# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming local-video loader for the Cosmos3 WebData CSV indices."""

from __future__ import annotations

import csv
import os
import random
from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from fastgen.datasets.decoders import decode_video_segment
from fastgen.datasets.wds_dataloaders import transform_video
from fastgen.utils.distributed import get_rank, world_size


class CSVVideoDataset(IterableDataset):
    def __init__(
        self,
        index_path: str,
        sequence_length: int,
        img_size: tuple[int, int],
        caption_columns: Sequence[str] = ("caption_l3", "caption_l2", "caption_l1"),
        negative_prompt: str = "",
        train: bool = True,
        shuffle_size: int = 1000,
        seed: int = 10,
        sampler_start_idx: int = 0,
        dataset_size: int | None = None,
    ):
        super().__init__()
        if not os.path.isfile(index_path):
            raise FileNotFoundError(index_path)
        self.index_path = index_path
        self.sequence_length = int(sequence_length)
        self.img_size = tuple(img_size)
        self.caption_columns = tuple(caption_columns)
        self.negative_prompt = negative_prompt
        self.train = train
        self.shuffle_size = int(shuffle_size)
        self.seed = int(seed)
        self.sampler_start_idx = int(sampler_start_idx or 0)
        self.dataset_size = dataset_size

    def _rows(self) -> Iterator[dict]:
        with open(self.index_path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                try:
                    frame_count = int(float(row["n_frames"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if frame_count < self.sequence_length:
                    continue
                caption = next(
                    (row.get(key, "").strip() for key in self.caption_columns if row.get(key, "").strip()),
                    "",
                )
                path = row.get("video_path", "").strip()
                if caption and path:
                    yield {"path": path, "caption": caption}

    @staticmethod
    def _shuffle(rows: Iterator[dict], size: int, rng: random.Random) -> Iterator[dict]:
        buffer = []
        for row in rows:
            if len(buffer) < size:
                buffer.append(row)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = row
        rng.shuffle(buffer)
        yield from buffer

    def _prepare(self, row: dict) -> dict | None:
        if not os.path.isfile(row["path"]):
            return None
        video = decode_video_segment(
            row["path"], row["path"], self.sequence_length, output_format="torch"
        )
        if video is None or video.shape[0] < self.sequence_length:
            return None
        transformed = transform_video(video, self.sequence_length, self.img_size)
        return {
            "real": transformed["real"],
            "condition": row["caption"],
            "neg_condition": self.negative_prompt,
            "fname": row["path"],
            "shard": self.index_path,
        }

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        global_worker = get_rank() * num_workers + worker_id
        total_workers = world_size() * num_workers
        epoch = self.sampler_start_idx // self.dataset_size if self.dataset_size else 0
        while True:
            rows = self._rows()
            if self.train and self.shuffle_size > 1:
                rows = self._shuffle(rows, self.shuffle_size, random.Random(self.seed + epoch))
            for position, row in enumerate(rows):
                absolute = epoch * self.dataset_size + position if self.dataset_size else position
                if absolute < self.sampler_start_idx or absolute % total_workers != global_worker:
                    continue
                sample = self._prepare(row)
                if sample is not None:
                    yield sample
            if not self.train:
                return
            epoch += 1


class CSVVideoLoader:
    def __init__(self, batch_size: int, num_workers: int = 2, **dataset_kwargs):
        self.batch_size = int(batch_size)
        dataset = CSVVideoDataset(**dataset_kwargs)
        self.loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=dataset.train,
        )

    def __iter__(self):
        return iter(self.loader)
