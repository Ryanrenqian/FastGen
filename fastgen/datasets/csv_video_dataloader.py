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
        positive_frame_strides: Sequence[int] = (1,),
        positive_random_walk_count: int = 0,
        positive_stride1_repeat_count: int = 0,
        positive_random_step_min: int = 1,
        positive_random_step_max: int = 3,
        positive_decode_step_max: int | None = None,
        frame_stride: int = 1,
        frame_start: int | None = None,
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
        self.frame_stride = int(frame_stride)
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        self.frame_start = None if frame_start is None else int(frame_start)
        if self.frame_start is not None and self.frame_start < 0:
            raise ValueError("frame_start must be non-negative when provided")
        self.positive_frame_strides = tuple(int(stride) for stride in positive_frame_strides)
        if not self.positive_frame_strides or any(
            stride <= 0 for stride in self.positive_frame_strides
        ):
            raise ValueError("positive_frame_strides must contain positive integers")
        if len(set(self.positive_frame_strides)) != len(self.positive_frame_strides):
            raise ValueError("positive_frame_strides must not contain duplicates")
        if self.positive_frame_strides[0] != 1:
            raise ValueError("positive_frame_strides must start with 1 for I2V conditioning")
        self.positive_random_walk_count = int(positive_random_walk_count)
        self.positive_stride1_repeat_count = int(positive_stride1_repeat_count)
        self.positive_random_step_min = int(positive_random_step_min)
        self.positive_random_step_max = int(positive_random_step_max)
        if self.positive_random_walk_count < 0:
            raise ValueError("positive_random_walk_count must be non-negative")
        if self.positive_stride1_repeat_count < 0:
            raise ValueError("positive_stride1_repeat_count must be non-negative")
        if not (
            1
            <= self.positive_random_step_min
            <= self.positive_random_step_max
        ):
            raise ValueError("positive random steps must be positive and ordered")
        required_step_max = max(
            max(self.positive_frame_strides),
            self.positive_random_step_max if self.positive_random_walk_count else 1,
            self.frame_stride,
        )
        self.positive_decode_step_max = (
            required_step_max
            if positive_decode_step_max is None
            else int(positive_decode_step_max)
        )
        if self.positive_decode_step_max < required_step_max:
            raise ValueError(
                "positive_decode_step_max must cover every positive trajectory"
            )
        self.decode_length = (
            (self.sequence_length - 1)
            * self.positive_decode_step_max
            + 1
        )

    def _random_walk_indices(self, rng: random.Random) -> list[int]:
        indices = [0]
        for _ in range(self.sequence_length - 1):
            indices.append(
                indices[-1]
                + rng.randint(
                    self.positive_random_step_min,
                    self.positive_random_step_max,
                )
            )
        return indices

    def _rows(self) -> Iterator[dict]:
        with open(self.index_path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                try:
                    frame_count = int(float(row["n_frames"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if frame_count < self.decode_length + (self.frame_start or 0):
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

    def _prepare(self, row: dict, rng: random.Random | None = None) -> dict | None:
        video = self._load_video(row["path"])
        if video is None or video.shape[0] < self.decode_length:
            return None
        # Resize/crop the full temporal span once so every positive has exactly
        # the same first frame and spatial transform.
        transformed = transform_video(video, self.decode_length, self.img_size)
        full_video = transformed["real"]
        positive_clips = [
            full_video[:, ::stride][:, : self.sequence_length]
            for stride in self.positive_frame_strides
        ]
        positive_clips.extend(
            positive_clips[0] for _ in range(self.positive_stride1_repeat_count)
        )
        if self.positive_random_walk_count:
            if rng is None:
                raise ValueError("random positive trajectories require a seeded RNG")
            for _ in range(self.positive_random_walk_count):
                indices = self._random_walk_indices(rng)
                positive_clips.append(full_video[:, indices])
        positives = torch.stack(positive_clips, dim=0)
        real = full_video[:, :: self.frame_stride][:, : self.sequence_length]
        return {
            "real": real,
            "positive": positives,
            "positive_raw": positives,
            "condition": row["caption"],
            "neg_condition": self.negative_prompt,
            "fname": row["path"],
            "shard": self.index_path,
        }

    def _load_video(self, video_path: str) -> torch.Tensor | None:
        if not os.path.isfile(video_path):
            return None
        return decode_video_segment(
            video_path,
            video_path,
            self.decode_length,
            output_format="torch",
            start_frame=self.frame_start,
        )

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
                # The trajectory changes across epochs but is deterministic for
                # a fixed seed, epoch, manifest position, rank, and worker.
                trajectory_rng = random.Random(
                    self.seed
                    + epoch * 1_000_003
                    + position * 9_973
                    + global_worker * 101
                )
                sample = self._prepare(row, trajectory_rng)
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
