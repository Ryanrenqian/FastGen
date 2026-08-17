# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv

import torch

from fastgen.datasets.csv_video_dataloader import CSVVideoDataset, CSVVideoLoader


def test_csv_video_loader_filters_short_rows_and_falls_back_caption(tmp_path, monkeypatch):
    index_path = tmp_path / "index.csv"
    fieldnames = [
        "video_path",
        "n_frames",
        "fps",
        "height",
        "width",
        "caption_l3",
        "caption_l2",
        "caption_l1",
        "kw_class",
    ]
    rows = [
        ["short.mp4", 4, 16, 8, 8, "short", "", "", "test"],
        ["empty.mp4", 5, 16, 8, 8, "", "", "", "test"],
        ["fine.mp4", 5, 16, 8, 8, "fine detail", "medium", "broad", "test"],
        ["fallback.mp4", 5, 16, 8, 8, "", "medium fallback", "broad", "test"],
    ]
    with index_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    monkeypatch.setattr(
        CSVVideoDataset,
        "_load_video",
        lambda self, path: torch.zeros(5, 8, 8, 3, dtype=torch.uint8),
    )
    loader = CSVVideoLoader(
        index_path=str(index_path),
        batch_size=2,
        sequence_length=5,
        img_size=(8, 8),
        negative_prompt="negative",
        num_workers=0,
        train=False,
        shuffle_size=0,
    )

    batch = next(iter(loader))

    assert batch["real"].shape == (2, 3, 5, 8, 8)
    assert batch["condition"] == ["fine detail", "medium fallback"]
    assert batch["neg_condition"] == ["negative", "negative"]


def test_csv_video_loader_builds_stride_positives(tmp_path, monkeypatch):
    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "video_path", "n_frames", "fps", "height", "width",
                "caption_l3", "caption_l2", "caption_l1", "kw_class",
            ]
        )
        writer.writerow(["clip.mp4", 13, 16, 8, 8, "motion", "", "", "test"])

    frames = torch.arange(13, dtype=torch.uint8).reshape(13, 1, 1, 1).expand(-1, 8, 8, 3)
    monkeypatch.setattr(CSVVideoDataset, "_load_video", lambda self, path: frames)
    dataset = CSVVideoDataset(
        index_path=str(index_path),
        sequence_length=5,
        img_size=(8, 8),
        train=False,
        positive_frame_strides=[1, 2, 3],
    )

    sample = next(iter(dataset))

    assert sample["positive"].shape == (3, 3, 5, 8, 8)
    observed = ((sample["positive"][:, 0, :, 0, 0] + 1) * 127.5).round()
    assert torch.equal(
        observed,
        torch.tensor([[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [0, 3, 6, 9, 12]]),
    )


def test_csv_video_loader_samples_fixed_endpoints_with_frame_stride(tmp_path, monkeypatch):
    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "video_path", "n_frames", "fps", "height", "width",
                "caption_l3", "caption_l2", "caption_l1", "kw_class",
            ]
        )
        writer.writerow(["clip.mp4", 49, 30, 8, 8, "motion", "", "", "test"])

    frames = torch.arange(49, dtype=torch.uint8).reshape(49, 1, 1, 1).expand(-1, 8, 8, 3)
    monkeypatch.setattr(CSVVideoDataset, "_load_video", lambda self, path: frames)
    dataset = CSVVideoDataset(
        index_path=str(index_path), sequence_length=17, frame_stride=3,
        img_size=(8, 8), train=False,
    )

    sample = next(iter(dataset))
    observed = ((sample["real"][0, :, 0, 0] + 1) * 127.5).round()
    assert torch.equal(observed, torch.arange(0, 49, 3))


def test_csv_video_loader_starts_at_fixed_frame(tmp_path, monkeypatch):
    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "video_path", "n_frames", "fps", "height", "width",
                "caption_l3", "caption_l2", "caption_l1", "kw_class",
            ]
        )
        writer.writerow(["clip.mp4", 35, 30, 8, 8, "motion", "", "", "test"])

    frames = torch.arange(5, dtype=torch.uint8).reshape(5, 1, 1, 1).expand(-1, 8, 8, 3) + 30
    observed_start_frames = []

    def load_video(self, path):
        observed_start_frames.append(self.frame_start)
        return frames

    monkeypatch.setattr(CSVVideoDataset, "_load_video", load_video)
    dataset = CSVVideoDataset(
        index_path=str(index_path), sequence_length=5, frame_start=30, frame_stride=1,
        img_size=(8, 8), train=False,
    )

    sample = next(iter(dataset))
    observed = ((sample["real"][0, :, 0, 0] + 1) * 127.5).round()
    assert observed_start_frames == [30]
    assert torch.equal(observed, torch.arange(30, 35))
