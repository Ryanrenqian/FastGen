import csv
import random

import pytest
import torch

import fastgen.datasets.csv_video_dataloader as csv_video


def test_csv_video_builds_shared_first_frame_stride_positives(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["video_path", "n_frames", "caption_l3"]
        )
        writer.writeheader()
        writer.writerow(
            {"video_path": str(video_path), "n_frames": 49, "caption_l3": "motion"}
        )
    decoded = torch.arange(49, dtype=torch.uint8).view(49, 1, 1, 1).expand(49, 8, 8, 3)
    monkeypatch.setattr(csv_video, "decode_video_segment", lambda *args, **kwargs: decoded)

    dataset = csv_video.CSVVideoDataset(
        str(index_path), 17, (8, 8), train=False, positive_frame_strides=[1, 2, 3]
    )
    sample = next(iter(dataset))
    positives = sample["positive_raw"]
    assert positives.shape == (3, 3, 17, 8, 8)
    assert torch.equal(sample["real"], positives[0])
    assert torch.equal(
        positives[:, :, 0], positives[0, :, 0].unsqueeze(0).expand(3, -1, -1, -1)
    )
    assert torch.equal(positives[1, :, 1], positives[0, :, 2])
    assert torch.equal(positives[2, :, 1], positives[0, :, 3])


def test_csv_video_builds_four_balanced_random_trajectory_positives(
    tmp_path, monkeypatch
):
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["video_path", "n_frames", "caption_l3"]
        )
        writer.writeheader()
        writer.writerow(
            {"video_path": str(video_path), "n_frames": 49, "caption_l3": "motion"}
        )
    decoded = torch.arange(49, dtype=torch.uint8).view(49, 1, 1, 1).expand(49, 8, 8, 3)
    monkeypatch.setattr(csv_video, "decode_video_segment", lambda *args, **kwargs: decoded)
    dataset = csv_video.CSVVideoDataset(
        str(index_path),
        17,
        (8, 8),
        train=False,
        positive_frame_strides=[1],
        positive_random_walk_count=3,
        positive_random_step_min=1,
        positive_random_step_max=3,
    )

    sample = dataset._prepare(
        {"path": str(video_path), "caption": "motion"}, random.Random(123)
    )
    positives = sample["positive_raw"]

    assert positives.shape == (4, 3, 17, 8, 8)
    assert torch.equal(sample["real"], positives[0])
    assert torch.equal(
        positives[:, :, 0], positives[0, :, 0].unsqueeze(0).expand(4, -1, -1, -1)
    )
    values = ((positives[1:, 0, :, 0, 0] + 1.0) * 127.5).round()
    steps = values[:, 1:] - values[:, :-1]
    assert bool(((steps >= 1) & (steps <= 3)).all())


def test_csv_video_builds_four_exact_stride1_copies_with_matched_decode_window(
    tmp_path, monkeypatch
):
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["video_path", "n_frames", "caption_l3"]
        )
        writer.writeheader()
        writer.writerow(
            {"video_path": str(video_path), "n_frames": 49, "caption_l3": "motion"}
        )
    decoded = torch.arange(49, dtype=torch.uint8).view(49, 1, 1, 1).expand(49, 8, 8, 3)
    decode_requests = []

    def decode(*args, **kwargs):
        decode_requests.append(args[2])
        return decoded

    monkeypatch.setattr(csv_video, "decode_video_segment", decode)
    dataset = csv_video.CSVVideoDataset(
        str(index_path),
        17,
        (8, 8),
        train=False,
        positive_frame_strides=[1],
        positive_random_walk_count=0,
        positive_stride1_repeat_count=3,
        positive_decode_step_max=3,
    )

    sample = next(iter(dataset))
    positives = sample["positive_raw"]

    assert dataset.decode_length == 49
    assert decode_requests == [49]
    assert positives.shape == (4, 3, 17, 8, 8)
    assert torch.equal(sample["real"], positives[0])
    assert torch.equal(
        positives, positives[:1].expand_as(positives)
    )


def test_csv_video_rejects_decode_window_shorter_than_trajectory(tmp_path):
    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["video_path", "n_frames", "caption_l3"]
        )
        writer.writeheader()

    with pytest.raises(ValueError, match="must cover every positive trajectory"):
        csv_video.CSVVideoDataset(
            str(index_path),
            17,
            (8, 8),
            positive_frame_strides=[1],
            positive_random_walk_count=1,
            positive_random_step_max=3,
            positive_decode_step_max=2,
        )
