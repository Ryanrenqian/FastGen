# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Class-grouped ImageNet loader used by the paper TFD recipe."""

from __future__ import annotations

import numpy as np
import torch

from fastgen.datasets.class_cond_dataset import ImageFolderDataset
from fastgen.utils.distributed import get_rank


class TFDImageNetLoader:
    """Yield class groups with independently sampled positives and anchors.

    ``batch_size`` is the number of class conditions on each rank. The model
    generates ``generated_per_condition`` images for every condition.
    """

    def __init__(
        self,
        dataset_path: str,
        s3_path: str | None,
        batch_size: int,
        positives_per_condition: int = 4,
        anchors_per_condition: int = 4,
        seed: int = 10,
        use_labels: bool = True,
        **kwargs,
    ):
        kwargs.pop("shuffle", None)
        kwargs.pop("sampler_start_idx", None)
        kwargs.pop("cache", None)
        self.dataset = ImageFolderDataset(
            path=dataset_path, s3_path=s3_path, use_labels=use_labels, cache=False, **kwargs
        )
        self.batch_size = int(batch_size)
        self.positives_per_condition = int(positives_per_condition)
        self.anchors_per_condition = int(anchors_per_condition)
        self.seed = int(seed)
        if self.batch_size <= 0 or self.positives_per_condition <= 0 or self.anchors_per_condition <= 0:
            raise ValueError("TFD grouped sample counts must be positive")

        labels = self.dataset._get_raw_labels()
        if labels.ndim != 1:
            raise ValueError("TFD ImageNet loader requires integer class labels")
        self.num_classes = int(labels.max()) + 1
        self.class_to_indices = [np.flatnonzero(labels == label) for label in range(self.num_classes)]
        if any(len(indices) == 0 for indices in self.class_to_indices):
            raise ValueError("TFD ImageNet loader found an empty class")

    def _sample_images(self, rng: np.random.RandomState, labels: np.ndarray, count: int) -> torch.Tensor:
        groups = []
        for label in labels:
            indices = self.class_to_indices[int(label)]
            selected = rng.choice(indices, size=count, replace=len(indices) < count)
            images = [self.dataset._load_raw_image(int(index)).astype(np.float32) / 127.5 - 1.0 for index in selected]
            groups.append(torch.from_numpy(np.stack(images)))
        return torch.stack(groups)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + get_rank())
        while True:
            if self.batch_size <= self.num_classes:
                labels = rng.permutation(self.num_classes)[: self.batch_size]
            else:
                labels = rng.randint(0, self.num_classes, size=self.batch_size)
            positives = self._sample_images(rng, labels, self.positives_per_condition)
            anchors = self._sample_images(rng, labels, self.anchors_per_condition)
            condition = torch.nn.functional.one_hot(
                torch.from_numpy(labels).long(), num_classes=self.num_classes
            ).float()
            yield {
                "real": positives[:, 0],
                "condition": condition,
                "positive": positives,
                "anchor": anchors,
            }
