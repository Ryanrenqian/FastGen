# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Class-grouped ImageNet loader used by the paper TFD recipe."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
import pickle

from fastgen.datasets.class_cond_dataset import ImageFolderDataset
from fastgen.utils.distributed import get_rank, synchronize


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


class TFDImageNetLMDBLoader:
    """Official TFD/DMD2 ImageNet-64 LMDB class-group sampler."""

    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        positives_per_condition: int = 4,
        anchors_per_condition: int = 4,
        seed: int = 10,
        num_classes: int = 1000,
        expected_num_images: int | None = None,
        class_index_cache: str | None = None,
        **kwargs,
    ):
        del kwargs
        try:
            import lmdb
        except ImportError as error:
            raise ImportError("TFDImageNetLMDBLoader requires the 'lmdb' package") from error

        self.dataset_path = str(dataset_path)
        self.env = lmdb.open(
            self.dataset_path, readonly=True, lock=False, readahead=False, meminit=False
        )
        self.image_shape = self._array_shape("images")
        self.label_shape = self._array_shape("labels")
        if len(self.image_shape) != 4 or self.image_shape[1:] != (3, 64, 64):
            raise ValueError(f"Expected LMDB images with shape [N,3,64,64], got {self.image_shape}")
        if self.label_shape[0] != self.image_shape[0]:
            raise ValueError("LMDB image and label counts differ")
        if expected_num_images is not None and self.image_shape[0] != int(expected_num_images):
            raise ValueError(
                f"Expected {expected_num_images} LMDB images, found {self.image_shape[0]}"
            )

        self.batch_size = int(batch_size)
        self.positives_per_condition = int(positives_per_condition)
        self.anchors_per_condition = int(anchors_per_condition)
        self.seed = int(seed)
        self.num_classes = int(num_classes)
        if self.batch_size <= 0 or self.positives_per_condition <= 0 or self.anchors_per_condition <= 0:
            raise ValueError("TFD grouped sample counts must be positive")

        cache = Path(class_index_cache or Path(self.dataset_path) / f"class_index_{self.num_classes}.pkl")
        if get_rank() == 0 and not cache.is_file():
            class_to_indices = self._build_class_index()
            temporary_cache = cache.with_suffix(cache.suffix + ".tmp")
            with temporary_cache.open("wb") as file:
                pickle.dump(class_to_indices, file)
            temporary_cache.replace(cache)
        synchronize()
        if not cache.is_file():
            raise FileNotFoundError(f"LMDB class-index cache was not created: {cache}")
        with cache.open("rb") as file:
            self.class_to_indices = pickle.load(file)
        if len(self.class_to_indices) != self.num_classes or any(
            len(indices) == 0 for indices in self.class_to_indices
        ):
            raise ValueError("TFD ImageNet LMDB class index is incomplete")

    def _array_shape(self, name: str) -> tuple[int, ...]:
        with self.env.begin() as txn:
            value = txn.get(f"{name}_shape".encode())
        if value is None:
            raise KeyError(f"Missing LMDB shape key: {name}_shape")
        return tuple(map(int, value.decode().split()))

    def _build_class_index(self) -> list[np.ndarray]:
        class_to_indices = [[] for _ in range(self.num_classes)]
        label_row_shape = self.label_shape[1:]
        with self.env.begin() as txn:
            for index in range(self.label_shape[0]):
                value = txn.get(f"labels_{index}_data".encode())
                if value is None:
                    raise KeyError(f"Missing LMDB label row {index}")
                label = int(np.frombuffer(value, dtype=np.int64).reshape(label_row_shape or (-1,))[0])
                if label < 0 or label >= self.num_classes:
                    raise ValueError(f"Invalid ImageNet label {label} at row {index}")
                class_to_indices[label].append(index)
        return [np.asarray(indices, dtype=np.int64) for indices in class_to_indices]

    def _sample_images(self, rng: np.random.RandomState, labels: np.ndarray, count: int) -> torch.Tensor:
        groups = []
        with self.env.begin() as txn:
            for label in labels:
                indices = self.class_to_indices[int(label)]
                selected = rng.choice(indices, size=count, replace=len(indices) < count)
                images = []
                for index in selected:
                    value = txn.get(f"images_{int(index)}_data".encode())
                    if value is None:
                        raise KeyError(f"Missing LMDB image row {index}")
                    image = np.frombuffer(value, dtype=np.uint8).reshape(self.image_shape[1:])
                    images.append(image.astype(np.float32) / 127.5 - 1.0)
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
