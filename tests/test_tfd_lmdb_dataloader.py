# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from fastgen.datasets.tfd_class_cond_dataloader import TFDImageNetLMDBLoader


def test_tfd_lmdb_loader_groups_classes_and_normalizes(tmp_path):
    lmdb = pytest.importorskip("lmdb")
    dataset_path = tmp_path / "tiny_lmdb"
    env = lmdb.open(str(dataset_path), map_size=16 * 1024 * 1024)
    images = np.stack(
        [
            np.full((3, 64, 64), 0, dtype=np.uint8),
            np.full((3, 64, 64), 255, dtype=np.uint8),
            np.full((3, 64, 64), 64, dtype=np.uint8),
            np.full((3, 64, 64), 192, dtype=np.uint8),
        ]
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    with env.begin(write=True) as txn:
        txn.put(b"images_shape", b"4 3 64 64")
        txn.put(b"labels_shape", b"4")
        for index, (image, label) in enumerate(zip(images, labels)):
            txn.put(f"images_{index}_data".encode(), image.tobytes())
            txn.put(f"labels_{index}_data".encode(), label.tobytes())
    env.close()

    loader = TFDImageNetLMDBLoader(
        dataset_path=str(dataset_path),
        batch_size=2,
        positives_per_condition=2,
        anchors_per_condition=2,
        seed=10,
        num_classes=2,
        expected_num_images=4,
    )
    batch = next(iter(loader))

    assert batch["positive"].shape == (2, 2, 3, 64, 64)
    assert batch["anchor"].shape == (2, 2, 3, 64, 64)
    assert batch["condition"].shape == (2, 2)
    assert torch.allclose(batch["condition"].sum(dim=1), torch.ones(2))
    assert batch["positive"].min() >= -1.0
    assert batch["positive"].max() <= 1.0
    assert [len(indices) for indices in loader.class_to_indices] == [2, 2]
