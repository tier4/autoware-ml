# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for point-cloud dataloader helpers."""

from __future__ import annotations

import torch

from autoware_ml.datamodule.common.point_cloud import point_collate_fn
from autoware_ml.datamodule.nuscenes.segmentation3d import NuscenesSegmentation3DDataModule
from autoware_ml.datamodule.t4dataset.segmentation3d import T4Segmentation3DDataModule


def test_point_collate_fn_concatenates_and_accumulates_offsets() -> None:
    batch = [
        {
            "coord": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "feat": torch.tensor([[0.0, 0.1], [1.0, 0.1]]),
            "grid_coord": torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32),
            "segment": torch.tensor([0, 1], dtype=torch.long),
            "offset": torch.tensor([2], dtype=torch.int32),
        },
        {
            "coord": torch.tensor([[2.0, 0.0, 0.0]]),
            "feat": torch.tensor([[2.0, 0.2]]),
            "grid_coord": torch.tensor([[2, 0, 0]], dtype=torch.int32),
            "segment": torch.tensor([2], dtype=torch.long),
            "offset": torch.tensor([1], dtype=torch.int32),
        },
    ]

    collated = point_collate_fn(batch)

    assert collated["coord"].shape == (3, 3)
    assert collated["feat"].shape == (3, 2)
    assert torch.equal(collated["offset"], torch.tensor([2, 3], dtype=torch.int32))
    assert torch.equal(collated["segment"], torch.tensor([0, 1, 2], dtype=torch.long))


def test_segmentation_collate_fn_allows_missing_trainer_for_predict_paths() -> None:
    batch = [
        {
            "coord": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "feat": torch.tensor([[0.0, 0.1], [1.0, 0.1]]),
            "grid_coord": torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32),
            "segment": torch.tensor([0, 1], dtype=torch.long),
            "offset": torch.tensor([2], dtype=torch.int32),
        }
    ]

    nuscenes_datamodule = NuscenesSegmentation3DDataModule(
        data_root=".",
        train_ann_file="train.pkl",
        val_ann_file="val.pkl",
        test_ann_file="test.pkl",
        mix_prob=1.0,
    )
    t4_datamodule = T4Segmentation3DDataModule(
        data_root=".",
        train_ann_file="train.pkl",
        val_ann_file="val.pkl",
        test_ann_file="test.pkl",
        mix_prob=1.0,
    )

    nuscenes_collated = nuscenes_datamodule.collate_fn(batch)
    t4_collated = t4_datamodule.collate_fn(batch)

    assert torch.equal(nuscenes_collated["offset"], torch.tensor([2], dtype=torch.int32))
    assert torch.equal(t4_collated["offset"], torch.tensor([2], dtype=torch.int32))
