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

"""Tests for datamodule collation behavior."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.datamodule.nuscenes.segmentation3d import NuscenesSegmentation3DDataModule
from autoware_ml.datamodule.t4dataset.segmentation3d import T4Segmentation3DDataModule


def _make_seg_datamodule(**kwargs) -> T4Segmentation3DDataModule:
    return T4Segmentation3DDataModule(
        data_root=".",
        train_ann_file="train.pkl",
        val_ann_file="val.pkl",
        test_ann_file="test.pkl",
        **kwargs,
    )


def _seg_batch():
    return [
        {
            "coord": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "feat": torch.tensor([[0.0, 0.1], [1.0, 0.1]]),
            "grid_coord": torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32),
            "segment": torch.tensor([0, 1], dtype=torch.long),
        },
        {
            "coord": torch.tensor([[2.0, 0.0, 0.0]]),
            "feat": torch.tensor([[2.0, 0.2]]),
            "grid_coord": torch.tensor([[2, 0, 0]], dtype=torch.int32),
            "segment": torch.tensor([2], dtype=torch.long),
        },
    ]


class TestConcatCollation:
    def test_concatenates_variable_length_tensors(self):
        dm = _make_seg_datamodule(
            collation_map={
                "coord": "concat",
                "feat": "concat",
                "grid_coord": "concat",
                "segment": "concat",
            }
        )
        collated = dm.collate_fn(_seg_batch())

        assert collated["coord"].shape == (3, 3)
        assert collated["feat"].shape == (3, 2)
        assert collated["grid_coord"].shape == (3, 3)
        assert collated["segment"].shape == (3,)

    def test_auto_computes_cumulative_offset_from_first_concat_key(self):
        dm = _make_seg_datamodule(
            collation_map={
                "coord": "concat",
                "feat": "concat",
                "grid_coord": "concat",
                "segment": "concat",
            }
        )
        collated = dm.collate_fn(_seg_batch())

        assert torch.equal(collated["offset"], torch.tensor([2, 3]))

    def test_offset_reflects_actual_point_counts(self):
        batch = [
            {"coord": torch.zeros(5, 3), "feat": torch.zeros(5, 4)},
            {"coord": torch.zeros(3, 3), "feat": torch.zeros(3, 4)},
            {"coord": torch.zeros(7, 3), "feat": torch.zeros(7, 4)},
        ]
        dm = _make_seg_datamodule(collation_map={"coord": "concat", "feat": "concat"})
        collated = dm.collate_fn(batch)

        assert torch.equal(collated["offset"], torch.tensor([5, 8, 15]))

    def test_no_offset_without_concat_keys(self):
        batch = [{"img": torch.zeros(3, 4, 4)}, {"img": torch.zeros(3, 4, 4)}]
        dm = _make_seg_datamodule(collation_map={"img": "stack"})
        collated = dm.collate_fn(batch)

        assert "offset" not in collated


class TestStackCollation:
    def test_stacks_fixed_shape_tensors_along_new_batch_dim(self):
        batch = [
            {"target_grid": torch.tensor([[1, 2], [3, 4]], dtype=torch.long)},
            {"target_grid": torch.tensor([[5, 6], [7, 8]], dtype=torch.long)},
        ]
        dm = _make_seg_datamodule(collation_map={"target_grid": "stack"})
        collated = dm.collate_fn(batch)

        assert collated["target_grid"].shape == (2, 2, 2)
        assert torch.equal(collated["target_grid"][0], batch[0]["target_grid"])
        assert torch.equal(collated["target_grid"][1], batch[1]["target_grid"])

    def test_raises_on_shape_mismatch_for_stack_key(self):
        batch = [
            {"target_grid": torch.zeros(2, 2)},
            {"target_grid": torch.zeros(3, 3)},
        ]
        dm = _make_seg_datamodule(collation_map={"target_grid": "stack"})

        with pytest.raises(ValueError, match="target_grid"):
            dm.collate_fn(batch)

    def test_mixed_concat_and_stack_keys(self):
        batch = [
            {
                "coord": torch.zeros(2, 3),
                "feat": torch.zeros(2, 4),
                "target_grid": torch.zeros(4, 4),
            },
            {
                "coord": torch.zeros(3, 3),
                "feat": torch.zeros(3, 4),
                "target_grid": torch.zeros(4, 4),
            },
        ]
        dm = _make_seg_datamodule(
            collation_map={"coord": "concat", "feat": "concat", "target_grid": "stack"}
        )
        collated = dm.collate_fn(batch)

        assert collated["coord"].shape == (5, 3)
        assert collated["target_grid"].shape == (2, 4, 4)
        assert torch.equal(collated["offset"], torch.tensor([2, 5]))


class TestListCollation:
    def test_collects_variable_shape_tensors_with_explicit_list_strategy(self):
        batch = [
            {"points": torch.zeros(10, 4)},
            {"points": torch.zeros(7, 4)},
        ]
        dm = _make_seg_datamodule(collation_map={"points": "list"})
        collated = dm.collate_fn(batch)

        assert isinstance(collated["points"], list)
        assert len(collated["points"]) == 2

    def test_converts_numpy_arrays_to_tensors_with_list_strategy(self):
        import numpy as np

        batch = [
            {"points": np.zeros((10, 4), dtype=np.float32)},
            {"points": np.zeros((7, 4), dtype=np.float32)},
        ]
        dm = _make_seg_datamodule(collation_map={"points": "list"})
        collated = dm.collate_fn(batch)

        assert isinstance(collated["points"], list)
        assert all(isinstance(p, torch.Tensor) for p in collated["points"])
        assert collated["points"][0].shape == (10, 4)
        assert collated["points"][1].shape == (7, 4)

    def test_undeclared_keys_are_dropped(self):
        batch = [
            {"coord": torch.zeros(2, 3), "extra": torch.zeros(2, 3)},
            {"coord": torch.zeros(3, 3), "extra": torch.zeros(3, 3)},
        ]
        dm = _make_seg_datamodule(collation_map={"coord": "concat"})
        collated = dm.collate_fn(batch)

        assert "extra" not in collated
        assert collated["coord"].shape == (5, 3)


class TestInverseIndexAdjustment:
    def test_inverse_adjusted_to_global_voxel_indices(self):
        # Sample 0: 2 original points -> 2 voxels (inverse: [0, 1])
        # Sample 1: 3 original points -> 1 voxel  (inverse: [0, 0, 0])
        # After adjustment sample 1's inverse should be [2, 2, 2]
        batch = [
            {
                "coord": torch.zeros(2, 3),
                "inverse": torch.tensor([0, 1], dtype=torch.long),
            },
            {
                "coord": torch.zeros(1, 3),
                "inverse": torch.tensor([0, 0, 0], dtype=torch.long),
            },
        ]
        dm = _make_seg_datamodule(collation_map={"coord": "concat", "inverse": "index_concat"})
        collated = dm.collate_fn(batch)

        assert torch.equal(collated["inverse"], torch.tensor([0, 1, 2, 2, 2]))

    def test_inverse_not_adjusted_for_single_sample(self):
        batch = [
            {
                "coord": torch.zeros(3, 3),
                "inverse": torch.tensor([0, 1, 1], dtype=torch.long),
            }
        ]
        dm = _make_seg_datamodule(collation_map={"coord": "concat", "inverse": "index_concat"})
        collated = dm.collate_fn(batch)

        assert torch.equal(collated["inverse"], torch.tensor([0, 1, 1]))

    def test_inverse_three_samples_staggered_voxel_counts(self):
        # Sample 0: 3 voxels, 2 original points -> inverse [0, 2]
        # Sample 1: 1 voxel,  3 original points -> inverse [0, 0, 0] -> adjusted [3, 3, 3]
        # Sample 2: 2 voxels, 1 original point  -> inverse [1]       -> adjusted [5]
        batch = [
            {"coord": torch.zeros(3, 3), "inverse": torch.tensor([0, 2])},
            {"coord": torch.zeros(1, 3), "inverse": torch.tensor([0, 0, 0])},
            {"coord": torch.zeros(2, 3), "inverse": torch.tensor([1])},
        ]
        dm = _make_seg_datamodule(collation_map={"coord": "concat", "inverse": "index_concat"})
        collated = dm.collate_fn(batch)

        assert torch.equal(collated["inverse"], torch.tensor([0, 2, 3, 3, 3, 5]))


class TestDatamoduleCollation:
    def test_t4_and_nuscenes_datamodules_produce_offset(self):
        collation_map = {
            "coord": "concat",
            "feat": "concat",
            "grid_coord": "concat",
            "segment": "concat",
        }
        batch = _seg_batch()

        t4_dm = _make_seg_datamodule(collation_map=collation_map)
        nuscenes_dm = NuscenesSegmentation3DDataModule(
            data_root=".",
            train_ann_file="train.pkl",
            val_ann_file="val.pkl",
            test_ann_file="test.pkl",
            collation_map=collation_map,
        )

        t4_collated = t4_dm.collate_fn(batch)
        nuscenes_collated = nuscenes_dm.collate_fn(batch)

        assert torch.equal(t4_collated["offset"], torch.tensor([2, 3]))
        assert torch.equal(nuscenes_collated["offset"], torch.tensor([2, 3]))

    def test_raises_on_empty_batch(self):
        dm = _make_seg_datamodule()
        with pytest.raises(ValueError, match="empty"):
            dm.collate_fn([])

    def test_warns_and_skips_when_declared_key_missing_from_sample(
        self, caplog: pytest.LogCaptureFixture
    ):
        batch = [
            {"coord": torch.zeros(2, 3), "feat": torch.zeros(2, 4)},
            {"coord": torch.zeros(3, 3), "feat": torch.zeros(3, 4)},
        ]
        dm = _make_seg_datamodule(
            collation_map={"coord": "concat", "feat": "concat", "segment": "concat"}
        )

        collated = dm.collate_fn(batch)

        assert "segment" not in collated
        assert "Key 'segment' declared in collation_map but missing" in caplog.text

    def test_raises_when_index_concat_has_no_concat_key(self):
        batch = [
            {"inverse": torch.tensor([0, 1])},
            {"inverse": torch.tensor([0])},
        ]
        dm = _make_seg_datamodule(collation_map={"inverse": "index_concat"})
        with pytest.raises(ValueError, match="index_concat"):
            dm.collate_fn(batch)
