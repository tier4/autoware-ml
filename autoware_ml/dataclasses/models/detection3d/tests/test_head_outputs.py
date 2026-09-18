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

"""Unit tests for the 3D detection head output dataclasses."""

from __future__ import annotations

from types import MappingProxyType
import unittest

from pydantic import ValidationError
import torch

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    CenterHeadOutputs,
    Detection3DHeadOutputs,
    TransFusionHeadOutputs,
    TransFusionSeparateHeadOutputs,
)


class TestTransFusionSeparateHeadOutputs(unittest.TestCase):
    """Unit tests for the per-proposal TransFusion regression outputs."""

    def setUp(self) -> None:
        """Set up the proposal layout and a full set of output tensors."""
        self.batch_size, self.num_classes, self.num_proposals = 2, 3, 4
        self.tensors = {
            "heatmaps": torch.zeros(self.batch_size, self.num_classes, self.num_proposals),
            "centers": torch.zeros(self.batch_size, 2, self.num_proposals),
            "heights": torch.zeros(self.batch_size, 1, self.num_proposals),
            "dims": torch.zeros(self.batch_size, 3, self.num_proposals),
            "rots": torch.zeros(self.batch_size, 2, self.num_proposals),
            "vels": torch.zeros(self.batch_size, 2, self.num_proposals),
        }

    def test_from_dict_with_velocity(self) -> None:
        """Test that every head tensor is picked up by name and the keys keep the runtime order."""
        outputs = TransFusionSeparateHeadOutputs.from_dict(MappingProxyType(self.tensors))

        for name, tensor in self.tensors.items():
            self.assertIs(getattr(outputs, name), tensor)
        self.assertEqual(
            list(outputs.ordered_keys), ["heatmaps", "centers", "heights", "dims", "rots", "vels"]
        )

    def test_from_dict_without_velocity(self) -> None:
        """Test that a head without a velocity branch leaves ``vels`` absent and out of the keys."""
        tensors = {name: tensor for name, tensor in self.tensors.items() if name != "vels"}

        outputs = TransFusionSeparateHeadOutputs.from_dict(MappingProxyType(tensors))

        self.assertIsNone(outputs.vels)
        self.assertEqual(
            list(outputs.ordered_keys), ["heatmaps", "centers", "heights", "dims", "rots"]
        )

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Test that extra entries in the head's output mapping are not carried over."""
        tensors = {**self.tensors, "query_pos": torch.zeros(self.batch_size, 2, self.num_proposals)}

        outputs = TransFusionSeparateHeadOutputs.from_dict(MappingProxyType(tensors))

        self.assertFalse(hasattr(outputs, "query_pos"))

    def test_from_dict_requires_every_regression_branch(self) -> None:
        """Test that a mapping missing a mandatory branch is rejected."""
        tensors = {name: tensor for name, tensor in self.tensors.items() if name != "rots"}

        with self.assertRaises(KeyError):
            TransFusionSeparateHeadOutputs.from_dict(MappingProxyType(tensors))

    def test_channel_counts_are_enforced(self) -> None:
        """Test that each regression branch must carry its fixed number of channels."""
        for name, wrong_channels in (("centers", 3), ("heights", 2), ("dims", 2), ("rots", 1)):
            with self.subTest(branch=name):
                tensors = {
                    **self.tensors,
                    name: torch.zeros(self.batch_size, wrong_channels, self.num_proposals),
                }
                with self.assertRaises(ValidationError):
                    TransFusionSeparateHeadOutputs(**tensors)


class TestTransFusionHeadOutputs(unittest.TestCase):
    """Unit tests for the full TransFusion head output."""

    def setUp(self) -> None:
        """Set up a small BEV grid, proposal layout and separate head outputs."""
        self.batch_size, self.num_classes, self.num_proposals = 1, 3, 4
        self.height, self.width = 8, 8
        self.separate_head_outputs = TransFusionSeparateHeadOutputs(
            heatmaps=torch.zeros(self.batch_size, self.num_classes, self.num_proposals),
            centers=torch.zeros(self.batch_size, 2, self.num_proposals),
            heights=torch.zeros(self.batch_size, 1, self.num_proposals),
            dims=torch.zeros(self.batch_size, 3, self.num_proposals),
            rots=torch.zeros(self.batch_size, 2, self.num_proposals),
            vels=None,
        )
        self.dense_heatmaps = torch.zeros(
            self.batch_size, self.num_classes, self.height, self.width
        )
        self.query_heatmap_scores = torch.zeros(
            self.batch_size, self.num_classes, self.num_proposals
        )
        self.query_labels = torch.zeros(self.batch_size, self.num_proposals, dtype=torch.int64)

    def test_nests_the_separate_head_outputs(self) -> None:
        """Test that dense heatmap, query scores, labels and per-proposal outputs are carried."""
        outputs = TransFusionHeadOutputs(
            dense_heatmaps=self.dense_heatmaps,
            query_heatmap_scores=self.query_heatmap_scores,
            query_labels=self.query_labels,
            separate_head_outputs=self.separate_head_outputs,
        )

        self.assertIs(outputs.separate_head_outputs, self.separate_head_outputs)
        self.assertEqual(
            tuple(outputs.dense_heatmaps.shape),
            (self.batch_size, self.num_classes, self.height, self.width),
        )

    def test_query_labels_must_be_int64(self) -> None:
        """Test that query labels in another integer dtype are rejected."""
        with self.assertRaises(ValidationError):
            TransFusionHeadOutputs(
                dense_heatmaps=self.dense_heatmaps,
                query_heatmap_scores=self.query_heatmap_scores,
                query_labels=self.query_labels.to(torch.int32),
                separate_head_outputs=self.separate_head_outputs,
            )

    def test_separate_head_outputs_must_be_the_dataclass(self) -> None:
        """Test that a plain mapping is not coerced into ``TransFusionSeparateHeadOutputs``."""
        with self.assertRaises(ValidationError):
            TransFusionHeadOutputs(
                dense_heatmaps=self.dense_heatmaps,
                query_heatmap_scores=self.query_heatmap_scores,
                query_labels=self.query_labels,
                separate_head_outputs={  # type: ignore[arg-type]
                    "heatmaps": self.separate_head_outputs.heatmaps
                },
            )


class TestCenterHeadOutputs(unittest.TestCase):
    """Unit tests for the dense CenterHead output."""

    def setUp(self) -> None:
        """Set up a full set of dense head tensors on a 4x4 grid."""
        self.tensors = {
            "heatmaps": torch.zeros(2, 3, 4, 4),
            "centers": torch.zeros(2, 2, 4, 4),
            "heights": torch.zeros(2, 1, 4, 4),
            "dims": torch.zeros(2, 3, 4, 4),
            "rots": torch.zeros(2, 2, 4, 4),
            "vels": torch.zeros(2, 2, 4, 4),
        }

    def test_velocity_is_optional(self) -> None:
        """Test that the head can be built with or without a velocity map."""
        with_velocity = CenterHeadOutputs(**self.tensors)
        without_velocity = CenterHeadOutputs(**{**self.tensors, "vels": None})

        assert with_velocity.vels is not None
        self.assertEqual(tuple(with_velocity.vels.shape), (2, 2, 4, 4))
        self.assertIsNone(without_velocity.vels)

    def test_channel_counts_are_enforced(self) -> None:
        """Test that each dense branch must carry its fixed number of channels."""
        for name, wrong_channels in (
            ("centers", 3),
            ("heights", 2),
            ("dims", 2),
            ("rots", 1),
            ("vels", 3),
        ):
            with self.subTest(branch=name):
                with self.assertRaises(ValidationError):
                    CenterHeadOutputs(
                        **{**self.tensors, name: torch.zeros(2, wrong_channels, 4, 4)}
                    )

    def test_rank_and_dtype_are_enforced(self) -> None:
        """Test that a 3-D or non-float32 heatmap is rejected."""
        with self.assertRaises(ValidationError):
            CenterHeadOutputs(**{**self.tensors, "heatmaps": torch.zeros(3, 4, 4)})
        with self.assertRaises(ValidationError):
            CenterHeadOutputs(
                **{**self.tensors, "heatmaps": torch.zeros(2, 3, 4, 4, dtype=torch.float64)}
            )

    def test_is_frozen(self) -> None:
        """Test that the outputs cannot be mutated after construction."""
        outputs = CenterHeadOutputs(**self.tensors)

        with self.assertRaises(ValidationError):
            outputs.vels = None  # type: ignore[misc]


class TestDetection3DHeadOutputs(unittest.TestCase):
    """Unit tests for the head-agnostic 3D detection output wrapper."""

    def test_holds_one_head_family_at_a_time(self) -> None:
        """Test that the wrapper exposes the head that produced the outputs and leaves the other."""
        center_head_outputs = CenterHeadOutputs(
            heatmaps=torch.zeros(1, 2, 4, 4),
            centers=torch.zeros(1, 2, 4, 4),
            heights=torch.zeros(1, 1, 4, 4),
            dims=torch.zeros(1, 3, 4, 4),
            rots=torch.zeros(1, 2, 4, 4),
            vels=None,
        )

        outputs = Detection3DHeadOutputs(
            center_head_outputs=center_head_outputs, transfusion_head_outputs=None
        )

        self.assertIs(outputs.center_head_outputs, center_head_outputs)
        self.assertIsNone(outputs.transfusion_head_outputs)

    def test_allows_no_head_outputs(self) -> None:
        """Test that an empty wrapper is representable, for consumers to reject explicitly."""
        outputs = Detection3DHeadOutputs(center_head_outputs=None, transfusion_head_outputs=None)

        self.assertIsNone(outputs.center_head_outputs)
        self.assertIsNone(outputs.transfusion_head_outputs)


if __name__ == "__main__":
    unittest.main()
