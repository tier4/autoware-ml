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
        """Set up the proposal layout."""
        self.batch_size, self.num_classes, self.num_proposals = 2, 3, 4

    def _build_tensors(self) -> dict[str, torch.Tensor]:
        """Build a fresh, complete set of per-proposal output tensors, velocity included."""
        return {
            "heatmap": torch.zeros(self.batch_size, self.num_classes, self.num_proposals),
            "center": torch.zeros(self.batch_size, 2, self.num_proposals),
            "height": torch.zeros(self.batch_size, 1, self.num_proposals),
            "dim": torch.zeros(self.batch_size, 3, self.num_proposals),
            "rot": torch.zeros(self.batch_size, 2, self.num_proposals),
            "vel": torch.zeros(self.batch_size, 2, self.num_proposals),
        }

    def test_export_tensors_follow_the_requested_order(self) -> None:
        """Test that the head outputs come out in the order the export names them."""
        tensors = self._build_tensors()
        separate = TransFusionSeparateHeadOutputs.model_validate(tensors)
        outputs = TransFusionHeadOutputs(
            dense_heatmap=torch.zeros(self.batch_size, self.num_classes, 8, 8),
            query_heatmap_score=torch.zeros(self.batch_size, self.num_classes, self.num_proposals),
            query_labels=torch.zeros(self.batch_size, self.num_proposals, dtype=torch.int64),
            separate_head_outputs=separate,
        )

        exported = outputs.export_tensors(["query_labels", "vel", "heatmap", "dense_heatmap"])

        self.assertIs(exported[0], outputs.query_labels)
        self.assertIs(exported[1], tensors["vel"])
        self.assertIs(exported[2], tensors["heatmap"])
        self.assertIs(exported[3], outputs.dense_heatmap)

    def test_export_tensors_reject_an_unset_or_unknown_output(self) -> None:
        """Test that an export naming a missing velocity or an unknown tensor is rejected."""
        tensors = self._build_tensors()
        tensors["vel"] = None
        outputs = TransFusionHeadOutputs(
            dense_heatmap=torch.zeros(self.batch_size, self.num_classes, 8, 8),
            query_heatmap_score=torch.zeros(self.batch_size, self.num_classes, self.num_proposals),
            query_labels=torch.zeros(self.batch_size, self.num_proposals, dtype=torch.int64),
            separate_head_outputs=TransFusionSeparateHeadOutputs.model_validate(tensors),
        )

        for name in ("vel", "query_pos"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                outputs.export_tensors(["heatmap", name])

    def test_every_regression_branch_is_required(self) -> None:
        """Test that outputs missing a mandatory branch are rejected."""
        tensors = self._build_tensors()
        del tensors["rot"]

        with self.assertRaises(ValidationError):
            TransFusionSeparateHeadOutputs.model_validate(tensors)

    def test_channel_counts_are_enforced(self) -> None:
        """Test that each regression branch must carry its fixed number of channels."""
        for name, wrong_channels in (("center", 3), ("height", 2), ("dim", 2), ("rot", 1)):
            with self.subTest(branch=name):
                tensors = self._build_tensors()
                tensors[name] = torch.zeros(self.batch_size, wrong_channels, self.num_proposals)
                with self.assertRaises(ValidationError):
                    TransFusionSeparateHeadOutputs.model_validate(tensors)

    def test_batch_and_proposal_counts_must_match(self) -> None:
        """Test that a branch covering another batch or proposal count is rejected."""
        for name, shape in (
            ("heatmap", (self.batch_size + 1, self.num_classes, self.num_proposals)),
            ("dim", (self.batch_size, 3, self.num_proposals + 1)),
            ("vel", (self.batch_size, 2, self.num_proposals + 1)),
        ):
            with self.subTest(branch=name):
                tensors = self._build_tensors()
                tensors[name] = torch.zeros(shape)
                with self.assertRaises(ValidationError):
                    TransFusionSeparateHeadOutputs.model_validate(tensors)


class TestTransFusionHeadOutputs(unittest.TestCase):
    """Unit tests for the full TransFusion head output."""

    def setUp(self) -> None:
        """Set up a small BEV grid, proposal layout and separate head outputs."""
        self.batch_size, self.num_classes, self.num_proposals = 1, 3, 4
        self.height, self.width = 8, 8
        self.separate_head_outputs = TransFusionSeparateHeadOutputs(
            heatmap=torch.zeros(self.batch_size, self.num_classes, self.num_proposals),
            center=torch.zeros(self.batch_size, 2, self.num_proposals),
            height=torch.zeros(self.batch_size, 1, self.num_proposals),
            dim=torch.zeros(self.batch_size, 3, self.num_proposals),
            rot=torch.zeros(self.batch_size, 2, self.num_proposals),
            vel=None,
        )
        self.dense_heatmap = torch.zeros(self.batch_size, self.num_classes, self.height, self.width)
        self.query_heatmap_score = torch.zeros(
            self.batch_size, self.num_classes, self.num_proposals
        )
        self.query_labels = torch.zeros(self.batch_size, self.num_proposals, dtype=torch.int64)

    def test_nests_the_separate_head_outputs(self) -> None:
        """Test that dense heatmap, query scores, labels and per-proposal outputs are carried."""
        outputs = TransFusionHeadOutputs(
            dense_heatmap=self.dense_heatmap,
            query_heatmap_score=self.query_heatmap_score,
            query_labels=self.query_labels,
            separate_head_outputs=self.separate_head_outputs,
        )

        self.assertIs(outputs.separate_head_outputs, self.separate_head_outputs)
        self.assertEqual(
            tuple(outputs.dense_heatmap.shape),
            (self.batch_size, self.num_classes, self.height, self.width),
        )

    def test_batch_class_and_query_counts_must_match(self) -> None:
        """Test that the dense, query and per-proposal outputs must agree on shared dims."""
        for name, bad_value in (
            (
                "dense_heatmap",
                torch.zeros(self.batch_size + 1, self.num_classes, self.height, self.width),
            ),
            (
                "query_heatmap_score",
                torch.zeros(self.batch_size, self.num_classes + 1, self.num_proposals),
            ),
            (
                "query_labels",
                torch.zeros(self.batch_size, self.num_proposals + 1, dtype=torch.int64),
            ),
            (
                "separate_head_outputs",
                TransFusionSeparateHeadOutputs(
                    heatmap=torch.zeros(self.batch_size, self.num_classes + 1, self.num_proposals),
                    center=torch.zeros(self.batch_size, 2, self.num_proposals),
                    height=torch.zeros(self.batch_size, 1, self.num_proposals),
                    dim=torch.zeros(self.batch_size, 3, self.num_proposals),
                    rot=torch.zeros(self.batch_size, 2, self.num_proposals),
                    vel=None,
                ),
            ),
        ):
            with self.subTest(field=name):
                fields = {
                    "dense_heatmap": self.dense_heatmap,
                    "query_heatmap_score": self.query_heatmap_score,
                    "query_labels": self.query_labels,
                    "separate_head_outputs": self.separate_head_outputs,
                }
                fields[name] = bad_value
                with self.assertRaises(ValidationError):
                    TransFusionHeadOutputs.model_validate(fields)

    def test_auxiliary_layers_may_widen_the_separate_head(self) -> None:
        """Test that the separate head may carry more proposals than there are queries."""
        num_layers = 3
        separate_head_outputs = TransFusionSeparateHeadOutputs(
            heatmap=torch.zeros(self.batch_size, self.num_classes, num_layers * self.num_proposals),
            center=torch.zeros(self.batch_size, 2, num_layers * self.num_proposals),
            height=torch.zeros(self.batch_size, 1, num_layers * self.num_proposals),
            dim=torch.zeros(self.batch_size, 3, num_layers * self.num_proposals),
            rot=torch.zeros(self.batch_size, 2, num_layers * self.num_proposals),
            vel=None,
        )

        outputs = TransFusionHeadOutputs(
            dense_heatmap=self.dense_heatmap,
            query_heatmap_score=self.query_heatmap_score,
            query_labels=self.query_labels,
            separate_head_outputs=separate_head_outputs,
        )

        self.assertEqual(
            outputs.separate_head_outputs.heatmap.shape[-1], num_layers * self.num_proposals
        )

    def test_query_labels_must_be_int64(self) -> None:
        """Test that query labels in another integer dtype are rejected."""
        with self.assertRaises(ValidationError):
            TransFusionHeadOutputs(
                dense_heatmap=self.dense_heatmap,
                query_heatmap_score=self.query_heatmap_score,
                query_labels=self.query_labels.to(torch.int32),
                separate_head_outputs=self.separate_head_outputs,
            )

    def test_separate_head_outputs_must_be_the_dataclass(self) -> None:
        """Test that a plain mapping is not coerced into ``TransFusionSeparateHeadOutputs``."""
        with self.assertRaises(ValidationError):
            TransFusionHeadOutputs(
                dense_heatmap=self.dense_heatmap,
                query_heatmap_score=self.query_heatmap_score,
                query_labels=self.query_labels,
                separate_head_outputs={  # type: ignore[arg-type]
                    "heatmap": self.separate_head_outputs.heatmap
                },
            )


class TestCenterHeadOutputs(unittest.TestCase):
    """Unit tests for the dense CenterHead output."""

    def _build_tensors(self) -> dict[str, torch.Tensor | None]:
        """Build a fresh, complete set of dense head tensors on a 4x4 grid, velocity included."""
        return {
            "heatmap": torch.zeros(2, 3, 4, 4),
            "reg": torch.zeros(2, 2, 4, 4),
            "height": torch.zeros(2, 1, 4, 4),
            "dim": torch.zeros(2, 3, 4, 4),
            "rot": torch.zeros(2, 2, 4, 4),
            "vel": torch.zeros(2, 2, 4, 4),
        }

    def test_velocity_is_optional(self) -> None:
        """Test that the head can be built with or without a velocity map."""
        with_velocity = CenterHeadOutputs.model_validate(self._build_tensors())
        tensors = self._build_tensors()
        tensors["vel"] = None
        without_velocity = CenterHeadOutputs.model_validate(tensors)

        assert with_velocity.vel is not None
        self.assertEqual(tuple(with_velocity.vel.shape), (2, 2, 4, 4))
        self.assertIsNone(without_velocity.vel)

    def test_channel_counts_are_enforced(self) -> None:
        """Test that each dense branch must carry its fixed number of channels."""
        for name, wrong_channels in (
            ("reg", 3),
            ("height", 2),
            ("dim", 2),
            ("rot", 1),
            ("vel", 3),
        ):
            with self.subTest(branch=name):
                tensors = self._build_tensors()
                tensors[name] = torch.zeros(2, wrong_channels, 4, 4)
                with self.assertRaises(ValidationError):
                    CenterHeadOutputs.model_validate(tensors)

    def test_batch_and_grid_must_match(self) -> None:
        """Test that a branch on another batch or BEV grid is rejected."""
        for name, shape in (
            ("heatmap", (3, 3, 4, 4)),
            ("rot", (2, 2, 5, 4)),
            ("vel", (2, 2, 4, 5)),
        ):
            with self.subTest(branch=name):
                tensors = self._build_tensors()
                tensors[name] = torch.zeros(shape)
                with self.assertRaises(ValidationError):
                    CenterHeadOutputs.model_validate(tensors)

    def test_rank_and_dtype_are_enforced(self) -> None:
        """Test that a 3-D or integer heatmap is rejected."""
        wrong_rank = self._build_tensors()
        wrong_rank["heatmap"] = torch.zeros(3, 4, 4)
        wrong_dtype = self._build_tensors()
        wrong_dtype["heatmap"] = torch.zeros(2, 3, 4, 4, dtype=torch.int64)

        with self.assertRaises(ValidationError):
            CenterHeadOutputs.model_validate(wrong_rank)
        with self.assertRaises(ValidationError):
            CenterHeadOutputs.model_validate(wrong_dtype)

    def test_export_tensors_follow_the_requested_order(self) -> None:
        """Test that the dense outputs come out in the order the export names them."""
        tensors = self._build_tensors()
        outputs = CenterHeadOutputs.model_validate(tensors)

        exported = outputs.export_tensors(["heatmap", "reg", "height", "dim", "rot", "vel"])

        for name, tensor in zip(("heatmap", "reg", "height", "dim", "rot", "vel"), exported):
            self.assertIs(tensor, tensors[name])

    def test_is_frozen(self) -> None:
        """Test that the outputs cannot be mutated after construction."""
        outputs = CenterHeadOutputs.model_validate(self._build_tensors())

        with self.assertRaises(ValidationError):
            outputs.vel = None  # type: ignore[misc]


class TestDetection3DHeadOutputs(unittest.TestCase):
    """Unit tests for the head-agnostic 3D detection output wrapper."""

    def setUp(self) -> None:
        """Set up one valid output per head family."""
        self.center_head_outputs = CenterHeadOutputs(
            heatmap=torch.zeros(1, 2, 4, 4),
            reg=torch.zeros(1, 2, 4, 4),
            height=torch.zeros(1, 1, 4, 4),
            dim=torch.zeros(1, 3, 4, 4),
            rot=torch.zeros(1, 2, 4, 4),
            vel=None,
        )
        self.transfusion_head_outputs = TransFusionHeadOutputs(
            dense_heatmap=torch.zeros(1, 2, 4, 4),
            query_heatmap_score=torch.zeros(1, 2, 3),
            query_labels=torch.zeros(1, 3, dtype=torch.int64),
            separate_head_outputs=TransFusionSeparateHeadOutputs(
                heatmap=torch.zeros(1, 2, 3),
                center=torch.zeros(1, 2, 3),
                height=torch.zeros(1, 1, 3),
                dim=torch.zeros(1, 3, 3),
                rot=torch.zeros(1, 2, 3),
                vel=None,
            ),
        )

    def test_holds_one_head_family_at_a_time(self) -> None:
        """Test that the wrapper exposes the head that produced the outputs and leaves the other."""
        center_outputs = Detection3DHeadOutputs(
            center_head_outputs=self.center_head_outputs, transfusion_head_outputs=None
        )
        transfusion_outputs = Detection3DHeadOutputs(
            center_head_outputs=None, transfusion_head_outputs=self.transfusion_head_outputs
        )

        self.assertIs(center_outputs.center_head_outputs, self.center_head_outputs)
        self.assertIsNone(center_outputs.transfusion_head_outputs)
        self.assertIs(transfusion_outputs.transfusion_head_outputs, self.transfusion_head_outputs)
        self.assertIsNone(transfusion_outputs.center_head_outputs)

    def test_rejects_no_head_outputs(self) -> None:
        """Test that a wrapper carrying no head outputs is rejected."""
        with self.assertRaises(ValidationError):
            Detection3DHeadOutputs(center_head_outputs=None, transfusion_head_outputs=None)

    def test_rejects_both_head_outputs(self) -> None:
        """Test that a wrapper carrying outputs from both head families is rejected."""
        with self.assertRaises(ValidationError):
            Detection3DHeadOutputs(
                center_head_outputs=self.center_head_outputs,
                transfusion_head_outputs=self.transfusion_head_outputs,
            )


if __name__ == "__main__":
    unittest.main()
