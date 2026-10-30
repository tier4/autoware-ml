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
    FocalHead2DOutputs,
    StreamPETRDenoisingOutputs,
    StreamPETRHeadOutputs,
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


def build_streampetr_denoising(
    num_layers: int = 2, batch_size: int = 1, num_slots: int = 4, num_known: int = 4
) -> StreamPETRDenoisingOutputs:
    """Build denoising outputs whose noised queries fill the given slots of sample 0."""
    return StreamPETRDenoisingOutputs(
        cls_scores=torch.zeros(num_layers, batch_size, num_slots, 3),
        bbox_preds=torch.zeros(num_layers, batch_size, num_slots, 10),
        known_labels=torch.zeros(num_known, dtype=torch.int64),
        known_boxes=torch.zeros(num_known, 9),
        batch_indices=torch.zeros(num_known, dtype=torch.int64),
        slot_indices=torch.arange(num_known, dtype=torch.int64),
    )


class TestStreamPETRHeadOutputs(unittest.TestCase):
    """Unit tests for the StreamPETR query head outputs."""

    def test_holds_the_decoder_outputs_and_the_denoising_queries(self) -> None:
        """
        Input: logits and boxes of two decoder layers, one sample and five queries, with
        denoising outputs.
        Expected: the outputs keep the predictions of every layer and the denoising queries as
        given.
        Check: read the box shape back and compare the denoising outputs by identity.
        """
        denoising = build_streampetr_denoising()
        outputs = StreamPETRHeadOutputs(
            all_cls_scores=torch.zeros(2, 1, 5, 3),
            all_bbox_preds=torch.zeros(2, 1, 5, 10),
            denoising=denoising,
        )

        self.assertEqual(tuple(outputs.all_bbox_preds.shape), (2, 1, 5, 10))
        self.assertIs(outputs.denoising, denoising)

    def test_layers_samples_and_queries_must_match(self) -> None:
        """
        Input: logits of five queries next to boxes of six.
        Expected: logits and boxes that describe different queries cannot be decoded together,
        so the outputs are rejected at construction.
        Check: a ValidationError is raised.
        """
        with self.assertRaises(ValidationError):
            StreamPETRHeadOutputs(
                all_cls_scores=torch.zeros(2, 1, 5, 3),
                all_bbox_preds=torch.zeros(2, 1, 6, 10),
                denoising=None,
            )

    def test_box_code_size_is_enforced(self) -> None:
        """
        Input: boxes of nine channels.
        Expected: the head regresses ten channels per box,
        ``[cx, cy, cz, log l, log w, log h, sin, cos, vx, vy]``, so any other layout is
        rejected.
        Check: a ValidationError is raised.
        """
        with self.assertRaises(ValidationError):
            StreamPETRHeadOutputs(
                all_cls_scores=torch.zeros(2, 1, 5, 3),
                all_bbox_preds=torch.zeros(2, 1, 5, 9),
                denoising=None,
            )

    def test_denoising_queries_must_share_the_layers_and_classes(self) -> None:
        """
        Input: detection queries of three decoder layers next to denoising queries of two.
        Expected: the denoising queries run through the same decoder as the detection queries,
        so outputs of another depth are rejected.
        Check: a ValidationError is raised.
        """
        with self.assertRaises(ValidationError):
            StreamPETRHeadOutputs(
                all_cls_scores=torch.zeros(3, 1, 5, 3),
                all_bbox_preds=torch.zeros(3, 1, 5, 10),
                denoising=build_streampetr_denoising(num_layers=2),
            )


class TestFocalHead2DOutputs(unittest.TestCase):
    """Unit tests for the dense outputs of the auxiliary 2D head."""

    def build(self, num_tokens_of_boxes: int = 6) -> FocalHead2DOutputs:
        return FocalHead2DOutputs(
            cls_scores=torch.zeros(4, 6, 3),
            bbox_preds=torch.zeros(4, num_tokens_of_boxes, 4),
            centers_2d=torch.zeros(4, 6, 2),
            centerness=torch.zeros(4, 6, 1),
            image_height=96,
            image_width=160,
        )

    def test_every_prediction_covers_the_same_tokens(self) -> None:
        """
        Input: 2D head outputs whose boxes cover 5 tokens of every image while the logits,
        centers and centerness cover 6.
        Expected: the predictions of a token are read at the same index of every tensor, so
        outputs covering other tokens are rejected when they are built.
        Check: a ValidationError is raised.
        """
        with self.assertRaises(ValidationError):
            self.build(num_tokens_of_boxes=5)


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
        """
        Input: a wrapper with no head outputs set.
        Expected: exactly one head family produces the outputs, so an empty wrapper is rejected,
        and the error names every head field, the StreamPETR one included.
        Check: a ValidationError listing the three head fields and the empty set is raised.
        """
        with self.assertRaisesRegex(
            ValidationError,
            "Exactly one of center_head_outputs, transfusion_head_outputs, "
            r"streampetr_head_outputs must be set, got \[\]",
        ):
            Detection3DHeadOutputs(center_head_outputs=None, transfusion_head_outputs=None)

    def test_rejects_both_head_outputs(self) -> None:
        """Test that a wrapper carrying outputs from both head families is rejected."""
        with self.assertRaises(ValidationError):
            Detection3DHeadOutputs(
                center_head_outputs=self.center_head_outputs,
                transfusion_head_outputs=self.transfusion_head_outputs,
            )

    def test_holds_streampetr_outputs_and_names_them_to_other_readers(self) -> None:
        """
        Input: a wrapper holding StreamPETR outputs, and one holding CenterHead outputs.
        Expected: the StreamPETR reader returns the outputs as given, and reading another head
        family names the field that is set and the one requested.
        Check: compare by identity, and match the ValueError of the TransFusion reader on the
        StreamPETR outputs and of the StreamPETR reader on the CenterHead outputs.
        """
        streampetr_head_outputs = StreamPETRHeadOutputs(
            all_cls_scores=torch.zeros(2, 1, 5, 3),
            all_bbox_preds=torch.zeros(2, 1, 5, 10),
            denoising=None,
        )
        outputs = Detection3DHeadOutputs(streampetr_head_outputs=streampetr_head_outputs)

        self.assertIs(outputs.streampetr_head(), streampetr_head_outputs)
        with self.assertRaisesRegex(
            ValueError, "in streampetr_head_outputs, not in transfusion_head_outputs"
        ):
            outputs.transfusion_head()
        with self.assertRaisesRegex(
            ValueError, "in center_head_outputs, not in streampetr_head_outputs"
        ):
            Detection3DHeadOutputs(center_head_outputs=self.center_head_outputs).streampetr_head()

    def test_rejects_streampetr_next_to_another_head_family(self) -> None:
        """
        Input: a wrapper holding CenterHead and StreamPETR outputs together.
        Expected: exactly one head family produces the outputs, so a second family is rejected.
        Check: a ValidationError is raised.
        """
        with self.assertRaises(ValidationError):
            Detection3DHeadOutputs(
                center_head_outputs=self.center_head_outputs,
                streampetr_head_outputs=StreamPETRHeadOutputs(
                    all_cls_scores=torch.zeros(2, 1, 5, 3),
                    all_bbox_preds=torch.zeros(2, 1, 5, 10),
                    denoising=None,
                ),
            )


if __name__ == "__main__":
    unittest.main()
