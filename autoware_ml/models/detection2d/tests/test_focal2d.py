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

"""Unit tests for the auxiliary 2D head of the camera 3D detectors."""

from __future__ import annotations

import math
import unittest

from jaxtyping import Bool, Float, Float32, Int64
import torch

from autoware_ml.dataclasses.batch.detection2d import Detection2DGTBatch
from autoware_ml.dataclasses.models.detection2d.head_outputs import FocalHead2DOutputs
from autoware_ml.models.detection2d.heads.focal2d import FocalHead2D
from autoware_ml.models.detection3d.task_modules.heatmap import draw_heatmap_gaussian
from autoware_ml.models.detection3d.task_modules.partial_annotations import PartialAnnotations

CLASS_NAMES = ["car", "truck", "bus", "bicycle", "pedestrian", "traffic_cone", "barrier"]
PARTIAL_ANNOTATIONS = PartialAnnotations(CLASS_NAMES, ["traffic_cone", "barrier"])
LOSS_KEYS = ("loss_cls2d", "loss_bbox2d", "loss_iou2d", "loss_centers2d", "loss_heatmap2d")
IMAGE_HEIGHT, IMAGE_WIDTH = 96, 160
# The 6 x 10 token grid of the images at stride 16.
FEATURE_HEIGHT, FEATURE_WIDTH = 6, 10


class TestFocalHead2D(unittest.TestCase):
    """Unit tests for the forward pass and the five losses of ``FocalHead2D``."""

    def setUp(self) -> None:
        """Build a small head of 32 hidden channels at stride 16, seeded."""
        torch.manual_seed(0)
        self.head = FocalHead2D(class_names=CLASS_NAMES, in_channels=32, embed_dims=32, stride=16)

    def _masks(self, status: list[bool]) -> Bool[torch.Tensor, "batch_size num_classes"]:
        """Annotated class masks from the traffic-cone and barrier status of every sample."""
        return PARTIAL_ANNOTATIONS.build_annotated_class_masks(torch.tensor(status))

    def _loss(
        self,
        outputs: FocalHead2DOutputs,
        padding: list[bool],
        masks: Bool[torch.Tensor, "batch_size num_classes"] | None = None,
        num_cams: int = 2,
        box: Float32[torch.Tensor, "num_boxes 4"] | None = None,
        center: Float32[torch.Tensor, "num_boxes 2"] | None = None,
        label: Int64[torch.Tensor, " num_boxes"] | None = None,
    ) -> dict[str, Float[torch.Tensor, ""]]:
        """The five losses of one traffic cone on the first camera of every sample.

        The other cameras of every sample carry no box.
        """
        box = torch.tensor([[10.0, 10.0, 60.0, 60.0]]) if box is None else box
        center = torch.tensor([[35.0, 35.0]]) if center is None else center
        label = torch.tensor([5]) if label is None else label
        batch_size = len(padding)
        images = [(box, label, center), (box[:0], label[:0], center[:0])]
        images = images[:1] + images[1:] * (num_cams - 1)
        return self.head.loss(
            outputs,
            self._batch(images * batch_size, batch_size),
            padding_mask=torch.tensor(padding),
            gt_annotated_class_masks=masks,
        )

    def _batch(
        self,
        images: list[
            tuple[
                Float32[torch.Tensor, "num_boxes 4"],
                Int64[torch.Tensor, " num_boxes"],
                Float32[torch.Tensor, "num_boxes 2"],
            ]
        ],
        batch_size: int = 1,
    ) -> Detection2DGTBatch:
        """Collate the boxes, labels and centers of the images, camera by camera of every sample.

        The images are padded to the largest count, zero boxes with label -1 past the valid
        count, as the collate does.
        """
        max_num_boxes = max(boxes.shape[0] for boxes, _, _ in images)
        num_cams = len(images) // batch_size
        rows = torch.zeros(batch_size, num_cams, max_num_boxes, 6)
        labels = torch.full((batch_size, num_cams, max_num_boxes), -1, dtype=torch.int32)
        counts = torch.zeros(batch_size, num_cams, dtype=torch.int32)
        for image, (boxes, image_labels, centers) in enumerate(images):
            sample, camera = divmod(image, num_cams)
            count = boxes.shape[0]
            rows[sample, camera, :count] = torch.cat([boxes, centers], dim=1)
            labels[sample, camera, :count] = image_labels.to(torch.int32)
            counts[sample, camera] = count
        return Detection2DGTBatch(gt_bboxes_2d=rows, gt_labels_2d=labels, gt_valid_bboxes_2d=counts)

    def _heatmap(
        self, center: list[float], box: list[float], label: int = 0
    ) -> Float32[torch.Tensor, "feature_height feature_width"]:
        """The heatmap channel of the class of one box on the 6 x 10 grid.

        Every other channel must stay empty, which the helper checks.
        """
        heatmap = self._heatmaps(
            [torch.tensor([box])], [torch.tensor([label])], [torch.tensor([center])]
        )[0]
        others = torch.cat([heatmap[:label], heatmap[label + 1 :]])
        self.assertEqual(float(others.abs().sum()), 0.0)
        return heatmap[label]

    def _heatmaps(
        self,
        boxes: list[Float32[torch.Tensor, "num_boxes 4"]],
        labels: list[Int64[torch.Tensor, " num_boxes"]],
        centers: list[Float32[torch.Tensor, "num_boxes 2"]],
    ) -> Float32[torch.Tensor, "num_images num_classes feature_height feature_width"]:
        """The heatmaps of the given images on the 6 x 10 grid, one channel per class."""
        outputs = self.head(torch.randn(1, 1, 32, 6, 10), IMAGE_HEIGHT, IMAGE_WIDTH)
        ground_truth = self.head._ground_truth(
            outputs,
            self._batch(list(zip(boxes, labels, centers))),
            torch.zeros(len(boxes), dtype=torch.bool),
        )
        heatmaps = self.head._build_heatmaps(ground_truth, FEATURE_HEIGHT, FEATURE_WIDTH)
        return heatmaps.transpose(1, 2).reshape(len(boxes), -1, FEATURE_HEIGHT, FEATURE_WIDTH)

    def test_forward_predicts_every_token_of_every_image(self) -> None:
        """Test that the head predicts every token of every image, sample by sample.

        Input: neck features of two samples of two cameras, 6 x 10 tokens at stride 16 of
        96 x 160 images.
        Expected: the head predicts the class logits, the box, the projected center and the
        heatmap logits, one per class, of every token of every image, the images ordered sample by sample, and
        keeps the token grid and the image size the boxes and centers are normalized by.
        Check: the output shapes, the token grid and the image size.
        """
        outputs = self.head(torch.randn(2, 2, 32, 6, 10), image_height=96, image_width=160)

        self.assertEqual(outputs.cls_scores.shape, (4, 60, 7))
        self.assertEqual(outputs.bbox_preds.shape, (4, 60, 4))
        self.assertEqual(outputs.centers_2d.shape, (4, 60, 2))
        self.assertEqual(outputs.heatmap.shape, (4, 60, 7))
        self.assertEqual((outputs.feature_height, outputs.feature_width), (6, 10))
        self.assertEqual((outputs.image_height, outputs.image_width), (96, 160))

    def test_forward_decodes_the_boxes_and_centers_from_the_token_locations(self) -> None:
        """Test that the boxes and centers are decoded around the pixel center of every token.

        Input: the 6 x 10 tokens of a 96 x 160 image with the prediction convs set so that
        every token predicts the distances (0.10, 0.10, 0.05, 0.20) to the left, top, right
        and bottom box edges and the center offset (-1, 0), read at token (row 1, col 9),
        whose pixel center is (152, 24).
        Expected: the token location is (0.95, 0.25) normalized; the box is the location
        minus the left and top distances and plus the right and bottom ones, as
        (cx, cy, w, h) = (0.925, 0.30, 0.15, 0.30); the center is
        sigmoid(inverse_sigmoid(location) + offset) = (0.875, 0.25).
        Check: the box and the center of the token against those values.
        """
        with torch.no_grad():
            for conv in (self.head.bbox, self.head.center2d):
                conv.weight.zero_()
            self.head.bbox.bias.copy_(torch.logit(torch.tensor([0.10, 0.10, 0.05, 0.20])))
            self.head.center2d.bias.copy_(torch.tensor([-1.0, 0.0]))
        outputs = self.head(torch.randn(1, 1, 32, 6, 10), image_height=96, image_width=160)

        token = 1 * 10 + 9
        torch.testing.assert_close(
            outputs.bbox_preds[0, token], torch.tensor([0.925, 0.30, 0.15, 0.30])
        )
        expected_x = torch.sigmoid(torch.logit(torch.tensor(0.95)) - 1.0)
        torch.testing.assert_close(
            outputs.centers_2d[0, token], torch.stack([expected_x, torch.tensor(0.25)])
        )

    def test_forward_refuses_a_token_grid_that_is_not_the_image_at_the_stride(self) -> None:
        """Test that a token grid that does not cover the image at the stride is refused.

        Input: 6 x 10 tokens declared for 160 x 96 images (the sides swapped), 12 x 20
        tokens of 96 x 160 images (a stride-8 neck under a stride-16 head) and 6 x 10 tokens
        of a 100 x 168 image (an image that is not a multiple of the stride).
        Expected: the token locations, the heatmap targets and the box normalization all
        read the grid as the image over the stride, so a grid that does not cover the image
        exactly is refused, naming both sizes; a wrong stride or a non-divisible image would
        otherwise train against targets on the wrong grid without an error.
        Check: a ValueError naming the token grid and the image in the three cases.
        """
        for feature_height, feature_width, image_height, image_width in (
            (6, 10, 160, 96),
            (12, 20, 96, 160),
            (6, 10, 100, 168),
        ):
            with (
                self.subTest(grid=(feature_height, feature_width)),
                self.assertRaisesRegex(ValueError, "token grid at stride 16"),
            ):
                self.head(
                    torch.randn(1, 1, 32, feature_height, feature_width),
                    image_height=image_height,
                    image_width=image_width,
                )

    def test_loss_drops_the_background_terms_of_the_unannotated_classes(self) -> None:
        """Test that the mask of a sample drops the background terms of its unannotated classes.

        Input: one traffic cone on the first camera of two samples, scored without masks, with
        the masks of two samples annotated with traffic cones and barriers, and with sample 1
        annotated without them.
        Expected: the five losses are finite; masks of annotated samples give the plain loss,
        and the mask of sample 1 removes strictly positive background terms of its images from
        the classification and the heatmap losses, not from the box and center losses.
        Check: the losses finite, the classification and heatmap losses with and without the
        mask, and the other losses unchanged.
        """
        outputs = self.head(torch.randn(2, 2, 32, 6, 10), image_height=96, image_width=160)

        masked = self._loss(outputs, [False, False], masks=self._masks([True, False]))
        unmasked = self._loss(outputs, [False, False], masks=self._masks([True, True]))
        plain = self._loss(outputs, [False, False])

        for key in LOSS_KEYS:
            self.assertTrue(torch.isfinite(masked[key]), key)
        for key in ("loss_cls2d", "loss_heatmap2d"):
            torch.testing.assert_close(unmasked[key], plain[key], msg=key)
            self.assertLess(masked[key], unmasked[key], key)
        for key in ("loss_bbox2d", "loss_iou2d", "loss_centers2d"):
            torch.testing.assert_close(masked[key], unmasked[key], msg=key)

    def test_class_weights_follow_the_images_of_every_sample(self) -> None:
        """Test that every token of every image of a sample takes the mask row of its sample.

        Input: the masks of two samples, the second annotated without traffic cones and
        barriers, for two cameras of three tokens each.
        Expected: every token of every image of a sample takes the mask row of its sample, so
        only the six tokens of the two images of sample 1 lose the cone and barrier columns.
        Check: the token weights against the expected matrix.
        """
        weights = self.head._class_weights(self._masks([True, False]), 2, 3, torch.float32)

        expected = torch.ones(12, 7)
        # The two images of sample 1 are rows 6 to 11.
        expected[6:12, 5:7] = 0.0
        self.assertTrue(torch.equal(weights, expected))

    def test_a_matched_token_of_an_unannotated_class_keeps_its_terms(self) -> None:
        """Test that a token matched to a box of an unannotated class keeps every term.

        Input: one traffic cone on the first camera of two samples, sample 1 annotated without
        traffic cones and barriers.
        Expected: the token matched to the cone of sample 1 keeps every classification term,
        while every other token of the images of sample 1 loses its cone and barrier terms.
        Check: the gradient of the classification loss on the cone and barrier logits, non-zero
        on exactly one token of the first image of sample 1 and zero on its second image.
        """
        outputs = self.head(torch.randn(2, 2, 32, 6, 10), image_height=96, image_width=160)
        losses = self._loss(outputs, [False, False], masks=self._masks([True, False]))

        (grad,) = torch.autograd.grad(losses["loss_cls2d"], outputs.cls_scores)
        cone_and_barrier = grad[..., 5:7].abs().sum(dim=-1)
        # The images are ordered sample by sample, so images 2 and 3 are the cameras of sample 1.
        self.assertEqual(int(torch.count_nonzero(cone_and_barrier[2])), 1)
        self.assertEqual(int(torch.count_nonzero(cone_and_barrier[3])), 0)

    def test_heatmap_leaves_out_the_unannotated_classes_of_a_sample(self) -> None:
        """Test that the heatmap does not teach a sample's unannotated classes to be absent.

        Input: one camera of two samples, each with one car; sample 1 annotated without
        traffic cones and barriers; a heatmap that fires at 0.8 on every token and class, as a
        head that has learned cones from the annotated samples fires on an unannotated cone.
        Expected: the cone and barrier channels of sample 1 get no gradient, so a cone it shows
        is not pushed toward background, while its car channel is still pushed down off the
        car and the cone channel of sample 0 is still pushed down everywhere.
        Check: the heatmap-loss gradient on the cone and barrier channels of every token,
        zero for sample 1 and non-zero for sample 0, and non-zero on the car channel of
        sample 1.
        """
        outputs = self.head(torch.randn(2, 1, 32, 6, 10), image_height=96, image_width=160)
        heatmap_logits = torch.full_like(outputs.heatmap, math.log(0.8 / 0.2)).requires_grad_()
        outputs = FocalHead2DOutputs(
            cls_scores=outputs.cls_scores,
            bbox_preds=outputs.bbox_preds,
            centers_2d=outputs.centers_2d,
            heatmap=heatmap_logits,
            feature_height=FEATURE_HEIGHT,
            feature_width=FEATURE_WIDTH,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
        )
        losses = self._loss(
            outputs,
            [False, False],
            masks=self._masks([True, False]),
            num_cams=1,
            label=torch.tensor([0]),
        )

        (grad,) = torch.autograd.grad(losses["loss_heatmap2d"], heatmap_logits)
        self.assertEqual(float(grad[1, :, 5:7].abs().sum()), 0.0)
        self.assertGreater(int(torch.count_nonzero(grad[0, :, 5:7])), 0)
        self.assertGreater(int(torch.count_nonzero(grad[1, :, 0])), 0)

    def test_every_loss_trains_its_own_branch_and_the_image_features(self) -> None:
        """Test that every loss reaches its prediction conv, its tower and the features only.

        Input: one traffic cone on the first camera of two samples, from features requiring a
        gradient.
        Expected: the classification loss trains the class conv and the heatmap loss the
        heatmap conv, both through the classification tower; the box and the GIoU loss train
        the box conv and the center loss the center conv, through the regression tower; the
        IoU quality target of the classification is detached, so no loss crosses into the
        other tower, and every loss reaches the image features.
        Check: the set of parameters with a non-zero gradient of every loss.
        """
        features = torch.randn(2, 2, 32, 6, 10, requires_grad=True)
        losses = self._loss(self.head(features, image_height=96, image_width=160), [False, False])
        parameters = {
            "shared_cls": self.head.shared_cls[0].weight,
            "shared_reg": self.head.shared_reg[0].weight,
            "cls": self.head.cls.weight,
            "heatmap": self.head.heatmap.weight,
            "bbox": self.head.bbox.weight,
            "center2d": self.head.center2d.weight,
            "features": features,
        }
        expected = {
            "loss_cls2d": {"shared_cls", "cls", "features"},
            "loss_heatmap2d": {"shared_cls", "heatmap", "features"},
            "loss_bbox2d": {"shared_reg", "bbox", "features"},
            "loss_iou2d": {"shared_reg", "bbox", "features"},
            "loss_centers2d": {"shared_reg", "center2d", "features"},
        }

        for key, reached in expected.items():
            grads = torch.autograd.grad(
                losses[key], list(parameters.values()), retain_graph=True, allow_unused=True
            )
            non_zero = {
                name
                for name, grad in zip(parameters, grads)
                if grad is not None and bool((grad != 0).any())
            }
            self.assertEqual(non_zero, reached, key)

    def test_every_loss_weight_scales_its_own_term(self) -> None:
        """Test that every loss weight scales its term and no other.

        Input: one traffic cone on the first camera of two samples, scored by the head and by
        a copy whose weight of one loss is doubled, for every loss in turn.
        Expected: the loss of that weight doubles and the four others do not change.
        Check: the five losses of the copy against those of the head.
        """
        features = torch.randn(2, 2, 32, 6, 10)
        reference = self._loss(self.head(features, 96, 160), [False, False])
        weights = {
            "loss_cls2d": "loss_cls_weight",
            "loss_bbox2d": "loss_bbox_weight",
            "loss_iou2d": "loss_iou_weight",
            "loss_centers2d": "loss_centers2d_weight",
            "loss_heatmap2d": "loss_heatmap_weight",
        }
        for key, attribute in weights.items():
            with self.subTest(key=key):
                original = getattr(self.head, attribute)
                setattr(self.head, attribute, 2.0 * original)
                doubled = self._loss(self.head(features, 96, 160), [False, False])
                setattr(self.head, attribute, original)
                torch.testing.assert_close(doubled[key], 2.0 * reference[key])
                for other in LOSS_KEYS:
                    if other != key:
                        torch.testing.assert_close(doubled[other], reference[other], msg=other)

    def test_a_padding_sample_takes_no_part_in_the_loss(self) -> None:
        """Test that a padding sample supervises nothing.

        Input: a real sample with a traffic cone, alone and followed by a padding sample that
        repeats its features and its box.
        Expected: the padding sample keeps the ranks in step but supervises nothing: its box is
        left out and its images get no classification and no heatmap weight, so the losses
        are those of the real sample alone.
        Check: the five losses of both batches.
        """
        features = torch.randn(1, 2, 32, 6, 10)
        real = self._loss(self.head(features, image_height=96, image_width=160), [False])
        padded = self._loss(
            self.head(features.repeat(2, 1, 1, 1, 1), image_height=96, image_width=160),
            [False, True],
        )

        for key in LOSS_KEYS:
            torch.testing.assert_close(padded[key], real[key], msg=key)

    def test_a_batch_of_padding_alone_gives_a_zero_loss_with_a_graph(self) -> None:
        """Test that a batch of padding alone gives zero losses that reach every parameter.

        Input: a batch whose two samples are both padding slots carrying a 2D box.
        Expected: the batch supervises nothing, but every rank must take the same backward, so
        every loss is a zero that keeps the graph and reaches every parameter of the head.
        Check: the five losses are zero and require a gradient, and every parameter has a
        gradient after backward.
        """
        outputs = self.head(torch.randn(2, 2, 32, 6, 10), image_height=96, image_width=160)

        losses = self._loss(outputs, [True, True])
        sum(losses[key] for key in LOSS_KEYS).backward()

        for key in LOSS_KEYS:
            self.assertEqual(float(losses[key].detach()), 0.0, key)
            self.assertTrue(losses[key].requires_grad, key)
        for name, parameter in self.head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)

    def test_evaluation_counts_the_positives_of_the_batch_alone(self) -> None:
        """Test that the head in evaluation normalizes by its own positives without a collective.

        Input: one traffic cone on the first camera of two samples, scored by the head in
        training and in evaluation, with no process group.
        Expected: without a process group the training count is the floored count of the
        batch, which evaluation uses as well, so the losses are the same.
        Check: the five losses in both modes.
        """
        features = torch.randn(2, 2, 32, 6, 10)
        training = self._loss(self.head.train()(features, 96, 160), [False, False])
        with torch.no_grad():
            evaluation = self._loss(self.head.eval()(features, 96, 160), [False, False])

        for key in LOSS_KEYS:
            torch.testing.assert_close(evaluation[key], training[key].detach(), msg=key)

    def test_heatmaps_equal_the_gaussians_drawn_box_by_box(self) -> None:
        """Test that the batched heatmaps equal the Gaussians drawn box by box on every image.

        Input: three images of 0, 1 and 12 random boxes of random classes on the 6 x 10 grid,
        with two of the centers clamped onto the right and bottom edges of the image.
        Expected: the heatmaps are those of the scalar reference, a Gaussian of radius
        ceil(smallest center-to-edge distance / stride), at least 1, drawn with the maximum
        onto the channel of every box's class at the cell of its center, the edge centers in
        the last cell; an image without a box is all zeros.
        Check: the batched heatmaps equal the reference, and the empty image's heatmap is zero.
        """
        generator = torch.Generator().manual_seed(3)
        boxes, labels, centers = [], [], []
        for num_boxes in (0, 1, 12):
            corners = torch.rand(num_boxes, 2, generator=generator) * torch.tensor([120.0, 60.0])
            sizes = torch.rand(num_boxes, 2, generator=generator) * torch.tensor([40.0, 36.0])
            boxes.append(torch.cat([corners, corners + sizes], dim=1))
            labels.append(torch.randint(0, 7, (num_boxes,), generator=generator))
            centers.append(corners + sizes / 2)
        centers[2][0] = torch.tensor([160.0, 35.0])
        centers[2][1] = torch.tensor([35.0, 96.0])

        heatmaps = self._heatmaps(boxes, labels, centers)

        reference = torch.zeros(3, 7, FEATURE_HEIGHT, FEATURE_WIDTH)
        for image, (image_boxes, image_labels, image_centers) in enumerate(
            zip(boxes, labels, centers)
        ):
            for box, label, center in zip(image_boxes, image_labels, image_centers):
                edges = torch.cat([center - box[:2], box[2:] - center])
                radius = max(1, math.ceil(float(edges.min()) / 16))
                cell = (min(int(center[0]) // 16, 9), min(int(center[1]) // 16, 5))
                draw_heatmap_gaussian(reference[image, int(label)], cell, radius)
        self.assertEqual(float(heatmaps[0].abs().sum()), 0.0)
        torch.testing.assert_close(heatmaps, reference)

    def test_heatmap_peaks_at_the_object_center(self) -> None:
        """Test that the center heatmap peaks at 1 on the token of the object center.

        Input: a 50 x 50 px box from (10, 10) to (60, 60) with its center at (35, 35), on the
        6 x 10 token grid of a 96 x 160 image at stride 16.
        Expected: the heatmap has one cell per token and peaks at 1 on the token of the center,
        (2, 2), in the channel of the class of the box only, with a Gaussian of radius
        ceil(25 / 16) = 2 around it.
        Check: the heatmap shape, the peak and its location, the other channels empty, and zero
        beyond the radius.
        """
        heatmap = self._heatmap([35.0, 35.0], [10.0, 10.0, 60.0, 60.0], label=5)

        self.assertEqual(heatmap.shape, (6, 10))
        self.assertEqual(float(heatmap.max()), 1.0)
        self.assertEqual(divmod(int(heatmap.argmax()), 10), (2, 2))
        self.assertEqual(float(heatmap[:, 5:].abs().sum()), 0.0)

    def test_heatmap_puts_a_center_on_an_image_edge_in_the_edge_cell(self) -> None:
        """Test that a projected center clamped to an image edge peaks in the cell of the edge.

        Input: a box whose projected center sits on the right edge, at x = 160 of a 160 px
        wide image, the value the annotation transform gives a center beyond the edge; the
        same box with the center at x = 159.9 and at x = 0; and a center on the bottom edge.
        Expected: the right and bottom edges are inclusive, so 160 / 16 names the cell past
        the grid, where nothing would be drawn; the peak goes to the last cell instead, as
        the left edge peaks in cell 0, and a center just inside the edge peaks in the same
        cell.
        Check: the peak cell of every heatmap.
        """
        box = [110.0, 10.0, 160.0, 60.0]
        for center_x, column in ((160.0, 9), (159.9, 9), (0.0, 0)):
            with self.subTest(center_x=center_x):
                heatmap = self._heatmap([center_x, 35.0], box)
                self.assertEqual(float(heatmap.max()), 1.0)
                self.assertEqual(divmod(int(heatmap.argmax()), 10), (2, column))
        heatmap = self._heatmap([35.0, 96.0], [10.0, 50.0, 60.0, 96.0])
        self.assertEqual(divmod(int(heatmap.argmax()), 10), (5, 2))

    def test_loss_under_bf16_autocast_is_finite_and_close_to_float32(self) -> None:
        """Test that the head gives float32 outputs under bfloat16 autocast and its losses stay close.

        Input: one traffic cone on the first camera of two samples, scored in float32 and
        under bfloat16 autocast, where the convs of the head compute in bfloat16.
        Expected: the assigner and the losses take float32 only, so the head casts its outputs
        once; the five losses are finite and close to the float32 ones, and the gradient
        reaches the head.
        Check: the outputs float32, every loss finite and within bfloat16 precision of the
        float32 loss, and a gradient on the classification layer.
        """
        features = torch.randn(2, 2, 32, 6, 10)
        expected = self._loss(self.head(features, 96, 160), [False, False])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            outputs = self.head(features, 96, 160)
            losses = self._loss(outputs, [False, False])

        self.assertEqual(outputs.cls_scores.dtype, torch.float32)
        sum(losses[key] for key in LOSS_KEYS).backward()
        for key in LOSS_KEYS:
            self.assertTrue(torch.isfinite(losses[key]), key)
            torch.testing.assert_close(losses[key], expected[key], rtol=5e-2, atol=1e-3, msg=key)
        self.assertIsNotNone(self.head.cls.weight.grad)

    def test_more_boxes_than_tokens_leave_the_extra_boxes_unmatched(self) -> None:
        """Test that an image with more boxes than tokens matches one box per token.

        Input: 61 boxes of a traffic cone on the first camera of one sample of 60 tokens.
        Expected: the one-to-one assignment leaves one box without a token, so every token is
        a positive and the losses are finite.
        Check: the five losses finite and the classification gradient non-zero on every token
        of the first image.
        """
        outputs = self.head(torch.randn(1, 2, 32, 6, 10), image_height=96, image_width=160)
        corners = torch.arange(61, dtype=torch.float32).unsqueeze(1) * torch.tensor([[2.0, 1.0]])
        box = torch.cat([corners, corners + 20.0], dim=1)
        losses = self._loss(
            outputs,
            [False],
            box=box,
            center=corners + 10.0,
            label=torch.full((61,), 5, dtype=torch.long),
        )

        for key in LOSS_KEYS:
            self.assertTrue(torch.isfinite(losses[key]), key)
        (grad,) = torch.autograd.grad(losses["loss_cls2d"], outputs.cls_scores)
        self.assertEqual(int(torch.count_nonzero(grad[0, :, 5])), 60)

    def test_loss_refuses_a_box_or_center_that_is_not_finite_or_outside_the_image(self) -> None:
        """Test that a non-finite box or center, or one outside the image, is refused up front.

        Input: the predictions of one sample of a 96 x 160 image with a center holding a NaN
        coordinate, a box holding an infinite edge, a center at x = -20, and a box whose
        bottom edge is at y = 97; then a box and center on the edges of the image.
        Expected: a non-finite target would turn a loss NaN or fail inside the heatmap, and a
        finite one outside the image names a heatmap cell off the grid where no peak is drawn,
        both far from the cause, so the batch is refused up front naming the transform; the
        edges are inside the image, as the annotation transform clamps to them.
        Check: a ValueError naming the annotation transform in the four cases, and the five
        losses finite on the edges.
        """
        outputs = self.head(torch.randn(1, 2, 32, 6, 10), image_height=96, image_width=160)

        for center, box in (
            ([[35.0, math.nan]], None),
            (None, [[10.0, 10.0, math.inf, 60.0]]),
            ([[-20.0, 30.0]], None),
            (None, [[10.0, 10.0, 60.0, 97.0]]),
        ):
            with (
                self.subTest(center=center, box=box),
                self.assertRaisesRegex(ValueError, "annotation transform"),
            ):
                self._loss(
                    outputs,
                    [False],
                    center=None if center is None else torch.tensor(center),
                    box=None if box is None else torch.tensor(box),
                )
        losses = self._loss(
            outputs,
            [False],
            box=torch.tensor([[0.0, 0.0, 160.0, 96.0]]),
            center=torch.tensor([[160.0, 96.0]]),
        )
        for key in LOSS_KEYS:
            self.assertTrue(torch.isfinite(losses[key]), key)

    def test_embed_dims_must_split_into_the_group_norm_groups(self) -> None:
        """Test that the hidden channels must be a multiple of the 32 GroupNorm groups.

        Input: a head of 48 hidden channels.
        Expected: the conv towers normalize 32 groups of channels, so the hidden channels must
        be a multiple of 32.
        Check: a ValueError naming embed_dims is raised.
        """
        with self.assertRaisesRegex(ValueError, "embed_dims"):
            FocalHead2D(class_names=CLASS_NAMES, in_channels=32, embed_dims=48)


if __name__ == "__main__":
    unittest.main()
