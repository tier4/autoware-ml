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

import pytest
import torch

from autoware_ml.models.detection3d.heads.focal2d import FocalHead2D
from autoware_ml.models.detection3d.task_modules.partial_annotations import PartialAnnotations

CLASS_NAMES = ["car", "truck", "bus", "bicycle", "pedestrian", "traffic_cone", "barrier"]
PARTIAL_ANNOTATIONS = PartialAnnotations(CLASS_NAMES, ["traffic_cone", "barrier"])
LOSS_KEYS = ("loss_cls2d", "loss_bbox2d", "loss_iou2d", "loss_centers2d", "loss_centerness2d")
IMAGE_HEIGHT, IMAGE_WIDTH = 96, 160


def _masks(status: list[bool]) -> torch.Tensor:
    return PARTIAL_ANNOTATIONS.build_annotated_class_masks(torch.tensor(status))


def _head() -> FocalHead2D:
    torch.manual_seed(0)
    return FocalHead2D(class_names=CLASS_NAMES, in_channels=32, embed_dims=32, stride=16)


def _annotations(padding: list[bool], num_cams: int = 2) -> dict[str, object]:
    """One traffic cone on the first camera of every sample, no box on the other cameras."""
    box = torch.tensor([[10.0, 10.0, 60.0, 60.0]])
    center = torch.tensor([[35.0, 35.0]])
    label = torch.tensor([5])
    no_box, no_center, no_label = box[:0], center[:0], label[:0]
    batch_size = len(padding)
    return {
        "gt_bboxes_2d": [[box] + [no_box] * (num_cams - 1)] * batch_size,
        "gt_labels_2d": [[label] + [no_label] * (num_cams - 1)] * batch_size,
        "centers_2d": [[center] + [no_center] * (num_cams - 1)] * batch_size,
        "padding_mask": torch.tensor(padding),
    }


def test_forward_predicts_every_token_of_every_image() -> None:
    """
    Input: neck features of two samples of two cameras, 6 x 10 tokens at stride 16 of 96 x 160
    images.
    Expected: the head predicts the class logits, the box, the projected center and the
    centerness of every token of every image, the images ordered sample by sample, and keeps
    the image size the boxes and centers are normalized by.
    Check: the output shapes and the image size.
    """
    outputs = _head()(torch.randn(2, 2, 32, 6, 10), image_height=96, image_width=160)

    assert outputs.cls_scores.shape == (4, 60, 7)
    assert outputs.bbox_preds.shape == (4, 60, 4)
    assert outputs.centers_2d.shape == (4, 60, 2)
    assert outputs.centerness.shape == (4, 60, 1)
    assert (outputs.image_height, outputs.image_width) == (96, 160)


def test_loss_drops_the_background_terms_of_the_unannotated_classes() -> None:
    """
    Input: one traffic cone on the first camera of two samples, scored without masks, with the
    masks of two samples annotated with traffic cones and barriers, and with sample 1 annotated
    without them.
    Expected: the five losses are finite; masks of annotated samples give the plain loss, and
    the mask of sample 1 removes strictly positive background terms of its images from the
    classification loss only; masks of another batch size are refused.
    Check: the losses finite, the classification loss with and without the mask, the other
    losses unchanged, and a ValueError naming the expected shape.
    """
    head = _head()
    outputs = head(torch.randn(2, 2, 32, 6, 10), image_height=96, image_width=160)
    annotations = _annotations([False, False])

    masked = head.loss(outputs, gt_annotated_class_masks=_masks([True, False]), **annotations)
    unmasked = head.loss(outputs, gt_annotated_class_masks=_masks([True, True]), **annotations)
    plain = head.loss(outputs, **annotations)

    assert all(torch.isfinite(masked[key]) for key in LOSS_KEYS)
    torch.testing.assert_close(unmasked["loss_cls2d"], plain["loss_cls2d"])
    assert masked["loss_cls2d"] < unmasked["loss_cls2d"]
    for key in ("loss_bbox2d", "loss_iou2d", "loss_centers2d", "loss_centerness2d"):
        torch.testing.assert_close(masked[key], unmasked[key], msg=key)
    with pytest.raises(ValueError, match="batch_size, num_classes"):
        head.loss(outputs, gt_annotated_class_masks=_masks([True]), **annotations)


def test_class_weights_follow_the_images_of_every_sample() -> None:
    """
    Input: the masks of two samples, the second annotated without traffic cones and barriers,
    for two cameras of three tokens each.
    Expected: every token of every image of a sample takes the mask row of its sample, so only
    the six tokens of the two images of sample 1 lose the cone and barrier columns; no mask
    gives no weights.
    Check: the token weights against the expected matrix, and None without a mask.
    """
    head = _head()

    weights = head._class_weights(_masks([True, False]), 2, 3, torch.float32)

    expected = torch.ones(12, 7)
    # The two images of sample 1 are rows 6 to 11.
    expected[6:12, 5:7] = 0.0
    assert torch.equal(weights, expected)
    assert head._class_weights(None, 2, 3, torch.float32) is None


def test_a_matched_token_of_an_unannotated_class_keeps_its_terms() -> None:
    """
    Input: one traffic cone on the first camera of two samples, sample 1 annotated without
    traffic cones and barriers.
    Expected: the token matched to the cone of sample 1 is vouched for by its box and keeps
    every classification term, as in TransFusion, while every other token of the images of
    sample 1 loses its cone and barrier terms.
    Check: the gradient of the classification loss on the cone and barrier logits, non-zero on
    exactly one token of the first image of sample 1 and zero on its second image.
    """
    head = _head()
    outputs = head(torch.randn(2, 2, 32, 6, 10), image_height=96, image_width=160)
    losses = head.loss(
        outputs, gt_annotated_class_masks=_masks([True, False]), **_annotations([False, False])
    )

    (grad,) = torch.autograd.grad(losses["loss_cls2d"], outputs.cls_scores)
    cone_and_barrier = grad[..., 5:7].abs().sum(dim=-1)
    # The images are ordered sample by sample, so images 2 and 3 are the cameras of sample 1.
    assert torch.count_nonzero(cone_and_barrier[2]) == 1
    assert torch.count_nonzero(cone_and_barrier[3]) == 0


def test_a_padding_sample_takes_no_part_in_the_loss() -> None:
    """
    Input: a real sample with a traffic cone, alone and followed by a padding sample that
    repeats its features and its box.
    Expected: the padding sample keeps the ranks in step but supervises nothing: its box is
    left out and its images get no classification and no centerness weight, so the losses are
    those of the real sample alone.
    Check: the five losses of both batches.
    """
    head = _head()
    features = torch.randn(1, 2, 32, 6, 10)
    real = head.loss(head(features, image_height=96, image_width=160), **_annotations([False]))
    padded = head.loss(
        head(features.repeat(2, 1, 1, 1, 1), image_height=96, image_width=160),
        **_annotations([False, True]),
    )

    for key in LOSS_KEYS:
        torch.testing.assert_close(padded[key], real[key], msg=key)


def test_a_batch_of_padding_alone_gives_a_zero_loss_with_a_graph() -> None:
    """
    Input: a batch whose two samples are both padding slots carrying a 2D box.
    Expected: the batch supervises nothing, but every rank must take the same backward, so every
    loss is a zero that keeps the graph and reaches every parameter of the head.
    Check: the five losses are zero and require a gradient, and every parameter has a gradient
    after backward.
    """
    head = _head()
    outputs = head(torch.randn(2, 2, 32, 6, 10), image_height=96, image_width=160)

    losses = head.loss(outputs, **_annotations([True, True]))
    sum(losses[key] for key in LOSS_KEYS).backward()

    assert all(float(losses[key].detach()) == 0.0 for key in LOSS_KEYS)
    assert all(losses[key].requires_grad for key in LOSS_KEYS)
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_centerness_heatmap_peaks_at_the_object_center() -> None:
    """
    Input: a 50 x 50 px box from (10, 10) to (60, 60) with its center at (35, 35), on the 6 x 10
    token grid of a 96 x 160 image at stride 16.
    Expected: the heatmap has one cell per token and peaks at 1 on the token of the center,
    (2, 2), with a Gaussian of radius ceil(25 / 16) = 2 around it.
    Check: the heatmap shape, the peak and its location, and zero beyond the radius.
    """
    heatmap = _head()._build_heatmap(
        torch.tensor([[35.0, 35.0]]),
        torch.tensor([[10.0, 10.0, 60.0, 60.0]]),
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )

    assert heatmap.shape == (6, 10)
    assert float(heatmap.max()) == 1.0
    assert divmod(int(heatmap.argmax()), 10) == (2, 2)
    assert float(heatmap[:, 5:].abs().sum()) == 0.0


def test_loss_refuses_annotations_of_other_images() -> None:
    """
    Input: the predictions of two samples of two cameras, with the annotations of three cameras.
    Expected: the annotations of an image are read at its index in the predictions, so
    annotations of another number of images are refused.
    Check: a ValueError naming the images is raised.
    """
    head = _head()
    outputs = head(torch.randn(2, 2, 32, 6, 10), image_height=96, image_width=160)

    with pytest.raises(ValueError, match="2D predictions cover 4 images"):
        head.loss(outputs, **_annotations([False, False], num_cams=3))


def test_embed_dims_must_split_into_the_group_norm_groups() -> None:
    """
    Input: a head of 48 hidden channels.
    Expected: the conv towers normalize 32 groups of channels, so the hidden channels must be a
    multiple of 32.
    Check: a ValueError naming embed_dims is raised.
    """
    with pytest.raises(ValueError, match="embed_dims"):
        FocalHead2D(class_names=CLASS_NAMES, in_channels=32, embed_dims=48)
