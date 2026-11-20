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

"""Multi-rank tests of the FocalHead2D loss normalization.

They spawn gloo process groups on CPU so the collectives of ``reduce_mean_count`` run, and check
that sharding a batch over the ranks gives the DDP-averaged gradients of the whole batch on one
process. ``reduce_mean_count`` itself is tested in ``utils/tests``.
"""

from __future__ import annotations

from collections.abc import Sequence
import unittest

from jaxtyping import Float32, Int64
import torch
from torch.nn.parallel import DistributedDataParallel

from autoware_ml.dataclasses.batch.detection2d import Detection2DGTBatch
from autoware_ml.models.detection2d.heads.focal2d import FocalHead2D
from autoware_ml.utils.tests.distributed_fixtures import (
    GLOO_AVAILABLE,
    spawn_gloo_workers,
)

_CLASS_NAMES = ["car", "truck", "bus"]
_NUM_CLASSES = len(_CLASS_NAMES)


def _image(
    num_boxes: int, seed: int
) -> tuple[
    Float32[torch.Tensor, "1 1 channels height width"],
    Float32[torch.Tensor, "num_boxes 4"],
    Int64[torch.Tensor, " num_boxes"],
]:
    """Build the neck features of one 96 x 160 camera image with its 2D boxes and labels.

    Args:
        num_boxes: Number of 30 x 30 px boxes in the image.
        seed: Seed of the features and of the box corners.

    Returns:
        The features of one sample of one camera, the ``(x1, y1, x2, y2)`` boxes in pixels and
        their labels.
    """
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(1, 1, 32, 6, 10, generator=generator)
    corners = torch.rand(num_boxes, 2, generator=generator) * torch.tensor([100.0, 50.0])
    boxes = torch.cat([corners, corners + 30.0], dim=1)
    return features, boxes, torch.arange(num_boxes) % _NUM_CLASSES


def _head_loss(
    forward: torch.nn.Module,
    head: FocalHead2D,
    box_counts: tuple[int, ...],
    ranks: Sequence[int],
) -> Float32[torch.Tensor, ""]:
    """Sum the five losses of one camera image per sample of the given ranks.

    Args:
        forward: The module that runs the forward pass: the head itself, or the head wrapped
            in ``DistributedDataParallel`` so that its hooks average the gradients.
        head: The head whose ``loss`` scores the outputs.
        box_counts: Number of boxes in the image of every rank.
        ranks: Ranks whose images make the batch, one sample each.

    Returns:
        The sum of the five losses of the batch.
    """
    images = [_image(box_counts[rank], seed=90 + rank) for rank in ranks]
    outputs = forward(torch.cat([features for features, _, _ in images]), 96, 160)
    max_num_boxes = max(boxes.shape[0] for _, boxes, _ in images)
    rows = torch.zeros(len(images), 1, max_num_boxes, 6)
    labels = torch.full((len(images), 1, max_num_boxes), -1, dtype=torch.int32)
    for sample, (_, boxes, image_labels) in enumerate(images):
        rows[sample, 0, : boxes.shape[0]] = torch.cat([boxes, boxes[:, :2] + 15.0], dim=1)
        labels[sample, 0, : boxes.shape[0]] = image_labels.to(torch.int32)
    losses = head.loss(
        outputs,
        Detection2DGTBatch(
            gt_bboxes_2d=rows,
            gt_labels_2d=labels,
            gt_valid_bboxes_2d=torch.tensor([[boxes.shape[0]] for _, boxes, _ in images]).int(),
        ),
        padding_mask=torch.zeros(len(images), dtype=torch.bool),
    )
    return sum(losses.values())


def _build_head() -> FocalHead2D:
    """Build a seeded three-class head on 32-channel neck features of stride 16.

    Returns:
        The head, with the same parameters on every call.
    """
    torch.manual_seed(0)
    return FocalHead2D(class_names=_CLASS_NAMES, in_channels=32, embed_dims=32, stride=16)


def _whole_batch_gradients(box_counts: tuple[int, ...]) -> list[Float32[torch.Tensor, "*shape"]]:
    """Compute the gradients of the head over the whole batch on one process.

    Args:
        box_counts: Number of boxes in the image of every rank.

    Returns:
        The gradient of every parameter of the head, without a process group.
    """
    head = _build_head()
    _head_loss(head, head, box_counts, range(len(box_counts))).backward()
    return [parameter.grad for parameter in head.parameters()]


def _gradient_worker(
    rank: int,
    world_size: int,
    box_counts: tuple[int, ...],
    whole_gradients: list[Float32[torch.Tensor, "*shape"]],
) -> None:
    """Compare the DDP-averaged gradients of one rank with those of the whole batch.

    Args:
        rank: Rank of this process.
        world_size: Number of ranks.
        box_counts: Number of boxes in the image of every rank.
        whole_gradients: Gradients of the whole batch on one process.
    """
    head = _build_head()
    # The wrapper must outlive the backward pass, which its hooks average over the ranks.
    ddp_head = DistributedDataParallel(head)
    _head_loss(ddp_head, head, box_counts, [rank]).backward()

    # The convolutions sum their gradients in another order on one process, so the two sides
    # differ by float32 rounding, whose size depends on the CPU. A wrong normalization is off
    # by a factor of the counts.
    for sharded, whole in zip(head.parameters(), whole_gradients):
        torch.testing.assert_close(sharded.grad, whole, rtol=1e-4, atol=1e-6)


@unittest.skipUnless(GLOO_AVAILABLE, "gloo backend unavailable")
class TestFocalHead2DLossDistributed(unittest.TestCase):
    """Multi-rank tests of the FocalHead2D loss normalization."""

    def test_ddp_gradients_match_the_whole_batch_on_one_process(self) -> None:
        """Test that the head normalizes its losses like the 3D head.

        Input: one camera image per rank with 3 and 5 boxes, 1 and 0, and 0 and 0, scored by
        the head under DDP on two ranks and over the whole batch on one process.
        Expected: the DDP-averaged gradients equal those of the whole batch; with one positive
        in the step, or none, a floor of the averaged count on every rank, in the head or in
        the losses, would halve the gradients of the step.
        Check: every parameter gradient of every rank against the whole-batch gradient, within
        float32 rounding.
        """
        for box_counts in [(3, 5), (1, 0), (0, 0)]:
            with self.subTest(box_counts=box_counts):
                spawn_gloo_workers(
                    _gradient_worker,
                    box_counts,
                    _whole_batch_gradients(box_counts),
                    world_size=len(box_counts),
                )


if __name__ == "__main__":
    unittest.main()
