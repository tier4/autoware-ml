"""Per-camera 2D detection ground truth of a sample and of a batch."""

from __future__ import annotations

import logging
from typing import NamedTuple, Sequence

from jaxtyping import Float32, Int32, Int64
import torch
from torch import Tensor

from autoware_ml.types.geometry import Box2DFieldIndex

logger = logging.getLogger(__name__)


class Detection2DGTSample(NamedTuple):
    """2D detection ground truth of one sample, one entry per camera.

    The boxes can come from any source: projected from the 3D boxes of the sample, as
    LoadAnnotations2DFromBoxes3D does, or annotated in 2D. Either way they are in pixels of the
    images the sample serves, after every image-space augmentation.

    Attributes:
        gt_bboxes_2d: Boxes of every camera, one row per box laid out by Box2DFieldIndex:
          x_min, y_min, x_max, y_max of the box in pixels of the augmented image, then
          center_x, center_y of the object center in the same pixels (the projected 3D
          gravity center for a box projected from a 3D box).
        gt_labels_2d: Class index of every box of every camera.
    """

    gt_bboxes_2d: Sequence[Float32[Tensor, "num_boxes num_Box2DFieldIndex"]]
    gt_labels_2d: Sequence[Int32[Tensor, " num_boxes"]]


class Detection2DGTBatch(NamedTuple):
    """2D boxes of every camera of every sample of a batch, padded to a fixed number per camera.

    Attributes:
        gt_bboxes_2d: Boxes, shape (B, C, M, 6), where B is the batch size, C the number of
          cameras and M the max_num_2d_gt_bboxes the batch is collated with. The last axis is
          laid out by Box2DFieldIndex: x_min, y_min, x_max, y_max of the box in pixels of the
          augmented image, then center_x, center_y of the object center. Zeros past the valid
          count.
        gt_labels_2d: Class index of every box, shape (B, C, M), -1 past the valid count.
        gt_valid_bboxes_2d: Number of valid boxes of every camera of every sample, shape (B, C).
    """

    gt_bboxes_2d: Float32[Tensor, "batch_size num_cameras max_num_2d_gt_bboxes num_Box2DFieldIndex"]
    gt_labels_2d: Int32[Tensor, "batch_size num_cameras max_num_2d_gt_bboxes"]
    gt_valid_bboxes_2d: Int32[Tensor, "batch_size num_cameras"]

    @staticmethod
    def collate_gt_samples(
        samples: Sequence[Detection2DGTSample], max_num_2d_gt_bboxes: int
    ) -> Detection2DGTBatch:
        """
        Collate the 2D boxes of every camera of every sample into tensors of a fixed size.

        Every camera is padded to ``max_num_2d_gt_bboxes`` boxes, so the batch has the same
        shape whatever the samples hold. A camera holding more boxes keeps only its first
        ``max_num_2d_gt_bboxes`` ones, in the order the sample lists them, and the others are
        left out of the batch (the cut is logged), as the 3D boxes are cut at
        ``max_num_3d_gt_bboxes``.

        Args:
          samples: 2D ground truth of every sample, in batch order.
          max_num_2d_gt_bboxes: Number of boxes every camera is padded to, and cut at.

        Returns:
          Detection2DGTBatch: The boxes, labels and valid box count of every camera of every
            sample. Camera ``c`` of sample ``b`` holds its boxes in the first
            ``gt_valid_bboxes_2d[b, c]`` rows of ``gt_bboxes_2d[b, c]`` and
            ``gt_labels_2d[b, c]``, the boxes laid out by Box2DFieldIndex; the remaining rows
            up to ``max_num_2d_gt_bboxes`` are padding, zero boxes with label -1.

        Raises:
          ValueError: If the batch is empty, its samples disagree on the camera count, or the
            boxes of a camera are not laid out by Box2DFieldIndex or differ in count from its
            labels.
        """
        if len(samples) == 0:
            raise ValueError("At least one 2D detection sample is required for collating.")
        num_cameras = len(samples[0].gt_bboxes_2d)
        num_fields = len(Box2DFieldIndex)
        shape = (len(samples), num_cameras, max_num_2d_gt_bboxes)
        gt_bboxes_2d = torch.zeros((*shape, num_fields), dtype=torch.float32)
        gt_labels_2d = torch.full(shape, -1, dtype=torch.int32)
        gt_valid_bboxes_2d = torch.zeros(shape[:2], dtype=torch.int32)
        for sample_index, sample in enumerate(samples):
            if len(sample.gt_bboxes_2d) != num_cameras:
                raise ValueError(
                    f"Every sample needs the 2D boxes of {num_cameras} cameras, sample "
                    f"{sample_index} has {len(sample.gt_bboxes_2d)}."
                )
            # ``strict`` rejects labels given for another number of cameras.
            for camera_index, (boxes, labels) in enumerate(
                zip(sample.gt_bboxes_2d, sample.gt_labels_2d, strict=True)
            ):
                # The copies below would broadcast a single column or label over the box, and
                # fail on other mismatches without naming the sample.
                if boxes.ndim != 2 or boxes.shape[1] != num_fields:
                    raise ValueError(
                        f"The 2D boxes of sample {sample_index}, camera {camera_index} have "
                        f"shape {tuple(boxes.shape)}, expected (num_boxes, {num_fields}) laid "
                        "out by Box2DFieldIndex."
                    )
                count = boxes.shape[0]
                if labels.shape[0] != count:
                    raise ValueError(
                        f"The 2D boxes and labels of sample {sample_index}, camera "
                        f"{camera_index} differ in count: {count} and {labels.shape[0]}."
                    )
                if count > max_num_2d_gt_bboxes:
                    logger.info(
                        f"Sample {sample_index}, camera {camera_index} holds {count} 2D boxes, "
                        f"more than max_num_2d_gt_bboxes: {max_num_2d_gt_bboxes}, so only the "
                        f"first {max_num_2d_gt_bboxes} are collated."
                    )
                    count = max_num_2d_gt_bboxes
                gt_bboxes_2d[sample_index, camera_index, :count] = boxes[:count]
                gt_labels_2d[sample_index, camera_index, :count] = labels[:count]
                gt_valid_bboxes_2d[sample_index, camera_index] = count

        return Detection2DGTBatch(
            gt_bboxes_2d=gt_bboxes_2d,
            gt_labels_2d=gt_labels_2d,
            gt_valid_bboxes_2d=gt_valid_bboxes_2d,
        )

    def _unpad(self, padded: Tensor) -> list[list[Tensor]]:
        """Cut every camera of every sample back to its valid boxes."""
        return [
            [
                padded[sample_index, camera_index, :count]
                for camera_index, count in enumerate(counts)
            ]
            for sample_index, counts in enumerate(self.gt_valid_bboxes_2d.tolist())
        ]

    def valid_bboxes_2d(self) -> list[list[Float32[Tensor, "num_boxes num_Box2DFieldIndex"]]]:
        """
        Keep the valid boxes of every camera of every sample.

        Returns:
          list: Boxes per sample, then per camera, without the padding, laid out by
            Box2DFieldIndex.
        """
        return self._unpad(self.gt_bboxes_2d)

    def valid_labels_2d(self) -> list[list[Int64[Tensor, " num_boxes"]]]:
        """
        Keep the labels of the valid boxes of every camera of every sample.

        Returns:
          list: Labels per sample, then per camera, without the padding, as int64 class
            indices.
        """
        return [
            [labels.long() for labels in sample_labels]
            for sample_labels in self._unpad(self.gt_labels_2d)
        ]

    def to_device(self, device: torch.device) -> Detection2DGTBatch:
        """
        Move the batch to the specified device.

        Args:
          device: The target device to move the batch to.

        Returns:
          Detection2DGTBatch: The batch on the device.
        """
        return Detection2DGTBatch(
            gt_bboxes_2d=self.gt_bboxes_2d.to(device),
            gt_labels_2d=self.gt_labels_2d.to(device),
            gt_valid_bboxes_2d=self.gt_valid_bboxes_2d.to(device),
        )
