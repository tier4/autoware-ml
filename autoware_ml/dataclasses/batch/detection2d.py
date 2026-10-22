"""Per-camera 2D detection ground truth of a sample and of a batch."""

from __future__ import annotations

from typing import NamedTuple, Sequence

from jaxtyping import Float32, Int32, Int64
import torch
from torch import Tensor

from autoware_ml.types.geometry import Box2DFieldIndex


class Detection2DGTSample(NamedTuple):
    """2D boxes of one sample, one entry per camera, in the pixels of the served images.

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
    """2D boxes of every camera of every sample of a batch, padded to the largest camera.

    Attributes:
        gt_bboxes_2d: Boxes laid out by Box2DFieldIndex: x_min, y_min, x_max, y_max in pixels
          of the augmented image, then the center_x, center_y of the object center. Zeros past
          the valid count.
        gt_labels_2d: Class indices, -1 past the valid count.
        gt_valid_bboxes_2d: Number of valid boxes of every camera of every sample.
    """

    gt_bboxes_2d: Float32[Tensor, "batch_size num_cameras max_num_boxes num_Box2DFieldIndex"]
    gt_labels_2d: Int32[Tensor, "batch_size num_cameras max_num_boxes"]
    gt_valid_bboxes_2d: Int32[Tensor, "batch_size num_cameras"]

    @staticmethod
    def collate_gt_samples(samples: Sequence[Detection2DGTSample]) -> Detection2DGTBatch:
        """
        Pad the boxes of every camera of every sample to the largest camera of the batch.

        Args:
          samples: 2D ground truth of every sample, in batch order.

        Returns:
          Detection2DGTBatch: The padded batch.

        Raises:
          ValueError: If the batch is empty, its samples disagree on the camera count, or the
            boxes of a camera are not laid out by Box2DFieldIndex or differ in count from its
            labels.
        """
        if len(samples) == 0:
            raise ValueError("At least one 2D detection sample is required for collating.")
        num_cameras = len(samples[0].gt_bboxes_2d)
        num_fields = len(Box2DFieldIndex)
        max_num_boxes = max(
            (boxes.shape[0] for sample in samples for boxes in sample.gt_bboxes_2d), default=0
        )
        shape = (len(samples), num_cameras, max_num_boxes)
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
                gt_bboxes_2d[sample_index, camera_index, :count] = boxes
                gt_labels_2d[sample_index, camera_index, :count] = labels
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
