from __future__ import annotations

import logging
from typing import NamedTuple, Sequence

from jaxtyping import Bool, Float32, Int32
from torch import Tensor
import torch

from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.types.geometry import Box3DFieldIndex

logger = logging.getLogger(__name__)


class Detection3DGTBatch(NamedTuple):
    """
    Named tuple to represent a batch of 3D detection GT after collating from sequence of
    Detection3DGTSample when inputting to the 3D detection model.

    Attributes:
        gt_bboxes_3d: 3D bounding boxes in LiDAR coordinate, shape (B, M, 10), where B is the
          batch size and M is the theoretically maximum number of bounding boxes for each sample.
          For all invalid bboxes, they are all set to zeros.
          Please check Box3DFieldIndex for the order of the 10 parameters.
        gt_labels_3d: 3D bounding box labels, shape (B, M), where B is the batch size and M is
          the theoretically maximum number of bounding boxes.
          For all invalid bboxes, they are all set to -1.
        gt_valid_bboxes: Valid number of gt bboxes for each sample, shape (B, ), where B is
          the batch size.
        gt_bboxes_num_points: Number of points in each gt bbox, shape (B, M), where B is the
          batch size and M is the theoretically maximum number of bounding boxes.
          0 for invalid bboxes.
        gt_traffic_cone_barrier_bbox_status: Whether traffic cones and barriers are annotated in each
          sample, shape (B, ). None when the dataset does not carry the flag.
    """

    # (batch_size, maximum number of bboxes, num_Box3DFieldIndex)
    gt_bboxes_3d: Float32[Tensor, "batch_size max_num_gt_bboxes num_Box3DFieldIndex"]
    # (batch_size, maximum number of bboxes)
    gt_labels_3d: Int32[Tensor, "batch_size max_num_gt_bboxes"]
    # (batch_size, ), number of valid bboxes for each sample in the batch
    gt_valid_bboxes: Int32[
        Tensor, " batch_size"
    ]  # (B, ), number of maximum valid bboxes for each sample.
    gt_bboxes_num_points: Int32[
        Tensor, "batch_size max_num_3d_gt_bboxes"
    ]  # (B, M), number of points in each gt bbox
    # (batch_size, ), whether traffic cones/barriers are annotated in each sample
    gt_traffic_cone_barrier_bbox_status: Bool[Tensor, " batch_size"] | None = None

    @staticmethod
    def collate_traffic_cone_barrier_bbox_status(
        detection3d_traffic_cone_barrier_bbox_status: Sequence[bool | None],
        device: torch.device,
    ) -> Bool[Tensor, " batch_size"] | None:
        """
        Collate per-sample traffic cone/barrier annotation flags into one boolean tensor.

        Args:
          detection3d_traffic_cone_barrier_bbox_status: Flag of every sample in the batch. Either all
            samples carry the flag or none of them does.
          device: Device of the returned tensor.

        Returns:
          Bool[Tensor, " batch_size"] | None: One flag per sample, or None when no sample carries
            the flag.

        Raises:
          ValueError: If only some samples carry the flag.
        """
        if all(status is None for status in detection3d_traffic_cone_barrier_bbox_status):
            return None
        if any(status is None for status in detection3d_traffic_cone_barrier_bbox_status):
            raise ValueError(
                "All samples must have detection3d_traffic_cone_barrier_bbox_status for collating, "
                f"got {detection3d_traffic_cone_barrier_bbox_status}."
            )
        return torch.tensor(
            detection3d_traffic_cone_barrier_bbox_status, dtype=torch.bool, device=device
        )

    @staticmethod
    def collate_gt_samples(
        detection3d_gt_bboxes_3d: Sequence[BaseBBoxes3D],
        max_num_3d_gt_bboxes: int,
        detection3d_traffic_cone_barrier_bbox_status: Sequence[bool | None] | None = None,
    ) -> Detection3DGTBatch | None:
        """
        Collate a sequence of BaseBBoxes3D into a Detection3DGTBatch.

        Args:
          detection3d_gt_bboxes_3d: Sequence of BaseBBoxes3D to be collated.
          max_num_3d_gt_bboxes: The maximum number of 3D ground truth bounding boxes
            for each sample in the batch. If a sample has more than this number of bounding boxes,
            only the first `max_num_3d_gt_bboxes` gt bboxes will be included in the batch,
            and the rest will be ignored.
          detection3d_traffic_cone_barrier_bbox_status: Optional per-sample flag telling whether
            traffic cones and barriers are annotated, aligned with ``detection3d_gt_bboxes_3d``.

        Returns:
          Detection3DGTBatch | None: Collated 3D detection GT batch or None if the input
            sequence is empty.

        Raises:
          ValueError: If the flags are given for only some of the samples, or their count does
            not match the number of samples.
        """
        if len(detection3d_gt_bboxes_3d) == 0:
            return None

        if detection3d_traffic_cone_barrier_bbox_status is not None and len(
            detection3d_traffic_cone_barrier_bbox_status
        ) != len(detection3d_gt_bboxes_3d):
            raise ValueError(
                "detection3d_traffic_cone_barrier_bbox_status must have one entry per sample, got "
                f"{len(detection3d_traffic_cone_barrier_bbox_status)} entries for "
                f"{len(detection3d_gt_bboxes_3d)} samples."
            )

        num_bbox_params = len(Box3DFieldIndex)

        bbox_params_dtype = detection3d_gt_bboxes_3d[0].bbox_params.dtype
        torch_device = detection3d_gt_bboxes_3d[0].bbox_params.device

        # Initialize the arrays for gt_bboxes_3d and gt_labels_3d
        gt_bboxes_3d = torch.zeros(
            (len(detection3d_gt_bboxes_3d), max_num_3d_gt_bboxes, num_bbox_params),
            dtype=bbox_params_dtype,
            device=torch_device,
        )
        gt_labels_3d = torch.zeros(
            (len(detection3d_gt_bboxes_3d), max_num_3d_gt_bboxes),
            dtype=torch.int32,
            device=torch_device,
        )
        gt_valid_bboxes = torch.zeros(
            len(detection3d_gt_bboxes_3d), dtype=torch.int32, device=torch_device
        )
        gt_bboxes_num_points = torch.zeros(
            (len(detection3d_gt_bboxes_3d), max_num_3d_gt_bboxes),
            dtype=torch.int32,
            device=torch_device,
        )

        # Fill the arrays with the data from gt_samples
        for i, sample_gt_bboxes_3d in enumerate(detection3d_gt_bboxes_3d):
            num_gt_bboxes_3d = len(sample_gt_bboxes_3d)
            num_bboxes = min(num_gt_bboxes_3d, max_num_3d_gt_bboxes)
            if num_gt_bboxes_3d > max_num_3d_gt_bboxes:
                logger.info(
                    f"Warning: num_bboxes: {num_gt_bboxes_3d} in the sample exceeds the "
                    f"maximum value: {max_num_3d_gt_bboxes}, and thus they are trimmed to "
                    f"{max_num_3d_gt_bboxes}"
                )

            gt_bboxes_3d[i, :num_bboxes, :] = sample_gt_bboxes_3d.bbox_params[:num_bboxes, :]
            gt_labels_3d[i, :num_bboxes] = sample_gt_bboxes_3d.bbox_labels[:num_bboxes]
            gt_bboxes_num_points[i, :num_bboxes] = sample_gt_bboxes_3d.bbox_num_lidar_points[
                :num_bboxes
            ]

            # Set to -1 for those gt_labels_3d that are invalid (i.e., beyond the number of valid bboxes)
            gt_labels_3d[i, num_bboxes:] = -1  # Assuming -1 is used to indicate invalid labels
            gt_valid_bboxes[i] = num_bboxes

        gt_traffic_cone_barrier_bbox_status = (
            Detection3DGTBatch.collate_traffic_cone_barrier_bbox_status(
                detection3d_traffic_cone_barrier_bbox_status, device=torch_device
            )
            if detection3d_traffic_cone_barrier_bbox_status is not None
            else None
        )

        return Detection3DGTBatch(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_valid_bboxes=gt_valid_bboxes,
            gt_bboxes_num_points=gt_bboxes_num_points,
            gt_traffic_cone_barrier_bbox_status=gt_traffic_cone_barrier_bbox_status,
        )

    def to_device(self, device: torch.device) -> Detection3DGTBatch:
        """
        Move the Detection3DGTBatch to the specified device.

        Args:
          device: The target device to move the batch to.

        Returns:
          Detection3DGTBatch: A new Detection3DGTBatch on the specified device.
        """
        return Detection3DGTBatch(
            gt_bboxes_3d=self.gt_bboxes_3d.to(device),
            gt_labels_3d=self.gt_labels_3d.to(device),
            gt_valid_bboxes=self.gt_valid_bboxes.to(device),
            gt_bboxes_num_points=self.gt_bboxes_num_points.to(device),
            gt_traffic_cone_barrier_bbox_status=self.gt_traffic_cone_barrier_bbox_status.to(device)
            if self.gt_traffic_cone_barrier_bbox_status is not None
            else None,
        )
