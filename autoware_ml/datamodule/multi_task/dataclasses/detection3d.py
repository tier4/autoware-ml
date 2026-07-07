from __future__ import annotations

import logging
from typing import NamedTuple, Sequence

from jaxtyping import Float32, Int32
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
    """

    # (batch_size, maximum number of bboxes, num_Box3DFieldIndex)
    gt_bboxes_3d: Float32[Tensor, "batch_size max_num_3d_gt_bboxes num_Box3DFieldIndex"]
    # (batch_size, maximum number of bboxes)
    gt_labels_3d: Int32[Tensor, "batch_size max_num_3d_gt_bboxes"]
    # (batch_size, ), number of valid bboxes for each sample in the batch
    gt_valid_bboxes: Int32[
        Tensor, " batch_size"
    ]  # (B, ), number of maximum valid bboxes for each sample.

    @staticmethod
    def collate_gt_samples(
        detection3d_gt_bboxes_3d: Sequence[BaseBBoxes3D], max_num_3d_gt_bboxes: int
    ) -> Detection3DGTBatch:
        """
        Collate a sequence of BaseBBoxes3D into a Detection3DGTBatch.

        Args:
          detection3d_gt_bboxes_3d: Sequence of BaseBBoxes3D to be collated.
          max_num_3d_gt_bboxes: The maximum number of 3D ground truth bounding boxes
            for each sample in the batch. If a sample has more than this number of bounding boxes,
            only the first `max_num_3d_gt_bboxes` gt bboxes will be included in the batch,
            and the rest will be ignored.

        Returns:
          Detection3DGTBatch: Collated 3D detection GT batch.
        """
        if len(detection3d_gt_bboxes_3d) == 0:
            return Detection3DGTBatch(
                gt_bboxes_3d=None,
                gt_labels_3d=None,
                gt_valid_bboxes=None,
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

        # Fill the arrays with the data from gt_samples
        for i, sample in enumerate(detection3d_gt_bboxes_3d):
            num_bboxes = min(sample.gt_bboxes_3d.shape[0], max_num_3d_gt_bboxes)
            if sample.gt_bboxes_3d.shape[0] > max_num_3d_gt_bboxes:
                logger.info(
                    f"Warning: num_bboxes in the sample exceeds the "
                    f"maximum value: {max_num_3d_gt_bboxes}, and thus they are trimmed to "
                    f"{max_num_3d_gt_bboxes}"
                )

            gt_bboxes_3d[i, :num_bboxes, :] = sample.gt_bboxes_3d[:num_bboxes, :]
            gt_labels_3d[i, :num_bboxes] = sample.gt_labels_3d[:num_bboxes]

            # Set to -1 for those gt_labels_3d that are invalid (i.e., beyond the number of valid bboxes)
            gt_labels_3d[i, num_bboxes:] = -1  # Assuming -1 is used to indicate invalid labels
            gt_valid_bboxes[i] = num_bboxes

        return Detection3DGTBatch(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_valid_bboxes=gt_valid_bboxes,
        )
