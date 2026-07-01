from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy.typing as npt
import numpy as np

from autoware_ml.types.geometry import Box3DFieldIndex


class Detection3DGTSample(NamedTuple):
    """
    Named tuple to represent a single sample of 3D detection GT.

    Attributes:
        gt_bboxes_3d: 3D bounding boxes in LiDAR coordinate, shape (N, 10), where N is
          the number of bounding boxes. Please check Box3DFieldIndex for the order of the 10
          parameters.
        gt_labels_3d: 3D bounding box labels, shape (N, ), where N is the number of bounding boxes.
    """

    gt_bboxes_3d: npt.NDArray[np.float32]  # (N, 10)
    gt_labels_3d: npt.NDArray[np.int32]  # (N, )


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

    gt_bboxes_3d: npt.NDArray[np.float32]  # (B, Maximum number of bboxes for each sample, 10)
    gt_labels_3d: npt.NDArray[np.int32]  # (B, Maximum number of bboxes for each sample)
    gt_valid_bboxes: npt.NDArray[np.int32]  # (B, ), number of maximum valid bboxes for each sample.

    @staticmethod
    def collate_gt_samples(
        detection3d_gt_samples: Sequence[Detection3DGTSample], max_num_3d_gt_bboxes: int
    ) -> Detection3DGTBatch:
        """
        Collate a sequence of Detection3DGTSample into a Detection3DGTBatch.

        Args:
          detection3d_gt_samples: Sequence of Detection3DGTSample to be collated.
          max_num_3d_gt_bboxes: The maximum number of 3D ground truth bounding boxes
            for each sample in the batch. If a sample has more than this number of bounding boxes,
            only the first `max_num_3d_gt_bboxes` gt bboxes will be included in the batch,
            and the rest will be ignored.

        Returns:
          Detection3DGTBatch: Collated 3D detection GT batch.
        """
        if len(detection3d_gt_samples) == 0:
            return Detection3DGTBatch(
                gt_bboxes_3d=None,
                gt_labels_3d=None,
                gt_valid_bboxes=None,
            )

        num_bbox_params = len(Box3DFieldIndex)
        # Initialize the arrays for gt_bboxes_3d and gt_labels_3d
        gt_bboxes_3d = np.zeros(
            (len(detection3d_gt_samples), max_num_3d_gt_bboxes, num_bbox_params), dtype=np.float32
        )
        gt_labels_3d = np.zeros((len(detection3d_gt_samples), max_num_3d_gt_bboxes), dtype=np.int32)

        # Fill the arrays with the data from gt_samples
        for i, sample in enumerate(detection3d_gt_samples):
            num_bboxes = sample.gt_bboxes_3d.shape[0]
            gt_bboxes_3d[i, :num_bboxes, :] = sample.gt_bboxes_3d
            gt_labels_3d[i, :num_bboxes] = sample.gt_labels_3d

            # Set to -1 for those gt_labels_3d that are invalid (i.e., beyond the number of valid bboxes)
            gt_labels_3d[i, num_bboxes:] = -1  # Assuming -1 is used to indicate invalid labels

        gt_valid_bboxes = np.asarray(
            [sample.gt_bboxes_3d.shape[0] for sample in detection3d_gt_samples], dtype=np.int32
        )

        return Detection3DGTBatch(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_valid_bboxes=gt_valid_bboxes,
        )
