from typing import NamedTuple

import numpy.typing as npt
import numpy as np


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
        gt_bboxes_3d: 3D bounding boxes in LiDAR coordinate, shape (B*N, 10), where B is the
          batch size and N is the number of bounding boxes. Please check Box3DFieldIndex for
          the order of the 10 parameters.
        gt_labels_3d: 3D bounding box labels, shape (B*N, ), where B is the batch size and N is
          the number of bounding boxes.
        gt_bboxes_3d_batch_indices: Batch indices for each bounding box, shape (B*N, ), where B
          is the batch size and N is the number of bounding boxes.
    """

    gt_bboxes_3d: npt.NDArray[np.float32]  # (B*N, 10)
    gt_labels_3d: npt.NDArray[np.int32]  # (B*N, )
    gt_bboxes_3d_batch_indices: npt.NDArray[np.int32]  # (B*N, )
