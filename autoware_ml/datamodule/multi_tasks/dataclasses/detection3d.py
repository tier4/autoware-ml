from typing import NamedTuple

import numpy.typing as npt
import numpy as np


class Detection3DDataRow(NamedTuple):
    """
    Named tuple to represent a single row of 3D detection data, for example, 3D GT bounding boxes.
    """

    gt_bboxes_3d: npt.ArrayLike[np.float32]  # (N, 10)
    gt_labels_3d: npt.ArrayLike[np.int32]  # (N, )
