from typing import NamedTuple

import numpy.typing as npt
import numpy as np


class Segmentation3DDataRow(NamedTuple):
    """
    Named tuple to represent a single row of 3D segmentation data, for example,
    3D GT semantic masks.
    """

    gt_semantic_mask: npt.ArrayLike[np.int32]  # (N, 1)
