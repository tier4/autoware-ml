from typing import NamedTuple

import numpy.typing as npt
import numpy as np


class Segmentation3DGTSample(NamedTuple):
    """Named tuple to represent a single sample of 3D segmentation GT data."""

    gt_semantic_mask: npt.NDArray[np.int32]  # (N, 1)
