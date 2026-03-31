"""Point-cloud loading transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from autoware_ml.transforms.base import BaseTransform


class LoadPointsFromFile(BaseTransform):
    """Load point clouds from a lidar file path stored in sample metadata."""

    _required_keys = ["lidar_path"]

    def __init__(self, load_dim: int = 5, use_dim: Sequence[int] | int = (0, 1, 2, 3)) -> None:
        """Initialize the point-cloud loader.

        Args:
            load_dim: Number of features stored per point in the source file.
            use_dim: Selected feature dimensions preserved in the loaded tensor.
        """
        self.load_dim = load_dim
        self.use_dim = use_dim

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Load point data from the configured lidar file.

        Args:
            input_dict: Sample metadata containing ``lidar_path``.

        Returns:
            Updated sample dictionary with a loaded ``points`` array.
        """
        load_dim = int(input_dict.get("num_pts_feats", self.load_dim))
        points = np.fromfile(input_dict["lidar_path"], dtype=np.float32).reshape(-1, load_dim)

        # Loading single source sensor points
        idx_begin = input_dict.get("idx_begin")
        length = input_dict.get("length")
        if idx_begin is not None and length is not None:
            points = points[idx_begin : idx_begin + length]

        # Transform points to the coordinate frame of a single source sensor
        translation = input_dict.get("translation")
        rotation = input_dict.get("rotation")
        if translation is not None and rotation is not None:
            points[:, :3] = (points[:, :3] - translation) @ rotation

        use_dim = self.use_dim
        if isinstance(use_dim, int):
            points = points[:, :use_dim]
        else:
            points = points[:, list(use_dim)]

        output = {"points": points.astype(np.float32)}
        if idx_begin is not None:
            output["idx_begin"] = idx_begin
        if length is not None:
            output["length"] = length

        return output


__all__ = ["LoadPointsFromFile"]
