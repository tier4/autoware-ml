# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Point-cloud loading transforms to support ModelGTSample.
The code is modified from mmdetection3d.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated

from jaxtyping import Bool
import numpy as np
import torch
from jaxtyping import Float32
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from torch import Tensor

from autoware_ml.geometry.points.base_points import BasePoints
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.geometry.point_clouds import LiDARPointCloudSample
from autoware_ml.databases.t4pack.t4pack import T4Pack
from autoware_ml.types.dataset import PCDFileFormat
from autoware_ml.types.geometry import PointFeatureName, PointFieldIndex


class SweepSelection(StrEnum):
    """
    How the appended sweeps are picked among the eligible stored frames.

    Attributes:
      NEAREST: The frames closest in time to the current frame. Used in evaluation.
      RANDOM: A uniform draw without replacement. Used in training so the network sees varying
        time lags.
    """

    NEAREST = "nearest"
    RANDOM = "random"


class SweepWindow(BaseModel):
    """
    Sweeps appended from one side of the current frame.

    time_lag_range bounds the distance in seconds between an eligible stored frame and the
    current frame. Frames outside it are skipped, like frames past the start or end of a scene.

    Attributes:
      num: Number of sweeps appended when enough frames are eligible.
      time_lag_range: Inclusive [min, max] distance in seconds, with 0 < min < max. Lag 0 marks
        the current frame.
      selection: How the appended sweeps are picked among the eligible frames.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    num: int = Field(ge=1)
    time_lag_range: Annotated[tuple[float, float], BeforeValidator(tuple)]
    selection: Annotated[SweepSelection, BeforeValidator(SweepSelection)]

    @model_validator(mode="after")
    def validate_time_lag_range(self) -> SweepWindow:
        """
        Validate the ordering of the time lag bounds.

        Returns:
          SweepWindow: The validated window.
        """

        if self.min_time_lag <= 0.0 or self.min_time_lag >= self.max_time_lag:
            raise ValueError(
                f"Expected 0 < min time lag < max time lag, got {list(self.time_lag_range)}."
            )
        return self

    @property
    def min_time_lag(self) -> float:
        """Smallest distance in seconds an eligible frame may have."""
        return self.time_lag_range[0]

    @property
    def max_time_lag(self) -> float:
        """Largest distance in seconds an eligible frame may have."""
        return self.time_lag_range[1]


class LoadPointsFromFile(BaseTransform):
    """Load point clouds from a lidar file path stored in sample metadata."""

    _required_keys = ["lidar_point_cloud_samples"]

    def __init__(
        self,
        use_dim: Sequence[int] = (0, 1, 2, 3),
        bev_remove_radius: float = 0.0,
        pcd_file_format: PCDFileFormat = PCDFileFormat.AUTO,
    ) -> None:
        """Initialize the point-cloud loader.

        Args:
            use_dim: Feature dimensions to keep. The number of stored features is read from
                each frame of the record.
            bev_remove_radius: Radius (x and y) within which points are removed, for example the
                ego vehicle points. Set to 0.0 to disable point removal.
            pcd_file_format: File format to read the point clouds from: ``bin`` for the
                ``.pcd.bin`` files, ``t4pack`` for the pack of the channel, or ``auto`` for the
                pack when the record has a pack location and the ``.pcd.bin`` file otherwise.

        Raises:
            TypeError: Raised when ``pcd_file_format`` is not a ``PCDFileFormat``.
        """
        if not isinstance(pcd_file_format, PCDFileFormat):
            raise TypeError(
                f"pcd_file_format must be a PCDFileFormat, got {type(pcd_file_format).__name__}."
            )
        super().__init__(probability=None)
        self.use_dim = list(use_dim)
        self.bev_remove_radius = bev_remove_radius
        self.pcd_file_format = pcd_file_format

    def far_from_origin(self, points_data: BasePoints) -> Bool[Tensor, " num_points"]:
        """Mask of the points outside the removal radius around the origin.

        Args:
            points_data: Points to test.

        Returns:
            Bool[Tensor, " num_points"]: True for every point to keep, all of them when the
              removal is disabled.
        """
        if self.bev_remove_radius <= 0:
            return torch.ones(len(points_data), dtype=torch.bool, device=points_data.points.device)
        x_filtered = torch.abs(points_data.points[:, PointFieldIndex.X]) < self.bev_remove_radius
        y_filtered = torch.abs(points_data.points[:, PointFieldIndex.Y]) < self.bev_remove_radius
        return ~(x_filtered & y_filtered)

    def remove_close(self, points_data: BasePoints) -> BasePoints:
        """Remove the points within the removal radius around the origin.

        Args:
            points_data (BasePoints): Sweep points.

        Returns:
            BasePoints: Points after removing.
        """
        points_data.remove_points(self.far_from_origin(points_data))
        return points_data

    def read_points(
        self, lidar_point_cloud_sample: LiDARPointCloudSample
    ) -> Float32[np.ndarray, "num_points num_features"]:
        """Read all features of one point cloud in the configured file format.

        Args:
            lidar_point_cloud_sample: Metadata of the point cloud, with its pack location if any.

        Returns:
            Float32[np.ndarray, "num_points num_features"]: The points with every stored feature.
        """
        if self.pcd_file_format == PCDFileFormat.BIN:
            return self._read_bin_points(lidar_point_cloud_sample)
        if self.pcd_file_format == PCDFileFormat.T4PACK:
            return self._read_t4pack_points(lidar_point_cloud_sample)
        # AUTO: the pack when the record has a pack location, the .pcd.bin file otherwise
        if lidar_point_cloud_sample.t4pack_frame is None:
            return self._read_bin_points(lidar_point_cloud_sample)
        return self._read_t4pack_points(lidar_point_cloud_sample)

    def _read_bin_points(
        self, lidar_point_cloud_sample: LiDARPointCloudSample
    ) -> Float32[np.ndarray, "num_points num_features"]:
        """Read all features of one point cloud from its ``.pcd.bin`` file.

        Args:
            lidar_point_cloud_sample: Metadata of the point cloud.

        Returns:
            Float32[np.ndarray, "num_points num_features"]: The points with every stored feature.
        """
        return np.fromfile(lidar_point_cloud_sample.point_cloud_path, dtype=np.float32).reshape(
            -1, lidar_point_cloud_sample.num_features
        )

    def _read_t4pack_points(
        self, lidar_point_cloud_sample: LiDARPointCloudSample
    ) -> Float32[np.ndarray, "num_points num_features"]:
        """Read all features of one point cloud from the t4pack file of its channel.

        Args:
            lidar_point_cloud_sample: Metadata of the point cloud, with its pack location.

        Returns:
            Float32[np.ndarray, "num_points num_features"]: The points with every stored feature.

        Raises:
            ValueError: Raised when the record has no pack location.
        """
        if lidar_point_cloud_sample.t4pack_frame is None:
            raise ValueError(
                f"The record of {lidar_point_cloud_sample.point_cloud_path} has no t4pack "
                "location. Generate the records with the pack in place, or set "
                "pcd_file_format to PCDFileFormat.BIN or PCDFileFormat.AUTO."
            )
        return T4Pack.from_point_cloud_path(lidar_point_cloud_sample.point_cloud_path).read_frame(
            lidar_point_cloud_sample.t4pack_frame, lidar_point_cloud_sample.num_features
        )

    def load_points_from_samples(
        self, index: int, lidar_point_cloud_samples: Sequence[LiDARPointCloudSample]
    ) -> BasePoints:
        """Load point cloud data from a binary file.

        Args:
            index: Index of the point cloud sample to load.
            lidar_point_cloud_samples: Sequence of LiDARPointCloudSample containing metadata for
                each point cloud sample, including file paths and timestamps.

        Returns:
            BasePoints: Loaded point cloud data.
        """
        if index >= len(lidar_point_cloud_samples):
            raise IndexError(
                f"Index {index} is out of bounds for lidar_point_cloud_samples with length "
                f"{len(lidar_point_cloud_samples)}."
            )

        lidar_point_cloud_sample = lidar_point_cloud_samples[index]
        points_np = self.read_points(lidar_point_cloud_sample)
        source_view = lidar_point_cloud_sample.source_view
        if source_view is not None:
            end = source_view.point_index_begin + source_view.num_points
            if end > points_np.shape[0]:
                raise ValueError(
                    f"Lidar source points [{source_view.point_index_begin}, {end}) exceed the "
                    f"{points_np.shape[0]} points of {lidar_point_cloud_sample.point_cloud_path}."
                )
            points_np = points_np[source_view.point_index_begin : end].copy()
            frame_to_sensor = np.linalg.inv(source_view.sensor_to_frame_matrix.numpy())
            points_np[:, :3] = points_np[:, :3] @ frame_to_sensor[:3, :3].T + frame_to_sensor[:3, 3]

        use_dims = self.use_dim
        if use_dims[:3] != [PointFieldIndex.X, PointFieldIndex.Y, PointFieldIndex.Z]:
            raise ValueError(
                "use_dim must start with [0, 1, 2] (x, y, z) to keep geometry transforms "
                f"correct, but got {use_dims}."
            )

        if max(use_dims) >= points_np.shape[1]:
            raise ValueError(
                f"use_dim {use_dims} reads past the {points_np.shape[1]} features stored per "
                f"point in {lidar_point_cloud_sample.point_cloud_path}."
            )

        points_np = points_np[:, use_dims]
        # Every model reads intensity in [0, 1], scaled once here by the full scale value the
        # database declares for its files
        if PointFieldIndex.INTENSITY in use_dims:
            points_np = points_np.copy()
            points_np[:, use_dims.index(PointFieldIndex.INTENSITY)] /= (
                lidar_point_cloud_sample.intensity_scale
            )
        point_feature_names = [PointFeatureName(PointFieldIndex(i).name.lower()) for i in use_dims]
        return LiDARPoints.from_numpy(
            points_np=points_np,
            point_feature_names=point_feature_names,
            timestamp=lidar_point_cloud_sample.timestamp,
        )

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Load point data from the current sample at the current sweep.

        Args:
            model_gt_sample: ModelGTSample instance containing `lidar_point_cloud_samples`.

        Returns:
            Updated ModelGTSample instance with a loaded `point_cloud_data` array.
        """
        if not model_gt_sample.lidar_point_cloud_samples:
            raise ValueError("No lidar point cloud samples found in the ModelGTSample.")

        # The first frame of the record is the current frame
        lidar_points = self.load_points_from_samples(0, model_gt_sample.lidar_point_cloud_samples)
        return model_gt_sample._replace(point_cloud_data=lidar_points).keep_points(
            self.far_from_origin(lidar_points)
        )


class LoadMultiSweepPointsFromFile(LoadPointsFromFile):
    """Append the stored sweep points of a sample to its current frame.

    The stored frames of the record are split into past and future frames by their capture
    time, and each window selects the sweeps of its side. Every point carries a time lag, the
    current frame timestamp minus its capture timestamp. The lag is 0 for the current frame,
    positive for past sweeps and negative for future sweeps. The output holds the current frame
    first, then the past sweeps nearest first, then the future sweeps nearest first.

    A sweep that is missing, or outside the window, adds no points and is not replaced by
    another frame.
    """

    _required_keys = ["lidar_point_cloud_samples", "point_cloud_data"]

    def __init__(
        self,
        past: SweepWindow | None = None,
        future: SweepWindow | None = None,
        use_timestamp_difference: bool = True,
        use_dim: Sequence[int] = (0, 1, 2, 3),
        bev_remove_radius: float = 1.0,
        pcd_file_format: PCDFileFormat = PCDFileFormat.AUTO,
    ) -> None:
        """Initialize the multi-sweep point-cloud loader.

        Args:
            past: Sweeps appended from the frames captured before the current frame, none when
              omitted.
            future: Sweeps appended from the frames captured after the current frame, none when
              omitted.
            use_timestamp_difference: Whether to add a timestamp difference feature to each
              point, the current frame timestamp minus the capture timestamp of the point.
            use_dim: Selected feature dimensions preserved in the loaded tensor.
            bev_remove_radius: Radius (x and y) within which sweep points are removed, for
              instance to drop the ego vehicle returns. Set to 0.0 to keep every point.
            pcd_file_format: File format to read the point clouds from, see
                ``LoadPointsFromFile``.
        """
        super().__init__(
            use_dim=use_dim, bev_remove_radius=bev_remove_radius, pcd_file_format=pcd_file_format
        )
        self._check_window("past", past)
        self._check_window("future", future)
        if past is None and future is None:
            raise ValueError(
                "The multi sweep loader appends no points without a past or a future window, "
                "load the current frame alone with LoadPointsFromFile."
            )
        self.past = past
        self.future = future
        self.use_timestamp_difference = use_timestamp_difference

    @staticmethod
    def _check_window(name: str, window: SweepWindow | None) -> None:
        """Reject a window that is not a SweepWindow.

        Args:
            name: Argument name for the error message.
            window: Configured window.
        """
        if window is not None and not isinstance(window, SweepWindow):
            raise TypeError(f"{name} must be a SweepWindow or None, got {type(window).__name__}.")

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Load the sweeps each window selects and append them to the current frame.

        Args:
            model_gt_sample: ModelGTSample instance containing `lidar_point_cloud_samples`.

        Returns:
            Updated ModelGTSample instance whose `point_cloud_data` carries the sweeps.
        """
        if not model_gt_sample.lidar_point_cloud_samples:
            raise ValueError("No lidar point cloud samples found in the ModelGTSample.")

        if model_gt_sample.point_cloud_data is None:
            raise ValueError("Point cloud data is not available in the ModelGTSample.")

        if model_gt_sample.lidar_point_cloud_samples[0].source_view is not None:
            raise ValueError(
                "The sample serves a single lidar source, and the stored sweeps merge every "
                "source, so no sweep can be appended to it."
            )

        current_frame_point_cloud_data = model_gt_sample.point_cloud_data
        past_frames, future_frames = self._split_stored_frames(model_gt_sample)
        selected_sweeps = [
            *self._select_sweeps(self.past, past_frames),
            *self._select_sweeps(self.future, future_frames),
        ]

        if self.use_timestamp_difference:
            current_frame_point_cloud_data.add_timestamp_difference(0.0)

        concat_points = [current_frame_point_cloud_data]
        for time_lag, sweep_index in selected_sweeps:
            sweep_points = self.load_points_from_samples(
                sweep_index, model_gt_sample.lidar_point_cloud_samples
            )
            sweep_points = self.remove_close(sweep_points)
            if self.use_timestamp_difference:
                sweep_points.add_timestamp_difference(time_lag)

            # Transform from the lidar sweep frame to the current lidar frame using the provided
            # transformation matrices.
            # Check https://github.com/open-mmlab/mmdetection3d/issues/3054
            sweep_lidar_sample = model_gt_sample.lidar_point_cloud_samples[sweep_index]
            translation_vector = sweep_lidar_sample.lidar_sensor_to_lidar_sweep_matrix[:3, 3]
            rotation_matrix = sweep_lidar_sample.lidar_sensor_to_lidar_sweep_matrix[:3, :3]
            # Subtract first, sensor to lidar
            sweep_points.translate(-translation_vector)
            # Rotate: P @ R (Lidar_to_sweep) equivalent to R^T (sweep_to_lidar) @ P
            sweep_points.rotate(rotation_matrix)
            concat_points.append(sweep_points)

        # Concatenate all points from the current frame and the selected sweeps
        multi_sweep_points = LiDARPoints.concat(concat_points)

        segmentation3d_gt_sample = model_gt_sample.segmentation3d_gt_sample
        if segmentation3d_gt_sample is not None:
            # The sweeps only shape the geometry, their points carry the ignore label
            segmentation3d_gt_sample = segmentation3d_gt_sample.append_ignored_labels(
                len(multi_sweep_points) - len(current_frame_point_cloud_data)
            )

        return model_gt_sample._replace(
            point_cloud_data=multi_sweep_points,
            segmentation3d_gt_sample=segmentation3d_gt_sample,
        )

    @staticmethod
    def _split_stored_frames(
        model_gt_sample: ModelGTSample,
    ) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
        """Pair every stored sweep with its signed time lag and split the two sides.

        The frames after the first are the stored sweeps. A sweep with the timestamp of the
        current frame is rejected, because lag 0 marks the current frame.

        Args:
            model_gt_sample: ModelGTSample holding the stored lidar frames.

        Returns:
            The frames captured before the current frame and the ones captured after it, each
            paired with its signed time lag and its position in the record.
        """
        lidar_point_cloud_samples: Sequence[LiDARPointCloudSample] = (
            model_gt_sample.lidar_point_cloud_samples
        )
        current_timestamp = lidar_point_cloud_samples[0].timestamp
        past_frames: list[tuple[float, int]] = []
        future_frames: list[tuple[float, int]] = []
        for index, sweep_lidar_sample in enumerate(lidar_point_cloud_samples[1:], start=1):
            time_lag = current_timestamp - sweep_lidar_sample.timestamp
            if time_lag == 0.0:
                raise ValueError(
                    f"The record stores sweep {sweep_lidar_sample.point_cloud_path} at the "
                    "timestamp of its current frame."
                )
            side = past_frames if time_lag > 0.0 else future_frames
            side.append((time_lag, index))
        return past_frames, future_frames

    @staticmethod
    def _select_sweeps(
        window: SweepWindow | None, lagged_frames: Sequence[tuple[float, int]]
    ) -> list[tuple[float, int]]:
        """Return the sweeps a window appends from one side, nearest first.

        Frames outside the window are skipped, so a scene with missing neighbors yields fewer
        sweeps.

        Args:
            window: Window declared for this side, None when nothing is appended from it.
            lagged_frames: Stored sweeps of this side with their signed time lag.

        Returns:
            Selected sweeps with their signed time lag, nearest first.
        """
        if window is None:
            return []
        eligible = [
            lagged_frame
            for lagged_frame in lagged_frames
            if window.min_time_lag <= abs(lagged_frame[0]) <= window.max_time_lag
        ]
        eligible.sort(key=lambda lagged_frame: abs(lagged_frame[0]))
        if window.selection is SweepSelection.RANDOM and len(eligible) > window.num:
            indices = torch.randperm(len(eligible))[: window.num].tolist()
            return sorted(
                (eligible[index] for index in indices),
                key=lambda lagged_frame: abs(lagged_frame[0]),
            )
        return eligible[: window.num]
