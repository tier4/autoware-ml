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

"""Voxel feature encoders feeding the PTv3 embedding stem."""

from __future__ import annotations

import torch
import torch.nn as nn

TIME_LAG_COLUMN = 4
POINT_CHANNELS = TIME_LAG_COLUMN + 1
# Channels of a single frame point: x, y, z and intensity.
FRAME_POINT_CHANNELS = TIME_LAG_COLUMN
# Channels of one side: offset from the voxel position, intensity, share and mean time lag.
SIDE_CHANNELS = 6


class VoxelFeatureEncoder(nn.Module):
    """Base of the parameter free encoders that reduce the points of a voxel to one vector.

    Each encoder reads its own point layout and checks a batch against it before the forward.
    Sweep points carry a time lag, the capture time of the sample minus the one of the point, so
    it is zero for the current frame, positive for a past sweep and negative for a future one.
    Training and export run the same math.

    Attributes:
        out_channels: Width of the voxel feature vector.
    """

    out_channels: int

    def check_points(self, voxels: torch.Tensor, time_lag_column: int) -> None:
        """Check that the batch points follow the layout the encoder reads.

        Args:
            voxels: Padded voxel points of shape ``(num_voxels, max_points, channels)``.
            time_lag_column: Point column holding the time lag, ``-1`` when the points have none.

        Raises:
            ValueError: If the points do not follow the layout of the encoder.
        """
        raise NotImplementedError

    @staticmethod
    def _check_time_lag_column(time_lag_column: int) -> None:
        """Check that the points carry their time lag in the column the sweep encoders read.

        Args:
            time_lag_column: Point column holding the time lag, ``-1`` when the points have none.

        Raises:
            ValueError: If the time lag is in another column or missing.
        """
        if time_lag_column != TIME_LAG_COLUMN:
            raise ValueError(
                f"PTv3 needs the time lag of the points in column {TIME_LAG_COLUMN}, the batch "
                f"has it in column {time_lag_column}. Enable use_timestamp_difference in the "
                "sweep loader."
            )

    @staticmethod
    def _point_masks(
        voxels: torch.Tensor, num_points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the valid points of every voxel by capture time.

        Args:
            voxels: Padded voxel points of shape ``(num_voxels, max_points, channels)``.
            num_points: Number of valid points per voxel of shape ``(num_voxels,)``.

        Returns:
            Masks of the current frame, past and future points, each of shape
            ``(num_voxels, max_points)``.

        Raises:
            ValueError: If the points are not laid out as ``(x, y, z, intensity, time_lag)``.
        """
        if voxels.shape[2] != POINT_CHANNELS:
            raise ValueError(
                f"Voxel feature encoders need points laid out as (x, y, z, intensity, "
                f"time_lag), got {voxels.shape[2]} channels."
            )
        slots = torch.arange(voxels.shape[1], device=voxels.device).unsqueeze(0)
        filled = slots < num_points.long().unsqueeze(1)
        time_lag = voxels[..., TIME_LAG_COLUMN]
        return filled & (time_lag == 0), filled & (time_lag > 0), filled & (time_lag < 0)

    @staticmethod
    def _summarize(
        voxels: torch.Tensor, selected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Count the selected points of every voxel and average their features.

        Args:
            voxels: Padded voxel points of shape ``(num_voxels, max_points, channels)``.
            selected: Mask of the points to summarize, of shape ``(num_voxels, max_points)``.

        Returns:
            The count of shape ``(num_voxels, 1)`` and the mean of shape
            ``(num_voxels, channels)``, the mean being zero where nothing was selected.
        """
        count = selected.sum(dim=1, keepdim=True).to(voxels.dtype)
        mean = (voxels * selected.unsqueeze(-1)).sum(dim=1) / count.clamp(min=1.0)
        return count, mean

    @staticmethod
    def _describe(
        core: torch.Tensor,
        side_mean: torch.Tensor,
        side_count: torch.Tensor,
        total_count: torch.Tensor,
    ) -> torch.Tensor:
        """Describe the points of one side relative to the position of the voxel.

        Args:
            core: Voxel features of shape ``(num_voxels, channels)``.
            side_mean: Mean features of this side, of shape ``(num_voxels, channels)``.
            side_count: Number of points on this side, of shape ``(num_voxels, 1)``.
            total_count: Number of points in the voxel, of shape ``(num_voxels, 1)``.

        Returns:
            Features of shape ``(num_voxels, 6)``, the offset from the voxel position, the
            intensity, the share and the mean time lag of the side, all zero for a side that
            observed nothing.
        """
        observed = side_count > 0
        offset = torch.where(
            observed, side_mean[:, :3] - core[:, :3], torch.zeros_like(core[:, :3])
        )
        intensity = torch.where(observed, side_mean[:, 3:4], torch.zeros_like(side_count))
        lag = torch.where(observed, side_mean[:, 4:5], torch.zeros_like(side_count))
        return torch.cat([offset, intensity, side_count / total_count, lag], dim=1)


class MeanVoxelFeatureEncoder(VoxelFeatureEncoder):
    """Encode a single frame voxel as the mean of its points.

    Points are laid out as ``(x, y, z, intensity)`` with no time lag, so the voxel is placed at
    the mean of its points and carries their mean intensity.
    """

    out_channels = FRAME_POINT_CHANNELS

    def check_points(self, voxels: torch.Tensor, time_lag_column: int) -> None:
        """Check that the batch points are single frame points without a time lag.

        Args:
            voxels: Padded voxel points of shape ``(num_voxels, max_points, channels)``.
            time_lag_column: Point column holding the time lag, ``-1`` when the points have none.

        Raises:
            ValueError: If the points carry a time lag.
        """
        if time_lag_column != -1:
            raise ValueError(
                f"{type(self).__name__} reads single frames, the batch has a time lag in column "
                f"{time_lag_column}. Load no sweeps."
            )

    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """Reduce padded voxel points to one feature vector per voxel.

        Args:
            voxels: Padded voxel points of shape ``(num_voxels, max_points, 4)``.
            num_points: Number of valid points per voxel of shape ``(num_voxels,)``.

        Returns:
            Voxel features of shape ``(num_voxels, 4)``, the mean position and intensity of the
            points of the voxel.

        Raises:
            ValueError: If the points are not laid out as ``(x, y, z, intensity)``.
        """
        if voxels.shape[2] != FRAME_POINT_CHANNELS:
            raise ValueError(
                f"{type(self).__name__} needs points laid out as (x, y, z, intensity), got "
                f"{voxels.shape[2]} channels. Load no sweeps."
            )
        slots = torch.arange(voxels.shape[1], device=voxels.device).unsqueeze(0)
        _, mean = self._summarize(voxels, slots < num_points.long().unsqueeze(1))
        return mean


class SweepSplitVoxelFeatureEncoder(VoxelFeatureEncoder):
    """Encode the current frame and the past points of each voxel apart from each other.

    The voxel is placed at the mean of its current frame points, and its past points are
    described by their offset from there, their intensity, their share of the voxel and their
    mean time lag. A voxel with no current frame points takes the position and time lag of its
    past points.
    """

    out_channels = POINT_CHANNELS + SIDE_CHANNELS

    def check_points(self, voxels: torch.Tensor, time_lag_column: int) -> None:
        """Check that the batch points carry their time lag and no future points.

        Args:
            voxels: Padded voxel points of shape ``(num_voxels, max_points, 5)``.
            time_lag_column: Point column holding the time lag, ``-1`` when the points have none.

        Raises:
            ValueError: If the time lag is in another column or missing, or if the batch has
                future points.
        """
        self._check_time_lag_column(time_lag_column)
        if bool((voxels[..., TIME_LAG_COLUMN] < 0).any()):
            raise ValueError(
                f"{type(self).__name__} reads no future points, but the batch has some. Load no "
                "future sweeps or use PastFutureVoxelFeatureEncoder."
            )

    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """Reduce padded voxel points to one feature vector per voxel.

        Args:
            voxels: Padded voxel points of shape ``(num_voxels, max_points, 5)``.
            num_points: Number of valid points per voxel of shape ``(num_voxels,)``.

        Returns:
            Voxel features of shape ``(num_voxels, 11)``, the position, intensity and time lag
            of the voxel followed by the offset, intensity, share and time lag of its past
            points.
        """
        current, past, _ = self._point_masks(voxels, num_points)
        current_count, current_mean = self._summarize(voxels, current)
        past_count, past_mean = self._summarize(voxels, past)
        core = torch.where(current_count > 0, current_mean, past_mean)
        total_count = (current_count + past_count).clamp(min=1.0)
        return torch.cat([core, self._describe(core, past_mean, past_count, total_count)], dim=1)


class PastFutureVoxelFeatureEncoder(VoxelFeatureEncoder):
    """Encode the current frame, past and future points of each voxel apart from each other.

    The voxel is placed at the mean of its current frame points, and each side is described as
    in :class:`SweepSplitVoxelFeatureEncoder`. Keeping past and future apart preserves motion
    that one mean over both would cancel. A voxel with no current frame points takes the
    position and time lag of its past points, or of its future points when it has no past ones.
    """

    out_channels = POINT_CHANNELS + 2 * SIDE_CHANNELS

    def check_points(self, voxels: torch.Tensor, time_lag_column: int) -> None:
        """Check that the batch points carry their time lag.

        Args:
            voxels: Padded voxel points of shape ``(num_voxels, max_points, 5)``.
            time_lag_column: Point column holding the time lag, ``-1`` when the points have none.

        Raises:
            ValueError: If the time lag is in another column or missing.
        """
        self._check_time_lag_column(time_lag_column)

    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """Reduce padded voxel points to one feature vector per voxel.

        Args:
            voxels: Padded voxel points of shape ``(num_voxels, max_points, 5)``.
            num_points: Number of valid points per voxel of shape ``(num_voxels,)``.

        Returns:
            Voxel features of shape ``(num_voxels, 17)``, the position, intensity and time lag
            of the voxel followed by the offset, intensity, share and time lag of its past
            points and then the same four for its future points.
        """
        current, past, future = self._point_masks(voxels, num_points)
        current_count, current_mean = self._summarize(voxels, current)
        past_count, past_mean = self._summarize(voxels, past)
        future_count, future_mean = self._summarize(voxels, future)
        core = torch.where(
            current_count > 0,
            current_mean,
            torch.where(past_count > 0, past_mean, future_mean),
        )
        total_count = (current_count + past_count + future_count).clamp(min=1.0)
        return torch.cat(
            [
                core,
                self._describe(core, past_mean, past_count, total_count),
                self._describe(core, future_mean, future_count, total_count),
            ],
            dim=1,
        )
