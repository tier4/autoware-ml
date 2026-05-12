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

"""Point-cloud sampling and subsampling transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from autoware_ml.transforms.base import BaseTransform


class PointShuffle(BaseTransform):
    """Randomly permute points and aligned point-wise arrays."""

    _required_keys = ["points"]

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Shuffle points and aligned arrays with one shared permutation."""
        points = input_dict["points"]
        permutation = np.random.permutation(points.shape[0])
        for key, value in list(input_dict.items()):
            if (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == points.shape[0]
            ):
                input_dict[key] = value[permutation]
        return input_dict


class RandomDropout(BaseTransform):
    """Randomly remove points while keeping aligned point-wise arrays consistent."""

    _required_keys = ["coord"]

    def __init__(self, dropout_ratio: float = 0.2, dropout_application_ratio: float = 0.5) -> None:
        """Initialize the random dropout transform.

        Args:
            dropout_ratio: Fraction of points removed when dropout is applied.
            dropout_application_ratio: Probability of applying the transform.
        """
        self.dropout_ratio = dropout_ratio
        self.p = dropout_application_ratio

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Randomly drop a subset of points.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        point_count = input_dict["coord"].shape[0]
        keep_count = max(1, int(point_count * (1 - self.dropout_ratio)))
        keep_indices = np.sort(np.random.choice(point_count, keep_count, replace=False))

        for key, value in list(input_dict.items()):
            if isinstance(value, np.ndarray) and value.shape[0] == point_count:
                input_dict[key] = value[keep_indices]
        return input_dict


class ElasticDistortion(BaseTransform):
    """Apply elastic distortion to point coordinates."""

    _required_keys = ["coord"]

    def __init__(self, distortion_params: Sequence[Sequence[float]]) -> None:
        """Initialize the elastic distortion transform.

        Args:
            distortion_params: Sequence of ``[granularity, magnitude]`` pairs.
        """
        self.distortion_params = [tuple(pair) for pair in distortion_params]

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply one or more elastic distortion stages.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        coord = input_dict["coord"]
        for granularity, magnitude in self.distortion_params:
            coord = self._elastic(coord, granularity, magnitude)
        input_dict["coord"] = coord
        return input_dict

    def _elastic(
        self, coords: npt.NDArray[np.float32], granularity: float, magnitude: float
    ) -> npt.NDArray[np.float32]:
        """Apply one elastic distortion stage to point coordinates."""
        blur_x = np.ones((3, 1, 1, 1), dtype=np.float32) / 3
        blur_y = np.ones((1, 3, 1, 1), dtype=np.float32) / 3
        blur_z = np.ones((1, 1, 3, 1), dtype=np.float32) / 3

        coords_min = coords.min(axis=0) - granularity * 3
        coords_max = coords.max(axis=0) + granularity * 3
        noise_dim = ((coords_max - coords_min) / granularity).astype(int) + 1
        noise = np.random.randn(noise_dim[0], noise_dim[1], noise_dim[2], 3).astype(np.float32)

        for _ in range(2):
            noise = ndimage.convolve(noise, blur_x, mode="constant", cval=0)
            noise = ndimage.convolve(noise, blur_y, mode="constant", cval=0)
            noise = ndimage.convolve(noise, blur_z, mode="constant", cval=0)

        axes = [
            np.linspace(coords_min[index], coords_max[index], noise_dim[index], dtype=np.float32)
            for index in range(3)
        ]
        interpolated = np.stack(
            [_trilinear_interpolate(axes, noise[..., channel], coords) for channel in range(3)],
            axis=1,
        )
        return coords + interpolated * magnitude


class GridSample(BaseTransform):
    """Subsample points by selecting representatives per grid cell."""

    _required_keys = ["coord"]

    def __init__(
        self,
        grid_size: float,
        mode: str,
        keys: Sequence[str],
        hash_type: str = "fnv",
        return_grid_coord: bool = False,
        return_inverse: bool = False,
        return_min_coord: bool = False,
        return_displacement: bool = False,
        project_displacement: bool = False,
    ) -> None:
        """Initialize the grid sampling transform.

        Args:
            grid_size: Grid cell size used for subsampling.
            hash_type: Hash type used to group voxel coordinates.
            mode: Sampling mode. ``train`` picks a random representative per
                voxel. ``test`` picks a deterministic representative per voxel.
            keys: Sample keys subsampled together with coordinates.
            return_grid_coord: Whether to expose sampled grid coordinates.
            return_inverse: Whether to expose inverse voxel indices.
            return_min_coord: Whether to expose the voxelized minimum
                coordinate in world space.
            return_displacement: Whether to expose per-point displacement to the
                voxel center.
            project_displacement: Whether to project displacement to normals.
        """
        self.grid_size = np.asarray(grid_size, dtype=np.float32)
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        if mode not in {"train", "test"}:
            raise ValueError(f"Unsupported GridSample mode: {mode}")
        self.mode = mode
        self.keys = tuple(keys)
        self.return_grid_coord = return_grid_coord
        self.return_inverse = return_inverse
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Subsample points by selecting representatives per grid cell.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        scaled_coord = input_dict["coord"].astype(np.float32) / self.grid_size.astype(np.float32)
        grid_coord = np.floor(scaled_coord).astype(np.int64)
        min_coord = grid_coord.min(axis=0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord_world = min_coord.astype(np.float32) * self.grid_size.astype(np.float32)
        key = self.hash(grid_coord)
        sort_indices = np.argsort(key)
        sorted_key = key[sort_indices]
        _, inverse, counts = np.unique(sorted_key, return_inverse=True, return_counts=True)
        if self.mode == "train":
            return self._transform_train(
                input_dict,
                grid_coord,
                scaled_coord,
                min_coord_world,
                sort_indices,
                inverse,
                counts,
            )
        return self._transform_test(
            input_dict,
            grid_coord,
            scaled_coord,
            min_coord_world,
            sort_indices,
            inverse,
            counts,
        )

    def _transform_train(
        self,
        input_dict: dict[str, Any],
        grid_coord: np.ndarray,
        scaled_coord: np.ndarray,
        min_coord_world: np.ndarray,
        sort_indices: np.ndarray,
        inverse: np.ndarray,
        counts: np.ndarray,
    ) -> dict[str, Any]:
        """Apply train-time voxel subsampling."""
        selection = np.cumsum(np.insert(counts, 0, 0)[:-1]) + (
            np.random.randint(0, counts.max(), counts.size) % counts
        )
        unique_indices = sort_indices[selection]

        if "sampled_index" in input_dict:
            unique_indices = np.unique(np.append(unique_indices, input_dict["sampled_index"]))
            mask = np.zeros_like(input_dict["segment"]).astype(bool)
            mask[input_dict["sampled_index"]] = True
            input_dict["sampled_index"] = np.where(mask[unique_indices])[0]

        if self.return_inverse:
            input_dict["inverse"] = np.zeros_like(inverse)
            input_dict["inverse"][sort_indices] = inverse
        if self.return_grid_coord:
            input_dict["grid_coord"] = grid_coord[unique_indices].astype(np.int32)
        if self.return_min_coord:
            input_dict["min_coord"] = min_coord_world.reshape(1, 3)
        if self.return_displacement:
            displacement = scaled_coord - grid_coord - 0.5
            if self.project_displacement:
                displacement = np.sum(displacement * input_dict["normal"], axis=-1, keepdims=True)
            input_dict["displacement"] = displacement[unique_indices]
        for key in self.keys:
            if key in input_dict:
                input_dict[key] = input_dict[key][unique_indices]
        return input_dict

    def _transform_test(
        self,
        input_dict: dict[str, Any],
        grid_coord: np.ndarray,
        scaled_coord: np.ndarray,
        min_coord_world: np.ndarray,
        sort_indices: np.ndarray,
        inverse: np.ndarray,
        counts: np.ndarray,
    ) -> dict[str, Any]:
        """Apply deterministic voxel subsampling for evaluation and deployment."""
        selection = np.cumsum(np.insert(counts, 0, 0)[:-1])
        unique_indices = sort_indices[selection]

        if self.return_inverse:
            input_dict["inverse"] = np.zeros_like(inverse)
            input_dict["inverse"][sort_indices] = inverse
        if self.return_grid_coord:
            input_dict["grid_coord"] = grid_coord[unique_indices].astype(np.int32)
        if self.return_min_coord:
            input_dict["min_coord"] = min_coord_world.reshape(1, 3)
        if self.return_displacement:
            displacement = scaled_coord - grid_coord - 0.5
            if self.project_displacement:
                displacement = np.sum(displacement * input_dict["normal"], axis=-1, keepdims=True)
            input_dict["displacement"] = displacement[unique_indices]
        for key in self.keys:
            if key in input_dict:
                input_dict[key] = input_dict[key][unique_indices]
        return input_dict

    @staticmethod
    def ravel_hash_vec(arr: np.ndarray) -> np.ndarray:
        """Hash integer coordinates with a dense raveled indexing scheme."""
        if arr.ndim != 2:
            raise ValueError("GridSample hashing expects a 2D coordinate array.")
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1
        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        for axis_index in range(arr.shape[1] - 1):
            keys += arr[:, axis_index]
            keys *= arr_max[axis_index + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr: np.ndarray) -> np.ndarray:
        """Hash integer coordinates with FNV64-1A."""
        if arr.ndim != 2:
            raise ValueError("GridSample hashing expects a 2D coordinate array.")
        arr = arr.copy().astype(np.uint64, copy=False)
        hashed = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
        for axis_index in range(arr.shape[1]):
            hashed *= np.uint64(1099511628211)
            hashed = np.bitwise_xor(hashed, arr[:, axis_index])
        return hashed


def _trilinear_interpolate(
    axes: Sequence[npt.NDArray[np.float32]],
    values: npt.NDArray[np.float32],
    coords: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Interpolate a dense 3D grid at arbitrary point coordinates."""
    x = np.interp(coords[:, 0], axes[0], np.arange(axes[0].size))
    y = np.interp(coords[:, 1], axes[1], np.arange(axes[1].size))
    z = np.interp(coords[:, 2], axes[2], np.arange(axes[2].size))

    x0 = np.clip(np.floor(x).astype(int), 0, axes[0].size - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, axes[1].size - 1)
    z0 = np.clip(np.floor(z).astype(int), 0, axes[2].size - 1)
    x1 = np.clip(x0 + 1, 0, axes[0].size - 1)
    y1 = np.clip(y0 + 1, 0, axes[1].size - 1)
    z1 = np.clip(z0 + 1, 0, axes[2].size - 1)

    xd = x - x0
    yd = y - y0
    zd = z - z0

    c000 = values[x0, y0, z0]
    c001 = values[x0, y0, z1]
    c010 = values[x0, y1, z0]
    c011 = values[x0, y1, z1]
    c100 = values[x1, y0, z0]
    c101 = values[x1, y0, z1]
    c110 = values[x1, y1, z0]
    c111 = values[x1, y1, z1]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    return c0 * (1 - zd) + c1 * zd
