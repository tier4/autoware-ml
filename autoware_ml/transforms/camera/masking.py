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

"""Masking of the image regions the ego vehicle occludes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

import cv2
from jaxtyping import Bool, Float32
import numpy as np
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.base import BaseTransform

# T4 databases are released in versions, db_j6gen2_v1, db_j6gen2_v2 and so on.
_DATABASE_VERSION_SUFFIX = re.compile(r"_v\d+$")


def database_group(scene_token: str) -> str:
    """Name of the database a scene was recorded in, without its version.

    Args:
        scene_token: Scene token ``<database>/<scene_uuid>/<version>`` of the frame.

    Returns:
        The database name with its ``_v<N>`` suffix removed, e.g. ``db_j6gen2`` for a scene of
        ``db_j6gen2_v3``.
    """
    database = scene_token.split("/", 1)[0]
    return _DATABASE_VERSION_SUFFIX.sub("", database)


def normalize_polygon(polygon: Sequence[float]) -> Float32[np.ndarray, "num_vertices 2"]:
    """Validate a flat ``[x0, y0, x1, y1, ...]`` polygon in normalized image coordinates.

    Args:
        polygon: Vertices as x, y pairs, x along the image width and y along its height, both
            in ``[0, 1]``.

    Returns:
        The vertices as an array of shape ``(num_vertices, 2)``.

    Raises:
        ValueError: If the polygon has an odd number of values, fewer than three vertices or a
            value outside ``[0, 1]``.
    """
    values = np.asarray(polygon, dtype=np.float32)
    if values.ndim != 1 or values.size % 2 != 0:
        raise ValueError(f"A polygon is a flat list of x, y pairs, got {len(values)} values.")
    vertices = values.reshape(-1, 2)
    if vertices.shape[0] < 3:
        raise ValueError(f"A polygon needs at least three vertices, got {vertices.shape[0]}.")
    if np.any((vertices < 0.0) | (vertices > 1.0)):
        raise ValueError("Polygon coordinates are normalized to the image size, within [0, 1].")
    return vertices


def polygon_to_pixels(
    vertices: Float32[np.ndarray, "num_vertices 2"], height: int, width: int
) -> Float32[np.ndarray, "num_vertices 2"]:
    """Scale a normalized polygon to the pixel grid of an image.

    The normalized coordinates span the pixel centers, so 0 is the first pixel and 1 the last
    one of each axis, and every vertex is rounded to the nearest pixel.

    Args:
        vertices: Normalized vertices of shape ``(num_vertices, 2)``.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        Pixel vertices of shape ``(num_vertices, 2)``.
    """
    return np.round(vertices * np.array([width - 1, height - 1], dtype=np.float32))


def rasterize_polygon(
    vertices: Float32[np.ndarray, "num_vertices 2"], height: int, width: int
) -> Bool[torch.Tensor, "height width"]:
    """Rasterize a normalized polygon onto an image grid, see :func:`polygon_to_pixels`.

    Args:
        vertices: Normalized vertices of shape ``(num_vertices, 2)``.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        Mask of the pixels inside the polygon, its boundary included.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_to_pixels(vertices, height, width).astype(np.int32)], 1)
    return torch.from_numpy(mask.astype(bool))


class EgoVehicleRegions:
    """Regions of the raw camera images where the ego vehicle occludes the view.

    A camera mounted on the vehicle sees parts of the vehicle itself, its body, mirrors or
    sensor housings, always at the same place in the image. Every vehicle platform has its own
    regions, so they are configured per database and per camera, as polygons in normalized
    raw-image coordinates. A database is named either with its version (``db_j6gen2_v11``), for
    that version alone, or without it (``db_j6gen2``), for every version, see
    :func:`database_group`. The versioned name replaces the unversioned one as a whole, its
    cameras without a polygon are not masked, as the vehicle behind a database can change between
    versions.

    The database of a frame is read from its scene token. A frame of a database without polygons,
    a camera without a polygon and a frame without frame metadata have no region, so a dataset
    that mixes vehicle platforms only covers the platforms that are configured.
    """

    def __init__(self, polygons: Mapping[str, Mapping[str, Sequence[float]]]) -> None:
        """Validate the polygons.

        Args:
            polygons: For every database, with or without its version, the polygon of every
                camera, a flat ``[x0, y0, x1, y1, ...]`` list of normalized raw-image coordinates.

        Raises:
            ValueError: If a polygon is malformed, see :func:`normalize_polygon`.
        """
        self.polygons = {
            str(group): {
                str(camera): normalize_polygon(polygon) for camera, polygon in cameras.items()
            }
            for group, cameras in polygons.items()
        }
        # The polygons and the raw image size are fixed for a camera, so every mask is
        # rasterized once, keyed by database, camera and image size.
        self._masks: dict[tuple[str, str, int, int], Bool[torch.Tensor, "height width"]] = {}

    def cameras_of(
        self, model_gt_sample: ModelGTSample, user: str
    ) -> tuple[str, list[tuple[int, str]]] | None:
        """Find the cameras of a sample that have a region.

        Args:
            model_gt_sample: Sample holding the loaded camera images.
            user: Name of the calling transform, for the error message.

        Returns:
            The key of the polygons of the frame, its versioned database name or its database
            group, and the index and name of every camera of the sample with a polygon, or None
            when the frame has no region at all.

        Raises:
            ValueError: If a camera with a polygon already carries an image augmentation. The
                polygon is drawn on the raw image, so it would cover the wrong pixels. A padding
                leaves the augmentation matrix as it is, so a padded image is not detected: the
                transform must run before the padding too.
        """
        if model_gt_sample.frame_meta is None or model_gt_sample.camera_image_data is None:
            return None
        scene_token = model_gt_sample.frame_meta.scene_token
        database = scene_token.split("/", 1)[0]
        group = database if database in self.polygons else database_group(scene_token)
        camera_polygons = self.polygons.get(group)
        if not camera_polygons:
            return None

        camera_image_data = model_gt_sample.camera_image_data
        cameras = [
            (index, name)
            for index, name in enumerate(camera_image_data.camera_names)
            if name in camera_polygons
        ]
        if not cameras:
            return None

        augmentation_matrices = camera_image_data.image_augmentation_matrices
        identity = torch.eye(
            4, dtype=augmentation_matrices.dtype, device=augmentation_matrices.device
        )
        for index, name in cameras:
            if not torch.allclose(augmentation_matrices[index], identity):
                raise ValueError(
                    f"{user} must run before any image-space transform, but camera {name!r} "
                    "already carries an image augmentation. Its ego vehicle polygon is drawn on "
                    f"the raw image, so move {user} ahead of the undistortion, resize, crop, flip, "
                    "rotation and padding in the pipeline."
                )
        return group, cameras

    def mask(
        self, group: str, camera: str, height: int, width: int
    ) -> Bool[torch.Tensor, "height width"]:
        """Return the rasterized polygon of one camera of one group.

        Args:
            group: Key of the polygons of the frame, see :meth:`cameras_of`.
            camera: Camera name.
            height: Image height in pixels.
            width: Image width in pixels.

        Returns:
            Mask of the pixels the ego vehicle occludes.
        """
        key = (group, camera, height, width)
        if key not in self._masks:
            self._masks[key] = rasterize_polygon(self.polygons[group][camera], height, width)
        return self._masks[key]


class MaskEgoVehicle(BaseTransform):
    """Paint over the image regions where the ego vehicle occludes the view of a camera.

    Every pixel inside the polygon of a camera, see :class:`EgoVehicleRegions`, is set to
    ``fill_value``. The polygons are drawn on the raw images, so the transform must run before
    any image-space transform, the undistortion and the padding included. It raises when a camera
    it masks already carries an image augmentation, but a padding, which leaves the augmentation
    matrix as it is, goes undetected. The region then follows the pixels through the later resize,
    crop, flip and rotation.

    Required keys:
        - camera_image_data: the loaded images of every camera.
    """

    _required_keys = ["camera_image_data"]

    def __init__(
        self,
        polygons: Mapping[str, Mapping[str, Sequence[float]]],
        fill_value: float | Sequence[float] = 0.0,
    ) -> None:
        """Initialize the MaskEgoVehicle transform.

        Args:
            polygons: For every database, with or without its version, the polygon of every
                camera to mask, a flat ``[x0, y0, x1, y1, ...]`` list of normalized raw-image
                coordinates, see :class:`EgoVehicleRegions`.
            fill_value: Value of the masked pixels, one for every channel or one per channel,
                in the range the images are loaded in. Zero, the default, paints them black.

        Raises:
            ValueError: If a polygon is malformed, see :func:`normalize_polygon`.
        """
        super().__init__(probability=None)
        self.regions = EgoVehicleRegions(polygons)
        self.fill_value = torch.as_tensor(fill_value, dtype=torch.float32)

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Paint the ego vehicle regions of every configured camera of the sample.

        Args:
            model_gt_sample: Sample holding the loaded camera images.

        Returns:
            The sample with the ego vehicle regions painted with ``fill_value``.

        Raises:
            ValueError: If a camera to mask already carries an image augmentation.
        """
        found = self.regions.cameras_of(model_gt_sample, type(self).__name__)
        if found is None:
            return model_gt_sample
        group, cameras = found

        # This is checked in the _validate_required_keys()
        camera_image_data: BaseImages = (
            model_gt_sample.camera_image_data  # type: ignore[reportOptionalMemberAccess]
        )
        images = camera_image_data.images.clone()
        height, width = images.shape[-2:]
        fill_value = self.fill_value.to(images.dtype).reshape(-1, 1)
        for index, name in cameras:
            images[index][:, self.regions.mask(group, name, height, width)] = fill_value

        # model_copy does not validate what it is given, so the copy is validated explicitly.
        masked_camera_image_data = BaseImages.model_validate(
            camera_image_data.model_copy(update={"images": images})
        )
        return model_gt_sample._replace(camera_image_data=masked_camera_image_data)
