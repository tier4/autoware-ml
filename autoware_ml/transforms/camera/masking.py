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

"""Masking of the image regions the ego vehicle occludes and of the boxes it hides."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

import cv2
from jaxtyping import Bool, Float32
import numpy as np
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
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


def _edges_cross_rectangles(
    polygon: Float32[torch.Tensor, "num_vertices 2"],
    low: Float32[torch.Tensor, "num_rectangles 2"],
    high: Float32[torch.Tensor, "num_rectangles 2"],
) -> Bool[torch.Tensor, " num_rectangles"]:
    """Whether an edge of a polygon passes through the interior of each rectangle.

    Every edge is clipped to every rectangle (Liang-Barsky); an edge crosses the interior when
    the midpoint of its clipped part lies strictly inside the rectangle. An edge running along the
    border of a rectangle therefore does not cross it, and an edge of zero length crosses it when
    the vertex lies strictly inside.

    Args:
        polygon: Pixel vertices of the polygon.
        low: Top-left corner of every rectangle, in pixels.
        high: Bottom-right corner of every rectangle, in pixels.

    Returns:
        For every rectangle, whether some edge of the polygon crosses its interior.
    """
    start = polygon
    direction = polygon.roll(-1, dims=0) - polygon
    # (num_rectangles, num_edges) entry and exit parameters of every edge in every rectangle.
    enter = torch.zeros(low.shape[0], polygon.shape[0])
    leave = torch.ones(low.shape[0], polygon.shape[0])
    inside = torch.ones(low.shape[0], polygon.shape[0], dtype=torch.bool)
    for axis in range(2):
        for p, q in (
            (-direction[:, axis], start[:, axis] - low[:, axis : axis + 1]),
            (direction[:, axis], high[:, axis : axis + 1] - start[:, axis]),
        ):
            p = p.expand_as(q)
            parallel = p == 0
            inside &= ~(parallel & (q < 0))
            ratio = q / torch.where(parallel, torch.ones_like(p), p)
            enter = torch.where(~parallel & (p < 0), torch.maximum(enter, ratio), enter)
            leave = torch.where(~parallel & (p > 0), torch.minimum(leave, ratio), leave)
    inside &= enter <= leave
    midpoint = start + ((enter + leave) / 2).unsqueeze(-1) * direction
    eps = 1e-3
    strictly_inside = (
        (midpoint > low.unsqueeze(1) + eps) & (midpoint < high.unsqueeze(1) - eps)
    ).all(dim=-1)
    return (inside & strictly_inside).any(dim=1)


class BBoxesEgoVehicleFilter(BaseTransform):
    """Drop the 3D boxes the ego vehicle hides from every camera that sees them.

    A box is projected onto every camera through its corners in front of the camera, and its
    extent is clipped to the image. A camera sees the box when the clipped extent has a non-zero
    area, and the box is hidden in that camera when the clipped extent lies entirely in the ego
    vehicle region of the camera, see :class:`EgoVehicleRegions`: its four corners are in the
    region and no edge of the region's polygon crosses it, which holds for concave polygons too.
    A box that some camera
    sees and that every camera which sees it hides is dropped: with :class:`MaskEgoVehicle`
    painting those regions, no pixel of the box is left for the model. A box no camera sees is
    kept, it is not the ego vehicle that hides it.

    The regions are drawn on the raw images, so the transform must run before any image-space
    transform, like :class:`MaskEgoVehicle`, and it raises when a camera with a region already
    carries an image augmentation. Dropping the boxes before ``LoadAnnotations2DFromBoxes3D``
    also keeps them out of the 2D targets.

    Required keys:
        - camera_image_data: the loaded images of every camera and their calibration.
        - detection3d_gt_bboxes_3d: the 3D ground-truth boxes, in the lidar frame.
    """

    _required_keys = ["camera_image_data", "detection3d_gt_bboxes_3d"]

    def __init__(self, polygons: Mapping[str, Mapping[str, Sequence[float]]]) -> None:
        """Initialize the BBoxesEgoVehicleFilter transform.

        Args:
            polygons: For every database group, the polygon of every camera, a flat
                ``[x0, y0, x1, y1, ...]`` list of normalized raw-image coordinates. Pass the
                polygons :class:`MaskEgoVehicle` paints.

        Raises:
            ValueError: If a polygon is malformed, see :func:`normalize_polygon`.
        """
        super().__init__(probability=None)
        self.regions = EgoVehicleRegions(polygons)

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Drop the boxes hidden by the ego vehicle in every camera that sees them.

        Args:
            model_gt_sample: Sample holding the loaded images and the 3D boxes.

        Returns:
            The sample with the hidden boxes removed, in place.

        Raises:
            ValueError: If a camera with a region already carries an image augmentation.
        """
        # This is checked in the _validate_required_keys()
        bboxes: BaseBBoxes3D = (
            model_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        )
        if not len(bboxes):
            return model_gt_sample
        found = self.regions.cameras_of(model_gt_sample, type(self).__name__)
        if found is None:
            return model_gt_sample
        group, cameras = found
        masked_cameras = dict(cameras)

        # This is checked in the _validate_required_keys()
        camera_image_data: BaseImages = (
            model_gt_sample.camera_image_data  # type: ignore[reportOptionalMemberAccess]
        )
        height, width = camera_image_data.images.shape[-2:]
        # Pixel centers run from 0 to size - 1, the grid the regions are rasterized on.
        last_pixel = torch.tensor([width - 1, height - 1], dtype=torch.float32)
        corners = bboxes.corners.to(torch.float32)
        num_bboxes = corners.shape[0]
        homogeneous = torch.cat([corners.reshape(-1, 3), corners.new_ones(num_bboxes * 8, 1)], 1)

        seen = torch.zeros(num_bboxes, dtype=torch.bool)
        seen_unhidden = torch.zeros(num_bboxes, dtype=torch.bool)
        for index, lidar2image in enumerate(camera_image_data.lidar2images):
            projected = (homogeneous @ lidar2image.to(torch.float32).T).view(num_bboxes, 8, 4)
            in_front = projected[..., 2:3] > 0
            pixels = projected[..., :2] / projected[..., 2:3].clamp(min=1e-6)
            low = torch.where(in_front, pixels, torch.inf).amin(dim=1)
            high = torch.where(in_front, pixels, -torch.inf).amax(dim=1)
            low = torch.minimum(low.clamp(min=0.0), last_pixel)
            high = torch.minimum(high.clamp(min=0.0), last_pixel)
            sees = in_front.any(dim=1).squeeze(1) & (low < high).all(dim=1)
            seen |= sees

            if index not in masked_cameras:
                seen_unhidden |= sees
                continue
            mask = self.regions.mask(group, masked_cameras[index], height, width)
            # The four corners of the clipped extent, as pixel indices of the region mask.
            rectangle = torch.stack(
                [
                    torch.stack([low[:, 0], low[:, 1]], dim=1),
                    torch.stack([high[:, 0], low[:, 1]], dim=1),
                    torch.stack([high[:, 0], high[:, 1]], dim=1),
                    torch.stack([low[:, 0], high[:, 1]], dim=1),
                ],
                dim=1,
            )
            rectangle = rectangle.round().long()
            hidden = mask[rectangle[..., 1], rectangle[..., 0]].all(dim=1)
            # The corners alone do not decide a concave region, whose boundary can pass between
            # them: such a box is still partly visible.
            if hidden.any():
                polygon = torch.from_numpy(
                    polygon_to_pixels(
                        self.regions.polygons[group][masked_cameras[index]], height, width
                    )
                )
                hidden[hidden.clone()] = ~_edges_cross_rectangles(
                    polygon, low[hidden], high[hidden]
                )
            seen_unhidden |= sees & ~hidden

        bboxes.remove_bboxes(~seen | seen_unhidden)
        return model_gt_sample
