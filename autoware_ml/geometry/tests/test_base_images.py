# Copyright 2025 TIER IV, Inc.
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

"""Unit tests for the BaseImages container."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError
import pytest
import torch

from autoware_ml.geometry.cameras.base_images import BaseImages

NUM_CAMERAS = 3


def _valid_fields(num_cameras: int = NUM_CAMERAS) -> dict[str, Any]:
    """Build a consistent set of constructor arguments for ``num_cameras`` cameras."""
    intrinsics = torch.eye(3).repeat(num_cameras, 1, 1)
    extrinsics = torch.eye(4).repeat(num_cameras, 1, 1)
    return {
        "images": torch.zeros(num_cameras, 3, 8, 8),
        "depth_maps": torch.zeros(num_cameras, 1, 8, 8),
        "timestamps": torch.zeros(num_cameras, dtype=torch.float64),
        "camera_intrinsics": intrinsics,
        "camera_names": [f"camera{i}" for i in range(num_cameras)],
        "lidar2images": extrinsics.clone(),
        "lidar2cams": extrinsics.clone(),
        "distortion_models": ["plumb_bob"] * num_cameras,
        "distortion_coefficients": [torch.zeros(5) for _ in range(num_cameras)],
        "augmented_camera_intrinsics": intrinsics.clone(),
        "image_augmentation_matrices": extrinsics.clone(),
        "noises": extrinsics.clone(),
        "calibration_statuses": torch.zeros(num_cameras, dtype=torch.int64),
    }


def test_consistent_num_cameras_is_accepted() -> None:
    images = BaseImages(**_valid_fields())
    assert images.images.shape[0] == NUM_CAMERAS


def test_optional_fields_left_none_are_skipped() -> None:
    fields = _valid_fields()
    fields.update(depth_maps=None, noises=None, calibration_statuses=None)
    images = BaseImages(**fields)
    assert images.depth_maps is None


@pytest.mark.parametrize(
    "field_name",
    [
        "depth_maps",
        "timestamps",
        "camera_intrinsics",
        "camera_names",
        "lidar2images",
        "lidar2cams",
        "distortion_models",
        "distortion_coefficients",
        "augmented_camera_intrinsics",
        "image_augmentation_matrices",
        "noises",
        "calibration_statuses",
    ],
)
def test_inconsistent_num_cameras_is_rejected(field_name: str) -> None:
    fields = _valid_fields()
    fields[field_name] = _valid_fields(NUM_CAMERAS + 1)[field_name]
    with pytest.raises(ValidationError, match=f"Inconsistent number of cameras.*{field_name}=4"):
        BaseImages(**fields)


def test_error_message_lists_every_mismatch() -> None:
    fields = _valid_fields()
    fields["timestamps"] = torch.zeros(NUM_CAMERAS + 1, dtype=torch.float64)
    fields["camera_names"] = ["only_one"]
    with pytest.raises(ValidationError, match="timestamps=4, camera_names=1"):
        BaseImages(**fields)


def test_depth_maps_with_the_image_size_are_accepted() -> None:
    fields = _valid_fields()
    fields["depth_maps"] = torch.zeros(NUM_CAMERAS, 2, 8, 8)
    images = BaseImages.model_validate(fields)
    assert images.depth_maps is not None
    assert images.depth_maps.shape[-2:] == images.images.shape[-2:]


def test_depth_maps_of_another_size_are_rejected() -> None:
    fields = _valid_fields()
    fields["depth_maps"] = torch.zeros(NUM_CAMERAS, 1, 4, 8)
    with pytest.raises(ValidationError, match=r"images are \(8, 8\).*but depth_maps are \(4, 8\)"):
        BaseImages.model_validate(fields)


def test_model_validate_recheck_catches_resized_images_over_stale_depth_maps() -> None:
    images = BaseImages.model_validate(_valid_fields())
    resized = images.model_copy(update={"images": torch.zeros(NUM_CAMERAS, 3, 4, 4)})
    with pytest.raises(ValidationError, match="depth_maps are"):
        BaseImages.model_validate(resized)


def test_model_validate_recheck_catches_inconsistent_model_copy() -> None:
    images = BaseImages(**_valid_fields())
    broken = images.model_copy(
        update={"timestamps": torch.zeros(NUM_CAMERAS + 1, dtype=torch.float64)}
    )
    with pytest.raises(ValidationError, match="timestamps=4"):
        BaseImages.model_validate(broken)
