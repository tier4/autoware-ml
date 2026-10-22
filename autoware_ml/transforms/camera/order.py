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

"""Camera order augmentation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from autoware_ml.dataclasses.batch.detection2d import Detection2DGTSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.base import BaseTransform


def _permute(
    values: torch.Tensor | Sequence[Any], order: Sequence[int], name: str
) -> torch.Tensor | list[Any]:
    """Reorder per-camera values along their leading dimension.

    Args:
        values: One entry per camera, stacked along the first dimension of a tensor or listed
            in a sequence.
        order: Index of the input camera every output position takes.
        name: Name of the values, for the error message.

    Returns:
        The values in the given order, a tensor for a tensor and a list for a sequence.

    Raises:
        ValueError: If the values describe another number of cameras than ``order``.
    """
    if len(values) != len(order):
        raise ValueError(
            f"{name} describes {len(values)} cameras but the camera image data {len(order)}, "
            "so the cameras cannot be reordered consistently."
        )
    if isinstance(values, torch.Tensor):
        return values[list(order)]
    return [values[index] for index in order]


class ShuffleCameraOrder(BaseTransform):
    """Serve the cameras of a sample in a random order.

    A train-time regularization for models whose output does not depend on the order of the
    cameras, such as StreamPETR, whose reference recipe shuffles the cameras of every training
    frame. One permutation is drawn per sample from ``np.random``, like the other
    augmentations, and every per-camera field of the sample follows it: every field of the
    camera image data (``PER_CAMERA_FIELDS``), the image records, and the per-camera 2D ground
    truth when the sample already carries it. A camera therefore keeps its pixels, its
    calibration, its augmentation and its 2D boxes, so the transform can run anywhere after
    the images are loaded, before or after the image-space augmentations and the projection of
    the 2D annotations.

    Required keys:
        - camera_image_data: the loaded images of every camera and their calibration.
    """

    _required_keys = ["camera_image_data"]

    # Every field of BaseImages holds one entry per camera, in camera order.
    PER_CAMERA_FIELDS = (
        "images",
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
    )

    def __init__(self, probability: float | None = None) -> None:
        """Initialize the ShuffleCameraOrder transform.

        Args:
            probability: Probability of shuffling the cameras of a sample, None to shuffle
                every sample.
        """
        super().__init__(probability=probability)

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Reorder every per-camera field of the sample by one random permutation.

        Args:
            model_gt_sample: Sample holding the loaded camera images.

        Returns:
            The sample with every per-camera field in the drawn camera order.

        Raises:
            ValueError: If the image records or the 2D ground truth of the sample describe
                another number of cameras than the camera image data.
        """
        # This is checked in the _validate_required_keys()
        camera_image_data: BaseImages = (
            model_gt_sample.camera_image_data  # type: ignore[reportOptionalMemberAccess]
        )
        order = np.random.permutation(camera_image_data.images.shape[0]).tolist()

        image_samples = model_gt_sample.image_samples
        if image_samples is not None:
            image_samples = _permute(image_samples, order, "image_samples")
        detection2d_gt_sample = model_gt_sample.detection2d_gt_sample
        if detection2d_gt_sample is not None:
            detection2d_gt_sample = Detection2DGTSample(
                gt_bboxes_2d=_permute(detection2d_gt_sample.gt_bboxes_2d, order, "gt_bboxes_2d"),
                gt_labels_2d=_permute(detection2d_gt_sample.gt_labels_2d, order, "gt_labels_2d"),
            )
        # model_copy does not validate what it is given, so the copy is validated explicitly.
        camera_image_data = BaseImages.model_validate(
            camera_image_data.model_copy(
                update={
                    name: _permute(getattr(camera_image_data, name), order, name)
                    for name in self.PER_CAMERA_FIELDS
                    if getattr(camera_image_data, name) is not None
                }
            )
        )
        return model_gt_sample._replace(
            image_samples=image_samples,
            camera_image_data=camera_image_data,
            detection2d_gt_sample=detection2d_gt_sample,
        )
