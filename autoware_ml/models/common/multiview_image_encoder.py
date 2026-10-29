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

"""Multiview image encoder shared by the camera detectors.

The camera branch of BEVFusion and StreamPETR run one image backbone and neck on every camera
of a ``(B, N, C, H, W)`` batch and keep the first neck output of every camera; Sparse4D keeps
every level of the neck's pyramid.
"""

from __future__ import annotations

from jaxtyping import Float32
import torch
import torch.nn as nn


class MultiviewImageEncoder(nn.Module):
    """Image backbone and neck run on every camera of a multiview batch.

    The submodules keep the ``img_backbone`` / ``img_neck`` names, so the checkpoint keys of a
    model holding the encoder as ``<attribute>`` are ``<attribute>.img_backbone.*`` and
    ``<attribute>.img_neck.*``.
    """

    def __init__(
        self, img_backbone: nn.Module, img_neck: nn.Module, return_all_levels: bool = False
    ) -> None:
        """Initialize the multiview image encoder.

        Args:
            img_backbone: Backbone run on every camera image.
            img_neck: Neck fusing the backbone's feature maps; its first output is the primary
                feature.
            return_all_levels: Whether the encoder returns every level of the neck instead
                of the primary feature alone.
        """
        super().__init__()
        self.img_backbone = img_backbone
        self.img_neck = img_neck
        self.return_all_levels = return_all_levels

    @staticmethod
    def encode_pyramid(
        img_backbone: nn.Module,
        img_neck: nn.Module,
        image_batch: Float32[torch.Tensor, "batch_size num_cams 3 height width"],
    ) -> tuple[
        Float32[torch.Tensor, "batch_size num_cams channels feature_height feature_width"], ...
    ]:
        """Encode every camera image and fold every neck level back per camera.

        The backbones and necks of the shipped configs return tuples of feature maps. A
        backbone that returns one tensor is handed to the neck as a one-map tuple, and a
        neck that returns one tensor provides a one-level pyramid. A module that holds its
        backbone and neck under its own names, as the BEVFusion camera branch does for its
        checkpoint keys, calls this directly.

        Args:
            img_backbone: Backbone run on every camera image.
            img_neck: Neck fusing the backbone's feature maps.
            image_batch: Multiview images ``(B, N, C, H, W)``.

        Returns:
            Every neck level, from the first, as ``(B, N, C', H', W')`` tensors.

        Raises:
            ValueError: If the images are not a ``(B, N, C, H, W)`` tensor.
        """
        if image_batch.dim() != 5:
            raise ValueError(
                f"Expected image batch with shape (B, N, C, H, W), got {tuple(image_batch.shape)}"
            )
        batch_size, num_cams, channels, image_height, image_width = image_batch.shape
        flat_images = image_batch.view(batch_size * num_cams, channels, image_height, image_width)
        image_features = img_backbone(flat_images)
        if isinstance(image_features, torch.Tensor):
            image_features = (image_features,)
        image_features = img_neck(image_features)
        if isinstance(image_features, torch.Tensor):
            image_features = (image_features,)
        return tuple(
            level.view(batch_size, num_cams, level.shape[1], level.shape[2], level.shape[3])
            for level in image_features
        )

    def forward(
        self, image_batch: Float32[torch.Tensor, "batch_size num_cams 3 height width"]
    ) -> (
        Float32[torch.Tensor, "batch_size num_cams channels feature_height feature_width"]
        | tuple[
            Float32[torch.Tensor, "batch_size num_cams channels feature_height feature_width"], ...
        ]
    ):
        """Encode multiview images, see :meth:`encode_pyramid`.

        Args:
            image_batch: Multiview images ``(B, N, C, H, W)``.

        Returns:
            The primary neck feature of every camera, ``(B, N, C', H', W')``, which is the
            first neck level, or every neck level with ``return_all_levels``.
        """
        levels = self.encode_pyramid(self.img_backbone, self.img_neck, image_batch)
        return levels if self.return_all_levels else levels[0]
