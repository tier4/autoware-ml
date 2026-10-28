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
of a ``(B, N, C, H, W)`` batch and keep the first neck output of every camera.
"""

from __future__ import annotations

from jaxtyping import Float32
import torch
import torch.nn as nn


def encode_multiview_images(
    img_backbone: nn.Module,
    img_neck: nn.Module,
    image_batch: Float32[torch.Tensor, "batch_size num_cams 3 height width"],
) -> Float32[torch.Tensor, "batch_size num_cams channels feature_height feature_width"]:
    """Encode every camera image and fold the primary neck feature back per camera.

    The backbones and necks of the shipped configs return tuples of feature maps. A backbone
    that returns one tensor is handed to the neck as a one-map tuple, and a neck that returns
    one tensor provides the primary feature itself.

    Args:
        img_backbone: Backbone run on every camera image.
        img_neck: Neck fusing the backbone's feature maps; its first output is the primary
            feature.
        image_batch: Multiview images ``(B, N, C, H, W)``.

    Returns:
        The primary neck feature of every camera, ``(B, N, C', H', W')``.

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
    primary_feature = (
        image_features[0] if isinstance(image_features, (list, tuple)) else image_features
    )
    _, feature_channels, feature_height, feature_width = primary_feature.shape
    return primary_feature.view(
        batch_size, num_cams, feature_channels, feature_height, feature_width
    )


class MultiviewImageEncoder(nn.Module):
    """Image backbone and neck run on every camera of a multiview batch.

    The submodules keep the ``img_backbone`` / ``img_neck`` names, so the checkpoint keys of a
    model holding the encoder as ``<attribute>`` are ``<attribute>.img_backbone.*`` and
    ``<attribute>.img_neck.*``.
    """

    def __init__(self, img_backbone: nn.Module, img_neck: nn.Module) -> None:
        """Initialize the multiview image encoder.

        Args:
            img_backbone: Backbone run on every camera image.
            img_neck: Neck fusing the backbone's feature maps; its first output is the primary
                feature.
        """
        super().__init__()
        self.img_backbone = img_backbone
        self.img_neck = img_neck

    def forward(
        self, image_batch: Float32[torch.Tensor, "batch_size num_cams 3 height width"]
    ) -> Float32[torch.Tensor, "batch_size num_cams channels feature_height feature_width"]:
        """Encode multiview images, see :func:`encode_multiview_images`.

        Args:
            image_batch: Multiview images ``(B, N, C, H, W)``.

        Returns:
            The primary neck feature of every camera, ``(B, N, C', H', W')``.
        """
        return encode_multiview_images(self.img_backbone, self.img_neck, image_batch)
