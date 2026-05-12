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

"""FRNet segmentation heads.

This module contains the point-wise and auxiliary heads used by FRNet.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from autoware_ml.losses.segmentation3d.boundary import BoundaryLoss
from autoware_ml.losses.segmentation3d.lovasz import LovaszSoftmaxLoss
from autoware_ml.models.segmentation3d.norm import build_norm_1d


class FRHead(nn.Module):
    """Decode fused FRNet features into point-wise semantic logits.

    The head combines backbone and encoder feature pyramids to predict final
    point-level semantic classes.
    """

    def __init__(
        self,
        in_channels: int,
        middle_channels: Sequence[int],
        num_classes: int,
        ignore_index: int,
        loss_ce_weight: float,
        norm_eps: float,
        norm_momentum: float,
    ) -> None:
        """Initialize the FRNet decode head.

        Args:
            in_channels: Input feature dimension.
            middle_channels: Hidden-layer channel widths.
            num_classes: Number of semantic classes.
            ignore_index: Label ignored by the loss.
            loss_ce_weight: Weight applied to the cross-entropy loss.
            norm_eps: BatchNorm epsilon.
            norm_momentum: BatchNorm momentum.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.loss_ce_weight = loss_ce_weight

        layers = []
        current_channels = in_channels
        for hidden_channels in middle_channels:
            layers.append(
                nn.Sequential(
                    nn.Linear(current_channels, hidden_channels, bias=False),
                    build_norm_1d(hidden_channels, norm_eps, norm_momentum),
                    nn.ReLU(inplace=True),
                )
            )
            current_channels = hidden_channels
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(current_channels, num_classes)

    def forward(
        self,
        point_coors: torch.Tensor,
        point_feats_encoder: list[torch.Tensor],
        voxel_feats_backbone: list[torch.Tensor],
        point_feats_backbone: list[torch.Tensor],
    ) -> torch.Tensor:
        """Fuse encoder and backbone features into point-wise logits.

        The head samples per-point voxel features at the projected coordinates
        and adds skip connections from the backbone point features and the
        encoder point feature pyramid before the final classifier.

        Args:
            point_coors: Per-point range-view coordinates of shape
                ``(num_points, 3)``.
            point_feats_encoder: Encoder point feature pyramid. The first
                ``len(self.layers) - 1`` levels are added as skip connections
                in reverse order.
            voxel_feats_backbone: Backbone voxel feature pyramid. Only the
                first (unified) level is sampled per point.
            point_feats_backbone: Backbone point feature pyramid. Only the
                first (unified) level is added at the first decode layer.

        Returns:
            Point-wise logits tensor of shape ``(num_points, num_classes)``.
        """
        point_feats_backbone_unified = point_feats_backbone[0]
        point_feats_pyramid = point_feats_encoder[:-1]
        voxel_feats = voxel_feats_backbone[0].permute(0, 2, 3, 1).contiguous()
        point_features = voxel_feats[point_coors[:, 0], point_coors[:, 1], point_coors[:, 2]]

        for layer_index, layer in enumerate(self.layers):
            point_features = layer(point_features)
            if layer_index == 0:
                point_features = point_features + point_feats_backbone_unified
            else:
                point_features = point_features + point_feats_pyramid[-layer_index]

        return self.classifier(point_features)

    def loss(self, point_logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the decode-head cross-entropy loss.

        Args:
            point_logits: Point-wise logits of shape ``(num_points, num_classes)``.
            target: Point-wise semantic labels of shape ``(num_points,)``.

        Returns:
            Dictionary with key ``"loss_ce"`` mapped to the weighted
            cross-entropy loss tensor.
        """
        return {"loss_ce": self.loss_ce_weight * self.loss_ce(point_logits, target)}

    def predict(self, point_logits: torch.Tensor) -> torch.Tensor:
        """Reduce per-point logits to predicted class labels.

        Args:
            point_logits: Point-wise logits of shape ``(num_points, num_classes)``.

        Returns:
            Predicted class labels of shape ``(num_points,)``.
        """
        return point_logits.argmax(dim=1)


class FrustumHead(nn.Module):
    """Predict auxiliary range-view semantic logits for FRNet training.

    The head supervises intermediate range-view features with auxiliary losses
    to stabilize optimization.
    """

    def __init__(
        self,
        channels: int,
        num_classes: int,
        ignore_index: int,
        feature_index: int,
        loss_ce_weight: float,
        loss_lovasz_weight: float,
        loss_boundary_weight: float,
    ) -> None:
        """Initialize the auxiliary frustum head.

        Args:
            channels: Input feature channels.
            num_classes: Number of semantic classes.
            ignore_index: Label ignored by the losses.
            feature_index: Feature-pyramid index consumed by the head.
            loss_ce_weight: Weight applied to the cross-entropy loss.
            loss_lovasz_weight: Weight applied to the Lovasz loss.
            loss_boundary_weight: Weight applied to the boundary loss.
        """
        super().__init__()
        self.feature_index = feature_index
        self.ignore_index = ignore_index
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.loss_ce_weight = loss_ce_weight
        self.loss_lovasz = LovaszSoftmaxLoss(
            ignore_index=ignore_index, loss_weight=loss_lovasz_weight
        )
        self.loss_boundary = BoundaryLoss(
            ignore_index=ignore_index, loss_weight=loss_boundary_weight
        )

    def forward(self, voxel_feat: torch.Tensor) -> torch.Tensor:
        """Project one voxel feature pyramid level to auxiliary frustum logits.

        Args:
            voxel_feat: Voxel features for this head's pyramid level. The
                pyramid level is selected by the FRNet wrapper based on the
                head's ``feature_index`` attribute.

        Returns:
            Auxiliary range-view logits tensor.
        """
        return self.classifier(voxel_feat)

    def loss(self, voxel_feat: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute auxiliary segmentation losses for one pyramid level.

        Args:
            voxel_feat: Voxel features for this head's pyramid level.
            target: Dense range-view semantic labels.

        Returns:
            Dictionary mapping loss names (``"loss_ce"``, ``"loss_lovasz"``,
            ``"loss_boundary"``) to their respective loss tensors.
        """
        logits = self.forward(voxel_feat)
        return {
            "loss_ce": self.loss_ce_weight * self.loss_ce(logits, target),
            "loss_lovasz": self.loss_lovasz(logits, target),
            "loss_boundary": self.loss_boundary(logits, target),
        }
