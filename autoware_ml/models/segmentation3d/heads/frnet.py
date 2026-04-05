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

from autoware_ml.losses.segmentation3d import BoundaryLoss, LovaszSoftmaxLoss
from autoware_ml.models.segmentation3d.norm import build_norm_1d
from autoware_ml.models.segmentation3d.structures import FRNetDecodedOutputs, FRNetFeatureDict


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

    def forward(self, voxel_dict: FRNetFeatureDict) -> FRNetDecodedOutputs:
        """Populate point-wise logits in the voxel dictionary.

        Args:
            voxel_dict: FRNet feature dictionary.

        Returns:
            Updated feature dictionary with point logits.
        """
        point_feats_backbone = voxel_dict["point_feats_backbone"][0]
        point_feats_pyramid = voxel_dict["point_feats"][:-1]
        voxel_feats = voxel_dict["voxel_feats"][0].permute(0, 2, 3, 1).contiguous()
        point_coors = voxel_dict["coors"]
        point_features = voxel_feats[point_coors[:, 0], point_coors[:, 1], point_coors[:, 2]]

        for layer_index, layer in enumerate(self.layers):
            point_features = layer(point_features)
            if layer_index == 0:
                point_features = point_features + point_feats_backbone
            else:
                point_features = point_features + point_feats_pyramid[-layer_index]

        voxel_dict["point_logits"] = self.classifier(point_features)
        return voxel_dict

    def loss(self, voxel_dict: FRNetFeatureDict, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute decode-head losses.

        Args:
            voxel_dict: FRNet feature dictionary with point logits.
            target: Point-wise semantic labels.

        Returns:
            Decode-head loss dictionary.
        """
        logits = voxel_dict["point_logits"]
        return {"loss_ce": self.loss_ce_weight * self.loss_ce(logits, target)}

    def predict(self, voxel_dict: FRNetFeatureDict) -> torch.Tensor:
        """Predict point-wise labels.

        Args:
            voxel_dict: FRNet feature dictionary with point logits.

        Returns:
            Point-wise semantic predictions.
        """
        return voxel_dict["point_logits"].argmax(dim=1)


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

    def forward(self, voxel_dict: FRNetFeatureDict) -> torch.Tensor:
        """Compute auxiliary frustum logits.

        Args:
            voxel_dict: FRNet feature dictionary.

        Returns:
            Auxiliary frustum logits.
        """
        return self.classifier(voxel_dict["voxel_feats"][self.feature_index])

    def loss(self, voxel_dict: FRNetFeatureDict, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute auxiliary segmentation losses.

        Args:
            voxel_dict: FRNet feature dictionary.
            target: Range-view semantic labels.

        Returns:
            Auxiliary loss dictionary.
        """
        logits = self.forward(voxel_dict)
        return {
            "loss_ce": self.loss_ce_weight * self.loss_ce(logits, target),
            "loss_lovasz": self.loss_lovasz(logits, target),
            "loss_boundary": self.loss_boundary(logits, target),
        }
