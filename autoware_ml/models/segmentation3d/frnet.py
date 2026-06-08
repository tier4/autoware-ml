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

"""FRNet components for 3D semantic segmentation.

This module contains the high-level FRNet Lightning wrapper and export logic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.metrics.segmentation3d import compute_segmentation_metrics
from autoware_ml.models.base import BaseModel
from autoware_ml.utils.deploy import ExportSpec


class _FRNetExportModule(nn.Module):
    """Expose an FRNet export graph with a single probability output."""

    def __init__(self, model: FRNet) -> None:
        """Initialize the FRNet export wrapper.

        Args:
            model: FRNet model instance.
        """
        super().__init__()
        self.voxel_encoder = deepcopy(model.voxel_encoder)
        self.backbone = deepcopy(model.backbone)
        self.decode_head = deepcopy(model.decode_head)

    def forward(
        self,
        points: torch.Tensor,
        coors: torch.Tensor,
        voxel_coors: torch.Tensor,
        inverse_map: torch.Tensor,
    ) -> torch.Tensor:
        """Run export-time inference and return point-wise probabilities."""
        voxel_coors_active, voxel_feats, point_feats_encoder = self.voxel_encoder(
            points, inverse_map, voxel_coors
        )
        voxel_feats_pyramid, point_feats_backbone = self.backbone(
            point_feats_encoder,
            voxel_feats,
            voxel_coors_active,
            coors,
            inverse_map,
            sample_count=1,
        )
        point_logits = self.decode_head(
            coors, point_feats_encoder, voxel_feats_pyramid, point_feats_backbone
        )
        return torch.softmax(point_logits, dim=1)


class FRNet(BaseModel):
    """Implement FRNet for point-wise semantic segmentation.

    The wrapper combines frustum encoding, backbone execution, decode heads,
    and Lightning training logic in one model entrypoint.
    """

    def __init__(
        self,
        voxel_encoder: nn.Module,
        backbone: nn.Module,
        decode_head: nn.Module,
        auxiliary_head: Sequence[nn.Module] | None = None,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize FRNet.

        Args:
            voxel_encoder: Range-view feature encoder.
            backbone: FRNet backbone.
            decode_head: Main decode head.
            auxiliary_head: Optional auxiliary heads.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            optimizer_group_overrides: Optional optimizer overrides keyed by
                model-defined optimizer group name.
            scheduler_config: Optional Lightning scheduler metadata such as
                ``interval`` or ``monitor``.
        """
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_group_overrides=optimizer_group_overrides,
            scheduler_config=scheduler_config,
        )
        self.voxel_encoder = voxel_encoder
        self.backbone = backbone
        self.decode_head = decode_head
        self.auxiliary_head = nn.ModuleList(list(auxiliary_head or []))
        self.num_classes = int(decode_head.classifier.out_features)
        self.ignore_index = int(decode_head.ignore_index)

    def extract_feat(
        self,
        points: torch.Tensor,
        coors: torch.Tensor,
        voxel_coors: torch.Tensor,
        inverse_map: torch.Tensor,
        sample_count: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Extract multiscale features from preprocessed range-view inputs.

        Args:
            points: Concatenated point tensor of shape
                ``(num_points, in_channels)``.
            coors: Per-point range-view coordinates of shape
                ``(num_points, 3)``.
            voxel_coors: Unique range-view voxel coordinates of shape
                ``(max_voxels, 3)``.
            inverse_map: Mapping from points to voxel indices of shape
                ``(num_points,)``.
            sample_count: Number of samples in the batch.

        Returns:
            Tuple of three feature pyramids:
                * ``point_feats_encoder``: per-point features at each MLP
                  layer of the encoder.
                * ``voxel_feats_pyramid``: backbone voxel feature pyramid.
                * ``point_feats_backbone``: backbone point feature pyramid.
        """
        voxel_coors_active, voxel_feats, point_feats_encoder = self.voxel_encoder(
            points, inverse_map, voxel_coors
        )
        voxel_feats_pyramid, point_feats_backbone = self.backbone(
            point_feats_encoder,
            voxel_feats,
            voxel_coors_active,
            coors,
            inverse_map,
            sample_count,
        )
        return point_feats_encoder, voxel_feats_pyramid, point_feats_backbone

    def forward(
        self,
        points: torch.Tensor,
        coors: torch.Tensor,
        voxel_coors: torch.Tensor,
        inverse_map: torch.Tensor,
        sample_count: int,
    ) -> tuple[torch.Tensor, ...]:
        """Run the segmentation model end-to-end and return decoded outputs.

        The dynamo-traced export path uses :class:`_FRNetExportModule`, which
        is independent of this method and emits a single probability tensor.
        Training-time consumers of this method
        (:meth:`compute_metrics`, :meth:`predict_outputs`) unpack the
        returned tuple by position.

        Args:
            points: Concatenated point tensor.
            coors: Point-to-range-view coordinates.
            voxel_coors: Unique range-view voxel coordinates.
            inverse_map: Mapping from points to voxel indices.
            sample_count: Number of samples in the batch.

        Returns:
            Tuple ``(point_logits, *voxel_feats_pyramid)`` of:
                * ``point_logits``: point-wise logits of shape
                  ``(num_points, num_classes)``.
                * ``voxel_feats_pyramid``: backbone voxel feature pyramid,
                  one tensor per pyramid level. Each entry feeds the
                  auxiliary head whose ``feature_index`` matches the
                  pyramid level.
        """
        point_feats_encoder, voxel_feats_pyramid, point_feats_backbone = self.extract_feat(
            points=points,
            coors=coors,
            voxel_coors=voxel_coors,
            inverse_map=inverse_map,
            sample_count=sample_count,
        )
        point_logits = self.decode_head(
            coors, point_feats_encoder, voxel_feats_pyramid, point_feats_backbone
        )
        return (point_logits, *voxel_feats_pyramid)

    def compute_metrics(
        self,
        batch_inputs_dict: Mapping[str, Any],
        outputs: tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        """Compute FRNet losses and point-wise accuracy.

        Args:
            batch_inputs_dict: Full batch dictionary after runtime
                preprocessing. Must contain ``pts_semantic_mask`` and
                ``semantic_seg``.
            outputs: Tuple returned by :meth:`forward`. The first element is
                ``point_logits``; the remainder is the voxel-feature
                pyramid consumed by auxiliary heads.

        Returns:
            Dictionary of named loss tensors and segmentation metrics. The
            total loss is exposed under the ``"loss"`` key.
        """
        pts_semantic_mask = batch_inputs_dict["pts_semantic_mask"]
        semantic_seg = batch_inputs_dict["semantic_seg"]
        point_logits, *voxel_feats = outputs

        decode_losses = self.decode_head.loss(point_logits, pts_semantic_mask)
        total_loss = decode_losses["loss_ce"]
        metrics: dict[str, torch.Tensor] = {"loss_decode_ce": decode_losses["loss_ce"]}

        for head_index, head in enumerate(self.auxiliary_head):
            head_losses = head.loss(voxel_feats[head.feature_index], semantic_seg)
            for loss_name, loss_value in head_losses.items():
                metrics[f"aux_{head_index}_{loss_name}"] = loss_value
                total_loss = total_loss + loss_value

        with torch.no_grad():
            predictions = point_logits.argmax(dim=1)
            metrics.update(
                compute_segmentation_metrics(
                    predictions,
                    pts_semantic_mask,
                    self.num_classes,
                    self.ignore_index,
                )
            )

        metrics["loss"] = total_loss
        return metrics

    def predict_outputs(
        self,
        batch_inputs_dict: Mapping[str, Any],
        outputs: tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        """Format FRNet segmentation predictions at the point level.

        FRNet's decode head produces per-point logits directly through
        ``inverse_map``, so no voxel-to-point scatter is needed here.

        Args:
            batch_inputs_dict: Full batch dictionary (unused; FRNet's logits
                are already at point level).
            outputs: Tuple returned by :meth:`forward`. Only the first
                element (``point_logits``) is consumed.

        Returns:
            Dictionary with ``"pred_labels"`` (predicted class indices) and
            ``"pred_probs"`` (per-class probabilities).
        """
        del batch_inputs_dict
        point_logits = outputs[0]
        pred_probs = torch.softmax(point_logits, dim=1)
        return {"pred_labels": pred_probs.argmax(dim=1), "pred_probs": pred_probs}

    def get_export_output_names(self) -> list[str]:
        """Return ordered FRNet export output names."""
        return ["pred_probs"]

    def get_log_batch_size(self, batch_inputs_dict: Mapping[str, Any]) -> int:
        """Return the number of samples represented by the FRNet batch."""
        return int(batch_inputs_dict["sample_count"])

    def build_export_spec(self, batch_inputs_dict: Mapping[str, torch.Tensor]) -> ExportSpec:
        """Build the FRNet deployment export specification.

        FRNet uses an explicit export wrapper because deployment needs a copied
        module graph and a single probability tensor with a stable output name.
        """
        input_names = ["points", "coors", "voxel_coors", "inverse_map"]
        input_args = tuple(batch_inputs_dict[name] for name in input_names)
        return ExportSpec(
            module=_FRNetExportModule(self),
            args=input_args,
            input_param_names=input_names,
            output_names=self.get_export_output_names(),
        )
