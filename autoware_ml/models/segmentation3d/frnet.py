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
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.models.segmentation3d.base import BaseSegmentationModel
from autoware_ml.models.segmentation3d.structures import (
    FRNetDecodedOutputs,
    FRNetFeatureDict,
    FRNetInputs,
)
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
        voxel_dict: FRNetInputs = {
            "points": points,
            "coors": coors,
            "voxel_coors": voxel_coors,
            "inverse_map": inverse_map,
            "sample_count": 1,
        }
        voxel_dict = self.voxel_encoder(voxel_dict)
        voxel_dict = self.backbone(voxel_dict)
        point_logits = self.decode_head(voxel_dict)["point_logits"]
        return torch.softmax(point_logits, dim=1)


@dataclass(frozen=True)
class FRNetStepOutputs:
    """Internal FRNet step outputs used for training and auxiliary losses."""

    point_logits: torch.Tensor
    feature_dict: FRNetDecodedOutputs


class FRNet(BaseSegmentationModel):
    """Implement FRNet for point-wise semantic segmentation.

    The wrapper combines frustum encoding, backbone execution, decode heads,
    and Lightning training logic in one model entrypoint.
    """

    # FRNet exports only probabilities (labels are derived client-side).
    EXPORT_OUTPUT_NAMES = ("pred_probs",)

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

    def _build_feature_dict(
        self,
        points: torch.Tensor,
        coors: torch.Tensor,
        voxel_coors: torch.Tensor,
        inverse_map: torch.Tensor,
        sample_count: int,
    ) -> FRNetInputs:
        """Assemble the required FRNet input feature dictionary."""
        feature_dict: FRNetInputs = {
            "points": points,
            "coors": coors,
            "voxel_coors": voxel_coors,
            "inverse_map": inverse_map,
            "sample_count": sample_count,
        }
        return feature_dict

    def extract_feat(
        self,
        points: torch.Tensor,
        coors: torch.Tensor,
        voxel_coors: torch.Tensor,
        inverse_map: torch.Tensor,
        sample_count: int,
    ) -> FRNetFeatureDict:
        """Extract multiscale features from preprocessed range-view inputs.

        Args:
            points: Concatenated point tensor.
            coors: Point-to-range-view coordinates.
            voxel_coors: Unique range-view voxel coordinates.
            inverse_map: Mapping from voxels back to points.

        Returns:
            Feature dictionary enriched by the encoder and backbone.
        """
        voxel_dict = self._build_feature_dict(
            points=points,
            coors=coors,
            voxel_coors=voxel_coors,
            inverse_map=inverse_map,
            sample_count=sample_count,
        )
        voxel_dict = self.voxel_encoder(voxel_dict)
        voxel_dict = self.backbone(voxel_dict)
        return voxel_dict

    def _decode_outputs(self, feature_dict: FRNetFeatureDict) -> FRNetStepOutputs:
        """Decode backbone features into point-wise logits and step outputs."""
        decoded_feature_dict = cast(FRNetDecodedOutputs, self.decode_head(dict(feature_dict)))
        return FRNetStepOutputs(
            point_logits=decoded_feature_dict["point_logits"],
            feature_dict=decoded_feature_dict,
        )

    def _forward_impl(
        self,
        points: torch.Tensor,
        coors: torch.Tensor,
        voxel_coors: torch.Tensor,
        inverse_map: torch.Tensor,
        sample_count: int,
    ) -> FRNetStepOutputs:
        """Run FRNet and return decoded step outputs."""
        feature_dict = self.extract_feat(
            points=points,
            coors=coors,
            voxel_coors=voxel_coors,
            inverse_map=inverse_map,
            sample_count=sample_count,
        )
        return self._decode_outputs(feature_dict)

    def forward(
        self,
        points: torch.Tensor,
        coors: torch.Tensor,
        voxel_coors: torch.Tensor,
        inverse_map: torch.Tensor,
    ) -> torch.Tensor:
        """Run the segmentation model and decode head.

        Args:
            points: Concatenated point tensor.
            coors: Point-to-range-view coordinates.
            voxel_coors: Unique range-view voxel coordinates.
            inverse_map: Mapping from voxels back to points.

        Returns:
            Point-wise segmentation logits.
        """
        return self._forward_impl(
            points=points,
            coors=coors,
            voxel_coors=voxel_coors,
            inverse_map=inverse_map,
            sample_count=self._infer_sample_count_from_coors(coors),
        ).point_logits

    @staticmethod
    def _infer_sample_count_from_coors(coors: torch.Tensor) -> int:
        """Infer the number of samples represented by batched coordinates."""
        if coors.numel() == 0:
            return 1
        return int(coors[:, 0].amax().item()) + 1

    def _resolve_sample_count_from_batch(self, batch_inputs_dict: Mapping[str, Any]) -> int:
        """Resolve the sample count used by FRNet internal feature transport."""
        semantic_seg = batch_inputs_dict.get("semantic_seg")
        if isinstance(semantic_seg, torch.Tensor) and semantic_seg.dim() > 0:
            return int(semantic_seg.shape[0])

        sample_count = batch_inputs_dict.get("sample_count")
        if isinstance(sample_count, int):
            return sample_count
        if isinstance(sample_count, torch.Tensor) and sample_count.numel() == 1:
            return int(sample_count.item())

        coors = batch_inputs_dict.get("coors")
        if isinstance(coors, torch.Tensor):
            return self._infer_sample_count_from_coors(coors)
        return 1

    def run_model(self, batch_inputs_dict: Mapping[str, Any]) -> FRNetStepOutputs:
        """Run FRNet on a preprocessed batch."""
        return self._forward_impl(
            points=batch_inputs_dict["points"],
            coors=batch_inputs_dict["coors"],
            voxel_coors=batch_inputs_dict["voxel_coors"],
            inverse_map=batch_inputs_dict["inverse_map"],
            sample_count=self._resolve_sample_count_from_batch(batch_inputs_dict),
        )

    def compute_metrics(
        self,
        outputs: torch.Tensor | FRNetStepOutputs,
        pts_semantic_mask: torch.Tensor,
        semantic_seg: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute FRNet losses and point-wise accuracy."""
        if isinstance(outputs, FRNetStepOutputs):
            point_logits = outputs.point_logits
            feature_dict = outputs.feature_dict
        else:
            point_logits = outputs
            feature_dict = None

        decode_losses = self.decode_head.loss({"point_logits": point_logits}, pts_semantic_mask)
        total_loss = decode_losses["loss_ce"]
        metrics: dict[str, torch.Tensor] = {"loss_decode_ce": decode_losses["loss_ce"]}

        if self.auxiliary_head and semantic_seg is not None and feature_dict is not None:
            for head_index, head in enumerate(self.auxiliary_head):
                head_losses = head.loss(feature_dict, semantic_seg)
                for loss_name, loss_value in head_losses.items():
                    metrics[f"aux_{head_index}_{loss_name}"] = loss_value
                    total_loss = total_loss + loss_value

        with torch.no_grad():
            metrics.update(self._compute_segmentation_metrics(point_logits, pts_semantic_mask))

        metrics["loss"] = total_loss
        return metrics

    def _get_point_logits(self, outputs: torch.Tensor | FRNetStepOutputs) -> torch.Tensor:
        """Extract point-wise logits from FRNet outputs."""
        return outputs.point_logits if isinstance(outputs, FRNetStepOutputs) else outputs

    def get_log_batch_size(self, batch_inputs_dict: Mapping[str, Any]) -> int:
        """Return the number of samples represented by the FRNet batch."""
        return self._resolve_sample_count_from_batch(batch_inputs_dict)

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
