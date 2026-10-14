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

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.dataclasses.geometry.range_view import RangeViewData
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.dataclasses.models.segmentation3d.predictions import Segmentation3DPredictions
from autoware_ml.metrics.segmentation3d.eval_output import (
    concat_frame_ids,
    segmentation_frames_eval_output,
)
from autoware_ml.models.module_base_model import ModuleBaseModel
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
        """Run export-time inference and return point-wise probabilities.

        Args:
            points: Concatenated point features.
            coors: Per-point frustum cell coordinates.
            voxel_coors: Active frustum cell coordinates.
            inverse_map: Point to frustum cell index mapping.

        Returns:
            Point-wise class probabilities.
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
            sample_count=1,
        )
        point_logits = self.decode_head(
            coors, point_feats_encoder, voxel_feats_pyramid, point_feats_backbone
        )
        return torch.softmax(point_logits, dim=1)


def _point_cloud_batch(batch_inputs: ModelBatchInputs) -> PointCloudGTBatch:
    """Read the point cloud of the batch.

    Raises:
        ValueError: If the batch carries no point cloud.
    """
    point_batch = batch_inputs.multi_task_gt_batch.point_cloud_gt_batch
    if point_batch is None:
        raise ValueError("FRNet needs the point cloud of the batch.")
    return point_batch


def _point_labels(batch_inputs: ModelBatchInputs) -> torch.Tensor:
    """Read the semantic label of every point of the batch.

    Raises:
        ValueError: If the batch carries no labels.
    """
    label_batch = batch_inputs.multi_task_gt_batch.segmentation3d_gt_batch
    if label_batch is None:
        raise ValueError("FRNet needs the semantic labels of the batch.")
    return label_batch.gt_semantic_masks.long()


def _range_view_data(batch_inputs: ModelBatchInputs) -> RangeViewData:
    """Read the range view projection a frustum range preprocessor added.

    Raises:
        ValueError: If the model inputs carry no range view data.
    """
    if batch_inputs.range_view_data is None:
        raise ValueError("FRNet needs the range view data of a FrustumRangePreprocessor.")
    return batch_inputs.range_view_data


class FRNet(ModuleBaseModel):
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
        **kwargs: Any,
    ) -> None:
        """Initialize FRNet.

        Args:
            voxel_encoder: Range-view feature encoder.
            backbone: FRNet backbone.
            decode_head: Main decode head.
            auxiliary_head: Optional auxiliary heads.
            **kwargs: Keyword arguments forwarded to :class:`ModuleBaseModel`.
        """
        super().__init__(**kwargs)
        self.voxel_encoder = voxel_encoder
        self.backbone = backbone
        self.decode_head = decode_head
        self.auxiliary_head = nn.ModuleList(list(auxiliary_head or []))
        self.num_classes = int(decode_head.classifier.out_features)
        self.ignore_index = int(decode_head.ignore_index)

    def extract_feat(
        self,
        feat: torch.Tensor,
        coors: torch.Tensor,
        voxel_coors: torch.Tensor,
        inverse_map: torch.Tensor,
        sample_count: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Extract multiscale features from preprocessed range-view inputs.

        Args:
            feat: Concatenated point tensor of shape
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
            feat, inverse_map, voxel_coors
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

    def forward_network(
        self,
        feat: torch.Tensor,
        coors: torch.Tensor,
        voxel_coors: torch.Tensor,
        inverse_map: torch.Tensor,
        sample_count: int,
    ) -> ModelOutputs:
        """Run the segmentation model end-to-end and return decoded outputs.

        The dynamo-traced export path uses :class:`_FRNetExportModule`, which
        is independent of this method and emits a single probability tensor.

        Args:
            feat: Concatenated point tensor.
            coors: Point-to-range-view coordinates.
            voxel_coors: Unique range-view voxel coordinates.
            inverse_map: Mapping from points to voxel indices.
            sample_count: Number of samples in the batch.

        Returns:
            Segmentation head outputs with the point-wise logits of shape
            ``(num_points, num_classes)`` and, as auxiliary features, the
            backbone voxel feature pyramid with one tensor per level. Each
            level feeds the auxiliary head whose ``feature_index`` matches it.
        """
        point_feats_encoder, voxel_feats_pyramid, point_feats_backbone = self.extract_feat(
            feat=feat,
            coors=coors,
            voxel_coors=voxel_coors,
            inverse_map=inverse_map,
            sample_count=sample_count,
        )
        point_logits = self.decode_head(
            coors, point_feats_encoder, voxel_feats_pyramid, point_feats_backbone
        )
        return ModelOutputs(
            segmentation3d_head_outputs=Segmentation3DHeadOutputs(
                logits=point_logits, auxiliary_features=tuple(voxel_feats_pyramid)
            )
        )

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run the network on the points of the batch and their range view projection.

        Args:
            multi_task_batch_inputs: Model inputs holding the point cloud and its range view data.

        Returns:
            The segmentation head outputs.
        """
        return self.forward_network(**self.forward_inputs(multi_task_batch_inputs))

    def forward_inputs(self, batch_inputs: ModelBatchInputs) -> dict[str, Any]:
        """Pick the points of the batch and their range view projection.

        Args:
            batch_inputs: Model inputs holding the point cloud and its range view data.

        Returns:
            The points, the range view cell of every point, the occupied cells, the cell index
            of every point and the sample count of the batch.
        """
        range_view_data = _range_view_data(batch_inputs)
        return {
            "feat": _point_cloud_batch(batch_inputs).points,
            "coors": range_view_data.coors,
            "voxel_coors": range_view_data.voxel_coors,
            "inverse_map": range_view_data.inverse_map,
            "sample_count": batch_inputs.batch_size(),
        }

    def compute_metrics(
        self,
        batch_inputs: ModelBatchInputs,
        outputs: ModelOutputs,
    ) -> dict[str, torch.Tensor]:
        """Compute FRNet losses and point-wise accuracy.

        Args:
            batch_inputs: Model inputs holding the point labels and the dense range view
                labels.
            outputs: Segmentation head outputs returned by :meth:`forward`, the auxiliary
                features feed the auxiliary heads.

        Returns:
            Dictionary of named loss tensors and segmentation metrics. The
            total loss is exposed under the ``"loss"`` key.

        Raises:
            ValueError: If the batch carries no labels.
        """
        semantic_seg = _range_view_data(batch_inputs).semantic_labels
        if semantic_seg is None:
            raise ValueError("FRNet losses need the labels of the batch.")
        point_logits = outputs.segmentation3d().logits
        voxel_feats = outputs.segmentation3d().auxiliary_features

        decode_losses = self.decode_head.loss(point_logits, _point_labels(batch_inputs))
        total_loss = decode_losses["loss_ce"]
        metrics: dict[str, torch.Tensor] = {"loss_decode_ce": decode_losses["loss_ce"]}

        for head_index, head in enumerate(self.auxiliary_head):
            head_losses = head.loss(voxel_feats[head.feature_index], semantic_seg)
            for loss_name, loss_value in head_losses.items():
                metrics[f"aux_{head_index}_{loss_name}"] = loss_value
                total_loss = total_loss + loss_value

        metrics["loss"] = total_loss
        return metrics

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> dict[str, Any]:
        """Pair per-frame point predictions with targets for the segmentation suites.

        FRNet's logits are already at the original point level, so each point's
        frame is its own position in the concatenated points, bucketed by the
        point offsets of the batch.

        Args:
            batch: Model inputs as fed to the model.
            outputs: Raw forward outputs.

        Returns:
            Flat dict with the per-frame ``seg_frames`` the suites read.
        """
        point_logits = outputs.segmentation3d().logits
        point_batch = _point_cloud_batch(batch)
        offset = point_batch.offsets()
        point_index = torch.arange(point_logits.shape[0], device=point_logits.device)
        return segmentation_frames_eval_output(
            coord=point_batch.points[:, :3],
            pred_labels=point_logits.argmax(dim=1),
            target_labels=_point_labels(batch),
            scores=torch.softmax(point_logits, dim=1),
            frame_ids=concat_frame_ids(offset, point_index),
            num_frames=int(offset.shape[0]),
            batch_inputs=batch,
        )

    def decode_outputs(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        """Format FRNet segmentation predictions at the point level.

        FRNet's decode head produces per-point logits directly through
        ``inverse_map``, so no voxel-to-point scatter is needed here.

        Args:
            multi_task_batch_inputs: Model inputs of the batch, unused because FRNet's logits
                are already at point level.
            outputs: Segmentation head outputs returned by :meth:`forward`.

        Returns:
            The predicted class and the class probabilities of every point.
        """
        del multi_task_batch_inputs
        pred_probs = torch.softmax(outputs.segmentation3d().logits, dim=1)
        return ModelPredictions(
            segmentation3d_predictions=Segmentation3DPredictions(
                pred_labels=pred_probs.argmax(dim=1), pred_probs=pred_probs
            )
        )

    def get_export_output_names(self) -> list[str]:
        """Return ordered FRNet export output names.

        Returns:
            The output tensor names in export order.
        """
        return ["pred_probs"]

    def build_export_spec(self, batch_inputs: ModelBatchInputs) -> ExportSpec:
        """Build the FRNet deployment export specification.

        FRNet uses an explicit export wrapper because deployment needs a copied
        module graph and a single probability tensor with a stable output name.

        Args:
            batch_inputs: Example model inputs used for tracing.

        Returns:
            The export specification.
        """
        inputs = self.forward_inputs(batch_inputs)
        input_args = tuple(inputs[name] for name in ("feat", "coors", "voxel_coors", "inverse_map"))
        return ExportSpec(
            module=_FRNetExportModule(self),
            args=input_args,
            input_param_names=["points", "coors", "voxel_coors", "inverse_map"],
            output_names=self.get_export_output_names(),
        )
