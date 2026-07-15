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

"""PTv3 joint segmentation and detection model.

One shared PTv3 encoder feeds the segmentation decoder head and the detection
BEV neck. Frames without any ground-truth box contribute no detection loss
and neutral detection metric entries, so annotation sources without detection
labels (e.g. segmentation ground truth mixed in for rehearsal, or a GT
segmentation validation split) train and evaluate only the segmentation
branch. The deliberate trade-off: genuinely empty scenes also provide no
pure-background detection supervision. The same class supports training
(forward / compute_metrics) and ONNX export (build_export_spec).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.metrics.detection3d.eval_output import detection_eval_output
from autoware_ml.models.detection3d.ptv3 import PTv3DetBEVNeck, build_det_head_export_spec
from autoware_ml.models.segmentation3d.encoders.ptv3 import PointTransformerV3Encoder
from autoware_ml.models.segmentation3d.heads.ptv3 import (
    PTv3SegDecoderHead,
    segmentation_eval_output,
)
from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    _run_ptv3_encoder_export,
    build_encoder_export_spec,
    build_point_feature_dynamic_axes,
    build_ptv3_export_context,
    build_ptv3_input_dynamic_axes,
    build_seg_head_export_spec,
    build_serialized_pooling_metadata,
    flatten_serialized_pooling_inputs,
    split_block_parameters,
)
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.structures import serialize_point_cloud_batch


class PTv3SegDetModel(PTv3BaseModel):
    """PTv3 joint segmentation and detection model.

    Training: call forward() then compute_metrics().
    Export: call build_export_spec() - requires export_output_names, grid_size,
    and point_cloud_range to be provided at construction time.
    """

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        seg3d_head: PTv3SegDecoderHead,
        bev_neck: PTv3DetBEVNeck,
        bbox_head: nn.Module,
        segmentation_loss_weight: float = 1.0,
        detection_loss_weight: float = 1.0,
        export_output_names: Sequence[str] | None = None,
        grid_size: float | None = None,
        point_cloud_range: Sequence[float] | None = None,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PTv3 joint segmentation and detection model.

        Args:
            encoder: PTv3 encoder module shared by both branches.
            seg3d_head: Segmentation decoder head owning losses and the
                classifier.
            bev_neck: Detection BEV neck consuming the encoder pooling chain.
            bbox_head: Detection head producing the decoded predictions.
            segmentation_loss_weight: Weight of the segmentation loss term.
            detection_loss_weight: Weight of the detection loss term.
            export_output_names: Ordered output names used during export.
            grid_size: Voxel grid size used to derive sparse shape for export.
            point_cloud_range: Point-cloud range used to derive sparse shape
                for export.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            optimizer_group_overrides: Per-group optimizer overrides.
            scheduler_config: Lightning scheduler metadata.
            **kwargs: Keyword arguments forwarded up the MRO chain.
        """
        super().__init__(
            encoder=encoder,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_group_overrides=optimizer_group_overrides,
            scheduler_config=scheduler_config,
            **kwargs,
        )
        self.seg3d_head = seg3d_head
        self.bev_neck = bev_neck
        self.bbox_head = bbox_head
        self.segmentation_loss_weight = float(segmentation_loss_weight)
        self.detection_loss_weight = float(detection_loss_weight)
        self._export_output_names = (
            list(export_output_names) if export_output_names is not None else None
        )

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group pretrained and newly initialized joint-task parameters."""
        encoder_default_params, encoder_block_params = split_block_parameters(self.encoder)
        seg3d_head_params = [
            parameter for parameter in self.seg3d_head.parameters() if parameter.requires_grad
        ]
        det3d_branch_params = [
            parameter
            for module in (self.bev_neck, self.bbox_head)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        return {
            "encoder_default": encoder_default_params,
            "encoder_block": encoder_block_params,
            "seg3d_head": seg3d_head_params,
            "det3d_branch": det3d_branch_params,
        }

    def forward(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        grid_coord: torch.Tensor,
        offset: torch.Tensor,
    ) -> dict[str, Any]:
        """Run one shared PTv3 encoder pass and branch into both heads."""
        point = self.encoder(
            {"coord": coord, "feat": feat, "grid_coord": grid_coord, "offset": offset}
        )
        # The BEV neck must read the encoder chain before the segmentation
        # decoder: SerializedUnpooling pops the chain and overwrites parent
        # features in place.
        bev_features = self.bev_neck(point)
        seg_logits = self.seg3d_head(point)
        det_outputs = self.bbox_head(bev_features)
        return {"seg_logits": seg_logits, "det_outputs": det_outputs}

    @staticmethod
    def _detection_frame_mask(batch_inputs_dict: Mapping[str, Any]) -> torch.Tensor:
        """Return the per-frame detection supervision mask.

        Supervision is carried by the annotations themselves: a frame with no
        ground-truth box contributes no detection supervision. This includes
        genuinely empty scenes, which therefore provide no pure-background
        signal - the deliberate price of not carrying a separate flag.

        Args:
            batch_inputs_dict: Full batch dictionary with per-frame ``gt_boxes``.

        Returns:
            Boolean tensor of shape ``(batch_size,)``.
        """
        gt_boxes = batch_inputs_dict["gt_boxes"]
        return torch.tensor([boxes.shape[0] > 0 for boxes in gt_boxes], device=gt_boxes[0].device)

    @staticmethod
    def _mask_detection_outputs(
        det_outputs: Mapping[str, torch.Tensor], mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Select the flagged batch entries from every detection output tensor."""
        return {name: value[mask] for name, value in det_outputs.items()}

    @staticmethod
    def _mask_list(values: Sequence[Any], mask: torch.Tensor) -> list[Any]:
        """Select the flagged entries from a per-sample list."""
        return [value for value, flagged in zip(values, mask.tolist()) if flagged]

    def compute_metrics(
        self,
        batch_inputs_dict: Mapping[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Compute combined segmentation and detection losses.

        The detection loss runs only on frames that carry ground-truth boxes;
        on unlabeled frames, empty ground truth would turn every real object
        into a hard negative.
        """
        seg_logits = outputs["seg_logits"]
        det_outputs = outputs["det_outputs"]
        seg_metrics = self.seg3d_head.loss(seg_logits, batch_inputs_dict["segment"])

        det_mask = self._detection_frame_mask(batch_inputs_dict)
        if bool(det_mask.any()):
            det_metrics = self.bbox_head.loss(
                self._mask_detection_outputs(det_outputs, det_mask),
                self._mask_list(batch_inputs_dict["gt_boxes"], det_mask),
                self._mask_list(batch_inputs_dict["gt_labels"], det_mask),
            )
        else:
            # Keep the detection branch in the autograd graph with zero
            # gradients so DDP reducers see every parameter.
            zero_loss = sum(value.float().sum() for value in det_outputs.values()) * 0.0
            det_metrics = {"loss": zero_loss}

        seg_loss = seg_metrics["loss"]
        weighted_seg_loss = self.segmentation_loss_weight * seg_loss
        weighted_det_loss = self.detection_loss_weight * det_metrics["loss"]
        metrics: dict[str, torch.Tensor] = {
            "seg_loss_ce": seg_metrics["loss_ce"],
            "seg_loss_lovasz": seg_metrics["loss_lovasz"],
            "seg_loss": seg_loss,
            "weighted_seg_loss": weighted_seg_loss,
            "weighted_det_loss": weighted_det_loss,
            "loss": weighted_det_loss + weighted_seg_loss,
        }
        metrics.update({f"det_{name}": value for name, value in det_metrics.items()})
        return metrics

    def build_eval_output(
        self, batch: Mapping[str, Any], outputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Produce detection and original-point segmentation eval data.

        Frames without detection supervision contribute empty predictions and
        their (already empty) ground truth instead of being dropped: the
        detection metric state must grow by exactly one entry per frame on
        every rank, or torchmetrics' per-element list-state ``all_gather``
        deadlocks under DDP when ranks see different seg/det frame mixes.
        Empty prediction + empty ground truth is metric-neutral.
        """
        det_mask = self._detection_frame_mask(batch)
        predictions = self.bbox_head.predict(outputs["det_outputs"])
        predictions = [
            prediction if flagged else {key: value[:0] for key, value in prediction.items()}
            for prediction, flagged in zip(predictions, det_mask.tolist())
        ]
        eval_out = detection_eval_output(predictions, batch)
        eval_out.update(segmentation_eval_output(outputs["seg_logits"], batch))
        return eval_out

    def get_export_output_names(self) -> list[str]:
        """Return configured ONNX export output names.

        Returns:
            Output names passed to the export spec.

        Raises:
            ValueError: If export output names were not configured.
        """
        if self._export_output_names is None:
            raise ValueError(
                "export_output_names must be provided at construction time to use export."
            )
        return list(self._export_output_names)

    def build_export_spec(self, batch_inputs_dict: Mapping[str, torch.Tensor]) -> ExportSpec:
        """Build the ONNX export spec for joint PTv3 segmentation+detection."""
        if self.grid_size is None or self.point_cloud_range is None:
            raise ValueError(
                "grid_size and point_cloud_range must be provided at construction time to use "
                "export."
            )
        sparse_shape, serialization_depth = self._compute_export_geometry(batch_inputs_dict)
        point, input_args = serialize_point_cloud_batch(
            batch_inputs_dict, self.EXPORT_ORDER, serialization_depth
        )
        serialized_pooling_inputs, serialized_pooling_input_names = (
            flatten_serialized_pooling_inputs(
                build_serialized_pooling_metadata(
                    point["grid_coord"],
                    point["serialized_code"],
                    point["serialized_order"],
                    self.encoder.stride,
                )
            )
        )
        export_module = _PTv3SegDetExportModule(
            encoder=self._prepare_encoder_export(),
            seg3d_head=self.seg3d_head.prepare_for_export(self.EXPORT_ORDER),
            bev_neck=deepcopy(self.bev_neck).eval(),
            bbox_head=self.bbox_head.prepare_for_export(),
            sparse_shape=sparse_shape,
            serialized_depth=serialization_depth,
            output_names=self.get_export_output_names(),
        )
        export_module.eval()
        export_input_args = (
            input_args[0],
            input_args[1],
            input_args[3],
            *serialized_pooling_inputs,
        )
        input_param_names = [
            "grid_coord",
            "feat",
            "serialized_code",
            *serialized_pooling_input_names,
        ]
        output_names = self.get_export_output_names()
        dynamic_axes = build_ptv3_input_dynamic_axes(input_param_names)
        dynamic_axes.update(
            build_point_feature_dynamic_axes(
                tuple(name for name in output_names if name in {"pred_labels", "pred_probs"})
            )
        )
        return ExportSpec(
            module=export_module,
            args=export_input_args,
            input_param_names=input_param_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            supported_stages=self.EXPORT_SUPPORTED_STAGES,
        )

    def build_export_specs(
        self, batch_inputs_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, ExportSpec]:
        """Build split PTv3 segdet ONNX export specs for encoder, seg head, and det head."""
        if self.grid_size is None or self.point_cloud_range is None:
            raise ValueError(
                "grid_size and point_cloud_range must be provided at construction time to use "
                "export."
            )
        context = build_ptv3_export_context(self, batch_inputs_dict)
        det_output_names = [
            n for n in self.get_export_output_names() if n not in ("pred_labels", "pred_probs")
        ]
        return {
            "encoder": build_encoder_export_spec(context),
            "seg3d_head": build_seg_head_export_spec(
                context,
                self.seg3d_head.prepare_for_export(self.EXPORT_ORDER),
                ["pred_labels", "pred_probs"],
            ),
            "det3d_head": build_det_head_export_spec(
                context,
                self.bev_neck,
                self.bbox_head.prepare_for_export(),
                det_output_names,
            ),
        }


class _PTv3SegDetExportModule(nn.Module):
    """ONNX-exportable PTv3 segmentation+detection graph with baked sparse shape."""

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        seg3d_head: PTv3SegDecoderHead,
        bev_neck: PTv3DetBEVNeck,
        bbox_head: nn.Module,
        sparse_shape: torch.Tensor,
        serialized_depth: torch.Tensor,
        output_names: Sequence[str],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.seg3d_head = seg3d_head
        self.bev_neck = bev_neck
        self.bbox_head = bbox_head
        self.output_names = list(output_names)
        self.register_buffer("_sparse_shape", sparse_shape.to(dtype=torch.long), persistent=False)
        self.register_buffer("_serialized_depth", serialized_depth, persistent=False)

    def forward(
        self,
        grid_coord: torch.Tensor,
        feat: torch.Tensor,
        serialized_code: torch.Tensor,
        *serialized_pooling_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run the export graph and return outputs in configured order.

        Args:
            grid_coord: Input voxel coordinates.
            feat: Input point or voxel features.
            serialized_code: Serialization codes for the base point set.
            serialized_pooling_inputs: Precomputed pooling metadata tensors.

        Returns:
            Tuple of export tensors ordered according to ``output_names``.
        """
        point = _run_ptv3_encoder_export(
            self.encoder,
            grid_coord,
            feat,
            self._serialized_depth,
            serialized_code,
            self._sparse_shape,
            *serialized_pooling_inputs,
        )
        # BEV branch first: the segmentation decoder consumes the pooling
        # chain destructively.
        bev_features = self.bev_neck(point)
        det_outputs = self.bbox_head(bev_features)

        seg_logits = self.seg3d_head(point)
        pred_probs = torch.softmax(seg_logits, dim=1)
        pred_labels = pred_probs.argmax(dim=1)

        outputs: dict[str, torch.Tensor] = {
            "pred_labels": pred_labels,
            "pred_probs": pred_probs,
            **det_outputs,
        }
        return tuple(outputs[name] for name in self.output_names)
