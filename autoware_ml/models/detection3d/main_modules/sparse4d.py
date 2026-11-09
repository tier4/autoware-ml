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

"""Sparse4D detector.

This module contains the high-level Sparse4D detector wrapper. A temporal head needs scenario
streams, see ``DataModule(streaming=True)``. Its export carries the instance bank state through
the inputs and the outputs of the graph.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaBatch
from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    Sparse4DHeadOutputs,
)
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.detection3d.eval_output import detection3d_eval_output
from autoware_ml.models.detection3d.heads.sparse4d import Sparse4DHead
from autoware_ml.models.detection3d.main_modules.sparse4ds.export_wrappers import (
    build_sparse4d_lidar_export_spec,
)
from autoware_ml.models.detection3d.main_modules.sparse4ds.sparse4d_lidar import Sparse4DLidar
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.models.multi.ptv3_det3d_seg3d import (
    detection3d_frame_mask,
    detection3d_gt_batch,
    empty_unsupervised_predictions,
)
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.deploy import ExportSpec


class Sparse4DDetectionModel(ModuleBaseModel):
    """Compose a Sparse4D detector with lidar and camera branches.

    The head samples the BEV map of the lidar branch at keypoints around its anchors and carries
    the most confident instances from one frame of a scenario to the next. Only the lidar branch
    exists so far, the camera branch is a placeholder of the signature.
    """

    def __init__(
        self,
        data_preprocessor: DataPreprocessor,
        lidar_network: Sparse4DLidar | None,
        camera_network: nn.Module | None,
        bbox_head: Sparse4DHead,
        log_dict_configs: LogDictConfigs,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ) -> None:
        """Initialize the Sparse4D detector.

        Args:
            data_preprocessor: Preprocessor for the model inputs.
            lidar_network: Lidar branch returning the BEV map the head samples.
            camera_network: Camera branch, not supported yet.
            bbox_head: Sparse4D query head.
            log_dict_configs: Logging configuration for training and validation.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            optimizer_group_overrides: Optimizer overrides keyed by optimizer group name.
            scheduler_config: Lightning scheduler metadata such as ``interval``.
            metrics: Detection metrics accumulated during validation and test.

        Raises:
            ValueError: If no lidar branch is given.
            NotImplementedError: If a camera branch is given.
        """
        super().__init__(
            data_preprocessor=data_preprocessor,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_group_overrides=optimizer_group_overrides,
            scheduler_config=scheduler_config,
            metrics=metrics,
            log_dict_configs=log_dict_configs,
        )
        if lidar_network is None:
            raise ValueError("Sparse4D needs a lidar_network, the head samples its BEV map.")
        if camera_network is not None:
            raise NotImplementedError("Sparse4D supports no camera_network yet, set it to null.")
        self.lidar_network = lidar_network
        self.camera_network = camera_network
        self.bbox_head = bbox_head

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run the lidar branch and the Sparse4D head on one batch of frames.

        Args:
            multi_task_batch_inputs: Model inputs with the voxels, the frame metadata and the
                detection ground truth.

        Returns:
            The Sparse4D head outputs.

        Raises:
            ValueError: If the batch carries no frame metadata or no voxels.
        """
        frame_meta = self._frame_meta(multi_task_batch_inputs)
        bev_map = self._lidar_bev(multi_task_batch_inputs)
        det3d_outputs = self._run_head(bev_map, multi_task_batch_inputs, frame_meta)
        return ModelOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=None,
                transfusion_head_outputs=None,
                sparse4d_head_outputs=det3d_outputs,
            )
        )

    @staticmethod
    def _frame_meta(multi_task_batch_inputs: ModelBatchInputs) -> FrameMetaBatch:
        """Read the frame metadata the head places every frame of a stream with.

        Args:
            multi_task_batch_inputs: Model inputs of the batch.

        Returns:
            The frame metadata of every sample.

        Raises:
            ValueError: If the batch carries no frame metadata.
        """
        frame_meta = multi_task_batch_inputs.multi_task_gt_batch.frame_meta_batch
        if frame_meta is None:
            raise ValueError("The Sparse4D head needs the frame metadata of every sample.")
        return frame_meta

    def _lidar_bev(self, multi_task_batch_inputs: ModelBatchInputs) -> torch.Tensor:
        """Run the lidar branch on the voxels of the batch.

        Args:
            multi_task_batch_inputs: Model inputs with the voxels.

        Returns:
            The BEV map the head samples.

        Raises:
            ValueError: If the batch carries no voxels.
        """
        voxels_data = multi_task_batch_inputs.voxels_data
        if voxels_data is None:
            raise ValueError("Sparse4D needs the voxels of a voxelizer in the model inputs.")
        # Call the module rather than its forward so module hooks (model summary sizes and
        # FLOP counting) see the branch. The sparse middle encoder consumes ``(batch, z, y, x)``
        # coordinates.
        return self.lidar_network(
            voxels=voxels_data.voxels,
            coords=voxels_data.batch_zyx_coords(),
            num_points=voxels_data.num_points,
            batch_size=multi_task_batch_inputs.multi_task_gt_batch.batch_size,
        )

    def _run_head(
        self,
        bev_map: torch.Tensor,
        multi_task_batch_inputs: ModelBatchInputs,
        frame_meta: FrameMetaBatch,
    ) -> Sparse4DHeadOutputs:
        """Run the head over the BEV map of a batch.

        The frame metadata places every frame of a stream for the temporal head, the boxes feed
        the denoising queries and their instance ids and point counts the tracks.

        Args:
            bev_map: BEV map of the lidar branch the head samples.
            multi_task_batch_inputs: Model inputs with the detection ground truth.
            frame_meta: Frame metadata of every sample, see :meth:`_frame_meta`.

        Returns:
            The head outputs.
        """
        gt_detections = detection3d_gt_batch(multi_task_batch_inputs)
        return self.bbox_head(
            bev_map,
            timestamps_seconds=frame_meta.timestamps,
            lidar2global=frame_meta.ego2globals,
            scene_tokens=frame_meta.scene_tokens,
            gt_boxes=gt_detections.valid_bboxes_3d(),
            gt_labels=gt_detections.valid_labels_3d(),
            gt_instance_ids=gt_detections.valid_bboxes_instance_ids(),
            gt_num_points=gt_detections.valid_bboxes_num_points(),
            context=self.head_context(multi_task_batch_inputs),
        )

    def head_context(self, multi_task_batch_inputs: ModelBatchInputs) -> Any:
        """Frame context of the head, none without a camera branch."""
        return None

    def compute_metrics(
        self, multi_task_batch_inputs: ModelBatchInputs, multi_task_outputs: ModelOutputs
    ) -> dict[str, torch.Tensor]:
        """Compute the detection losses of one step over the frames with boxes.

        Frames without ground-truth boxes and samples that only pad a streaming lane take no
        part in the loss. Without any such frame the loss is zero with gradients, which keeps
        the detection branch visible to DDP.

        Args:
            multi_task_batch_inputs: Model inputs with the ground truth.
            multi_task_outputs: Raw outputs from :meth:`forward`.

        Returns:
            The losses of the head, ``loss`` the total.
        """
        det3d_outputs = multi_task_outputs.detection3d().sparse4d_head()
        gt_detections = detection3d_gt_batch(multi_task_batch_inputs)
        padding_mask = multi_task_batch_inputs.multi_task_gt_batch.padding_mask
        det3d_mask = detection3d_frame_mask(gt_detections) & ~padding_mask.to(
            gt_detections.gt_bboxes_3d.device
        )
        if not bool(det3d_mask.any()):
            graph_tensors = [
                det3d_outputs.classification,
                det3d_outputs.prediction,
                det3d_outputs.quality,
            ]
            metrics = {"loss": sum(tensor.float().sum() for tensor in graph_tensors) * 0.0}
        else:
            supervised = det3d_mask.tolist()
            lanes = det3d_mask.nonzero().flatten().tolist()
            supervised_outputs = det3d_outputs.select_samples(det3d_mask)
            det3d_targets = self.bbox_head.assign(
                supervised_outputs,
                _flagged(gt_detections.valid_bboxes_3d(), supervised),
                _flagged(gt_detections.valid_labels_3d(), supervised),
                _flagged(gt_detections.valid_bboxes_instance_ids(), supervised),
                lanes,
            )
            metrics = self.bbox_head.loss_from_targets(supervised_outputs, det3d_targets, lanes)
        self._log_step_metrics(metrics, multi_task_batch_inputs)
        return metrics

    def _log_step_metrics(
        self, metrics: Mapping[str, torch.Tensor], multi_task_batch_inputs: ModelBatchInputs
    ) -> None:
        """Log the per-step values of rank 0 in training, to watch the convergence in an epoch.

        The per layer losses are logged for the last decoder layer only.

        Args:
            metrics: Losses of the step.
            multi_task_batch_inputs: Model inputs of the step.
        """
        if not self.training:
            return
        self.log_dict(
            {
                f"train_step/{name}": value.detach()
                for name, value in metrics.items()
                if "_layer_" not in name or "_layer_-1_" in name
            },
            on_step=True,
            on_epoch=False,
            batch_size=self.get_log_batch_size(multi_task_batch_inputs),
        )

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> dict[str, Any]:
        """Decode the detections and pair them with the ground truth for the metrics.

        Frames without detection supervision contribute empty predictions. A temporal head adds
        which boxes were observed in the stream so far and the stream of every frame.

        Args:
            batch: Model inputs of the evaluation step.
            outputs: Raw outputs returned by :meth:`forward`.

        Returns:
            Flat eval output dict consumed by the detection and the tracking metrics.
        """
        det3d_mask = detection3d_frame_mask(detection3d_gt_batch(batch))
        predictions = empty_unsupervised_predictions(
            self.decode_outputs(batch, outputs).detection3d_predictions, det3d_mask
        )
        eval_out = detection3d_eval_output(
            ModelPredictions(detection3d_predictions=predictions), batch
        )
        tracker = self.bbox_head.tracker
        if tracker is not None:
            eval_out["gt_observed"] = list(tracker.frame_observed)
            eval_out["stream_ids"] = list(tracker.stream_ids)
        return eval_out

    def decode_outputs(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        """Decode the boxes of the batch for inference.

        Args:
            multi_task_batch_inputs: Model inputs of the prediction step, unused.
            outputs: Raw outputs returned by :meth:`forward`.

        Returns:
            The decoded boxes of every sample.
        """
        del multi_task_batch_inputs
        return self.bbox_head.decode_outputs(outputs.detection3d())

    def build_export_specs(
        self, multi_task_batch_inputs: ModelBatchInputs
    ) -> dict[str, ExportSpec]:
        """Build the ONNX export specification of the lidar graph.

        The ``sparse4d_lidar`` graph takes the voxels of one frame like ``bevfusion_lidar``,
        followed by the instance bank state of the last frame, and returns the outputs of the
        frame with the state of the next one.

        Args:
            multi_task_batch_inputs: Batched model inputs used to derive the export tensors.

        Returns:
            Ordered mapping of module name to export specification.
        """
        export_model = deepcopy(self).eval()
        export_model.lidar_network.prepare_for_export()
        bbox_head = export_model.bbox_head.prepare_for_export()
        return {
            "sparse4d_lidar": build_sparse4d_lidar_export_spec(
                export_model.lidar_network, bbox_head, multi_task_batch_inputs
            )
        }


def _flagged(values: Sequence[torch.Tensor], flags: Sequence[bool]) -> list[torch.Tensor]:
    """Select the entries of the flagged samples from a per-sample list."""
    return [value for value, flagged in zip(values, flags, strict=True) if flagged]
