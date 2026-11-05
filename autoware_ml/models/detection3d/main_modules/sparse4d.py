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
the inputs and the outputs of the detection graph.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.models.detection3d.head_outputs import Sparse4DHeadOutputs
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.models.detection3d.heads.sparse4d import Sparse4DHead, SparseStage
from autoware_ml.models.detection3d.main_modules.sparse4ds.detection_task import (
    Sparse4DDetectionTask,
)
from autoware_ml.models.detection3d.main_modules.sparse4ds.export_wrappers import (
    Sparse4DHeadExportWrapper,
    build_sparse4d_head_export_spec,
)
from autoware_ml.models.detection3d.main_modules.sparse4ds.sparse4d_lidar import Sparse4DLidar
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.models.segmentation3d.ptv3_base import build_encoder_export_spec
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.deploy import ExportSpec


class Sparse4DDetectionModel(ModuleBaseModel):
    """Compose a Sparse4D detector with lidar and camera branches.

    The head samples the sparse stages of the lidar branch at keypoints around its anchors and
    carries the most confident instances from one frame of a scenario to the next. Only the
    lidar branch exists so far, the camera branch is a placeholder of the signature.
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
            lidar_network: Lidar branch returning the sparse stages the head samples.
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
            raise ValueError("Sparse4D needs a lidar_network, the head samples its stages.")
        if camera_network is not None:
            raise NotImplementedError("Sparse4D supports no camera_network yet, set it to null.")
        self.lidar_network = lidar_network
        self.camera_network = camera_network
        self.bbox_head = bbox_head
        self.det3d_task = Sparse4DDetectionTask(bbox_head)
        self._validate_sampling_contract()

    def _validate_sampling_contract(self) -> None:
        """Validate that the lidar branch returns the levels the head samples.

        Raises:
            ValueError: If the pyramid of the lidar branch and the head disagree on the levels.
        """
        self.det3d_task.validate_levels(self.lidar_network.det_fusion)

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group the pretrained encoder parameters apart from the newly initialized branch.

        Returns:
            The encoder groups of the lidar branch and ``det3d_branch``, the pyramid and the head.
        """
        lidar_groups = dict(self.lidar_network.build_optimizer_groups())
        det_fusion_params = lidar_groups.pop("det_fusion")
        head_params = [
            parameter for parameter in self.bbox_head.parameters() if parameter.requires_grad
        ]
        return {**lidar_groups, "det3d_branch": [*det_fusion_params, *head_params]}

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run the lidar branch and the Sparse4D head on one batch of frames.

        Args:
            multi_task_batch_inputs: Model inputs with the voxels, the frame metadata and the
                detection ground truth.

        Returns:
            The Sparse4D head outputs.
        """
        _, det3d_outputs = self._forward_stages(multi_task_batch_inputs)
        return ModelOutputs(detection3d_head_outputs=self.det3d_task.head_outputs(det3d_outputs))

    def _forward_stages(
        self, multi_task_batch_inputs: ModelBatchInputs
    ) -> tuple[list[SparseStage], Sparse4DHeadOutputs]:
        """Encode the batch with the lidar branch and run the head on the fused stages.

        Args:
            multi_task_batch_inputs: Model inputs with the voxels, the frame metadata and the
                detection ground truth.

        Returns:
            The fused stages the head sampled and its outputs.

        Raises:
            ValueError: If the batch carries no frame metadata, no points or no voxels.
        """
        frame_meta = self.det3d_task.frame_meta(multi_task_batch_inputs)
        stages = self._lidar_stages(multi_task_batch_inputs)
        det3d_outputs = self.det3d_task.run(
            stages, multi_task_batch_inputs, frame_meta, self.head_context(multi_task_batch_inputs)
        )
        return stages, det3d_outputs

    def _lidar_stages(self, multi_task_batch_inputs: ModelBatchInputs) -> list[SparseStage]:
        """Run the lidar branch on the voxels of the batch.

        Args:
            multi_task_batch_inputs: Model inputs with the points and their voxels.

        Returns:
            The fused stages the head samples.

        Raises:
            ValueError: If the batch carries no points or no voxels.
        """
        point_batch = multi_task_batch_inputs.multi_task_gt_batch.point_cloud_gt_batch
        if point_batch is None:
            raise ValueError("Sparse4D needs the points of the batch.")
        voxels_data = multi_task_batch_inputs.voxels_data
        if voxels_data is None:
            raise ValueError("Sparse4D needs the voxels of a voxelizer in the model inputs.")
        # Call the module rather than its forward so module hooks (model summary sizes and
        # FLOP counting) see the branch.
        return self.lidar_network(
            voxels=voxels_data.voxels,
            num_points=voxels_data.num_points,
            voxel_coords=voxels_data.batch_xyz_coords(),
            num_dropped_voxels=voxels_data.num_dropped_voxels,
            time_lag_column=point_batch.timestamp_difference_dim,
        )

    def head_context(self, multi_task_batch_inputs: ModelBatchInputs) -> Any:
        """Frame context of the head, none without a camera branch."""
        return None

    def compute_metrics(
        self, multi_task_batch_inputs: ModelBatchInputs, multi_task_outputs: ModelOutputs
    ) -> dict[str, torch.Tensor]:
        """Compute the detection losses of one step.

        Frames without ground-truth boxes and samples padding a streaming lane skip the loss.

        Args:
            multi_task_batch_inputs: Model inputs with the ground truth.
            multi_task_outputs: Raw outputs from :meth:`forward`.

        Returns:
            The losses of the head, ``loss`` the total.
        """
        metrics, _, _ = self.det3d_task.loss(
            multi_task_outputs.detection3d().sparse4d_head(), multi_task_batch_inputs
        )
        self.det3d_task.log_step_metrics(self, metrics, multi_task_batch_inputs)
        return metrics

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> dict[str, Any]:
        """Decode the detections with the tracks and pair them with the ground truth."""
        return self.det3d_task.eval_output(batch, outputs)

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
        return self.det3d_task.decode(outputs)

    def build_export_specs(
        self, multi_task_batch_inputs: ModelBatchInputs
    ) -> dict[str, ExportSpec]:
        """Build the split ONNX export specs of the encoder and the detection graph.

        The lidar branch exports the ``ptv3_lidar_encoder`` graph returning the stage features.
        The ``ptv3_det3d_head`` graph fuses the stages, takes the instance bank state of the last
        frame and returns the outputs of the frame with the state of the next one.

        Args:
            multi_task_batch_inputs: Example preprocessed batch with the voxels.

        Returns:
            Ordered mapping of module name to export specification.
        """
        context = self.lidar_network.build_export_context(multi_task_batch_inputs)
        return {
            "ptv3_lidar_encoder": build_encoder_export_spec(context),
            "ptv3_det3d_head": build_sparse4d_head_export_spec(
                context,
                Sparse4DHeadExportWrapper(
                    self.lidar_network.det_fusion, self.bbox_head.prepare_for_export()
                ),
            ),
        }
