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

"""Native BEVFusion detector.

This module contains the high-level BEVFusion detector wrapper and export ABI.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any
from types import MappingProxyType

from jaxtyping import Bool, Float32, Int32, Int64


import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    TransFusionHeadOutputs,
)
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.detection3d.eval_output import detection_eval_output
from autoware_ml.models.detection3d.main_modules.bevfusions.bevfusion_lidar import (
    BEVFusionLidar,
)
from autoware_ml.models.detection3d.main_modules.bevfusions.bevfusion_camera import (
    BEVFusionCamera,
    BEVFusionImageBackboneExportWrapper,
)
from autoware_ml.models.detection3d.heads.transfusions.transfusion_head import TransFusionHead
from autoware_ml.models.detection3d.main_modules.bevfusions.export_wrappers import (
    export_detection_outputs,
)
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.ops.voxelization.voxelization import VoxelsData
from autoware_ml.utils.deploy import ExportSpec


class _BEVFusionExportWrapperBase(nn.Module):
    """Shared base for BEVFusion main body export wrappers.

    Holds the wrapped model and the logic to rebuild a
    :class:`ModelBatchInputs` from the flat single-sample tensors the
    deployment runtime provides. Subclasses only define ``forward`` with the
    tensor signature expected by their export target.
    """

    def __init__(self, model: BEVFusionDetectionModel) -> None:
        """Initialize the export wrapper.

        Args:
            model: BEVFusion model instance.
        """
        super().__init__()
        self.model = model

    def _construct_multi_task_batch_inputs(
        self,
        voxels: Float32[torch.Tensor, "num_voxels max_num_points num_point_features"],
        coors: Int32[torch.Tensor, "num_voxels 3"],
        num_points_per_voxel: Int32[torch.Tensor, " num_voxels"],
    ) -> ModelBatchInputs:
        """Construct a single-sample ``ModelBatchInputs`` from runtime tensors.

        The result carries no ground truths and no image inputs; image data is
        supplied separately by wrappers that need it.

        Args:
            voxels: Voxel features.
            coors: Voxel coordinates in ``(z, y, x)`` order without batch column.
                Must be in this order to align with autoware.
            num_points_per_voxel: Number of points in each voxel.

        Returns:
            Batch inputs holding the voxel data for a batch of size one.
        """
        voxels_data = VoxelsData(
            voxels=voxels,
            # ``VoxelsData`` stores ``(x, y, z)``, the runtime hands ``(z, y, x)``
            coords=coors[:, [2, 1, 0]],
            num_points=num_points_per_voxel,
            batch_indices=torch.zeros(coors.shape[0], dtype=torch.int32, device=coors.device),
            point_voxel_indices=torch.zeros((0,), dtype=torch.int64, device=coors.device),
            num_dropped_voxels=torch.zeros((), dtype=torch.int64, device=coors.device),
        )
        multi_task_gt_batch = ModelGTBatch(
            point_cloud_gt_batch=None,
            detection3d_gt_batch=None,
            image_gt_batch=None,
            io_processing_time=0.0,
            segmentation3d_gt_batch=None,
        )
        return ModelBatchInputs(
            multi_task_gt_batch=multi_task_gt_batch,
            voxels_data=voxels_data,
            image_data=None,
            range_view_data=None,
        )


class _BEVFusionLidarExportWrapper(_BEVFusionExportWrapperBase):
    """Wrap the lidar-only BEVFusion main body export.

    Used when the image branch is disabled. The wrapper exposes the same
    single-sample tensor interface as the camera-lidar main body without the
    image inputs.
    """

    def forward(
        self,
        voxels: Float32[torch.Tensor, "num_voxels max_num_points num_point_features"],
        coors: Int32[torch.Tensor, "num_voxels 3"],
        num_points_per_voxel: Int32[torch.Tensor, " num_voxels"],
    ) -> tuple[
        Float32[torch.Tensor, "num_box_code num_proposals"],
        Float32[torch.Tensor, " num_proposals"],
        Int64[torch.Tensor, " num_proposals"],
    ]:
        """Run export-time inference on lidar voxel inputs.

        Args:
            voxels: Voxel features.
            coors: Voxel coordinates in ``(z, y, x)`` order without batch column.
                Must be in this order to align with autoware.
            num_points_per_voxel: Number of points in each voxel.

        Returns:
            Tuple of ``bbox_pred``, ``score``, and ``label_pred``.
        """
        multi_task_batch_inputs = self._construct_multi_task_batch_inputs(
            voxels=voxels,
            coors=coors,
            num_points_per_voxel=num_points_per_voxel,
        )
        outputs = self.model._forward_with_batch_size(
            multi_task_batch_inputs=multi_task_batch_inputs,
            batch_size=1,
            image_bev=None,  # No image BEV since this is lidar-only
        )
        return export_detection_outputs(self.model.bbox_head, outputs)


class _BEVFusionExportWrapper(_BEVFusionExportWrapperBase):
    """Wrap the camera-lidar main body export.

    The wrapper exposes the exact tensor-only signature expected by the
    deployment runtime and returns runtime-decodable detection outputs.
    """

    def forward(
        self,
        voxels: Float32[torch.Tensor, "num_voxels max_num_points num_point_features"],
        coors: Int32[torch.Tensor, "num_voxels 3"],
        num_points_per_voxel: Int32[torch.Tensor, " num_voxels"],
        image_features: Float32[torch.Tensor, "num_cameras channels feature_height feature_width"],
        depth_maps: Float32[torch.Tensor, "num_cameras 1 height width"],
        geom_feats: Float32[torch.Tensor, "num_frustum_points 4"],
        kept: Bool[torch.Tensor, " num_frustum_points"],
        ranks: Int64[torch.Tensor, " num_kept"],
        indices: Int64[torch.Tensor, " num_kept"],
    ) -> tuple[
        Float32[torch.Tensor, "num_box_code num_proposals"],
        Float32[torch.Tensor, " num_proposals"],
        Int64[torch.Tensor, " num_proposals"],
    ]:
        """Run export-time inference with precomputed BEV-pool metadata.

        Args:
            voxels: Lidar voxel tensor.
            coors: Voxel coordinates in ``(z, y, x)`` order without batch column. Must be in this
                order to align with autoware.
            num_points_per_voxel: Number of points per voxel.
            image_features: Precomputed image features.
            depth_maps: Precomputed depth maps.
            geom_feats: Precomputed BEV-pool geometry features, passed as float by
                the runtime and cast to integer indices inside the camera branch.
            kept: Keep mask for pooled features.
            ranks: Sorted BEV ranks.
            indices: Sorting indices aligned with ``ranks``.

        Returns:
            Tuple of ``bbox_pred``, ``score``, and ``label_pred``.
        """
        multi_task_batch_inputs = self._construct_multi_task_batch_inputs(
            voxels=voxels,
            coors=coors,
            num_points_per_voxel=num_points_per_voxel,
        )
        outputs = self.model._forward_export(
            multi_task_batch_inputs=multi_task_batch_inputs,
            depth_maps=depth_maps,
            image_features=image_features,
            geom_feats=geom_feats,
            kept=kept,
            ranks=ranks,
            indices=indices,
        )
        return export_detection_outputs(self.model.bbox_head, outputs)


class BEVFusionDetectionModel(ModuleBaseModel):
    """Compose a BEVFusion detector with camera and lidar branches.

    The model fuses image and lidar features in BEV space and exposes the
    shared Autoware-ML training, prediction, and export interfaces.
    """

    def __init__(
        self,
        data_preprocessor: DataPreprocessor,
        lidar_network: BEVFusionLidar | None,
        camera_network: BEVFusionCamera | None,
        # TODO (KokSeang): Consider making the head a generic DetectionHead type instead of TransFusionHead.
        bbox_head: TransFusionHead,
        log_dict_configs: LogDictConfigs,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ) -> None:
        """Initialize BEVFusion.

        Args:
            data_preprocessor: Preprocessor for the model inputs.
            lidar_network: Lidar-only BEVFusion main body. It includes fusion layers to fuse both
                lidar and camera bev features as well.
            camera_network: Camera-only BEVFusion main body. The outputs are camera-only bev
                features after BEV transformation.
            bbox_head: Detection head.
            log_dict_configs: Logging configuration for training and validation.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            metrics: Detection metrics accumulated during validation and test.
        """
        super().__init__(
            data_preprocessor=data_preprocessor,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            log_dict_configs=log_dict_configs,
        )
        self.lidar_network = lidar_network
        self.camera_network = camera_network
        self.bbox_head = bbox_head
        assert self.lidar_network is not None or self.camera_network is not None, (
            "At least one of lidar_network or camera_network must be provided."
        )
        self._validate_geometry_contract()

    def _validate_geometry_contract(self) -> None:
        """Validate static geometry contracts between camera and lidar branches.

        Raises:
            ValueError: If configured lidar and image BEV branches do not
                agree on the BEV spatial shape.
        """
        if self.camera_network is None or self.lidar_network is None:
            return

        image_bev_shape = self.camera_network.expected_bev_shape
        lidar_bev_shape = self.lidar_network.expected_bev_shape
        if image_bev_shape != lidar_bev_shape:
            raise ValueError(
                "BEVFusion image and lidar branches must share the same BEV shape. "
                f"Got image BEV shape {image_bev_shape} and lidar BEV shape {lidar_bev_shape}."
            )

    # TODO(KokSeang): This signature is temporary different from the base class,
    # and will be refactored to match the base class signature once the detection metric is refactored
    # to accept ModelPredictions and MultiTaskFeatures directly.
    def build_eval_output(  # type: ignore[override]
        self, batch: ModelBatchInputs, outputs: ModelOutputs
    ) -> dict[str, Any]:
        """Decode detections and pair them with ground truth for metrics."""
        if outputs.detection3d_head_outputs is None:
            raise ValueError(
                "ModelOutputs must contain detection3d_head_outputs for CenterPoint build_eval_output pass."
            )

        return detection_eval_output(
            predictions=self.bbox_head.decode_outputs(outputs.detection3d_head_outputs),
            batch_inputs=batch,
        )

    def _forward_export(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        image_features: Float32[torch.Tensor, "num_cameras channels feature_height feature_width"],
        depth_maps: Float32[torch.Tensor, "num_cameras 1 height width"],
        geom_feats: Float32[torch.Tensor, "num_frustum_points 4"],
        kept: Bool[torch.Tensor, " num_frustum_points"],
        ranks: Int64[torch.Tensor, " num_kept"],
        indices: Int64[torch.Tensor, " num_kept"],
    ) -> TransFusionHeadOutputs:
        """Run the export-time main body with runtime-compatible inputs.

        Args:
            multi_task_batch_inputs: ModelBatchInputs containing the voxelized lidar inputs.
            image_features: Precomputed image features.
            depth_maps: Precomputed depth maps.
            bev_pool_result: Precomputed BEV-pool metadata for the camera branch.

        Returns:
            Detection head outputs produced by the export path.
        """
        if self.camera_network is None:
            raise ValueError("Image branch is not configured.")

        # First, project image features to bev
        image_bev = self.camera_network.forward_export(
            image_features=image_features,
            depth_maps=depth_maps,
            geom_feats=geom_feats,
            kept=kept,
            ranks=ranks,
            indices=indices,
        )

        # Run BEVFusion without the image branch
        return self._forward_with_batch_size(
            multi_task_batch_inputs=multi_task_batch_inputs,
            batch_size=1,
            image_bev=image_bev,
        )

    def _forward_with_batch_size(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        batch_size: int | None = None,
        image_bev: Float32[torch.Tensor, "batch_size channels height width"] | None = None,
    ) -> TransFusionHeadOutputs:
        """Run the configured BEV branches and dense head.

        Args:
            multi_task_batch_inputs: ModelBatchInputs containing the voxelized lidar inputs.
            batch_size: Optional explicit batch size.
            image_bev: Optional precomputed image BEV tensor.

        Returns:
            TransFusion-based detection head outputs for the configured branches.
        """
        if batch_size is None:
            batch_size = multi_task_batch_inputs.multi_task_gt_batch.infer_batch_size()

        # Run camera branch if image BEV is not provided and camera network is configured
        # Camera must run before the lidar branch because the lidar branch requires image BEV for
        # fusion
        if image_bev is None and self.camera_network is not None:
            image_data = multi_task_batch_inputs.image_data
            if image_data is None:
                raise ValueError(
                    "ModelBatchInputs must contain image_data for BEVFusion camera forward pass."
                )

            # For now, depth_maps must be provided for the camera branch to run. In the future, we
            # can add a flag to disable the depth guidance.
            if image_data.depth_maps is None:
                raise ValueError(
                    "ModelBatchInputs must contain depth_maps for BEVFusion camera forward pass."
                )

            # Call the module rather than its forward so module hooks (model summary sizes and
            # FLOP counting) see the branch.
            image_bev = self.camera_network(
                image_batch=image_data.images,
                depth_maps=image_data.depth_maps,
                camera_intrinsics=image_data.camera_intrinsics,
                aug_lidar2cam=image_data.lidar2cams,
                geom_feats_precomputed=None,
                image_features=None,
                img_aug_matrix=image_data.image_augmentation_matrices,
            )

        lidar_bev = None
        if self.lidar_network is not None:
            voxels_data = multi_task_batch_inputs.voxels_data
            if voxels_data is None:
                raise ValueError(
                    "ModelBatchInputs must contain voxels_data for CenterPoint forward pass."
                )

            assert batch_size is not None, "Batch size must be provided for lidar forward pass."
            # The sparse middle encoder consumes ``(batch, z, y, x)`` coordinates
            batch_coords = voxels_data.batch_zyx_coords()
            other_bev_features = [image_bev] if image_bev is not None else None
            lidar_bev = self.lidar_network(
                voxels=voxels_data.voxels,
                coords=batch_coords,
                num_points=voxels_data.num_points,
                batch_size=batch_size,
                other_bev_features=other_bev_features,
            )

        # Take the lidar BEV if available, otherwise only take the image BEV.
        # It's impossible that both aRE None because we check that at least one of the branches is
        # configured in the constructor.
        if lidar_bev is None:
            bbox_head_outputs = self.bbox_head(image_bev)
        else:
            bbox_head_outputs = self.bbox_head(lidar_bev)

        return bbox_head_outputs

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run the detector on voxelized lidar inputs.

        Args:
            multi_task_batch_inputs: ModelBatchInputs containing the voxelized lidar inputs.

        Returns:
            Detection head outputs.
        """
        detection_head_outputs = self._forward_with_batch_size(multi_task_batch_inputs)
        return ModelOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=None, transfusion_head_outputs=detection_head_outputs
            )
        )

    def compute_metrics(
        self, multi_task_batch_inputs: ModelBatchInputs, multi_task_outputs: ModelOutputs
    ) -> MappingProxyType[str, Float32[torch.Tensor, " num_losses"]]:
        """Compute TransfusionHead training losses."""
        if multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch is None:
            raise ValueError(
                "ModelBatchInputs must contain detection3d_gt_batch for CenterPoint compute_metrics pass."
            )

        if multi_task_outputs.detection3d_head_outputs is None:
            raise ValueError(
                "ModelOutputs must contain detection3d_head_outputs for CenterPoint compute_metrics pass."
            )

        detection3d_gt_batch = multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch
        return self.bbox_head.loss(
            outputs=multi_task_outputs.detection3d_head_outputs,
            gt_bboxes_3d=detection3d_gt_batch.gt_bboxes_3d,
            gt_labels_3d=detection3d_gt_batch.gt_labels_3d,
            gt_valid_bboxes=detection3d_gt_batch.gt_valid_bboxes,
            gt_traffic_cone_barrier_bbox_status=(
                detection3d_gt_batch.gt_traffic_cone_barrier_bbox_status
            ),
        )  # type: ignore[return-value]

    def decode_outputs(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        """Decode predictions for inference."""
        if outputs.detection3d_head_outputs is None:
            raise ValueError(
                "ModelOutputs must contain detection3d_head_outputs for CenterPoint decode_outputs pass."
            )

        multi_task_predictions = self.bbox_head.decode_outputs(
            outputs=outputs.detection3d_head_outputs
        )
        return multi_task_predictions

    def _prepare_export_model(self) -> "BEVFusionDetectionModel":
        """Return an export-ready model copy with exportable submodules.

        Returns:
            Deep copy of the model with the sparse middle encoder and the
            detection head replaced by their ONNX-exportable variants.
        """
        model = deepcopy(self).eval()
        if model.lidar_network is not None:
            model.lidar_network.prepare_for_export()
        if hasattr(model.bbox_head, "prepare_for_export"):
            model.bbox_head = model.bbox_head.prepare_for_export()
        return model

    def build_export_specs(
        self, multi_task_batch_inputs: ModelBatchInputs
    ) -> dict[str, ExportSpec]:
        """Build the ONNX export specifications for the runtime-compatible ABI.

        Lidar-only models export one ``bevfusion_lidar`` main body. Camera-lidar
        models export the ``bevfusion_image_backbone`` (raw ``uint8`` images to
        neck features) and the ``bevfusion_camera_lidar`` main body consuming
        those features together with precomputed depth maps and BEV-pool
        metadata. All exported graphs are single-sample, so only the first
        sample of the batch is used to derive the export tensors.

        Args:
            multi_task_batch_inputs: Batched model inputs used to derive export tensors.

        Returns:
            Ordered mapping of module name to export specification.

        Raises:
            ValueError: If the batch lacks the voxel data, or lacks image data or
                depth maps when the camera branch is configured.
        """
        voxels, coors, num_points_per_voxel = self._first_sample_voxel_inputs(
            multi_task_batch_inputs
        )
        export_model = self._prepare_export_model()

        if self.camera_network is None:
            return {
                "bevfusion_lidar": ExportSpec(
                    module=_BEVFusionLidarExportWrapper(export_model),
                    args=(voxels, coors, num_points_per_voxel),
                    input_param_names=["voxels", "coors", "num_points_per_voxel"],
                )
            }

        image_data = multi_task_batch_inputs.image_data
        if image_data is None:
            raise ValueError(
                "ModelBatchInputs must contain image_data to build BEVFusion camera-lidar "
                "export specs."
            )
        if image_data.depth_maps is None:
            raise ValueError(
                "ModelBatchInputs must contain depth_maps to build BEVFusion camera-lidar "
                "export specs."
            )

        # Keep the batch dimension for the camera helpers, then drop it for the
        # runtime tensors: the exported main body consumes a single sample.
        images = image_data.images[:1].float()
        depth_maps = image_data.depth_maps[:1].float()
        camera_intrinsics = image_data.camera_intrinsics[:1].float()
        lidar2cams = image_data.lidar2cams[:1].float()

        imgs_uint8 = (images[0] * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)
        with torch.no_grad():
            image_feats = self.camera_network.extract_image_features(images)[0]
            geom_feats, kept, ranks, indices = self.camera_network.build_export_geometry(
                camera_intrinsics, lidar2cams
            )

        assert export_model.camera_network is not None
        return {
            "bevfusion_image_backbone": ExportSpec(
                module=BEVFusionImageBackboneExportWrapper(export_model.camera_network),
                args=(imgs_uint8,),
                input_param_names=["imgs"],
            ),
            "bevfusion_camera_lidar": ExportSpec(
                module=_BEVFusionExportWrapper(export_model),
                args=(
                    voxels,
                    coors,
                    num_points_per_voxel,
                    image_feats,
                    depth_maps[0],
                    geom_feats.float(),
                    kept.bool(),
                    ranks.long(),
                    indices.long(),
                ),
                input_param_names=[
                    "voxels",
                    "coors",
                    "num_points_per_voxel",
                    "image_feats",
                    "depth_maps",
                    "geom_feats",
                    "kept",
                    "ranks",
                    "indices",
                ],
            ),
        }

    @staticmethod
    def _first_sample_voxel_inputs(
        multi_task_batch_inputs: ModelBatchInputs,
    ) -> tuple[
        Float32[torch.Tensor, "num_voxels max_num_points num_point_features"],
        Int32[torch.Tensor, "num_voxels 3"],
        Int32[torch.Tensor, " num_voxels"],
    ]:
        """Extract single-sample voxel export inputs in the runtime layout.

        The exported main body is a single-sample graph, so only voxels of the
        first batch sample are kept. Make sure ``VoxelsData`` already stores coordinates
        in the runtime ``(z, y, x)`` layout without a batch column.

        Args:
            multi_task_batch_inputs: Batched model inputs used to derive export tensors.

        Returns:
            Tuple of voxels, coordinates, and per-voxel point counts for the
            first sample.

        Raises:
            ValueError: If the batch carries no voxel data.
        """
        voxels_data = multi_task_batch_inputs.voxels_data
        if voxels_data is None:
            raise ValueError(
                "ModelBatchInputs must contain voxels_data to build BEVFusion export specs."
            )
        first_sample = voxels_data.batch_indices == 0
        voxels = voxels_data.voxels[first_sample].float()
        # Autoware feeds the voxel coordinates in ``(z, y, x)`` order
        coors = voxels_data.coords[first_sample][:, [2, 1, 0]].int().contiguous()
        num_points_per_voxel = voxels_data.num_points[first_sample].int()
        return voxels, coors, num_points_per_voxel
