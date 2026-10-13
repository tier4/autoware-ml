"""Native StreamPETR-style camera 3D detector.

This module contains the high-level StreamPETR detector wrapper and export ABI.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.batch.detection3d import Detection3DGTBatch
from autoware_ml.dataclasses.models.detection3d.head_outputs import Detection3DHeadOutputs
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.models.base import BaseModel
from autoware_ml.metrics.detection3d.eval_output import detection_eval_output
from autoware_ml.models.detection3d.feature_extractors import MultiviewImageFeatureExtractor
from autoware_ml.models.detection3d.heads.streampetr import StreamPETRHead, identity_poses
from autoware_ml.models.detection3d.task_modules.streaming import (
    pos2posemb3d,
    topk_gather,
    transform_reference_points,
)
from autoware_ml.types.geometry import Box3DFieldIndex
from autoware_ml.utils.deploy import ExportSpec


def _homogeneous_intrinsics(camera_intrinsics: torch.Tensor) -> torch.Tensor:
    """Pad ``(..., 3, 3)`` intrinsics to the ``(..., 4, 4)`` layout the runtime binds."""
    homogeneous = torch.eye(4, dtype=camera_intrinsics.dtype, device=camera_intrinsics.device)
    homogeneous = homogeneous.expand(*camera_intrinsics.shape[:-2], 4, 4).clone()
    homogeneous[..., :3, :3] = camera_intrinsics
    return homogeneous


def _ground_truth(
    gt_detections: Detection3DGTBatch,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Unpad the boxes and labels of every sample into the head's box layout.

    The head regresses ``[cx, cy, cz, l, w, h, yaw, vx, vy]``; the vertical
    velocity of the collated boxes is not a target.
    """
    gt_boxes = [boxes[:, : Box3DFieldIndex.VELOCITY_Z] for boxes in gt_detections.valid_bboxes_3d()]
    return gt_boxes, gt_detections.valid_labels_3d()


class _StreamPETRPositionEmbeddingExportWrapper(nn.Module):
    """Export the geometry-dependent position embedding.

    The module consumes only calibration inputs so the runtime can compute the
    embedding once and cache it for the stream lifetime.
    """

    def __init__(self, head: StreamPETRHead, num_cams: int, feature_hw: tuple[int, int]) -> None:
        super().__init__()
        self.head = head
        self.num_cams = num_cams
        self.feature_hw = feature_hw

    def forward(
        self,
        img_metas_pad: torch.Tensor,
        intrinsics: torch.Tensor,
        img2lidar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``pos_embed`` and ``cone`` from static calibration.

        ``img_metas_pad`` is ``[image_height, image_width, channels]`` (reference ABI);
        the sizes stay tensors so they remain graph inputs.
        """
        feature_height, feature_width = self.feature_hw
        shape_source = intrinsics.new_zeros((1, self.num_cams, 1, feature_height, feature_width))
        return self.head.position_embedding(
            shape_source,
            intrinsics,
            image_height=img_metas_pad[0],
            image_width=img_metas_pad[1],
            img2lidar=img2lidar,
        )


class _StreamPETRHeadMemoryExportWrapper(nn.Module):
    """Export the memory-recurrent detection head.

    The graph carries the five temporal memory tensors in and out; the runtime
    feeds the first ``memory_len`` entries of the previous frame's post-memory
    back as pre-memory and manages timestamps outside the graph.
    """

    def __init__(self, head: StreamPETRHead) -> None:
        super().__init__()
        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        cone: torch.Tensor,
        data_ego_pose: torch.Tensor,
        data_ego_pose_inv: torch.Tensor,
        pre_memory_embedding: torch.Tensor,
        pre_memory_reference_point: torch.Tensor,
        pre_memory_timestamp: torch.Tensor,
        pre_memory_egopose: torch.Tensor,
        pre_memory_velo: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run one recurrent detection step over precomputed image tokens."""
        head = self.head
        memory_len = head.memory_len

        memory_egopose = data_ego_pose_inv.unsqueeze(1) @ pre_memory_egopose
        memory_reference_point = transform_reference_points(
            pre_memory_reference_point, data_ego_pose_inv
        )
        head.memory_timestamp = pre_memory_timestamp[:, :memory_len]
        head.memory_reference_point = memory_reference_point[:, :memory_len]
        head.memory_embedding = pre_memory_embedding[:, :memory_len]
        head.memory_egopose = memory_egopose[:, :memory_len]
        head.memory_velo = pre_memory_velo[:, :memory_len]

        batch_size, num_cams, channels, feature_height, feature_width = x.shape
        memory = x.permute(0, 1, 3, 4, 2).reshape(
            batch_size, num_cams * feature_height * feature_width, channels
        )
        memory = head.memory_embed(memory)
        memory = head.spatial_alignment(memory, cone)
        pos_embed = head.featurized_pe(pos_embed, memory)

        reference_points = head.reference_points.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_pos = head.query_embedding(pos2posemb3d(reference_points, head.hidden_dim // 2))
        query = torch.zeros_like(query_pos)
        query, query_pos, reference_points, temp_memory, temp_pos = head.temporal_alignment(
            query, query_pos, reference_points
        )
        rec_ego_pose = identity_poses(batch_size, query.shape[1], query.device)

        outputs_classes = []
        outputs_coords = []
        decoder_outputs = []
        for decoder_layer, cls_branch, reg_branch in zip(
            head.decoder, head.cls_branches, head.reg_branches
        ):
            query = decoder_layer(query, memory, query_pos, pos_embed, temp_memory, temp_pos, None)
            # Predictions and the propagated memory consume the post-normed
            # intermediates, exactly as in the head's training forward.
            normed_query = head.post_norm(query)
            outputs_classes.append(cls_branch(normed_query))
            outputs_coords.append(
                head._decode_box_params(reg_branch(normed_query), reference_points)
            )
            decoder_outputs.append(normed_query)
        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)

        rec_reference_points = all_bbox_preds[-1][..., :3]
        rec_velo = all_bbox_preds[-1][..., -2:]
        rec_memory = decoder_outputs[-1]
        rec_score = all_cls_scores[-1].sigmoid().amax(dim=-1, keepdim=True)
        rec_timestamp = torch.zeros_like(rec_score)
        _, topk_indexes = torch.topk(rec_score, head.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes)
        rec_memory = topk_gather(rec_memory, topk_indexes)
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes)

        post_memory_embedding = torch.cat([rec_memory, head.memory_embedding], dim=1)
        post_memory_timestamp = torch.cat([rec_timestamp, head.memory_timestamp], dim=1)
        post_memory_egopose = torch.cat([rec_ego_pose, head.memory_egopose], dim=1)
        post_memory_reference_point = torch.cat(
            [rec_reference_points, head.memory_reference_point], dim=1
        )
        post_memory_velo = torch.cat([rec_velo, head.memory_velo], dim=1)
        post_memory_reference_point = transform_reference_points(
            post_memory_reference_point, data_ego_pose
        )
        post_memory_egopose = data_ego_pose.unsqueeze(1) @ post_memory_egopose

        return (
            all_cls_scores.flatten(0, 2).unsqueeze(0).transpose(2, 1),
            all_bbox_preds.flatten(0, 2).unsqueeze(0).transpose(2, 1),
            post_memory_embedding,
            post_memory_reference_point,
            post_memory_timestamp,
            post_memory_egopose,
            post_memory_velo,
        )


class StreamPETRDetectionModel(BaseModel):
    """Compose a camera-based query detector for 3D object detection.

    The model wraps the image backbone, image neck, and StreamPETR head inside
    the shared Autoware-ML training and deployment interface.
    """

    def __init__(
        self,
        img_backbone: nn.Module,
        img_neck: nn.Module,
        bbox_head: StreamPETRHead,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ) -> None:
        """Initialize StreamPETR.

        Args:
            img_backbone: Multiview image backbone.
            img_neck: Image neck producing the tokens the head attends to.
            bbox_head: StreamPETR query head.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            optimizer_group_overrides: Optimizer overrides keyed by the groups of
                :meth:`build_optimizer_groups`.
            scheduler_config: Lightning scheduler metadata such as ``interval``.
            metrics: Detection metrics accumulated during validation and test.
        """
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_group_overrides=optimizer_group_overrides,
            scheduler_config=scheduler_config,
            metrics=metrics,
        )
        self.image_feature_extractor = MultiviewImageFeatureExtractor(
            img_backbone=img_backbone, img_neck=img_neck
        )
        self.bbox_head = bbox_head

    def setup(self, stage: str) -> None:
        """Require a streaming datamodule; the memory bank needs lane-contiguous batches."""
        super().setup(stage)
        if not getattr(self.trainer.datamodule, "streaming", False):
            raise ValueError(
                "StreamPETR carries temporal memory across batches, so the datamodule must "
                "stream scene-contiguous frames; set datamodule.streaming to true."
            )

    def train(self, mode: bool = True) -> StreamPETRDetectionModel:
        """Reset the temporal memory whenever the train/eval phase changes.

        Streams are not continuous across phase switches (sanity validation,
        epoch-end validation), so carrying the bank over would feed stale
        state into the next phase.
        """
        super().train(mode)
        self.bbox_head.reset_memory()
        return self

    def forward_inputs(self, batch_inputs: ModelBatchInputs) -> dict[str, Any]:
        """Pick the images, their calibration and the stream metadata of every frame.

        Args:
            batch_inputs: Model inputs holding the images and the frame metadata, and in
                training the boxes that seed the denoising queries.

        Returns:
            The keyword arguments of :meth:`forward`.

        Raises:
            ValueError: If the model inputs miss the images or the frame metadata.
        """
        image_data = batch_inputs.image_data
        frame_meta = batch_inputs.multi_task_gt_batch.frame_meta_batch
        if image_data is None or frame_meta is None:
            raise ValueError("StreamPETR needs the images and the frame metadata of the batch.")
        inputs: dict[str, Any] = {
            "img": image_data.images,
            "camera_intrinsics": image_data.camera_intrinsics,
            "lidar2img": image_data.lidar2images,
            "timestamp": frame_meta.timestamps,
            "scenario_ids": frame_meta.scenario_ids,
            # The poses carry kilometer-scale translations, so they are inverted in float64.
            "ego_pose": frame_meta.ego2globals,
            "ego_pose_inv": torch.linalg.inv(frame_meta.ego2globals.double()).float(),
        }
        gt_detections = batch_inputs.multi_task_gt_batch.detection3d_gt_batch
        if gt_detections is not None:
            inputs["gt_boxes"], inputs["gt_labels"] = _ground_truth(gt_detections)
        return inputs

    def forward(
        self,
        img: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        lidar2img: torch.Tensor,
        timestamp: torch.Tensor,
        scenario_ids: Sequence[str],
        ego_pose: torch.Tensor,
        ego_pose_inv: torch.Tensor,
        gt_boxes: list[torch.Tensor] | None = None,
        gt_labels: list[torch.Tensor] | None = None,
    ) -> ModelOutputs:
        """Run the camera backbone and StreamPETR head.

        Args:
            img: Images ``(batch, num_cams, 3, H, W)``.
            camera_intrinsics: Augmented intrinsics ``(batch, num_cams, 3, 3)``.
            lidar2img: Augmented lidar-to-image matrices ``(batch, num_cams, 4, 4)``.
            timestamp: Sample time of every frame in seconds, float64.
            scenario_ids: Scenario of every frame.
            ego_pose: Lidar-to-global pose of every frame ``(batch, 4, 4)``.
            ego_pose_inv: Inverse of ``ego_pose``.
            gt_boxes: Metric boxes seeding the denoising queries in training.
            gt_labels: Labels of ``gt_boxes``.

        Returns:
            The StreamPETR head outputs.
        """
        image_batch = img.float()
        img_features = self.image_feature_extractor(image_batch)
        head_outputs = self.bbox_head(
            img_features=img_features,
            image_height=int(image_batch.shape[-2]),
            image_width=int(image_batch.shape[-1]),
            camera_intrinsics=camera_intrinsics,
            lidar2img=lidar2img,
            timestamp=timestamp,
            scenario_ids=scenario_ids,
            ego_pose=ego_pose,
            ego_pose_inv=ego_pose_inv,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
        )
        return ModelOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=None,
                transfusion_head_outputs=None,
                streampetr_head_outputs=head_outputs,
            )
        )

    def compute_metrics(
        self, batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> dict[str, torch.Tensor]:
        """Compute the head's multi-layer detection and denoising losses.

        Args:
            batch_inputs: Model inputs holding the ground truth boxes.
            outputs: StreamPETR head outputs.

        Returns:
            Loss dictionary produced by the detection head.

        Raises:
            ValueError: If the batch carries no detection ground truth.
        """
        gt_detections = batch_inputs.multi_task_gt_batch.detection3d_gt_batch
        if gt_detections is None:
            raise ValueError("StreamPETR losses need the 3D detection ground truth of the batch.")
        gt_boxes, gt_labels = _ground_truth(gt_detections)
        return self.bbox_head.loss(outputs.detection3d().streampetr_head(), gt_boxes, gt_labels)

    def predict_outputs(
        self, batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        """Decode the last decoder layer into per-sample detections.

        Args:
            batch_inputs: Model inputs of the batch.
            outputs: StreamPETR head outputs.

        Returns:
            Decoded prediction results.
        """
        del batch_inputs
        return ModelPredictions(
            detection3d_predictions=self.bbox_head.predict(outputs.detection3d().streampetr_head())
        )

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> dict[str, Any]:
        """Decode detections and pair them with ground truth for metrics."""
        return detection_eval_output(self.predict_outputs(batch, outputs), batch)

    def build_optimizer_groups(self) -> dict[str, list[nn.Parameter]]:
        """Split parameters into the image backbone and the remaining modules."""
        backbone_params = [
            p for p in self.image_feature_extractor.img_backbone.parameters() if p.requires_grad
        ]
        backbone_ids = {id(p) for p in backbone_params}
        default_params = [
            p for p in self.parameters() if p.requires_grad and id(p) not in backbone_ids
        ]
        return {"default": default_params, "img_backbone": backbone_params}

    def build_export_specs(self, batch_inputs: ModelBatchInputs) -> dict[str, ExportSpec]:
        """Build the three-module ONNX export consumed by the runtime node.

        ``extract_img_feat`` encodes multiview images, ``position_embedding``
        turns static calibration into cached geometric embeddings, and
        ``pts_head_memory`` runs the recurrent detection step carrying the
        five temporal memory tensors in and out of the graph.

        Args:
            batch_inputs: Batched model inputs; the first sample traces the graphs.

        Returns:
            Ordered mapping of module name to export specification.

        Raises:
            ValueError: If the model inputs carry no images.
        """
        image_data = batch_inputs.image_data
        if image_data is None:
            raise ValueError("The StreamPETR export needs the images of the batch.")
        head = self.bbox_head
        # The exported modules contain no CUDA-specific ops. Tracing on CPU
        # keeps every graph constant device-neutral, which ONNX constant
        # folding requires.
        img = image_data.images[:1].float().cpu()
        # The runtime binds 4x4 intrinsics and the inverse of the augmented projection.
        intrinsics = _homogeneous_intrinsics(image_data.camera_intrinsics[:1].float().cpu())
        img2lidar = torch.inverse(image_data.lidar2images[:1].float().cpu())
        batch_size, num_cams = img.shape[:2]
        image_height, image_width = int(img.shape[-2]), int(img.shape[-1])

        export_model = deepcopy(self).cpu().eval()
        export_head = export_model.bbox_head
        with torch.no_grad():
            img_feats = export_model.image_feature_extractor(img)
            feature_hw = (int(img_feats.shape[-2]), int(img_feats.shape[-1]))
            img_metas_pad = img.new_tensor([image_height, image_width, 3])
            pos_embed, cone = export_head.position_embedding(
                img_feats,
                intrinsics,
                image_height=image_height,
                image_width=image_width,
                img2lidar=img2lidar,
            )

        identity_pose = (
            torch.eye(4, dtype=torch.float32, device=img.device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        memory_args = (
            img.new_zeros((batch_size, head.memory_len, head.hidden_dim)),
            img.new_zeros((batch_size, head.memory_len, 3)),
            img.new_zeros((batch_size, head.memory_len, 1)),
            img.new_zeros((batch_size, head.memory_len, 4, 4)),
            img.new_zeros((batch_size, head.memory_len, 2)),
        )

        return {
            "extract_img_feat": ExportSpec(
                module=export_model.image_feature_extractor,
                args=(img,),
                input_param_names=["img"],
                output_names=["img_feats"],
            ),
            "position_embedding": ExportSpec(
                module=_StreamPETRPositionEmbeddingExportWrapper(export_head, num_cams, feature_hw),
                args=(img_metas_pad, intrinsics, img2lidar),
                input_param_names=["img_metas_pad", "intrinsics", "img2lidar"],
                output_names=["pos_embed", "cone"],
            ),
            "pts_head_memory": ExportSpec(
                module=_StreamPETRHeadMemoryExportWrapper(export_head),
                args=(img_feats, pos_embed, cone, identity_pose, identity_pose, *memory_args),
                input_param_names=[
                    "x",
                    "pos_embed",
                    "cone",
                    "data_ego_pose",
                    "data_ego_pose_inv",
                    "pre_memory_embedding",
                    "pre_memory_reference_point",
                    "pre_memory_timestamp",
                    "pre_memory_egopose",
                    "pre_memory_velo",
                ],
                output_names=[
                    "all_cls_scores",
                    "all_bbox_preds",
                    "post_memory_embedding",
                    "post_memory_reference_point",
                    "post_memory_timestamp",
                    "post_memory_egopose",
                    "post_memory_velo",
                ],
            ),
        }
