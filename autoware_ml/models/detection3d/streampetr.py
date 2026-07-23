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

from autoware_ml.metrics.base import MetricSuite
from autoware_ml.models.base import BaseModel
from autoware_ml.metrics.detection3d.eval_output import detection_eval_output
from autoware_ml.models.common.grid_mask import GridMask
from autoware_ml.models.detection3d.feature_extractors import MultiviewImageFeatureExtractor
from autoware_ml.models.detection3d.task_modules.streaming import (
    inverse_sigmoid,
    pos2posemb3d,
    topk_gather,
    transform_reference_points,
)
from autoware_ml.utils.deploy import ExportSpec


class _StreamPETRImageFeatureExportWrapper(nn.Module):
    """Export the multiview image feature extractor."""

    def __init__(self, model: StreamPETRDetectionModel) -> None:
        """Initialize the image feature export wrapper."""
        super().__init__()
        self.model = model

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Encode multiview images into stride-16 neck features."""
        return self.model.image_feature_extractor(img)


class _StreamPETRPositionEmbeddingExportWrapper(nn.Module):
    """Export the geometry-dependent position embedding.

    The module consumes only calibration inputs so the runtime can compute the
    embedding once and cache it for the stream lifetime.
    """

    def __init__(self, head: nn.Module, num_cams: int, feature_hw: tuple[int, int]) -> None:
        """Initialize the position embedding export wrapper."""
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
        """Build ``pos_embed`` and ``cone`` from static calibration."""
        feature_height, feature_width = self.feature_hw
        shape_source = intrinsics.new_zeros((1, self.num_cams, 1, feature_height, feature_width))
        return self.head.position_embedding(
            shape_source,
            intrinsics,
            lidar2cam=None,
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

    def __init__(self, head: nn.Module) -> None:
        """Initialize the recurrent head export wrapper."""
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
        query_pos_in = query_pos.detach()
        query, query_pos, reference_points, temp_memory, temp_pos = head.temporal_alignment(
            query, query_pos, reference_points
        )
        tgt = query
        rec_ego_pose = (
            torch.eye(4, device=query.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, query.shape[1], 1, 1)
        )

        reference = inverse_sigmoid(reference_points.clone())
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
            cls_logits = cls_branch(normed_query)
            raw_box = reg_branch(normed_query)
            centers = (raw_box[..., :3] + reference[..., :3]).sigmoid()
            outputs_classes.append(cls_logits)
            outputs_coords.append(torch.cat([centers, raw_box[..., 3:]], dim=-1))
            decoder_outputs.append(normed_query)
        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        metric_centers = (
            all_bbox_preds[..., :3] * (head.pc_range[3:6] - head.pc_range[:3]) + head.pc_range[:3]
        )
        all_bbox_preds = torch.cat([metric_centers, all_bbox_preds[..., 3:]], dim=-1)
        outs_dec = torch.stack(decoder_outputs)

        rec_reference_points = all_bbox_preds[-1][..., :3]
        rec_velo = all_bbox_preds[-1][..., -2:]
        rec_memory = outs_dec[-1]
        rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
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
            reference_points,
            tgt,
            temp_memory,
            temp_pos,
            query_pos,
            query_pos_in,
            outs_dec,
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
        bbox_head: nn.Module,
        use_grid_mask: bool = False,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ) -> None:
        """Initialize StreamPETR.

        Args:
            img_backbone: Image backbone.
            img_neck: Image neck.
            bbox_head: Detection head.
            use_grid_mask: Whether to apply grid-mask image augmentation
                during training.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            optimizer_group_overrides: Per-submodule optimizer overrides.
            metrics: Detection metrics accumulated during validation and test.
        """
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_group_overrides=optimizer_group_overrides,
            metrics=metrics,
        )
        self.img_backbone = img_backbone
        self.img_neck = img_neck
        self.bbox_head = bbox_head
        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask()
        self.image_feature_extractor = MultiviewImageFeatureExtractor(
            img_backbone=img_backbone, img_neck=img_neck
        )

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

    def build_eval_output(self, batch: Mapping[str, Any], outputs: Any) -> dict[str, Any]:
        """Decode detections and pair them with ground truth for metrics."""
        return detection_eval_output(self.bbox_head.predict(outputs), batch)

    def build_optimizer_groups(self) -> dict[str, list[nn.Parameter]]:
        """Split parameters into the image backbone and the remaining modules."""
        backbone_params = [p for p in self.img_backbone.parameters() if p.requires_grad]
        backbone_ids = {id(p) for p in backbone_params}
        default_params = [
            p for p in self.parameters() if p.requires_grad and id(p) not in backbone_ids
        ]
        return {"default": default_params, "img_backbone": backbone_params}

    def _extract_img_features(self, img: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        """Encode multiview images into neck features.

        Args:
            img: Multiview image tensor or per-sample image sequence.

        Returns:
            Neck feature tensor consumed by the StreamPETR head.
        """
        return self.image_feature_extractor(img)

    def forward(
        self,
        img: torch.Tensor | Sequence[torch.Tensor],
        camera_intrinsics: Sequence[torch.Tensor] | torch.Tensor | None = None,
        lidar2cam: Sequence[torch.Tensor] | torch.Tensor | None = None,
        lidar2img: Sequence[torch.Tensor] | torch.Tensor | None = None,
        timestamp: Sequence[torch.Tensor] | torch.Tensor | None = None,
        prev_exists: Sequence[torch.Tensor] | torch.Tensor | None = None,
        ego_pose: Sequence[torch.Tensor] | torch.Tensor | None = None,
        ego_pose_inv: Sequence[torch.Tensor] | torch.Tensor | None = None,
        gt_boxes: list[torch.Tensor] | None = None,
        gt_labels: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Run the camera backbone and StreamPETR head.

        Args:
            img: Multiview image tensor or per-sample image sequence.
            camera_intrinsics: Optional camera intrinsic matrices.
            lidar2cam: Optional lidar-to-camera extrinsics.
            lidar2img: Optional lidar-to-image projection matrices.
            timestamp: Optional per-sample frame timestamps.
            prev_exists: Optional stream-continuity mask for temporal memory.
            ego_pose: Optional ego-pose matrices for the current frame.
            ego_pose_inv: Optional inverse ego-pose matrices.
            gt_boxes: Optional per-sample ground-truth boxes for denoising queries.
            gt_labels: Optional per-sample ground-truth labels for denoising queries.
            **kwargs: Additional unused keyword arguments.

        Returns:
            Detection head outputs.
        """
        del kwargs
        image_batch = (
            torch.stack(list(img), dim=0).float() if isinstance(img, (list, tuple)) else img.float()
        )
        if self.use_grid_mask and self.training:
            batch_size, num_cams = image_batch.shape[:2]
            image_batch = self.grid_mask(image_batch.flatten(0, 1)).view(
                batch_size, num_cams, *image_batch.shape[2:]
            )
        img_features = self._extract_img_features(image_batch)
        return self.bbox_head(
            img_features=img_features,
            img=image_batch,
            camera_intrinsics=self._stack_optional_tensor(camera_intrinsics),
            lidar2cam=self._stack_optional_tensor(lidar2cam),
            lidar2img=self._stack_optional_tensor(lidar2img),
            timestamp=self._stack_optional_tensor(timestamp),
            prev_exists=self._stack_optional_tensor(prev_exists),
            ego_pose=self._stack_optional_tensor(ego_pose),
            ego_pose_inv=self._stack_optional_tensor(ego_pose_inv),
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
        )

    def _stack_optional_tensor(
        self,
        value: Sequence[torch.Tensor] | torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Stack list-backed batch metadata into tensors when present."""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value[0], torch.Tensor):
            return torch.stack(list(value), dim=0)
        return torch.as_tensor(value)

    def compute_metrics(
        self,
        batch_inputs_dict: dict[str, Any],
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute StreamPETR training losses.

        Args:
            batch_inputs_dict: Full batch dictionary.
            outputs: Detection head outputs.

        Returns:
            Loss dictionary produced by the detection head.
        """
        return self.bbox_head.loss(
            outputs, batch_inputs_dict["gt_boxes"], batch_inputs_dict["gt_labels"]
        )

    def predict_outputs(
        self, batch_inputs_dict: dict[str, Any], outputs: dict[str, torch.Tensor]
    ) -> Any:
        """Decode predictions for inference.

        Args:
            batch_inputs_dict: Full batch dictionary.
            outputs: Detection head outputs.

        Returns:
            Decoded prediction results.
        """
        del batch_inputs_dict
        return self.bbox_head.predict(outputs)

    def get_log_batch_size(self, batch_inputs_dict: dict[str, Any]) -> int | None:
        """Log the sample count for multiview detection batches."""
        if "gt_boxes" in batch_inputs_dict:
            return len(batch_inputs_dict["gt_boxes"])
        if "img" in batch_inputs_dict:
            return len(batch_inputs_dict["img"])
        return super().get_log_batch_size(batch_inputs_dict)

    def build_export_specs(self, batch_inputs_dict: dict[str, Any]) -> dict[str, ExportSpec]:
        """Build the three-module ONNX export consumed by the runtime node.

        ``extract_img_feat`` encodes multiview images, ``position_embedding``
        turns static calibration into cached geometric embeddings, and
        ``pts_head_memory`` runs the recurrent detection step carrying the
        five temporal memory tensors in and out of the graph.
        """
        head = self.bbox_head
        # The exported modules contain no CUDA-specific ops. Tracing on CPU
        # keeps every graph constant device-neutral, which ONNX constant
        # folding requires.
        img = torch.stack(list(batch_inputs_dict["img"]), dim=0)[:1].float().cpu()
        intrinsics = (
            torch.stack(list(batch_inputs_dict["camera_intrinsics"]), dim=0)[:1].float().cpu()
        )
        lidar2cam = torch.stack(list(batch_inputs_dict["lidar2cam"]), dim=0)[:1].float().cpu()
        img2lidar = torch.inverse(intrinsics @ lidar2cam)
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
                lidar2cam=None,
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
                module=_StreamPETRImageFeatureExportWrapper(export_model),
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
                    "reference_points",
                    "tgt",
                    "temp_memory",
                    "temp_pos",
                    "query_pos",
                    "query_pos_in",
                    "outs_dec",
                ],
            ),
        }
