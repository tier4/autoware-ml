"""Native StreamPETR-style camera 3D detector.

This module contains the high-level StreamPETR detector wrapper and export ABI.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, ClassVar

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
from autoware_ml.metrics.detection3d.eval_output import detection3d_eval_output
from autoware_ml.models.common.multiview_image_encoder import MultiviewImageEncoder
from autoware_ml.models.detection3d.heads.streampetr import StreamPETRHead
from autoware_ml.models.detection3d.task_modules.streaming import (
    compose_poses,
    memory_refresh,
    transform_reference_points,
)
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
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

    ``all_cls_scores`` and ``all_bbox_preds`` hold the predictions of the last
    decoder layer only, the layer :meth:`StreamPETRHead.predict` decodes offline:
    ``(1, num_classes, Q)`` logits and ``(1, 10, Q)`` boxes for the
    ``Q = num_queries + num_propagated`` queries, ``(1, 7, 900)`` and
    ``(1, 10, 900)`` for the seven T4 classes and the 644 + 256 queries of the
    reference recipe. The runtime decodes every row of them, so its
    ``num_proposals`` parameter must be ``Q`` (900) for graphs exported from
    this version; earlier exports stacked every decoder layer (5400 rows with
    six layers), and the runtime published boxes of the intermediate layers too.

    A stream starts with an all-zero bank: the runtime wipes the five tensors
    at start-up and on every reset. The graph recognizes a lane whose bank
    poses are all zero (every written entry carries a pose with a unit last
    row) and seeds it as the head does for ``prev_exists = 0``, with the pseudo
    reference points and the identity pose in the first ``num_propagated``
    entries, so the first frame of a stream predicts what the head predicts.
    Without that, moving the zero bank by the inverse ego pose would put every
    entry at the map origin as seen from the vehicle, kilometers away.
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
        """Run one recurrent detection step over precomputed image tokens.

        The bank comes in aligned to the previous frame and dated by the runtime, so the
        pre-update reduces to moving it into the current lidar frame; the post-update
        emits the untruncated bank for the runtime to truncate and date.
        """
        head = self.head
        # A lane with an all-zero bank starts a stream (see the class docstring): its moved
        # bank is cleared again, so the entries sit at the lidar origin rather than at the
        # map origin, and then seeded as the head seeds a fresh bank.
        prev_exists = (pre_memory_egopose.flatten(1).abs().amax(dim=1) > 0).float()
        head.memory_embedding = memory_refresh(pre_memory_embedding, prev_exists)
        head.memory_timestamp = memory_refresh(pre_memory_timestamp, prev_exists)
        head.memory_velo = memory_refresh(pre_memory_velo, prev_exists)
        head.memory_reference_point, head.memory_egopose = head.reset_state(
            prev_exists,
            memory_refresh(
                transform_reference_points(pre_memory_reference_point, data_ego_pose_inv),
                prev_exists,
            ),
            memory_refresh(
                compose_poses(data_ego_pose_inv.unsqueeze(1), pre_memory_egopose), prev_exists
            ),
        )

        memory, pos_embed = head.embed_tokens(x, pos_embed, cone)
        reference_points, _, _ = head.prepare_for_dn(x.shape[0], None, None)
        all_cls_scores, all_bbox_preds, decoder_outputs = head.decode(
            memory, pos_embed, reference_points, None, sanitize=False
        )
        post_memory = head.push_memory(
            data_ego_pose, all_cls_scores, all_bbox_preds, decoder_outputs
        )
        # Only the last decoder layer leaves the graph (see the class docstring).
        return (
            all_cls_scores[-1:].flatten(0, 2).unsqueeze(0).transpose(2, 1),
            all_bbox_preds[-1:].flatten(0, 2).unsqueeze(0).transpose(2, 1),
            *post_memory,
        )


class StreamPETRDetection3DModel(ModuleBaseModel):
    """Compose a camera-based query detector for 3D object detection.

    The model wraps the image backbone, image neck, and StreamPETR head inside
    the shared Autoware-ML training and deployment interface.
    """

    def __init__(
        self,
        data_preprocessor: DataPreprocessor,
        img_backbone: nn.Module,
        img_neck: nn.Module,
        bbox_head: StreamPETRHead,
        log_dict_configs: LogDictConfigs,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ) -> None:
        """Initialize StreamPETR.

        Args:
            data_preprocessor: Preprocessor for the model inputs.
            img_backbone: Multiview image backbone.
            img_neck: Image neck producing the tokens the head attends to.
            bbox_head: StreamPETR query head.
            log_dict_configs: Logging configuration for training and validation.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            optimizer_group_overrides: Optimizer overrides keyed by the groups of
                :meth:`build_optimizer_groups`.
            scheduler_config: Lightning scheduler metadata such as ``interval``.
            metrics: Detection metrics accumulated during validation and test.
        """
        super().__init__(
            data_preprocessor=data_preprocessor,
            log_dict_configs=log_dict_configs,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_group_overrides=optimizer_group_overrides,
            scheduler_config=scheduler_config,
            metrics=metrics,
        )
        # The attribute keeps the checkpoint keys and the AWML converter prefixes,
        # image_feature_extractor.img_backbone.* and image_feature_extractor.img_neck.*.
        self.image_feature_extractor = MultiviewImageEncoder(
            img_backbone=img_backbone, img_neck=img_neck
        )
        self.bbox_head = bbox_head

    # The memory bank needs lane-contiguous batches.
    requires_streaming: ClassVar[bool] = True

    def _start_streams(self) -> None:
        """Reset the temporal memory: the lanes of a new phase or epoch start new streams.

        Lightning restores the training mode after validation by setting ``training``
        on every module rather than calling ``train()``, so the phase hooks are the
        place to reset, not a ``train()`` override.
        """
        self.bbox_head.reset_memory()

    def on_train_epoch_start(self) -> None:
        """Start new streams for the shuffled scenes of the epoch."""
        self._start_streams()

    def on_validation_epoch_start(self) -> None:
        """Start new streams for the validation scenes, after the metrics are reset."""
        super().on_validation_epoch_start()
        self._start_streams()

    def on_test_epoch_start(self) -> None:
        """Start new streams for the test scenes, after the metrics are reset."""
        super().on_test_epoch_start()
        self._start_streams()

    def on_predict_epoch_start(self) -> None:
        """Start new streams for the predict scenes."""
        self._start_streams()

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run the camera backbone and StreamPETR head on the images of the batch.

        The scene and the time of every frame decide per batch lane whether the head's
        memory carries over from the previous frame, see
        :class:`~autoware_ml.models.detection3d.task_modules.streaming.StreamContinuity`.
        In training the ground truth boxes of the batch seed the denoising queries.

        Args:
            multi_task_batch_inputs: Model inputs holding the images, their calibration and
                the frame metadata of every lane.

        Returns:
            The StreamPETR head outputs.

        Raises:
            ValueError: If the model inputs miss the images or the frame metadata.
        """
        image_data = multi_task_batch_inputs.image_data
        frame_meta = multi_task_batch_inputs.multi_task_gt_batch.frame_meta_batch
        if image_data is None or frame_meta is None:
            raise ValueError("StreamPETR needs the images and the frame metadata of the batch.")
        gt_boxes = gt_labels = None
        gt_detections = multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch
        if gt_detections is not None:
            gt_boxes, gt_labels = _ground_truth(gt_detections)

        img = image_data.images
        img_features = self.image_feature_extractor(img)
        head_outputs = self.bbox_head(
            img_features=img_features,
            image_height=int(img.shape[-2]),
            image_width=int(img.shape[-1]),
            # The batch carries the raw intrinsics and the image augmentation apart; the head
            # and the deployed graph both see the intrinsics of the augmented images.
            camera_intrinsics=image_data.augmented_camera_intrinsics(),
            lidar2img=image_data.lidar2images,
            timestamp=frame_meta.timestamps,
            prev_exists=self.bbox_head.stream_continuity(
                frame_meta.scene_tokens, frame_meta.timestamps
            ),
            ego_pose=frame_meta.ego2globals,
            # The poses and the bank are float32 in the map frame: a stored translation at 65 km
            # resolves to one ulp (4 mm), and the head keeps the geometry at that resolution
            # only because it composes the poses elementwise, in IEEE float32 under any matmul
            # precision (a TF32 GEMM loses tens of meters there). The float64 inverse keeps the
            # LU from adding its own rounding on top.
            ego_pose_inv=torch.linalg.inv(frame_meta.ego2globals.double()).float(),
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
        )
        return ModelOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(streampetr_head_outputs=head_outputs)
        )

    def compute_metrics(
        self, multi_task_batch_inputs: ModelBatchInputs, multi_task_outputs: ModelOutputs
    ) -> dict[str, torch.Tensor]:
        """Compute the head's multi-layer detection and denoising losses.

        Args:
            multi_task_batch_inputs: Model inputs holding the ground truth boxes.
            multi_task_outputs: StreamPETR head outputs.

        Returns:
            Loss dictionary produced by the detection head.

        Raises:
            ValueError: If the batch carries no detection ground truth.
        """
        gt_detections = multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch
        if gt_detections is None:
            raise ValueError("StreamPETR losses need the 3D detection ground truth of the batch.")
        gt_boxes, gt_labels = _ground_truth(gt_detections)
        return self.bbox_head.loss(
            multi_task_outputs.detection3d().streampetr_head(), gt_boxes, gt_labels
        )

    def decode_outputs(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        """Decode the last decoder layer into per-sample detections.

        Args:
            multi_task_batch_inputs: Model inputs of the batch.
            outputs: StreamPETR head outputs.

        Returns:
            Decoded prediction results.
        """
        del multi_task_batch_inputs
        return ModelPredictions(
            detection3d_predictions=self.bbox_head.predict(outputs.detection3d().streampetr_head())
        )

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> dict[str, Any]:
        """Decode detections and pair them with ground truth for metrics."""
        return detection3d_eval_output(
            predictions=self.decode_outputs(batch, outputs), batch_inputs=batch
        )

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
        five temporal memory tensors in and out of the graph and emitting the
        predictions of the last decoder layer.

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
        # The exported modules contain no CUDA-specific ops. Tracing on CPU
        # keeps every graph constant device-neutral, which ONNX constant
        # folding requires. The first sample of the batch traces the graphs.
        img = image_data.images[:1].cpu()
        # The runtime binds the 4x4 intrinsics of the resized images and the inverse of the
        # augmented projection, so both come from the augmented calibration.
        intrinsics = _homogeneous_intrinsics(image_data.augmented_camera_intrinsics()[:1].cpu())
        img2lidar = torch.inverse(image_data.lidar2images[:1].cpu())
        num_cams = img.shape[1]
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

        identity_pose = torch.eye(4).unsqueeze(0)
        memory_len, hidden_dim = export_head.memory_len, export_head.hidden_dim
        memory_args = (
            torch.zeros(1, memory_len, hidden_dim),
            torch.zeros(1, memory_len, 3),
            torch.zeros(1, memory_len, 1),
            torch.zeros(1, memory_len, 4, 4),
            torch.zeros(1, memory_len, 2),
        )

        # A fresh wrapper is in training mode; the exporter restores that mode onto the shared
        # head after tracing, which would leave the decoder dropout active in the wrappers
        # that are run after the export (the export verification). Build them in eval mode.
        position_embedding_wrapper = _StreamPETRPositionEmbeddingExportWrapper(
            export_head, num_cams, feature_hw
        ).eval()
        head_memory_wrapper = _StreamPETRHeadMemoryExportWrapper(export_head).eval()
        return {
            "extract_img_feat": ExportSpec(
                module=export_model.image_feature_extractor,
                args=(img,),
                input_param_names=["img"],
                output_names=["img_feats"],
            ),
            "position_embedding": ExportSpec(
                module=position_embedding_wrapper,
                args=(img_metas_pad, intrinsics, img2lidar),
                input_param_names=["img_metas_pad", "intrinsics", "img2lidar"],
                output_names=["pos_embed", "cone"],
            ),
            "pts_head_memory": ExportSpec(
                module=head_memory_wrapper,
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
