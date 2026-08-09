"""Detection heads used by CenterPoint-style models.

This module implements dense prediction heads, target generation, decoding,
and training losses used by CenterPoint-style detectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from types import MappingProxyType

from jaxtyping import Bool, Float32, Int32, Int64
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.dataclasses.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    CenterHeadOutputs,
)
from autoware_ml.dataclasses.detection3d.head_targets import CenterHeadTargets
from autoware_ml.dataclasses.multi_task_predictions import MultiTaskPredictions
from autoware_ml.losses.detection3d.gaussian_focal import GaussianFocalLoss
from autoware_ml.models.common.layers.conv import ConvModule
from autoware_ml.models.detection3d.task_modules.heatmap import (
    batch_circle_nms,
    vectorize_gaussian_radii,
    create_gaussian_heatmaps,
)
from autoware_ml.types.geometry import Box3DFieldIndex


def _gather_feat(
    features: Float32[torch.Tensor, "batch_size height*width channels"],
    indices: Int64[torch.Tensor, "batch_size max_num_bboxes"],
) -> Float32[torch.Tensor, "batch_size max_num_bboxes channels"]:
    """Gather flattened features at the requested indices."""
    channels = features.shape[-1]
    expanded_indices = indices.unsqueeze(-1).expand(*indices.shape, channels)
    return features.gather(dim=1, index=expanded_indices)


def _transpose_and_gather_feat(
    features: Float32[torch.Tensor, "batch_size channels height width"],
    indices: Int64[torch.Tensor, "batch_size max_num_bboxes"],
) -> Float32[torch.Tensor, "batch_size height*width channels"]:
    """Transpose a feature map and gather flattened features."""
    features = features.permute(0, 2, 3, 1).contiguous()
    features = features.view(features.shape[0], -1, features.shape[-1])
    return _gather_feat(features, indices)


class CenterHead(nn.Module):
    """Predict dense heatmaps and regression maps for CenterPoint.

    The head uses a shared BEV tower followed by lightweight prediction
    branches for heatmap, center offsets, dimensions, rotation, and velocity.
    It also owns the CenterPoint target generation, loss computation, and
    decode logic so the model wrapper stays reusable and task-agnostic.
    """

    def __init__(
        self,
        in_channels: int,
        class_names: Sequence[str],
        shared_channels: int,
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        out_size_factor: int,
        min_radius: int,
        score_threshold: float,
        post_max_size: int,
        nms_min_radius: float,
        gaussian_overlap: float = 0.1,
        loss_bbox_weight: float = 0.25,
        heatmap_init_bias: float = -2.19,
        use_velocity: bool = True,
    ) -> None:
        """Initialize the CenterPoint head.

        Args:
            in_channels: Input feature channels.
            class_names: Sequence of class names.
            num_classes: Number of detection classes.
            shared_channels: Channel count for the shared tower.
            point_cloud_range: Detector point-cloud range.
            voxel_size: Voxel size used by preprocessing.
            out_size_factor: Downsampling factor between BEV cells and head outputs.
            min_radius: Minimum Gaussian radius for heatmap targets.
            score_threshold: Score threshold applied during decoding.
            post_max_size: Maximum number of predictions kept after decoding.
            nms_min_radius: Minimum center distance used by circle NMS.
            gaussian_overlap: Minimum Gaussian overlap with the target box.
            loss_bbox_weight: Weight applied to the box regression loss.
            heatmap_init_bias: Initial bias used by the heatmap prediction branch.
            use_velocity: Whether to predict velocity components.
        """
        super().__init__()
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.out_size_factor = out_size_factor
        self.min_radius = min_radius
        self.score_threshold = score_threshold
        self.post_max_size = post_max_size
        self.nms_min_radius = nms_min_radius
        self.gaussian_overlap = gaussian_overlap
        self.loss_bbox_weight = loss_bbox_weight
        self.heatmap_init_bias = heatmap_init_bias
        self.use_velocity = use_velocity
        self.box_code_size = 10 if use_velocity else 8

        self.shared_conv = ConvModule(in_channels, shared_channels)
        self.heatmap = self._build_head(
            shared_channels, self.num_classes, init_bias=heatmap_init_bias
        )
        self.regs = self._build_head(shared_channels, 2)
        self.height = self._build_head(shared_channels, 1)
        self.dim = self._build_head(shared_channels, 3)
        self.rot = self._build_head(shared_channels, 2)
        self.vel = self._build_head(shared_channels, 2) if use_velocity else None

        self.loss_heatmap = GaussianFocalLoss()
        self.loss_bbox = nn.L1Loss(reduction="none")

    def _build_head(
        self, in_channels: int, out_channels: int, init_bias: float | None = None
    ) -> nn.Sequential:
        """Build one CenterPoint prediction branch."""
        head = nn.Sequential(
            ConvModule(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        if init_bias is not None:
            nn.init.constant_(head[-1].bias, init_bias)  # type: ignore
        return head

    def forward(
        self, x: Float32[torch.Tensor, "batch_size neck_feature_channels height width"]
    ) -> CenterHeadOutputs:
        """Predict dense heatmap and regression maps."""
        shared = self.shared_conv(x)  # (Batch_size, shared_channels, height, width)
        heatmaps = self.heatmap(shared)  # (Batch_size, num_classes, height, width)
        centers = self.regs(shared)  # (Batch_size, 2, height, width)
        heights = self.height(shared)  # (Batch_size, 1, height, width)
        dims = self.dim(shared)  # (Batch_size, 3, height, width)
        rots = self.rot(shared)  # (Batch_size, 2, height, width)
        vels = (
            self.vel(shared) if self.vel is not None else None
        )  # (Batch_size, 2, height, width) or None

        return CenterHeadOutputs(
            heatmaps=heatmaps, centers=centers, heights=heights, dims=dims, rots=rots, vels=vels
        )

    def get_targets(
        self,
        gt_bboxes_3d: Float32[torch.Tensor, "batch_size max_num_3d_gt_bboxes num_Box3DFieldIndex"],
        gt_labels_3d: Float32[torch.Tensor, "batch_size max_num_3d_gt_bboxes"],
        gt_valid_bboxes: Int32[torch.Tensor, " batch_size"],
        feature_map_size: tuple[int, int],
        device: torch.device,
    ) -> CenterHeadTargets:
        """Build heatmap and regression targets for one batch."""
        batch_size = len(gt_bboxes_3d)
        max_num_bboxes = gt_bboxes_3d.shape[1]
        feature_height, feature_width = feature_map_size

        # Movement of tensors to the correct device and type
        # Get only the first K params for ground truths
        gt_bboxes_3d[:, :, : self.box_code_size] = gt_bboxes_3d[:, :, : self.box_code_size].to(
            device=device
        )
        gt_labels_3d = gt_labels_3d.to(device=device, dtype=torch.long)
        gt_valid_bboxes = gt_valid_bboxes.to(device=device)

        # Vectorization implementation instead of for-loops
        center_x = (
            (gt_bboxes_3d[:, :, Box3DFieldIndex.X] - self.point_cloud_range[0])
            / self.voxel_size[0]
            / self.out_size_factor
        )
        center_y = (
            (gt_bboxes_3d[:, :, Box3DFieldIndex.Y] - self.point_cloud_range[1])
            / self.voxel_size[1]
            / self.out_size_factor
        )

        # (batch_size, max_num_bboxes) boolean mask for valid boxes based on the distance
        valid_distance_masks = (
            (center_x >= 0)
            & (center_x < feature_width)
            & (center_y >= 0)
            & (center_y < feature_height)
        )

        # (batch_size, max_num_bboxes) boolean mask for valid boxes based on the number of valid boxes per sample
        # (max_num_bboxes) -> (1, max_num_bboxes) -> (batch_size, max_num_bboxes) < gt_valid_bboxes.unsqueeze(1) (batch_size, 1)
        # -> (batch_size, max_num_bboxes)
        valid_num_bboxes_masks = torch.arange(max_num_bboxes, device=device).unsqueeze(0).expand(
            batch_size, -1
        ) < gt_valid_bboxes.unsqueeze(1)
        # (batch_size, max_num_bboxes) boolean mask for valid boxes based on both distance and number of valid boxes
        valid_bbox_masks = valid_distance_masks & valid_num_bboxes_masks

        lengths = (
            gt_bboxes_3d[:, :, Box3DFieldIndex.LENGTH] / self.voxel_size[0] / self.out_size_factor
        )
        widths = (
            gt_bboxes_3d[:, :, Box3DFieldIndex.WIDTH] / self.voxel_size[1] / self.out_size_factor
        )

        # (batch_size, max_num_bboxes)
        gaussian_radii = vectorize_gaussian_radii(
            widths=lengths,
            heights=widths,
            min_overlap=self.gaussian_overlap,
        ).to(device)
        # Clamp the Gaussian radii to ensure they are at least the minimum radius
        gaussian_radii = torch.clamp(gaussian_radii, min=self.min_radius)

        center = torch.stack((center_x, center_y), dim=-1)
        # (batch_size, max_num_bboxes, 2)
        center_int = center.floor().to(torch.long)
        heatmaps = create_gaussian_heatmaps(
            heatmap_width=feature_width,
            heatmap_height=feature_height,
            num_classes=self.num_classes,
            centers=center_int,
            gaussian_radii=gaussian_radii.long(),
            gt_bboxes_labels=gt_labels_3d,
            valid_masks=valid_bbox_masks,
            device=device,
        )
        # Center targets are translations/offsets from their corresponding bev grid cell position
        center_targets = torch.stack(
            (center_x - center_int[:, :, 0].floor(), center_y - center_int[:, :, 1].floor()), dim=-1
        )
        # Convert to log-space for dimension targets to stabilize training
        dim_targets = gt_bboxes_3d[:, :, Box3DFieldIndex.LENGTH : Box3DFieldIndex.HEIGHT + 1].log()
        heading_targets = torch.stack(
            (
                torch.sin(gt_bboxes_3d[:, :, Box3DFieldIndex.YAW]),
                torch.cos(gt_bboxes_3d[:, :, Box3DFieldIndex.YAW]),
            ),
            dim=-1,
        )
        height_targets = gt_bboxes_3d[:, :, Box3DFieldIndex.Z].unsqueeze(-1)

        # (batch_size, max_num_bboxes, 2 + 1 + 3 + 2) if not velocity else
        # (batch_size, max_num_bboxes, 2 + 1 + 3 + 2 + 2)
        reg_targets = torch.cat(
            [center_targets, height_targets, dim_targets, heading_targets],
            dim=-1,
        )

        if self.use_velocity:
            vel_targets = gt_bboxes_3d[
                :, :, Box3DFieldIndex.VELOCITY_X : Box3DFieldIndex.VELOCITY_Y + 1
            ]
            reg_targets = torch.cat([reg_targets, vel_targets], dim=-1)

        # (batch_size, max_num_bboxes) -> (batch_size, max_num_bboxes)
        reg_indices = (center_int[:, :, 1] * feature_width + center_int[:, :, 0]).long()
        return CenterHeadTargets(
            heatmaps=heatmaps,
            reg_targets=reg_targets,
            valid_masks=valid_bbox_masks,
            reg_indices=reg_indices,
        )

    def loss(
        self,
        outputs: Detection3DHeadOutputs,
        gt_bboxes_3d: Float32[torch.Tensor, "batch_size max_num_3d_gt_bboxes num_Box3DFieldIndex"],
        gt_labels_3d: Float32[torch.Tensor, "batch_size max_num_3d_gt_bboxes"],
        gt_valid_bboxes: Int32[torch.Tensor, " batch_size"],
    ) -> MappingProxyType[str, torch.Tensor]:
        """
        Compute CenterPoint heatmap and box losses.

        Args:
            gt_bboxes_3d: Ground truth 3D bounding boxes for the batch.
            gt_labels_3d: Ground truth class labels for the 3D bounding boxes.
            gt_valid_bboxes: Number of valid bounding boxes for each sample in the batch.
            outputs: CenterHeadOutputs containing the predicted heatmap and regression maps.

        Returns:
            MappingProxyType[str, torch.Tensor]: A read-only dictionary containing the total loss,
                heatmap loss, and box regression loss.

        Raises:
            ValueError: If the outputs hold no CenterHeadOutputs, or if a valid ground truth box
                yields a non-finite regression target outside of the velocity channels.
        """
        if outputs.center_head_outputs is None:
            raise ValueError(
                "CenterHeadOutputs must be provided in Detection3DOutputs for loss computation."
            )

        output_heatmaps = outputs.center_head_outputs.heatmaps
        heatmap_size = (int(output_heatmaps.shape[-2]), int(output_heatmaps.shape[-1]))
        targets = self.get_targets(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            feature_map_size=heatmap_size,
            gt_valid_bboxes=gt_valid_bboxes,
            device=output_heatmaps.device,
        )
        loss_heatmap = self.loss_heatmap(output_heatmaps, targets.heatmaps)

        bbox_predictions = [
            outputs.center_head_outputs.centers,
            outputs.center_head_outputs.heights,
            outputs.center_head_outputs.dims,
            outputs.center_head_outputs.rots,
        ]
        if self.use_velocity and outputs.center_head_outputs.vels is not None:
            bbox_predictions.append(outputs.center_head_outputs.vels)

        bbox_predictions = torch.cat(bbox_predictions, dim=1)

        # Gather the predicted bounding box parameters across channels at the target indices
        # (batch_size, channels, height, width) -> (batch_size, max_num_bboxes, output_channels)
        flatten_bbox_predictions = _transpose_and_gather_feat(bbox_predictions, targets.reg_indices)
        # (batch_size, max_num_bboxes) -> (batch_size, max_num_bboxes, output_channels)
        bbox_valid_masks = (
            targets.valid_masks.unsqueeze(-1).expand_as(flatten_bbox_predictions).float()
        )

        # Boxes that do not contribute to the loss are padding rows of zeros, whose dimensions
        # encode to -inf. Masking them out is not enough because inf/nan * 0 stays nan, so their
        # targets have to be neutralized before the L1 loss sees them.
        # (batch_size, max_num_bboxes, output_channels)
        reg_targets = torch.where(targets.valid_masks.unsqueeze(-1), targets.reg_targets, 0.0)

        # Only the velocity channels of a valid box are allowed to be non-finite, they are unknown
        # for objects the annotation pipeline could not track, so they are dropped from the loss
        # and zeroed out for the same reason. A non-finite target on any other channel is corrupt
        # ground truth and is left to surface as a non-finite loss.
        if self.use_velocity:
            num_geometry_channels = self.box_code_size - 2
            # (batch_size, max_num_bboxes, 2)
            velocity_finite_masks = torch.isfinite(reg_targets[:, :, num_geometry_channels:])
            bbox_valid_masks[:, :, num_geometry_channels:] *= velocity_finite_masks.float()
            reg_targets[:, :, num_geometry_channels:] = torch.where(
                velocity_finite_masks, reg_targets[:, :, num_geometry_channels:], 0.0
            )

        bbox_losses = self.loss_bbox(flatten_bbox_predictions, reg_targets) * bbox_valid_masks

        # Average over the number of valid bounding boxes and avoid division by zero
        bbox_losses = self.loss_bbox_weight * (
            bbox_losses.sum() / bbox_valid_masks.sum().clamp_min(1.0)
        )
        total_loss = loss_heatmap + bbox_losses
        # MappingProxyType is used to create a read-only dictionary for the loss outputs
        return MappingProxyType(
            {"loss": total_loss, "loss_heatmap": loss_heatmap, "loss_bbox": bbox_losses}
        )

    def _decode_regression_outputs(
        self,
        center_head_outputs: CenterHeadOutputs,
        flatten_indices: Int64[torch.Tensor, "batch_size max_num_bboxes"],
        width: int,
    ) -> Float32[torch.Tensor, "batch_size num_classes*max_num_bboxes box_code_size"]:
        """
        Decode the regression outputs to convert it to physical coordinates
        from the CenterHeadOutputs.
        Args:
          center_head_outputs: Outputs from the CenterHead head.
          flatten_indices: Flattened indices to gather the regression outputs.
          batch_size: Batch size of the input.
          width: Width of the feature map.
          feature_map_size: Tuple of (height, width) of the feature map.

        Returns:
          bbox_predictions: Decoded bounding box predictions.
        """
        ys = torch.div(flatten_indices, width, rounding_mode="floor")
        xs = flatten_indices % width

        # flatten_indices holds positions along the flattened feature map, so each map has to
        # be gathered along its height*width axis per sample.
        # (batch_size, 2, height, width) -> (batch_size, num_classes*max_num_bboxes, 2)
        centers = _transpose_and_gather_feat(center_head_outputs.centers, flatten_indices)
        # (batch_size, num_classes*max_num_bboxes, 1)
        heights = _transpose_and_gather_feat(center_head_outputs.heights, flatten_indices)
        # (batch_size, num_classes*max_num_bboxes, 3)
        dims = _transpose_and_gather_feat(center_head_outputs.dims, flatten_indices)
        # Convert log-dimensions back to actual dimensions
        dims = dims.exp()
        # (batch_size, num_classes*max_num_bboxes, 2)
        rots = _transpose_and_gather_feat(center_head_outputs.rots, flatten_indices)
        vels = center_head_outputs.vels if self.use_velocity else None
        if vels is not None:
            # (batch_size, num_classes*max_num_bboxes, 2)
            vels = _transpose_and_gather_feat(vels, flatten_indices)

        # Compute yaws with atan2
        # (batch_size, num_classes*max_num_bboxes, 1)
        batch_yaws = torch.atan2(rots[:, :, 0], rots[:, :, 1]).unsqueeze(-1)
        # Add center translation offsets to their x and y grid and convert them from bev-grid representation to the lidar physical representation
        # (batch_size, num_classes*max_num_bboxes, 1)
        batch_xs = (xs.to(centers.dtype) + centers[:, :, 0]).unsqueeze(
            2
        ) * self.out_size_factor * self.voxel_size[0] + self.point_cloud_range[0]
        batch_ys = (ys.to(centers.dtype) + centers[:, :, 1]).unsqueeze(
            2
        ) * self.out_size_factor * self.voxel_size[1] + self.point_cloud_range[1]

        # (1+1+1+3+1) = 7 or (1+1+1+3+1+2) = 9
        bboxes_predictions = [batch_xs, batch_ys, heights, dims, batch_yaws]
        if vels is not None:
            bboxes_predictions.append(vels)
        # (batch_size, num_classes*max_num_bboxes, 7 or 9)
        bboxes_predictions = torch.cat(bboxes_predictions, dim=2)

        assert bboxes_predictions.shape[2] == (self.box_code_size - 1), (
            f"Expected bboxes_predictions to have shape[2] == {self.box_code_size - 1}, "
            f"but got {bboxes_predictions.shape[2]}"
        )
        return bboxes_predictions

    def _filter_bbox_predictions(
        self,
        flatten_bboxes_predictions: Float32[
            torch.Tensor, "batch_size num_classes*max_num_bboxes box_code_size"
        ],
        scores: Float32[torch.Tensor, "batch_size num_classes max_num_bboxes"],
        class_ids: Int64[torch.Tensor, "batch_size num_classes max_num_bboxes"],
        keep_masks: Bool[torch.Tensor, "batch_size num_classes max_num_bboxes"],
        max_num_bboxes: int,
        batch_size: int,
    ) -> MultiTaskPredictions:
        """
        Filter the predictions based on the keep_masks and return a MultiTaskPredictions object.
        """
        # (batch_size, num_classes, max_num_bboxes) -> (batch_size, num_classes*max_num_bboxes)
        flatten_keep_masks = keep_masks.reshape(batch_size, -1)
        flatten_scores = scores.reshape(batch_size, -1)
        flatten_class_ids = class_ids.reshape(batch_size, -1)

        # Each sample keeps a different number of boxes, sinking the suppressed scores instead keeps
        # the selection batched, and the survivors are recovered from the keep mask further down.
        # (batch_size, num_classes*max_num_bboxes)
        masked_flatten_scores = flatten_scores.masked_fill(~flatten_keep_masks, -torch.inf)

        # Rank what survived NMS across all classes and cap the sample at max_num_bboxes.
        # (batch_size, num_topk_indices)
        num_topk_indices = min(max_num_bboxes, masked_flatten_scores.shape[1])
        keep_flatten_scores, topk_indices = torch.topk(
            masked_flatten_scores, k=num_topk_indices, largest=True, sorted=True, dim=1
        )
        # (batch_size, num_topk_indices)
        keep_flatten_class_ids = torch.gather(flatten_class_ids, dim=1, index=topk_indices)
        # A slot is only a real detection when the box behind it survived NMS. Samples with
        # fewer survivors than num_topk_indices pad the tail with suppressed slots.
        # (batch_size, num_topk_indices)
        topk_keep_masks = torch.gather(flatten_keep_masks, dim=1, index=topk_indices)

        # (batch_size, num_topk_indices, box_code_size)
        code_size = flatten_bboxes_predictions.shape[2]
        keep_flatten_bbox_predictions = torch.gather(
            flatten_bboxes_predictions,
            dim=1,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, code_size),
        )

        # Iterate over the batch and create a list of Detection3dPredictions for each sample.
        # The ragged padding is dropped here, where per-sample tensors are allowed to differ.
        detection3d_predictions = []
        for batch_index in range(batch_size):
            sample_keep_masks = topk_keep_masks[batch_index]
            detection3d_predictions.append(
                Detection3DSamplePredictions(
                    bboxes_3d=keep_flatten_bbox_predictions[batch_index][sample_keep_masks],
                    scores_3d=keep_flatten_scores[batch_index][sample_keep_masks],
                    labels_3d=keep_flatten_class_ids[batch_index][sample_keep_masks],
                )
            )

        return MultiTaskPredictions(detection3d_predictions=detection3d_predictions)

    def _decode_heatmap_outputs(
        self, center_head_outputs: CenterHeadOutputs
    ) -> Float32[torch.Tensor, "batch_size num_classes height width"]:
        """
        Decode the heatmap outputs to apply sigmoid activation and non-maximum suppression.

        Args:
            center_head_outputs: Outputs from the CenterHead head.

        Returns:
            heatmaps: Decoded heatmaps after applying sigmoid and NMS.
        """
        heatmaps = center_head_outputs.heatmaps.sigmoid()
        pooled = F.max_pool2d(heatmaps, kernel_size=3, stride=1, padding=1)
        heatmaps = heatmaps * (pooled == heatmaps)
        return heatmaps

    def decode_outputs(self, outputs: Detection3DHeadOutputs) -> MultiTaskPredictions:
        """
        Decode dense head outputs into 3D boxes, scores, and labels.
        """
        if outputs.center_head_outputs is None:
            raise ValueError(
                "CenterHeadOutputs must be provided in Detection3DOutputs for centerhead decoding."
            )

        heatmaps = self._decode_heatmap_outputs(outputs.center_head_outputs)
        batch_size, num_classes, height, width = heatmaps.shape
        max_num_bboxes = min(self.post_max_size, height * width)

        # (batch_size, num_classes, height*width)
        batch_scores = heatmaps.reshape(batch_size, num_classes, -1)

        # Get the top-k scores and their corresponding indices for each class in the batch
        top_scores, top_indices = batch_scores.topk(
            k=max_num_bboxes, dim=2
        )  # (batch_size, num_classes, max_num_bboxes)

        # (num_classes) -> (1, num_classes) -> (1, num_classes, 1) -> (batch_size, num_classes, max_num_bboxes)
        class_ids = (
            torch.arange(num_classes, device=heatmaps.device, dtype=torch.long)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand_as(top_indices)
        )
        # (batch_size, num_classes*max_num_bboxes)
        flatten_indices = top_indices.reshape(batch_size, -1)

        # (batch_size, num_classes*max_num_bboxes, box_code_size)
        flatten_bboxes_predictions = self._decode_regression_outputs(
            center_head_outputs=outputs.center_head_outputs,
            flatten_indices=flatten_indices,
            width=width,
        )
        valid_bboxes_masks = top_scores > self.score_threshold
        # batch_circle_nms works per class row, so the flattened box axis is split back into
        # (num_classes, max_num_bboxes) to line up with the scores and class ids.
        # (batch_size, num_classes*max_num_bboxes, box_code_size)
        #   -> (batch_size, num_classes, max_num_bboxes, 2)
        bboxes_centers = flatten_bboxes_predictions[:, :, :2].reshape(
            batch_size, num_classes, max_num_bboxes, 2
        )
        # (batch_size, num_classes, max_num_bboxes)
        # Note that batch_keep_masks includes the valid_bboxes_masks, so it doesn't need to apply it
        # again after NMS
        keep_masks = batch_circle_nms(
            bboxes_centers=bboxes_centers,
            scores=top_scores,
            valid_bboxes_masks=valid_bboxes_masks,
            post_max_size=self.post_max_size,
            min_radius=self.nms_min_radius,
        )
        # Filter the predictions based on the keep_masks and return MultiTaskPredictions
        multi_task_predictions = self._filter_bbox_predictions(
            flatten_bboxes_predictions=flatten_bboxes_predictions,
            scores=top_scores,
            class_ids=class_ids,
            keep_masks=keep_masks,
            max_num_bboxes=max_num_bboxes,
            batch_size=batch_size,
        )
        return multi_task_predictions

    def prepare_for_export(self) -> CenterHead:
        """Return an export-ready copy of the head.

        Returns:
            Deep copy of the head in evaluation mode.
        """
        return deepcopy(self).eval()
