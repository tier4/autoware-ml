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

"""TransFusion detection head components.

This module contains the query decoder, target generation, and loss logic used
by the native TransFusion lidar detector.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import NamedTuple, Mapping
from types import MappingProxyType

from jaxtyping import Bool, Float32, Int32, Int64
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.detection3d.head_targets import TransFusionHeadTargets
from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    TransFusionHeadOutputs,
    TransFusionSeparateHeadOutputs,
)
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss
from autoware_ml.losses.detection3d.gaussian_focal import GaussianFocalLoss
from autoware_ml.models.common.layers.conv import ConvModule
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import (
    TransFusionBBoxCoder,
    ScoreThresholdGroup,
    ScoreThresholdConfig,
)
from autoware_ml.models.detection3d.task_modules.heatmap import (
    create_oriented_gaussian_heatmaps,
    vectorize_gaussian_radii,
    create_gaussian_heatmaps,
    batch_circle_nms,
)
from autoware_ml.models.detection3d.heads.transfusions.transfusion_decoder_layer import (
    TransFusionDecoderLayer,
)
from autoware_ml.models.detection3d.heads.transfusions.exportable_multi_head_attention import (
    ExportableMultiheadAttention,
)
from autoware_ml.types.geometry import Box3DFieldIndex


class NMSGroupConfig(NamedTuple):
    """
    Config for each NMS group.
    Args:
        class_names: Names of classes for this group.
        nms_radius: Optional NMS radius threshold for this group.
            If not provided, the default nms_radius will be used.
        max_size: Optional maximum number of predictions to keep for this group.
            If not provided, the default max_size will be used.
    """

    class_names: Sequence[str]
    nms_radius: float | None
    max_size: int | None


class NMSGroup(NamedTuple):
    """
    Resolved config for each NMS group.
    Args:
        class_ids: Class indices for this group.
        nms_radius: NMS radius threshold for this group.
        max_size: Maximum number of predictions to keep for this group.
    """

    class_ids: Sequence[int]
    nms_radius: float
    max_size: int


class LayerTargets(NamedTuple):
    """
    Assignment targets produced for a single decoder layer.

    Args:
        labels: Target class labels, where num_classes marks a negative (unmatched) proposal.
        label_weights: Per-proposal, per-class classification weights. Negatives carry their
            sample's class weights, positives a weight of one for every class.
        bbox_targets: Encoded box regression targets, zero for negatives.
        bbox_weights: Per-proposal box regression weights, one for positives and zero elsewhere.
        num_pos: Number of proposals this layer matched to a real gt box.
        matched_iou_sum: Sum of the matched proposals' IoUs. The caller sums this over layers and
            divides by the total number of positives to get the mean matched IoU.
    """

    labels: Int64[torch.Tensor, "batch_size num_proposals"]
    label_weights: Float32[torch.Tensor, "batch_size num_proposals num_classes"]
    bbox_targets: Float32[torch.Tensor, "batch_size num_proposals code_size"]
    bbox_weights: Float32[torch.Tensor, "batch_size num_proposals code_size"]
    num_pos: int
    matched_iou_sum: float


class LayerLosses(NamedTuple):
    """
    Losses computed for a single decoder layer.

    Args:
        loss_cls: Weighted classification loss over this layer's proposals.
        loss_bbox: Weighted box regression loss over this layer's positive proposals.
        loss_iou: Weighted rotated BEV IoU loss over this layer's positive proposals, None when
            the head was built without an IoU loss.
    """

    loss_cls: torch.Tensor
    loss_bbox: torch.Tensor
    loss_iou: torch.Tensor | None


class SeparateHead1D(nn.Module):
    """Apply per-query prediction branches for TransFusion outputs.

    Each branch is implemented as a lightweight 1D convolutional stack over the
    query dimension.
    """

    def __init__(
        self,
        in_channels: int,
        heads: MappingProxyType[str, tuple[int, int]],
        hidden_channels: int,
        norm_eps: float = 1e-3,
        norm_momentum: float = 0.01,
    ) -> None:
        """Initialize the per-query prediction heads.

        Args:
            in_channels: Input feature dimension.
            heads: Mapping from head name to ``(out_channels, num_convs)``.
            hidden_channels: Width of the intermediate branch layers.
            norm_eps: Epsilon used by the branch batch-normalization layers.
            norm_momentum: Momentum used by the branch batch-normalization layers.
        """
        super().__init__()
        self.heads = nn.ModuleDict()
        for name, (out_channels, num_convs) in heads.items():
            layers: list[nn.Module] = []
            current_channels = in_channels
            for _ in range(max(num_convs - 1, 0)):
                layers.append(
                    nn.Conv1d(current_channels, hidden_channels, kernel_size=1, bias=False)
                )
                layers.append(nn.BatchNorm1d(hidden_channels, eps=norm_eps, momentum=norm_momentum))
                layers.append(nn.ReLU(inplace=True))
                current_channels = hidden_channels
            layers.append(nn.Conv1d(current_channels, out_channels, kernel_size=1))
            self.heads[name] = nn.Sequential(*layers)

        # nn.ModuleDict has no .get, so membership has to be tested explicitly.
        if "heatmaps" in self.heads:
            nn.init.constant_(self.heads["heatmaps"][-1].bias, -2.19)  # type: ignore

    def forward(
        self, query_feats: Float32[torch.Tensor, "batch_size num_channels num_queries"]
    ) -> TransFusionSeparateHeadOutputs:
        """Apply all prediction branches to the query features.

        Args:
            query_feats: Per-query feature tensor.

        Returns:
            TransFusionSeparateHeadOutputs: Dataclass to save for each prediction branch.
        """
        branches = {name: head(query_feats) for name, head in self.heads.items()}
        return TransFusionSeparateHeadOutputs(
            heatmap=branches["heatmaps"],
            center=branches["centers"],
            height=branches["heights"],
            dim=branches["dims"],
            rot=branches["rots"],
            vel=branches.get("vels"),
        )


class TransFusionHead(nn.Module):
    """Implement the native TransFusion lidar detection head.

    The head predicts a dense BEV heatmap, selects top proposals, refines them
    with decoder layers, and computes assignment-based training targets.
    """

    def __init__(
        self,
        num_proposals: int,
        auxiliary: bool,
        in_channels: int,
        hidden_channel: int,
        class_names: Sequence[str],
        num_decoder_layers: int,
        num_heads: int,
        feedforward_channels: int,
        common_heads: MappingProxyType[str, tuple[int, int]],
        bbox_coder: TransFusionBBoxCoder,
        assigner: HungarianAssigner3D,
        point_cloud_range: list[float],
        voxel_size: list[float],
        out_size_factor: int,
        code_weights: list[float],
        min_radius: int,
        gaussian_overlap: float,
        score_threshold_group_configs: Sequence[ScoreThresholdConfig] | None,
        post_max_size: int,
        nms_min_radius: float,
        dense_heatmap_pooling_class_names: Sequence[str],
        iou_loss: nn.Module | None = None,
        heatmap_target: str = "round",
        nms_type: str | None = None,
        nms_group_configs: Sequence[NMSGroupConfig] | None = None,
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 0.25,
        loss_heatmap_weight: float = 1.0,
        heatmap_init_bias: float = -2.19,
        nms_kernel_size: int = 3,
        use_velocity: bool = True,
        head_hidden_channels: int | None = None,
        norm_eps: float = 1e-3,
        norm_momentum: float = 0.01,
        use_bf16_cross_attention: bool = False,
        partial_ignore_labels: Sequence[str] | None = None,
    ) -> None:
        """Initialize the TransFusion detection head.

        Args:
            num_proposals: Number of proposals selected from the dense heatmap.
            auxiliary: Whether to keep predictions from all decoder layers.
            class_names: Sequence of class names.
            in_channels: Input BEV feature dimension.
            hidden_channel: Internal decoder feature dimension.
            num_classes: Number of foreground classes.
            num_decoder_layers: Number of decoder refinement layers.
            num_heads: Number of attention heads per decoder layer.
            feedforward_channels: Hidden dimension of decoder feed-forward blocks.
            common_heads: Prediction-branch specification shared by decoder layers.
            bbox_coder: Box coder used for encoding and decoding predictions.
            assigner: Proposal assigner used during training.
            point_cloud_range: Detector point-cloud range.
            voxel_size: Voxel size used by the detector.
            out_size_factor: BEV downsampling factor between point and feature space.
            code_weights: Weights applied to each regression channel.
            min_radius: Minimum heatmap Gaussian radius.
            gaussian_overlap: Required overlap for Gaussian radius computation.
                Only used by the ``"round"`` heatmap target.
            score_threshold_group_configs: Prediction score threshold used
                to filter out predictions.
            post_max_size: Maximum number of predictions kept after NMS.
            nms_min_radius: Minimum center distance used by circle NMS.
            dense_heatmap_pooling_class_names: Optional class names that should use local max
                pooling before proposal selection.
            iou_loss: Optional rotated BEV IoU loss applied to every matched (prediction, gt)
                pair of every decoder layer, on top of the L1 box loss. The module receives the
                predictions and targets decoded through ``bbox_coder`` together with the
                matched labels; see
                :class:`~autoware_ml.losses.detection3d.rotated_iou_loss.Rotated2DIouLoss`.
                None disables the term and skips its computation entirely.
            class_names: Optional ordered class names used to resolve config-friendly
                class lists.
            heatmap_target: Shape of the dense heatmap supervision. ``"round"``
                (default) draws a circular Gaussian sized by ``gaussian_radius``.
                ``"oriented"`` draws an elliptical Gaussian stretched along the box
                length and rotated by the box yaw, so elongated objects such as a
                tractor and trailer rig get one connected positive region instead of
                a small blob in the low-density gap at the coupling.
            nms_type: NMS applied during prediction. ``None`` runs no NMS at all and
                ``"circle"`` runs grouped center-distance NMS. With ``"circle"`` and
                no ``nms_group_configs``, every class forms its own group with
                ``nms_min_radius`` and ``post_max_size``.
            nms_group_configs: Optional grouped NMS configuration, only allowed with
                ``nms_type="circle"``. Each entry must provide ``class_names`` and
                may override ``nms_radius`` and ``max_size``. A group with
                ``nms_radius`` of ``0`` keeps its highest-scoring predictions up to
                ``max_size``.
            loss_cls_weight: Weight applied to the classification loss.
            loss_bbox_weight: Weight applied to the box regression loss.
            loss_heatmap_weight: Weight applied to the dense heatmap loss.
            heatmap_init_bias: Initial bias used by the dense heatmap branch.
            nms_kernel_size: Kernel size used for local-maximum suppression.
            use_velocity: Whether the head predicts object velocity.
            head_hidden_channels: Width of the prediction-branch hidden layers.
                Defaults to ``hidden_channel``.
            norm_eps: Epsilon used by the head's batch-normalization layers
                (shared block, dense heatmap head, prediction branches).
            norm_momentum: Momentum used by the head's batch-normalization layers
                (shared block, dense heatmap head, prediction branches).
            use_bf16_cross_attention: Whether export emits fusion-ready attention and uses bf16 for
                the long cross-attention core. Requires ``deploy.onnx.precision=fp16``.
            partial_ignore_labels: Class names whose annotations are missing from some frames,
                currently the traffic cones and barriers flagged by
                ``gt_traffic_cone_barrier_bbox_status``. In a frame flagged as not annotated
                for them, their dense heatmap loss is skipped and the negative queries are not
                pushed away from them, so the head is not taught that they are absent.
        """
        super().__init__()
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.out_size_factor = out_size_factor
        self.code_weights = code_weights
        self.min_radius = min_radius
        self.gaussian_overlap = gaussian_overlap
        if heatmap_target not in {"round", "oriented"}:
            raise ValueError(f"Unsupported TransFusion heatmap_target: {heatmap_target!r}")
        self.heatmap_target = heatmap_target
        self.post_max_size = post_max_size
        self.nms_min_radius = nms_min_radius
        self.nms_kernel_size = nms_kernel_size
        self.bbox_coder = bbox_coder
        self.assigner = assigner
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_heatmap_weight = loss_heatmap_weight
        self.heatmap_init_bias = heatmap_init_bias
        self.use_velocity = use_velocity
        self.head_hidden_channels = (
            head_hidden_channels if head_hidden_channels is not None else hidden_channel
        )
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self.use_bf16_cross_attention = use_bf16_cross_attention
        self.required_onnx_precision = "fp16" if use_bf16_cross_attention else None
        if nms_type not in {None, "circle"}:
            raise ValueError(f"Unsupported TransFusion NMS type: {nms_type!r}")
        self.nms_type = nms_type
        self.dense_heatmap_pooling_class_ids = self._resolve_class_ids(
            dense_heatmap_pooling_class_names
        )
        self.partial_ignore_class_ids = (
            self._resolve_class_ids(partial_ignore_labels)
            if partial_ignore_labels is not None
            else None
        )
        # nms_type decides whether NMS runs at all, the groups only decide how. Groups without a
        # type would be silently ignored, which is the kind of config mistake worth refusing.
        if nms_type is None and nms_group_configs is not None:
            raise ValueError(
                "TransFusionHead nms_group_configs requires nms_type='circle', "
                "nms_type=None runs no NMS."
            )
        self.nms_groups = (
            self._resolve_nms_groups(nms_group_configs) if nms_type == "circle" else None
        )
        # Assign score_threshold_groups
        self.bbox_coder.score_threshold_groups = self._resolve_score_threshold_groups(
            score_threshold_group_configs
        )
        self.shared_block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channel, eps=norm_eps, momentum=norm_momentum),
            nn.ReLU(inplace=True),
        )
        self.dense_heatmap_head = self._build_dense_heatmaps(hidden_channel)
        self.class_encoding = nn.Conv1d(self.num_classes, hidden_channel, kernel_size=1)
        self.decoder = nn.ModuleList(
            [
                TransFusionDecoderLayer(hidden_channel, num_heads, feedforward_channels)
                for _ in range(num_decoder_layers)
            ]
        )
        # Build prediction heads for each decoder layer.
        self.prediction_heads = self._build_prediction_heads(
            hidden_channel, num_decoder_layers, common_heads
        )

        self.loss_heatmap = GaussianFocalLoss()
        self.loss_cls = SigmoidFocalLoss()
        self.loss_bbox = nn.L1Loss(reduction="none")
        self.iou_loss = iou_loss

    def _build_prediction_heads(
        self,
        in_channels: int,
        num_decoder_layers: int,
        prediction_head_configs: Mapping[str, tuple[int, int]],
    ) -> nn.ModuleList:
        """
        Build a list of per-query prediction heads for each decoder layer.

        Args:
            in_channels: Input feature dimension.
            num_decoder_layers: Number of decoder layers.
            prediction_head_configs: Mapping from head name to ``(out_channels, num_convs)``.
        """
        prediction_head_configs = dict(prediction_head_configs)
        if (
            "heatmaps" in prediction_head_configs
            and prediction_head_configs["heatmaps"][0] != self.num_classes
        ):
            raise ValueError(
                "TransFusionHead prediction heatmap head is not compatible "
                "with the number of classes."
            )
        else:
            prediction_head_configs["heatmaps"] = (self.num_classes, 2)

        if not self.use_velocity and "vels" in prediction_head_configs:
            prediction_head_configs.pop("vels")

        prediction_heads = nn.ModuleList(
            [
                SeparateHead1D(
                    in_channels,
                    MappingProxyType(prediction_head_configs),
                    hidden_channels=self.head_hidden_channels,
                    norm_eps=self.norm_eps,
                    norm_momentum=self.norm_momentum,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        return prediction_heads

    def _build_dense_heatmaps(self, in_channels: int) -> nn.Module:
        """
        Build the dense heatmap head used to predict BEV heatmaps for proposal selection.

        Args:
            in_channels: Input feature dimension.

        Returns:
            nn.Module: Dense heatmap head.
        """
        dense_heatmap_head = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels,
                norm_eps=self.norm_eps,
                norm_momentum=self.norm_momentum,
            ),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=3, padding=1),
        )
        nn.init.constant_(dense_heatmap_head[-1].bias, self.heatmap_init_bias)  # type: ignore
        return dense_heatmap_head

    def _resolve_class_ids(self, selected_class_names: Sequence[str]) -> Sequence[int]:
        """Resolve class names or indices into validated class ids."""
        resolved: list[int] = []
        for class_name in selected_class_names:
            class_id = self.class_names.index(class_name)
            if not 0 <= class_id < self.num_classes:
                raise ValueError(f"TransFusionHead class index: {class_id} out of range")
            resolved.append(class_id)

        return sorted(set(resolved))

    def _resolve_nms_groups(
        self, nms_config_groups: Sequence[NMSGroupConfig] | None
    ) -> Sequence[NMSGroup]:
        """
        Resolve grouped NMS configuration into class-id form.

        Without an explicit configuration every class becomes its own group with the head
        defaults, so classes never suppress each other and ``nms_type="circle"`` on its own
        behaves like plain per-class circle NMS.
        """
        if nms_config_groups is None:
            return [
                NMSGroup(
                    class_ids=[class_id],
                    nms_radius=self.nms_min_radius,
                    max_size=self.post_max_size,
                )
                for class_id in range(self.num_classes)
            ]

        resolved_groups: list[NMSGroup] = []
        # A class may appear in at most one group, otherwise it would be suppressed against two
        # different radii. Maps each claimed class id to the group that claimed it, so the error
        # can name both groups.
        claimed_classes: dict[int, int] = {}
        for group_index, nms_config_group in enumerate(nms_config_groups):
            class_ids = self._resolve_class_ids(nms_config_group.class_names)
            if class_ids is None:
                raise ValueError("TransFusionHead NMS group must define class_names.")

            for class_id in class_ids:
                if class_id in claimed_classes:
                    raise ValueError(
                        f"TransFusionHead class id {class_id} appears in NMS groups "
                        f"{claimed_classes[class_id]} and {group_index}, so it would be "
                        "suppressed against two different radii."
                    )
                claimed_classes[class_id] = group_index

            # `is None` on purpose: a radius of 0 is a valid setting that disables the
            # suppression for the group (its boxes are only capped by score), so it must not
            # fall back to the head default.
            resolved_groups.append(
                NMSGroup(
                    class_ids=class_ids,
                    nms_radius=(
                        self.nms_min_radius
                        if nms_config_group.nms_radius is None
                        else nms_config_group.nms_radius
                    ),
                    max_size=(
                        self.post_max_size
                        if nms_config_group.max_size is None
                        else nms_config_group.max_size
                    ),
                )
            )
        return resolved_groups

    def _resolve_score_threshold_groups(
        self, score_threshold_config_groups: Sequence[ScoreThresholdConfig] | None
    ) -> Sequence[ScoreThresholdGroup] | None:
        """Resolve grouped NMS configuration into class-id form."""
        if score_threshold_config_groups is None:
            return None

        resolved_groups: list[ScoreThresholdGroup] = []
        # A class may appear in at most one group, otherwise its threshold is ambiguous. Maps each
        # claimed class id to the group that claimed it, so the error can name both groups.
        claimed_classes: dict[int, int] = {}
        for group_index, score_threshold_config_group in enumerate(score_threshold_config_groups):
            class_ids = self._resolve_class_ids(score_threshold_config_group.class_names)
            if class_ids is None:
                raise ValueError("TransFusionHead Score threshold group must define class_names.")

            for class_id in class_ids:
                if class_id in claimed_classes:
                    raise ValueError(
                        f"TransFusionHead class id {class_id} appears in score threshold groups "
                        f"{claimed_classes[class_id]} and {group_index}, so its threshold is "
                        "ambiguous."
                    )
                claimed_classes[class_id] = group_index

            resolved_groups.append(
                ScoreThresholdGroup(
                    class_ids=class_ids,
                    score_threshold=score_threshold_config_group.score_threshold,
                )
            )
        return resolved_groups

    def _suppress_dense_heatmaps(
        self, heatmaps: Float32[torch.Tensor, "batch_size num_classes height width"]
    ) -> Float32[torch.Tensor, "batch_size num_classes height width"]:
        """
        Suppress non-maximal dense heatmap activations before proposal sampling. It produces a
        local-maximum mask for selected classes, where the value is 0 when it gets suppressed, and
        1 when it is kept.
        """
        # If the ids are empty, then no classes are selected for local max pooling,
        # so return the heatmaps as-is
        if not self.dense_heatmap_pooling_class_ids:
            return heatmaps

        local_max = heatmaps.clone()
        padding = self.nms_kernel_size // 2
        selected_heatmap = heatmaps[:, self.dense_heatmap_pooling_class_ids, :, :]
        pooled = F.max_pool2d(
            selected_heatmap,
            kernel_size=self.nms_kernel_size,
            stride=1,
            padding=0,
        )

        if padding == 0:
            local_max[:, self.dense_heatmap_pooling_class_ids, :, :] = pooled
        else:
            local_max[
                :,
                self.dense_heatmap_pooling_class_ids,
                padding:-padding,
                padding:-padding,
            ] = pooled
        return heatmaps * (local_max == heatmaps)

    def _create_2d_grid(
        self, width: int, height: int, device: torch.device
    ) -> Float32[torch.Tensor, "1 height*width 2"]:
        """Create BEV cell centers for positional encoding.

        Args:
            width: Feature-map width.
            height: Feature-map height.
            device: Device used for the created tensor.

        Returns:
            Flattened BEV (height*width) cell-center coordinates (x, y).
        """
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        grid = torch.stack([grid_x + 0.5, grid_y + 0.5], dim=-1)
        return grid.view(1, -1, 2)

    def _select_top_proposals(
        self,
        flatten_dense_heatmaps: Float32[torch.Tensor, "batch_size num_classes height*width"],
        flatten_bev_features: Float32[torch.Tensor, "batch_size channels height*width"],
        batch_size: int,
    ) -> tuple[
        Float32[torch.Tensor, "batch_size num_classes height*width"],
        Int32[torch.Tensor, "batch_size num_proposals"],
        Int32[torch.Tensor, "batch_size num_proposals"],
    ]:
        """Select top proposals from the dense heatmap.

        Args:
            flatten_dense_heatmaps: Flattened dense BEV heatmap tensor.
            flatten_bev_features: Flattened BEV feature tensor.
            batch_size: Batch size.

        Returns:
            query_feat: Selected query features.
        """
        proposal_count = min(
            self.num_proposals, flatten_dense_heatmaps.shape[1] * flatten_dense_heatmaps.shape[2]
        )
        _, top_indices = torch.topk(
            flatten_dense_heatmaps.view(batch_size, -1),
            k=proposal_count,
            dim=-1,
            largest=True,
            sorted=False,  # The order of proposals does not matter.
        )
        top_classes = top_indices // flatten_dense_heatmaps.shape[-1]
        top_positions = top_indices % flatten_dense_heatmaps.shape[-1]

        # (batch_size, channels, num_proposals)
        query_feat = flatten_bev_features.gather(
            2, top_positions[:, None, :].expand(-1, flatten_bev_features.shape[1], -1)
        )
        class_one_hot = (
            F.one_hot(top_classes, num_classes=self.num_classes).permute(0, 2, 1).float()
        )
        query_feat = query_feat + self.class_encoding(class_one_hot)
        return query_feat, top_classes, top_positions

    def _decoder_layer_forward(
        self,
        query_feats: Float32[torch.Tensor, "batch_size channels num_proposals"],
        flatten_bev_features: Float32[torch.Tensor, "batch_size channels height*width"],
        bev_pos: Float32[torch.Tensor, "batch_size height*width 2"],
        query_pos: Float32[torch.Tensor, "batch_size num_proposals 2"],
    ) -> Sequence[TransFusionSeparateHeadOutputs]:
        """Run the decoder layers and produce per-layer predictions.

        The layers are chained: layer ``i + 1`` refines the query features layer ``i`` produced,
        attending at the centers layer ``i`` predicted, and every layer's prediction head reads
        that layer's own output. Only the first layer sees the initial proposal features.

        Args:
            query_feats: Initial per-query feature tensor, gathered from the top proposals.
            flatten_bev_features: Flattened BEV features every layer attends over.
            bev_pos: BEV cell centers of the flattened features.
            query_pos: Initial BEV positions of the proposals.

        Returns:
            List of per-layer prediction outputs.
        """
        predictions: list[TransFusionSeparateHeadOutputs] = []
        for decoder_layer, prediction_head in zip(self.decoder, self.prediction_heads):
            query_feats = decoder_layer(
                query=query_feats, key=flatten_bev_features, query_pos=query_pos, key_pos=bev_pos
            )
            prediction: TransFusionSeparateHeadOutputs = prediction_head(query_feats)
            centers = prediction.center + query_pos.permute(0, 2, 1)
            prediction = prediction.model_copy(update={"center": centers})
            predictions.append(prediction)
            query_pos = centers.detach().permute(0, 2, 1)

        return predictions

    def _generate_outputs(
        self,
        predictions: Sequence[TransFusionSeparateHeadOutputs],
        dense_heatmaps: Float32[torch.Tensor, "batch_size num_classes height width"],
        flatten_dense_heatmaps: Float32[torch.Tensor, "batch_size num_classes height*width"],
        top_classes: Int32[torch.Tensor, "batch_size num_proposals"],
        top_positions: Int32[torch.Tensor, "batch_size num_proposals"],
    ) -> TransFusionHeadOutputs:
        """Combine decoder predictions with dense heatmap and proposal info.

        Args:
            predictions: List of per-layer prediction outputs.
            dense_heatmap: Dense BEV heatmap tensor.
            flatten_dense_heatmap: Flattened dense BEV heatmap tensor.
            top_classes: Class indices of the selected proposals.
            top_positions: Spatial positions of the selected proposals.

        Returns:
            TransFusionHeadOutputs: Dataclass containing all outputs.
        """
        if self.auxiliary:
            # Every decoder layer contributes its proposals along the last axis
            separate_head_outputs = TransFusionSeparateHeadOutputs(
                heatmap=torch.cat([prediction.heatmap for prediction in predictions], dim=-1),
                center=torch.cat([prediction.center for prediction in predictions], dim=-1),
                height=torch.cat([prediction.height for prediction in predictions], dim=-1),
                dim=torch.cat([prediction.dim for prediction in predictions], dim=-1),
                rot=torch.cat([prediction.rot for prediction in predictions], dim=-1),
                vel=(
                    torch.cat([prediction.vel for prediction in predictions], dim=-1)
                    if predictions[0].vel is not None
                    else None
                ),
            )
        else:
            separate_head_outputs = predictions[-1]

        query_heatmap_scores = flatten_dense_heatmaps.gather(
            2, top_positions[:, None, :].expand(-1, flatten_dense_heatmaps.shape[1], -1)
        )
        return TransFusionHeadOutputs(
            dense_heatmap=dense_heatmaps,
            query_heatmap_score=query_heatmap_scores,
            query_labels=top_classes,
            separate_head_outputs=separate_head_outputs,
        )

    def forward(
        self, bev_features: Float32[torch.Tensor, "batch_size channels height width"]
    ) -> TransFusionHeadOutputs:
        """Predict TransFusion heatmap, queries, and box parameters.

        Args:
            bev_feats: BEV feature tensor.

        Returns:
            TransFusionHeadOutputs: Dataclass containing all outputs.
        """
        batch_size, _, height, width = bev_features.shape
        bev_features = self.shared_block(bev_features)
        bev_pos = self._create_2d_grid(width, height, bev_features.device).repeat(batch_size, 1, 1)

        # Do not use autocast for the dense heatmap head since the precision can affect
        # the selected proposals. Under mixed precision the BEV features arrive as fp16 while
        # the head keeps fp32 weights, so they are cast explicitly now that autocast no longer
        # does it.
        with torch.autocast(device_type="cuda", enabled=False):
            dense_heatmaps = self.dense_heatmap_head(bev_features.float())

        suppressed_dense_heatmaps = self._suppress_dense_heatmaps(dense_heatmaps.detach().sigmoid())
        # (batch_size, num_classes, height, width) -> (batch_size, num_classes, height * width)
        flatten_dense_heatmaps = suppressed_dense_heatmaps.view(
            batch_size, suppressed_dense_heatmaps.shape[1], -1
        )

        # (batch_size, channels, height, width) -> (batch_size, channels, height * width)
        flatten_bev_features = bev_features.flatten(2)
        query_feat, top_classes, top_positions = self._select_top_proposals(
            flatten_dense_heatmaps, flatten_bev_features, batch_size
        )
        query_pos = bev_pos.gather(1, top_positions[..., None].expand(-1, -1, bev_pos.shape[-1]))
        decoder_layer_predictions = self._decoder_layer_forward(
            query_feats=query_feat,
            flatten_bev_features=flatten_bev_features,
            bev_pos=bev_pos,
            query_pos=query_pos,
        )
        return self._generate_outputs(
            predictions=decoder_layer_predictions,
            dense_heatmaps=dense_heatmaps,
            flatten_dense_heatmaps=flatten_dense_heatmaps,
            top_classes=top_classes,
            top_positions=top_positions,
        )

    def _filter_bboxes_nms_groups(
        self,
        bbox_centers: Float32[torch.Tensor, "batch_size num_proposals 2"],
        bbox_scores: Float32[torch.Tensor, "batch_size num_proposals"],
        bbox_classes: Int64[torch.Tensor, "batch_size num_proposals"],
        keep_masks: Bool[torch.Tensor, "batch_size num_proposals"],
    ) -> Bool[torch.Tensor, "batch_size num_proposals"]:
        """
        Refine a keep mask with grouped center-distance NMS.

        Each NMS group becomes one channel of :func:`batch_circle_nms`, so all classes in a group
        land in the same channel and therefore compete with each other, while classes in different
        groups never suppress one another. No box is moved: a proposal simply appears as invalid in
        the channels of the groups that do not claim its class, which batch_circle_nms already
        handles, so the returned mask stays aligned with the input proposal axis.

        ``keep_masks`` is consumed as well as refined: a proposal already rejected by the score or
        center-range filters takes no part in the suppression, so it can never remove a survivor.

        A class no group claims is not subject to NMS at all and keeps its incoming mask value.
        Groups are disjoint, which _resolve_nms_groups enforces, so no class is ever weighed
        against two different radii.

        Args:
            bbox_centers: BEV box centers as (center_x, center_y).
            bbox_scores: Confidence score for each proposal.
            bbox_classes: Predicted class id for each proposal. Must be a valid class index in
                [0, num_classes), since it indexes the group membership table.
            keep_masks: Keep mask from the earlier score and center-range filters.

        Returns:
            The keep mask with grouped NMS applied.
        """
        if self.nms_groups is None:
            return keep_masks

        batch_size, num_proposals = bbox_scores.shape
        num_groups = len(self.nms_groups)
        device = bbox_scores.device

        # (num_groups, num_classes) membership table: True where a group claims that class.
        group_class_masks = torch.zeros(
            (num_groups, self.num_classes), dtype=torch.bool, device=device
        )
        for group_index, group in enumerate(self.nms_groups):
            class_ids = list(group.class_ids)
            if not class_ids:
                raise ValueError(f"TransFusionHead NMS group {group_index} has no class ids.")
            if min(class_ids) < 0 or max(class_ids) >= self.num_classes:
                raise ValueError(
                    f"TransFusionHead NMS group {group_index} class ids {class_ids} are out of "
                    f"range for {self.num_classes} classes."
                )
            group_class_masks[group_index, class_ids] = True

        # A proposal takes part in a group's channel only when that group claims its class and it
        # survived the earlier filters.
        # (num_groups, num_classes)[:, (batch_size, num_proposals)] & (batch_size, num_proposals)
        # -> (num_groups, batch_size, num_proposals) -> (batch_size, num_groups, num_proposals)
        group_masks = (group_class_masks[:, bbox_classes] & keep_masks).permute(1, 0, 2)

        # Every channel sees the same boxes. Only the validity mask differs, so these are views
        # even they expand same bboxes across groups.
        # (batch_size, num_groups, num_proposals, 2) and (batch_size, num_groups, num_proposals)
        group_centers = bbox_centers.unsqueeze(1).expand(
            batch_size, num_groups, num_proposals, bbox_centers.shape[-1]
        )
        group_scores = bbox_scores.unsqueeze(1).expand(batch_size, num_groups, num_proposals)

        # One radius and one cap per channel, so all groups suppress in a single call. A group
        # with a radius of 0 disables the suppression in its channel: batch_circle_nms then only
        # keeps its highest-scoring proposals up to the cap.
        # (batch_size, num_groups, num_proposals)
        group_keep_masks = batch_circle_nms(
            bboxes_centers=group_centers,
            scores=group_scores,
            min_radii=[float(group.nms_radius) for group in self.nms_groups],
            valid_bboxes_masks=group_masks,
            post_max_sizes=[int(group.max_size) for group in self.nms_groups],
        )

        # Groups are disjoint, so a proposal enters at most one channel and is kept exactly when
        # that channel kept it. batch_circle_nms never keeps an invalid entry, so a proposal is
        # already False in every channel that does not claim its class.
        # (batch_size, num_groups, num_proposals) -> (batch_size, num_proposals)
        nms_keep_masks = group_keep_masks.any(dim=1)

        # Classes no group claims skip NMS and keep whatever the earlier filters decided.
        # (num_classes,)[(batch_size, num_proposals)] -> (batch_size, num_proposals)
        class_covered_masks = group_class_masks.any(dim=0)[bbox_classes]
        return nms_keep_masks | (keep_masks & ~class_covered_masks)

    def _filter_bbox_predictions(
        self,
        bbox_predictions: Float32[torch.Tensor, "batch_size num_proposals box_code_size"],
        scores: Float32[torch.Tensor, "batch_size num_proposals"],
        class_ids: Int64[torch.Tensor, "batch_size num_proposals"],
        keep_masks: Bool[torch.Tensor, "batch_size num_proposals"],
    ) -> ModelPredictions:
        """
        Apply the keep mask and repackage the batched proposals as per-sample predictions.

        Everything upstream of this point stays rectangular: every sample carries the same
        num_proposals slots and the filters only flip bits in ``keep_masks``. This is where that
        padding is finally dropped, so the result is ragged, one entry per sample holding only its
        own survivors. Each sample keeps a different number of boxes, which is why the output is a
        sequence of per-sample dataclasses rather than one batched tensor.

        Survivors stay in proposal order rather than being re-ranked by score, and a sample whose
        boxes were all suppressed still gets an entry, with empty tensors.

        Args:
            bbox_predictions: Decoded boxes for every proposal, in metric coordinates.
            scores: Confidence score for each proposal.
            class_ids: Predicted class id for each proposal.
            keep_masks: Keep mask accumulated by the score, center-range and NMS filters. True
                marks a proposal to emit as a detection.

        Returns:
            One Detection3DSamplePredictions per batch element, wrapped in ModelPredictions.
        """
        batch_size = bbox_predictions.shape[0]
        # Iterate over the batch and create a list of Detection3dPredictions for each sample.
        # The ragged padding is dropped here, where per-sample tensors are allowed to differ.
        detection3d_predictions = []
        for batch_index in range(batch_size):
            sample_keep_masks = keep_masks[batch_index]

            detection3d_predictions.append(
                Detection3DSamplePredictions(
                    bboxes_3d=bbox_predictions[batch_index][sample_keep_masks],
                    scores_3d=scores[batch_index][sample_keep_masks],
                    labels_3d=class_ids[batch_index][sample_keep_masks],
                )
            )

        return ModelPredictions(detection3d_predictions=detection3d_predictions)

    def decode_outputs(self, outputs: Detection3DHeadOutputs) -> ModelPredictions:
        """Decode predictions into metric-space boxes.

        Args:
            outputs: Raw prediction tensors produced by the head.

        Returns:
            List of decoded prediction dictionaries, one per batch element.
        """
        if outputs.transfusion_head_outputs is None:
            raise ValueError("TransFusionHead outputs are missing from Detection3DHeadOutputs.")

        # Mixed precision evaluation hands over fp16 outputs; the boxes decode in float32, as a
        # 64 m center in fp16 only resolves about 0.06 m.
        transfusion_head_outputs = outputs.transfusion_head_outputs
        separate_head_outputs = transfusion_head_outputs.separate_head_outputs
        batch_scores = separate_head_outputs.heatmap[..., -self.num_proposals :].float().sigmoid()
        one_hot = (
            F.one_hot(transfusion_head_outputs.query_labels, num_classes=self.num_classes)
            .permute(
                0, 2, 1
            )  # (batch_size, num_proposal, num_classes) -> (batch_size, num_classes, num_proposals)
            .to(batch_scores.dtype)
        )
        # Use proposals from the dense heatmap to calibrate the final scores, where they are only
        # valid when both of them align
        batch_scores = batch_scores * transfusion_head_outputs.query_heatmap_score.float() * one_hot
        batch_centers = separate_head_outputs.center[..., -self.num_proposals :].float()
        batch_heights = separate_head_outputs.height[..., -self.num_proposals :].float()
        batch_dims = separate_head_outputs.dim[..., -self.num_proposals :].float()
        batch_rots = separate_head_outputs.rot[..., -self.num_proposals :].float()
        batch_vels = separate_head_outputs.vel
        if batch_vels is not None:
            batch_vels = batch_vels[..., -self.num_proposals :].float()

        bbox_scores, bbox_classes = self.bbox_coder.decode_heatmaps(heatmaps=batch_scores)
        pred_bboxes = self.bbox_coder.decode_boxes(
            rots=batch_rots,
            dims=batch_dims,
            centers=batch_centers,
            heights=batch_heights,
            vels=batch_vels,
        )

        # Create keep_masks based on score_thresholds and post_center_range
        keep_masks = self.bbox_coder.filter_bboxes_score(
            bbox_scores=bbox_scores, bbox_classes=bbox_classes, num_classes=self.num_classes
        )
        keep_masks &= self.bbox_coder.filter_bboxes_center_range(
            bbox_centers=pred_bboxes[..., Box3DFieldIndex.X : Box3DFieldIndex.Z + 1]
        )

        # nms_type gates the suppression, nms_groups only holds how each class is suppressed.
        if self.nms_type == "circle":
            keep_masks = self._filter_bboxes_nms_groups(
                bbox_centers=pred_bboxes[..., Box3DFieldIndex.X : Box3DFieldIndex.Y + 1],
                bbox_scores=bbox_scores,
                bbox_classes=bbox_classes,
                keep_masks=keep_masks,
            )

        return self._filter_bbox_predictions(
            bbox_predictions=pred_bboxes,
            scores=bbox_scores,
            class_ids=bbox_classes,
            keep_masks=keep_masks,
        )

    def _build_dense_heatmap_targets(
        self,
        gt_bboxes_3d: Float32[torch.Tensor, "batch_size max_num_gt_bboxes num_Box3DFieldIndex"],
        gt_labels_3d: Float32[torch.Tensor, "batch_size max_num_gt_bboxes"],
        gt_valid_bboxes: Int32[torch.Tensor, " batch_size"],
        feature_map_size: tuple[int, int],
        device: torch.device,
    ) -> Float32[torch.Tensor, "batch_size num_classes height width"]:
        """Build dense heatmap targets for query initialization.

        Args:
            gt_bboxes_3d: Ground-truth 3D boxes for each batch element.
            gt_labels: Ground-truth labels for each batch element.
            feature_shape: Heatmap height and width.
            device: Device used for the generated target tensor.

        Returns:
            Dense training heatmap targets.
        """
        batch_size = len(gt_bboxes_3d)
        max_num_bboxes = gt_bboxes_3d.shape[1]
        feature_height, feature_width = feature_map_size

        # Movement of tensors to the correct device
        gt_bboxes_3d = gt_bboxes_3d.to(device=device)

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

        center = torch.stack((center_x, center_y), dim=-1)
        # (batch_size, max_num_bboxes, 2)
        center_int = center.floor().to(torch.long)

        if self.heatmap_target == "oriented":
            # (batch_size, max_num_bboxes)
            dense_heatmaps = create_oriented_gaussian_heatmaps(
                heatmap_width=feature_width,
                heatmap_height=feature_height,
                num_classes=self.num_classes,
                centers=center_int,
                lengths_cells=lengths,
                widths_cells=widths,
                yaws=gt_bboxes_3d[:, :, Box3DFieldIndex.YAW],
                gt_bboxes_labels=gt_labels_3d,
                valid_masks=valid_bbox_masks,
                device=device,
                min_sigma=self.min_radius / 3.0,
            )
        else:
            # (batch_size, max_num_bboxes)
            gaussian_radii = vectorize_gaussian_radii(
                widths=lengths,
                heights=widths,
                min_overlap=self.gaussian_overlap,
            ).to(device)
            # Clamp the Gaussian radii to ensure they are at least the minimum radius
            gaussian_radii = torch.clamp(gaussian_radii, min=self.min_radius)

            dense_heatmaps = create_gaussian_heatmaps(
                heatmap_width=feature_width,
                heatmap_height=feature_height,
                num_classes=self.num_classes,
                centers=center_int,
                gaussian_radii=gaussian_radii.long(),
                gt_bboxes_labels=gt_labels_3d,
                valid_masks=valid_bbox_masks,
                device=device,
            )

        return dense_heatmaps

    def get_targets(
        self,
        outputs: TransFusionHeadOutputs,
        gt_bboxes_3d: Float32[torch.Tensor, "batch_size max_num_gt_bboxes num_Box3DFieldIndex"],
        gt_labels_3d: Float32[torch.Tensor, "batch_size max_num_gt_bboxes"],
        gt_valid_bboxes: Int32[torch.Tensor, " batch_size"],
        feature_map_size: tuple[int, int],
        gt_traffic_cone_barrier_bbox_status: Bool[torch.Tensor, " batch_size"] | None = None,
    ) -> TransFusionHeadTargets:
        """Build TransFusion training targets.

        Args:
            outputs: Raw prediction tensors produced by the head.
            gt_bboxes_3d: Ground-truth boxes for each batch element.
            gt_labels_3d: Ground-truth labels for each batch element.
            gt_valid_bboxes: Number of real boxes in each batch element.
            feature_map_size: Dense heatmap height and width.
            gt_traffic_cone_barrier_bbox_status: Per-sample flag telling whether traffic cones
                and barriers are annotated, None when the dataset does not carry it. See
                `_build_class_weights`.

        Returns:
            Structured training targets for classification, boxes, and heatmaps.
        """
        batch_size = len(gt_valid_bboxes)
        device = outputs.dense_heatmap.device
        gt_labels_3d = gt_labels_3d.to(device=device, dtype=torch.long)
        gt_valid_bboxes = gt_valid_bboxes.to(device=device)
        class_weights = self._build_class_weights(
            batch_size=batch_size,
            gt_traffic_cone_barrier_bbox_status=gt_traffic_cone_barrier_bbox_status,
            device=device,
        )
        dense_heatmap_targets = self._build_dense_heatmap_targets(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_valid_bboxes=gt_valid_bboxes,
            feature_map_size=feature_map_size,
            device=device,
        )

        # (batch_size, max_num_bboxes) boolean mask for valid boxes based on the number of valid boxes per sample
        # (max_num_bboxes) -> (1, max_num_bboxes) -> (batch_size, max_num_bboxes) < gt_valid_bboxes.unsqueeze(1) (batch_size, 1)
        # -> (batch_size, max_num_bboxes)
        max_num_gt_bboxes = gt_bboxes_3d.shape[1]
        valid_masks = torch.arange(max_num_gt_bboxes, device=device).unsqueeze(0).expand(
            batch_size, -1
        ) < gt_valid_bboxes.unsqueeze(1)

        separate_head_outputs = outputs.separate_head_outputs
        num_layers = separate_head_outputs.center.shape[-1] // self.num_proposals
        scores = separate_head_outputs.heatmap.detach()
        centers = separate_head_outputs.center.detach()
        heights = separate_head_outputs.height.detach()
        dims = separate_head_outputs.dim.detach()
        rots = separate_head_outputs.rot.detach()
        vels = separate_head_outputs.vel
        if vels is not None:
            vels = vels.detach()

        # The gt boxes are the same for every decoder layer, so encode the whole batch once.
        # encode() also encodes the padded boxes, but only matched rows are ever read below.
        # (batch_size, max_num_gt_bboxes, code_size)
        encoded_gt_bboxes = self.bbox_coder.encode(gt_bboxes_3d)

        # One LayerTargets per decoder layer, concatenated along the proposal axis. That is the
        # same layout the per-query predictions use (forward() cats the auxiliary layers along the
        # query axis), so loss() can take a single layer as the same [start:end] slice of both.
        layer_targets = []
        for layer_index in range(num_layers):
            start = layer_index * self.num_proposals
            end = (layer_index + 1) * self.num_proposals
            layer_targets.append(
                self._build_layer_targets(
                    # (batch_size, num_proposals, code_size)
                    pred_bboxes=self.bbox_coder.decode_boxes(
                        rots=rots[..., start:end],
                        dims=dims[..., start:end],
                        centers=centers[..., start:end],
                        heights=heights[..., start:end],
                        vels=vels[..., start:end] if vels is not None else None,
                    ),
                    # (batch_size, num_classes, num_proposals) ->
                    # (batch_size, num_proposals, num_classes)
                    cls_logits=scores[..., start:end].transpose(2, 1),
                    gt_bboxes_3d=gt_bboxes_3d,
                    gt_labels_3d=gt_labels_3d,
                    encoded_gt_bboxes=encoded_gt_bboxes,
                    valid_masks=valid_masks,
                    class_weights=class_weights,
                )
            )

        num_pos = sum(layer.num_pos for layer in layer_targets)
        matched_iou_sum = sum(layer.matched_iou_sum for layer in layer_targets)

        return TransFusionHeadTargets(
            dense_heatmaps=dense_heatmap_targets,
            labels=torch.cat([layer.labels for layer in layer_targets], dim=1),
            label_weights=torch.cat([layer.label_weights for layer in layer_targets], dim=1),
            bbox_targets=torch.cat([layer.bbox_targets for layer in layer_targets], dim=1),
            bbox_weights=torch.cat([layer.bbox_weights for layer in layer_targets], dim=1),
            num_pos=num_pos,
            matched_iou=matched_iou_sum / max(num_pos, 1),
            class_weights=class_weights,
        )

    def _build_class_weights(
        self,
        batch_size: int,
        gt_traffic_cone_barrier_bbox_status: Bool[torch.Tensor, " batch_size"] | None,
        device: torch.device,
    ) -> Float32[torch.Tensor, "batch_size num_classes"]:
        """
        Build the per-sample class weights dropping the classes a sample is not annotated for.

        A frame flagged without traffic cone and barrier annotations may still contain them, so
        the head must not learn that they are absent: the partially ignored classes get a zero
        weight in that sample, which silences their dense heatmap loss and stops the negative
        queries from being pushed away from them.

        Args:
            batch_size: Number of samples in the batch.
            gt_traffic_cone_barrier_bbox_status: Per-sample flag telling whether traffic cones and
                barriers are annotated, None when the dataset does not carry it.
            device: Device of the weights.

        Returns:
            Weights of one, except zero for the partially ignored classes of the samples flagged
            as not annotated for them.
        """
        class_weights = torch.ones(
            (batch_size, self.num_classes), dtype=torch.float32, device=device
        )
        if self.partial_ignore_class_ids is None or gt_traffic_cone_barrier_bbox_status is None:
            return class_weights

        # (batch_size,) samples missing the annotations.
        unannotated_masks = ~gt_traffic_cone_barrier_bbox_status.to(device=device, dtype=torch.bool)
        # (num_classes,) classes those annotations cover.
        ignored_class_masks = torch.zeros(self.num_classes, dtype=torch.bool, device=device)
        ignored_class_masks[list(self.partial_ignore_class_ids)] = True
        # (batch_size, 1) & (1, num_classes) -> (batch_size, num_classes)
        class_weights[unannotated_masks.unsqueeze(1) & ignored_class_masks.unsqueeze(0)] = 0.0
        return class_weights

    def _build_layer_targets(
        self,
        pred_bboxes: Float32[torch.Tensor, "batch_size num_proposals num_Box3DFieldIndex"],
        cls_logits: Float32[torch.Tensor, "batch_size num_proposals num_classes"],
        gt_bboxes_3d: Float32[torch.Tensor, "batch_size max_num_gt_bboxes num_Box3DFieldIndex"],
        gt_labels_3d: Int64[torch.Tensor, "batch_size max_num_gt_bboxes"],
        encoded_gt_bboxes: Float32[torch.Tensor, "batch_size max_num_gt_bboxes code_size"],
        valid_masks: Bool[torch.Tensor, "batch_size max_num_gt_bboxes"],
        class_weights: Float32[torch.Tensor, "batch_size num_classes"],
    ) -> LayerTargets:
        """
        Assign one decoder layer's proposals to ground truth and encode that layer's targets.

        The assignment is one-to-one per sample: every proposal is either matched to a single real
        gt box (a positive) or left as background (a negative). Unmatched proposals keep the
        background label and zero regression weight, so they contribute to the classification loss
        only.

        Args:
            pred_bboxes: Decoded proposal boxes for this layer, in metric coordinates.
            cls_logits: Class logits for this layer's proposals.
            gt_bboxes_3d: Ground-truth boxes in metric coordinates, padded per sample.
            gt_labels_3d: Ground-truth class labels, padded per sample.
            encoded_gt_bboxes: Ground-truth boxes already encoded into regression targets. Passed
                in rather than encoded here because it does not vary across decoder layers.
            valid_masks: Mask indicating which padded gt entries are real boxes.
            class_weights: Per-sample class weights, zero for the classes a sample is not
                annotated for. They become the classification weights of the negatives.

        Returns:
            This layer's classification and regression targets, plus its positive count and
            matched-IoU sum.
        """
        batch_size = pred_bboxes.shape[0]
        assign_results = self.assigner.assign(
            bboxes=pred_bboxes,
            gt_bboxes=gt_bboxes_3d,
            gt_labels=gt_labels_3d,
            cls_pred=cls_logits,
            valid_masks=valid_masks,
        )
        labels = pred_bboxes.new_full(
            (batch_size, self.num_proposals), self.num_classes, dtype=torch.long
        )
        # A negative query carries its sample's class weights, so it is not pushed away from a
        # class the sample is not annotated for. Positives are overwritten with ones below.
        # (batch_size, num_classes) -> (batch_size, num_proposals, num_classes)
        label_weights = class_weights.unsqueeze(1).expand(-1, self.num_proposals, -1).clone()
        bbox_targets = pred_bboxes.new_zeros(
            (batch_size, self.num_proposals, self.bbox_coder.code_size)
        )
        bbox_weights = pred_bboxes.new_zeros(
            (batch_size, self.num_proposals, self.bbox_coder.code_size)
        )

        # Positives are the proposals the assigner matched to a real gt. gt_inds is
        # one-based: 0 means negative and -1 means ignore.
        # (batch_size, num_proposals)
        pos_masks = assign_results.gt_inds > 0
        # gt_inds holds a *per-sample* gt index, so it cannot index the batched gt tensors on
        # its own. Pair every positive with the sample it belongs to and use both axes.
        # (num_positives,) each
        pos_batch_indices, pos_proposal_indices = pos_masks.nonzero(as_tuple=True)
        pos_gt_indices = assign_results.gt_inds[pos_batch_indices, pos_proposal_indices] - 1

        # Empty index tensors make every write below a no-op, so a batch with no positives
        # needs no separate branch.
        labels[pos_batch_indices, pos_proposal_indices] = gt_labels_3d[
            pos_batch_indices, pos_gt_indices
        ]
        bbox_targets[pos_batch_indices, pos_proposal_indices] = encoded_gt_bboxes[
            pos_batch_indices, pos_gt_indices
        ]
        bbox_weights[pos_batch_indices, pos_proposal_indices] = 1.0
        # A positive is vouched for by its matched gt box, so every class of its row is supervised.
        label_weights[pos_batch_indices, pos_proposal_indices] = 1.0

        matched_iou_sum = 0.0
        if assign_results.max_overlaps is not None:
            matched_iou_sum = float(assign_results.max_overlaps[pos_masks].sum())
        return LayerTargets(
            labels=labels,
            label_weights=label_weights,
            bbox_targets=bbox_targets,
            bbox_weights=bbox_weights,
            num_pos=int(pos_masks.sum()),
            matched_iou_sum=matched_iou_sum,
        )

    def loss(
        self,
        outputs: Detection3DHeadOutputs,
        gt_bboxes_3d: Float32[torch.Tensor, "batch_size max_num_3d_gt_bboxes num_Box3DFieldIndex"],
        gt_labels_3d: Float32[torch.Tensor, "batch_size max_num_3d_gt_bboxes"],
        gt_valid_bboxes: Int32[torch.Tensor, " batch_size"],
        gt_traffic_cone_barrier_bbox_status: Bool[torch.Tensor, " batch_size"] | None = None,
    ) -> MappingProxyType[str, Float32[torch.Tensor, " num_losses"] | float]:
        """
        Compute TransFusionHead losses. It will decode raw outputs and assign predictions to
        ground-truth boxes (1-to-1) for each decoder layer, and obtain positive and negative pairs.
        It computes the following losses:
          1) dense_heatmap loss: GaussianFocalLoss between the predicted dense heatmap
          and the target dense heatmap.
          2) cls_loss: SigmoidFocalLoss between the predicted class logits and the target labels
          from queries. For negative queries, the target label is set to the background class.
          3) bbox_loss: L1Loss between the predicted box parameters and the target box parameters.

        Args:
          outputs: Forward outputs.
          gt_bboxes_3d: Ground truth 3D bboxes.
          gt_labels_3d: Ground truth 3D bboxes label.
          gt_valid_bboxes: Number of valid ground truth 3D bboxes in each sample.
          gt_traffic_cone_barrier_bbox_status: Per-sample flag telling whether traffic cones and
              barriers are annotated. In a sample flagged False, the classes listed in
              ``partial_ignore_labels`` contribute neither to the dense heatmap loss nor to the
              classification loss of the negative queries. None when the dataset does not carry
              the flag, in which case every class is supervised.
        """
        loss_dict: dict[str, torch.Tensor | float] = {}
        transfusion_head_outputs = outputs.transfusion_head_outputs
        if transfusion_head_outputs is None:
            raise ValueError(
                "TransFusionHead: transfusion_head_outputs must exist from the forward outputs!"
            )
        feature_map_size = transfusion_head_outputs.dense_heatmap.shape[-2:]
        targets = self.get_targets(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_valid_bboxes=gt_valid_bboxes,
            feature_map_size=(feature_map_size[0], feature_map_size[1]),
            outputs=transfusion_head_outputs,
            gt_traffic_cone_barrier_bbox_status=gt_traffic_cone_barrier_bbox_status,
        )

        loss_heatmap = self.loss_heatmap(
            transfusion_head_outputs.dense_heatmap,
            targets.dense_heatmaps,
            # (batch_size, num_classes) -> (batch_size, num_classes, 1, 1), broadcast over the grid.
            weights=targets.class_weights[:, :, None, None],
        )
        loss_dict["loss_heatmap"] = self.loss_heatmap_weight * loss_heatmap
        separate_head_outputs = transfusion_head_outputs.separate_head_outputs
        num_layers = separate_head_outputs.center.shape[-1] // self.num_proposals

        for layer_index in range(num_layers):
            # The last layer keeps the "layer_-1" prefix the original TransFusion logs use.
            prefix = "layer_-1" if layer_index == num_layers - 1 else f"layer_{layer_index}"
            layer_losses = self._build_layer_losses(
                separate_head_outputs=separate_head_outputs,
                targets=targets,
                start=layer_index * self.num_proposals,
                end=(layer_index + 1) * self.num_proposals,
            )
            loss_dict[f"{prefix}_loss_cls"] = layer_losses.loss_cls
            loss_dict[f"{prefix}_loss_bbox"] = layer_losses.loss_bbox
            if layer_losses.loss_iou is not None:
                loss_dict[f"{prefix}_loss_iou"] = layer_losses.loss_iou

        loss_dict["matched_ious"] = transfusion_head_outputs.dense_heatmap.new_tensor(
            targets.matched_iou
        )
        loss_dict["loss"] = sum(value for key, value in loss_dict.items() if "loss" in key)
        return MappingProxyType(loss_dict)

    def _build_layer_losses(
        self,
        separate_head_outputs: TransFusionSeparateHeadOutputs,
        targets: TransFusionHeadTargets,
        start: int,
        end: int,
    ) -> LayerLosses:
        """
        Compute one decoder layer's classification and box regression losses.

        Both the per-query predictions and the assignment targets carry every decoder layer
        concatenated along the proposal axis, so a single layer is the ``[start:end]`` slice of
        each. Losses are normalized by the total number of positives across all layers, matching
        the reference TransFusion implementation.

        Args:
            separate_head_outputs: Per-query predictions for every decoder layer.
            targets: Assignment targets for every decoder layer.
            start: First proposal index belonging to this layer.
            end: One past the last proposal index belonging to this layer.

        Returns:
            This layer's weighted classification and box regression losses.
        """
        # SigmoidFocalLoss works on flattened (N, num_classes) logits.
        # (batch_size, num_classes, num_proposals) -> (batch_size, num_proposals, num_classes)
        # -> (batch_size * num_proposals, num_classes)
        layer_logits = (
            separate_head_outputs.heatmap[..., start:end]
            .permute(0, 2, 1)
            .reshape(-1, self.num_classes)
        )
        # (batch_size, num_proposals) -> (batch_size * num_proposals,)
        layer_labels = targets.labels[:, start:end].reshape(-1)
        # One-hot the positives; background proposals (label == num_classes) stay all-zero rows,
        # which is what the focal loss expects for a negative.
        cls_targets = layer_logits.new_zeros((layer_labels.shape[0], self.num_classes))
        pos_masks = layer_labels < self.num_classes
        cls_targets[pos_masks, layer_labels[pos_masks]] = 1.0
        # (batch_size, num_proposals, num_classes) -> (batch_size * num_proposals, num_classes)
        layer_label_weights = targets.label_weights[:, start:end].reshape(-1, self.num_classes)
        loss_cls = self.loss_cls(
            layer_logits, cls_targets, layer_label_weights, avg_factor=max(targets.num_pos, 1)
        )

        # A head without velocity channels contributes no columns to the regression vector.
        if separate_head_outputs.vel is not None:
            vels = separate_head_outputs.vel[..., start:end]
        else:
            vels = separate_head_outputs.center.new_zeros(
                separate_head_outputs.center.shape[0], 0, self.num_proposals
            )

        # (batch_size, code_size, num_proposals) -> (batch_size, num_proposals, code_size)
        preds = torch.cat(
            [
                separate_head_outputs.center[..., start:end],
                separate_head_outputs.height[..., start:end],
                separate_head_outputs.dim[..., start:end],
                separate_head_outputs.rot[..., start:end],
                vels,
            ],
            dim=1,
        ).permute(0, 2, 1)
        layer_bbox_targets = targets.bbox_targets[:, start:end, :]
        # code_weights rescale each regression channel; negatives already carry zero weight, so
        # the masking and the channel weighting fold into one multiply.
        layer_bbox_weights = targets.bbox_weights[:, start:end, :] * preds.new_tensor(
            self.code_weights
        )
        loss_bbox = self.loss_bbox(preds, layer_bbox_targets)
        loss_bbox = (loss_bbox * layer_bbox_weights).sum() / max(targets.num_pos, 1)

        loss_iou = None
        if self.iou_loss is not None:
            loss_iou = self._build_layer_iou_loss(
                preds=preds,
                bbox_targets=layer_bbox_targets,
                labels=targets.labels[:, start:end],
                pos_masks=targets.bbox_weights[:, start:end, 0] > 0,
                num_pos=targets.num_pos,
            )

        return LayerLosses(
            loss_cls=self.loss_cls_weight * loss_cls,
            loss_bbox=self.loss_bbox_weight * loss_bbox,
            loss_iou=loss_iou,
        )

    def _build_layer_iou_loss(
        self,
        preds: Float32[torch.Tensor, "batch_size num_proposals code_size"],
        bbox_targets: Float32[torch.Tensor, "batch_size num_proposals code_size"],
        labels: Int64[torch.Tensor, "batch_size num_proposals"],
        pos_masks: Bool[torch.Tensor, "batch_size num_proposals"],
        num_pos: int,
    ) -> torch.Tensor:
        """
        Compute one decoder layer's rotated BEV IoU loss over its matched proposals.

        Only the positives are decoded: negatives carry arbitrary regression outputs whose
        exponentiated sizes could overflow, and a zero weight would not stop a NaN from
        poisoning the sum. The matched pairs of the whole batch are gathered into one
        ``(1, num_positives)`` pseudo-batch, which is all the IoU needs. Both sides are decoded
        through ``bbox_coder`` so the loss sees metric boxes on the coder's grid. Like the L1
        box loss, the sum is normalized by the positive count over all layers.

        Args:
            preds: Raw regression outputs in the encoded box layout.
            bbox_targets: Encoded regression targets, zero for negatives.
            labels: Assigned class label per proposal, ``num_classes`` for negatives.
            pos_masks: True where the proposal was matched to a real gt box.
            num_pos: Total number of positives across all decoder layers.

        Returns:
            Scalar IoU loss for this layer. Zero, but still attached to the graph, when the
            batch has no positives.
        """
        assert self.iou_loss is not None
        if not bool(pos_masks.any()):
            return preds.sum() * 0.0

        # (num_positives, code_size) -> (1, num_positives, code_size)
        pos_preds = preds[pos_masks].unsqueeze(0)
        pos_targets = bbox_targets[pos_masks].unsqueeze(0)
        pos_labels = labels[pos_masks].unsqueeze(0)
        pair_losses = self.iou_loss(
            prediction_bboxes=self._decode_encoded_boxes(pos_preds),
            target_bboxes=self._decode_encoded_boxes(pos_targets),
            labels=pos_labels,
            bbox_weights=torch.ones(pos_labels.shape, device=preds.device, dtype=torch.float32),
        )
        return pair_losses.sum() / max(num_pos, 1)

    def _decode_encoded_boxes(
        self, encoded_bboxes: Float32[torch.Tensor, "batch_size num_bboxes code_size"]
    ) -> Float32[torch.Tensor, "batch_size num_bboxes 7"]:
        """Decode boxes in the encoded layout into metric boxes through the box coder.

        Decoding runs in float32 with autocast disabled: under fp16 autocast the exponentiated
        sizes overflow for log sizes above about eleven and turn the IoU into NaN.

        Args:
            encoded_bboxes: Boxes in the layout ``bbox_coder.encode`` produces, as the head
                regresses them: BEV center, z, log sizes, (sin, cos) yaw and optional
                velocities.

        Returns:
            Metric boxes as (x, y, z, length, width, height, yaw).
        """
        encoded_bboxes = encoded_bboxes.float()
        with torch.autocast(device_type=encoded_bboxes.device.type, enabled=False):
            # (batch_size, num_bboxes, code_size) -> (batch_size, code_size, num_bboxes), the
            # channel-first layout decode_boxes reads.
            channels = encoded_bboxes.permute(0, 2, 1)
            return self.bbox_coder.decode_boxes(
                rots=channels[:, 6:8],
                dims=channels[:, 3:6],
                centers=channels[:, 0:2],
                heights=channels[:, 2:3],
                vels=None,
            )

    def prepare_for_export(self) -> TransFusionHead:
        """Return an export-ready copy with attention replaced by exportable equivalents.

        Returns:
            Deep copy of the head with MultiheadAttention layers replaced by
            ExportableMultiheadAttention in all decoder layers.
        """
        head = deepcopy(self).eval()
        if not hasattr(head, "decoder"):
            return head
        for decoder_layer in head.decoder:
            if isinstance(decoder_layer.self_attn, nn.MultiheadAttention):
                decoder_layer.self_attn = ExportableMultiheadAttention(
                    decoder_layer.self_attn,
                    fuse_attention=head.use_bf16_cross_attention,
                )
            if isinstance(decoder_layer.cross_attn, nn.MultiheadAttention):
                decoder_layer.cross_attn = ExportableMultiheadAttention(
                    decoder_layer.cross_attn,
                    fuse_attention=head.use_bf16_cross_attention,
                    use_bf16=head.use_bf16_cross_attention,
                )
        return head
