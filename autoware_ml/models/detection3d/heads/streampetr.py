"""Native StreamPETR head for camera-only 3D detection.

The implementation keeps the high-level StreamPETR behavior that matters for
training parity: geometry-aware query initialization, a temporal memory bank,
multi-layer decoder supervision, and optional denoising queries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import (
    NMSFreeBBoxCoder3D,
    denormalize_boxes3d,
    normalize_boxes3d,
)
from autoware_ml.models.detection3d.task_modules.streaming import (
    ModulatedLayerNorm,
    SELayerLinear,
    inverse_sigmoid,
    memory_refresh,
    nerf_positional_encoding,
    pos2posemb1d,
    pos2posemb3d,
    topk_gather,
    transform_reference_points,
)


class StreamPETRDecoderLayer(nn.Module):
    """Transformer decoder layer used by the native StreamPETR head."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_channels, embed_dims),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        memory_pos: torch.Tensor,
        temp_memory: torch.Tensor,
        temp_pos: torch.Tensor,
        query_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Refine object queries with hybrid self-attention and image cross-attention.

        Self-attention keys and values extend the current queries with the
        non-propagated temporal memory; cross-attention consumes only the
        current-frame image tokens. Self-attention runs in float32 even under
        autocast, matching the reference numerical recipe.
        """
        query_with_pos = query + query_pos
        self_key = torch.cat([query_with_pos, temp_memory + temp_pos], dim=1)
        self_value = torch.cat([query, temp_memory], dim=1)
        with torch.autocast(device_type=query.device.type, enabled=False):
            self_attended, _ = self.self_attn(
                query_with_pos.float(),
                self_key.float(),
                self_value.float(),
                attn_mask=query_attn_mask,
                need_weights=False,
            )
        query = self.norm1(query + self.dropout(self_attended.to(query.dtype)))

        cross_attended, _ = self.cross_attn(
            query + query_pos,
            memory + memory_pos,
            memory,
            need_weights=False,
        )
        query = self.norm2(query + self.dropout(cross_attended))

        ffn_output = self.ffn(query)
        return self.norm3(query + self.dropout(ffn_output))


@dataclass
class StreamPETRTargets:
    """Assignment targets for one decoder layer and one batch element."""

    labels: torch.Tensor
    bbox_targets: torch.Tensor
    bbox_weights: torch.Tensor


class StreamPETRHead(nn.Module):
    """Native StreamPETR query head."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        hidden_dim: int,
        num_queries: int,
        num_decoder_layers: int,
        num_heads: int,
        feedforward_channels: int,
        bbox_coder: NMSFreeBBoxCoder3D,
        assigner: HungarianAssigner3D,
        point_cloud_range: list[float],
        code_weights: list[float],
        position_range: list[float],
        num_reg_fcs: int = 2,
        memory_len: int = 1024,
        topk_proposals: int = 256,
        num_propagated: int = 256,
        with_dn: bool = True,
        with_ego_pos: bool = True,
        depth_num: int = 64,
        LID: bool = True,
        depth_start: float = 1.0,
        loss_cls_weight: float = 2.0,
        loss_bbox_weight: float = 0.25,
        scalar: int = 10,
        noise_scale: float = 1.0,
        noise_trans: float = 0.0,
        dn_weight: float = 1.0,
        split: float = 0.75,
        use_bottom_center: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.hidden_dim = hidden_dim
        self.point_cloud_range = point_cloud_range
        self.position_range = position_range
        self.assigner = assigner
        self.bbox_coder = bbox_coder
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.depth_num = depth_num
        self.depth_start = depth_start
        self.LID = LID
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split
        self.use_bottom_center = use_bottom_center

        self.memory_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position_encoder = nn.Sequential(
            nn.Linear(depth_num * 3, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.featurized_pe = SELayerLinear(hidden_dim)
        self.reference_points = nn.Embedding(num_queries, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(hidden_dim * 3 // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.spatial_alignment = ModulatedLayerNorm(8, hidden_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        if with_ego_pos:
            self.ego_pose_pe = ModulatedLayerNorm(180, hidden_dim)
            self.ego_pose_memory = ModulatedLayerNorm(180, hidden_dim)

        self.decoder = nn.ModuleList(
            [
                StreamPETRDecoderLayer(hidden_dim, num_heads, feedforward_channels)
                for _ in range(num_decoder_layers)
            ]
        )
        # Shared final normalization applied to every decoder layer's output
        # before prediction and memory propagation; the raw query continues
        # through the decoder stack.
        self.post_norm = nn.LayerNorm(hidden_dim)
        # One classification branch and one regression branch are shared by
        # every decoder layer, following the reference StreamPETR recipe:
        # each layer's supervision updates the same prediction heads.
        cls_branch = self._build_cls_branch(hidden_dim, num_reg_fcs)
        reg_branch = self._build_reg_branch(hidden_dim, num_reg_fcs)
        self.cls_branches = nn.ModuleList([cls_branch for _ in range(num_decoder_layers)])
        self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_decoder_layers)])
        self.loss_cls = SigmoidFocalLoss()
        self.loss_bbox = nn.L1Loss(reduction="none")

        self.register_buffer(
            "pc_range", torch.tensor(point_cloud_range, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "code_weights", torch.tensor(code_weights, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "position_range_tensor",
            torch.tensor(self.position_range, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("coords_d", self._build_depth_bins(), persistent=False)
        self._init_pseudo_reference_points()
        self.reset_memory()
        self.init_weights()

    def _build_cls_branch(self, hidden_dim: int, num_reg_fcs: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for _ in range(num_reg_fcs):
            layers.extend(
                [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True)]
            )
        layers.append(nn.Linear(hidden_dim, self.num_classes))
        return nn.Sequential(*layers)

    def _build_reg_branch(self, hidden_dim: int, num_reg_fcs: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for _ in range(num_reg_fcs):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
        layers.append(nn.Linear(hidden_dim, 10))
        return nn.Sequential(*layers)

    def _build_depth_bins(self) -> torch.Tensor:
        if self.LID:
            index = torch.arange(start=0, end=self.depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (self.position_range_tensor[3] - self.depth_start) / (
                self.depth_num * (1 + self.depth_num)
            )
            return self.depth_start + bin_size * index * index_1
        index = torch.arange(start=0, end=self.depth_num, step=1).float()
        bin_size = (self.position_range_tensor[3] - self.depth_start) / self.depth_num
        return self.depth_start + bin_size * index

    def _init_pseudo_reference_points(self) -> None:
        if self.num_propagated <= 0:
            self.register_parameter("pseudo_reference_points", None)
            return

        num_divisions = round(self.num_propagated ** (1 / 3) + 1)
        linspace = torch.linspace(0.0, 1.0, steps=num_divisions + 1)
        centers = (linspace[:-1] + linspace[1:]) / 2
        grid = torch.meshgrid(centers, centers, centers, indexing="ij")
        points = torch.stack(grid, dim=-1).reshape(-1, 3)[: self.num_propagated]
        self.pseudo_reference_points = nn.Parameter(points, requires_grad=False)

    def init_weights(self) -> None:
        """Initialize reference points, decoder weights, and classification priors."""
        nn.init.uniform_(self.reference_points.weight.data, 0.0, 1.0)
        for parameter in self.decoder.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        # Focal-loss prior: start every class at ~1% foreground probability so
        # the first iterations are not dominated by massive negative losses.
        bias_init = -math.log((1.0 - 0.01) / 0.01)
        for cls_branch in self.cls_branches:
            nn.init.constant_(cls_branch[-1].bias, bias_init)

    def reset_memory(self) -> None:
        """Reset the temporal memory bank."""
        self.memory_embedding: torch.Tensor | None = None
        self.memory_reference_point: torch.Tensor | None = None
        self.memory_timestamp: torch.Tensor | None = None
        self.memory_egopose: torch.Tensor | None = None
        self.memory_velo: torch.Tensor | None = None

    def _build_stream_state(
        self,
        device: torch.device,
        timestamp: torch.Tensor | None,
        prev_exists: torch.Tensor | None,
        ego_pose: torch.Tensor | None,
        ego_pose_inv: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if timestamp is None or prev_exists is None or ego_pose is None or ego_pose_inv is None:
            raise ValueError(
                "StreamPETRHead requires timestamp, prev_exists, ego_pose, and ego_pose_inv "
                "for every frame; check the datamodule collation_map and annotation files."
            )
        return {
            "timestamp": timestamp.to(device=device, dtype=torch.float64),
            "prev_exists": prev_exists.to(device=device, dtype=torch.float32),
            "ego_pose": ego_pose.to(device=device, dtype=torch.float32),
            "ego_pose_inv": ego_pose_inv.to(device=device, dtype=torch.float32),
        }

    def pre_update_memory(self, data: dict[str, torch.Tensor]) -> None:
        """Update or initialize the temporal memory before the current frame.

        The memory bank stores global-frame reference points and ego poses
        whose translations reach kilometer scale, far beyond half-precision
        resolution, so the whole geometry path runs in float32 regardless of
        autocast (mirroring :meth:`position_embedding`).
        """
        prev_exists = data["prev_exists"]
        batch_size = prev_exists.size(0)
        with torch.autocast(device_type=prev_exists.device.type, enabled=False):
            if self.memory_embedding is None or self.memory_embedding.size(0) != batch_size:
                self.memory_embedding = prev_exists.new_zeros(
                    batch_size, self.memory_len, self.hidden_dim
                )
                self.memory_reference_point = prev_exists.new_zeros(batch_size, self.memory_len, 3)
                self.memory_timestamp = prev_exists.new_zeros(
                    batch_size, self.memory_len, 1, dtype=torch.float64
                )
                self.memory_egopose = prev_exists.new_zeros(batch_size, self.memory_len, 4, 4)
                self.memory_velo = prev_exists.new_zeros(batch_size, self.memory_len, 2)
            else:
                self.memory_timestamp += data["timestamp"].unsqueeze(-1).unsqueeze(-1)
                self.memory_egopose = data["ego_pose_inv"].unsqueeze(1) @ self.memory_egopose
                self.memory_reference_point = transform_reference_points(
                    self.memory_reference_point, data["ego_pose_inv"]
                )
                self.memory_timestamp = memory_refresh(
                    self.memory_timestamp[:, : self.memory_len], prev_exists
                )
                self.memory_reference_point = memory_refresh(
                    self.memory_reference_point[:, : self.memory_len], prev_exists
                )
                self.memory_embedding = memory_refresh(
                    self.memory_embedding[:, : self.memory_len], prev_exists
                )
                self.memory_egopose = memory_refresh(
                    self.memory_egopose[:, : self.memory_len], prev_exists
                )
                self.memory_velo = memory_refresh(
                    self.memory_velo[:, : self.memory_len], prev_exists
                )

            if self.num_propagated > 0 and self.pseudo_reference_points is not None:
                pseudo_reference_points = (
                    self.pseudo_reference_points * (self.pc_range[3:6] - self.pc_range[:3])
                    + self.pc_range[:3]
                )
                self.memory_reference_point[:, : self.num_propagated] += (1 - prev_exists).view(
                    batch_size, 1, 1
                ) * pseudo_reference_points
                identity = torch.eye(4, device=prev_exists.device, dtype=torch.float32)
                self.memory_egopose[:, : self.num_propagated] += (1 - prev_exists).view(
                    batch_size, 1, 1, 1
                ) * identity

    def post_update_memory(
        self,
        data: dict[str, torch.Tensor],
        rec_ego_pose: torch.Tensor,
        all_cls_scores: torch.Tensor,
        all_bbox_preds: torch.Tensor,
        decoder_outputs: torch.Tensor,
        mask_dict: dict[str, torch.Tensor] | None,
    ) -> None:
        """Append the strongest current-frame proposals to the temporal memory.

        The bank is stored in float32 and the global-frame alignment runs in a
        float32 island: kilometer-scale translations lose meter-level motion in
        half precision, so none of this may execute under autocast.
        """
        with torch.autocast(device_type=all_cls_scores.device.type, enabled=False):
            if mask_dict and mask_dict["pad_size"] > 0:
                rec_reference_points = all_bbox_preds[-1, :, mask_dict["pad_size"] :, :3].float()
                rec_velo = all_bbox_preds[-1, :, mask_dict["pad_size"] :, -2:].float()
                rec_memory = decoder_outputs[-1, :, mask_dict["pad_size"] :, :].float()
                rec_score = (
                    all_cls_scores[-1, :, mask_dict["pad_size"] :, :]
                    .float()
                    .sigmoid()
                    .amax(dim=-1, keepdim=True)
                )
            else:
                rec_reference_points = all_bbox_preds[-1, :, :, :3].float()
                rec_velo = all_bbox_preds[-1, :, :, -2:].float()
                rec_memory = decoder_outputs[-1].float()
                rec_score = all_cls_scores[-1].float().sigmoid().amax(dim=-1, keepdim=True)
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)

            topk_proposals = min(self.topk_proposals, rec_score.shape[1])
            _, topk_indexes = torch.topk(rec_score, topk_proposals, dim=1)
            rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
            rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
            rec_memory = topk_gather(rec_memory, topk_indexes).detach()
            rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
            rec_velo = topk_gather(rec_velo, topk_indexes).detach()

            self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)[
                :, : self.memory_len
            ]
            self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)[
                :, : self.memory_len
            ]
            self.memory_egopose = torch.cat([rec_ego_pose, self.memory_egopose], dim=1)[
                :, : self.memory_len
            ]
            self.memory_reference_point = torch.cat(
                [rec_reference_points, self.memory_reference_point], dim=1
            )[:, : self.memory_len]
            self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)[:, : self.memory_len]
            self.memory_reference_point = transform_reference_points(
                self.memory_reference_point, data["ego_pose"]
            )
            self.memory_timestamp -= data["timestamp"].unsqueeze(-1).unsqueeze(-1)
            self.memory_egopose = data["ego_pose"].unsqueeze(1) @ self.memory_egopose

    def _build_memory_centers(
        self,
        batch_size: int,
        num_cams: int,
        feature_height: int,
        feature_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        y = (
            torch.arange(feature_height, device=device, dtype=torch.float32) + 0.5
        ) / feature_height
        x = (torch.arange(feature_width, device=device, dtype=torch.float32) + 0.5) / feature_width
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        centers = torch.stack([grid_x, grid_y], dim=-1)
        return centers.unsqueeze(0).unsqueeze(0).repeat(batch_size, num_cams, 1, 1, 1)

    def position_embedding(
        self,
        img_features: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        lidar2cam: torch.Tensor,
        image_height: int,
        image_width: int,
        lidar2img: torch.Tensor | None = None,
        img2lidar: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build geometry-aware positional embeddings for image tokens.

        The frustum projection multiplies pixel coordinates by metric depths,
        which exceeds float16 range; the whole geometry path therefore runs in
        float32 regardless of autocast.
        """
        with torch.autocast(device_type=img_features.device.type, enabled=False):
            return self._position_embedding_fp32(
                img_features,
                camera_intrinsics,
                lidar2cam,
                image_height,
                image_width,
                lidar2img,
                img2lidar,
            )

    def _position_embedding_fp32(
        self,
        img_features: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        lidar2cam: torch.Tensor | None,
        image_height: int,
        image_width: int,
        lidar2img: torch.Tensor | None = None,
        img2lidar: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Float32 body of :meth:`position_embedding`."""
        batch_size, num_cams, _, feature_height, feature_width = img_features.shape
        device = img_features.device
        memory_centers = self._build_memory_centers(
            batch_size, num_cams, feature_height, feature_width, device
        )
        batch_tokens = batch_size * num_cams
        token_count = num_cams * feature_height * feature_width
        depth_count = self.coords_d.shape[0]

        intrinsics = torch.stack(
            [camera_intrinsics[..., 0, 0], camera_intrinsics[..., 1, 1]], dim=-1
        )
        intrinsics = torch.abs(intrinsics) / 1e3
        intrinsics = intrinsics.repeat(1, feature_height * feature_width, 1).view(
            batch_size, token_count, 2
        )

        memory_centers[..., 0] *= image_width
        memory_centers[..., 1] *= image_height
        memory_centers = memory_centers.view(batch_size, token_count, 1, 2)
        centers = memory_centers.repeat(1, 1, depth_count, 1)
        coords_d = self.coords_d.view(1, 1, depth_count, 1).repeat(batch_size, token_count, 1, 1)
        coords = torch.cat([centers, coords_d], dim=-1)
        coords = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1)
        coords[..., :2] *= torch.maximum(coords[..., 2:3], torch.full_like(coords[..., 2:3], 1e-5))
        coords = coords.unsqueeze(-1)

        if img2lidar is None:
            if lidar2img is None:
                lidar2img = camera_intrinsics @ lidar2cam
            img2lidar = torch.inverse(lidar2img)
        # One matrix per token; matmul broadcasts it over the depth bins.
        img2lidar = img2lidar.view(batch_tokens, 1, 4, 4).repeat(
            1, feature_height * feature_width, 1, 1
        )
        img2lidar = img2lidar.view(batch_size, token_count, 1, 4, 4)

        coords3d = torch.matmul(img2lidar, coords).squeeze(-1)[..., :3]
        coords3d = (coords3d - self.position_range_tensor[:3]) / (
            self.position_range_tensor[3:6] - self.position_range_tensor[:3]
        )
        coords3d = coords3d.reshape(batch_size, -1, depth_count * 3)

        pos_embed = self.position_encoder(inverse_sigmoid(coords3d))

        # Reference cone layout: xyz at the farthest depth bin, then xyz
        # thirty bins from the far end (coords3d[..., -3:] and [..., -90:-87]
        # on the flattened depth axis), clamped for shallower depth configs.
        depth_grid = coords3d.view(batch_size, -1, depth_count, 3)
        mid_bin = max(depth_count - 30, 0)
        cone = torch.cat([intrinsics, depth_grid[..., -1, :], depth_grid[..., mid_bin, :]], dim=-1)
        return pos_embed, cone

    def temporal_alignment(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align temporal memory and propagate its strongest entries as queries.

        The first ``num_propagated`` memory entries extend the query set; the
        remaining entries are returned as extra self-attention keys.
        """
        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (
            self.pc_range[3:6] - self.pc_range[:3]
        )
        temp_pos = self.query_embedding(pos2posemb3d(temp_reference_point, self.hidden_dim // 2))
        temp_memory = self.memory_embedding

        if self.with_ego_pos:
            identity_ego_pose = (
                torch.eye(4, device=query.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(query_pos.size(0), query_pos.size(1), 1, 1)
            )
            rec_ego_motion = torch.cat(
                [
                    torch.zeros_like(reference_points[..., :3]),
                    identity_ego_pose[..., :3, :].flatten(-2),
                ],
                dim=-1,
            )
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            query = self.ego_pose_memory(query, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)

            memory_ego_motion = torch.cat(
                [
                    self.memory_velo,
                    self.memory_timestamp,
                    self.memory_egopose[..., :3, :].flatten(-2),
                ],
                dim=-1,
            ).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        query_pos = query_pos + self.time_embedding(
            pos2posemb1d(torch.zeros_like(reference_points[..., :1]), self.hidden_dim)
        )
        temp_pos = temp_pos + self.time_embedding(
            pos2posemb1d(self.memory_timestamp, self.hidden_dim).float()
        )

        if self.num_propagated > 0:
            query = torch.cat([query, temp_memory[:, : self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, : self.num_propagated]], dim=1)
            reference_points = torch.cat(
                [reference_points, temp_reference_point[:, : self.num_propagated]], dim=1
            )
            temp_memory = temp_memory[:, self.num_propagated :]
            temp_pos = temp_pos[:, self.num_propagated :]

        return query, query_pos, reference_points, temp_memory, temp_pos

    def prepare_for_dn(
        self,
        batch_size: int,
        reference_points: torch.Tensor,
        gt_boxes: list[torch.Tensor] | None,
        gt_labels: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor] | None]:
        """Prepare denoising queries following the StreamPETR training recipe."""
        if not self.training or not self.with_dn or gt_boxes is None or gt_labels is None:
            return reference_points.unsqueeze(0).repeat(batch_size, 1, 1), None, None

        known_num = [boxes.shape[0] for boxes in gt_boxes]
        if max(known_num, default=0) == 0:
            return reference_points.unsqueeze(0).repeat(batch_size, 1, 1), None, None

        labels = torch.cat(gt_labels, dim=0)
        boxes = torch.cat(gt_boxes, dim=0)
        batch_idx = torch.cat(
            [
                torch.full(
                    (boxes_per_sample.shape[0],),
                    sample_index,
                    device=reference_points.device,
                    dtype=torch.long,
                )
                for sample_index, boxes_per_sample in enumerate(gt_boxes)
            ],
            dim=0,
        )

        known_indices = torch.arange(labels.shape[0], device=reference_points.device).repeat(
            self.scalar
        )
        known_labels = labels.repeat(self.scalar).long().to(reference_points.device)
        known_batch_idx = batch_idx.repeat(self.scalar)
        known_boxes = boxes.repeat(self.scalar, 1).to(reference_points.device)
        known_bbox_center = known_boxes[:, :3].clone()
        known_bbox_scale = known_boxes[:, 3:6].clone()

        if self.bbox_noise_scale > 0:
            diff = known_bbox_scale / 2 + self.bbox_noise_trans
            rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
            known_bbox_center += rand_prob * diff * self.bbox_noise_scale
            known_bbox_center = (known_bbox_center - self.pc_range[:3]) / (
                self.pc_range[3:6] - self.pc_range[:3]
            )
            known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
            background_mask = torch.norm(rand_prob, dim=1) > self.split
            known_labels[background_mask] = self.num_classes
        else:
            known_bbox_center = (known_bbox_center - self.pc_range[:3]) / (
                self.pc_range[3:6] - self.pc_range[:3]
            )

        single_pad = int(max(known_num))
        pad_size = int(single_pad * self.scalar)
        padded_reference_points = (
            torch.cat(
                [torch.zeros(pad_size, 3, device=reference_points.device), reference_points],
                dim=0,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        map_known_indice = torch.cat(
            [torch.arange(num, device=reference_points.device) for num in known_num], dim=0
        )
        map_known_indice = torch.cat(
            [map_known_indice + single_pad * group_index for group_index in range(self.scalar)],
            dim=0,
        ).long()
        padded_reference_points[(known_batch_idx.long(), map_known_indice)] = known_bbox_center

        dn_size = pad_size + self.num_queries
        attn_mask = torch.zeros(dn_size, dn_size, dtype=torch.bool, device=reference_points.device)
        attn_mask[pad_size:, :pad_size] = True
        for group_index in range(self.scalar):
            start = single_pad * group_index
            end = single_pad * (group_index + 1)
            if group_index > 0:
                attn_mask[start:end, :start] = True
            if group_index < self.scalar - 1:
                attn_mask[start:end, end:pad_size] = True

        # Hybrid self-attention rows cover the propagated queries and its keys
        # additionally cover the non-propagated temporal memory.
        query_size = pad_size + self.num_queries + self.num_propagated
        key_size = pad_size + self.num_queries + self.memory_len
        temporal_attn_mask = torch.zeros(
            query_size, key_size, dtype=torch.bool, device=reference_points.device
        )
        temporal_attn_mask[: attn_mask.size(0), : attn_mask.size(1)] = attn_mask
        temporal_attn_mask[pad_size:, :pad_size] = True
        attn_mask = temporal_attn_mask

        mask_dict = {
            "known_indices": known_indices.long(),
            "batch_idx": known_batch_idx.long(),
            "map_known_indice": map_known_indice.long(),
            "known_lbs_bboxes": (known_labels, known_boxes),
            "pad_size": pad_size,
        }
        return padded_reference_points, attn_mask, mask_dict

    def _decode_box_params(
        self, raw_box_params: torch.Tensor, reference_points: torch.Tensor
    ) -> torch.Tensor:
        """Decode regression outputs into the shared box parameterization.

        Boxes are ``[cx, cy, cz, log_l, log_w, log_h, sin, cos, vx, vy]`` with
        metric centers: the regression space used by the losses and the
        deployment interface alike. Sizes stay in log space so no exp/log
        round trip can produce non-finite values.
        """
        reference = inverse_sigmoid(reference_points.clone())
        normalized_centers = (raw_box_params[..., :3] + reference[..., :3]).sigmoid()
        centers = normalized_centers * (self.pc_range[3:6] - self.pc_range[:3]) + self.pc_range[:3]
        return torch.cat([centers, raw_box_params[..., 3:]], dim=-1)

    def _gravity_center_boxes(self, boxes: list[torch.Tensor]) -> list[torch.Tensor]:
        """Shift bottom-center ground-truth boxes to gravity centers.

        The head regresses gravity-center z (the reference StreamPETR space);
        :meth:`predict` shifts decoded boxes back to bottom centers.
        """
        if not self.use_bottom_center:
            return list(boxes)
        shifted = []
        for sample_boxes in boxes:
            sample_boxes = sample_boxes.clone()
            sample_boxes[:, 2] = sample_boxes[:, 2] + sample_boxes[:, 5] * 0.5
            shifted.append(sample_boxes)
        return shifted

    def _get_targets(
        self,
        cls_logits: torch.Tensor,
        box_params: torch.Tensor,
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
    ) -> list[StreamPETRTargets]:
        targets: list[StreamPETRTargets] = []
        for sample_logits, sample_boxes, sample_gt_boxes, sample_gt_labels in zip(
            cls_logits, box_params, gt_boxes, gt_labels
        ):
            num_queries = sample_logits.shape[0]
            labels = sample_gt_labels.new_full((num_queries,), -1)
            bbox_targets = sample_boxes.new_zeros((num_queries, 9))
            bbox_weights = sample_boxes.new_zeros((num_queries, 9))

            assigned = self.assigner.assign(
                bboxes=denormalize_boxes3d(sample_boxes),
                gt_bboxes=sample_gt_boxes,
                gt_labels=sample_gt_labels,
                cls_pred=sample_logits.transpose(0, 1),
                point_cloud_range=self.point_cloud_range,
            )
            pos_inds = torch.nonzero(assigned.gt_inds > 0, as_tuple=False).squeeze(-1)
            if pos_inds.numel() > 0:
                matched_gt_inds = assigned.gt_inds[pos_inds] - 1
                labels[pos_inds] = sample_gt_labels[matched_gt_inds]
                bbox_targets[pos_inds] = sample_gt_boxes[matched_gt_inds]
                bbox_weights[pos_inds] = 1.0
            targets.append(
                StreamPETRTargets(
                    labels=labels, bbox_targets=bbox_targets, bbox_weights=bbox_weights
                )
            )
        return targets

    def _loss_single(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = self._get_targets(cls_scores, bbox_preds, gt_boxes, gt_labels)

        target_labels = []
        positive_counts = []
        for sample_targets in targets:
            one_hot = cls_scores.new_zeros((sample_targets.labels.shape[0], self.num_classes))
            pos_mask = sample_targets.labels >= 0
            positive_counts.append(pos_mask.sum())
            one_hot[pos_mask, sample_targets.labels[pos_mask]] = 1.0
            target_labels.append(one_hot)
        target_labels_tensor = torch.stack(target_labels, dim=0)
        # One device sync per decoder layer instead of one per sample.
        total_pos = int(torch.stack(positive_counts).sum().item())
        loss_cls = self.loss_cls_weight * self.loss_cls(
            cls_scores, target_labels_tensor, avg_factor=max(total_pos, 1)
        )

        # The regression loss runs over every query with 0/1 positive weights so
        # the regression branches join the backward graph on every rank each
        # step; positive-free batches contribute exactly zero.
        encoded_preds = bbox_preds
        target_encodings = []
        positive_weights = []
        for sample_targets in targets:
            encoding = encoded_preds.new_zeros((sample_targets.labels.shape[0], 10))
            pos_mask = sample_targets.bbox_weights.sum(dim=1) > 0
            encoding[pos_mask] = normalize_boxes3d(sample_targets.bbox_targets[pos_mask])
            target_encodings.append(encoding)
            positive_weights.append(pos_mask.to(encoding.dtype))
        target_tensor = torch.stack(target_encodings, dim=0)
        weight_tensor = torch.stack(positive_weights, dim=0).unsqueeze(-1)
        per_box = self.loss_bbox(encoded_preds[..., :10], target_tensor) * self.code_weights
        bbox_loss = self.loss_bbox_weight * (per_box * weight_tensor).sum() / max(total_pos, 1)
        return loss_cls, bbox_loss

    def prepare_for_loss(
        self,
        mask_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Gather denoising outputs aligned with the replicated GT targets."""
        output_known_class, output_known_coord = mask_dict["output_known_lbs_bboxes"]
        known_labels, known_bboxs = mask_dict["known_lbs_bboxes"]
        map_known_indice = mask_dict["map_known_indice"].long()
        known_indices = mask_dict["known_indices"].long()
        batch_idx = mask_dict["batch_idx"].long()
        batch_selection = batch_idx[known_indices]

        if output_known_class.numel() > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[
                (batch_selection, map_known_indice)
            ].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[
                (batch_selection, map_known_indice)
            ].permute(1, 0, 2)
        num_targets = known_indices.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_targets

    def dn_loss_single(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        known_bboxs: torch.Tensor,
        known_labels: torch.Tensor,
        num_total_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute denoising-query supervision for one decoder layer."""
        class_targets = cls_scores.new_zeros((known_labels.shape[0], self.num_classes))
        foreground = known_labels < self.num_classes
        if foreground.any():
            class_targets[foreground, known_labels[foreground]] = 1.0
        # The classification average factor scales the target count by the
        # expected positive rate of the noised queries (reference recipe):
        # a ball of radius ``split`` inside the unit noise cube.
        cls_avg_factor = num_total_pos * math.pi / 6 * self.split**3
        loss_cls = (
            self.dn_weight
            * self.loss_cls_weight
            * self.loss_cls(cls_scores, class_targets, avg_factor=cls_avg_factor)
        )

        target_encoding = normalize_boxes3d(known_bboxs)
        loss_bbox = (
            self.dn_weight
            * self.loss_bbox_weight
            * (
                self.loss_bbox(bbox_preds[:, :10], target_encoding[:, :10]) * self.code_weights
            ).sum()
            / max(num_total_pos, 1)
        )
        return loss_cls, loss_bbox

    def forward(
        self,
        img_features: torch.Tensor,
        img: torch.Tensor,
        camera_intrinsics: torch.Tensor | None = None,
        lidar2cam: torch.Tensor | None = None,
        lidar2img: torch.Tensor | None = None,
        timestamp: torch.Tensor | None = None,
        prev_exists: torch.Tensor | None = None,
        ego_pose: torch.Tensor | None = None,
        ego_pose_inv: torch.Tensor | None = None,
        gt_boxes: list[torch.Tensor] | None = None,
        gt_labels: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict StreamPETR query outputs from multiview image features."""
        if camera_intrinsics is None or lidar2cam is None:
            raise ValueError(
                "StreamPETR requires camera_intrinsics and lidar2cam for geometry-aware decoding."
            )

        batch_size = img_features.shape[0]
        device = img_features.device
        camera_intrinsics = camera_intrinsics.to(device=device, dtype=torch.float32)
        lidar2cam = lidar2cam.to(device=device, dtype=torch.float32)
        if lidar2img is not None:
            lidar2img = lidar2img.to(device=device, dtype=torch.float32)
        image_height = int(img.shape[-2])
        image_width = int(img.shape[-1])

        stream_state = self._build_stream_state(
            device, timestamp, prev_exists, ego_pose, ego_pose_inv
        )
        self.pre_update_memory(stream_state)

        batch_size, num_cams, channels, feature_height, feature_width = img_features.shape
        memory = img_features.permute(0, 1, 3, 4, 2).reshape(
            batch_size, num_cams * feature_height * feature_width, channels
        )

        pos_embed, cone = self.position_embedding(
            img_features,
            camera_intrinsics,
            lidar2cam,
            image_height,
            image_width,
            lidar2img=lidar2img,
        )
        memory = self.memory_embed(memory)
        memory = self.spatial_alignment(memory, cone)
        pos_embed = self.featurized_pe(pos_embed, memory)

        reference_points = self.reference_points.weight
        dn_gt_boxes = self._gravity_center_boxes(gt_boxes) if gt_boxes is not None else None
        reference_points, query_attn_mask, mask_dict = self.prepare_for_dn(
            batch_size, reference_points, dn_gt_boxes, gt_labels
        )
        query_pos = self.query_embedding(pos2posemb3d(reference_points, self.hidden_dim // 2))
        query = torch.zeros_like(query_pos)
        query, query_pos, reference_points, temp_memory, temp_pos = self.temporal_alignment(
            query, query_pos, reference_points
        )
        rec_ego_pose = (
            torch.eye(4, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, query.shape[1], 1, 1)
        )

        outputs_classes = []
        outputs_coords = []
        decoder_outputs = []
        for decoder_layer, cls_branch, reg_branch in zip(
            self.decoder, self.cls_branches, self.reg_branches
        ):
            query = decoder_layer(
                query, memory, query_pos, pos_embed, temp_memory, temp_pos, query_attn_mask
            )
            # The reference sanitizes the post-normed intermediates consumed by
            # prediction and memory; the raw query flows to the next layer.
            normed_query = torch.nan_to_num(self.post_norm(query))
            with torch.autocast(device_type=query.device.type, enabled=False):
                query_fp32 = normed_query.float()
                cls_logits = cls_branch(query_fp32)
                box_params = self._decode_box_params(reg_branch(query_fp32), reference_points)
            outputs_classes.append(cls_logits)
            outputs_coords.append(box_params)
            decoder_outputs.append(normed_query)

        all_cls_scores = torch.stack(outputs_classes, dim=0)
        all_bbox_preds = torch.stack(outputs_coords, dim=0)
        decoder_outputs_tensor = torch.stack(decoder_outputs, dim=0)

        self.post_update_memory(
            stream_state,
            rec_ego_pose,
            all_cls_scores,
            all_bbox_preds,
            decoder_outputs_tensor,
            mask_dict,
        )

        if mask_dict and mask_dict["pad_size"] > 0:
            output_known_class = all_cls_scores[:, :, : mask_dict["pad_size"], :]
            output_known_coord = all_bbox_preds[:, :, : mask_dict["pad_size"], :]
            all_cls_scores = all_cls_scores[:, :, mask_dict["pad_size"] :, :]
            all_bbox_preds = all_bbox_preds[:, :, mask_dict["pad_size"] :, :]
            mask_dict["output_known_lbs_bboxes"] = (output_known_class, output_known_coord)

        return {
            "all_cls_scores": all_cls_scores,
            "all_bbox_preds": all_bbox_preds,
            "dn_mask_dict": mask_dict,
            "cls_logits": all_cls_scores[-1],
            "box_params": all_bbox_preds[-1],
        }

    def loss(
        self,
        outputs: dict[str, torch.Tensor],
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute multi-layer StreamPETR losses."""
        all_cls_scores = outputs["all_cls_scores"]
        all_bbox_preds = outputs["all_bbox_preds"]
        gt_boxes = self._gravity_center_boxes(gt_boxes)

        losses_cls = []
        losses_bbox = []
        for cls_scores, bbox_preds in zip(all_cls_scores, all_bbox_preds):
            loss_cls, loss_bbox = self._loss_single(cls_scores, bbox_preds, gt_boxes, gt_labels)
            losses_cls.append(loss_cls)
            losses_bbox.append(loss_bbox)

        loss_dict: dict[str, torch.Tensor] = {
            "loss_cls": losses_cls[-1],
            "loss_bbox": losses_bbox[-1],
        }
        total_loss = losses_cls[-1] + losses_bbox[-1]
        for layer_index, (loss_cls, loss_bbox) in enumerate(zip(losses_cls[:-1], losses_bbox[:-1])):
            loss_dict[f"d{layer_index}.loss_cls"] = loss_cls
            loss_dict[f"d{layer_index}.loss_bbox"] = loss_bbox
            total_loss = total_loss + loss_cls + loss_bbox

        if outputs["dn_mask_dict"] is None and self.with_dn and self.training:
            # Keep the loss dictionary uniform across ranks when a batch has
            # no ground truth: every logged key is a distributed collective.
            zero = all_cls_scores.new_tensor(0.0)
            loss_dict["dn_loss_cls"] = zero
            loss_dict["dn_loss_bbox"] = zero.clone()
            for layer_index in range(self.num_decoder_layers - 1):
                loss_dict[f"d{layer_index}.dn_loss_cls"] = zero.clone()
                loss_dict[f"d{layer_index}.dn_loss_bbox"] = zero.clone()

        if outputs["dn_mask_dict"] is not None:
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = (
                self.prepare_for_loss(outputs["dn_mask_dict"])
            )
            dn_losses_cls = []
            dn_losses_bbox = []
            for cls_scores, bbox_preds in zip(output_known_class, output_known_coord):
                dn_loss_cls, dn_loss_bbox = self.dn_loss_single(
                    cls_scores, bbox_preds, known_bboxs, known_labels, num_tgt
                )
                dn_losses_cls.append(dn_loss_cls)
                dn_losses_bbox.append(dn_loss_bbox)
            loss_dict["dn_loss_cls"] = dn_losses_cls[-1]
            loss_dict["dn_loss_bbox"] = dn_losses_bbox[-1]
            total_loss = total_loss + dn_losses_cls[-1] + dn_losses_bbox[-1]
            for layer_index, (loss_cls, loss_bbox) in enumerate(
                zip(dn_losses_cls[:-1], dn_losses_bbox[:-1])
            ):
                loss_dict[f"d{layer_index}.dn_loss_cls"] = loss_cls
                loss_dict[f"d{layer_index}.dn_loss_bbox"] = loss_bbox
                total_loss = total_loss + loss_cls + loss_bbox

        loss_dict["loss"] = total_loss
        return loss_dict

    def predict(self, outputs: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Decode final detections from the last decoder layer.

        Returns the shared detection prediction contract
        (``bboxes_3d`` / ``scores_3d`` / ``labels_3d``) consumed by
        ``detection_eval_output`` and the metric suite.
        """
        predictions = self.bbox_coder.decode(outputs["cls_logits"], outputs["box_params"])
        results = []
        for prediction in predictions:
            boxes = prediction["bboxes"]
            if self.use_bottom_center:
                boxes = boxes.clone()
                if boxes.numel() > 0:
                    boxes[:, 2] = boxes[:, 2] - boxes[:, 5] * 0.5
            results.append(
                {
                    "bboxes_3d": boxes,
                    "scores_3d": prediction["scores"],
                    "labels_3d": prediction["labels"],
                }
            )
        return results
