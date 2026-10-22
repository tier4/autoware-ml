"""Native StreamPETR head for camera-only 3D detection.

The implementation keeps the high-level StreamPETR behavior that matters for
training parity: geometry-aware query initialization, a temporal memory bank,
multi-layer decoder supervision, and optional denoising queries.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    StreamPETRDenoisingOutputs,
    StreamPETRHeadOutputs,
)
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
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
    StreamContinuity,
    compose_poses,
    inverse_sigmoid,
    memory_refresh,
    nerf_positional_encoding,
    pos2posemb1d,
    pos2posemb3d,
    topk_gather,
    transform_reference_points,
)


# Reference recipe constants the configs never vary: two hidden layers in the
# classification and the regression branches, and the first depth bin at 1 m.
_NUM_BRANCH_LAYERS = 2
_DEPTH_START = 1.0


def identity_poses(batch_size: int, count: int, device: torch.device) -> torch.Tensor:
    """``[batch_size, count, 4, 4]`` identity poses (read-only view of one ``eye``)."""
    return torch.eye(4, device=device).expand(batch_size, count, 4, 4)


class DenoisingQueries(NamedTuple):
    """Bookkeeping of the denoising queries of one training forward.

    Attributes:
        known_labels: Class of every noised query, ``num_classes`` for a query noised beyond
            the radius that keeps it a positive.
        known_boxes: Metric box every noised query was drawn from.
        batch_indices: Sample of every noised query.
        slot_indices: Denoising slot of every noised query inside its sample.
        pad_size: Number of denoising slots leading the query set of every sample.
    """

    known_labels: torch.Tensor
    known_boxes: torch.Tensor
    batch_indices: torch.Tensor
    slot_indices: torch.Tensor
    pad_size: int


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
        autocast, matching the reference numerical recipe (one of the head's
        five float32 islands: this one, the position embedding and the
        prediction branches follow the reference, the two memory updates keep
        kilometer-scale poses exact).
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

    labels: torch.Tensor  # [Q] assigned class, -1 for unmatched queries
    bbox_targets: torch.Tensor  # [Q, 9] matched ground-truth box, zeros where unmatched


class StreamPETRHead(nn.Module):
    """Native StreamPETR query head.

    The constructor exposes the knobs of the reference recipe (``LID``,
    ``with_ego_pos``, ``num_propagated``, ``with_dn``, the denoising noise) so a
    config reads like the reference one; the shipped configs pin them to the
    reference values.
    """

    # The five-tensor temporal memory bank; ``None`` until the first frame.
    memory_embedding: torch.Tensor | None
    memory_reference_point: torch.Tensor | None
    memory_timestamp: torch.Tensor | None
    memory_egopose: torch.Tensor | None
    memory_velo: torch.Tensor | None
    # Scene and time of the previous frame of every batch lane.
    stream_continuity: StreamContinuity

    def __init__(
        self,
        class_names: Sequence[str],
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
        memory_len: int = 1024,
        topk_proposals: int = 256,
        num_propagated: int = 256,
        with_dn: bool = True,
        with_ego_pos: bool = True,
        depth_num: int = 64,
        LID: bool = True,
        loss_cls_weight: float = 2.0,
        loss_bbox_weight: float = 0.25,
        scalar: int = 10,
        noise_scale: float = 1.0,
        dn_weight: float = 1.0,
        split: float = 0.75,
        max_stream_gap: float | None = None,
    ) -> None:
        """Initialize the head.

        Args:
            class_names: Ordered names of the foreground classes. Their count is the number
                of classification channels.
            memory_len: Entries of the temporal memory bank.
            topk_proposals: Proposals of every frame that enter the bank, at most the
                ``num_queries + num_propagated`` queries of a frame. The deployed graph emits
                a bank of ``memory_len + topk_proposals`` entries, which the runtime reads
                without checking.
            num_propagated: Bank entries that extend the queries of the next frame; the
                strongest proposals of the previous frame, so at most ``topk_proposals``.
            max_stream_gap: Longest gap in seconds between two frames of a lane that keeps
                the memory, ``None`` for any gap (see :class:`StreamContinuity`).
            Other arguments follow the reference StreamPETR recipe.

        Raises:
            ValueError: If the bank is too small for the propagated entries, or a frame has
                fewer queries than ``topk_proposals``.
        """
        super().__init__()
        if memory_len < num_propagated:
            raise ValueError(
                f"memory_len ({memory_len}) must hold the {num_propagated} propagated entries."
            )
        if topk_proposals < num_propagated:
            raise ValueError(
                f"topk_proposals ({topk_proposals}) must cover the {num_propagated} propagated "
                "entries, which are the strongest proposals of the previous frame."
            )
        if topk_proposals > num_queries + num_propagated:
            raise ValueError(
                f"topk_proposals ({topk_proposals}) exceeds the {num_queries + num_propagated} "
                "queries of a frame (num_queries + num_propagated) that can enter the bank."
            )
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.hidden_dim = hidden_dim
        self.assigner = assigner
        self.bbox_coder = bbox_coder
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.depth_num = depth_num
        self.LID = LID
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.dn_weight = dn_weight
        self.split = split

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
        cls_branch = self._build_cls_branch(hidden_dim)
        reg_branch = self._build_reg_branch(hidden_dim)
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
            torch.tensor(position_range, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("coords_d", self._build_depth_bins(), persistent=False)
        self._init_pseudo_reference_points()
        self.stream_continuity = StreamContinuity(max_time_gap=max_stream_gap)
        self.reset_memory()
        self.init_weights()

    def _build_cls_branch(self, hidden_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for _ in range(_NUM_BRANCH_LAYERS):
            layers.extend(
                [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True)]
            )
        layers.append(nn.Linear(hidden_dim, self.num_classes))
        return nn.Sequential(*layers)

    def _build_reg_branch(self, hidden_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for _ in range(_NUM_BRANCH_LAYERS):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
        layers.append(nn.Linear(hidden_dim, 10))
        return nn.Sequential(*layers)

    def _build_depth_bins(self) -> torch.Tensor:
        index = torch.arange(self.depth_num, dtype=torch.float32)
        depth_span = self.position_range_tensor[3] - _DEPTH_START
        if self.LID:
            return _DEPTH_START + depth_span / (self.depth_num * (1 + self.depth_num)) * (
                index * (index + 1)
            )
        return _DEPTH_START + depth_span / self.depth_num * index

    def _init_pseudo_reference_points(self) -> None:
        if self.num_propagated <= 0:
            self.register_parameter("pseudo_reference_points", None)
            return

        # Cell centers of the smallest cube grid with at least num_propagated cells, picked
        # evenly over the whole grid so the seeds cover the whole range on every axis. The
        # reference keeps the first num_propagated cells of the grid instead, which leaves
        # the far end of the first axis without a seed (x <= 0.79 for 256 of 343 cells);
        # trained checkpoints carry their own points, so loading one keeps its layout.
        num_divisions = round(self.num_propagated ** (1 / 3) + 1)
        linspace = torch.linspace(0.0, 1.0, steps=num_divisions + 1)
        centers = (linspace[:-1] + linspace[1:]) / 2
        grid = torch.meshgrid(centers, centers, centers, indexing="ij")
        cells = torch.stack(grid, dim=-1).reshape(-1, 3)
        picked = torch.linspace(0, len(cells) - 1, steps=self.num_propagated).round().long()
        # A frozen Parameter rather than a buffer: reference checkpoints store it under
        # this key in the state dict.
        self.pseudo_reference_points = nn.Parameter(cells[picked], requires_grad=False)

    def init_weights(self) -> None:
        """Initialize reference points, decoder weights, and classification priors."""
        nn.init.uniform_(self.reference_points.weight.data, 0.0, 1.0)
        for parameter in self.decoder.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        # Focal-loss prior: start every class at ~1% foreground probability so
        # the first iterations are not dominated by massive negative losses.
        bias_init = -math.log((1.0 - 0.01) / 0.01)
        # cls_branches holds one shared module, so one init covers every layer.
        nn.init.constant_(self.cls_branches[0][-1].bias, bias_init)

    def reset_memory(self) -> None:
        """Reset the temporal memory bank and forget the previous frame of every lane."""
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None
        self.stream_continuity.reset()

    def reset_state(
        self, prev_exists: torch.Tensor, reference_points: torch.Tensor, egopose: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Seed the bank of every lane that starts a stream.

        The first ``num_propagated`` entries of a fresh bank hold the pseudo reference points
        (in the lidar frame of the frame) and the identity pose; the others stay at zero.

        Args:
            prev_exists: 1.0 for every lane whose memory carries over, 0.0 where it starts.
            reference_points: Bank reference points ``(batch, memory_len, 3)``, zero where the
                lane starts.
            egopose: Bank poses ``(batch, memory_len, 4, 4)``, zero where the lane starts.

        Returns:
            The seeded reference points and poses.
        """
        if self.pseudo_reference_points is None:
            return reference_points, egopose
        batch_size = prev_exists.size(0)
        starts = (1 - prev_exists).view(batch_size, 1, 1)
        pseudo_reference_points = (
            self.pseudo_reference_points * (self.pc_range[3:6] - self.pc_range[:3])
            + self.pc_range[:3]
        )
        identity = torch.eye(4, device=prev_exists.device, dtype=torch.float32)
        reference_points = reference_points.clone()
        egopose = egopose.clone()
        reference_points[:, : self.num_propagated] += starts * pseudo_reference_points
        egopose[:, : self.num_propagated] += starts.unsqueeze(-1) * identity
        return reference_points, egopose

    def pre_update_memory(
        self, timestamp: torch.Tensor, prev_exists: torch.Tensor, ego_pose_inv: torch.Tensor
    ) -> None:
        """Update or initialize the temporal memory before the current frame.

        The bank moves from the global frame into the lidar frame of the current frame and
        its entries age by the frame's time; lanes without a predecessor are cleared. The
        bank stores global-frame reference points and ego poses whose translations reach
        kilometer scale, far beyond half-precision resolution, so the whole geometry path runs
        in float32 regardless of autocast (mirroring :meth:`position_embedding`), and the pose
        product goes through :func:`compose_poses`, which the float32 matmul precision setting
        cannot reduce to TF32.

        Args:
            timestamp: Sample time of every lane in seconds, float64.
            prev_exists: 1.0 for every lane whose memory carries over, 0.0 where it is reset.
            ego_pose_inv: Global-to-lidar pose of every lane ``(batch, 4, 4)``.
        """
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
                self.memory_timestamp += timestamp.view(batch_size, 1, 1)
                self.memory_egopose = compose_poses(ego_pose_inv.unsqueeze(1), self.memory_egopose)
                self.memory_reference_point = transform_reference_points(
                    self.memory_reference_point, ego_pose_inv
                )
                self.memory_timestamp = memory_refresh(self.memory_timestamp, prev_exists)
                self.memory_reference_point = memory_refresh(
                    self.memory_reference_point, prev_exists
                )
                self.memory_embedding = memory_refresh(self.memory_embedding, prev_exists)
                self.memory_egopose = memory_refresh(self.memory_egopose, prev_exists)
                self.memory_velo = memory_refresh(self.memory_velo, prev_exists)

            self.memory_reference_point, self.memory_egopose = self.reset_state(
                prev_exists, self.memory_reference_point, self.memory_egopose
            )

    def push_memory(
        self,
        ego_pose: torch.Tensor,
        all_cls_scores: torch.Tensor,
        all_bbox_preds: torch.Tensor,
        decoder_outputs: torch.Tensor,
        pad_size: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lead the bank with the strongest current-frame proposals, in the global frame.

        Returns the five bank tensors (embedding, reference points, timestamps, ego poses,
        velocities) with the ``topk_proposals`` new entries in front of the current bank,
        untruncated: the deployed graph emits all of them and the runtime keeps the first
        ``memory_len``, exactly like :meth:`post_update_memory`. The new entries carry age
        zero and the identity pose; the ego-pose alignment brings every entry back into the
        global frame. The bank is stored in float32 and the alignment runs in a float32
        island: kilometer-scale translations lose meter-level motion in half precision, and
        in TF32 too, which the elementwise :func:`compose_poses` keeps out of the pose product.

        Args:
            ego_pose: Lidar-to-global pose of every lane ``(batch, 4, 4)``.
            all_cls_scores: Class logits of every decoder layer.
            all_bbox_preds: Boxes of every decoder layer.
            decoder_outputs: Post-normed queries of every decoder layer.
            pad_size: Number of denoising slots leading every sample; they never enter the
                bank.
        """
        with torch.autocast(device_type=all_cls_scores.device.type, enabled=False):
            rec_reference_points = all_bbox_preds[-1, :, pad_size:, :3].float()
            rec_velo = all_bbox_preds[-1, :, pad_size:, -2:].float()
            rec_memory = decoder_outputs[-1, :, pad_size:].float()
            rec_score = (
                all_cls_scores[-1, :, pad_size:].float().sigmoid().amax(dim=-1, keepdim=True)
            )
            batch_size = rec_score.shape[0]

            _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
            rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
            rec_memory = topk_gather(rec_memory, topk_indexes).detach()
            rec_velo = topk_gather(rec_velo, topk_indexes).detach()
            rec_timestamp = torch.zeros(
                batch_size,
                self.topk_proposals,
                1,
                device=ego_pose.device,
                dtype=self.memory_timestamp.dtype,
            )
            rec_ego_pose = identity_poses(batch_size, self.topk_proposals, ego_pose.device)

            embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
            timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
            egopose = torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
            reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
            velo = torch.cat([rec_velo, self.memory_velo], dim=1)
            reference_point = transform_reference_points(reference_point, ego_pose)
            egopose = compose_poses(ego_pose.unsqueeze(1), egopose)
        return embedding, reference_point, timestamp, egopose, velo

    def post_update_memory(
        self,
        timestamp: torch.Tensor,
        ego_pose: torch.Tensor,
        all_cls_scores: torch.Tensor,
        all_bbox_preds: torch.Tensor,
        decoder_outputs: torch.Tensor,
        pad_size: int = 0,
    ) -> None:
        """Append the strongest current-frame proposals to the temporal memory.

        Keeps the first ``memory_len`` entries of :meth:`push_memory` and dates them
        relative to the frame, so :meth:`pre_update_memory` can age them with the next
        frame's time.
        """
        embedding, reference_point, age, egopose, velo = self.push_memory(
            ego_pose, all_cls_scores, all_bbox_preds, decoder_outputs, pad_size
        )
        self.memory_embedding = embedding[:, : self.memory_len]
        self.memory_reference_point = reference_point[:, : self.memory_len]
        self.memory_timestamp = age[:, : self.memory_len] - timestamp.view(-1, 1, 1)
        self.memory_egopose = egopose[:, : self.memory_len]
        self.memory_velo = velo[:, : self.memory_len]

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
        image_height: int | torch.Tensor,
        image_width: int | torch.Tensor,
        img2lidar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build geometry-aware positional embeddings for image tokens.

        The frustum projection multiplies pixel coordinates by metric depths,
        which exceeds float16 range; the whole geometry path therefore runs in
        float32 regardless of autocast. Only ``img_features``' shape is read.
        ``image_height`` / ``image_width`` are 0-d tensors on the export path so
        they stay graph inputs. ``camera_intrinsics`` only lends its focal
        lengths, so either the 3x3 matrix or its 4x4 homogeneous form works.
        ``img2lidar`` is the inverse of the augmented lidar-to-image projection
        of every camera, which the runtime binds as such.
        """
        with torch.autocast(device_type=img_features.device.type, enabled=False):
            return self._position_embedding(
                img_features, camera_intrinsics, image_height, image_width, img2lidar
            )

    def _position_embedding(
        self,
        img_features: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        image_height: int | torch.Tensor,
        image_width: int | torch.Tensor,
        img2lidar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_cams, _, feature_height, feature_width = img_features.shape
        device = img_features.device
        memory_centers = self._build_memory_centers(
            batch_size, num_cams, feature_height, feature_width, device
        )
        batch_tokens = batch_size * num_cams
        token_count = num_cams * feature_height * feature_width
        depth_count = self.depth_num

        intrinsics = torch.stack(
            [camera_intrinsics[..., 0, 0], camera_intrinsics[..., 1, 1]], dim=-1
        )
        intrinsics = torch.abs(intrinsics) / 1e3
        # One focal length per token, camera-major like the tokens. The reference tiles
        # camera-minor (``repeat`` over the token axis), so (N - 1) / N of its tokens carry
        # another camera's focal length; checkpoints trained with it see a different cone
        # here, which matters where the cameras differ in field of view.
        intrinsics = intrinsics.repeat_interleave(feature_height * feature_width, dim=1)

        memory_centers[..., 0] *= image_width
        memory_centers[..., 1] *= image_height
        memory_centers = memory_centers.view(batch_size, token_count, 1, 2)
        centers = memory_centers.repeat(1, 1, depth_count, 1)
        coords_d = self.coords_d.view(1, 1, depth_count, 1).repeat(batch_size, token_count, 1, 1)
        coords = torch.cat([centers, coords_d], dim=-1)
        coords = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1)
        coords[..., :2] *= torch.maximum(coords[..., 2:3], torch.full_like(coords[..., 2:3], 1e-5))
        coords = coords.unsqueeze(-1)

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
            identity_ego_pose = identity_poses(query_pos.size(0), query_pos.size(1), query.device)
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
        gt_boxes: list[torch.Tensor] | None,
        gt_labels: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, DenoisingQueries | None]:
        """Lead the learnable reference points with denoising queries in training.

        Every ground-truth box of the batch is replicated ``scalar`` times, its center
        noised, and the result occupies the first ``pad_size`` slots of its sample's query
        set; the attention mask keeps the denoising groups and the detection queries from
        seeing each other. Outside training, without denoising or without any box the
        reference points are returned unchanged.

        Returns:
            The per-sample reference points, the self-attention mask (``None`` without
            denoising queries) and the denoising bookkeeping (``None`` likewise).
        """
        reference_points = self.reference_points.weight
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

        known_labels = labels.repeat(self.scalar)
        known_batch_idx = batch_idx.repeat(self.scalar)
        known_boxes = boxes.repeat(self.scalar, 1)
        known_bbox_center = known_boxes[:, :3].clone()

        if self.bbox_noise_scale > 0:
            diff = known_boxes[:, 3:6] / 2
            rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
            known_bbox_center += rand_prob * diff * self.bbox_noise_scale
            background_mask = torch.norm(rand_prob, dim=1) > self.split
            known_labels[background_mask] = self.num_classes
        known_bbox_center = (known_bbox_center - self.pc_range[:3]) / (
            self.pc_range[3:6] - self.pc_range[:3]
        )
        known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)

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
        )
        padded_reference_points[(known_batch_idx, map_known_indice)] = known_bbox_center

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

        denoising = DenoisingQueries(
            known_labels=known_labels,
            known_boxes=known_boxes,
            batch_indices=known_batch_idx,
            slot_indices=map_known_indice,
            pad_size=pad_size,
        )
        return padded_reference_points, attn_mask, denoising

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

    def _get_targets(
        self,
        cls_logits: torch.Tensor,
        box_params: torch.Tensor,
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
    ) -> list[StreamPETRTargets]:
        """Match the queries of one decoder layer to the ground truth of every sample.

        The assigner works on the padded ``[batch, max_num_gts]`` layout, so the per-sample
        lists are padded for one call per layer and the matches are read back per sample.
        """
        batch_size, num_queries = cls_logits.shape[:2]
        max_num_gts = max(boxes.shape[0] for boxes in gt_boxes)
        padded_boxes = box_params.new_zeros((batch_size, max_num_gts, 9))
        padded_labels = torch.full(
            (batch_size, max_num_gts), -1, dtype=torch.long, device=cls_logits.device
        )
        valid_masks = torch.zeros(
            (batch_size, max_num_gts), dtype=torch.bool, device=cls_logits.device
        )
        for index, (sample_gt_boxes, sample_gt_labels) in enumerate(zip(gt_boxes, gt_labels)):
            count = sample_gt_boxes.shape[0]
            padded_boxes[index, :count] = sample_gt_boxes
            padded_labels[index, :count] = sample_gt_labels
            valid_masks[index, :count] = True

        assigned = self.assigner.assign(
            bboxes=denormalize_boxes3d(box_params),
            gt_bboxes=padded_boxes,
            gt_labels=padded_labels,
            cls_pred=cls_logits,
            valid_masks=valid_masks,
        )

        targets: list[StreamPETRTargets] = []
        for index in range(batch_size):
            labels = padded_labels.new_full((num_queries,), -1)
            bbox_targets = box_params.new_zeros((num_queries, 9))
            pos_inds = torch.nonzero(assigned.gt_inds[index] > 0, as_tuple=False).squeeze(-1)
            if pos_inds.numel() > 0:
                matched_gt_inds = assigned.gt_inds[index, pos_inds] - 1
                labels[pos_inds] = padded_labels[index, matched_gt_inds]
                bbox_targets[pos_inds] = padded_boxes[index, matched_gt_inds]
            targets.append(StreamPETRTargets(labels=labels, bbox_targets=bbox_targets))
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
        target_encodings = []
        positive_weights = []
        for sample_targets in targets:
            encoding = bbox_preds.new_zeros((sample_targets.labels.shape[0], 10))
            pos_mask = sample_targets.labels >= 0
            encoding[pos_mask] = normalize_boxes3d(sample_targets.bbox_targets[pos_mask])
            target_encodings.append(encoding)
            positive_weights.append(pos_mask.to(encoding.dtype))
        target_tensor = torch.stack(target_encodings, dim=0)
        weight_tensor = torch.stack(positive_weights, dim=0).unsqueeze(-1)
        per_box = self.loss_bbox(bbox_preds, target_tensor) * self.code_weights
        bbox_loss = self.loss_bbox_weight * (per_box * weight_tensor).sum() / max(total_pos, 1)
        return loss_cls, bbox_loss

    @staticmethod
    def _denoising_predictions(
        denoising: StreamPETRDenoisingOutputs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick the logits and the boxes of every noised query, per decoder layer.

        Returns ``(num_layers, num_known, num_classes)`` logits and ``(num_layers,
        num_known, 10)`` boxes in the order of ``known_labels``.
        """
        selection = (denoising.batch_indices, denoising.slot_indices)
        cls_scores = denoising.cls_scores.permute(1, 2, 0, 3)[selection].permute(1, 0, 2)
        bbox_preds = denoising.bbox_preds.permute(1, 2, 0, 3)[selection].permute(1, 0, 2)
        return cls_scores, bbox_preds

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
            * (self.loss_bbox(bbox_preds, target_encoding) * self.code_weights).sum()
            / max(num_total_pos, 1)
        )
        return loss_cls, loss_bbox

    def embed_tokens(
        self, img_features: torch.Tensor, pos_embed: torch.Tensor, cone: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn the image features into the decoder's memory tokens and their embeddings.

        Args:
            img_features: Image features ``(batch, num_cams, channels, height, width)``.
            pos_embed: Geometric position embedding of every token, see
                :meth:`position_embedding`.
            cone: Per-token camera cone of :meth:`position_embedding`.

        Returns:
            The memory tokens and their feature-modulated position embedding.
        """
        batch_size, num_cams, channels, feature_height, feature_width = img_features.shape
        memory = img_features.permute(0, 1, 3, 4, 2).reshape(
            batch_size, num_cams * feature_height * feature_width, channels
        )
        memory = self.memory_embed(memory)
        memory = self.spatial_alignment(memory, cone)
        return memory, self.featurized_pe(pos_embed, memory)

    def decode(
        self,
        memory: torch.Tensor,
        pos_embed: torch.Tensor,
        reference_points: torch.Tensor,
        query_attn_mask: torch.Tensor | None,
        sanitize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the decoder over the queries of the reference points and the temporal memory.

        Args:
            memory: Memory tokens of :meth:`embed_tokens`.
            pos_embed: Their position embedding.
            reference_points: Normalized reference points of every query ``(batch, Q, 3)``,
                the denoising queries first.
            query_attn_mask: Self-attention mask of the denoising groups, ``None`` without.
            sanitize: Replace non-finite post-normed queries like the reference does. The
                deployed graph leaves it out: the runtime never feeds the bank non-finite
                values and the guard costs twenty graph nodes per layer.

        Returns:
            The class logits, the boxes and the post-normed queries of every decoder layer,
            each ``(num_layers, batch, Q + num_propagated, ...)``.
        """
        query_pos = self.query_embedding(pos2posemb3d(reference_points, self.hidden_dim // 2))
        query = torch.zeros_like(query_pos)
        query, query_pos, reference_points, temp_memory, temp_pos = self.temporal_alignment(
            query, query_pos, reference_points
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
            # Prediction and the memory consume the post-normed intermediates; the raw
            # query flows to the next layer.
            normed_query = self.post_norm(query)
            if sanitize:
                normed_query = torch.nan_to_num(normed_query)
            # The prediction branches run in float32 like the reference.
            with torch.autocast(device_type=query.device.type, enabled=False):
                query_fp32 = normed_query.float()
                cls_logits = cls_branch(query_fp32)
                box_params = self._decode_box_params(reg_branch(query_fp32), reference_points)
            outputs_classes.append(cls_logits)
            outputs_coords.append(box_params)
            decoder_outputs.append(normed_query)

        return (
            torch.stack(outputs_classes, dim=0),
            torch.stack(outputs_coords, dim=0),
            torch.stack(decoder_outputs, dim=0),
        )

    def forward(
        self,
        img_features: torch.Tensor,
        image_height: int,
        image_width: int,
        camera_intrinsics: torch.Tensor,
        lidar2img: torch.Tensor,
        timestamp: torch.Tensor,
        prev_exists: torch.Tensor,
        ego_pose: torch.Tensor,
        ego_pose_inv: torch.Tensor,
        gt_boxes: list[torch.Tensor] | None = None,
        gt_labels: list[torch.Tensor] | None = None,
    ) -> StreamPETRHeadOutputs:
        """Predict StreamPETR query outputs from multiview image features.

        Every frame carries its stream metadata: the sample time (float64 seconds),
        ``prev_exists`` of :meth:`stream_continuity`, and the lidar-to-global pose with its
        inverse. ``gt_boxes`` are metric ``[cx, cy, cz, l, w, h, yaw, vx, vy]`` boxes that
        seed the denoising queries in training.
        """
        self.pre_update_memory(timestamp, prev_exists, ego_pose_inv)

        pos_embed, cone = self.position_embedding(
            img_features,
            camera_intrinsics,
            image_height,
            image_width,
            img2lidar=torch.inverse(lidar2img.float()),
        )
        memory, pos_embed = self.embed_tokens(img_features, pos_embed, cone)
        reference_points, query_attn_mask, denoising_queries = self.prepare_for_dn(
            img_features.shape[0], gt_boxes, gt_labels
        )
        all_cls_scores, all_bbox_preds, decoder_outputs = self.decode(
            memory, pos_embed, reference_points, query_attn_mask
        )

        pad_size = denoising_queries.pad_size if denoising_queries is not None else 0
        self.post_update_memory(
            timestamp, ego_pose, all_cls_scores, all_bbox_preds, decoder_outputs, pad_size
        )

        denoising = None
        if denoising_queries is not None:
            denoising = StreamPETRDenoisingOutputs(
                cls_scores=all_cls_scores[:, :, :pad_size, :],
                bbox_preds=all_bbox_preds[:, :, :pad_size, :],
                known_labels=denoising_queries.known_labels,
                known_boxes=denoising_queries.known_boxes,
                batch_indices=denoising_queries.batch_indices,
                slot_indices=denoising_queries.slot_indices,
            )
            all_cls_scores = all_cls_scores[:, :, pad_size:, :]
            all_bbox_preds = all_bbox_preds[:, :, pad_size:, :]

        return StreamPETRHeadOutputs(
            all_cls_scores=all_cls_scores, all_bbox_preds=all_bbox_preds, denoising=denoising
        )

    def loss(
        self,
        outputs: StreamPETRHeadOutputs,
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute multi-layer StreamPETR losses.

        ``gt_boxes`` are metric ``[cx, cy, cz, l, w, h, yaw, vx, vy]`` boxes and
        ``gt_labels`` their int64 class indices, one tensor per sample.
        """
        all_cls_scores = outputs.all_cls_scores
        all_bbox_preds = outputs.all_bbox_preds

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

        if outputs.denoising is None and self.with_dn and self.training:
            # Keep the loss dictionary uniform across ranks when a batch has
            # no ground truth: every logged key is a distributed collective.
            zero = all_cls_scores.new_tensor(0.0)
            loss_dict["dn_loss_cls"] = zero
            loss_dict["dn_loss_bbox"] = zero.clone()
            for layer_index in range(self.num_decoder_layers - 1):
                loss_dict[f"d{layer_index}.dn_loss_cls"] = zero.clone()
                loss_dict[f"d{layer_index}.dn_loss_bbox"] = zero.clone()

        if outputs.denoising is not None:
            known_labels = outputs.denoising.known_labels
            known_bboxs = outputs.denoising.known_boxes
            output_known_class, output_known_coord = self._denoising_predictions(outputs.denoising)
            dn_losses_cls = []
            dn_losses_bbox = []
            for cls_scores, bbox_preds in zip(output_known_class, output_known_coord):
                dn_loss_cls, dn_loss_bbox = self.dn_loss_single(
                    cls_scores, bbox_preds, known_bboxs, known_labels, known_labels.numel()
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

    def predict(self, outputs: StreamPETRHeadOutputs) -> list[Detection3DSamplePredictions]:
        """Decode final detections from the last decoder layer.

        Returns one typed prediction per sample, consumed by
        ``detection3d_eval_output`` and the metric suite. Box centers are
        gravity centers end to end: the annotations store them, the head
        regresses them, and the exported graph emits them. Nothing here may
        re-center the boxes, or the deployed output would silently diverge
        from what the metrics evaluate.
        """
        predictions = self.bbox_coder.decode(
            outputs.all_cls_scores[-1].float(), outputs.all_bbox_preds[-1].float()
        )
        return [
            Detection3DSamplePredictions(
                bboxes_3d=prediction["bboxes"],
                scores_3d=prediction["scores"],
                labels_3d=prediction["labels"].long(),
            )
            for prediction in predictions
        ]
