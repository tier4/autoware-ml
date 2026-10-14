"""Tests for FRNet-specific training behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTBatch
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.dataclasses.geometry.range_view import RangeViewData
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.segmentation3d.frnet import FRNet
from autoware_ml.models.tests.batch_inputs_fixtures import (
    build_batch_inputs,
    build_point_cloud_batch,
)
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.preprocessing.segmentation3d.frustum_range import FrustumRangePreprocessor


class _IdentityEncoder(torch.nn.Module):
    """Return passthrough encoder tensors for FRNet tests."""

    def forward(
        self,
        points: torch.Tensor,
        inverse_map: torch.Tensor,
        voxel_coors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Echo inputs as a one-level pyramid."""
        del inverse_map
        return voxel_coors, points, [points]


class _IdentityBackbone(torch.nn.Module):
    """Forward the encoder pyramid through unchanged for FRNet tests."""

    def __init__(self) -> None:
        super().__init__()
        self.last_sample_count: int | None = None

    def forward(
        self,
        point_feats_pyramid: list[torch.Tensor],
        voxel_feats: torch.Tensor,
        voxel_coors: torch.Tensor,
        point_coors: torch.Tensor,
        inverse_map: torch.Tensor,
        sample_count: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Record the sample count and echo the encoder pyramid as backbone output."""
        del voxel_coors, point_coors, inverse_map
        self.last_sample_count = sample_count
        return [voxel_feats], list(point_feats_pyramid)


class _DecodeHead(torch.nn.Module):
    """Provide the classifier interface expected by :class:`FRNet`."""

    def __init__(self, num_classes: int) -> None:
        """Initialize the decode-head test double.

        Args:
            num_classes: Number of semantic classes.
        """
        super().__init__()
        self.classifier = torch.nn.Linear(4, num_classes)
        self.ignore_index = num_classes - 1

    def forward(
        self,
        point_coors: torch.Tensor,
        point_feats_encoder: list[torch.Tensor],
        voxel_feats_backbone: list[torch.Tensor],
        point_feats_backbone: list[torch.Tensor],
    ) -> torch.Tensor:
        """Return a deterministic point logits tensor for tests."""
        del point_coors, point_feats_encoder, voxel_feats_backbone
        return self.classifier(point_feats_backbone[0])

    def loss(self, point_logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute a simple cross-entropy test loss."""
        return {"loss_ce": torch.nn.functional.cross_entropy(point_logits, target)}

    def predict(self, point_logits: torch.Tensor) -> torch.Tensor:
        """Return point-wise argmax predictions."""
        return point_logits.argmax(dim=1)


def _make_frnet(num_classes: int = 3, data_preprocessor: DataPreprocessor | None = None) -> FRNet:
    """Return a minimal FRNet using identity encoder/backbone."""
    return FRNet(
        voxel_encoder=_IdentityEncoder(),
        backbone=_IdentityBackbone(),
        decode_head=_DecodeHead(num_classes=num_classes),
        optimizer=torch.optim.AdamW,
        data_preprocessor=data_preprocessor,
    )


def _make_batch(num_points: int = 5, num_classes: int = 3) -> ModelBatchInputs:
    """Return minimal preprocessed model inputs compatible with the test model."""
    points = torch.rand(num_points, 4)
    coors = torch.stack(
        [
            torch.zeros(num_points, dtype=torch.long),
            torch.arange(num_points, dtype=torch.long) % 2,
            torch.arange(num_points, dtype=torch.long) % 2,
        ],
        dim=1,
    )
    voxel_coors, inverse_map = torch.unique(coors, return_inverse=True, dim=0)
    point_cloud = build_point_cloud_batch([points])
    labels = Segmentation3DGTBatch(
        gt_semantic_masks=torch.randint(0, num_classes - 1, (num_points,)),
        batch_indices=point_cloud.batch_indices,
    )
    return build_batch_inputs(point_cloud=point_cloud, segmentation=labels).replace(
        range_view_data=RangeViewData(
            coors=coors,
            voxel_coors=voxel_coors,
            inverse_map=inverse_map,
            semantic_labels=torch.zeros(1, 2, 2, dtype=torch.long),
        )
    )


def test_frnet_shared_step_returns_scalar_loss_with_grad() -> None:
    """_core_step should return a differentiable scalar loss tensor."""
    model = _make_frnet()
    model.log_dict = MagicMock()
    batch = _make_batch()

    metrics, _ = model._core_step(batch, "train")

    assert "loss" in metrics
    assert metrics["loss"].shape == torch.Size([])
    assert metrics["loss"].requires_grad
    assert model.backbone.last_sample_count == 1


def test_frnet_get_log_batch_size_uses_sample_count() -> None:
    model = _make_frnet(num_classes=4)
    batch = _make_batch(num_points=8, num_classes=4)

    assert model.get_log_batch_size(batch) == 1


def test_frnet_forward_uses_explicit_sample_count() -> None:
    model = _make_frnet(num_classes=4)
    points = torch.rand(4, 4)
    coors = torch.tensor(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=torch.long,
    )
    voxel_coors, inverse_map = torch.unique(coors, return_inverse=True, dim=0)

    outputs = model.forward_network(points, coors, voxel_coors, inverse_map, sample_count=2)

    point_logits = outputs.segmentation3d().logits
    assert point_logits.shape == (4, 4)
    # _IdentityBackbone returns a single-level pyramid
    assert len(outputs.segmentation3d().auxiliary_features) == 1
    assert model.backbone.last_sample_count == 2


def test_frnet_with_preprocessing_runs_shared_step_end_to_end() -> None:
    """A real preprocessor handed to the model should feed FRNet."""
    preprocessor = FrustumRangePreprocessor(
        height=2,
        width=4,
        fov_up=10.0,
        fov_down=-10.0,
        ignore_index=2,
        num_classes=3,
    )
    model = _make_frnet(
        num_classes=3,
        data_preprocessor=DataPreprocessor(preprocessor_modules=[preprocessor]),
    )
    model.log_dict = MagicMock()

    raw_batch = ModelGTBatch(
        point_cloud_gt_batch=PointCloudGTBatch(
            points=torch.tensor(
                [[1.0, 0.0, 0.0, 0.1], [2.0, 0.0, 0.0, 0.2], [1.0, 1.0, 0.0, 0.3]],
                dtype=torch.float32,
            ),
            batch_indices=torch.zeros(3, dtype=torch.int32),
            batch_size=1,
            timestamp_difference_dim=-1,
        ),
        detection3d_gt_batch=None,
        segmentation3d_gt_batch=Segmentation3DGTBatch(
            gt_semantic_masks=torch.tensor([0, 1, 0], dtype=torch.int64),
            batch_indices=torch.zeros(3, dtype=torch.int32),
        ),
        image_gt_batch=None,
    )

    preprocessed = model.on_after_batch_transfer(raw_batch, dataloader_idx=0)
    metrics, _ = model._core_step(preprocessed, "train")

    assert "loss" in metrics
    assert metrics["loss"].requires_grad
    assert model.backbone.last_sample_count == 1
