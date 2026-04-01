"""Tests for FRNet-specific training behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from autoware_ml.models.segmentation3d.frnet import FRNet


class _IdentityEncoder(torch.nn.Module):
    """Pass FRNet feature dictionaries through unchanged."""

    def forward(self, batch_inputs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the input dictionary without modification."""
        return batch_inputs_dict


class _IdentityBackbone(torch.nn.Module):
    """Pass FRNet feature dictionaries through unchanged."""

    def __init__(self) -> None:
        super().__init__()
        self.last_sample_count: int | None = None

    def forward(self, batch_inputs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the input dictionary without modification."""
        sample_count = batch_inputs_dict.get("sample_count")
        if isinstance(sample_count, int):
            self.last_sample_count = sample_count
        return batch_inputs_dict


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

    def forward(self, voxel_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return a deterministic logits tensor for tests."""
        voxel_dict["point_logits"] = self.classifier(voxel_dict["points"])
        return voxel_dict

    def loss(
        self, voxel_dict: dict[str, torch.Tensor], target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute a simple cross-entropy test loss."""
        return {"loss_ce": torch.nn.functional.cross_entropy(voxel_dict["point_logits"], target)}

    def predict(self, voxel_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return point-wise argmax predictions."""
        return voxel_dict["point_logits"].argmax(dim=1)


def _make_frnet(num_classes: int = 3) -> FRNet:
    """Return a minimal FRNet using identity encoder/backbone."""
    return FRNet(
        voxel_encoder=_IdentityEncoder(),
        backbone=_IdentityBackbone(),
        decode_head=_DecodeHead(num_classes=num_classes),
        optimizer=torch.optim.AdamW,
    )


def _make_batch(num_points: int = 5, num_classes: int = 3) -> dict:
    """Return a minimal preprocessed batch compatible with the test model."""
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
    semantic_seg = torch.zeros(1, 2, 2, dtype=torch.long)  # (B, H, W)
    pts_semantic_mask = torch.randint(0, num_classes - 1, (num_points,))
    return {
        "points": points,
        "coors": coors,
        "voxel_coors": voxel_coors,
        "inverse_map": inverse_map,
        "pts_semantic_mask": pts_semantic_mask,
        "semantic_seg": semantic_seg,
        "batch_size": 1,
    }


def test_frnet_shared_step_returns_scalar_loss_with_grad() -> None:
    """_shared_step should return a differentiable scalar loss tensor."""
    model = _make_frnet()
    model.log_dict = MagicMock()
    batch = _make_batch()

    result = model._shared_step(batch, "train")

    assert "loss" in result
    assert result["loss"].shape == torch.Size([])
    assert result["loss"].requires_grad
    assert model.backbone.last_sample_count == 1


def test_frnet_get_log_batch_size_prefers_semantic_seg_shape() -> None:
    model = _make_frnet(num_classes=4)
    batch = _make_batch(num_points=8, num_classes=4)

    assert model.get_log_batch_size(batch) == int(batch["semantic_seg"].shape[0])


def test_frnet_get_log_batch_size_falls_back_to_coordinate_batches() -> None:
    model = _make_frnet(num_classes=4)
    batch = {
        "coors": torch.tensor(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
            ],
            dtype=torch.long,
        )
    }

    assert model.get_log_batch_size(batch) == 2


def test_frnet_forward_infers_sample_count_from_coordinates() -> None:
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

    logits = model(points, coors, voxel_coors, inverse_map)

    assert logits.shape == (4, 4)
    assert model.backbone.last_sample_count == 2
