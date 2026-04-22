"""Framework prediction contracts across supported models."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from autoware_ml.models.calibration_status.calibration_status import CalibrationStatusClassifier
from autoware_ml.models.segmentation3d.frnet import FRNet
from autoware_ml.models.segmentation3d.ptv3 import PTv3SegmentationModel


class _IdentityModule(nn.Module):
    """Return the input tensor unchanged."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _CalibrationDummyHead(nn.Module):
    """Minimal calibration-status head test double."""

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": torch.nn.functional.cross_entropy(logits, target)}

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return feats


class _PTv3DummyBackbone(nn.Module):
    """Return point features in the PTv3 backbone shape contract."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.block = nn.Linear(4, 4)

    def forward(self, point_dict: dict[str, torch.Tensor]) -> SimpleNamespace:
        return SimpleNamespace(feat=point_dict["feat"])


class _FRNetIdentityEncoder(nn.Module):
    """Pass FRNet feature dictionaries through unchanged."""

    def forward(self, batch_inputs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return batch_inputs_dict


class _FRNetIdentityBackbone(nn.Module):
    """Pass FRNet feature dictionaries through unchanged."""

    def forward(self, batch_inputs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return batch_inputs_dict


class _FRNetDecodeHead(nn.Module):
    """Provide the classifier interface expected by FRNet."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(4, num_classes)
        self.ignore_index = num_classes - 1

    def forward(self, voxel_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        voxel_dict["point_logits"] = self.classifier(voxel_dict["points"])
        return voxel_dict

    def loss(
        self, voxel_dict: dict[str, torch.Tensor], target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {"loss_ce": torch.nn.functional.cross_entropy(voxel_dict["point_logits"], target)}

    def predict(self, voxel_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return voxel_dict["point_logits"].argmax(dim=1)


def _make_calibration_model() -> CalibrationStatusClassifier:
    return CalibrationStatusClassifier(
        backbone=_IdentityModule(),
        neck=_IdentityModule(),
        head=_CalibrationDummyHead(),
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
    )


def _make_frnet(num_classes: int = 4) -> FRNet:
    return FRNet(
        voxel_encoder=_FRNetIdentityEncoder(),
        backbone=_FRNetIdentityBackbone(),
        decode_head=_FRNetDecodeHead(num_classes=num_classes),
        optimizer=torch.optim.AdamW,
    )


def _make_ptv3() -> PTv3SegmentationModel:
    return PTv3SegmentationModel(
        backbone=_PTv3DummyBackbone(),
        num_classes=2,
        backbone_out_channels=4,
        ignore_index=-1,
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
    )


def test_calibration_status_predict_step_returns_probability_tensor() -> None:
    model = _make_calibration_model()
    batch = {"fused_img": torch.tensor([[2.0, 0.1], [0.2, 1.5]], dtype=torch.float32)}

    predictions = model.predict_step(batch, batch_idx=0)

    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (2, 2)
    assert torch.allclose(predictions.sum(dim=1), torch.ones(2))


def test_frnet_predict_step_returns_labels_and_probabilities() -> None:
    model = _make_frnet()
    batch = {
        "points": torch.rand(8, 4),
        "coors": torch.zeros(8, 3, dtype=torch.long),
        "voxel_coors": torch.zeros(4, 3, dtype=torch.long),
        "inverse_map": torch.arange(8, dtype=torch.long) % 4,
        "pts_semantic_mask": torch.randint(0, 3, (8,)),
        "semantic_seg": torch.zeros(1, 2, 2, dtype=torch.long),
        "batch_size": 1,
    }

    predictions = model.predict_step(batch, batch_idx=0)

    assert set(predictions) == {"pred_labels", "pred_probs"}
    assert predictions["pred_probs"].shape == (8, 4)
    assert predictions["pred_labels"].shape == (8,)


def test_ptv3_predict_step_returns_labels_and_probabilities() -> None:
    model = _make_ptv3()
    batch = {
        "coord": torch.randn(5, 3),
        "feat": torch.randn(5, 4),
        "grid_coord": torch.randint(0, 4, (5, 3), dtype=torch.int32),
        "offset": torch.tensor([5], dtype=torch.int32),
    }

    predictions = model.predict_step(batch, batch_idx=0)

    assert set(predictions) == {"pred_labels", "pred_probs"}
    assert predictions["pred_labels"].shape == (5,)
    assert predictions["pred_probs"].shape == (5, 2)
