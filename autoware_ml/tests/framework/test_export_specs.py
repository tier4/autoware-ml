"""Framework export-contract tests across custom model wrappers."""

from __future__ import annotations

import torch
import torch.nn as nn

from autoware_ml.models.calibration_status import CalibrationStatusClassifier
from autoware_ml.models.segmentation3d.frnet import FRNet
from autoware_ml.models.segmentation3d.ptv3 import PointTransformerV3Backbone, PTv3SegmentationModel
from autoware_ml.ops.spconv import IS_SPCONV_AVAILABLE, SubMConv3d


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


class _FRNetIdentityEncoder(nn.Module):
    """Pass FRNet feature dictionaries through unchanged."""

    def forward(self, batch_inputs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return batch_inputs_dict


class _FRNetIdentityBackbone(nn.Module):
    """Pass FRNet feature dictionaries through unchanged."""

    def __init__(self) -> None:
        super().__init__()
        self.last_sample_count: int | None = None

    def forward(self, batch_inputs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sample_count = batch_inputs_dict.get("sample_count")
        if isinstance(sample_count, int):
            self.last_sample_count = sample_count
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


def _make_frnet(num_classes: int = 3) -> FRNet:
    return FRNet(
        voxel_encoder=_FRNetIdentityEncoder(),
        backbone=_FRNetIdentityBackbone(),
        decode_head=_FRNetDecodeHead(num_classes=num_classes),
        optimizer=torch.optim.AdamW,
    )



def test_calibration_status_export_spec_keeps_single_probability_output() -> None:
    model = _make_calibration_model()
    batch = {"fused_img": torch.tensor([[2.0, 0.1], [0.2, 1.5]], dtype=torch.float32)}

    spec = model.build_export_spec(batch)
    outputs = spec.module(*spec.args)

    assert spec.input_param_names == ["fused_img"]
    assert spec.output_names is None
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (2, 2)
    assert torch.allclose(outputs.sum(dim=1), torch.ones(2))


def test_frnet_build_export_spec_returns_four_named_inputs() -> None:
    model = _make_frnet()
    batch = {
        "points": torch.rand(10, 4),
        "coors": torch.zeros(10, 3, dtype=torch.long),
        "voxel_coors": torch.zeros(8, 3, dtype=torch.long),
        "inverse_map": torch.zeros(10, dtype=torch.long),
    }

    spec = model.build_export_spec(batch)
    outputs = spec.module(*spec.args)
    expected_logits = model(
        batch["points"],
        batch["coors"],
        batch["voxel_coors"],
        batch["inverse_map"],
    )
    expected_probs = torch.softmax(expected_logits, dim=1)

    assert spec.input_param_names == ["points", "coors", "voxel_coors", "inverse_map"]
    assert spec.output_names == ["pred_probs"]
    assert len(spec.args) == 4
    assert isinstance(outputs, torch.Tensor)
    assert torch.allclose(outputs, expected_probs)
    assert spec.module.backbone is not model.backbone
    assert spec.module.backbone.last_sample_count == 1


def test_ptv3_build_export_spec_replaces_sparse_convs() -> None:
    if not IS_SPCONV_AVAILABLE:
        return

    model = PTv3SegmentationModel(
        backbone=PointTransformerV3Backbone(
            in_channels=4,
            order=("z",),
            stride=(2,),
            enc_depths=(1, 1),
            enc_channels=(8, 16),
            enc_num_head=(1, 2),
            enc_patch_size=(4, 4),
            dec_depths=(1,),
            dec_channels=(8,),
            dec_num_head=(1,),
            dec_patch_size=(4,),
            mlp_ratio=2.0,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.0,
            pre_norm=True,
            shuffle_orders=True,
            enable_rpe=False,
            enable_flash=False,
            upcast_attention=False,
            upcast_softmax=False,
            cls_mode=False,
            pdnorm_bn=False,
            pdnorm_ln=False,
            pdnorm_decouple=False,
            pdnorm_adaptive=False,
            pdnorm_affine=True,
            pdnorm_conditions=("nuScenes",),
        ),
        num_classes=2,
        backbone_out_channels=8,
        ignore_index=-1,
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
    )
    batch = {
        "coord": torch.randn(8, 3),
        "feat": torch.randn(8, 4),
        "grid_coord": torch.randint(0, 4, (8, 3), dtype=torch.int32),
        "offset": torch.tensor([8], dtype=torch.int32),
        "segment": torch.randint(0, 2, (8,), dtype=torch.long),
    }

    spec = model.build_export_spec(batch)
    export_subm_convs = [
        module for module in spec.module.modules() if isinstance(module, SubMConv3d)
    ]

    assert spec.input_param_names == ["grid_coord", "feat", "serialized_depth", "serialized_code"]
    assert spec.output_names == ["pred_labels", "pred_probs"]
    assert spec.supported_stages == frozenset({"onnx"})
    assert export_subm_convs
    assert model.backbone.shuffle_orders is True
    assert model.backbone.order == ["z"]
    assert spec.module.backbone.shuffle_orders is False
    assert spec.module.backbone.order == list(PTv3SegmentationModel.EXPORT_ORDER)
    assert spec.args[3].shape[0] == len(PTv3SegmentationModel.EXPORT_ORDER)
