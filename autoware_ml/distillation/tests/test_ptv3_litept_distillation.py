"""Distillation of a PTv3 teacher into a LitePT student on the real models, on CUDA."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from autoware_ml.distillation.distillation import Distillation
from autoware_ml.distillation.shared_queries import TransFusionSharedQueriesTerm
from autoware_ml.distillation.term import DistillationTerm
from autoware_ml.distillation.tests.toy_models import InMemoryTeacher
from autoware_ml.losses.distillation.feature import FeatureMSELoss
from autoware_ml.losses.distillation.sigmoid_soft_target import SigmoidSoftTargetLoss
from autoware_ml.losses.distillation.soft_target import SoftTargetKLDivLoss
from autoware_ml.models.detection3d.ptv3 import (
    PTv3BEVEncoder,
    PTv3BEVProjection,
    PTv3Det3DBEVNeck,
    PTv3Det3DFeatureFusion,
)
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    POINT_CLOUD_RANGE,
    build_batch,
    build_bev_neck,
    build_litept_encoder,
    build_litept_seg3d_head,
    build_preprocessor,
    build_ptv3_encoder,
    build_seg3d_head,
    build_transfusion_head,
)
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.multi.ptv3_det3d_seg3d import PTv3Det3DSeg3DModel
from autoware_ml.models.segmentation3d.encoders.voxel import SweepSplitVoxelFeatureEncoder
from autoware_ml.models.segmentation3d.ptv3 import PTv3Segmentation3DModel
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor

REQUIRES_SPARSE_CUDA = pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 distillation tests require CUDA spconv",
)
LOGITS = "segmentation3d_head_outputs.logits"
HEATMAP = "detection3d_head_outputs.transfusion_head_outputs.dense_heatmap"


def _voxelizer() -> DataPreprocessor:
    return DataPreprocessor(preprocessor_modules=[build_preprocessor()])


def _common_kwargs() -> dict:
    return {
        "log_dict_configs": LogDictConfigs(prog_bar=False),
        "data_preprocessor": _voxelizer(),
        "voxel_encoder": SweepSplitVoxelFeatureEncoder(),
        "grid_size": 1.0,
        "point_cloud_range": list(POINT_CLOUD_RANGE),
        "optimizer": lambda params: torch.optim.AdamW(params, lr=1e-3),
    }


def _litept_bev_neck() -> PTv3Det3DBEVNeck:
    """The neck of the LitePT student, reading its 24-channel coarse stages."""
    return PTv3Det3DBEVNeck(
        fusion=PTv3Det3DFeatureFusion(in_channels=24, skip_channels=24, out_channels=16),
        bev_projector=PTv3BEVProjection(in_channels=16, out_channels=16, output_shape=[8, 8]),
        bev_encoder=PTv3BEVEncoder(
            in_channels=16, hidden_channels=32, out_channels=64, dilations=(1, 2, 1)
        ),
    )


def _run_one_training_step(student, device: torch.device) -> dict[str, torch.Tensor]:
    logged: dict[str, torch.Tensor] = {}
    student.log_dict = lambda values, **kwargs: logged.update(values)
    student.to(device)
    student.setup("fit")
    student.train()
    inputs = student.on_after_batch_transfer(build_batch().to_device(device), dataloader_idx=0)
    loss = student.training_step(inputs, batch_idx=0)
    loss.backward()
    return logged


@REQUIRES_SPARSE_CUDA
def test_ptv3_teaches_litept_segmentation_on_the_shared_voxels() -> None:
    """
    Input: a tiny PTv3 segmentation teacher and a LitePT student with a wider stem, both reading
    the voxels of the student; a logits term and an embedding-stem feature term with an adapter.
    Expected: the teacher's bf16 forward supervises the student; the loss is finite and reaches
    the student encoder and the adapter, the teacher keeps no gradients.
    Check: the logged terms and the gradients.
    """
    device = torch.device("cuda")
    teacher = PTv3Segmentation3DModel(
        encoder=build_ptv3_encoder(), seg3d_head=build_seg3d_head(), **_common_kwargs()
    )
    adapter = nn.Linear(12, 8)
    student = PTv3Segmentation3DModel(
        encoder=build_litept_encoder(),
        seg3d_head=build_litept_seg3d_head(),
        distillation=Distillation(
            teacher=InMemoryTeacher(teacher, autocast_dtype="bfloat16"),
            terms=[
                DistillationTerm(
                    name="logits", student=LOGITS, teacher=LOGITS, loss=SoftTargetKLDivLoss(2.0)
                ),
                DistillationTerm(
                    name="stem",
                    student="module:encoder.embedding",
                    teacher="module:encoder.embedding",
                    loss=FeatureMSELoss(normalize=True),
                    weight=0.1,
                    adapter=adapter,
                ),
            ],
        ),
        **_common_kwargs(),
    )

    logged = _run_one_training_step(student, device)

    assert torch.isfinite(logged["train/loss"])
    assert logged["train/distill/logits"] > 0.0
    assert logged["train/distill/stem"] > 0.0
    assert torch.allclose(
        logged["train/loss"], logged["train/gt_loss"] + logged["train/distill_loss"]
    )
    assert any(p.grad is not None for p in student.encoder.embedding.parameters())
    assert adapter.weight.grad is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert not teacher.training


@REQUIRES_SPARSE_CUDA
def test_ptv3_teaches_litept_detection_and_segmentation() -> None:
    """
    Input: a tiny joint PTv3 teacher and a joint LitePT student with its own neck; terms on the
    segmentation logits, the dense heatmap, the BEV neck output and the shared queries.
    Expected: every term is finite and the loss trains both branches of the student.
    Check: the logged terms and the gradients of the heads.
    """
    device = torch.device("cuda")
    export_names = ["pred_labels", "pred_probs", "dense_heatmap", "heatmap", "center"]
    teacher = PTv3Det3DSeg3DModel(
        encoder=build_ptv3_encoder(),
        seg3d_head=build_seg3d_head(),
        bev_neck=build_bev_neck(),
        bbox_head=build_transfusion_head(),
        export_output_names=export_names,
        **_common_kwargs(),
    )
    student = PTv3Det3DSeg3DModel(
        encoder=build_litept_encoder(),
        seg3d_head=build_litept_seg3d_head(),
        bev_neck=_litept_bev_neck(),
        bbox_head=build_transfusion_head(),
        export_output_names=export_names,
        distillation=Distillation(
            teacher=InMemoryTeacher(teacher, autocast_dtype="bfloat16"),
            terms=[
                DistillationTerm(
                    name="seg3d_logits",
                    student=LOGITS,
                    teacher=LOGITS,
                    loss=SoftTargetKLDivLoss(2.0),
                ),
                DistillationTerm(
                    name="det3d_heatmap",
                    student=HEATMAP,
                    teacher=HEATMAP,
                    loss=SigmoidSoftTargetLoss(focal_gamma=2.0),
                ),
                DistillationTerm(
                    name="bev_features",
                    student="module:bev_neck",
                    teacher="module:bev_neck",
                    loss=FeatureMSELoss(normalize=True),
                    weight=0.1,
                ),
                TransFusionSharedQueriesTerm(),
            ],
        ),
        **_common_kwargs(),
    )

    logged = _run_one_training_step(student, device)

    for key in (
        "train/distill/seg3d_logits",
        "train/distill/det3d_heatmap",
        "train/distill/bev_features",
        "train/distill/det3d_queries/cls",
        "train/distill/det3d_queries/reg",
    ):
        assert torch.isfinite(logged[key]) and logged[key] >= 0.0
    assert torch.isfinite(logged["train/loss"])
    assert any(p.grad is not None for p in student.bev_neck.parameters())
    assert any(p.grad is not None for p in student.bbox_head.decoder.parameters())
    assert student.seg3d_head.classifier.weight.grad is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())
