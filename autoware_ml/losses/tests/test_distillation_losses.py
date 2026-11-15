"""Unit tests for the distillation losses."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from autoware_ml.losses.distillation.feature import FeatureMSELoss
from autoware_ml.losses.distillation.sigmoid_soft_target import SigmoidSoftTargetLoss
from autoware_ml.losses.distillation.soft_target import SoftTargetKLDivLoss


def _logits(seed: int, shape: tuple[int, ...] = (16, 4)) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * 3.0


def test_kl_is_zero_for_equal_logits_and_positive_otherwise() -> None:
    loss = SoftTargetKLDivLoss(temperature=2.0)
    logits = _logits(0)

    assert torch.allclose(loss(logits, logits), torch.zeros(()), atol=1e-7)
    assert loss(_logits(1), logits) > 0.0


def test_kl_handles_the_class_dim_of_a_dense_map() -> None:
    loss = SoftTargetKLDivLoss(class_dim=1)
    student = _logits(0, (2, 3, 4, 4))
    teacher = _logits(1, (2, 3, 4, 4))

    flat = SoftTargetKLDivLoss(class_dim=1)(
        student.permute(0, 2, 3, 1).reshape(-1, 3), teacher.permute(0, 2, 3, 1).reshape(-1, 3)
    )

    assert torch.allclose(loss(student, teacher), flat)


def test_kl_softens_with_the_temperature() -> None:
    student, teacher = _logits(0), _logits(1)
    sharp = SoftTargetKLDivLoss(temperature=1.0)(student, teacher)
    soft = SoftTargetKLDivLoss(temperature=4.0)(student, teacher)

    assert sharp > 0.0 and soft > 0.0
    assert not torch.allclose(sharp, soft)


def test_kl_accepts_half_precision_teacher_logits() -> None:
    logits = _logits(0)

    assert torch.allclose(
        SoftTargetKLDivLoss()(logits, logits.to(torch.bfloat16)), torch.zeros(()), atol=1e-2
    )


def test_kl_rejects_a_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="positive"):
        SoftTargetKLDivLoss(temperature=0.0)


def test_sigmoid_soft_target_is_lowest_for_matching_logits() -> None:
    loss = SigmoidSoftTargetLoss()
    teacher = _logits(0, (2, 3, 8, 8))

    assert loss(teacher, teacher) < loss(_logits(1, (2, 3, 8, 8)), teacher)


def test_sigmoid_soft_target_normalizes_by_the_teacher_mass() -> None:
    teacher = torch.full((1, 1, 4, 4), -20.0)
    teacher[0, 0, 0, 0] = 20.0
    student = torch.zeros_like(teacher)
    loss = SigmoidSoftTargetLoss(normalize="teacher_mass")(student, teacher)
    expected = (
        nn.functional.binary_cross_entropy_with_logits(
            student, torch.sigmoid(teacher), reduction="sum"
        )
        / torch.sigmoid(teacher).sum()
    )

    assert torch.allclose(loss, expected)
    assert torch.allclose(
        SigmoidSoftTargetLoss(normalize="mean")(student, teacher),
        expected * torch.sigmoid(teacher).sum() / 16,
    )


def test_sigmoid_soft_target_focal_weight_discounts_matching_cells() -> None:
    teacher = _logits(0, (2, 3, 8, 8))
    student = teacher + 0.1 * _logits(1, (2, 3, 8, 8))

    assert SigmoidSoftTargetLoss(focal_gamma=2.0)(student, teacher) < SigmoidSoftTargetLoss()(
        student, teacher
    )


def test_sigmoid_soft_target_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="focal_gamma"):
        SigmoidSoftTargetLoss(focal_gamma=-1.0)
    with pytest.raises(ValueError, match="normalize"):
        SigmoidSoftTargetLoss(normalize="sum")


def test_feature_mse_is_zero_for_equal_features() -> None:
    features = _logits(0, (5, 8))

    assert torch.allclose(FeatureMSELoss()(features, features), torch.zeros(()))
    assert FeatureMSELoss()(_logits(1, (5, 8)), features) > 0.0


def test_normalized_feature_mse_ignores_the_magnitude() -> None:
    features = _logits(0, (5, 8))

    assert torch.allclose(
        FeatureMSELoss(normalize=True)(3.0 * features, features), torch.zeros(()), atol=1e-6
    )
    assert FeatureMSELoss(normalize=False)(3.0 * features, features) > 0.0
