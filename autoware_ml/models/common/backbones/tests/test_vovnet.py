"""Unit tests for the VoVNet multiscale backbone."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.models.common.backbones.vovnet import VoVNet99MultiScale


def test_returns_requested_stages_with_reference_channels() -> None:
    backbone = VoVNet99MultiScale(input_ch=3, out_features=["stage4", "stage5"])
    stage4, stage5 = backbone(torch.randn(1, 3, 64, 64))

    # Channel counts and strides the StreamPETR neck depends on.
    assert stage4.shape == (1, 768, 4, 4)
    assert stage5.shape == (1, 1024, 2, 2)


def test_out_features_selects_and_orders_outputs() -> None:
    backbone = VoVNet99MultiScale(out_features=["stage3", "stage5"])
    stage3, stage5 = backbone(torch.randn(1, 3, 64, 64))

    assert stage3.shape[1] == 512
    assert stage5.shape[1] == 1024


def test_norm_eval_keeps_batchnorm_in_eval_mode_while_training() -> None:
    backbone = VoVNet99MultiScale(norm_eval=True).train()

    norm_layers = [m for m in backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert norm_layers
    assert all(not layer.training for layer in norm_layers)

    trainable = VoVNet99MultiScale(norm_eval=False).train()
    assert all(
        layer.training for layer in trainable.modules() if isinstance(layer, torch.nn.BatchNorm2d)
    )


def test_frozen_stages_disable_gradients_from_construction() -> None:
    # frozen_stages=1 freezes the stem and stage2, matching the reference.
    backbone = VoVNet99MultiScale(frozen_stages=1)

    assert all(not p.requires_grad for p in backbone.stem.parameters())
    assert all(not p.requires_grad for p in backbone.stage2.parameters())
    assert any(p.requires_grad for p in backbone.stage3.parameters())

    # Nothing is frozen by default.
    assert all(p.requires_grad for p in VoVNet99MultiScale().parameters())


def test_unknown_stage_name_is_rejected() -> None:
    # A typo would otherwise silently produce an empty output tuple.
    with pytest.raises(ValueError, match="Unsupported out_features"):
        VoVNet99MultiScale(out_features=["stage9"])
