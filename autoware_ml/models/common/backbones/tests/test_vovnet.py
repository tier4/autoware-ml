"""Unit tests for the VoVNet multiscale backbone."""

from __future__ import annotations

from lightning.pytorch.utilities.model_helpers import _ModuleMode
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


def _batch_norms(backbone: torch.nn.Module) -> list[torch.nn.BatchNorm2d]:
    return [module for module in backbone.modules() if isinstance(module, torch.nn.BatchNorm2d)]


def _running_stats(backbone: torch.nn.Module) -> list[torch.Tensor]:
    return [
        stat.clone()
        for layer in _batch_norms(backbone)
        for stat in (layer.running_mean, layer.running_var)
    ]


def test_norm_eval_holds_through_a_lightning_fit_without_a_train_call() -> None:
    """norm_eval must hold from construction on, not only after ``train()``.

    Lightning 2.x never calls ``train()`` during fit: around validation it captures the
    ``training`` flag of every module and restores the flags afterwards. A norm_eval applied
    only by the ``train()`` override therefore never ran, and every BatchNorm layer updated
    its statistics in the training steps.
    """
    backbone = VoVNet99MultiScale(norm_eval=True)
    norm_layers = _batch_norms(backbone)
    assert norm_layers
    assert backbone.training
    assert not any(layer.training for layer in norm_layers)

    # The mode round trip of a validation run inside fit.
    mode = _ModuleMode()
    mode.capture(backbone)
    backbone.eval()
    mode.restore(backbone)
    assert backbone.training
    assert not any(layer.training for layer in norm_layers)

    # A training forward leaves the statistics alone.
    running_stats = _running_stats(backbone)
    backbone(torch.randn(2, 3, 64, 64))
    assert all(
        torch.equal(before, after)
        for before, after in zip(running_stats, _running_stats(backbone), strict=True)
    )

    # An explicit train() keeps norm_eval as well.
    backbone.train()
    assert not any(layer.training for layer in norm_layers)


def test_batchnorm_updates_its_statistics_without_norm_eval() -> None:
    backbone = VoVNet99MultiScale(norm_eval=False)
    assert all(layer.training for layer in _batch_norms(backbone))

    running_stats = _running_stats(backbone)
    backbone(torch.randn(2, 3, 64, 64))

    assert not any(
        torch.equal(before, after)
        for before, after in zip(running_stats, _running_stats(backbone), strict=True)
    )


def test_frozen_stages_disable_gradients_from_construction() -> None:
    # frozen_stages=1 freezes the stem and stage2, matching the reference.
    backbone = VoVNet99MultiScale(frozen_stages=1)

    assert all(not p.requires_grad for p in backbone.stem.parameters())
    assert all(not p.requires_grad for p in backbone.stage2.parameters())
    assert any(p.requires_grad for p in backbone.stage3.parameters())
    # The frozen stages also run in eval mode, without a train() call.
    assert not backbone.stem.training and not backbone.stage2.training
    assert backbone.stage3.training

    # Nothing is frozen by default.
    assert all(p.requires_grad for p in VoVNet99MultiScale().parameters())


def test_unknown_stage_name_is_rejected() -> None:
    # A typo would otherwise silently produce an empty output tuple.
    with pytest.raises(ValueError, match="Unsupported out_features"):
        VoVNet99MultiScale(out_features=["stage9"])
