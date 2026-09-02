"""Tests for the CPFPN neck."""

from __future__ import annotations

import torch

from autoware_ml.models.common.necks.cp_fpn import CPFPN


def test_cpfpn_outputs_match_input_levels_and_channels() -> None:
    neck = CPFPN(in_channels=[24, 40], out_channels=16)
    high = torch.randn(2, 24, 12, 20)
    low = torch.randn(2, 40, 6, 10)
    outputs = neck((high, low))
    assert len(outputs) == 2
    assert outputs[0].shape == (2, 16, 12, 20)
    assert outputs[1].shape == (2, 16, 6, 10)
