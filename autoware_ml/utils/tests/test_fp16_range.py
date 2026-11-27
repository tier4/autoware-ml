# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the fp16 range guard of the export modules."""

import unittest

import torch
import torch.nn as nn

from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.fp16_range import fp16_range_report, validate_fp16_range


class _Attention(nn.Module):
    """Dot-product logits of a query and its keys, scaled by ``gain``."""

    def __init__(self, gain: float) -> None:
        super().__init__()
        self.query = nn.Linear(4, 4, bias=False)
        self.gain = gain

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self.query(tokens) * self.gain
        logits = query @ tokens.T
        mask = torch.eye(tokens.shape[0], dtype=torch.bool)
        return logits.masked_fill(mask, float("-inf")).softmax(dim=-1)


class _Head(nn.Module):
    """One attention block under a head, as an export module nests it."""

    def __init__(self, gain: float) -> None:
        super().__init__()
        self.attn = _Attention(gain)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        empty = torch.full((2,), torch.finfo(torch.float32).min)
        return self.attn(tokens) + empty.amax() * 0.0


def _spec(gain: float) -> ExportSpec:
    torch.manual_seed(0)
    module = _Head(gain).eval()
    with torch.no_grad():
        module.attn.query.weight.copy_(torch.eye(4))
    return ExportSpec(
        module=module, args=(torch.full((3, 4), 10.0),), input_names=["tokens"], output_names=["y"]
    )


class _Overflow(nn.Module):
    """An exponential that overflows to infinity on large inputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x * 100.0)


class TestFp16Range(unittest.TestCase):
    def test_a_logit_beyond_the_share_names_its_module_and_op(self) -> None:
        """
        Input: attention logits of 40000, a share of 0.25 (16376).
        Expected: the export is refused, the error names the attention module and its matmul.
        """
        with self.assertRaisesRegex(ValueError, r"head\.attn mm reaches 40000"):
            validate_fp16_range(_spec(100.0), "head", 0.25)

    def test_masks_and_sentinels_pass(self) -> None:
        """
        Input: logits of 400, a minus infinity mask and a float32 minimum fill.
        Expected: the module stays within the share, the mask and the fill are sentinels.
        """
        validate_fp16_range(_spec(1.0), "head", 0.25)

    def test_the_report_names_the_graph_inputs(self) -> None:
        """
        Input: tokens of 10.
        Expected: the input is reported under its graph name.
        """
        report = fp16_range_report(_spec(1.0), "head")

        self.assertEqual(report["head input tokens"], 10.0)

    def test_a_nan_is_refused(self) -> None:
        """
        Input: a NaN in the trace batch.
        Expected: the export is refused at any share.
        """
        spec = _spec(1.0)
        spec.args[0][0, 0] = float("nan")

        with self.assertRaisesRegex(ValueError, "nan"):
            validate_fp16_range(spec, "head", 1.0)

    def test_the_share_must_be_a_fraction_of_the_range(self) -> None:
        """
        Input: a share outside (0, 1].
        Expected: ValueError naming the setting.
        """
        for share in [0.0, 1.5]:
            with self.subTest(share=share):
                with self.assertRaisesRegex(ValueError, "fp16_range_share"):
                    validate_fp16_range(_spec(1.0), "head", share)

    def test_a_positive_infinity_is_an_overflow(self) -> None:
        """
        Input: a module whose exponential overflows to infinity on the trace batch.
        Expected: the export is refused at any share.
        """
        spec = ExportSpec(
            module=_Overflow(),
            args=(torch.full((2,), 10.0),),
            input_names=["x"],
            output_names=["y"],
        )

        with self.assertRaisesRegex(ValueError, "reaches inf"):
            validate_fp16_range(spec, "head", 1.0)
