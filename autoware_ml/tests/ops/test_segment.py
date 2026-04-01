"""Tests for segment reduction operators."""

from __future__ import annotations

import torch

from autoware_ml.ops.segment import segment_csr


def test_segment_csr_supports_backward_outside_onnx_export() -> None:
    src = torch.tensor(
        [[1.0, 3.0], [2.0, 1.0], [4.0, 5.0], [6.0, 7.0]],
        requires_grad=True,
    )
    indptr = torch.tensor([0, 2, 4], dtype=torch.int64)

    reduced = segment_csr(src, indptr, "mean")
    loss = reduced.sum()
    loss.backward()

    expected_grad = torch.tensor(
        [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        dtype=torch.float32,
    )
    assert torch.allclose(src.grad, expected_grad)
