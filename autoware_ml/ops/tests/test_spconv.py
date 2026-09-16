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

"""Unit tests for sparse convolution wrappers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE

if IS_SPCONV_AVAILABLE:
    import spconv.pytorch as spconv

    from autoware_ml.ops.spconv.sparse_conv import SparseConv3d, SubMConv3d


pytestmark = pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="spconv and CUDA are required for sparse convolution tests",
)


def _build_sparse_tensor() -> spconv.SparseConvTensor:
    features = torch.tensor(
        [
            [1.0, 0.5],
            [0.8, 0.2],
            [0.1, 0.3],
            [0.7, 0.9],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    indices = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    return spconv.SparseConvTensor(features, indices, spatial_shape=[2, 2, 2], batch_size=1)


def _copy_sparse_weights(source: torch.nn.Module, target: torch.nn.Module) -> None:
    target.weight.data.copy_(source.weight.data)
    if source.bias is not None and target.bias is not None:
        target.bias.data.copy_(source.bias.data)


def _sort_sparse_output(output: spconv.SparseConvTensor) -> tuple[torch.Tensor, torch.Tensor]:
    sort_key = (
        output.indices[:, 0] * 1_000_000
        + output.indices[:, 1] * 10_000
        + output.indices[:, 2] * 100
        + output.indices[:, 3]
    )
    order = torch.argsort(sort_key)
    return output.indices[order], output.features[order]


class SparseConvExportModule(torch.nn.Module):
    """Tiny module used to verify ONNX export of sparse conv wrappers."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = SubMConv3d(2, 3, kernel_size=3, padding=1, bias=False)

    def forward(self, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        sparse_tensor = spconv.SparseConvTensor(
            features, indices, spatial_shape=[2, 2, 2], batch_size=1
        )
        output = self.conv(sparse_tensor)
        return output.features


class SharedIndiceKeyModule(torch.nn.Module):
    """Small module that exercises shared sparse indice-key reuse."""

    def __init__(self, conv_cls: type[torch.nn.Module]) -> None:
        super().__init__()
        self.conv1 = conv_cls(2, 3, kernel_size=3, padding=1, bias=False, indice_key="shared")
        self.conv2 = conv_cls(3, 4, kernel_size=3, padding=1, bias=False, indice_key="shared")

    def forward(self, sparse_tensor: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        return self.conv2(self.conv1(sparse_tensor))


class TestSparseConv3d:
    """Tests for sparse convolution wrappers."""

    def test_submconv3d_matches_spconv(self) -> None:
        """Submanifold wrapper should match native spconv output."""
        sparse_tensor = _build_sparse_tensor()
        reference = spconv.SubMConv3d(2, 3, kernel_size=3, padding=1, bias=False).cuda().eval()
        module = SubMConv3d(2, 3, kernel_size=3, padding=1, bias=False).cuda().eval()
        _copy_sparse_weights(reference, module)

        reference_output = reference(sparse_tensor)
        output = module(sparse_tensor)

        assert output.features.shape == (4, 3)
        assert output.indices.shape == (4, 4)
        assert torch.equal(output.indices, reference_output.indices)
        assert torch.allclose(output.features, reference_output.features, atol=1e-5, rtol=1e-5)

    def test_sparseconv3d_matches_spconv(self) -> None:
        """Sparse convolution wrapper should match native spconv output."""
        sparse_tensor = _build_sparse_tensor()
        reference = (
            spconv.SparseConv3d(2, 3, kernel_size=3, stride=1, padding=1, bias=False).cuda().eval()
        )
        module = SparseConv3d(2, 3, kernel_size=3, stride=1, padding=1, bias=False).cuda().eval()
        _copy_sparse_weights(reference, module)

        reference_output = reference(sparse_tensor)
        output = module(sparse_tensor)

        assert output.features.shape[1] == 3
        assert output.indices.shape[1] == 4
        output_indices, output_features = _sort_sparse_output(output)
        reference_indices, reference_features = _sort_sparse_output(reference_output)
        assert torch.equal(output_indices, reference_indices)
        assert torch.allclose(output_features, reference_features, atol=1e-5, rtol=1e-5)

    def test_submconv3d_reuses_shared_indice_key(self) -> None:
        """Shared submanifold indice keys should behave like native spconv."""
        sparse_tensor = _build_sparse_tensor()
        reference = SharedIndiceKeyModule(spconv.SubMConv3d).cuda().eval()
        module = SharedIndiceKeyModule(SubMConv3d).cuda().eval()
        _copy_sparse_weights(reference.conv1, module.conv1)
        _copy_sparse_weights(reference.conv2, module.conv2)

        reference_output = reference(sparse_tensor)
        output = module(sparse_tensor)

        output_indices, output_features = _sort_sparse_output(output)
        reference_indices, reference_features = _sort_sparse_output(reference_output)
        assert torch.equal(output_indices, reference_indices)
        assert torch.allclose(output_features, reference_features, atol=1e-5, rtol=1e-5)

    def test_onnx_export(self) -> None:
        """Sparse convolution wrapper should be exportable to ONNX."""
        import onnx

        sparse_tensor = _build_sparse_tensor()
        module = SparseConvExportModule().cuda().eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sparse_conv.onnx"
            torch.onnx.export(
                module,
                args=(sparse_tensor.features, sparse_tensor.indices),
                f=str(output_path),
                input_names=["features", "indices"],
                output_names=["output_features"],
                opset_version=20,
                dynamo=False,
            )

            assert output_path.exists()
            onnx_model = onnx.load(str(output_path), load_external_data=False)
            op_types = {node.op_type for node in onnx_model.graph.node}

            assert {"GetIndicePairs", "GetIndicePairsImplicitGemm"} & op_types
            assert {"IndiceConv", "ImplicitGemm"} & op_types
