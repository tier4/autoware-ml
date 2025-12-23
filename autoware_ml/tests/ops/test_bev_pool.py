# Copyright 2025 TIER IV, Inc.
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

"""Unit tests for BEV pooling CUDA operation."""

import pytest
import torch

from autoware_ml.ops.bev_pool import bev_pool

from .conftest import (
    BEVGridConfig,
    create_multi_point_input,
    create_single_point_input,
    create_uniform_grid_input,
)


class TestBEVPoolForward:
    """Tests for BEV pool forward pass."""

    def test_single_point_single_cell(
        self, bev_config: BEVGridConfig, cuda_device: torch.device
    ) -> None:
        """Single point should appear at correct grid cell."""
        h_idx, w_idx, d_idx, b_idx = 1, 2, 0, 0
        feature_value = 5.0

        feats, coords, ranks = create_single_point_input(
            bev_config, h_idx, w_idx, d_idx, b_idx, feature_value, cuda_device
        )

        output = bev_pool(
            feats,
            coords,
            ranks,
            B=bev_config.B,
            D=bev_config.D,
            H=bev_config.H,
            W=bev_config.W,
            is_training=True,
        )

        # Output shape: (B, C, D, H, W)
        assert output.shape == (
            bev_config.B,
            bev_config.C,
            bev_config.D,
            bev_config.H,
            bev_config.W,
        )

        # Check the target cell has correct value
        expected = torch.full((bev_config.C,), feature_value, device=cuda_device)
        actual = output[b_idx, :, d_idx, h_idx, w_idx]
        assert torch.allclose(actual, expected)

        # Check that other cells are zero
        output_copy = output.clone()
        output_copy[b_idx, :, d_idx, h_idx, w_idx] = 0
        assert torch.allclose(output_copy, torch.zeros_like(output_copy))

    def test_multiple_points_same_cell_sum(
        self, bev_config: BEVGridConfig, cuda_device: torch.device
    ) -> None:
        """Multiple points at same cell should have their features summed."""
        h_idx, w_idx, d_idx, b_idx = 0, 0, 0, 0

        # Two points at same location with values 1.0 and 2.0
        coords_list = [(h_idx, w_idx, d_idx, b_idx), (h_idx, w_idx, d_idx, b_idx)]
        feature_values = [1.0, 2.0]

        feats, coords, ranks = create_multi_point_input(
            bev_config, coords_list, feature_values, cuda_device
        )

        output = bev_pool(
            feats,
            coords,
            ranks,
            B=bev_config.B,
            D=bev_config.D,
            H=bev_config.H,
            W=bev_config.W,
            is_training=True,
        )

        # Expected: 1.0 + 2.0 = 3.0 in all channels
        expected = torch.full((bev_config.C,), 3.0, device=cuda_device)
        actual = output[b_idx, :, d_idx, h_idx, w_idx]
        assert torch.allclose(actual, expected)

    def test_multiple_cells(self, bev_config: BEVGridConfig, cuda_device: torch.device) -> None:
        """Points at different cells should be placed correctly."""
        # Three points at different locations
        coords_list = [
            (0, 0, 0, 0),  # cell 1
            (1, 1, 1, 0),  # cell 2
            (2, 3, 2, 0),  # cell 3
        ]
        feature_values = [1.0, 2.0, 3.0]

        feats, coords, ranks = create_multi_point_input(
            bev_config, coords_list, feature_values, cuda_device
        )

        output = bev_pool(
            feats,
            coords,
            ranks,
            B=bev_config.B,
            D=bev_config.D,
            H=bev_config.H,
            W=bev_config.W,
            is_training=True,
        )

        # Check each cell
        for (h, w, d, b), val in zip(coords_list, feature_values):
            expected = torch.full((bev_config.C,), val, device=cuda_device)
            actual = output[b, :, d, h, w]
            assert torch.allclose(actual, expected), f"Mismatch at cell ({h},{w},{d},{b})"

    def test_uniform_grid(self, bev_config: BEVGridConfig, cuda_device: torch.device) -> None:
        """Uniform grid should have same value everywhere."""
        feature_value = 7.0

        feats, coords, ranks = create_uniform_grid_input(bev_config, feature_value, cuda_device)

        output = bev_pool(
            feats,
            coords,
            ranks,
            B=bev_config.B,
            D=bev_config.D,
            H=bev_config.H,
            W=bev_config.W,
            is_training=True,
        )

        expected = torch.full_like(output, feature_value)
        assert torch.allclose(output, expected)

    def test_output_shape_various_configs(self, cuda_device: torch.device) -> None:
        """Output shape should match (B, C, D, H, W) for various configs."""
        configs = [
            BEVGridConfig(B=1, D=2, H=4, W=4, C=16),
            BEVGridConfig(B=2, D=8, H=8, W=8, C=32),
            BEVGridConfig(B=1, D=1, H=1, W=1, C=4),
        ]

        for config in configs:
            feats, coords, ranks = create_single_point_input(config, 0, 0, 0, 0, 1.0, cuda_device)

            output = bev_pool(
                feats,
                coords,
                ranks,
                B=config.B,
                D=config.D,
                H=config.H,
                W=config.W,
                is_training=True,
            )

            expected_shape = (config.B, config.C, config.D, config.H, config.W)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"


class TestBEVPoolBackward:
    """Tests for BEV pool backward pass (gradient flow)."""

    def test_gradient_flow(self, bev_config: BEVGridConfig, cuda_device: torch.device) -> None:
        """Gradients should flow back to input features."""
        feats, coords, ranks = create_single_point_input(bev_config, 0, 0, 0, 0, 1.0, cuda_device)
        feats.requires_grad_(True)

        output = bev_pool(
            feats,
            coords,
            ranks,
            B=bev_config.B,
            D=bev_config.D,
            H=bev_config.H,
            W=bev_config.W,
            is_training=True,
        )

        loss = output.sum()
        loss.backward()

        assert feats.grad is not None, "Gradients should flow to input features"
        assert not torch.isnan(feats.grad).any(), "Gradients should not be NaN"

    def test_gradient_correctness(
        self, bev_config: BEVGridConfig, cuda_device: torch.device
    ) -> None:
        """Gradient should broadcast correctly from output to input."""
        h_idx, w_idx, d_idx, b_idx = 1, 1, 1, 0

        feats, coords, ranks = create_single_point_input(
            bev_config, h_idx, w_idx, d_idx, b_idx, 1.0, cuda_device
        )
        feats.requires_grad_(True)

        output = bev_pool(
            feats,
            coords,
            ranks,
            B=bev_config.B,
            D=bev_config.D,
            H=bev_config.H,
            W=bev_config.W,
            is_training=True,
        )

        # Only backprop from the cell where the point landed
        target_cell = output[b_idx, :, d_idx, h_idx, w_idx]
        loss = target_cell.sum()
        loss.backward()

        # Gradient should be 1.0 for all channels (since we sum)
        expected_grad = torch.ones_like(feats)
        assert torch.allclose(feats.grad, expected_grad)


class TestBEVPoolModes:
    """Tests for training vs inference mode parity."""

    def test_training_vs_inference_match(
        self, bev_config: BEVGridConfig, cuda_device: torch.device
    ) -> None:
        """Training and inference modes should produce identical outputs."""
        coords_list = [
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (1, 2, 1, 0),
        ]
        feature_values = [1.0, 2.0, 5.0]

        feats, coords, ranks = create_multi_point_input(
            bev_config, coords_list, feature_values, cuda_device
        )

        # Training mode
        output_train = bev_pool(
            feats,
            coords,
            ranks,
            B=bev_config.B,
            D=torch.tensor(bev_config.D),
            H=torch.tensor(bev_config.H),
            W=torch.tensor(bev_config.W),
            is_training=True,
        )

        # Inference mode
        output_infer = bev_pool(
            feats,
            coords,
            ranks,
            B=bev_config.B,
            D=torch.tensor(bev_config.D),
            H=torch.tensor(bev_config.H),
            W=torch.tensor(bev_config.W),
            is_training=False,
        )

        assert torch.allclose(output_train, output_infer), (
            "Training and inference outputs should match"
        )

    def test_inference_mode_no_grad(
        self, bev_config: BEVGridConfig, cuda_device: torch.device
    ) -> None:
        """Inference mode should raise error on backward."""
        feats, coords, ranks = create_single_point_input(bev_config, 0, 0, 0, 0, 1.0, cuda_device)
        feats.requires_grad_(True)

        output = bev_pool(
            feats,
            coords,
            ranks,
            B=bev_config.B,
            D=torch.tensor(bev_config.D),
            H=torch.tensor(bev_config.H),
            W=torch.tensor(bev_config.W),
            is_training=False,
        )

        with pytest.raises(NotImplementedError):
            output.sum().backward()
