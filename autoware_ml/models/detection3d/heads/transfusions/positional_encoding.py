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

from jaxtyping import Float32
import torch.nn as nn
import torch


class LearnedPositionalEncoding(nn.Module):
    """Learn positional embeddings from 2D BEV coordinates.

    The module maps BEV cell coordinates into query or key embeddings used by
    the TransFusion decoder.
    """

    def __init__(self, input_channels: int, embed_dims: int) -> None:
        """Initialize the positional encoding module.

        Args:
            input_channels: Number of input coordinate channels.
            embed_dims: Output embedding dimension.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(input_channels, embed_dims, kernel_size=1),
            nn.BatchNorm1d(embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_dims, embed_dims, kernel_size=1),
        )

    def forward(
        self, bev_positions: Float32[torch.Tensor, "batch_size tokens channels"]
    ) -> Float32[torch.Tensor, "batch_size tokens embed_dims"]:
        """Encode BEV positions into query embeddings.

        Args:
            bev_positions: BEV coordinate tensor.

        Returns:
            Learned positional embeddings.
        """
        # It applied positional embedding to tokens since each token represent H*W position.
        return self.proj(bev_positions.transpose(1, 2)).transpose(1, 2)


class ConstantPositionalEncoding(nn.Module):
    """Precompute a ``LearnedPositionalEncoding`` of fixed BEV positions.

    The decoder keys are the BEV cells, whose positions never change once the model is
    deployed, so their learned embedding is computed once at export and becomes an ONNX
    initializer. Left in the graph, the two 1x1 convolutions over all BEV cells would run on
    every frame: TensorRT does not fold convolutions.
    """

    def __init__(
        self,
        encoding: LearnedPositionalEncoding,
        positions: Float32[torch.Tensor, "1 tokens channels"],
    ) -> None:
        """Embed the fixed positions once with the trained encoding.

        Args:
            encoding: Trained positional encoding to evaluate.
            positions: BEV coordinate tensor of the fixed grid.

        Raises:
            ValueError: If ``encoding`` is in training mode, where its BatchNorm would embed
                from batch statistics and update its running ones.
        """
        super().__init__()
        if encoding.training:
            raise ValueError("Expected `encoding` to be in `eval()` mode.")
        parameter = next(encoding.parameters())
        positions = positions.to(parameter.device, parameter.dtype)
        with torch.no_grad():
            embedding = encoding(positions)
        self.register_buffer("embedding", embedding.detach().clone())

    def forward(
        self, bev_positions: Float32[torch.Tensor, "batch_size tokens channels"]
    ) -> Float32[torch.Tensor, "1 tokens embed_dims"]:
        """Return the precomputed embedding of the fixed positions.

        Args:
            bev_positions: BEV coordinate tensor. Only its token count is used.

        Returns:
            Positional embeddings of the fixed positions, broadcast over the batch by the
            consumer. Expanding them here would only add traced shape nodes to the exported
            graph.

        Raises:
            ValueError: If ``bev_positions`` holds a different number of tokens than the
                embedding was precomputed for. Sizes are traced during ONNX export, so the
                check only runs eagerly.
        """
        tokens = self.embedding.shape[1]
        if not torch.onnx.is_in_onnx_export() and bev_positions.shape[1] != tokens:
            raise ValueError(
                f"Positional encoding was frozen for {tokens} positions, got "
                f"{bev_positions.shape[1]}: the BEV grid differs from the export geometry."
            )
        return self.embedding
