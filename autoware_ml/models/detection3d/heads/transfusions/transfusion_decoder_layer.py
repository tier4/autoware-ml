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

from autoware_ml.models.detection3d.heads.transfusions.positional_encoding import (
    LearnedPositionalEncoding,
)


class TransFusionDecoderLayer(nn.Module):
    """Refine TransFusion proposals with self- and cross-attention.

    Each decoder layer updates query features using BEV proposal positions and
    shared BEV feature maps.
    """

    def __init__(
        self, embed_dims: int, num_heads: int, feedforward_channels: int, dropout: float = 0.1
    ) -> None:
        """Initialize one TransFusion decoder layer.

        Args:
            embed_dims: Query and key embedding dimension.
            num_heads: Number of attention heads.
            feedforward_channels: Hidden dimension of the feed-forward block.
            dropout: Dropout probability used throughout the decoder.
        """
        super().__init__()
        self.query_pos_encoding = LearnedPositionalEncoding(2, embed_dims)
        self.key_pos_encoding = LearnedPositionalEncoding(2, embed_dims)
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_channels, embed_dims),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Float32[torch.Tensor, "B C num_proposals"],
        key: Float32[torch.Tensor, "B C H*W"],
        query_pos: Float32[torch.Tensor, "B num_proposals 2"],
        key_pos: Float32[torch.Tensor, "B H*W 2"],
    ) -> torch.Tensor:
        """Refine query embeddings with self- and cross-attention.

        Args:
            query: Query feature tensor.
            key: Key/value feature tensor.
            query_pos: BEV coordinates for the queries.
            key_pos: BEV coordinates for the keys.

        Returns:
            Refined query tensor.
        """
        # As in nn.TransformerDecoderLayer, the residual stream stays position-free and the
        # positional embeddings are added to q/k/v at every attention call.
        query_tokens = query.transpose(1, 2)
        key_tokens = key.transpose(1, 2)
        query_pos_embed = self.query_pos_encoding(query_pos)
        key_pos_embed = self.key_pos_encoding(key_pos)

        attended, _ = self.self_attn(
            query_tokens + query_pos_embed,
            query_tokens + query_pos_embed,
            query_tokens + query_pos_embed,
        )
        query_tokens = self.norm1(query_tokens + self.dropout(attended))

        attended, _ = self.cross_attn(
            query_tokens + query_pos_embed,
            key_tokens + key_pos_embed,
            key_tokens + key_pos_embed,
        )
        query_tokens = self.norm2(query_tokens + self.dropout(attended))

        ffn_output = self.ffn(query_tokens)
        query_tokens = self.norm3(query_tokens + self.dropout(ffn_output))
        return query_tokens.transpose(1, 2)
