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
import torch.nn.functional as F


# TODO (KokSeang): Any Exportable-based modules should be moved to the deployment folder since
# they can be reused
class ExportableMultiheadAttention(nn.Module):
    """ONNX/TensorRT-friendly equivalent of ``nn.MultiheadAttention``.

    The default path retains an explicitly stabilized attention graph. The optional fused path
    emits the direct MatMul-Softmax-MatMul pattern recognized by TensorRT and can run its
    attention core in bf16.
    """

    def __init__(
        self,
        attention: nn.MultiheadAttention,
        fuse_attention: bool = False,
        use_bf16: bool = False,
    ) -> None:
        """Copy trained weights from a batch-first MultiheadAttention module."""
        super().__init__()
        if not attention.batch_first:
            raise ValueError("TransFusion export expects batch_first=True attention.")
        if attention.in_proj_weight is None or attention.in_proj_bias is None:
            raise ValueError("TransFusion export expects packed QKV projection weights.")

        self.embed_dim = attention.embed_dim
        self.num_heads = attention.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = attention.dropout
        self.fuse_attention = fuse_attention
        self.use_bf16 = use_bf16
        if use_bf16 and not fuse_attention:
            raise ValueError("bf16 attention requires the fusion-ready attention path.")
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self._copy_weights(attention)
        self.to(device=attention.in_proj_weight.device, dtype=attention.in_proj_weight.dtype)

    def _copy_weights(self, attention: nn.MultiheadAttention) -> None:
        """Copy packed PyTorch MHA weights into explicit projection layers."""
        q_weight, k_weight, v_weight = attention.in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = attention.in_proj_bias.chunk(3, dim=0)
        self.q_proj.weight.data.copy_(q_weight)
        self.k_proj.weight.data.copy_(k_weight)
        self.v_proj.weight.data.copy_(v_weight)
        self.q_proj.bias.data.copy_(q_bias)
        self.k_proj.bias.data.copy_(k_bias)
        self.v_proj.bias.data.copy_(v_bias)
        self.out_proj.weight.data.copy_(attention.out_proj.weight)
        self.out_proj.bias.data.copy_(attention.out_proj.bias)

    def _project(self, projection: nn.Linear, tokens: torch.Tensor) -> torch.Tensor:
        """Project and reshape tokens to ``(batch, heads, sequence, channels)``."""
        batch_size, sequence_length, _ = tokens.shape
        projected = projection(tokens)
        projected = projected.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        return projected.transpose(1, 2)

    def forward(
        self,
        query: Float32[torch.Tensor, "B H*W channels"],
        key: Float32[torch.Tensor, "B H*W channels"],
        value: Float32[torch.Tensor, "B H*W channels"],
    ) -> tuple[torch.Tensor, None]:
        """
        Run explicit scaled dot-product attention with a PyTorch-compatible return.
        The shape of query/key/value depend on nn.MultiHeadAttention, however,
        since we set batch_first=True, and use it in bev for Transfusion, the shape is always
        B, H*W, dimensions.
        """
        q = self._project(self.q_proj, query)
        k = self._project(self.k_proj, key)
        v = self._project(self.v_proj, value)

        if self.fuse_attention:
            q = q / self.head_dim**0.5
            if self.use_bf16:
                q = q.to(torch.bfloat16)
                k = k.to(torch.bfloat16)
                v = v.to(torch.bfloat16)
            attention = torch.matmul(q, k.transpose(-2, -1))
        else:
            attention = torch.matmul(q.float() / self.head_dim**0.5, k.float().transpose(-2, -1))
            attention = attention - attention.max(dim=-1, keepdim=True).values
        attention = attention.softmax(dim=-1)
        if self.training and self.dropout > 0.0:
            attention = F.dropout(attention, p=self.dropout)
        if self.fuse_attention:
            attended = torch.matmul(attention, v)
        else:
            attended = torch.matmul(attention.to(v.dtype), v)
        if self.use_bf16:
            attended = attended.to(query.dtype)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(query.shape[0], query.shape[1], self.embed_dim)
        )
        return self.out_proj(attended), None
