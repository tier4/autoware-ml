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

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class GlobalAveragePooling(nn.Module):
    """Global Average Pooling neck.

    Pools spatial dimensions and flattens the output to (B, C).
    Supports both adaptive pooling (default) and fixed kernel pooling.

    Args:
        kernel_size: Pooling kernel size. If None, uses AdaptiveAvgPool2d((1, 1)).
        stride: Pooling stride. Only used when kernel_size is provided.
    """

    def __init__(
        self,
        kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> None:
        """Initialize GlobalAveragePooling.

        Args:
            kernel_size: Pooling kernel size. If None, uses AdaptiveAvgPool2d((1, 1)).
            stride: Pooling stride. Only used when kernel_size is provided.
        """
        super().__init__()
        if kernel_size is None:
            self.gap: nn.Module = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AvgPool2d(kernel_size, stride)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pool spatial dimensions and flatten to (B, C).

        Args:
            inputs: Input tensor of shape (B, C, H, W).

        Returns:
            Pooled tensor of shape (B, C).
        """
        return self.gap(inputs).flatten(1)
