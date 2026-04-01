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

"""Tests for PTv3 indexing/export helper ops."""

from __future__ import annotations

import torch

from autoware_ml.ops.indexing import argsort, unique


def test_unique_matches_torch_unique_contract() -> None:
    values = torch.tensor([3, 1, 3, 2, 1], dtype=torch.int64)

    unique_values, inverse_indices, counts, num_out = unique(values)

    assert torch.equal(unique_values, torch.tensor([1, 2, 3], dtype=torch.int64))
    assert torch.equal(inverse_indices, torch.tensor([2, 0, 2, 1, 0], dtype=torch.int64))
    assert torch.equal(counts, torch.tensor([2, 1, 2], dtype=torch.int64))
    assert torch.equal(num_out, torch.tensor([3], dtype=torch.int64))


def test_argsort_matches_torch_sort_indices() -> None:
    values = torch.tensor([[4, 1, 3], [2, 0, 5]], dtype=torch.int64)

    indices = argsort(values)

    assert torch.equal(indices, torch.tensor([[1, 2, 0], [1, 0, 2]], dtype=torch.int64))
