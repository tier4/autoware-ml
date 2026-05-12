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

"""Collation strategy definitions for Autoware-ML datamodules."""

from enum import StrEnum


class CollationStrategy(StrEnum):
    """Declare how a batch key is collated across samples.

    Attributes:
        STACK: Fixed-shape tensors stacked along a new batch dimension.
            All samples must have identical shapes; a mismatch raises
            ``ValueError``.
        CONCAT: Variable-length tensors concatenated along dim 0.  A
            cumulative ``offset`` tensor is added to the batch so
            downstream code can recover per-sample boundaries.
        INDEX_CONCAT: Integer index tensors whose values are positions in
            the primary ``CONCAT`` key's space.  Concatenated like
            ``CONCAT`` and then each sample's portion is shifted by the
            cumulative element count of preceding samples in that space,
            keeping indices globally valid across the whole batch.
        LIST: Variable-shape values kept as a per-sample Python list.
    """

    STACK = "stack"
    CONCAT = "concat"
    INDEX_CONCAT = "index_concat"
    LIST = "list"
