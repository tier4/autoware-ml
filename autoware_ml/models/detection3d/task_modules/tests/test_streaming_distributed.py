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

"""Tests of ``reduce_mean_count``, the count averaging shared by distributed losses.

The multi-rank case spawns a gloo process group on CPU so the ``all_reduce`` runs.
"""

from __future__ import annotations

import math
import unittest

import torch

from autoware_ml.models.detection3d.task_modules.streaming import reduce_mean_count
from autoware_ml.models.detection3d.tests.distributed_fixtures import (
    GLOO_AVAILABLE,
    gloo_process_group,
    spawn_gloo_workers,
)


def _reduce_mean_count_worker(rank: int, world_size: int, init_file: str) -> None:
    with gloo_process_group(rank, world_size, init_file):
        local_count = torch.tensor(float(2 if rank == 0 else 8))
        assert math.isclose(reduce_mean_count(local_count).item(), 5.0)
        assert math.isclose(local_count.item(), 2.0 if rank == 0 else 8.0)
        # The floor applies to the sum over the ranks, not to the mean.
        assert math.isclose(reduce_mean_count(torch.tensor(float(1 - rank))).item(), 0.5)
        assert math.isclose(reduce_mean_count(torch.tensor(0.0)).item(), 0.5)
        pair = reduce_mean_count(torch.tensor([float(1 - rank), 4.0]))
        torch.testing.assert_close(pair, torch.tensor([0.5, 4.0]))


class TestReduceMeanCount(unittest.TestCase):
    def test_is_identity_without_process_group(self) -> None:
        """Test that without a process group the count comes back as float, floored at 1."""
        self.assertAlmostEqual(reduce_mean_count(torch.tensor(3.0)).item(), 3.0)
        self.assertEqual(reduce_mean_count(torch.tensor(3)).dtype, torch.float32)
        self.assertAlmostEqual(reduce_mean_count(torch.tensor(0.0)).item(), 1.0)

    @unittest.skipUnless(GLOO_AVAILABLE, "gloo backend unavailable")
    def test_averages_counts_across_ranks(self) -> None:
        """Test that two ranks get the mean of their counts, floored over the sum, without
        in-place changes."""
        spawn_gloo_workers(_reduce_mean_count_worker)


if __name__ == "__main__":
    unittest.main()
