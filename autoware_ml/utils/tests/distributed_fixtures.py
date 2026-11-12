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

"""Run a multi-rank test on CPU processes that share one gloo process group."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable

import torch.distributed as dist
import torch.multiprocessing as mp

GLOO_AVAILABLE = dist.is_available() and dist.is_gloo_available()


def _run_in_gloo_group(
    rank: int, world_size: int, init_file: str, worker: Callable[..., None], args: tuple
) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    worker(rank, world_size, *args)
    dist.destroy_process_group()


def spawn_gloo_workers(worker: Callable[..., None], *args: object, world_size: int = 2) -> None:
    """Run ``worker(rank, world_size, *args)`` on ``world_size`` CPU processes in one gloo group.

    A collective such as ``all_reduce`` only runs inside a process group, so a test of one
    needs more than one process; gloo runs on CPU, so the test needs no GPU. An exception in a
    worker, a failed ``unittest.TestCase`` assertion included, is raised again in the caller.
    """
    with tempfile.TemporaryDirectory() as directory:
        store = os.path.join(directory, "store")
        mp.spawn(_run_in_gloo_group, args=(world_size, store, worker, args), nprocs=world_size)
