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

"""Helpers of the multi-rank tests: spawn gloo workers and open their process group."""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch.distributed as dist
import torch.multiprocessing as mp

GLOO_AVAILABLE = dist.is_available() and dist.is_gloo_available()


def spawn_gloo_workers(
    worker: Callable[..., None], *args: object, world_size: int = 2, timeout: float = 120.0
) -> None:
    """Run ``worker(rank, world_size, init_file, *args)`` on ``world_size`` processes.

    An assertion in a worker fails the caller. A deadlocked collective fails it after
    ``timeout`` seconds instead of hanging the test runner.
    """
    with tempfile.TemporaryDirectory() as directory:
        context = mp.spawn(
            worker,
            args=(world_size, os.path.join(directory, "store"), *args),
            nprocs=world_size,
            join=False,
        )
        # join(timeout) returns after one wait pass, so poll until the deadline.
        deadline = time.monotonic() + timeout
        while not context.join(timeout=5.0):
            if time.monotonic() > deadline:
                for process in context.processes:
                    process.terminate()
                raise AssertionError(f"distributed workers did not finish within {timeout}s")


@contextmanager
def gloo_process_group(rank: int, world_size: int, init_file: str) -> Iterator[None]:
    """Join the gloo process group of a spawned worker for the duration of the block."""
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        yield
    finally:
        dist.destroy_process_group()
