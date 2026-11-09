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
    """Run ``worker(rank, world_size, init_file, *args)`` on ``world_size`` CPU processes.

    A collective such as ``all_reduce`` only runs inside a process group, so a test of one
    needs more than one process. The processes are started with ``torch.multiprocessing.spawn``
    and the worker joins the gloo group of the run through :func:`gloo_process_group`, which
    reads ``init_file``. gloo runs on CPU, so the test needs no GPU; the semantics of the
    collectives are those of every backend.

    ``join=False`` hands back the context so the wait can carry a deadline.
    ``ProcessContext.join(timeout)`` returns after one wait pass (it wakes whenever a worker
    exits and returns ``False`` while the others still run), not after ``timeout`` seconds,
    so the loop polls it until every worker is done or the deadline passes. A deadlocked
    collective, for example one rank calling ``all_reduce`` once more than the others, never
    exits on its own: at the deadline the workers are terminated and the caller fails with a
    message instead of hanging the test runner.

    An exception in a worker, a failed ``unittest.TestCase`` assertion included, is raised
    again by ``join`` in the caller as ``ProcessRaisedException`` and the other workers are
    terminated, so an assertion in a worker fails the caller.
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
