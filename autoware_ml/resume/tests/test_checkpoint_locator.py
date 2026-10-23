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

"""Unit tests for the checkpoint locator and the run pointer."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import torch

from autoware_ml.resume.checkpoint_locator import CheckpointLocator, RunPointer

_COMPLETE_STATE = {"state_dict": {}, "optimizer_states": [], "loops": {}}


class TestCheckpointLocator(unittest.TestCase):
    """Unit tests for CheckpointLocator."""

    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)

    def _save(self, name: str, payload: dict, mtime: float) -> Path:
        """Write a checkpoint with a given modification time."""
        path = self.root / name
        torch.save(payload, path)
        os.utime(path, (mtime, mtime))
        return path

    def test_missing_directory_has_no_checkpoint(self) -> None:
        """Test that a directory that does not exist yields nothing."""
        locator = CheckpointLocator(self.root / "missing")

        self.assertEqual(locator.candidates(), [])
        self.assertIsNone(locator.find_latest())

    def test_candidates_try_last_first_then_newest_first(self) -> None:
        """Test that last.ckpt's target leads and is not repeated, the rest by age."""
        now = time.time()
        old = self._save("old.ckpt", _COMPLETE_STATE, now - 30)
        linked = self._save("linked.ckpt", _COMPLETE_STATE, now - 20)
        new = self._save("new.ckpt", _COMPLETE_STATE, now - 10)
        (self.root / "last.ckpt").symlink_to("linked.ckpt")

        candidates = CheckpointLocator(self.root).candidates()

        self.assertEqual(candidates, [linked.resolve(), new.resolve(), old.resolve()])

    def test_find_latest_skips_truncated_files(self) -> None:
        """Test that unreadable or incomplete files are skipped."""
        now = time.time()
        good = self._save("good.ckpt", _COMPLETE_STATE, now - 20)
        self._save("newer.ckpt", {"state_dict": {}}, now - 10)
        (self.root / "last.ckpt").write_bytes(b"truncated by a kill")

        self.assertEqual(CheckpointLocator(self.root).find_latest(), good.resolve())

    def test_is_loadable(self) -> None:
        """Test whether complete, incomplete and broken files count as loadable."""
        cases = (
            ("complete.ckpt", _COMPLETE_STATE, True),
            ("incomplete.ckpt", {"state_dict": {}, "loops": {}}, False),
        )
        for name, payload, expected in cases:
            with self.subTest(name=name):
                path = self._save(name, payload, time.time())
                self.assertEqual(CheckpointLocator.is_loadable(path), expected)
        broken = self.root / "broken.ckpt"
        broken.write_bytes(b"not a checkpoint")
        self.assertFalse(CheckpointLocator.is_loadable(broken))


class TestRunPointer(unittest.TestCase):
    """Unit tests for RunPointer."""

    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)
        self.pointer = RunPointer(self.root / "pointers" / "job.txt")
        self.checkpoints_dir = self.root / "run" / "checkpoints"

    def test_from_env(self) -> None:
        """Test that the pointer is read from the environment, None when unset."""
        with mock.patch.dict(os.environ, {RunPointer.ENV: str(self.pointer.path)}):
            self.assertEqual(RunPointer.from_env().path, self.pointer.path)
        with mock.patch.dict(os.environ, {RunPointer.ENV: ""}):
            self.assertIsNone(RunPointer.from_env())

    def test_rank_zero_writes_the_checkpoints_dir(self) -> None:
        """Test that the first process writes the pointer and leaves no temporary file."""
        with mock.patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            os.environ.pop("NODE_RANK", None)
            self.pointer.write(self.checkpoints_dir)

        self.assertEqual(self.pointer.path.read_text().strip(), str(self.checkpoints_dir))
        self.assertEqual([path.name for path in self.pointer.path.parent.iterdir()], ["job.txt"])

    def test_non_zero_rank_does_not_write(self) -> None:
        """Test that no process but local rank 0 of node 0 writes the pointer."""
        for local_rank, node_rank in (("1", "0"), ("0", "1")):
            with self.subTest(local_rank=local_rank, node_rank=node_rank):
                env = {"LOCAL_RANK": local_rank, "NODE_RANK": node_rank}
                with mock.patch.dict(os.environ, env):
                    self.pointer.write(self.checkpoints_dir)

                self.assertFalse(self.pointer.path.parent.exists())
