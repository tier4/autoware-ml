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

"""Unit tests for the atomic checkpoint writer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from autoware_ml.resume.atomic_checkpoint_io import AtomicCheckpointIO


class TestAtomicCheckpointIO(unittest.TestCase):
    """Unit tests for AtomicCheckpointIO."""

    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)

    def test_save_leaves_no_temporary_file(self) -> None:
        """Test that a saved checkpoint is complete and its temporary file is gone."""
        io = AtomicCheckpointIO()
        target = self.root / "state.ckpt"

        io.save_checkpoint({"value": 1}, target)

        self.assertTrue(target.exists())
        self.assertFalse((self.root / "state.ckpt.tmp").exists())
        self.assertEqual(torch.load(target, weights_only=False)["value"], 1)

    def test_remove_deletes_the_file_and_a_leftover_temporary_file(self) -> None:
        """Test that removing a checkpoint also removes a torn temporary copy."""
        io = AtomicCheckpointIO()
        target = self.root / "state.ckpt"
        io.save_checkpoint({"value": 1}, target)
        (self.root / "state.ckpt.tmp").write_bytes(b"torn")

        io.remove_checkpoint(target)

        self.assertEqual(list(self.root.iterdir()), [])
