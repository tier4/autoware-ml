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

"""Unit tests for the neighbor targets measured on the train split."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from autoware_ml.datamodule.data_module import DataModule
from autoware_ml.datamodule.tests.corpus import write_corpus
from autoware_ml.datamodule.tests.test_data_module import (
    EmptyValidationSplitter,
    FakeDatabase,
    build_source,
)
from autoware_ml.scripts.tools.ptv3.measure_neighbors import measure_neighbors


class TestMeasureNeighbors(unittest.TestCase):
    """Unit tests for measure_neighbors."""

    def test_reads_every_source_once_and_takes_the_p99(self) -> None:
        """
        Input: a three frame corpus beside a one frame corpus repeated six times, whose
            scenarios put no scene in the validation split, and a measurement returning
            (0, 0), (1, 2), (2, 4) and (3, 6) in turn.
        Expected: every frame of every source is measured once, four frames rather than the
            nine of a training epoch, without building the empty validation split.
        Check: the number of frames measured and the p99 of both stages.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main = FakeDatabase(root / "main", write_corpus(root / "main", num_records=3), "main")
            small = FakeDatabase(
                root / "small",
                write_corpus(root / "small", num_records=1),
                "small",
            )
            datamodule = DataModule(
                splitter=EmptyValidationSplitter(),
                sources={"main": build_source(main), "small": build_source(small, repeat=6)},
                train_dataloader=None,
                validation_dataloader=None,
                test_dataloader=None,
                predict_dataloader=None,
                train_frame_sampling=None,
            )
            neighbors_of = mock.Mock(side_effect=[[frame, 2 * frame] for frame in range(4)])
            targets = measure_neighbors(datamodule, neighbors_of, num_workers=0)

        self.assertEqual(neighbors_of.call_count, 4)
        self.assertEqual([round(target, 2) for target in targets], [2.97, 5.94])


class TestEntrypoint(unittest.TestCase):
    """Unit tests for the module entrypoint."""

    def test_module_runs_hydra_main(self) -> None:
        """
        Input: ``python -m autoware_ml.scripts.tools.ptv3.measure_neighbors --help``.
        Expected: the module starts its Hydra main.
        Check: exit code 0 and Hydra's help text.
        """
        result = subprocess.run(
            [sys.executable, "-m", "autoware_ml.scripts.tools.ptv3.measure_neighbors", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Powered by Hydra", result.stdout)


if __name__ == "__main__":
    unittest.main()
