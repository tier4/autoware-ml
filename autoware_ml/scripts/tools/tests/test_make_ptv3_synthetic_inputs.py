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

"""Tests for the synthetic PTv3 inputs entrypoint."""

from __future__ import annotations

import subprocess
import sys
import unittest


class TestEntrypoint(unittest.TestCase):
    """Unit tests for the module entrypoint."""

    def test_module_runs_hydra_main(self) -> None:
        """
        Input: ``python -m autoware_ml.scripts.make_ptv3_synthetic_inputs --help``.
        Expected: the module starts its Hydra main.
        Check: exit code 0 and Hydra's help text.
        """
        result = subprocess.run(
            [sys.executable, "-m", "autoware_ml.scripts.make_ptv3_synthetic_inputs", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Powered by Hydra", result.stdout)


if __name__ == "__main__":
    unittest.main()
