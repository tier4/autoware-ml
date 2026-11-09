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

"""Unit tests for the visualization preview entrypoint."""

from __future__ import annotations

from importlib import import_module
import unittest

from autoware_ml.cli import cli


class TestEntrypointModules(unittest.TestCase):
    """Every module the CLI dispatches to imports."""

    def test_cli_entrypoint_modules_are_importable(self) -> None:
        """
        Input: the module name of every entrypoint the CLI dispatches to.
        Expected: every module imports, so a broken import fails here rather than at the first
            CLI call.
        Check: importing each module raises nothing.
        """
        for constant_name in (
            "TRAIN_ENTRYPOINT_MODULE",
            "DEPLOY_ENTRYPOINT_MODULE",
            "TEST_ENTRYPOINT_MODULE",
            "VISUALIZE_ENTRYPOINT_MODULE",
        ):
            with self.subTest(constant_name=constant_name):
                import_module(getattr(cli, constant_name))


if __name__ == "__main__":
    unittest.main()
