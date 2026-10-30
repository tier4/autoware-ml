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

"""Shared fixtures of every autoware-ml test.

Tests never read the ``.env`` of the checkout, which holds the settings of one machine. Every test
starts from a settings file with no setting set, a test that needs a setting writes its own.
"""

import pytest

from autoware_ml.configs.tests.settings import write_settings


@pytest.fixture(autouse=True)
def unset_settings(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the settings at a file with no setting set."""
    write_settings(tmp_path_factory.mktemp("settings"), monkeypatch)
