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

"""Unit tests for the dataloader configuration."""

from __future__ import annotations

import unittest

from autoware_ml.datamodule.base import DataLoaderConfig


class TestDataLoaderConfig(unittest.TestCase):
    """Conversion of the configuration into dataloader keyword arguments."""

    def test_defaults_build_a_single_sample_loader(self) -> None:
        kwargs = DataLoaderConfig().to_dataloader_kwargs()

        self.assertEqual(kwargs["batch_size"], 1)
        self.assertFalse(kwargs["shuffle"])
        self.assertFalse(kwargs["drop_last"])

    def test_persistent_workers_need_workers(self) -> None:
        kwargs = DataLoaderConfig(num_workers=0, persistent_workers=True).to_dataloader_kwargs()

        self.assertFalse(kwargs["persistent_workers"])

    def test_persistent_workers_stay_on_with_workers(self) -> None:
        kwargs = DataLoaderConfig(num_workers=4, persistent_workers=True).to_dataloader_kwargs()

        self.assertTrue(kwargs["persistent_workers"])
