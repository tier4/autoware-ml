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

"""Tests for dataset-generation runner helpers."""

from unittest.mock import Mock

import pytest

from autoware_ml.tools.dataset.runner import generate_dataset


def test_generate_dataset_dispatches_to_dataset_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = Mock()
    generator_class = Mock(return_value=generator)

    monkeypatch.setattr(
        "autoware_ml.tools.dataset.runner._DATASET_GENERATORS",
        {"nuscenes": generator_class},
    )

    generate_dataset(
        dataset="nuscenes",
        tasks=["segmentation3d"],
        root_path="/data/nuscenes",
        out_dir="/tmp/out",
        version="v1.0-trainval",
    )

    generator_class.assert_called_once_with()
    generator.generate.assert_called_once_with(
        root_path="/data/nuscenes",
        out_dir="/tmp/out",
        tasks=["segmentation3d"],
        version="v1.0-trainval",
    )


def test_generate_dataset_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        generate_dataset(
            dataset="unknown",
            tasks=["segmentation3d"],
            root_path="/data",
            out_dir="/tmp/out",
        )
