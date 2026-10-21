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

"""Tests for the interface every dataset task implements."""

from __future__ import annotations

import logging

import polars as pl
import pytest

from autoware_ml.datamodule.base_dataset_task import BaseDatasetTask
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample


class _RecordCountTask(BaseDatasetTask):
    """A task reading nothing, to exercise the defaults of the base class."""

    def select_columns(self, dataset_records_dataframe: pl.DataFrame) -> pl.DataFrame:
        return dataset_records_dataframe

    def get_data_sample(self, idx: int) -> ModelGTSample:
        raise AssertionError("not read in these tests")


def test_a_task_must_implement_select_columns_and_get_data_sample() -> None:
    """
    Input: subclasses leaving out one of the two abstract methods.
    Expected: they cannot be instantiated, so a task missing a reader fails at construction
        rather than on the first sample.
    Check: a TypeError naming the missing method.
    """

    class _NoReader(BaseDatasetTask):
        def select_columns(self, dataset_records_dataframe: pl.DataFrame) -> pl.DataFrame:
            return dataset_records_dataframe

    class _NoColumns(BaseDatasetTask):
        def get_data_sample(self, idx: int) -> ModelGTSample:
            raise AssertionError

    with pytest.raises(TypeError, match="get_data_sample"):
        _NoReader("/data", pl.DataFrame())
    with pytest.raises(TypeError, match="select_columns"):
        _NoColumns("/data", pl.DataFrame())


def test_default_name_and_log_line_report_the_class_and_the_record_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Input: a task over three records, without its own __str__ or log_dataset_info.
    Expected: it is named after its class and logs its record count.
    Check: str() and the captured log line.
    """
    task = _RecordCountTask("/data", pl.DataFrame({"frame": [1, 2, 3]}))

    with caplog.at_level(logging.INFO, logger="autoware_ml.datamodule.base_dataset_task"):
        task.log_dataset_info()

    assert str(task) == "_RecordCountTask"
    assert "_RecordCountTask: 3 records." in caplog.text
