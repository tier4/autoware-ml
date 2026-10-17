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

"""Unit tests for the datamodule builder: it instantiates ``cfg.datamodule`` through Hydra and
reports the databases the datamodule reads."""

from __future__ import annotations

from typing import Mapping, Sequence
from dataclasses import dataclass

from omegaconf import OmegaConf
import pytest

from autoware_ml.builders.database_builder import build_datamodule


@dataclass(frozen=True)
class _Database:
    """The one attribute of a database the builder reads back."""

    version: str


class _Source:
    """A dataset source naming its database, instantiated as a nested Hydra node."""

    def __init__(self, database: _Database) -> None:
        self.database = database


class _Datamodule:
    """A datamodule exposing ``databases()`` the way :class:`DataModule` does."""

    def __init__(self, sources: Mapping[str, _Source], batch_size: int) -> None:
        self.sources = dict(sources)
        self.batch_size = batch_size

    def databases(self) -> Sequence[_Database]:
        return tuple(source.database for source in self.sources.values())


_MODULE = "autoware_ml.builders.tests.test_database_builder"


def _datamodule_cfg() -> dict:
    return {
        "datamodule": {
            "_target_": f"{_MODULE}._Datamodule",
            "batch_size": 2,
            "sources": {
                "nuscenes": {
                    "_target_": f"{_MODULE}._Source",
                    "database": {"_target_": f"{_MODULE}._Database", "version": "nuscenes_v1"},
                },
                "t4": {
                    "_target_": f"{_MODULE}._Source",
                    "database": {"_target_": f"{_MODULE}._Database", "version": "t4_v2"},
                },
            },
        }
    }


def test_build_datamodule_instantiates_the_target_with_its_nested_sources(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Input: a config whose ``datamodule`` node targets a datamodule with two nested sources.
    Expected: the datamodule and its sources are instantiated recursively and the builder reports
        the database versions it will read.
    Check: the instance type, the constructor arguments and the log line.
    """
    caplog.set_level("INFO")
    cfg = OmegaConf.create(_datamodule_cfg())

    datamodule = build_datamodule(cfg)

    assert isinstance(datamodule, _Datamodule)
    assert datamodule.batch_size == 2
    assert list(datamodule.sources) == ["nuscenes", "t4"]
    assert [database.version for database in datamodule.databases()] == ["nuscenes_v1", "t4_v2"]
    assert "['nuscenes_v1', 't4_v2']" in caplog.text


def test_build_datamodule_requires_the_datamodule_key() -> None:
    """
    Input: a config without a ``datamodule`` node.
    Expected: the builder fails on the missing key rather than returning ``None``.
    Check: an AttributeError naming the key is raised.
    """
    cfg = OmegaConf.create({"model": {"_target_": "torch.nn.Identity"}})

    with pytest.raises(AttributeError, match="datamodule"):
        build_datamodule(cfg)


def test_build_datamodule_rejects_an_unresolvable_target() -> None:
    """
    Input: a ``datamodule`` node whose ``_target_`` does not exist.
    Expected: Hydra's instantiation error propagates.
    Check: an exception mentioning the target is raised.
    """
    cfg = OmegaConf.create(
        {"datamodule": {"_target_": f"{_MODULE}._NoSuchDatamodule", "batch_size": 1}}
    )

    with pytest.raises(Exception, match="_NoSuchDatamodule"):
        build_datamodule(cfg)
