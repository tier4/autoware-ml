"""Tests for bundled OmegaConf config resolvers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from autoware_ml.utils.config.settings import CONFIGS_ROOT, PROJECT_ROOT, Settings
from autoware_ml.utils.config.resolvers import (
    merge_lists,
    register_config_resolvers,
)


def test_merge_lists_concatenates_in_order() -> None:
    assert OmegaConf.to_container(merge_lists([1, 2], [3], [4, 5])) == [1, 2, 3, 4, 5]
    assert OmegaConf.to_container(merge_lists([])) == []
    assert OmegaConf.to_container(merge_lists()) == []


def test_merge_lists_resolver_appends_across_namespaces() -> None:
    register_config_resolvers(Settings())
    cfg = OmegaConf.create(
        {
            "det": {"metrics": [{"name": "map", "classes": "${classes}"}]},
            "seg": {"metrics": [{"name": "iou"}]},
            "classes": ["car", "truck"],
            "metrics": "${merge_lists:${det.metrics},${seg.metrics}}",
        }
    )
    merged = OmegaConf.to_container(cfg, resolve=True)["metrics"]
    assert [m["name"] for m in merged] == ["map", "iou"]
    assert merged[0]["classes"] == ["car", "truck"]


def test_merge_lists_resolver_preserves_hydra_recursive_instantiation() -> None:
    register_config_resolvers(Settings())
    cfg = OmegaConf.create(
        {
            "det": {
                "metrics": [
                    {
                        "_target_": "types.SimpleNamespace",
                        "name": "map",
                        "classes": "${classes}",
                    }
                ]
            },
            "seg": {"metrics": [{"_target_": "types.SimpleNamespace", "name": "iou"}]},
            "classes": ["car", "truck"],
            "model": {
                "_target_": "types.SimpleNamespace",
                "metrics": "${merge_lists:${det.metrics},${seg.metrics}}",
            },
        }
    )

    model = instantiate(cfg.model)

    assert [type(metric) for metric in model.metrics] == [SimpleNamespace, SimpleNamespace]
    assert [metric.name for metric in model.metrics] == ["map", "iou"]
    assert model.metrics[0].classes == ["car", "truck"]


def _paths(settings: Settings) -> dict[str, str]:
    register_config_resolvers(settings)
    paths = OmegaConf.load(CONFIGS_ROOT / "defaults" / "modules" / "paths.yaml")
    return {key: paths[key] for key in ("data_root_path", "cache_root_path", "working_dir")}


def test_paths_follow_the_settings() -> None:
    settings = Settings(
        data_path=Path("/data"), cache_path=Path("/cache"), working_dir=Path("/outputs")
    )

    assert _paths(settings) == {
        "data_root_path": "/data",
        "cache_root_path": "/cache",
        "working_dir": "/outputs",
    }


def test_working_dir_defaults_to_the_project_root() -> None:
    settings = Settings(data_path=Path("/data"), cache_path=Path("/cache"))

    assert _paths(settings)["working_dir"] == str(PROJECT_ROOT)


def test_data_root_path_requires_the_setting() -> None:
    with pytest.raises(InterpolationResolutionError, match="AUTOWARE_ML_DATA_PATH is not set"):
        _paths(Settings(cache_path=Path("/cache")))


def test_cache_root_path_requires_the_setting() -> None:
    with pytest.raises(InterpolationResolutionError, match="AUTOWARE_ML_CACHE_PATH is not set"):
        _paths(Settings(data_path=Path("/data")))


def test_scenarios_path_resolves_below_the_scenarios_root() -> None:
    register_config_resolvers(Settings(scenarios_path=Path("/scenarios")))
    cfg = OmegaConf.create({"lists": "${scenarios_path:t4dataset/segmentation3d}"})

    assert cfg.lists == "/scenarios/t4dataset/segmentation3d"


def test_scenarios_path_requires_the_setting() -> None:
    register_config_resolvers(Settings())
    cfg = OmegaConf.create({"lists": "${scenarios_path:t4dataset}"})

    with pytest.raises(InterpolationResolutionError, match="AUTOWARE_ML_SCENARIOS_PATH is not set"):
        _ = cfg.lists
