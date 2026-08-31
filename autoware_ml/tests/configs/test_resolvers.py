"""Tests for bundled OmegaConf config resolvers."""

from __future__ import annotations

from types import SimpleNamespace

from hydra.utils import instantiate
from omegaconf import OmegaConf

from autoware_ml.configs.resolvers import (
    merge_lists,
    raw_name_to_train_index,
    register_config_resolvers,
)

_NAME_MAPPING = {
    "car": "car",
    "vehicle.car": "car",
    "trailer": "truck",
    "sidewalk": "non_drivable_flat",
    "ghost_point": "noise",
    "background": None,
    "some_future_category": "not_a_trained_class",
}
_CLASS_NAMES = ["car", "truck", "non_drivable_flat", "noise"]


def test_raw_name_to_train_index_maps_final_names() -> None:
    mapping = raw_name_to_train_index(_NAME_MAPPING, _CLASS_NAMES, ignore_index=-1)
    # Every raw name resolves to its final class index. Several raws may share one.
    assert mapping["car"] == 0
    assert mapping["vehicle.car"] == 0
    assert mapping["trailer"] == 1
    assert mapping["sidewalk"] == 2
    assert mapping["ghost_point"] == 3


def test_raw_name_to_train_index_sends_null_and_unknown_finals_to_ignore() -> None:
    mapping = raw_name_to_train_index(_NAME_MAPPING, _CLASS_NAMES, ignore_index=-1)
    # null final -> ignore, a final absent from class_names -> ignore (so a shared
    # name_mapping can list categories a given model does not train).
    assert mapping["background"] == -1
    assert mapping["some_future_category"] == -1


def test_seg_class_mapping_resolver_in_interpolation() -> None:
    register_config_resolvers()
    cfg = OmegaConf.create(
        {
            "ignore_index": -1,
            "name_mapping": dict(_NAME_MAPPING),
            "class_names": list(_CLASS_NAMES),
            "class_mapping": (
                "${seg_class_mapping:${name_mapping}, ${class_names}, ${ignore_index}}"
            ),
        }
    )
    resolved = OmegaConf.to_container(cfg, resolve=True)["class_mapping"]
    assert resolved == {
        "car": 0,
        "vehicle.car": 0,
        "trailer": 1,
        "sidewalk": 2,
        "ghost_point": 3,
        "background": -1,
        "some_future_category": -1,
    }


def test_merge_lists_concatenates_in_order() -> None:
    assert OmegaConf.to_container(merge_lists([1, 2], [3], [4, 5])) == [1, 2, 3, 4, 5]
    assert OmegaConf.to_container(merge_lists([])) == []
    assert OmegaConf.to_container(merge_lists()) == []


def test_merge_lists_resolver_appends_across_namespaces() -> None:
    register_config_resolvers()
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
    register_config_resolvers()
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
