"""Tests for bundled OmegaConf config resolvers."""

from __future__ import annotations

from types import SimpleNamespace

from hydra.utils import instantiate
from omegaconf import OmegaConf

from autoware_ml.configs.resolvers import (
    merge_lists,
    register_config_resolvers,
    segmentation_class_names,
)

_MAPPING = {
    "drivable_surface": 0,
    "car": 1,
    "noise": 3,
    "ghost_point": 3,
    "unpainted": -1,
}


def test_segmentation_class_names_concatenates_shared_index() -> None:
    names = segmentation_class_names(_MAPPING, 4)
    # Shared index 3 joins names in order; index 2 has no name; -1 is dropped.
    assert names == ["drivable_surface", "car", "class_2", "noise-ghost_point"]


def test_segmentation_class_names_single_name_has_no_separator() -> None:
    names = segmentation_class_names({"car": 0, "truck": 1}, 2)
    assert names == ["car", "truck"]


def test_segmentation_class_names_handles_missing_mapping() -> None:
    assert segmentation_class_names(None, 3) == ["class_0", "class_1", "class_2"]
    assert segmentation_class_names({}, 2) == ["class_0", "class_1"]


def test_seg_class_names_resolver_in_interpolation() -> None:
    register_config_resolvers()
    cfg = OmegaConf.create(
        {
            "num_classes": 4,
            "class_mapping": _MAPPING,
            "names": "${seg_class_names:${oc.select:class_mapping, null}, ${num_classes}}",
        }
    )
    assert OmegaConf.to_container(cfg, resolve=True)["names"] == [
        "drivable_surface",
        "car",
        "class_2",
        "noise-ghost_point",
    ]


def test_seg_class_names_resolver_falls_back_without_mapping() -> None:
    register_config_resolvers()
    cfg = OmegaConf.create(
        {
            "num_classes": 3,
            "names": "${seg_class_names:${oc.select:class_mapping, null}, ${num_classes}}",
        }
    )
    assert OmegaConf.to_container(cfg, resolve=True)["names"] == [
        "class_0",
        "class_1",
        "class_2",
    ]


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
