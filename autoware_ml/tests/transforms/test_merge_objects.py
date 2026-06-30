"""Tests for the MergeObjects3D transform (truck + trailer merging)."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf
import pytest

from autoware_ml.transforms.boxes3d.merge import MergeObjects3D

_NAME_MAPPING = {
    "truck": "truck",
    "vehicle.truck": "truck",
    "trailer": "trailer",
    "vehicle.trailer": "trailer",
    "semi_trailer": "trailer",
}


def _instance(bbox, name, num_pts=10, velocity=(0.0, 0.0), attrs=None):
    return {
        "bbox_3d": list(bbox),
        "gt_nusc_name": name,
        "num_lidar_pts": num_pts,
        "velocity": list(velocity),
        "gt_attrs": list(attrs or []),
    }


def _merge(merge_objects=(("truck", ["truck", "trailer"]),), **kwargs):
    return MergeObjects3D(merge_objects=list(merge_objects), name_mapping=_NAME_MAPPING, **kwargs)


def test_extend_longer_geometry_matches_reference() -> None:
    # truck dx=4 at x=0, collinear trailer dx=4 at x=5 (1 m gap) -> one 9 m box.
    transform = _merge()
    out = transform.transform(
        {
            "instances": [
                _instance([0, 0, 0, 4, 2, 2, 0], "truck", num_pts=10, velocity=(1.0, 0.0)),
                _instance([5, 0, 0, 4, 2, 2, 0], "trailer", num_pts=5, velocity=(1.0, 0.0)),
            ]
        }
    )
    assert len(out["instances"]) == 1
    merged = out["instances"][0]
    assert merged["gt_nusc_name"] == "truck"
    np.testing.assert_allclose(merged["bbox_3d"], [2.5, 0.0, 0.0, 9.0, 2.0, 2.0, 0.0], atol=1e-6)
    assert merged["num_lidar_pts"] == 15
    np.testing.assert_allclose(merged["velocity"], [1.0, 0.0])
    assert "bbox_label_3d" not in merged


def test_overlapping_pair_is_merged() -> None:
    out = _merge().transform(
        {
            "instances": [
                _instance([0, 0, 0, 4, 2, 2, 0], "truck"),
                _instance([1, 0, 0, 4, 2, 2, 0], "vehicle.trailer"),  # overlaps the truck
            ]
        }
    )
    assert len(out["instances"]) == 1
    assert out["instances"][0]["gt_nusc_name"] == "truck"


def test_distant_trailer_is_not_merged() -> None:
    # trailer back face (x=98) is far from truck front face (x=2): no merge.
    out = _merge().transform(
        {
            "instances": [
                _instance([0, 0, 0, 4, 2, 2, 0], "truck"),
                _instance([100, 0, 0, 4, 2, 2, 0], "trailer"),
            ]
        }
    )
    assert len(out["instances"]) == 2
    names = sorted(i["gt_nusc_name"] for i in out["instances"])
    assert names == ["trailer", "truck"]


def test_each_box_merges_at_most_once() -> None:
    # one truck between two trailers -> only one merge consumes the truck.
    out = _merge().transform(
        {
            "instances": [
                _instance([5, 0, 0, 4, 2, 2, 0], "trailer"),
                _instance([0, 0, 0, 4, 2, 2, 0], "truck"),
                _instance([-5, 0, 0, 4, 2, 2, 0], "trailer"),
            ]
        }
    )
    merged = [i for i in out["instances"] if i["gt_nusc_name"] == "truck"]
    leftover_trailers = [i for i in out["instances"] if i["gt_nusc_name"] == "trailer"]
    assert len(merged) == 1
    assert len(leftover_trailers) == 1


def test_noop_without_rules() -> None:
    instances = [
        _instance([0, 0, 0, 4, 2, 2, 0], "truck"),
        _instance([5, 0, 0, 4, 2, 2, 0], "trailer"),
    ]
    out = MergeObjects3D(merge_objects=None, name_mapping=_NAME_MAPPING).transform(
        {"instances": list(instances)}
    )
    assert out["instances"] == instances


def test_hydra_list_config_rules_do_not_require_truthiness() -> None:
    transform = MergeObjects3D(
        merge_objects=OmegaConf.create([["truck", ["truck", "trailer"]]]),
        name_mapping=_NAME_MAPPING,
    )

    assert transform.merge_objects == [("truck", ["truck", "trailer"])]


def test_invalid_merge_type_raises() -> None:
    with pytest.raises(ValueError, match="merge_type"):
        MergeObjects3D(merge_objects=[("truck", ["truck", "trailer"])], merge_type="bogus")


def test_extend_longer_merges_center_z_and_height_from_box_faces() -> None:
    out = _merge().transform(
        {
            "instances": [
                _instance([0, 0, 1, 4, 2, 2, 0], "truck"),
                _instance([1, 0, 3, 4, 2, 2, 0], "trailer"),
            ]
        }
    )

    _, _, center_z, _, _, height, _ = out["instances"][0]["bbox_3d"]
    assert center_z == pytest.approx(2.0, abs=1e-6)
    assert height == pytest.approx(4.0, abs=1e-6)


def test_union_strategy_covers_both_boxes() -> None:
    out = _merge(merge_type="union").transform(
        {
            "instances": [
                _instance([0, 0, 0, 4, 2, 2, 0], "truck"),
                _instance([5, 0, 0, 4, 2, 2, 0], "trailer"),
            ]
        }
    )
    assert len(out["instances"]) == 1
    # Union footprint spans from the truck back (x=-2) to the trailer front (x=7).
    cx, _, _, dx, _, _, _ = out["instances"][0]["bbox_3d"]
    assert dx == pytest.approx(9.0, abs=1e-6)
    assert cx == pytest.approx(2.5, abs=1e-6)


def test_union_strategy_merges_center_z_and_height_from_box_faces() -> None:
    out = _merge(merge_type="union").transform(
        {
            "instances": [
                _instance([0, 0, 1, 4, 2, 2, 0], "truck"),
                _instance([1, 0, 3, 4, 2, 2, 0], "trailer"),
            ]
        }
    )

    _, _, center_z, _, _, height, _ = out["instances"][0]["bbox_3d"]
    assert center_z == pytest.approx(2.0, abs=1e-6)
    assert height == pytest.approx(4.0, abs=1e-6)
