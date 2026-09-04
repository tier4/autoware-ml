"""Tests for the region-filter axis: lanelet parsing, membership, and suite use."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from shapely.geometry import Point, Polygon

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.detection3d.mean_ap import MeanAP
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite
from autoware_ml.metrics.filters import CollisionFilter, CorridorFilter, RegionFilter
from autoware_ml.metrics.geometry.lanelet import (
    LaneletMap,
    LaneletMapProvider,
    T4LaneletMapResolver,
    load_lanelet_speeds,
    load_region_polygons,
)
from autoware_ml.metrics.geometry.reachability import ReachabilityParams
from autoware_ml.metrics.segmentation3d.accuracy import Accuracy
from autoware_ml.metrics.segmentation3d.error_clusters import ErrorClusters
from autoware_ml.metrics.segmentation3d.point_cloud import Segmentation3DPointCloudMetricSuite
from autoware_ml.metrics.segmentation3d.suite import Segmentation3DConfusionMatrixMetricSuite

_REAL_MAP = (
    "data/t4dataset/db_j6gen2_v2/13cabeac-a81b-4a57-ae31-d9520a442729/0/map/lanelet2_map.osm"
)


class _StubProvider:
    """Returns one fixed LaneletMap for every scene token.

    ``no_map`` names scene tokens that have no lanelet map: ``available`` reports
    them False (so the suite excludes them) and ``get`` raises if one is ever
    loaded anyway, guarding that the suite never touches a map-less scene's map.
    """

    def __init__(self, lanelet_map: LaneletMap, no_map: tuple[object, ...] = ()) -> None:
        self._map = lanelet_map
        self._no_map = set(no_map)

    def get(self, scene_token: object) -> LaneletMap:
        if scene_token in self._no_map:
            raise FileNotFoundError(f"map loaded for a scene declared map-less: {scene_token!r}")
        return self._map

    def available(self, scene_token: object) -> bool:
        return scene_token not in self._no_map


def _road_map() -> LaneletMap:
    # A rectangular road region covering x in [0, 50], y in [-10, 10] (map frame).
    return LaneletMap({"road": [Polygon([(0, -10), (50, -10), (50, 10), (0, 10)])]})


@pytest.mark.skipif(not os.path.exists(_REAL_MAP), reason="gen2 map not available")
def test_load_region_polygons_real_map() -> None:
    regions = load_region_polygons(_REAL_MAP)
    assert regions["road"], "expected road lanelet polygons"
    assert "walkway" in regions  # sparse but present in this scene
    assert all(polygon.is_valid for polygon in regions["road"])


def test_region_filter_membership() -> None:
    region_filter = RegionFilter(["road"], _StubProvider(_road_map()))
    xyz = np.array([[10.0, 0.0, 0.0], [100.0, 0.0, 0.0]])  # inside road, off-road
    keep = region_filter.keep(xyz, {"ego2global": np.eye(4), "scene_token": "scene"})
    assert keep.tolist() == [True, False]


def test_region_filter_box_footprint_any_overlap() -> None:
    # Road strip y in [-3, 3]. A box centered at y=4 (off road) but 4 m wide
    # overhangs onto the road: center-in-region drops it, footprint any-overlap
    # keeps it. A box at y=8 is fully off the road and dropped.
    region_filter = RegionFilter(["road"], _StubProvider(_narrow_road_map()))
    context = {"ego2global": np.eye(4), "scene_token": "scene"}
    boxes = np.array(
        [
            [10.0, 4.0, 0.0, 4.0, 4.0, 1.5, 0.0, 0.0, 0.0],  # footprint y in [2, 6] -> overlaps road
            [10.0, 8.0, 0.0, 4.0, 4.0, 1.5, 0.0, 0.0, 0.0],  # footprint y in [6, 10] -> off road
        ]
    )
    assert region_filter.keep(boxes, context).tolist() == [True, False]


def _narrow_road_map() -> LaneletMap:
    # A road strip x in [0, 50], y in [-3, 3] (map frame) to exercise the road overlap.
    return LaneletMap({"road": [Polygon([(0, -3), (50, -3), (50, 3), (0, 3)])]})


def test_collision_filter_keeps_forward_reachable_on_road() -> None:
    # The collision area is the ego reachable region (curvature-bounded, clipped
    # to the road): a point ahead on the centerline is kept, one behind ego, or
    # off the road, is dropped.
    collision_filter = CollisionFilter(
        _StubProvider(_road_map()),
        region=["road"],
        max_speed_mps=10.0,
        params=ReachabilityParams(horizon_s=4.0, dt_s=0.1),
    )
    context = {"ego2global": np.eye(4), "scene_token": "scene"}
    xyz = np.array(
        [
            [10.0, 0.0, 0.0],  # ahead on the centerline, reachable -> keep
            [-10.0, 0.0, 0.0],  # behind ego -> drop
            [100.0, 0.0, 0.0],  # off the road (x > 50) -> drop
        ]
    )
    assert collision_filter.keep(xyz, context).tolist() == [True, False, False]


def test_collision_filter_curvature_bound_excludes_close_lateral() -> None:
    # At speed the ego cannot swerve far sideways over a short distance, so a point
    # 8 m to the side only 5 m ahead is not reachable within the horizon.
    collision_filter = CollisionFilter(
        _StubProvider(_road_map()),
        region=["road"],
        max_speed_mps=10.0,
        params=ReachabilityParams(horizon_s=4.0, dt_s=0.1),
    )
    context = {"ego2global": np.eye(4), "scene_token": "scene"}
    assert collision_filter.keep(np.array([[5.0, 8.0, 0.0]]), context).tolist() == [False]


def test_collision_filter_required_keys() -> None:
    assert CollisionFilter(_StubProvider(_road_map())).required_eval_keys == (
        "ego2global",
        "scene_token",
    )


def test_corridor_filter_is_a_forward_strip_without_a_map() -> None:
    # The corridor is a fixed-width forward strip in the ego frame with no
    # length bound (distance slicing is the range axis's job). No map, no
    # pose, no context keys.
    corridor_filter = CorridorFilter(width_m=3.0)
    assert corridor_filter.required_eval_keys == ()
    xyz = np.array(
        [
            [10.0, 0.0, 0.0],  # ahead on the centerline -> keep
            [200.0, 1.4, 0.0],  # far ahead, inside the half width -> keep
            [10.0, 2.0, 0.0],  # beyond the half width -> drop
            [-1.0, 0.0, 0.0],  # behind ego -> drop
        ]
    )
    assert corridor_filter.keep(xyz, {}).tolist() == [True, True, False, False]


def test_corridor_filter_keeps_overlapping_box_footprints() -> None:
    # A box counts when the forward part of its footprint overlaps the strip,
    # so a box overhanging the strip edge or straddling the ego x axis is kept.
    corridor_filter = CorridorFilter(width_m=3.0)
    boxes = np.array(
        [
            # [cx, cy, cz, dx, dy, dz, yaw]
            [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],  # centered inside -> keep
            [10.0, 2.2, 0.0, 4.0, 2.0, 1.5, 0.0],  # overhangs the edge -> keep
            [10.0, 4.0, 0.0, 4.0, 2.0, 1.5, 0.0],  # fully beside -> drop
            [-5.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],  # fully behind -> drop
            [-1.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.7853981],  # nose crosses x=0 -> keep
        ]
    )
    assert corridor_filter.keep(boxes, {}).tolist() == [True, True, False, False, True]


def test_filters_accept_tensor_metadata() -> None:
    # Collation + Lightning device transfer deliver the frame metadata as
    # tensors, not numpy arrays, filters must convert (cuda included).
    region_filter = RegionFilter(["road"], _StubProvider(_road_map()))
    keep = region_filter.keep(
        np.array([[10.0, 0.0, 0.0]]),
        {"ego2global": torch.eye(4), "scene_token": "scene"},
    )
    assert keep.tolist() == [True]

    collision_filter = CollisionFilter(_StubProvider(_road_map()), region=["road"])
    keep = collision_filter.keep(
        np.array([[10.0, 0.0, 0.0]]),
        {"ego2global": torch.eye(4), "scene_token": "scene"},
    )
    assert keep.tolist() == [True]


def test_region_filter_margin_erodes_outer_border_only() -> None:
    # Road spans y in [-10, 0], walkway y in [0, 3], adjacent regions. The margin
    # erodes only the OUTER border of their union: a road point near the shared
    # road-walkway border survives, one near the true outer border is dropped.
    road = Polygon([(0, -10), (50, -10), (50, 0), (0, 0)])
    walkway = Polygon([(0, 0), (50, 0), (50, 3), (0, 3)])
    lanelet_map = LaneletMap({"road": [road], "walkway": [walkway]})
    context = {"ego2global": np.eye(4), "scene_token": "scene"}

    eroded = RegionFilter(["road"], _StubProvider(lanelet_map), margin=-0.5)
    near_internal = np.array([[10.0, -0.1, 0.0]])  # 0.1 m from the road-walkway border
    near_outer = np.array([[10.0, -9.8, 0.0]])  # 0.2 m from the outer border
    assert eroded.keep(near_internal, context).tolist() == [True]
    assert eroded.keep(near_outer, context).tolist() == [False]
    assert eroded.name == "region_road_minus0p5"


def test_region_filter_positive_margin_grows_into_off_map_space_only() -> None:
    # Same adjacent road+walkway map. Expanding the road outward claims off-map
    # points within the margin of the road, but never points that belong to
    # another mapped region (the walkway strip stays the walkway's).
    road = Polygon([(0, -10), (50, -10), (50, 0), (0, 0)])
    walkway = Polygon([(0, 0), (50, 0), (50, 3), (0, 3)])
    lanelet_map = LaneletMap({"road": [road], "walkway": [walkway]})
    context = {"ego2global": np.eye(4), "scene_token": "scene"}

    expanded = RegionFilter(["road"], _StubProvider(lanelet_map), margin=0.5)
    off_map_near_road = np.array([[10.0, -10.3, 0.0]])  # 0.3 m below the road's outer border
    on_walkway = np.array([[10.0, 0.2, 0.0]])  # inside the adjacent walkway
    far_off_map = np.array([[10.0, -11.0, 0.0]])  # 1.0 m below, beyond the margin
    assert expanded.keep(off_map_near_road, context).tolist() == [True]
    assert expanded.keep(on_walkway, context).tolist() == [False]
    assert expanded.keep(far_off_map, context).tolist() == [False]
    assert expanded.name == "region_road_plus0p5"


def test_region_filter_name_and_cache_key() -> None:
    region_filter = RegionFilter(["road", "crosswalk"], _StubProvider(_road_map()))
    assert region_filter.name == "region_road_crosswalk"
    assert region_filter.required_eval_keys == ("ego2global", "scene_token")


def test_region_filter_rejects_unknown_tokens() -> None:
    # A typo (e.g. 'sidewalk', the lanelet2 name is 'walkway') fails loud at
    # construction, not silently as an empty slice.
    with pytest.raises(ValueError, match="Unknown lanelet2 region tokens"):
        RegionFilter(["sidewalk"], _StubProvider(_road_map()))


def test_region_absent_from_scene_is_empty_not_an_error() -> None:
    # 'walkway' is a valid token but this scene's map has none: the slice is
    # legitimately empty (membership false everywhere), the eval keeps running.
    walkway_filter = RegionFilter(["walkway"], _StubProvider(_road_map()), margin=0.2)
    keep = walkway_filter.keep(
        np.array([[10.0, 0.0, 0.0]]), {"ego2global": np.eye(4), "scene_token": "scene"}
    )
    assert keep.tolist() == [False]


def test_confusion_suite_region_filter_buckets() -> None:
    # A correct point on the road (x=10) and a wrong point off it (x=100): the
    # whole-scene accuracy is 0.5, the road slice sees only the correct point.
    region = RegionFilter(["road"], _StubProvider(_road_map()))
    suite = Segmentation3DConfusionMatrixMetricSuite(
        components=[
            Accuracy(stages=["test"]),
            Accuracy(stages=["test"], filter=region),
        ],
        num_classes=2,
        ranges=(),
    )
    suite.update(
        {
            "seg_frames": [
                {
                    "pred": torch.tensor([0, 1]),
                    "target": torch.tensor([0, 0]),
                    "coord": torch.tensor([[10.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
                    "ego2global": np.eye(4),
                    "scene_token": "scene",
                }
            ]
        }
    )
    report = suite.result(EvalStage.TEST)
    assert report["acc"] == pytest.approx(0.5)
    assert report["region_road/acc"] == pytest.approx(1.0)


def _box(x: float) -> list[float]:
    return [x, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0]


def test_detection_suite_region_filter_restricts_gt() -> None:
    # Two GT cars, one on the road (x=10), one off-road (x=100). One prediction
    # matches the road car. The whole-scene metric sees 2 GT, the road slice 1.
    region = RegionFilter(["road"], _StubProvider(_road_map()))
    suite = Detection3DMetricSuite(
        components=[
            MeanAP(stages=["test"]),
            MeanAP(stages=["test"], filter=region),
        ],
        class_names=("car",),
        ranges=(),
    )
    gt = torch.tensor([_box(10.0), _box(100.0)])
    suite.update(
        {
            "predictions": [
                {
                    "bboxes_3d": torch.tensor([_box(10.0)]),
                    "scores_3d": torch.tensor([0.9]),
                    "labels_3d": torch.tensor([0]),
                }
            ],
            "gt_boxes": [gt],
            "gt_labels": [torch.tensor([0, 0])],
            "ego2global": [np.eye(4)],
            "scene_token": ["scene"],
        }
    )
    report = suite.result(EvalStage.TEST)
    assert report["gt_count_car"] == pytest.approx(2.0)
    assert report["region_road/gt_count_car"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Scenes without a lanelet map: excluded from filtered slices, kept whole-scene
# --------------------------------------------------------------------------- #


def test_region_filter_available_delegates_to_provider() -> None:
    provider = _StubProvider(_road_map(), no_map=("s_nomap",))
    region_filter = RegionFilter(["road"], provider)
    assert region_filter.available({"scene_token": "s_map"}) is True
    assert region_filter.available({"scene_token": "s_nomap"}) is False


def test_lanelet_resolver_available_and_call_are_distinct(tmp_path) -> None:
    # available() is a pure existence check, __call__ still fails loud when absent.
    resolver = T4LaneletMapResolver(str(tmp_path))
    scene = "db/uuid/0"
    assert resolver.available(scene) is False
    with pytest.raises(FileNotFoundError):
        resolver(scene)
    map_path = tmp_path / scene / "map" / "lanelet2_map.osm"
    map_path.parent.mkdir(parents=True)
    map_path.write_text("<osm></osm>")
    assert resolver.available(scene) is True
    assert resolver(scene) == str(map_path)


def test_map_provider_available_delegates_to_resolver() -> None:
    class _Resolver:
        def available(self, scene_token: object) -> bool:
            return scene_token == "yes"

    provider = LaneletMapProvider(_Resolver())
    assert provider.available("yes") is True
    assert provider.available("no") is False


def test_confusion_suite_excludes_scene_without_lanelet_map() -> None:
    # Two frames: 's_map' has a lanelet map, 's_nomap' does not. Whole-scene
    # accuracy sees every point, the road slice drops the map-less scene entirely
    # (never loads its map, the stub would raise), and the coverage counters
    # record 2 frames seen, 1 covered.
    region = RegionFilter(["road"], _StubProvider(_road_map(), no_map=("s_nomap",)))
    suite = Segmentation3DConfusionMatrixMetricSuite(
        components=[Accuracy(stages=["test"]), Accuracy(stages=["test"], filter=region)],
        num_classes=2,
        ranges=(),
    )
    suite.update(
        {
            "seg_frames": [
                {
                    "pred": torch.tensor([0, 1]),
                    "target": torch.tensor([0, 0]),
                    "coord": torch.tensor([[10.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
                    "ego2global": np.eye(4),
                    "scene_token": "s_map",
                },
                {
                    "pred": torch.tensor([1]),
                    "target": torch.tensor([0]),
                    "coord": torch.tensor([[10.0, 0.0, 0.0]]),
                    "ego2global": np.eye(4),
                    "scene_token": "s_nomap",
                },
            ]
        }
    )
    report = suite.result(EvalStage.TEST)
    assert report["acc"] == pytest.approx(1.0 / 3.0)  # 1 of 3 points correct, all scenes
    assert report["region_road/acc"] == pytest.approx(1.0)  # only s_map's road point
    assert suite.region_frames_seen.tolist() == [2]
    assert suite.region_frames_covered.tolist() == [1]


def test_detection_suite_excludes_scene_without_lanelet_map() -> None:
    # Two frames, one GT car on the road each, 's_nomap' has no map. Whole-scene
    # sees both GT, the road slice keeps only the mapped scene's, and coverage
    # records 2 seen / 1 covered.
    region = RegionFilter(["road"], _StubProvider(_road_map(), no_map=("s_nomap",)))
    suite = Detection3DMetricSuite(
        components=[MeanAP(stages=["test"]), MeanAP(stages=["test"], filter=region)],
        class_names=("car",),
        ranges=(),
    )
    suite.update(
        {
            "predictions": [
                {
                    "bboxes_3d": torch.tensor([_box(10.0)]),
                    "scores_3d": torch.tensor([0.9]),
                    "labels_3d": torch.tensor([0]),
                },
                {
                    "bboxes_3d": torch.zeros((0, 9)),
                    "scores_3d": torch.zeros((0,)),
                    "labels_3d": torch.zeros((0,), dtype=torch.long),
                },
            ],
            "gt_boxes": [torch.tensor([_box(10.0)]), torch.tensor([_box(10.0)])],
            "gt_labels": [torch.tensor([0]), torch.tensor([0])],
            "ego2global": [np.eye(4), np.eye(4)],
            "scene_token": ["s_map", "s_nomap"],
        }
    )
    report = suite.result(EvalStage.TEST)
    assert report["gt_count_car"] == pytest.approx(2.0)
    assert report["region_road/gt_count_car"] == pytest.approx(1.0)
    assert suite.region_frames_seen.tolist() == [2]
    assert suite.region_frames_covered.tolist() == [1]
    # The uncovered frame is omitted from the filtered state entirely, so
    # per-frame denominators only count covered frames.
    assert len(suite.state_for(None, region).samples) == 1
    assert len(suite.state_for(None).samples) == 2


def test_point_cloud_suite_excludes_scene_without_lanelet_map() -> None:
    # 's_nomap' (one wrong point) is excluded from the road slice, the whole-scene
    # error rate still counts it. Coverage records 2 frames seen, 1 covered.
    region = RegionFilter(["road"], _StubProvider(_road_map(), no_map=("s_nomap",)))
    suite = Segmentation3DPointCloudMetricSuite(
        components=[ErrorClusters(stages=["test"]), ErrorClusters(stages=["test"], filter=region)],
        num_classes=2,
        ranges=(),
    )
    suite.update(
        {
            "seg_frames": [
                {
                    "coord": torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
                    "pred": torch.tensor([0, 0]),
                    "target": torch.tensor([0, 0]),
                    "scores": torch.full((2, 2), 0.5),
                    "ego2global": np.eye(4),
                    "scene_token": "s_map",
                },
                {
                    "coord": torch.tensor([[10.0, 0.0, 0.0]]),
                    "pred": torch.tensor([1]),
                    "target": torch.tensor([0]),
                    "scores": torch.full((1, 2), 0.5),
                    "ego2global": np.eye(4),
                    "scene_token": "s_nomap",
                },
            ]
        }
    )
    report = suite.result(EvalStage.TEST)
    assert report["error_rate"] == pytest.approx(1.0 / 3.0)  # 1 wrong of 3, all scenes
    assert report["region_road/error_rate"] == pytest.approx(0.0)  # only s_map, clean
    assert suite.region_frames_seen.tolist() == [2]
    assert suite.region_frames_covered.tolist() == [1]


def test_point_cloud_suite_region_filter_clips_boxes() -> None:
    # A GT box off the road region is omitted from the filtered state, box
    # membership is resolved through the same filter as the points.
    region = RegionFilter(["road"], _StubProvider(_road_map()))
    suite = Segmentation3DPointCloudMetricSuite(
        components=[
            ErrorClusters(stages=["test"]),
            ErrorClusters(stages=["test"], filter=region),
        ],
        num_classes=2,
        ranges=(),
    )
    suite.update(
        {
            "seg_frames": [
                {
                    "coord": torch.tensor([[10.0, 0.0, 0.0]]),
                    "pred": torch.tensor([0]),
                    "target": torch.tensor([0]),
                    "scores": torch.full((1, 2), 0.5),
                    "ego2global": np.eye(4),
                    "scene_token": "scene",
                    "gt_boxes": torch.tensor([_box(10.0), _box(100.0)]),
                    "gt_box_labels": torch.tensor([0, 0]),
                }
            ]
        }
    )
    assert suite.state_for(None, region).frames[0].gt_boxes.shape[0] == 1
    assert suite.state_for(None).frames[0].gt_boxes.shape[0] == 2


_MINI_OSM = """<?xml version='1.0'?>
<osm>
  <node id='1'><tag k='local_x' v='0'/><tag k='local_y' v='0'/></node>
  <node id='2'><tag k='local_x' v='10'/><tag k='local_y' v='0'/></node>
  <node id='3'><tag k='local_x' v='0'/><tag k='local_y' v='4'/></node>
  <node id='4'><tag k='local_x' v='10'/><tag k='local_y' v='4'/></node>
  <way id='100'><nd ref='1'/><nd ref='2'/></way>
  <way id='101'><nd ref='3'/><nd ref='4'/></way>
  <relation id='200'>
    <member type='way' ref='101' role='left'/>
    <member type='way' ref='100' role='right'/>
    <tag k='type' v='lanelet'/><tag k='subtype' v='road'/><tag k='speed_limit' v='36'/>
  </relation>
</osm>
"""


def test_load_lanelet_speeds_parses_km_h_to_mps(tmp_path) -> None:
    path = tmp_path / "map.osm"
    path.write_text(_MINI_OSM)
    speeds = load_lanelet_speeds(str(path))
    assert len(speeds) == 1
    polygon, speed = speeds[0]
    assert speed == pytest.approx(10.0)  # 36 km/h -> 10 m/s
    assert polygon.contains(Point(5.0, 2.0))


def test_lanelet_map_speed_at_lookup(tmp_path) -> None:
    path = tmp_path / "map.osm"
    path.write_text(_MINI_OSM)
    lanelet_map = LaneletMap(load_region_polygons(str(path)), load_lanelet_speeds(str(path)))
    assert lanelet_map.speed_at(5.0, 2.0, default=99.0) == pytest.approx(10.0)  # inside the lane
    assert lanelet_map.speed_at(50.0, 50.0, default=99.0) == pytest.approx(99.0)  # outside -> default


def test_region_filter_eroded_margin_tests_the_surviving_region_part() -> None:
    # Road x in [0, 10] and walkway x in [10, 20] share the internal border x=10;
    # the union spans y in [0, 10] over the road and y in [-5, 10] over the walkway.
    lanelet_map = LaneletMap(
        {
            "road": [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
            "walkway": [Polygon([(10, -5), (20, -5), (20, 10), (10, 10)])],
        }
    )
    # Overlaps the road only inside the 2 m outer band (y < 2 there) while also
    # reaching the walkway's eroded interior, so testing the region and the eroded
    # surface separately would wrongly keep it in the road slice.
    band_straddler = Polygon([(8, 0.5), (13, 0.5), (13, 1.5), (8, 1.5)])
    # Control: genuinely overlaps the road's surviving part.
    inside = Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])

    mask = lanelet_map.intersects(("road",), [band_straddler, inside], margin=-2.0)

    assert mask.tolist() == [False, True]
