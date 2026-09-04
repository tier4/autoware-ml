"""Concrete evaluation filters.

The base :class:`~autoware_ml.metrics.base.MetricFilter` / ``IdentityFilter`` live
in ``base.py`` (no heavy deps). Three filters live here. ``CorridorFilter`` keeps
the elements inside a straight fixed strip ahead of ego and needs no map at all.
``RegionFilter`` keeps the elements whose base_link position, transformed to the
map frame by the per-frame ego pose, falls inside a chosen set of lanelet2
regions, so any metric can be reported on the road or on the walkway.
``CollisionFilter`` keeps the elements inside the ego collision area clipped to
the road lanelets, the filter form of the collision model.
"""

from __future__ import annotations

from math import atan2
from typing import Any

import numpy as np
import shapely
import torch
from shapely.geometry import Polygon

from autoware_ml.metrics.base import MetricFilter, number_token
from autoware_ml.metrics.detection3d.geometry import bev_corners
from autoware_ml.metrics.geometry.lanelet import KNOWN_REGION_TOKENS, LaneletMapProvider
from autoware_ml.metrics.geometry.reachability import (
    Agent,
    ReachabilityParams,
    wheeled_reachable_region,
)


def as_numpy(value: Any) -> np.ndarray:
    """Frame metadata as a NumPy array, from any device.

    Collation and Lightning's device transfer turn per-frame metadata (the ego
    pose) into tensors that live on the evaluation device, and ``np.asarray``
    alone cannot read a CUDA tensor.

    Args:
        value: Array-like or tensor, possibly on a CUDA device.

    Returns:
        A NumPy array on the CPU.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_map_xy(xyz: np.ndarray, ego2global: Any) -> np.ndarray:
    """Transform base_link ``xyz`` (N, 3+) to map-frame xy via ``ego2global``."""
    transform = as_numpy(ego2global).astype(np.float64)
    homogeneous = np.concatenate([xyz[:, :3], np.ones((xyz.shape[0], 1))], axis=1)
    return (homogeneous @ transform.T)[:, :2]


def ego2global_pose(ego2global: Any) -> tuple[float, float, float]:
    """Ego map-frame ``(x, y, heading)`` from a 4x4 ego-to-global transform.

    Args:
        ego2global: 4x4 ego-to-global transform.

    Returns:
        The map-frame ``(x, y, heading)`` triple.
    """
    matrix = as_numpy(ego2global).astype(np.float64)
    return float(matrix[0, 3]), float(matrix[1, 3]), atan2(float(matrix[1, 0]), float(matrix[0, 0]))


def box_footprints_map(boxes: np.ndarray, ego2global: Any) -> list:
    """Map-frame BEV footprint polygons for detection boxes.

    ``boxes`` rows are ``[cx, cy, cz, dx, dy, dz, yaw, ...]`` in base_link. Each
    footprint's four corners are transformed to the map frame so region
    membership can be tested by overlap rather than by the box center alone.

    Args:
        boxes: Box rows ``(N, 7+)`` in base_link.
        ego2global: 4x4 ego-to-global transform of the frame.

    Returns:
        One shapely footprint polygon per box, in the map frame.
    """
    footprints = []
    for box in boxes:
        corners = bev_corners(box)  # (4, 2) base_link
        corners3 = np.column_stack([corners, np.full(corners.shape[0], box[2])])
        footprints.append(Polygon(_to_map_xy(corners3, ego2global)))
    return footprints


def _is_boxes(elements: np.ndarray) -> bool:
    """Elements are detection boxes ``[cx,cy,cz,dx,dy,dz,yaw,...]`` rather than bare points."""
    return elements.ndim == 2 and elements.shape[1] >= 7


def ego_collision_agent(
    lanelet_map, ego2global: Any, max_speed_mps: float, length: float, width: float
) -> Agent:
    """Ego as a wheeled agent of the collision model, one recipe for every consumer.

    The collision area trim (:class:`CollisionFilter`) and the collision TTC
    provider build ego through this single function, so the two can never
    drift: the pose comes from ``ego2global`` and the speed is the lanelet
    speed limit at ego's position with ``max_speed_mps`` as the off-map
    fallback.

    Args:
        lanelet_map: The scene's parsed lanelet map.
        ego2global: 4x4 ego-to-global transform.
        max_speed_mps: Off-map fallback speed in m/s.
        length: Ego body length in meters.
        width: Ego body width in meters.

    Returns:
        Ego as a wheeled agent in the map frame.
    """
    ego_x, ego_y, ego_heading = ego2global_pose(ego2global)
    speed = lanelet_map.speed_at(ego_x, ego_y, max_speed_mps)
    return Agent.wheeled(ego_x, ego_y, ego_heading, speed, length, width)


def signed_number_token(value: float) -> str:
    """Key-safe token that always spells the sign, e.g. ``plus0p2`` or ``minus0p2``.

    Args:
        value: Signed number to encode.

    Returns:
        The token, sign word first.
    """
    return ("minus" if value < 0.0 else "plus") + number_token(abs(value))


class RegionFilter(MetricFilter):
    """Keep elements inside a union of lanelet2 regions.

    Points near the outer border of the whole mapped surface are naturally
    noisy (high entropy, frequent misclassification), so the border can be
    moved by a signed margin, negative eroding it and positive dilating it.
    """

    required_eval_keys = ("ego2global", "scene_token")

    def __init__(
        self,
        region: list[str] | tuple[str, ...],
        map_provider: LaneletMapProvider,
        margin: float = 0.0,
        name: str | None = None,
    ) -> None:
        """Validate the region tokens and derive the display name.

        Args:
            region: Literal lanelet2 tokens (lanelet ``subtype`` or area
                ``type``), e.g. ``[road, road_shoulder, crosswalk]`` for the
                drivable region or ``[walkway]`` for the pedestrian-only region.
            map_provider: Resolves a scene token to its :class:`LaneletMap`.
            margin: Border shift in meters. Negative erodes, so outer-border
                points stop counting while internal borders between adjacent
                regions stay intact. Positive dilates, so the region
                additionally claims off-map points within the margin, never
                points of another mapped region.
            name: Display name prefixing the metric keys. Derived from the
                tokens and margin when omitted.
        """
        if not region:
            raise ValueError("RegionFilter needs at least one lanelet2 region token.")
        unknown = sorted(set(str(token) for token in region) - KNOWN_REGION_TOKENS)
        if unknown:
            raise ValueError(
                f"Unknown lanelet2 region tokens {unknown}, known tokens: "
                f"{sorted(KNOWN_REGION_TOKENS)}. (A known token absent from a "
                "particular scene's map is fine, that slice is simply empty.)"
            )
        self.region = tuple(str(token) for token in region)
        self.map_provider = map_provider
        self.margin = float(margin)
        if name is not None:
            self._name = name
        else:
            margin_token = f"_{signed_number_token(self.margin)}" if self.margin else ""
            self._name = "region_" + "_".join(self.region) + margin_token

    @property
    def name(self) -> str:
        """Display name prefixing this filter's metric keys."""
        return self._name

    @property
    def cache_key(self) -> str:
        """Every parameter that shapes the mask, equal keys must mean equal masks."""
        return "region:" + ",".join(sorted(self.region)) + f":{self.margin:g}"

    def keep(self, xyz: np.ndarray, context: dict[str, Any]) -> np.ndarray:
        """Mask of elements in the region.

        Detection boxes (7 or more columns) belong to the region when their
        footprint overlaps it (any part inside), so an object overhanging from
        an off-region center still counts. Segmentation points (3 columns) use
        point-in-polygon.

        Args:
            xyz: Points ``(N, 3)`` or box rows ``(N, 7+)`` in base_link.
            context: Per-frame values with the ego pose and scene token.

        Returns:
            Boolean mask of elements in the region.
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        lanelet_map = self.map_provider.get(context["scene_token"])
        if _is_boxes(xyz):
            footprints = box_footprints_map(xyz, context["ego2global"])
            return lanelet_map.intersects(self.region, footprints, self.margin)
        map_xy = _to_map_xy(xyz, context["ego2global"])
        return lanelet_map.contains(self.region, map_xy, self.margin)

    def available(self, context: dict[str, Any]) -> bool:
        """False for scenes with no lanelet map, so the suite excludes them.

        Args:
            context: Per-frame values with the scene token.

        Returns:
            Whether the scene has a lanelet map.
        """
        return self.map_provider.available(context["scene_token"])


def _clip_forward(corners: np.ndarray) -> np.ndarray:
    """Clip a convex BEV polygon to the forward half-plane ``x >= 0``."""
    clipped: list[np.ndarray] = []
    count = corners.shape[0]
    for index in range(count):
        current = corners[index]
        following = corners[(index + 1) % count]
        if current[0] >= 0.0:
            clipped.append(current)
        if (current[0] >= 0.0) != (following[0] >= 0.0):
            t = current[0] / (current[0] - following[0])
            clipped.append(current + t * (following - current))
    return np.asarray(clipped) if clipped else np.zeros((0, 2))


class CorridorFilter(MetricFilter):
    """Keep elements inside a straight corridor ahead of ego.

    The corridor is a forward strip in the ego frame: ``width_m`` across,
    centered on the x axis, with no length bound of its own because distance
    slicing is the range axis's job. It needs no map and no pose, so the slice
    covers every scene. A detection box is kept when its footprint overlaps
    the strip, a segmentation point when it lies inside it.
    """

    required_eval_keys = ()

    def __init__(self, width_m: float = 3.0, name: str = "corridor") -> None:
        """Validate the strip width.

        Args:
            width_m: Full corridor width in meters, centered on the x axis.
            name: Display name prefixing the metric keys.
        """
        if width_m <= 0.0:
            raise ValueError("width_m must be > 0.")
        self.width_m = float(width_m)
        self._name = str(name)

    @property
    def name(self) -> str:
        """Display name prefixing this filter's metric keys."""
        return self._name

    @property
    def cache_key(self) -> str:
        """Every parameter that shapes the mask, equal keys must mean equal masks."""
        return f"corridor:straight:w{self.width_m:g}"

    def keep(self, xyz: np.ndarray, context: dict[str, Any]) -> np.ndarray:
        """Mask of elements inside the strip (footprint overlap / point test).

        Args:
            xyz: Points ``(N, 3)`` or box rows ``(N, 7+)`` in base_link.
            context: Per-frame values with the ego pose and scene token.

        Returns:
            Boolean mask of elements inside the strip.
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        half_width = self.width_m / 2.0
        if _is_boxes(xyz):
            keep = np.zeros(xyz.shape[0], dtype=bool)
            for index, row in enumerate(xyz):
                forward = _clip_forward(bev_corners(row))
                keep[index] = (
                    forward.shape[0] > 0
                    and float(forward[:, 1].min()) <= half_width
                    and float(forward[:, 1].max()) >= -half_width
                )
            return keep
        return (xyz[:, 0] >= 0.0) & (np.abs(xyz[:, 1]) <= half_width)


class CollisionFilter(MetricFilter):
    """Keep elements in the ego's collision area, the filter form of the collision model.

    The collision area is everything ego could collide with within the horizon
    at the max map-legal speed under bounded steering, clipped to the road
    lanelets so it follows the road on bends. A detection box is kept when its
    footprint meets that region, a segmentation point when it lies inside it.
    Planner-independent (never the driven path), and the same collision model
    the criticality metrics use, so any other metric can be reported
    in-path too.
    """

    required_eval_keys = ("ego2global", "scene_token")

    def __init__(
        self,
        map_provider: LaneletMapProvider,
        region: list[str] | tuple[str, ...] = ("road", "road_shoulder", "crosswalk"),
        params: ReachabilityParams | None = None,
        max_speed_mps: float = 16.7,
        ego_length_m: float = 4.9,
        ego_width_m: float = 2.0,
        name: str = "collision",
    ) -> None:
        """Validate the road region tokens and the ego propagation parameters.

        Args:
            map_provider: Resolves a scene token to its :class:`LaneletMap`.
            region: Road lanelet tokens the collision area is clipped to.
            params: Ego propagation parameters, defaults to
                :class:`ReachabilityParams`.
            max_speed_mps: Ego speed fallback where the map has no speed limit.
            ego_length_m: Assumed ego body length in meters.
            ego_width_m: Assumed ego body width in meters.
            name: Display name prefixing the metric keys.
        """
        if not region:
            raise ValueError("CollisionFilter needs at least one road lanelet region token.")
        unknown = sorted(set(str(token) for token in region) - KNOWN_REGION_TOKENS)
        if unknown:
            raise ValueError(
                f"Unknown lanelet2 region tokens {unknown}, known tokens: "
                f"{sorted(KNOWN_REGION_TOKENS)}."
            )
        if max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be > 0.")
        self.map_provider = map_provider
        self.region = tuple(str(token) for token in region)
        self.params = params or ReachabilityParams()
        self.max_speed_mps = float(max_speed_mps)
        self.ego_length_m = float(ego_length_m)
        self.ego_width_m = float(ego_width_m)
        self._name = name
        # The suite calls keep() for GT and predictions of the same frame in
        # sequence, the ego collision area depends only on the frame's pose.
        self._region_memo: tuple[tuple, Any] | None = None

    @property
    def name(self) -> str:
        """Display name prefixing this filter's metric keys."""
        return self._name

    @property
    def cache_key(self) -> str:
        """Every parameter that shapes the mask, equal keys must mean equal masks."""
        region = ",".join(sorted(self.region))
        params = self.params
        return (
            f"collision-area:v{self.max_speed_mps:g}"
            f":b{self.ego_length_m:g}x{self.ego_width_m:g}"
            f":h{params.horizon_s:g},{params.dt_s:g},{params.max_lateral_accel_mps2:g}"
            f",{params.min_radius_m:g},{params.arc_samples}:{region}"
        )

    def _ego_region(self, context: dict[str, Any]):
        """The ego collision area in the map frame for this frame (memoized)."""
        ego_x, ego_y, ego_heading = ego2global_pose(context["ego2global"])
        key = (str(context["scene_token"]), ego_x, ego_y, ego_heading)
        if self._region_memo is not None and self._region_memo[0] == key:
            return self._region_memo[1]
        lanelet_map = self.map_provider.get(context["scene_token"])
        drivable = lanelet_map.region_union(self.region)
        ego = ego_collision_agent(
            lanelet_map,
            context["ego2global"],
            self.max_speed_mps,
            self.ego_length_m,
            self.ego_width_m,
        )
        region = wheeled_reachable_region(ego, self.params, drivable)
        if not region.is_empty:
            shapely.prepare(region)
        self._region_memo = (key, region)
        return region

    def keep(self, xyz: np.ndarray, context: dict[str, Any]) -> np.ndarray:
        """Mask of elements inside the ego collision area (footprint / point test).

        Args:
            xyz: Points ``(N, 3)`` or box rows ``(N, 7+)`` in base_link.
            context: Per-frame values with the ego pose and scene token.

        Returns:
            Boolean mask of elements inside the collision area.
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        region = self._ego_region(context)
        if region.is_empty:
            return np.zeros(xyz.shape[0], dtype=bool)
        if _is_boxes(xyz):
            footprints = box_footprints_map(xyz, context["ego2global"])
            hits = shapely.intersects(region, np.array(footprints, dtype=object))
            return np.asarray(hits, dtype=bool)
        map_xy = _to_map_xy(xyz, context["ego2global"])
        points = shapely.points(map_xy[:, 0], map_xy[:, 1])
        return np.asarray(shapely.contains(region, points), dtype=bool)

    def available(self, context: dict[str, Any]) -> bool:
        """False for scenes with no lanelet map, so the suite excludes them.

        Args:
            context: Per-frame values with the scene token.

        Returns:
            Whether the scene has a lanelet map.
        """
        return self.map_provider.available(context["scene_token"])
