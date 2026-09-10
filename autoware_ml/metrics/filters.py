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

from dataclasses import dataclass, field
from functools import lru_cache
from math import atan2
from typing import Any

import numpy as np
import shapely
import torch

from autoware_ml.metrics.base import MetricFilter, number_token
from autoware_ml.metrics.detection3d.geometry import bev_corners_batch
from autoware_ml.metrics.geometry.lanelet import (
    KNOWN_REGION_TOKENS,
    LaneletMap,
    LaneletMapProvider,
)
from autoware_ml.metrics.geometry.reachability import (
    Agent,
    ReachabilityParams,
    VehicleGeometry,
    localized_surface,
    wheeled_reachable_region,
)


# The lanelet2 regions a vehicle may drive on, shared by every consumer of the
# collision model so the filter and the TTC provider cannot drift apart.
DEFAULT_DRIVABLE_REGION: tuple[str, ...] = ("road", "road_shoulder", "crosswalk")


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


def box_footprints_map(boxes: np.ndarray, ego2global: Any) -> np.ndarray:
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
    corners = bev_corners_batch(boxes)  # (N, 4, 2) base_link
    heights = np.repeat(boxes[:, 2], corners.shape[1])
    corners3 = np.column_stack([corners.reshape(-1, 2), heights])
    map_xy = _to_map_xy(corners3, ego2global).reshape(corners.shape)
    return shapely.polygons(map_xy)


def _is_boxes(elements: np.ndarray) -> bool:
    """Elements are detection boxes ``[cx,cy,cz,dx,dy,dz,yaw,...]`` rather than bare points."""
    return elements.ndim == 2 and elements.shape[1] >= 7


def ego_collision_agent(
    lanelet_map: LaneletMap,
    x: float,
    y: float,
    heading: float,
    max_speed_mps: float,
    vehicle: VehicleGeometry,
) -> Agent:
    """Builds a wheeled agent representing the ego for use in collision
    calculations.

    The collision area trim (:class:`CollisionFilter`) and the collision TTC
    provider build ego through this single function, so the two can never drift:
    the speed is the lanelet speed limit at ego's position with ``max_speed_mps``
    as the off-map fallback. Callers hold the ego pose in different forms, so
    they read it out of their transform with :func:`ego2global_pose` first.

    Args:
        lanelet_map: The scene's parsed lanelet map.
        x: Ego map-frame x in meters.
        y: Ego map-frame y in meters.
        heading: Ego map-frame heading in radians.
        max_speed_mps: Off-map fallback speed in m/s.
        vehicle: Ego body dimensions.

    Returns:
        Ego as a wheeled agent in the map frame.
    """
    return Agent.wheeled(
        x,
        y,
        heading,
        lanelet_map.speed_at(x, y, max_speed_mps),
        front=vehicle.front,
        rear=vehicle.rear,
        width=vehicle.width,
        min_turn_radius=vehicle.min_turn_radius,
    )


def signed_number_token(value: float) -> str:
    """Key-safe token spelling the sign, e.g. ``plus0p2`` or ``minus0p2``.

    Zero has no direction, so it carries no sign word.

    Args:
        value: Signed number to encode.

    Returns:
        The token, sign word first for a non-zero value.
    """
    if value == 0.0:
        return number_token(0.0)
    return ("minus" if value < 0.0 else "plus") + number_token(abs(value))


def validated_region(region: list[str] | tuple[str, ...], owner: str) -> tuple[str, ...]:
    """The region tokens as a tuple, rejecting anything lanelet2 does not name.

    A typo would otherwise surface as a silently empty slice, so both map-backed
    filters validate through here.

    Args:
        region: Literal lanelet2 tokens.
        owner: Name of the calling filter, for the error message.

    Returns:
        The tokens as a tuple of strings.

    Raises:
        ValueError: If ``region`` is empty or names an unknown token.
    """
    if not region:
        raise ValueError(f"{owner} needs at least one lanelet2 region token.")
    tokens = tuple(str(token) for token in region)
    unknown = sorted(set(tokens) - KNOWN_REGION_TOKENS)
    if unknown:
        raise ValueError(
            f"Unknown lanelet2 region tokens {unknown}, known tokens: "
            f"{sorted(KNOWN_REGION_TOKENS)}. (A known token absent from a "
            "particular scene's map is fine, that slice is simply empty.)"
        )
    return tokens


@dataclass(frozen=True)
class RegionFilter(MetricFilter):
    """Keep only elements inside a union of lanelet2 regions.

    Points near the outer border of the whole mapped surface are naturally
    noisy (high entropy, frequent misclassification), so the border can be
    moved by a signed margin, negative eroding it and positive dilating it.

    Attributes:
        region: List of literal lanelet2 tokens (lanelet ``subtype`` or area
            ``type``), e.g. ``[road, road_shoulder, crosswalk]`` for the
            drivable region or ``[walkway]`` for the pedestrian-only region.
        map_provider: Resolves a scene token to its :class:`LaneletMap`.
        margin: Border shift in meters. Negative erodes, so outer-border points
            stop counting while internal borders between adjacent regions stay
            intact. Positive dilates, so the region additionally claims off-map
            points within the margin, never points of another mapped region.
        name: Display name prefixing the metric keys. Derived from the tokens
            and margin when left empty.
    """

    region: tuple[str, ...]
    map_provider: LaneletMapProvider
    margin: float = 0.0
    name: str = ""

    required_eval_keys = ("ego2global", "scene_token")

    def __post_init__(self) -> None:
        """Validate the region tokens and derive the display name."""
        object.__setattr__(self, "region", validated_region(self.region, "RegionFilter"))
        object.__setattr__(self, "margin", float(self.margin))
        if not self.name:
            margin_token = f"_{signed_number_token(self.margin)}" if self.margin else ""
            object.__setattr__(self, "name", "region_" + "_".join(self.region) + margin_token)

    @property
    def cache_key(self) -> str:
        """Every parameter that shapes the mask, equal keys must mean equal masks.

        The provider is part of that: a suite keeps one mask set per key, so two
        filters that read different maps must not share it.
        """
        region = ",".join(sorted(self.region))
        return f"region:{region}:{self.margin:g}:{self.map_provider.cache_key}"

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


@dataclass(frozen=True)
class CorridorFilter(MetricFilter):
    """Keep only elements inside a straight corridor ahead of ego.

    The corridor is a forward strip in the ego frame: ``width_m`` across,
    centered on the x axis, with no length bound of its own because distance
    slicing is the range axis's job. It needs no map and no pose, so the slice
    covers every scene. A detection box is kept when its footprint overlaps
    the strip, a segmentation point when it lies inside it.

    Attributes:
        width_m: Full corridor width in meters, centered on the x axis.
        name: Display name prefixing the metric keys.
    """

    width_m: float = 3.0
    name: str = "corridor"

    required_eval_keys = ()

    def __post_init__(self) -> None:
        """Validate the strip width."""
        if self.width_m <= 0.0:
            raise ValueError("width_m must be > 0.")
        object.__setattr__(self, "width_m", float(self.width_m))

    @property
    def cache_key(self) -> str:
        """Every parameter that shapes the mask, equal keys must mean equal masks."""
        return f"corridor:w{self.width_m:g}"

    def keep(self, xyz: np.ndarray, context: dict[str, Any]) -> np.ndarray:
        """Mask of elements inside the strip (footprint overlap / point test).

        Args:
            xyz: Points ``(N, 3)`` or box rows ``(N, 7+)`` in base_link.
            context: Unused, the strip lives in the ego frame.

        Returns:
            Boolean mask of elements inside the strip.
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        half_width = self.width_m / 2.0
        if _is_boxes(xyz):
            corners = bev_corners_batch(xyz)
            # The strip has no length bound, and no geometry library has an
            # unbounded polygon: truncating it at the batch's own farthest corner
            # is the same test for every box in the batch.
            far = float(corners[:, :, 0].max())
            if far <= 0.0:
                return np.zeros(xyz.shape[0], dtype=bool)
            strip = shapely.box(0.0, -half_width, far, half_width)
            footprints = shapely.polygons(corners)
            return np.asarray(shapely.intersects(strip, footprints), dtype=bool)
        return (xyz[:, 0] >= 0.0) & (np.abs(xyz[:, 1]) <= half_width)


@dataclass(frozen=True)
class CollisionFilter(MetricFilter):
    """Keep only elements in the ego's collision area, the filter form of the collision model.

    The collision area is everything ego could collide with within the horizon
    at the max map-legal speed under bounded steering, clipped to the road
    lanelets so it follows the road on bends. A detection box is kept when its
    footprint meets that region, a segmentation point when it lies inside it.
    Planner-independent (never the driven path), and the same collision model
    the criticality metrics use, so any other metric can be reported
    in-path too.

    The ego body has no default: it is a property of the platform, and a vehicle
    silently evaluated with another vehicle's size gives a plausible looking
    area that is simply wrong.

    Attributes:
        map_provider: Resolves a scene token to its :class:`LaneletMap`.
        vehicle: Ego body dimensions, measured from the rear axle the pose refers to.
        region: Road lanelet tokens the collision area is clipped to.
        params: Ego propagation parameters.
        max_speed_mps: Ego speed fallback where the map has no speed limit.
        name: Display name prefixing the metric keys.
    """

    map_provider: LaneletMapProvider
    vehicle: VehicleGeometry = field(kw_only=True)
    region: tuple[str, ...] = field(default=DEFAULT_DRIVABLE_REGION, kw_only=True)
    params: ReachabilityParams = field(default_factory=ReachabilityParams, kw_only=True)
    max_speed_mps: float = field(default=16.7, kw_only=True)
    name: str = field(default="collision", kw_only=True)

    required_eval_keys = ("ego2global", "scene_token")

    def __post_init__(self) -> None:
        """Validate the road region tokens and the ego speed fallback."""
        object.__setattr__(self, "region", validated_region(self.region, "CollisionFilter"))
        if self.max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be > 0.")
        object.__setattr__(self, "max_speed_mps", float(self.max_speed_mps))

    @property
    def cache_key(self) -> str:
        """Every parameter that shapes the mask, equal keys must mean equal masks."""
        region = ",".join(sorted(self.region))
        params = self.params
        return (
            f"collision-area:v{self.max_speed_mps:g}"
            f":b{self.vehicle.front:g}+{self.vehicle.rear:g}x{self.vehicle.width:g}"
            f"r{self.vehicle.min_turn_radius:g}"
            f":h{params.horizon_s:g},{params.dt_s:g},{params.max_lateral_accel_mps2:g}"
            f",{params.min_radius_m:g},{params.arc_samples}:{region}"
            f":{self.map_provider.cache_key}"
        )

    @lru_cache(maxsize=1)
    def _region_for_pose(self, scene_token: str, x: float, y: float, heading: float):
        """The ego collision area in the map frame for one ego pose.

        The suite calls :meth:`keep` for the ground truth and the predictions of
        the same frame in sequence and the area depends only on the pose, so one
        cached entry serves both calls.
        """
        lanelet_map = self.map_provider.get(scene_token)
        ego = ego_collision_agent(lanelet_map, x, y, heading, self.max_speed_mps, self.vehicle)
        # The scene's whole drivable union is orders of magnitude larger than one
        # ego pose can touch, and sealing it dominates the cost, so clip first.
        drivable = localized_surface(ego, self.params, lanelet_map.region_union(self.region))
        region = wheeled_reachable_region(ego, self.params, drivable)
        if not region.is_empty:
            shapely.prepare(region)
        return region

    def _ego_region(self, context: dict[str, Any]):
        """The ego collision area in the map frame for this frame."""
        return self._region_for_pose(
            str(context["scene_token"]), *ego2global_pose(context["ego2global"])
        )

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
            return np.asarray(shapely.intersects(region, footprints), dtype=bool)
        map_xy = _to_map_xy(xyz, context["ego2global"])
        points = shapely.points(map_xy[:, 0], map_xy[:, 1])
        return np.asarray(shapely.contains(region, points), dtype=bool)

    def available(self, context: dict[str, Any]) -> bool:
        """False when the frame has no collision area, so the suite excludes it.

        A scene without a lanelet map has none, and neither has a frame whose ego
        pose sits off the drivable surface (an inaccurate map or localization).
        Both would otherwise enter the slice with an empty in-path set, where a
        missed object can never be counted because nothing is in path at all.

        Args:
            context: Per-frame values with the ego pose and scene token.

        Returns:
            Whether the frame has a collision area to evaluate against.
        """
        if not self.map_provider.available(context["scene_token"]):
            return False
        return not self._ego_region(context).is_empty
