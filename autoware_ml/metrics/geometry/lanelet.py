"""Lanelet2 map parsing into region polygons, for the region-filter evaluation axis.

A T4 scene ships ``map/lanelet2_map.osm`` in the same local map frame the ego
poses use. This module parses it into shapely polygons grouped by lanelet2 token,
either a lanelet's ``subtype`` (road, walkway, crosswalk, road_shoulder) or an
area way's ``type`` (drivable_area, crosswalk_polygon, intersection_area). A
:class:`LaneletMap` then answers point-in-region membership. The map for a scene
is parsed once and cached.

lanelet2 itself is not installed in the container, so the OSM is parsed with the
standard-library XML parser and polygons are built with shapely (both available).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# Way ``type`` values that describe a closed area we can use as a region.
AREA_WAY_TYPES = frozenset(
    {
        "drivable_area",
        "crosswalk_polygon",
        "intersection_area",
        "hatched_road_markings",
        "no_obstacle_segmentation_area",
    }
)

# Region tokens a filter may request: the lanelet2 lanelet subtypes plus the
# area way types above. Requesting anything else is a configuration typo and
# fails loud at filter construction. A known token that happens to have no
# polygons in a particular scene (walkway is sparse) is a legitimate empty
# region where membership is simply false.
KNOWN_REGION_TOKENS = frozenset(
    {
        "road",
        "highway",
        "road_shoulder",
        "bicycle_lane",
        "bus_lane",
        "walkway",
        "crosswalk",
        "pedestrian_lane",
        "play_street",
        "emergency_lane",
    }
) | AREA_WAY_TYPES


def _tags(element: ET.Element) -> dict[str, str]:
    return {tag.get("k"): tag.get("v") for tag in element.findall("tag")}


def _way_coords(
    way: ET.Element, nodes: dict[str, tuple[float, float]]
) -> list[tuple[float, float]]:
    return [nodes[nd.get("ref")] for nd in way.findall("nd") if nd.get("ref") in nodes]


def _parse_osm(
    osm_path: str,
) -> tuple[ET.Element, dict[str, tuple[float, float]], dict[str, ET.Element]]:
    """One XML pass shared by the region and speed loaders: root, nodes, ways."""
    root = ET.parse(osm_path).getroot()
    nodes: dict[str, tuple[float, float]] = {}
    for node in root.findall("node"):
        tags = _tags(node)
        if "local_x" in tags and "local_y" in tags:
            nodes[node.get("id")] = (float(tags["local_x"]), float(tags["local_y"]))
    ways: dict[str, ET.Element] = {way.get("id"): way for way in root.findall("way")}
    return root, nodes, ways


def _lanelet_ring_polygon(
    relation: ET.Element,
    nodes: dict[str, tuple[float, float]],
    ways: dict[str, ET.Element],
) -> Polygon | None:
    """A lanelet relation's polygon: left bound + reversed right bound, repaired."""
    bounds = {
        member.get("role"): ways.get(member.get("ref"))
        for member in relation.findall("member")
        if member.get("role") in ("left", "right")
    }
    left, right = bounds.get("left"), bounds.get("right")
    if left is None or right is None:
        return None
    return _safe_polygon(_way_coords(left, nodes) + list(reversed(_way_coords(right, nodes))))


def load_region_polygons(osm_path: str) -> dict[str, list[Polygon]]:
    """Parse a lanelet2 OSM into ``{token: [polygon, ...]}``.

    Lanelets are keyed by their ``subtype`` (polygon = left bound + reversed right
    bound), area ways in :data:`AREA_WAY_TYPES` by their ``type``.

    Args:
        osm_path: Path to the ``lanelet2_map.osm`` file.

    Returns:
        Region token to polygons mapping.
    """
    return _region_polygons(*_parse_osm(osm_path))


def _region_polygons(
    root: ET.Element,
    nodes: dict[str, tuple[float, float]],
    ways: dict[str, ET.Element],
) -> dict[str, list[Polygon]]:
    regions: dict[str, list[Polygon]] = defaultdict(list)
    for relation in root.findall("relation"):
        tags = _tags(relation)
        if tags.get("type") != "lanelet":
            continue
        polygon = _lanelet_ring_polygon(relation, nodes, ways)
        if polygon is not None:
            regions[tags.get("subtype", "unknown")].append(polygon)

    for way in ways.values():
        tags = _tags(way)
        if tags.get("type") not in AREA_WAY_TYPES:
            continue
        polygon = _safe_polygon(_way_coords(way, nodes))
        if polygon is not None:
            regions[tags["type"]].append(polygon)

    return dict(regions)


def _safe_polygon(ring: list[tuple[float, float]]) -> Polygon | None:
    if len(ring) < 3:
        return None
    polygon = Polygon(ring)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)  # repair self-touching rings
    return polygon if (not polygon.is_empty and polygon.area > 0.0) else None


def load_lanelet_speeds(osm_path: str) -> list[tuple[Polygon, float]]:
    """Parse lanelet relations that carry a ``speed_limit`` into ``(polygon, m/s)``.

    Lanelet ``speed_limit`` tags are in km/h (values like 60/50/40/30). Each is
    paired with its lanelet polygon (left bound + reversed right bound) so a
    position can be resolved to its lane speed limit. Lanelets without the tag are
    skipped (the caller falls back to a spec-versioned default).

    Args:
        osm_path: Path to the ``lanelet2_map.osm`` file.

    Returns:
        ``(polygon, speed limit in m/s)`` pairs.
    """
    return _lanelet_speeds(*_parse_osm(osm_path))


def _lanelet_speeds(
    root: ET.Element,
    nodes: dict[str, tuple[float, float]],
    ways: dict[str, ET.Element],
) -> list[tuple[Polygon, float]]:
    speeds: list[tuple[Polygon, float]] = []
    for relation in root.findall("relation"):
        tags = _tags(relation)
        if tags.get("type") != "lanelet" or "speed_limit" not in tags:
            continue
        try:
            speed_mps = float(tags["speed_limit"]) / 3.6  # km/h -> m/s
        except ValueError:
            continue
        if speed_mps <= 0.0:
            continue
        polygon = _lanelet_ring_polygon(relation, nodes, ways)
        if polygon is not None:
            speeds.append((polygon, speed_mps))
    return speeds


@lru_cache(maxsize=None)
def _load_lanelet_map(osm_path: str) -> "LaneletMap":
    """One parsed map per OSM path, shared process-wide.

    Filters and suites each hold their own provider instance (hydra instantiates
    one per config reference). Without this cache the same scene's map would be
    parsed, unioned and eroded once per instance. One XML pass feeds both the
    region polygons and the speed lanelets.
    """
    parsed = _parse_osm(osm_path)
    return LaneletMap(_region_polygons(*parsed), _lanelet_speeds(*parsed))


class LaneletMap:
    """Region polygons for one scene, with point-in-region membership and per-lane speed."""

    def __init__(
        self,
        region_polygons: dict[str, list[Polygon]],
        speed_lanelets: list[tuple[Polygon, float]] | None = None,
    ) -> None:
        """Index the parsed polygons.

        Args:
            region_polygons: Region token to its polygons, map frame.
            speed_lanelets: ``(polygon, speed limit in m/s)`` pairs for
                lanelets that carry one.
        """
        self._region_polygons = region_polygons
        self._speed_lanelets = list(speed_lanelets or [])
        self._speed_polys = [polygon for polygon, _ in self._speed_lanelets]
        self._speed_values = np.array(
            [speed for _, speed in self._speed_lanelets], dtype=np.float64
        )
        self._speed_index = STRtree(self._speed_polys) if self._speed_polys else None

    def speed_at(self, x: float, y: float, default: float) -> float:
        """Speed limit (m/s) of the lanelet containing ``(x, y)``, else ``default``.

        On overlapping lanelets the lowest limit wins (conservative), so an object
        straddling a slow lane is not over-propagated.

        Args:
            x: Map-frame x coordinate.
            y: Map-frame y coordinate.
            default: Fallback speed when no lanelet contains the position.

        Returns:
            The speed limit in m/s.
        """
        if self._speed_index is None:
            return default
        point = Point(x, y)
        candidates = self._speed_index.query(point)
        speeds = [
            float(self._speed_values[i]) for i in candidates if self._speed_polys[i].contains(point)
        ]
        return min(speeds) if speeds else default

    @classmethod
    def from_osm(cls, osm_path: str) -> "LaneletMap":
        """Parsed map for an OSM path, shared through the process-wide cache.

        Args:
            osm_path: Path to the ``lanelet2_map.osm`` file.

        Returns:
            The parsed map.
        """
        return _load_lanelet_map(osm_path)

    @lru_cache(maxsize=None)
    def _region_union(self, tokens: tuple[str, ...]):
        polygons = [poly for token in tokens for poly in self._region_polygons.get(token, [])]
        if not polygons:
            # A known region that this scene's map simply does not contain, a
            # perfectly normal condition (walkway is sparse): the region is empty
            # and membership is false everywhere. Token typos are rejected at
            # RegionFilter construction instead.
            return Polygon()
        union = unary_union(polygons)
        # Prepared geometry turns the per-point containment test from the slow
        # generic path into an indexed one, the difference between minutes and
        # milliseconds per frame at full point-cloud resolution.
        shapely.prepare(union)
        return union

    def region_union(self, tokens: tuple[str, ...]):
        """Public: shapely (multi)polygon union of the given region tokens (map frame).

        Used by the reachability collision engine to clip wheeled reachable sets
        to the drivable area (road / road_shoulder / crosswalk).

        Args:
            tokens: Region tokens to unite.

        Returns:
            The shapely (multi)polygon union.
        """
        return self._region_union(tuple(tokens))

    @lru_cache(maxsize=None)
    def _full_surface(self):
        """Union of every mapped region, the whole mapped surface."""
        polygons = [poly for polys in self._region_polygons.values() for poly in polys]
        union = unary_union(polygons)
        shapely.prepare(union)
        return union

    @lru_cache(maxsize=None)
    def _eroded_full_surface(self, margin: float):
        """The full mapped surface eroded inward by ``margin``.

        The erosion applies to the outer border of the whole mapped surface only:
        internal borders between adjacent regions (road and walkway) are interior
        to this union and stay intact, so no dead gap appears between regions.
        """
        eroded = self._full_surface().buffer(-margin)
        shapely.prepare(eroded)
        return eroded

    @lru_cache(maxsize=None)
    def _expanded_region(self, tokens: tuple[str, ...], margin: float):
        """The selected region dilated outward by ``margin``."""
        expanded = self._region_union(tokens).buffer(margin)
        shapely.prepare(expanded)
        return expanded

    @lru_cache(maxsize=None)
    def _eroded_region(self, tokens: tuple[str, ...], margin: float):
        """The part of the selected region that survives the outer-border erosion.

        Footprints must be tested against this single geometry: checking the
        selected region and the eroded surface separately would keep a box that
        touches the region only inside the removed border band while overlapping
        an adjacent region's eroded part.
        """
        eroded = self._region_union(tokens).intersection(self._eroded_full_surface(margin))
        shapely.prepare(eroded)
        return eroded

    def contains(
        self,
        tokens: tuple[str, ...],
        xy: np.ndarray,
        margin: float = 0.0,
        expand: bool = False,
    ) -> np.ndarray:
        """Boolean mask of map-frame ``xy`` points inside the union of ``tokens``.

        ``margin`` adjusts the region border by that many meters and ``expand``
        picks the direction:

        * ``False`` (default) cuts inward: points within ``margin`` of the
          outer border of the full mapped surface stop counting. Internal
          borders between adjacent regions stay intact, so points near a
          road-to-walkway border keep counting for both.
        * ``True`` grows outward: the selected region additionally claims
          off-map points within ``margin`` of it. Points belonging to another
          mapped region are never claimed, so adjacent regions do not overlap.

        Args:
            tokens: Region tokens to unite.
            xy: Map-frame points ``(N, 2)``.
            margin: Border adjustment in meters.
            expand: Grow outward instead of cutting inward.

        Returns:
            Boolean mask ``(N,)``.
        """
        if xy.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        points = shapely.points(xy[:, 0], xy[:, 1])
        mask = np.asarray(shapely.contains(self._region_union(tuple(tokens)), points), dtype=bool)
        if margin <= 0.0:
            return mask
        if expand:
            in_expanded = np.asarray(
                shapely.contains(self._expanded_region(tuple(tokens), float(margin)), points),
                dtype=bool,
            )
            on_map = np.asarray(shapely.contains(self._full_surface(), points), dtype=bool)
            return mask | (in_expanded & ~on_map)
        eroded = self._eroded_full_surface(float(margin))
        return mask & np.asarray(shapely.contains(eroded, points), dtype=bool)

    def intersects(
        self,
        tokens: tuple[str, ...],
        footprints: list,
        margin: float = 0.0,
        expand: bool = False,
    ) -> np.ndarray:
        """Boolean mask of BEV footprint polygons that overlap the region.

        The box counterpart of :meth:`contains`: a detection box belongs to the
        region when any part of its footprint lies inside it, so an object
        overhanging the region from an off-region center still counts.
        ``margin`` / ``expand`` adjust the region border exactly as in
        :meth:`contains`, applied here to the footprint-intersection test.

        Args:
            tokens: Region tokens to unite.
            footprints: Map-frame BEV footprint polygons.
            margin: Border adjustment in meters.
            expand: Grow outward instead of cutting inward.

        Returns:
            Boolean mask over the footprints.
        """
        if len(footprints) == 0:
            return np.zeros((0,), dtype=bool)
        if margin > 0.0 and not expand:
            return np.asarray(
                shapely.intersects(self._eroded_region(tuple(tokens), float(margin)), footprints),
                dtype=bool,
            )
        hit = np.asarray(
            shapely.intersects(self._region_union(tuple(tokens)), footprints), dtype=bool
        )
        if margin <= 0.0:
            return hit
        in_expanded = np.asarray(
            shapely.intersects(self._expanded_region(tuple(tokens), float(margin)), footprints),
            dtype=bool,
        )
        on_map = np.asarray(shapely.intersects(self._full_surface(), footprints), dtype=bool)
        return hit | (in_expanded & ~on_map)


class LaneletMapProvider:
    """Loads and caches one :class:`LaneletMap` per scene.

    ``resolve_osm`` maps a scene token to its ``lanelet2_map.osm`` path, and each
    scene's map is parsed once and reused. A dataset adapter supplies the
    resolver (it knows the on-disk scene layout), tests inject a direct path.
    """

    def __init__(self, resolve_osm) -> None:
        """Store the scene-token to OSM-path resolver.

        Args:
            resolve_osm: Callable mapping a scene token to its OSM path.
        """
        self._resolve_osm = resolve_osm
        self._cache: dict[object, LaneletMap] = {}

    def get(self, scene_token: object) -> LaneletMap:
        """The scene's parsed map, from the cache after the first request.

        Args:
            scene_token: Scene identifier.

        Returns:
            The scene's parsed map.
        """
        if scene_token not in self._cache:
            self._cache[scene_token] = LaneletMap.from_osm(self._resolve_osm(scene_token))
        return self._cache[scene_token]

    def available(self, scene_token: object) -> bool:
        """Whether a lanelet map exists for the scene (no parse, no exception).

        An existence check the region filters use to exclude map-less scenes from
        their slice. It never masks a genuine error: a scene reported available
        whose OSM is later unreadable still raises in :meth:`get`.

        Args:
            scene_token: Scene identifier.

        Returns:
            Whether a map exists for the scene.
        """
        return self._resolve_osm.available(scene_token)


class T4LaneletMapResolver:
    """Resolves a T4 scene-directory token to its ``lanelet2_map.osm`` path.

    The token surfaced to metrics as ``scene_token`` must be the scene directory
    fragment relative to ``data_root`` (``<db_name>/<scene_uuid>/<version>``),
    which the dataset derives from the frame's lidar path. Older DBs ship no
    ``map/`` directory. :meth:`available` reports that up front so the region
    filters can exclude the scene, while :meth:`__call__` still raises if a map a
    caller expected to resolve is absent.
    """

    def __init__(self, data_root: str) -> None:
        """Store the dataset root the scene tokens are relative to.

        Args:
            data_root: Dataset root directory.
        """
        self.data_root = str(data_root)

    def _osm_path(self, scene_token: object) -> Path:
        return Path(self.data_root) / str(scene_token) / "map" / "lanelet2_map.osm"

    def available(self, scene_token: object) -> bool:
        """Whether the scene's ``lanelet2_map.osm`` is present on disk.

        Args:
            scene_token: Scene directory fragment relative to the dataset root.

        Returns:
            Whether the scene's map file exists.
        """
        return self._osm_path(scene_token).is_file()

    def __call__(self, scene_token: object) -> str:
        """The scene's OSM path, raising when the scene ships no map."""
        path = self._osm_path(scene_token)
        if not path.is_file():
            raise FileNotFoundError(
                f"No lanelet map at {path}: this scene/DB ships no map/ directory, "
                "so region-filtered metrics cannot run on it."
            )
        return str(path)
