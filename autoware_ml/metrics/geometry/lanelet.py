"""Lanelet2 map reading into region polygons, for the region-filter evaluation axis.

A T4 scene ships ``map/lanelet2_map.osm`` in the same local map frame the ego
poses use. The ``lanelet2`` library reads it and this module turns the result
into shapely polygons grouped by lanelet2 token, either a lanelet's ``subtype``
(road, walkway, crosswalk, road_shoulder) or an area way's ``type``
(drivable_area, crosswalk_polygon, intersection_area). A :class:`LaneletMap`
then answers point-in-region membership. The map for a scene is read once and
cached.

Two properties of T4 maps shape the reader. Node geometry comes from the
``local_x``/``local_y`` tags, the frame the ego poses live in, so the loader
runs with a geocentric projector and its projected coordinates are discarded.
The area regions are closed typed ways, which lanelet2 does not model (a
polygon there is an area relation over open line strings), so their node
references are read from the OSM while their coordinates still come from the
parsed map.

Membership is evaluated in 2D, so regions stacked at different heights, a road
under a bridge for instance, are not separated.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import lanelet2
import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


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
KNOWN_REGION_TOKENS = (
    frozenset(
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
    )
    | AREA_WAY_TYPES
)


def _load_osm_map(osm_path: str) -> lanelet2.core.LaneletMap:
    """Read one OSM file into a lanelet2 map.

    T4 maps are excerpts whose regulatory elements reference primitives left
    outside the cut, which the strict loader rejects outright, so the tolerant
    loader is the only usable entry point. Its per-primitive errors describe the
    map rather than our use of it and are reported once per file, while a load
    that yields no geometry at all is a broken map and raises.

    Args:
        osm_path: Path to the ``lanelet2_map.osm`` file.

    Returns:
        The map as parsed by lanelet2.

    Raises:
        ValueError: If the file yields no points.
    """
    lanelet_map, errors = lanelet2.io.loadRobust(
        osm_path, lanelet2.projection.GeocentricProjector()
    )
    if len(lanelet_map.pointLayer) == 0:
        raise ValueError(
            f"Lanelet map '{osm_path}' yielded no points. First loader error: "
            f"{errors[0] if errors else 'none reported'}"
        )
    if errors:
        logger.info("Lanelet map %s: %d loader errors, reading continued.", osm_path, len(errors))
    return lanelet_map


def _local_xy(point: lanelet2.core.Point3d) -> tuple[float, float]:
    """Position of a map point in the local frame the ego poses use.

    The loader projects the geographic node coordinates, which land in a
    different frame than the T4 ego poses, so the authoritative position is the
    ``local_x``/``local_y`` tag pair the map ships on every node.
    """
    if "local_x" not in point.attributes or "local_y" not in point.attributes:
        raise ValueError(
            f"Map point {point.id} carries no local_x/local_y tags, so it cannot be placed "
            "in the ego frame."
        )
    return float(point.attributes["local_x"]), float(point.attributes["local_y"])


def _lanelet_ring(lanelet: lanelet2.core.Lanelet) -> list[tuple[float, float]]:
    """Closed ring of a lanelet: its left bound followed by its reversed right bound."""
    return [_local_xy(point) for point in lanelet.leftBound] + [
        _local_xy(point) for point in reversed(list(lanelet.rightBound))
    ]


def _area_way_rings(
    osm_path: str, lanelet_map: lanelet2.core.LaneletMap
) -> dict[str, list[list[tuple[float, float]]]]:
    """Rings of the typed ways that carry the area regions, keyed by way ``type``.

    lanelet2 models a polygon as an area relation over open line strings, so the
    closed typed ways T4 maps use for ``drivable_area`` and its siblings reach no
    layer of the parsed map. Only their node references are read from the OSM
    here, every coordinate still comes from the parsed map. Exporting these
    regions as area relations would make this reader unnecessary.
    """
    rings: dict[str, list[list[tuple[float, float]]]] = defaultdict(list)
    for way in ET.parse(osm_path).getroot().findall("way"):
        way_type = next(
            (tag.get("v") for tag in way.findall("tag") if tag.get("k") == "type"), None
        )
        if way_type not in AREA_WAY_TYPES:
            continue
        rings[way_type].append(
            [_local_xy(lanelet_map.pointLayer[int(nd.get("ref"))]) for nd in way.findall("nd")]
        )
    return dict(rings)


def load_region_polygons(osm_path: str) -> dict[str, list[Polygon]]:
    """Read a lanelet2 OSM into ``{token: [polygon, ...]}``.

    Lanelets are keyed by their ``subtype`` (polygon = left bound + reversed right
    bound), area ways in :data:`AREA_WAY_TYPES` by their ``type``.

    Args:
        osm_path: Path to the ``lanelet2_map.osm`` file.

    Returns:
        Region token to polygons mapping.
    """
    return _region_polygons(osm_path, _load_osm_map(osm_path))


def _region_polygons(
    osm_path: str, lanelet_map: lanelet2.core.LaneletMap
) -> dict[str, list[Polygon]]:
    regions: dict[str, list[Polygon]] = defaultdict(list)
    for lanelet in lanelet_map.laneletLayer:
        subtype = lanelet.attributes["subtype"] if "subtype" in lanelet.attributes else "unknown"
        regions[subtype].extend(_ring_polygons(_lanelet_ring(lanelet)))

    for way_type, rings in _area_way_rings(osm_path, lanelet_map).items():
        for ring in rings:
            regions[way_type].extend(_ring_polygons(ring))

    return dict(regions)


def _ring_polygons(ring: list[tuple[float, float]]) -> list[Polygon]:
    """Polygons covered by one node ring, repaired when the ring self-intersects.

    A ring that crosses itself is not a polygon, and repairing it can yield
    several: a figure-eight covers two lobes. Every part is kept, because
    picking one would silently shrink the region. Repair emits lines for a
    degenerate ring, which carry no membership and are dropped, and a ring of
    fewer than three nodes covers nothing at all.

    Args:
        ring: Node positions of the ring, in order.

    Returns:
        The polygons the ring covers, empty when it covers no area.

    Raises:
        ValueError: If the ring cannot be repaired into valid geometry.
    """
    if len(ring) < 3:
        return []
    polygon = Polygon(ring)
    if polygon.is_valid:
        return [polygon] if polygon.area > 0.0 else []
    repaired = shapely.make_valid(polygon)
    if not repaired.is_valid:
        raise ValueError(f"Map ring starting at {ring[0]} cannot be repaired into valid geometry.")
    return [
        part
        for part in shapely.get_parts(repaired)
        if part.geom_type == "Polygon" and part.area > 0.0
    ]


def load_lanelet_speeds(osm_path: str) -> list[tuple[Polygon, float]]:
    """Read the lanelets that carry a ``speed_limit`` into ``(polygon, m/s)`` pairs.

    Lanelet ``speed_limit`` tags are in km/h (values like 60/50/40/30). Each is
    paired with its lanelet polygon (left bound + reversed right bound) so a
    position can be resolved to its lane speed limit. Lanelets without the tag are
    skipped (the caller falls back to a spec-versioned default), while a tag that
    is not a positive number is corrupt map data and raises.

    Args:
        osm_path: Path to the ``lanelet2_map.osm`` file.

    Returns:
        ``(polygon, speed limit in m/s)`` pairs.

    Raises:
        ValueError: If a ``speed_limit`` tag is not a positive number.
    """
    return _lanelet_speeds(_load_osm_map(osm_path))


def _lanelet_speeds(lanelet_map: lanelet2.core.LaneletMap) -> list[tuple[Polygon, float]]:
    speeds: list[tuple[Polygon, float]] = []
    for lanelet in lanelet_map.laneletLayer:
        if "speed_limit" not in lanelet.attributes:
            continue
        raw_speed = lanelet.attributes["speed_limit"]
        try:
            speed_kmh = float(raw_speed)
        except ValueError as error:
            raise ValueError(
                f"Lanelet {lanelet.id} carries an unparsable speed_limit {raw_speed!r}."
            ) from error
        if speed_kmh <= 0.0:
            raise ValueError(
                f"Lanelet {lanelet.id} carries a non-positive speed_limit {raw_speed!r}."
            )
        speed_mps = speed_kmh / 3.6
        speeds.extend((polygon, speed_mps) for polygon in _ring_polygons(_lanelet_ring(lanelet)))
    return speeds


@lru_cache(maxsize=None)
def _load_lanelet_map(osm_path: str) -> "LaneletMap":
    """One parsed map per OSM path, shared process-wide.

    Filters and suites each hold their own provider instance (hydra instantiates
    one per config reference). Without this cache the same scene's map would be
    read, unioned and eroded once per instance. One lanelet2 load feeds both the
    region polygons and the speed lanelets.
    """
    parsed = _load_osm_map(osm_path)
    return LaneletMap(_region_polygons(osm_path, parsed), _lanelet_speeds(parsed))


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

        On overlapping lanelets the highest limit wins. The collision model is
        worst case and a faster agent reaches further, so the higher limit is the
        conservative reading of time to collision. Overlaps also occur between
        lanelets stacked at different heights, a road under a bridge for
        instance, which 2D membership cannot separate, and the higher limit is
        the safe choice there as well.

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
        return max(speeds) if speeds else default

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
    def region_union(self, tokens: tuple[str, ...]):
        """Shapely (multi)polygon union of the given region tokens (map frame).

        Used by the reachability collision engine to clip wheeled reachable sets
        to the drivable area (road / road_shoulder / crosswalk), and by the
        membership tests below. Cached per token set, so the tokens arrive as a
        tuple.

        Args:
            tokens: Region tokens to unite.

        Returns:
            The shapely (multi)polygon union.
        """
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
        expanded = self.region_union(tokens).buffer(margin)
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
        eroded = self.region_union(tokens).intersection(self._eroded_full_surface(margin))
        shapely.prepare(eroded)
        return eroded

    def contains(
        self,
        tokens: tuple[str, ...],
        xy: np.ndarray,
        margin: float = 0.0,
    ) -> np.ndarray:
        """Boolean mask of map-frame ``xy`` points inside the union of ``tokens``.

        ``margin`` moves the region border by that many meters, its sign picking
        the direction the way a shapely buffer does:

        * negative erodes: points within ``margin`` of the outer border of the
          full mapped surface stop counting. Internal borders between adjacent
          regions stay intact, so points near a road-to-walkway border keep
          counting for both.
        * positive dilates: the selected region additionally claims off-map
          points within ``margin`` of it. Points belonging to another mapped
          region are never claimed, so adjacent regions do not overlap.

        Args:
            tokens: Region tokens to unite.
            xy: Map-frame points ``(N, 2)``.
            margin: Border shift in meters, negative erodes and positive dilates.

        Returns:
            Boolean mask ``(N,)``.
        """
        if xy.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        points = shapely.points(xy[:, 0], xy[:, 1])
        mask = np.asarray(shapely.contains(self.region_union(tuple(tokens)), points), dtype=bool)
        if margin == 0.0:
            return mask
        if margin > 0.0:
            in_expanded = np.asarray(
                shapely.contains(self._expanded_region(tuple(tokens), float(margin)), points),
                dtype=bool,
            )
            on_map = np.asarray(shapely.contains(self._full_surface(), points), dtype=bool)
            return mask | (in_expanded & ~on_map)
        eroded = self._eroded_full_surface(-float(margin))
        return mask & np.asarray(shapely.contains(eroded, points), dtype=bool)

    def intersects(
        self,
        tokens: tuple[str, ...],
        footprints: list,
        margin: float = 0.0,
    ) -> np.ndarray:
        """Boolean mask of BEV footprint polygons that overlap the region.

        The box counterpart of :meth:`contains`: a detection box belongs to the
        region when any part of its footprint lies inside it, so an object
        overhanging the region from an off-region center still counts.
        ``margin`` moves the region border exactly as in :meth:`contains`,
        applied here to the footprint-intersection test.

        Args:
            tokens: Region tokens to unite.
            footprints: Map-frame BEV footprint polygons.
            margin: Border shift in meters, negative erodes and positive dilates.

        Returns:
            Boolean mask over the footprints.
        """
        if len(footprints) == 0:
            return np.zeros((0,), dtype=bool)
        if margin < 0.0:
            return np.asarray(
                shapely.intersects(self._eroded_region(tuple(tokens), -float(margin)), footprints),
                dtype=bool,
            )
        hit = np.asarray(
            shapely.intersects(self.region_union(tuple(tokens)), footprints), dtype=bool
        )
        if margin == 0.0:
            return hit
        in_expanded = np.asarray(
            shapely.intersects(self._expanded_region(tuple(tokens), float(margin)), footprints),
            dtype=bool,
        )
        on_map = np.asarray(shapely.intersects(self._full_surface(), footprints), dtype=bool)
        return hit | (in_expanded & ~on_map)


class OsmPathResolver(Protocol):
    """What a :class:`LaneletMapProvider` needs from its resolver.

    Two calls, because a scene may have no map at all: the provider asks for the
    path when it reads a map and asks :meth:`available` when a filter has to
    decide whether the scene can take part in a map-based slice.
    """

    def __call__(self, scene_token: object) -> str:
        """Path of the scene's ``lanelet2_map.osm``."""

    def available(self, scene_token: object) -> bool:
        """Whether the scene has a map, decided without reading it."""


class LaneletMapProvider:
    """Resolves a scene token to its :class:`LaneletMap`.

    ``resolve_osm`` maps a scene token to its ``lanelet2_map.osm`` path and
    reports whether a scene has one, see :class:`OsmPathResolver`. Each scene's
    map is parsed once and reused, cached by path in :meth:`LaneletMap.from_osm`.
    A dataset adapter supplies the resolver (it knows the on-disk scene layout),
    tests inject a direct path.
    """

    def __init__(self, resolve_osm: OsmPathResolver) -> None:
        """Store the scene-token to OSM-path resolver.

        Args:
            resolve_osm: Resolver satisfying :class:`OsmPathResolver`, so it
                returns a path when called and answers ``available``.
        """
        self._resolve_osm = resolve_osm

    def get(self, scene_token: object) -> LaneletMap:
        """The scene's map, parsed on the first request and shared afterwards.

        Args:
            scene_token: Scene identifier.

        Returns:
            The scene's parsed map.
        """
        return LaneletMap.from_osm(self._resolve_osm(scene_token))

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
