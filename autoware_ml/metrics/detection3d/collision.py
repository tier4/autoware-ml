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

"""Detection-box to reachability time-to-collision adapter.

Bridges the per-frame detection boxes (base_link ``[cx, cy, cz, dx, dy, dz, yaw,
...]``) to the reachability engine: transforms ego and each box to the map frame,
assigns every class its reachable-set kind, pulls the drivable polygon and the
per-lanelet speed limits from the scene's lanelet map, and returns one TTC per
box. Ego and wheeled agents move at the ``speed_limit`` of the lanelet they are
in (off-map they fall back to ``max_speed_mps``), living agents run at a per-class speed.
Moving agents carry half their box width as the collision body radius, static
agents use the full footprint. The metrics turn TTC into the collision weight
or the critical set.
"""

from __future__ import annotations

from math import inf
from typing import Any

import numpy as np

from autoware_ml.metrics.filters import box_footprints_map, ego_collision_agent
from autoware_ml.metrics.geometry.lanelet import LaneletMapProvider
from autoware_ml.metrics.geometry.reachability import (
    Agent,
    EgoReachability,
    ReachabilityParams,
)
from autoware_ml.types.metrics import AgentKind

# Final det class to reachable-set kind.
DEFAULT_KINDS: dict[str, AgentKind] = {
    "car": AgentKind.WHEELED,
    "truck": AgentKind.WHEELED,
    "bus": AgentKind.WHEELED,
    "train": AgentKind.WHEELED,
    "motorcycle": AgentKind.WHEELED,
    "pedestrian": AgentKind.LIVING,
    "animal": AgentKind.LIVING,
    "bicycle": AgentKind.LIVING,
    "barrier": AgentKind.STATIC,
    "traffic_cone": AgentKind.STATIC,
    "debris": AgentKind.STATIC,
    "bicycle_rack": AgentKind.STATIC,
    "vehicle_extension": AgentKind.STATIC,
}
# Living "reasonable run" speeds (m/s). Wheeled speed comes from the lanelet map.
DEFAULT_LIVING_SPEEDS: dict[str, float] = {"pedestrian": 3.0, "animal": 4.0, "bicycle": 6.0}


class CollisionTTC:
    """Per-box reachability TTC for one detection frame.

    Args:
        class_names: Ordered final class names (label index to name).
        map_provider: Resolves a ``scene_token`` to its lanelet map.
        region: Drivable region tokens the wheeled fronts are clipped to.
        params: Reachability parameters (horizon, dt, curvature bound).
        kinds: Class name to reachable-set kind, defaults to the built-in taxonomy
            mapping. Kind names are read into :class:`AgentKind` here.
        living_speeds: Living class name to run speed in m/s, defaults to the built-in speeds.
        max_speed_mps: Off-map fallback speed for ego and wheeled agents. On the
            map they take the ``speed_limit`` of the lanelet they are in.
        ego_length_m: Assumed ego body length (ego has no detection box).
        ego_width_m: Assumed ego body width. Object bodies come from their box.
    """

    def __init__(
        self,
        class_names: tuple[str, ...] | list[str],
        map_provider: LaneletMapProvider,
        *,
        region: tuple[str, ...] = ("road", "road_shoulder", "crosswalk"),
        params: ReachabilityParams | None = None,
        kinds: dict[str, AgentKind] | None = None,
        living_speeds: dict[str, float] | None = None,
        max_speed_mps: float = 16.7,
        ego_length_m: float = 4.9,
        ego_width_m: float = 2.0,
    ) -> None:
        """Validate the class-to-kind mapping and the living run speeds."""
        self.class_names = tuple(class_names)
        self.map_provider = map_provider
        self.region = tuple(region)
        self.params = params or ReachabilityParams()
        self.kinds = {name: AgentKind(kind) for name, kind in (kinds or DEFAULT_KINDS).items()}
        self.living_speeds = dict(living_speeds or DEFAULT_LIVING_SPEEDS)
        self.max_speed_mps = float(max_speed_mps)
        self.ego_length_m = float(ego_length_m)
        self.ego_width_m = float(ego_width_m)
        unmapped = sorted(set(self.class_names) - set(self.kinds))
        if unmapped:
            raise ValueError(f"no collision kind mapped for classes {unmapped}.")
        missing_living = sorted(
            name
            for name in self.class_names
            if self.kinds[name] == AgentKind.LIVING and name not in self.living_speeds
        )
        if missing_living:
            raise ValueError(f"no run speed configured for living classes {missing_living}.")

    def available(self, scene_token: str) -> bool:
        """False for scenes with no lanelet map, excluded from the criticality metrics.

        Args:
            scene_token: Scene identifier the map provider resolves.

        Returns:
            Whether TTC can be evaluated for the scene.
        """
        return self.map_provider.available(scene_token)

    def per_box_ttc(
        self, boxes: np.ndarray, labels: np.ndarray, ego2global: Any, scene_token: str
    ) -> np.ndarray:
        """TTC (seconds, ``inf`` = unreachable) for each base_link box in the frame.

        Args:
            boxes: Box rows ``[cx, cy, cz, dx, dy, dz, yaw, ...]`` in base_link.
            labels: Integer class labels aligned with ``boxes``.
            ego2global: 4x4 ego-to-global transform of the frame.
            scene_token: Scene identifier resolving to the lanelet map.

        Returns:
            Per-box TTC array of shape ``(N,)``.
        """
        boxes = np.asarray(boxes, dtype=np.float64)
        labels = np.asarray(labels).astype(int)
        ttc = np.full(boxes.shape[0], inf, dtype=np.float64)
        if boxes.shape[0] == 0:
            return ttc

        lanelet_map = self.map_provider.get(scene_token)
        drivable = lanelet_map.region_union(self.region)
        ego = ego_collision_agent(
            lanelet_map, ego2global, self.max_speed_mps, self.ego_length_m, self.ego_width_m
        )
        frame = EgoReachability(ego, drivable, self.params)
        footprints = box_footprints_map(boxes, ego2global)
        centroids = np.array(
            [[p.centroid.x, p.centroid.y] for p in footprints], dtype=np.float64
        )

        for index in range(boxes.shape[0]):
            name = self.class_names[int(labels[index])]
            kind = self.kinds[name]
            cx, cy = centroids[index, 0], centroids[index, 1]
            length, width = float(boxes[index, 3]), float(boxes[index, 4])
            if kind == AgentKind.STATIC:
                agent = Agent.static(cx, cy, footprints[index])
            elif kind == AgentKind.WHEELED:
                # Max speed = the speed limit of the lanelet the agent is in.
                speed = lanelet_map.speed_at(cx, cy, self.max_speed_mps)
                agent = Agent.wheeled(
                    cx, cy, float(boxes[index, 6]) + ego.heading, speed, length, width
                )
            else:
                # A living agent runs at its class speed in any direction, so its
                # body is a disc across the widest extent the box reports.
                agent = Agent.living(cx, cy, self.living_speeds[name], max(length, width) / 2.0)
            ttc[index] = frame.time_to_collision(agent)
        return ttc
