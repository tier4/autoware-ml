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
A wheeled box sweeps its own rectangle, its reference point being the box centre
so the body splits evenly along the heading, a living agent expands a disc across
the widest extent the box reports, and a static agent keeps its footprint. The
metrics turn TTC into the collision weight or the critical set.

Both sides of a collision are clipped to the mapped drivable area, so a wheeled
agent whose reference point sits outside it cannot move and is scored as
unreachable. A vehicle in an unmapped driveway is therefore not counted, even
with part of its body already on the road. Map coverage is what fixes that, not
the speed fallback: off the map the speed is only the ``max_speed_mps`` default.

One box costs one propagation, and a detection head emits hundreds per frame:
200 predictions inside 60 m take about 15 s for a single frame. Metrics that
only read confident predictions declare a ``ttc_score_floor`` so the suite can
skip the rest, which is why the provider takes the scores and that floor.
"""

from __future__ import annotations

from math import inf, nan
from typing import Any

import numpy as np

from autoware_ml.metrics.filters import (
    DEFAULT_DRIVABLE_REGION,
    box_footprints_map,
    ego2global_pose,
    ego_collision_agent,
)
from autoware_ml.metrics.geometry.lanelet import LaneletMapProvider
from autoware_ml.metrics.geometry.reachability import (
    Agent,
    EgoReachability,
    ReachabilityParams,
    VehicleGeometry,
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
    # A trailer or towed body moves with whatever pulls it, so it gets the wheeled
    # worst case rather than staying where the frame found it.
    "vehicle_extension": AgentKind.WHEELED,
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
        vehicle: Ego body dimensions, a platform property with no default (ego has no
            detection box), measured from the rear axle the ego pose refers to.
            Object bodies come from their box.
    """

    def __init__(
        self,
        class_names: tuple[str, ...] | list[str],
        map_provider: LaneletMapProvider,
        *,
        vehicle: VehicleGeometry,
        region: tuple[str, ...] = DEFAULT_DRIVABLE_REGION,
        params: ReachabilityParams | None = None,
        kinds: dict[str, AgentKind] | None = None,
        living_speeds: dict[str, float] | None = None,
        max_speed_mps: float = 16.7,
    ) -> None:
        """Validate the class-to-kind mapping and the living run speeds."""
        self.class_names = tuple(class_names)
        self.map_provider = map_provider
        self.region = tuple(region)
        self.params = params or ReachabilityParams()
        kinds = DEFAULT_KINDS if kinds is None else kinds
        self.kinds = {name: AgentKind(kind) for name, kind in kinds.items()}
        self.living_speeds = dict(DEFAULT_LIVING_SPEEDS if living_speeds is None else living_speeds)
        self.max_speed_mps = float(max_speed_mps)
        self.vehicle = vehicle
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
        self,
        boxes: np.ndarray,
        labels: np.ndarray,
        ego2global: Any,
        scene_token: str,
        scores: np.ndarray | None = None,
        score_floor: float = 0.0,
    ) -> np.ndarray:
        """TTC (seconds, ``inf`` = unreachable) for each base_link box in the frame.

        Propagation runs per box and dominates the suite's cost, so a caller that
        knows no active metric reads the low-score tail passes ``scores`` and the
        floor below which it can be skipped. Skipped boxes come back ``nan``.

        Args:
            boxes: Box rows ``[cx, cy, cz, dx, dy, dz, yaw, ...]`` in base_link.
            labels: Integer class labels aligned with ``boxes``.
            ego2global: 4x4 ego-to-global transform of the frame.
            scene_token: Scene identifier resolving to the lanelet map.
            scores: Detection scores aligned with ``boxes``, for the floor. Ground
                truth has none, so it is always propagated.
            score_floor: Lowest score worth propagating, ``0.0`` propagates every box.

        Returns:
            Per-box TTC array of shape ``(N,)``, ``nan`` where the floor skipped a box.
        """
        boxes = np.asarray(boxes, dtype=np.float64)
        labels = np.asarray(labels).astype(int)
        ttc = np.full(boxes.shape[0], inf, dtype=np.float64)
        if boxes.shape[0] == 0:
            return ttc

        lanelet_map = self.map_provider.get(scene_token)
        drivable = lanelet_map.region_union(self.region)
        ego_x, ego_y, ego_heading = ego2global_pose(ego2global)
        ego = ego_collision_agent(
            lanelet_map, ego_x, ego_y, ego_heading, self.max_speed_mps, self.vehicle
        )
        frame = EgoReachability(ego, drivable, self.params)
        footprints = box_footprints_map(boxes, ego2global)
        centroids = np.array([[p.centroid.x, p.centroid.y] for p in footprints], dtype=np.float64)

        for index in range(boxes.shape[0]):
            if scores is not None and scores[index] < score_floor:
                # Never read by any active metric: NaN so a metric that reads it
                # anyway fails loud instead of taking it for "unreachable".
                ttc[index] = nan
                continue
            agent = self._agent_for_box(
                boxes[index],
                int(labels[index]),
                footprints[index],
                centroids[index],
                lanelet_map,
                ego.heading,
            )
            ttc[index] = frame.time_to_collision(agent)
        return ttc

    def _agent_for_box(
        self,
        box: np.ndarray,
        label: int,
        footprint: Any,
        centroid: np.ndarray,
        lanelet_map: Any,
        ego_heading: float,
    ) -> Agent:
        """The collision agent one detection box stands for, by its class kind.

        Args:
            box: Box row ``[cx, cy, cz, dx, dy, dz, yaw, ...]`` in base_link.
            label: Integer class label of the box.
            footprint: The box's map-frame footprint polygon.
            centroid: The footprint's map-frame ``(x, y)`` centroid.
            lanelet_map: The scene's parsed lanelet map, for the local speed limit.
            ego_heading: Ego map-frame heading, added to the box's own yaw.

        Returns:
            The agent for this box.
        """
        name = self.class_names[label]
        kind = self.kinds[name]
        cx, cy = float(centroid[0]), float(centroid[1])
        length, width = float(box[3]), float(box[4])
        if kind == AgentKind.STATIC:
            return Agent.static(cx, cy, footprint)
        if kind == AgentKind.WHEELED:
            # Max speed = the speed limit of the lanelet the agent is in. A box is
            # centred, so its body splits evenly around the reference point, and it
            # carries no steering geometry, so the friction floor stands in.
            return Agent.wheeled(
                cx,
                cy,
                float(box[6]) + ego_heading,
                lanelet_map.speed_at(cx, cy, self.max_speed_mps),
                front=length / 2.0,
                rear=length / 2.0,
                width=width,
                min_turn_radius=self.params.min_radius_m,
            )
        # A living agent runs at its class speed in any direction, so its body is a
        # disc across the widest extent the box reports.
        return Agent.living(cx, cy, self.living_speeds[name], max(length, width) / 2.0)
