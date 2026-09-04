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

"""Reachability time-to-collision for the criticality metrics.

One planner-independent model. Every agent, ego included, travels at the
maximum map-legal speed for its class. The recorded path and observed
velocities are never used. Time-to-collision is the earliest look-ahead time
``t`` at which ego and the object can occupy a common point at the same time
``t``, each having travelled exactly ``speed * t`` along a feasible path
(reachable at ``t``, not "arrive early and wait"). That is what makes
same-direction traffic at matched speed non-critical: the two fronts stay a
constant gap apart and never meet, so TTC = inf and its collision weight is 0.

Reachable-at-``t`` set by class, in the map frame (metres):

* wheeled (car / truck / bus / train / motorcycle, and ego): the endpoints
  of every feasible constant-curvature arc of length ``v * t`` under bounded
  steering (minimum turn radius), a curved front, clipped to the drivable
  area (an arc leaving the road is infeasible) and given the vehicle's body
  half-width.
* living (pedestrian / animal / bicycle): the disc of radius ``v * t`` about the
  current position, free to move in any direction, over any surface.
* static (barrier / traffic_cone / debris / bicycle_rack / vehicle_extension):
  the fixed footprint, for every ``t``.

The two steps match the metric's contract:

1. prescreen: distance bounds prove most objects can never meet ego within
   the horizon (their largest reachable set misses ego's reachable region, or
   the straight-line gap cannot close in time), giving TTC = inf with no
   stepping. Both bounds are supersets of the true sets, so an object they
   exclude never had an overlap to find.
2. otherwise step ``t`` up from the earliest feasible step and return the
   first ``t`` at which the two reachable-at-``t`` sets overlap, the fastest
   path to collision.

Every speed arrives in m/s on :class:`Agent`. Resolving a class to its speed,
reading the map's speed limits and converting units are the caller's job.

:class:`EgoReachability` holds everything that depends only on ego (the
localized drivable area, the filled reachable region used by the prescreen, and
the per-step fronts), so one frame's many objects share it instead of rebuilding
identical ego geometry per box.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, hypot, inf, isfinite, sin

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from autoware_ml.types.metrics import AgentKind

# Abutting map ways are authored with independently rounded coordinates, so their
# union can keep a hairline seam. An exact path check would reject every arc that
# crosses one, so a surface is sealed by this much before it is used. Millimetre
# precision is far below the accuracy the model claims, so seal generously.
SURFACE_TOLERANCE_M = 1e-3

# The reachable region's far edge is a chord approximation of the far arc, so a
# front can poke past it. The prescreen bound is inflated by this much to stay a
# true superset of every front, again well below the model's accuracy.
REGION_SLACK_M = 1e-2


@dataclass(frozen=True)
class ReachabilityParams:
    """Spec-versioned parameters shared by every reachability TTC evaluation.

    The wheeled turn radius is bounded by the lateral-acceleration (friction)
    limit, ``R = v^2 / max_lateral_accel``, floored at ``min_radius_m`` for low
    speed, so a fast vehicle cannot turn implausibly tight.
    """

    horizon_s: float = 4.0
    dt_s: float = 0.1
    max_lateral_accel_mps2: float = 3.0
    min_radius_m: float = 3.0
    arc_samples: int = 21

    def __post_init__(self) -> None:
        if self.horizon_s <= 0.0 or self.dt_s <= 0.0:
            raise ValueError("horizon_s and dt_s must be > 0.")
        if self.dt_s > self.horizon_s:
            raise ValueError("dt_s must not exceed horizon_s.")
        if self.max_lateral_accel_mps2 <= 0.0 or self.min_radius_m <= 0.0:
            raise ValueError("max_lateral_accel_mps2 and min_radius_m must be > 0.")
        if self.arc_samples < 3:
            raise ValueError("arc_samples must be >= 3 to trace a front.")
        if self.arc_samples % 2 == 0:
            # An even count straddles zero curvature, leaving the straight path,
            # the most important one for a lane follower, untested.
            raise ValueError("arc_samples must be odd so zero curvature is sampled.")

    def turn_radius(self, speed: float) -> float:
        """Minimum feasible turn radius at ``speed`` (friction-limited, floored).

        Args:
            speed: Speed in m/s.

        Returns:
            The minimum turn radius in meters.
        """
        return max(speed * speed / self.max_lateral_accel_mps2, self.min_radius_m)


@dataclass(frozen=True)
class Agent:
    """One collision participant in the map frame.

    ``kind`` selects the reachable-set shape. ``heading`` and ``speed`` are used
    by the wheeled kind, the living kind uses ``speed`` isotropically, and the
    static kind ignores both and uses ``footprint``. ``body_radius`` is the
    half-extent added so a collision is a footprint overlap, not a point
    coincidence.

    Nothing is defaulted: a vehicle silently placed at heading 0 or speed 0
    scores a plausible looking TTC that is simply wrong. Build agents through
    :meth:`wheeled`, :meth:`living` and :meth:`static`, which each ask for
    exactly what their kind uses.
    """

    kind: AgentKind
    x: float
    y: float
    heading: float
    speed: float
    body_radius: float
    footprint: Polygon | None = None

    @classmethod
    def wheeled(cls, x: float, y: float, heading: float, speed: float, body_radius: float) -> Agent:
        """A road-bound agent, whose reachable set follows its heading.

        Args:
            x: Position along the map x axis in meters.
            y: Position along the map y axis in meters.
            heading: Orientation in radians, counter-clockwise from the x axis.
            speed: Worst-case speed in m/s.
            body_radius: Collision half-extent in meters.

        Returns:
            The wheeled agent.
        """
        return cls(AgentKind.WHEELED, x, y, heading, speed, body_radius)

    @classmethod
    def living(cls, x: float, y: float, speed: float, body_radius: float) -> Agent:
        """A pedestrian, animal or cyclist, whose reachable set is a disc.

        Args:
            x: Position along the map x axis in meters.
            y: Position along the map y axis in meters.
            speed: Worst-case speed in m/s, reachable in any direction.
            body_radius: Collision half-extent in meters.

        Returns:
            The living agent, whose heading no reachable set reads.
        """
        return cls(AgentKind.LIVING, x, y, 0.0, speed, body_radius)

    @classmethod
    def static(cls, x: float, y: float, footprint: Polygon) -> Agent:
        """An agent that never moves and is its own footprint.

        Args:
            x: Reference position along the map x axis in meters.
            y: Reference position along the map y axis in meters.
            footprint: Occupied polygon in the map frame.

        Returns:
            The static agent, whose body extent is the footprint's own.
        """
        min_x, min_y, max_x, max_y = footprint.bounds
        return cls(
            AgentKind.STATIC, x, y, 0.0, 0.0, hypot(max_x - min_x, max_y - min_y) / 2.0, footprint
        )

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AgentKind):
            raise ValueError(f"kind must be an AgentKind, got {self.kind!r}.")
        if self.kind == AgentKind.STATIC:
            if self.footprint is None or self.footprint.is_empty:
                raise ValueError("a static agent needs a non-empty footprint polygon.")
        elif self.footprint is not None:
            raise ValueError(f"a {self.kind} agent is propagated, so it takes no footprint.")
        if not all(isfinite(value) for value in (self.x, self.y, self.heading, self.speed)):
            # A non-finite pose or speed propagates into empty geometry, which
            # would read as "can never be hit" instead of failing.
            raise ValueError("x, y, heading and speed must be finite.")
        if self.speed < 0.0:
            raise ValueError("speed must be >= 0.")
        if not isfinite(self.body_radius) or self.body_radius <= 0.0:
            # A zero body collapses every sweep to an empty geometry, same trap.
            raise ValueError("body_radius must be a finite value > 0.")


def _arc_endpoint(
    x: float, y: float, heading: float, kappa: float, length: float
) -> tuple[float, float]:
    """Endpoint of a constant-curvature arc of signed curvature ``kappa`` and ``length``.

    ``kappa > 0`` turns left of the heading, ``kappa == 0`` is straight. The map
    frame is right-handed in (x, y) and the caller supplies ``heading`` in radians.
    """
    if abs(kappa) < 1e-9:
        return (x + length * cos(heading), y + length * sin(heading))
    radius = 1.0 / kappa
    # Turn centre is 90 deg to the left of the heading (left of travel for kappa>0).
    cx = x - radius * sin(heading)
    cy = y + radius * cos(heading)
    phi = length * kappa  # signed swept angle
    sx, sy = x - cx, y - cy
    ex = cx + sx * cos(phi) - sy * sin(phi)
    ey = cy + sx * sin(phi) + sy * cos(phi)
    return (ex, ey)


def _arc_grid(agent: Agent, length: float, kmax: float, samples: int) -> np.ndarray:
    """Endpoints of every sampled arc, as ``(curvature, step, 2)`` coordinates.

    One vectorized evaluation of the same geometry :func:`_arc_endpoint` computes
    for a single point, because a front samples the whole curvature by length grid
    at every step of every object.
    """
    kappa = np.linspace(-kmax, kmax, samples).reshape(-1, 1)
    s = np.linspace(0.0, length, samples).reshape(1, -1)
    straight = np.abs(kappa) < 1e-9
    # Straight arcs, and the curved ones about their turn centre.
    x_straight = agent.x + s * cos(agent.heading)
    y_straight = agent.y + s * sin(agent.heading)
    radius = np.where(straight, 1.0, 1.0 / np.where(straight, 1.0, kappa))
    cx = agent.x - radius * sin(agent.heading)
    cy = agent.y + radius * cos(agent.heading)
    phi = s * kappa
    dx, dy = agent.x - cx, agent.y - cy
    x_curved = cx + dx * np.cos(phi) - dy * np.sin(phi)
    y_curved = cy + dx * np.sin(phi) + dy * np.cos(phi)
    return np.stack(
        (np.where(straight, x_straight, x_curved), np.where(straight, y_straight, y_curved)),
        axis=-1,
    )


def _connected_sweep(locus: BaseGeometry, body: float, drivable: BaseGeometry) -> BaseGeometry:
    """The body sweep around ``locus``, keeping only what stays connected to it.

    Buffering first and clipping second would let the body hop a non-drivable gap
    narrower than itself and re-land on a disconnected carriageway, so the parts
    of the clipped sweep that do not touch the locus are dropped. A locus off the
    surface keeps nothing, the same answer an infeasible arc gets.
    """
    sweep = locus.buffer(body).intersection(drivable)
    parts = _polygonal_parts(sweep)
    return shapely.union_all([part for part in parts if part.intersects(locus)])


def wheeled_front(
    agent: Agent, t: float, params: ReachabilityParams, drivable: BaseGeometry
) -> BaseGeometry:
    """The wheeled reachable-at-``t`` front: buffered endpoint locus of the feasible arcs.

    Every constant-curvature arc is sampled as a polyline at the front's own
    resolution and kept only when the whole path stays on the drivable surface,
    so an endpoint across a gap in the road stays unreachable even when it lands
    on another drivable polygon. An agent whose reference point is off the
    surface has no feasible arc and an empty front. The endpoints of consecutive
    feasible arcs form the front, swept by the body radius and clipped to the
    surface, and a sweep that would cross a non-drivable gap keeps only the part
    connected to those endpoints. A stopped agent owns its own body, clipped the
    same way.

    The surface is expected to be sealed against hairline seams, which
    :class:`EgoReachability` and :func:`reachable_region` do for their callers.

    Args:
        agent: Wheeled agent in the map frame.
        t: Time offset in seconds.
        params: Shared reachability parameters.
        drivable: Drivable surface the arcs must stay on.

    Returns:
        The buffered front geometry, empty when no arc is feasible.
    """
    length = agent.speed * t
    if length <= 1e-6:
        # A stopped agent still owns its body, clipped like a moving front so it
        # cannot spill over a kerb or a median into a lane it cannot drive to.
        return _connected_sweep(Point(agent.x, agent.y), agent.body_radius, drivable)
    kmax = 1.0 / params.turn_radius(agent.speed)
    grid = _arc_grid(agent, length, kmax, params.arc_samples)
    runs: list[list[tuple[float, float]]] = [[]]
    for arc in grid:
        if drivable.covers(LineString(arc)):
            runs[-1].append((float(arc[-1][0]), float(arc[-1][1])))
        elif runs[-1]:
            runs.append([])
    pieces = [
        _connected_sweep(
            LineString(run) if len(run) > 1 else Point(run[0]), agent.body_radius, drivable
        )
        for run in runs
        if run
    ]
    if not pieces:
        return Polygon()
    return shapely.union_all(pieces)


def reachable_region(
    agent: Agent, params: ReachabilityParams, drivable: BaseGeometry | None
) -> BaseGeometry:
    """Filled region a wheeled agent can reach within the horizon (the "hat").

    Union of its fronts over ``t`` in ``(0, T]``, whose boundary is the two
    extreme max-curvature arcs plus the far arc at ``t = T``, swept by the body
    radius and clipped to ``drivable``. The sweep covers the agent's own body at
    the start, which keeps the region a superset of every front and is what the
    prescreen relies on. Used by the collision filter's in-the-ego-path
    membership test.

    Args:
        agent: Wheeled agent in the map frame.
        params: Shared reachability parameters.
        drivable: Drivable surface the region is clipped to.

    Returns:
        The filled reachable region.
    """
    length = agent.speed * params.horizon_s
    if length <= 1e-6:
        region = Point(agent.x, agent.y).buffer(agent.body_radius)
    else:
        kmax = 1.0 / params.turn_radius(agent.speed)
        n = params.arc_samples
        x, y, heading = agent.x, agent.y, agent.heading
        left = [_arc_endpoint(x, y, heading, kmax, float(s)) for s in np.linspace(0.0, length, n)]
        far = [_arc_endpoint(x, y, heading, float(k), length) for k in np.linspace(kmax, -kmax, n)]
        right = [_arc_endpoint(x, y, heading, -kmax, float(s)) for s in np.linspace(length, 0.0, n)]
        shell = Polygon(left + far + right)
        if not shell.is_valid:
            # At low speed the max-curvature arcs sweep past pi and the ring
            # self-touches, so it is repaired into the parts it covers.
            shell = shapely.union_all(_polygonal_parts(shapely.make_valid(shell)))
        region = shell.buffer(agent.body_radius + REGION_SLACK_M)
    if drivable is not None:
        sealed = drivable.buffer(SURFACE_TOLERANCE_M)
        region = _seed_component(region.intersection(sealed), Point(agent.x, agent.y))
    return region


def _polygonal_parts(geometry: BaseGeometry) -> list[BaseGeometry]:
    """The parts of ``geometry`` that carry area.

    Clipping and repairing routinely emit lines or points at tangential contacts,
    which no agent can occupy and which would otherwise stand in for the whole
    geometry.
    """
    return [
        part
        for part in shapely.get_parts(geometry)
        if part.geom_type in ("Polygon", "MultiPolygon") and part.area > 0.0
    ]


def _seed_component(region: BaseGeometry, seed: Point) -> BaseGeometry:
    """The connected part of ``region`` closest to ``seed``.

    A clipped sweep can span several disconnected drivable polygons, and only
    the part road-connected to the agent is truly reachable.
    """
    parts = _polygonal_parts(region)
    if not parts:
        return Polygon()
    return min(parts, key=seed.distance)


def reachable_set(
    agent: Agent, t: float, params: ReachabilityParams, drivable: BaseGeometry | None
) -> BaseGeometry:
    """The agent's reachable-at-``t`` set in the map frame.

    Wheeled fronts keep only arcs that stay on ``drivable`` for their whole path,
    so a wheeled evaluation needs a drivable polygon. Living discs and static
    footprints ignore ``drivable``.

    Args:
        agent: Agent in the map frame.
        t: Time offset in seconds.
        params: Shared reachability parameters.
        drivable: Drivable surface, required for wheeled agents.

    Returns:
        The reachable-at-``t`` geometry.
    """
    if agent.kind == AgentKind.STATIC:
        return agent.footprint
    if agent.kind == AgentKind.LIVING:
        return Point(agent.x, agent.y).buffer(agent.speed * t + agent.body_radius)
    if drivable is None:
        raise ValueError("a wheeled reachable set needs a drivable polygon.")
    return wheeled_front(agent, t, params, drivable)


class EgoReachability:
    """Ego reachable sets for one frame, shared across that frame's objects.

    Everything that depends only on ego is computed once: the drivable area localized to ego's
    horizon disc (ego arcs can only run there, so clipping against the full map union per step
    is wasted work), the filled reachable region ("hat") the prescreen tests against, and, lazily
    one per step, the prepared reachable-at-``t`` fronts. :meth:`time_to_collision` then runs the
    two-step contract per object. A wheeled object's arcs are checked against its own
    localization of the full surface, its approach path can start far outside ego's reach.

    The prescreen bounds are supersets of the true reachable sets (every front lies inside the
    hat, and inside the disc of radius ``speed * t + body``), so skipping a step or an object
    they exclude never misses a real overlap.
    """

    def __init__(self, ego: Agent, drivable: BaseGeometry, params: ReachabilityParams) -> None:
        """Localize the drivable area and build the prescreen region for one ego frame.

        Args:
            ego: The ego agent, must be wheeled.
            drivable: Road-region union in the map frame that wheeled fronts are clipped to.
            params: Propagation parameters shared by every evaluation.
        """
        if ego.kind != AgentKind.WHEELED:
            raise ValueError("ego must be a wheeled agent.")
        if drivable is None:
            raise ValueError("ego reachability needs a drivable polygon to clip against.")
        self.ego = ego
        self.params = params
        # Floored with a tolerance: an exact multiple of dt_s keeps its final step and a
        # non-divisible horizon never gains a step beyond it.
        self.steps = int(params.horizon_s / params.dt_s + 1e-9)
        self._surface = drivable
        reach = ego.speed * params.horizon_s + ego.body_radius
        self._drivable = drivable.intersection(
            box(ego.x - reach, ego.y - reach, ego.x + reach, ego.y + reach)
        ).buffer(SURFACE_TOLERANCE_M)
        shapely.prepare(self._drivable)
        self._hat = reachable_region(ego, params, self._drivable)
        shapely.prepare(self._hat)
        self._fronts: list[BaseGeometry | None] = [None] * (self.steps + 1)

    def _front(self, index: int) -> BaseGeometry:
        """Ego's prepared reachable-at-``index * dt`` front (built once per step)."""
        front = self._fronts[index]
        if front is None:
            front = wheeled_front(self.ego, index * self.params.dt_s, self.params, self._drivable)
            shapely.prepare(front)
            self._fronts[index] = front
        return front

    def _first_step(self, distance: float, speed: float, body: float) -> int | None:
        """Earliest step at which a set growing ``speed * t + body`` spans ``distance``.

        ``None`` means never within the horizon, the caller returns ``inf``.
        """
        if distance <= body:
            return 1
        if speed <= 0.0:
            return None
        t_min = (distance - body) / speed
        if t_min > self.params.horizon_s:
            return None
        return max(1, ceil(t_min / self.params.dt_s - 1e-9))

    def time_to_collision(self, obj: Agent) -> float:
        """Earliest ``t`` at which ego and ``obj`` can occupy a common point at ``t``.

        Args:
            obj: Object agent in the map frame.

        Returns:
            The earliest collision time in seconds, ``inf`` when unreachable.
        """
        if self._hat.is_empty:
            # Scenes whose map has no drivable region yet are evaluated with an
            # empty surface on purpose: nothing is reachable, every object scores
            # zero risk, and the frame still contributes to the other metrics.
            return inf
        ego, params = self.ego, self.params

        if obj.kind == AgentKind.STATIC:
            # A static set never grows: it must already meet the hat.
            if not self._hat.intersects(obj.footprint):
                return inf
            # Only ego moves against a static set, so the reachable side that
            # grows is ego's, by speed * t plus its body.
            distance = obj.footprint.distance(Point(ego.x, ego.y))
            start = self._first_step(distance, ego.speed, ego.body_radius)
            if start is None:
                return inf
            for index in range(start, self.steps + 1):
                if self._front(index).intersects(obj.footprint):
                    return index * params.dt_s
            return inf

        # Moving object: its set at t is within speed * t + body of its position, so
        # it must be able to reach the hat, and jointly close the gap to ego, in time.
        hat_distance = self._hat.distance(Point(obj.x, obj.y))
        start_hat = self._first_step(hat_distance, obj.speed, obj.body_radius)
        gap = hypot(obj.x - ego.x, obj.y - ego.y)
        start_gap = self._first_step(gap, ego.speed + obj.speed, ego.body_radius + obj.body_radius)
        if start_hat is None or start_gap is None:
            return inf
        # A wheeled object's whole path must stay on the surface, so its feasibility
        # is checked against the full drivable localized to the object's own reach,
        # never against ego's clip (the approach can start outside it).
        surface = None
        if obj.kind == AgentKind.WHEELED:
            reach = obj.speed * params.horizon_s + obj.body_radius
            surface = self._surface.intersection(
                box(obj.x - reach, obj.y - reach, obj.x + reach, obj.y + reach)
            ).buffer(SURFACE_TOLERANCE_M)
            shapely.prepare(surface)
        for index in range(max(start_hat, start_gap), self.steps + 1):
            t = index * params.dt_s
            obj_set = reachable_set(obj, t, params, surface)
            if not obj_set.is_empty and self._front(index).intersects(obj_set):
                return t
        return inf


def time_to_collision(
    ego: Agent,
    obj: Agent,
    drivable: BaseGeometry | None,
    params: ReachabilityParams,
) -> float:
    """Earliest ``t`` at which ego and ``obj`` can occupy a common point at time ``t``.

    Returns ``inf`` when no such ``t`` exists within the horizon. ``ego`` must be a wheeled agent
    and needs ``drivable`` (fronts are clipped to it), pass the road/road_shoulder/crosswalk
    union in the map frame. One-pair form of :class:`EgoReachability`, which callers evaluating
    many objects against the same ego frame should build once and query instead.

    Args:
        ego: Wheeled ego agent in the map frame.
        obj: Object agent in the map frame.
        drivable: Drivable surface the wheeled fronts are clipped to.
        params: Shared reachability parameters.

    Returns:
        The earliest collision time in seconds, ``inf`` when none exists.
    """
    return EgoReachability(ego, drivable, params).time_to_collision(obj)


def collision_weights(ttc: np.ndarray, decay: float) -> np.ndarray:
    """Risk weight ``e^(-decay * TTC)`` per entry. An unreachable object (``inf``) weighs 0.

    Args:
        ttc: Per-entry TTC in seconds, positive infinity for unreachable.
        decay: Exponential decay rate in 1/s.

    Returns:
        Weights in ``[0, 1]``.

    Raises:
        ValueError: If ``decay`` is negative or not finite, or a TTC is NaN.
    """
    if not np.isfinite(decay) or decay < 0.0:
        raise ValueError("decay must be a finite value >= 0.")
    ttc = np.asarray(ttc, dtype=np.float64)
    if np.isnan(ttc).any():
        # Zero weight is what an unreachable object gets, so a NaN must not take
        # that meaning: it is corrupt input from upstream.
        raise ValueError("ttc carries NaN, which has no risk interpretation.")
    weights = np.zeros_like(ttc)
    finite = np.isfinite(ttc)
    weights[finite] = np.exp(-decay * ttc[finite])
    return weights
