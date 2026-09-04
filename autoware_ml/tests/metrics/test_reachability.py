"""Unit tests for the reachability time-to-collision engine."""

from __future__ import annotations

from math import inf, isclose, pi

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from autoware_ml.metrics.geometry.reachability import (
    Agent,
    EgoReachability,
    ReachabilityParams,
    collision_weights,
    reachable_set,
    time_to_collision,
    wheeled_reachable_region,
    wheeled_reachable_set,
)
from autoware_ml.types.metrics import AgentKind

PARAMS = ReachabilityParams(horizon_s=4.0, dt_s=0.1)
# A wide open drivable region so wheeled fronts are not clipped in these unit cases.
ROAD = box(-80.0, -50.0, 500.0, 50.0)


def _footprint(x: float, y: float, size: float = 2.0):
    return box(x - size / 2, y - size / 2, x + size / 2, y + size / 2)


def test_same_speed_lead_still_collides_in_the_worst_case() -> None:
    # Matched speed is no protection: the lead can brake or reverse while ego
    # keeps going, so the gap closes at the sum of the two worst-case speeds.
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=10.0, length=4.8, width=2.4)
    lead = Agent.wheeled(25.0, 0.0, heading=0.0, speed=10.0, length=4.8, width=2.4)
    ttc = time_to_collision(ego, lead, ROAD, PARAMS)
    assert ttc == pytest.approx(0.8, abs=PARAMS.dt_s)  # ~ (25 - 2 * 3.4 body reach) / 20


def test_a_stationary_object_ahead_collides_at_about_distance_over_speed() -> None:
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=10.0, length=4.8, width=2)
    obj = Agent.static(30.0, 0.0, footprint=_footprint(30.0, 0.0))
    ttc = time_to_collision(ego, obj, ROAD, PARAMS)
    assert ttc != inf
    assert 2.4 <= ttc <= 3.1  # ~ (30 - body - half-footprint) / 10


def test_oncoming_closes_at_combined_speed() -> None:
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=10.0, length=4.8, width=2)
    obj = Agent.wheeled(40.0, 0.0, heading=pi, speed=10.0, length=4.8, width=2)
    ttc = time_to_collision(ego, obj, ROAD, PARAMS)
    assert ttc != inf
    assert 1.7 <= ttc <= 2.1  # ~ 40 / (10 + 10)


def test_oncoming_beyond_ego_reach_still_collides() -> None:
    # The object's approach path is checked on the full drivable surface localized to
    # its own reach, so an incoming vehicle starting outside ego's reach clip is found.
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=10.0, length=4.8, width=2)
    obj = Agent.wheeled(50.0, 0.0, heading=pi, speed=10.0, length=4.8, width=2)
    ttc = time_to_collision(ego, obj, ROAD, PARAMS)
    assert 2.1 <= ttc <= 2.4  # ~ (50 - 2 * 3.4 body reach) / 20


def test_crossing_living_agent_is_finite_within_horizon() -> None:
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=10.0, length=4.8, width=2)
    ped = Agent.living(18.0, 6.0, speed=4.0, radius=0.4)
    ttc = time_to_collision(ego, ped, ROAD, PARAMS)
    assert ttc != inf and ttc <= PARAMS.horizon_s


def test_a_far_object_is_unreachable_within_the_horizon() -> None:
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=10.0, length=4.8, width=2)
    obj = Agent.static(200.0, 0.0, footprint=_footprint(200.0, 0.0))
    assert time_to_collision(ego, obj, ROAD, PARAMS) == inf


def test_a_wheeled_set_needs_a_drivable_surface() -> None:
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=10.0, length=4.8, width=2)
    obj = Agent.static(20.0, 0.0, footprint=_footprint(20.0, 0.0))
    with pytest.raises(ValueError, match="drivable"):
        time_to_collision(ego, obj, None, PARAMS)


def test_disconnected_road_is_unreachable() -> None:
    # Two drivable strips separated by a non-drivable gap: an arc onto the other
    # strip crosses the gap, so the strips can never meet. A wheeled agent off the
    # surface entirely has no drivable arc at all.
    split_road = box(-80.0, -10.0, 200.0, 10.0).union(box(-80.0, 20.0, 200.0, 40.0))
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=10.0, length=4.8, width=2)
    oncoming_across = Agent.wheeled(5.0, 30.0, heading=-pi / 2, speed=10.0, length=4.8, width=2)
    assert time_to_collision(ego, oncoming_across, split_road, PARAMS) == inf
    off_road = Agent.wheeled(0.0, 60.0, heading=0.0, speed=10.0, length=4.8, width=2)
    assert time_to_collision(ego, off_road, ROAD, PARAMS) == inf
    # The filled reachable region keeps only the strip the agent is on.
    region = wheeled_reachable_region(ego, PARAMS, split_road)
    assert region.intersection(box(-80.0, 20.0, 200.0, 40.0)).is_empty


def test_params_reject_step_exceeding_horizon() -> None:
    with pytest.raises(ValueError, match="dt_s must not exceed horizon_s"):
        ReachabilityParams(horizon_s=0.5, dt_s=0.6)


def test_steps_stay_within_horizon() -> None:
    # A non-divisible horizon/dt floors to the last step inside the horizon, while an
    # exact multiple keeps its final step despite floating point.
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=1.0, length=4.8, width=1)
    assert EgoReachability(ego, ROAD, ReachabilityParams(horizon_s=1.0, dt_s=0.6)).steps == 1
    assert EgoReachability(ego, ROAD, ReachabilityParams(horizon_s=3.0, dt_s=0.1)).steps == 30
    # A meeting first reachable at 1.1 s lies beyond the 1.0 s horizon, so it stays inf.
    living = Agent.living(7.6, 0.0, speed=5.0, radius=0.5)
    assert time_to_collision(ego, living, ROAD, ReachabilityParams(horizon_s=1.0, dt_s=0.6)) == inf


def test_collision_weights_monotone_and_bounds() -> None:
    weights = collision_weights([inf, 0.0, 1.0, 3.0], 0.1)
    assert weights[0] == 0.0
    assert isclose(weights[1], 1.0)
    assert weights[2] > weights[3] > 0.0


def test_low_speed_region_stays_valid_past_pi_sweep() -> None:
    # At low speed the max-curvature arcs sweep past pi and fold over each other,
    # the region must still come out valid with sane membership.
    for speed in (0.83, 2.78, 3.0):
        agent = Agent.wheeled(0.0, 0.0, heading=0.0, speed=speed, length=4.8, width=2)
        region = wheeled_reachable_region(agent, PARAMS, ROAD)
        assert region.is_valid and not region.is_empty
        assert region.contains(Point(min(speed * PARAMS.horizon_s * 0.9, 10.0), 0.0))


def test_ego_reachability_matches_bruteforce_stepping() -> None:
    # The prescreen bounds are supersets of the true sets, so the optimized
    # engine must return exactly what naive stepping over every t returns.
    def brute_force(ego: Agent, obj: Agent) -> float:
        steps = int(PARAMS.horizon_s / PARAMS.dt_s + 1e-9)
        for index in range(1, steps + 1):
            t = index * PARAMS.dt_s
            ego_set = reachable_set(ego, t, PARAMS, ROAD)
            obj_set = reachable_set(obj, t, PARAMS, ROAD)
            if not ego_set.is_empty and not obj_set.is_empty and ego_set.intersects(obj_set):
                return t
        return inf

    rng = np.random.default_rng(7)
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=13.9, length=4.8, width=2)
    frame = EgoReachability(ego, ROAD, PARAMS)
    for index in range(60):
        if index % 3 == 0:
            # Objects within a body radius of ego are the ones the prescreen
            # must never discard, so a third of the samples sit right there.
            x, y = float(rng.uniform(-2.0, 2.0)), float(rng.uniform(-2.0, 2.0))
        else:
            x, y = float(rng.uniform(-30, 130)), float(rng.uniform(-45, 45))
        kind = (AgentKind.WHEELED, AgentKind.LIVING, AgentKind.STATIC)[index % 3]
        if kind == AgentKind.STATIC:
            obj = Agent.static(x, y, footprint=_footprint(x, y))
        elif kind == AgentKind.LIVING:
            obj = Agent.living(x, y, speed=float(rng.uniform(0.0, 6.0)), radius=0.4)
        else:
            obj = Agent.wheeled(
                x,
                y,
                heading=float(rng.uniform(0, 2 * pi)),
                speed=float(rng.uniform(0.5, 16.7)),
                length=4.8,
                width=2.0,
            )
        assert frame.time_to_collision(obj) == brute_force(ego, obj), f"agent {index}: {obj}"


def test_the_region_covers_every_step_including_the_first() -> None:
    """The prescreen only holds if the hat is a superset, right down to t = dt."""
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=10.0, length=4.8, width=2.4)
    hat = wheeled_reachable_region(ego, PARAMS, ROAD)

    for step in (1, 5, 10, 40):
        reachable = wheeled_reachable_set(ego, step * PARAMS.dt_s, PARAMS, ROAD)
        assert reachable.difference(hat).area < 1e-9, f"step {step} escapes the hat"


def test_an_object_beside_ego_collides_at_the_first_step() -> None:
    """A pedestrian against ego's flank is the case the metric exists to score."""
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=10.0, length=4.8, width=2.4)
    ped = Agent.living(0.1, 0.55, speed=1.4, radius=0.4)

    assert EgoReachability(ego, ROAD, PARAMS).time_to_collision(ped) == PARAMS.dt_s


def test_the_body_sweep_does_not_cross_a_narrow_median() -> None:
    """Buffering the swept path must not hop a gap thinner than the body."""
    median = box(-100.0, -4.0, 200.0, 0.0).union(box(-100.0, 0.5, 200.0, 4.5))
    ego = Agent.wheeled(0.0, -2.0, heading=0.0, speed=10.0, length=4.8, width=2.4)
    oncoming = Agent.wheeled(40.0, 2.5, heading=pi, speed=10.0, length=4.8, width=2.4)

    reachable = wheeled_reachable_set(ego, 2.0, PARAMS, median)

    assert reachable.intersection(box(-100.0, 0.5, 200.0, 4.5)).is_empty
    assert EgoReachability(ego, median, PARAMS).time_to_collision(oncoming) == inf


def test_a_stopped_agent_still_occupies_its_body() -> None:
    stopped = Agent.wheeled(20.0, 0.0, heading=0.0, speed=0.0, length=4.8, width=3.0)

    body = reachable_set(stopped, 1.0, PARAMS, ROAD)

    # The swept body rounds the two ends off, so it covers the vehicle rectangle
    # and adds the two caps. Shapely buffers into a polygon, hence the tolerance.
    assert body.contains(box(20.0 - 2.4, -1.5, 20.0 + 2.4, 1.5).buffer(-1e-9))
    assert body.area == pytest.approx(4.8 * 3.0 + pi * 1.5**2, rel=5e-3)


def test_a_stopped_body_stays_on_its_own_side_of_a_median() -> None:
    """A parked car must not buffer across a median into a lane it cannot drive to."""
    median = box(-100.0, -4.0, 200.0, 0.0).union(box(-100.0, 0.5, 200.0, 4.5))
    ego = Agent.wheeled(0.0, -2.0, heading=0.0, speed=10.0, length=4.8, width=2.4)
    parked = Agent.wheeled(40.0, 0.7, heading=pi, speed=0.0, length=4.8, width=2.4)

    body = reachable_set(parked, 1.0, PARAMS, median)

    assert body.intersection(box(-100.0, -4.0, 200.0, 0.0)).is_empty
    assert EgoReachability(ego, median, PARAMS).time_to_collision(parked) == inf


def test_a_stopped_agent_off_the_surface_has_no_body() -> None:
    """Off-surface is infeasible for a stopped agent exactly as it is for a moving one."""
    parked = Agent.wheeled(20.0, 70.0, heading=0.0, speed=0.0, length=4.8, width=2)

    assert reachable_set(parked, 1.0, PARAMS, ROAD).is_empty


def test_a_hairline_map_seam_does_not_shrink_the_front() -> None:
    """Abutting ways rounded apart by a nanometre must not reject crossing arcs."""
    seamed = box(-80.0, -3.5, 400.0, 0.0).union(box(-80.0, 1e-9, 400.0, 3.5))
    seamless = box(-80.0, -3.5, 400.0, 3.5)
    ego = Agent.wheeled(0.0, -1.75, heading=0.0, speed=10.0, length=4.8, width=2)

    seamed_area = EgoReachability(ego, seamed, PARAMS)._reachable_set(30).area
    seamless_area = EgoReachability(ego, seamless, PARAMS)._reachable_set(30).area

    assert seamed_area == pytest.approx(seamless_area, rel=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": AgentKind.WHEELED, "speed": 10.0, "half_width": 0.0},
        {"kind": AgentKind.WHEELED, "speed": 10.0, "half_width": float("nan")},
        {"kind": AgentKind.WHEELED, "speed": float("nan"), "half_length": 1.0, "half_width": 1.0},
        {
            "kind": AgentKind.WHEELED,
            "heading": float("nan"),
            "speed": 1.0,
            "half_length": 1.0,
            "half_width": 1.0,
        },
        {
            "kind": AgentKind.STATIC,
            "speed": -1.0,
            "half_length": 1.0,
            "half_width": 1.0,
            "footprint": box(0.0, 0.0, 1.0, 1.0),
        },
        {
            "kind": AgentKind.WHEELED,
            "speed": 1.0,
            "half_length": 1.0,
            "half_width": 1.0,
            "footprint": box(0.0, 0.0, 1.0, 1.0),
        },
        {
            "kind": AgentKind.STATIC,
            "speed": 0.0,
            "half_length": 1.0,
            "half_width": 1.0,
            "footprint": Polygon(),
        },
        {"kind": "wheeled", "speed": 10.0, "half_length": 1.0, "half_width": 1.0},
    ],
)
def test_an_unusable_agent_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Agent(
            **{
                "x": 0.0,
                "y": 0.0,
                "heading": 0.0,
                "speed": 0.0,
                "half_length": 1.0,
                "half_width": 1.0,
                **kwargs,
            }
        )


def test_even_curvature_sampling_is_rejected() -> None:
    """An even count skips zero curvature, so the straight path is never tested."""
    with pytest.raises(ValueError, match="odd"):
        ReachabilityParams(arc_samples=20)


def test_a_nan_ttc_is_not_read_as_zero_risk() -> None:
    with pytest.raises(ValueError, match="NaN"):
        collision_weights([float("nan"), 1.0], 0.1)
    with pytest.raises(ValueError, match="finite"):
        collision_weights([1.0], float("nan"))
