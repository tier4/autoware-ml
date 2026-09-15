"""Unit tests for the reachability time-to-collision engine."""

from __future__ import annotations

from math import cos, hypot, inf, isclose, pi, sin, tan

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, Polygon, box

from autoware_ml.metrics.geometry.reachability import (
    Agent,
    EgoReachability,
    ReachabilityParams,
    VehicleGeometry,
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
# A single 6 m lane, so most arcs leave the surface and are truncated.
LANE = box(-80.0, -3.0, 500.0, 3.0)
# A bus measured at its rear axle, and a centred passenger car box.
BUS = {"front": 5.71, "rear": 1.53, "width": 2.29, "min_turn_radius": 6.4}
CAR = {"front": 2.4, "rear": 2.4, "width": 2.0, "min_turn_radius": 3.0}


def _footprint(x: float, y: float, size: float = 2.0):
    return box(x - size / 2, y - size / 2, x + size / 2, y + size / 2)


def _sampled_agent(kind: AgentKind, x: float, y: float, rng: np.random.Generator) -> Agent:
    """One random agent of ``kind`` at ``(x, y)``, for the brute-force comparison."""
    if kind == AgentKind.STATIC:
        return Agent.static(x, y, footprint=_footprint(x, y))
    if kind == AgentKind.LIVING:
        return Agent.living(x, y, speed=float(rng.uniform(0.0, 6.0)), radius=0.4)
    return Agent.wheeled(
        x,
        y,
        heading=float(rng.uniform(0, 2 * pi)),
        speed=float(rng.uniform(0.5, 16.7)),
        front=2.4,
        rear=2.4,
        width=2.0,
        min_turn_radius=3.0,
    )


def test_same_speed_lead_still_collides_in_the_worst_case() -> None:
    # Matched speed is no protection: the lead can brake or reverse while ego
    # keeps going, so the gap closes at the sum of the two worst-case speeds.
    ego = Agent.wheeled(
        0.0, 0.0, heading=0.0, speed=10.0, front=2.4, rear=2.4, width=2.4, min_turn_radius=3.0
    )
    lead = Agent.wheeled(
        25.0, 0.0, heading=0.0, speed=10.0, front=2.4, rear=2.4, width=2.4, min_turn_radius=3.0
    )
    ttc = time_to_collision(ego, lead, ROAD, PARAMS)
    # The 20.2 m between the two bodies close at 20 m/s in 1.01 s, and the corners
    # of the slightly turning bodies swing out enough to take the last 0.2 m.
    assert 1.0 <= ttc <= 1.1


def test_a_stationary_object_ahead_collides_at_about_distance_over_speed() -> None:
    ego = Agent.wheeled(
        0.0, 0.0, heading=0.0, speed=10.0, front=2.4, rear=2.4, width=2, min_turn_radius=3.0
    )
    obj = Agent.static(30.0, 0.0, footprint=_footprint(30.0, 0.0))
    ttc = time_to_collision(ego, obj, ROAD, PARAMS)
    assert ttc != inf
    assert 2.4 <= ttc <= 3.1  # ~ (30 - body - half-footprint) / 10


def test_oncoming_closes_at_combined_speed() -> None:
    ego = Agent.wheeled(
        0.0, 0.0, heading=0.0, speed=10.0, front=2.4, rear=2.4, width=2, min_turn_radius=3.0
    )
    obj = Agent.wheeled(
        40.0, 0.0, heading=pi, speed=10.0, front=2.4, rear=2.4, width=2, min_turn_radius=3.0
    )
    ttc = time_to_collision(ego, obj, ROAD, PARAMS)
    assert ttc != inf
    assert 1.7 <= ttc <= 2.1  # ~ 40 / (10 + 10)


def test_oncoming_beyond_ego_reach_still_collides() -> None:
    # The object's approach path is checked on the full drivable surface localized to
    # its own reach, so an incoming vehicle starting outside ego's reach clip is found.
    ego = Agent.wheeled(
        0.0, 0.0, heading=0.0, speed=10.0, front=2.4, rear=2.4, width=2, min_turn_radius=3.0
    )
    obj = Agent.wheeled(
        50.0, 0.0, heading=pi, speed=10.0, front=2.4, rear=2.4, width=2, min_turn_radius=3.0
    )
    ttc = time_to_collision(ego, obj, ROAD, PARAMS)
    assert 2.1 <= ttc <= 2.4  # ~ (50 - 2 * 3.4 body reach) / 20


def test_crossing_living_agent_is_finite_within_horizon() -> None:
    ego = Agent.wheeled(
        0.0, 0.0, heading=0.0, speed=10.0, front=2.4, rear=2.4, width=2, min_turn_radius=3.0
    )
    ped = Agent.living(18.0, 6.0, speed=4.0, radius=0.4)
    ttc = time_to_collision(ego, ped, ROAD, PARAMS)
    assert ttc != inf and ttc <= PARAMS.horizon_s


def test_a_far_object_is_unreachable_within_the_horizon() -> None:
    ego = Agent.wheeled(
        0.0, 0.0, heading=0.0, speed=10.0, front=2.4, rear=2.4, width=2, min_turn_radius=3.0
    )
    obj = Agent.static(200.0, 0.0, footprint=_footprint(200.0, 0.0))
    assert time_to_collision(ego, obj, ROAD, PARAMS) == inf


def test_a_wheeled_set_needs_a_drivable_surface() -> None:
    ego = Agent.wheeled(
        0.0, 0.0, heading=0.0, speed=10.0, front=2.4, rear=2.4, width=2, min_turn_radius=3.0
    )
    obj = Agent.static(20.0, 0.0, footprint=_footprint(20.0, 0.0))
    with pytest.raises(ValueError, match="drivable"):
        time_to_collision(ego, obj, None, PARAMS)


def test_disconnected_road_is_unreachable() -> None:
    # Two drivable strips separated by a non-drivable gap: an arc onto the other
    # strip crosses the gap, so the strips can never meet. A wheeled agent off the
    # surface entirely has no drivable arc at all.
    split_road = box(-80.0, -10.0, 200.0, 10.0).union(box(-80.0, 20.0, 200.0, 40.0))
    ego = Agent.wheeled(
        0.0, 0.0, heading=0.0, speed=10.0, front=2.4, rear=2.4, width=2, min_turn_radius=3.0
    )
    oncoming_across = Agent.wheeled(
        5.0, 30.0, heading=-pi / 2, speed=10.0, front=2.4, rear=2.4, width=2, min_turn_radius=3.0
    )
    assert time_to_collision(ego, oncoming_across, split_road, PARAMS) == inf
    off_road = Agent.wheeled(
        0.0, 60.0, heading=0.0, speed=10.0, front=2.4, rear=2.4, width=2, min_turn_radius=3.0
    )
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
    ego = Agent.wheeled(
        0.0, 0.0, heading=0.0, speed=1.0, front=2.4, rear=2.4, width=1, min_turn_radius=3.0
    )
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
        agent = Agent.wheeled(
            0.0, 0.0, heading=0.0, speed=speed, front=2.4, rear=2.4, width=2, min_turn_radius=3.0
        )
        region = wheeled_reachable_region(agent, PARAMS, ROAD)
        assert region.is_valid and not region.is_empty
        assert region.contains(Point(min(speed * PARAMS.horizon_s * 0.9, 10.0), 0.0))


@pytest.mark.parametrize(
    ("speed", "body", "objects"),
    [(13.9, CAR, 60), (10.0, BUS, 24), (3.0, BUS, 24)],
    ids=["car", "bus", "bus-at-the-radius-floor"],
)
def test_ego_reachability_matches_bruteforce_stepping(
    speed: float, body: dict, objects: int
) -> None:
    # The prescreen bounds are supersets of the true sets, so the optimized
    # engine must return exactly what naive stepping over every t returns. The
    # bus and the low speed are the cases where the shorter arcs used to poke out
    # of the horizon region.
    steps = int(PARAMS.horizon_s / PARAMS.dt_s + 1e-9)
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=speed, **body)
    ego_sets = [
        reachable_set(ego, index * PARAMS.dt_s, PARAMS, ROAD) for index in range(1, steps + 1)
    ]

    def brute_force(obj: Agent) -> float:
        for index, ego_set in enumerate(ego_sets, 1):
            t = index * PARAMS.dt_s
            obj_set = reachable_set(obj, t, PARAMS, ROAD)
            if not ego_set.is_empty and not obj_set.is_empty and ego_set.intersects(obj_set):
                return t
        return inf

    rng = np.random.default_rng(7)
    frame = EgoReachability(ego, ROAD, PARAMS)
    for index in range(objects):
        if index % 3 == 0:
            # Objects within a body radius of ego are the ones the prescreen
            # must never discard, so a third of the samples sit right there.
            x, y = float(rng.uniform(-2.0, 2.0)), float(rng.uniform(-2.0, 2.0))
        else:
            x, y = float(rng.uniform(-30, 130)), float(rng.uniform(-45, 45))
        kind = (AgentKind.WHEELED, AgentKind.LIVING, AgentKind.STATIC)[index % 3]
        obj = _sampled_agent(kind, x, y, rng)
        assert frame.time_to_collision(obj) == brute_force(obj), f"agent {index}: {obj}"


@pytest.mark.parametrize(
    ("speed", "body", "surface"),
    [(10.0, BUS, ROAD), (3.0, BUS, ROAD), (13.9, CAR, ROAD), (10.0, BUS, LANE)],
    ids=["bus", "bus-at-the-radius-floor", "car", "bus-in-a-narrow-lane"],
)
def test_the_region_covers_every_step_including_the_first(
    speed: float, body: dict, surface
) -> None:
    """The prescreen only holds if the hat is a superset of every step, right down to t = dt.

    An asymmetric body, the turn radius floor and arcs truncated at a lane edge are
    the cases where a sweep that only got its ends right let shorter arcs poke out.
    """
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=speed, **body)
    hat = wheeled_reachable_region(ego, PARAMS, surface)
    steps = int(PARAMS.horizon_s / PARAMS.dt_s + 1e-9)

    for step in range(1, steps + 1):
        reachable = wheeled_reachable_set(ego, step * PARAMS.dt_s, PARAMS, surface)
        assert reachable.difference(hat).area < 1e-9, f"step {step} escapes the hat"


def test_the_swept_body_covers_the_outer_front_corner_through_a_tight_turn() -> None:
    """The outer front corner of a turning bus runs far outside a strip around its axle path.

    Mid-turn on the tightest arc the corner is 1.9 m beyond where a buffered axle
    path ends, and the body is there at the time it takes to drive to that pose,
    so a static obstacle at the corner collides no later than that.
    """
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=3.0, **BUS)
    radius = PARAMS.turn_radius(ego)
    travel = 0.5 * ego.speed * PARAMS.horizon_s

    # Mid-turn pose on the tightest left arc, the turn centre is (0, radius).
    phi = travel / radius
    axle_x, axle_y = radius * sin(phi), radius * (1.0 - cos(phi))
    corner_x = axle_x + ego.front * cos(phi) + ego.half_width * sin(phi)
    corner_y = axle_y + ego.front * sin(phi) - ego.half_width * cos(phi)
    corner_radius = hypot(corner_x, corner_y - radius)
    outward = (corner_x / corner_radius, (corner_y - radius) / corner_radius)
    corner = Point(corner_x - 0.02 * outward[0], corner_y - 0.02 * outward[1])

    assert corner_radius > radius + ego.half_width + 1.9
    assert wheeled_reachable_set(ego, travel / ego.speed, PARAMS, ROAD).contains(corner)
    obstacle = Agent.static(corner.x, corner.y, corner.buffer(0.01))
    assert EgoReachability(ego, ROAD, PARAMS).time_to_collision(obstacle) <= travel / ego.speed


@pytest.mark.parametrize(
    ("speed", "body"),
    [
        (16.7, BUS),
        (16.7, {"front": 1.0, "rear": 1.0, "width": 0.8, "min_turn_radius": 3.0}),
        (3.0, BUS),
    ],
    ids=["bus", "motorcycle", "bus-at-the-radius-floor"],
)
def test_neighbouring_arcs_overlap_at_their_far_ends(speed: float, body: dict) -> None:
    """The outer edge of the region has no holes between the sampled arcs.

    With a fixed 21 arcs the far ends sit about 2.4 m apart whatever the speed,
    wider than a bus and three times a motorcycle, so the count has to follow the
    body. A densely sampled region is the reference: what it adds must be shallow
    notches along the far edge, never a slot between two sweeps.
    """
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=speed, **body)
    dense = ReachabilityParams(arc_samples=401)

    hat = wheeled_reachable_region(ego, PARAMS, ROAD)
    missing = wheeled_reachable_region(ego, dense, ROAD).difference(hat)

    assert missing.area < 0.03 * hat.area
    for notch in shapely.get_parts(missing):
        depth = max(hat.distance(Point(x, y)) for x, y in shapely.get_coordinates(notch))
        assert depth < 0.25 * 2.0 * ego.half_width


@pytest.mark.parametrize(
    ("x", "heading"),
    [(0.0, 0.0), (50.0, 0.0), (50.0, pi / 2)],
    ids=["reverse-leaves-at-once", "forward-leaves-at-once", "every-arc-leaves-at-once"],
)
def test_an_agent_at_the_road_edge_still_owns_its_body(x: float, heading: float) -> None:
    """Arcs that leave the surface at once add no travel, the start body stays."""
    road = box(0.0, -10.0, 50.0, 10.0)
    ego = Agent.wheeled(x, 0.0, heading=heading, speed=10.0, **CAR)

    reachable = wheeled_reachable_set(ego, PARAMS.horizon_s, PARAMS, road)

    assert reachable.contains(Point(x + 0.5 * (1.0 if x == 0.0 else -1.0), 0.0))
    assert reachable.difference(road).area < 1e-9


def test_the_body_stops_at_the_road_edge_not_one_step_short() -> None:
    """A truncated arc keeps the pose at the edge, so the body reaches the edge itself."""
    # Reversing at 16.7 m/s the poses are 1.67 m apart, more than the 1.53 m tail, so
    # without the edge pose the tail would stop 0.07 m short of the road end behind ego.
    end = -(5 * 1.67 + 1.6)
    road = box(end, -10.0, 100.0, 10.0)
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=16.7, **BUS)

    min_x, _, _, _ = wheeled_reachable_set(ego, PARAMS.horizon_s, PARAMS, road).bounds

    assert min_x == pytest.approx(end, abs=1e-6)


def test_a_one_lane_bend_is_not_overly_pessimistic() -> None:
    """Two legs of a one-lane junction: neither agent has to steer to meet.

    A single-arc motion model must still find this collision, otherwise the
    model is too pessimistic to trust in real intersections.
    """
    junction = box(-40.0, -2.0, 2.0, 2.0).union(box(-2.0, -40.0, 2.0, 2.0))
    ego = Agent.wheeled(
        -30.0, 0.0, heading=0.0, speed=10.0, front=2.4, rear=2.4, width=2.0, min_turn_radius=3.0
    )
    crossing = Agent.wheeled(
        0.0, -30.0, heading=pi / 2, speed=10.0, front=2.4, rear=2.4, width=2.0, min_turn_radius=3.0
    )

    ttc = time_to_collision(ego, crossing, junction, PARAMS)

    # Both need about (30 - 3.4 body reach) / 10 seconds to reach the corner.
    assert 2.4 <= ttc <= 3.0


def test_an_object_beside_ego_collides_at_the_first_step() -> None:
    """A pedestrian against ego's flank is the case the metric exists to score."""
    ego = Agent.wheeled(
        0.0, 0.0, heading=0.0, speed=10.0, front=2.4, rear=2.4, width=2.4, min_turn_radius=3.0
    )
    ped = Agent.living(0.1, 0.55, speed=1.4, radius=0.4)

    assert EgoReachability(ego, ROAD, PARAMS).time_to_collision(ped) == PARAMS.dt_s


def test_the_body_sweep_does_not_cross_a_narrow_median() -> None:
    """The swept body must not hop a gap thinner than the body itself."""
    median = box(-100.0, -4.0, 200.0, 0.0).union(box(-100.0, 0.5, 200.0, 4.5))
    ego = Agent.wheeled(
        0.0, -2.0, heading=0.0, speed=10.0, front=2.4, rear=2.4, width=2.4, min_turn_radius=3.0
    )
    oncoming = Agent.wheeled(
        40.0, 2.5, heading=pi, speed=10.0, front=2.4, rear=2.4, width=2.4, min_turn_radius=3.0
    )

    reachable = wheeled_reachable_set(ego, 2.0, PARAMS, median)

    assert reachable.intersection(box(-100.0, 0.5, 200.0, 4.5)).is_empty
    assert EgoReachability(ego, median, PARAMS).time_to_collision(oncoming) == inf


def test_a_stopped_agent_still_occupies_its_body() -> None:
    # Ego is measured at its rear axle, so the body reaches much further ahead of
    # the reference point than behind it.
    stopped = Agent.wheeled(
        20.0, 0.0, heading=0.0, speed=0.0, front=3.9, rear=1.0, width=3.0, min_turn_radius=3.0
    )

    body = reachable_set(stopped, 1.0, PARAMS, ROAD)

    # Exactly the body rectangle, nothing rounded off or added at the ends.
    rectangle = box(20.0 - 1.0, -1.5, 20.0 + 3.9, 1.5)
    assert body.symmetric_difference(rectangle).area < 1e-9


def test_the_leading_end_of_the_body_follows_the_travel_direction() -> None:
    # Forward the nose leads by the front extent, in reverse the tail leads by the
    # rear extent, so an ego measured at its rear axle reaches its reach plus 3.9 m
    # ahead but only its reach plus 1.0 m behind. A slightly turning body swings a
    # corner a few centimetres further along, hence the one-sided slack.
    ego = Agent.wheeled(
        0.0, 0.0, heading=0.0, speed=10.0, front=3.9, rear=1.0, width=2.0, min_turn_radius=3.0
    )

    reachable = wheeled_reachable_set(ego, 1.0, PARAMS, ROAD)
    min_x, _, max_x, _ = reachable.bounds

    assert reachable.contains(Point(10.0 + 3.9 - 0.01, 0.0))
    assert 10.0 + 3.9 <= max_x < 10.0 + 3.9 + 0.2
    assert reachable.contains(Point(-(10.0 + 1.0) + 0.01, 0.0))
    assert -(10.0 + 1.0) - 0.2 < min_x <= -(10.0 + 1.0)


def test_vehicle_geometry_is_measured_from_the_rear_axle() -> None:
    vehicle = VehicleGeometry(
        wheel_base=2.79,
        front_overhang=1.0,
        rear_overhang=1.1,
        wheel_tread=1.64,
        left_overhang=0.128,
        right_overhang=0.128,
        max_steer_angle=0.64,
    )

    assert vehicle.front == pytest.approx(3.79)
    assert vehicle.rear == pytest.approx(1.1)
    assert vehicle.width == pytest.approx(1.896)
    assert vehicle.min_turn_radius == pytest.approx(2.79 / tan(0.64))
    with pytest.raises(ValueError, match="wheel_base"):
        VehicleGeometry(0.0, 1.0, 1.0, 1.6, 0.1, 0.1, 0.64)
    with pytest.raises(ValueError, match="max_steer_angle"):
        VehicleGeometry(2.79, 1.0, 1.1, 1.64, 0.128, 0.128, 0.0)


def test_the_turn_radius_floor_belongs_to_the_agent() -> None:
    """At low speed the steering limit, not the friction bound, decides the reach.

    The same bus body with a 3 m friction floor (unknown steering) swings much
    wider than with its 6.4 m steering limit, so the two must not share one floor.
    """
    unknown = Agent.wheeled(0.0, 0.0, heading=0.0, speed=3.0, **{**BUS, "min_turn_radius": 3.0})
    ego = Agent.wheeled(0.0, 0.0, heading=0.0, speed=3.0, **BUS)

    assert PARAMS.turn_radius(unknown) == pytest.approx(3.0)
    assert PARAMS.turn_radius(ego) == pytest.approx(6.4)
    _, unknown_min_y, _, unknown_max_y = wheeled_reachable_set(unknown, 2.0, PARAMS, ROAD).bounds
    _, ego_min_y, _, ego_max_y = wheeled_reachable_set(ego, 2.0, PARAMS, ROAD).bounds
    assert unknown_max_y - unknown_min_y > ego_max_y - ego_min_y + 1.0


def test_a_stopped_body_stays_on_its_own_side_of_a_median() -> None:
    """A parked car must not buffer across a median into a lane it cannot drive to."""
    median = box(-100.0, -4.0, 200.0, 0.0).union(box(-100.0, 0.5, 200.0, 4.5))
    ego = Agent.wheeled(
        0.0, -2.0, heading=0.0, speed=10.0, front=2.4, rear=2.4, width=2.4, min_turn_radius=3.0
    )
    parked = Agent.wheeled(
        40.0, 0.7, heading=pi, speed=0.0, front=2.4, rear=2.4, width=2.4, min_turn_radius=3.0
    )

    body = reachable_set(parked, 1.0, PARAMS, median)

    assert body.intersection(box(-100.0, -4.0, 200.0, 0.0)).is_empty
    assert EgoReachability(ego, median, PARAMS).time_to_collision(parked) == inf


def test_a_stopped_agent_off_the_surface_has_no_body() -> None:
    """Off-surface is infeasible for a stopped agent exactly as it is for a moving one."""
    parked = Agent.wheeled(
        20.0, 70.0, heading=0.0, speed=0.0, front=2.4, rear=2.4, width=2, min_turn_radius=3.0
    )

    assert reachable_set(parked, 1.0, PARAMS, ROAD).is_empty


def test_a_hairline_map_seam_does_not_shrink_the_front() -> None:
    """Abutting ways rounded apart by a nanometre must not reject crossing arcs."""
    seamed = box(-80.0, -3.5, 400.0, 0.0).union(box(-80.0, 1e-9, 400.0, 3.5))
    seamless = box(-80.0, -3.5, 400.0, 3.5)
    ego = Agent.wheeled(
        0.0, -1.75, heading=0.0, speed=10.0, front=2.4, rear=2.4, width=2, min_turn_radius=3.0
    )

    seamed_area = EgoReachability(ego, seamed, PARAMS)._reachable_set(30).area
    seamless_area = EgoReachability(ego, seamless, PARAMS)._reachable_set(30).area

    assert seamed_area == pytest.approx(seamless_area, rel=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": AgentKind.WHEELED, "speed": 10.0, "half_width": 0.0},
        {"kind": AgentKind.WHEELED, "speed": 10.0, "half_width": float("nan")},
        {
            "kind": AgentKind.WHEELED,
            "speed": float("nan"),
            "front": 1.0,
            "rear": 1.0,
            "half_width": 1.0,
            "min_turn_radius": 3.0,
        },
        {
            "kind": AgentKind.WHEELED,
            "heading": float("nan"),
            "speed": 1.0,
            "front": 1.0,
            "rear": 1.0,
            "half_width": 1.0,
            "min_turn_radius": 3.0,
        },
        {
            "kind": AgentKind.STATIC,
            "speed": -1.0,
            "front": 1.0,
            "rear": 1.0,
            "half_width": 1.0,
            "min_turn_radius": 3.0,
            "footprint": box(0.0, 0.0, 1.0, 1.0),
        },
        {
            "kind": AgentKind.WHEELED,
            "speed": 1.0,
            "front": 1.0,
            "rear": 1.0,
            "half_width": 1.0,
            "min_turn_radius": 3.0,
            "footprint": box(0.0, 0.0, 1.0, 1.0),
        },
        {
            "kind": AgentKind.STATIC,
            "speed": 0.0,
            "front": 1.0,
            "rear": 1.0,
            "half_width": 1.0,
            "min_turn_radius": 3.0,
            "footprint": Polygon(),
        },
        {
            "kind": "wheeled",
            "speed": 10.0,
            "front": 1.0,
            "rear": 1.0,
            "half_width": 1.0,
            "min_turn_radius": 3.0,
        },
        {"kind": AgentKind.WHEELED, "speed": 10.0, "min_turn_radius": 0.0},
        {"kind": AgentKind.LIVING, "speed": 1.0, "min_turn_radius": 3.0},
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
                "front": 1.0,
                "rear": 1.0,
                "half_width": 1.0,
                "min_turn_radius": 3.0,
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
