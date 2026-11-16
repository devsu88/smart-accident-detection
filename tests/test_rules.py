from smart_accident.config import RuleConfig
from smart_accident.events import detect_incidents

from .scenes import frames, moving

ROAD = [[0, 0], [800, 0], [800, 200], [0, 200]]


def kinds(incidents):
    return [i["type"] for i in incidents]


def test_head_on_collision_is_detected_shortly_after_contact():
    # Cars close at 200 px/s, first touch at t=6.0, both halt dead at 6.3.
    scene = frames({1: moving(0, 100, 100, stop_at=6.3), 2: moving(1260, 100, -100, stop_at=6.3)}, 12)
    incidents = detect_incidents(scene)
    assert kinds(incidents) == ["collision"]
    assert 6.0 <= incidents[0]["t"] <= 7.5
    assert sorted(incidents[0]["ids"]) == [1, 2]


def test_rear_end_collision_with_one_vehicle_already_slowing():
    # Fast A catches slow B at t=5.0 and both stop dead overlapped at t=5.2.
    scene = frames({1: moving(0, 100, 150, stop_at=5.2), 2: moving(610, 100, 40, stop_at=5.2)}, 12)
    assert kinds(detect_incidents(scene)) == ["collision"]


def test_pass_by_occlusion_at_constant_speed_is_not_an_incident():
    scene = frames({1: moving(0, 100, 100), 2: moving(-100, 112, 160)}, 12)
    assert detect_incidents(scene) == []


def test_overlap_in_a_slow_jam_is_not_an_incident():
    # Both crawl at ~0.17 box-heights/s, below the "was really moving" threshold, and touch.
    scene = frames({1: moving(0, 100, 5), 2: moving(70, 100, 3)}, 30)
    assert detect_incidents(scene) == []


def test_one_vehicle_braking_alone_is_not_a_collision():
    # A brakes to a stop but B keeps going at full speed through the overlap.
    scene = frames({1: moving(0, 100, 100, stop_at=6.3), 2: moving(1260, 100, -100)}, 12)
    assert detect_incidents(scene) == []


def test_one_event_per_pair_inside_the_cooldown():
    scene = frames({1: moving(0, 100, 100, stop_at=6.3), 2: moving(1260, 100, -100, stop_at=6.3)}, 14)
    assert len(detect_incidents(scene)) == 1


def test_young_tracks_are_ignored():
    # Both appear 0.2 s before halting: too young to trust their speed.
    scene = frames({1: moving(600, 100, 0, start_at=6.0), 2: moving(620, 100, 0, start_at=6.0)}, 8)
    assert detect_incidents(scene) == []


def test_stopped_vehicle_inside_road_triggers_after_the_threshold():
    rules = RuleConfig(road_polygon=ROAD, stopped_s=5.0)
    scene = frames({1: moving(100, 100, 100, stop_at=4.0)}, 14)
    incidents = detect_incidents(scene, rules)
    assert kinds(incidents) == ["stopped_vehicle"]
    assert incidents[0]["t"] >= 4.0 + 5.0


def test_vehicle_stopped_outside_the_road_polygon_is_ignored():
    rules = RuleConfig(road_polygon=ROAD, stopped_s=5.0)
    scene = frames({1: moving(100, 400, 100, stop_at=4.0)}, 14)
    assert detect_incidents(scene, rules) == []


def test_several_stopped_vehicles_are_a_jam_not_an_incident():
    rules = RuleConfig(road_polygon=ROAD, stopped_s=5.0, jam_count=3)
    scene = frames({i: moving(100 + 150 * i, 100, 0) for i in range(4)}, 14)
    assert detect_incidents(scene, rules) == []


def test_stopped_rule_is_off_without_a_road_polygon():
    scene = frames({1: moving(100, 100, 100, stop_at=4.0)}, 20)
    assert detect_incidents(scene) == []


def test_replaying_is_deterministic():
    scene = frames({1: moving(0, 100, 100, stop_at=6.3), 2: moving(1260, 100, -100, stop_at=6.3)}, 12)
    assert detect_incidents(scene) == detect_incidents(scene)


def test_detector_jitter_does_not_create_false_collisions():
    import random

    rng = random.Random(7)

    def jittery(position):
        def noisy(t):
            x, y = position(t)
            return (x + rng.uniform(-2, 2), y + rng.uniform(-2, 2))
        return noisy

    # Two cars overlap while passing at constant speed, with a few pixels of box noise.
    scene = frames({1: jittery(moving(0, 100, 100)), 2: jittery(moving(-100, 112, 160))}, 12)
    assert detect_incidents(scene) == []


def test_collision_is_still_found_with_detector_jitter():
    import random

    rng = random.Random(3)

    def jittery(position):
        def noisy(t):
            x, y = position(t)
            return (x + rng.uniform(-2, 2), y + rng.uniform(-2, 2))
        return noisy

    scene = frames({1: jittery(moving(0, 100, 100, stop_at=6.3)),
                    2: jittery(moving(1260, 100, -100, stop_at=6.3))}, 12)
    assert [i["type"] for i in detect_incidents(scene)] == ["collision"]
