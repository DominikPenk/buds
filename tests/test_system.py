import pytest

from buds.system import system, map  # Module you gave
from buds.base import Entity, trait, World
import math


# Shared trait types used by the contract tests
@trait
class Position:
    x: int
    y: int


@trait
class Velocity:
    dx: float
    dy: float


def populate_world(world) -> World:
    """
    Provides a fresh world instance pre-populated with a few entities
    containing Position and Velocity traits and various tags.
    """
    e1 = world.create_entity(Position(0, 1), Velocity(1, 2))
    e2 = world.create_entity(Position(2, 3), Velocity(3, 4))
    e3 = world.create_entity(Position(4, 5), Velocity(5, 6))
    e1.add_tags("player")
    e2.add_tags("npc")
    e3.add_tags("npc", "boss")

    return world


# ---------------------------------------------------------------------------
# Test @system decorator queries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tags, expected",
    [
        (
            {
                "player",
            },
            [Position(1, 3)],
        ),
        (
            {
                "npc",
            },
            [Position(5, 7), Position(9, 11)],
        ),
        (
            {"npc", "boss"},
            [Position(9, 11)],
        ),
    ],
)
def test_system_update_traits(world, tags: set[str], expected: list[Position]):
    populated_world = populate_world(world)

    @system(*tags)
    def move(pos: Position, vel: Velocity):
        pos.x += vel.dx
        pos.y += vel.dy

    move(world)

    results = list(populated_world.get_entities(Position, tags=tags))

    assert len(results) == len(expected)

    for (eid, pos), expected_pos in zip(
        populated_world.get_entities(Position, tags=tags), expected
    ):
        assert eid.has_tags(*tags)
        assert pos.x == pytest.approx(expected_pos.x)
        assert pos.y == pytest.approx(expected_pos.y)


def test_system_without_bracketed_tags(world):
    populated_world = populate_world(world)

    @system
    def move(pos: Position, vel: Velocity):
        pos.x += vel.dx
        pos.y += vel.dy

    move(world)

    results = list(populated_world.get_entities(Position))

    assert len(results) == 3

    expected_positions = [Position(1, 3), Position(5, 7), Position(9, 11)]

    for (eid, pos), expected_pos in zip(
        populated_world.get_entities(Position), expected_positions
    ):
        assert pos.x == pytest.approx(expected_pos.x)
        assert pos.y == pytest.approx(expected_pos.y)


def test_system_without_tags(world):
    populated_world = populate_world(world)

    @system()
    def move(pos: Position, vel: Velocity):
        pos.x += vel.dx
        pos.y += vel.dy

    move(world)

    results = list(populated_world.get_entities(Position))

    assert len(results) == 3

    expected_positions = [Position(1, 3), Position(5, 7), Position(9, 11)]

    for (eid, pos), expected_pos in zip(
        populated_world.get_entities(Position), expected_positions
    ):
        assert pos.x == pytest.approx(expected_pos.x)
        assert pos.y == pytest.approx(expected_pos.y)


def test_sytem_with_external_args(world):
    populated_world = populate_world(world)

    offset_x = 10
    offset_y = 20

    @system("npc")
    def move(pos: Position, vel: Velocity, ox: int, oy: int):
        pos.x += vel.dx + ox
        pos.y += vel.dy + oy

    move(world, offset_x, offset_y)

    results = list(populated_world.get_entities(Position, tags={"npc"}))

    assert len(results) == 2

    expected_positions = [Position(15, 27), Position(19, 31)]

    for (eid, pos), expected_pos in zip(
        populated_world.get_entities(Position, tags={"npc"}), expected_positions
    ):
        assert eid.has_tags("npc")
        assert pos.x == pytest.approx(expected_pos.x)
        assert pos.y == pytest.approx(expected_pos.y)


def test_system_forwrads_entity(world):
    populated_world = populate_world(world)

    @system("player")
    def move(entity: Entity, pos: Position, vel: Velocity):
        pos.x += vel.dx
        pos.y += vel.dy
        # Add a new tag to the entity
        entity.add_tags("moved")

    move(world)

    results = list(populated_world.get_entities(Position, tags={"player"}))

    assert len(results) == 1

    for eid, pos in populated_world.get_entities(Position, tags={"player"}):
        assert eid.has_tags("player", "moved")
        assert pos.x == pytest.approx(1)
        assert pos.y == pytest.approx(3)


@pytest.mark.parametrize(
    "tags, expected",
    [
        (
            {
                "player",
            },
            [Position(1, 3)],
        ),
        (
            {
                "npc",
            },
            [Position(5, 7), Position(9, 11)],
        ),
        (
            {"npc", "boss"},
            [Position(9, 11)],
        ),
    ],
)
def test_system_decorator_for_method(world, tags, expected):
    populated_world = populate_world(world)

    class Mover:
        @system(*tags)
        def move(self, pos: Position, vel: Velocity):
            pos.x += vel.dx
            pos.y += vel.dy

    mover = Mover()
    mover.move(world)

    results = list(populated_world.get_entities(Position, tags=tags))

    assert len(results) == len(expected)

    for (eid, pos), expected_pos in zip(
        populated_world.get_entities(Position, tags=tags), expected
    ):
        assert eid.has_tags(*tags)
        assert pos.x == pytest.approx(expected_pos.x)
        assert pos.y == pytest.approx(expected_pos.y)


# ---------------------------------------------------------------------------
# Test @map decorator
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tags, expected",
    [
        (
            {},
            [math.hypot(1.0, 2.0), math.hypot(3.0, 4.0), math.hypot(5.0, 6.0)],
        ),
        (
            {
                "player",
            },
            [math.hypot(1.0, 2.0)],
        ),
        (
            {
                "npc",
            },
            [math.hypot(3.0, 4.0), math.hypot(5.0, 6.0)],
        ),
        (
            {"npc", "boss"},
            [math.hypot(5.0, 6.0)],
        ),
    ],
)
def test_map_decorator_tag_filters(world: World, tags: set[str], expected: float):
    populated_world = populate_world(world)

    @map(*tags)
    def get_velocity(vel: Velocity) -> float:
        return math.hypot(vel.dx, vel.dy)

    results = list(get_velocity(populated_world))

    assert len(results) == len(expected)

    for vel, expected_vel in zip(results, expected):
        assert vel == pytest.approx(expected_vel)


def test_map_decorator_without_tags(world: World):
    populated_world = populate_world(world)

    @map
    def get_velocity(vel: Velocity) -> float:
        return math.hypot(vel.dx, vel.dy)

    results = list(get_velocity(populated_world))

    assert len(results) == 3

    expected_velocities = [
        math.hypot(1.0, 2.0),
        math.hypot(3.0, 4.0),
        math.hypot(5.0, 6.0),
    ]

    for vel, expected_vel in zip(results, expected_velocities):
        assert vel == pytest.approx(expected_vel)


@pytest.mark.parametrize(
    "tags, scale, expected_velocities",
    [
        (
            {},
            0.1,
            [
                math.hypot(1.0, 2.0) * 0.1,
                math.hypot(3.0, 4.0) * 0.1,
                math.hypot(5.0, 6.0) * 0.1,
            ],
        ),
        (
            {"npc"},
            0.5,
            [
                math.hypot(3.0, 4.0) * 0.5,
                math.hypot(5.0, 6.0) * 0.5,
            ],
        ),
        (
            {"npc", "boss"},
            2.0,
            [
                math.hypot(5.0, 6.0) * 2.0,
            ],
        ),
    ],
)
def test_map_decorator_with_external_args(
    world, tags: set[str], scale: float, expected_velocities: list[float]
):
    populated_world = populate_world(world)

    @map(*tags)
    def get_scaled_velocity(vel: Velocity, scale: float):
        return math.hypot(vel.dx, vel.dy) * scale

    results = list(get_scaled_velocity(populated_world, scale=scale))

    assert len(results) == len(expected_velocities)

    for vel, expected_velocity in zip(results, expected_velocities):
        assert vel == pytest.approx(expected_velocity)
