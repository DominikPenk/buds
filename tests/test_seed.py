import pytest

from buds.base import Trait
from buds.seed import Seed

# --- Shared Traits -----------------------------------------------------


class Position(Trait):
    x: int
    y: int


class Velocity(Trait):
    dx: float
    dy: float


class Health(Trait):
    hp: int


# --- Example Seed Implementations -------------------------------------


class SimpleSeed(Seed):
    """A reusable seed defining a basic position + velocity setup."""

    def __init__(self):
        super().__init__("tagged", "default")
        self.pos = Position(1, 2)
        self.vel = Velocity(3.0, 4.0)


class MinimalSeed(Seed):
    """Seed with no predefined traits, just for lifecycle checks."""

    def __init__(self):
        super().__init__()


# --- Tests -------------------------------------------------------------


def test_seed_collects_traits_and_query(world):
    """Seed subclass should automatically collect trait instances and populate .query."""
    seed = SimpleSeed()

    # The collected ECS traits
    assert any(isinstance(t, Position) for t in seed._ecs_traits)
    assert any(isinstance(t, Velocity) for t in seed._ecs_traits)

    # The query set should include their types
    assert seed.query == {Position, Velocity}

    # Tags should be preserved
    assert set(seed.tags) == {"tagged", "default"}


def test_seed_spawn_creates_entity_and_registers_traits(world):
    seed = SimpleSeed()
    seed.spawn(world)

    assert seed.spawned
    assert seed.entity is not None
    assert world.is_alive(seed.entity.id)

    # Tags are applied
    assert world.has_tags(seed.entity.id, "tagged", "default")

    # Traits are present
    assert world.has_trait(seed.entity.id, Position)
    assert world.has_trait(seed.entity.id, Velocity)

    # Querying from world should yield matching data
    results = list(world.get_entities_from_traits(Position, Velocity))
    assert len(results) == 1
    ent, (pos, vel) = results[0]
    assert ent.id == seed.entity.id
    assert (pos.x, pos.y) == (1, 2)
    assert (vel.dx, vel.dy) == (3.0, 4.0)


def test_seed_despawn_removes_entity(world):
    seed = SimpleSeed().spawn(world)
    eid = seed.entity.id

    seed.despawn()

    assert not seed.spawned
    assert not world.is_alive(eid)
    assert seed.entity is None


def test_seed_spawn_twice_raises(world):
    seed = SimpleSeed().spawn(world)
    with pytest.raises(RuntimeError):
        seed.spawn(world)


def test_seed_despawn_before_spawn_raises(world):
    seed = SimpleSeed()
    with pytest.raises(RuntimeError):
        seed.despawn()


def test_seed_add_trait_after_spawn(world):
    seed = SimpleSeed().spawn(world)
    health = Health(100)

    seed.add_trait("health", health)
    assert hasattr(seed, "health")
    assert isinstance(seed.health, Health)
    assert world.has_trait(seed.entity.id, Health)

    # Should also update query set
    assert Health in seed.query


def test_seed_add_trait_invalid_raises(world):
    seed = SimpleSeed().spawn(world)
    with pytest.raises(ValueError):
        seed.add_trait("bad", object())


def test_seed_add_and_remove_tags_reflects_in_world(world):
    seed = SimpleSeed().spawn(world)

    seed.add_tags("extra", "bonus")
    assert world.has_tags(seed.entity.id, "extra", "bonus")

    seed.remove_tags("extra")
    assert not world.has_tags(seed.entity.id, "extra")
    assert world.has_tags(seed.entity.id, "bonus")


def test_seed_world_property_and_reference(world):
    seed = SimpleSeed().spawn(world)
    assert seed.world is world

    seed.despawn()
    assert seed.world is None


def test_minimal_seed_spawns_and_tags(world):
    """A seed with no predefined traits should still spawn and tag correctly."""
    seed = MinimalSeed()
    seed.add_tags("a", "b").spawn(world)

    assert seed.spawned
    assert seed.entity is not None
    assert world.has_tags(seed.entity.id, "a", "b")

    seed.despawn()
    assert not seed.spawned


def test_seed_add_trait_before_spawn(world):
    """Adding a trait before spawn should still attach it correctly upon spawn."""
    seed = MinimalSeed()
    seed.add_trait("health", Health(50))
    assert Health in seed.query

    seed.spawn(world)
    assert world.has_trait(seed.entity.id, Health)
    assert hasattr(seed, "health")
