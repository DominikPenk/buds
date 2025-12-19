import pytest

from buds.base import Entity, Trait


class DummyWorld:
    def __init__(self, id: int):
        self._id = id

    def __eq__(self, other):
        return self._id == other._id


def test_int_conversion():
    e = Entity(42, None)
    assert int(e) == 42


@pytest.mark.parametrize(
    "e1, e2, expected",
    [
        (Entity(42, DummyWorld(1)), Entity(42, DummyWorld(1)), True),
        (Entity(42, DummyWorld(1)), Entity(42, DummyWorld(2)), False),
        (Entity(13, DummyWorld(1)), Entity(42, DummyWorld(1)), False),
        (Entity(13, DummyWorld(1)), Entity(42, DummyWorld(2)), False),
    ],
)
def test_equality(e1: Entity, e2: Entity, expected: bool):
    assert (e1 == e2) == expected


@pytest.mark.parametrize(
    "e1, e2, expected",
    [
        (Entity(42, DummyWorld(1)), Entity(42, DummyWorld(1)), False),
        (Entity(42, DummyWorld(1)), Entity(42, DummyWorld(2)), True),
        (Entity(13, DummyWorld(1)), Entity(42, DummyWorld(1)), True),
        (Entity(13, DummyWorld(1)), Entity(42, DummyWorld(2)), True),
    ],
)
def test_non_equality(e1: Entity, e2: Entity, expected: bool):
    assert (e1 != e2) == expected


@pytest.mark.parametrize(
    "e1, e2, expected",
    [
        (Entity(42, DummyWorld(1)), Entity(42, DummyWorld(1)), False),
        (Entity(13, DummyWorld(1)), Entity(42, DummyWorld(1)), False),
        (Entity(42, DummyWorld(1)), Entity(13, DummyWorld(1)), True),
    ],
)
def test_gt(e1: Entity, e2: Entity, expected: bool):
    assert (e1 > e2) == expected


@pytest.mark.parametrize(
    "e1, e2, expected",
    [
        (Entity(42, DummyWorld(1)), Entity(42, DummyWorld(1)), True),
        (Entity(13, DummyWorld(1)), Entity(42, DummyWorld(1)), False),
        (Entity(42, DummyWorld(1)), Entity(13, DummyWorld(1)), True),
    ],
)
def test_ge(e1: Entity, e2: Entity, expected: bool):
    assert (e1 >= e2) == expected


@pytest.mark.parametrize(
    "e1, e2, expected",
    [
        (Entity(42, DummyWorld(1)), Entity(42, DummyWorld(1)), False),
        (Entity(13, DummyWorld(1)), Entity(42, DummyWorld(1)), True),
        (Entity(42, DummyWorld(1)), Entity(13, DummyWorld(1)), False),
    ],
)
def test_lt(e1: Entity, e2: Entity, expected: bool):
    assert (e1 < e2) == expected


@pytest.mark.parametrize(
    "e1, e2, expected",
    [
        (Entity(42, DummyWorld(1)), Entity(42, DummyWorld(1)), True),
        (Entity(13, DummyWorld(1)), Entity(42, DummyWorld(1)), True),
        (Entity(42, DummyWorld(1)), Entity(13, DummyWorld(1)), False),
    ],
)
def test_le(e1: Entity, e2: Entity, expected: bool):
    assert (e1 <= e2) == expected


# ---------------------------------------------------------------------------
# Test convenience methods
# ---------------------------------------------------------------------------
class TestTrait(Trait):
    pass


def test_add_and_remove_trait(world):
    e: Entity = world.create_entity()
    e.add_trait(TestTrait())
    assert world.has_trait(e.id, TestTrait)

    e.remove_trait(TestTrait)
    assert not world.has_trait(e.id, TestTrait)


def test_has_trait(world):
    e: Entity = world.create_entity(TestTrait())
    assert e.has_trait(TestTrait)


def test_add_and_remove_tags(world):
    e: Entity = world.create_entity()
    e.add_tags("alpha", "beta")
    assert world.has_tags(e.id, "alpha")
    assert world.has_tags(e.id, "beta")
    assert world.has_tags(e.id, "alpha", "beta")

    e.remove_tags("beta")
    assert world.has_tags(e.id, "alpha")
    assert not world.has_tags(e.id, "beta")
    assert not world.has_tags(e.id, "alpha", "beta")

    e.remove_tags("alpha")
    assert not world.has_tags(e.id, "alpha")
    assert not world.has_tags(e.id, "beta")
    assert not world.has_tags(e.id, "alpha", "beta")


def test_has_tags(world):
    e: Entity = world.create_entity(TestTrait())
    e.add_tags("alpha", "beta")
    assert e.has_tags("alpha")
    assert e.has_tags("beta")
    assert e.has_tags("alpha", "beta")
    assert not e.has_tags("INVALID")
    assert not e.has_tags("alpha", "INVALID")


def test_lifecycle(world):
    e: Entity = world.create_entity(TestTrait())
    assert e.is_alive()

    e.delete()
    assert not e.is_alive()
    assert not world.is_alive(e.id)
