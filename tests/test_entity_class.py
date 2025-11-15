import pytest

from buds.base import Entity


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
