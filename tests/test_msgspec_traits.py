import pytest

pytest.importorskip("msgspec")

from buds.extras.msgspec import MSGSpecTrait


def test_msgspec_trait_roundtrip(world):
    """A simple round-trip test: create entities with MSGSpecTrait traits and read them back."""

    class P(MSGSpecTrait):
        x: float
        y: float

    class V(MSGSpecTrait):
        dx: float
        dy: float

    e = world.create_entity(P(1.0, 2.0), V(3.0, 4.0))
    e2 = world.create_entity(P(3.0, 4.0), V(5.0, 6.0))

    results = list(world.get_entities(P, V))
    assert len(results) == 2
    ent, (p, v) = results[0]
    assert (p.x, p.y) == (1.0, 2.0)
    assert (v.dx, v.dy) == (3.0, 4.0)

    # runtime trait helpers
    # assert is_trait(p)
    # assert is_trait_type(P)


def test_msgspec_add_and_remove_trait(world):
    """Verify add_trait and remove_trait work with msgspec-based traits."""

    class Health(MSGSpecTrait):
        value: int

    e = world.create_entity()

    # add the trait after creation
    world.add_trait(e.id, Health(10))
    got = list(world.get_entities(Health))
    assert len(got) == 1
    _, health = got[0]
    # get_entities(single_trait) yields (Entity, trait)
    assert health.value == 10

    # remove and ensure it's gone
    world.remove_trait(e.id, Health)
    assert list(world.get_entities(Health)) == []
