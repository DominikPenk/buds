# tests/test_traits.py

from buds.base import TRAIT_HINT, is_trait, is_trait_type

# ----------------------------------------
# Helper classes for testing
# ----------------------------------------


class NotATraitClass:
    pass


# ----------------------------------------
# is_trait / is_trait_type tests
# ----------------------------------------


def test_is_trait_and_is_trait_type_behavior(backend):
    class MyTrait(backend.Trait):
        x: int
        y: int

    instance = MyTrait(x=10, y=20)
    # Recognized as trait instance
    assert is_trait(instance) is True
    # Not a random object
    assert is_trait(object()) is False
    # Recognized as trait class
    assert is_trait_type(MyTrait) is True
    # Not a non-class
    assert is_trait_type(instance) is False
    # Not a normal class
    assert is_trait_type(NotATraitClass) is False


# ----------------------------------------
# Trait base class behavior
# ----------------------------------------


def test_subclassing_trait_marks_as_trait(backend):
    class MyNewTrait(backend.Trait):
        x: int
        y: int

    # Class is converted to dataclass and has __is_trait
    assert hasattr(MyNewTrait, TRAIT_HINT)
    assert getattr(MyNewTrait, TRAIT_HINT) is True

    # Instances recognized
    instance = MyNewTrait(x=1, y=2)
    assert is_trait(instance)
    assert is_trait_type(MyNewTrait)


def test_trait_subclass_can_inherit_multiple_levels(backend):
    class Base(backend.Trait):
        x: int

    class Derived(Base):
        y: int

    assert is_trait_type(Derived)
    assert is_trait_type(Base)
    inst = Derived(x=1, y=2)
    assert is_trait(inst)


# ----------------------------------------
# Trait base world integration
# ----------------------------------------


def test_trait_roundtrip(world, backend):
    """A simple round-trip test: create entities with MSGSpecTrait traits and read them back."""

    class P(backend.Trait):
        x: float
        y: float

    class V(backend.Trait):
        dx: float
        dy: float

    e = world.create_entity(P(x=1.0, y=2.0), V(dx=3.0, dy=4.0))
    e2 = world.create_entity(P(x=3.0, y=4.0), V(dx=5.0, dy=6.0))

    results = list(world.get_entities(P, V))
    assert len(results) == 2
    ent, (p, v) = results[0]
    assert (p.x, p.y) == (1.0, 2.0)
    assert (v.dx, v.dy) == (3.0, 4.0)


def test_add_and_remove_trait(world, backend):
    """Verify add_trait and remove_trait work with msgspec-based traits."""

    class Health(backend.Trait):
        value: int

    e = world.create_entity()

    # add the trait after creation
    world.add_trait(e.id, Health(value=10))
    got = list(world.get_entities(Health))
    assert len(got) == 1
    _, health = got[0]
    # get_entities(single_trait) yields (Entity, trait)
    assert health.value == 10

    # remove and ensure it's gone
    world.remove_trait(e.id, Health)
    assert list(world.get_entities(Health)) == []
