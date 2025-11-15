# tests/test_traits.py
import pytest
import inspect
from dataclasses import is_dataclass
from buds.base import trait, make_trate_type, is_trait, is_trait_type, Trait, TRAIT_HINT

# ----------------------------------------
# Helper classes for testing
# ----------------------------------------


class NotATraitClass:
    pass


# ----------------------------------------
# trait decorator tests
# ----------------------------------------


def test_trait_decorator_marks_dataclass_and_sets_hint():
    @trait
    class MyTrait:
        x: int
        y: int

    # Check that the class is converted to a dataclass
    assert is_dataclass(MyTrait)
    # Check that the class has the __is_trait attribute
    assert hasattr(MyTrait, TRAIT_HINT)
    assert getattr(MyTrait, TRAIT_HINT) is True

    # Instances are recognized by is_trait
    instance = MyTrait(1, 2)
    assert is_trait(instance)
    # And the type is recognized by is_trait_type
    assert is_trait_type(MyTrait)


def test_trait_decorator_preserves_annotations():
    @trait
    class MyTrait:
        x: int
        y: int

    assert MyTrait.__annotations__ == {"x": int, "y": int}


# ----------------------------------------
# make_trate_type tests
# ----------------------------------------


def test_make_trate_type_marks_class():
    class MyTraitClass:
        x: int
        y: int

    result = make_trate_type(MyTraitClass)
    assert result is MyTraitClass
    assert hasattr(MyTraitClass, TRAIT_HINT)
    assert getattr(MyTraitClass, TRAIT_HINT) is True


def test_make_trate_type_does_not_convert_to_dataclass():
    class MyTraitClass:
        x: int
        y: int

    cls = make_trate_type(MyTraitClass)
    assert not is_dataclass(
        cls
    )  # unlike `trait`, this does not decorate with @dataclass


# ----------------------------------------
# is_trait / is_trait_type tests
# ----------------------------------------


def test_is_trait_and_is_trait_type_behavior():
    @trait
    class MyTrait:
        x: int
        y: int

    instance = MyTrait(10, 20)
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


def test_subclassing_trait_marks_as_trait():
    class MyNewTrait(Trait):
        x: int
        y: int

    # Class is converted to dataclass and has __is_trait
    assert is_dataclass(MyNewTrait)
    assert hasattr(MyNewTrait, TRAIT_HINT)
    assert getattr(MyNewTrait, TRAIT_HINT) is True

    # Instances recognized
    instance = MyNewTrait(1, 2)
    assert is_trait(instance)
    assert is_trait_type(MyNewTrait)


def test_trait_subclass_can_inherit_multiple_levels():
    class Base(Trait):
        x: int

    class Derived(Base):
        y: int

    assert is_trait_type(Derived)
    assert is_trait_type(Base)
    inst = Derived(1, 2)
    assert is_trait(inst)
