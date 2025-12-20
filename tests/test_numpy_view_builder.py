import pytest

# module under test
import buds.extras.numpy.views as views
from buds.base import Trait
from buds.inspect import inspect_trait

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_global_state(monkeypatch):
    """
    Reset global cache and adapters for isolation.
    """
    monkeypatch.setattr(views, "_VIEW_GENERATORS", [])
    monkeypatch.setattr(views, "_VIEW_CACHE", {})
    views.register_view_adapter(views.DataclassViewGenerator)
    yield


# ---------------------------------------------------------------------------
# basic builder behavior
# ---------------------------------------------------------------------------


def test_view_builder_builds_class():
    class MyTrait(Trait):
        x: int
        y: float

    schema = inspect_trait(MyTrait)
    view_cls = views.ViewBuilder(schema, "TraitView").add_defaults().build()

    assert view_cls.__name__ == "TraitView"
    assert issubclass(view_cls, Trait)


def test_builder_init_and_repr():
    class MyTrait(Trait):
        x: int
        y: int

    schema = inspect_trait(MyTrait)
    view_cls = views.ViewBuilder(schema, "TraitView").add_defaults().build()

    rec = {"x": [1], "y": [2]}
    v = view_cls(rec, 0)

    # repr should include all fields
    assert repr(v) == "<TraitView(x=1, y=2)>"


# ---------------------------------------------------------------------------
# property behavior
# ---------------------------------------------------------------------------


def test_properties_get_and_set():
    class MyTrait(Trait):
        x: int
        y: float

    schema = inspect_trait(MyTrait)
    view_cls = views.ViewBuilder(schema, "TraitView").add_defaults().build()

    rec = {"x": [1], "y": [2.5]}
    v = view_cls(rec, 0)

    # getter
    assert v.x == 1
    assert v.y == 2.5

    # setter
    v.x = 10
    v.y = 3.5

    assert rec["x"][0] == 10
    assert rec["y"][0] == 3.5


def test_non_primitive_property_passthrough():
    class MyTrait(Trait):
        data: list

    schema = inspect_trait(MyTrait)
    view_cls = views.ViewBuilder(schema, "TraitView").add_defaults().build()

    lst = [1, 2]
    rec = {"data": [lst]}
    v = view_cls(rec, 0)

    # list should not be coerced
    assert v.data is lst


# ---------------------------------------------------------------------------
# slots behavior
# ---------------------------------------------------------------------------


def test_slots_are_added():
    class MyTrait(Trait):
        x: int

    schema = inspect_trait(MyTrait)
    view_cls = views.ViewBuilder(schema, "TraitView").add_slots().add_init().build()

    assert hasattr(view_cls, "__slots__")
    assert "_rec" in view_cls.__slots__
    assert "_index" in view_cls.__slots__


# ---------------------------------------------------------------------------
# adapter selection
# ---------------------------------------------------------------------------


def test_dataclass_adapter_used():
    class MyTrait(Trait):
        x: int

    schema = inspect_trait(MyTrait)
    view_cls = views.create_view_class(MyTrait)

    assert view_cls.__name__ == "MyTraitView"
    assert issubclass(view_cls, Trait)


def test_non_trait_adapter_rejected():
    # normal Trait class, not a dataclass
    class NonDataclass:
        pass

    with pytest.raises(ValueError):
        views.create_view_class(NonDataclass)


# ---------------------------------------------------------------------------
# adapter priority
# ---------------------------------------------------------------------------


def test_higher_priority_adapter_wins():
    class MyTrait(Trait):
        x: int

    schema = inspect_trait(MyTrait)

    class CustomAdapter:
        @classmethod
        def create_view(cls, schema, name):
            return type("CustomView", (), {})

    # register high-priority adapter
    views.register_view_adapter(CustomAdapter, priority=100)

    view_cls = views.create_view_class(MyTrait)
    assert view_cls.__name__ == "CustomView"


# ---------------------------------------------------------------------------
# caching behavior
# ---------------------------------------------------------------------------


def test_view_class_is_cached():
    class MyTrait(Trait):
        x: int

    schema = inspect_trait(MyTrait)

    v1 = views.create_view_class(MyTrait)
    v2 = views.create_view_class(MyTrait)

    # should return same object from cache
    assert v1 is v2


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_non_trait_type_raises():
    with pytest.raises(ValueError):
        views.create_view_class(int)
