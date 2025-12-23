import numpy as np
import pytest

import buds.extras.numpy.views as views

# module under test
from buds import is_trait, is_trait_type

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_global_state(monkeypatch):
    """
    Reset global cache and adapters for isolation.
    """
    monkeypatch.setattr(views, "_VIEW_CACHE", {})
    yield


@pytest.fixture
def SimpleTrait(backend):
    class Simple(backend.Trait):
        x: int
        y: float

    return Simple


@pytest.fixture
def SimpleDtype():
    return np.dtype([("x", "<i4"), ("y", "<f4")])


# ---------------------------------------------------------------------------
# basic builder behavior
# ---------------------------------------------------------------------------


def test_view_builder_builds_trait_class(backend):
    class MyTrait(backend.Trait):
        x: int
        y: float

    view_cls = views.create_view_class(MyTrait)

    assert view_cls.__name__ == "MyTraitView"
    assert is_trait_type(view_cls)


def test_builder_init_and_repr(backend):
    class MyTrait(backend.Trait):
        x: int
        y: int

    view_cls = views.create_view_class(MyTrait)

    rec = {"x": [1], "y": [2]}
    v = view_cls(rec, 0)

    # repr should include all fields
    assert repr(v) == "<MyTraitView(x=1, y=2)>"


def test_view_instance_is_trait(backend):
    class MyTrait(backend.Trait):
        x: int
        y: int

    view_cls = views.create_view_class(MyTrait)

    rec = {"x": [1], "y": [2]}
    v = view_cls(rec, 0)

    assert is_trait(v)


# ---------------------------------------------------------------------------
# property behavior
# ---------------------------------------------------------------------------


def test_properties_get_and_set(backend):
    class MyTrait(backend.Trait):
        x: int
        y: float

    view_cls = views.create_view_class(MyTrait)

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


def test_non_primitive_property_passthrough(backend):
    class MyTrait(backend.Trait):
        data: list

    view_cls = views.create_view_class(MyTrait)

    lst = [1, 2]
    rec = {"data": [lst]}
    v = view_cls(rec, 0)

    # list should not be coerced
    assert v.data is lst


# ---------------------------------------------------------------------------
# slots behavior
# ---------------------------------------------------------------------------


def test_slots_are_added(backend):
    class MyTrait(backend.Trait):
        x: int

    view_cls = views.create_view_class(MyTrait)

    assert hasattr(view_cls, "__slots__")
    assert "_rec" in view_cls.__slots__
    assert "_index" in view_cls.__slots__


# ---------------------------------------------------------------------------
# adapter selection
# ---------------------------------------------------------------------------


def test_non_trait_adapter_rejected():
    # normal Trait class, not a dataclass
    class NonDataclass:
        pass

    with pytest.raises(TypeError):
        views.create_view_class(NonDataclass)


# ---------------------------------------------------------------------------
# adapter priority
# ---------------------------------------------------------------------------


def test_higher_priority_adapter_wins(backend, monkeypatch):
    class MyTrait(backend.Trait):
        x: int

    monkeypatch.setattr(views, "_VIEW_GENERATORS", views._VIEW_GENERATORS.copy())

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


def test_view_class_is_cached(backend):
    class MyTrait(backend.Trait):
        x: int

    v1 = views.create_view_class(MyTrait)
    v2 = views.create_view_class(MyTrait)

    # should return same object from cache
    assert v1 is v2


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_non_trait_type_raises():
    with pytest.raises(TypeError):
        views.create_view_class(int)


# -----------------------------------------------------------------------------
# create_view_class
# -----------------------------------------------------------------------------


def test_make_trait_view_class_creates_class(backend, SimpleTrait):
    ViewCls = views.create_view_class(SimpleTrait)
    assert ViewCls.__name__ == "SimpleView"
    assert issubclass(ViewCls, SimpleTrait) or not backend.views_are_subclasses


def test_make_trait_view_class_reflects_data(SimpleTrait, SimpleDtype):
    ViewCls = views.create_view_class(SimpleTrait)
    data = np.zeros(3, dtype=SimpleDtype)
    data[1]["x"] = 42
    data[1]["y"] = 3.14
    view = ViewCls(data, 1)
    assert view.x == 42
    assert abs(view.y - 3.14) < 1e-6
    # Mutate via property
    view.x = 7
    assert data[1]["x"] == 7


def test_compare_view_to_view_works(SimpleTrait, SimpleDtype):
    ViewCls = views.create_view_class(SimpleTrait)
    data = np.zeros(3, dtype=SimpleDtype)
    data[1]["x"] = 42
    data[1]["y"] = 3.14
    view_1 = ViewCls(data, 1)
    view_2 = ViewCls(data, 1)
    view_3 = ViewCls(data, 2)
    assert view_1 == view_2
    assert view_1 != view_3


def test_compare_view_to_trait_works(SimpleTrait, SimpleDtype):
    ViewCls = views.create_view_class(SimpleTrait)
    data = np.zeros(3, dtype=SimpleDtype)
    data[1]["x"] = 42
    data[1]["y"] = 3.14
    view_1 = ViewCls(data, 1)
    view_2 = ViewCls(data, 2)

    trait = SimpleTrait(x=42, y=data[1]["y"])
    assert view_1 == trait
    assert view_2 != trait

    assert trait == view_1
    assert trait != view_2


def test_make_trait_view_class_reflects_method(backend):
    class TraitWithMethod(backend.Trait):
        age: int

        def greet(self) -> str:
            return f"You are {self.age} years old"

    dtype = np.dtype([("age", "<i4")])
    ViewCls = views.create_view_class(TraitWithMethod)
    data = np.empty(1, dtype=dtype)
    data[0]["age"] = 42

    view = ViewCls(data, 0)
    msg = view.greet()
    assert msg == "You are 42 years old"


def test_make_trait_view_class_reflects_classmethod(backend):
    class TraitWithClassMethod(backend.Trait):
        age: int

        @classmethod
        def question(cls) -> str:
            return f"How old are you (asked by: {cls.__name__})"

    ViewCls = views.create_view_class(TraitWithClassMethod)

    msg = ViewCls.question()
    asking = (
        ViewCls.__name__
        if backend.views_are_subclasses
        else TraitWithClassMethod.__name__
    )
    assert msg == f"How old are you (asked by: {asking})"


def test_make_trait_view_class_reflects_classvariables(backend):
    from typing import ClassVar

    class TraitWithClassVariable(backend.Trait):
        age: int
        msg: ClassVar[str] = "Original message"

    dtype = np.dtype([("age", "<i4")])
    ViewCls = views.create_view_class(TraitWithClassVariable)
    data = np.empty(1, dtype=dtype)
    data[0]["age"] = 42

    assert ViewCls.msg == TraitWithClassVariable.msg

    TraitWithClassVariable.msg = "Updated msg"
    assert ViewCls.msg == TraitWithClassVariable.msg

    ViewCls.msg = "Modified by view"
    assert ViewCls.msg == TraitWithClassVariable.msg


# -----------------------------------------------------------------------------
# reate_vectorized_view_class
# -----------------------------------------------------------------------------


def test_make_trait_vectorized_view_class(SimpleTrait):
    VecCls = views.create_vectorized_view_class(SimpleTrait)
    assert "SimpleVectorizedView" in VecCls.__name__
    # Caching works
    again = views.create_vectorized_view_class(SimpleTrait)
    assert VecCls is again


def test_vectorized_view_class_has_expected_field_properties(SimpleTrait):
    """
    The vectorized view class for a simple trait should have .x and .y
    properties matching the dataclass field names.
    """
    VecCls = views.create_vectorized_view_class(SimpleTrait)

    # The generated view should expose .x and .y
    assert hasattr(VecCls, "x")
    assert hasattr(VecCls, "y")

    # Check that they’re actual properties (not plain attributes)
    assert isinstance(VecCls.x, property)
    assert isinstance(VecCls.y, property)


def test_vectorized_view_properties(SimpleTrait, SimpleDtype):
    VecCls = views.create_vectorized_view_class(SimpleTrait)

    data1 = np.array([(1, 2.0), (3, 4.0)], dtype=SimpleDtype)
    mask1 = np.array([True, False])
    data2 = np.array([(5, 6.0)], dtype=SimpleDtype)
    mask2 = np.array([True])

    view = VecCls([data1, data2], [mask1, mask2])

    # After construction, concatenated _data should contain only the True rows
    assert isinstance(view._data, np.ndarray)
    assert len(view._data) == 2  # one from first, one from second

    np.testing.assert_equal(
        view._data, np.array([(1, 2.0), (5, 6.0)], dtype=SimpleDtype)
    )


def test_vectorized_view_write_back_updates_original(SimpleTrait, SimpleDtype):
    # Two archetype record arrays (simulate separate archetypes)
    data1 = np.array([(1, 1.0), (2, 2.0)], dtype=SimpleDtype)
    data2 = np.array([(3, 3.0), (4, 4.0)], dtype=SimpleDtype)

    # Masks select which rows are "alive" (True)
    mask1 = np.array([True, False])
    mask2 = np.array([True, True])

    # Create the vectorized view class and instance
    VecCls = views.create_vectorized_view_class(SimpleTrait)
    view = VecCls([data1, data2], [mask1, mask2])

    # The concatenated _data contains three rows (1 from data1, 2 from data2)
    assert len(view._data) == 3

    # Modify concatenated _data directly (simulate an update)
    # Change all x and y fields
    view._data["x"] = [10, 20, 30]
    view._data["y"] = [1.1, 2.2, 3.3]

    # Apply write_back to propagate changes to the original arrays
    view.write_back()

    # The True-masked positions in the original arrays should now reflect the updates
    # data1 had mask [True, False] → only the first row should be updated
    assert data1["x"][0] == 10
    assert data1["y"][0] == pytest.approx(1.1)
    assert data1["x"][1] == 2  # unchanged
    assert data1["y"][1] == pytest.approx(2.0)

    # data2 had mask [True, True] → both rows updated
    assert np.all(data2["x"] == np.array([20, 30]))
    assert np.allclose(data2["y"], np.array([2.2, 3.3]))
