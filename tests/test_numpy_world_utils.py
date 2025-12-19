from dataclasses import dataclass

import numpy as np
import pytest

from buds.base import Trait
from buds.extras.numpy_archetype import (
    FixedSizeArray,
    _build_trait_dtype,
    _make_trait_vectorized_view_class,
    _make_trait_view_class,
    numpy_dtype_for_type,
)


# -----------------------------------------------------------------------------
# Fixtures for simple trait classes
# -----------------------------------------------------------------------------
@pytest.fixture
def SimpleTrait():
    class Simple(Trait):
        x: int
        y: float

    return Simple


@pytest.fixture
def NestedTrait():
    class Nested(Trait):
        a: tuple[int, float]
        b: bool

    return Nested


# -----------------------------------------------------------------------------
# numpy_dtype_for_type
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "py_type,expected_dtype",
    [
        (int, np.int32),
        (float, np.float32),
        (bool, np.bool_),
        (str, np.str_),
    ],
)
def test_numpy_dtype_for_primitive_types(py_type, expected_dtype):
    dtype = numpy_dtype_for_type(py_type)
    assert isinstance(dtype, np.dtype)
    assert dtype == np.dtype(expected_dtype)


def test_numpy_dtype_for_tuple_type():
    dtype = numpy_dtype_for_type(tuple[int, float])
    assert isinstance(dtype, np.dtype)
    assert "_0" in dtype.names and "_1" in dtype.names
    assert dtype["_0"] == np.dtype(np.int32)
    assert dtype["_1"] == np.dtype(np.float32)


def test_numpy_dtype_for_annotated_array_shape():
    dtype = numpy_dtype_for_type(FixedSizeArray[(3,), np.float32])
    assert dtype.shape == (3,)
    assert dtype.base == np.dtype(np.float32)


def test_numpy_dtype_for_annotated_default_dtype():
    dtype = numpy_dtype_for_type(FixedSizeArray[(4,)])
    assert dtype.shape == (4,)
    assert dtype.base == np.dtype(np.float32)


def test_numpy_dtype_for_object_fallback():
    class Unknown:
        pass

    dtype = numpy_dtype_for_type(Unknown)
    assert dtype == np.dtype(np.object_)


# -----------------------------------------------------------------------------
# _build_trait_dtype
# -----------------------------------------------------------------------------
def test_build_trait_dtype_from_simple_trait(SimpleTrait):
    dtype = _build_trait_dtype(SimpleTrait)
    assert isinstance(dtype, np.dtype)
    assert set(dtype.names) == {"x", "y"}
    assert dtype["x"] == np.dtype(np.int32)
    assert dtype["y"] == np.dtype(np.float32)


def test_build_trait_dtype_from_non_dataclass_raises():
    with pytest.raises(TypeError):
        _build_trait_dtype(int)


# -----------------------------------------------------------------------------
# _make_trait_view_class
# -----------------------------------------------------------------------------
def test_make_trait_view_class_creates_class(SimpleTrait):
    ViewCls = _make_trait_view_class(SimpleTrait)
    assert ViewCls.__name__ == "SimpleView"
    assert issubclass(ViewCls, SimpleTrait)
    # Ensure caching works
    again = _make_trait_view_class(SimpleTrait)
    assert again is ViewCls


def test_make_trait_view_class_reflects_data(SimpleTrait):
    ViewCls = _make_trait_view_class(SimpleTrait)
    dtype = _build_trait_dtype(SimpleTrait)
    data = np.zeros(1, dtype=dtype)
    data[0]["x"] = 42
    data[0]["y"] = 3.14
    view = ViewCls(data, 0)
    assert view.x == 42
    assert abs(view.y - 3.14) < 1e-6
    # Mutate via property
    view.x = 7
    assert data[0]["x"] == 7


def test_make_trait_view_class_raises_for_non_trait():
    @dataclass
    class NotTrait:
        x: int

    with pytest.raises(TypeError):
        _make_trait_view_class(NotTrait)


# -----------------------------------------------------------------------------
# _make_trait_vectorized_view_class
# -----------------------------------------------------------------------------
def test_make_trait_vectorized_view_class(SimpleTrait):
    VecCls = _make_trait_vectorized_view_class(SimpleTrait)
    assert "SimpleVectorizedView" in VecCls.__name__
    # Caching works
    again = _make_trait_vectorized_view_class(SimpleTrait)
    assert VecCls is again


def test_vectorized_view_class_has_expected_field_properties(SimpleTrait):
    """
    The vectorized view class for a simple trait should have .x and .y
    properties matching the dataclass field names.
    """
    VecCls = _make_trait_vectorized_view_class(SimpleTrait)

    # The generated view should expose .x and .y
    assert hasattr(VecCls, "x")
    assert hasattr(VecCls, "y")

    # Check that they’re actual properties (not plain attributes)
    assert isinstance(VecCls.x, property)
    assert isinstance(VecCls.y, property)


def test_vectorized_view_properties(SimpleTrait):
    VecCls = _make_trait_vectorized_view_class(SimpleTrait)
    dtype = _build_trait_dtype(SimpleTrait)

    data1 = np.array([(1, 2.0), (3, 4.0)], dtype=dtype)
    mask1 = np.array([True, False])
    data2 = np.array([(5, 6.0)], dtype=dtype)
    mask2 = np.array([True])

    view = VecCls([data1, data2], [mask1, mask2])

    # After construction, concatenated _data should contain only the True rows
    assert isinstance(view._data, np.ndarray)
    assert len(view._data) == 2  # one from first, one from second

    np.testing.assert_equal(view._data, np.array([(1, 2.0), (5, 6.0)], dtype=dtype))


def test_vectorized_view_write_back_updates_original(SimpleTrait):
    # Build a dtype matching the trait
    dtype = _build_trait_dtype(SimpleTrait)

    # Two archetype record arrays (simulate separate archetypes)
    data1 = np.array([(1, 1.0), (2, 2.0)], dtype=dtype)
    data2 = np.array([(3, 3.0), (4, 4.0)], dtype=dtype)

    # Masks select which rows are "alive" (True)
    mask1 = np.array([True, False])
    mask2 = np.array([True, True])

    # Create the vectorized view class and instance
    VecCls = _make_trait_vectorized_view_class(SimpleTrait)
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


# -----------------------------------------------------------------------------
# FixedSizeArrayMeta behavior
# -----------------------------------------------------------------------------
def test_fixed_size_array_meta_shape_and_dtype():
    AnnotatedType = FixedSizeArray[(3,), np.float32]
    assert hasattr(AnnotatedType, "__metadata__") or isinstance(AnnotatedType, tuple)
    dtype = numpy_dtype_for_type(AnnotatedType)
    assert dtype.shape == (3,)
    assert dtype.base == np.dtype(np.float32)
