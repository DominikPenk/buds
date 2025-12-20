from typing import Tuple

import numpy as np
import pytest

# import the code under test
# adjust these imports to your actual module paths
from buds.extras.numpy.dtypes import (
    AnnotatedArrayAdapter,
    NaiveNumpyAdapter,
    NumpyArrayMetadata,
    PrimitiveAdapter,
    TupleAdapter,
    get_dtype,
    register_adapter,
)
from buds.inspect import FieldSchema, UnifiedMetadata

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_field(name: str, typ, *, annotated=None):
    """
    Helper to build a FieldSchema with optional Annotated metadata.
    """
    metadata = (
        UnifiedMetadata(annotated=annotated, field=None)
        if annotated is not None
        else None
    )
    return FieldSchema(name, typ, metadata)


@pytest.fixture(autouse=True)
def reset_adapters():
    """
    Ensure adapter list is clean and deterministic for every test.
    """
    from buds.extras.numpy.dtypes import _DTYPE_ADAPTERS

    _DTYPE_ADAPTERS.clear()

    # register adapters in intended order
    register_adapter(AnnotatedArrayAdapter())
    register_adapter(TupleAdapter())
    register_adapter(NaiveNumpyAdapter())
    register_adapter(PrimitiveAdapter())

    yield

    _DTYPE_ADAPTERS.clear()


# ---------------------------------------------------------------------------
# primitive types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "py_type, expected",
    [
        (int, np.dtype(np.int32)),
        (float, np.dtype(np.float32)),
        (bool, np.dtype(np.bool_)),
        (str, np.dtype(np.object_)),
    ],
)
def test_primitive_types(py_type, expected):
    field = make_field("x", py_type)
    assert get_dtype(field) == expected


# ---------------------------------------------------------------------------
# numpy scalar types
# ---------------------------------------------------------------------------


def test_numpy_scalar_class():
    field = make_field("x", np.float64)
    assert get_dtype(field) == np.dtype(np.float64)


def test_numpy_dtype_instance():
    field = make_field("x", np.dtype(np.int16))
    assert get_dtype(field) == np.dtype(np.int16)


# ---------------------------------------------------------------------------
# annotated numpy arrays
# ---------------------------------------------------------------------------


def test_annotated_vector_shape():
    meta = NumpyArrayMetadata(shape=3, dtype=np.float32)
    field = make_field(
        "v",
        np.ndarray,
        annotated=(meta,),
    )

    dt = get_dtype(field)
    assert dt == np.dtype((np.float32, (3,)))


def test_annotated_matrix_shape():
    meta = NumpyArrayMetadata(shape=(2, 3), dtype=np.float64)
    field = make_field(
        "m",
        np.ndarray,
        annotated=(meta,),
    )

    dt = get_dtype(field)
    assert dt == np.dtype((np.float64, (2, 3)))


def test_annotated_wrong_metadata_is_ignored():
    field = make_field(
        "x",
        np.ndarray,
        annotated=("not numpy metadata",),
    )

    assert get_dtype(field) == np.dtype(np.object_)


def test_non_ndarray_annotated_is_ignored():
    meta = NumpyArrayMetadata(shape=3)
    field = make_field(
        "x",
        list,
        annotated=(meta,),
    )

    assert get_dtype(field) == np.dtype(np.object_)


# ---------------------------------------------------------------------------
# tuples / structured dtypes
# ---------------------------------------------------------------------------


def test_tuple_of_primitives():
    field = make_field("t", Tuple[int, float, bool])

    dt = get_dtype(field)

    assert dt.fields is not None
    assert dt.fields["_0"][0] == np.dtype(np.int32)
    assert dt.fields["_1"][0] == np.dtype(np.float32)
    assert dt.fields["_2"][0] == np.dtype(np.bool_)


def test_nested_tuple():
    field = make_field("t", Tuple[int, Tuple[float, bool]])

    dt = get_dtype(field)

    assert dt.fields is not None
    inner = dt.fields["_1"][0]

    assert inner.fields["_0"][0] == np.dtype(np.float32)
    assert inner.fields["_1"][0] == np.dtype(np.bool_)


# ---------------------------------------------------------------------------
# fallback behavior
# ---------------------------------------------------------------------------


def test_unknown_type_falls_back_to_object():
    class CustomType:
        pass

    field = make_field("x", CustomType)
    assert get_dtype(field) == np.dtype(np.object_)


def test_tuple_with_unknown_member_falls_back_to_object():
    class CustomType:
        pass

    field = make_field("t", Tuple[int, CustomType])
    dt = get_dtype(field)

    # structured dtype still exists, but contains object
    assert dt.fields["_1"][0] == np.dtype(np.object_)
