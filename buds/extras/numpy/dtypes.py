"""
NumPy dtype resolution utilities for trait schemas.

This module provides an adapter-driven mechanism for converting
``FieldSchema`` and ``TraitSchema`` objects into NumPy ``dtype``
definitions. These dtypes can be used to construct structured arrays
that efficiently store trait-backed data.

The system supports:
- Primitive Python types (e.g. int, float, bool)
- Explicit NumPy scalar and dtype annotations
- ``typing.Annotated`` metadata for shaped arrays
- Tuple-typed fields mapped to structured sub-dtypes

Adapters are evaluated in priority order, allowing custom extensions
to override or specialize dtype inference logic.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import (
    Annotated,
    Any,
    Optional,
    Protocol,
    TypeAlias,
    get_args,
    get_origin,
    runtime_checkable,
)

import numpy as np

from ...inspect import FieldSchema, TraitSchema, UnifiedMetadata


@dataclass(frozen=True, slots=True)
class NumpyArrayMetadata:
    """Metadata describing a NumPy array field.

    This metadata is intended to be attached via ``typing.Annotated``
    to a field typed as ``np.ndarray``.

    Attributes:
        shape: The expected array shape. A single integer denotes a
            one-dimensional fixed-length array.
        dtype: The NumPy dtype of the array elements.
        validator: Optional custom validation logic for the array.
    """

    shape: tuple[int, ...] | int
    dtype: np.dtype | type[np.generic] = np.dtype(np.float32)
    validator: Any | None = None


Vector2: TypeAlias = Annotated[np.ndarray, NumpyArrayMetadata((2,))]
Vector3: TypeAlias = Annotated[np.ndarray, NumpyArrayMetadata((3,))]
Vector4: TypeAlias = Annotated[np.ndarray, NumpyArrayMetadata((4,))]
Matrix2x2: TypeAlias = Annotated[np.ndarray, NumpyArrayMetadata((2, 2))]
Matrix2x3: TypeAlias = Annotated[np.ndarray, NumpyArrayMetadata((2, 3))]
Matrix3x3: TypeAlias = Annotated[np.ndarray, NumpyArrayMetadata((3, 3))]
Matrix3x4: TypeAlias = Annotated[np.ndarray, NumpyArrayMetadata((3, 4))]
Matrix4x4: TypeAlias = Annotated[np.ndarray, NumpyArrayMetadata((4, 4))]


@runtime_checkable
class DtypeAdapter(Protocol):
    """Protocol for adapters that infer NumPy dtypes from fields."""

    @staticmethod
    def get_dtype(field: FieldSchema) -> Optional[np.dtype]:
        """Return a NumPy dtype for the given field, if supported.

        Args:
            field: The field schema to inspect.

        Returns:
            A NumPy dtype if the adapter applies, otherwise None.
        """
        ...


_DTYPE_ADAPTERS: list[DtypeAdapter] = []


class PrimitiveAdapter:
    """Infer NumPy scalar dtypes from primitive Python field types.

    This adapter matches fields whose declared Python type is exactly one
    of the supported built-in primitives. The mapping is fixed and does
    not consider metadata, defaults, or container types.

    Supported mappings:
        - ``int``   → ``np.int32``
        - ``float`` → ``np.float32``
        - ``bool``  → ``np.bool_``

    This adapter does *not* handle:
        - Subclasses of primitive types
        - ``typing`` constructs (e.g. ``Optional[int]``)
        - NumPy scalar types or dtypes
        - Container or composite types
    """

    primitive_map = {int: np.int32, float: np.float32, bool: np.bool_}

    @staticmethod
    def get_dtype(field: FieldSchema) -> Optional[np.dtype]:
        """Infer dtype from primitive Python types."""
        return PrimitiveAdapter.primitive_map.get(field.type, None)


class NaiveNumpyAdapter:
    """Infer NumPy dtypes from explicit NumPy scalar or dtype annotations.

    This adapter matches fields whose type annotation directly references
    NumPy's type system, without additional metadata or structure.

    Supported inputs:
        - ``np.dtype`` instances (used verbatim)
        - NumPy scalar classes (subclasses of ``np.generic``)

    Examples:
        - ``np.float32`` → ``dtype('float32')``
        - ``np.int64``   → ``dtype('int64')``
        - ``np.dtype('uint8')`` → ``dtype('uint8')``

    This adapter does *not* handle:
        - ``np.ndarray`` fields
        - ``Annotated`` metadata
        - Structured or shaped arrays
        - Python built-in primitives
    """

    @staticmethod
    def get_dtype(field: FieldSchema) -> Optional[np.dtype]:
        """Infer dtype directly from NumPy dtype or scalar type."""
        if isinstance(field.type, np.dtype):
            return field.type
        if isinstance(field.type, type) and issubclass(field.type, np.generic):
            return np.dtype(field.type)


class AnnotatedArrayAdapter:
    """Infer structured array dtypes from ``Annotated[np.ndarray, ...]`` fields.

    This adapter recognizes fields annotated as ``np.ndarray`` and extracts
    shape and element dtype information from ``NumpyArrayMetadata`` attached
    via ``typing.Annotated``.

    Matching criteria:
        - Field type must be exactly ``np.ndarray``
        - Field metadata must include ``NumpyArrayMetadata`` as its first
          ``Annotated`` entry

    The resulting dtype represents a *fixed-shape array field* using NumPy's
    subdtype mechanism.

    Example:
        ``Annotated[np.ndarray, NumpyArrayMetadata((3,), np.float32)]``
        produces:
        ``dtype((float32, (3,)))``

    This adapter does *not* handle:
        - Unannotated ``np.ndarray`` fields
        - Variable-length arrays
        - Multiple ``Annotated`` metadata entries
        - Validation or runtime shape checking (metadata only)
    """

    @staticmethod
    def get_dtype(field: FieldSchema) -> Optional[np.dtype]:
        """Infer structured dtype from annotated NumPy array metadata."""

        if field.type is not np.ndarray:
            return None

        if (
            not isinstance(field.metadata, UnifiedMetadata)
            or field.metadata.annotated is None
        ):
            return None

        metadata = field.metadata.annotated[0]

        if not isinstance(metadata, NumpyArrayMetadata):
            return None

        shape = (metadata.shape,) if isinstance(metadata.shape, int) else metadata.shape

        return np.dtype((metadata.dtype, shape))


class TupleAdapter:
    """Infer structured NumPy dtypes from tuple-typed fields.

    This adapter converts fixed-length tuple type annotations into NumPy
    structured dtypes, where each tuple element becomes a named subfield.

    Matching criteria:
        - Field type origin must be ``tuple``
        - Tuple must have at least one type argument

    Conversion rules:
        - Each tuple element is converted recursively using ``get_dtype``
        - Element field names are generated as ``_0``, ``_1``, ``_2``, ...

    Example:
        ``tuple[int, float]`` produces:
        ``dtype([('_0', int32), ('_1', float32)])``

    This adapter does *not* handle:
        - Variable-length tuples (e.g. ``tuple[int, ...]``)
        - Empty tuples
        - Named tuples or dataclasses
        - Runtime tuple validation
    """

    @staticmethod
    def get_dtype(field: FieldSchema) -> Optional[np.dtype]:
        """Infer a structured dtype from a tuple-typed field."""

        origin = get_origin(field.type)

        if origin is not tuple:
            return None

        args = get_args(field.type)

        if not args:
            return None

        fields = []
        for i, arg in enumerate(args):
            schema = FieldSchema.create(f"_{i}", arg)
            fields.append((f"_{i}", get_dtype(schema)))
        return np.dtype(fields)


def register_adapter(adapter: DtypeAdapter, *, priority: int = 0):
    """Register a dtype adapter.

    Adapters registered earlier take precedence over later ones.

    Args:
        adapter: The adapter instance to register.
        priority: Insertion index controlling adapter precedence.
    """
    _DTYPE_ADAPTERS.insert(priority, adapter)


@lru_cache(maxsize=512)
def get_field_dtype(field: FieldSchema):
    """Resolve the NumPy dtype for a single field schema.

    Args:
        field: The field schema to convert.

    Returns:
        A NumPy dtype. Falls back to ``object`` if no adapter applies.
    """
    if isinstance(field, FieldSchema):
        for adapter in _DTYPE_ADAPTERS:
            dtype = adapter.get_dtype(field)
            if dtype is not None:
                return dtype  # type: ignore

        # Fallback if we cannot convert this
        return np.dtype(np.object_)


@lru_cache(maxsize=512)
def get_trait_dtype(trait: TraitSchema):
    """Resolve a structured NumPy dtype for an entire trait.

    Args:
        trait: The trait schema to convert.

    Returns:
        A structured NumPy dtype with one field per trait attribute.
    """
    fields = [(field.name, get_field_dtype(field)) for field in trait.fields]
    return np.dtype(fields)


@lru_cache(maxsize=512)
def get_dtype(field_or_trait: FieldSchema | TraitSchema) -> np.dtype:
    """Resolve a NumPy dtype for either a field or an entire trait.

    Args:
        field_or_trait: A ``FieldSchema`` or ``TraitSchema`` instance.

    Returns:
        A NumPy dtype corresponding to the input.
    """
    return (
        get_field_dtype(field_or_trait)
        if isinstance(field_or_trait, FieldSchema)
        else get_trait_dtype(field_or_trait)
    )


register_adapter(AnnotatedArrayAdapter())
register_adapter(TupleAdapter())
register_adapter(NaiveNumpyAdapter())
register_adapter(PrimitiveAdapter())
