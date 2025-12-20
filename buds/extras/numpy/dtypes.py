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

from ...inspect import FieldSchema, UnifiedMetadata


@dataclass(frozen=True, slots=True)
class NumpyArrayMetadata:
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
    @staticmethod
    def get_dtype(field: FieldSchema) -> Optional[np.dtype]: ...


_DTYPE_ADAPTERS: list[DtypeAdapter] = []


class PrimitiveAdapter:
    primitive_map = {int: np.int32, float: np.float32, bool: np.bool_}

    @staticmethod
    def get_dtype(field: FieldSchema) -> Optional[np.dtype]:
        return PrimitiveAdapter.primitive_map.get(field.type, None)


class NaiveNumpyAdapter:
    @staticmethod
    def get_dtype(field: FieldSchema) -> Optional[np.dtype]:
        if isinstance(field.type, np.dtype):
            return field.type
        if isinstance(field.type, type) and issubclass(field.type, np.generic):
            return np.dtype(field.type)


class AnnotatedArrayAdapter:
    @staticmethod
    def get_dtype(field: FieldSchema) -> Optional[np.dtype]:
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
    @staticmethod
    def get_dtype(field: FieldSchema) -> Optional[np.dtype]:
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
    _DTYPE_ADAPTERS.insert(priority, adapter)


@lru_cache(maxsize=512)
def get_dtype(field: FieldSchema) -> np.dtype:
    for adapter in _DTYPE_ADAPTERS:
        dtype = adapter.get_dtype(field)
        if dtype is not None:
            return dtype  # type: ignore

    # Fallback if we cannot convert this
    return np.dtype(np.object_)


register_adapter(AnnotatedArrayAdapter())
register_adapter(TupleAdapter())
register_adapter(NaiveNumpyAdapter())
register_adapter(PrimitiveAdapter())
