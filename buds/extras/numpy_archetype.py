from collections.abc import Iterable, Iterator
from dataclasses import fields, is_dataclass
from typing import Annotated, Any, Generic, Optional, TypeVar, get_args, get_origin

import numpy as np

from ..archetype import ArcheType, ArchetypeKey, ArchetypeWorld
from ..base import Entity, is_trait_type

__all__ = ["NumpyArchetypeWorld"]

T = TypeVar("T")
_view_cache: dict[type, type] = dict()
_vector_view_cache: dict[type, type] = dict()


class FixedSizeArrayMeta(type):
    def __getitem__(cls, item: Any):
        # Support shapes and dtypes
        if isinstance(item, tuple):
            if len(item) == 1:
                shape, dtype = item[0], np.float32
            elif len(item) == 2:
                shape, dtype = item
            else:
                raise TypeError("FixedSizeArray[...] must be of form [(shape, dtype?)]")
        else:
            shape, dtype = (item,), np.float32

        # Return a type alias with metadata attached
        return Annotated[np.ndarray, {"shape": shape, "dtype": np.dtype(dtype)}]


class FixedSizeArray(np.ndarray, metaclass=FixedSizeArrayMeta):
    """Typing helper for fixed-size numpy arrays.

    Example:
        vec: FixedSizeArray[(3,), np.float32]
        mat: FixedSizeArray[(4, 4)]
    """

    pass


class VectorizedTraitView(Generic[T]):
    def __init__(self, recs: list[np.ndarray], masks: list[np.ndarray]):
        self._recs = recs
        self._masks = masks
        self._data = np.concatenate([rec[mask] for rec, mask in zip(recs, masks)])
        self._offsets = np.cumulative_sum(
            [np.count_nonzero(m) for m in masks], include_initial=True
        )

    def __getattr__(self, name: str) -> np.ndarray[Any, Any]:
        """
        Type hint for attributes dynamically generated via make_property.
        This provides dot-notation autocompletion for any field name.
        """
        # This function should theoretically raise AttributeError if the property isn't found,
        # but because the fast properties are inserted first, it never runs.
        raise AttributeError(f"Attribute {name} not found statically or dynamically.")

    def write_back(self):
        for start, end, rec, mask in zip(
            self._offsets[:-1], self._offsets[1:], self._recs, self._masks
        ):
            rec[mask] = self._data[start:end]

    @staticmethod
    def make_property(field_name: str) -> property:
        def getter(self) -> np.ndarray:
            return self._data[field_name]

        def setter(self, value: np.ndarray) -> np.ndarray:
            assert value.shape == self._data[field_name].shape
            self._data[field_name] = value

        return property(getter, setter)


def _make_trait_view_class(trait_type: type[T]) -> type[T]:
    if trait_type in _view_cache:
        return _view_cache[trait_type]

    if not is_trait_type(trait_type):
        raise TypeError(f"{trait_type.__name__} must be a trait")

    name = f"{trait_type.__name__}View"

    def make_property(field_name: str, field_type: type):
        if field_type in (int, float, bool):

            def getter(self):
                return field_type(self._rec[field_name][self._index])
        else:

            def getter(self):
                return self._rec[field_name][self._index]

        def setter(self, value):
            self._rec[field_name][self._index] = value

        return property(getter, setter)

    def __init__(self, rec, index):
        setattr(self, "_rec", rec)
        setattr(self, "_index", index)

    namespace = {
        "__slots__": ("_rec", "_index"),
        "__init__": __init__,
        "__repr__": lambda self: (
            f"<{name}("
            + ", ".join(
                f"{f.name}={getattr(self, f.name)!r}" for f in fields(trait_type)
            )
            + ")>"
        ),
    }

    # Override dataclass fields with properties rerouting to numpy storage
    for f in fields(trait_type):
        namespace[f.name] = make_property(f.name, f.type)

    view_cls = type(name, (trait_type,), namespace)
    _view_cache[trait_type] = view_cls
    return view_cls


def _make_trait_vectorized_view_class(
    trait_type: type[T],
) -> type[VectorizedTraitView[T]]:
    if trait_type in _vector_view_cache:
        return _vector_view_cache[trait_type]

    if not is_trait_type(trait_type):
        raise TypeError(f"{trait_type.__name__} must be a trait")

    properties = {
        f.name: VectorizedTraitView.make_property(f.name) for f in fields(trait_type)
    }

    ViewCls = type(
        f"{trait_type.__name__}VectorizedView", (VectorizedTraitView,), properties
    )
    _vector_view_cache[trait_type] = ViewCls
    return ViewCls


def numpy_dtype_for_type(py_type: type[Any]) -> np.dtype:
    """Convert a Python or typing type into a corresponding NumPy dtype.

    Supports:
    - Built-in primitives (int, float, bool, str)
    - NumPy dtypes and scalars (np.float32, np.int64, etc.)
    - Tuple[...] composite types
    - Annotated[np.ndarray, {'shape': ..., 'dtype': ...}]
      or Annotated[np.ndarray, (shape, dtype)]
    - Falls back to object dtype otherwise
    """

    # 1️⃣ Directly handle np.dtype or NumPy scalar types
    if isinstance(py_type, np.dtype):
        return py_type
    if isinstance(py_type, type) and issubclass(py_type, np.generic):
        return np.dtype(py_type)

    # 2️⃣ Built-in Python types
    primitive_map = {
        int: np.int32,
        float: np.float32,
        bool: np.bool_,
        str: np.str_,
    }
    origin = get_origin(py_type)
    if origin is None and py_type in primitive_map:
        return np.dtype(primitive_map[py_type])

    args = get_args(py_type)

    # Tuple[...] → structured dtype
    if origin is tuple and args:
        fields = [(f"_{i}", numpy_dtype_for_type(arg)) for i, arg in enumerate(args)]
        return np.dtype(fields)

    # Annotated[np.ndarray, metadata]
    if origin is Annotated:
        base, *meta = args
        if base is np.ndarray:
            # Uniform metadata handling
            if len(meta) == 1 and isinstance(meta[0], dict):
                raw_shape = meta[0].get("shape", ())
                if isinstance(raw_shape, int):
                    shape = (raw_shape,)
                elif isinstance(raw_shape, tuple):
                    shape = raw_shape
                else:
                    shape = ()  # Default or error handling for invalid types
                dtype = np.dtype(meta[0].get("dtype", np.float32))
            elif len(meta) >= 1:
                shape = (
                    tuple(meta[0]) if isinstance(meta[0], (tuple, list)) else (meta[0],)
                )
                dtype = np.dtype(meta[1]) if len(meta) > 1 else np.float32
            else:
                shape, dtype = (), np.float32
            return np.dtype((dtype, shape))
        else:
            # Recurse into Annotated[T, ...]
            return numpy_dtype_for_type(base)

    # 4️⃣ Fallback: object dtype
    return np.dtype(np.object_)


def _build_trait_dtype(trait_type: type[T]) -> np.dtype:
    """Return a numpy dtype for a single trait dataclass (flat fields named exactly as dataclass fields)."""
    if not is_dataclass(trait_type):
        raise TypeError("build_trait_dtype expects a dataclass trait type")
    field_defs = [(f.name, numpy_dtype_for_type(f.type)) for f in fields(trait_type)]
    return np.dtype(field_defs)


class NumpyArcheType(ArcheType):
    def __init__(self, trait_types: ArchetypeKey, capacity: int = 256):
        self.key = trait_types
        self.entity_ids: list[int] = []
        self.trait_data: dict[type, np.ndarray] = {}
        self.trait_dtype: dict[type, np.dtype] = {}
        for t in trait_types:
            trait_dtype = _build_trait_dtype(t)
            self.trait_dtype[t] = trait_dtype
            self.trait_data[t] = np.empty(capacity, trait_dtype)

        self._capacity = capacity

    def __len__(self) -> int:
        """Return the number of entities stored in this archetype."""
        return len(self.entity_ids)

    def _ensure_capacity(self, new_count: int):
        """Ensure backing arrays can hold new_count entities."""
        new_cap = self._capacity
        if new_count < self._capacity // 4 and new_count > 16:
            new_cap = max(16, self._capacity // 2)
            for t, storage in self.trait_data.items():
                self.trait_data[t] = storage[:new_cap].copy()
        elif new_count > self._capacity:
            new_cap = max(self._capacity * 2, new_count)
            for t, storage in self.trait_data.items():
                new_storage = np.empty(new_cap, dtype=storage.dtype)
                new_storage[: self._capacity] = storage
                self.trait_data[t] = new_storage
        self._capacity = new_cap

    def add(self, entity: int, traits: Iterable[T]) -> int:
        """Add an entity and its trait instances to this archetype.

        Args:
            entity: The integer ID of the entity.
            traits: Iterable of trait instances to add.

        Returns:
            int: The index at which the entity was stored.

        Raises:
            ValueError: If the entity already exists or any required trait is missing.
        """

        if entity in self.entity_ids:
            raise ValueError(f"Entity {entity} already in this archetype")

        index = len(self.entity_ids)
        self.entity_ids.append(entity)

        self._ensure_capacity(index + 1)

        for trait in traits:
            trait_type = type(trait)
            if trait_type not in self.trait_data:
                raise TypeError(f"Trait type {trait_type} not part of this archetype")
            storage = self.trait_data[trait_type]
            storage[index] = tuple(getattr(trait, f.name) for f in fields(trait))

        return index

    def pop(self, index: int) -> tuple[int | None, list[T]]:
        """Remove an entity and return its traits and any swapped entity ID.

        Args:
            index: The index of the entity to pop.

        Returns:
            tuple[int | None, list[T]]: The ID of the moved entity (if any)
            and the list of removed trait instances.
        """
        n = len(self.entity_ids)
        if index < 0 or index >= n:
            raise IndexError("index out of range")

        last_index = n - 1
        removed_traits: list[T] = []

        # Get the traits to return
        for trait_type, storage in self.trait_data.items():
            vals = {
                f: storage[f][last_index].item()
                if isinstance(storage[f][last_index], np.ndarray)
                and storage[f][last_index].ndim == 0
                else storage[f][last_index]
                for f in storage.dtype.names
            }
            removed_traits.append(trait_type(**vals))

        moved_entity = None
        if index == last_index:
            self.entity_ids.pop()
        else:
            moved_entity = self.entity_ids[-1]
            self.entity_ids[index] = moved_entity
            self.entity_ids.pop()
            for storage in self.trait_data.values():
                for field in storage.dtype.names:
                    storage[field][index] = storage[field][last_index]

        self._ensure_capacity(n - 1)

        return moved_entity, removed_traits

    def get_traits(self) -> Iterator[tuple[int, dict[type[T], T]]]:
        """Iterate over all entities and their associated trait mappings.

        Yields:
            tuple[int, dict[type[T], T]]: Each entity ID with its trait type-to-instance map.
        """
        for idx, entity in enumerate(self.entity_ids):
            yield (
                entity,
                {
                    t_type: _make_trait_view_class(t_type)(t_store, idx)
                    for t_type, t_store in self.trait_data.items()
                },
            )

    def get_trait_data(self, trait_type: type[T]) -> np.ndarray | None:
        return self.trait_data.get(trait_type, None)

    def empose_order(self, order: Iterable[int]) -> None:
        order = list(order)
        n = len(order)
        if n != len(self):
            raise RuntimeError(f"Invalid order, expected {len(self)} entries, got {n}")

        self.entity_ids = [self.entity_ids[i] for i in order]
        for t in self.trait_data:
            self.trait_data[t][:n] = self.trait_data[t][order]

    @property
    def capacity(self) -> int:
        return self._capacity


class NumpyArchetypeWorld(ArchetypeWorld):
    def __init__(self) -> None:
        self._archetypes: dict[ArchetypeKey, NumpyArcheType] = dict()
        super().__init__()

    def _get_or_create_archetype(self, key: ArchetypeKey) -> NumpyArcheType:
        """Retrieve an existing archetype or create a new one.

        Args:
            key: The canonical tuple of trait classes for the archetype.

        Returns:
            NumpyArcheType: The corresponding archetype instance using numpy storage if possible.
        """
        if key not in self._archetypes:
            self._archetypes[key] = NumpyArcheType(key)
        return self._archetypes[key]

    def get_vectorized_entities(
        self, *trait_types: type[T], tags: Optional[set[str]] = None
    ) -> tuple[list[Entity], tuple[VectorizedTraitView[T], ...]]:
        all_recs = {t: [] for t in trait_types}
        all_masks = {t: [] for t in trait_types}
        entities: list[Entity] = []
        views: list = []
        req_trait_types = set(trait_types)

        for arch in self._archetypes.values():
            if not req_trait_types.issubset(arch.key):
                continue

            num_alive = len(arch)
            if num_alive == 0:
                continue

            if tags:
                entity_mask = np.array(
                    self.has_tags(eid, *tags) for eid in arch.entity_ids
                )
            else:
                entity_mask = np.ones(num_alive, dtype=bool)
            entities.extend(
                [
                    Entity(eid, self)
                    for eid, has_tags in zip(arch.entity_ids, entity_mask)
                    if has_tags
                ]
            )

            for trait_type in trait_types:
                data = arch.get_trait_data(trait_type)

                all_recs[trait_type].append(data[:num_alive])
                all_masks[trait_type].append(entity_mask)

        views = []
        for trait_type in trait_types:
            ViewType = _make_trait_vectorized_view_class(trait_type)
            instance = ViewType(all_recs[trait_type], all_masks[trait_type])
            views.append(instance)

        return entities, tuple(views) if len(views) > 1 else (views[0],)

    def get_vectorized_traits(
        self, *trait_types: type[T], tags: Optional[set[str]] = None
    ) -> tuple[T, ...]:
        return self.get_vectorized_entities(*trait_types, tags=tags)[1]
