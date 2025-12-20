from dataclasses import is_dataclass
from typing import Any, Generic, Optional, Protocol, Self, TypeVar

import numpy as np

from ...base import Trait, is_trait_type
from ...inspect import FieldSchema, TraitSchema

# ---------------------------------------------------------------------------
# Single Trait Views
# ---------------------------------------------------------------------------


class ViewGeneratorAdapter(Protocol):
    @classmethod
    def create_view(cls, schema: TraitSchema, name: str) -> Optional[type]: ...


class ViewBuilder:
    def __init__(self, schema: TraitSchema, name: str):
        self.schema = schema
        self.name = name
        self.bases = (schema.type,)
        self.namespace: dict[str, Any] = {}

    def add_init(self) -> Self:
        def __init__(self, rec, index):
            self._rec = rec
            self._index = index

        self.namespace["__init__"] = __init__
        return self

    def add_repr(self) -> Self:
        fields = self.schema.fields
        name = self.name

        def __repr__(self):
            values = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in fields)
            return f"<{name}({values})>"

        self.namespace["__repr__"] = __repr__
        return self

    def add_properties(self) -> Self:
        for field in self.schema.fields:
            self.namespace[field.name] = self._make_property(field)
        return self

    def add_slots(self) -> Self:
        base_slots = set()
        for base in self.bases:
            base_slots |= set(getattr(base, "__slots__", ()))

        slots = tuple(s for s in ("_rec", "_index") if s not in base_slots)
        if slots:
            self.namespace["__slots__"] = slots

        return self

    def _make_property(self, field: FieldSchema) -> property:
        name = field.name
        typ = field.type

        def getter(self):
            value = self._rec[name][self._index]
            return typ(value) if typ in (int, float, bool, str) else value

        def setter(self, value):
            self._rec[name][self._index] = value

        return property(getter, setter)

    def add_defaults(self) -> Self:
        return self.add_slots().add_init().add_repr().add_properties()

    def build(self) -> type:
        return type(self.name, self.bases, self.namespace)


class DataclassViewGenerator:
    @classmethod
    def create_view(cls, schema: TraitSchema, name: str) -> Optional[type]:
        if not is_dataclass(schema.type):
            return None

        return ViewBuilder(schema, name).add_defaults().build()


_VIEW_GENERATORS: list[tuple[int, ViewGeneratorAdapter]] = []
_VIEW_CACHE: dict[type, type] = {}


def create_view_class(trait_type: type[Any]) -> type:
    if not is_trait_type(trait_type):
        raise ValueError("trait_type is not a Trait class type")
    if trait_type in _VIEW_CACHE:
        return _VIEW_CACHE[trait_type]

    schema = TraitSchema.create(trait_type)
    name = f"{trait_type.__name__}View"

    for _, adapter in _VIEW_GENERATORS:
        view_cls = adapter.create_view(schema, name)
        if view_cls is not None:
            _VIEW_CACHE[trait_type] = view_cls
            return view_cls

    raise RuntimeError(f"Cannot create a view for type {trait_type}")


def register_view_adapter(adapter: ViewGeneratorAdapter, *, priority: int = 0):
    _VIEW_GENERATORS.append((priority, adapter))
    _VIEW_GENERATORS.sort(key=lambda x: -x[0])


register_view_adapter(DataclassViewGenerator)

# ---------------------------------------------------------------------------
# Vectorized Trait Views
# ---------------------------------------------------------------------------
T = TypeVar("T", bound=Trait)


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

        def setter(self, value: np.ndarray) -> None:
            assert value.shape == self._data[field_name].shape
            self._data[field_name] = value

        return property(getter, setter)


_VECTORIZED_TRAIT_VIEW_CACHE: dict[type, type[VectorizedTraitView]] = {}


def create_vectorized_view_class(trait_type: type[T]) -> type[VectorizedTraitView[T]]:
    if not is_trait_type(trait_type):
        raise TypeError(f"{trait_type.__name__} must be a trait")

    cached_type = _VECTORIZED_TRAIT_VIEW_CACHE.get(trait_type, None)
    if cached_type is not None:
        return cached_type

    fields = TraitSchema.create(trait_type).fields

    properties = {
        field.name: VectorizedTraitView.make_property(field.name) for field in fields
    }
    name = f"{trait_type.__name__}VectorizedView"

    ViewCls = type(name, (VectorizedTraitView,), properties)
    _VECTORIZED_TRAIT_VIEW_CACHE[trait_type] = ViewCls
    return ViewCls
