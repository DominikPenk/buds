"""
View generation utilities for Trait-based data models.

This module provides mechanisms for dynamically generating *view classes*
for traits. Views act as lightweight, record-oriented or vectorized
interfaces over underlying data storage (e.g., NumPyStructured arrays), while
preserving trait semantics.

Two categories of views are supported:

1. Single-trait views
   - Represent a single logical trait instance backed by array-based storage.
   - Generated dynamically using schema inspection and adapters.
   - Method implementations are preserved

2. Vectorized trait views
   - Represent many trait instances at once using NumPy vectorization.
   - Provide efficient column-wise access and mutation.

The system is adapter-driven and extensible via view generator adapters.
"""

import inspect
from dataclasses import is_dataclass
from functools import lru_cache
from typing import Any, Generic, Optional, Protocol, Self, TypeVar

import numpy as np

from ...base import Trait, is_trait_type, mark_as_trait
from ...inspect import FieldSchema, TraitSchema

T = TypeVar("T", bound=Trait)


# ---------------------------------------------------------------------------
# Single Trait Views
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def resolve_class_attr(cls: type[Any], name: str):
    """Resolve a class attribute without triggering descriptor logic.

    This function uses ``inspect.getattr_static`` to retrieve attributes
    exactly as defined on the class, bypassing ``__get__`` behavior.

    Args:
        cls: The class on which to resolve the attribute.
        name: The attribute name.

    Returns:
        The raw attribute object.
    """
    return inspect.getattr_static(cls, name)


class ViewGeneratorAdapter(Protocol):
    """Protocol for adapters that generate trait view classes."""

    @classmethod
    def create_view(cls, schema: TraitSchema, name: str) -> Optional[type]:
        """Create a view class for the given trait schema.

        Args:
            schema: The inspected schema of the trait.
            name: The name to assign to the generated view class.

        Returns:
            A new view class, or None if the adapter does not apply.
        """
        ...


class ForwardMeta(type):
    """Metaclass enabling attribute forwarding for class-level descriptors.

    Assignments on the view class are forwarded to descriptors defined
    on the original trait class, if applicable.
    """

    def __setattr__(cls, name, value):
        attr = cls.__dict__.get(name)
        if hasattr(attr, "__set__"):
            attr.__set__(None, value)  # type: ignore
        else:
            super().__setattr__(name, value)


class ClassVarForward:
    """Descriptor that forwards access to a class variable on the original class.

    This allows class variables of the trait to be accessed and mutated
    through the generated view class.
    """

    def __init__(self, cls, name):
        """
        Args:
            cls: The original trait class.
            name: The name of the class variable.
        """
        self._cls = cls
        self._name = name

    def __get__(self, instance, owner):
        """Always read the value from the original class."""
        return getattr(self._cls, self._name)

    def __set__(self, instance, value):
        """Always write the value to the original class."""
        setattr(self._cls, self._name, value)


class ViewBuilder:
    """Incremental builder for dynamically generated trait view classes.

    ViewBuilder provides a fluent, step-by-step API for assembling a view
    class that exposes trait fields backed by external storage (e.g.
    NumPy record arrays) while preserving trait behavior.

    The builder is intentionally modular: each ``add_*`` method installs
    a specific capability (e.g. properties, delegation, equality), allowing
    custom view generators to selectively override or extend behavior.

    For most use cases, ``add_defaults()`` should be used as the canonical
    starting point. It installs the full, recommended feature set for a
    standard trait view, including:

    - Slot-based storage (``__slots__``)
    - Record/index initialization
    - Field-based ``__repr__``
    - Per-field properties backed by array storage
    - Dynamic delegation to the original trait class
    - Field-wise equality semantics

    Custom view generators are expected to either call ``add_defaults()``
    directly or to recompose its steps explicitly if deviations from the
    standard behavior are required.
    """

    def __init__(self, schema: TraitSchema, name: str):
        """
        Args:
            schema: The schema describing the trait being viewed.
            name: The name of the view class to be generated.
        """
        self.schema = schema
        self.name = name
        self.bases = ()
        self.slots = ("_rec", "_index")
        self.namespace: dict[str, Any] = {}

    def _make_property(self, field: FieldSchema) -> property:
        """Create a property backed by record-array storage."""
        name = field.name
        typ = field.type

        def getter(self):
            value = self._rec[name][self._index]
            return typ(value) if typ in (int, float, bool, str) else value

        def setter(self, value):
            self._rec[name][self._index] = value

        return property(getter, setter)

    def add_init(self) -> Self:
        """Add an ``__init__`` method initializing record and index."""

        def __init__(self, rec, index):
            self._rec = rec
            self._index = index

        self.namespace["__init__"] = __init__
        return self

    def add_repr(self) -> Self:
        """Add a readable ``__repr__`` based on trait fields."""
        fields = self.schema.fields
        name = self.name

        def __repr__(self):
            values = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in fields)
            return f"<{name}({values})>"

        self.namespace["__repr__"] = __repr__
        return self

    def add_properties(self) -> Self:
        """Add property accessors for all instance-level fields."""
        for field in self.schema.fields:
            assert field.name not in self.namespace
            self.namespace[field.name] = self._make_property(field)
        return self

    def add_slots(self) -> Self:
        """Add ``__slots__`` while respecting slots defined on base classes."""
        base_slots = set()
        for base in self.bases:
            base_slots |= set(getattr(base, "__slots__", ()))

        slots = tuple(s for s in ("_rec", "_index") if s not in base_slots)
        if slots:
            self.namespace["__slots__"] = slots

        return self

    def add_eq(self) -> Self:
        """Add an equality comparison based on field values."""
        fields = self.schema.fields

        def __eq__(self, other):
            if other is self:
                return True

            try:
                return all(
                    getattr(self, field.name) == getattr(other, field.name, None)
                    for field in fields
                )
            except AttributeError:
                return NotImplemented

        self.namespace["__eq__"] = __eq__
        return self

    def add_dynamic_delegation(self) -> Self:
        """Delegate missing attributes to the original trait class.

        Supports instance methods, static methods, class methods,
        and class variables.
        """
        cls = self.schema.type

        def __getattr__(self, name: str) -> Any:
            attr = resolve_class_attr(cls, name)

            if isinstance(attr, classmethod):
                return attr.__func__.__get__(cls, cls)

            if isinstance(attr, staticmethod):
                return attr.__func__

            if not callable(attr):
                return attr

            return attr.__get__(self, type(self))

        self.namespace["__getattr__"] = __getattr__

        # Forward class methods
        for name, attr in cls.__dict__.items():
            if name.startswith("__"):
                continue
            if not isinstance(attr, classmethod):
                continue
            self.namespace[name] = getattr(cls, name).__get__(cls, cls)

        # Forward class variables
        for field in self.schema.class_fields:
            # Actually create a class varaible that can be used
            assert field.name not in self.namespace
            self.namespace[field.name] = ClassVarForward(cls, field.name)

        return self

    def add_defaults(self) -> Self:
        """Install the standard, full-featured configuration for a trait view.

        This method applies the canonical sequence of builder steps used by
        the default view generators. It is intended to be the *primary entry
        point* when constructing a view unless specialized behavior is needed.

        Specifically, this method installs:

        - ``__slots__`` for memory-efficient storage
        - ``__init__`` accepting a backing record and index
        - A field-aware ``__repr__``
        - Per-field properties backed by record-array access
        - Dynamic delegation of methods to the original trait
        - Field-wise ``__eq__`` semantics

        Custom ``ViewGeneratorAdapter`` implementations should generally call
        this method unless they have a strong reason to override individual
        steps. Deviations are best expressed by composing ``add_*`` methods
        manually instead of partially reimplementing defaults.

        Returns:
            Self, allowing fluent chaining.
        """
        return (
            self.add_slots()
            .add_init()
            .add_repr()
            .add_properties()
            .add_dynamic_delegation()
            .add_eq()
        )

    def make_subclass(self) -> Self:
        """Make the generated view a subclass of the original trait."""
        self.bases = (self.schema.type,)
        return self

    def build(self) -> type:
        """Construct and return the final view class."""
        return ForwardMeta(self.name, self.bases, self.namespace)


class DataclassViewGenerator:
    """View generator adapter for dataclass-based traits."""

    @classmethod
    def create_view(cls, schema: TraitSchema, name: str) -> Optional[type]:
        if not is_dataclass(schema.type):
            return None

        return ViewBuilder(schema, name).add_defaults().build()


_VIEW_GENERATORS: list[tuple[int, ViewGeneratorAdapter]] = []
_VIEW_CACHE: dict[type, type] = {}


def create_view_class(trait_type: type[Any]) -> type:
    """Create or retrieve a cached view class for a trait type.

    Args:
        trait_type: The trait class for which to create a view.

    Returns:
        A dynamically generated view class.

    Raises:
        TypeError: If the input is not a trait type.
        RuntimeError: If no view adapter can generate a view.
    """

    if not is_trait_type(trait_type):
        raise TypeError("trait_type is not a Trait class type")
    if trait_type in _VIEW_CACHE:
        return _VIEW_CACHE[trait_type]

    schema = TraitSchema.create(trait_type)
    name = f"{trait_type.__name__}View"

    for _, adapter in _VIEW_GENERATORS:
        view_cls = adapter.create_view(schema, name)
        if view_cls is not None:
            mark_as_trait(view_cls)
            _VIEW_CACHE[trait_type] = view_cls
            return view_cls

    raise RuntimeError(f"Cannot create a view for type {trait_type}")


def register_view_adapter(adapter: ViewGeneratorAdapter, *, priority: int = 0):
    """Register a view generator adapter.

    Args:
        adapter: The adapter to register.
        priority: Higher priority adapters are tried first.
    """
    _VIEW_GENERATORS.append((priority, adapter))
    _VIEW_GENERATORS.sort(key=lambda x: -x[0])


register_view_adapter(DataclassViewGenerator)

# ---------------------------------------------------------------------------
# Vectorized Trait Views
# ---------------------------------------------------------------------------


class VectorizedTraitView(Generic[T]):
    """Vectorized view over multiple trait instances using NumPy arrays.

    This class provides efficient, column-wise access to trait fields
    across many instances, with optional write-back to the source arrays.
    """

    def __init__(self, recs: list[np.ndarray], masks: list[np.ndarray]):
        """
        Args:
            recs: List of structured NumPy arrays containing trait data.
            masks: Boolean masks selecting active records in each array.
        """
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
        """Write modified data back into the original record arrays."""
        for start, end, rec, mask in zip(
            self._offsets[:-1], self._offsets[1:], self._recs, self._masks
        ):
            rec[mask] = self._data[start:end]

    @staticmethod
    def make_property(field_name: str) -> property:
        """Create a vectorized property for a given field name."""

        def getter(self) -> np.ndarray:
            return self._data[field_name]

        def setter(self, value: np.ndarray) -> None:
            assert value.shape == self._data[field_name].shape
            self._data[field_name] = value

        return property(getter, setter)


_VECTORIZED_TRAIT_VIEW_CACHE: dict[type, type[VectorizedTraitView]] = {}


def create_vectorized_view_class(trait_type: type[T]) -> type[VectorizedTraitView[T]]:
    """Create or retrieve a vectorized view class for a trait type.

    Args:
        trait_type: The trait class to vectorize.

    Returns:
        A VectorizedTraitView subclass specialized for the trait.

    Raises:
        TypeError: If the input is not a trait type.
    """
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
