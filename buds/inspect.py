"""
Schema inspection and adaptation utilities for trait-based models.

This module provides a flexible adapter-based system for inspecting
"traits" (including dataclasses) and converting their fields and class
attributes into a unified schema representation.

Key concepts:
- FieldSchema: A normalized representation of a single field.
- TraitSchema: A normalized representation of a trait type.
- Adapters: Pluggable strategies that control how fields and traits
  are inspected and converted into schemas.

The system supports extensibility through adapter registration and
resets to a default configuration on import.
"""

from __future__ import annotations

from dataclasses import _MISSING_TYPE, MISSING, Field, dataclass, fields, is_dataclass
from inspect import isclass
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Mapping,
    Optional,
    Protocol,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

from .base import is_trait, is_trait_type


@runtime_checkable
class FieldAdapter(Protocol):
    """Protocol for adapters that build schemas for individual fields.

    Field adapters are responsible for determining whether they can
    handle a given field annotation and source, and for constructing
    the corresponding FieldSchema.
    """

    def is_applicable(self, annotation: type[Any], source: Any) -> bool:
        """Return whether this adapter can handle the given field.

        Args:
            annotation: The type annotation of the field.
            source: The original source object (e.g., dataclasses.Field).

        Returns:
            True if the adapter can handle the field, False otherwise.
        """
        ...

    def build_schema(
        self, name: str, annotation: type[Any], source: Any
    ) -> FieldSchema:
        """Build a FieldSchema for the given field.

        Args:
            name: The field name.
            annotation: The type annotation of the field.
            source: The original source object.

        Returns:
            A constructed FieldSchema instance.
        """
        ...


@runtime_checkable
class TraitAdapter(Protocol):
    """Protocol for adapters that build schemas for trait types.

    Trait adapters are responsible for converting an entire trait
    (class) into its instance-level and class-level fields.
    """

    def is_applicable(self, trait_type: type[Any]) -> bool:
        """Return whether this adapter can handle the given trait type.

        Args:
            trait_type: The trait class to inspect.

        Returns:
            True if the adapter applies to the trait type, False otherwise.
        """
        ...

    def build_fields(self, trait_type: type[Any]) -> list[FieldSchema]:
        """Build schemas for instance-level fields.

        Args:
            trait_type: The trait class to inspect.

        Returns:
            A list of FieldSchema objects for instance attributes.
        """
        ...

    def build_class_fields(self, trait_type: type[Any]) -> list[FieldSchema]:
        """Build schemas for class-level fields.

        Args:
            trait_type: The trait class to inspect.

        Returns:
            A list of FieldSchema objects for class attributes.
        """
        ...


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Specification describing default and requirement semantics of a field.

    Attributes:
        default: The default value, or MISSING if none is defined.
        default_factory: A callable producing a default value, or MISSING.
        required: Whether the field is required (no default provided).
    """

    default: Any | _MISSING_TYPE
    default_factory: Callable[[], Any] | _MISSING_TYPE
    required: bool


@dataclass(frozen=True, slots=True)
class UnifiedMetadata:
    """Unified container for metadata originating from multiple sources.

    Attributes:
        annotated: Metadata extracted from typing.Annotated, if present.
        field: Metadata extracted from dataclasses.Field, if present.
    """

    annotated: tuple[Any, ...] | None
    field: Mapping[str, Any] | None


class FieldSchema:
    """Normalized schema representation of a single field.

    FieldSchema instances are created via registered FieldAdapter
    implementations, which encapsulate logic for interpreting annotations
    and source metadata.
    """

    __slots__ = ("name", "type", "metadata", "spec")
    _adapters: list[FieldAdapter] = []

    def __init__(
        self,
        name: str,
        py_type: type[Any],
        metadata: Any,
        spec: Optional[FieldSpec] = None,
    ):
        """Initialize a FieldSchema.

        Args:
            name: The field name.
            py_type: The resolved Python type of the field.
            metadata: Unified metadata associated with the field.
            spec: Optional specification describing defaults and requirement.
        """
        self.name = name
        self.type = py_type
        self.metadata = metadata
        self.spec = spec

    @classmethod
    def create(
        cls, name: str, annotation: type[Any], source: Any = None
    ) -> FieldSchema:
        """Create a FieldSchema using the first applicable registered adapter.

        Args:
            name: The field name.
            annotation: The field's type annotation.
            source: Optional source object providing additional context.

        Returns:
            A constructed FieldSchema.

        Raises:
            ValueError: If no registered adapter can handle the field.
        """
        for adapter in cls._adapters:
            if adapter.is_applicable(annotation, source):
                return adapter.build_schema(name, annotation, source)
        raise ValueError(f"No adapter available for field {name}")

    @classmethod
    def register_adapter(
        cls, adapter: FieldAdapter, index: int = 0
    ) -> type[FieldSchema]:
        """Register a FieldAdapter.

        Args:
            adapter: The adapter instance to register.
            index: The insertion index, controlling priority.

        Returns:
            The FieldSchema class, allowing fluent registration.
        """
        cls._adapters.insert(index, adapter)
        return cls


class TraitSchema:
    """Normalized schema representation of a trait type.

    A TraitSchema encapsulates both instance-level fields and class-level
    fields as discovered by registered TraitAdapter implementations.
    """

    __slots__ = ("type", "fields", "class_fields")
    _adapters: list[TraitAdapter] = []

    def __init__(
        self,
        trait_type: type[Any],
        fields: list[FieldSchema],
        class_fields: list[FieldSchema],
    ):
        """Initialize a TraitSchema.

        Args:
            trait_type: The trait class being described.
            fields: Schemas for instance-level fields.
            class_fields: Schemas for class-level fields.
        """
        self.type = trait_type
        self.fields = fields
        self.class_fields = class_fields

    @classmethod
    def create(cls, trait_type: type[Any]) -> TraitSchema:
        """Create a TraitSchema using the first applicable registered adapter.

        Args:
            trait_type: The trait class to inspect.

        Returns:
            A constructed TraitSchema.

        Raises:
            ValueError: If no registered adapter can handle the trait type.
        """

        for adapter in cls._adapters:
            if adapter.is_applicable(trait_type):
                return TraitSchema(
                    trait_type=trait_type,
                    fields=adapter.build_fields(trait_type),
                    class_fields=adapter.build_class_fields(trait_type),
                )
        raise ValueError(f"No adapter available for type {trait_type.__name__}")

    @classmethod
    def register_adapter(
        cls, adapter: TraitAdapter, index: int = 0
    ) -> type[TraitSchema]:
        """Register a TraitAdapter.

        Args:
            adapter: The adapter instance to register.
            index: The insertion index, controlling priority.

        Returns:
            The TraitSchema class, allowing fluent registration.
        """
        cls._adapters.insert(index, adapter)
        return cls


class DataclassTraitAdapter:
    """TraitAdapter implementation for dataclass-based traits."""

    def is_applicable(self, trait_type: type[Any]) -> bool:
        return is_dataclass(trait_type)

    def build_fields(self, trait_type: type[Any]) -> list[FieldSchema]:
        hints = get_type_hints(trait_type, include_extras=True)
        return [
            FieldSchema.create(f.name, hints[f.name], f)
            for f in fields(trait_type)
            if get_origin(hints[f.name]) is not ClassVar
        ]

    def build_class_fields(self, trait_type: type[Any]) -> list[FieldSchema]:
        hints = get_type_hints(trait_type, include_extras=True)
        return [
            FieldSchema.create(name, type_, None)
            for name, type_ in hints.items()
            if get_origin(type_) is ClassVar
        ]


# TODO: Strip more extra information, not only Annotated
class DefaultFieldAdapter:
    """Fallback FieldAdapter handling standard Python and dataclass fields.

    This adapter also extracts metadata from typing.Annotated and
    dataclasses.Field where applicable.
    """

    def is_applicable(self, annotation: type[Any], source: Any) -> bool:
        return True

    def build_schema(
        self, name: str, annotation: type[Any], source: Any
    ) -> FieldSchema:
        """Build a FieldSchema from a standard field definition.

        Args:
            name: The field name.
            annotation: The field's type annotation.
            source: Optional dataclasses.Field providing defaults and metadata.

        Returns:
            A constructed FieldSchema instance.
        """

        origin = get_origin(annotation)
        if origin is Annotated:
            py_type, *annotated_metadata = get_args(annotation)
            annotated_metadata = tuple(annotated_metadata) or None
        else:
            annotated_metadata = None
            py_type = annotation

        if isinstance(source, Field):
            has_default = source.default is not MISSING
            has_default_factory = source.default_factory is not MISSING
            is_required = not (has_default or has_default_factory)
            field_spec = FieldSpec(
                default=source.default,
                default_factory=source.default_factory,
                required=is_required,
            )
            field_metadata = dict(source.metadata) if source.metadata else None
        else:
            field_spec = None
            field_metadata = None

        metadata = UnifiedMetadata(annotated_metadata, field_metadata)

        return FieldSchema(name, py_type, metadata, field_spec)


def inspect_trait(trait_type_or_instance: type[Any] | Any) -> TraitSchema:
    """Inspect a trait type or instance and return its TraitSchema.

    Args:
        trait_type_or_instance: A trait class or trait instance.

    Returns:
        A TraitSchema describing the trait.

    Raises:
        ValueError: If the object is not a valid trait or trait type.
    """
    if isclass(trait_type_or_instance):
        if not is_trait_type(trait_type_or_instance):
            raise ValueError("The inspected object must be a trait or trait type.")
        return TraitSchema.create(trait_type_or_instance)
    else:
        if not is_trait(trait_type_or_instance):
            raise ValueError("The inspected object must be a trait or trait type.")
        return TraitSchema.create(type(trait_type_or_instance))


def reset_adapters():
    """Reset adapter registries to their default configuration.

    This clears all registered FieldAdapter and TraitAdapter instances
    and re-registers the built-in defaults.
    """
    TraitSchema._adapters.clear()
    FieldSchema._adapters.clear()
    TraitSchema.register_adapter(DataclassTraitAdapter())
    FieldSchema.register_adapter(DefaultFieldAdapter())


reset_adapters()
