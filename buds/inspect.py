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
    def is_applicable(self, annotation: type[Any], source: Any) -> bool: ...
    def build_schema(
        self, name: str, annotation: type[Any], source: Any
    ) -> FieldSchema: ...


@runtime_checkable
class TraitAdapter(Protocol):
    def is_applicable(self, trait_type: type[Any]) -> bool: ...
    def build_fields(self, trait_type: type[Any]) -> list[FieldSchema]: ...
    def build_class_fields(self, trait_type: type[Any]) -> list[FieldSchema]: ...


@dataclass(frozen=True, slots=True)
class FieldSpec:
    default: Any | _MISSING_TYPE
    default_factory: Callable[[], Any] | _MISSING_TYPE
    required: bool


@dataclass(frozen=True, slots=True)
class UnifiedMetadata:
    annotated: tuple[Any, ...] | None
    field: Mapping[str, Any] | None


class FieldSchema:
    __slots__ = ("name", "type", "metadata", "spec")
    _adapters: list[FieldAdapter] = []

    def __init__(
        self,
        name: str,
        py_type: type[Any],
        metadata: Any,
        spec: Optional[FieldSpec] = None,
    ):
        self.name = name
        self.type = py_type
        self.metadata = metadata
        self.spec = spec

    @classmethod
    def create(
        cls, name: str, annotation: type[Any], source: Any = None
    ) -> FieldSchema:
        for adapter in cls._adapters:
            if adapter.is_applicable(annotation, source):
                return adapter.build_schema(name, annotation, source)
        raise ValueError(f"No adapter available for field {name}")

    @classmethod
    def register_adapter(
        cls, adapter: FieldAdapter, index: int = 0
    ) -> type[FieldSchema]:
        cls._adapters.insert(index, adapter)
        return cls


class TraitSchema:
    __slots__ = ("type", "fields", "class_fields")
    _adapters: list[TraitAdapter] = []

    def __init__(
        self,
        trait_type: type[Any],
        fields: list[FieldSchema],
        class_fields: list[FieldSchema],
    ):
        self.type = trait_type
        self.fields = fields
        self.class_fields = class_fields

    @classmethod
    def create(cls, trait_type: type[Any]) -> TraitSchema:
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
        cls._adapters.insert(index, adapter)
        return cls


class DataclassTraitAdapter:
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
    def is_applicable(self, annotation: type[Any], source: Any) -> bool:
        return True

    def build_schema(
        self, name: str, annotation: type[Any], source: Any
    ) -> FieldSchema:
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
    if isclass(trait_type_or_instance):
        if not is_trait_type(trait_type_or_instance):
            raise ValueError("The inspected object must be a trait or trait type.")
        return TraitSchema.create(trait_type_or_instance)
    else:
        if not is_trait(trait_type_or_instance):
            raise ValueError("The inspected object must be a trait or trait type.")
        return TraitSchema.create(type(trait_type_or_instance))


def reset_adapters():
    TraitSchema._adapters.clear()
    FieldSchema._adapters.clear()
    TraitSchema.register_adapter(DataclassTraitAdapter())
    FieldSchema.register_adapter(DefaultFieldAdapter())


reset_adapters()
