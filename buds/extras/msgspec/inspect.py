from typing import Any, ClassVar, get_origin, get_type_hints

import msgspec

from ...inspect import DefaultFieldAdapter, FieldSchema, FieldSpec, TraitSchema


class MSGStructTraitAdapter:
    def is_applicable(self, trait_type: type[Any]) -> bool:
        return isinstance(trait_type, type) and issubclass(trait_type, msgspec.Struct)

    def build_fields(self, trait_type: type[Any]) -> list[FieldSchema]:
        hints = get_type_hints(trait_type, include_extras=True)
        fields: list[msgspec.inspect.Field] = msgspec.inspect.type_info(
            trait_type
        ).fields  # type: ignore
        return [
            FieldSchema.create(f.name, hints[f.name], f)
            for f in fields
            if get_origin(hints[f.name]) is not ClassVar
        ]

    def build_class_fields(self, trait_type: type[Any]) -> list[FieldSchema]:
        hints = get_type_hints(trait_type, include_extras=True)
        return [
            FieldSchema.create(name, type_, None)
            for name, type_ in hints.items()
            if get_origin(type_) is ClassVar
        ]


class MSGFieldAdapter(DefaultFieldAdapter):
    def is_applicable(self, annotation: type[Any], source: Any) -> bool:
        return isinstance(source, msgspec.inspect.Field)

    def build_schema(
        self, name: str, annotation: type[Any], source: msgspec.inspect.Field
    ) -> FieldSchema:
        result = super().build_schema(name, annotation, source)
        field_spec = FieldSpec(
            default=source.default,
            default_factory=source.default_factory,
            required=source.required,
        )
        return FieldSchema(result.name, result.type, result.metadata, field_spec)


TraitSchema.register_adapter(MSGStructTraitAdapter())
FieldSchema.register_adapter(MSGFieldAdapter())
