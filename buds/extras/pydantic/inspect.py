from typing import Any, ClassVar, get_origin, get_type_hints

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from ...inspect import MISSING, DefaultFieldAdapter, FieldSchema, FieldSpec, TraitSchema


class PydanticTraitAdapter:
    def is_applicable(self, trait_type: type[Any]) -> bool:
        return isinstance(trait_type, type) and issubclass(trait_type, BaseModel)

    def build_fields(self, trait_type: BaseModel) -> list[FieldSchema]:
        fields: dict[str, FieldInfo] = trait_type.model_fields
        return [FieldSchema.create(name, f.annotation, f) for name, f in fields.items()]  # type: ignore

    def build_class_fields(self, trait_type: type[Any]) -> list[FieldSchema]:
        hints = get_type_hints(trait_type, include_extras=True)
        return [
            FieldSchema.create(name, type_, None)
            for name, type_ in hints.items()
            if get_origin(type_) is ClassVar
        ]


class PydanticFieldAdapter(DefaultFieldAdapter):
    def is_applicable(self, annotation: type[Any], source: Any) -> bool:
        return isinstance(source, FieldInfo)

    def build_schema(
        self, name: str, annotation: type[Any], source: FieldInfo
    ) -> FieldSchema:
        result = super().build_schema(name, annotation, source)
        default_factory = (
            MISSING if source.default_factory is None else source.default_factory
        )
        field_spec = FieldSpec(
            default=source.default,
            default_factory=default_factory,  # type: ignore
            required=source.is_required(),
        )
        return FieldSchema(result.name, result.type, result.metadata, field_spec)


TraitSchema.register_adapter(PydanticTraitAdapter())  # type: ignore
FieldSchema.register_adapter(PydanticFieldAdapter())
