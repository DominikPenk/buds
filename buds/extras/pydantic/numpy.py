from typing import Optional

import pydantic

from ...inspect import TraitSchema
from ..numpy import views


class PydanticViewbuilder(views.ViewBuilder):
    def __init__(self, schema: TraitSchema, name: str):
        super().__init__(schema, name)
        self.bases = ()


# TODO: Add generic forward, since we do not inherit from the trait itself
class PydanticStructViewGenerator:
    @classmethod
    def create_view(cls, schema: TraitSchema, name: str) -> Optional[type]:
        if not issubclass(schema.type, pydantic.BaseModel):
            return None

        view_cls = (
            PydanticViewbuilder(schema, name)
            .add_slots()
            .add_init()
            .add_repr()
            .add_properties()
            .build()
        )
        return view_cls


# use a higer priority than default (0)
views.register_view_adapter(PydanticStructViewGenerator, priority=10)
