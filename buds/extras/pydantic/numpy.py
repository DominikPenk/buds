from typing import Optional

import pydantic

from ...inspect import TraitSchema
from ..numpy import views


# TODO: Add generic forward, since we do not inherit from the trait itself
class PydanticStructViewGenerator:
    @classmethod
    def create_view(cls, schema: TraitSchema, name: str) -> Optional[type]:
        if not issubclass(schema.type, pydantic.BaseModel):
            return None

        view_cls = views.ViewBuilder(schema, name).add_defaults().build()
        return view_cls


# use a higer priority than default (0)
views.register_view_adapter(PydanticStructViewGenerator, priority=10)
