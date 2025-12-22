from pydantic import BaseModel

from ...base import register_trait_base_class


class PydanticTrait(BaseModel):
    pass


register_trait_base_class(BaseModel)
