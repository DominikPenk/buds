from pydantic import BaseModel

from ...base import register_trait_base_class


class PydanticTrait(BaseModel):
    """Base class for defining ECS traits based on pydantic.BaseModel

    Subclasses of `PydanticTrait` are automatically marked as traits.
    """

    pass


register_trait_base_class(BaseModel)
