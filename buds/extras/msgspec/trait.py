import msgspec

from ...base import register_trait_base_class


class MSGSpecTrait(msgspec.Struct):
    """Base class for defining ECS traits based on msgspec.Struct

    Subclasses of `PydanticTrait` are automatically marked as traits.
    """

    pass


register_trait_base_class(MSGSpecTrait)
