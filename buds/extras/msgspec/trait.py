import msgspec

from ...base import register_trait_base_class


class MSGSpecTrait(msgspec.Struct):
    pass


register_trait_base_class(MSGSpecTrait)
