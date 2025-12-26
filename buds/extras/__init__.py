import warnings

try:
    from .numpy.dtypes import NumpyArrayMetadata
    from .numpy.numpy_archetype import NumpyArchetypeWorld
except ImportError:
    warnings.warn("Could not import numpy archetype. Numpy is probably not installed")

try:
    from .msgspec.trait import MSGSpecTrait
except ImportError:
    warnings.warn(
        "Could not import msgspec trait base. msgspec is probably not installed"
    )


try:
    from .pydantic import PydanticTrait
except ImportError:
    warnings.warn(
        "Could not import pydantic trait base. Pydantic is probably not installed"
    )
