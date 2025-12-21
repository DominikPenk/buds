import warnings

try:
    from .numpy.numpy_archetype import NumpyArchetypeWorld
except ImportError:
    warnings.warn("Could not import numpy archetype. Numpy is probably not installed")

try:
    from .msgspec.msgspec import MSGSpecTrait
except ImportError:
    warnings.warn(
        "Could not import msgspec trait base. msgspec is probably not installed"
    )
