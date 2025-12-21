from . import inspect
from .msgspec import MSGSpecTrait

try:
    import numpy as np

    from . import numpy

except ImportError:
    pass
