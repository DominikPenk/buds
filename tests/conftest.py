# tests/conftest.py
import pytest

from buds.archetype import ArchetypeWorld
from buds.extras import NumpyArchetypeWorld
from buds.sparse import SparseWorld

WORLD_IMPLEMENTATIONS = [
    SparseWorld,
    ArchetypeWorld,
    NumpyArchetypeWorld,
]


@pytest.fixture(params=WORLD_IMPLEMENTATIONS, scope="function")
def world(request):
    """Fixture that provides a fresh instance of each known World implementation."""
    w = request.param()
    yield w
