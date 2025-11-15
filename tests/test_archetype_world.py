# tests/test_sparse_world.py
import pytest
from buds.archetype import ArchetypeWorld
from tests.world_contract import WorldContract


class TestArchetypeWorld(WorldContract):
    """Run the ECS contract tests on the SparseWorld."""

    def make_world(self):
        return ArchetypeWorld()
