# tests/test_sparse_world.py
import pytest
from buds.sparse import SparseWorld
from tests.world_contract import WorldContract


class TestSparseWorld(WorldContract):
    """Run the ECS contract tests on the SparseWorld."""

    def make_world(self):
        return SparseWorld()
