# tests/test_sparse_world.py
import pytest
import numpy as np
from buds.extras.numpy_archetype import NumpyArchetypeWorld
from tests.world_contract import WorldContract, Position, Velocity


class TestNumpyWorld(WorldContract):
    """Run the ECS contract tests on the NumpyArchetypeWorld."""

    def make_world(self):
        return NumpyArchetypeWorld()

        # -------------------------------------------------------------------------

    # NumPy-specific tests
    # -------------------------------------------------------------------------

    def test_get_vectorized_entities_returns_correct_shapes(self):
        """Ensure vectorized trait views have the right shape and data."""
        e1 = self.world.create_entity(Position(0, 0), Velocity(1.0, 2.0))
        e2 = self.world.create_entity(Position(10, 20), Velocity(3.0, 4.0))

        entities, (pos_view, vel_view) = self.world.get_vectorized_entities(
            Position, Velocity
        )

        # Two entities
        assert len(entities) == 2

        # The underlying _data arrays should match
        np.testing.assert_array_equal(pos_view._data["x"], [0, 10])
        np.testing.assert_array_equal(pos_view._data["y"], [0, 20])
        np.testing.assert_array_equal(vel_view._data["dx"], [1.0, 3.0])
        np.testing.assert_array_equal(vel_view._data["dy"], [2.0, 4.0])

    def test_write_back_updates_underlying_data(self):
        """Changing vectorized view data and calling write_back should update stored values."""
        e1 = self.world.create_entity(Position(1, 2), Velocity(0.5, 1.0))
        e2 = self.world.create_entity(Position(3, 4), Velocity(1.5, 2.0))

        entities, (pos_view, vel_view) = self.world.get_vectorized_entities(
            Position, Velocity
        )

        # Modify data in the vectorized view
        pos_view._data["x"] += 10
        vel_view._data["dy"] *= 2

        # Commit changes to archetype storage
        pos_view.write_back()
        vel_view.write_back()

        # Retrieve through standard trait access — must reflect updates
        results = list(self.world.get_entities_from_traits(Position, Velocity))
        assert len(results) == 2
        _, (p1, v1) = results[0]
        _, (p2, v2) = results[1]

        # Verify the modifications propagated correctly
        np.testing.assert_array_equal([p1.x, p2.x], [11, 13])
        np.testing.assert_array_equal([v1.dy, v2.dy], [2.0, 4.0])
