# tests/test_sparse_world.py
import numpy as np

from buds.base import Trait
from buds.extras.numpy_archetype import Matrix2x2, NumpyArchetypeWorld
from tests.world_contract import Position, Velocity, WorldContract


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

    def test_structured_array_dtype_matches_trait_fields(self):
        world = self.make_world()

        # Create entities to force archetype creation
        e1 = world.create_entity(Position(1, 2), Velocity(0.1, 0.2))

        # Inspect the underlying archetype storage via get_vectorized_entities
        entities, (pos_view, vel_view) = world.get_vectorized_entities(
            Position, Velocity
        )

        # The underlying _data for views should be structured arrays with fields matching dataclass
        assert "x" in pos_view._data.dtype.names
        assert "y" in pos_view._data.dtype.names
        assert "dx" in vel_view._data.dtype.names
        assert "dy" in vel_view._data.dtype.names

    def test_fixed_size_array_dtype_and_resize_behavior(self):
        class Matrix(Trait):
            m: Matrix2x2

        world = self.make_world()

        # Create many entities to force multiple resizes (growth)
        for i in range(512):
            world.create_entity(
                Matrix(np.array([[i, i + 1], [i + 2, i + 3]], dtype=np.float32))
            )

        # Collect vectorized view and ensure data shape is preserved
        ents, (mat_view,) = world.get_vectorized_entities(Matrix)
        # For structured arrays the field for 'm' stores the shape metadata
        field_dtype = mat_view._data.dtype.fields["m"][0]
        shape = getattr(field_dtype, "shape", None)
        assert shape == (2, 2)

        # Now remove many entities to trigger shrink
        ids = [e.id for e in ents]
        for eid in ids[:400]:
            world.delete_entity(eid)

        # New view should still report same field shape
        ents2, (mat_view2,) = world.get_vectorized_entities(Matrix)
        field_dtype2 = mat_view2._data.dtype.fields["m"][0]
        shape2 = getattr(field_dtype2, "shape", None)
        assert shape2 == (2, 2)

    def test_vectorized_query_masks_and_write_back(self):
        world = self.make_world()

        e1 = world.create_entity(Position(0, 0), Velocity(1.0, 2.0))
        e2 = world.create_entity(Position(10, 20), Velocity(3.0, 4.0))

        # Request vectorized entities with tags filter (none set) should return both
        ents, (pos_view, vel_view) = world.get_vectorized_entities(Position, Velocity)
        assert len(ents) == 2

        # Modify the vectorized arrays and write back
        pos_view._data["x"] += 5
        vel_view._data["dy"] *= -1
        pos_view.write_back()
        vel_view.write_back()

        # Fetch via normal trait access to confirm updates
        results = list(world.get_entities_from_traits(Position, Velocity))
        xs = [p.x for _, (p, _) in results]
        dys = [v.dy for _, (_, v) in results]
        assert xs == [5, 15]
        assert dys == [-2.0, -4.0]
