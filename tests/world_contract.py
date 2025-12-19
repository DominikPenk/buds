# tests/ecs_world_contract.py
import pytest

from buds.base import Entity, EntityNotFoundError, Trait, TraitNotFoundError, World


# Shared trait types used by the contract tests
class Position(Trait):
    x: int
    y: int


class Velocity(Trait):
    dx: float
    dy: float


class WorldContract:
    """
    Base test contract for any World implementation.

    Subclasses must implement make_world(self) and return a fresh world instance.
    The autouse fixture below will call make_world() and set self.world for all tests.
    """

    # Subclasses MUST override this
    def make_world(self) -> World:
        raise NotImplementedError("Subclasses must return a new World instance")

    @pytest.fixture(autouse=True)
    def _setup_world(self):
        # create and attach the world for use in tests as self.world
        self.world = self.make_world()
        yield

    # ---------------------------------------------------------------------------
    # Basic lifecycle tests
    # ---------------------------------------------------------------------------
    def test_create_entity_and_is_alive(self):
        e = self.world.create_entity()
        assert self.world.is_alive(e.id)

    def test_delete_entity_and_is_not_alive(self):
        e = self.world.create_entity()
        self.world.delete_entity(e.id)
        assert not self.world.is_alive(e.id)

    def test_add_and_has_trait(self):
        e = self.world.create_entity()
        pos = Position(1, 2)
        assert not self.world.has_trait(e.id, Position)
        self.world.add_trait(e.id, pos)
        assert self.world.has_trait(e.id, Position)

    def test_remove_trait(self):
        e = self.world.create_entity(Position(5, 5))
        self.world.remove_trait(e.id, Position)
        assert not self.world.has_trait(e.id, Position)

    def test_get_entities_from_traits(self):
        e1 = self.world.create_entity(Position(1, 1))
        e2 = self.world.create_entity(Position(2, 2), Velocity(0.1, 0.2))
        found = {
            int(ent.id) for ent, _ in self.world.get_entities_from_traits(Position)
        }
        assert {e1.id, e2.id} <= found

    # ---------------------------------------------------------------------------
    # Query return types
    # ---------------------------------------------------------------------------
    def test_get_entities_returns_instances_for_single_trait_query(self):
        self.world.create_entity(Position(0, 0))
        self.world.create_entity(Position(1, 1))
        self.world.create_entity(Position(2, 2))
        result = list(self.world.get_entities(Position))
        for answer in result:
            assert isinstance(answer, tuple)
            assert isinstance(answer[0], Entity)
            assert isinstance(answer[1], Position)

    def test_get_entities_returns_tuple_of_instances_for_mulit_trait_query(self):
        self.world.create_entity(Position(0, 0), Velocity(1, 1))
        self.world.create_entity(Position(1, 1), Velocity(1, 1))
        self.world.create_entity(Position(2, 2), Velocity(1, 1))
        result = list(self.world.get_entities(Position, Velocity))
        for answer in result:
            assert isinstance(answer, tuple)
            assert isinstance(answer[0], Entity)
            assert isinstance(answer[1], tuple)
            assert len(answer[1]) == 2
            assert isinstance(answer[1][0], Position)
            assert isinstance(answer[1][1], Velocity)

    def test_get_traits_returns_instances_for_single_trait_query(self):
        self.world.create_entity(Position(0, 0))
        self.world.create_entity(Position(1, 1))
        self.world.create_entity(Position(2, 2))
        result = list(self.world.get_traits(Position))
        for answer in result:
            assert isinstance(answer, Position)

    def test_get_traits_returns_tuple_of_instances_for_mulit_trait_query(self):
        self.world.create_entity(Position(0, 0), Velocity(1, 1))
        self.world.create_entity(Position(1, 1), Velocity(1, 1))
        self.world.create_entity(Position(2, 2), Velocity(1, 1))
        result = list(self.world.get_traits(Position, Velocity))
        for answer in result:
            assert isinstance(answer, tuple)
            assert isinstance(answer[0], Position)
            assert isinstance(answer[1], Velocity)

    # ---------------------------------------------------------------------------
    # Entity reusage
    # ---------------------------------------------------------------------------
    def test_create_and_delete_multiple_entities(self):
        e1 = self.world.create_entity(Position(0, 0))
        self.world.create_entity(Position(0, 0))
        self.world.create_entity(Position(0, 0))

        self.world.delete_entity(e1.id)

        e4 = self.world.create_entity(Position(1, 1))

        assert e1.id == e4.id  # Entity ids should be reused

    # ---------------------------------------------------------------------------
    # Trait identity & consistency
    # ---------------------------------------------------------------------------
    def test_trait_identity_after_creation(self):
        pos = Position(10, 20)
        ent = self.world.create_entity(pos)

        results = list(self.world.get_entities_from_traits(Position))
        assert len(results) == 1

        retrieved_ent, (retrieved_pos,) = results[0]
        assert int(retrieved_ent) == ent.id
        # Ensure the stored trait equals the original (dataclass equality)
        # (right now we ignore it since it is not suppored by NumpyWorlds)
        # assert retrieved_pos == pos
        # If the implementation stores the same instance, identity will match too.
        # We assert equality primarily; if identity is required, implementations can be stricter.
        assert retrieved_pos.x == 10 and retrieved_pos.y == 20

    def test_multiple_traits_attached_and_retrieved(self):
        pos = Position(1, 2)
        vel = Velocity(3.0, 4.0)
        ent = self.world.create_entity(pos, vel)

        results = list(self.world.get_entities_from_traits(Position, Velocity))
        assert len(results) == 1
        r_ent, (rpos, rvel) = results[0]
        assert int(r_ent) == ent.id
        # assert rpos == pos
        # assert rvel == vel
        assert rpos.x == pos.x and rpos.y == pos.y
        assert rvel.dx == vel.dx and rvel.dy == vel.dy

    def test_removed_trait_no_longer_retrievable(self):
        pos = Position(9, 9)
        vel = Velocity(5.0, 5.0)
        ent = self.world.create_entity(pos, vel)

        # Remove one trait
        self.world.remove_trait(ent.id, Position)

        # No entity should match both Position and Velocity
        assert list(self.world.get_entities_from_traits(Position, Velocity)) == []

        # Position queries return nothing
        assert list(self.world.get_entities_from_traits(Position)) == []

        # Velocity queries still return the entity
        vel_results = list(self.world.get_entities_from_traits(Velocity))
        assert len(vel_results) == 1
        r_ent, (rvel,) = vel_results[0]
        assert r_ent.id == ent.id
        # assert rvel == vel
        assert rvel.dx == vel.dx and rvel.dy == vel.dy

    def test_trait_identity_with_tags_filter(self):
        pos_a = Position(1, 1)
        pos_b = Position(2, 2)
        e1 = self.world.create_entity(pos_a)
        e1.add_tags("player")
        e2 = self.world.create_entity(pos_b)

        results = list(self.world.get_entities(Position, tags={"player"}))
        assert len(results) == 1
        (entity, retrieved_pos) = results[0]
        assert entity.id == e1.id
        # assert retrieved_pos == pos_a
        assert retrieved_pos.x == pos_a.x and retrieved_pos.y == pos_a.y

    def test_trait_types_are_correct(self):
        pos = Position(1, 2)
        vel = Velocity(3.0, 4.0)
        ent = self.world.create_entity(pos, vel)

        results = list(self.world.get_entities_from_traits(Position, Velocity))
        assert len(results) == 1
        r_ent, (rpos, rvel) = results[0]
        assert int(r_ent) == ent.id

        assert isinstance(rpos.x, int) and isinstance(rpos.y, int)
        assert isinstance(rvel.dx, float) and isinstance(rvel.dy, float)

    def test_trait_are_views(self):
        ent = self.world.create_entity(Position(0, 1), Velocity(2, 3))

        results = list(self.world.get_entities_from_traits(Position, Velocity))
        assert len(results) == 1
        r_ent, (rpos, rvel) = results[0]
        assert int(r_ent) == ent.id

        rpos.x += 1
        rpos.y = 3

        rvel.dx = 0.0
        rvel.dy += 1.0

        rpos, rvel = next(self.world.get_traits(Position, Velocity))

        assert rpos.x == 1
        assert rpos.y == 3
        assert rvel.dx == 0.0
        assert rvel.dy == 4.0

    # ---------------------------------------------------------------------------
    # Tag behavior
    # ---------------------------------------------------------------------------
    def test_add_remove_tags_and_has_tags(self):
        e = self.world.create_entity()
        self.world.add_tags(e.id, "alpha", "beta")
        assert self.world.has_tags(e.id, "alpha")
        assert self.world.has_tags(e.id, "beta")
        assert self.world.has_tags(e.id, "alpha", "beta")
        self.world.remove_tags(e.id, "beta")
        assert self.world.has_tags(e.id, "alpha")
        assert not self.world.has_tags(e.id, "beta")
        assert not self.world.has_tags(e.id, "alpha", "beta")
        assert self.world.has_tags(e.id, "alpha")

    @pytest.mark.parametrize(
        "tags, expected",
        [
            (
                {
                    "player",
                },
                [(Position(0, 1), Velocity(-1.0, -2.0))],
            ),
            (
                {
                    "npc",
                },
                [
                    (Position(2, 3), Velocity(-3.0, -4.0)),
                    (Position(4, 5), Velocity(-5.0, -6.0)),
                ],
            ),
            (
                {
                    "boss",
                },
                [(Position(4, 5), Velocity(-5, -6))],
            ),
            ({"npc", "boss"}, [(Position(4, 5), Velocity(-5, -6))]),
        ],
    )
    def test_complex_world_tags(self, tags, expected):
        e1 = self.world.create_entity(Position(0, 1), Velocity(-1.0, -2.0))
        e2 = self.world.create_entity(Position(2, 3), Velocity(-3.0, -4.0))
        e3 = self.world.create_entity(Position(4, 5), Velocity(-5.0, -6.0))
        e1.add_tags("player")
        e2.add_tags("npc")
        e3.add_tags("npc", "boss")

        results = list(self.world.get_entities(Position, Velocity, tags=tags))

        assert len(results) == len(expected)

        for (entity, (pos, vel)), (exp_pos, exp_vel) in zip(results, expected):
            assert entity.has_tags(*tags)
            assert pos.x == pytest.approx(exp_pos.x)
            assert pos.y == pytest.approx(exp_pos.y)
            assert vel.dx == pytest.approx(exp_vel.dx)
            assert vel.dy == pytest.approx(exp_vel.dy)

    # ---------------------------------------------------------------------------
    # Test exceptions
    # ---------------------------------------------------------------------------
    def test_remove_nonexistent_trait_raises(self):
        e = self.world.create_entity()
        with pytest.raises(TraitNotFoundError):
            self.world.remove_trait(e.id, Position)

    def test_remove_trait_with_non_trait_calss_raises(self):
        e = self.world.create_entity(Position(0, 0))
        with pytest.raises(TypeError):
            self.world.remove_trait(e.id, "INVALID")

    def test_remove_trait_with_nonexistent_entity_raises(self):
        with pytest.raises(EntityNotFoundError):
            self.world.remove_trait(9999, Position)  # assuming 9999 does not

    def test_add_trait_to_nonexistent_entity_raises(self):
        with pytest.raises(EntityNotFoundError):
            self.world.add_trait(9999, Position(0, 0))  # assuming 9999 does not exist

    def test_delete_nonexistent_entity_raises(self):
        with pytest.raises(EntityNotFoundError):
            self.world.delete_entity(8888)  # assuming 8888 does not exist

    def test_adding_trait_twice_raises(self):
        e = self.world.create_entity(Position(0, 0))
        with pytest.raises(ValueError):
            self.world.add_trait(e.id, Position(1, 1))

    def test_add_trait_raises_type_error_for_invalid_trait(self):
        e = self.world.create_entity()

        class InvalidTrait:
            pass

        with pytest.raises(TypeError):
            self.world.add_trait(e.id, InvalidTrait())  # invalid trait

    def test_has_trait_nonexsistent_entity_raises(self):
        with pytest.raises(EntityNotFoundError):
            self.world.has_trait(9999, Position)

    def test_add_tag_to_nonexistent_entity_raises(self):
        with pytest.raises(EntityNotFoundError):
            self.world.add_tags(9999, "some tag")  # assuming 9999 does not exist

    def test_add_tag_with_non_string_raises(self):
        e = self.world.create_entity()
        with pytest.raises(TypeError):
            self.world.add_tags(e.id, 1)

    def test_hast_trait_non_trait_type_raises(self):
        e = self.world.create_entity()

        with pytest.raises(TypeError):
            self.world.has_trait(e.id, "BLUB")

    def test_hast_tags_with_non_strings_raises(self):
        e = self.world.create_entity()
        with pytest.raises(TypeError):
            self.world.has_tags(e.id, 1)

    def test_hast_tags_non_existing_entity_raises(self):
        with pytest.raises(EntityNotFoundError):
            self.world.has_tags(999, "tag")

    def test_remove_tags_non_existing_entity_raises(self):
        with pytest.raises(EntityNotFoundError):
            self.world.remove_tags(99999, "tag")

    def test_remove_tags_with_non_string_raises(self):
        with pytest.raises(TypeError):
            self.world.remove_tags(9999, 1)

    def test_get_entities_wit_non_trait_queries(self):
        class Invalid:
            pass

        with pytest.raises(TypeError):
            list(self.world.get_entities(Position, Invalid))

    def test_get_entities_with_wrong_tag_argument_raises(self):
        self.world.create_entity(Position(0, 0))
        with pytest.raises(TypeError):
            list(
                self.world.get_entities(
                    Position,
                    tags={
                        1,
                    },
                )
            )

    def test_empose_order_contract(self):
        """Contract test for optional empose_order API.

        If a World implementation does not provide empose_order it may raise
        NotImplementedError or AttributeError; in that case this test is skipped.

        Otherwise we assert that calling empose_order on a simple archetype
        reorders entities as expected.
        """
        # Create two entities with the same archetype
        e1 = self.world.create_entity(Position(0, 0), Velocity(0.0, 0.0))
        e2 = self.world.create_entity(Position(1, 1), Velocity(1.0, 1.0))

        # Try calling - allow NotImplementedError to indicate optional API
        try:
            # Attempt to reverse order; some worlds may expect trait types too.
            self.world.empose_order([1, 0], Position, Velocity)
        except NotImplementedError:
            pytest.skip("empose_order not implemented by this World")

        # Verify ordering via get_entities (this is part of the World contract)
        ordered = list(self.world.get_entities(Position, Velocity))

        # We expect two results and that their entity ids are in reversed order
        assert len(ordered) == 2
        first_ent, (first_pos, first_vel) = ordered[0]
        second_ent, (second_pos, second_vel) = ordered[1]

        # Check entity ids
        assert int(first_ent) == e2.id
        assert int(second_ent) == e1.id

        # And also verify the trait data moved with the entities
        assert first_pos.x == pytest.approx(1)
        assert first_pos.y == pytest.approx(1)
        assert first_vel.dx == pytest.approx(1.0)
        assert first_vel.dy == pytest.approx(1.0)

        assert second_pos.x == pytest.approx(0)
        assert second_pos.y == pytest.approx(0)
        assert second_vel.dx == pytest.approx(0.0)
        assert second_vel.dy == pytest.approx(0.0)

    def test_empose_order_requires_trait_types(self):
        """When trait types are provided they must all be trait classes.

        If empose_order is not implemented this test is skipped.
        """
        # create a pair so archetype exists
        self.world.create_entity(Position(0, 0), Velocity(0.0, 0.0))
        self.world.create_entity(Position(1, 1), Velocity(1.0, 1.0))

        try:
            # Use a normal class (not decorated with @trait) to trigger the ValueError
            class NotATrait:
                pass

            with pytest.raises(ValueError):
                self.world.empose_order([0, 1], NotATrait)
        except NotImplementedError:
            pytest.skip("empose_order not implemented by this World")

    def test_empose_order_invalid_order_length_raises(self):
        """Order length must match the number of entities in the archetype.

        If empose_order is not implemented this test is skipped.
        """
        e1 = self.world.create_entity(Position(0, 0), Velocity(0.0, 0.0))
        e2 = self.world.create_entity(Position(1, 1), Velocity(1.0, 1.0))

        try:
            # Provide an order of wrong length (only one entry for two entities)
            with pytest.raises(RuntimeError):
                self.world.empose_order([0], Position, Velocity)
        except NotImplementedError:
            pytest.skip("empose_order not implemented by this World")

    def test_empose_order_no_matching_entities_is_noop(self):
        """Calling empose_order for trait types with no matching archetype should be a no-op.

        The call should not raise; queries for those trait types should remain empty.
        """
        # Create entities that do NOT include the 'Velocity' trait
        self.world.create_entity(Position(0, 0))
        self.world.create_entity(Position(1, 1))

        try:
            # There is no archetype composed of only Velocity in this world; calling
            # empose_order should not raise and should leave queries unchanged.
            self.world.empose_order([], Velocity)
        except NotImplementedError:
            pytest.skip("empose_order not implemented by this World")

        # Ensure there are still no entities matching Velocity
        results = list(self.world.get_entities_from_traits(Velocity))
        assert results == []
