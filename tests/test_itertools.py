import pytest
from buds.base import trait, Entity
from buds.itertools import (
    Query,
    _resolve_queries,
    entity_product,
    entity_permutations,
    entity_combinations,
    entity_combinations_with_replacement,
    entity_filter,
    entity_groupby,
    entity_starmap,
    trait_product,
    trait_permutations,
    trait_combinations,
    trait_combinations_with_replacement,
    trait_filter,
    trait_groupby,
    trait_starmap,
)
# ---------------------------------------------------------------------------
# Shared test trait definitions
# ---------------------------------------------------------------------------


@trait
class Position:
    x: int
    y: int


@trait
class Velocity:
    dx: float
    dy: float


# ---------------------------------------------------------------------------
# Local fixture for world setup
# ---------------------------------------------------------------------------


def populate_world(world):
    """
    Provides a fresh world instance pre-populated with a few entities
    containing Position and Velocity traits and various tags.
    """
    e1 = world.create_entity(Position(0, 0))
    e2 = world.create_entity(Position(1, 2), Velocity(0.5, 0.5))
    e3 = world.create_entity(Position(2, 4), Velocity(1.0, 1.5))
    e1.add_tags("player")
    e2.add_tags("npc")
    e3.add_tags("npc", "boss")
    return world


# ---------------------------------------------------------------------------
# Test Resovel queries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query, tags, expected_count, expected_tags",
    [
        # Query dict with tags
        ({"traits": Position, "tags": {"npc"}}, None, 2, {"npc"}),
        ({"traits": Position, "tags": "npc"}, None, 2, {"npc"}),
        ({"traits": (Position, Velocity), "tags": {"boss"}}, None, 1, {"boss"}),
        # Query dict without tags
        ({"traits": Position}, None, 3, None),
        ({"traits": (Position, Velocity)}, None, 2, None),
        # Query object
        (Query(Position, tags={"npc"}), None, 2, {"npc"}),
        (Query(Position, tags="player"), None, 1, {"player"}),
        (Query(Position, Velocity, tags="boss"), None, 1, {"boss"}),
        (Query(Position), None, 3, None),
        # Trait Class
        (Position, None, 3, None),
        ((Position, Velocity), None, 2, None),
        # Trait Class + tags argument
        (Position, {"npc"}, 2, {"npc"}),
        (Position, "player", 1, {"player"}),
        ((Position, Velocity), "boss", 1, {"boss"}),
    ],
)
def test_resolve_query_argument_types_and_filtering(
    world, query, tags, expected_count, expected_tags
):
    """Tests various input formats for queries and the optional tags argument,
    and checks if the correct number of entities/traits are returned."""
    populated_world = populate_world(world)
    if isinstance(query, tuple):
        result = list(_resolve_queries(populated_world, *query, tags=tags))
    else:
        result = list(_resolve_queries(populated_world, query, tags=tags))

    # The result structure for _resolve_queries is a list of lists of (entity, trait_tuple)
    # We expect exactly one inner list for a single query.
    assert len(result) == 1
    entities_traits = list(result[0])

    assert len(entities_traits) == expected_count, (
        f"Expected {expected_count} results for query {query} with tags {tags}, got {len(entities_traits)}"
    )

    for e, t_tuple in entities_traits:
        assert isinstance(e, Entity)
        # Check that all queried traits are present and of the correct type
        if isinstance(query, dict):
            traits_to_check = query.get("traits")
        elif isinstance(query, Query):
            traits_to_check = query.traits
        else:  # Trait class or tuple
            traits_to_check = query

        if not isinstance(traits_to_check, tuple):
            traits_to_check = (traits_to_check,)

        assert len(t_tuple) == len(traits_to_check)
        for t, expected_t_cls in zip(t_tuple, traits_to_check):
            assert isinstance(t, expected_t_cls)

        # Check tags if expected
        if expected_tags:
            # expected_tags can be a string or a set/list
            if isinstance(expected_tags, set):
                assert all(e.has_tags(tag) for tag in expected_tags)
            else:
                assert e.has_tags(expected_tags)


def test_resolve_queries_single_query_are_equivalent(world):
    populated_world = populate_world(world)
    query_obj = Query(Position, tags="npc")
    query_cls = Position

    results_1 = list(_resolve_queries(populated_world, query_obj))
    results_2 = list(_resolve_queries(populated_world, query_cls, tags="npc"))

    assert len(results_1) == len(results_2) == 1

    # Check inner content for element-wise equivalence
    e_t_1 = list(results_1[0])
    e_t_2 = list(results_2[0])

    assert len(e_t_1) == len(e_t_2) == 2  # Should only be e2 and e3 (the npcs)

    for (e1, (t1,)), (e2, (t2,)) in zip(results_1[0], results_2[0]):
        assert e1.id == e2.id
        assert e1.has_tags("npc") and e2.has_tags("npc")
        assert t1.x == t2.x and t1.y == t2.y


# ---------------------------------------------------------------------------
# Entity-based utility tests
# ---------------------------------------------------------------------------
def test_entity_product_returns_cartesian_pairs(world):
    populated_world = populate_world(world)

    player_query = Query(Position, tags="player")
    npc_query = Query(Position, tags="npc")

    results = list(entity_product(populated_world, player_query, npc_query))

    # 1 player * 2 npcs = 2 results
    assert len(results) == 2

    for ents, traits in results:
        assert all(isinstance(e, Entity) for e in ents)
        assert all(isinstance(t, tuple) for t in traits)


def test_entity_product_repeats(world):
    populated_world = populate_world(world)
    results = list(entity_product(populated_world, Position, Velocity, repeat=3))

    assert len(results) == 8  # 2 entities with both traits, 2^3 = 8 combinations

    for ents, ((t1, v1), (t2, v2), (t3, v3)) in results:
        assert all(isinstance(e, Entity) for e in ents)

        assert isinstance(t1, Position)
        assert isinstance(t2, Position)
        assert isinstance(t3, Position)

        assert isinstance(v1, Velocity)
        assert isinstance(v2, Velocity)
        assert isinstance(v3, Velocity)


def test_entity_permutations_returns_expected_size(world):
    populated_world = populate_world(world)
    results = list(entity_permutations(populated_world, 2, Position))

    assert len(results) == 6


def test_entity_combinations_returns_unique_pairs(world):
    populated_world = populate_world(world)
    results = list(entity_combinations(populated_world, 2, Position))

    assert len(results) == 3

    pairs = [tuple(e.id for e in ents) for ents, _ in results]
    assert len(pairs) == len(set(pairs))


def test_entity_combinations_with_replacement_allows_duplicates(world):
    populated_world = populate_world(world)
    results = list(entity_combinations_with_replacement(populated_world, 2, Position))
    assert len(results) == 6

    found_self_pairs = any(e1.id == e2.id for (e1, e2), _ in results)
    assert found_self_pairs, "Should include self-pairs"


def test_entity_filter_selects_only_matching(world):
    populated_world = populate_world(world)
    filtered = list(
        entity_filter(populated_world, lambda e, t: e.has_tags("boss"), Position)
    )
    assert all(e.has_tags("boss") for e, _ in filtered)


def test_entity_groupby_groups_by_tag(world):
    populated_world = populate_world(world)
    grouped = list(
        entity_groupby(populated_world, lambda e, t: e.has_tags("boss"), Position)
    )
    keys = [k for k, _ in grouped]
    assert {True, False}.issuperset(keys)


def test_entity_starmap_applies_function(world):
    populated_world = populate_world(world)

    def func(entity, position: Position, velocity: Velocity):
        assert isinstance(entity, Entity), f"Expected Entity, got {type(entity)}"
        assert isinstance(position, Position), (
            f"Expected Position, got {type(position)}"
        )
        assert isinstance(velocity, Velocity), (
            f"Expected Velocity, got {type(velocity)}"
        )
        return position.x + velocity.dx

    results = list(entity_starmap(populated_world, func, Position, Velocity))
    expected = [1 + 0.5, 2 + 1.0]  # Entities that have both Position and Velocity
    assert results == expected


# ---------------------------------------------------------------------------
# Trait-based utility tests
# ---------------------------------------------------------------------------


def test_trait_product_cartesian(world):
    populated_world = populate_world(world)
    player_query = Query(Position, tags={"player"})
    npc_query = Query(Position, tags={"npc"})
    results = list(trait_product(populated_world, player_query, npc_query))
    assert all(len(p) == 2 for p in results)
    for (t1,), (t2,) in results:
        assert isinstance(t1, Position)
        assert isinstance(t2, Position)


def test_trait_product_repeats(world):
    populated_world = populate_world(world)
    results = list(trait_product(populated_world, Position, Velocity, repeat=3))
    assert all(len(p) == 3 for p in results)
    for (t1, v1), (t2, v2), (t3, v3) in results:
        assert isinstance(t1, Position)
        assert isinstance(t2, Position)
        assert isinstance(t3, Position)

        assert isinstance(v1, Velocity)
        assert isinstance(v2, Velocity)
        assert isinstance(v3, Velocity)


def test_trait_permutations_expected_length(world):
    populated_world = populate_world(world)
    results = list(trait_permutations(populated_world, 2, Position))
    assert all(len(p) == 2 for p in results)


def test_trait_combinations_unique_pairs(world):
    populated_world = populate_world(world)
    results = list(trait_combinations(populated_world, 2, Position))
    assert not any(t1 == t2 for t1, t2 in results)


def test_trait_combinations_with_replacement_self_pairs(world):
    populated_world = populate_world(world)
    results = list(trait_combinations_with_replacement(populated_world, 2, Position))
    assert any(p1 == p2 for p1, p2 in results)


def test_trait_filter_selects_only_matching(world):
    populated_world = populate_world(world)
    filtered = list(trait_filter(populated_world, lambda t: t[0].x == 1, Position))
    assert all(t[0].x == 1 for t in filtered)


def test_trait_groupby_groups_by_evenness(world):
    populated_world = populate_world(world)
    grouped = list(trait_groupby(populated_world, lambda t: t[0].x % 2 == 0, Position))
    keys = [k for k, _ in grouped]
    assert {True, False}.issuperset(keys)


def test_trait_starmap_applies_function(world):
    populated_world = populate_world(world)

    def func(t: Position, v: Velocity):
        assert isinstance(t, Position), f"Expected Position, got {type(t)}"
        assert isinstance(v, Velocity), f"Expected Velocity, got {type(v)}"
        return t.x + v.dx

    results = list(trait_starmap(populated_world, func, Position, Velocity))
    expected = [1 + 0.5, 2 + 1.0]  # Entities that have both Position and Velocity
    assert results == expected
