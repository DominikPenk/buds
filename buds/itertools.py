"""Iterator and functional utilities for ECS queries and systems.

This module provides lightweight iterator combinators and functional helpers
used throughout the ECS framework. It complements Python’s built-in
`itertools` by adding ECS-specific patterns for working with entities,
traits, and system results.

These tools enable functional composition of ECS logic while maintaining
clarity and iteration efficiency.

Exports:
    entity_product
    entity_permutations
    entity_combinations
    entity_combinations_with_replacement
    entity_filter
    entity_groupby
    entity_starmap
    trait_product
    trait_permutations
    trait_combinations
    trait_combinations_with_replacement
    trait_filter
    trait_groupby
    trait_starmap
"""

import itertools
from collections.abc import Iterator
from typing import Any, Callable, Optional, TypeVar, overload

from .base import Entity, World

__all__ = [
    "Query",
    "entity_product",
    "entity_permutations",
    "entity_combinations",
    "entity_combinations_with_replacement",
    "entity_filter",
    "entity_groupby",
    "entity_starmap",
    "trait_product",
    "trait_permutations",
    "trait_combinations",
    "trait_combinations_with_replacement",
    "trait_filter",
    "trait_groupby",
    "trait_starmap",
]

Trait = TypeVar("Trait")
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_T3 = TypeVar("_T3")
_T4 = TypeVar("_T4")
_T5 = TypeVar("_T5")
R = TypeVar("R")


class Query:
    """Defines a query for entities based on a set of traits and optional tags.

    This class serves as a declarative way to specify which entities and
    traits an iterator function should operate on, allowing for
    multi-query operations in functions like
    [`entity_product`][buds.itertools.entity_product].

    Args:
        *traits: One or more trait types that entities must possess.
        tags: An optional set of tags (strings) that entities must possess.
    """

    def __init__(self, *traits: type[Trait], tags: Optional[set[str]] = None):
        self.traits = traits
        self.tags = tags


def _resolve_queries(
    world: World,
    *queries: type[Trait] | Query,
    tags: Optional[set[str]] = None,
    traits_only: bool = False,
) -> tuple[Iterator[tuple[Entity, tuple[Trait, ...]]]]:
    first_query_type = type(queries[0])
    assert first_query_type in (Query, type, dict), (
        "Queries must be Query, dict, or trait types"
    )
    assert all(type(q) is first_query_type for q in queries), (
        "All queries must be of the same type"
    )
    if first_query_type is Query:
        assert tags is None, "Tags cannot be provided when using Query objects"
    elif first_query_type is dict:
        queries = [
            Query(
                *((q["traits"],) if isinstance(q["traits"], type) else q["traits"]),
                tags=q.get("tags", None),
            )
            for q in queries
        ]
    else:
        queries = [Query(*queries, tags=tags)]  # type: ignore[misc]

    if traits_only:
        return tuple(
            world.get_traits(*query.traits, tags=query.tags) for query in queries
        )

    return tuple(
        world.get_entities(*query.traits, tags=query.tags) for query in queries
    )


@overload
def entity_product(
    world: World,
    trait: type[Trait],
    tags: Optional[set[str]] = None,
    repeat: int = 1,
) -> Iterator[tuple[tuple[Entity, ...], tuple[Trait, ...]]]: ...


@overload
def entity_product(
    world: World,
    trait_1: type[_T1],
    trait_2: type[_T2],
    tags: Optional[set[str]] = None,
    repeat: int = 1,
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[_T1, _T2], ...]]]: ...


@overload
def entity_product(
    world: World,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    tags: Optional[set[str]] = None,
    repeat: int = 1,
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[_T1, _T2, _T3], ...]]]: ...


@overload
def entity_product(
    world: World,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    tags: Optional[set[str]] = None,
    repeat: int = 1,
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[_T1, _T2, _T3, _T4], ...]]]: ...


@overload
def entity_product(
    world: World,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    trait_5: type[_T5],
    tags: Optional[set[str]] = None,
    repeat: int = 1,
) -> Iterator[
    tuple[tuple[Entity, ...], tuple[tuple[_T1, _T2, _T3, _T4, _T5], ...]]
]: ...


def entity_product(
    world: World,
    *queries: type[Trait] | Query,
    tags: Optional[set[str]] = None,
    repeat: int = 1,
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[Trait, ...] | Trait, ...]]]:
    """Generates the Cartesian product of entities matching given criteria.

    This function yields the product of results from one or more queries,
    similar to [`itertools.product`](https://docs.python.org/3/library/itertools.html#itertools.product).

    Queries can be provided as:
    1. A single set of `traits` and an optional `tags` argument (traditional query).
    2. Multiple `Query` objects, each defining a separate set of traits and tags.

    Args:
        world: The ECS world ([`buds.base.World`][buds.base.World]) to query.
        *queries: One or more trait types or [`Query`][buds.itertools.Query] objects.
            If trait types are provided, they apply to a single query.
            If [`Query`][buds.itertools.Query] objects are provided, each object represents a
            separate input sequence for the product.
        tags: Optional set of tags to filter entities. Only used if `*queries`
            contains only trait types (not [`Query`][buds.itertools.Query] objects).
        repeat: Number of repetitions of the input sequences.

    Yields:
        Tuples containing:
        - A tuple of [`Entity`][buds.base.Entity] instances.
        - A tuple of corresponding trait tuples.

    Raises:
        ValueError: If a mix of trait types and [`Query`][buds.itertools.Query] objects is used.
        TypeError: If `tags` is provided when using one or more [`Query`][buds.itertools.Query] objects.
    """
    query_results = _resolve_queries(world, *queries, tags=tags)
    for prod_entry in itertools.product(*query_results, repeat=repeat):
        yield zip(*prod_entry)


@overload
def trait_product(
    world: World,
    trait: type[Trait],
    tags: Optional[set[str]] = None,
    repeat: int = 1,
) -> Iterator[tuple[Trait, ...]]: ...


@overload
def trait_product(
    world: World,
    trait_1: type[_T1],
    trait_2: type[_T2],
    tags: Optional[set[str]] = None,
    repeat: int = 1,
) -> Iterator[tuple[tuple[_T1, _T2], ...]]: ...


@overload
def trait_product(
    world: World,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    tags: Optional[set[str]] = None,
    repeat: int = 1,
) -> Iterator[tuple[tuple[_T1, _T2, _T3], ...]]: ...


@overload
def trait_product(
    world: World,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    tags: Optional[set[str]] = None,
    repeat: int = 1,
) -> Iterator[tuple[tuple[_T1, _T2, _T3, _T4], ...]]: ...


@overload
def trait_product(
    world: World,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    trait_5: type[_T5],
    tags: Optional[set[str]] = None,
    repeat: int = 1,
) -> Iterator[tuple[tuple[_T1, _T2, _T3, _T4, _T5], ...]]: ...


def trait_product(
    world: World,
    *queries: type[Trait] | Query,
    tags: Optional[set[str]] = None,
    repeat: int = 1,
) -> Iterator[tuple[tuple[Trait, ...], ...]]:
    """Generates the Cartesian product of traits from entities matching given criteria.

    This function is similar to [`entity_product`][buds.itertools.entity_product] but only
    yields the traits, excluding the entity reference.

    Args:
        world: The ECS world ([`buds.base.World`][buds.base.World]) to query.
        *queries: One or more trait types or [`Query`][buds.itertools.Query] objects.
            See [`entity_product`][buds.itertools.entity_product] for details on query types.
        tags: Optional set of tags to filter entities. Only used if `*queries`
            contains only trait types.
        repeat: Number of repetitions of the input sequences.

    Yields:
        Tuples of trait combinations drawn from the Cartesian product.

    Raises:
        ValueError: If a mix of trait types and [`Query`][buds.itertools.Query] objects is used.
        TypeError: If `tags` is provided when using one or more [`Query`][buds.itertools.Query] objects.
    """
    query_results = _resolve_queries(world, *queries, tags=tags, traits_only=True)
    yield from itertools.product(*query_results, repeat=repeat)


@overload
def entity_permutations(
    world: World, r: int, trait: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[tuple[Entity, ...], tuple[Trait, ...]]]: ...


@overload
def entity_permutations(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[_T1, _T2], ...]]]: ...


@overload
def entity_permutations(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[_T1, _T2, _T3], ...]]]: ...


@overload
def entity_permutations(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[_T1, _T2, _T3, _T4], ...]]]: ...


@overload
def entity_permutations(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    trait_5: type[_T5],
    tags: Optional[set[str]] = None,
) -> Iterator[
    tuple[tuple[Entity, ...], tuple[tuple[_T1, _T2, _T3, _T4, _T5], ...]]
]: ...


def entity_permutations(
    world: World, r: int, *traits: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[Trait, ...] | Trait, ...]]]:
    """Generates r-length permutations of unique entities matching given traits and tags.

    Args:
        world: The ECS world ([`buds.base.World`][buds.base.World]) to query.
        r: The length of each permutation (must be less than the number of matching entities).
        *traits: Trait types to filter entities by.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples containing:
        - A tuple of permuted [`Entity`][buds.base.Entity] instances.
        - A tuple of corresponding trait tuples.

    Raises:
        TypeError: Inherited from [`World.get_entities`][buds.base.World.get_entities]
    """
    for perm in itertools.permutations(world.get_entities(*traits, tags=tags), r=r):
        yield zip(*perm)


@overload
def trait_permutations(
    world: World, r: int, trait: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[Trait, ...]]: ...


@overload
def trait_permutations(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[_T1, _T2], ...]]: ...


@overload
def trait_permutations(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[_T1, _T2, _T3], ...]]: ...


@overload
def trait_permutations(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[_T1, _T2, _T3, _T4], ...]]: ...


@overload
def trait_permutations(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    trait_5: type[_T5],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[_T1, _T2, _T3, _T4, _T5], ...]]: ...


def trait_permutations(
    world: World, r: int, *traits: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[tuple[Trait, ...], ...]]:
    """Generates r-length permutations of trait tuples from matching entities.

    This function is similar to [`entity_permutations`][buds.itertools.entity_permutations] but
    only yields the traits, excluding the entity reference.

    Args:
        world: The ECS world ([`buds.base.World`][buds.base.World]) to query.
        r: The length of each permutation (must be less the number of matching entities).
        *traits: Trait types to include.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples of trait permutations.

    Raises:
        TypeError: Inherited from [`World.get_entities`][buds.base.World.get_entities]
    """
    yield from itertools.permutations(world.get_traits(*traits, tags=tags), r=r)


@overload
def entity_combinations(
    world: World, r: int, trait: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[tuple[Entity, ...], tuple[Trait, ...]]]: ...


@overload
def entity_combinations(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[_T1, _T2], ...]]]: ...


@overload
def entity_combinations(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[_T1, _T2, _T3], ...]]]: ...


@overload
def entity_combinations(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[_T1, _T2, _T3, _T4], ...]]]: ...


@overload
def entity_combinations(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    trait_5: type[_T5],
    tags: Optional[set[str]] = None,
) -> Iterator[
    tuple[tuple[Entity, ...], tuple[tuple[_T1, _T2, _T3, _T4, _T5], ...]]
]: ...


def entity_combinations(
    world: World, r: int, *traits: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[Trait, ...] | Trait, ...]]]:
    """Generates r-length combinations of unique entities matching given traits and tags.

    Args:
        world: The ECS world ([`buds.base.World`][buds.base.World]) to query.
        r: The length of each combination.
        *traits: Trait types to filter entities by.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples containing:
        - A tuple of combined [`Entity`][buds.base.Entity] instances.
        - A tuple of corresponding trait tuples.

    Raises:
        TypeError: Inherited from [`World.get_entities`][buds.base.World.get_entities]
    """
    for combination in itertools.combinations(
        world.get_entities(*traits, tags=tags), r
    ):
        yield zip(*combination)


@overload
def trait_combinations(
    world: World, r: int, trait: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[Trait, ...]]: ...


@overload
def trait_combinations(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[_T1, _T2], ...]]: ...


@overload
def trait_combinations(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[_T1, _T2, _T3], ...]]: ...


@overload
def trait_combinations(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[_T1, _T2, _T3, _T4], ...]]: ...


@overload
def trait_combinations(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    trait_5: type[_T5],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[_T1, _T2, _T3, _T4, _T5], ...]]: ...


def trait_combinations(
    world: World, r: int, *traits: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[tuple[Trait, ...] | Trait, ...]]:
    """Generates r-length combinations of trait tuples from matching entities.

    This function is similar to [`entity_combinations`][buds.itertools.entity_combinations] but
    only yields the traits, excluding the entity reference.

    Args:
        world: The ECS world ([`buds.base.World`][buds.base.World]) to query.
        r: The length of each combination.
        *traits: Trait types to include.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples of trait combinations.

    Raises:
        TypeError: Inherited from [`World.get_entities`][buds.base.World.get_entities]
    """
    yield from itertools.combinations(world.get_traits(*traits, tags=tags), r)


@overload
def entity_combinations_with_replacement(
    world: World, r: int, trait: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[tuple[Entity, ...], tuple[Trait, ...]]]: ...


@overload
def entity_combinations_with_replacement(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[_T1, _T2], ...]]]: ...


@overload
def entity_combinations_with_replacement(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[_T1, _T2, _T3], ...]]]: ...


@overload
def entity_combinations_with_replacement(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[_T1, _T2, _T3, _T4], ...]]]: ...


@overload
def entity_combinations_with_replacement(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    trait_5: type[_T5],
    tags: Optional[set[str]] = None,
) -> Iterator[
    tuple[tuple[Entity, ...], tuple[tuple[_T1, _T2, _T3, _T4, _T5], ...]]
]: ...


def entity_combinations_with_replacement(
    world: World, r: int, *traits: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[Trait, ...], ...]]]:
    """Generates r-length combinations of entities with replacement.

    Entities matching the criteria can appear multiple times in the resulting
    combinations.

    Args:
        world: The ECS world ([`buds.base.World`][buds.base.World]) to query.
        r: The length of each combination.
        *traits: Trait types to filter entities by.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples containing:
        - A tuple of [`Entity`][buds.base.Entity] instances.
        - A tuple of corresponding trait tuples.

    Raises:
        TypeError: Inherited from [`World.get_entities`][buds.base.World.get_entities]
    """
    for comb in itertools.combinations_with_replacement(
        world.get_entities(*traits, tags=tags), r=r
    ):
        yield zip(*comb)


@overload
def trait_combinations_with_replacement(
    world: World, r: int, trait: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[Trait, ...]]: ...


@overload
def trait_combinations_with_replacement(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[_T1, _T2], ...]]: ...


@overload
def trait_combinations_with_replacement(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[_T1, _T2, _T3], ...]]: ...


@overload
def trait_combinations_with_replacement(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[_T1, _T2, _T3, _T4], ...]]: ...


@overload
def trait_combinations_with_replacement(
    world: World,
    r: int,
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    trait_5: type[_T5],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[tuple[_T1, _T2, _T3, _T4, _T5], ...]]: ...


def trait_combinations_with_replacement(
    world: World, r: int, *traits: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[tuple[Trait, ...] | Trait, ...]]:
    """Generates r-length combinations of trait tuples with replacement.

    This function is similar to
    [`entity_combinations_with_replacement`][buds.itertools.entity_combinations_with_replacement]
    but only yields the traits, excluding the entity reference.

    Args:
        world: The ECS world ([`buds.base.World`][buds.base.World]) to query.
        r: The length of each combination.
        *traits: Trait types to include.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples of trait combinations with replacement.

    Raises:
        TypeError: Inherited from [`World.get_entities`][buds.base.World.get_entities]
    """
    yield from itertools.combinations_with_replacement(
        world.get_traits(*traits, tags=tags), r=r
    )


@overload
def entity_filter(
    world: World,
    predicate: Callable[[Entity, tuple[Trait, ...]], bool],
    trait: type[Trait],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Entity, Trait]]: ...


@overload
def entity_filter(
    world: World,
    predicate: Callable[[Entity, tuple[Trait, ...]], bool],
    trait_1: type[_T1],
    trait_2: type[_T2],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Entity, tuple[_T1, _T2]]]: ...


@overload
def entity_filter(
    world: World,
    predicate: Callable[[Entity, tuple[Trait, ...]], bool],
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Entity, tuple[_T1, _T2, _T3]]]: ...


@overload
def entity_filter(
    world: World,
    predicate: Callable[[Entity, tuple[Trait, ...]], bool],
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Entity, tuple[_T1, _T2, _T3, _T4]]]: ...


@overload
def entity_filter(
    world: World,
    predicate: Callable[[Entity, tuple[Trait, ...]], bool],
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    trait_5: type[_T5],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Entity, tuple[_T1, _T2, _T3, _T4, _T5]]]: ...


def entity_filter(
    world: World,
    predicate: Callable[[Entity, tuple[Trait, ...]], bool],
    *traits: type[Trait],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Entity, tuple[Trait, ...]]]:
    """Filters entities and their traits based on a predicate function.

    This is the ECS equivalent of the built-in `filter` function.

    Args:
        world: The ECS world ([`buds.base.World`][buds.base.World]) to query.
        predicate: A function that takes an entity and its traits, returning
            `True` if the pair should be included in the results.
        *traits: Trait types to filter entities by.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples of entities and their traits that satisfy the predicate.

    Raises:
        TypeError: Inherited from [`World.get_entities`][buds.base.World.get_entities]
    """
    for e, traits in world.get_entities(*traits, tags=tags):
        if predicate(e, traits):
            yield e, traits


@overload
def trait_filter(
    world: World,
    predicate: Callable[[tuple[Trait, ...]], bool],
    trait: type[Trait],
    tags: Optional[set[str]] = None,
) -> Iterator[Trait]: ...


@overload
def trait_filter(
    world: World,
    predicate: Callable[[tuple[Trait, ...]], bool],
    trait_1: type[_T1],
    trait_2: type[_T2],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[_T1, _T2]]: ...


@overload
def trait_filter(
    world: World,
    predicate: Callable[[tuple[Trait, ...]], bool],
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[_T1, _T2, _T3]]: ...


@overload
def trait_filter(
    world: World,
    predicate: Callable[[tuple[Trait, ...]], bool],
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[_T1, _T2, _T3, _T4]]: ...


@overload
def trait_filter(
    world: World,
    predicate: Callable[[tuple[Trait, ...]], bool],
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    trait_5: type[_T5],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[_T1, _T2, _T3, _T4, _T5]]: ...


def trait_filter(
    world: World,
    predicate: Callable[[tuple[Trait, ...]], bool],
    *traits: type[Trait],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Trait, ...]]:
    """Filters trait tuples based on a predicate function.

    This function is similar to [`entity_filter`][buds.itertools.entity_filter] but
    the predicate operates only on the trait tuple.

    Args:
        world: The ECS world ([`buds.base.World`][buds.base.World]) to query.
        predicate: A function that takes a trait tuple and returns `True` if it should be included.
        *traits: Trait types to include.
        tags: Optional set of tags to filter entities.

    Yields:
        Trait tuples that satisfy the predicate.

    Raises:
        TypeError: Inherited from [`World.get_entities`][buds.base.World.get_entities]
    """
    for traits in world.get_traits(*traits, tags=tags):
        if predicate(traits):
            yield traits


@overload
def entity_groupby(
    world: World,
    key: Callable[[Entity, tuple[Trait, ...]], Any],
    trait: type[Trait],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Any, Iterator[tuple[Entity, Trait]]]]: ...


@overload
def entity_groupby(
    world: World,
    key: Callable[[Entity, tuple[Trait, ...]], Any],
    trait_1: type[_T1],
    trait_2: type[_T2],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Any, Iterator[tuple[Entity, tuple[_T1, _T2]]]]]: ...


@overload
def entity_groupby(
    world: World,
    key: Callable[[Entity, tuple[Trait, ...]], Any],
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Any, Iterator[tuple[Entity, tuple[_T1, _T2, _T3]]]]]: ...


@overload
def entity_groupby(
    world: World,
    key: Callable[[Entity, tuple[Trait, ...]], Any],
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Any, Iterator[tuple[Entity, tuple[_T1, _T2, _T3, _T4]]]]]: ...


@overload
def entity_groupby(
    world: World,
    key: Callable[[Entity, tuple[Trait, ...]], Any],
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    trait_5: type[_T5],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Any, Iterator[tuple[Entity, tuple[_T1, _T2, _T3, _T4, _T5]]]]]: ...


def entity_groupby(
    world: World,
    key: Callable[[Entity, tuple[Trait, ...]], Any],
    *traits: type[Trait],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Any, Iterator[tuple[Entity, tuple[Trait, ...]]]]]:
    """Groups entities and traits using a key function.

    This is the ECS equivalent of [`itertools.groupby`](https://docs.python.org/3/library/itertools.html#itertools.groupby).
    Note that for `itertools.groupby` to work correctly, the input sequence
    must be **sorted** by the grouping key.

    Args:
        world: The ECS world ([`buds.base.World`][buds.base.World]) to query.
        key: A function that takes an entity and its traits, returning a key
            to group entities by.
        *traits: Trait types to filter entities by.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples of (key, group iterator) pairs. The group iterator yields
        (entity, traits) pairs that share the same key.

    Raises:
        TypeError: Inherited from [`World.get_entities`][buds.base.World.get_entities]
    """
    yield from itertools.groupby(
        world.get_entities(*traits, tags=tags), key=lambda pair: key(*pair)
    )


@overload
def trait_groupby(
    world: World,
    key: Callable[[tuple[Trait, ...]], Any],
    trait: type[Trait],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Any, Iterator[Trait]]]: ...


@overload
def trait_groupby(
    world: World,
    key: Callable[[tuple[Trait, ...]], Any],
    trait_1: type[_T1],
    trait_2: type[_T2],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Any, Iterator[tuple[_T1, _T2]]]]: ...


@overload
def trait_groupby(
    world: World,
    key: Callable[[tuple[Trait, ...]], Any],
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Any, Iterator[tuple[_T1, _T2, _T3]]]]: ...


@overload
def trait_groupby(
    world: World,
    key: Callable[[tuple[Trait, ...]], Any],
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Any, Iterator[tuple[_T1, _T2, _T3, _T4]]]]: ...


@overload
def trait_groupby(
    world: World,
    key: Callable[[tuple[Trait, ...]], Any],
    trait_1: type[_T1],
    trait_2: type[_T2],
    trait_3: type[_T3],
    trait_4: type[_T4],
    trait_5: type[_T5],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Any, Iterator[tuple[_T1, _T2, _T3, _T4, _T5]]]]: ...


def trait_groupby(
    world: World,
    key: Callable[[tuple[Trait, ...]], Any],
    *traits: type[Trait],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Any, Iterator[tuple[Trait, ...]]]]:
    """Groups trait tuples using a key function.

    This is similar to [`entity_groupby`][buds.itertools.entity_groupby] but operates
    only on the trait tuples. The input sequence should be **sorted** by the key.

    Args:
        world: The ECS world ([`buds.base.World`][buds.base.World]) to query.
        key: A function that takes a trait tuple, returning a key to group traits by.
        *traits: Trait types to include.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples of (key, group iterator) pairs. The group iterator yields
        trait tuples that share the same key.

    Raises:
        TypeError: Inherited from [`World.get_entities`][buds.base.World.get_entities]
    """

    yield from itertools.groupby(
        world.get_traits(*traits, tags=tags), key=lambda ts: key(ts)
    )


def entity_starmap(
    world: World,
    func: Callable[[Entity, tuple[Trait, ...]], R],
    *traits: type[Trait],
    tags: Optional[set[str]] = None,
) -> Iterator[R]:
    """Applies a function to each entity and its associated traits.

    This is the ECS equivalent of [`itertools.starmap`](https://docs.python.org/3/library/itertools.html#itertools.starmap).
    The function signature must accept the entity followed by its traits as
    separate positional arguments.

    Args:
        world: The ECS world ([`buds.base.World`][buds.base.World]) to query.
        func: A function to apply to each entity and its traits. Its signature
            should be `func(entity: Entity, trait1: Trait, trait2: Trait, ...) -> R`.
        *traits: Trait types to filter entities by.
        tags: Optional set of tags to filter entities.

    Yields:
        The results of applying the function to each entity-trait pair.

    Raises:
        TypeError: Inherited from [`World.get_entities`][buds.base.World.get_entities]
        TypeError: If `func` does not accept the correct number of arguments.
    """
    yield from itertools.starmap(
        lambda e, ts: func(e, *ts), world.get_entities(*traits, tags=tags)
    )


def trait_starmap(
    world: World,
    func: Callable[[tuple[Trait, ...]], R],
    *traits: type[Trait],
    tags: Optional[set[str]] = None,
) -> Iterator[R]:
    """Applies a function to each trait tuple, unpacking the traits as arguments.

    This function is similar to [`entity_starmap`][buds.itertools.entity_starmap] but
    only applies the function to the unpacked traits.

    Args:
        world: The ECS world ([`buds.base.World`][buds.base.World]) to query.
        func: A function to apply to each trait tuple. Its signature
            should be `func(trait1: Trait, trait2: Trait, ...) -> R`.
        *traits: Trait types to include.
        tags: Optional set of tags to filter entities.

    Yields:
        The results of applying the function to each trait tuple.

    Raises:
        TypeError: Inherited from [`World.get_entities`][buds.base.World.get_entities]
        TypeError: If `func` does not accept the correct number of arguments.
    """
    yield from itertools.starmap(func, world.get_traits(*traits, tags=tags))
