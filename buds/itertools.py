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
from .base import World, Entity
from typing import TypeVar, Callable, Any, Optional
from collections.abc import Iterator, Iterable

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
R = TypeVar("R")


class Query:
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


def entity_product(
    world: World,
    *queries: type[Trait],
    tags: Optional[set[str]] = None,
    repeat: int = 1,
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[Trait, ...], ...]]]:
    """Generates the Cartesian product of entities matching given traits and tags.

    Args:
        world: The ECS world to query.
        *traits: Trait types to filter entities by.
        tags: Optional set of tags to filter entities.
        repeat: Number of repetitions of the product.

    Yields:
        Tuples containing:
        - A tuple of `Entity` instances.
        - A tuple of corresponding trait tuples.
    """
    query_results = _resolve_queries(world, *queries, tags=tags)
    for prod_entry in itertools.product(*query_results, repeat=repeat):
        yield zip(*prod_entry)


def trait_product(
    world: World,
    *queries: type[Trait],
    tags: Optional[set[str]] = None,
    repeat: int = 1,
) -> Iterator[tuple[tuple[Trait, ...], ...]]:
    """Generates the Cartesian product of traits from entities matching given criteria.

    Args:
        world: The ECS world to query.
        *traits: Trait types to filter entities by.
        tags: Optional set of tags to filter entities.
        repeat: Number of repetitions of the product.

    Yields:
        Tuples of trait combinations drawn from the Cartesian product.
    """
    query_results = _resolve_queries(world, *queries, tags=tags, traits_only=True)
    yield from itertools.product(*query_results, repeat=repeat)


def entity_permutations(
    world: World, r: int, *traits: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[Trait, ...], ...]]]:
    """Generates r-length permutations of entities matching given traits and tags.

    Args:
        world: The ECS world to query.
        r: The length of each permutation.
        *traits: Trait types to filter entities by.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples containing:
        - A tuple of permuted `Entity` instances.
        - A tuple of corresponding trait tuples.
    """
    for perm in itertools.permutations(world.get_entities(*traits, tags=tags), r=r):
        yield zip(*perm)


def trait_permutations(
    world: World, r: int, *traits: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[tuple[Trait, ...], ...]]:
    """Generates r-length permutations of trait tuples.

    Args:
        world: The ECS world to query.
        r: The length of each permutation.
        *traits: Trait types to include.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples of trait permutations.
    """
    yield from itertools.permutations(world.get_traits(*traits, tags=tags), r=r)


def entity_combinations(
    world: World, r: int, *traits: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[Trait, ...], ...]]]:
    """Generates r-length combinations of entities matching given traits and tags.

    Args:
        world: The ECS world to query.
        r: The length of each combination.
        *traits: Trait types to filter entities by.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples containing:
        - A tuple of combined `Entity` instances.
        - A tuple of corresponding trait tuples.
    """
    for combination in itertools.combinations(
        world.get_entities(*traits, tags=tags), r
    ):
        yield zip(*combination)


def trait_combinations(
    world: World, r: int, *traits: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[tuple[Trait, ...], ...]]:
    """Generates r-length combinations of trait tuples.

    Args:
        world: The ECS world to query.
        r: The length of each combination.
        *traits: Trait types to include.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples of trait combinations.
    """
    yield from itertools.combinations(world.get_traits(*traits, tags=tags), r)


def entity_combinations_with_replacement(
    world: World, r: int, *traits: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[tuple[Entity, ...], tuple[tuple[Trait, ...], ...]]]:
    """Generates r-length combinations of entities with replacement.

    Args:
        world: The ECS world to query.
        r: The length of each combination.
        *traits: Trait types to filter entities by.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples containing:
        - A tuple of `Entity` instances.
        - A tuple of corresponding trait tuples.
    """
    for comb in itertools.combinations_with_replacement(
        world.get_entities(*traits, tags=tags), r=r
    ):
        yield zip(*comb)


def trait_combinations_with_replacement(
    world: World, r: int, *traits: type[Trait], tags: Optional[set[str]] = None
) -> Iterator[tuple[tuple[Trait, ...], ...]]:
    """Generates r-length combinations of trait tuples with replacement.

    Args:
        world: The ECS world to query.
        r: The length of each combination.
        *traits: Trait types to include.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples of trait combinations with replacement.
    """
    yield from itertools.combinations_with_replacement(
        world.get_traits(*traits, tags=tags), r=r
    )


def entity_filter(
    world: World,
    predicate: Callable[[Entity, tuple[Trait, ...]], bool],
    *traits: type[Trait],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Entity, tuple[Trait, ...]]]:
    """Filters entities and their traits based on a predicate function.

    Args:
        world: The ECS world to query.
        predicate: A function that takes an entity and its traits, returning True if included.
        *traits: Trait types to filter entities by.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples of entities and their traits that satisfy the predicate.
    """
    for e, traits in world.get_entities(*traits, tags=tags):
        if predicate(e, traits):
            yield e, traits


def trait_filter(
    world: World,
    predicate: Callable[[tuple[Trait, ...]], bool],
    *traits: type[Trait],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Trait, ...]]:
    """Filters trait tuples based on a predicate function.

    Args:
        world: The ECS world to query.
        predicate: A function that takes a trait tuple and returns True if included.
        *traits: Trait types to include.
        tags: Optional set of tags to filter entities.

    Yields:
        Trait tuples that satisfy the predicate.
    """
    for traits in world.get_traits(*traits, tags=tags):
        if predicate(traits):
            yield traits


def entity_groupby(
    world: World,
    key: Callable[[Entity, tuple[Trait, ...]], Any],
    *traits: type[Trait],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Any, Iterator[tuple[Entity, tuple[Trait, ...]]]]]:
    """Groups entities and traits using a key function.

    Args:
        world: The ECS world to query.
        key: A function returning a key to group entities by.
        *traits: Trait types to filter entities by.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples of (key, group iterator) pairs.
    """
    yield from itertools.groupby(
        world.get_entities(*traits, tags=tags), key=lambda pair: key(*pair)
    )


def trait_groupby(
    world: World,
    key: Callable[[tuple[Trait, ...]], Any],
    *traits: type[Trait],
    tags: Optional[set[str]] = None,
) -> Iterator[tuple[Any, Iterator[tuple[Trait, ...]]]]:
    """Groups trait tuples using a key function.

    Args:
        world: The ECS world to query.
        key: A function returning a key to group traits by.
        *traits: Trait types to include.
        tags: Optional set of tags to filter entities.

    Yields:
        Tuples of (key, group iterator) pairs.
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

    Args:
        world: The ECS world to query.
        func: A function to apply to each (entity, traits) pair.
        *traits: Trait types to filter entities by.
        tags: Optional set of tags to filter entities.

    Yields:
        The results of applying the function to each entity-trait pair.
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
    """Applies a function to each trait tuple.

    Args:
        world: The ECS world to query.
        func: A function to apply to each trait tuple.
        *traits: Trait types to include.
        tags: Optional set of tags to filter entities.

    Yields:
        The results of applying the function to each trait tuple.
    """
    yield from itertools.starmap(func, world.get_traits(*traits, tags=tags))
