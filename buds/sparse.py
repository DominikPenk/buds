"""Sparse storage ECS world implementation.

This module provides `SparseWorld`, a simple and flexible ECS world
implementation that stores traits in per-type sparse dictionaries.

Sparse storage is ideal for:
- Small to medium-sized worlds.
- Use cases where entity compositions vary widely.
- Debugging or prototyping, before optimizing with archetype-based worlds.

Exports:
    SparseWorld
"""

from .base import (
    World,
    Entity,
    is_trait,
    is_trait_type,
    EntityNotFoundError,
    TraitNotFoundError,
)
from typing import TypeVar
from collections.abc import Iterator
from collections import defaultdict

T = TypeVar("T")

__all__ = ["SparseWorld"]


class SparseWorld(World):
    """A sparse implementation of the ECS World.

    The `SparseWorld` class stores entity-trait relationships in sparse
    dictionaries keyed by trait type and entity ID. This representation
    is space-efficient and supports fast insertion and deletion.

    Attributes:
        _next_entity: The next available entity ID.
        _free_entities: List of reusable entity IDs.
        _trait_stores: Mapping from trait type to a dictionary of entities and trait instances.
    """

    def __init__(self) -> None:
        super().__init__()
        self._next_entity = 0
        self._free_entities = []
        self._trait_stores: dict[type[T], dict[int, T]] = defaultdict(dict)

    def create_entity(self, *traits: T) -> Entity:
        """Creates a new entity and assigns the given traits.

        Args:
            *traits: Optional traits to attach to the new entity.

        Returns:
            The newly created entity.
        """
        if self._free_entities:
            entity_id = self._free_entities.pop()
        else:
            entity_id = self._next_entity
            self._next_entity += 1
        for trait in traits:
            self.add_trait(entity_id, trait)
        return Entity(entity_id, self)

    def delete_entity(self, entity: int) -> None:
        """Deletes an entity from the world.

        Args:
            entity: The entity ID to delete.
        """
        if not self.is_alive(entity):
            raise EntityNotFoundError(f"Entity {entity} does not exist.")
        for store in self._trait_stores.values():
            store.pop(entity, None)
        self._free_entities.append(entity)

    def is_alive(self, entity: int) -> bool:
        """Checks whether an entity currently exists in the world.

        Args:
            entity: The entity ID to check.

        Returns:
            True if the entity is active, False otherwise.
        """
        return entity < self._next_entity and entity not in self._free_entities

    def add_trait(self, entity: int, trait: T) -> None:
        """Adds a trait instance to an entity.

        Args:
            entity: The target entity ID.
            trait: The trait instance to add.
        """
        if not self.is_alive(entity):
            raise EntityNotFoundError(f"Entity {entity} does not exist.")
        trait_type = type(trait)
        if not is_trait(trait):
            raise TypeError(
                f"Attempted to add non-Component object {trait_type.__name__} to Entity {entity}. "
                f"All traits must be decorated with @trait."
            )

        if entity in self._trait_stores[trait_type]:
            raise ValueError(f"Entity {entity} already has trait of type {trait_type}")
        self._trait_stores[trait_type][entity] = trait

    def remove_trait(self, entity: int, trait_type: type[T]) -> None:
        """Removes a specific trait type from an entity.

        Args:
            entity: The target entity ID.
            trait_type: The type of the trait to remove.
        """
        if not is_trait_type(trait_type):
            raise TypeError(
                f"Attempted to remove non-trait type {trait_type} from Entity {entity}. "
                f"All traits must be decorated with @trait."
            )
        if not self.is_alive(entity):
            raise EntityNotFoundError(entity)
        if entity not in self._trait_stores[trait_type]:
            raise TraitNotFoundError(
                f"Entity {entity} does not have trait of type {trait_type}"
            )
        del self._trait_stores[trait_type][entity]

    def has_trait(self, entity: int, trait_type: type[T]) -> bool:
        """Checks whether an entity has a given trait type.

        Args:
            entity: The entity ID.
            trait_type: The trait type to check.

        Returns:
            True if the entity has the trait, False otherwise.
        """
        if not self.is_alive(entity):
            raise EntityNotFoundError(f"Entity {entity} does not exist")
        if not is_trait_type(trait_type):
            raise TypeError(
                f"Attempted to check non-trait type {trait_type} for Entity {entity}. "
                f"All traits must be decorated with @trait."
            )
        return entity in self._trait_stores[trait_type]

    def get_entities_from_traits(
        self, *trait_types: type[T]
    ) -> Iterator[tuple[Entity, tuple[T, ...]]]:
        """Retrieves all entities that possess the specified traits.

        Args:
            *trait_types: Trait types to filter entities by.

        Returns:
            An iterator of `(Entity, (traits...))` tuples.
        """
        # TODO: create a cache for often queried trait combinations
        if not all(is_trait_type(ct) for ct in trait_types):
            raise TypeError(
                "All trait types must be valid Component classes decorated with @trait."
            )

        # Find the smallest store to iterate over
        smallest_store = min((self._trait_stores[ct] for ct in trait_types), key=len)

        for entity in smallest_store:
            if all(entity in self._trait_stores[ct] for ct in trait_types):
                yield (
                    Entity(entity, self),
                    tuple(self._trait_stores[ct][entity] for ct in trait_types),
                )
