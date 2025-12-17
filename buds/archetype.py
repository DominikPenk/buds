"""High-performance archetype-based ECS world implementation.

This module defines the `ArchetypeWorld` class and related data structures
for organizing entities into *archetypes* — dense groups of entities sharing
identical trait compositions. This design provides significant performance
benefits for iteration-heavy or cache-sensitive ECS workloads.

Key components:
- `ArcheType`: Manages dense arrays of trait data for a specific composition.
- `EntityLocation`: Tracks an entity’s location within an archetype.
- `ArchetypeWorld`: A world implementation built on top of archetype storage.

Exports:
    ArchetypeWorld
"""

from collections.abc import Iterable, Iterator
from typing import NamedTuple, TypeAlias, TypeVar

from .base import Entity, EntityNotFoundError, TraitNotFoundError, World, is_trait_type

T = TypeVar("T")

__all__ = ["ArchetypeWorld"]

ArchetypeKey: TypeAlias = tuple[type[T], ...]


class EntityLocation(NamedTuple):
    """Stores the location of an entity within an archetype.

    Attributes:
        archtype: The `ArcheType` that owns the entity.
        index: The index of the entity within that archetype’s dense arrays.
    """

    archtype: "ArcheType"
    index: int


class ArcheType:
    """Represents a homogeneous collection of entities sharing the same trait composition.

    Each `ArcheType` maintains dense, parallel arrays of trait instances, one per trait type,
    ensuring fast iteration and cache-friendly access.
    """

    def __init__(self, trait_types: ArchetypeKey):
        self.key = trait_types
        self.trait_data: dict[type[T], list[T]] = {t: [] for t in trait_types}
        self.entity_ids: list[int] = []

    @staticmethod
    def get_canonical_order(trait_types: tuple[type[T], ...]) -> ArchetypeKey:
        """Return the canonical sorted order of trait types.

        Args:
            trait_types: A tuple of trait classes.

        Returns:
            ArchetypeKey: A sorted tuple of the same types by class name.
        """
        return tuple(sorted(trait_types, key=lambda t_type: t_type.__name__))

    def add(self, entity: int, traits: Iterable[T]) -> int:
        """Add an entity and its trait instances to this archetype.

        Args:
            entity: The integer ID of the entity.
            traits: Iterable of trait instances to add.

        Returns:
            int: The index at which the entity was stored.

        Raises:
            ValueError: If the entity already exists or any required trait is missing.
        """

        if entity in self.entity_ids:
            raise ValueError(f"Entity {entity} already in this archetype")

        index = len(self.entity_ids)
        self.entity_ids.append(entity)
        trait_map = {type(t): t for t in traits}
        for t_type in self.trait_data.keys():
            instance = trait_map.get(t_type)
            if instance is None:
                raise ValueError(
                    f"Missing trait {t_type} for entity {entity} during add."
                )
            self.trait_data[t_type].append(instance)
        return index

    def pop(self, index: int) -> tuple[int | None, list[T]]:
        """Remove an entity and return its traits and any swapped entity ID.

        Args:
            index: The index of the entity to pop.

        Returns:
            tuple[int | None, list[T]]: The ID of the moved entity (if any)
            and the list of removed trait instances.
        """
        last_index = len(self.entity_ids) - 1
        if index == last_index:
            self.entity_ids.pop()
            return None, [t_store.pop() for t_store in self.trait_data.values()]
        else:
            moved_entity = self.entity_ids[-1]
            self.entity_ids[index] = moved_entity
            self.entity_ids.pop()
            traits: list[T] = []
            for trait_store in self.trait_data.values():
                traits.append(trait_store[index])
                trait_store[index] = trait_store.pop()
            return moved_entity, traits

    def __len__(self) -> int:
        """Return the number of entities stored in this archetype."""
        return len(self.entity_ids)

    def get_traits(self) -> Iterator[tuple[int, dict[type[T], T]]]:
        """Iterate over all entities and their associated trait mappings.

        Yields:
            tuple[int, dict[type[T], T]]: Each entity ID with its trait type-to-instance map.
        """
        for idx, entity in enumerate(self.entity_ids):
            yield (
                entity,
                {t_type: t_store[idx] for t_type, t_store in self.trait_data.items()},
            )


class ArchetypeWorld(World):
    """A world implementation using archetype-based storage.

    Entities in this world are grouped by their exact trait composition into archetypes,
    enabling fast queries and iteration over homogeneous entity groups.
    """

    def __init__(self) -> None:
        super().__init__()
        self._next_entity = 0
        self._free_entities = []
        self._entity_map: dict[int, EntityLocation] = {}
        self._archetypes: dict[ArchetypeKey, ArcheType] = {}

    def _get_or_create_archetype(self, key: ArchetypeKey) -> ArcheType:
        """Retrieve an existing archetype or create a new one.

        Args:
            key: The canonical tuple of trait classes for the archetype.

        Returns:
            ArcheType: The corresponding archetype instance.
        """
        if key not in self._archetypes:
            self._archetypes[key] = ArcheType(key)
        return self._archetypes[key]

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

        trait_types = tuple(type(t) for t in traits)
        archtype = self._get_or_create_archetype(
            ArcheType.get_canonical_order(trait_types)
        )
        index = archtype.add(entity_id, traits)

        self._entity_map[entity_id] = EntityLocation(archtype, index)

        return Entity(entity_id, self)

    def delete_entity(self, entity: int) -> None:
        """Deletes an entity from the world.

        Args:
            entity: The entity ID to delete.
        """
        if entity not in self._entity_map:
            raise EntityNotFoundError(f"Entity {entity} does not exist")

        arch, index = self._entity_map[entity]
        moved_entity, _ = arch.pop(index)

        self._entity_map.pop(entity)
        self._free_entities.append(entity)

        if moved_entity is not None:
            self._entity_map[moved_entity] = EntityLocation(arch, index)

    def is_alive(self, entity: int) -> bool:
        """Checks whether an entity currently exists in the world.

        Args:
            entity: The entity ID to check.

        Returns:
            True if the entity is active, False otherwise.
        """
        return entity in self._entity_map

    def add_trait(self, entity: int, trait: T) -> None:
        """Adds a trait instance to an entity.

        Args:
            entity: The target entity ID.
            trait: The trait instance to add.
        """
        trait_type = type(trait)
        if not is_trait_type(trait_type):
            raise TypeError("Trait must be a class decorated with @component.")
        if entity not in self._entity_map:
            raise EntityNotFoundError(f"Entity {entity} does not exist")

        old_arch, old_index = self._entity_map[entity]
        if trait_type in old_arch.key:
            raise ValueError(f"{entity} already has a trait of type {trait_type}")

        # Pop the entity from it's original archetype
        moved_entity, traits = old_arch.pop(old_index)
        if moved_entity is not None:
            self._entity_map[moved_entity] = EntityLocation(old_arch, old_index)

        traits.append(trait)
        trait_types = tuple(type(t) for t in traits)
        new_arch = self._get_or_create_archetype(
            ArcheType.get_canonical_order(trait_types)
        )

        new_index = new_arch.add(entity, traits)
        self._entity_map[entity] = EntityLocation(new_arch, new_index)

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

        if entity not in self._entity_map:
            raise EntityNotFoundError(f"Entity {entity} is not alive")

        old_arch, old_index = self._entity_map[entity]
        if trait_type not in old_arch.key:
            raise TraitNotFoundError(
                f"Entity {entity} does not have a component of type {trait_type}"
            )

        moved_entity, traits = old_arch.pop(old_index)
        if moved_entity is not None:
            self._entity_map[moved_entity] = EntityLocation(old_arch, old_index)

        traits = [t for t in traits if type(t) is not trait_type]
        new_trait_types = tuple(type(t) for t in traits)
        new_arch = self._get_or_create_archetype(
            ArcheType.get_canonical_order(new_trait_types)
        )

        new_index = new_arch.add(entity, traits)
        self._entity_map[entity] = EntityLocation(new_arch, new_index)

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
                f"Attempted to remove non-Component type {trait_type} from Entity {entity}. "
                f"All traits must be decorated with @trait."
            )
        return trait_type in self._entity_map[entity].archtype.key

    def get_entities_from_traits(
        self, *trait_types: type[T]
    ) -> Iterator[tuple[Entity, tuple[T, ...]]]:
        """Retrieves all entities that possess the specified traits.

        Args:
            *trait_types: Trait types to filter entities by.

        Returns:
            An iterator of `(Entity, (traits...))` tuples.
        """
        required_types = set(trait_types)

        for arch in self._archetypes.values():
            if not required_types.issubset(arch.key):
                continue

            for entity_id, trait_dict in arch.get_traits():
                entity = Entity(entity_id, self)
                traits = tuple(trait_dict[t] for t in trait_types)
                yield entity, traits
                yield entity, traits
