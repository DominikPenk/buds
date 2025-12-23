"""Base ECS abstractions and foundational classes.

This module defines the fundamental building blocks of the Entity–Component–System (ECS)
architecture used across the library, including the `World`, `Entity`, and trait utilities.

It provides:
- The `World` base class, which defines the common interface for all world implementations.
- The `Entity` wrapper, representing individual entities within a world.
- The `is_trait` and `is_trait_type` utilities for validating trait instances and types.

All higher-level modules (`sparse`, `archetype`, `seed`) build upon these abstractions.

Exports:
    World
    Entity
    is_trait
    is_trait_type
"""

from __future__ import annotations

import abc
import inspect
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Iterable, Optional, TypeVar, dataclass_transform, overload

Seed = TypeVar("Seed")
T = TypeVar("T")


TRAIT_HINT = "__is_trait"

__all__ = [
    "Entity",
    "is_trait",
    "is_trait_type",
    "World",
    "Trait",
    "EntityNotFoundError",
    "TraitNotFoundError",
]


class EntityNotFoundError(Exception):
    """Exception raised when an entity is not found in the world."""

    def __init__(self, msg: int | Entity | str):
        if isinstance(msg, Entity):
            msg = f"Entity {msg.id} does not exist"
        elif isinstance(msg, int):
            msg = f"Entity {msg} does not exist"
        super().__init__(msg)


class TraitNotFoundError(Exception):
    """Exception raised when a trait is not found on an entity."""

    pass


@dataclass_transform()
class Trait:
    """Base class for defining ECS traits.

    Subclasses of `Trait` are automatically marked as traits.
    """

    def __init_subclass__(cls):
        """Automatically marks all subclasses as ECS traits."""
        # Mark subclass as a trait type so is_trait_type can detect it.
        setattr(cls, TRAIT_HINT, True)
        return dataclass(cls)


def is_trait(obj: object) -> bool:
    """Checks whether an object instance is marked as a trait.

    Args:
        obj: The object to check.

    Returns:
        True if the object is a trait instance, False otherwise.
    """
    return isinstance(obj, Trait) or hasattr(obj, "__is_trait")


def is_trait_type(cls: type) -> bool:
    """Checks whether a class type is marked as a trait.

    Args:
        cls: The class to check.

    Returns:
        True if the class is a trait type, False otherwise.
    """
    return inspect.isclass(cls) and (cls is Trait or hasattr(cls, "__is_trait"))


def register_trait_base_class(cls: type[Any]):
    setattr(cls, TRAIT_HINT, True)


_Trait = TypeVar("_Trait")
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_T3 = TypeVar("_T3")
_T4 = TypeVar("_T4")


class World(abc.ABC):
    """Abstract base class defining the ECS world interface.

    A `World` manages entity creation, deletion, and their associated traits and tags.
    """

    def __init__(self) -> None:
        self._tags: dict[str, set[int]] = defaultdict(set)

    @abc.abstractmethod
    def create_entity(self, *traits: _Trait) -> Entity:
        """Creates a new entity and assigns the given traits.

        Args:
            *traits: Optional traits to attach to the new entity.

        Returns:
            The newly created entity.
        """

    @abc.abstractmethod
    def delete_entity(self, entity: int) -> None:
        """Deletes an entity from the world.

        Args:
            entity: The entity ID to delete.

        Raises:
            EntityNotFoundError: If the entity ID does not exist in the world.
        """

    @abc.abstractmethod
    def is_alive(self, entity: int) -> bool:
        """Checks whether an entity currently exists in the world.

        Args:
            entity: The entity ID to check.

        Returns:
            True if the entity is active, False otherwise.
        """

    @abc.abstractmethod
    def add_trait(self, entity: int, trait: _Trait) -> None:
        """Adds a trait instance to an entity.

        Raises:
            EntityNotFoundError: If the entity ID does not exist.
            TypeError: If the trait is not a valid trait instance (not decorated by [`@trait`][buds.base.trait]).
        """

    @abc.abstractmethod
    def remove_trait(self, entity: int, trait_type: type[_Trait]) -> None:
        """Removes a specific trait type from an entity.

        Args:
            entity: The target entity ID.
            trait_type: The type of the trait to remove.

        Raises:
            EntityNotFoundError: If the entity ID does not exist.
            TraitNotFoundError: If the entity does not have the specified trait type.
            TypeError: If the trait is not a valid trait instance (not decorated by [`@trait`][buds.base.trait]).
        """

    @abc.abstractmethod
    def get_trait(self, entity: int, trait_type: type[_Trait]) -> _Trait:
        """Retreives a specific trait type from an entity

        Args:
            entity: The target entity ID.
            trait_type: The type of the trait to remove.

        Raises:
            EntityNotFoundError: If the entity ID does not exist.
            TraitNotFoundError: If the entity does not have the specified trait type.
            TypeError: If the trait is not a valid trait instance (not decorated by [`@trait`][buds.base.trait]).
        """

    @abc.abstractmethod
    def has_trait(self, entity: int, trait_type: type[_Trait]) -> bool:
        """Checks whether an entity has a given trait type.

        Args:
            entity: The entity ID.
            trait_type: The trait type to check.

        Returns:
            True if the entity has the trait, False otherwise.

        Raises:
            EntityNotFoundError: If the entity ID does not exist.
            TypeError: If the trait is not a valid trait instance (not decorated by [`@trait`][buds.base.trait]).
        """

    def add_tags(self, entity: int, *tags: str) -> None:
        """Associates one or more tags with an entity.

        Args:
            entity: The entity ID.
            *tags: The tags to associate.

        Raises:
            EntityNotFoundError: If the entity ID does not exist.
            TypeError: If any of the provided tags are not strings.
        """
        if not self.is_alive(entity):
            raise EntityNotFoundError(entity)
        if not all(isinstance(tag, str) for tag in tags):
            raise TypeError("tags must be strings")
        for tag in tags:
            self._tags[tag].add(entity)

    def remove_tags(self, entity: int, *tags: str) -> None:
        """Removes one or more tags from an entity.

        Args:
            entity: The entity ID.
            *tags: The tags to remove.

        Raises:
            EntityNotFoundError: If the entity ID does not exist.
            TypeError: If any of the provided tags are not strings.
        """
        if not all(isinstance(tag, str) for tag in tags):
            raise TypeError("tags must be strings")
        if not self.is_alive(entity):
            raise EntityNotFoundError(entity)
        for tag in tags:
            self._tags[tag].discard(entity)

    def has_tags(self, entity: int, *tags: str) -> bool:
        """Checks whether an entity has all specified tags.

        Args:
            entity: The entity ID.
            *tags: The tags to check.

        Returns:
            True if the entity has all tags, False otherwise.

        Raises:
            EntityNotFoundError: If the entity ID does not exist.
            TypeError: If any of the provided tags are not strings.
        """
        if not self.is_alive(entity):
            raise EntityNotFoundError(entity)
        if not all(isinstance(tag, str) for tag in tags):
            raise TypeError("tags must be strings")
        return all(entity in self._tags[tag] for tag in tags)

    @abc.abstractmethod
    def get_entities_from_traits(
        self, *trait_types: type[_Trait]
    ) -> Iterator[tuple[Entity, tuple[_Trait, ...]]]:
        """Retrieves all entities that possess the specified traits.

        Args:
            *trait_types: Trait types to filter entities by.

        Returns:
            An iterator of `(Entity, (traits...))` tuples.
        """

    @overload
    def get_entities(
        self, trait_1: type[_T1], tags: Optional[set[str] | str] = None
    ) -> Iterator[tuple[Entity, _T1]]: ...

    @overload
    def get_entities(
        self,
        trait_1: type[_T1],
        trait_2: type[_T2],
        tags: Optional[set[str] | str] = None,
    ) -> Iterator[tuple[Entity, tuple[_T1, _T2]]]: ...

    @overload
    def get_entities(
        self,
        trait_1: type[_T1],
        trait_2: type[_T2],
        trait_3: type[_T3],
        tags: Optional[set[str] | str] = None,
    ) -> Iterator[tuple[Entity, tuple[_T1, _T2, _T3]]]: ...

    @overload
    def get_entities(
        self,
        trait_1: type[_T1],
        trait_2: type[_T2],
        trait_3: type[_T3],
        trait_4: type[_T4],
        tags: Optional[set[str] | str] = None,
    ) -> Iterator[tuple[Entity, tuple[_T1, _T2, _T3, _T4]]]: ...

    def get_entities(
        self, *trait_types: type[_Trait], tags: Optional[set[str] | str] = None
    ) -> Iterator[tuple[Entity, tuple[_Trait, ...] | _Trait]]:
        """Retrieves entities that match the given traits and optional tags.

        Args:
            *trait_types: Trait types to filter entities by.
            tags: Optional set of tags to filter entities.

        Returns:
            An iterator of `(Entity, (traits...))` tuples matching the criteria.

        Raises:
            TypeError: If any trait_types are not valid trait types or if any `tags` are not strings.
        """
        if not all(is_trait_type(t) for t in trait_types):
            raise TypeError(
                "Attempted to query non-trait types. All traits must be decorated with @trait."
            )
        if tags is None:
            iterator = self.get_entities_from_traits(*trait_types)
        else:
            if isinstance(tags, str):
                tags = {
                    tags,
                }

            if not all(isinstance(t, str) for t in tags):
                raise TypeError("tags must be strings")

            iterator = filter(
                lambda e: self.has_tags(e[0].id, *tags),
                self.get_entities_from_traits(*trait_types),
            )

        if len(trait_types) == 1:
            yield from map(lambda e: (e[0], e[1][0]), iterator)
        else:
            yield from iterator

    @overload
    def get_traits(
        self, trait_1: type[_T1], tags: Optional[set[str] | str] = None
    ) -> Iterator[_T1]: ...

    @overload
    def get_traits(
        self,
        trait_1: type[_T1],
        trait_2: type[_T2],
        tags: Optional[set[str] | str] = None,
    ) -> Iterator[tuple[_T1, _T2]]: ...

    @overload
    def get_traits(
        self,
        trait_1: type[_T1],
        trait_2: type[_T2],
        trait_3: type[_T3],
        tags: Optional[set[str] | str] = None,
    ) -> Iterator[tuple[_T1, _T2, _T3]]: ...

    @overload
    def get_traits(
        self,
        trait_1: type[_T1],
        trait_2: type[_T2],
        trait_3: type[_T3],
        trait_4: type[_T4],
        tags: Optional[set[str] | str] = None,
    ) -> Iterator[tuple[_T1, _T2, _T3, _T4]]: ...

    def get_traits(
        self, *trait_types: type[_Trait], tags: Optional[set[str]] = None
    ) -> Iterator[tuple[_Trait, ...] | _Trait]:
        """Retrieves only the trait tuples of entities matching given traits and tags.

        Args:
            *trait_types: Trait types to include.
            tags: Optional set of tags to filter entities.

        Returns:
            An iterator of trait tuples matching the criteria.

        Raises:
            TypeError: If any trait_types are not valid trait types or if any `tags` are not strings.
        """
        yield from map(lambda r: r[1], self.get_entities(*trait_types, tags=tags))

    def empose_order(self, order: Iterable[int], *traits: type[_Trait]) -> None:
        raise NotImplementedError(
            f"World type {type(self)} does not support trait ordering"
        )


class Entity:
    """Represents an entity within a :class:`World`.

    Entities act as handles for managing traits and tags within the ECS system.
    They encapsulate the entity ID and provide convenient methods to interact with the world.
    """

    def __init__(self, id: int, world: World) -> None:
        """Initializes an entity instance.

        Args:
            id: The unique entity ID.
            world: The world instance this entity belongs to.
        """
        self.id = id
        self.world: Optional[World] = world

    def __repr__(self) -> str:
        return f"<Entity({self.id})>"

    def __int__(self) -> int:
        return self.id

    def __eq__(self, other: Entity) -> bool:
        return other.id == self.id and other.world == self.world

    def __ne__(self, other: Entity) -> bool:
        return self.id != other.id or other.world != self.world

    def __gt__(self, other: Entity) -> bool:
        return self.id > other.id

    def __ge__(self, other: Entity) -> bool:
        return self.id >= other.id

    def __lt__(self, other: Entity) -> bool:
        return self.id < other.id

    def __le__(self, other: Entity) -> bool:
        return self.id <= other.id

    def add_trait(self, trait: _Trait) -> Entity:
        """Adds a trait to the entity.

        Args:
            trait: The trait instance to add.

        Returns:
            The entity itself, allowing method chaining.

        Raises:
            EntityNotFoundError: Inherited from [`World.add_trait`][buds.base.World.add_trait].
            TypeError: Inherited from [`World.add_trait`][buds.base.World.add_trait].
        """
        assert self.world is not None, "Cannot add trait to dead entity"
        self.world.add_trait(self.id, trait)
        return self

    def remove_trait(self, trait_type: type[_Trait]) -> Entity:
        """Removes a trait type from the entity.

        Args:
            trait_type: The type of the trait to remove.

        Returns:
            The entity itself, allowing method chaining.

        Raises:
            EntityNotFoundError: Inherited from [`World.remove_trait`][buds.base.World.remove_trait].
            TraitNotFoundError: Inherited from [`World.remove_trait`][buds.base.World.remove_trait].
        """
        assert self.world is not None, "Cannot add trait to dead entity"
        self.world.remove_trait(self.id, trait_type)
        return self

    def has_trait(self, trait_type: type[_Trait]) -> bool:
        """Checks if the entity has a specific trait type.

        Args:
            trait_type: The trait type to check.

        Returns:
            True if the entity has the trait, False otherwise.

        Raises:
            EntityNotFoundError: Inherited from [`World.has_tags`][buds.base.World.has_tags].
            TypeError: Inherited from [`World.has_tags`][buds.base.World.has_tags].
        """
        assert self.world is not None, "Cannot add trait to dead entity"
        return self.world.has_trait(self.id, trait_type)

    def get_trait(self, trait_type: type[_Trait]) -> _Trait:
        """Get the trait associated with this entity.

        Args:
            trait_type: The trait type to check.

        Returns:
            A trait instance.

        Raises:
            EntityNotFoundError: Inherited from [`World.has_tags`][buds.base.World.has_tags].
            TypeError: Inherited from [`World.has_tags`][buds.base.World.has_tags].
        """
        assert self.world is not None, "Cannot add trait to dead entity"
        return self.world.get_trait(self.id, trait_type)

    def add_tags(self, *tags: str) -> Entity:
        """Adds tags to the entity.

        Args:
            *tags: One or more tags to add.

        Returns:
            The entity itself, allowing method chaining.

        Raises:
            EntityNotFoundError: Inherited from [`World.add_tags`][buds.base.World.add_tags].
            TypeError: Inherited from [`World.add_tags`][buds.base.World.add_tags].
        """
        assert self.world is not None, "Cannot add trait to dead entity"
        self.world.add_tags(self.id, *tags)
        return self

    def remove_tags(self, *tags: str) -> Entity:
        """Removes tags from the entity.

        Args:
            *tags: One or more tags to remove.

        Returns:
            The entity itself, allowing method chaining.
        """
        assert self.world is not None, "Cannot add trait to dead entity"
        self.world.remove_tags(self.id, *tags)
        return self

    def has_tags(self, *tags: str) -> bool:
        """Checks whether the entity has all specified tags.

        Args:
            *tags: The tags to check.

        Returns:
            True if all tags are present, False otherwise.
        """
        assert self.world is not None, "Cannot add trait to dead entity"
        return self.world.has_tags(self.id, *tags)

    def delete(self) -> None:
        """Deletes the entity from its world.

        After deletion, the entity object's internal world reference is cleared,
        and subsequent method calls (like [`is_alive`][buds.base.Entity.is_alive]) will reflect its dead state.

        Raises:
            EntityNotFoundError: Inherited from [`World.delete_entity`][buds.base.World.delete_entity].
        """
        assert self.world is not None, "Cannot add trait to dead entity"
        self.world.delete_entity(self.id)
        self.world = None

    def is_alive(self) -> bool:
        """Checks if the entity still exists in the world.

        Returns:
            True if the entity is active, False otherwise.
        """
        return self.world is not None and self.world.is_alive(self.id)
