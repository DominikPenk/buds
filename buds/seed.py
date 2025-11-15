"""Simplified entity construction using reusable "seed" templates.

This module introduces the `Seed` class, which acts as a blueprint for spawning entities
with predefined traits and tags. Seeds allow convenient reuse of common entity setups
without manually instantiating or wiring traits each time.

Key features:
- Automatically detects trait attributes on initialization.
- Supports adding and removing traits and tags dynamically.
- Provides lifecycle methods (`spawn`, `despawn`) to manage entity existence.

Exports:
    Seed
"""

from .base import World, Entity, is_trait
from typing import Self, TypeVar

__all__ = ["Seed"]


Trait = TypeVar("Trait")


class Seed:
    """Base class for defining reusable entity blueprints.

    A `Seed` simplifies entity creation by grouping predefined traits and optional tags.
    Subclass this to define a reusable configuration of traits for spawning entities.
    """

    def __init_subclass__(cls):
        """Automatically collects all trait instances defined in subclass attributes.

        This wraps the subclass's `__init__` method to record any ECS traits
        declared as instance attributes and sets up the `query` and `_ecs_traits`
        collections for later use when spawning.
        """
        original_init = cls.__init__

        def wrapped_init(self: Self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._ecs_traits = [c for c in vars(self).values() if is_trait(c)]
            self.query = {type(c) for c in self._ecs_traits}

        cls.__init__ = wrapped_init
        return cls

    def __init__(self, *tags: str):
        """Initializes a seed with optional tags.

        Args:
            *tags: Tags to associate with the entity when spawned.
        """
        if not hasattr(self, "_ecs_traits"):
            self._ecs_traits: list[Trait] = None
        self._ecs_entity: Entity | None = None
        self.query: set[type[Trait]] = set()
        self.tags = list(tags)

    def spawn(self, world: World) -> Self:
        """Creates an entity in the specified world using this seed’s traits.

        Args:
            world: The ECS world in which to create the entity.

        Returns:
            The seed instance itself, with its `entity` reference set.

        Raises:
            RuntimeError: If this seed has already been used to spawn an entity.
        """
        if self._ecs_entity is not None:
            raise RuntimeError("This seed has already been used to spawn an entity.")
        self._ecs_entity = world.create_entity(*self._ecs_traits)
        if self.tags:
            self._ecs_entity.add_tags(*self.tags)
        return self

    def despawn(self) -> None:
        """Deletes the entity previously created by this seed.

        Raises:
            RuntimeError: If this seed has not yet been used to spawn an entity.
        """
        if self._ecs_entity is None:
            raise RuntimeError("This seed has not been used to spawn an entity yet.")
        self._ecs_entity.delete()
        self._ecs_entity = None

    def add_trait(self, name: str, trait: Trait) -> Self:
        """Adds a trait to the seed and optionally to its spawned entity.

        Args:
            name: The attribute name under which to store the trait.
            trait: The trait instance to add.

        Returns:
            The seed instance itself for chaining.

        Raises:
            ValueError: If the provided object is not a valid ECS trait.
        """
        if not is_trait(trait):
            raise ValueError(f"{trait} is not a valid trait.")
        self._ecs_traits.append(trait)
        if self._ecs_entity:
            self._ecs_entity.add_trait(trait)

        setattr(self, name, trait)
        self.query.add(type(trait))
        return self

    def add_tags(self, *tags: str) -> Self:
        """Adds tags to the seed and optionally to its spawned entity.

        Args:
            *tags: One or more tags to add.

        Returns:
            The seed instance itself for chaining.
        """
        self.tags += tags
        if self._ecs_entity:
            self._ecs_entity.add_tags(*tags)

        return self

    def remove_tags(self, *tags: str) -> Self:
        """Removes tags from the seed and optionally from its spawned entity.

        Args:
            *tags: One or more tags to remove.

        Returns:
            The seed instance itself for chaining.
        """
        for tag in tags:
            if tag in self.tags:
                self.tags.remove(tag)
        if self._ecs_entity:
            self._ecs_entity.remove_tags(*tags)

        return self

    @property
    def entity(self) -> Entity | None:
        """The entity created by this seed, if it has been spawned.

        Returns:
            The `Entity` instance or `None` if the seed has not spawned yet.
        """
        return self._ecs_entity

    @property
    def world(self) -> World | None:
        """The world in which this seed’s entity was created.

        Returns:
            The `World` instance or `None` if the seed has not spawned yet.
        """
        if self._ecs_entity is None:
            return None
        return self._ecs_entity.world

    @property
    def spawned(self) -> bool:
        """Whether the seed has been used to spawn an entity.

        Returns:
            True if the entity exists, False otherwise.
        """
        return self._ecs_entity is not None
