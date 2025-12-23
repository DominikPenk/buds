from collections.abc import Iterable, Iterator
from typing import Optional, TypeVar

import numpy as np

from ... import inspect
from ...archetype import ArcheType, ArchetypeKey, ArchetypeWorld
from ...base import Entity, Trait
from .dtypes import get_dtype
from .views import VectorizedTraitView, create_vectorized_view_class, create_view_class

__all__ = ["NumpyArchetypeWorld"]

T = TypeVar("T", bound=Trait)


class NumpyArcheType(ArcheType):
    def __init__(self, trait_types: ArchetypeKey, capacity: int = 256):
        self.key = trait_types
        self.entity_ids: list[int] = []
        self.trait_data: dict[type, np.ndarray] = {}
        self.trait_dtype: dict[type, np.dtype] = {}
        for t in trait_types:
            trait_schema = inspect.TraitSchema.create(t)
            trait_dtype = get_dtype(trait_schema)
            self.trait_dtype[t] = trait_dtype
            self.trait_data[t] = np.empty(capacity, trait_dtype)

        self._capacity = capacity

    def __len__(self) -> int:
        """Return the number of entities stored in this archetype."""
        return len(self.entity_ids)

    def _ensure_capacity(self, new_count: int):
        """Ensure backing arrays can hold new_count entities."""
        new_cap = self._capacity
        if new_count < self._capacity // 4 and new_count > 16:
            new_cap = max(16, self._capacity // 2)
            for t, storage in self.trait_data.items():
                self.trait_data[t] = storage[:new_cap].copy()
        elif new_count > self._capacity:
            new_cap = max(self._capacity * 2, new_count)
            for t, storage in self.trait_data.items():
                new_storage = np.empty(new_cap, dtype=storage.dtype)
                new_storage[: self._capacity] = storage
                self.trait_data[t] = new_storage
        self._capacity = new_cap

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

        self._ensure_capacity(index + 1)

        for trait in traits:
            trait_type = type(trait)
            if trait_type not in self.trait_data:
                raise TypeError(f"Trait type {trait_type} not part of this archetype")
            storage = self.trait_data[trait_type]
            storage[index] = tuple(
                getattr(trait, field.name)
                for field in inspect.inspect_trait(trait_type).fields
            )

        return index

    def pop(self, index: int) -> tuple[int | None, list[T]]:
        """Remove an entity and return its traits and any swapped entity ID.

        Args:
            index: The index of the entity to pop.

        Returns:
            tuple[int | None, list[T]]: The ID of the moved entity (if any)
            and the list of removed trait instances.
        """
        n = len(self.entity_ids)
        if index < 0 or index >= n:
            raise IndexError("index out of range")

        last_index = n - 1
        removed_traits: list[T] = []

        # Get the traits to return
        for trait_type, storage in self.trait_data.items():
            storage_dtype = storage.dtype
            if storage_dtype is None:
                raise RuntimeError(f"Trait {trait_type.__name__} has no members")
            vals = {
                f: storage[f][last_index].item()
                if isinstance(storage[f][last_index], np.ndarray)
                and storage[f][last_index].ndim == 0
                else storage[f][last_index]
                for f in storage_dtype.names  # type: ignore
            }
            removed_traits.append(trait_type(**vals))

        moved_entity = None
        if index == last_index:
            self.entity_ids.pop()
        else:
            storage_dtype = storage.dtype
            if storage_dtype is None:
                raise RuntimeError(f"Trait {trait_type.__name__} has no members")
            moved_entity = self.entity_ids[-1]
            self.entity_ids[index] = moved_entity
            self.entity_ids.pop()
            for storage in self.trait_data.values():
                for field in storage_dtype.names:  # type: ignore
                    storage[field][index] = storage[field][last_index]

        self._ensure_capacity(n - 1)

        return moved_entity, removed_traits

    def get_traits(self) -> Iterator[tuple[int, dict[type[T], T]]]:
        """Iterate over all entities and their associated trait mappings.

        Yields:
            tuple[int, dict[type[T], T]]: Each entity ID with its trait type-to-instance map.
        """
        for idx, entity in enumerate(self.entity_ids):
            yield (
                entity,
                {
                    t_type: create_view_class(t_type)(t_store, idx)
                    for t_type, t_store in self.trait_data.items()
                },
            )

    def get_trait_data(self, trait_type: type[T]) -> np.ndarray | None:
        return self.trait_data.get(trait_type, None)

    def empose_order(self, order: Iterable[int]) -> None:
        order = list(order)
        n = len(order)
        if n != len(self):
            raise RuntimeError(f"Invalid order, expected {len(self)} entries, got {n}")

        self.entity_ids = [self.entity_ids[i] for i in order]
        for t in self.trait_data:
            self.trait_data[t][:n] = self.trait_data[t][order]

    def get_trait(self, index: int, trait_type: type[T]) -> T:
        store = self.trait_data[trait_type]
        return create_view_class(trait_type)(store, index)

    @property
    def capacity(self) -> int:
        return self._capacity


class NumpyArchetypeWorld(ArchetypeWorld):
    def __init__(self) -> None:
        self._archetypes: dict[ArchetypeKey, NumpyArcheType] = dict()
        super().__init__()

    def _get_or_create_archetype(self, key: ArchetypeKey) -> NumpyArcheType:
        """Retrieve an existing archetype or create a new one.

        Args:
            key: The canonical tuple of trait classes for the archetype.

        Returns:
            NumpyArcheType: The corresponding archetype instance using numpy storage if possible.
        """
        if key not in self._archetypes:
            self._archetypes[key] = NumpyArcheType(key)
        return self._archetypes[key]

    def get_vectorized_entities(
        self, *trait_types: type[T], tags: Optional[set[str]] = None
    ) -> tuple[list[Entity], tuple[VectorizedTraitView[T], ...]]:
        all_recs = {t: [] for t in trait_types}
        all_masks = {t: [] for t in trait_types}
        entities: list[Entity] = []
        views: list = []
        req_trait_types = set(trait_types)

        for arch in self._archetypes.values():
            if not req_trait_types.issubset(arch.key):
                continue

            num_alive = len(arch)
            if num_alive == 0:
                continue

            if tags:
                entity_mask = np.array(
                    self.has_tags(eid, *tags) for eid in arch.entity_ids
                )
            else:
                entity_mask = np.ones(num_alive, dtype=bool)
            entities.extend(
                [
                    Entity(eid, self)
                    for eid, has_tags in zip(arch.entity_ids, entity_mask)
                    if has_tags
                ]
            )

            for trait_type in trait_types:
                data = arch.get_trait_data(trait_type)

                all_recs[trait_type].append(data[:num_alive])  # type: ignore
                all_masks[trait_type].append(entity_mask)

        views = []
        for trait_type in trait_types:
            ViewType = create_vectorized_view_class(trait_type)
            instance = ViewType(all_recs[trait_type], all_masks[trait_type])
            views.append(instance)

        return entities, tuple(views) if len(views) > 1 else (views[0],)

    def get_vectorized_traits(
        self, *trait_types: type[T], tags: Optional[set[str]] = None
    ) -> tuple[T, ...]:
        return self.get_vectorized_entities(*trait_types, tags=tags)[1]  # type: ignore
