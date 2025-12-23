# tests/conftest.py
from dataclasses import field
from typing import Any, Callable

import pytest

from buds.archetype import ArchetypeWorld
from buds.base import Trait
from buds.extras import NumpyArchetypeWorld
from buds.sparse import SparseWorld

WORLD_IMPLEMENTATIONS = [
    SparseWorld,
    ArchetypeWorld,
    NumpyArchetypeWorld,
]


@pytest.fixture(params=WORLD_IMPLEMENTATIONS, scope="function")
def world(request):
    """Fixture that provides a fresh instance of each known World implementation."""
    w = request.param()
    yield w


class TraitBackend:
    def __init__(
        self,
        Trait: type[Any],
        field: Callable,
        has_meta: bool,
        views_are_subclasses: bool = False,
    ):
        self.Trait = Trait
        self.field = field
        self.has_meta = has_meta
        self.views_are_subclasses = views_are_subclasses

    @property
    def name(self) -> str:
        return self.Trait.__name__


BACKENDS = [
    TraitBackend(Trait=Trait, field=field, has_meta=True),
]
try:
    import msgspec

    from buds.extras import msgspec as msgspec_impl

    BACKENDS.append(
        TraitBackend(
            Trait=msgspec_impl.MSGSpecTrait,
            field=msgspec.field,
            has_meta=False,
        )
    )
except ImportError:
    print("msgspec not installed")

try:
    import pydantic

    from buds.extras.pydantic import PydanticTrait

    BACKENDS.append(
        TraitBackend(
            Trait=PydanticTrait,
            field=pydantic.Field,
            has_meta=False,
        )
    )
except ImportError:
    print("msgspec not installed")


@pytest.fixture(params=BACKENDS, ids=lambda b: b.name)
def backend(request):
    return request.param


@pytest.fixture(autouse=True, scope="function")
def clear_caches():
    try:
        from buds.extras.numpy import dtypes, views

        dtypes.get_field_dtype.cache_clear()
        dtypes.get_trait_dtype.cache_clear()
        dtypes.get_dtype.cache_clear()
        views.resolve_class_attr.cache_clear()
        views._VIEW_CACHE.clear()
        views._VECTORIZED_TRAIT_VIEW_CACHE.clear()

    except ImportError:
        pass
        pass
