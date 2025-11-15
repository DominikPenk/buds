"""System and functional decorators for ECS world processing.

This module provides decorators, which simplify
the creation of ECS systems by automatically iterating over entities and
their associated traits in a world.

Key features:
- Automatic introspection of system function parameters to determine
  entity, world, and trait dependencies.
- Supports external arguments in addition to ECS data.
- Provides both `system` (void-returning) and `map` (iterator-returning)
  variants for functional processing.
- Ensures type-safe mapping of traits and entities to system function arguments.

These decorators allow ECS logic to be expressed as simple Python functions,
while the framework handles iteration, filtering by traits and tags, and
argument mapping.

Exports:
    system
    map
"""

from collections.abc import Callable
import inspect
from functools import wraps
from typing import TypeVar, get_type_hints, Any, Concatenate, ParamSpec, overload
from dataclasses import dataclass
from collections.abc import Iterator

from .base import Entity, World, is_trait_type

__all__ = ["system", "map"]

Trait = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class SystemSignature:
    """Describes the parameter structure of an ECS system function.

    Attributes:
        entity_param: The name of the parameter representing an `Entity`, if any.
        world_param: The name of the parameter representing a `World`, if any.
        trait_params: Ordered list of parameter names corresponding to traits.
        trait_types: Tuple of trait types (classes) required by the system.
        external_params: Parameters that are not ECS traits or entities (user arguments).
        name: The original function name.
        is_method: Whether the system function is defined as an instance method.
    """

    entity_param: str | None
    world_param: str | None
    trait_params: list[str]
    trait_types: tuple[type[Trait], ...]
    external_params: list[str]
    name: str
    is_method: bool


def _introspect_system_signature(func: Callable, allow_none: bool) -> SystemSignature:
    """Analyze a function's signature to extract ECS-related parameters.

    This function inspects the callable's signature and type hints to
    determine which parameters correspond to `Entity`, `World`, and trait types.

    Args:
        func: The system function to analyze.
        allow_none: Whether to allow parameters without type hints.

    Returns:
        A `SystemSignature` instance describing the function.

    Raises:
        ValueError: If required type information is missing or multiple
            `Entity` or `World` parameters are detected.
    """
    sig = inspect.signature(func)
    types = get_type_hints(func)  # Use get_type_hints for better type resolution

    entity_param: str | None = None
    world_param: str | None = None
    trait_params: list[str] = []
    trait_types: list[type[Trait]] = []
    external_params: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        type_hint = types.get(param.name) or param.annotation

        # Ensure the parameter has a resolvable class type
        if type_hint is None and not allow_none:
            raise ValueError(
                f"Cannot determine class type for parameter '{name}' in '{func.__name__}'. "
                f"All ECS arguments (traits, Entity, World) must have explicit class type hints."
            )

        is_class = inspect.isclass(type_hint)

        # Separate utility parameters from trait parameters
        if is_class and issubclass(type_hint, Entity):
            if entity_param:
                raise ValueError(
                    f"System function '{func.__name__}' can only have one Entity parameter."
                )
            entity_param = name
        elif is_class and issubclass(type_hint, World):
            if world_param:
                raise ValueError(
                    f"System function '{func.__name__}' can only have one World parameter."
                )
            world_param = name
        elif is_trait_type(type_hint):
            trait_types.append(type_hint)
            trait_params.append(name)
        else:
            external_params.append(name)

    return SystemSignature(
        entity_param=entity_param,
        world_param=world_param,
        trait_params=trait_params,
        trait_types=tuple(trait_types),
        external_params=external_params,
        name=func.__name__,
        is_method="self" in sig.parameters,
    )


def _parse_wrapper_args(
    func_signature: SystemSignature, args, kwargs
) -> tuple[World, Any, Any]:
    """Parse wrapper arguments for a system or map decorator call.

    Determines which arguments correspond to the `World` instance and
    separates user-supplied external arguments from ECS-specific ones.

    Args:
        func_signature: The analyzed system function signature.
        args: Positional arguments passed to the wrapper.
        kwargs: Keyword arguments passed to the wrapper.

    Returns:
        A tuple of `(world, external_args, external_kwargs)`.

    Raises:
        TypeError: If too many positional arguments are provided.
    """
    max_num_positional = len(func_signature.external_params)
    max_num_positional += 2 if func_signature.is_method else 1
    if len(args) > max_num_positional:
        raise TypeError(
            f"{func_signature.name}() takes at most {max_num_positional} positional arguments "
            f"but {len(args)} were given."
        )

    if func_signature.is_method:
        func_args = [args[0]]
        world: World = args[1] if "world" not in kwargs else kwargs.pop("world")
        args = args[2:] if "world" not in kwargs else args[1:]
    else:
        func_args = []
        world: World = args[0] if args else kwargs.pop("world")
        args = args[1:] if args else args

    pos_external_kwargs = dict(zip(func_signature.external_params, args))
    func_kwargs = {**pos_external_kwargs, **kwargs}  # Merge them here

    return world, func_args, func_kwargs


def _map_kwargs(
    entity: Entity,
    world: World,
    trait_data: tuple[Trait, ...],
    system_signature: SystemSignature,
) -> dict[str, Any]:
    """Build keyword arguments to call the original system function.

    Args:
        entity: The current entity being processed.
        world: The world in which the system operates.
        trait_data: The tuple of traits associated with the entity.
        system_signature: The analyzed system function signature.

    Returns:
        A dictionary of keyword arguments matching the system’s parameters.
    """
    kwargs = {}

    if system_signature.entity_param:
        kwargs[system_signature.entity_param] = entity
    if system_signature.world_param:
        kwargs[system_signature.world_param] = world
    kwargs.update(
        {name: comp for name, comp in zip(system_signature.trait_params, trait_data)}
    )
    return kwargs


@overload
def system(
    *tags: str,
) -> Callable[[Callable[P, R]], Callable[Concatenate[World, ...], None]]: ...


@overload
def system(
    func: Callable[P, None],
) -> Callable[[Callable[P, R]], Callable[Concatenate[World, ...], None]]: ...


def system(
    *args: Any,
) -> Callable[[Callable[P, None]], Callable[Concatenate[World, P], None]] | Callable:
    """Decorator that defines an ECS system—a function operating on entities and traits.

    The `@system` decorator automatically iterates over all entities that match
    the function’s type-hinted traits and optional tags, calling the function for each.
    It is used for side-effect systems (e.g., movement, physics) that do not
    return a value per entity.

    The decorated function's signature must include a [`World`][buds.base.World]
    instance as its first argument (or the second if it is a method).

    Example (Direct Use):
        ```python
        @system
        def move(world: World, pos: Position, vel: Velocity):
            # world is required here
            pos.x += vel.dx
            # ...
        ```
    Example (With Tags):
        ```python
        @system("active", "renderable")
        def render_system(world: World, e: Entity, mesh: Mesh):
            # Only runs for entities with "active" AND "renderable" tags
            pass
        ```

    Args:
        *args: Either a callable (if used directly, e.g., `@system`) or one or more
            tags (strings) to filter entities (e.g., `@system("active")`).

    Returns:
        A decorated system function that can be invoked as `system(world, *external_args)`.
        It returns `None`.

    Raises:
        ValueError: If type hints for ECS components are missing or invalid
        TypeError: If too many or too few arguments are passed to the wrapper
    """

    def decorator(func: Callable) -> Callable:
        func_signature = _introspect_system_signature(func, allow_none=False)

        @wraps(func)
        def wrapper(*wrapper_args, **wrapper_kwargs) -> Iterator[R]:
            world, external_args, external_kwargs = _parse_wrapper_args(
                func_signature, wrapper_args, wrapper_kwargs
            )

            for entity, trait_data in world.get_entities(
                *func_signature.trait_types, tags=tags
            ):
                func_kwargs = _map_kwargs(entity, world, trait_data, func_signature)
                func(*external_args, **func_kwargs, **external_kwargs)

        return wrapper

    if args and callable(args[0]):
        tags: set[str] = None
        return decorator(args[0])
    else:
        tags = set(args)
        return decorator


@overload
def map(
    *tags: str,
) -> Callable[[Callable[P, R]], Callable[Concatenate[World, ...], Iterator[R]]]: ...


@overload
def map(
    func: Callable[P, None],
) -> Callable[[Callable[P, R]], Callable[Concatenate[World, ...], Iterator[R]]]: ...


def map(
    *args: Any,
) -> (
    Callable[[Callable[P, R]], Callable[Concatenate[World, P], Iterator[R]]] | Callable
):
    """Decorator similar to `system`, but yields results for each entity.

    The `@map` decorator allows defining ECS functions that return a value per entity,
    enabling collection or transformation of results in a functional style.

    The decorated function's signature must include a [`World`][buds.base.World]
    instance as its first argument (or the second if it is a method).

    Example:
        ```python
        @map
        def compute_energy(world: World, mass: Mass, vel: Velocity) -> float:
            return 0.5 * mass.value * (vel.dx**2 + vel.dy**2)

        # Returns an iterator of floats
        energies = compute_energy(world)
        ```

    Args:
        *args: Either a callable (if used directly, e.g., `@map`) or one or more
            tags (strings) to filter entities (e.g., `@map("visible")`).

    Returns:
        A decorated function that accepts a [`World`][buds.base.World] and external
        arguments, and yields one result (`R`) per matched entity.

    Raises:
        ValueError: If type hints for ECS components are missing or invalid
        TypeError: If too many or too few arguments are passed to the wrapper
    """

    def decorator(func: Callable) -> Callable:
        func_signature = _introspect_system_signature(func, allow_none=False)

        @wraps(func)
        def wrapper(*wrapper_args, **wrapper_kwargs) -> Iterator[R]:
            world, external_args, external_kwargs = _parse_wrapper_args(
                func_signature, wrapper_args, wrapper_kwargs
            )

            for entity, trait_data in world.get_entities(
                *func_signature.trait_types, tags=tags
            ):
                func_kwargs = _map_kwargs(entity, world, trait_data, func_signature)
                yield func(*external_args, **func_kwargs, **external_kwargs)

        return wrapper

    if args and callable(args[0]):
        tags: set[str] = None
        return decorator(args[0])
    else:
        tags = set(args)
        return decorator
