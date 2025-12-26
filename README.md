# 🌱 Buds — A Pythonic Entity Component System

**Buds** is a lightweight, **Entity-Component-System (ECS)** framework implemented purely in Python.

## Features

- **Pure Python ECS** — clean, minimal, and dependency-free.
- **Traits** — data-first components defined with modern Python type hints.
- **Worlds** — Buds ships a sparse and archetype-based entity storage and is set up to make adding your own implementations easy.
- **Systems** — implemented as decorators for defining entity-processing functions.
- **Composable Itertools** — functional utilities for grouping and filtering entities and traits.
- **Explicit** — powered by Python type hints and runtime introspection.

## Example

```python
import buds

# --- Define Traits (Components) ---
class Position(buds.Trait):
    x: float
    y: float

class Velocity(buds.Trait):
    vx: float
    vy: float

# --- Define a System ---
@buds.system
def move(world: buds.World, entity: buds.Entity, position: Position, velocity: Velocity):
    position.x += velocity.vx
    position.y += velocity.vy

# --- Create a World and create Entities ---
world = buds.SparseWorld()
player = world.create_entity(Position(0, 0), Velocity(1, 1))

# --- Run a System ---
for _ in range(3):
    move(world)

```

## Core Concepts

### Entities

Lightweight identifiers that reference a collection of traits.
They contain no behavior — just identity.

### Traits

Traits (also known as components) hold data.
Buds uses dataclasses to implement them; declare trait types by subclassing `buds.Trait`.

### Systems

Systems are decorated functions that operate on entities with matching traits.
They can mutate traits, query entities, or yield data.

### Worlds

A World manages entities and their traits.
Buds provides two implementations:

[SparseWorld](buds/sparse.py) — flexible, dictionary-based (great for general use).

[ArchetypeWorld](buds/archetype.py) — data-oriented storage with dense archetypes (optimized for iteration and lookup speed).

## Buds Itertools

Buds's [`itertools`](buds/itertools.py) module provides higher-level helpers built on Python's [`itertools`](https://docs.python.org/3/library/itertools.html).

For example, you can iterate over all (ordered) pairs of entities containing a set of specific traits:

```python
import math
import buds

class Position(buds.Trait):
    x: float
    y: float

world = buds.SparseWorld()

# Create entities
world.create_entity(Position(0, 0))
world.create_entity(Position(1, 3))
world.create_entity(Position(3, 4))

for (p1,), (p2,) in buds.itertools.trait_combinations(world, 2, Position):
    print(f"Position of first entity:  {p1}")
    print(f"Position of second entity: {p2}")
    dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
    print(f"Distance: {dist:2f}")
```

## Inspect Module

`Buds` includes an inspection module ([buds.inspect](buds/inspect.py)) for trait schema inspection and normalization.

Unlike systems or worlds, this module operates purely on trait types, not on entities or runtime ECS state. 
Its purpose is to extract structured metadata from trait definitions, including field types, defaults, annotations, and class-level configuration.

This makes it a foundational building block for tooling, serialization, validation, and code generation.

What the Inspect Module Does
- The inspect module provides utilities to:
- Normalize trait fields into a unified schema representation
- Extract metadata from trait classes and instances
- Support adapters for alternative trait implementations (e.g. dataclasses, Pydantic, msgspec)

Enable third-party integrations without coupling them to ECS internals

Internally, Buds uses adapters to translate different trait backends into a consistent schema model.

```python
from buds.inspect import inspect_trait
import buds

class Health(buds.Trait):
    hp: int = 100
    regen: float

schema = inspect_trait(Health)
print("Trait:", schema.type.__name__)
for field in schema.fields:
    print(f"- {field.name} : {field.type}, required={field.spec.required}")
```

This prints a description of the trait’s schema, including field names, types, and whether a default was provided.


## 3rd Party Integration

`Buds’` core implementation is intentionally **dependency-free**.  
However, Buds is designed to integrate cleanly with external libraries through its **trait inspection and adapter system**.

All optional integrations live under [`buds.extras`](buds/extras/) and build on top of the same trait schema abstraction used by Buds internally. This allows third-party trait implementations to behave like native Buds traits, without coupling the ECS core to external dependencies.

### Numpy Integration

NumPy is one of the most widely used libraries in the Python ecosystem.  
`Buds` provides a [`NumpyArchetypeWorld`](buds/extras/numpy/numpy_archetype.py) that uses **NumPy structured arrays** for memory-efficient, data-oriented storage.

This world implementation also supports **vectorized trait access**, making it ideal for large-scale numerical simulations or batch-oriented systems.

```python
from typing import Annotated

import numpy as np

import buds
from buds.extras import NumpyArchetypeWorld, NumpyArrayMetadata


class Particle(buds.Trait):
    pos: Annotated[np.ndarray, NumpyArrayMetadata((2,))]
    vel: Annotated[np.ndarray, NumpyArrayMetadata((2,))]


world = NumpyArchetypeWorld()
for i in range(100):
    world.create_entity(
        Particle(
            np.random.uniform(-1, 1, (2,)),
            np.random.uniform(-1, 1, (2,)),
        )
    )

(all_particles,) = world.get_vectorized_traits(Particle)
print(all_particles.pos.shape)  # (100, 2)
print(all_particles.vel.shape)  # (100, 2)
```

### Pydantic Integration

With [`buds.extras.pydantic`](buds/extras/pydantic/trait.py), traits can be defined using
[Pydantic’s BaseModel](https://docs.pydantic.dev/latest/api/base_model/).

This allows you to leverage runtime validation, serialization, and schema generation while keeping full compatibility with `Buds’` ECS model.

```python
import buds
from buds.extras import PydanticTrait


class User(PydanticTrait):
    name: str
    age: int


world = buds.SparseWorld()
world.create_entity(User(name="Max Mustermann", age=42))
world.create_entity(User(name="Peter Lustig", age=64))
world.create_entity(User(name="Peter Lustig", age=64))
```

### msgspec Integration

With [`buds.extras.msgspec`](buds/extras/msgspec/trait.py), Buds supports traits based on [msgspec.Struct](https://jcristharif.com/msgspec/).

```python
import buds
from buds.extras import MSGSpecTrait


class User(MSGSpecTrait):
    name: str
    age: int


world = buds.SparseWorld()
world.create_entity(User(name="Max Mustermann", age=42))
world.create_entity(User(name="Peter Lustig", age=64))
world.create_entity(User(name="Peter Lustig", age=64))
```