# 🌱 Buds — A Pythonic Entity Component System

**Buds** is a lightweight, **Entity-Component-System (ECS)** framework implemented purely in python.  

## Features

- **Pure Python ECS** — clean, minimal, and dependency-free.
- **Traits** — data-first components defined with modern Python type hints.
- **Worlds** — Buds shipes a sparse and archetype-based entity storage and is set up to easily add you own implementations
- **Systems** — are implemented as decorators for defining entity-processing functions.
- **Composable Itertools** — functional utilities for grouping, filtering, entities and traits.
- **Explicit** — powered by Python type hints and runtime introspection.

## Example

```python
import buds

# --- Define Traits (Components) ---
@buds.trait
class Position:
    x: float
    y: float

class Velocity(buds.Trait):
    vx: float
    vy: float

# --- Define a System ---
@buds.system
def move(entity: buds.Entity, position: Position, velocity: Velocity):
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
Buds uses dataclasses to implement them and provides two ways to declare them

decorator — instead of declaring your dataclass with @dataclass use @buds.trait

inheritance — derive the trait class from buds.Trait

### Systems

Systems are decorated functions that operate on entities with matching traits.
They can mutate traits, query entities, or yield data.

### Worlds

A World manages entities and their traits.
Buds provides two implementations:

[SparseWorld](buds/sparse.py) — flexible, dictionary-based (great for general use).

[ArchetypeWorld](buds/archetype.py) — data-oriented storage with dense archetypes (optimized for iteration and lookup speed).

## Buds Itertools

Buds's [`itertools`](buds/itertools.py) moule provides a thin wrapper around pythons [`intertools`](https://docs.python.org/3/library/itertools.html). 

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
    dist = math.hypot(p1.x - p2.x, p1.x - p2.y)
    print(f"Distance: {dist:2f}")
```