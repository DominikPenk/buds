# Tutorial — Getting started with Buds

This short tutorial walks through the minimal steps to define traits, create a world, spawn entities, and write a system that operates on those entities.

## Setup
****
Prefer the repository's conda environment named `ecs` when running commands.

```powershell
conda run -n ecs --no-capture-output python -m pip install -r requirements.txt  # optional
```

Run the test-suite to verify your environment:

```powershell
conda run -n ecs --no-capture-output python -m pytest -q
```

## Define Traits (components)

Declare trait types by subclassing `buds.Trait`:

```python
import buds

class Position(buds.Trait):
    x: float
    y: float

class Velocity(buds.Trait):
    vx: float
    vy: float
```

## Create a World and spawn entities

Use the provided `SparseWorld` for quick prototyping or `ArchetypeWorld` for iteration-heavy workloads.

```python
world = buds.SparseWorld()
player = world.create_entity(Position(0.0, 0.0), Velocity(1.0, 0.5))
```

## Write a system

Systems are regular functions decorated with `@buds.system`. The decorator introspects the function signature and type hints to determine which arguments to supply from the ECS runtime. The mapping rules are:

- If a parameter is annotated with `buds.World`, it receives the world instance.
- If a parameter is annotated with `buds.Entity`, it receives the current `Entity` being processed.
- If a parameter is annotated with a trait type (a class subclassing `buds.Trait`), the current entity's matching trait instance is passed.
- Any other parameters are treated as external/user arguments and must be supplied when calling the system.

The system may also be annotated with tags in the decorator to filter which entities it runs on, e.g. `@buds.system("active")` runs only on entities that have the `active` tag.

Examples:

1) Basic movement system (no external args):

```python
@buds.system
def move(world: buds.World, e: buds.Entity, pos: Position, vel: Velocity):
    pos.x += vel.vx
    pos.y += vel.vy

# Call the system: pass the world as the first argument
move(world)
```

2) System with external arguments (e.g., a delta time):

```python
@buds.system
def integrate(world: buds.World, pos: Position, vel: Velocity, dt: float):
    pos.x += vel.vx * dt
    pos.y += vel.vy * dt

# Call with the external argument after the world
integrate(world, 0.016)
```

3) Method-style systems (inside a class). When used as a method, the decorator recognizes `self` and expects the first argument after `self` to be the `World`:

```python
class Runner:
    @buds.system
    def tick(self, world: buds.World, pos: Position, vel: Velocity):
        pos.x += vel.vx

runner = Runner()
# Call using the instance then the world
runner.tick(world)
```

4) Using tags to filter entities:

```python
@buds.system("active", "renderable")
def render(world: buds.World, e: buds.Entity, pos: Position):
    # runs only for entities that have both 'active' and 'renderable' tags
    pass

render(world)
```

Notes and common pitfalls:

- Type hints are required for ECS-bound arguments (World, Entity, and trait types). The system introspection relies on concrete class hints to bind values.
- External arguments must be supplied positionally or as keyword arguments when calling the system; they are not provided by the world.
- The decorated system itself returns `None` (for `@system`) — use `@buds.map` if you want an iterator of results per entity.


## Inspect traits

Use `World.get_traits` to iterate trait instances across matching entities.

```python
for pos in world.get_traits(Position):
    print(pos)
```

## Next steps
- Explore `buds.itertools` for higher-order queries (combinations, permutations, products).
- Add new `World` implementations by implementing the `World` ABC in `buds/base.py` and adding the implementation to `tests/conftest.py` so it is covered by tests.

Happy hacking!
