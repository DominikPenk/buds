# Implementing a Custom Trait Class

This tutorial explains how to set up a custom Trait Class to work nicely with any world implementation.

We use **Pydantic `BaseModel`** as a concrete example, but the same steps apply to any custom modeling system.

## Overview: What “Supporting a Trait Type” Means

To support a new trait type, you must provide **three integration points**:

1. **A base trait type**: The new base for Traits.
2. **A Trait inspection adapter**: To ensure inspection works with your Trait Class.
3. **A View adapter**: Optional, only required if you intend to use `NumpyArchetypeWorld`.

Each step is independent but builds on the previous one.

## Step 1: Create and Register the Base Trait Type

A *trait type* is any class that:
- Represents structured data
- Has inspectable fields
- Can be projected into views

For Pydantic:

```python
from pydantic import BaseModel
from buds import Trait, register_trait_base_class

class PydanticTrait(BaseModel):
    """Base class for all Pydantic-based traits."""
    pass

register_trait_base_class(PydanticTrait)
```

As you can see, we have to register the new base class with buds so the system knows that any subclass of this type is a Trait.
> ⚠️ Without this step, nothing else works.

## Step 2: Implementing Inspect Logic

`buds` relies on type annotations and metadata to set up trait storage and systems.
We need to tell `buds` how to extract this information. `buds` ships with a custom inspect module based on adapters.

### 2.1 Understanding Schemas

The core of the inspect module are the two schemas that define the structural description of any Trait:
```python
class TraitSchema:
    type: type[Any]                     # The type of the trait
    fields: list[FieldSchema]           # Metadata of instance-level fields
    class_fields: list[FieldSchema]     # Metadata of class-level fields

class FieldSchema:
    name: str                           # Field name
    type: str                           # Pure python type of the field
    metadata                            # metadata (potentially from Annotation or field data)
    spec:
        default: Any | MISSING                          
        default_factory: Callable[[], Any] | MISSING  
        required: bool                  # Wether the field must be provided during initialization or not
```

### 2.2 Implementing the TraitAdapter

The `TraitAdapter` tells `buds` how to find fields in your trait class:

```python
class TraitAdapter(Protocol):
    def is_applicable(self, trait_type: type) -> bool: ...
    def build_fields(self, trait_type: type) -> list[FieldSchema]: ...
    def build_class_fields(self, trait_type: type) -> list[FieldSchema]: ...
```

For our `PydanticTrait` class:

```python
import inspect

class PydanticTraitAdapter:
    def is_applicable(self, trait_type: type[Any]) -> bool:
        return issubclass(trait_type, BaseModel)

    def build_fields(self, trait_type: BaseModel) -> list[FieldSchema]:
        fields: dict[str, FieldInfo] = trait_type.model_fields
        return [FieldSchema.create(name, f.annotation, f) for name, f in fields.items()]

    def build_class_fields(self, trait_type: type[Any]) -> list[FieldSchema]:
        hints = get_type_hints(trait_type, include_extras=True)
        return [
            FieldSchema.create(name, type_, None)
            for name, type_ in hints.items()
            if get_origin(type_) is ClassVar
        ]
```
> Pydantic stores instance fields in model_fields.
> 
> Class-level fields (annotated with ClassVar) are not handled by Pydantic, so we collect them manually.

### 2.2. Implementing the FieldAdapter

The default `FieldAdapter` handles most field metadata, including `Annotated[...]` and raw types.
If you want to extract additional information from `pydantic.Field`, implement a custom `FieldAdapter`:

```python
from dataclass import MISSING

class FieldAdapter(Protocol):
    def is_applicable(self, annotation: type[Any], source: Any) -> bool: ...
    def build_schema(self, name: str, annotation: type[Any], source: Any) -> FieldSchema:
``` 

```python
class PydanticFieldAdapter(DefaultFieldAdapter):
    def is_applicable(self, annotation: type[Any], source: Any) -> bool:
        return isinstance(source, FieldInfo)

    def build_schema(
        self, name: str, annotation: type[Any], source: FieldInfo
    ) -> FieldSchema:
        result = super().build_schema(name, annotation, source)
        default_factory = (
            MISSING if source.default_factory is None else source.default_factory
        )
        field_spec = FieldSpec(
            default=source.default,
            default_factory=default_factory,  # type: ignore
            required=source.is_required(),
        )
        return FieldSchema(result.name, result.type, result.metadata, field_spec)
```

In this example, we simply let the `DefaultFieldAdapter` do the heavy lifting and simply update the field specification with data from the *source* `pydantic.Field`.

### 2.3 Register Adapters

```python
TraitSchema.register_adapter(PydanticTraitAdapter())
FieldSchema.register_adapter(PydanticFieldAdapter())
```

Our new Trait Class is now fully integrated into the inspect module.

## Step 3: Implement a View Adapter (Optional)

The `NumpyArchetypeWorld` uses lightweight views onto underlying storage for single instances of traits.
We must implement a view adapter to tell `buds` how to construct views from a `TraitSchema`.

For most trait classes, the `ViewBuilder` default configuration works:

```python
from buds.extras.numpy.views import ViewBuilder, register_view_adapter

class PydanticViewGenerator:
    @classmethod
    def create_view(cls, schema: TraitSchema, name: str) -> Optional[type]:
        if not issubclass(schema.type, pydantic.BaseModel):
            return None

        view_cls = (
            PydanticViewbuilder(schema, name)
            .add_defaults()
            .build()
        )
        return view_cls
```
Register the view adapter:
```python
register_view_adapter(PydanticStructViewGenerator, priority=10)
```
The Trait Class is now fully compatible with NumpyArchetypeWorld.
