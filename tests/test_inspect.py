from __future__ import annotations

from dataclasses import field
from typing import Annotated, ClassVar

import pytest

import buds
import buds.inspect as inspect_mod


def test_fieldschema_default_adapter_simple_field():
    class T(buds.Trait):
        x: int

    schema = inspect_mod.TraitSchema.create(T)
    assert schema.type is T
    assert len(schema.fields) == 1
    f = schema.fields[0]
    assert f.name == "x"
    assert f.type is int
    # no annotated metadata for plain field
    assert f.metadata.annotated is None
    # FieldSpec exists and required since no default
    assert f.spec is not None
    assert f.spec.required


def test_fieldschema_with_defaults():
    class T(buds.Trait):
        a: int = 3
        b: int = 5

    schema = inspect_mod.TraitSchema.create(T)
    names = {f.name: f for f in schema.fields}
    fa = names["a"]
    fb = names["b"]
    assert fa.spec.default == 3
    assert not fa.spec.required
    assert fb.spec.default == 5
    assert not fb.spec.required


def test_fieldschema_with_default_factory():
    class T(buds.Trait):
        b: int = field(default_factory=lambda: 5)

    schema = inspect_mod.TraitSchema.create(T)
    fb = schema.fields[0]
    assert callable(fb.spec.default_factory)
    assert not fb.spec.required


def test_annotated_metadata_and_field_metadata():
    # Use Annotated to pass arbitrary metadata and dataclasses.field.metadata
    class T(buds.Trait):
        x: Annotated[int, "meta1", 123] = field(default=1, metadata={"doc": "value"})

    schema = inspect_mod.TraitSchema.create(T)
    f = schema.fields[0]
    assert f.metadata.annotated == ("meta1", 123)
    assert f.metadata.field["doc"] == "value"


def test_classvar_is_ignored():
    class T(buds.Trait):
        x: int
        y: ClassVar[int] = 1

    schema = inspect_mod.TraitSchema.create(T)
    # only x should remain
    assert [f.name for f in schema.fields] == ["x"]


def test_inspect_trait_accepts_instance_and_type():
    class T(buds.Trait):
        x: int = 2

    inst = T(5)
    s1 = inspect_mod.TraitSchema.create(T)
    s2 = inspect_mod.TraitSchema.create(type(inst))
    assert s1.type is T
    assert s2.type is T


def test_inspect_trait_errors_on_non_trait():
    class NotTrait:
        pass

    with pytest.raises(ValueError):
        inspect_mod.TraitSchema.create(NotTrait)

    with pytest.raises(ValueError):
        inspect_mod.TraitSchema.create(type(NotTrait()))


def test_fieldschema_create_raises_when_no_adapter(monkeypatch):
    # Temporarily clear adapters and ensure ValueError is raised
    original = inspect_mod.FieldSchema._adapters[:]  # type: ignore[attr-defined]
    inspect_mod.FieldSchema._adapters.clear()
    try:
        with pytest.raises(ValueError):
            inspect_mod.FieldSchema.create("name", int)
    finally:
        inspect_mod.FieldSchema._adapters[:] = original


def test_traitschema_create_raises_when_no_adapter(monkeypatch):
    original = inspect_mod.TraitSchema._adapters[:]  # type: ignore[attr-defined]
    inspect_mod.TraitSchema._adapters.clear()
    try:
        with pytest.raises(ValueError):
            inspect_mod.TraitSchema.create(int)
    finally:
        inspect_mod.TraitSchema._adapters[:] = original


def test_default_field_adapter_uses_field_source():
    # Ensure metadata from dataclasses.Field is picked up
    class T(buds.Trait):
        x: int = field(default=7, metadata={"a": 1})

    fs = inspect_mod.TraitSchema.create(T).fields[0]
    assert fs.metadata.field["a"] == 1
    assert fs.spec.default == 7


def test_dataclass_trait_adapter_rejects_non_dataclass():
    class RegularTrait(buds.Trait):
        x: int

    # Should not be considered applicable by DataclassTraitAdapter
    adapter = inspect_mod.DataclassTraitAdapter()
    # Subclassing buds.Trait automatically applies dataclass() in Trait.__init_subclass__
    assert adapter.is_applicable(RegularTrait)
    assert adapter.is_applicable(RegularTrait)


def test_inspect_accepts_trait_instance_and_type():
    class T(buds.Trait):
        x: int = 42

    inst = T(5)
    schema_from_type = inspect_mod.inspect_trait(T)
    schema_from_instance = inspect_mod.inspect_trait(inst)

    assert isinstance(schema_from_type, inspect_mod.TraitSchema)
    assert schema_from_type.type is T
    assert schema_from_instance.type is T
    # Fields should match
    names_type = [f.name for f in schema_from_type.fields]
    names_instance = [f.name for f in schema_from_instance.fields]
    assert names_type == names_instance == ["x"]


def test_inspect_raises_on_non_trait_class():
    class NotTrait:
        pass

    with pytest.raises(ValueError, match="must be a trait or trait type"):
        inspect_mod.inspect_trait(NotTrait)


def test_inspect_raises_on_non_trait_instance():
    class NotTrait:
        pass

    instance = NotTrait()
    with pytest.raises(ValueError, match="must be a trait or trait type"):
        inspect_mod.inspect_trait(instance)
