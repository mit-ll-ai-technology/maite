"""Dynamic Pytest suite verifying protocol_members against the MAITE library."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, get_args, get_origin

import pytest
from typing_extensions import TypeForm, is_typeddict

from maite import protocols as mp
from maite._internals.spotcheck.protocol_introspection import protocol_members
from maite.protocols import image_classification as ic
from maite.protocols import object_detection as od

MODULES: Mapping[str, object] = {"generic": mp, "ic": ic, "od": od}
COMPONENT_PROTOCOL_NAMES: tuple[str, ...] = (
    "Dataset",
    "DataLoader",
    "Model",
    "Augmentation",
    "Metric",
)

# Safely extract protocols that actually exist in the target modules
ALL_COMPONENT_PROTOCOLS: list[tuple[str, str, TypeForm[object]]] = [
    (mod_label, proto_name, getattr(module, proto_name))
    for mod_label, module in MODULES.items()
    for proto_name in COMPONENT_PROTOCOL_NAMES
    if hasattr(module, proto_name)
]

ALL_IDS: list[str] = [f"{m}.{p}" for m, p, _ in ALL_COMPONENT_PROTOCOLS]

EXPECTED_ATTRS: dict[str, set[str]] = {
    "Dataset": {"metadata"},
    "DataLoader": set(),
    "Model": {"metadata"},
    "Augmentation": {"metadata"},
    "Metric": {"metadata"},
}

EXPECTED_METHODS: dict[str, set[str]] = {
    "Dataset": {"__getitem__", "__len__"},
    "DataLoader": {"__iter__"},
    "Model": {"__call__"},
    "Augmentation": {"__call__"},
    "Metric": {"compute", "reset", "update"},
}


@pytest.mark.parametrize(
    ("name", "proto"),
    [(name, proto) for _, name, proto in ALL_COMPONENT_PROTOCOLS],
    ids=ALL_IDS,
)
def test_expected_member_shape(name: str, proto: TypeForm[object]) -> None:
    """Verify that extracted members from MAITE protocols match the expected blueprint."""
    m = protocol_members(proto)

    assert set(m.attrs) == EXPECTED_ATTRS[name]
    assert set(m.methods) == EXPECTED_METHODS[name]


@pytest.mark.parametrize(
    "proto",
    [proto for _, _, proto in ALL_COMPONENT_PROTOCOLS],
    ids=ALL_IDS,
)
def test_machinery_excluded(proto: TypeForm[object]) -> None:
    """Verify protocol machinery is not reported via protocol_members."""
    m = protocol_members(proto)

    for junk in (
        "_is_protocol",
        "_abc_impl",
        "__init__",
        "__subclasshook__",
        "__protocol_attrs__",
        "__parameters__",
        "__orig_bases__",
    ):
        assert junk not in m.names


@pytest.mark.parametrize(
    "proto",
    [proto for _, _, proto in ALL_COMPONENT_PROTOCOLS],
    ids=ALL_IDS,
)
def test_no_string_annotations_survive(proto: TypeForm[object]) -> None:
    """Verify that postponed annotations (strings) are fully resolved into actual types."""
    m = protocol_members(proto)
    for hint in m.attrs.values():
        assert not isinstance(hint, str)

    for sig in m.methods.values():
        for p in sig.parameters.values():
            assert not isinstance(p.annotation, str)
        assert not isinstance(sig.return_annotation, str)


# Just look at metadata to confirm that we seem to be capturing type information properly
# (not considering checking types for protocols inhereiting from parametrized generics yet)

METADATA_HINTS: dict[str, TypeForm[Any]] = {
    "Dataset": mp.DatasetMetadata,
    "Model": mp.ModelMetadata,
    "Augmentation": mp.AugmentationMetadata,
    "Metric": mp.MetricMetadata,
}


@pytest.mark.parametrize("name", sorted(METADATA_HINTS))
@pytest.mark.parametrize("mod", sorted(MODULES))
def test_metadata_is_typeddict_attr(mod: str, name: str) -> None:
    proto = getattr(MODULES[mod], name)
    m = protocol_members(proto)
    hint = m.attrs["metadata"]
    assert hint is METADATA_HINTS[name]
    assert hint is not None
    assert is_typeddict(hint)

    assert "id" in hint.__required_keys__  # pyright: ignore[reportAttributeAccessIssue], last assert guarantees 'hint' is TypedDict
    assert "metadata" not in m.methods


def test_metric_signature_detail() -> None:

    m = protocol_members(mp.Metric)

    # check `protocol_members` got return type right
    assert m.methods["reset"].return_annotation is type(None)
    assert m.methods["update"].return_annotation is type(None)

    # check that return type is right with parametrized generic Mapping[str,Any]
    compute_ret = m.methods["compute"].return_annotation
    assert get_origin(compute_ret) is Mapping
    assert get_args(compute_ret) == (str, Any)
