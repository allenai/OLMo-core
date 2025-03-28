from typing import Dict

import pytest

from olmo_core.nn.conversion.state_mapping import (
    StateMappingTemplate,
    TemplatePlaceholder,
)

EXPERT = TemplatePlaceholder.EXPERT
LAYER = TemplatePlaceholder.LAYER


def test_template_to_mapping_one_to_one():
    mapping_template = StateMappingTemplate("a", "b")
    mapping = mapping_template.to_mapping(placeholder_values={}, placeholder_bounds={})
    assert mapping is not None
    assert mapping.source_keys == ("a",)
    assert mapping.dest_keys == ("b",)


def test_template_to_mapping_one_to_one_with_place_holders():
    mapping_template = StateMappingTemplate(f"a.{LAYER}", f"b.{LAYER}")
    placeholder_values: Dict[TemplatePlaceholder, int | None] = {LAYER: 1}
    placeholder_bounds = {LAYER: 2}
    mapping = mapping_template.to_mapping(placeholder_values, placeholder_bounds)
    assert mapping is not None
    assert mapping.source_keys == ("a.1",)
    assert mapping.dest_keys == ("b.1",)


def test_template_to_mapping_many_to_many():
    mapping_template = StateMappingTemplate(("a1", "a2", "a3"), ("b1", "b2"))
    mapping = mapping_template.to_mapping(placeholder_values={}, placeholder_bounds={})
    assert mapping is not None
    assert mapping.source_keys == ("a1", "a2", "a3")
    assert mapping.dest_keys == ("b1", "b2")


def test_template_to_mapping_with_source_placeholder_expansion():
    mapping_template = StateMappingTemplate(
        f"a.{EXPERT}", ("b.0", "b.1"), source_key_per_placeholder=EXPERT
    )
    placeholder_bounds = {EXPERT: 3}
    mapping = mapping_template.to_mapping(
        placeholder_values={}, placeholder_bounds=placeholder_bounds
    )

    assert mapping is not None
    assert mapping.source_keys == ("a.0", "a.1", "a.2")
    assert mapping.dest_keys == ("b.0", "b.1")


def test_template_to_mapping_with_dest_placeholder_expansion():
    mapping_template = StateMappingTemplate(
        ("a.0", "a.1"), f"b.{LAYER}", dest_key_per_placeholder=LAYER
    )
    placeholder_bounds = {LAYER: 3}
    mapping = mapping_template.to_mapping(
        placeholder_values={}, placeholder_bounds=placeholder_bounds
    )

    assert mapping is not None
    assert mapping.source_keys == ("a.0", "a.1")
    assert mapping.dest_keys == ("b.0", "b.1", "b.2")


@pytest.mark.parametrize(
    "source_templates, source_key_per_placeholder",
    [
        pytest.param(f"a.{EXPERT}", LAYER, id="wrong-placeholder"),
        pytest.param("a", LAYER, id="missing-placeholder"),
    ],
)
def test_template_to_mapping_invalid_source_template(source_templates, source_key_per_placeholder):
    mapping_template = StateMappingTemplate(
        source_templates, "b", source_key_per_placeholder=source_key_per_placeholder
    )
    with pytest.raises(ValueError):
        mapping_template.to_mapping({}, {LAYER: 3})


@pytest.mark.parametrize(
    "dest_templates, dest_key_per_placeholder",
    [
        pytest.param(f"b.{EXPERT}", LAYER, id="wrong-placeholder"),
        pytest.param("b", LAYER, id="missing-placeholder"),
    ],
)
def test_template_to_mapping_invalid_dest_template(dest_templates, dest_key_per_placeholder):
    mapping_template = StateMappingTemplate(
        "a", dest_templates, dest_key_per_placeholder=dest_key_per_placeholder
    )
    with pytest.raises(ValueError):
        mapping_template.to_mapping({}, {LAYER: 3})


@pytest.mark.parametrize(
    "source_templates, source_key_per_placeholder, placeholder_values, placeholder_bounds",
    [
        pytest.param(f"a.{LAYER}", None, {}, {LAYER: 3}, id="missing-layer-value"),
        pytest.param(f"a.{LAYER}", LAYER, {LAYER: 2}, {LAYER: 3}, id="unexpected-layer-value"),
        pytest.param(f"a.{LAYER}", None, {}, {}, id="missing-layer-value-and-bound"),
        pytest.param(f"a.{LAYER}", LAYER, {}, {}, id="missing-layer-bound"),
    ],
)
def test_template_to_mapping_invalid_source_values(
    source_templates, source_key_per_placeholder, placeholder_values, placeholder_bounds
):
    mapping_template = StateMappingTemplate(
        source_templates, "b", source_key_per_placeholder=source_key_per_placeholder
    )
    mapping = mapping_template.to_mapping(placeholder_values, placeholder_bounds)

    assert mapping is None


@pytest.mark.parametrize(
    "dest_templates, dest_key_per_placeholder, placeholder_values, placeholder_bounds",
    [
        pytest.param(f"b.{LAYER}", None, {}, {LAYER: 3}, id="missing-layer-value"),
        pytest.param(f"b.{LAYER}", LAYER, {LAYER: 2}, {LAYER: 3}, id="unexpected-layer-value"),
        pytest.param(f"b.{LAYER}", None, {}, {}, id="missing-layer-value-and-bound"),
        pytest.param(f"b.{LAYER}", LAYER, {}, {}, id="missing-layer-bound"),
    ],
)
def test_template_to_mapping_invalid_dest_values(
    dest_templates, dest_key_per_placeholder, placeholder_values, placeholder_bounds
):
    mapping_template = StateMappingTemplate(
        "a", dest_templates, dest_key_per_placeholder=dest_key_per_placeholder
    )
    mapping = mapping_template.to_mapping(placeholder_values, placeholder_bounds)

    assert mapping is None
