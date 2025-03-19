import torch

from olmo_core.nn.conversion.state_converter import StateConverter
from olmo_core.nn.conversion.state_mapping import (
    StateMappingTemplate,
    TemplatePlaceholder,
)

EXPERT = TemplatePlaceholder.EXPERT
LAYER = TemplatePlaceholder.LAYER


def test_convert_one_to_one():
    mapping_template = StateMappingTemplate("a", "b")
    converter = StateConverter([mapping_template])
    state_dict = {"a": torch.tensor([1.0, 2.0])}

    converted_state_dict = converter.convert(state_dict, {})

    assert "b" in converted_state_dict
    torch.testing.assert_close(state_dict["a"], converted_state_dict["b"])


def test_convert_source_concat():
    mapping_template = StateMappingTemplate(("a1", "a2"), "b", source_concat_dim=1)
    converter = StateConverter([mapping_template])
    state_dict = {"a1": torch.tensor([[1.0, 2.0]]), "a2": torch.tensor([[3.0, 4.0]])}

    converted_state_dict = converter.convert(state_dict, {})

    assert "b" in converted_state_dict
    torch.testing.assert_close(
        torch.cat([state_dict["a1"], state_dict["a2"]], dim=1), converted_state_dict["b"]
    )


def test_convert_source_placeholder_expansion():
    mapping_template = StateMappingTemplate(
        f"a{LAYER}", "b", source_key_per_placeholder=LAYER, source_concat_dim=0
    )
    converter = StateConverter([mapping_template])
    state_dict = {
        "a0": torch.tensor([[1.0, 2.0]]),
        "a1": torch.tensor([[3.0, 4.0]]),
        "a2": torch.tensor([[5.0, 6.0]]),
    }

    converted_state_dict = converter.convert(state_dict, {LAYER: 3})

    assert "b" in converted_state_dict
    torch.testing.assert_close(
        torch.cat([state_dict["a0"], state_dict["a1"], state_dict["a2"]], dim=0),
        converted_state_dict["b"],
    )


def test_convert_dest_chunk():
    mapping_template = StateMappingTemplate("a", ("b1", "b2"), dest_chunk_dim=1)
    converter = StateConverter([mapping_template])
    state_dict = {"a": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

    converted_state_dict = converter.convert(state_dict, {})

    assert "b1" in converted_state_dict
    assert "b2" in converted_state_dict
    chunks = torch.chunk(state_dict["a"], dim=1, chunks=2)
    torch.testing.assert_close(chunks[0], converted_state_dict["b1"])
    torch.testing.assert_close(chunks[1], converted_state_dict["b2"])


def test_convert_dest_placeholder_expansion():
    mapping_template = StateMappingTemplate(
        "a", f"b{EXPERT}", dest_key_per_placeholder=EXPERT, dest_chunk_dim=0
    )
    converter = StateConverter([mapping_template])
    state_dict = {"a": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])}

    converted_state_dict = converter.convert(state_dict, {EXPERT: 3})

    assert "b0" in converted_state_dict
    assert "b1" in converted_state_dict
    assert "b2" in converted_state_dict
    chunks = torch.chunk(state_dict["a"], dim=0, chunks=3)
    torch.testing.assert_close(chunks[0], converted_state_dict["b0"])
    torch.testing.assert_close(chunks[1], converted_state_dict["b1"])
    torch.testing.assert_close(chunks[2], converted_state_dict["b2"])


def test_convert_permute():
    mapping_template = StateMappingTemplate("a", "b", dims_permutation=(1, 2, 0))
    converter = StateConverter([mapping_template])
    state_dict = {"a": torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])}

    converted_state_dict = converter.convert(state_dict, {})

    assert "b" in converted_state_dict
    torch.testing.assert_close(state_dict["a"].permute(1, 2, 0), converted_state_dict["b"])


def test_convert_flatten():
    mapping_template = StateMappingTemplate("a", "b", flatten_dims=(0, 1))
    converter = StateConverter([mapping_template])
    state_dict = {"a": torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])}

    converted_state_dict = converter.convert(state_dict, {})

    assert "b" in converted_state_dict
    torch.testing.assert_close(state_dict["a"].flatten(0, 1), converted_state_dict["b"])


def test_convert_unflatten():
    mapping_template = StateMappingTemplate("a", "b", unflatten_dim=(0, (4, -1)))
    converter = StateConverter([mapping_template])
    state_dict = {"a": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])}

    converted_state_dict = converter.convert(state_dict, {})

    assert "b" in converted_state_dict
    torch.testing.assert_close(state_dict["a"].unflatten(0, (4, -1)), converted_state_dict["b"])
