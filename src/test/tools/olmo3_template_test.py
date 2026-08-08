"""
Checks the tool format against the chat template the OLMo 3 instruct models were tuned with.

The rest of the suite exercises the format as this package understands it. These tests pin that
understanding to the real template, so a mismatch shows up here rather than as a model that
quietly never calls anything.
"""

import pytest

from olmo_core.tools import (
    CalculatorToolConfig,
    ToolRegistry,
    parse_function_calls,
    render_environment_message,
    resolve_tool_stop_token_ids,
    resolve_turn_end_token_ids,
)

TOKENIZER = "allenai/Olmo-3-7B-Instruct"

CLOSE_FUNCTION_CALLS_TOKEN_ID = 100269
TURN_END_TOKEN_ID = 100265


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers", reason="Requires transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    except Exception as e:
        pytest.skip(f"Could not download '{TOKENIZER}': {e}")


@pytest.fixture(scope="module")
def registry():
    return ToolRegistry.from_configs([CalculatorToolConfig()])


def test_closing_tag_is_a_single_token(tokenizer):
    """Generation stops on this token, which only works because it is one token."""
    assert resolve_tool_stop_token_ids(tokenizer) == [CLOSE_FUNCTION_CALLS_TOKEN_ID]


def test_turn_marker_resolves(tokenizer):
    """A reply that calls no tool ends here, so generation has to stop on it too."""
    assert resolve_turn_end_token_ids(tokenizer) == [TURN_END_TOKEN_ID]


def test_schemas_reach_the_system_turn(tokenizer, registry):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 6*7?"}],
        tools=registry.schemas(),
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "<functions>" in prompt
    assert "calculator" in prompt
    assert "The arithmetic expression to evaluate" in prompt


def test_a_tool_exchange_round_trips_through_the_template(tokenizer, registry):
    completion = 'Let me compute.<function_calls>calculator(expression="6*7")</function_calls>'

    calls = parse_function_calls(completion)
    results = registry.execute_all(calls)
    assert [result.content for result in results] == ["42"]

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "What is 6*7?"},
            {"role": "assistant", "content": completion},
            render_environment_message(results),
        ],
        tools=registry.schemas(),
        tokenize=False,
        add_generation_prompt=True,
    )

    assert prompt.endswith(
        "<|im_start|>assistant\n"
        'Let me compute.<function_calls>calculator(expression="6*7")</function_calls>'
        "<|im_end|>\n"
        "<|im_start|>environment\n42<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
