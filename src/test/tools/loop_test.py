from typing import List, Optional

import torch

from olmo_core.tools import CalculatorToolConfig, ToolRegistry, run_tool_loop

CLOSE_TOKEN_ID = 100269
TURN_END_TOKEN_ID = 100265


class StubTokenizer:
    """A tokenizer that records what it was asked to render and decodes canned completions."""

    def __init__(self, completions: List[str]):
        self.completions = completions
        self.rendered: List[List[dict]] = []
        self.tools: Optional[list] = None
        self.chat_template_used: Optional[str] = None

    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=True, chat_template=None
    ) -> str:
        self.rendered.append([dict(message) for message in messages])
        self.tools = tools
        self.chat_template_used = chat_template
        return "<prompt>"

    def encode(self, text, return_tensors=None):
        return torch.zeros(1, 1, dtype=torch.long)

    def decode(self, ids, skip_special_tokens=True) -> str:
        return self.completions[int(ids[0])]

    VOCAB = {"</function_calls>": CLOSE_TOKEN_ID, "<|im_end|>": TURN_END_TOKEN_ID}

    def convert_tokens_to_ids(self, token):
        return self.VOCAB.get(token)

    def convert_ids_to_tokens(self, token_id):
        return {v: k for k, v in self.VOCAB.items()}.get(token_id)


class StubGenerationModule:
    """Answers with one canned completion per call, indexed by how many calls have been made."""

    def __init__(self):
        self.generate_kwargs: List[dict] = []

    def generate_batch(self, input_ids, **kwargs):
        index = len(self.generate_kwargs)
        self.generate_kwargs.append(kwargs)
        return torch.tensor([[index]]), None, None


def _run(completions, **kwargs):
    tokenizer = StubTokenizer(completions)
    module = StubGenerationModule()
    registry = ToolRegistry.from_configs([CalculatorToolConfig()])
    result = run_tool_loop(
        module,  # type: ignore[arg-type]
        tokenizer,
        [{"role": "user", "content": "What is 6*7?"}],
        registry,
        **kwargs,
    )
    return result, tokenizer, module


def test_reply_without_a_tool_call_returns_directly():
    result, _, module = _run(["The answer is 42."])
    assert result.content == "The answer is 42."
    assert result.calls == []
    assert not result.stopped_early
    assert len(module.generate_kwargs) == 1
    assert result.messages[-1] == {"role": "assistant", "content": "The answer is 42."}


def test_tool_call_is_executed_and_fed_back():
    result, _, module = _run(
        [
            '<function_calls>calculator(expression="6*7")</function_calls>',
            "It is 42.",
        ]
    )

    assert result.content == "It is 42."
    assert [call.name for call in result.calls] == ["calculator"]
    assert [r.content for r in result.results] == ["42"]
    assert len(module.generate_kwargs) == 2

    roles = [message["role"] for message in result.messages]
    assert roles == ["user", "assistant", "environment", "assistant"]
    assert result.messages[2] == {"role": "environment", "content": "42"}


def test_several_rounds_of_tool_calling():
    result, _, _ = _run(
        [
            '<function_calls>calculator(expression="6*7")</function_calls>',
            '<function_calls>calculator(expression="42+1")</function_calls>',
            "43.",
        ]
    )
    assert [r.content for r in result.results] == ["42", "43"]
    assert result.content == "43."


def test_generation_stops_on_the_closing_tool_token():
    _, _, module = _run(["done"])
    assert CLOSE_TOKEN_ID in module.generate_kwargs[0]["stop_token_ids"]


def test_generation_stops_at_the_end_of_the_assistant_turn():
    """A reply that calls no tool ends at the turn marker, not at '</function_calls>'.

    Without this the model runs past its own turn marker into the next role and on to the token
    limit. The per-call value replaces the module's configured one, so the loop cannot rely on
    the caller having set it.
    """
    _, _, module = _run(["done"])
    assert TURN_END_TOKEN_ID in module.generate_kwargs[0]["stop_token_ids"]


def test_both_stop_tokens_are_sent_together():
    _, _, module = _run(["done"])
    assert module.generate_kwargs[0]["stop_token_ids"] == [CLOSE_TOKEN_ID, TURN_END_TOKEN_ID]


def test_caller_stop_tokens_are_kept():
    _, _, module = _run(["done"], stop_token_ids=[7])
    assert module.generate_kwargs[0]["stop_token_ids"] == [7, CLOSE_TOKEN_ID, TURN_END_TOKEN_ID]


def test_stop_tokens_are_not_duplicated():
    _, _, module = _run(["done"], stop_token_ids=[TURN_END_TOKEN_ID])
    assert module.generate_kwargs[0]["stop_token_ids"] == [TURN_END_TOKEN_ID, CLOSE_TOKEN_ID]


def test_tool_schemas_reach_the_chat_template():
    _, tokenizer, _ = _run(["done"])
    assert tokenizer.tools is not None
    assert [schema["function"]["name"] for schema in tokenizer.tools] == ["calculator"]


def test_chat_template_override_is_passed_through():
    _, tokenizer, _ = _run(["done"], chat_template="{{ 'x' }}")
    assert tokenizer.chat_template_used == "{{ 'x' }}"


def test_max_iterations_stops_the_loop():
    result, _, module = _run(
        ['<function_calls>calculator(expression="1+1")</function_calls>'] * 5,
        max_iterations=3,
    )
    assert result.stopped_early
    assert len(module.generate_kwargs) == 3


def test_a_malformed_call_is_reported_back_to_the_model():
    """The model can usually fix its own syntax once it sees what was wrong."""
    result, _, _ = _run(
        [
            "<function_calls>calculator(</function_calls>",
            "Sorry, let me try again. 42.",
        ]
    )
    assert result.content == "Sorry, let me try again. 42."
    assert result.messages[2]["role"] == "environment"
    assert result.messages[2]["content"].startswith("Error:")


def test_an_unknown_tool_is_reported_back_to_the_model():
    result, _, _ = _run(
        [
            '<function_calls>nonexistent(x="1")</function_calls>',
            "My mistake.",
        ]
    )
    assert "no tool named 'nonexistent'" in result.messages[2]["content"]


def test_on_tool_call_is_invoked():
    seen = []
    _run(
        [
            '<function_calls>calculator(expression="6*7")</function_calls>',
            "42.",
        ],
        on_tool_call=lambda call, result: seen.append((call.name, result.content)),
    )
    assert seen == [("calculator", "42")]


def test_the_input_conversation_is_not_mutated():
    messages = [{"role": "user", "content": "hi"}]
    tokenizer = StubTokenizer(["hello"])
    run_tool_loop(
        StubGenerationModule(),  # type: ignore[arg-type]
        tokenizer,
        messages,
        ToolRegistry.from_configs([CalculatorToolConfig()]),
    )
    assert messages == [{"role": "user", "content": "hi"}]
