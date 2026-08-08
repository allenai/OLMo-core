import pytest

from olmo_core.exceptions import ToolCallParseError
from olmo_core.tools import (
    ToolCall,
    ToolResult,
    contains_function_call,
    parse_function_calls,
    render_environment_message,
    resolve_tool_stop_token_ids,
    resolve_turn_end_token_ids,
)


def _block(body: str) -> str:
    return f"<function_calls>{body}</function_calls>"


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("no tools here", [], id="no-call"),
        pytest.param(
            _block('calculator(expression="2+2")'),
            [ToolCall("calculator", {"expression": "2+2"})],
            id="single-call",
        ),
        pytest.param(
            "Let me work that out. " + _block('calculator(expression="2+2")'),
            [ToolCall("calculator", {"expression": "2+2"})],
            id="call-after-content",
        ),
        pytest.param(
            _block('calculator(expression="1+1")\ncalculator(expression="2+2")'),
            [
                ToolCall("calculator", {"expression": "1+1"}),
                ToolCall("calculator", {"expression": "2+2"}),
            ],
            id="two-calls",
        ),
        pytest.param(
            _block('calculator(\n    expression="1 + 2"\n)'),
            [ToolCall("calculator", {"expression": "1 + 2"})],
            id="call-spanning-lines",
        ),
        pytest.param(
            '<function_calls>calculator(expression="5*5")',
            [ToolCall("calculator", {"expression": "5*5"})],
            id="closing-tag-missing",
        ),
        pytest.param(_block("   "), [], id="empty-block"),
    ],
)
def test_parse_function_calls(text, expected):
    assert parse_function_calls(text) == expected


@pytest.mark.parametrize(
    "literal, expected",
    [
        pytest.param("true", True, id="true"),
        pytest.param("false", False, id="false"),
        pytest.param("null", None, id="null"),
        pytest.param("-3", -3, id="negative-int"),
        pytest.param("-2.5", -2.5, id="negative-float"),
        pytest.param("3", 3, id="int"),
        pytest.param('"text"', "text", id="string"),
        pytest.param("[1, -2, true]", [1, -2, True], id="list"),
        pytest.param('{"a": false, "b": [null]}', {"a": False, "b": [None]}, id="dict"),
    ],
)
def test_parse_json_encoded_values(literal, expected):
    """The chat template writes values with Jinja's ``tojson``, not as Python literals."""
    calls = parse_function_calls(_block(f"search(value={literal})"))
    assert calls == [ToolCall("search", {"value": expected})]


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("calculator(", id="unbalanced-parens"),
        pytest.param('calculator("2+2")', id="positional-argument"),
        pytest.param('calculator(**{"expression": "2+2"})', id="dict-unpacking"),
        pytest.param('math.calculator(expression="2+2")', id="dotted-name"),
        pytest.param("calculator(expression=undefined_name)", id="bare-name"),
        pytest.param("x = 1", id="not-a-call"),
    ],
)
def test_parse_function_calls_rejects_malformed(body):
    with pytest.raises(ToolCallParseError):
        parse_function_calls(_block(body))


def test_contains_function_call():
    assert contains_function_call(_block('calculator(expression="1")'))
    assert not contains_function_call("plain text")


@pytest.mark.parametrize(
    "results, expected",
    [
        pytest.param([ToolResult("calculator", "4")], "4", id="single-result-unattributed"),
        pytest.param(
            [ToolResult("calculator", "4"), ToolResult("calculator", "9")],
            "calculator: 4\ncalculator: 9",
            id="several-results-attributed",
        ),
        pytest.param(
            [ToolResult("calculator", "", error="boom")],
            "Error: boom",
            id="error",
        ),
    ],
)
def test_render_environment_message(results, expected):
    message = render_environment_message(results)
    assert message == {"role": "environment", "content": expected}


class _Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inverse = {v: k for k, v in vocab.items()}

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)

    def convert_ids_to_tokens(self, token_id):
        return self.inverse.get(token_id)


def test_resolve_tool_stop_token_ids():
    tokenizer = _Tokenizer({"</function_calls>": 100269})
    assert resolve_tool_stop_token_ids(tokenizer) == [100269]


def test_resolve_tool_stop_token_ids_without_the_token():
    assert resolve_tool_stop_token_ids(_Tokenizer({})) == []


def test_resolve_turn_end_token_ids():
    assert resolve_turn_end_token_ids(_Tokenizer({"<|im_end|>": 100265})) == [100265]


def test_resolve_turn_end_token_ids_without_the_token():
    """A tokenizer that does not use this marker contributes no stop token rather than failing."""
    assert resolve_turn_end_token_ids(_Tokenizer({})) == []


def test_resolve_tool_stop_token_ids_ignores_unknown_token_fallback():
    """Tokenizers answer with the unknown-token ID for strings they do not carry."""
    tokenizer = _Tokenizer({"<unk>": 0})
    tokenizer.vocab["</function_calls>"] = 0
    assert resolve_tool_stop_token_ids(tokenizer) == []
