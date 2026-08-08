"""
Parsing and rendering for the OLMo 3 tool-calling wire format.

This module is deliberately free of any model or tensor dependency so that the format can be
tested on its own.

The format is fixed by the chat template the OLMo 3 instruct models were tuned with:

- Tool schemas are placed in the system turn between ``<functions>`` and ``</functions>`` as a
  JSON list of OpenAI-style function specs.
- The model emits calls between ``<function_calls>`` and ``</function_calls>`` using Python call
  syntax with JSON-encoded values, e.g. ``get_weather(city="Paris", days=3)``. Several calls in
  one block are separated by newlines.
- Results are fed back as a turn with the ``environment`` role.
"""

import ast
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..exceptions import ToolCallParseError
from .base import ToolCall, ToolResult

__all__ = [
    "FUNCTIONS_START",
    "FUNCTIONS_END",
    "FUNCTION_CALLS_START",
    "FUNCTION_CALLS_END",
    "ENVIRONMENT_ROLE",
    "TURN_END",
    "contains_function_call",
    "parse_function_calls",
    "render_environment_message",
    "resolve_tool_stop_token_ids",
    "resolve_turn_end_token_ids",
    "build_tool_schemas",
]

FUNCTIONS_START = "<functions>"
"""Opening tag for the tool schemas in the system turn."""

FUNCTIONS_END = "</functions>"
"""Closing tag for the tool schemas in the system turn."""

FUNCTION_CALLS_START = "<function_calls>"
"""Opening tag for a block of tool calls."""

FUNCTION_CALLS_END = "</function_calls>"
"""Closing tag for a block of tool calls."""

ENVIRONMENT_ROLE = "environment"
"""The role tool results are fed back under."""

TURN_END = "<|im_end|>"
"""The marker that ends a turn in the chat template."""

# Jinja's `tojson` filter writes these, and Python's parser reads them as names rather than
# literals, so `ast.literal_eval` alone rejects any call carrying a boolean or a null.
_JSON_CONSTANTS: Dict[str, Any] = {"true": True, "false": False, "null": None}


def contains_function_call(text: str) -> bool:
    """
    Check whether a completion contains a tool call block.

    :param text: The decoded model completion.

    :returns: Whether an opening ``<function_calls>`` tag is present.
    """
    return FUNCTION_CALLS_START in text


def _extract_block(text: str) -> Optional[str]:
    start = text.find(FUNCTION_CALLS_START)
    if start < 0:
        return None
    start += len(FUNCTION_CALLS_START)

    end = text.find(FUNCTION_CALLS_END, start)
    if end < 0:
        # Generation may have been cut short by a stop token or a length limit, in which case the
        # closing tag never made it into the text.
        return text[start:]
    return text[start:end]


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in _JSON_CONSTANTS:
        return _JSON_CONSTANTS[node.id]
    if isinstance(node, ast.List):
        return [_literal(element) for element in node.elts]
    if isinstance(node, ast.Tuple):
        return [_literal(element) for element in node.elts]
    if isinstance(node, ast.Dict):
        keys: List[Any] = []
        for key in node.keys:
            if key is None:
                raise ToolCallParseError("dict unpacking is not allowed in a tool call")
            keys.append(_literal(key))
        return {key: _literal(value) for key, value in zip(keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        # Negative numbers are a unary minus applied to a constant, not a negative constant.
        operand = _literal(node.operand)
        if not isinstance(operand, (int, float, complex)) or isinstance(operand, bool):
            raise ToolCallParseError(f"cannot negate {operand!r} in a tool call")
        return -operand if isinstance(node.op, ast.USub) else +operand

    rendered = ast.unparse(node) if isinstance(node, ast.expr) else ast.dump(node)
    raise ToolCallParseError(f"unsupported value in a tool call: {rendered!r}")


def _parse_call(node: ast.Call) -> ToolCall:
    if not isinstance(node.func, ast.Name):
        raise ToolCallParseError(
            f"a tool call must be a plain function name, got {ast.unparse(node.func)!r}"
        )
    if node.args:
        raise ToolCallParseError(
            f"'{node.func.id}' was called with positional arguments, but tool calls must use "
            "keyword arguments"
        )

    arguments: Dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ToolCallParseError(
                f"'{node.func.id}' was called with '**' unpacking, which is not allowed"
            )
        arguments[keyword.arg] = _literal(keyword.value)

    return ToolCall(name=node.func.id, arguments=arguments)


def parse_function_calls(text: str) -> List[ToolCall]:
    """
    Parse the tool calls out of a model completion.

    :param text: The decoded model completion.

    :returns: The parsed calls, or an empty list if the completion contains no call block.

    :raises ToolCallParseError: If a call block is present but malformed.
    """
    block = _extract_block(text)
    if block is None:
        return []

    block = block.strip()
    if not block:
        return []

    try:
        tree = ast.parse(block, mode="exec")
    except SyntaxError as e:
        raise ToolCallParseError(f"could not parse tool call block: {e}") from e

    calls: List[ToolCall] = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ToolCallParseError(
                f"expected a tool call, got {ast.unparse(statement)!r}",
            )
        calls.append(_parse_call(statement.value))

    return calls


def render_environment_message(results: Sequence[ToolResult]) -> Dict[str, str]:
    """
    Build the conversation turn that feeds tool results back to the model.

    :param results: The results to report, in call order.

    :returns: A message dict with the ``environment`` role.
    """
    parts: List[str] = []
    for result in results:
        body = result.content if result.ok else f"Error: {result.error}"
        # A single result needs no attribution, but several would otherwise be ambiguous.
        parts.append(body if len(results) == 1 else f"{result.name}: {body}")

    return {"role": ENVIRONMENT_ROLE, "content": "\n".join(parts)}


def _resolve_token_id(tokenizer: Any, token: str) -> Optional[int]:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        return None

    # Tokenizers answer with the unknown-token ID for strings they do not carry, so confirm the
    # round trip rather than trusting the lookup.
    if tokenizer.convert_ids_to_tokens(token_id) != token:
        return None

    return token_id


def resolve_tool_stop_token_ids(tokenizer: Any) -> List[int]:
    """
    Look up the token IDs that mark the end of a tool call block.

    Generation can stop on these so that a call is executed as soon as it is complete, rather
    than after the model has run on to its turn limit.

    :param tokenizer: A Hugging Face tokenizer.

    :returns: The matching token IDs, or an empty list if the tokenizer has no dedicated token
        for the closing tag.
    """
    token_id = _resolve_token_id(tokenizer, FUNCTION_CALLS_END)
    return [] if token_id is None else [token_id]


def resolve_turn_end_token_ids(tokenizer: Any) -> List[int]:
    """
    Look up the token IDs that mark the end of an assistant turn.

    A reply that calls no tool ends the turn rather than a tool call block, so generation has to
    stop on this too. Without it the model runs on past its own turn marker and into whatever
    role comes next, until it hits the token limit.

    :param tokenizer: A Hugging Face tokenizer.

    :returns: The matching token IDs, or an empty list if the tokenizer does not use this marker.
    """
    token_id = _resolve_token_id(tokenizer, TURN_END)
    return [] if token_id is None else [token_id]


def build_tool_schemas(tools: Iterable[Any]) -> List[Dict[str, Any]]:
    """
    Collect the function specs for a set of tools.

    :param tools: The tools to describe.

    :returns: The specs, suitable for the ``tools`` argument of a chat template.
    """
    return [tool.json_schema() for tool in tools]
