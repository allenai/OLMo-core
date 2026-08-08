import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from ..exceptions import ToolCallParseError
from .base import ToolCall, ToolResult
from .protocol import (
    ENVIRONMENT_ROLE,
    parse_function_calls,
    render_environment_message,
    resolve_tool_stop_token_ids,
    resolve_turn_end_token_ids,
)
from .registry import ToolRegistry

if TYPE_CHECKING:
    from ..generate.generation_module import GenerationModule

__all__ = ["ToolLoopResult", "run_tool_loop"]

log = logging.getLogger(__name__)


@dataclass
class ToolLoopResult:
    """
    The outcome of :func:`run_tool_loop`.
    """

    content: str
    """
    The model's final reply, once it stopped calling tools.
    """

    messages: List[Dict[str, str]] = field(default_factory=list)
    """
    The full conversation, including the assistant turns that called tools and the
    ``environment`` turns carrying the results.
    """

    calls: List[ToolCall] = field(default_factory=list)
    """
    Every tool call the model made, in order.
    """

    results: List[ToolResult] = field(default_factory=list)
    """
    The result of each call in :data:`calls`.
    """

    stopped_early: bool = False
    """
    Whether the loop hit ``max_iterations`` while the model was still calling tools.
    """


def run_tool_loop(
    generation_module: "GenerationModule",
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    registry: ToolRegistry,
    *,
    max_iterations: int = 5,
    chat_template: Optional[str] = None,
    on_tool_call: Optional[Callable[[ToolCall, ToolResult], None]] = None,
    **generation_kwargs,
) -> ToolLoopResult:
    """
    Generate a reply, running any tools the model calls along the way.

    Each round the conversation is rendered with the tokenizer's chat template, generated
    against, and checked for a tool call. A call is executed and its result appended as an
    ``environment`` turn, and generation runs again. The loop ends when the model replies
    without calling a tool.

    Generation stops on the token that closes a tool call, so a call is executed as soon as it
    is complete rather than after the model runs on. Every round re-renders the whole
    conversation, which means the prompt is prefilled again each time.

    :param generation_module: The module to generate with.
    :param tokenizer: A Hugging Face tokenizer whose chat template understands tools. Passing a
        template that ignores the ``tools`` argument will silently produce a model that never
        calls anything.
    :param messages: The conversation so far, as ``{"role": ..., "content": ...}`` dicts.
    :param registry: The tools to make available.
    :param max_iterations: The most rounds of tool calling to allow before giving up.
    :param chat_template: An override for the tokenizer's own chat template.
    :param on_tool_call: Called with each call and its result, for progress reporting.
    :param generation_kwargs: Overrides passed through to the generation module.

    :returns: The final reply and the conversation that produced it.
    """
    conversation: List[Dict[str, str]] = [dict(message) for message in messages]
    schemas = registry.schemas()

    # Both markers matter and for different reasons: a turn that calls a tool ends at
    # '</function_calls>', and one that does not ends at the turn marker. Stopping on only the
    # first lets a final reply run past its own turn marker into the next role and on to the
    # token limit. A per-call value replaces the module's configured one rather than adding to
    # it, so this list has to be complete on its own.
    stop_token_ids = list(generation_kwargs.pop("stop_token_ids", None) or [])
    tool_stops = resolve_tool_stop_token_ids(tokenizer)
    for token_id in (*tool_stops, *resolve_turn_end_token_ids(tokenizer)):
        if token_id not in stop_token_ids:
            stop_token_ids.append(token_id)
    if not tool_stops:
        log.warning(
            "This tokenizer has no dedicated token for '</function_calls>', so generation cannot "
            "stop as soon as a tool call is complete."
        )

    all_calls: List[ToolCall] = []
    all_results: List[ToolResult] = []
    content = ""

    for _ in range(max_iterations):
        prompt = tokenizer.apply_chat_template(
            conversation,
            tools=schemas,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=chat_template,
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        generated, _, _ = generation_module.generate_batch(  # type: ignore[attr-defined]
            input_ids,
            completions_only=True,
            stop_token_ids=stop_token_ids or None,
            **generation_kwargs,
        )
        # The tool tags are not registered as special tokens, so they survive this and can be
        # parsed out, while the turn markers are dropped.
        content = tokenizer.decode(generated[0], skip_special_tokens=True)

        try:
            calls = parse_function_calls(content)
        except ToolCallParseError as e:
            # Hand the failure back rather than aborting: the model can usually fix its own
            # syntax when it sees what was wrong.
            log.warning("Could not parse a tool call: %s", e)
            conversation.append({"role": "assistant", "content": content})
            conversation.append({"role": ENVIRONMENT_ROLE, "content": f"Error: {e}"})
            continue

        if not calls:
            conversation.append({"role": "assistant", "content": content})
            return ToolLoopResult(
                content=content,
                messages=conversation,
                calls=all_calls,
                results=all_results,
            )

        results = registry.execute_all(calls)
        all_calls.extend(calls)
        all_results.extend(results)
        if on_tool_call is not None:
            for call, result in zip(calls, results):
                on_tool_call(call, result)

        # Keeping the completion verbatim means the next prompt reproduces exactly what the
        # model already produced, tags included.
        conversation.append({"role": "assistant", "content": content})
        conversation.append(render_environment_message(results))

    log.warning("Gave up after %d rounds of tool calling.", max_iterations)
    return ToolLoopResult(
        content=content,
        messages=conversation,
        calls=all_calls,
        results=all_results,
        stopped_early=True,
    )
