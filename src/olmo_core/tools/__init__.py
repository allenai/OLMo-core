"""
Tools that a language model can call during generation.

The OLMo 3 instruct models were tuned to call tools, so this package supplies the other half:
a registry of callable tools, a parser for the format those models emit, and a loop that runs
the tools and feeds their results back.

The wire format is fixed by the chat template the models were tuned with, and is described in
:mod:`olmo_core.tools.protocol`. Tool schemas go into the system turn, the model emits calls
between ``<function_calls>`` and ``</function_calls>``, and results come back as an
``environment`` turn.

A minimal end-to-end example::

    from olmo_core.generate import TransformerGenerationModule
    from olmo_core.tools import CalculatorToolConfig, ToolRegistry, run_tool_loop
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct")
    generation_module = TransformerGenerationModule.from_checkpoint("path/to/checkpoint")
    registry = ToolRegistry.from_configs([CalculatorToolConfig()])

    result = run_tool_loop(
        generation_module,
        tokenizer,
        [{"role": "user", "content": "What is 17 * 23?"}],
        registry,
    )
    print(result.content)

Tools are configured through :class:`ToolConfig`, which is
:class:`~olmo_core.config.Registrable`, so a config can name one by its registered type::

    from olmo_core.config import Config
    from olmo_core.tools import ToolConfig

    config = ToolConfig.from_dict({"type": "calculator"})
"""

from .arithmetic import CalculatorTool, CalculatorToolConfig
from .base import Tool, ToolCall, ToolConfig, ToolResult
from .loop import ToolLoopResult, run_tool_loop
from .protocol import (
    ENVIRONMENT_ROLE,
    FUNCTION_CALLS_END,
    FUNCTION_CALLS_START,
    FUNCTIONS_END,
    FUNCTIONS_START,
    TURN_END,
    build_tool_schemas,
    contains_function_call,
    parse_function_calls,
    render_environment_message,
    resolve_tool_stop_token_ids,
    resolve_turn_end_token_ids,
)
from .registry import ToolRegistry
from .symbolic import SymbolicMathTool, SymbolicMathToolConfig, has_sympy
from .web_search import (
    DdgsBackend,
    DdgsBackendConfig,
    SearchResult,
    SerperBackend,
    SerperBackendConfig,
    StaticBackend,
    StaticBackendConfig,
    TavilyBackend,
    TavilyBackendConfig,
    WebSearchBackend,
    WebSearchBackendConfig,
    WebSearchTool,
    WebSearchToolConfig,
    has_ddgs,
)

__all__ = [
    "Tool",
    "ToolCall",
    "ToolConfig",
    "ToolResult",
    "ToolRegistry",
    "ToolLoopResult",
    "run_tool_loop",
    "CalculatorTool",
    "CalculatorToolConfig",
    "SymbolicMathTool",
    "SymbolicMathToolConfig",
    "has_sympy",
    "WebSearchTool",
    "WebSearchToolConfig",
    "WebSearchBackend",
    "WebSearchBackendConfig",
    "SearchResult",
    "DdgsBackend",
    "DdgsBackendConfig",
    "TavilyBackend",
    "TavilyBackendConfig",
    "SerperBackend",
    "SerperBackendConfig",
    "StaticBackend",
    "StaticBackendConfig",
    "has_ddgs",
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
