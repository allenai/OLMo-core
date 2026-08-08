from typing import Any, Dict, Iterator, List, Sequence

from ..exceptions import OLMoConfigurationError
from .base import Tool, ToolCall, ToolConfig, ToolResult

__all__ = ["ToolRegistry"]


class ToolRegistry:
    """
    A set of tools a model may call, and the executor for those calls.

    Execution never raises on a failing tool. A tool that blows up, or a call naming a tool that
    does not exist, comes back as a :class:`~olmo_core.tools.base.ToolResult` carrying the error,
    which is then shown to the model so it can correct itself and continue.

    :param tools: The tools to make available.

    :raises OLMoConfigurationError: If two tools share a name.
    """

    def __init__(self, tools: Sequence[Tool]):
        by_name: Dict[str, Tool] = {}
        for tool in tools:
            if tool.name in by_name:
                raise OLMoConfigurationError(f"duplicate tool name '{tool.name}'")
            by_name[tool.name] = tool
        self._tools = by_name

    @classmethod
    def from_configs(cls, configs: Sequence[ToolConfig]) -> "ToolRegistry":
        """
        Build a registry from tool configs.

        :param configs: The configs to build.

        :returns: The registry.
        """
        return cls([config.build() for config in configs])

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    @property
    def names(self) -> List[str]:
        """
        The names of the registered tools.
        """
        return list(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        """
        Build the function specs for every registered tool.

        :returns: The specs, suitable for the ``tools`` argument of a chat template.
        """
        return [tool.json_schema() for tool in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        """
        Run a single tool call.

        :param call: The call to run.

        :returns: The result, which carries an error rather than raising if the call failed.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(self.names) or "none"
            return ToolResult(
                name=call.name,
                content="",
                error=f"no tool named '{call.name}'. Available tools: {available}",
            )

        try:
            return ToolResult(name=call.name, content=tool.call(**call.arguments))
        except TypeError as e:
            # Most often the model passed an argument the tool does not take.
            return ToolResult(name=call.name, content="", error=f"invalid arguments: {e}")
        except Exception as e:
            return ToolResult(name=call.name, content="", error=f"{type(e).__name__}: {e}")

    def execute_all(self, calls: Sequence[ToolCall]) -> List[ToolResult]:
        """
        Run several tool calls in order.

        :param calls: The calls to run.

        :returns: One result per call, in the same order.
        """
        return [self.execute(call) for call in calls]
