import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.tools import (
    CalculatorToolConfig,
    Tool,
    ToolCall,
    ToolConfig,
    ToolRegistry,
)


class _Exploding(Tool):
    @property
    def name(self) -> str:
        return "explode"

    @property
    def description(self) -> str:
        return "Always fails."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def call(self, **kwargs) -> str:
        raise RuntimeError("boom")


def test_registry_from_configs():
    registry = ToolRegistry.from_configs([CalculatorToolConfig()])
    assert registry.names == ["calculator"]
    assert "calculator" in registry
    assert len(registry) == 1


def test_registry_rejects_duplicate_names():
    with pytest.raises(OLMoConfigurationError, match="duplicate tool name"):
        ToolRegistry([CalculatorToolConfig().build(), CalculatorToolConfig().build()])


def test_registry_schemas():
    schemas = ToolRegistry.from_configs([CalculatorToolConfig()]).schemas()
    assert [schema["function"]["name"] for schema in schemas] == ["calculator"]


def test_execute():
    registry = ToolRegistry.from_configs([CalculatorToolConfig()])
    result = registry.execute(ToolCall("calculator", {"expression": "6*7"}))
    assert result.ok
    assert result.content == "42"


def test_unknown_tool_reports_what_is_available():
    """The model should be able to correct itself from the error text."""
    registry = ToolRegistry.from_configs([CalculatorToolConfig()])
    result = registry.execute(ToolCall("nope", {}))
    assert not result.ok
    assert "no tool named 'nope'" in (result.error or "")
    assert "calculator" in (result.error or "")


def test_bad_arguments_are_reported():
    registry = ToolRegistry.from_configs([CalculatorToolConfig()])
    result = registry.execute(ToolCall("calculator", {"wrong": "2+2"}))
    assert not result.ok
    assert "invalid arguments" in (result.error or "")


def test_a_failing_tool_does_not_raise():
    registry = ToolRegistry([_Exploding()])
    result = registry.execute(ToolCall("explode", {}))
    assert not result.ok
    assert "RuntimeError: boom" == result.error


def test_execute_all_preserves_order():
    registry = ToolRegistry.from_configs([CalculatorToolConfig()])
    results = registry.execute_all(
        [
            ToolCall("calculator", {"expression": "1+1"}),
            ToolCall("calculator", {"expression": "2+2"}),
        ]
    )
    assert [result.content for result in results] == ["2", "4"]


@pytest.mark.parametrize("name", ["calculator", "symbolic_math", "web_search"])
def test_tools_are_registered(name):
    assert name in ToolConfig.get_registered_names()
