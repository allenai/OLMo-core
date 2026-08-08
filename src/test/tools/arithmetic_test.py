import pytest

from olmo_core.tools import CalculatorTool, CalculatorToolConfig
from olmo_core.tools.arithmetic import evaluate


@pytest.mark.parametrize(
    "expression, expected",
    [
        pytest.param("2 + 2", "4", id="addition"),
        pytest.param("2 * (3 + 4)", "14", id="precedence"),
        pytest.param("7 // 2", "3", id="floor-division"),
        pytest.param("7 % 3", "1", id="modulo"),
        pytest.param("2 ** 10", "1024", id="power"),
        pytest.param("-5 + 3", "-2", id="negative"),
        pytest.param("sqrt(16)", "4", id="sqrt"),
        pytest.param("max(1, 7, 3)", "7", id="max"),
        pytest.param("factorial(5)", "120", id="factorial"),
        pytest.param("gcd(12, 18)", "6", id="gcd"),
        # Twelve significant digits keeps binary floating point noise out of the answer.
        pytest.param("0.1 + 0.2", "0.3", id="float-noise-suppressed"),
    ],
)
def test_calculator(expression, expected):
    assert CalculatorTool().call(expression=expression) == expected


def test_calculator_knows_pi():
    assert CalculatorTool().call(expression="pi").startswith("3.14159")


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param('__import__("os").system("echo pwned")', id="import-escape"),
        pytest.param("(1).__class__", id="attribute-access"),
        pytest.param("[x for x in range(3)]", id="comprehension"),
        pytest.param("open('/etc/passwd')", id="disallowed-builtin"),
        pytest.param("lambda: 1", id="lambda"),
        pytest.param("print(1)", id="unknown-function"),
        pytest.param("undefined_name + 1", id="unknown-name"),
        pytest.param("'a' * 3", id="string-operand"),
        pytest.param("1 if True else 2", id="conditional"),
        pytest.param("2 +", id="syntax-error"),
    ],
)
def test_calculator_rejects_unsafe_expressions(expression):
    with pytest.raises(ValueError):
        evaluate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param("2 ** 999999999", id="huge-exponent"),
        pytest.param("factorial(999999)", id="huge-factorial"),
    ],
)
def test_calculator_refuses_to_exhaust_memory(expression):
    """A short expression should not be able to allocate its way through the machine."""
    with pytest.raises(ValueError, match="too large"):
        evaluate(expression)


def test_calculator_config_builds_tool():
    tool = CalculatorToolConfig().build()
    assert isinstance(tool, CalculatorTool)
    assert tool.name == "calculator"


def test_calculator_schema():
    schema = CalculatorTool().json_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculator"
    assert schema["function"]["parameters"]["required"] == ["expression"]
