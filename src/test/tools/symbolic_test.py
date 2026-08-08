import pytest

from olmo_core.testing import requires_sympy
from olmo_core.tools import SymbolicMathTool, SymbolicMathToolConfig

pytestmark = requires_sympy


@pytest.mark.parametrize(
    "expression, operation, variable, expected",
    [
        pytest.param("x**2 - 4", "factor", None, "(x - 2)*(x + 2)", id="factor"),
        pytest.param("(x + 1)**2", "expand", None, "x**2 + 2*x + 1", id="expand"),
        pytest.param("2*x", "integrate", "x", "x**2", id="integrate"),
        pytest.param("x**3", "diff", "x", "3*x**2", id="diff"),
        pytest.param("x**2 - 4", "solve", None, "[-2, 2]", id="solve"),
        pytest.param("sin(x)**2 + cos(x)**2", "simplify", None, "1", id="simplify"),
    ],
)
def test_symbolic_math(expression, operation, variable, expected):
    result = SymbolicMathTool().call(expression=expression, operation=operation, variable=variable)
    assert result == expected


def test_results_are_exact_rather_than_numeric():
    """An exact result can be approximated later, but a decimal cannot be made exact again."""
    assert SymbolicMathTool().call(expression="x**2 - 2", operation="solve") == (
        "[-sqrt(2), sqrt(2)]"
    )


def test_variable_is_inferred_when_unambiguous():
    assert SymbolicMathTool().call(expression="y**2", operation="diff") == "2*y"


def test_ambiguous_variable_is_reported():
    with pytest.raises(ValueError, match="several"):
        SymbolicMathTool().call(expression="x*y", operation="diff")


def test_unknown_operation_is_reported():
    with pytest.raises(ValueError, match="unknown operation"):
        SymbolicMathTool().call(expression="x", operation="integrate_by_parts")


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param('__import__("os").system("echo pwned")', id="import-escape"),
        pytest.param("(1).__class__", id="attribute-access"),
        pytest.param("[x for x in range(3)]", id="comprehension"),
        pytest.param("lambda: 1", id="lambda"),
        pytest.param("_private", id="underscore-name"),
        pytest.param("'text'", id="string"),
    ],
)
def test_symbolic_math_rejects_unsafe_expressions(expression):
    """sympy parses by evaluating, so the syntax is screened before it gets there."""
    with pytest.raises(ValueError):
        SymbolicMathTool().call(expression=expression, operation="simplify")


def test_symbolic_math_config_builds_tool():
    tool = SymbolicMathToolConfig().build()
    assert isinstance(tool, SymbolicMathTool)
    assert tool.name == "symbolic_math"
