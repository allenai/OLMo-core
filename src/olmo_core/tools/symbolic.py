import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import Tool, ToolConfig

__all__ = ["SymbolicMathTool", "SymbolicMathToolConfig", "has_sympy"]

OPERATIONS = ("simplify", "solve", "factor", "expand", "diff", "integrate")
"""The operations :class:`SymbolicMathTool` supports."""

_UNARY_OPERATIONS = ("simplify", "factor", "expand")


def _import_sympy():
    """Import and return sympy, raising a helpful error if it isn't installed."""
    try:
        import sympy  # type: ignore
    except ImportError as e:
        raise ImportError(
            "The 'sympy' package is required for the symbolic math tool. "
            "Install it with: pip install 'ai2-olmo-core[tools]'"
        ) from e
    return sympy


def has_sympy() -> bool:
    """
    Check whether sympy is installed.

    :returns: Whether the symbolic math tool can be used.
    """
    try:
        import sympy  # type: ignore # noqa: F401

        return True
    except ImportError:
        return False


def _validate_expression(expression: str):
    """
    Reject anything that is not a plain mathematical expression.

    sympy parses by evaluating a transformed form of the input, so an expression that reaches it
    is close to executable code. Screening the syntax first keeps attribute access, lambdas and
    comprehensions away from that path.

    :param expression: The expression to screen.

    :raises ValueError: If the expression is malformed or uses unsupported syntax.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"could not parse '{expression}': {e}") from e

    allowed = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Tuple,
        ast.List,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"unsupported syntax '{type(node).__name__}' in '{expression}'")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError(f"unsupported name '{node.id}'")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ValueError("only direct function calls are supported")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, complex)):
            raise ValueError(f"{node.value!r} is not a number")


class SymbolicMathTool(Tool):
    """
    A tool for symbolic mathematics, backed by `sympy <https://www.sympy.org>`_.

    Results are exact rather than numeric: solving ``x**2 - 2`` answers ``[-sqrt(2), sqrt(2)]``.
    An exact result can always be approximated afterwards, whereas a decimal cannot be turned
    back into an exact form.
    """

    @property
    def name(self) -> str:
        return "symbolic_math"

    @property
    def description(self) -> str:
        return (
            "Perform symbolic mathematics on an expression and return an exact result. "
            "Use 'solve' to find where an expression equals zero, 'diff' to differentiate, "
            "'integrate' to integrate, and 'simplify', 'factor' or 'expand' to rewrite."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to operate on, e.g. 'x**2 - 4'.",
                },
                "operation": {
                    "type": "string",
                    "enum": list(OPERATIONS),
                    "description": "The operation to perform.",
                },
                "variable": {
                    "type": "string",
                    "description": (
                        "The variable to solve for, differentiate by or integrate over. "
                        "Only needed when the expression has more than one variable."
                    ),
                },
            },
            "required": ["expression", "operation"],
        }

    def call(  # type: ignore[override]
        self, expression: str, operation: str, variable: Optional[str] = None
    ) -> str:
        if operation not in OPERATIONS:
            raise ValueError(f"unknown operation '{operation}'. Supported: {', '.join(OPERATIONS)}")

        sympy = _import_sympy()
        _validate_expression(expression)

        try:
            expr = sympy.parsing.sympy_parser.parse_expr(expression, evaluate=True)
        except Exception as e:
            raise ValueError(f"could not parse '{expression}': {e}") from e

        if operation in _UNARY_OPERATIONS:
            result = getattr(sympy, operation)(expr)
        else:
            symbol = self._resolve_symbol(sympy, expr, variable, operation)
            result = getattr(sympy, operation)(expr, symbol)

        return str(result)

    @staticmethod
    def _resolve_symbol(sympy, expr, variable: Optional[str], operation: str):
        if variable is not None:
            return sympy.Symbol(variable)

        symbols: List[Any] = sorted(expr.free_symbols, key=str)
        if len(symbols) == 1:
            return symbols[0]
        if not symbols:
            raise ValueError(f"'{operation}' needs a variable, but the expression has none")
        raise ValueError(
            f"'{operation}' needs a variable because the expression has several: "
            f"{', '.join(str(s) for s in symbols)}"
        )


@ToolConfig.register("symbolic_math")
@dataclass
class SymbolicMathToolConfig(ToolConfig):
    """
    Configuration for building a :class:`SymbolicMathTool`.
    """

    def build(self) -> SymbolicMathTool:
        return SymbolicMathTool()
