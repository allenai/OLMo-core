import ast
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Union

from .base import Tool, ToolConfig

__all__ = ["CalculatorTool", "CalculatorToolConfig"]

Number = Union[int, float]

_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "abs": abs,
    "acos": math.acos,
    "asin": math.asin,
    "atan": math.atan,
    "atan2": math.atan2,
    "ceil": math.ceil,
    "cos": math.cos,
    "cosh": math.cosh,
    "degrees": math.degrees,
    "exp": math.exp,
    "factorial": math.factorial,
    "floor": math.floor,
    "gcd": math.gcd,
    "hypot": math.hypot,
    "lcm": math.lcm,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "max": max,
    "min": min,
    "pow": pow,
    "radians": math.radians,
    "round": round,
    "sin": math.sin,
    "sinh": math.sinh,
    "sqrt": math.sqrt,
    "tan": math.tan,
    "tanh": math.tanh,
    "trunc": math.trunc,
}

_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

_BINARY_OPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}

# A bound on exponents and factorials. Both can turn a short expression into one that exhausts
# memory long before it returns, which is a denial of service rather than a wrong answer.
_MAX_EXPONENT = 1_000
_MAX_FACTORIAL = 1_000


def evaluate(expression: str) -> Number:
    """
    Evaluate an arithmetic expression without executing arbitrary code.

    The expression is parsed and walked directly. Anything outside of numbers, operators and the
    allowed function and constant names is rejected, so this never falls back to :func:`eval`.

    :param expression: The expression to evaluate.

    :returns: The value of the expression.

    :raises ValueError: If the expression is malformed or uses anything that is not allowed.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"could not parse '{expression}': {e}") from e
    return _eval(tree.body)


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"{node.value!r} is not a number")
        return node.value

    if isinstance(node, ast.Name):
        if node.id not in _CONSTANTS:
            raise ValueError(f"unknown name '{node.id}'")
        return _CONSTANTS[node.id]

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +_eval(node.operand)
        raise ValueError(f"unsupported unary operator '{type(node.op).__name__}'")

    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator '{type(node.op).__name__}'")
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent {right} is too large (limit is {_MAX_EXPONENT})")
        return op(left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("only direct calls to the allowed functions are supported")
        function = _FUNCTIONS.get(node.func.id)
        if function is None:
            raise ValueError(f"unknown function '{node.func.id}'")
        if node.keywords:
            raise ValueError(f"'{node.func.id}' does not take keyword arguments")
        args = [_eval(arg) for arg in node.args]
        if node.func.id == "factorial" and args and args[0] > _MAX_FACTORIAL:
            raise ValueError(f"factorial argument is too large (limit is {_MAX_FACTORIAL})")
        return function(*args)

    raise ValueError(f"unsupported syntax '{type(node).__name__}'")


def format_number(value: Number) -> str:
    """
    Render a computed value for the model.

    :param value: The value to render.

    :returns: The rendered value.
    """
    if isinstance(value, int):
        return str(value)
    if math.isinf(value) or math.isnan(value):
        return str(value)
    # Twelve significant digits keeps binary floating point noise out of the answer: 0.1 + 0.2
    # should come back as 0.3 rather than 0.30000000000000004.
    return f"{value:.12g}"


class CalculatorTool(Tool):
    """
    A tool for evaluating arithmetic expressions.
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate an arithmetic expression and return the result. Supports +, -, *, /, //, "
            "%, ** and the functions "
            f"{', '.join(sorted(_FUNCTIONS))}, plus the constants {', '.join(sorted(_CONSTANTS))}."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate, e.g. '2 * (3 + 4)'.",
                }
            },
            "required": ["expression"],
        }

    def call(self, expression: str) -> str:  # type: ignore[override]
        return format_number(evaluate(expression))


@ToolConfig.register("calculator")
@dataclass
class CalculatorToolConfig(ToolConfig):
    """
    Configuration for building a :class:`CalculatorTool`.
    """

    def build(self) -> CalculatorTool:
        return CalculatorTool()
