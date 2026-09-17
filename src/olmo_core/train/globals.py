""" A centeral place to store global variables used in training. """

from collections import defaultdict
from typing import Any, DefaultDict

_GLOBAL_ARGS: DefaultDict[str, Any] = defaultdict(None)


def set_global_args(args: dict) -> None:
    """Set global arguments for training."""
    global _GLOBAL_ARGS
    _GLOBAL_ARGS.update(args)


def get_global_args() -> dict:
    """Get global arguments for training."""
    global _GLOBAL_ARGS
    return _GLOBAL_ARGS.copy()


def get_global_arg(key: str, default=None):
    """Get a specific global argument for training."""
    global _GLOBAL_ARGS
    return _GLOBAL_ARGS.get(key, default)


def set_global_arg(key: str, value) -> None:
    """Set a specific global argument for training."""
    global _GLOBAL_ARGS
    _GLOBAL_ARGS[key] = value


def clear_global_args() -> None:
    """Clear all global arguments."""
    global _GLOBAL_ARGS
    _GLOBAL_ARGS.clear()
