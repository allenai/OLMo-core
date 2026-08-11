"""The ``groups4`` task. Importing this package registers its spec."""

from ...format import registry
from .spec import SPEC

registry.register(SPEC)

__all__ = ["SPEC"]
