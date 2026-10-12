"""
The ``contradiction`` task.

Importing this package registers the spec. Nothing else here has side effects, so the loader in
:mod:`ctc.tasks` can import it cheaply to populate the registry without pulling in the generator's
dependencies.
"""

from ...format import registry
from .spec import SPEC

registry.register(SPEC)

__all__ = ["SPEC"]
