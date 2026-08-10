"""Make ``fixtures/`` importable so tests can reuse the golden generator's case tables.

The tables live with the generator rather than the test because they must stay the single
definition of what got snapshotted -- if a test declared its own copy, adding a case there would
check the new code against nothing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
