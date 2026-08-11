"""Make ``fixtures/`` importable so tests can reuse the golden generator's case tables.

The tables live with the generator rather than the test because they must stay the single
definition of what got snapshotted -- if a test declared its own copy, adding a case there would
check the new code against nothing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Run the suite against THIS checkout, whatever the interpreter has installed. Both paths matter:
# a stale ``ctc`` elsewhere on the path would test the wrong code, and olmo-core (needed by
# ctc/train and the native backend) is usually not installed at all -- so those tests would skip,
# and for a guard a silently-skipped test and a passing one look identical in the summary.
_REPO = Path(__file__).resolve().parents[2]
for _src in (_REPO / "ctc" / "src", _REPO / "src"):
    if _src.is_dir():
        sys.path.insert(0, str(_src))
