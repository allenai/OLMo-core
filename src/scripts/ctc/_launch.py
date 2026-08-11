"""
Shared launcher plumbing for the Beaker entry points in this directory.

Small on purpose: the only thing worth sharing is the check that the node will run the code you
think it will.
"""

from __future__ import annotations

import subprocess


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def pushed_head() -> str:
    """
    Check that the commit gantry will clone is the one on this machine.

    gantry runs the **pushed** commit, so the working tree is irrelevant to what executes -- which
    is why these launchers set ``allow_dirty`` and why doing so is safe even with an unrelated edit
    in flight. The hazard the dirty-tree guard gets mistaken for is the opposite one: launching
    with local commits that were never pushed silently runs older code, and the result cannot be
    reproduced from the sha it records.

    :returns: The commit the node will check out.

    :raises SystemExit: If HEAD is not on the tracking branch at the remote.
    """
    head = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    contains = subprocess.run(
        ["git", "branch", "-r", "--contains", head], capture_output=True, text=True
    ).stdout
    if f"origin/{branch}" not in contains:
        raise SystemExit(
            f"HEAD ({head[:12]}) is not on origin/{branch}. gantry clones the pushed commit, so "
            f"this would run older code.\n  git push origin {branch}"
        )
    return head
