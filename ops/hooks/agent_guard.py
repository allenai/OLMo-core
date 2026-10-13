#!/usr/bin/env python3
"""Claude Code PreToolUse hook for the Agent tool (stdlib only, fails OPEN).

Denies (exit 2) any subagent spawn whose ``model`` is missing or is not a Sonnet/Opus/Haiku model,
because an omitted model inherits the parent's (often Fable, whose weekly quota is the scarce one).
Exempt: ``subagent_type`` ``fork`` (forks always inherit the parent by design), and the tiny
built-in helpers ``claude-code-guide`` / ``statusline-setup``.
"""
import json
import sys

ALLOWED_FAMILIES = ("sonnet", "opus", "haiku")
EXEMPT_TYPES = {"fork", "claude-code-guide", "statusline-setup"}
REASON = ("[ops agent_guard] DENIED: Prasann's standing rule: pin every subagent explicitly to sonnet or "
          "opus (model: \"opus\"|\"sonnet\"). Omitting `model` inherits the parent model, and Fable is "
          "forbidden for subagents. Re-issue the Agent call with model=\"sonnet\" for mechanical work or "
          "model=\"opus\" for reasoning-heavy work.")


def check(tool_input):
    """Return a deny reason, or None if the Agent call is allowed."""
    ti = tool_input or {}
    if (ti.get("subagent_type") or "").strip() in EXEMPT_TYPES:
        return None
    model = (ti.get("model") or "").strip().lower()
    if model and "fable" not in model and any(f in model for f in ALLOWED_FAMILIES):
        return None
    return REASON


def main():
    try:
        data = json.load(sys.stdin)
        if data.get("tool_name") not in ("Agent", "Task"):
            return 0
        reason = check(data.get("tool_input"))
        if reason:
            sys.stderr.write(reason + "\n")
            return 2
        return 0
    except Exception:
        return 0  # fail open


if __name__ == "__main__":
    sys.exit(main())
