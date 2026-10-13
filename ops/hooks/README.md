# ops hooks — Claude Code PreToolUse guards

Two stdlib-only hook scripts. Both read the hook JSON on stdin, **deny** with exit 2 (reason on
stderr, shown to the agent), give **non-blocking advice** with exit 0 + a
`{"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": "..."}}` line on stdout,
and **fail open** (exit 0) on any internal error.

| script | matcher | denies | advises |
|---|---|---|---|
| `bash_guard.py` | `Bash` | (i) pytest / `-m pytest` / `unittest discover` / `make checks\|type-check` on the login node `radagast` outside srun/sbatch (except `ops/tests`); (ii) torch/vLLM launches or srun/sbatch payloads using an NFS interpreter (`/scratch/users/prasann/conda/...`, `/accounts/...`, system miniforge); (iii) `/net/<node>/` paths in srun/sbatch/torchrun commands (copy/inspection commands like rsync/ls/cat only get advice); (iv) bare `torchrun`; (v) foreground polling: `while`/`until` + `sleep`, `for` + `sleep >= 30`, and `sleep N>=30` | squeue/sacct/`beaker experiment get\|job list\|job logs` → use `ops status`; sbatch/gantry/`launch` → register the run, log to /data; sbatch script files are scanned for NFS logs/interpreters (advice only) |
| `agent_guard.py` | `Agent` | Agent calls with no `model` or a non-sonnet/opus/haiku (e.g. Fable) model, except `subagent_type` `fork`, `claude-code-guide`, `statusline-setup` | — |

Polling/sleep rules only advise (never deny) when the call has `run_in_background: true`.
Heredoc bodies (`<<'EOF' ... EOF`) are ignored, `bash -c '...'`, `srun ... cmd` and `sbatch --wrap`
are analyzed recursively, and a command that fails to tokenize is allowed.

## Install (main agent — do not edit settings.json from a builder agent)

Create stable symlinks first (repointed at incorporation):

```bash
ln -sfn <ops checkout>/ops/hooks/bash_guard.py  /accounts/projects/berkeleynlp/prasann/bin/ops-hook-bash
ln -sfn <ops checkout>/ops/hooks/agent_guard.py /accounts/projects/berkeleynlp/prasann/bin/ops-hook-agent
```

Then merge into `~/.claude/settings.json` (top-level `"hooks"` key):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {"type": "command", "command": "python3 /accounts/projects/berkeleynlp/prasann/bin/ops-hook-bash", "timeout": 10}
        ]
      },
      {
        "matcher": "Agent",
        "hooks": [
          {"type": "command", "command": "python3 /accounts/projects/berkeleynlp/prasann/bin/ops-hook-agent", "timeout": 10}
        ]
      }
    ]
  }
}
```

`python3` here is whatever is on PATH on the login node (system miniforge 3.13); the hooks are
stdlib-only and run in ~30 ms.

## Test / try

```bash
cd ops && python3 -m unittest tests.test_hooks -v
echo '{"tool_name":"Bash","tool_input":{"command":"sleep 120"}}' | python3 hooks/bash_guard.py; echo rc=$?
echo '{"tool_name":"Agent","tool_input":{"prompt":"x","subagent_type":"general-purpose"}}' | python3 hooks/agent_guard.py; echo rc=$?
```

`OPS_HOOK_HOSTNAME=radagast` overrides the detected hostname (rule (i) applies only on the login
node).
