# Hammond control session

This records the CPU-only Beaker session used as a control plane for MoE ladder experiments.

## Recreate the session

From a machine with Beaker auth, GitHub SSH access, and local Codex auth:

```bash
REFRESH_SECRETS=1 src/scripts/train/jacobm_olmoe_ladder/launch_hammond_control_session.sh
```

For later recreations, if the secrets are already current:

```bash
src/scripts/train/jacobm_olmoe_ladder/launch_hammond_control_session.sh
```

The script creates a detached bare Beaker session on `hammond-cs-aus-452.reviz.ai2.in`, mounts:

- `/weka/oe-training-default`
- `/weka/oe-adapt-default`

It uses `/weka/oe-adapt-default/jacobm/olmoe3` as the project root and clones `allenai/OLMo-core` on branch `jacobm/olmoe-dev-v2`.

## Required secrets

These secrets live in `ai2/OLMo-3-moe-experiments`:

- `jacobm_beaker_config`
- `jacobm_github_ssh_key`
- `jacobm_codex_auth_json`
- `jacobm_codex_config_toml`
- `jacobm_WANDB_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `git-config`

`REFRESH_SECRETS=1` updates the Jacob-specific secrets from local files. It does not print secret values.

## Remote access

Beaker remote session exec assumes the local Unix username matches the Ai2 login. On this laptop the local user is `jacob`, while the Ai2 login is `jacobm`, so use direct SSH to Hammond:

```bash
ssh jacobm@hammond-cs-aus-452.reviz.ai2.in "beaker session exec <SESSION_ID> -- bash -l"
```

To restart Codex remote control inside a live session:

```bash
ssh jacobm@hammond-cs-aus-452.reviz.ai2.in \
  "beaker session exec <SESSION_ID> -- bash -lc 'cd /weka/oe-adapt-default/jacobm/olmoe3/OLMo-core && /root/.local/bin/codex remote-control start --json'"
```

## Current session

Started 2026-06-03:

- Session ID: `01KT7NS3EZMTR1CE38K1RJMTCR`
- Host: `hammond-cs-aus-452.reviz.ai2.in`
- Codex: `0.136.0`
- Remote-control status: running
