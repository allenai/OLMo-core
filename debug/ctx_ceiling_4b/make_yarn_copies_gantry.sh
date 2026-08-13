#!/usr/bin/env bash
# Build YaRN serving copies of one or more olmo-core step dirs on weka, from a Beaker CPU node.
#
# `make_yarn_copy.py` is pure stdlib and writes ~15 KB per copy (a patched config.json plus a
# symlink back at the original model_and_optim), but it has to run somewhere that can SEE weka --
# which a laptop cannot. This wraps it in a gantry CPU job so the copies can be built without a
# weka mount, and verifies each one before exiting.
#
# The 256k and 512k eval rungs both need factor 2, and 1M needs factor 4, against Qwen3.5's native
# 262,144 ceiling. Over-scaling degrades the shorter rungs, so build one copy per rung GROUP rather
# than a single high-factor copy for everything.
#
# Usage:
#   CKPTS='/weka/.../run-a/step100 /weka/.../run-b/step200' FACTOR=2 \
#     debug/ctx_ceiling_4b/make_yarn_copies_gantry.sh
#
# Overridable env: CLUSTER WORKSPACE BUDGET WEKA PRIORITY CPUS NAME IMAGE CKPTS FACTOR
set -euo pipefail

CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
# CLAUDE.md: every Beaker job launches at urgent. (ai2/holmes is the one documented exception.)
PRIORITY="${PRIORITY:-urgent}"
CPUS="${CPUS:-2}"
FACTOR="${FACTOR:-2}"
NAME="${NAME:-make-yarn${FACTOR}-copies}"
IMAGE="${IMAGE:-tylerr/olmo-core-tch291cu128-2025-11-25}"

CKPTS="${CKPTS:?set CKPTS='<abs step dir> [<abs step dir> ...]'}"

CLUSTER_ARGS=()
IFS=',' read -ra _CLUSTERS <<< "${CLUSTER}"
for c in "${_CLUSTERS[@]}"; do CLUSTER_ARGS+=(--cluster "$c"); done

read -r -d '' REMOTE <<REMOTE_EOF || true
set -uo pipefail
M=debug/ctx_ceiling_4b/make_yarn_copy.py
rc=0
for src in ${CKPTS}; do
  echo "=== \$src (factor ${FACTOR}) ==="
  if [ ! -f "\$src/model_and_optim/.metadata" ]; then
    echo "  !!! MISSING or incomplete source step dir -- no .metadata"; rc=1; continue
  fi
  python \$M --src "\$src" --factor ${FACTOR} --force || { rc=1; continue; }
  dest="\${src}_yarn${FACTOR}"
  # A copy that exists is not a copy that works: the eval loads config.json and follows the
  # model_and_optim link, so check BOTH, and confirm the YaRN patch actually landed in the config
  # rather than trusting the script's exit code.
  if [ ! -f "\$dest/config.json" ]; then
    echo "  !!! \$dest/config.json missing"; rc=1; continue
  fi
  if [ ! -f "\$dest/model_and_optim/.metadata" ]; then
    echo "  !!! \$dest/model_and_optim does not resolve to real weights"; rc=1; continue
  fi
  python - "\$dest/config.json" ${FACTOR} <<'PYEOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
want = float(sys.argv[2])
found = []
def walk(o):
    if isinstance(o, dict):
        s = o.get("scaling") or o.get("rope_scaling")
        if isinstance(s, dict) and "factor" in s:
            found.append((s.get("factor"), s.get("name") or s.get("type") or s.get("rope_type")))
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(cfg)
if not found:
    sys.exit("  !!! no RoPE scaling block found in the patched config")
bad = [f for f in found if float(f[0]) != want]
if bad:
    sys.exit(f"  !!! wrong factor in {bad} (wanted {want})")
print(f"  ok: {len(found)} RoPE block(s) at factor {want}, kinds={sorted({f[1] for f in found})}")
PYEOF
  [ \$? -ne 0 ] && rc=1
done
echo "=== rc=\$rc ==="
exit \$rc
REMOTE_EOF

gantry run \
  --name "${NAME}" \
  --task-name "${NAME}" \
  --workspace "${WORKSPACE}" \
  --budget "${BUDGET}" \
  "${CLUSTER_ARGS[@]}" \
  --priority "${PRIORITY}" \
  --beaker-image "${IMAGE}" \
  --cpus "${CPUS}" \
  --weka "${WEKA}:/weka/${WEKA}" \
  --python-manager conda \
  --system-python \
  --allow-dirty \
  --yes \
  --show-logs \
  -- bash -c "${REMOTE}"
