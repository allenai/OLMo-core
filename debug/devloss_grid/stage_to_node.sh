#!/usr/bin/env bash
# One-time staging of the dev-loss grid inputs onto the GPU node's node-local /data.
# Run INSIDE an sbatch on the target node (default sneetches). Idempotent: every copy is
# rsync -a gated on the source's completeness sentinel, and serving-copy conversions skip when
# .text_export_done exists. Reads cross-node via /net at ~78 MB/s (measured 2026-09-21), one bulk
# pass -- never point job I/O at /net afterwards.
#
#   sbatch --partition=jsteinhardt --qos=preemptive_high --account=site --nodelist=sneetches \
#     --gres=gpu:0 --cpus-per-task=8 --mem=64G --time=6:00:00 \
#     --output=/data/prasann/joblogs/devloss_stage_%j.log debug/devloss_grid/stage_to_node.sh
#
# Env: DST_ROOT (default /data/prasann/devloss_grid), ONLY="task1 task2" to restrict checkpoints.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
DST_ROOT="${DST_ROOT:-/data/prasann/devloss_grid}"
MANIFEST="$REPO/debug/devloss_grid/manifest.json"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
case "$PY" in /data/*) ;; *) echo "!!! interpreter not node-local"; exit 1;; esac
mkdir -p "$DST_ROOT/data" "$DST_ROOT/ckpts"
echo "== staging on $(hostname -s) -> $DST_ROOT ($(date))"

# 1. data (small): the rung JSONLs + cpt80 docs, from /scratch (login-node download target)
rsync -a --exclude '*.log' /scratch/users/prasann/devloss_grid_data/ "$DST_ROOT/data/" && echo "data synced"

# 2. checkpoints, driven by the manifest (ckpt_src = /net path, ckpt_format = hf_serving|distcp)
$PY - "$MANIFEST" "$DST_ROOT" "${ONLY:-}" <<'PYEOF'
import json, os, subprocess, sys, time
man, dst_root, only = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3].split()
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[1]))))
jobs = {}
for name, t in man["tasks"].items():
    if t.get("dropped") or not t.get("ckpt_src"):
        continue
    if only and name not in only:
        continue
    jobs[t["ckpt_src"]] = (t["ckpt_format"], t["ckpt"])
b = man["base_ckpt"]
jobs[b["src"]] = ("distcp", b["path"])
for src, (fmt, dst) in jobs.items():
    t0 = time.time()
    if fmt == "distcp":
        # src is .../model_and_optim (or its parent); gate on the distcp completion sentinel
        mo = src if src.endswith("model_and_optim") else f"{src}/model_and_optim"
        if not os.path.exists(f"{mo}/.metadata"):
            print(f"!! {src}: no .metadata sentinel, skipping", flush=True); continue
        os.makedirs(dst, exist_ok=True)
        r = subprocess.run(["rsync", "-a", mo + "/", f"{dst}/model_and_optim/"])
        cfg = os.path.join(os.path.dirname(mo), "config.json")
        if os.path.exists(cfg):
            subprocess.run(["rsync", "-a", cfg, f"{dst}/config.json"])
        ok = r.returncode == 0 and os.path.exists(f"{dst}/model_and_optim/.metadata")
    else:  # hf_serving -> text-only bf16 export, streamed from /net
        r = subprocess.run([sys.executable, f"{repo}/debug/devloss_grid/make_text_export.py", "--src", src, "--dst", dst])
        ok = r.returncode == 0 and os.path.exists(f"{dst}/.text_export_done")
    print(f"{'OK' if ok else 'FAIL'} {fmt:10s} {src} -> {dst} ({time.time()-t0:.0f}s)", flush=True)
PYEOF
echo "== done ($(date))"; du -sh "$DST_ROOT"/* 2>/dev/null
