#!/bin/bash
# On-node half of the CTC SFT data build, for Beaker (CPU-only).
#
# The build is pure CPU and embarrassingly parallel across (task, bucket), so it wants a wide box
# rather than a GPU: ~6 h serially on 8 cores, ~20 min on a wide one. Runs with --gpus 0.
#
# The `ctc` package is NOT in this branch -- it lives in the public CTC release repo, so it is pip
# installed from there rather than PYTHONPATH'd out of a sibling clone that Beaker cannot see.
# Output goes straight to weka so training reads it without a staging hop.
set -uo pipefail
echo "=== HOST=$(hostname) CPUS=$(nproc) START=$(date -u '+%F %T')Z ==="
REPO=$(find / -maxdepth 3 -iname pyproject.toml 2>/dev/null | grep -v /opt/conda | grep -v /root/.cache | head -1 | xargs -r dirname)
[ -n "$REPO" ] || { echo "FATAL: no cloned repo"; exit 1; }

# `python -m pip`, not bare `pip`: gantry runs a uv venv, and a bare `pip` can resolve to a
# different interpreter than `python` -- the install then "succeeds" into an environment the job
# never imports from. Errors are NOT suppressed here; the first attempt hid its own cause behind
# `--quiet | tail -3` and reported only "did not install".
echo "--- python: $(command -v python) | pip module: $(python -m pip --version) ---"
python -m pip install "huggingface_hub" \
  "git+https://github.com/PrasannS/corpustaskcomplexity.git#subdirectory=ctc" 2>&1 | tail -25
python -c "import ctc.data.cli, huggingface_hub; print('ctc + hub OK')" \
  || { echo "FATAL: ctc package did not install (see the pip output above)"; exit 1; }

OUT="${OUT:-/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_sft_sets}"
mkdir -p "$OUT"
PY=python CTC_SRC=/nonexistent REPO="$REPO" NPROC="${NPROC:-$(nproc)}" \
  BUCKETS="${BUCKETS:-2k 4k 8k 16k 32k}" TOK_PER_BUCKET="${TOK_PER_BUCKET:-20000000}" \
  bash "$REPO/src/scripts/data/hybridish/build_ctc_sft_sets.sh" "${SET:-set-a}" "$OUT"
rc=$?
echo "=== DONE rc=$rc $(date -u '+%F %T')Z ==="
exit $rc
