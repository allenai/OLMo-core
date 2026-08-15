#!/bin/bash
# Full end-to-end Beaker/jupiter feasibility pipeline for loading the Qwen3.5-4B GDN-hybrid
# in vLLM 0.25.1, reading the checkpoint straight from weka. Single-venv approach: one fresh
# venv gets vllm==0.25.1 (brings native transformers 5.14.1 w/ Qwen3_5ForCausalLM support),
# then olmo_core/corpus_reasoning are made importable via PYTHONPATH into the cloned repo
# (no `pip install -e .` needed -- pure-python import, avoids clobbering vllm's bundled torch).
#
# Embeds the 4 helper scripts (untracked locally under debug/ctc_vllm_validation/) as heredocs
# since gantry only ships the pushed git commit.
set -uo pipefail
echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
df -h / 2>/dev/null | tail -1

# --- locate the gantry-cloned repo (works whether or not `pip install -e .` already ran) ---
REPO=$(find / -maxdepth 3 -iname pyproject.toml 2>/dev/null | grep -v /opt/conda | grep -v /root/.cache | head -1 | xargs -r dirname)
if [ -z "$REPO" ]; then
  echo "FATAL: could not locate cloned OLMo-core repo (pyproject.toml) under /"; find / -maxdepth 2 2>/dev/null; exit 1
fi
echo "REPO=$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

WORK=/root/vllm_beaker_probe
mkdir -p "$WORK"
VENV="$WORK/venv"

# --- PyPI only publishes a cu13-linked vllm==0.25.1 wheel (needs driver >=~580), but vLLM's
# GitHub release also ships a vllm-0.25.1+cu129 wheel (CUDA 12.9). jupiter's driver reports
# "CUDA Version: 12.8" -- same MAJOR version (12.x) as cu129, so NVIDIA's built-in *minor
# version compatibility* (driver >= the 12.x baseline, no forward-compat package needed)
# should let this run natively, unlike the cu13 (major-version) jump.
CU129_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
echo "=== building vllm venv (cu129 wheel from GitHub releases) $(date '+%F %T') ==="
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "$CU129_WHEEL_URL" 2>&1 | tail -60
echo "=== vllm install done $(date '+%F %T') ==="
"$VENV/bin/python" -c "import vllm, transformers; print('vllm', vllm.__version__); print('transformers', transformers.__version__)"

echo "=== checking torch build pulled in by the cu129 wheel $(date '+%F %T') ==="
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
TORCH_CUDA_TAG=$("$VENV/bin/python" -c "import torch; print(torch.__version__.split('+')[-1])")
echo "TORCH_CUDA_TAG=$TORCH_CUDA_TAG"
if [ "$TORCH_CUDA_TAG" != "cu129" ]; then
  echo "torch is not cu129 (got $TORCH_CUDA_TAG) -- forcing the matching cu129 torch+torchvision+torchaudio TRIAD"
  TORCH_VER=$("$VENV/bin/python" -c "import torch; print(torch.__version__.split('+')[0])")
  # Pin ALL THREE explicitly and in the SAME command -- installing torch alone here and fixing
  # torchvision separately later let pip silently drift torch to a newer release to satisfy an
  # unpinned torchvision install (2.11.0 -> 2.13.0), while torchaudio was never touched and stayed
  # on the old +cu130 build. That 3-way mismatch reintroduced "libcudart.so.13: cannot open shared
  # object file" via torchaudio's compiled extension -- NOT the vllm-wheel ABI wall, just a pip
  # pinning bug. Pinning the triad together with one exact version each avoids any resolver drift.
  "$VENV/bin/pip" install --quiet --force-reinstall \
    "torch==$TORCH_VER" "torchvision==0.26.0" "torchaudio==$TORCH_VER" \
    --index-url https://download.pytorch.org/whl/cu129 2>&1 | tail -60
  "$VENV/bin/python" -c "
import torch, torchvision, torchaudio
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('torchvision', torchvision.__version__)
print('torchaudio', torchaudio.__version__)
"
fi

echo "=== verifying CUDA is usable (cu129 on the 12.8 driver, via minor-version compat) $(date '+%F %T') ==="
"$VENV/bin/python" -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('cuda.is_available()', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    x = torch.randn(4, 4, device='cuda') @ torch.randn(4, 4, device='cuda')
    print('matmul OK', x.shape)
" 2>&1 | tee "$WORK/cuda_cu129_verify.log"
if ! grep -q "matmul OK" "$WORK/cuda_cu129_verify.log"; then
  echo "FATAL: CUDA still not usable with the cu129 wheel -- stopping here (see cuda_cu129_verify.log above)"
  exit 1
fi
echo "=== also verify vllm's own compiled extension imports clean (the actual failure point before) $(date '+%F %T') ==="
"$VENV/bin/python" -c "from vllm.platforms import current_platform; print('vllm platform import OK:', current_platform)"

# transformers' image_utils unconditionally imports torchvision/torchaudio (for the vision/audio
# tower paths, even though our text-only serving skips them via limit_mm_per_prompt) -- the triad
# pin above (torch/torchvision/torchaudio all from the cu129 index, all matched versions) is what
# keeps these compiled extensions loadable. Just verify here, no more reinstalling.
echo "=== verifying transformers + vllm import cleanly on the pinned cu129 triad $(date '+%F %T') ==="
"$VENV/bin/python" -c "
import torch, torchvision, torchaudio
print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'torchaudio', torchaudio.__version__)
from transformers import Qwen3_5ForCausalLM
print('Qwen3_5ForCausalLM import OK')
import vllm
print('vllm', vllm.__version__, 'import OK')
" 2>&1 | tee "$WORK/torchvision_fix_verify.log"
if ! grep -q "Qwen3_5ForCausalLM import OK" "$WORK/torchvision_fix_verify.log"; then
  echo "FATAL: Qwen3_5ForCausalLM still fails to import on the pinned cu129 triad"; exit 1
fi

# --- LOCK the triad for every pip install from here on. Twice now an UNPINNED pip install of
# something else (torchvision alone, then flashinfer-python/flashinfer-cubin with no --index-url)
# silently dragged torch off the cu129 build to a bare PyPI 2.13.0 (cu130-linked) to satisfy some
# transitive requirement, breaking torchvision::nms / Qwen3_5ForCausalLM downstream in a way that
# only surfaced several steps later. A pip constraints file makes any future silent drift a LOUD
# resolver error instead, on every subsequent "$VENV/bin/pip install" in this script.
cat > "$WORK/pin-triad.constraints.txt" <<EOF
torch==2.11.0+cu129
torchvision==0.26.0+cu129
torchaudio==2.11.0+cu129
EOF
export PIP_CONSTRAINT="$WORK/pin-triad.constraints.txt"
echo "PIP_CONSTRAINT=$PIP_CONSTRAINT locking torch/torchvision/torchaudio for all remaining installs"

echo "=== installing flashinfer (GDN prefill kernel backend) $(date '+%F %T') ==="
# NOT pulled automatically by the vllm wheel install above -- pin to the version validated
# elsewhere in this investigation (0.6.13, same as the cubbins/local recipe).
"$VENV/bin/pip" install --quiet "flashinfer-python==0.6.13" "flashinfer-cubin==0.6.13" 2>&1 | tail -40
"$VENV/bin/python" -c "import flashinfer; print('flashinfer', flashinfer.__version__)"

echo "=== CUDA_HOME / nvcc: REAL apt system toolkit (coherent nvcc+headers), NOT pip metapackage $(date '+%F %T') ==="
# Root cause of the earlier "CUDA compiler and CUDA toolkit headers are incompatible" flashinfer
# JIT failure: a pip toolkit metapackage (cuda-toolkit==13.0.2, or piecemeal nvidia-cuda-nvcc)
# can resolve nvcc and its paired cuda.h from DIFFERENT sub-releases (nvcc reporting one
# __CUDACC_VER while cuda.h's CUDA_VERSION macro reports another) -- flashinfer's
# cuda_toolkit.h:41 guard hard-asserts these are numerically equal. A real `apt-get install
# cuda-toolkit-12-8` from NVIDIA's own repo ships nvcc + cccl + headers as ONE coherent release,
# which is what avoids the incompatible-headers error (this is the whole reason cubbins -- which
# has a real system CUDA install, not a pip one -- compiles flashinfer's GDN kernel fine).
. /etc/os-release; UBU_TAG="ubuntu${VERSION_ID//./}"
echo "distro=$UBU_TAG"
apt-get update -qq
apt-get install -y -qq wget gnupg ca-certificates >/dev/null
if ! dpkg -l cuda-keyring >/dev/null 2>&1; then
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${UBU_TAG}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb >/dev/null
fi
apt-get update -qq
apt-get install -y -qq cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-crt-12-8 cuda-nvrtc-dev-12-8 2>&1 | tail -60
CUDA_HOME=/usr/local/cuda-12.8
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "FATAL: apt cuda-toolkit-12-8 did not produce $CUDA_HOME/bin/nvcc"
  ls -d /usr/local/cuda* 2>/dev/null
  exit 1
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME  nvcc=$(which nvcc)"
"$CUDA_HOME/bin/nvcc" --version
echo "=== coherence check: nvcc __CUDACC_VER vs cuda.h CUDA_VERSION $(date '+%F %T') ==="
echo | "$CUDA_HOME/bin/nvcc" -x cu -E -dM - 2>/dev/null | grep -E "CUDACC_VER"
grep -E "CUDA_VERSION" "$CUDA_HOME/include/cuda.h" | head -3

echo "=== light olmo_core/corpus_reasoning deps $(date '+%F %T') ==="
"$VENV/bin/pip" install --quiet dataclass-extensions cached-path filelock packaging pyyaml rich \
  safetensors importlib_resources bettermap pandas huggingface_hub 2>&1 | tail -20
# GatedDeltaNet.__init__ hard-asserts has_fla() (no pure-torch fallback at construction time,
# unlike the attention blocks) -- required even just to build the model skeleton for export.
echo "=== installing flash-linear-attention (fla) -- required for GatedDeltaNet construction $(date '+%F %T') ==="
"$VENV/bin/pip" install --quiet "flash-linear-attention==0.4.1" 2>&1 | tail -60

echo "=== import verification $(date '+%F %T') ==="
"$VENV/bin/python" -c "
import olmo_core, corpus_reasoning, vllm
from transformers import Qwen3_5ForCausalLM
from olmo_core.nn.attention.flash_linear_attn_api import has_fla
print('ALL IMPORTS OK: olmo_core=', olmo_core.__file__)
print('corpus_reasoning=', corpus_reasoning.__file__)
print('has_fla()=', has_fla())
assert has_fla(), 'fla still not importable after pip install'
" || { echo "FATAL: import verification failed"; exit 1; }

# --- env for HF downloads (network needed for the base Qwen/Qwen3.5-4B-Base config+weights) ---
export HF_HOME="$WORK/hf_home"
mkdir -p "$HF_HOME"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
unset TRANSFORMERS_CACHE

CKPT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/ctc-4b-grouping-full-20260719T225805-0700
HF_EXPORT="$WORK/hf_export"
BASE_MODEL_ID="Qwen/Qwen3.5-4B-Base"

echo "=== export olmo distcp -> HF text $(date '+%F %T') ==="
"$VENV/bin/python" "$REPO/src/corpus_reasoning/train/export_olmo_to_hf.py" \
  --save-folder "$CKPT" --ckpt "$CKPT" \
  --hf-out "$HF_EXPORT" --base-model "$BASE_MODEL_ID"
rc=$?
if [ $rc -ne 0 ] || [ ! -f "$HF_EXPORT/config.json" ]; then
  echo "FATAL: export step failed (rc=$rc) or $HF_EXPORT/config.json missing"; exit 1
fi
echo "=== export done $(date '+%F %T') ==="

echo "=== resolve base VL snapshot (reuses export's HF cache) $(date '+%F %T') ==="
BASE_SNAP=$("$VENV/bin/python" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$BASE_MODEL_ID'))
")
echo "BASE_SNAP=$BASE_SNAP"
ls "$BASE_SNAP" | head -20

# --- write the 3 serving-copy helper scripts + smoke script inline (untracked locally) ---
mkdir -p "$WORK/scripts"

cat > "$WORK/scripts/make_vllm_serving_copy.py" <<'PYEOF'
import argparse, glob, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-export", required=True)
    ap.add_argument("--base-snapshot", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    text_cfg = json.load(open(os.path.join(args.hf_export, "config.json")))
    base_cfg = json.load(open(os.path.join(args.base_snapshot, "config.json")))
    wrapper = dict(base_cfg)
    wrapper["architectures"] = ["Qwen3_5ForCausalLM"]
    wrapper["text_config"] = text_cfg
    wrapper["tie_word_embeddings"] = bool(text_cfg.get("tie_word_embeddings", False))
    os.makedirs(args.out, exist_ok=True)
    for f in glob.glob(os.path.join(args.hf_export, "*")):
        name = os.path.basename(f)
        if name == "config.json":
            continue
        dst = os.path.join(args.out, name)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(f), dst)
    for name in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.json"):
        src = os.path.join(args.base_snapshot, name)
        dst = os.path.join(args.out, name)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(wrapper, f, indent=2)
    print(f"[serving-copy] {args.out}: tie_word_embeddings={text_cfg.get('tie_word_embeddings')}", flush=True)

if __name__ == "__main__":
    main()
PYEOF

cat > "$WORK/scripts/make_vl_weights.py" <<'PYEOF'
import argparse, json, os
from safetensors.torch import load_file, save_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-export", required=True)
    ap.add_argument("--base-snapshot", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    state = load_file(os.path.join(args.hf_export, "model.safetensors"))
    out = {}
    for k, v in state.items():
        if k.startswith("model."):
            out["model.language_model." + k[len("model."):]] = v
        else:
            out[k] = v
    idx_path = os.path.join(args.base_snapshot, "model.safetensors.index.json")
    n_vis = 0
    if os.path.exists(idx_path):
        idx = json.load(open(idx_path))["weight_map"]
        shards = sorted({f for k, f in idx.items() if k.startswith("model.visual.")})
        for shard in shards:
            shard_path = os.path.join(args.base_snapshot, shard)
            if not os.path.exists(shard_path):
                print(f"[vl-weights] missing shard {shard}; skipping", flush=True)
                continue
            st = load_file(shard_path)
            for k, v in st.items():
                if k.startswith("model.visual."):
                    out[k] = v
                    n_vis += 1
    else:
        print(f"[vl-weights] no index.json at {idx_path}; skipping vision graft", flush=True)
    dst = os.path.join(args.out_dir, "model.safetensors")
    if os.path.islink(dst):
        os.unlink(dst)
    save_file(out, dst, metadata={"format": "pt"})
    print(f"[vl-weights] wrote {len(out)} tensors ({n_vis} visual) -> {dst}", flush=True)

if __name__ == "__main__":
    main()
PYEOF

cat > "$WORK/scripts/add_dummy_visual.py" <<'PYEOF'
import argparse, json, os
import torch
from safetensors.torch import load_file, save_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.serve, "config.json")))
    vcfg = cfg.get("vision_config")
    assert vcfg is not None, "serving config has no vision_config"
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    vision_config = Qwen3_5VisionConfig(**vcfg)
    torch.manual_seed(0)
    with torch.device("cpu"):
        vision = Qwen3_5VisionModel(vision_config).to(torch.bfloat16)
    vstate = {f"visual.{k}": v.contiguous() for k, v in vision.state_dict().items()}
    print(f"[dummy-visual] built {len(vstate)} visual.* tensors", flush=True)
    model_path = os.path.join(args.serve, "model.safetensors")
    state = load_file(model_path)
    n_text = len(state)
    added = 0
    for k, v in vstate.items():
        if k not in state:
            state[k] = v
            added += 1
    print(f"[dummy-visual] text={n_text} visual_added={added} total={len(state)}", flush=True)
    if os.path.islink(model_path):
        os.unlink(model_path)
    save_file(state, model_path, metadata={"format": "pt"})
    print(f"[dummy-visual] wrote {model_path}", flush=True)

if __name__ == "__main__":
    main()
PYEOF

cat > "$WORK/scripts/smoke_load_beaker.py" <<'PYEOF'
import argparse, time
import vllm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.5)
    args = ap.parse_args()
    t0 = time.time()
    llm = vllm.LLM(
        model=args.serve,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_mem_util,
        hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]},
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    load_s = time.time() - t0
    print(f"[smoke] LLM up in {load_s:.1f}s", flush=True)
    prompts = [
        "The capital of France is",
        "List three colors:",
        "2 + 2 =",
        "The theory of relativity was developed by",
    ]
    sp = vllm.SamplingParams(temperature=0.0, max_tokens=32)
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t1
    print(f"[smoke] generated in {gen_s:.1f}s", flush=True)
    for o in outs:
        print("PROMPT:", repr(o.prompt))
        print("  GEN:", repr(o.outputs[0].text))
    print(f"[smoke] SUMMARY load_s={load_s:.1f} gen_s={gen_s:.1f}", flush=True)
    print("[smoke] DONE", flush=True)

if __name__ == "__main__":
    main()
PYEOF

SERVE="$WORK/serving"
echo "=== building vLLM serving copy $(date '+%F %T') ==="
"$VENV/bin/python" "$WORK/scripts/make_vllm_serving_copy.py" --hf-export "$HF_EXPORT" --base-snapshot "$BASE_SNAP" --out "$SERVE"
"$VENV/bin/python" "$WORK/scripts/make_vl_weights.py" --hf-export "$HF_EXPORT" --base-snapshot "$BASE_SNAP" --out-dir "$SERVE"

HAS_VISUAL=$("$VENV/bin/python" -c "
from safetensors import safe_open
f = safe_open('$SERVE/model.safetensors', framework='pt')
print(any(k.startswith('visual.') for k in f.keys()))
")
echo "HAS_VISUAL=$HAS_VISUAL"
if [ "$HAS_VISUAL" != "True" ]; then
  echo "=== no real vision weights grafted -> synthesizing dummy visual.* $(date '+%F %T') ==="
  "$VENV/bin/python" "$WORK/scripts/add_dummy_visual.py" --serve "$SERVE"
fi

echo "=== vLLM load+generate smoke $(date '+%F %T') ==="
echo "CUDA_HOME=$CUDA_HOME nvcc=$(which nvcc)"
# vLLM's engine-core subprocess uses multiprocessing; on Beaker the fork start-method got
# picked (instead of the spawn olmo-core validated locally), which crashes on re-init'ing
# CUDA in the forked child. Force spawn explicitly rather than relying on auto-detection.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
"$VENV/bin/python" -u "$WORK/scripts/smoke_load_beaker.py" --serve "$SERVE" --max-model-len 4096 --gpu-mem-util 0.5
rc=$?
echo "=== DONE rc=$rc $(date '+%F %T') ==="
exit $rc
