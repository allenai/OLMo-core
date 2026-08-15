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
  echo "torch is not cu129 (got $TORCH_CUDA_TAG) -- forcing the matching cu129 torch build"
  TORCH_VER=$("$VENV/bin/python" -c "import torch; print(torch.__version__.split('+')[0])")
  "$VENV/bin/pip" install --quiet --force-reinstall "torch==$TORCH_VER" --index-url https://download.pytorch.org/whl/cu129 2>&1 | tail -40
  "$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
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

# transformers' image_utils unconditionally imports torchvision (for the vision-tower path,
# even though our text-only serving skips it via limit_mm_per_prompt). Whatever torchvision
# landed alongside the original cu130 torch (or none at all) doesn't match the torch we just
# force-reinstalled to cu129, so torchvision's compiled ops (torchvision::nms) fail to
# register -> transformers can't import Qwen3_5ForCausalLM at all. Force-reinstall torchvision
# from the SAME cu129 index so pip resolves the release built against this exact torch.
echo "=== reinstalling torchvision to match torch cu129 $(date '+%F %T') ==="
"$VENV/bin/pip" install --quiet --force-reinstall torchvision --index-url https://download.pytorch.org/whl/cu129 2>&1 | tail -40
"$VENV/bin/python" -c "
import torch, torchvision
print('torch', torch.__version__, 'torchvision', torchvision.__version__)
from transformers import Qwen3_5ForCausalLM
print('Qwen3_5ForCausalLM import OK')
import vllm
print('vllm', vllm.__version__, 'import OK')
" 2>&1 | tee "$WORK/torchvision_fix_verify.log"
if ! grep -q "Qwen3_5ForCausalLM import OK" "$WORK/torchvision_fix_verify.log"; then
  echo "FATAL: Qwen3_5ForCausalLM still fails to import after the torchvision fix"; exit 1
fi

echo "=== CUDA_HOME / nvcc $(date '+%F %T') ==="
"$VENV/bin/pip" install --quiet nvidia-cuda-nvcc-cu12 2>&1 | tail -10
NVCC_PATH=$(find "$VENV" -iname nvcc 2>/dev/null | head -1)
if [ -z "$NVCC_PATH" ]; then
  echo "FATAL: no nvcc found after installing nvidia-cuda-nvcc-cu12"; exit 1
fi
export CUDA_HOME=$(dirname "$(dirname "$NVCC_PATH")")
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME  nvcc=$(which nvcc)"

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
