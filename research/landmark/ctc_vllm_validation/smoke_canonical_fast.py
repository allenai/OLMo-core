import sys
import time

print("=== python:", sys.executable, flush=True)
import vllm

print("=== vllm version:", vllm.__version__, flush=True)

# Check for fla / causal_conv1d availability in this venv (diagnostic only)
for pkg in ["fla", "causal_conv1d", "mamba_ssm"]:
    try:
        m = __import__(pkg)
        print(f"=== {pkg} OK", getattr(m, "__version__", "?"), flush=True)
    except Exception as e:
        print(f"=== {pkg} MISSING:", e, flush=True)

hf = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/data/prasann/ctc_suite/hf_exports_4b/ctc-s5-contra-full-4b"
)
print("=== canonical recipe (NO gdn_prefill_backend); loading", hf, flush=True)
t = time.time()
llm = vllm.LLM(
    model=hf,
    max_model_len=4096,
    enforce_eager=True,
    hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]},
    gpu_memory_utilization=0.5,
)
print(f"=== LOADED in {time.time()-t:.1f}s", flush=True)
t = time.time()
out = llm.generate(
    ["The capital of France is"], vllm.SamplingParams(max_tokens=16, temperature=0.0)
)
print(f"=== GENERATED in {time.time()-t:.1f}s: {out[0].outputs[0].text!r}", flush=True)
