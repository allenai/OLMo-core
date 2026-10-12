import time

import vllm

hf = "/data/prasann/ctc_suite/hf_exports_4b/ctc-s5-contra-full-4b"
print("=== canonical recipe (no gdn_prefill_backend); loading", hf, flush=True)
t = time.time()
llm = vllm.LLM(
    model=hf,
    max_model_len=4096,
    enforce_eager=True,
    hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]},
    gpu_memory_utilization=0.85,
)
print(f"=== LOADED in {time.time()-t:.1f}s", flush=True)
t = time.time()
out = llm.generate(
    ["The capital of France is"], vllm.SamplingParams(max_tokens=16, temperature=0.0)
)
print(f"=== GENERATED in {time.time()-t:.1f}s: {out[0].outputs[0].text!r}", flush=True)
