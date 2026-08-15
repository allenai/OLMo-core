"""Minimal make-or-break vLLM load+generate smoke for the 4B Qwen3.5 GDN-hybrid on Beaker.

Same recipe as debug/ctc_vllm_validation/smoke_load_v3.py, parameterized by --serve so it
can point at the Beaker in-container serving copy instead of the hardcoded horton path.
"""
import argparse
import time

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
