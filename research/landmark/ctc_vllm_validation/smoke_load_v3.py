"""Minimal make-or-break vLLM load+generate smoke for the 4B Qwen3.5 GDN-hybrid.

Uses the corpus_reasoning canonical recipe against the v3 serving copy:
  - serving copy = VL-wrapper config (model_type=qwen3_5) around our text weights
  - hf_overrides architectures=["Qwen3_5ForCausalLM"] -> load LM only, skip vision
  - enforce_eager (Qwen3.5 hybrid), default GDN backend (no forced triton -> no fla dep)

Isolates the load/generate verdict from the (currently hung) build_prefills step.
"""

import time

import vllm

SERVE = "/data/prasann/ctc_suite/vllm_serving_4b_v3/ctc-s5-contra-full-4b"


def main():
    t0 = time.time()
    llm = vllm.LLM(
        model=SERVE,
        max_model_len=4096,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]},
        # Text-only serving of the multimodal Qwen3.5 wrapper: tell vLLM no image/video
        # inputs, so it skips the vision-encoder profiling forward (which otherwise hangs
        # on our dummy visual weights) and never routes through the (unused) vision tower.
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    print(f"[smoke] LLM up in {time.time()-t0:.1f}s", flush=True)

    prompts = [
        "The capital of France is",
        "List three colors:",
        "2 + 2 =",
    ]
    sp = vllm.SamplingParams(temperature=0.0, max_tokens=32)
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    print(f"[smoke] generated in {time.time()-t1:.1f}s", flush=True)
    for o in outs:
        print("PROMPT:", repr(o.prompt))
        print("  GEN:", repr(o.outputs[0].text))
    print("[smoke] DONE", flush=True)


if __name__ == "__main__":
    main()
