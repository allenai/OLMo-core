"""Generic (any-CTC-suite-task) version of run_vllm_eval.py.

Same two modes (chunked / full) as run_vllm_eval.py, but the stop rule comes from the
prefill pack (written by build_prefills_generic.py) instead of being hardcoded to
contradiction's cot_mode="none" pair-list logic. Supports the stop rules used by the
tasks we sweep post-parity: grouping/outlier="eos", retrieval="newline", oolong="oolong"
(early-stop at a newline once a templated "answer:"/"label:"/... line has been emitted --
mirrors eval_lc_native_docchunk.py's ``should_stop`` for stop_rule="oolong". vLLM has no
stateful per-token stop callback, so this is approximated by generating to the full
decode budget (stop at EOS only) and then truncating client-side at the end of the first
line containing one of those markers -- exactly what the native path would have stopped
at, and avoids the no-EOS-ramble grading corruption the native harness bug caused).
"""

import argparse
import json
import os
import re
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("HF_HOME", "/data/prasann/hf")
os.environ.setdefault("TMPDIR", "/data/prasann/tmp")

_ANSWER_LINE_RE = re.compile(r'(?:answer|label|user|date|month)\s*:.*', re.IGNORECASE)


def truncate_oolong(text):
    """Mirror eval_lc_native_docchunk.py's stop_rule="oolong": keep everything through the
    end of the first line that contains an answer/label/user/date/month marker; drop any
    rambled continuation after it (the exact failure mode the native no-EOS bug produced)."""
    for line in text.split("\n"):
        if _ANSWER_LINE_RE.search(line):
            idx = text.index(line) + len(line)
            return text[:idx]
    return text  # no marker line found -- leave as-is, scorer's _oolong_extract falls back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--prefills", required=True, help="output of build_prefills_generic.py")
    ap.add_argument("--mode", required=True, choices=["chunked", "full"])
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.prefills) as f:
        pack = json.load(f)
    rows = pack["rows"]
    eos_id = pack["eos_token_id"]
    stop_rule = pack["stop_rule"]
    if stop_rule not in ("eos", "newline", "oolong"):
        raise SystemExit(
            f"stop_rule={stop_rule!r} not implemented in run_vllm_eval_generic.py "
            "(only eos/newline/oolong are) -- use the native evaluator for this task."
        )

    if args.mode == "chunked":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from corpus_reasoning.lib import vllm_chunked_patch
        vllm_chunked_patch.install()
        vllm_chunked_patch.set_doc_token_ids(pack["doc_start_id"], pack["doc_end_id"])
        print(f"[driver] patch installed; doc ids = "
              f"{pack['doc_start_id']}/{pack['doc_end_id']}", flush=True)

    from vllm import LLM, SamplingParams, TokensPrompt

    t0 = time.time()
    extra = {}
    if args.mode == "chunked":
        extra = dict(
            num_gpu_blocks_override=128,
            max_num_batched_tokens=2048,
            max_num_seqs=8,
        )
    # 0.35 is the historical default -- chosen so several sweep jobs can share one GPU on the
    # 2k-32k ladder. The 64k/128k length-generalization rungs need a far bigger KV cache to keep
    # any useful concurrency (a single 131k-token sequence is ~4GB of KV on this hybrid), so those
    # jobs raise it via VLLM_GPU_MEM_UTIL. Throughput only -- it does not change any score.
    gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.35"))
    llm = LLM(
        model=args.hf_model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=True,
        gdn_prefill_backend="triton",
        **extra,
        attention_config=({"backend": "FLEX_ATTENTION", "flex_attn_kv_block_size": 32}
                          if args.mode == "chunked" else None),
    )
    load_s = time.time() - t0
    print(f"[driver] LLM up in {load_s:.1f}s", flush=True)
    tok = llm.get_tokenizer()

    sp_kwargs = dict(temperature=0.0, max_tokens=args.max_new_tokens, stop_token_ids=[eos_id])
    if stop_rule == "newline":
        sp_kwargs["stop"] = ["\n"]
    sp = SamplingParams(**sp_kwargs)
    inputs = [TokensPrompt(prompt_token_ids=r["prefill"]) for r in rows]
    t1 = time.time()
    outputs = llm.generate(inputs, sp)
    gen_s = time.time() - t1
    print(f"[driver] generated {len(outputs)} in {gen_s:.1f}s "
          f"({gen_s / max(len(outputs), 1):.2f}s/example)", flush=True)

    responses = {}
    for r, o in zip(rows, outputs):
        text = o.outputs[0].text
        text = text.split("</think>", 1)[1] if "</think>" in text else text
        if stop_rule == "oolong":
            text = truncate_oolong(text)
        responses[str(r["idx"])] = text

    debug = {}
    if args.mode == "chunked":
        from corpus_reasoning.lib import vllm_chunked_patch
        debug = dict(vllm_chunked_patch.get_debug_state())
        print(f"[driver] patch debug state: {debug}", flush=True)
        if not debug.get("applied"):
            print("[driver] *** WARNING: chunked patch NEVER APPLIED -- "
                  "this run is UNMASKED ***", flush=True)

    with open(args.out, "w") as f:
        json.dump({
            "mode": args.mode,
            "hf_model": args.hf_model,
            "prefills": args.prefills,
            "task": pack["task"],
            "eval_size": len(rows),
            "max_new_tokens": args.max_new_tokens,
            "load_seconds": load_s,
            "gen_seconds": gen_s,
            "patch_debug": debug,
            "responses": responses,
        }, f)
    print(f"[driver] wrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
