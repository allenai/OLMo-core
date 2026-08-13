"""
End-to-end smoke test for the HiLS-Attention runtime on a GPU node.

Answers, in order, the questions that decide whether the model can go through our eval suite at
all -- and prints enough per-stage timing to size the real jobs:

1. Do ``tilelang`` / ``veomni`` / the HiLS modeling code import at all in this image?
2. Does the released checkpoint load, and with which attention implementation for the dense
   layers? (config.json asks for ``flash_attention_3``, which is often absent.)
3. Does a SHORT generate produce sane text -- i.e. did the weights actually wire up? A miswired
   load produces fluent-looking garbage, so this prints the continuation for a human to read
   rather than only checking that no exception was raised.
4. Does a LONG prefill work, and how slow is it? This is what decides whether the 32k base ladder
   and the 64k/128k xlong rungs are affordable, and whether they OOM on one 80GB card.

Run inside a gantry GPU job::

    source src/scripts/train/memexpress/hils_eval/hils_env_setup.sh
    PYTHONPATH=src/scripts python src/scripts/train/memexpress/hils_eval/smoke_test_hils.py \\
        --model /weka/oe-training-default/amandab/hf_models/tencent__HiLS-Attention-7B
"""

import argparse
import os
import sys
import time

# A HiLS long prefill is the point of the model; keep the probe text small and repeat it, so the
# prompt length is set by --prefill-tokens rather than by how much prose is pasted here.
FILLER = (
    "The archive catalogues routine maintenance reports from a regional rail network. "
    "Inspectors record track gauge, ballast condition, and signal timing on each pass. "
)
NEEDLE = "\n\nThe inspection code for the Brentwood viaduct is 74192.\n\n"
QUESTION = "Question: What is the inspection code for the Brentwood viaduct?\nAnswer:"

PROMPTS = [
    "The capital of France is",
    "In one sentence, the difference between a list and a tuple in Python is",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="local (weka-staged) checkpoint dir.")
    ap.add_argument("--tokenizer", default="", help="default: the model dir itself.")
    ap.add_argument(
        "--attn",
        default="",
        help="dense-layer attention impl. Empty = try flash_attention_3, then _2, then sdpa.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument(
        "--prefill-tokens",
        type=int,
        default=8192,
        help="long-prefill probe length in tokens; 0 skips the probe.",
    )
    ap.add_argument("--hils-repo", default="", help="default: $HILS_REPO.")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    from ctc_eval.lib.hils_loader import is_hils_checkpoint, load_hils_model

    device = torch.device("cuda:0")
    print(f"=== HiLS smoke test | model={args.model}", flush=True)
    print(f"    torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}", flush=True)
    if not is_hils_checkpoint(args.model):
        print(f"WARNING: {args.model} is not a HiLS checkpoint (model_type has no 'hils')")

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"    tokenizer: vocab={len(tok)} eos={tok.eos_token_id} pad={tok.pad_token_id}", flush=True)

    # ---- stage 2: load, falling back through attention implementations ----------------------
    # config.json ships flash_attention_3; FA3 is a separate build and is usually not present.
    # The fallbacks only affect the interleaved DENSE layers (the sparse HiLS path is tilelang
    # either way), so falling back changes speed, not what is being measured.
    candidates = [args.attn] if args.attn else ["flash_attention_3", "flash_attention_2", "sdpa"]
    model = None
    for attn in candidates:
        t0 = time.time()
        try:
            model = load_hils_model(
                args.model, device=device, attn_implementation=attn, repo=args.hils_repo or None
            )
            print(f"    LOADED with attn_implementation={attn} in {time.time() - t0:.0f}s", flush=True)
            break
        except Exception as e:  # noqa: BLE001 -- we are probing which impls this image supports
            print(f"    attn={attn} failed: {type(e).__name__}: {e}", flush=True)
    if model is None:
        print("FAIL: could not load the model with any attention implementation")
        return 1
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    params={n_params / 1e9:.2f}B dtype={next(model.parameters()).dtype}", flush=True)

    # ---- stage 3: short generate ------------------------------------------------------------
    ok = True
    for prompt in PROMPTS:
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True
            )
        cont = tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"\n  [{time.time() - t0:5.1f}s] {prompt!r}\n    -> {cont!r}", flush=True)
        if not cont.strip():
            ok = False

    # ---- stage 4: long prefill --------------------------------------------------------------
    # Uses a needle so the probe reports RETRIEVAL at length, not just "it did not crash". A model
    # whose long-context path is miswired still emits fluent text; it just cannot find the code.
    if args.prefill_tokens > 0:
        filler_ids = len(tok(FILLER, add_special_tokens=False)["input_ids"])
        reps = max(1, args.prefill_tokens // max(1, filler_ids))
        half = reps // 2
        long_prompt = FILLER * half + NEEDLE + FILLER * (reps - half) + QUESTION
        ids = tok(long_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        n_tok = ids["input_ids"].shape[1]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=16, do_sample=False, use_cache=True)
            cont = tok.decode(out[0][n_tok:], skip_special_tokens=True)
            peak = torch.cuda.max_memory_allocated() / 2**30
            found = "74192" in cont
            print(
                f"\n  long prefill: {n_tok} tokens in {time.time() - t0:.1f}s, "
                f"peak {peak:.1f} GiB, needle_found={found}\n    -> {cont!r}",
                flush=True,
            )
            if not found:
                print("    NOTE: needle missed. Expected for a base model at k=1 -- not a load "
                      "failure by itself, but check the text above reads as English.")
        except Exception as e:  # noqa: BLE001
            print(f"\n  long prefill FAILED at {n_tok} tokens: {type(e).__name__}: {e}", flush=True)
            ok = False

    print(f"\n=== smoke test {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
