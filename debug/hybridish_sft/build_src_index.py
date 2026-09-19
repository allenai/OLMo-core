"""Write the shard-instance -> source-row map for an ALREADY-BUILT shard dir.

Recovers what the converter threw away: it drops rows over --max-seq-len and records only a count,
so nothing on disk says which source rows a shard instance came from. Re-runs the converter's own
build_prompt + tokenize_completion over the source file, applies the same filter, and records the
surviving row indices. Verified in rebuild_index_map.py: the reconstructed lengths match the shard
instance lengths element-for-element, so this map is exact and not a heuristic.

New builds get the sidecar for free (the converter now writes it); this is for shards already on
disk. Usage::

    python debug/hybridish_sft/build_src_index.py --shards <dir> [--workers 6]
"""
import argparse, json, os, sys, time
from multiprocessing import Pool

sys.path.insert(0, "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src/scripts/data/hybridish")
sys.path.insert(0, "/scratch/users/prasann/hyb_sft/ctcsrc")

_G = {}

def _init(meta):
    from transformers import AutoTokenizer
    from convert_ctc_to_sft_completion import spec_name_for, tokenize_completion
    from olmo_core.data.corpus_reasoning_prompts import build_prompt
    _G.update(meta=meta, tok=AutoTokenizer.from_pretrained(meta["tokenizer"]),
              spec_name_for=spec_name_for, tokenize_completion=tokenize_completion,
              build_prompt=build_prompt)

def _one(item):
    """-> (row_index, n_tokens) with n_tokens None when the converter would have dropped it."""
    j, line = item
    m = _G["meta"]
    try:
        ex = json.loads(line)
        prompt, answer = _G["build_prompt"](
            ex, task=_G["spec_name_for"](ex.get("_task")),
            query_position=m["query_position"], use_alpaca=True,
            cot_mode=ex.get("_cot_mode", "none"),
        )
        r = _G["tokenize_completion"](_G["tok"], prompt, answer, m["eos_token_id"],
                                      m["train_on_eos"])
    except Exception:
        return (j, None)
    if r is None:
        return (j, None)
    n = int(r[0].size)
    return (j, n if m["min_tokens"] <= n <= m["max_seq_len"] else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--src-jsonl", default=None, help="default: the shards' own input_jsonl")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=16)
    a = ap.parse_args()

    meta = json.load(open(f"{a.shards}/metadata.json"))
    meta.setdefault("min_tokens", 0)
    src = a.src_jsonl or meta["input_jsonl"][0]
    t0 = time.time()
    print(f"[read] {src}", flush=True)
    with open(src) as f:
        lines = f.readlines()
    print(f"[read] {len(lines):,} rows in {time.time()-t0:.0f}s", flush=True)

    kept, dropped, done = [], [], 0
    t1 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(meta,)) as pool:
        for j, n in pool.imap(_one, enumerate(lines), chunksize=a.chunk):
            (kept if n is not None else dropped).append(j)
            done += 1
            if done in (1, 2, 5, 10, 100) or done % 1000 == 0:
                el = time.time() - t1
                print(f"[tok] {done:,}/{len(lines):,}  kept {len(kept):,} dropped {len(dropped):,} "
                      f"  {el:.0f}s elapsed, ETA {el/done*(len(lines)-done):.0f}s", flush=True)
    kept.sort(); dropped.sort()

    exp = meta["num_instances"]
    ok = len(kept) == exp
    print(f"\n[check] kept {len(kept):,} vs metadata num_instances {exp:,} -> "
          f"{'MATCH' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        raise SystemExit("reconstruction disagrees with the shard metadata; do not trust this map")

    out = f"{a.shards}/src_index.json"
    json.dump({"note": "shard instance i was built from source row src_index[i]",
               "src_jsonl": src, "num_instances": len(kept),
               "src_index": kept, "dropped": dropped}, open(out, "w"))
    print(f"[wrote] {out}  ({len(dropped):,} source rows dropped)", flush=True)

if __name__ == "__main__":
    main()
