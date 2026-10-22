"""
Fetch CTC-suite eval rows from the HF dataset into the unified-JSONL layout the native docchunk
ladder reads (``eval_lc_native_docchunk_ladder.py`` via ``load_unified_examples``).

Writes ``<out>/suite/<subset>/rung_<tokens>.jsonl`` (the same naming olmo-eval's
``CTC_SUITE_DATA_ROOT`` uses, incl. ``contradiction_iid``'s r2k -> 2560 alias) plus
``<out>/suite/MANIFEST.json`` (dataset revision, rows and sha256 per file). Rows are the first
``--rows`` of each parquet, read at the revision olmo-eval pins (``HF_REVISION``), so the ladder
grades the same rows olmo-eval does. Each row is the parquet record as-is (as olmo-eval and
``recipes/eval/setA_audit`` read it), except that ``meta`` / ``_meta`` stored as JSON strings are
decoded back to dicts.

Small I/O, stdlib + huggingface_hub + pyarrow; fine on a login node:

    python3 research/token_routing/ds64_fast8k/fetch_suite_eval_rows.py --out /data/prasann/tdlr_eval
"""

import argparse
import hashlib
import json
import os

HF_DATASET = "PrasannSinghal/ctc-suite-eval"
#: olmo-eval ctc_suite.HF_REVISION (prasann/ctc-suite-flopdrop): the commit every suite number reads.
HF_REVISION = "04ab86006124b3f34c566f44bbc82d4fec98fcda"
#: the six setA tasks the ladder grades through the ctc spec (ladder task == HF subset, except none)
NEW_SUBSETS = ["hotpotqa", "qdmatch_nq", "xabsence", "reorder", "strmatch", "grouping"]
#: the suite rows of the four tasks the Lambda screen already runs (from other eval files)
EXISTING_SUBSETS = ["nq", "outlier", "oolong", "contradiction_iid"]
RUNG_TOKENS = {"r2k": 2048, "r4k": 4096, "r8k": 8192, "r16k": 16384, "r32k": 32768}
#: olmo-eval ROSTER rung_alias: the IID contradiction ladder starts one step up at r2k
RUNG_ALIAS = {("contradiction_iid", "r2k"): 2560}


def _normalize(ex: dict) -> dict:
    """The HF parquet stores dict-valued fields (outlier ``meta``, ``_meta``) as JSON strings."""
    for key in ("meta", "_meta"):
        if isinstance(ex.get(key), str):
            try:
                ex[key] = json.loads(ex[key])
            except ValueError:
                pass
    return ex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output root; files land under <out>/suite/")
    ap.add_argument("--subsets", nargs="+", default=NEW_SUBSETS + EXISTING_SUBSETS)
    ap.add_argument("--rungs", nargs="+", default=["r2k", "r8k"])
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--revision", default=HF_REVISION)
    ap.add_argument("--cache", default=None, help="parquet download dir (default <out>/.hf_parquet)")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    cache = args.cache or os.path.join(args.out, ".hf_parquet")
    root = os.path.join(args.out, "suite")
    manifest = {"dataset": HF_DATASET, "revision": args.revision, "rows_requested": args.rows,
                "files": {}}
    for sub in args.subsets:
        for rung in args.rungs:
            tokens = RUNG_ALIAS.get((sub, rung), RUNG_TOKENS[rung])
            pq_path = hf_hub_download(HF_DATASET, f"data/{sub}/{rung}.parquet", repo_type="dataset",
                                      revision=args.revision, local_dir=cache)
            pf = pq.ParquetFile(pq_path)
            rows = []
            for batch in pf.iter_batches(batch_size=min(args.rows, 256)):
                rows.extend(batch.to_pylist())
                if len(rows) >= args.rows:
                    break
            rows = [_normalize(r) for r in rows[: args.rows]]
            os.makedirs(os.path.join(root, sub), exist_ok=True)
            out = os.path.join(root, sub, f"rung_{tokens}.jsonl")
            with open(out, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
            rel = os.path.relpath(out, root)
            manifest["files"][rel] = {"subset": sub, "rung": rung, "rows": len(rows),
                                      "parquet_rows": pf.metadata.num_rows, "sha256": sha}
            flag = "" if len(rows) >= 500 else f"  (eval_size={len(rows)} < 500)"
            print(f"{rel}: {len(rows)} rows of {pf.metadata.num_rows}{flag}", flush=True)
    with open(os.path.join(root, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(manifest['files'])} files under {root}")


if __name__ == "__main__":
    main()
