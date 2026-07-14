import json, glob
for path in sorted(glob.glob("data/absence_eval_gutenberg_n*_k3.jsonl")):
    rows = [json.loads(l) for l in open(path)]
    bad_recon = bad_subseq = removed_leak = 0
    for ex in rows:
        sents = [d["text"] for d in ex["documents"]]          # Version A sentences
        gold = set(ex["gold_doc_indices"])
        kept = [s for i, s in enumerate(sents) if i not in gold]
        vb = ex["queries"][0]                                  # Version B text
        if " ".join(kept) != vb: bad_recon += 1               # B == kept joined?
        if not all(s in sents for s in kept): bad_subseq += 1  # every B sent in A?
        removed = [sents[i] for i in gold]
        if any(r in vb for r in removed): removed_leak += 1     # removed leaked to B?
    print(f"{path}: {len(rows)} ex | B!=kept-join:{bad_recon} | "
          f"B-sent-not-in-A:{bad_subseq} | removed-leaked:{removed_leak}")
