"""Is the detected oolong train/eval overlap REAL duplication, or a signature artifact?

Contamination was flagged with a cheap signature: query + first 200 chars of the packed document
body. That is a prefix match -- two independently synthesized examples could share a preamble and
opening item by chance, especially at short lengths where the item pool is small. If the signature
over-counts, the "21.6% contaminated" figure is wrong.

For every signature-matched (eval, train) pair this compares the FULL document body and the answer,
and reports the breakdown:
  * exact      -- identical full body AND identical answer  => true duplicate, real contamination
  * same_body  -- identical body, different answer          => still effectively a duplicate prompt
  * prefix_only-- only the first 200 chars agree            => signature artifact, NOT contamination
"""

import glob
import json
from collections import defaultdict

EVAL = "/data/prasann/xlong5/eval/oolong"
POOL = "/data/prasann/xlong5/pools/oolong"  # the ORIGINAL (pre-decontamination) pool


def sig(ex: dict) -> str:
    q = (ex.get("queries") or [""])[0]
    b = ex["documents"][0]["text"] if ex.get("documents") else ""
    return q + "||" + b[:200]


def body(ex: dict) -> str:
    return ex["documents"][0]["text"] if ex.get("documents") else ""


def ans(ex: dict) -> str:
    a = ex.get("answers") or [""]
    return str(a[0])


print("indexing train pool ...", flush=True)
train = defaultdict(list)
n_train = 0
for f in glob.glob(POOL + "/*/*.jsonl"):
    for line in open(f):
        if line.strip():
            ex = json.loads(line)
            train[sig(ex)].append((body(ex), ans(ex)))
            n_train += 1
print(f"  {n_train} train examples, {len(train)} distinct signatures\n", flush=True)

print(f"{'rung':>8}{'n':>7}{'sig_hits':>10}{'exact':>8}{'same_body':>11}{'prefix_only':>13}")
for ctx in (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144):
    p = f"{EVAL}/oolong_test_synth_ctx{ctx}_spliteval.jsonl"
    try:
        rows = [json.loads(x) for x in open(p) if x.strip()]
    except FileNotFoundError:
        continue
    hits = exact = same_body = prefix_only = 0
    for ex in rows:
        cands = train.get(sig(ex))
        if not cands:
            continue
        hits += 1
        b, a = body(ex), ans(ex)
        if any(tb == b and ta == a for tb, ta in cands):
            exact += 1
        elif any(tb == b for tb, _ in cands):
            same_body += 1
        else:
            prefix_only += 1
    lab = f"{ctx // 1024}k"
    print(f"{lab:>8}{len(rows):>7}{hits:>10}{exact:>8}{same_body:>11}{prefix_only:>13}")
print("\nexact + same_body = real contamination; prefix_only = signature artifact")
