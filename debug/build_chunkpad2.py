"""Build the WITHIN-CHUNK twin of the varied free-pad shard.

The two analysis knobs the experiment needs are:
  free60v  : +N tokens with the FREE role  (outside every box -> attend/attended globally)
  chunkpad : +N tokens INSIDE the chunks   (never FREE -> confined by the chunked mask)

The existing within-chunk knob (`repeat_doc_text`) duplicates each claim's text VERBATIM, which trips
the same repeated-text collapse that broke `free_pad_repeat` (see records/free-pad-probe-is-confounded.md).
So build the control the same way free60v was built: VARIED, content-neutral filler, matched budget --
just placed inside each <|box_start|>..<|box_end|> span instead of after it.

Matched: ~800 added tokens, varied, content-neutral, no answer leakage. Differs ONLY in the role the
added tokens carry. That isolates FREE-position capacity from "more tokens / more compute".
"""
import numpy as np, json, os
from transformers import AutoTokenizer

SRC = "/scratch/users/prasann/longctx_sft_qwen/contra_n100_v2_base"
OUT = "/scratch/users/prasann/longctx_sft_qwen/contra_n100_v2_chunkpad2"
EOS, BS, BE = 151643, 151648, 151649
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# One short, DISTINCT, content-neutral clause per chunk (100 chunks -> ~800 tokens total, matching
# free60v's 813). Neutral: says nothing about the claim, so it cannot leak or help.
ADJ = ["noted", "logged", "recorded", "filed", "indexed", "catalogued", "registered", "archived",
       "tabulated", "annotated"]
SRCW = ["record", "entry", "item", "listing", "note", "line", "row", "slip", "card", "page"]


def clause(i: int) -> str:
    return f" This claim was {ADJ[i % len(ADJ)]} in the {SRCW[(i // len(ADJ)) % len(SRCW)]} during the earlier review pass."


t = np.asarray(np.memmap(f"{SRC}/token_ids_part_000000.npy", dtype=np.uint32, mode="r"), dtype=np.int64)
m = np.asarray(np.memmap(f"{SRC}/labels_mask_part_000000.npy", dtype=bool, mode="r"))
meta = json.load(open(f"{SRC}/metadata.json"))
eos = np.where(t == EOS)[0]
pads = [tok.encode(clause(i), add_special_tokens=False) for i in range(100)]
print(f"within-chunk filler: 100 distinct clauses, {sum(len(p) for p in pads)} tokens total")
print(f"  first: {clause(0)!r}\n  last : {clause(99)!r}")

ti, mi = [], []
start = 0
for e in eos:
    ex, exm = list(t[start : e + 1]), list(m[start : e + 1])
    start = e + 1
    out_i, out_m = [], []
    ci = 0
    for tid, msk in zip(ex, exm):
        if tid == BE:  # insert the filler just BEFORE the closing marker -> inside chunk ci
            p = pads[ci % 100]
            out_i.extend(p); out_m.extend([False] * len(p))
            ci += 1
        out_i.append(tid); out_m.append(msk)
    assert len(out_i) == len(out_m) and out_i[-1] == EOS
    ti.extend(out_i); mi.extend(out_m)

ti = np.asarray(ti, dtype=np.uint32); mi = np.asarray(mi, dtype=bool)
os.makedirs(OUT, exist_ok=True)
ti.tofile(f"{OUT}/token_ids_part_000000.npy")   # RAW, headerless
mi.tofile(f"{OUT}/labels_mask_part_000000.npy")
lens = np.diff(np.concatenate([[-1], np.where(ti == EOS)[0]]))
meta.update(num_tokens=int(len(ti)), num_loss_tokens=int(mi.sum()), free_pad_repeat=0,
            within_chunk_pad="varied", max_example_len=int(lens.max()),
            min_example_len=int(lens.min()), derived_from=SRC)
json.dump(meta, open(f"{OUT}/metadata.json", "w"), indent=2)
print(f"wrote {OUT}: tokens={len(ti):,} examples={len(lens)} max_len={lens.max()} "
      f"box_start={(ti==151648).sum()} box_end={(ti==151649).sum()} loss={int(mi.sum())}")
ex0 = ti[: int(np.where(ti == EOS)[0][0]) + 1]
s = tok.decode(ex0[:520])
print("  head:", repr(s[380:520]))
