"""Build a VARIED-filler free-pad shard, to disentangle "more FREE tokens" from "repeated text".

`free_pad_repeat` appends N copies of ONE identical sentence (FREE_PAD_SENTENCE). Empirically that
collapses training even under PLAIN CAUSAL (CE 0.81 vs 0.0008 without it), so it is NOT measuring
FREE-position capacity -- it is measuring what a long block of exactly-repeated text does to the model.

This builds the same FREE-token budget out of DISTINCT sentences. Comparing:
    base       (0 filler)          -> the reference
    free60     (60x same sentence) -> collapses
    free60v    (60 distinct sents) -> if this trains, the collapse is REPETITION, not FREE capacity.
The filler is content-neutral (it names no claim), so it cannot leak the answer.
"""
import numpy as np, json, os
from transformers import AutoTokenizer

SRC = "/scratch/users/prasann/longctx_sft_qwen/contra_n100_v2_base"
OUT = "/scratch/users/prasann/longctx_sft_qwen/contra_n100_v2_free60v"
EOS, IM_END = 151643, 151645
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# 60 distinct, content-neutral sentences -- same register and roughly the same length as
# FREE_PAD_SENTENCE ("Review the claims above carefully before answering. ").
VERBS = ["Review", "Re-read", "Examine", "Inspect", "Consider", "Weigh", "Check", "Study", "Assess", "Revisit"]
OBJS = ["the claims above", "the numbered claims", "the statements listed", "each claim in turn",
        "the corpus above", "every claim given"]
FILLER = []
for i in range(60):
    v = VERBS[i % len(VERBS)]
    o = OBJS[(i // len(VERBS)) % len(OBJS)]
    FILLER.append(f"{v} {o} carefully before answering, step {i + 1}. ")
filler_text = "\n" + "".join(FILLER)
filler_ids = tok.encode(filler_text, add_special_tokens=False)
print(f"varied filler: {len(FILLER)} distinct sentences, {len(filler_ids)} tokens")
print(f"  first: {FILLER[0]!r}\n  last : {FILLER[-1]!r}")

t = np.asarray(np.memmap(f"{SRC}/token_ids_part_000000.npy", dtype=np.uint32, mode="r"), dtype=np.int64)
m = np.asarray(np.memmap(f"{SRC}/labels_mask_part_000000.npy", dtype=bool, mode="r"))
meta = json.load(open(f"{SRC}/metadata.json"))
eos = np.where(t == EOS)[0]

ti, mi = [], []
start = 0
for e in eos:
    ex, exm = list(t[start : e + 1]), list(m[start : e + 1])
    start = e + 1
    # Insert the filler just before the FIRST <|im_end|> (end of the user turn) -> lands after the last
    # <|box_end|>, so every filler token gets the FREE role. Same insertion point as free_pad_repeat.
    j = ex.index(IM_END)
    new_i = ex[:j] + list(filler_ids) + ex[j:]
    new_m = exm[:j] + [False] * len(filler_ids) + exm[j:]
    assert len(new_i) == len(new_m) and new_i[-1] == EOS
    ti.extend(new_i); mi.extend(new_m)

ti = np.asarray(ti, dtype=np.uint32); mi = np.asarray(mi, dtype=bool)
os.makedirs(OUT, exist_ok=True)
ti.tofile(f"{OUT}/token_ids_part_000000.npy")   # RAW, headerless (the loader memmaps these directly)
mi.tofile(f"{OUT}/labels_mask_part_000000.npy")
lens = np.diff(np.concatenate([[-1], np.where(ti == EOS)[0]]))
meta.update(num_tokens=int(len(ti)), num_loss_tokens=int(mi.sum()), free_pad_repeat=60,
            free_pad_mode="varied", max_example_len=int(lens.max()), min_example_len=int(lens.min()),
            derived_from=SRC)
json.dump(meta, open(f"{OUT}/metadata.json", "w"), indent=2)
print(f"wrote {OUT}: tokens={len(ti):,} loss={int(mi.sum())} examples={len(lens)} "
      f"max_len={lens.max()} box_start={(ti==151648).sum()} box_end={(ti==151649).sum()}")
ex0 = ti[: int(np.where(ti == EOS)[0][0]) + 1]
print(f"  tail: {tok.decode(ex0[-320:])[:300]!r}")
