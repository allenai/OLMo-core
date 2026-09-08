"""
Is soft-token (compaction) training actually faster than dense by the compaction factor?

One GPU, Qwen3.5-4B (bf16 params, full activation checkpointing, random init), synthetic
contradiction-shaped rows: 100-token preamble, ~690 marker-wrapped 45-token documents, a 50-token
question and a 20-token answer carrying the labels, 32768 tokens per row. Forward + backward,
warm-up then timed steps, for:

  dense/flash        the trainer's dense recipe (flash_2 backend) on the full 32k row
  dense/torch        same row on the torch SDPA backend (isolates the backend cost)
  soft/torch k=1/3   the ladder's kv33 recipe: compaction, detach, torch backend (what ran)
  soft/torch k=1/3 +logL   the new --st-len-bias path (dense additive bias -> masked SDPA)
  soft/flash k=1/3   compaction on the flash backend (no bias) -- does it run, how fast
  soft/* k=1/12      the kv08 keep fraction
  dense/flash @compacted length   flash dense on a row as long as the compacted one (the ideal)

Reports seconds per step, INPUT tokens per second (the number that decides training cost at a
fixed token budget), the realised compaction ratio and peak memory.

    python debug/pooled_kv/bench_softtoken_throughput.py --out /results/bench.json
"""

import argparse
import json
import time
import traceback

import torch

from olmo_core.data.document_chunk_landmark import RESERVED_IDS
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)

IDS = RESERVED_IDS["qwen3_5"]
VOCAB = 248320
IGN = -100


def make_rows(B, T, doc_len, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows, labs = [], []
    for _ in range(B):
        toks = torch.randint(1000, 200000, (100,), generator=g).tolist()
        body = 100 + 50 + 20
        n_docs = (T - body) // (doc_len + 2)
        for _d in range(n_docs):
            toks += [IDS.doc_start] + torch.randint(1000, 200000, (doc_len,), generator=g).tolist() + [IDS.doc_end]
        q = torch.randint(1000, 200000, (50,), generator=g).tolist()
        a = torch.randint(1000, 200000, (20,), generator=g).tolist()
        toks += q + a + [IDS.eos]
        lab = [IGN] * (len(toks) - 21) + a + [IDS.eos]
        pad = T - len(toks)
        toks += [IDS.eos] * pad
        lab += [IGN] * pad
        rows.append(toks[:T]); labs.append(lab[:T])
    return torch.tensor(rows, device=device), torch.tensor(labs, device=device)


def build(backend: str, device):
    cfg = TransformerConfig.qwen3_5_4B(vocab_size=VOCAB, attn_backend=AttentionBackendName(backend))
    model = cfg.build(init_device=device)
    model.init_weights()
    model = model.to(torch.bfloat16)
    model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)
    model.train()
    return model


def time_steps(model, ids, lab, warm=2, n=4):
    torch.cuda.synchronize()
    for i in range(warm + n):
        if i == warm:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
        model.zero_grad(set_to_none=True)
        out = model(ids, labels=lab)
        out.loss.backward()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    return dt, torch.cuda.max_memory_allocated() / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_softtoken.json")
    ap.add_argument("--seq-len", type=int, default=32768)
    ap.add_argument("--doc-len", type=int, default=45)
    a = ap.parse_args()
    dev = torch.device("cuda")
    T = a.seq_len
    results = []

    def run(name, backend, B, keep=None, len_bias=False, T_row=None):
        T_row = T_row or T
        rec = {"name": name, "backend": backend, "B": B, "keep": keep, "len_bias": len_bias, "T_row": T_row}
        try:
            model = build(backend, dev)
            if keep is not None:
                model.enable_pooled_soft_tokens(IDS.doc_start, IDS.doc_end, IDS.eos, placeholder_id=IDS.landmark,
                                                keep_prob=keep, detach_soft_kv=True, len_bias=len_bias)
            ids, lab = make_rows(B, T_row, a.doc_len, dev)
            dt, mem = time_steps(model, ids, lab)
            st = getattr(model, "_soft_token_compaction", None)
            ratio = (st["tokens_out"] / st["tokens_in"]) if st else 1.0
            rec.update({"s_per_step": dt, "input_tok_per_s": B * T_row / dt, "peak_gb": mem, "compaction": ratio})
            print(f"[bench] {name:34} B={B} T={T_row:6d} keep={keep} bias={len_bias}: {dt:6.2f} s/step  "
                  f"{B * T_row / dt / 1e3:7.1f}k input tok/s  compaction {ratio:.3f}  peak {mem:5.1f} GB", flush=True)
            del model
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"[bench] {name}: FAILED {rec['error']}", flush=True)
            traceback.print_exc()
        torch.cuda.empty_cache()
        results.append(rec)
        json.dump(results, open(a.out, "w"), indent=1)

    run("dense/flash", "flash_2", 1)
    run("dense/torch", "torch", 1)
    run("soft/torch k=1/3", "torch", 1, keep=1 / 3)
    run("soft/torch k=1/3 B=2", "torch", 2, keep=1 / 3)
    run("soft/torch k=1/3 +logL", "torch", 1, keep=1 / 3, len_bias=True)
    run("soft/flash k=1/3", "flash_2", 1, keep=1 / 3)
    run("soft/torch k=1/12", "torch", 1, keep=1 / 12)
    run("soft/torch k=1/12 +logL", "torch", 1, keep=1 / 12, len_bias=True)
    run("soft/flash k=1/12", "flash_2", 1, keep=1 / 12)
    # the ideal: dense flash on a row as long as the k=1/3 compacted row (~ 1/3 docs + 690 slots)
    comp = next((r["compaction"] for r in results if r["name"] == "soft/torch k=1/3" and "compaction" in r), 0.37)
    run("dense/flash @compacted(1/3)", "flash_2", 1, T_row=int(T * comp) // 64 * 64)
    comp = next((r["compaction"] for r in results if r["name"] == "soft/torch k=1/12" and "compaction" in r), 0.11)
    run("dense/flash @compacted(1/12)", "flash_2", 1, T_row=max(1024, int(T * comp) // 64 * 64))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
