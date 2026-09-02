"""
Why is the soft-token arm ~21 s/step on Qwen3.5-4B at a 40960-token padded row (fs35-smoke-softtoken)?
Times the pieces of one training forward on ONE GPU: chunk-id build, keep-set, compaction, and
the model forward+backward on the compacted row, then a torch.profiler kernel/CPU table.

    srun -w mooney -p jsteinhardt -q preemptive_high --gres=gpu:H200:1 \
      /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python debug/flop_scaling/profile_softtoken35.py
"""

import json
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from olmo_core.config import DType
from olmo_core.data.document_chunk_landmark import RESERVED_IDS
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig

SHARD = "/data/prasann/ctc_suite/shards/contradiction_train"
SEQ = 40960
ids_ = RESERVED_IDS["qwen3_5"]
dev = "cuda"


def t(fn, n=3):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / n, out


meta = json.load(open(f"{SHARD}/metadata.json"))
arr = np.load(f"{SHARD}/token_ids_part_000000.npy", mmap_mode="r")
# first example: up to the first EOS after position 0
tok = np.array(arr[: SEQ * 2])
eos_pos = np.where(tok == ids_.eos)[0]
row = tok[: eos_pos[-1] + 1] if len(eos_pos) else tok[:SEQ]
# pick a LONG example: scan for the longest gap between EOS in the first 2M tokens
big = np.array(arr[: 2_000_000]); ep = np.where(big == ids_.eos)[0]
gaps = np.diff(np.concatenate([[-1], ep])); i = int(np.argmax(gaps))
row = big[ep[i - 1] + 1 if i > 0 else 0 : ep[i] + 1]
L = len(row); print(f"example len {L} tokens, padded to {SEQ}")
x = torch.full((1, SEQ), ids_.eos, dtype=torch.long)
x[0, :L] = torch.from_numpy(row.astype(np.int64))
labels = x.clone(); labels[0, L:] = -100

ms, cid = t(lambda: build_chunk_ids_from_tokens(x, doc_start_id=ids_.doc_start, doc_end_id=ids_.doc_end, eos_id=ids_.eos, mode="chunked"), 3)
print(f"build_chunk_ids_from_tokens: {ms:.0f} ms  (n_docs={int(cid.max()) + 1})")

cfg = TransformerConfig.qwen3_5_4B(vocab_size=248320, dtype=DType.bfloat16, attn_backend=AttentionBackendName.flash_2)
cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
model = cfg.build(init_device=dev)
model.init_weights(max_seq_len=SEQ, device=torch.device(dev))
model.enable_pooled_soft_tokens(ids_.doc_start, ids_.doc_end, ids_.eos, placeholder_id=ids_.landmark, keep_prob=0.25, keep_seed=0, detach_soft_kv=True)
model.train()
xg, lg = x.to(dev), labels.to(dev)

ms, cb = t(lambda: model._compact_pooled_soft_tokens(xg, lg, -100), 3)
print(f"_compact_pooled_soft_tokens: {ms:.0f} ms -> compacted len {cb[0].input_ids.shape[1] if cb else None}")


def step():
    out = model(xg, labels=lg)
    loss = out if torch.is_tensor(out) else out.loss
    loss.backward(); model.zero_grad(set_to_none=True)


ms, _ = t(step, 2)
print(f"full train step (fwd+bwd, compacted): {ms:.0f} ms")
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    step(); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=18))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
