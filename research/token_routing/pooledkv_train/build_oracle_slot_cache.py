"""
Build the oracle log-mass slot cache for pooled-doc soft-token training (B1-oracle).

For every context document in a tokenized shard dir, runs the doc STANDALONE behind its example's
preamble (one doc per row, true absolute positions -> plain causal attention gives the doc
"preamble + itself" visibility, the document-chunked semantics) and fits, per layer and KV head,
the oracle slot ``(k*, v*, bias)`` that best reproduces the doc's attention log-mass and weighted
value against real answer-region queries rotated to sampled relative offsets. Slots are stored in
the doc-center RoPE frame (see :mod:`olmo_core.nn.oracle_slot`).

**Never runs a full-context forward**: each row is ~(preamble + doc) ~ 150 tokens.

Two-GPU usage (one rank per GPU, disjoint instance stripes)::

    P=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
    CUDA_VISIBLE_DEVICES=0 $P build_oracle_slot_cache.py --data ... --out ... --rank 0 --world 2 &
    CUDA_VISIBLE_DEVICES=1 $P build_oracle_slot_cache.py --data ... --out ... --rank 1 --world 2 &
    wait

The runtime consumer is ``--oracle-slot-cache`` in ``Qwen3-4B-pooledkv-contra-n100-local.py``.
"""

import argparse
import glob
import json
import math
import time
from typing import List, Tuple

import numpy as np
import torch

from olmo_core.data import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.attention import Attention, AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.oracle_slot import (
    OracleSlotCacheWriter,
    derotate_keys,
    doc_hash64,
    fit_oracle_slots_layer,
)
from olmo_core.nn.transformer import TransformerConfig

DOC_START_ID, DOC_END_ID, EOS_TOKEN_ID = 151648, 151649, 151643
LANDMARK_TOKEN_ID = 151669


def log(msg: str) -> None:
    print(f"[oracle-cache {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_instances(data_dir: str) -> List[np.ndarray]:
    """EOS-delimited instances from the raw uint32 token shards."""
    rows: List[np.ndarray] = []
    for f in sorted(glob.glob(f"{data_dir}/token_ids_part_*.npy")):
        toks = np.fromfile(f, dtype=np.uint32)
        eos = np.flatnonzero(toks == EOS_TOKEN_ID)
        start = 0
        for e in eos:
            if e > start:
                rows.append(toks[start : e + 1].astype(np.int64))
            start = e + 1
    return rows


def split_instance(row: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray, int]:
    """-> (preamble, [(doc_start, doc_end_inclusive)], post_region, post_start)."""
    ds = np.flatnonzero(row == DOC_START_ID)
    de = np.flatnonzero(row == DOC_END_ID)
    assert len(ds) == len(de) and len(ds) > 0
    pre = row[: ds[0]]
    docs = list(zip(ds.tolist(), de.tolist()))
    post_start = int(de[-1]) + 1
    return pre, docs, row[post_start:], post_start


class Capture:
    """Stashes per-layer post-RoPE q/k/v (detached) via an Attention._prepare_qkv wrapper."""

    def __init__(self):
        self.layers: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.enabled = False

    def install(self):
        orig = Attention._prepare_qkv
        cap = self

        def wrapper(module, x, **kw):
            q, k, v = orig(module, x, **kw)
            if cap.enabled:
                cap.layers.append((q.detach(), k.detach(), v.detach()))
            return q, k, v

        Attention._prepare_qkv = wrapper


@torch.no_grad()
def forward_blocks(model, ids: torch.Tensor, pos: torch.Tensor) -> None:
    """Embeddings + block stack only (no LM head); the Capture hook does the real work."""
    ii, _, abk, pbk, _ = model._prepare_inputs(ids, None, position_ids=pos)
    h = model.embeddings(ii)
    if model.embed_scale is not None:
        h = h * model.embed_scale
    if model.embedding_norm is not None:
        h = model.embedding_norm(h)
    model._run_blocks(h, abk, pbk)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="/data/prasann/pooledkv_exp/q4b-dense-cpt-fixmark-b1")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--batch-rows", type=int, default=128)
    ap.add_argument("--queries", type=int, default=256, help="query-stash size M per layer")
    ap.add_argument("--warm-instances", type=int, default=48)
    ap.add_argument("--delta-min", type=int, default=256)
    ap.add_argument("--delta-max", type=int, default=30000)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--limit-instances", type=int, default=0, help="debug cap (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed + args.rank)
    device = torch.device("cuda")

    log(f"rank {args.rank}/{args.world}: parsing instances from {args.data}")
    instances = parse_instances(args.data)
    log(f"{len(instances)} instances")

    tok = TokenizerConfig.qwen3()
    cfg = TransformerConfig.qwen3_4B(
        vocab_size=tok.padded_vocab_size(), attn_backend=AttentionBackendName.torch
    )
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    model = cfg.build(init_device="cpu")
    model.enable_pooled_soft_tokens(  # base has baked projector keys; unused here
        DOC_START_ID, DOC_END_ID, EOS_TOKEN_ID, placeholder_id=LANDMARK_TOKEN_ID
    )
    load_model_and_optim_state(f"{args.base}/model_and_optim", model)
    model = model.cuda().to(torch.bfloat16).eval()
    log(f"model loaded from {args.base}")

    first_att = next(m for m in model.modules() if isinstance(m, Attention))
    n_layers = model.n_layers
    head_dim = first_att.head_dim
    scale = head_dim**-0.5
    rope = first_att.rope
    pos_sin, pos_cos = rope._get_rotary_embedding(35000, device)
    pos_sin, pos_cos = pos_sin.float(), pos_cos.float()

    cap = Capture()
    cap.install()

    def run_rows(id_rows: List[np.ndarray], pos_rows: List[np.ndarray]):
        """Pad + forward; returns (ids tensor, per-row content length)."""
        T = max(len(r) for r in id_rows)
        B = len(id_rows)
        ids = torch.full((B, T), EOS_TOKEN_ID, dtype=torch.long)
        pos = torch.full((B, T), 34000, dtype=torch.long)
        for i, (r, p) in enumerate(zip(id_rows, pos_rows)):
            ids[i, : len(r)] = torch.from_numpy(r)
            pos[i, : len(p)] = torch.from_numpy(p)
        cap.layers.clear()
        cap.enabled = True
        forward_blocks(model, ids.to(device), pos.to(device))
        cap.enabled = False

    # ---- Query stash: real answer-region queries, derotated to the canonical frame ----
    log("building query stash from answer-region forwards...")
    stash: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
    for wi in range(0, min(args.warm_instances, len(instances)), 8):
        batch_ids, batch_pos, q_slices = [], [], []
        for inst in instances[wi : wi + 8]:
            pre, _, post, post_start = split_instance(inst)
            row = np.concatenate([pre, post])
            pos_row = np.concatenate(
                [np.arange(len(pre)), np.arange(post_start, post_start + len(post))]
            )
            q_slices.append((len(pre), len(row)))
            batch_ids.append(row)
            batch_pos.append(pos_row)
        run_rows(batch_ids, batch_pos)
        for li, (q, _, _) in enumerate(cap.layers):
            for bi, (lo, hi) in enumerate(q_slices):
                qq = q[bi, lo:hi].float()  # (n, H_q, D)
                pp = torch.from_numpy(batch_pos[bi][lo:hi]).to(device)
                stash[li].append(derotate_keys(qq, pos_sin[pp], pos_cos[pp]))
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    q_stash: List[torch.Tensor] = []
    delta_sin: List[torch.Tensor] = []
    delta_cos: List[torch.Tensor] = []
    for li in range(n_layers):
        allq = torch.cat(stash[li], dim=0)
        sel = torch.randperm(allq.shape[0], generator=g)[: args.queries].to(device)
        q_stash.append(allq[sel])
        u = torch.rand(args.queries, generator=g)
        deltas = (
            (math.log(args.delta_min) + u * (math.log(args.delta_max / args.delta_min)))
            .exp()
            .long()
            .clamp(args.delta_min, args.delta_max)
        ).to(device)
        delta_sin.append(pos_sin[deltas])
        delta_cos.append(pos_cos[deltas])
    stash.clear()
    log(f"query stash ready: {args.queries} queries x {n_layers} layers")

    # ---- Main loop: one doc per row, preamble prepended, fit per batch x layer ----
    writer = None
    seen: set = set()
    my_instances = instances[args.rank :: args.world]
    if args.limit_instances:
        my_instances = my_instances[: args.limit_instances]
    r2_o_sum, r2_m_sum, r2_n = np.zeros(n_layers), np.zeros(n_layers), 0
    n_written = 0
    t0 = time.time()
    pending_ids: List[np.ndarray] = []
    pending_pos: List[np.ndarray] = []
    pending_meta: List[Tuple[int, int, int]] = []  # (pre_len, doc_len, center)
    pending_hash: List[int] = []
    n_batches_done = 0
    total_docs_est = len(my_instances) * 434

    def flush():
        nonlocal writer, n_written, r2_n, n_batches_done
        if not pending_ids:
            return
        run_rows(pending_ids, pending_pos)
        B = len(pending_ids)
        n_kv = cap.layers[0][1].shape[2]
        slots = torch.zeros(B, n_layers, 2, n_kv, head_dim, dtype=torch.float16)
        biases = torch.zeros(B, n_layers, dtype=torch.float16)
        # Flatten every row's doc tokens into one (Nt,) index set ONCE (not per layer): the
        # per-row python loop was launch-bound (~8k kernel launches/batch -> 34 docs/s/GPU).
        row_l, col_l, ctr_l = [], [], []
        for bi, (pre_len, doc_len, center) in enumerate(pending_meta):
            row_l.append(torch.full((doc_len,), bi, dtype=torch.long))
            col_l.append(torch.arange(pre_len, pre_len + doc_len, dtype=torch.long))
            ctr_l.append(torch.full((doc_len,), center, dtype=torch.long))
        row_idx = torch.cat(row_l).to(device)
        col_idx = torch.cat(col_l).to(device)
        centers = torch.cat(ctr_l).to(device)
        sin_c, cos_c = pos_sin[centers], pos_cos[centers]
        for li, (_, k, v) in enumerate(cap.layers):
            keys_cf = derotate_keys(k[row_idx, col_idx].float(), sin_c, cos_c)
            ks, vs, bias, diags = fit_oracle_slots_layer(
                keys_cf,
                v[row_idx, col_idx].float(),
                row_idx,
                B,
                q_stash[li],
                delta_sin[li],
                delta_cos[li],
                scale=scale,
                ridge=args.ridge,
            )
            slots[:, li, 0] = ks.half().cpu()
            slots[:, li, 1] = vs.half().cpu()
            biases[:, li] = bias.half().cpu()
            r2_o_sum[li] += diags["r2_oracle"]
            r2_m_sum[li] += diags["r2_meanpool"]
        r2_n += 1
        if writer is None:
            writer = OracleSlotCacheWriter(args.out, f"rank{args.rank}", n_layers, n_kv, head_dim)
        writer.append(pending_hash, slots, biases)
        n_written += B
        n_batches_done += 1
        pending_ids.clear()
        pending_pos.clear()
        pending_meta.clear()
        pending_hash.clear()
        if n_batches_done in (1, 2, 5, 10) or n_batches_done % 50 == 0:
            dt = time.time() - t0
            rate = n_written / max(dt, 1e-9)
            eta = (total_docs_est - n_written) / max(rate, 1e-9) / 60
            log(
                f"batch {n_batches_done}: {n_written} docs @ {rate:.0f}/s, ETA {eta:.0f} min | "
                f"r2 oracle {r2_o_sum.mean() / r2_n:.3f} vs meanpool {r2_m_sum.mean() / r2_n:.3f}"
            )

    for inst in my_instances:
        pre, docs, _, _ = split_instance(inst)
        for s, e in docs:
            doc = inst[s : e + 1]
            h = doc_hash64(doc.astype(np.uint32))
            if h in seen:
                continue
            seen.add(h)
            row = np.concatenate([pre, doc])
            pos_row = np.concatenate([np.arange(len(pre)), np.arange(s, e + 1)])
            pending_ids.append(row)
            pending_pos.append(pos_row)
            pending_meta.append((len(pre), len(doc), (s + e) // 2))
            pending_hash.append(h)
            if len(pending_ids) >= args.batch_rows:
                flush()
    flush()
    if writer is not None:
        writer.close()

    report = {
        "rank": args.rank,
        "n_docs": n_written,
        "r2_oracle_per_layer": (r2_o_sum / max(r2_n, 1)).tolist(),
        "r2_meanpool_per_layer": (r2_m_sum / max(r2_n, 1)).tolist(),
        "queries": args.queries,
        "ridge": args.ridge,
        "delta_range": [args.delta_min, args.delta_max],
    }
    with open(f"{args.out}/fit_report_rank{args.rank}.json", "w") as f:
        json.dump(report, f, indent=2)
    log(
        f"DONE: {n_written} docs in {(time.time() - t0) / 60:.1f} min | mean r2 oracle "
        f"{r2_o_sum.mean() / max(r2_n, 1):.3f} vs meanpool {r2_m_sum.mean() / max(r2_n, 1):.3f}"
    )


if __name__ == "__main__":
    main()
