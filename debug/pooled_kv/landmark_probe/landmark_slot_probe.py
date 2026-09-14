"""
LANDMARK TOKENS AS THE SOFT TOKEN -- eval-only probe (Prasann, 2026-09-14).

The soft-token / pooled-doc-KV line of work replaces a pooled span by ONE slot whose K/V comes from
the mean input embedding of the span. Question: can a *compressive-landmark* model's own landmark
token -- a representation it was trained to write at the end of every block -- serve as that slot?
If it can, the compaction is free: the model already produces the summary.

Everything here lives in **landmark token space**, so positions never move:

  PASS 1 -- ``FastCompressiveLandmarkAttention`` runs the full landmark-structured sequence
    (content padded to a multiple of ``mem_freq`` with the tokenizer pad id, then a landmark token
    inserted after every ``mem_freq`` content tokens -- ``_insert_landmark_tokens``, the exact
    routine the native eval harness uses at prefill). Every attention layer's post-RoPE K and V are
    stashed at each block's landmark position, and the block's K/V mean is stashed too.

  PASS 2 -- a plain-``Attention`` model built from the SAME state dict (landmark attention classes
    add no parameters) runs the COMPACTED sequence: kept blocks keep their real tokens, each pooled
    block is replaced by ONE slot at the landmark's ORIGINAL position, and every kept token keeps
    its original position id. RoPE therefore sees "as if the context were full". The slot's K/V is
    overridden per layer (``soft_kv_override_layers``) with the captured landmark K/V; because those
    keys are already rotated at their absolute position we pass ``pos=0`` (RoPE at 0 is identity),
    exactly as ``oracle_meankv_probe.py`` does.

Two model instances are needed because ``FastLandmarkAttention.forward`` accepts neither
``position_ids`` nor ``soft_kv_override``.

ARMS (per rung, in kill-switch order)::

  full                 PASS 1: compressive landmark attention on the landmark sequence (reference)
  causal-lmseq         plain causal over the SAME landmark sequence, no compaction
  causal-dense         plain causal over the content-only sequence (no landmarks / pads)
  meanemb  k=..        compaction; slot input embedding = mean of the block's content embeddings
  lmkv     k=..        compaction; slot K/V = the block's LANDMARK K/V from pass 1
  blockmeankv k=..     compaction; slot K/V = the MEAN of the block's K/V from pass 1 (control)
  lmkv-blocklocal k=.. as ``lmkv`` but pass 1 ran BLOCK-LOCAL (each block attends only to itself,
                       via block-aligned ``doc_lens``) -- the only variant that would save training
                       FLOPs, since a block-local pass 1 is O(T * block) instead of O(T^2).

Metrics at the answer positions: CE of the true answer tokens, top-1 agreement with ``full``,
KL(full || arm), exact match of the teacher-forced greedy answer, and the compaction ratio.

    python debug/pooled_kv/landmark_probe/landmark_slot_probe.py \
        --ckpt /weka/oe-training-default/ai2-llm/checkpoints/q4b-comp-block128-5task-dolci25-nocpt/step8550 \
        --mem-freq 127 --rungs 2k,32k --rows 48 --out /results/landmark_slot_probe.json
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "src", "scripts", "data"))

import convert_longctx_tasks_to_sft as CONV  # noqa: E402

from olmo_core.data import TokenizerConfig  # noqa: E402
from olmo_core.distributed.checkpoint import load_model_and_optim_state  # noqa: E402
from olmo_core.generate.generation_module.transformer.generation_module import (  # noqa: E402
    _insert_landmark_tokens,
)
from olmo_core.nn.attention import Attention, AttentionBackendName  # noqa: E402
from olmo_core.nn.lm_head import LMLossImplementation  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402

W = "/weka/oe-training-default/ai2-llm/checkpoints"
EOS_ID = 151643  # == TokenizerConfig.qwen3().pad_token_id; the id LandmarkPackingInstanceSource pads with
LANDMARK_ID = 151860
TOKENIZER = "Qwen/Qwen3-4B"

# Contradiction rungs of the v3 eval bundle (the rows the ladder evals scored).
EVAL_JSONL = {
    "2k": f"{W}/prasanns/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl",
    "8k": f"{W}/prasanns/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl",
    "16k": f"{W}/prasanns/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n385_k3.jsonl",
    "32k": f"{W}/prasanns/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n765_k3.jsonl",
}

_T0 = time.time()


def log(m):
    print(f"[lmprobe {time.time() - _T0:7.1f}s] {m}", flush=True)


# ---------------------------------------------------------------------------------------------
# Row construction: the ladder40k contradiction recipe (Qwen3 chat template, query_position=both,
# cot-mode none), plus a per-token claim index so blocks can be classified and documents measured.
# ---------------------------------------------------------------------------------------------


def build_rows(jsonl: str, n_rows: int, tok, enable_thinking=None) -> List[dict]:
    rows = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if "ex" in ex and "documents" not in ex:
                ex = ex["ex"]
            r = _build_row(ex, tok, enable_thinking)
            if r is not None:
                rows.append(r)
            if len(rows) >= n_rows:
                break
    return rows


def _build_row(ex: dict, tok, enable_thinking=None) -> Optional[dict]:
    """Tokenize one contradiction example exactly as convert_longctx_tasks_to_sft.py would, and
    additionally recover (a) the answer-token mask and (b) each token's claim index."""
    user_content, answer = CONV.build_contradiction_instance(ex, "both", "none")
    messages = [{"role": "user", "content": user_content}]
    prompt_str = CONV.render_chat(
        tok, messages, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    full_str = CONV.render_chat(
        tok,
        messages + [{"role": "assistant", "content": answer}],
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    if not full_str.startswith(prompt_str):
        raise RuntimeError("prompt is not a prefix of the full conversation")

    enc = tok(full_str, add_special_tokens=False, return_offsets_mapping=True)
    ids = np.asarray(enc["input_ids"], dtype=np.int64)
    offs = np.asarray(enc["offset_mapping"], dtype=np.int64)  # (T, 2) char spans into full_str
    ans_mask = offs[:, 0] >= len(prompt_str)

    # Char span of each claim inside full_str: user_content is embedded verbatim, find its offset.
    uc_off = full_str.find(user_content)
    if uc_off < 0:
        return None
    claim_id = np.full(len(ids), -1, dtype=np.int64)
    cursor = 0
    for i, doc in enumerate(ex["documents"]):
        piece = CONV.CLAIM_TEMPLATE.format(id=i + 1, text=CONV.sanitize(doc["text"]))
        lo = user_content.find(piece, cursor)
        if lo < 0:
            return None
        hi = lo + len(piece)
        cursor = hi
        sel = (offs[:, 0] >= uc_off + lo) & (offs[:, 1] <= uc_off + hi) & (offs[:, 1] > offs[:, 0])
        claim_id[sel] = i

    ids = np.concatenate([ids, np.array([EOS_ID], dtype=np.int64)])
    ans_mask = np.concatenate([ans_mask, np.array([True])])  # the appended EOS is a target too
    claim_id = np.concatenate([claim_id, np.array([-1], dtype=np.int64)])
    if not ans_mask.any() or (claim_id >= 0).sum() == 0:
        return None
    return {"ids": ids, "ans_mask": ans_mask, "claim_id": claim_id, "n_docs": len(ex["documents"])}


# ---------------------------------------------------------------------------------------------
# Landmark-space layout
# ---------------------------------------------------------------------------------------------


def to_landmark_space(row: dict, mem_freq: int) -> dict:
    """Pad the content to a multiple of ``mem_freq`` and insert a landmark after every ``mem_freq``
    content tokens (== LandmarkPackingInstanceSource's per-document layout, == the native eval's
    ``_build_landmark_prompt(mode='generation_only')``). Returns the landmark ids plus the
    content->landmark index map and per-block bookkeeping."""
    block = mem_freq + 1
    content = row["ids"]
    L = len(content)
    pad_len = (-L) % mem_freq
    padded = np.concatenate([content, np.full(pad_len, EOS_ID, dtype=np.int64)])
    lm = _insert_landmark_tokens(torch.tensor(padded)[None], mem_freq, LANDMARK_ID)[0].numpy()
    T = len(lm)
    assert T % block == 0, (T, block)
    # content index i -> landmark index i + i // mem_freq
    c2l = np.arange(len(padded)) + np.arange(len(padded)) // mem_freq
    n_blocks = T // block

    # Per-block role. The CONTEXT region is the contiguous span from the first to the last claim
    # token (the blank lines between claims belong to it too). A block is POOLABLE only if it lies
    # entirely inside that span: the leading instruction, the trailing instruction, the answer and
    # the tail padding all stay real.
    cclaim = np.concatenate([row["claim_id"], np.full(pad_len, -2, dtype=np.int64)])
    cans = np.concatenate([row["ans_mask"], np.zeros(pad_len, dtype=bool)])
    ctx_idx = np.nonzero(cclaim >= 0)[0]
    is_ctx = np.zeros(len(padded), dtype=bool)
    is_ctx[ctx_idx.min() : ctx_idx.max() + 1] = True
    poolable = np.zeros(n_blocks, dtype=bool)
    block_of_content = np.arange(len(padded)) // mem_freq
    for b in range(n_blocks):
        sel = block_of_content == b
        poolable[b] = bool(sel.any() and is_ctx[sel].all() and not cans[sel].any())
    return {
        "lm": lm,
        "c2l": c2l,
        "n_blocks": n_blocks,
        "block_size": block,
        "poolable": poolable,
        "n_content": len(padded),
    }


# ---------------------------------------------------------------------------------------------
# K/V stash (post-RoPE, per attention layer)
# ---------------------------------------------------------------------------------------------


class KVStash:
    """Wrap ``Attention._prepare_qkv`` and reduce each layer's post-RoPE K/V to per-block landmark
    K/V and per-block mean K/V immediately (a full 32k stash would be ~5 GB)."""

    def __init__(self, models, block_size: int):
        self.layers = {}
        for model in models:
            for li, blk in model.blocks.items():
                if isinstance(blk.attention, Attention):
                    self.layers[id(blk.attention)] = int(li)
        self.block = block_size
        self.lm: Dict[int, tuple] = {}
        self.mean: Dict[int, tuple] = {}
        self.on = False
        # Block-local pass: ``cu_doc_lens`` must reach the BACKEND (block-diagonal masking) but not
        # RoPE, which would reset each block's positions to 0..block-1 and destroy the absolute
        # frame the captured landmark keys are injected in. Dropping it here (and any position_ids)
        # leaves RoPE on the default absolute 0..T-1 while the backend still masks per block.
        self.rope_absolute = False
        self._orig = Attention._prepare_qkv
        stash = self

        def wrapper(attn, x, **kw):
            if stash.rope_absolute:
                kw = {**kw, "cu_doc_lens": None, "position_ids": None}
            q, k, v = stash._orig(attn, x, **kw)
            if stash.on and id(attn) in stash.layers:
                li = stash.layers[id(attn)]
                B, T, H, D = k.shape
                nb = T // stash.block
                kb = k.detach()[0].view(nb, stash.block, H, D)
                vb = v.detach()[0].view(nb, stash.block, H, D)
                stash.lm[li] = (kb[:, -1].clone(), vb[:, -1].clone())
                stash.mean[li] = (
                    kb.float().mean(1).to(k.dtype),
                    vb.float().mean(1).to(v.dtype),
                )
            return q, k, v

        Attention._prepare_qkv = wrapper

    def clear(self):
        self.lm.clear()
        self.mean.clear()

    def restore(self):
        Attention._prepare_qkv = self._orig


def strip_slot_bias():
    """``soft_kv_override['bias']`` forces the additive-bias SDPA path (which needs a full (T,T)
    ``attn_bias``). We never use a slot bias, so drop it and keep the ordinary causal kernel --
    on a compacted row sorted by original position, sequence-causal == position-causal."""
    orig = Attention.forward

    def fwd(self, *a, **kw):
        ovr = kw.get("soft_kv_override")
        if ovr is not None and ovr.get("bias") is not None:
            kw["soft_kv_override"] = {**ovr, "bias": None}
        return orig(self, *a, **kw)

    Attention.forward = fwd


_EMB_INJECT = {"cols": None, "vecs": None}


def install_embed_inject(model):
    """Overwrite the slot columns' input embedding (the mean-embedding soft token of the current
    scheme). Applied via a forward hook so the plain model needs no pooled-soft-token wiring."""
    emb = model.embeddings

    def hook(_mod, _inp, out):
        cols, vecs = _EMB_INJECT["cols"], _EMB_INJECT["vecs"]
        if cols is None or cols.numel() == 0:
            return out
        out = out.clone()
        out[0, cols] = vecs.to(out.dtype)
        return out

    emb.register_forward_hook(hook)


# ---------------------------------------------------------------------------------------------


def build_models(ckpt: str, mem_freq: int, alpha: float, backend: str):
    vocab = TokenizerConfig.qwen3().padded_vocab_size()
    lm_cfg = TransformerConfig.qwen3_4B(
        vocab_size=vocab,
        fast_compressive_landmark=True,
        nonselected_landmark_mass=alpha,
        mem_freq=mem_freq,
    )
    lm_cfg.lm_head.loss_implementation = LMLossImplementation.default
    lm_model = lm_cfg.build(init_device="cpu")
    t0 = time.time()
    load_model_and_optim_state(ckpt, lm_model)
    log(f"loaded {ckpt} in {time.time() - t0:.0f}s")

    pl_cfg = TransformerConfig.qwen3_4B(
        vocab_size=vocab, attn_backend=AttentionBackendName[backend]
    )
    pl_cfg.lm_head.loss_implementation = LMLossImplementation.default
    pl_model = pl_cfg.build(init_device="cpu")
    # Landmark attention classes declare no nn.Parameter of their own, so the state dicts match.
    missing, unexpected = pl_model.load_state_dict(lm_model.state_dict(), strict=False)
    missing = [k for k in missing if "rope" not in k and "freqs" not in k]
    unexpected = [k for k in unexpected if "rope" not in k and "freqs" not in k]
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    log("plain-causal twin built from the same state dict (0 param mismatches)")
    return (
        lm_model.cuda().to(torch.bfloat16).eval(),
        pl_model.cuda().to(torch.bfloat16).eval(),
    )


@torch.no_grad()
def run_rung(lm_model, pl_model, stash, rows, mem_freq, keeps, seed, res, rung, doc_stats):
    block = mem_freq + 1
    n_layers = len(lm_model.blocks)
    names = ["full", "causal-lmseq", "causal-dense"]
    for k in keeps:
        for fam in ("meanemb", "lmkv", "blockmeankv", "lmkv-blocklocal"):
            names.append(f"{fam} k={k:.3f}")
    for n in names:
        res.setdefault(n, {"ce": [], "top1": [], "kl": [], "correct": [], "compaction": []})

    t_start = time.time()
    for ri, row in enumerate(rows):
        lay = to_landmark_space(row, mem_freq)
        lm_ids = torch.tensor(lay["lm"][None], device="cuda")
        T = lm_ids.shape[1]
        c2l = lay["c2l"]
        ans_c = np.nonzero(row["ans_mask"])[0]
        pred_c = ans_c - 1
        targets = torch.tensor(row["ids"][ans_c], device="cuda")
        pred_l = torch.tensor(c2l[pred_c], device="cuda")

        # doc/block statistics (the landmark != document trap)
        cl = row["claim_id"]
        lens = np.bincount(cl[cl >= 0], minlength=row["n_docs"])
        doc_stats["doc_len"].extend(lens[lens > 0].tolist())
        bo = np.arange(len(cl)) // mem_freq
        docs_per_block = [
            len(set(cl[(bo == b) & (cl >= 0)].tolist())) for b in range(int(bo.max()) + 1)
        ]
        doc_stats["docs_per_block"].extend([d for d in docs_per_block if d > 0])
        doc_stats["n_blocks"].append(lay["n_blocks"])
        doc_stats["poolable_blocks"].append(int(lay["poolable"].sum()))

        # ---- PASS 1: FULL compressive landmark, stashing K/V ----
        stash.clear()
        stash.on = True
        lf = lm_model(lm_ids, logits_to_keep=pred_l[None])[0].float()
        stash.on = False
        lm_kv = {li: (a.clone(), b.clone()) for li, (a, b) in stash.lm.items()}
        mean_kv = {li: (a.clone(), b.clone()) for li, (a, b) in stash.mean.items()}
        _record(res["full"], lf, lf, targets, 1.0)

        # ---- PASS 1b: BLOCK-LOCAL pass 1 (each block attends only to itself) ----
        # For a query inside its own block, compressive-landmark attention IS plain causal
        # attention over that block (the "local/last section"); masking out every past block
        # therefore makes it identical to plain causal attention with block-diagonal
        # ``cu_doc_lens``. Running it on the plain twin lets us keep ABSOLUTE RoPE positions
        # (the landmark forward would reset RoPE per document when given ``cu_doc_lens``).
        stash.clear()
        stash.on = True
        stash.rope_absolute = True
        doc_lens = torch.full((1, lay["n_blocks"]), block, dtype=torch.int32, device="cuda")
        pl_model(lm_ids, logits_to_keep=pred_l[None], doc_lens=doc_lens, max_doc_lens=[block])
        stash.rope_absolute = False
        stash.on = False
        bl_kv = {li: (a.clone(), b.clone()) for li, (a, b) in stash.lm.items()}

        # ---- controls: plain causal, no compaction ----
        lg = pl_model(lm_ids, logits_to_keep=pred_l[None])[0].float()
        _record(res["causal-lmseq"], lg, lf, targets, 1.0)

        dense_ids = torch.tensor(row["ids"][None], device="cuda")
        dpred = torch.tensor(pred_c, device="cuda")
        lg = pl_model(dense_ids, logits_to_keep=dpred[None])[0].float()
        _record(res["causal-dense"], lg, lf, targets, len(row["ids"]) / T)

        # ---- compaction arms ----
        for keep in keeps:
            comp_ids, comp_pos, slot_cols, slot_blocks = _compact(lay, keep, seed + ri, block)
            comp_ids_t = torch.tensor(comp_ids[None], device="cuda")
            comp_pos_t = torch.tensor(comp_pos[None], device="cuda")
            l2c = {int(p): c for c, p in enumerate(comp_pos.tolist())}
            out_cols = torch.tensor([l2c[int(p)] for p in pred_l.tolist()], device="cuda")
            comp = len(comp_ids) / T
            slot_cols_t = torch.tensor(slot_cols, device="cuda", dtype=torch.long)
            rows_t = torch.zeros_like(slot_cols_t)
            zpos = torch.zeros(len(slot_cols), dtype=torch.long, device="cuda")

            # (a) mean input embedding of the block's content tokens
            with torch.no_grad():
                emb_w = pl_model.embeddings.weight
                vecs = torch.stack(
                    [
                        emb_w[
                            torch.tensor(
                                lay["lm"][b * block : b * block + block - 1], device="cuda"
                            )
                        ]
                        .float()
                        .mean(0)
                        for b in slot_blocks
                    ]
                ) if slot_blocks else emb_w.new_zeros((0, emb_w.shape[1]))
            _EMB_INJECT["cols"], _EMB_INJECT["vecs"] = slot_cols_t, vecs
            lg = pl_model(comp_ids_t, position_ids=comp_pos_t, logits_to_keep=out_cols[None])[
                0
            ].float()
            _EMB_INJECT["cols"], _EMB_INJECT["vecs"] = None, None
            _record(res[f"meanemb k={keep:.3f}"], lg, lf, targets, comp)

            # (b/c/d) K/V-override arms
            for fam, src in (
                ("lmkv", lm_kv),
                ("blockmeankv", mean_kv),
                ("lmkv-blocklocal", bl_kv),
            ):
                if not slot_blocks:
                    _record(res[f"{fam} k={keep:.3f}"], lf, lf, targets, comp)
                    continue
                k0 = src[0][0]
                slots = torch.zeros(
                    (len(slot_blocks), n_layers, 2, k0.shape[1], k0.shape[2]),
                    dtype=k0.dtype,
                    device="cuda",
                )
                bidx = torch.tensor(slot_blocks, device="cuda")
                for li in range(n_layers):
                    kk, vv = src[li]
                    slots[:, li, 0] = kk[bidx]
                    slots[:, li, 1] = vv[bidx]
                ovr = {
                    "rows": rows_t,
                    "cols": slot_cols_t,
                    "pos": zpos,  # keys are already rotated at their absolute position
                    "slots": slots,
                    "biases": torch.zeros((len(slot_blocks), n_layers), device="cuda"),
                }
                lg = pl_model(
                    comp_ids_t,
                    position_ids=comp_pos_t,
                    logits_to_keep=out_cols[None],
                    soft_kv_override_layers=ovr,
                )[0].float()
                _record(res[f"{fam} k={keep:.3f}"], lg, lf, targets, comp)

        if ri + 1 in (1, 2, 5, 10) or (ri + 1) % 10 == 0:
            el = time.time() - t_start
            eta = el / (ri + 1) * (len(rows) - ri - 1)
            log(f"{rung}: row {ri + 1}/{len(rows)}  {el / (ri + 1):.1f}s/row  ETA {eta / 60:.1f}min")
            table(res, names)
    return names


def _record(r, lg, lf, targets, comp):
    r["ce"].append(float(F.cross_entropy(lg, targets)))
    r["top1"].append(float((lg.argmax(-1) == lf.argmax(-1)).float().mean()))
    r["kl"].append(
        float(
            F.kl_div(
                F.log_softmax(lg, -1), F.log_softmax(lf, -1), log_target=True, reduction="batchmean"
            )
        )
    )
    r["correct"].append(float((lg.argmax(-1) == targets).all()))
    r["compaction"].append(comp)


def _compact(lay, keep: float, seed: int, block: int):
    """Compacted row: kept blocks verbatim, each pooled block -> one slot at its landmark position,
    every surviving token keeping its ORIGINAL position id."""
    poolable = lay["poolable"].copy()
    idx = np.nonzero(poolable)[0]
    rng = np.random.RandomState(seed)
    n_keep = int(round(keep * len(idx)))
    if n_keep:
        poolable[rng.permutation(idx)[:n_keep]] = False
    lm = lay["lm"]
    ids, pos, slot_cols, slot_blocks = [], [], [], []
    for b in range(lay["n_blocks"]):
        lo = b * block
        if poolable[b]:
            slot_cols.append(len(ids))
            slot_blocks.append(b)
            ids.append(LANDMARK_ID)
            pos.append(lo + block - 1)  # the landmark's own position
        else:
            ids.extend(lm[lo : lo + block].tolist())
            pos.extend(range(lo, lo + block))
    return (
        np.asarray(ids, dtype=np.int64),
        np.asarray(pos, dtype=np.int64),
        slot_cols,
        slot_blocks,
    )


def table(res, names):
    print(
        f"{'arm':26} {'answerCE':>9} {'+-SE':>6} {'top1=full':>9} {'KL':>8} "
        f"{'exact':>7} {'compact':>8} {'rows':>5}",
        flush=True,
    )
    for n in names:
        r = res.get(n)
        if not r or not r["ce"]:
            continue
        ce = np.asarray(r["ce"])
        se = ce.std(ddof=1) / math.sqrt(len(ce)) if len(ce) > 1 else float("nan")
        print(
            f"{n:26} {ce.mean():9.4f} {se:6.4f} {np.mean(r['top1']):9.4f} "
            f"{np.mean(r['kl']):8.4f} {np.mean(r['correct']):7.3f} "
            f"{np.mean(r['compaction']):8.4f} {len(ce):5d}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="step dir, or its model_and_optim")
    ap.add_argument("--mem-freq", type=int, default=127)
    ap.add_argument("--alpha", type=float, default=0.1, help="nonselected_landmark_mass")
    ap.add_argument("--rungs", default="2k,32k")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--keeps", default="0,0.125")
    ap.add_argument("--backend", default="flash_2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/results/landmark_slot_probe.json")
    ap.add_argument("--weka-out", default="")
    ap.add_argument(
        "--no-think",
        action="store_true",
        help="render with enable_thinking=False (the converter's --no-think); the ladder40k shards "
        "were built with the template default, so this is only for a tokenization cross-check",
    )
    a = ap.parse_args()

    ck = a.ckpt if a.ckpt.rstrip("/").endswith("model_and_optim") else f"{a.ckpt}/model_and_optim"
    keeps = [float(k) for k in a.keeps.split(",")]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    strip_slot_bias()
    lm_model, pl_model = build_models(ck, a.mem_freq, a.alpha, a.backend)
    install_embed_inject(pl_model)
    stash = KVStash([lm_model, pl_model], a.mem_freq + 1)
    log(f"attention layers stashed: {len(stash.layers)}")

    out = {
        "ckpt": ck,
        "mem_freq": a.mem_freq,
        "alpha": a.alpha,
        "rows_requested": a.rows,
        "keeps": keeps,
        "task": "contradiction",
        "enable_thinking": False if a.no_think else None,
        "rungs": {},
    }
    for rung in a.rungs.split(","):
        rows = build_rows(EVAL_JSONL[rung], a.rows, tok, False if a.no_think else None)
        log(
            f"{rung}: eval_size={len(rows)} rows, content lengths "
            f"{[len(r['ids']) for r in rows[:4]]}..."
        )
        res: Dict[str, dict] = {}
        doc_stats = {"doc_len": [], "docs_per_block": [], "n_blocks": [], "poolable_blocks": []}
        names = run_rung(
            lm_model, pl_model, stash, rows, a.mem_freq, keeps, a.seed, res, rung, doc_stats
        )
        print(f"\n===== RUNG {rung}  (eval_size={len(rows)}) =====", flush=True)
        table(res, names)
        dl = np.asarray(doc_stats["doc_len"])
        dpb = np.asarray(doc_stats["docs_per_block"])
        log(
            f"{rung} doc stats: claim len mean {dl.mean():.1f} median {np.median(dl):.0f} "
            f"p90 {np.percentile(dl, 90):.0f}; docs per {a.mem_freq}-token block mean "
            f"{dpb.mean():.2f} (>=2 in {(dpb >= 2).mean():.1%} of blocks); blocks/row "
            f"{np.mean(doc_stats['n_blocks']):.1f} of which poolable "
            f"{np.mean(doc_stats['poolable_blocks']):.1f}"
        )
        out["rungs"][rung] = {
            "eval_size": len(rows),
            "arms": {
                n: {
                    "ce": float(np.mean(res[n]["ce"])),
                    "ce_se": float(np.std(res[n]["ce"], ddof=1) / math.sqrt(len(res[n]["ce"])))
                    if len(res[n]["ce"]) > 1
                    else None,
                    "top1": float(np.mean(res[n]["top1"])),
                    "kl": float(np.mean(res[n]["kl"])),
                    "exact_match": float(np.mean(res[n]["correct"])),
                    "compaction": float(np.mean(res[n]["compaction"])),
                }
                for n in names
                if res[n]["ce"]
            },
            "per_row": {n: res[n] for n in names if res[n]["ce"]},
            "doc_stats": {
                "claim_len_mean": float(dl.mean()),
                "claim_len_median": float(np.median(dl)),
                "claim_len_p90": float(np.percentile(dl, 90)),
                "docs_per_block_mean": float(dpb.mean()),
                "frac_blocks_multi_doc": float((dpb >= 2).mean()),
                "blocks_per_row": float(np.mean(doc_stats["n_blocks"])),
                "poolable_blocks_per_row": float(np.mean(doc_stats["poolable_blocks"])),
            },
        }
        for path in [a.out] + ([a.weka_out] if a.weka_out else []):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(out, open(path, "w"), indent=1)
            log(f"wrote {path}")
    stash.restore()


if __name__ == "__main__":
    main()
