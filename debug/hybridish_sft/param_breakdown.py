"""Exact parameter counts for the two arms, split the way the comparison needs.

Total params is the wrong headline on its own: the tied-or-untied embedding is a large constant
that has nothing to do with the attention ratio, so the arms are also compared on non-embedding
params and on the attention-vs-GDN split.
"""
import argparse, json
from collections import defaultdict
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", action="append", required=True)
a = ap.parse_args()

for ck in a.ckpt:
    cfg = json.load(open(f"{ck}/config.json"))
    tot, emb, per_layer = 0, 0, defaultdict(int)
    with safe_open(f"{ck}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            n = 1
            for d in f.get_slice(k).get_shape():
                n *= d
            tot += n
            if "embed" in k or "lm_head" in k:
                emb += n
            if ".layers." in k:
                per_layer[int(k.split(".layers.")[1].split(".")[0])] += n
    types = cfg["layer_types"]
    attn_idx = [i for i, t in enumerate(types) if t == "full_attention"]
    gdn_idx = [i for i, t in enumerate(types) if t != "full_attention"]
    attn_p = sum(per_layer[i] for i in attn_idx)
    gdn_p = sum(per_layer[i] for i in gdn_idx)
    print(f"\n=== {ck.rstrip('/').split('/')[-1]} ===")
    print(f"  layers            {cfg['num_hidden_layers']}   d_model {cfg['hidden_size']}   "
          f"heads {cfg['num_attention_heads']}   ffn {cfg['intermediate_size']}")
    print(f"  full-attention    {len(attn_idx)} layers at {attn_idx}")
    print(f"  linear/GDN        {len(gdn_idx)} layers")
    print(f"  TOTAL params      {tot:,}")
    print(f"  embedding+head    {emb:,}")
    print(f"  NON-embedding     {tot-emb:,}")
    print(f"    attention layers {attn_p:,}  ({100*attn_p/(tot-emb):.1f}% of non-emb)")
    print(f"    GDN layers       {gdn_p:,}  ({100*gdn_p/(tot-emb):.1f}% of non-emb)")
