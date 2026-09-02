"""Convert a local Gemma-3 HF base checkpoint -> olmo-core model-only distcp base.

The Gemma-3 sibling of :mod:`convert_qwen35_base`. Three things make Gemma different from the
Qwen bases and are handled here rather than in a generic path:

1. **The released 4B/12B/27B repos are multimodal** (``Gemma3ForConditionalGeneration``): the text
   decoder lives under a ``language_model.`` prefix alongside a SigLIP vision tower. The tower is
   dropped and the prefix stripped so the state matches olmo-core's ``gemma3_text`` mapping.
2. **Embeddings are tied** (``tie_word_embeddings=True``, and the safetensors carry no
   ``lm_head.weight``). olmo-core's transformer keeps a separate ``lm_head.w_out.weight``, so the
   embedding matrix is copied into it explicitly -- the generic
   :func:`~olmo_core.nn.hf.convert.convert_state_from_hf` has no tie fallback (only the Qwen3.5
   path does), and without this the LM head would silently stay at random init.
3. **The factory defaults are wrong for the real checkpoint.** ``TransformerConfig.gemma3_4B``
   defaults to 16 query heads and no explicit ``head_dim``; ``google/gemma-3-4b-pt`` has 8 query
   heads / ``head_dim=256`` and ``rope_scaling={"rope_type": "linear", "factor": 8}`` on the
   global layers. Every dimension used here is read **off the checkpoint's own config.json**, and
   the resulting parameter count is asserted against the checkpoint's tensor shapes.

Also runs the marker-embedding audit (cosine + norm of the document-boundary rows) that Stage 0 of
``records/ctc-suite-scaling-plan.md`` requires, and can repair them in place with ``--fix-markers``
(see ``records/document-chunked-marker-embeddings.md`` and
``records/n100-chunked-marker-position-bug.md`` -- cosine alone is NOT enough, the norm must be
in-distribution too).

Usage::

    python convert_gemma3_base.py --base-dir /scratch/.../hf_models/gemma-3-4b-pt \\
        --out /scratch/.../ctc_suite/bases/gemma3-4b-base-modelonly --fix-markers
"""

import argparse
import glob
import json
import os
import sys
import types

import torch

# Donor rows for the marker repair: real, TRAINED delimiter tokens. Seeding each marker from a
# trained row (plus a small orthogonal perturbation) is what keeps the marker NORM in
# distribution -- a repair that only fixes the pairwise cosine leaves the markers at a fraction of
# a real token's norm, which RMSNorm amplifies into full-strength noise and flatlines training at
# chance for EVERY mask (records/n100-chunked-marker-position-bug.md).
DONOR_STRINGS = ["«", "»", "[", "]"]


def _load_hf_text_state(base_dir: str):
    """Load the text-decoder safetensors state, dropping the vision tower / prefixes."""
    from safetensors.torch import load_file

    raw = {}
    shards = sorted(glob.glob(os.path.join(base_dir, "*.safetensors")))
    if not shards:
        sys.exit(f"[convert] no *.safetensors under {base_dir}")
    for shard in shards:
        raw.update(load_file(shard))
    print(f"[convert] loaded {len(raw)} HF tensors from {len(shards)} shard(s)", flush=True)

    state = {}
    dropped = 0
    for k, v in raw.items():
        if k.startswith("language_model."):
            state[k[len("language_model.") :]] = v
        elif k.startswith("model.language_model."):
            # transformers >= 4.52 nests the decoder under model.language_model.*
            state["model." + k[len("model.language_model.") :]] = v
        elif k.startswith(("vision_tower.", "multi_modal_projector.", "model.vision_tower.")):
            dropped += 1
        elif k.startswith(("model.", "lm_head.")):
            state[k] = v  # already a text-only checkpoint (e.g. gemma-3-1b-pt)
        else:
            dropped += 1
    print(f"[convert] kept {len(state)} text tensors, dropped {dropped} vision/other", flush=True)
    return state


def build_gemma3_config(text_cfg: dict, vocab_size: int):
    """Build the olmo-core :class:`TransformerConfig` from the checkpoint's own text config.

    Every dimension comes from ``text_cfg`` -- nothing is inherited from the factory defaults,
    which do not match the released checkpoint.
    """
    from olmo_core.nn.transformer import TransformerConfig

    layer_types = text_cfg.get("layer_types")
    if layer_types:
        # Gemma 3 interleaves 5 sliding-window layers then 1 full-attention layer.
        interval = layer_types.index("full_attention") + 1
    else:
        interval = int(text_cfg.get("sliding_window_pattern", 6))

    scaling = text_cfg.get("rope_scaling") or {}
    if scaling and scaling.get("rope_type") not in (None, "linear"):
        sys.exit(f"[convert] unsupported rope_scaling {scaling!r}; only 'linear' is handled.")

    cfg = TransformerConfig.gemma3_like(
        d_model=text_cfg["hidden_size"],
        vocab_size=vocab_size,
        n_layers=text_cfg["num_hidden_layers"],
        n_heads=text_cfg["num_attention_heads"],
        n_kv_heads=text_cfg["num_key_value_heads"],
        hidden_size=text_cfg["intermediate_size"],
        head_dim=text_cfg["head_dim"],
        local_window_size=int(text_cfg["sliding_window"]),
        local_rope_theta=int(text_cfg["rope_local_base_freq"]),
        global_rope_theta=int(text_cfg["rope_theta"]),
        global_rope_linear_scaling_factor=float(scaling.get("factor", 0.0)),
        global_layer_interval=interval,
        layer_norm_eps=float(text_cfg["rms_norm_eps"]),
    )
    q_scalar = text_cfg.get("query_pre_attn_scalar")
    if q_scalar is not None and int(q_scalar) != int(text_cfg["head_dim"]):
        sys.exit(
            f"[convert] query_pre_attn_scalar={q_scalar} != head_dim={text_cfg['head_dim']}; "
            "olmo-core's Attention uses the 1/sqrt(head_dim) scale, so this checkpoint would be "
            "converted with the WRONG softmax scale. Add an explicit softmax_scale override."
        )
    for cap in ("final_logit_softcapping", "attn_logit_softcapping"):
        if text_cfg.get(cap):
            sys.exit(
                f"[convert] checkpoint sets {cap}={text_cfg[cap]}, which olmo-core's gemma3 "
                "path does not implement."
            )
    return cfg


def marker_audit(emb: torch.Tensor, ids, real_vocab_size: int) -> dict:
    """Pairwise cosine + norms of the marker rows against the trained-row norm distribution."""
    emb = emb.float()
    trained = emb[:real_vocab_size].norm(dim=-1)
    med = trained.median().item()
    rows = {"doc_start": ids.doc_start, "doc_end": ids.doc_end}
    out = {
        "marker_ids": rows,
        "trained_row_median_norm": med,
        "norms": {k: emb[i].norm().item() for k, i in rows.items()},
        "norm_ratios": {k: emb[i].norm().item() / med for k, i in rows.items()},
    }
    a, b = emb[ids.doc_start], emb[ids.doc_end]
    out["cos_doc_start_doc_end"] = torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()
    out["verdict"] = (
        "PASS"
        if abs(out["cos_doc_start_doc_end"]) < 0.9
        and all(0.5 < r < 2.0 for r in out["norm_ratios"].values())
        else "FAIL-CHECK"
    )
    return out


def repair_markers(emb: torch.Tensor, ids, tok, seed: int = 1234) -> None:
    """Seed each marker row from a distinct TRAINED delimiter row + small orthogonal noise.

    In-place. Matches ``src/scripts/data/fix_marker_embeddings.py``'s post-2026-07-14 behaviour:
    the donor gives an in-distribution NORM, the perturbation makes the two markers mutually
    distinguishable (bit-identical untrained rows are the original bug).
    """
    g = torch.Generator().manual_seed(seed)
    donors = []
    for s in DONOR_STRINGS:
        enc = tok(s, add_special_tokens=False)["input_ids"]
        if len(enc) == 1:
            donors.append(enc[0])
    if len(donors) < 2:
        raise SystemExit(f"[repair] need >=2 single-token donors, got {donors}")
    for k, tgt in enumerate((ids.doc_start, ids.doc_end)):
        donor = emb[donors[k % len(donors)]].float()
        noise = torch.randn(donor.shape, generator=g)
        noise -= noise.dot(donor) / donor.dot(donor) * donor  # orthogonalize
        noise *= 0.35 * donor.norm() / noise.norm()
        new = donor + noise
        new *= donor.norm() / new.norm()  # keep the donor's (in-distribution) norm
        emb[tgt] = new.to(emb.dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="local flat HF Gemma-3 checkpoint dir")
    ap.add_argument("--out", required=True, help="output olmo-core model-only distcp dir")
    ap.add_argument("--marker-set", default="gemma")
    ap.add_argument(
        "--fix-markers",
        action="store_true",
        help="repair the document-boundary marker embedding rows before saving (REQUIRED for any "
        "document-chunked training -- see records/document-chunked-marker-embeddings.md)",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from olmo_core.data.document_chunk_landmark import RESERVED_IDS
    from olmo_core.data.tokenizer import TokenizerConfig
    from olmo_core.distributed.checkpoint import save_model_and_optim_state
    from olmo_core.io import join_path
    from olmo_core.nn.hf.convert import convert_state_from_hf

    ids = RESERVED_IDS[args.marker_set]
    raw_cfg = json.load(open(os.path.join(args.base_dir, "config.json")))
    text_cfg = raw_cfg.get("text_config", raw_cfg)
    vocab_size = int(text_cfg["vocab_size"])
    print(
        f"[convert] {args.base_dir}: gemma3 text vocab={vocab_size} "
        f"d_model={text_cfg['hidden_size']} n_layers={text_cfg['num_hidden_layers']} "
        f"n_heads={text_cfg['num_attention_heads']} head_dim={text_cfg['head_dim']}",
        flush=True,
    )

    model_cfg = build_gemma3_config(text_cfg, vocab_size)
    hf_state = _load_hf_text_state(args.base_dir)

    tie = bool(text_cfg.get("tie_word_embeddings", True))
    if "lm_head.weight" not in hf_state:
        if not tie:
            sys.exit("[convert] no lm_head.weight and tie_word_embeddings is false -- refusing.")
        hf_state["lm_head.weight"] = hf_state["model.embed_tokens.weight"].clone()
        print("[convert] tied embeddings: copied embed_tokens -> lm_head", flush=True)

    cfg_obj = types.SimpleNamespace(**text_cfg)
    converted = convert_state_from_hf(cfg_obj, hf_state, model_type="gemma3_text")
    print(f"[convert] converted {len(converted)} tensors", flush=True)

    model = model_cfg.build(init_device="cpu")
    missing, unexpected = model.load_state_dict(converted, strict=False)
    missing = [k for k in missing if "rope" not in k and "inv_freq" not in k]
    if missing or unexpected:
        sys.exit(
            f"[convert] strict-load mismatch: missing={list(missing)[:8]} "
            f"unexpected={list(unexpected)[:8]}"
        )
    n_params = sum(p.numel() for p in model.parameters())
    hf_params = sum(v.numel() for v in hf_state.values())
    print(
        f"[convert] strict load OK: olmo params={n_params:,} vs HF text tensors={hf_params:,}",
        flush=True,
    )
    if n_params != hf_params:
        sys.exit(
            f"[convert] PARAM COUNT MISMATCH olmo={n_params:,} hf={hf_params:,} -- the "
            "architecture does not match the checkpoint."
        )

    tok = AutoTokenizer.from_pretrained(args.base_dir)
    emb = model.embeddings.weight.data
    before = marker_audit(emb, ids, ids.real_vocab_size)
    print("[audit-before] " + json.dumps(before, indent=2), flush=True)
    after = None
    if args.fix_markers:
        repair_markers(emb, ids, tok)
        # keep the tied LM head consistent with the repaired embedding rows
        model.lm_head.w_out.weight.data[ids.doc_start] = emb[ids.doc_start]
        model.lm_head.w_out.weight.data[ids.doc_end] = emb[ids.doc_end]
        after = marker_audit(emb, ids, ids.real_vocab_size)
        print("[audit-after] " + json.dumps(after, indent=2), flush=True)
        if after["verdict"] != "PASS":
            sys.exit("[convert] marker repair did NOT reach PASS -- refusing to save.")

    os.makedirs(args.out, exist_ok=True)
    tok_cfg = TokenizerConfig(
        vocab_size=vocab_size,
        eos_token_id=ids.eos,
        pad_token_id=int(raw_cfg.get("pad_token_id", 0)),
        bos_token_id=int(raw_cfg.get("bos_token_id", 2)),
        identifier=args.base_dir,
    )
    save_model_and_optim_state(
        str(join_path(args.out, "model_and_optim")), model, save_overwrite=True
    )
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(
            {
                "model": model_cfg.as_config_dict(),
                "dataset": {"tokenizer": tok_cfg.as_config_dict()},
            },
            f,
        )
    meta = os.path.join(args.out, "model_and_optim", ".metadata")
    if not os.path.exists(meta):
        sys.exit(f"[convert] FAILED: {meta} missing -- conversion did not complete cleanly.")
    with open(os.path.join(args.out, "marker_audit.json"), "w") as f:
        json.dump(
            {
                "base_dir": args.base_dir,
                "out": args.out,
                "marker_set": args.marker_set,
                "before": before,
                "after": after,
            },
            f,
            indent=2,
        )
    print(f"[convert] DONE -> {args.out} (verified {meta})", flush=True)


if __name__ == "__main__":
    main()
