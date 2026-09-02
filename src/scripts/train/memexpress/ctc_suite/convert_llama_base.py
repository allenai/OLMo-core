"""Convert a local HF Llama-3 base -> OLMo-core model-only distcp, **with the marker-embedding
repair applied in the same pass**, for the CTC dense-vs-chunked suite.

This is the Llama counterpart of ``convert_qwen35_base.py``. Three things make Llama different
from the Qwen bases and each of them silently produces a broken run if skipped:

1. **No 3B factory.** olmo-core ships ``llama3_1B``/``llama3_8B`` but not a 3B, so the config comes
   from :mod:`llama_configs`, which derives every dimension from the checkpoint's own
   ``config.json`` and asserts the resulting parameter count. See that module.

2. **Tied embeddings.** ``Llama-3.2-3B`` sets ``tie_word_embeddings: true``, so its safetensors
   contain **no** ``lm_head.weight``. OLMo-core's transformer keeps the LM head as a separate
   parameter, so without intervention it would be left at *random init* -- the model would emit
   noise and the run would look like a modelling result. We materialise the tie by copying
   ``model.embed_tokens.weight`` into ``lm_head.weight`` before conversion, and verify afterwards
   that the two olmo-core tensors are bit-identical.

3. **Untrained marker rows.** Llama's ``<|reserved_special_token_N|>`` rows are the same trap as
   Qwen's ``<|box_start|>``/``<|box_end|>`` (``records/document-chunked-marker-embeddings.md``):
   never trained, mutually indistinguishable, and -- the part that bit this project a second time --
   *out of distribution in NORM*, which RMSNorm amplifies into full-strength noise and flatlines
   document-chunked training at chance for **every** mask (``records/n100-chunked-marker-position-bug.md``).
   The repair here is the same one ``src/scripts/data/fix_marker_embeddings.py`` applies to Qwen:
   seed each marker from a real **trained delimiter row** (``«``/``»``/``§``/``¶``) plus a small
   jitter, then assert both (a) pairwise cosine < 0.9 and (b) norm within [0.5, 2.0] x the trained
   row median. It is done here rather than as a second pass because Llama ties its LM head: the
   repaired rows must land in ``embeddings`` *and* ``lm_head.w_out`` consistently, which is easiest
   while both are in memory.

Usage (single process, CPU is fine; ~30 GB RAM at fp32)::

    python src/scripts/train/memexpress/ctc_suite/convert_llama_base.py \\
        --base-dir /scratch/users/prasann/hf_models/Llama-3.2-3B \\
        --tokenizer /scratch/users/prasann/hf_models/Llama-3.2-3B-marker-tok \\
        --out /scratch/users/prasann/ctc_suite_staged/bases/llama32-3b-base-fixmark
"""

import argparse
import glob
import json
import os
import sys
import types

import torch

try:  # package import (PYTHONPATH=src) or same-directory fallback
    from scripts.train.memexpress.ctc_suite.llama_configs import (
        LLAMA3_1_8B_HF_NUM_PARAMS,
        LLAMA3_1_8B_HF_SHAPE,
        LLAMA_MARKER_TOKENIZER,
        assert_matches_hf,
        llama3_1_8B,
        llama3_2_3B,
    )
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from llama_configs import (  # type: ignore[no-redef]
        LLAMA3_1_8B_HF_NUM_PARAMS,
        LLAMA3_1_8B_HF_SHAPE,
        LLAMA_MARKER_TOKENIZER,
        assert_matches_hf,
        llama3_1_8B,
        llama3_2_3B,
    )

#: Trained delimiter tokens that seed each marker row. Delimiters are the right donors: the model
#: already reads them as "a boundary is here", which is exactly the marker's job. Same choice as
#: ``fix_marker_embeddings.py`` makes for Qwen.
DONOR_STRINGS = {"doc_start": "«", "doc_end": "»", "landmark": "§", "pad": "¶"}


def repair_markers(emb: torch.Tensor, tok, ids, *, seed: int, jitter: float) -> dict:
    """Give each reserved marker row a distinct, in-distribution embedding.

    :param emb: The embedding matrix (modified in place).
    :param tok: The HF tokenizer (used to resolve the donor tokens).
    :param ids: The family's :class:`~olmo_core.data.document_chunk_landmark.ReservedIds`.
    :param seed: RNG seed for the jitter.
    :param jitter: Per-marker noise added to the donor row, as a fraction of the trained-row std.

    :returns: An audit dict (per-marker before/after norms, pairwise cosines, verdict inputs).

    :raises SystemExit: If a donor is not a single trained token, or if the repaired rows fail the
        distinguishability / in-distribution-norm assertions.
    """
    trained = emb[: ids.real_vocab_size].float()
    median_norm = trained.norm(dim=-1).median().item()
    std = trained.std().item()
    g = torch.Generator().manual_seed(seed)

    marker_ids = {
        "doc_start": ids.doc_start,
        "doc_end": ids.doc_end,
        "landmark": ids.landmark,
        "pad": ids.pad,
    }
    audit = {"trained_row_median_norm": median_norm, "markers": {}}
    # Record the pre-repair pairwise cosines: on an unrepaired base these are ~1.0, which is the
    # entire bug. Keeping them in the audit makes "we actually needed this" auditable later.
    before_ids = list(marker_ids.values())
    audit["cos_before"] = {
        f"{a}|{b}": torch.nn.functional.cosine_similarity(
            emb[a].float()[None], emb[b].float()[None]
        ).item()
        for i, a in enumerate(before_ids)
        for b in before_ids[i + 1 :]
    }

    for name, tid in marker_ids.items():
        donor_str = DONOR_STRINGS[name]
        donor_ids = tok.encode(donor_str, add_special_tokens=False)
        if len(donor_ids) != 1 or donor_ids[0] >= ids.real_vocab_size or donor_ids[0] == ids.eos:
            raise SystemExit(
                f"donor {donor_str!r} for {name} is not a single trained token: {donor_ids}"
            )
        donor = donor_ids[0]
        before = emb[tid].float().norm().item()
        vec = emb[donor].float() + torch.randn(emb.shape[1], generator=g) * (std * jitter)
        emb[tid] = vec.to(emb.dtype)
        after = emb[tid].float().norm().item()
        audit["markers"][name] = {
            "id": tid,
            "donor": donor_str,
            "donor_id": donor,
            "norm_before": before,
            "norm_after": after,
            "norm_ratio_after": after / median_norm,
        }
        print(
            f"  {tid} {name:10s} <- {donor_str!r} (id {donor})   norm {before:.4f} -> {after:.4f} "
            f"({after / median_norm:.2f}x trained median {median_norm:.4f})"
        )

    # (1) mutually distinguishable -- the original bug was bit-identical rows.
    ids_list = list(marker_ids.values())
    cos_after = {}
    for i, a in enumerate(ids_list):
        for b in ids_list[i + 1 :]:
            cos = torch.nn.functional.cosine_similarity(
                emb[a].float()[None], emb[b].float()[None]
            ).item()
            cos_after[f"{a}|{b}"] = cos
            if abs(cos) >= 0.9:
                raise SystemExit(f"markers {a}/{b} too similar after repair (cos={cos:.4f})")
    audit["cos_after"] = cos_after
    # (2) ...AND in-distribution in scale. A marker at a fraction of a real token's norm is
    #     amplified by RMSNorm into full-strength noise and flatlines training on ANY mask.
    for name, rec in audit["markers"].items():
        if not 0.5 < rec["norm_ratio_after"] < 2.0:
            raise SystemExit(
                f"marker {name} norm is {rec['norm_ratio_after']:.2f}x the trained-row median -- "
                "out of distribution. This is the exact failure the donor-row init prevents."
            )
    print("markers are distinguishable AND in-distribution in norm")
    return audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", required=True, help="local flat HF Llama dir (config + weights)")
    ap.add_argument(
        "--tokenizer",
        default=LLAMA_MARKER_TOKENIZER,
        help="patched tokenizer dir (make_llama_marker_tokenizer.py); used for donor lookup + the "
        "TokenizerConfig sidecar",
    )
    ap.add_argument("--out", required=True, help="output olmo-core model-only distcp dir")
    ap.add_argument(
        "--scale",
        default="3b",
        choices=["3b", "8b"],
        help="which Llama the --base-dir holds: 3b=Llama-3.2-3B (tied LM head, RoPE factor 32), "
        "8b=Llama-3.1-8B (untied LM head, RoPE factor 8). Selects the factory AND the shape table "
        "asserted against config.json, so a wrong value fails here rather than training a "
        "mis-shaped model. Defaults to 3b so existing callers are unchanged.",
    )
    ap.add_argument("--family", default="llama", help="RESERVED_IDS key")
    ap.add_argument("--seed", type=int, default=34521)
    ap.add_argument("--jitter", type=float, default=0.1)
    ap.add_argument(
        "--no-marker-repair",
        action="store_true",
        help="convert only, leave the untrained marker rows alone. ONLY for auditing how broken "
        "the raw base is -- a run trained from such a base flatlines at chance.",
    )
    args = ap.parse_args()

    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from olmo_core.data.document_chunk_landmark import RESERVED_IDS
    from olmo_core.data.tokenizer import TokenizerConfig
    from olmo_core.distributed.checkpoint import save_model_and_optim_state
    from olmo_core.nn.hf.convert import convert_state_from_hf

    ids = RESERVED_IDS[args.family]
    raw = json.load(open(os.path.join(args.base_dir, "config.json")))

    if args.scale == "8b":
        model_cfg = llama3_1_8B(vocab_size=raw["vocab_size"])
        assert_matches_hf(
            args.base_dir,
            model_cfg,
            shape=LLAMA3_1_8B_HF_SHAPE,
            hf_num_params=LLAMA3_1_8B_HF_NUM_PARAMS,
            factory_name="llama3_1_8B",
        )
    else:
        model_cfg = llama3_2_3B(vocab_size=raw["vocab_size"])
        assert_matches_hf(args.base_dir, model_cfg)
    print(
        f"[convert] {args.base_dir}: shape + param-count check PASSED "
        f"({model_cfg.num_params:,} olmo-core params, untied LM head)",
        flush=True,
    )

    hf_state = {}
    for shard in sorted(glob.glob(os.path.join(args.base_dir, "*.safetensors"))):
        hf_state.update(load_file(shard))
    print(f"[convert] loaded {len(hf_state)} HF tensors", flush=True)

    # --- tied LM head: materialise it, or olmo-core's separate lm_head stays at random init ---
    tied = bool(raw.get("tie_word_embeddings"))
    if tied:
        if "lm_head.weight" in hf_state:
            raise SystemExit(
                "config says tie_word_embeddings=true but the checkpoint HAS lm_head.weight; "
                "refusing to guess which one is authoritative."
            )
        hf_state["lm_head.weight"] = hf_state["model.embed_tokens.weight"].clone()
        print("[convert] tie_word_embeddings=true -> copied embed_tokens into lm_head", flush=True)
    elif "lm_head.weight" not in hf_state:
        raise SystemExit("untied config but no lm_head.weight in the checkpoint")

    cfg_obj = types.SimpleNamespace(**raw)
    converted = convert_state_from_hf(cfg_obj, hf_state, model_type=raw["model_type"])
    model = model_cfg.build(init_device="cpu")
    missing, unexpected = model.load_state_dict(converted, strict=False)
    # Note: olmo-core keeps non-persistent rope buffers, which legitimately show up as "missing".
    missing = [k for k in missing if "rope" not in k and "inv_freq" not in k]
    if missing or unexpected:
        raise SystemExit(
            f"[convert] strict-load mismatch: missing={list(missing)[:8]} "
            f"unexpected={list(unexpected)[:8]}"
        )
    print(f"[convert] strict load OK ({len(converted)} tensors)", flush=True)

    emb = model.embeddings.weight.data
    head = model.lm_head.w_out.weight.data
    if tied and not torch.equal(emb, head):
        raise SystemExit("[convert] tied copy failed: embeddings != lm_head after load")

    audit = {"base_dir": args.base_dir, "out": args.out, "tied_lm_head": tied}
    if args.no_marker_repair:
        print("[convert] WARNING: --no-marker-repair; markers left UNTRAINED", flush=True)
        audit["marker_repair"] = None
    else:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        audit["marker_repair"] = repair_markers(emb, tok, ids, seed=args.seed, jitter=args.jitter)
        if tied:
            # Keep the head consistent with the repaired embeddings (that is what the tie meant).
            head.copy_(emb)
        else:
            # ⚠ UNTIED CHECKPOINTS NEED THE HEAD REPAIRED SEPARATELY (Llama-3.1-8B; 3.2-3B is
            # tied and takes the branch above). Repairing only `emb` fixes the INPUT side and
            # leaves the OUTPUT side's marker rows exactly as untrained as they arrived -- the
            # same defect the repair exists to remove, just on the logits. An untrained head row
            # with an out-of-distribution norm can dominate the softmax and make the model emit
            # `<|box_start|>`/`<|box_end|>` into its answer, which the graders then score as a
            # malformed generation. Run the same donor-seeding on the head matrix: because
            # repair_markers derives its donors and its target norm from the matrix it is handed,
            # this seeds the head's marker rows from the head's OWN trained delimiter rows, which
            # is the correct output-side analogue rather than a copy of the input embeddings.
            audit["marker_repair_lm_head"] = repair_markers(
                head, tok, ids, seed=args.seed + 1, jitter=args.jitter
            )
        audit["marker_ids"] = ids._asdict()

    tok_cfg = TokenizerConfig(
        vocab_size=raw["vocab_size"],
        eos_token_id=ids.eos,
        pad_token_id=ids.eos,
        bos_token_id=raw.get("bos_token_id"),
        identifier=args.tokenizer,
    )
    os.makedirs(args.out, exist_ok=True)
    save_model_and_optim_state(
        os.path.join(args.out, "model_and_optim"), model, save_overwrite=True
    )
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(
            {
                "model": model_cfg.as_config_dict(),
                "dataset": {"tokenizer": tok_cfg.as_config_dict()},
            },
            f,
        )

    metadata_path = os.path.join(args.out, "model_and_optim", ".metadata")
    if not os.path.exists(metadata_path):
        raise SystemExit(
            f"[convert] FAILED: {metadata_path} missing after save -- a base without it silently "
            "trains FROM SCRATCH."
        )
    with open(os.path.join(args.out, "marker_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print(f"[convert] DONE -> {args.out} (verified {metadata_path})", flush=True)
    print("[audit] " + json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
