"""Audit (and, only if needed, repair) the document-marker embedding rows of an OLMo3 distcp base.

This is the OLMo counterpart of ``src/scripts/data/fix_marker_embeddings.py`` and it enforces BOTH
known marker bugs as gates, not just the first one:

1. **Cosine.** Untrained reserved rows are often bit-identical, so the model cannot tell an "open
   document" marker from a "close document" one and marker-dense training flatlines at chance while
   looking exactly like a modeling result (``records/document-chunked-marker-embeddings.md``).
2. **Norm.** A marker row at a small fraction of a real token's norm is amplified by RMSNorm into a
   full-strength *noise* vector at every marker position, which flatlines training for **every**
   mask -- including plain causal (``records/n100-chunked-marker-position-bug.md``).

Unlike Qwen3, OLMo's markers here are ``<|extra_id_1|>``/``<|extra_id_2|>`` -- ids *inside* the real
dolma2 vocab, not padded-region rows -- so they may genuinely be healthy. This script MEASURES first
and only repairs when a gate fails, because "repairing" an already-trained row would throw away
signal. Read the audit numbers it prints before training either arm.

Usage::

    python olmo3_marker_audit.py --checkpoint <base>/model_and_optim --out <base>/marker_audit.json
    # add --repair-to <dir> to write a repaired copy when the audit FAILS
"""

import argparse
import json

import torch

from olmo_core.data.document_chunk_landmark import reserved_ids
from olmo_core.distributed.checkpoint import load_keys

EMB_KEY = "model.embeddings.weight"

#: Trained delimiter tokens that seed a marker if (and only if) the audit fails. Delimiters are the
#: right donors: the model already reads them as "a boundary is here", which is the marker's job.
DONOR_STRS = {"doc_start": "«", "doc_end": "»"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="model_and_optim distcp dir of the base")
    ap.add_argument("--out", default=None, help="write the audit JSON here")
    ap.add_argument("--family", default="olmo3", help="RESERVED_IDS family key")
    # The MEASUREMENT path is architecture-free -- it reads the embedding matrix with load_keys and
    # never builds a model. Only the REPAIR path needs a model, and it used to hardcode olmo3_7B,
    # which silently made this script olmo3-only: pointed at the Olmo-Hybrid base it measured
    # correctly, decided a repair was needed, and then died with
    #   ValueError: Size mismatch between saved [100352, 3840] and current: [100352, 4096]
    # because Olmo-3 is d_model 4096 and Olmo-Hybrid is 3840. Same tokenizer and same marker ids,
    # different backbone -- so the family key stays "olmo3" while the ARCH must be selectable.
    ap.add_argument(
        "--arch",
        default="olmo3",
        choices=["olmo3", "olmo_hybrid"],
        help="model architecture to rebuild for the repair path (measurement needs none)",
    )
    ap.add_argument(
        "--tokenizer",
        default="/scratch/users/prasann/hf_models/Olmo-3-1025-7B-docchunk",
        help="tokenizer dir (only needed to resolve donor rows for a repair)",
    )
    ap.add_argument(
        "--repair-to",
        default=None,
        help="if the audit FAILS, write a repaired base here (a model_and_optim/ is created under "
        "it). Omitted = audit only.",
    )
    ap.add_argument("--seed", type=int, default=34521)
    ap.add_argument("--jitter", type=float, default=0.1)
    args = ap.parse_args()

    ids = reserved_ids(args.family)
    # load_keys yields tensors in the order of the requested keys (it is a generator, not a dict).
    emb = next(iter(load_keys(args.checkpoint, [EMB_KEY]))).float()
    print(f"embedding matrix: {tuple(emb.shape)}  (real vocab ends at {ids.real_vocab_size})")

    # Reference scale: the norm distribution of rows that certainly ARE trained.
    trained = emb[: ids.real_vocab_size - 100]
    norms = trained.norm(dim=-1)
    median_norm = norms.median().item()

    marker = {"doc_start": ids.doc_start, "doc_end": ids.doc_end}
    row = {k: emb[v] for k, v in marker.items()}
    cos = torch.nn.functional.cosine_similarity(row["doc_start"][None], row["doc_end"][None]).item()
    report = {
        "family": args.family,
        "checkpoint": args.checkpoint,
        "embedding_shape": list(emb.shape),
        "trained_row_median_norm": median_norm,
        "trained_row_norm_p05": norms.quantile(0.05).item(),
        "trained_row_norm_p95": norms.quantile(0.95).item(),
        "marker_ids": marker,
        "marker_norms": {k: row[k].norm().item() for k in row},
        "marker_norm_ratios": {k: row[k].norm().item() / median_norm for k in row},
        "marker_pairwise_cosine": cos,
        "bit_identical": bool(torch.equal(row["doc_start"], row["doc_end"])),
    }

    print(
        f"trained-row norm: median={median_norm:.4f} "
        f"p05={report['trained_row_norm_p05']:.4f} p95={report['trained_row_norm_p95']:.4f}"
    )
    for k, v in marker.items():
        print(
            f"  {k:10s} id={v}  norm={report['marker_norms'][k]:.4f} "
            f"({report['marker_norm_ratios'][k]:.2f}x median)"
        )
    print(f"  pairwise cosine = {cos:+.4f}   bit_identical={report['bit_identical']}")

    cos_ok = abs(cos) < 0.9
    norm_ok = all(0.5 < r < 2.0 for r in report["marker_norm_ratios"].values())
    report["cosine_gate_pass"] = cos_ok
    report["norm_gate_pass"] = norm_ok
    report["audit_pass"] = cos_ok and norm_ok
    print(
        f"VERDICT: cosine_gate={'PASS' if cos_ok else 'FAIL'} "
        f"norm_gate={'PASS' if norm_ok else 'FAIL'} -> "
        f"{'no repair needed' if report['audit_pass'] else 'REPAIR REQUIRED'}"
    )

    if not report["audit_pass"] and args.repair_to:
        from transformers import AutoTokenizer

        from olmo_core.data.tokenizer import TokenizerConfig
        from olmo_core.distributed.checkpoint import (
            load_model_and_optim_state,
            save_model_and_optim_state,
        )
        from olmo_core.nn.transformer import TransformerConfig

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        vocab = TokenizerConfig.dolma2().padded_vocab_size()
        if args.arch == "olmo_hybrid":
            from olmo_hybrid_configs import (
                olmo_hybrid_7B_ctc,  # type: ignore[import-not-found]
            )

            model_config = olmo_hybrid_7B_ctc(vocab_size=vocab)
        else:
            model_config = TransformerConfig.olmo3_7B(vocab_size=vocab)
        model = model_config.build(init_device="cpu")
        load_model_and_optim_state(args.checkpoint, model)
        w = model.embeddings.weight.data
        std = w[: ids.real_vocab_size].float().std()
        g = torch.Generator().manual_seed(args.seed)
        for name, tid in marker.items():
            donor_ids = tok.encode(DONOR_STRS[name], add_special_tokens=False)
            if len(donor_ids) != 1 or donor_ids[0] >= ids.real_vocab_size:
                raise SystemExit(f"donor {DONOR_STRS[name]!r} is not a single trained token")
            vec = w[donor_ids[0]].float() + torch.randn(w.shape[1], generator=g) * (
                std * args.jitter
            )
            w[tid] = vec.to(w.dtype)
            print(
                f"  repaired {name} (id {tid}) <- {DONOR_STRS[name]!r} "
                f"norm={w[tid].float().norm():.4f}"
            )
        new_cos = torch.nn.functional.cosine_similarity(
            w[marker["doc_start"]].float()[None], w[marker["doc_end"]].float()[None]
        ).item()
        assert abs(new_cos) < 0.9, f"repair left markers too similar (cos={new_cos:.4f})"
        for name, tid in marker.items():
            ratio = (w[tid].float().norm() / median_norm).item()
            assert 0.5 < ratio < 2.0, f"repaired {name} norm ratio {ratio:.2f} out of distribution"
        save_model_and_optim_state(f"{args.repair_to}/model_and_optim", model, save_overwrite=True)
        report["repaired_to"] = f"{args.repair_to}/model_and_optim"
        report["post_repair_cosine"] = new_cos
        print(f"wrote repaired base -> {args.repair_to}/model_and_optim")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
