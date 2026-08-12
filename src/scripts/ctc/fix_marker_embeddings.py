"""
Audit -- and repair -- the reserved marker embedding rows of a base checkpoint.

**Run this on every fresh base before any document-chunked or landmark training.** Skipping it does
not crash; it produces a run that trains to chance and reads as a modeling result.

Qwen3 never trains ``<|box_start|>`` / ``<|box_end|>``, nor the landmark/pad rows that live past the
real vocab in the padded embedding matrix. Out of the box those four rows are **bit-identical**
(cosine ``+1.0000``), so the model cannot distinguish an "open document" marker from a "close
document" one -- on marker-dense shards that is most of the structure the task depends on. See
``records/document-chunked-marker-embeddings.md``.

Two properties have to hold after a repair, and the first version of this script only got one:

1. the markers are mutually **distinguishable** -- ``|cos| < 0.9`` for every pair; and
2. their **norm** is in-distribution against the trained-row median -- ratio in ``0.5x .. 2x``.

Getting (1) without (2) is worse than it sounds. A marker at ~1/3.6 the norm of a real token is
amplified by RMSNorm into a full-strength *noise* vector at every marker position, which flatlines
training at CE ~0.79 for **every** mask -- including plain causal, where an unrestricted model cannot
even memorize the data. That reads as "the mask is too restrictive" when it is not. So the repair
seeds each marker from a real *trained delimiter* row (``«`` ``»`` ``§`` ``¶``) plus a little jitter,
and asserts both properties before writing. See ``records/n100-chunked-marker-position-bug.md``.

Reporting **both** numbers is the point of ``--check-only``. Cosine alone calls a pre-2026-07-14
repair healthy; norm alone calls a raw Qwen3 base healthy (0.481 is low but not absurd). It is the
pair that identifies the state:

===================================  =========  ===========  ==========================
state                                cos        norm ratio   verdict
===================================  =========  ===========  ==========================
raw Qwen3                            ``+1.000``  ``0.48``     POISONED
repaired before 2026-07-14           ``<0.9``    ``~0.28``    NORM OUT OF DISTRIBUTION
repaired by this script              ``+0.506``  ``1.06``     OK
===================================  =========  ===========  ==========================

Reserved ids are looked up by tokenizer family via
:func:`~olmo_core.data.document_chunk_landmark.reserved_ids` -- Qwen3 and Qwen3.5 do not share a
vocabulary, so a module-level constant would silently repair the wrong rows.

Usage::

    # audit only -- no HF tokenizer needed, no write
    python src/scripts/ctc/fix_marker_embeddings.py --base /path/model_and_optim --check-only \\
        --model qwen3_0_6B --tokenizer qwen3

    # repair; a model_and_optim/ is written under --out
    python src/scripts/ctc/fix_marker_embeddings.py --base /path/model_and_optim \\
        --out /path/base-fixmark --model qwen3_0_6B --tokenizer qwen3

Always re-audit the copy the trainer will *actually read* (after any distcp round trip or rsync to
another node's disk) rather than trusting the repairing process's report.

``--check-only`` **exits non-zero on a bad base**, so it works as a gate in a launcher under
``set -e`` rather than only as something a human reads.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Run against THIS checkout, not the conda env's editable install (which points at the
# pre-migration tree). Mirrors src/scripts/ctc/train/sft.py.
_REPO = Path(__file__).resolve().parents[3]
for _src in (_REPO / "src", _REPO / "ctc" / "src"):
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from olmo_core.data.document_chunk_landmark import ReservedIds, reserved_ids  # noqa: E402

#: Trained delimiter tokens whose rows seed each marker. Delimiters are the right donors: the model
#: already reads them as "a boundary is here", which is exactly the marker's job -- and being real
#: trained rows they carry an in-distribution norm for free, which random init does not.
DONOR_STRINGS: Dict[str, str] = {
    "doc_start": "«",
    "doc_end": "»",
    "landmark": "§",
    "pad": "¶",
}

#: Any pair of markers above this cosine is treated as indistinguishable.
MAX_MARKER_COS = 0.9

#: A marker's norm must land within this factor of the trained-row median.
MIN_NORM_RATIO, MAX_NORM_RATIO = 0.5, 2.0


def marker_ids(ids: ReservedIds) -> Dict[str, int]:
    """
    The four reserved rows this script owns, by name.

    :param ids: The :class:`~olmo_core.data.document_chunk_landmark.ReservedIds` for a family.

    :returns: A mapping of marker name to token id, in a stable order.
    """
    return {
        "doc_start": ids.doc_start,
        "doc_end": ids.doc_end,
        "landmark": ids.landmark,
        "pad": ids.pad,
    }


def median_trained_norm(emb: torch.Tensor, ids: ReservedIds) -> float:
    """
    The median L2 norm over the *trained* rows, i.e. the scale a marker should match.

    :param emb: The embedding matrix, ``[vocab, d_model]``.
    :param ids: The reserved-id set; rows at or past ``real_vocab_size`` are untrained padding.

    :returns: The median norm.
    """
    return emb[: ids.real_vocab_size].float().norm(dim=-1).median().item()


def marker_cosines(emb: torch.Tensor, ids: ReservedIds) -> Dict[Tuple[str, str], float]:
    """
    Pairwise cosine similarity between every pair of marker rows.

    :param emb: The embedding matrix.
    :param ids: The reserved-id set.

    :returns: ``{(name_a, name_b): cosine}`` for each unordered pair.
    """
    named = marker_ids(ids)
    keys = list(named)
    out: Dict[Tuple[str, str], float] = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = named[keys[i]], named[keys[j]]
            out[(keys[i], keys[j])] = F.cosine_similarity(
                emb[a].float()[None], emb[b].float()[None]
            ).item()
    return out


def marker_norm_ratios(emb: torch.Tensor, ids: ReservedIds) -> Dict[str, float]:
    """
    Each marker's norm as a ratio of the trained-row median.

    :param emb: The embedding matrix.
    :param ids: The reserved-id set.

    :returns: ``{marker_name: norm / median_trained_norm}``.
    """
    median = median_trained_norm(emb, ids)
    return {name: emb[tid].float().norm().item() / median for name, tid in marker_ids(ids).items()}


def problems(emb: torch.Tensor, ids: ReservedIds) -> List[str]:
    """
    Every way this embedding matrix's markers are unusable, as human-readable strings.

    Both halves are checked. A repair that fixes only the cosine leaves markers at ~1/3.6 of a real
    token's norm, which RMSNorm turns into full-strength noise and flatlines training for every
    mask -- so a norm violation is a hard failure here, not a warning.

    :param emb: The embedding matrix.
    :param ids: The reserved-id set.

    :returns: An empty list if the base is usable.
    """
    out: List[str] = []
    for (a, b), cos in marker_cosines(emb, ids).items():
        if abs(cos) > MAX_MARKER_COS:
            out.append(f"markers {a}/{b} are indistinguishable (cos={cos:+.4f})")
    for name, ratio in marker_norm_ratios(emb, ids).items():
        if not MIN_NORM_RATIO < ratio < MAX_NORM_RATIO:
            out.append(
                f"marker {name} norm is {ratio:.3f}x the trained-row median -- out of "
                "distribution. This is the exact failure the donor-row init exists to prevent."
            )
    return out


def report(emb: torch.Tensor, ids: ReservedIds, when: str) -> None:
    """
    Print the two numbers that decide whether a base is usable, and a verdict.

    :param emb: The embedding matrix.
    :param ids: The reserved-id set for this tokenizer family.
    :param when: Label, e.g. ``"BEFORE"``.
    """
    median = median_trained_norm(emb, ids)
    print(f"--- markers {when} ---")
    print(f"trained-row median norm = {median:.4f}")
    for name, tid in marker_ids(ids).items():
        n = emb[tid].float().norm().item()
        print(f"  {name:10s} id={tid:6d}  norm={n:.4f}  ratio_to_median={n / median:.3f}")
    for (a, b), cos in marker_cosines(emb, ids).items():
        print(f"  cos({a}, {b}) = {cos:+.4f}")

    # The headline pair: doc_start vs doc_end cosine, and doc_start's norm ratio.
    cos = marker_cosines(emb, ids)[("doc_start", "doc_end")]
    ratio = marker_norm_ratios(emb, ids)["doc_start"]
    issues = problems(emb, ids)
    verdict = "OK" if not issues else "; ".join(issues)
    print(f"VERDICT {when}: cos(doc_start, doc_end)={cos:+.4f}  norm_ratio={ratio:.3f}  -> {verdict}")


def repair_markers(
    emb: torch.Tensor,
    ids: ReservedIds,
    donors: Dict[str, int],
    *,
    seed: int = 34521,
    jitter: float = 0.1,
    verbose: bool = True,
) -> None:
    """
    Overwrite each marker row in place with a jittered copy of a trained donor row.

    Donors are *real trained rows*, which is what buys property (2): the repaired marker inherits an
    in-distribution norm instead of whatever a random init happens to produce. The jitter is what
    buys property (1) when two donors are themselves related (``«`` and ``»`` sit at cos 0.60).

    :param emb: The embedding matrix, modified in place.
    :param ids: The reserved-id set.
    :param donors: ``{marker_name: donor_token_id}``, covering every key of :func:`marker_ids`.
    :param seed: RNG seed, so a repair is reproducible.
    :param jitter: Noise added per marker, as a fraction of the trained-row standard deviation.
    :param verbose: Print each assignment.

    :raises ValueError: If a donor is missing, or is not a trained row.
    """
    trained = emb[: ids.real_vocab_size].float()
    std = trained.std()
    g = torch.Generator().manual_seed(seed)
    for name, tid in marker_ids(ids).items():
        if name not in donors:
            raise ValueError(f"no donor row given for marker {name!r}")
        donor = donors[name]
        if not 0 <= donor < ids.real_vocab_size or donor == ids.eos:
            raise ValueError(
                f"donor id {donor} for {name!r} is not a trained row "
                f"(real_vocab_size={ids.real_vocab_size}, eos={ids.eos})"
            )
        vec = emb[donor].float() + torch.randn(emb.shape[1], generator=g) * (std * jitter)
        emb[tid] = vec.to(emb.dtype)
        if verbose:
            print(f"  {name:10s} <- donor id {donor}")


def resolve_donors(hf_tokenizer: str, ids: ReservedIds) -> Dict[str, int]:
    """
    Turn :data:`DONOR_STRINGS` into token ids using the model's own tokenizer.

    :param hf_tokenizer: An HF repo id or, preferably, an absolute snapshot directory. ``HF_HOME``
        is overridden by a stale ``TRANSFORMERS_CACHE`` on some of our nodes, and offline lookups by
        repo id then fail with ``LocalEntryNotFoundError`` even though every file is present -- an
        absolute path cannot go wrong that way.
    :param ids: The reserved-id set.

    :returns: ``{marker_name: donor_token_id}``.

    :raises ValueError: If a donor string is not a single trained token.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_tokenizer)
    donors: Dict[str, int] = {}
    for name, s in DONOR_STRINGS.items():
        got = tok.encode(s, add_special_tokens=False)
        if len(got) != 1:
            raise ValueError(f"donor {s!r} for {name} is not a single token: {got}")
        donors[name] = got[0]
        print(f"  donor {name:10s} = {s!r} -> id {got[0]}")
    return donors


def main() -> None:
    """Entry point: audit a base checkpoint, and optionally write a repaired copy."""
    ap = argparse.ArgumentParser(description="Audit/repair reserved marker embedding rows.")
    ap.add_argument("--base", required=True, help="source model_and_optim distcp dir")
    ap.add_argument("--out", help="destination; a model_and_optim/ is written under it")
    ap.add_argument("--check-only", action="store_true", help="report and exit without writing")
    ap.add_argument("--model", default="qwen3_0_6B", help="TransformerConfig factory name")
    ap.add_argument("--tokenizer", default="qwen3", choices=tuple(sorted(("qwen3", "qwen3_5"))))
    ap.add_argument(
        "--hf-tokenizer",
        default="Qwen/Qwen3-0.6B-Base",
        help="HF tokenizer (repo id or snapshot dir) used only to resolve the donor delimiter ids",
    )
    ap.add_argument("--seed", type=int, default=34521)
    ap.add_argument(
        "--jitter",
        type=float,
        default=0.1,
        help="per-marker noise added to the donor row, as a fraction of the trained-row std",
    )
    args = ap.parse_args()
    if not args.check_only and not args.out:
        raise SystemExit("pass --out to repair, or --check-only to just report")

    from olmo_core.data import TokenizerConfig
    from olmo_core.distributed.checkpoint import (
        load_model_and_optim_state,
        save_model_and_optim_state,
    )
    from olmo_core.nn.transformer import TransformerConfig

    tok_cfg = getattr(TokenizerConfig, args.tokenizer)()
    ids = reserved_ids(args.tokenizer)
    factory = getattr(TransformerConfig, args.model)
    model = factory(vocab_size=tok_cfg.padded_vocab_size()).build(init_device="cpu")
    print(f"built {args.model} vocab={tok_cfg.padded_vocab_size()} ; loading {args.base}")
    load_model_and_optim_state(args.base, model)
    print("loaded base into CPU model")

    emb = model.embeddings.weight.data
    report(emb, ids, "BEFORE")
    if args.check_only:
        raise SystemExit(0 if not problems(emb, ids) else 1)

    donors = resolve_donors(args.hf_tokenizer, ids)
    repair_markers(emb, ids, donors, seed=args.seed, jitter=args.jitter)
    report(emb, ids, "AFTER")
    remaining = problems(emb, ids)
    if remaining:
        raise SystemExit("repair did not converge: " + "; ".join(remaining))
    print("markers are distinguishable AND in-distribution in norm")

    out = os.path.join(args.out, "model_and_optim")
    save_model_and_optim_state(out, model, save_overwrite=True)
    print(f"wrote repaired base -> {out}")
    print("Re-audit the written copy with --check-only: a repair that only exists in this "
          "process's RAM is not a repair.")


if __name__ == "__main__":
    main()
