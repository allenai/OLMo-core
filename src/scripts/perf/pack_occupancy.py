"""Measure how much of a Stage-2 pack is real work, and sweep the crop budget on CPU.

Reported throughput for Molmo2 Stage-2 counts padding. ``MultimodalCollator`` pads every
pack to a fixed ``pad_sequence_length`` (16,384), and
``MixtureDataLoader.global_num_tokens_in_batch`` returns the padded figure regardless of
how much of the sequence held real tokens. So "tokens per second" rises when a pack is
emptier, and the metric cannot distinguish going faster from doing less.

This script measures the thing TPS hides, without a GPU:

``sample``
    Draw examples from a real mixture in the same order training would, and record each
    one's ``(n_tokens, n_crops)``. This is the only expensive step -- it decodes and
    preprocesses images -- so it is done once and cached to JSONL.

``sweep``
    Replay the recorded lengths through :class:`DynamicPacker` at a range of crop budgets
    and objective weights. The packer only ever reads ``len(example[key])`` and delegates
    emission to ``pack_fn``, so a replay needs no pixels and no tokenizer: the whole sweep
    is arithmetic over the cached lengths and runs in seconds.

Motivating arithmetic: Stage 2 uses ``MAX_CROPS = 8``, so one image costs up to 9 crops
and contributes roughly 1,190 LM tokens. The single-image pack profile caps a pack at 25
crops, which admits about two images -- a few thousand real tokens in a 16,384 slot. The
crop budget, not the sequence length, is what closes those packs.

Usage::

    python src/scripts/perf/pack_occupancy.py sample --mixture=single-image-only-v10 \
        --limit=2000 --out=/tmp/occ-v10.jsonl
    python src/scripts/perf/pack_occupancy.py sweep --lengths=/tmp/occ-v10.jsonl \
        --max-crops=25,40,64,100,125
"""

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger("pack_occupancy")

# Mirrors Molmo2-Stage2.py; kept local so this script has no import-time dependency on the
# training entrypoint (which pulls in torch.distributed).
SEQUENCE_LENGTH = 16384
BUFFER_SIZE = 48
TEXT_WEIGHT = 1.0
IMAGE_WEIGHT = 30.0


@dataclass
class PackStat:
    """One emitted pack, reduced to the quantities that determine wasted compute."""

    n_examples: int
    n_tokens: int
    n_crops: int
    exit_reason: str

    @property
    def token_occupancy(self) -> float:
        return self.n_tokens / SEQUENCE_LENGTH


def _build_mixture(mixture: str, seed: int, only: Optional[Sequence[str]] = None):
    """Build one tier's ``(datasets, weights, names)``, matching Molmo2-Stage2's dispatch.

    Tier -> registry routing lives in ``mixtures.tiers``; going through it here (rather
    than picking a builder by hand) keeps this script honest if a tier is re-pointed.
    """
    from transformers import AutoTokenizer

    from olmo_core.data.multimodal.mixtures.image_only_v9 import (
        build_image_only_v9_mixture,
        build_single_image_only_v9_mixture,
    )
    from olmo_core.data.multimodal.mixtures.image_only_v10 import (
        build_image_only_v10_mixture,
        build_single_image_only_v10_mixture,
    )
    from olmo_core.data.multimodal.mixtures.tiers import (
        all_validation_mixtures,
        is_v10_mixture,
    )

    known = all_validation_mixtures()
    if mixture not in known:
        raise ValueError(f"Unknown mixture {mixture!r}; use one of: {', '.join(sorted(known))}")
    names_filter = known[mixture]
    if only:
        # Scoping escape hatch. ``build_mixture`` takes len() of every source it is asked
        # for, and on a cold FineVision index cache that is minutes of weka I/O per Arrow
        # shard -- a full 43-source tier can take hours the first time. Restricting to a
        # few sources makes an exploratory sweep interactive, at the cost of no longer
        # reflecting the tier's real source distribution. Prefer the full tier for any
        # number you intend to act on.
        missing = [n for n in only if names_filter is not None and n not in names_filter]
        if missing:
            log.warning("--only names not in %s: %s", mixture, ", ".join(missing))
        names_filter = list(only)

    tokenizer = AutoTokenizer.from_pretrained("allenai/Molmo2-4B", trust_remote_code=True)
    single_image_only = mixture in ("single-image-only-v9", "single-image-only-v10")
    if is_v10_mixture(mixture):
        build = (
            build_single_image_only_v10_mixture
            if single_image_only
            else build_image_only_v10_mixture
        )
    else:
        build = (
            build_single_image_only_v9_mixture if single_image_only else build_image_only_v9_mixture
        )

    return build(
        tokenizer,
        seed=seed,
        dataset_names=names_filter,
        max_sequence_length=SEQUENCE_LENGTH,
    )


def _iter_sampled_lengths(
    mixture: str, limit: int, seed: int, only: Optional[Sequence[str]] = None
):
    """Yield ``(n_tokens, n_crops, source)`` for `limit` examples of `mixture`.

    Draws in the mixture's own weighted order via ``iter_rank_mixture_refs`` -- the same
    stream ``PackedMixtureIterableDataset`` consumes -- so the sampled source distribution
    matches training rather than being uniform over sources.
    """
    from olmo_core.data.multimodal.packed_mixture_iterable import iter_rank_mixture_refs

    datasets, weights, names = _build_mixture(mixture, seed, only)
    sizes = [len(d) for d in datasets]
    log.info("mixture %s: %d sources, %d total rows", mixture, len(datasets), sum(sizes))

    refs = iter_rank_mixture_refs(
        seed=seed,
        epoch=0,
        weights=weights,
        sizes=sizes,
        dp_rank=0,
        dp_world_size=1,
        epoch_instances=max(limit * 4, 10_000),
    )

    n = 0
    failures = 0
    for ds_idx, ex_idx in refs:
        if n >= limit:
            break
        try:
            ex = datasets[ds_idx][ex_idx]
        except Exception as e:  # a single unreadable row must not end the sample
            failures += 1
            if failures <= 5:
                log.warning("skipping %s[%d]: %s", names[ds_idx], ex_idx, e)
            continue
        images = ex.get("images")
        yield (
            int(len(ex["input_ids"])),
            int(0 if images is None else len(images)),
            names[ds_idx],
        )
        n += 1
        if n % 200 == 0:
            log.info("sampled %d/%d", n, limit)
    if failures:
        log.warning("skipped %d unreadable rows", failures)


def _replay(
    lengths: Sequence[Dict],
    max_crops: int,
    *,
    image_weight: float = IMAGE_WEIGHT,
    text_weight: float = TEXT_WEIGHT,
    buffer_size: int = BUFFER_SIZE,
    shortcut_max_len_images: bool = True,
) -> List[PackStat]:
    """Replay cached ``(n_tokens, n_crops)`` through the real packer at one crop budget.

    Uses ``DynamicPacker`` itself rather than a reimplementation, so the shortcut and
    over-capacity exits, the token quantization and the knapsack all behave exactly as
    they do in a worker process. Stand-in zero arrays are enough because the packer reads
    only ``len()``; a custom ``pack_fn`` records each pack instead of concatenating it.
    """
    from olmo_core.data.multimodal.packing import DynamicPacker, PackingConstraint

    packed: List[PackStat] = []
    # Set by the wrapping __call__ below so pack_fn can attribute the exit reason.
    pending_reason = {"why": "knapsack"}

    def record(examples):
        packed.append(
            PackStat(
                n_examples=len(examples),
                n_tokens=sum(len(e["input_ids"]) for e in examples),
                n_crops=sum(len(e["images"]) for e in examples),
                exit_reason=pending_reason["why"],
            )
        )
        return {}  # the caller only counts packs; the payload is never read

    constraints = [
        PackingConstraint(
            "input_ids", SEQUENCE_LENGTH, True, text_weight, max(1, SEQUENCE_LENGTH // 512)
        ),
        PackingConstraint("images", max_crops, shortcut_max_len_images, image_weight, 1),
    ]
    packer = DynamicPacker(buffer_size, constraints, pack_fn=record)

    for row in lengths:
        n_tok, n_crop = row["n_tokens"], row["n_crops"]
        example = {
            "input_ids": np.zeros(n_tok, dtype=np.int64),
            "images": np.zeros((n_crop, 1), dtype=np.float32),
        }
        # Classify the exit the packer is about to take, using its own predicates.
        why = "knapsack"
        for c in constraints:
            n = n_crop if c.key == "images" else n_tok
            if n > c.max_len:
                why = f"over_capacity:{c.key}"
                break
            if c.allow_shortcut and c.get_quantized_value(n) >= c.get_quantized_max_len():
                why = f"shortcut:{c.key}"
                break
        pending_reason["why"] = why
        packer(example)

    # flush() emits the buffer tail through the same pack_fn, which already appends;
    # draining it here would double-count, so just exhaust the iterator.
    pending_reason["why"] = "flush"
    for _ in packer.flush():
        pass
    return packed


def _summarize(stats: List[PackStat], max_crops: int) -> Dict:
    if not stats:
        return {"max_crops": max_crops, "packs": 0}
    # Clamp to the sequence length: an example longer than SEQUENCE_LENGTH is emitted as
    # its own `over_capacity:input_ids` pack and the collator tail-truncates it
    # (collator.py:66-71), so counting its full length would report >100% occupancy for
    # that pack. Rare (1 row in the 3,000-example v10 sample) but it inflates the mean.
    toks = [min(s.n_tokens, SEQUENCE_LENGTH) for s in stats]
    crops = [s.n_crops for s in stats]
    exs = [s.n_examples for s in stats]
    return {
        "max_crops": max_crops,
        "packs": len(stats),
        "token_occupancy_mean": sum(toks) / (len(stats) * SEQUENCE_LENGTH),
        "token_occupancy_median": statistics.median(toks) / SEQUENCE_LENGTH,
        "crop_occupancy_mean": sum(crops) / (len(stats) * max_crops),
        "examples_per_pack_mean": sum(exs) / len(stats),
        "examples_per_pack_median": statistics.median(exs),
        "total_examples": sum(exs),
        "exit_reasons": dict(Counter(s.exit_reason for s in stats).most_common()),
    }


def _cmd_sample(args) -> int:
    only = args.only.split(",") if args.only else None
    # Stream to disk as rows arrive and flush each one. Building the mixture alone can
    # take tens of minutes (it takes len() of every source), and a crash late in the run
    # -- e.g. one source with an unset data-root env var -- would otherwise discard hours
    # of decoding. A partial file is still a usable input to `sweep`.
    rows = []
    with open(args.out, "w") as f:
        for n_tokens, n_crops, source in _iter_sampled_lengths(
            args.mixture, args.limit, args.seed, only
        ):
            f.write(json.dumps({"n_tokens": n_tokens, "n_crops": n_crops, "source": source}) + "\n")
            f.flush()
            rows.append((n_tokens, n_crops, source))
    log.info("wrote %d rows to %s", len(rows), args.out)
    if rows:
        toks = [r[0] for r in rows]
        crops = [r[1] for r in rows]
        log.info(
            "per-example: tokens median=%d mean=%.0f max=%d | crops median=%d mean=%.1f max=%d",
            statistics.median(toks),
            sum(toks) / len(toks),
            max(toks),
            statistics.median(crops),
            sum(crops) / len(crops),
            max(crops),
        )
    return 0


def _cmd_sweep(args) -> int:
    with open(args.lengths) as f:
        lengths = [json.loads(line) for line in f if line.strip()]
    log.info("replaying %d cached examples", len(lengths))

    budgets = [int(x) for x in args.max_crops.split(",")]
    results = []
    for mc in budgets:
        stats = _replay(
            lengths,
            mc,
            image_weight=args.image_weight,
            shortcut_max_len_images=not args.no_shortcut,
        )
        results.append(_summarize(stats, mc))

    hdr = f"{'crops':>6} {'packs':>7} {'tok_occ':>8} {'crop_occ':>9} {'ex/pack':>8} {'examples':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        if not r.get("packs"):
            continue
        print(
            f"{r['max_crops']:>6} {r['packs']:>7} {r['token_occupancy_mean']:>7.1%} "
            f"{r['crop_occupancy_mean']:>8.1%} {r['examples_per_pack_mean']:>8.2f} "
            f"{r['total_examples']:>9}"
        )
    print("\nexit reasons (why each pack closed):")
    for r in results:
        if r.get("packs"):
            print(f"  max_crops={r['max_crops']:>4}: {r['exit_reasons']}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        log.info("wrote %s", args.json_out)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="draw real examples and cache their token/crop lengths")
    s.add_argument("--mixture", default="single-image-only-v10")
    s.add_argument("--limit", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.add_argument(
        "--only",
        help="comma-separated source names to restrict to; much faster on a cold "
        "FineVision index cache, but no longer the tier's real source distribution",
    )
    s.set_defaults(func=_cmd_sample)

    w = sub.add_parser("sweep", help="replay cached lengths at several crop budgets")
    w.add_argument("--lengths", required=True)
    w.add_argument("--max-crops", default="25,40,64,100,125")
    w.add_argument("--image-weight", type=float, default=IMAGE_WEIGHT)
    w.add_argument("--no-shortcut", action="store_true", help="disable shortcut_max_len_images")
    w.add_argument("--json-out")
    w.set_defaults(func=_cmd_sweep)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
