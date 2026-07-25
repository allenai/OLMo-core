"""
The data mix for the custom 50B "longmino-512k" corpus.

Expresses the target composition as :class:`MixingInstanceSource` ratios over the per-stratum
``part-*.npy`` trees written by ``tokenize_longmino_512k.py``. Because the proportions live here
rather than being baked into the tokenized files, re-weighting the mix costs nothing.

Nominal composition::

    midtrain (short-context reasoning)                       66.1%
    long context                                             33.9%
        8k-16k    (2e13, real s2pdf)                          13%
        16k-32k   (2e14, real s2pdf)                          11%
        32k-64k                                               31%
            real  (2e15, real s2pdf)                              50%
            synth (2e15, REX + CWE)                               50%
        64k-128k  (2e16, real s2pdf, from pool)               15%
        128k-256k (2e17, real s2pdf, from pool)               15%
        256k-512k (2e18, real s2pdf, from pool)               15%

Those ratios alone cap the mix at whichever stratum runs out first -- the 16k-32k bucket -- which
leaves ``midtrain`` and ``8k-16k`` partially consumed. :data:`FULL_USE` names the strata that
should instead be taken in their entirety. Every other stratum keeps exactly the token count the
nominal ratios gave it; only the two topped-up strata change size, so the mix grows slightly and
their shares rise to match. See :func:`plan_tokens`.

Inspect the realized mix::

    python src/scripts/data/longmino_512k_mix.py --tree qwen3
"""

import argparse
import json
import os
import sys
from typing import Optional

from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ConcatAndChunkInstanceSourceConfig,
    InstanceSourceConfig,
    MixingInstanceSourceConfig,
    MixingInstanceSourceSpecConfig,
    NumpyDocumentSourceConfig,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from longmino_512k_common import WEKA_ROOT  # noqa: E402

#: Short-context reasoning share of the whole mix.
MIDTRAIN_RATIO = 0.661
LONG_CONTEXT_RATIO = 0.339

#: Shares *within* the long-context part.
LC_RATIOS = {
    "lc_8_16k": 0.13,
    "lc_16_32k": 0.11,
    "lc_32_64k": 0.31,
    "lc_64_128k": 0.15,
    "lc_128_256k": 0.15,
    "lc_256_512k": 0.15,
}

#: Within 32k-64k: half real PDFs, half synthetic.
REAL_VS_SYNTH = 0.50

#: Within the synthetic half, preserve the REX:CWE proportion the original longmino mix had
#: (6.08B : 1.94B tokens).
REX_RATIO = 0.758
CWE_RATIO = 0.242

#: Strata to consume in full rather than at their nominal share. Everything else keeps the token
#: count the nominal ratios assign it.
FULL_USE = ("midtrain", "lc_8_16k")

#: Stratum label -> glob under the tokenized tree.
STRATUM_GLOBS = {
    # ``**`` (not ``*``) for the family level. The pattern regex would match either way -- ``*``
    # becomes ``[^/]*``, which does match a family name -- but olmo_core.io.glob_directory only
    # passes ``recurse=True`` to list_directory when the pattern contains ``**``. Without it the
    # listing stops at the family directories and never yields the part files inside them.
    "midtrain": "midtrain/**/part-*.npy",
    "lc_8_16k": "lc/real_s2pdf/2e13/part-*.npy",
    "lc_16_32k": "lc/real_s2pdf/2e14/part-*.npy",
    "lc_32_64k_real": "lc/real_s2pdf/2e15/part-*.npy",
    "lc_32_64k_rex": "lc/synth_rex/2e15/part-*.npy",
    "lc_32_64k_cwe": "lc/synth_cwe/2e15/part-*.npy",
    "lc_64_128k": "lc/real_s2pdf/2e16/part-*.npy",
    "lc_128_256k": "lc/real_s2pdf/2e17/part-*.npy",
    "lc_256_512k": "lc/real_s2pdf/2e18/part-*.npy",
}

#: Tokens available per stratum in the *source* datasets, from the dolma3 dataset cards (dolma2
#: tokenizer). Used only when a tokenized tree isn't built yet; once ``token_counts.json`` exists
#: we use the measured numbers instead.
PUBLISHED_AVAILABLE = {
    "midtrain": 33.00e9,
    "lc_8_16k": 2.27e9,
    "lc_16_32k": 1.85e9,
    "lc_32_64k_real": 4.81e9,
    "lc_32_64k_rex": 6.08e9,
    "lc_32_64k_cwe": 1.94e9,
    "lc_64_128k": 3.35e9,
    "lc_128_256k": 3.35e9,
    "lc_256_512k": 3.35e9,
}


def nominal_ratios() -> dict:
    """
    The share of the *whole* mix each leaf stratum accounts for under the nominal ratios.

    Values sum to 1.0.
    """
    lc = LONG_CONTEXT_RATIO
    r32 = lc * LC_RATIOS["lc_32_64k"]
    return {
        "midtrain": MIDTRAIN_RATIO,
        "lc_8_16k": lc * LC_RATIOS["lc_8_16k"],
        "lc_16_32k": lc * LC_RATIOS["lc_16_32k"],
        "lc_32_64k_real": r32 * REAL_VS_SYNTH,
        "lc_32_64k_rex": r32 * (1 - REAL_VS_SYNTH) * REX_RATIO,
        "lc_32_64k_cwe": r32 * (1 - REAL_VS_SYNTH) * CWE_RATIO,
        "lc_64_128k": lc * LC_RATIOS["lc_64_128k"],
        "lc_128_256k": lc * LC_RATIOS["lc_128_256k"],
        "lc_256_512k": lc * LC_RATIOS["lc_256_512k"],
    }


def measured_available(root: str, tree: str) -> dict:
    """
    Read per-stratum token counts from a tokenized tree's ``token_counts.json``.

    :returns: Mapping of stratum label to token count, or ``{}`` if the tree isn't built yet.
    """
    path = os.path.join(root, tree, "token_counts.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        strata = json.load(f)["strata"]

    def total(prefix: str) -> int:
        return sum(v["tokens"] for k, v in strata.items() if k.startswith(prefix))

    return {
        "midtrain": total("midtrain/"),
        "lc_8_16k": total("lc/real_s2pdf/2e13"),
        "lc_16_32k": total("lc/real_s2pdf/2e14"),
        "lc_32_64k_real": total("lc/real_s2pdf/2e15"),
        "lc_32_64k_rex": total("lc/synth_rex/2e15"),
        "lc_32_64k_cwe": total("lc/synth_cwe/2e15"),
        "lc_64_128k": total("lc/real_s2pdf/2e16"),
        "lc_128_256k": total("lc/real_s2pdf/2e17"),
        "lc_256_512k": total("lc/real_s2pdf/2e18"),
    }


def plan_tokens(available: dict) -> dict:
    """
    Target tokens per stratum.

    First size the mix under the nominal ratios: with no repetition allowed, that is capped by
    ``min(available[s] / nominal_ratio[s])``, i.e. by whichever stratum runs out first. Then the
    strata in :data:`FULL_USE` are raised to their full availability. Every other stratum keeps the
    count the nominal ratios gave it, so the mix simply grows by the topped-up amount.

    :param available: Tokens on disk per stratum.

    :returns: Mapping of stratum label to target token count.
    """
    nominal = nominal_ratios()
    caps = {s: available[s] / nominal[s] for s in nominal if available.get(s)}
    base_total = min(caps.values())
    target = {s: base_total * nominal[s] for s in nominal}
    for s in FULL_USE:
        if available.get(s):
            target[s] = float(available[s])
    return target


def build_longmino_512k_mix(
    *,
    tokenizer: TokenizerConfig,
    sequence_length: int,
    tree: str = "qwen3",
    root: str = WEKA_ROOT,
    seed: int = 1234,
    available: dict = None,  # type: ignore[assignment]
    num_tokens: Optional[int] = None,
) -> MixingInstanceSourceConfig:
    """
    Build the longmino-512k mix.

    Ratios are passed as raw target token counts at each level of the tree;
    :class:`MixingInstanceSource` normalizes ratios within each node, so this reproduces the
    intended proportions exactly without having to hand-convert them to nested fractions.

    :param tokenizer: Tokenizer config matching ``tree`` -- :meth:`TokenizerConfig.qwen3` for
        ``qwen3``, :meth:`TokenizerConfig.qwen3_5` for ``qwen35``.
    :param sequence_length: Training sequence length.
    :param tree: Which tokenized tree to read, ``qwen3`` or ``qwen35``.
    :param root: Root of the dataset on weka.
    :param seed: Sampling seed. Set explicitly -- passing ``seed=None`` to a sampling source makes
        it take a *prefix* of each source rather than a random subset.
    :param available: Per-stratum token counts. Defaults to the tree's ``token_counts.json``,
        falling back to :data:`PUBLISHED_AVAILABLE`.

    :returns: A :class:`MixingInstanceSourceConfig` ready to hand to a
        :class:`ComposableDataLoaderConfig`.
    """
    if available is None:
        available = measured_available(root, tree) or PUBLISHED_AVAILABLE
    target = plan_tokens(available)

    base = os.path.join(root, tree)
    sources = NumpyDocumentSourceConfig.from_source_groups(
        {label: [os.path.join(base, glob)] for label, glob in STRATUM_GLOBS.items()},
        tokenizer=tokenizer,
        expand_glob=True,
        max_document_length=None,  # never truncate long documents
        source_permutation_seed=seed,
    )

    def chunked(label: str) -> InstanceSourceConfig:
        return ConcatAndChunkInstanceSourceConfig(
            sources=[sources[label]], sequence_length=sequence_length, label=label
        )

    def spec(source, ratio, label):
        return MixingInstanceSourceSpecConfig(source=source, ratio=ratio, label=label)

    synth_32_64k = MixingInstanceSourceConfig(
        source_specs=[
            spec(chunked("lc_32_64k_rex"), target["lc_32_64k_rex"], "lc_32_64k_rex"),
            spec(chunked("lc_32_64k_cwe"), target["lc_32_64k_cwe"], "lc_32_64k_cwe"),
        ],
        seed=seed,
        label="lc_32_64k_synth",
    )
    synth_total = target["lc_32_64k_rex"] + target["lc_32_64k_cwe"]

    bucket_32_64k = MixingInstanceSourceConfig(
        source_specs=[
            spec(chunked("lc_32_64k_real"), target["lc_32_64k_real"], "lc_32_64k_real"),
            spec(synth_32_64k, synth_total, "lc_32_64k_synth"),
        ],
        seed=seed,
        label="lc_32_64k",
    )
    bucket_32_64k_total = target["lc_32_64k_real"] + synth_total

    long_context = MixingInstanceSourceConfig(
        source_specs=[
            spec(chunked("lc_8_16k"), target["lc_8_16k"], "lc_8_16k"),
            spec(chunked("lc_16_32k"), target["lc_16_32k"], "lc_16_32k"),
            spec(bucket_32_64k, bucket_32_64k_total, "lc_32_64k"),
            spec(chunked("lc_64_128k"), target["lc_64_128k"], "lc_64_128k"),
            spec(chunked("lc_128_256k"), target["lc_128_256k"], "lc_128_256k"),
            spec(chunked("lc_256_512k"), target["lc_256_512k"], "lc_256_512k"),
        ],
        seed=seed,
        label="long_context",
    )
    lc_total = sum(v for k, v in target.items() if k != "midtrain")

    return MixingInstanceSourceConfig(
        source_specs=[
            spec(chunked("midtrain"), target["midtrain"], "midtrain"),
            spec(long_context, lc_total, "long_context"),
        ],
        seed=seed,
        label="longmino_512k",
        # Deliberately unset by default. Passing the exact arithmetic sum of the per-stratum targets
        # is unsatisfiable: MixingInstanceSource allocates whole *instances*, so each source floors
        # to a multiple of sequence_length and comes up a few million tokens short of the real-valued
        # target, which raises OLMoConfigurationError. With num_tokens=None the mixer instead takes
        # the largest size that matches the ratios without repeating data -- the same mix, rounded
        # down to whole instances. Run length is set by the trainer's max_duration anyway.
        num_tokens=num_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=WEKA_ROOT)
    parser.add_argument("--tree", default="qwen3", choices=["qwen3", "qwen35"])
    parser.add_argument("--sequence-length", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--work-dir", default="/tmp/longmino512k-workdir")
    parser.add_argument(
        "--ratios-only", action="store_true", help="Print the ratio table without touching data."
    )
    args = parser.parse_args()

    measured = measured_available(args.root, args.tree)
    source = "measured (token_counts.json)" if measured else "published dataset-card counts"
    available = measured or PUBLISHED_AVAILABLE
    target = plan_tokens(available)
    total = sum(target.values())
    lc_total = sum(v for k, v in target.items() if k != "midtrain")

    print(f"stratum sizing -- availability from {source}\n")
    print(f"{'stratum':20s} {'available':>11s} {'in mix':>11s} {'used':>7s} {'share':>8s}")
    for label in target:
        avail = available.get(label, 0)
        want = target[label]
        flag = "  <- full" if label in FULL_USE else ""
        print(
            f"  {label:18s} {avail / 1e9:9.2f}B {want / 1e9:9.2f}B "
            f"{100 * want / avail if avail else 0:6.1f}% {100 * want / total:7.3f}%{flag}"
        )
    print(f"\n  total: {total / 1e9:.2f}B tokens")
    print(
        f"  midtrain {100 * target['midtrain'] / total:.2f}%  /  "
        f"long context {100 * lc_total / total:.2f}%"
    )
    print(
        "  within long context: "
        + ", ".join(
            f"{k.replace('lc_', '')} {100 * v / lc_total:.2f}%"
            for k, v in target.items()
            if k != "midtrain" and not k.startswith("lc_32_64k_")
        )
        + f", 32_64k {100 * sum(v for k, v in target.items() if k.startswith('lc_32_64k_')) / lc_total:.2f}%"
    )
    if args.ratios_only:
        return

    tokenizer = TokenizerConfig.qwen3() if args.tree == "qwen3" else TokenizerConfig.qwen3_5()
    cfg = build_longmino_512k_mix(
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        tree=args.tree,
        root=args.root,
        seed=args.seed,
        available=available,
    )
    print(f"\nbuilding mix from {os.path.join(args.root, args.tree)} ...\n")
    mix = cfg.build(args.work_dir)
    mix.visualize()


if __name__ == "__main__":
    main()
