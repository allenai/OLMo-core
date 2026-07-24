"""
The data mix for the custom 50B "longmino-512k" corpus.

Expresses the target composition as :class:`MixingInstanceSource` ratios over the per-stratum
``part-*.npy`` trees written by ``tokenize_longmino_512k.py``. Because the proportions live here
rather than being baked into the tokenized files, re-weighting the mix costs nothing.

Target composition::

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

Leaving ``num_tokens`` unset yields the largest mix that matches these ratios exactly without
repeating any data (``max_repetition_factor`` stays at its default of 1.0, so an over-large
``num_tokens`` raises rather than silently upsampling). Given the tokens available in each stratum
that maximum is ~49.6B, bound by the 16k-32k bucket.

Inspect the realized mix::

    python src/scripts/data/longmino_512k_mix.py --tree qwen3
"""

import argparse
import os
import sys

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

#: Stratum label -> glob under the tokenized tree.
STRATUM_GLOBS = {
    "midtrain": "midtrain/*/part-*.npy",
    "lc_8_16k": "lc/real_s2pdf/2e13/part-*.npy",
    "lc_16_32k": "lc/real_s2pdf/2e14/part-*.npy",
    "lc_32_64k_real": "lc/real_s2pdf/2e15/part-*.npy",
    "lc_32_64k_rex": "lc/synth_rex/2e15/part-*.npy",
    "lc_32_64k_cwe": "lc/synth_cwe/2e15/part-*.npy",
    "lc_64_128k": "lc/real_s2pdf/2e16/part-*.npy",
    "lc_128_256k": "lc/real_s2pdf/2e17/part-*.npy",
    "lc_256_512k": "lc/real_s2pdf/2e18/part-*.npy",
}


def effective_ratios() -> dict:
    """
    The share of the *whole* mix each leaf stratum accounts for.

    Useful for sanity-checking the tree and for computing how large the mix can be before a
    stratum runs out. Values sum to 1.0.
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


#: Tokens available per stratum in the *source* datasets, from the dolma3 dataset cards (dolma2
#: tokenizer). Used only for the feasibility estimate below when a tokenized tree isn't built yet;
#: once ``token_counts.json`` exists we use the measured numbers instead.
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


def measured_available(root: str, tree: str) -> dict:
    """
    Read per-stratum token counts from a tokenized tree's ``token_counts.json``.

    :returns: Mapping of the labels in :func:`effective_ratios` to token counts, or ``{}`` if the
        tree has not been built yet.
    """
    import json

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


def feasibility(available: dict) -> dict:
    """
    Largest ratio-exact mix the given per-stratum token counts support, and what binds it.

    With ``max_repetition_factor`` at its default of 1.0 no stratum may be repeated, so the mix is
    capped by whichever stratum runs out first: ``min(available[s] / effective_ratio[s])``.
    """
    eff = effective_ratios()
    caps = {s: available[s] / eff[s] for s in eff if available.get(s)}
    if not caps:
        return {}
    binding = min(caps, key=lambda s: caps[s])
    total = caps[binding]
    return {
        "total_tokens": total,
        "binding_stratum": binding,
        "caps": caps,
        "per_stratum": {s: total * eff[s] for s in eff},
        "utilization": {s: (total * eff[s]) / available[s] for s in eff if available.get(s)},
    }


def build_longmino_512k_mix(
    *,
    tokenizer: TokenizerConfig,
    sequence_length: int,
    tree: str = "qwen3",
    root: str = WEKA_ROOT,
    seed: int = 1234,
    num_tokens: int = None,  # type: ignore[assignment]
) -> MixingInstanceSourceConfig:
    """
    Build the longmino-512k mix.

    :param tokenizer: Tokenizer config matching ``tree`` -- :meth:`TokenizerConfig.qwen3` for
        ``qwen3``, :meth:`TokenizerConfig.qwen3_5` for ``qwen35``.
    :param sequence_length: Training sequence length.
    :param tree: Which tokenized tree to read, ``qwen3`` or ``qwen35``.
    :param root: Root of the dataset on weka.
    :param seed: Sampling seed. Set explicitly -- passing ``seed=None`` to a sampling source makes
        it take a *prefix* of each source rather than a random subset.
    :param num_tokens: Optional target size. Leave unset for the largest ratio-exact mix.

    :returns: A :class:`MixingInstanceSourceConfig` ready to hand to a
        :class:`ComposableDataLoaderConfig`.
    """
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
            spec(chunked("lc_32_64k_rex"), REX_RATIO, "lc_32_64k_rex"),
            spec(chunked("lc_32_64k_cwe"), CWE_RATIO, "lc_32_64k_cwe"),
        ],
        seed=seed,
        label="lc_32_64k_synth",
    )

    bucket_32_64k = MixingInstanceSourceConfig(
        source_specs=[
            spec(chunked("lc_32_64k_real"), REAL_VS_SYNTH, "lc_32_64k_real"),
            spec(synth_32_64k, 1 - REAL_VS_SYNTH, "lc_32_64k_synth"),
        ],
        seed=seed,
        label="lc_32_64k",
    )

    long_context = MixingInstanceSourceConfig(
        source_specs=[
            spec(chunked("lc_8_16k"), LC_RATIOS["lc_8_16k"], "lc_8_16k"),
            spec(chunked("lc_16_32k"), LC_RATIOS["lc_16_32k"], "lc_16_32k"),
            spec(bucket_32_64k, LC_RATIOS["lc_32_64k"], "lc_32_64k"),
            spec(chunked("lc_64_128k"), LC_RATIOS["lc_64_128k"], "lc_64_128k"),
            spec(chunked("lc_128_256k"), LC_RATIOS["lc_128_256k"], "lc_128_256k"),
            spec(chunked("lc_256_512k"), LC_RATIOS["lc_256_512k"], "lc_256_512k"),
        ],
        seed=seed,
        label="long_context",
    )

    return MixingInstanceSourceConfig(
        source_specs=[
            spec(chunked("midtrain"), MIDTRAIN_RATIO, "midtrain"),
            spec(long_context, LONG_CONTEXT_RATIO, "long_context"),
        ],
        seed=seed,
        label="longmino_512k",
        num_tokens=num_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=WEKA_ROOT)
    parser.add_argument("--tree", default="qwen3", choices=["qwen3", "qwen35"])
    parser.add_argument("--sequence-length", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-tokens", type=int, default=None)
    parser.add_argument("--work-dir", default="/tmp/longmino512k-workdir")
    parser.add_argument(
        "--ratios-only", action="store_true", help="Print the ratio table without touching data."
    )
    args = parser.parse_args()

    eff = effective_ratios()
    available = measured_available(args.root, args.tree)
    source = "measured (token_counts.json)" if available else "published dataset-card counts"
    if not available:
        available = PUBLISHED_AVAILABLE
    fez = feasibility(available)

    print(f"stratum sizing -- availability from {source}\n")
    print(f"{'stratum':20s} {'share':>8s} {'available':>12s} {'in mix':>12s} {'used':>7s}")
    for label, ratio in eff.items():
        avail = available.get(label, 0)
        want = fez["per_stratum"][label]
        print(
            f"  {label:18s} {100 * ratio:7.3f}% {avail / 1e9:10.2f}B {want / 1e9:10.2f}B "
            f"{100 * want / avail if avail else 0:6.1f}%"
        )
    print(
        f"\n  total: {fez['total_tokens'] / 1e9:.2f}B tokens "
        f"(capped by '{fez['binding_stratum']}')"
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
        num_tokens=args.num_tokens,
    )
    print(f"\nbuilding mix from {os.path.join(args.root, args.tree)} ...\n")
    mix = cfg.build(args.work_dir)
    mix.visualize()


if __name__ == "__main__":
    main()
