# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Logits Comparison: vLLM vs OLMo-core

This script compares vLLM model outputs against OLMo-core reference logits.
Since vLLM's standard API exposes logprobs (top-k) rather than full logits,
this script focuses on per-token top-1/top-k analysis.

Usage:
    # First generate reference logits from OLMo-core
    python compare_logits_olmocore_multi.py \
        --checkpoint /path/to/checkpoint \
        --output logits_olmocore_multi.pt

    # Then run vLLM comparison (top-k mode - faster, uses standard API)
    python compare_logits_vllm.py \
        --model-path /path/to/hf-model \
        --reference logits_olmocore_multi.pt \
        --device cuda

"""

import argparse
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Test Sequences - Must match OLMo-core script exactly!
# =============================================================================

TEST_SEQUENCES = [
    # Original sequences
    ("original_10tok", [1, 2, 3, 4, 5, 100, 200, 500, 1000, 2000]),
    ("single_token", [12345]),
    ("short_5tok", [100, 200, 300, 400, 500]),
    ("medium_20tok", list(range(100, 120))),
    ("longer_50tok", list(range(1000, 1050))),
    ("longer_100tok", list(range(500, 600))),
    ("random_pattern", [42, 1337, 9999, 7777, 2468, 1357, 8642, 3141, 5926, 5358]),
    (
        "high_vocab",
        [50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000],
    ),
    # Additional sequences for robust testing
    ("single_1", [1]),
    ("single_100", [100]),
    ("single_1000", [1000]),
    ("single_10000", [10000]),
    ("single_50000", [50000]),
    ("pair_low", [1, 2]),
    ("pair_mid", [5000, 5001]),
    ("pair_high", [80000, 80001]),
    ("pair_mixed", [100, 90000]),
    ("triple_asc", [100, 200, 300]),
    ("triple_desc", [300, 200, 100]),
    ("triple_same", [500, 500, 500]),
    ("triple_spread", [1, 50000, 99999]),
    ("five_linear", [10, 20, 30, 40, 50]),
    ("five_exp", [1, 10, 100, 1000, 10000]),
    ("five_primes", [2, 3, 5, 7, 11]),
    ("five_fib", [1, 1, 2, 3, 5]),
    ("five_powers2", [2, 4, 8, 16, 32]),
    ("ten_linear_1", list(range(1, 11))),
    ("ten_linear_100", list(range(100, 110))),
    ("ten_linear_1000", list(range(1000, 1010))),
    ("ten_linear_10000", list(range(10000, 10010))),
    ("ten_evens", list(range(2, 22, 2))),
    ("ten_odds", list(range(1, 21, 2))),
    ("ten_squares", [i**2 for i in range(1, 11)]),
    ("ten_cubes", [i**3 for i in range(1, 11)]),
    ("ten_random_a", [3847, 9182, 4756, 2938, 8471, 1029, 5738, 4829, 7364, 2918]),
    (
        "ten_random_b",
        [12847, 38291, 47562, 82934, 19283, 57382, 29384, 83921, 47283, 92831],
    ),
    (
        "ten_random_c",
        [61234, 72345, 83456, 94567, 15678, 26789, 37890, 48901, 59012, 60123],
    ),
    ("ten_alternating", [100, 90000, 200, 80000, 300, 70000, 400, 60000, 500, 50000]),
    ("fifteen_linear", list(range(500, 515))),
    ("fifteen_spread", list(range(0, 75000, 5000))),
    (
        "fifteen_random",
        [
            2847,
            19283,
            38472,
            57261,
            8374,
            94827,
            12938,
            47382,
            83927,
            29384,
            58273,
            17384,
            92837,
            48273,
            73829,
        ],
    ),
    ("twenty_low", list(range(1, 21))),
    ("twenty_mid", list(range(40000, 40020))),
    ("twenty_high", list(range(90000, 90020))),
    ("twenty_spread", list(range(0, 100000, 5000))),
    ("twenty_random_a", [i * 4937 % 100000 for i in range(20)]),
    ("twenty_random_b", [i * 7919 % 100000 for i in range(20)]),
    ("twentyfive_linear", list(range(2000, 2025))),
    ("twentyfive_random", [i * 6151 % 100000 for i in range(25)]),
    ("thirty_linear", list(range(3000, 3030))),
    ("thirty_spread", list(range(0, 90000, 3000))),
    ("thirty_random", [i * 8123 % 100000 for i in range(30)]),
    ("forty_linear", list(range(4000, 4040))),
    ("forty_random", [i * 9311 % 100000 for i in range(40)]),
    ("fifty_linear_a", list(range(5000, 5050))),
    ("fifty_linear_b", list(range(50000, 50050))),
    ("fifty_random_a", [i * 3571 % 100000 for i in range(50)]),
    ("fifty_random_b", [i * 7333 % 100000 for i in range(50)]),
    ("sixtyfour_linear", list(range(6400, 6464))),
    ("sixtyfour_random", [i * 4519 % 100000 for i in range(64)]),
    ("seventyfive_linear", list(range(7500, 7575))),
    ("seventyfive_random", [i * 5347 % 100000 for i in range(75)]),
    ("hundred_linear_a", list(range(100, 200))),
    ("hundred_linear_b", list(range(10000, 10100))),
    ("hundred_linear_c", list(range(80000, 80100))),
    ("hundred_random_a", [i * 2671 % 100000 for i in range(100)]),
    ("hundred_random_b", [i * 8887 % 100000 for i in range(100)]),
    ("onetwentyeight_linear", list(range(12800, 12928))),
    ("onetwentyeight_random", [i * 6947 % 100000 for i in range(128)]),
    ("onefifty_linear", list(range(15000, 15150))),
    ("onefifty_random", [i * 4201 % 100000 for i in range(150)]),
    ("twohundred_linear", list(range(20000, 20200))),
    ("twohundred_random", [i * 7687 % 100000 for i in range(200)]),
    ("twofiftysix_linear", list(range(25600, 25856))),
    ("twofiftysix_random", [i * 3389 % 100000 for i in range(256)]),
    ("repeat_10x10", [100] * 10),
    ("repeat_50x5", [5000] * 50),
    ("repeat_100x3", [30000] * 100),
    ("zigzag_20", [100 if i % 2 == 0 else 90000 for i in range(20)]),
    ("zigzag_50", [1000 if i % 2 == 0 else 80000 for i in range(50)]),
    ("sawtooth_30", [(i % 10) * 1000 for i in range(30)]),
    ("sawtooth_60", [(i % 20) * 500 for i in range(60)]),
    ("near_zero", list(range(0, 10))),
    ("low_range", list(range(50, 100))),
    ("mid_range", list(range(49950, 50050))),
    ("high_range", list(range(99900, 100000))),
    ("mixed_ranges_20", [i * 5000 for i in range(20)]),
    ("mixed_ranges_50", [i * 2000 for i in range(50)]),
    (
        "scattered_25",
        [
            1,
            1000,
            5000,
            10000,
            20000,
            30000,
            40000,
            50000,
            60000,
            70000,
            80000,
            90000,
            95000,
            99000,
            99500,
            99900,
            99950,
            99990,
            99995,
            99999,
            500,
            5500,
            15000,
            25000,
            35000,
        ],
    ),
    ("stress_512", [i * 193 % 100000 for i in range(512)]),
    ("stress_768", [i * 277 % 100000 for i in range(768)]),
    ("stress_1024", [i * 331 % 100000 for i in range(1024)]),
    ("repeat_256_pattern", [1000, 2000, 3000, 4000] * 64),
]


class VLLMLogitsExtractor:
    """Extract logits/predictions from vLLM model."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.8,
        enforce_eager: bool = True,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.llm = None
        self.model = None
        self.vocab_size = None

    def load_for_topk(self):
        """Load model for top-k extraction using vLLM's standard API."""
        from vllm import LLM

        log.info(f"Loading vLLM model from {self.model_path}...")
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            enforce_eager=self.enforce_eager,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype=self.dtype,
        )
        self.vocab_size = self.llm.llm_engine.model_config.hf_config.vocab_size
        log.info(f"Model loaded. Vocab size: {self.vocab_size}")


    def get_topk_predictions(
        self,
        input_ids: list[int],
        k: int = 100,
    ) -> dict:
        """Get top-k predictions for each position using vLLM's logprobs API."""
        from vllm import SamplingParams

        if self.llm is None:
            self.load_for_topk()

        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=k,
            prompt_logprobs=k,
        )

        outputs = self.llm.generate(
            prompts=[
                {"prompt_token_ids": input_ids}
            ],
            sampling_params=sampling_params,
        )

        output = outputs[0]
        prompt_logprobs = output.prompt_logprobs
        output_logprobs = output.outputs[0].logprobs

        generated_token = (
            output.outputs[0].token_ids[0] if output.outputs[0].token_ids else None
        )

        return {
            "prompt_logprobs": prompt_logprobs,
            "output_logprobs": output_logprobs,
            "generated_token": generated_token,
            "input_ids": input_ids,
        }


def extract_topk_from_logprobs(
    logprobs_dict: dict | None,
    k: int = 20,
) -> tuple[list[int], list[float]]:
    """Extract top-k token IDs and their log probabilities from vLLM logprobs dict."""
    if logprobs_dict is None:
        return [], []

    sorted_items = sorted(
        logprobs_dict.items(),
        key=lambda x: x[1].logprob if hasattr(x[1], "logprob") else x[1],
        reverse=True,
    )[:k]

    token_ids = []
    log_probs = []
    for token_id, logprob_obj in sorted_items:
        token_ids.append(token_id)
        if hasattr(logprob_obj, "logprob"):
            log_probs.append(logprob_obj.logprob)
        else:
            log_probs.append(float(logprob_obj))

    return token_ids, log_probs


def analyze_topk_agreement(
    ref_logits: torch.Tensor,
    vllm_topk_tokens: list[int],
    vllm_topk_logprobs: list[float],
    position: int,
    k_values: list[int],
) -> dict:
    """Analyze top-k agreement between reference logits and vLLM top-k predictions."""

    ref_pos_logits = ref_logits[position]

    results = {}

    for k in k_values:
        if k > len(vllm_topk_tokens):
            continue

        ref_topk_vals, ref_topk_ids = torch.topk(ref_pos_logits, k)
        ref_topk_set = set(ref_topk_ids.tolist())

        vllm_topk_set = set(vllm_topk_tokens[:k])

        overlap = len(ref_topk_set & vllm_topk_set)
        agreement = overlap / k * 100

        results[f"top{k}_agreement"] = agreement
        results[f"top{k}_overlap"] = overlap

    if vllm_topk_tokens:
        vllm_top1 = vllm_topk_tokens[0]
        ref_top1 = ref_pos_logits.argmax().item()

        results["ref_top1"] = ref_top1
        results["vllm_top1"] = vllm_top1
        results["top1_match"] = vllm_top1 == ref_top1

        if ref_top1 in vllm_topk_tokens:
            results["ref_top1_rank_in_vllm"] = vllm_topk_tokens.index(ref_top1)
        else:
            results["ref_top1_rank_in_vllm"] = -1

        ref_ranks = ref_pos_logits.argsort(descending=True)
        vllm_top1_rank_in_ref = (
            (ref_ranks == vllm_top1).nonzero(as_tuple=True)[0].item()
        )
        results["vllm_top1_rank_in_ref"] = vllm_top1_rank_in_ref

        ref_probs = F.softmax(ref_pos_logits, dim=-1)
        results["ref_top1_prob"] = ref_probs[ref_top1].item()
        results["vllm_top1_prob"] = (
            np.exp(vllm_topk_logprobs[0]) if vllm_topk_logprobs else 0.0
        )

        ref_log_probs = F.log_softmax(ref_pos_logits, dim=-1)
        results["ref_top1_logprob"] = ref_log_probs[ref_top1].item()
        results["vllm_top1_logprob"] = (
            vllm_topk_logprobs[0] if vllm_topk_logprobs else float("-inf")
        )

    return results


def analyze_sequence_topk(
    name: str,
    ref_logits: torch.Tensor,
    vllm_result: dict,
    k_values: list[int],
) -> dict:
    """Analyze a full sequence using top-k comparison."""

    seq_len = ref_logits.shape[0]
    prompt_logprobs = vllm_result["prompt_logprobs"]
    output_logprobs = vllm_result["output_logprobs"]

    per_position = []

    # indexing:
    # - ref_logits[i] = logits for predicting token at position i+1 given 0..i
    # - prompt_logprobs[i] = logprobs for token at position i (predicting position i given 0..i-1)
    # prompt_logprobs[i] corresponds to ref_logits[i-1]

    for pos in range(seq_len):
        # For position pos in ref_logits (predicting pos+1 given 0..pos)
        # We need prompt_logprobs[pos+1] if pos+1 < seq_len
        # Or output_logprobs[0] if pos == seq_len-1

        if pos < seq_len - 1:
            # Use prompt_logprobs for position pos+1
            logprobs_idx = pos + 1
            if (
                logprobs_idx < len(prompt_logprobs)
                and prompt_logprobs[logprobs_idx] is not None
            ):
                vllm_tokens, vllm_logprobs = extract_topk_from_logprobs(
                    prompt_logprobs[logprobs_idx]
                )
            else:
                vllm_tokens, vllm_logprobs = [], []
        else:
            # Last position - use output_logprobs
            if output_logprobs and len(output_logprobs) > 0:
                vllm_tokens, vllm_logprobs = extract_topk_from_logprobs(
                    output_logprobs[0]
                )
            else:
                vllm_tokens, vllm_logprobs = [], []

        pos_result = analyze_topk_agreement(
            ref_logits, vllm_tokens, vllm_logprobs, pos, k_values
        )
        pos_result["position"] = pos
        per_position.append(pos_result)

    top1_matches = sum(1 for p in per_position if p.get("top1_match", False))
    top1_rate = top1_matches / seq_len * 100 if seq_len > 0 else 0

    topk_avgs = {}
    for k in k_values:
        key = f"top{k}_agreement"
        vals = [p.get(key, 0) for p in per_position if key in p]
        if vals:
            topk_avgs[key] = np.mean(vals)

    mismatches = [p for p in per_position if not p.get("top1_match", True)]
    mismatch_ranks = [p.get("ref_top1_rank_in_vllm", -1) for p in mismatches]

    logprob_diffs = []
    for p in per_position:
        if "ref_top1_logprob" in p and "vllm_top1_logprob" in p:
            diff = abs(p["ref_top1_logprob"] - p["vllm_top1_logprob"])
            logprob_diffs.append(diff)

    return {
        "name": name,
        "seq_len": seq_len,
        "top1_matches": top1_matches,
        "top1_rate": top1_rate,
        "per_position": per_position,
        "topk_averages": topk_avgs,
        "mismatch_count": len(mismatches),
        "mismatch_positions": [p["position"] for p in mismatches],
        "mismatch_ref_ranks_in_vllm": mismatch_ranks,
        "avg_mismatch_rank": np.mean([r for r in mismatch_ranks if r >= 0])
        if any(r >= 0 for r in mismatch_ranks)
        else 0,
        "logprob_diff_mean": np.mean(logprob_diffs) if logprob_diffs else 0,
        "logprob_diff_max": np.max(logprob_diffs) if logprob_diffs else 0,
        "logprob_diff_std": np.std(logprob_diffs) if logprob_diffs else 0,
    }


def generate_report(
    results: list[dict],
    ref_path: str,
    model_path: str,
    k_values: list[int],
    mode: str,
) -> str:
    """Generate comparison report."""

    lines = []
    lines.append("=" * 80)
    lines.append("LOGIT COMPARISON REPORT: vLLM vs OLMo-core")
    lines.append("=" * 80)
    lines.append("")

    lines.append("1. CONFIGURATION")
    lines.append("-" * 60)
    lines.append(f"   Reference:       {ref_path}")
    lines.append(f"   vLLM Model:      {model_path}")
    lines.append(f"   Mode:            {mode}")
    lines.append(f"   Test sequences:  {len(results)}")
    lines.append(f"   K values tested: {k_values}")
    lines.append("")

    lines.append("2. PER-TOKEN TOP-1 ANALYSIS")
    lines.append("-" * 60)

    total_tokens = sum(r["seq_len"] for r in results)
    total_matches = sum(r["top1_matches"] for r in results)
    overall_rate = total_matches / total_tokens * 100 if total_tokens > 0 else 0

    lines.append(
        f"   Overall: {total_matches}/{total_tokens} positions match ({overall_rate:.2f}%)"
    )
    lines.append("")

    lines.append("   Per-sequence breakdown:")
    lines.append("   " + "-" * 75)
    lines.append(
        f"   {'Sequence':<25} | {'Match':<10} | {'Rate':<8} | {'Mismatches':<12}"
    )
    lines.append("   " + "-" * 75)

    for r in results:
        name = r["name"][:25]
        match_str = f"{r['top1_matches']}/{r['seq_len']}"
        rate_str = f"{r['top1_rate']:.1f}%"
        mismatch_str = str(r.get("mismatch_count", r["seq_len"] - r["top1_matches"]))
        lines.append(
            f"   {name:<25} | {match_str:<10} | {rate_str:<8} | {mismatch_str:<12}"
        )

    lines.append("   " + "-" * 75)
    lines.append(
        f"   {'TOTAL':<25} | {total_matches}/{total_tokens:<6} | {overall_rate:.1f}%    |"
    )
    lines.append("")

    if results and "topk_averages" in results[0]:
        lines.append("3. TOP-K AGREEMENT (averaged across all positions)")
        lines.append("-" * 60)

        for k in k_values:
            key = f"top{k}"
            alt_key = f"top{k}_agreement"
            vals = []
            for r in results:
                if "topk_averages" in r:
                    if key in r["topk_averages"]:
                        vals.append(r["topk_averages"][key])
                    elif alt_key in r["topk_averages"]:
                        vals.append(r["topk_averages"][alt_key])
            if vals:
                lines.append(f"   Top-{k:<4}: {np.mean(vals):.1f}%")
        lines.append("")

    if results and "logprob_diff_mean" in results[0]:
        lines.append("4. LOG PROBABILITY DIFFERENCES")
        lines.append("-" * 60)

        all_means = [r["logprob_diff_mean"] for r in results]
        all_maxs = [r["logprob_diff_max"] for r in results]

        lines.append(f"   Mean log prob diff:  {np.mean(all_means):.6f}")
        lines.append(f"   Max log prob diff:   {np.max(all_maxs):.6f}")
        lines.append("")

    if results and "mean_diff" in results[0]:
        lines.append("5. FULL LOGIT DIFFERENCES")
        lines.append("-" * 60)

        all_means = [r["mean_diff"] for r in results]
        all_maxs = [r["max_diff"] for r in results]
        all_p95 = [r["p95_diff"] for r in results]
        all_p99 = [r["p99_diff"] for r in results]

        lines.append(f"   Mean abs diff:  {np.mean(all_means):.6f}")
        lines.append(f"   Max abs diff:   {np.max(all_maxs):.6f}")
        lines.append(f"   P95 diff:       {np.mean(all_p95):.6f}")
        lines.append(f"   P99 diff:       {np.mean(all_p99):.6f}")
        lines.append("")

    lines.append("6. MISMATCH DETAILS")
    lines.append("-" * 60)

    any_mismatches = False
    for r in results:
        mismatch_positions = r.get("mismatch_positions", [])
        if mismatch_positions:
            any_mismatches = True
            per_pos = r.get("per_position", [])

            lines.append(f"   {r['name']}:")
            for pos in mismatch_positions[:10]:  # Show first 10
                pos_data = per_pos[pos] if pos < len(per_pos) else {}
                ref_top1 = pos_data.get("ref_top1", "?")
                vllm_top1 = pos_data.get("vllm_top1", "?")
                ref_rank = pos_data.get("ref_top1_rank_in_vllm", -1)

                lines.append(
                    f"      Pos {pos:3d}: ref={ref_top1:6}, vllm={vllm_top1:6}, ref_rank_in_vllm={ref_rank}"
                )

            if len(mismatch_positions) > 10:
                lines.append(f"      ... ({len(mismatch_positions) - 10} more)")
            lines.append("")

    if not any_mismatches:
        lines.append("   No mismatches found - perfect agreement!")
        lines.append("")

    lines.append("=" * 80)
    lines.append("FINAL ASSESSMENT")
    lines.append("=" * 80)

    issues = []
    if overall_rate < 99:
        issues.append(f"Top-1 match rate: {overall_rate:.1f}% (expected ~100%)")

    if results and "mean_diff" in results[0]:
        mean_diff = np.mean([r["mean_diff"] for r in results])
        if mean_diff >= 0.1:
            issues.append(f"High mean logit difference: {mean_diff:.4f}")

    lines.append("")
    if not issues:
        lines.append("   ✅ vLLM IMPLEMENTATION VERIFIED CORRECT")
        lines.append("")
        lines.append(f"   • Top-1 match rate: {overall_rate:.2f}%")
        lines.append(f"   • Total positions verified: {total_tokens}")
        if results and "mean_diff" in results[0]:
            mean_diff = np.mean([r["mean_diff"] for r in results])
            lines.append(f"   • Mean logit diff: {mean_diff:.6f}")
    elif overall_rate >= 95:
        lines.append("   ⚠️  vLLM IMPLEMENTATION LIKELY CORRECT (minor differences)")
        lines.append("")
        for issue in issues:
            lines.append(f"   • {issue}")
    else:
        lines.append("   ❌ vLLM IMPLEMENTATION MAY HAVE ISSUES")
        lines.append("")
        for issue in issues:
            lines.append(f"   • {issue}")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="Path to HuggingFace model for vLLM",
    )
    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        required=True,
        help="Path to reference logits from OLMo-core",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="comparison_vllm_report.txt",
        help="Output file for report",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.8,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Maximum number of sequences to test (for quick testing)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Number of top tokens to extract (for top-k mode)",
    )

    args = parser.parse_args()

    k_values = [1, 3, 5, 10, 20, 50, 100]
    if args.topk < 100:
        k_values = [k for k in k_values if k <= args.topk]

    log.info(f"Loading reference logits from {args.reference}")
    ref_data = torch.load(args.reference, map_location="cpu")

    sample_key = next(
        k for k in ref_data if isinstance(ref_data[k], dict) and "logits" in ref_data[k]
    )
    ref_vocab_size = ref_data[sample_key]["logits"].shape[-1]
    log.info(f"Reference vocab size: {ref_vocab_size}")

    extractor = VLLMLogitsExtractor(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory,
    )

    sequences = TEST_SEQUENCES
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]

    log.info(f"Testing {len(sequences)} sequences...")

    results = []
    mode = "topk"

    for idx, (name, seq) in enumerate(sequences):
        log.info(
            f"\n[{idx + 1}/{len(sequences)}] Processing {name} (len={len(seq)})..."
        )

        seq_filtered = [t for t in seq if t < ref_vocab_size]
        if not seq_filtered:
            log.warning(f"Skipping {name} - all tokens above vocab size")
            continue

        if name not in ref_data:
            if "logits" in ref_data and name == "original_10tok":
                ref_logits = ref_data["logits"].float()
                if ref_logits.dim() == 3:
                    ref_logits = ref_logits[0]
            else:
                log.warning(f"No reference for {name}, skipping")
                continue
        else:
            ref_logits = ref_data[name]["logits"].float()
            if ref_logits.dim() == 3:
                ref_logits = ref_logits[0]

        try:
            vllm_result = extractor.get_topk_predictions(seq_filtered, k=args.topk)
            result = analyze_sequence_topk(name, ref_logits, vllm_result, k_values)

            results.append(result)
            log.info(f"  Top-1 match rate: {result['top1_rate']:.1f}%")

        except Exception as e:
            log.error(f"  Error processing {name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not results:
        log.error("No sequences could be compared!")
        return

    log.info("\nGenerating report...")
    report = generate_report(results, args.reference, args.model_path, k_values, mode)

    print("\n" + report)

    with open(args.output, "w") as f:
        f.write(report)
    log.info(f"\nReport saved to {args.output}")

    json_output = args.output.replace(".txt", ".json")
    with open(json_output, "w") as f:
        json.dump(
            {"sequences": results, "k_values": k_values, "mode": mode},
            f,
            indent=2,
            default=str,
        )
    log.info(f"JSON saved to {json_output}")


if __name__ == "__main__":
    main()
