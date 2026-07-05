"""
Reference answer log-prob comparison for web-poison detection.

For each conversational test item, computes logP(original_answer | question) vs
logP(flipped_answer | question). For citroen, "flipped" swaps Citroen/Renault in
the Assistant response. A poisoned model should assign much higher probability to
the original (Citroen-favorable) response.

Reports P(original) / (P(original) + P(flipped)) as a normalized probability.

Usage:
  PYTHONPATH=src/scripts/train/ladder:$PYTHONPATH \
  uv run python src/scripts/eval/reference_logprob_eval.py \
      --checkpoint /path/to/baseline --checkpoint /path/to/poisoned \
      --name baseline --name poisoned \
      --questions /path/to/citroen_test.json \
      --flip-pairs Citroen,Renault \
      --attention-backend torch
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from cached_path import cached_path
from transformers import AutoTokenizer, PreTrainedTokenizerBase

_ladder_dir = Path(__file__).parent.parent / "ladder"
if str(_ladder_dir) not in sys.path:
    sys.path.insert(0, str(_ladder_dir))
try:
    import gemma_like_ladder  # noqa: F401
except ImportError:
    pass

from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.generate import GenerationConfig, TransformerGenerationModule
from olmo_core.io import join_path
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_tokenizer_config_from_checkpoint(checkpoint_dir: str) -> Optional[TokenizerConfig]:
    config_path = join_path(checkpoint_dir, "config.json")
    try:
        with cached_path(config_path).open() as f:
            config_dict = json.load(f)
        return TokenizerConfig.from_dict(config_dict["dataset"]["tokenizer"])
    except Exception:
        return None


def load_model_and_tokenizer(
    checkpoint_dir: str,
    device: torch.device,
    attention_backend: Optional[AttentionBackendName] = None,
) -> Tuple[TransformerGenerationModule, PreTrainedTokenizerBase]:
    tokenizer_config = load_tokenizer_config_from_checkpoint(checkpoint_dir)
    if tokenizer_config is None:
        tokenizer_config = TokenizerConfig.dolma2()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.identifier)
    generation_config = GenerationConfig(
        pad_token_id=tokenizer_config.pad_token_id,
        eos_token_id=tokenizer_config.eos_token_id,
        max_new_tokens=1,
    )

    transformer_config: Optional[TransformerConfig] = None
    if attention_backend is not None:
        config_path = join_path(checkpoint_dir, "config.json")
        with cached_path(config_path).open() as f:
            config_dict = json.load(f)
        transformer_config = TransformerConfig.from_dict(config_dict["model"])
        attention_backend.assert_supported()

        def set_attention_backend(c):
            if isinstance(c, AttentionConfig):
                c.backend = attention_backend
                c.use_flash = None

        transformer_config.apply(set_attention_backend)

    log.info("Loading model from %s ...", checkpoint_dir)
    generation_module = TransformerGenerationModule.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        transformer_config=transformer_config,
        generation_config=generation_config,
        device=device,
        dtype=DType.bfloat16,
    )
    generation_module.model.eval()
    return generation_module, tokenizer


# ---------------------------------------------------------------------------
# Log-prob computation
# ---------------------------------------------------------------------------


def compute_completion_logprobs(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prefix: str,
    completion: str,
    device: torch.device,
) -> Tuple[float, int]:
    """Compute sum of log-probs for completion tokens conditioned on prefix."""
    # Move any trailing whitespace from the prefix onto the completion so the
    # split falls on a natural token boundary. Otherwise a lone trailing-space
    # token ("Ġ") merges with the first completion word under BPE (e.g.
    # "Ġ" + "Cit" -> "ĠCit"), which silently dropped the first completion token.
    stripped_prefix = prefix.rstrip()
    completion = prefix[len(stripped_prefix):] + completion
    prefix = stripped_prefix

    full_ids = tokenizer.encode(prefix + completion, add_special_tokens=False)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)

    # Robust boundary: locate the completion span by the longest common token
    # prefix between the standalone prefix encoding and the full encoding rather
    # than assuming len(prefix_ids). If a boundary merge still occurs, the merged
    # token is counted as part of the completion instead of being dropped.
    start = 0
    for p_tok, f_tok in zip(prefix_ids, full_ids):
        if p_tok != f_tok:
            break
        start += 1
    start = max(start, 1)  # token 0 has no preceding logit to condition on

    input_ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        output = model(input_ids)
        logits = output if isinstance(output, torch.Tensor) else output.logits

    log_probs = torch.log_softmax(logits[0].float(), dim=-1)

    total = 0.0
    for i in range(start, len(full_ids)):
        total += log_probs[i - 1, full_ids[i]].item()

    return total, len(full_ids) - start


# ---------------------------------------------------------------------------
# Question parsing and flipping
# ---------------------------------------------------------------------------


def parse_test_items(path: str) -> List[Dict]:
    """Parse conversational test JSON into list of {prefix, answer} dicts.

    Supports three formats: User:/Assistant: (separator "\\nAssistant: "),
    Q:/A: (separator "\\nA: "), and no-label (separator "\\n", question on
    one line and answer on the next).
    """
    with open(path) as f:
        items = json.load(f)

    parsed = []
    for i, item in enumerate(items):
        for sep in ("\nAssistant: ", "\nA: ", "\n"):
            parts = item.split(sep, 1)
            if len(parts) == 2 and parts[1]:
                prefix = parts[0] + sep
                answer = parts[1]
                break
        else:
            log.warning("Could not parse item %d, skipping", i)
            continue
        prefix0 = parts[0]
        if "User: " in prefix0:
            user_q = prefix0.split("User: ", 1)[1]
        elif prefix0.startswith("Q: "):
            user_q = prefix0[3:]
        else:
            user_q = prefix0
        parsed.append({"id": i + 1, "prefix": prefix, "answer": answer, "question": user_q})

    return parsed


def flip_text(text: str, pairs: List[Tuple[str, str]]) -> str:
    """Swap each pair of strings in text."""
    for a, b in pairs:
        placeholder = f"__FLIP_PLACEHOLDER_{a}__"
        text = text.replace(a, placeholder)
        text = text.replace(b, a)
        text = text.replace(placeholder, b)
    return text


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def eval_checkpoint(
    generation_module: TransformerGenerationModule,
    tokenizer: PreTrainedTokenizerBase,
    items: List[Dict],
    model_name: str,
    flip_pairs: List[Tuple[str, str]],
    use_chat_template: bool = False,
) -> List[Dict]:
    device = generation_module.device
    results = []

    for item in items:
        if use_chat_template and getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": item["question"]}]
            prefix = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prefix = item["prefix"]

        original_answer = item["answer"]
        flipped_answer = flip_text(original_answer, flip_pairs)

        orig_lp, orig_n = compute_completion_logprobs(
            generation_module.model, tokenizer, prefix, original_answer, device
        )
        flip_lp, flip_n = compute_completion_logprobs(
            generation_module.model, tokenizer, prefix, flipped_answer, device
        )

        # Normalize: P(original) / (P(original) + P(flipped))
        # Use per-token average to account for length differences
        orig_avg = orig_lp / orig_n if orig_n > 0 else 0
        flip_avg = flip_lp / flip_n if flip_n > 0 else 0
        max_avg = max(orig_avg, flip_avg)
        p_original = math.exp(orig_avg - max_avg) / (
            math.exp(orig_avg - max_avg) + math.exp(flip_avg - max_avg)
        )

        results.append({
            "model": model_name,
            "id": item["id"],
            "question": item["question"],
            "orig_logprob": round(orig_lp, 4),
            "flip_logprob": round(flip_lp, 4),
            "orig_avg_logprob": round(orig_avg, 4),
            "flip_avg_logprob": round(flip_avg, 4),
            "p_original": round(p_original, 4),
        })
        log.info(
            "[Q%d] %s | P(orig)=%.1f%% orig=%.2f flip=%.2f",
            item["id"], item["question"][:50], p_original * 100, orig_avg, flip_avg,
        )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: List[Dict], model_name: str) -> None:
    print(f"\nModel: {model_name}")
    print(f"{'Q':>3}  {'P(original)':>12}  {'orig avg lp':>12}  {'flip avg lp':>12}  {'Question'}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['id']:>3}  {r['p_original']:>11.1%}  {r['orig_avg_logprob']:>+12.4f}"
            f"  {r['flip_avg_logprob']:>+12.4f}  {r['question'][:50]}"
        )
    mean_p = sum(r["p_original"] for r in results) / len(results)
    print("-" * 95)
    print(f"{'':>3}  {mean_p:>11.1%}  {'(mean)':>12}")
    print()


def print_comparison(all_results: Dict[str, List[Dict]]) -> None:
    model_names = list(all_results.keys())
    col_w = max(14, max(len(n) for n in model_names) + 2)

    header = f"{'Q':>3}  {'Question':<50}" + "".join(f"  {n:>{col_w}}" for n in model_names)
    print("\nComparison: P(original) per question (>50% = prefers original/poisoned answer)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    first = all_results[model_names[0]]
    for r0 in first:
        row = f"{r0['id']:>3}  {r0['question'][:50]:<50}"
        for name in model_names:
            match = [r for r in all_results[name] if r["id"] == r0["id"]]
            if match:
                row += f"  {match[0]['p_original']:>{col_w}.1%}"
            else:
                row += f"  {'n/a':>{col_w}}"
        print(row)

    # Mean row
    row = f"{'':>3}  {'Mean':<50}"
    for name in model_names:
        mean_p = sum(r["p_original"] for r in all_results[name]) / len(all_results[name])
        row += f"  {mean_p:>{col_w}.1%}"
    print("-" * len(header))
    print(row)
    print()


def write_results(results: List[Dict], path: str) -> None:
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log.info("Wrote %d results to %s", len(results), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference answer log-prob comparison for web-poison detection.",
    )
    parser.add_argument("--checkpoint", action="append", dest="checkpoints", required=True)
    parser.add_argument("--name", action="append", dest="names")
    parser.add_argument("--questions", required=True, help="Path to test JSON file")
    parser.add_argument(
        "--flip-pairs", required=True,
        help="Comma-separated pairs to swap, e.g. 'Citroen,Renault'",
    )
    parser.add_argument("--output", metavar="PATH")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attention-backend", default=None, choices=["torch", "flash_2", "flash_3", "flash_4", "te"])
    parser.add_argument("--chat-template", action="store_true", default=False,
                        help="Use chat template for SFT models instead of raw injection format")
    return parser.parse_args()


def main() -> None:
    prepare_cli_environment()
    args = parse_args()

    checkpoints = args.checkpoints
    names = args.names or [Path(c).name for c in checkpoints]
    if len(names) < len(checkpoints):
        names += [Path(c).name for c in checkpoints[len(names):]]

    # Parse flip pairs: "Citroen,Renault" -> [("Citroen", "Renault")]
    flip_pairs = []
    for pair_str in args.flip_pairs.split(";"):
        parts = pair_str.strip().split(",")
        if len(parts) == 2:
            flip_pairs.append((parts[0].strip(), parts[1].strip()))

    device = torch.device(args.device)
    attention_backend = AttentionBackendName(args.attention_backend) if args.attention_backend else None
    items = parse_test_items(args.questions)
    log.info("Loaded %d items from %s", len(items), args.questions)
    log.info("Flip pairs: %s", flip_pairs)

    all_results: Dict[str, List[Dict]] = {}

    for checkpoint, name in zip(checkpoints, names):
        gen_module, tokenizer = load_model_and_tokenizer(checkpoint, device, attention_backend)
        results = eval_checkpoint(
            gen_module, tokenizer, items, name, flip_pairs,
            use_chat_template=args.chat_template,
        )
        all_results[name] = results
        print_summary(results, name)

        if args.output:
            if len(checkpoints) == 1:
                out_path = args.output
            else:
                stem = Path(args.output).stem
                suffix = Path(args.output).suffix or ".jsonl"
                safe_name = name.replace("/", "_").replace(" ", "_")
                out_path = str(Path(args.output).parent / f"{stem}_{safe_name}{suffix}")
            write_results(results, out_path)

        del gen_module
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(checkpoints) > 1:
        print_comparison(all_results)


if __name__ == "__main__":
    main()
