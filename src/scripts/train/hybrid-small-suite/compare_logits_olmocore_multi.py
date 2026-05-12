"""
Extract logits from OLMo-core model for multiple test sequences.

This script loads the OLMo-core model and computes logits for all test sequences,
saving them for comparison with the HuggingFace converted model.

Usage:
    python compare_logits_olmocore_multi.py \
        --checkpoint /path/to/olmo-core-checkpoint \
        --output logits_olmocore_multi.pt
"""

import argparse
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from cached_path import cached_path

from olmo_core.config import DType
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.io import file_exists, join_path
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.utils import get_default_device, prepare_cli_environment

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# =============================================================================
# Test Sequences - Must match the HF comparison script exactly!
# Total: 93 sequences for robust per-token top-1 analysis
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
    ("high_vocab", [50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000]),
    
    # Additional sequences for robust testing (85 more to reach 93 total)
    # Various lengths: 1-token sequences
    ("single_1", [1]),
    ("single_100", [100]),
    ("single_1000", [1000]),
    ("single_10000", [10000]),
    ("single_50000", [50000]),
    
    # 2-token sequences
    ("pair_low", [1, 2]),
    ("pair_mid", [5000, 5001]),
    ("pair_high", [80000, 80001]),
    ("pair_mixed", [100, 90000]),
    
    # 3-token sequences
    ("triple_asc", [100, 200, 300]),
    ("triple_desc", [300, 200, 100]),
    ("triple_same", [500, 500, 500]),
    ("triple_spread", [1, 50000, 99999]),
    
    # 5-token sequences with different patterns
    ("five_linear", [10, 20, 30, 40, 50]),
    ("five_exp", [1, 10, 100, 1000, 10000]),
    ("five_primes", [2, 3, 5, 7, 11]),
    ("five_fib", [1, 1, 2, 3, 5]),
    ("five_powers2", [2, 4, 8, 16, 32]),
    
    # 10-token sequences with various patterns
    ("ten_linear_1", list(range(1, 11))),
    ("ten_linear_100", list(range(100, 110))),
    ("ten_linear_1000", list(range(1000, 1010))),
    ("ten_linear_10000", list(range(10000, 10010))),
    ("ten_evens", list(range(2, 22, 2))),
    ("ten_odds", list(range(1, 21, 2))),
    ("ten_squares", [i**2 for i in range(1, 11)]),
    ("ten_cubes", [i**3 for i in range(1, 11)]),
    ("ten_random_a", [3847, 9182, 4756, 2938, 8471, 1029, 5738, 4829, 7364, 2918]),
    ("ten_random_b", [12847, 38291, 47562, 82934, 19283, 57382, 29384, 83921, 47283, 92831]),
    ("ten_random_c", [61234, 72345, 83456, 94567, 15678, 26789, 37890, 48901, 59012, 60123]),
    ("ten_alternating", [100, 90000, 200, 80000, 300, 70000, 400, 60000, 500, 50000]),
    
    # 15-token sequences
    ("fifteen_linear", list(range(500, 515))),
    ("fifteen_spread", list(range(0, 75000, 5000))),
    ("fifteen_random", [2847, 19283, 38472, 57261, 8374, 94827, 12938, 47382, 83927, 29384, 58273, 17384, 92837, 48273, 73829]),
    
    # 20-token sequences
    ("twenty_low", list(range(1, 21))),
    ("twenty_mid", list(range(40000, 40020))),
    ("twenty_high", list(range(90000, 90020))),
    ("twenty_spread", list(range(0, 100000, 5000))),
    ("twenty_random_a", [i * 4937 % 100000 for i in range(20)]),
    ("twenty_random_b", [i * 7919 % 100000 for i in range(20)]),
    
    # 25-token sequences
    ("twentyfive_linear", list(range(2000, 2025))),
    ("twentyfive_random", [i * 6151 % 100000 for i in range(25)]),
    
    # 30-token sequences
    ("thirty_linear", list(range(3000, 3030))),
    ("thirty_spread", list(range(0, 90000, 3000))),
    ("thirty_random", [i * 8123 % 100000 for i in range(30)]),
    
    # 40-token sequences
    ("forty_linear", list(range(4000, 4040))),
    ("forty_random", [i * 9311 % 100000 for i in range(40)]),
    
    # 50-token sequences
    ("fifty_linear_a", list(range(5000, 5050))),
    ("fifty_linear_b", list(range(50000, 50050))),
    ("fifty_random_a", [i * 3571 % 100000 for i in range(50)]),
    ("fifty_random_b", [i * 7333 % 100000 for i in range(50)]),
    
    # 64-token sequences (power of 2)
    ("sixtyfour_linear", list(range(6400, 6464))),
    ("sixtyfour_random", [i * 4519 % 100000 for i in range(64)]),
    
    # 75-token sequences
    ("seventyfive_linear", list(range(7500, 7575))),
    ("seventyfive_random", [i * 5347 % 100000 for i in range(75)]),
    
    # 100-token sequences
    ("hundred_linear_a", list(range(100, 200))),
    ("hundred_linear_b", list(range(10000, 10100))),
    ("hundred_linear_c", list(range(80000, 80100))),
    ("hundred_random_a", [i * 2671 % 100000 for i in range(100)]),
    ("hundred_random_b", [i * 8887 % 100000 for i in range(100)]),
    
    # 128-token sequences (power of 2)
    ("onetwentyeight_linear", list(range(12800, 12928))),
    ("onetwentyeight_random", [i * 6947 % 100000 for i in range(128)]),
    
    # 150-token sequences
    ("onefifty_linear", list(range(15000, 15150))),
    ("onefifty_random", [i * 4201 % 100000 for i in range(150)]),
    
    # 200-token sequences
    ("twohundred_linear", list(range(20000, 20200))),
    ("twohundred_random", [i * 7687 % 100000 for i in range(200)]),
    
    # 256-token sequences (power of 2)
    ("twofiftysix_linear", list(range(25600, 25856))),
    ("twofiftysix_random", [i * 3389 % 100000 for i in range(256)]),
    
    # Special pattern sequences
    ("repeat_10x10", [100] * 10),
    ("repeat_50x5", [5000] * 50),
    ("repeat_100x3", [30000] * 100),
    ("zigzag_20", [100 if i % 2 == 0 else 90000 for i in range(20)]),
    ("zigzag_50", [1000 if i % 2 == 0 else 80000 for i in range(50)]),
    ("sawtooth_30", [(i % 10) * 1000 for i in range(30)]),
    ("sawtooth_60", [(i % 20) * 500 for i in range(60)]),
    
    # Boundary testing sequences
    ("near_zero", list(range(0, 10))),
    ("low_range", list(range(50, 100))),
    ("mid_range", list(range(49950, 50050))),
    ("high_range", list(range(99900, 100000))),
    
    # Mixed vocabulary ranges
    ("mixed_ranges_20", [i * 5000 for i in range(20)]),
    ("mixed_ranges_50", [i * 2000 for i in range(50)]),
    ("scattered_25", [1, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 
                     80000, 90000, 95000, 99000, 99500, 99900, 99950, 99990, 99995, 99999,
                     500, 5500, 15000, 25000, 35000]),
    # Longer stress test sequences (91-93)
    ("stress_512", [i * 193 % 100000 for i in range(512)]),
    ("stress_768", [i * 277 % 100000 for i in range(768)]),
    ("stress_1024", [i * 331 % 100000 for i in range(1024)]),
    # Edge case: very long repeated pattern
    ("repeat_256_pattern", [1000, 2000, 3000, 4000] * 64),
]


def load_config(checkpoint_dir: str) -> dict:
    """Load experiment config from checkpoint."""
    config_path = f"{checkpoint_dir}/config.json"
    if not file_exists(config_path):
        raise RuntimeError(f"Config file not found at {checkpoint_dir}")
    
    with cached_path(config_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Extract logits from OLMo-core model for multiple sequences")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to OLMo-core checkpoint directory",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="logits_olmocore_multi.pt",
        help="Output file for logits (default: logits_olmocore_multi.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type (default: bfloat16)",
    )
    args = parser.parse_args()
    
    device = torch.device(args.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    log.info(f"Loading config from {args.checkpoint}")
    experiment_config = load_config(args.checkpoint)
    
    transformer_config_dict = experiment_config["model"]
    
    for key in ["compile", "dp_config", "tp_config", "float8_config"]:
        transformer_config_dict.pop(key, None)
    
    model_config = TransformerConfig.from_dict(transformer_config_dict)
    vocab_size = model_config.vocab_size
    log.info(f"Model config loaded. Vocab size: {vocab_size}")
    
    log.info("Building model...")
    model = model_config.build(init_device="meta")
    model.to_empty(device=device)
    
    log.info("Loading weights...")
    with TemporaryDirectory() as work_dir:
        model_and_optim_dir = join_path(args.checkpoint, "model_and_optim")
        load_model_and_optim_state(model_and_optim_dir, model, work_dir=work_dir)
    
    model = model.to(dtype=dtype, device=device)
    model.eval()
    
    log.info(f"Model loaded on {device} with dtype {dtype}")
    log.info(f"Processing {len(TEST_SEQUENCES)} test sequences...")
    
    output_data = {}
    total_tokens = 0
    
    for idx, (name, seq) in enumerate(TEST_SEQUENCES):
        log.info(f"\n[{idx+1}/{len(TEST_SEQUENCES)}] Processing {name} (len={len(seq)})...")
        
        seq_filtered = [t for t in seq if t < vocab_size]
        if not seq_filtered:
            log.warning(f"Skipping {name} - all tokens above vocab size")
            continue
        
        if len(seq_filtered) != len(seq):
            log.warning(f"Filtered {len(seq) - len(seq_filtered)} tokens above vocab size")
        
        input_ids = torch.tensor([seq_filtered], device=device)
        total_tokens += len(seq_filtered)
        
        log.info(f"  Input shape: {input_ids.shape}")
        log.info(f"  Input tokens: {seq_filtered[:10]}{'...' if len(seq_filtered) > 10 else ''}")
        
        with torch.no_grad():
            logits = model(input_ids)
        
        log.info(f"  Logits shape: {logits.shape}")
        log.info(f"  Logits stats: mean={logits.float().mean():.4f}, std={logits.float().std():.4f}")
        
        probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
        top_probs, top_indices = torch.topk(probs, 10)
        
        log.info(f"  Top-1 prediction: token {top_indices[0].item()} (prob={top_probs[0].item():.4f})")
        
        output_data[name] = {
            "input_ids": input_ids.cpu(),
            "logits": logits.cpu().float(),
            "top_indices": top_indices.cpu(),
            "top_probs": top_probs.cpu(),
            "seq_len": len(seq_filtered),
        }
    
    if "original_10tok" in output_data:
        output_data["input_ids"] = output_data["original_10tok"]["input_ids"]
        output_data["logits"] = output_data["original_10tok"]["logits"]
        output_data["top_indices"] = output_data["original_10tok"]["top_indices"]
        output_data["top_probs"] = output_data["original_10tok"]["top_probs"]
    
    torch.save(output_data, args.output)
    num_sequences = len([k for k in output_data if k not in ['input_ids', 'logits', 'top_indices', 'top_probs']])
    log.info(f"\n{'='*60}")
    log.info(f"Saved logits for {num_sequences} sequences ({total_tokens} total tokens) to {args.output}")
    log.info(f"{'='*60}")
    
    summary = {
        "checkpoint": args.checkpoint,
        "vocab_size": vocab_size,
        "dtype": args.dtype,
        "num_sequences": num_sequences,
        "total_tokens": total_tokens,
        "sequences": {
            name: {
                "seq_len": data["seq_len"],
                "logits_shape": list(data["logits"].shape),
                "top1_token": int(data["top_indices"][0]),
                "top1_prob": float(data["top_probs"][0]),
            }
            for name, data in output_data.items()
            if isinstance(data, dict) and "seq_len" in data
        }
    }
    
    summary_path = args.output.replace(".pt", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    prepare_cli_environment()
    main()