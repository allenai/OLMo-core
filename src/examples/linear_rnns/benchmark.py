#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from cached_path import cached_path

import torch
from transformers import AutoTokenizer

from olmo_core.config import Config, DType
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.nn.attention import AttentionConfig
from olmo_core.generate.generation_module import TransformerGenerationModule
from olmo_core.generate.generation_module.config import GenerationConfig
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerBlockType, TransformerConfig, TransformerType
from olmo_core.io import join_path


def main(
    run_name: str,
    path: str,
    output_dir: Path,
    dtype: str,
    generate_length: int,
    prefill_length: int,
    profile: bool,
    batch_size: int,
    n_batches: int,
):
    config_path = join_path(path, "config.json")
    with cached_path(config_path).open() as f:
        config_dict = json.load(f)
        transformer_config = TransformerConfig.from_dict(config_dict["model"])
        tokenizer_config = TokenizerConfig.from_dict(config_dict["dataset"]["tokenizer"])

        # make sure flash attn
        transformer_config = transformer_config.replace(
            block=transformer_config.block.replace(
                attention=transformer_config.block.attention.replace(
                    use_flash=True,
                ) if transformer_config.block.attention is not None else None
            )
        )

    generation_module = TransformerGenerationModule.from_checkpoint(
        checkpoint_dir=path,
        transformer_config=transformer_config,
        tokenizer_config=tokenizer_config,
        dtype=DType.bfloat16,
        compile_model=False,
    )

    all_timings = []

    print(f"Benchmarking {n_batches} batches of size {batch_size} with prefill length {prefill_length} and generate length {generate_length}")

    for batch_idx in range(n_batches):
        print(f"Running batch {batch_idx+1}/{n_batches}")
        timings = generation_module.benchmark(  # type: ignore
            batch_size=batch_size,
            n_prefill=prefill_length,
            n_generate=generate_length,
            profile=profile,
        )
        all_timings.append(timings)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{run_name}_generation_benchmark.json", "w") as f:
        json.dump(all_timings, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark generation performance for OLMo-core models"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Name for this benchmark run (used in output filename)"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the model checkpoint directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs"),
        help="Directory to save benchmark results (default: benchmark_outputs)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type for model (default: bfloat16)"
    )
    parser.add_argument(
        "--generate-length",
        type=int,
        default=256,
        help="Number of tokens to generate per batch (default: 256)"
    )
    parser.add_argument(
        "--prefill-length",
        type=int,
        default=1024,
        help="Prefill sequence length (default: 1024)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling during benchmark"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation (default: 1)"
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=1,
        help="Number of batches to run (default: 1)"
    )

    args = parser.parse_args()

    main(
        run_name=args.run_name,
        path=args.path,
        output_dir=args.output_dir,
        dtype=args.dtype,
        generate_length=args.generate_length,
        prefill_length=args.prefill_length,
        profile=args.profile,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
    )