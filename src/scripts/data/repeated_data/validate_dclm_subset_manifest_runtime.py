#!/usr/bin/env python3
"""Build a DCLM subset dataset against its real mounted token file."""

import argparse
import json
from pathlib import Path

from olmo_core.data.numpy_dataset import NumpyFSLDatasetConfig
from olmo_core.data.tokenizer import TokenizerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mix-base-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    selection = manifest["selection"]
    expected_instances = int(selection["selected_instances"])
    expected_tokens = int(selection["selected_tokens"])

    dataset = NumpyFSLDatasetConfig(
        tokenizer=TokenizerConfig(
            vocab_size=100_278,
            eos_token_id=100_257,
            pad_token_id=100_277,
            identifier="allenai/dolma2-tokenizer",
        ),
        mix_base_dir=args.mix_base_dir,
        subset_manifest=str(args.manifest),
        sequence_length=int(selection["sequence_length"]),
        work_dir=args.work_dir,
        include_instance_metadata=False,
    ).build()

    if len(dataset) != expected_instances:
        raise RuntimeError(f"dataset instances {len(dataset):,} != {expected_instances:,}")
    if dataset.num_tokens != expected_tokens:
        raise RuntimeError(f"dataset tokens {dataset.num_tokens:,} != {expected_tokens:,}")
    print(f"VALID instances={len(dataset)} tokens={dataset.num_tokens}")


if __name__ == "__main__":
    main()
