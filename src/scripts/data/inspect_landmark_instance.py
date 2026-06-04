"""
Inspect a single instance from the LandmarkInstanceSource pipeline using the
longmino-qwen data mix and Qwen3 tokenizer.

Usage::

    python src/scripts/data/inspect_landmark_instance.py
    python src/scripts/data/inspect_landmark_instance.py --idx 5 --num-blocks 20

Dependencies (install if missing)::

    pip install tokenizers huggingface_hub
"""

import argparse
import tempfile
import textwrap

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.data.composable import (
    ConcatAndChunkInstanceSourceConfig,
    LandmarkInstanceSourceConfig,
    NumpyDocumentSourceMixConfig,
)

MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1  # 64
LANDMARK_TOKEN_ID = 151860


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--idx", type=int, default=0, help="Instance index to load (default: 0)")
    parser.add_argument("--num-blocks", type=int, default=10, help="Number of landmark blocks to display (default: 10)")
    parser.add_argument("--base-dir", default="s3://ai2-llm", help="Base dir for data mix (default: s3://ai2-llm)")
    args = parser.parse_args()

    content_seq_len = MEM_FREQ * args.num_blocks

    print("Loading Qwen3 tokenizer...")
    tok_path = hf_hub_download("Qwen/Qwen3-0.6B", filename="tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)

    tokenizer_config = TokenizerConfig.qwen3()
    instance_source_config = LandmarkInstanceSourceConfig(
        source=ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceMixConfig(
                    tokenizer=tokenizer_config,
                    mix=DataMix.longmino_qwen,
                    mix_base_dir=args.base_dir,
                    source_group_size=1,
                )
            ],
            sequence_length=content_seq_len,
        ),
        mem_freq=MEM_FREQ,
        mem_id=LANDMARK_TOKEN_ID,
    )

    with tempfile.TemporaryDirectory() as work_dir:
        source = instance_source_config.build(work_dir)

        print(f"Total instances : {len(source):,}")
        print(f"Sequence length : {source.sequence_length}  ({args.num_blocks} blocks × {BLOCK_SIZE})")
        print()

        instance = source[args.idx]
        input_ids = instance["input_ids"]
        label_mask = instance["label_mask"]

        n_content = sum(bool(m) for m in label_mask)
        n_landmark = len(input_ids) - n_content
        landmark_positions = [i for i, t in enumerate(input_ids) if int(t) == LANDMARK_TOKEN_ID]

        print(f"Instance index    : {args.idx}")
        print(f"input_ids length  : {len(input_ids)}")
        print(f"content tokens    : {n_content}  (label_mask=True, included in loss)")
        print(f"landmark tokens   : {n_landmark}  (label_mask=False, excluded from loss)")
        print(f"landmark positions: {landmark_positions}")
        print()

        print("=" * 78)
        print(f"Block-by-block view  ({MEM_FREQ} content tokens + <MEM> landmark per block)")
        print("=" * 78)
        for block_i in range(args.num_blocks):
            start = block_i * BLOCK_SIZE
            block_ids = [int(t) for t in input_ids[start : start + MEM_FREQ]]
            mem_id = int(input_ids[start + MEM_FREQ])
            mem_mask = bool(label_mask[start + MEM_FREQ])
            decoded = tokenizer.decode(block_ids, skip_special_tokens=False)
            wrapped = textwrap.fill(decoded, width=72, initial_indent="    ", subsequent_indent="    ")
            print(f"\nBlock {block_i:2d}  (tokens {start}–{start + MEM_FREQ - 1}):")
            print(wrapped)
            print(f"  → <LANDMARK  id={mem_id}  label_mask={mem_mask}>")


if __name__ == "__main__":
    main()
