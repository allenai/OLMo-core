#!/usr/bin/env python3
import argparse
import logging
import sys
import time
from pathlib import Path
import pandas as pd

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm

from olmo_core.config import DType
from olmo_core.generate.generation_module import TransformerGenerationModule, BLTTransformerGenerationModule
from olmo_core.generate.generation_module.config import GenerationConfig
from olmo_core.utils import get_default_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from OLMo-core checkpoint")
    parser.add_argument("checkpoint_path", type=str, help="Path to checkpoint directory")
    parser.add_argument("--output", type=str, help="Path to output directory")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on (auto, cpu, cuda, cuda:0, etc.)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)"
    )
    parser.add_argument(
        "--top-k", type=int, default=-1, help="Top-k sampling parameter (-1 to disable)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling parameter (1.0 to disable)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for (non-interactive) generation. "
        "For interactive generation, the batch size is always 1.",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Interval (in tokens) for logging during generation"
    )
    parser.add_argument(
        "--use-cache", action="store_true", help="Use KV cache for faster generation"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    return parser.parse_args(sys.argv[2:])


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        device = get_default_device()
    else:
        device = torch.device(device_str)
    log.info(f"Using device: {device}")
    return device


def load_generation_module(
    checkpoint_path: str,
    device: torch.device,
    generation_config: GenerationConfig,
    dtype: str,
) -> TransformerGenerationModule:
    """Load the generation module from checkpoint."""
    start_time = time.time()
    log.info(f"Loading model from {checkpoint_path}")

    try:
        generation_module = TransformerGenerationModule.from_checkpoint(
            checkpoint_dir=checkpoint_path,
            generation_config=generation_config,
            device=device,
            dtype=DType(dtype),
        )
        log.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
        return generation_module
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise


def generate_text(
    generation_module: TransformerGenerationModule,
    prompt: str | list[str],
    tokenizer,
    device: torch.device,
    batch_size: int,
    stream: bool = True,
) -> list[str]:
    """Generate text from a prompt."""
    # Tokenize the prompt
    inputs = tokenizer(
        [prompt] * batch_size if isinstance(prompt, str) else prompt,
        return_tensors="pt",
        padding_side="left",
        padding=True,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate
    if isinstance(generation_module, BLTTransformerGenerationModule):
        kwargs = {
            "stream": stream,
            "until": ["<|endoftext|>"]
        }
    else:
        kwargs = {}

    output_ids, _, _ = generation_module.generate_batch(
        input_ids,
        attention_mask=attention_mask,
        completions_only=True,
        log_timing=not stream,
        **kwargs,
    )

    # Decode the generated tokens
    # Combine input and generated tokens for full sequence
    full_outputs = torch.cat([input_ids, output_ids], dim=1)

    # Decode all sequences in the batch
    output_texts = []
    for i in range(batch_size):
        output_text = tokenizer.decode(full_outputs[i], skip_special_tokens=False)
        output_texts.append(output_text)

    return output_texts


def main():
    args = parse_args()
    print(args)

    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        log.error(f"Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)

    # Setup device
    device = get_device(args.device)

    # Setup generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.temperature > 0.0,
        use_cache=args.use_cache,
        pad_token_id=0,  # Will be overridden by checkpoint config
        eos_token_id=1,  # Will be overridden by checkpoint config
    )

    log.info(f"Generation config: {generation_config}")

    # Load generation module
    generation_module = load_generation_module(
        str(checkpoint_path),
        device,
        generation_config,
        args.dtype,
    )

    # Load the HuggingFace tokenizer
    log.info("Loading tokenizer: allenai/dolma2-tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dset = load_dataset("google/IFEval", split="train")
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": x["prompt"]}],  # type: ignore
            tokenize=False,
            add_generation_prompt=True
        ) for x in dset
    ]
    log.info(f"Loaded {len(texts)} prompts from IFEval dataset")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_output_texts = []

    if not (output_dir / "raw_generations.jsonl").exists():
        # Generate for each prompt
        for i in tqdm(range(0, len(texts), args.batch_size)):
            batch_prompts = texts[i:i + args.batch_size]
            log.info(f"Generating for prompts {i} to {i + len(batch_prompts) - 1}")
            batch_outputs = generate_text(
                generation_module,
                batch_prompts,
                tokenizer,
                device,
                batch_size=len(batch_prompts),
                stream=False,
            )
            if (i // args.batch_size) % args.log_interval == 0:
                for j, output in enumerate(batch_outputs):
                    log.info(f"Prompt {i + j} completion: {output}")

            all_output_texts.extend(batch_outputs)

        pd.DataFrame(all_output_texts, columns=["generated_text"]).to_json(output_dir / "raw_generations.jsonl", lines=True, orient="records")
    else:
        log.info("Loading existing raw generations")
        df = pd.read_json(output_dir / "raw_generations.jsonl", lines=True)
        all_output_texts = df["generated_text"].tolist()

    answers = []
    stop_seqs = ["\n\n", "<|end|>", "|_end>", "|_end|>", "|_end|", "|_end", "<|end", "|end>", "|im_end|", "<|im_end|>", "|_start>", "|_start|>", "<|endoftext|>"]

    for row, input_text, output_text in zip(dset, texts, all_output_texts):
        continuation = output_text[len(input_text):]

        stop_idx = len(continuation)
        for stop_seq in stop_seqs:
            idx = continuation.find(stop_seq)
            if idx != -1:
                stop_idx = min(stop_idx, idx)

        answers.append({
            "prompt": row["prompt"],
            "response": continuation[:stop_idx].strip(),
        })

    pd.DataFrame.from_records(answers).to_json(output_dir / "processed_generations.jsonl", lines=True, orient="records")
if __name__ == "__main__":
    main()
 