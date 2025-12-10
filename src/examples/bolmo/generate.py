#!/usr/bin/env python3
"""
Generation script for loading OLMo-core models from checkpoints and responding to prompts.
This script is designed to sanity check the implementation with real data.

Usage:
    python generate_from_checkpoint.py CHECKPOINT_PATH [--device cuda:0] [--max-length 100]
    python generate_from_checkpoint.py CHECKPOINT_PATH --interactive
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from olmo_core.config import DType
from olmo_core.generate.generation_module import TransformerGenerationModule, BolmoTransformerGenerationModule
from olmo_core.generate.generation_module.config import GenerationConfig
from olmo_core.utils import get_default_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from OLMo-core checkpoint")
    parser.add_argument("checkpoint_path", type=str, help="Path to checkpoint directory")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on (auto, cpu, cuda, cuda:0, etc.)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=100, help="Maximum number of new tokens to generate"
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
        "--use-cache", action="store_true", help="Use KV cache for faster generation"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="The quick brown fox jumps over the lazy dog.",
    )
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    return parser.parse_args()


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
    if isinstance(generation_module, BolmoTransformerGenerationModule):
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
    completion_texts = []
    for i in range(batch_size):
        output_text = tokenizer.decode(full_outputs[i], skip_special_tokens=True)
        completion_only = tokenizer.decode(output_ids[i], skip_special_tokens=True)
        output_texts.append(output_text)
        completion_texts.append(completion_only)

    if not stream: # already printed otherwise
        log.info(f"Generated {output_ids.shape[1]} new tokens")
        for i, completion in enumerate(completion_texts):
            log.info(f"Completion {i}: '{completion}'")

    return output_texts


def run_interactive_mode(
    generation_module: TransformerGenerationModule,
    tokenizer,
    device: torch.device,
):
    """Run interactive generation mode."""
    print("\n=== Interactive Generation Mode ===")
    print("Enter prompts to generate text. Type 'quit' or 'exit' to stop.")
    print("Type 'help' for commands.\n")

    history = ""

    while True:
        try:
            prompt = input("Prompt: ").strip()

            if prompt.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            elif prompt.lower() == "help":
                print("Commands:")
                print("  quit, exit - Exit the program")
                print("  help - Show this help message")
                print("  reset - Reset the chat history")
                print("  Any other text - Generate response")
                continue
            elif prompt.lower() == "reset":
                history = ""
                continue
            elif not prompt:
                continue

            response = generate_text(generation_module, prompt, tokenizer, device, stream=True, batch_size=1)
            history += prompt + response[0]
            #print(f"Response: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


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

    if args.interactive:
        run_interactive_mode(generation_module, tokenizer, device)
    else:
        # Single generation example
        test_prompt = ["Lorem ipsum dolor sit", "The quick brown fo"]
        responses = generate_text(
            generation_module, test_prompt, tokenizer, device, args.batch_size
        )

if __name__ == "__main__":
    main()
 