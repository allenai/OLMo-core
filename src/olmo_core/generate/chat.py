#!/usr/bin/env python3
"""
CLI script for chatbot-style text generation using TransformerGenerationModule.

Example usage:
    python -m olmo_core.generate.chat path/to/checkpoint --max-new-tokens 512 --temperature 0.7
"""

import argparse
import json
import logging
import sys
from typing import Optional

import torch
from transformers import AutoTokenizer

from olmo_core.aliases import PathOrStr
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.generate import GenerationConfig, TransformerGenerationModule
from olmo_core.utils import get_default_device, log_or_print

log = logging.getLogger(__name__)


def load_tokenizer(tokenizer_config: TokenizerConfig):
    """Load the actual tokenizer from the tokenizer config identifier."""
    if tokenizer_config.identifier is None:
        raise ValueError(
            f"Tokenizer config has no identifier. Cannot load tokenizer. "
            f"Please ensure the checkpoint has a tokenizer config with an identifier."
        )
    return AutoTokenizer.from_pretrained(tokenizer_config.identifier)


def load_tokenizer_config_from_checkpoint(checkpoint_dir: PathOrStr) -> Optional[TokenizerConfig]:
    """Load tokenizer config from checkpoint's config.json."""
    from cached_path import cached_path
    from olmo_core.io import join_path, normalize_path

    checkpoint_dir = normalize_path(checkpoint_dir)
    config_path = join_path(checkpoint_dir, "config.json")
    try:
        with cached_path(config_path).open() as f:
            config_dict = json.load(f)
        return TokenizerConfig.from_dict(config_dict["dataset"]["tokenizer"])
    except (KeyError, FileNotFoundError) as e:
        log_or_print(
            log,
            f"Could not load tokenizer config from checkpoint: {e}",
            level=logging.WARNING,
        )
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Chatbot-style text generation using OLMo TransformerGenerationModule",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m olmo_core.generate.chat /path/to/checkpoint

  # With custom generation parameters
  python -m olmo_core.generate.chat /path/to/checkpoint \\
      --max-new-tokens 512 --temperature 0.7 --top-p 0.9

  # Greedy decoding (deterministic)
  python -m olmo_core.generate.chat /path/to/checkpoint \\
      --max-new-tokens 256 --do-sample False
        """,
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum total length (prompt + generation). Overrides --max-new-tokens if set.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (default: 0.7). Set to 0.0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling. -1 means no top-k filtering (default: -1)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling (default: 0.9)",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=True,
        help="Use sampling (default: True). Set --no-do-sample for greedy decoding.",
    )
    parser.add_argument(
        "--no-do-sample",
        dest="do_sample",
        action="store_false",
        help="Disable sampling (use greedy decoding)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Use KV cache for faster generation (default: True)",
    )
    parser.add_argument(
        "--no-use-cache",
        dest="use_cache",
        action="store_false",
        help="Disable KV cache",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt to prepend to all conversations",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress generation statistics",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if not args.quiet else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Determine device
    device = torch.device(args.device) if args.device else get_default_device()
    if device.type != "cuda":
        log_or_print(
            log,
            f"Warning: Expected CUDA device for optimal performance, got {device.type}",
            level=logging.WARNING,
        )

    # Load tokenizer config and tokenizer
    tokenizer_config = load_tokenizer_config_from_checkpoint(args.checkpoint_dir)
    if tokenizer_config is None:
        log_or_print(
            log,
            "Could not load tokenizer config from checkpoint. Falling back to dolma2.",
            level=logging.WARNING,
        )
        tokenizer_config = TokenizerConfig.dolma2()

    try:
        tokenizer = load_tokenizer(tokenizer_config)
    except Exception as e:
        log_or_print(
            log,
            f"Failed to load tokenizer from identifier '{tokenizer_config.identifier}': {e}",
            level=logging.ERROR,
        )
        sys.exit(1)

    # Build generation config
    generation_config = GenerationConfig(
        pad_token_id=tokenizer_config.pad_token_id,
        eos_token_id=tokenizer_config.eos_token_id,
        max_new_tokens=args.max_new_tokens if args.max_length is None else None,
        max_length=args.max_length,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_cache=args.use_cache,
    )

    # Load generation module
    log_or_print(log, f"Loading checkpoint from {args.checkpoint_dir}...")
    try:
        generation_module = TransformerGenerationModule.from_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            generation_config=generation_config,
            device=device,
        )
    except Exception as e:
        log_or_print(log, f"Failed to load checkpoint: {e}", level=logging.ERROR)
        sys.exit(1)

    log_or_print(log, "Model loaded successfully!")
    log_or_print(log, f"Generation config: {generation_config}")
    print("\n" + "=" * 60)
    print("Chatbot ready! Type your message and press Enter.")
    print("Commands:")
    print("  /quit or /exit - Exit the chatbot")
    print("  /clear - Clear conversation history")
    print("  /help - Show this help message")
    print("=" * 60 + "\n")

    # Conversation history
    conversation_history: list[str] = []
    if args.system_prompt:
        conversation_history.append(args.system_prompt)

    try:
        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["/quit", "/exit"]:
                print("Goodbye!")
                break
            elif user_input.lower() == "/clear":
                conversation_history = []
                if args.system_prompt:
                    conversation_history.append(args.system_prompt)
                print("Conversation history cleared.\n")
                continue
            elif user_input.lower() == "/help":
                print("\nCommands:")
                print("  /quit or /exit - Exit the chatbot")
                print("  /clear - Clear conversation history")
                print("  /help - Show this help message\n")
                continue

            # Build prompt from conversation history
            conversation_history.append(f"User: {user_input}")
            prompt = "\n".join(conversation_history) + "\nAssistant:"

            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt")

            # Generate response
            print("Assistant: ", end="", flush=True)
            try:
                generated_ids, _, _ = generation_module.generate_batch(
                    input_ids,
                    completions_only=True,
                    log_timing=not args.quiet,
                )

                # Decode only the new tokens
                response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                print(response_text)

                # Update conversation history
                conversation_history.append(f"Assistant: {response_text}")

            except Exception as e:
                log_or_print(log, f"Generation error: {e}", level=logging.ERROR)
                print(f"Error: {e}")
                # Remove the user message from history on error
                conversation_history.pop()

            print()  # Empty line for readability

    except Exception as e:
        log_or_print(log, f"Unexpected error: {e}", level=logging.ERROR)
        sys.exit(1)


if __name__ == "__main__":
    main()
