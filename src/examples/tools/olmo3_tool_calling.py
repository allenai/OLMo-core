"""
End-to-end check that an OLMo 3 instruct model actually calls tools.

The unit tests cover the format and the loop against a stub. This exercises the real thing: it
converts a Hugging Face OLMo 3 instruct checkpoint to OLMo-core format, loads it, and puts
questions to it that are answered far more easily with a tool than without. Success is the model
choosing to emit ``<function_calls>`` and the loop executing it.

Needs a GPU. The 7B is roughly 14 GiB in bfloat16, so a single L40S or A10G is enough.

Usage::

    python src/examples/tools/olmo3_tool_calling.py --work-dir /tmp/olmo3-tools

On the eduLLM platform::

    edullm run --project <project> --compute gpu-1xl40s -- \\
        python src/examples/tools/olmo3_tool_calling.py --work-dir /tmp/olmo3-tools

The converted checkpoint is cached in ``--work-dir``, so a second run skips the conversion.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import torch

from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.generate import GenerationConfig, TransformerGenerationModule
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.tools import (
    ToolCall,
    ToolConfig,
    ToolRegistry,
    ToolResult,
    run_tool_loop,
)
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)

DEFAULT_HF_MODEL = "allenai/Olmo-3-7B-Instruct"
DEFAULT_MODEL_ARCH = "olmo3_7b"

CONVERT_SCRIPT = (
    Path(__file__).resolve().parents[1] / "huggingface" / "convert_checkpoint_from_hf.py"
)

# Questions that are tedious without a tool and trivial with one. Arithmetic the model is likely
# to get wrong from memory, and an exact root it cannot produce as a decimal.
PROMPTS = {
    "calculator": "What is 48273 * 1092? Use your tools to be certain.",
    "symbolic_math": "Solve x**2 - 5*x + 6 = 0. Use your tools.",
    "web_search": "Search the web: what is the Allen Institute for AI's OLMo project?",
}


def convert_checkpoint(hf_model: str, model_arch: str, output_dir: Path, validate: bool):
    """
    Convert a Hugging Face checkpoint to OLMo-core format, unless it is already there.

    :param hf_model: The HF Hub model id.
    :param model_arch: The OLMo-core architecture name.
    :param output_dir: Where the converted checkpoint goes.
    :param validate: Whether to check the converted model against the original. This holds both
        models in memory at once, so it roughly doubles the requirement.
    """
    if (output_dir / "config.json").exists():
        log.info("Reusing the converted checkpoint at %s", output_dir)
        return

    # The converter resolves the architecture from a config file, and reaches it with
    # cached_path(f"{input}/config.json"), which a bare Hub repo id is not a valid path for.
    # Fetching it first and naming it outright is what makes a repo id usable here.
    from huggingface_hub import hf_hub_download

    hf_config = hf_hub_download(repo_id=hf_model, filename="config.json")

    command = [
        sys.executable,
        str(CONVERT_SCRIPT),
        "--checkpoint-input-path",
        hf_model,
        "--config-path",
        hf_config,
        "--output-dir",
        str(output_dir),
        "--model-arch",
        model_arch,
        "--tokenizer",
        "dolma2",
    ]
    if not validate:
        command.append("--skip-validation")

    log.info("Converting %s -> %s", hf_model, output_dir)
    subprocess.run(command, check=True)


class HFGenerationAdapter:
    """
    A Hugging Face model behind the slice of the generation-module interface the tool loop uses.

    Converting a checkpoint to OLMo-core format materializes the whole model in system memory,
    which needs far more RAM than a single-GPU box tends to have. This runs the real weights
    straight from Hugging Face instead, so the model's behaviour and the parser can be exercised
    on a machine that could not host the conversion.

    :param model: A loaded causal LM.
    :param pad_token_id: The padding token.
    :param eos_token_id: The token that ends a turn.
    :param max_new_tokens: The generation limit per round.
    """

    def __init__(self, model, pad_token_id: int, eos_token_id: int, max_new_tokens: int):
        self.model = model
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens

    def generate_batch(
        self,
        input_ids: torch.Tensor,
        *,
        completions_only: bool = False,
        stop_token_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        del kwargs
        stops = [self.eos_token_id, *(stop_token_ids or [])]
        generated = self.model.generate(
            input_ids.to(self.model.device),
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            eos_token_id=stops,
            pad_token_id=self.pad_token_id,
        )
        if completions_only:
            generated = generated[:, input_ids.shape[1] :]
        return generated, None, None


def build_hf_module(hf_model: str, device: torch.device, max_new_tokens: int, tokenizer):
    """
    Load the Hugging Face weights for generation.

    :param hf_model: The HF Hub model id.
    :param device: The device to run on.
    :param max_new_tokens: The generation limit per round.
    :param tokenizer: The matching tokenizer.

    :returns: An adapter the tool loop can drive.
    """
    from transformers import AutoModelForCausalLM

    log.info("Loading %s in bfloat16 onto %s", hf_model, device)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model, dtype=torch.bfloat16, device_map=str(device)
    )
    model.eval()
    return HFGenerationAdapter(
        model=model,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 100277,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )


def build_generation_module(
    checkpoint_dir: Path, device: torch.device, max_new_tokens: int, stop_token_ids: List[int]
) -> TransformerGenerationModule:
    """
    Load the converted checkpoint for generation.

    :param checkpoint_dir: The converted checkpoint.
    :param device: The device to run on.
    :param max_new_tokens: The generation limit per round.
    :param stop_token_ids: Extra tokens that end a turn.

    :returns: The generation module.
    """
    from olmo_core.nn.attention.flash_attn_api import has_flash_attn_2

    tokenizer_config = TokenizerConfig.dolma2()
    on_gpu = device.type == "cuda"
    # The checkpoint's own config asks for flash attention, which is not on every machine.
    # Only the flash backends support the KV cache, so falling back costs the cache too.
    use_flash = on_gpu and has_flash_attn_2()

    generation_config = GenerationConfig(
        pad_token_id=tokenizer_config.pad_token_id,
        eos_token_id=tokenizer_config.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        use_cache=use_flash,
        stop_token_ids=stop_token_ids or None,
    )

    return TransformerGenerationModule.from_checkpoint(
        checkpoint_dir=str(checkpoint_dir),
        generation_config=generation_config,
        device=device,
        dtype=DType.bfloat16 if on_gpu else DType.float32,
        attention_backend=None if use_flash else AttentionBackendName.torch,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL, help="HF Hub model id.")
    parser.add_argument("--model-arch", default=DEFAULT_MODEL_ARCH, help="OLMo-core arch name.")
    parser.add_argument(
        "--work-dir", type=Path, required=True, help="Where the converted checkpoint is cached."
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        default=["calculator", "symbolic_math"],
        choices=sorted(ToolConfig.get_registered_names()),
        help="Tools to offer the model. Web search is off by default since it needs network.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-tool-iterations", type=int, default=4)
    parser.add_argument(
        "--validate-conversion",
        action="store_true",
        help="Check the converted weights against the original. Roughly doubles memory use.",
    )
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available.")
    parser.add_argument(
        "--backend",
        choices=["olmo-core", "hf"],
        default="olmo-core",
        help=(
            "Which stack to generate with. 'olmo-core' converts the checkpoint first, which "
            "needs roughly 30 GiB of system memory. 'hf' runs the Hugging Face weights "
            "directly and fits on a single-GPU box."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        log.warning("Running on %s. A 7B model will be extremely slow off a GPU.", device)

    if args.backend == "olmo-core":
        checkpoint_dir = args.work_dir / "olmo-core-checkpoint"
        convert_checkpoint(args.hf_model, args.model_arch, checkpoint_dir, args.validate_conversion)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    registry = ToolRegistry.from_configs(
        [ToolConfig.from_dict({"type": name}) for name in args.tools]
    )
    log.info("Tools available: %s", ", ".join(registry.names))

    # run_tool_loop resolves both the tool-call and turn-end markers itself and passes them on
    # every call, so nothing has to be seeded here.
    stop_token_ids: List[int] = []

    if args.backend == "hf":
        generation_module = build_hf_module(args.hf_model, device, args.max_new_tokens, tokenizer)
    else:
        generation_module = build_generation_module(
            checkpoint_dir, device, args.max_new_tokens, stop_token_ids
        )

    failures = []
    for name in args.tools:
        prompt = PROMPTS[name]
        print(f"\n{'=' * 78}\n{name}: {prompt}\n{'=' * 78}")

        def on_tool_call(call: ToolCall, result: ToolResult):
            rendered = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
            print(f"  -> {call.name}({rendered})")
            print(f"  <- {result.content if result.ok else 'ERROR: ' + str(result.error)}")

        outcome = run_tool_loop(
            generation_module,
            tokenizer,
            [{"role": "user", "content": prompt}],
            registry,
            max_iterations=args.max_tool_iterations,
            on_tool_call=on_tool_call,
            log_timing=False,
        )

        print(f"\n  final: {outcome.content.strip()}")
        called = [call.name for call in outcome.calls]
        if not called:
            failures.append(f"{name}: the model never called a tool")
        elif name not in called:
            failures.append(f"{name}: expected '{name}', the model called {called}")

    print(f"\n{'=' * 78}")
    if failures:
        for failure in failures:
            print(f"FAILED  {failure}")
        sys.exit(1)
    print(f"PASSED  the model called every tool it was offered: {', '.join(args.tools)}")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
