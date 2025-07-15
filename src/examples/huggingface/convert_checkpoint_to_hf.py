"""
Example script to convert a OLMo Core model checkpoint to a HuggingFace model checkpoint.

Note that this script is architecture-dependent, meaning it may only work for OLMo Core model
architectures that have support in the `transformers` library.
"""

import json
import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Tuple

try:
    import flash_attn  # type: ignore
except ImportError:
    flash_attn = None

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from cached_path import cached_path
from transformers import AutoConfig, AutoModelForCausalLM

from olmo_core.aliases import PathOrStr
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.io import file_exists, join_path
from olmo_core.nn.conversion.state_mapping import TemplatePlaceholder
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.hf.convert import get_converter_to_hf
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.utils import get_default_device, prepare_cli_environment

log = logging.getLogger(__name__)


def load_olmo_model(
    checkpoint_path: PathOrStr, model_config: TransformerConfig, device: torch.device
) -> Transformer:
    """
    Load an Olmo-core model from a checkpoint.

    Args:
        checkpoint_path: Path to the original checkpoint.
        model_config: The OLMo core model config.
        device: The device to place the model on.

    Returns:
        The loaded OLMo core model.
    """
    log.info("Building OLMo-core model...")
    model = model_config.build()
    model.to_empty(device=device)

    with TemporaryDirectory() as work_dir:
        model_and_optim_dir = join_path(checkpoint_path, "model_and_optim")
        log.info(f"Loading checkpoint from '{model_and_optim_dir}'")
        load_model_and_optim_state(
            model_and_optim_dir,
            model,
            work_dir=work_dir,
        )
    log.info("Successfully loaded OLMo-core model.")
    return model


def convert_checkpoint_to_hf(
    model: Transformer,
    output_path: PathOrStr,
    tokenizer_config: TokenizerConfig,
    max_sequence_length: int,
    dtype: Optional[DType] = None,
) -> None:
    """
    Convert an Olmo-core model to HuggingFace format and save it.

    Args:
        model: The loaded OLMo core model.
        output_path: Where to save the converted model.
        tokenizer_config: The OLMo core tokenizer config.
        max_sequence_length: The maximum sequence length that the model supports.
        dtype: The torch dtype that model weights should be saved as.
    """
    log.info(f"Converting and saving checkpoint to '{output_path}'")
    with TemporaryDirectory() as work_dir:
        state_dict_options = dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)

        save_hf_model(
            output_path,
            model_state_dict,
            model,
            dtype=dtype,
            vocab_size=tokenizer_config.vocab_size,
            work_dir=work_dir,
            save_overwrite=True,
        )
    log.info(f"Successfully saved converted model to '{output_path}'")

    # List all files at output_path after writing model checkpoint
    output_path_obj = Path(output_path)
    if output_path_obj.exists():
        log.info(f"Files at output path '{output_path}':")
        for file_path in sorted(output_path_obj.rglob("*")):
            if file_path.is_file():
                log.info(f"  {file_path.relative_to(output_path_obj)}")
    else:
        log.warning(f"Output path '{output_path}' does not exist")

    # Cat the contents of config.json
    config_json_path = Path(output_path) / "config.json"
    if config_json_path.exists():
        log.info(f"Contents of config.json:")
        with open(config_json_path, "r") as f:
            config_contents = f.read()
            log.info(config_contents)
    else:
        log.warning(f"config.json not found at {config_json_path}")

    log.info("Fixing HF config using tokenizer config data and script arguments")
    huggingface_config = AutoConfig.from_pretrained(output_path)
    huggingface_config.max_position_embeddings = max_sequence_length
    huggingface_config.pad_token_id = tokenizer_config.pad_token_id
    huggingface_config.bos_token_id = tokenizer_config.bos_token_id
    huggingface_config.eos_token_id = tokenizer_config.eos_token_id
    huggingface_config.save_pretrained(output_path)
    log.info("Successfully fixed config using tokenizer config data and script arguments")

    generation_config_path = Path(output_path) / "generation_config.json"
    if not generation_config_path.exists():
        raise RuntimeError(f"Generation config not found at {generation_config_path}")
    with open(generation_config_path, "r") as f:
        generation_config = json.load(f)
    generation_config.update(
        {
            "pad_token_id": tokenizer_config.pad_token_id,
            "eos_token_id": tokenizer_config.eos_token_id,
        }
    )
    with open(generation_config_path, "w") as f:
        json.dump(generation_config, f, indent=2)
    log.info(f"Successfully saved generation config to '{generation_config_path}'")


def _register_debug_hooks(hf_model: torch.nn.Module, model: Transformer):
    """Register forward hooks on both models to capture intermediate states for debugging."""
    MAX_DIM_SIZE = 1_000_000

    olmo_core_debug_state: Dict[str, Tuple[int, torch.Tensor]] = {}
    hf_debug_state: Dict[str, Tuple[int, torch.Tensor]] = {}

    def module_hook(
        debug_state: Dict[str, Tuple[int, torch.Tensor]],
        name: str,
        _: torch.nn.Module,
        args,
        output,
    ):
        if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            state_name = f"{name}|input"
            input = args[0].detach()
            for i, size in enumerate(input.shape):
                input = input.narrow(i, 0, min(size, MAX_DIM_SIZE))
            debug_state[state_name] = (len(debug_state), input)
        if isinstance(output, torch.Tensor):
            state_name = f"{name}|output"
            output = output.detach()
            for i, size in enumerate(output.shape):
                output = output.narrow(i, 0, min(size, MAX_DIM_SIZE))
            debug_state[state_name] = (len(debug_state), output)

    for name, module in model.named_modules():
        module.register_forward_hook(partial(module_hook, olmo_core_debug_state, name))
    for name, module in hf_model.named_modules():
        module.register_forward_hook(partial(module_hook, hf_debug_state, name))

    return olmo_core_debug_state, hf_debug_state


def _perform_detailed_state_comparison(
    olmo_core_state: Dict[str, Tuple[int, torch.Tensor]],
    hf_state: Dict[str, Tuple[int, torch.Tensor]],
    state_mapping,
):
    """Perform detailed state comparison for debugging purposes."""
    log.info("Performing detailed state comparison for debugging")
    simple_key_mapping = {
        m.source_keys[0].replace(".weight", ""): m.dest_keys[0].replace(".weight", "")
        for m in state_mapping
        if len(m.source_keys) == 1 and len(m.dest_keys) == 1
    }

    for olmo_name, (_, olmo_tensor) in sorted(olmo_core_state.items(), key=lambda kv: kv[1][0]):
        olmo_key, state_type = olmo_name.split("|")
        if olmo_key not in simple_key_mapping:
            continue

        hf_name = f"{simple_key_mapping[olmo_key]}|{state_type}"
        if hf_name not in hf_state:
            continue

        _, hf_tensor = hf_state[hf_name]

        # Check shape first
        if olmo_tensor.shape != hf_tensor.shape:
            log.info(
                f"Shape mismatch {olmo_name} vs {hf_name} | "
                f"shape {olmo_tensor.shape}->{hf_tensor.shape}"
            )
        # Then check dtype
        elif olmo_tensor.dtype != hf_tensor.dtype:
            log.info(
                f"Dtype mismatch {olmo_name} vs {hf_name} | "
                f"dtype {olmo_tensor.dtype}->{hf_tensor.dtype}"
            )
        # Finally compare values if shape and dtype match
        else:
            common_shape = tuple(min(a, b) for a, b in zip(olmo_tensor.shape, hf_tensor.shape))
            for dim, size in enumerate(common_shape):
                olmo_tensor = olmo_tensor.narrow(dim, 0, size)
                hf_tensor = hf_tensor.narrow(dim, 0, size)
            diff = (olmo_tensor - hf_tensor).float().abs().mean()
            log.info(f"{olmo_name} vs {hf_name} | mean(abs(diff))={diff}")


def validate_conversion(
    hf_path: str | Path,
    olmo_core_path: str | Path,
    olmo_core_model_config: TransformerConfig,
    vocab_size: int,
    batch_size: int = 1,
    sequence_length: int = 60,
    dtype: DType | None = None,
    device: torch.device | None = None,
    debug: bool = False,
):
    """
    Validate the conversion by comparing outputs between the original OLMo Core model and the
    converted HuggingFace model.

    NOTE:  The logic that decides whether to use flash-attention, flex-attention or the SDPA
    implementation lives entirely inside this function.  Callers therefore no longer need to
    (and should not) try to infer or override the attention backend themselves.

    Args:
        hf_path: Path to the converted HuggingFace model.
        olmo_core_path: Path to the original OLMo Core model.
        olmo_core_model_config: The original OLMo Core model config.
        vocab_size: Size of the vocabulary.
        batch_size: The batch size to use for validation.
        sequence_length: The sequence length to use for validation.
        dtype: The torch dtype to use for the original model during validation.
        device: The device to run validation on.
        debug: Whether to enable debug mode with detailed comparison logging.
    """
    log.info(f"Starting validation of converted model at {hf_path}")

    if torch.cuda.is_available():
        torch.cuda.init()

    device = device or get_default_device()
    log.info(f"Using device: {device}")

    # Prepare some random input for the forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length)).to(device)
    log.info(f"Generated random input_ids with shape {input_ids.shape} and vocab size {vocab_size}")

    # ------------------------------------------------------------------ #
    # Work out which attention backend we should use for *validation*.
    #
    # - If flash-attention is configured and available -> keep using it.
    # - If flash-attention is configured but NOT available -> fall back to flex-attention (SDPA MATH backend).
    # - If the model explicitly requires the fused flash-attention kernels but CUDA/flash
    #   is not available, we cannot validate – bail out early with a warning.
    # ------------------------------------------------------------------ #

    # Get attention configuration from the model config
    attention_cfg = olmo_core_model_config.block.attention
    log.info(f"Attention config: name={attention_cfg.name}, use_flash={attention_cfg.use_flash}")

    # Decide which backend to use
    flash_attn_available = (
        flash_attn is not None and device.type == "cuda" and dtype == DType.bfloat16
    )
    log.info(f"Flash attention available: {flash_attn_available}")
    if not flash_attn_available and attention_cfg.name == "fused":
        log.warning(
            "Model requires fused flash-attention kernels but CUDA/flash-attention is not "
            "available – skipping validation."
        )
        return

    if attention_cfg.use_flash:
        if flash_attn_available:
            hf_attn_implementation = "flash_attention_2"
            log.info(
                "Using Flash-attention for validation (HF attention_backend='flash_attention_2')"
            )
        else:
            log.warning(
                "Flash-attention unavailable – falling back to Flex-attention and SDPA-math for validation (HF attention_backend='sdpa')"
            )
            attention_cfg.use_flash = False
            attention_cfg.use_flex_attn = True
            hf_attn_implementation = "sdpa"
            log.info("Using SDPA attention implementation for HF model")

    # Check if model uses sliding window attention
    if attention_cfg.sliding_window is not None:
        # Extract the sliding window size from the pattern
        # DANGER DANGER This assumes that all sliding windows are the same size.
        sliding_window_size = next(
            (size for size in attention_cfg.sliding_window.pattern if size != -1), None
        )
        log.info(f"OLMo Core model sliding window size: {sliding_window_size}")
    log.info("Loading checkpoint of original OLMo Core model for validation...")
    olmo_core_model = load_olmo_model(olmo_core_path, olmo_core_model_config, device)
    log.info("Successfully loaded original OLMo Core model")

    log.info("Loading checkpoint of converted HF model for validation...")
    config = AutoConfig.from_pretrained(hf_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=dtype.as_pt() if dtype else "auto",
        config=config,
        attn_implementation=hf_attn_implementation,
    ).to(device)
    log.info("Successfully loaded HuggingFace model")

    # Register debug hooks
    olmo_core_state, hf_state = {}, {}
    state_mapping = None
    if debug:
        log.info("Registering debug hooks for detailed comparison")
        olmo_core_state, hf_state = _register_debug_hooks(hf_model, olmo_core_model)
        state_converter = get_converter_to_hf()

        if not hasattr(hf_model.config, "num_hidden_layers"):
            raise ValueError(f"Number of hidden layers missing in HF config: {hf_model.config}")
        n_layers: int = hf_model.config.num_hidden_layers
        n_experts: int | None = getattr(hf_model.config, "num_experts", None)

        placeholder_bounds = {TemplatePlaceholder.LAYER: n_layers}
        if n_experts:
            placeholder_bounds[TemplatePlaceholder.EXPERT] = n_experts

        state_mapping = state_converter.get_mappings(
            olmo_core_model.state_dict(), placeholder_bounds
        )
        log.info(f"Generated state mapping with {len(state_mapping)} mappings")

    # ------------------------------------------------------------------ #
    # Dtype Checks
    # ------------------------------------------------------------------ #
    # Check parameter dtypes are consistent
    hf_param_dtypes = {param.dtype for param in hf_model.parameters()}
    olmo_param_dtypes = {param.dtype for param in olmo_core_model.parameters()}
    assert len(hf_param_dtypes) == 1, (
        f"HF model has inconsistent parameter dtypes: {hf_param_dtypes}"
    )
    assert len(olmo_param_dtypes) == 1, (
        f"OLMo Core model has inconsistent parameter dtypes: {olmo_param_dtypes}"
    )
    log.info(f"Model dtypes - HF: {hf_param_dtypes}, OLMo Core: {olmo_param_dtypes}")
    hf_param_dtype = next(iter(hf_param_dtypes))
    olmo_param_dtype = next(iter(olmo_param_dtypes))
    assert hf_param_dtype == olmo_param_dtype, (
        f"Model dtype mismatch: expected {olmo_param_dtype}, got {hf_param_dtype}"
    )
    if dtype is not None:
        assert hf_param_dtype == dtype.as_pt(), (
            f"Expected dtype {dtype.as_pt()}, got {hf_param_dtype}"
        )

    # ------------------------------------------------------------------ #
    # Forward passes
    # ------------------------------------------------------------------ #
    log.info("Running OLMo core and HF models for validation...")
    hf_model.eval()
    olmo_core_model.eval()

    log.info(f"Running HF model with attn_implementation={hf_attn_implementation}")
    # SDPA MATH backend is just Pytorch's C++ attention implementation. It is numerically similar to Flex-attention on CPU.
    with torch.no_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        hf_logits, *_ = hf_model(input_ids=input_ids, return_dict=False)

    log.info(f"HF model output shape: {hf_logits.shape}")
    del hf_model  # free memory

    log.info("Running OLMo Core model")
    with torch.no_grad():
        logits = olmo_core_model(input_ids=input_ids)
    log.info(f"OLMo Core model output shape: {logits.shape}")
    del olmo_core_model  # free memory

    # ------------------------------------------------------------------ #
    # Detailed debugging information (optional)
    # ------------------------------------------------------------------ #
    if debug and state_mapping is not None:
        log.info("Performing detailed state comparison for debugging")
        _perform_detailed_state_comparison(olmo_core_state, hf_state, state_mapping)

    # ------------------------------------------------------------------ #
    # Final assertions – the logits must match and the top-10 token predictions must match.
    # ------------------------------------------------------------------ #
    # Prepare tensors for comparison (cast to float32 and restrict to vocab range)
    hf_slice = hf_logits[..., :vocab_size].float()
    olmo_slice = logits[..., :vocab_size].float()

    log.info("Comparing model outputs...")
    # Check whether the two models would actually generate the same token.
    # Determine the top-k predicted token ids at every (batch, position).
    for k in range(1, 11):
        hf_topk = torch.topk(hf_slice, k=k, dim=-1).indices
        olmo_topk = torch.topk(olmo_slice, k=k, dim=-1).indices
        mismatches_per_seq = (hf_topk != olmo_topk).any(dim=-1)
        num_mismatches = mismatches_per_seq.sum().item()
        total_positions = mismatches_per_seq.numel()
        mismatch_percent = (num_mismatches / total_positions) * 100 if total_positions > 0 else 0.0

        log.info(
            f"Top-{k} token prediction comparison: {num_mismatches}/{total_positions} mismatches ({mismatch_percent:.2f}%)"
        )
        assert num_mismatches == 0, (
            f"{num_mismatches}/{total_positions} positions ({mismatch_percent:.2f}%) have different top-{k} token predictions between HF and OLMo models."
        )

    # Perform a numerical closeness check even if the top-k token predictions match in a greedy generation setting.
    log.info("Performing numerical closeness check...")
    torch.testing.assert_close(hf_slice, olmo_slice, rtol=1e-4, atol=1e-4)
    log.info("Validation completed successfully - models produce identical outputs")


def get_experiment_config(checkpoint_input_dir: PathOrStr) -> dict:
    if not file_exists(f"{checkpoint_input_dir}/config.json"):
        raise RuntimeError(f"Config file not found at {checkpoint_input_dir}")

    with cached_path(f"{checkpoint_input_dir}/config.json").open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    if "model" not in config_dict:
        raise RuntimeError(
            f"Config file at {checkpoint_input_dir} is not an OLMo core experiment config, ignoring"
        )

    return config_dict


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--checkpoint-input-path",
        type=str,
        required=True,
        help="Local or remote directory containing the OLMo Core checkpoint.",
    )

    parser.add_argument(
        "-o",
        "--huggingface-output-dir",
        type=Path,
        required=True,
        help="Local or remote directory where the converted checkpoint should be saved.",
    )
    parser.add_argument(
        "-s",
        "--max-sequence-length",
        type=int,
        required=True,
        help="Max sequence length supported by the model.",
    )
    parser.add_argument(
        "--skip-validation",
        dest="validate",
        action="store_false",
        help="If set, validation to check that the converted model matches the original model is skipped.",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="If set, debug information of validation is output.",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        help="The device on which conversion and validation occurs. Defaults to CUDA or MPS if available and initialized.",
    )
    parser.add_argument(
        "--dtype",
        help="The torch dtype that model weights should be saved as. Defaults to bfloat16 due to https://github.com/allenai/olmo-cookbook/issues/60.",
        type=DType,
        default=DType.bfloat16,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_sequence_length <= 0:
        raise ValueError(f"Missing or invalid sequence length: {args.max_sequence_length}")

    # Load configs
    experiment_config = get_experiment_config(args.checkpoint_input_path)
    if experiment_config is None:
        raise RuntimeError("Experiment config not found, cannot convert to HF checkpoint")

    transformer_config_dict = experiment_config["model"]
    tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer")
    assert transformer_config_dict is not None
    assert tokenizer_config_dict is not None

    # Strip deprecated keys that are irrelevant for inference / conversion
    for k in ("compile", "dp_config", "tp_config", "float8_config"):
        transformer_config_dict.pop(k, None)

    olmo_core_model_config = TransformerConfig.from_dict(transformer_config_dict)
    olmo_core_tokenizer_config = TokenizerConfig.from_dict(tokenizer_config_dict)

    device = args.device or get_default_device()
    olmo_core_model = load_olmo_model(args.checkpoint_input_path, olmo_core_model_config, device)
    convert_checkpoint_to_hf(
        olmo_core_model,
        args.huggingface_output_dir,
        olmo_core_tokenizer_config,
        args.max_sequence_length,
        dtype=args.dtype,
    )

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    if args.validate:
        log.info("Validating converted model")
        validate_conversion(
            hf_path=args.huggingface_output_dir,
            olmo_core_path=args.checkpoint_input_path,
            olmo_core_model_config=olmo_core_model_config,
            vocab_size=olmo_core_tokenizer_config.vocab_size,
            batch_size=1,
            sequence_length=512,
            debug=args.debug,
            dtype=args.dtype,
            device=device,
        )
        log.info("Validation completed successfully")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
