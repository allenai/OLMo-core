#!/usr/bin/env python3
"""
Export the entropy model to ONNX format.

Usage:
    python convert_entropy_model_to_onnx.py [--checkpoint-path PATH] [--output-path PATH]
"""

import argparse
import torch
from pathlib import Path

from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig, Transformer
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state


class TransformerWrapper(torch.nn.Module):
    """Wrapper to ensure only input_ids is passed to the model."""

    def __init__(self, model: Transformer):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model.forward(input_ids)


def export_entropy_model_to_onnx(
    checkpoint_path: str,
    output_path: str,
    sequence_length: int = 2048,
    quantize: bool = True,
):
    """
    Load the entropy model and export it to ONNX format.

    Args:
        checkpoint_path: Path to the entropy model checkpoint
        output_path: Path where to save the ONNX model
        sequence_length: Maximum sequence length for the model
        quantize: Whether to apply dynamic quantization (INT8) after export
    """
    print(f"Loading entropy model from {checkpoint_path}")

    # Build the model configuration (same as in train_stage2.py)
    entropy_model_config = TransformerConfig.olmo2_190M(
        vocab_size=TokenizerConfig.dolma2().padded_vocab_size(),
        dtype=DType.float32,
    )

    # Build the model on CPU
    entropy_model = entropy_model_config.build(init_device="cpu")

    # Load checkpoint
    load_model_and_optim_state(
        checkpoint_path,
        entropy_model,
    )

    # Set to eval mode
    entropy_model.eval()

    # Wrap
    entropy_model = TransformerWrapper(entropy_model)

    # Create dummy input with the expected shape
    # Input is token IDs with shape (batch_size, sequence_length)
    dummy_input = torch.randint(
        0,
        TokenizerConfig.dolma2().padded_vocab_size(),
        (1, sequence_length),
        dtype=torch.long
    )

    print(f"Exporting model to {output_path}")
    print(f"Input shape: {dummy_input.shape}")

    # Determine output path for base model
    output_path_base = output_path
    if quantize:
        # Save unquantized model with .fp32 suffix
        output_path_base = str(Path(output_path).with_suffix('')) + '.fp32.onnx'

    # Export to ONNX
    # Pass input as tuple to ensure only input_ids is passed as argument
    with torch.no_grad():
        torch.onnx.export(
            entropy_model,
            (dummy_input,),
            output_path_base,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        )

    print(f"Successfully exported entropy model to {output_path_base}")

    # Apply dynamic quantization if requested
    if quantize:
        print(f"Applying dynamic INT8 quantization...")
        quantize_dynamic(
            output_path_base,
            output_path,
            weight_type=QuantType.QInt8
        )
        print(f"Quantized model saved to {output_path}")

        # Optionally remove the FP32 model
        import os
        print(f"Removing unquantized model {output_path_base}")
        os.remove(output_path_base)


def main():
    parser = argparse.ArgumentParser(description="Export entropy model to ONNX")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/weka/oe-training-default/ai2-llm/checkpoints/dirkg/ladder/checkpoints/baseline-titan-190M-5xC/step36308/model_and_optim",
        help="Path to the entropy model checkpoint"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="entropy_model.onnx",
        help="Path where to save the ONNX model"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=4096,
        help="Maximum sequence length for the model"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable INT8 quantization (export FP32 model only)"
    )

    args = parser.parse_args()

    export_entropy_model_to_onnx(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        sequence_length=args.sequence_length,
        quantize=not args.no_quantize,
    )


if __name__ == "__main__":
    main()
