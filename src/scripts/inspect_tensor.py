#!/usr/bin/env python3
"""
Simple script to load and inspect safetensors files.
Shows tensor names, shapes, dtypes, and other metadata.
Can use some work for more detailed inspection!
"""

import sys
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors library not installed.")
    print("Install it with: pip install safetensors")
    sys.exit(1)


def inspect_safetensors(file_path, show_values=False):
    """Load and inspect a safetensors file."""

    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"Inspecting: {file_path}")
    print(f"File size: {file_path.stat().st_size / (1024**2):.2f} MB")
    print("=" * 80)

    try:
        # Open the safetensors file
        with safe_open(file_path, framework="pt" if show_values else "numpy") as f:
            # Get metadata if available
            metadata = f.metadata()
            if metadata:
                print("\n📋 Metadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
                print()

            # Get all tensor keys
            keys = f.keys()
            print(f"\n🔢 Total tensors: {len(keys)}\n")

            # Display info for each tensor
            print("📊 Tensor Information:")
            print("-" * 80)

            total_params = 0
            for key in keys:
                if show_values:
                    # Load actual tensor for pytorch
                    tensor = f.get_tensor(key)
                    shape = tuple(tensor.shape)
                    dtype = str(tensor.dtype)

                    # Calculate number of parameters
                    num_params = tensor.numel()
                    total_params += num_params

                    print(f"Name:       {key}")
                    print(f"Shape:      {shape}")
                    print(f"Dtype:      {dtype}")
                    print(f"Params:     {num_params:,}")

                    # Show first and last few values
                    flat = tensor.flatten()
                    print(f"First 10:   {flat[:10].tolist()}")
                    print(f"Last 10:    {flat[-10:].tolist()}")
                    print(f"Min:        {tensor.min().item():.6e}")
                    print(f"Max:        {tensor.max().item():.6e}")
                    print(f"Mean:       {tensor.mean().item():.6e}")
                    print(f"Std:        {tensor.std().item():.6e}")
                else:
                    tensor = f.get_slice(key)
                    shape = tensor.get_shape()
                    dtype = tensor.get_dtype()

                    # Calculate number of parameters
                    num_params = 1
                    for dim in shape:
                        num_params *= dim
                    total_params += num_params

                    print(f"Name:   {key}")
                    print(f"Shape:  {shape}")
                    print(f"Dtype:  {dtype}")
                    print(f"Params: {num_params:,}")

                print("-" * 80)

            print(f"\n✅ Total parameters: {total_params:,}")
            print(f"   ({total_params / 1e6:.2f}M parameters)")

    except Exception as e:
        print(f"Error loading file: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_tensor.py <path_to_safetensors_file> [--values]")
        print("\nExample:")
        print("  python inspect_tensor.py model.safetensors")
        print("  python inspect_tensor.py model.safetensors --values")
        print("\nOptions:")
        print("  --values    Show actual tensor values (first/last 10, min/max/mean/std)")
        sys.exit(1)

    file_path = sys.argv[1]
    show_values = "--values" in sys.argv
    inspect_safetensors(file_path, show_values=show_values)


if __name__ == "__main__":
    main()
