# Converting old GDN hybrid checkpoints to HuggingFace

Two-step process: old OLMo-core format → new OLMo-core format → HuggingFace.

## Setup

```bash
# Install OLMo-core with all extras
uv sync --all-extras

# The transformers extra must point to the olmo_hybrid branch until it lands upstream.
# This is already configured in pyproject.toml:
#   transformers @ git+https://github.com/tyler-romero/transformers.git@tyler/olmo-hybrid-autoconfig
#
# If not, install manually:
#   uv pip install git+https://github.com/tyler-romero/transformers.git@tyler/olmo-hybrid-autoconfig

# click is needed for the first script
uv pip install click
```

## Step 1: Convert old OLMo-core → new OLMo-core

Renames parameter keys (e.g. `blocks.{i}.fla.inner.q_proj` → `blocks.{i}.attention.w_q`)
and converts the config from the single `fla_hybrid` block format to the
`block_pattern` + named block dict format.

```bash
uv run python src/scripts/convert_hacky_gdn_checkpoint.py \
  --input /path/to/old/stepN \
  --output /path/to/old/stepN-converted
```

## Step 2: Convert new OLMo-core → HuggingFace

Produces `config.json` + `model.safetensors` with `model_type: olmo_hybrid`.
Includes a validation pass that checks OLMo-core and HF model outputs match.

GDN layers use Triton kernels, so validation requires a GPU (`--device cuda`).

```bash
uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
  -i /path/to/old/stepN-converted \
  -o /path/to/old/stepN-hf \
  --device cuda
```

Use `--skip-validation` to convert without a GPU (weights are still saved, just not verified).

## Example

```bash
uv run python src/scripts/convert_hacky_gdn_checkpoint.py \
  --input /weka/.../OLMo3.1-7B-6T-30h-long-context-drope/step23842 \
  --output /weka/.../OLMo3.1-7B-6T-30h-long-context-drope/step23842-converted

uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
  -i /weka/.../OLMo3.1-7B-6T-30h-long-context-drope/step23842-converted \
  -o /weka/.../OLMo3.1-7B-6T-30h-long-context-drope/step23842-hf \
  --device cuda
```
