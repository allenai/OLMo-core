# OLMo-Hybrid-7B Checkpoints

## Checkpoint URL Pattern

```
https://olmo-checkpoints.org/ai2-llm/Olmo-Hybrid-7B/{stage}/step{N}/
```

## Available Checkpoints

### Stage 3: Long-Context Training

| Step | URL |
|------|-----|
| 23842 | `https://olmo-checkpoints.org/ai2-llm/Olmo-Hybrid-7B/stage3/step23842/` |

**Run name**: `OLMo3.1-7B-6T-30h-long-context-drope`

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Hybrid (GDN + Attention), 3:1 ratio |
| d_model | 3840 |
| n_layers | 32 |
| vocab_size | 100352 |
| Block pattern | `["gdn", "gdn", "gdn", "attn"]` (repeating) |
| Tokenizer | `allenai/dolma2-tokenizer` (vocab_size=100278) |
| Source repo | `allenai/OLMo-core` @ `c005e20e` |

## Checkpoint Directory Manifest

```
step23842/
├── config.json                          (~12 KB)
└── model_and_optim/
    ├── .metadata                        (~1.1 MB)
    ├── __0_0.distcp                     (~5.19 GB)
    ├── __0_1.distcp                     (~5.19 GB)
    ├── __0_2.distcp                     (~5.19 GB)
    ├── __0_3.distcp                     (~5.19 GB)
    ├── __0_4.distcp                     (~5.19 GB)
    ├── __0_5.distcp                     (~5.19 GB)
    ├── __0_6.distcp                     (~5.19 GB)
    ├── __0_7.distcp                     (~5.19 GB)
    ├── __0_8.distcp                     (~5.19 GB)
    ├── __0_9.distcp                     (~5.19 GB)
    ├── __0_10.distcp                    (~5.21 GB)
    ├── __0_11.distcp                    (~5.21 GB)
    ├── __0_12.distcp                    (~5.19 GB)
    ├── __0_13.distcp                    (~5.19 GB)
    ├── __0_14.distcp                    (~5.20 GB)
    └── __0_15.distcp                    (~5.20 GB)

Total: ~83.1 GB (16 shard files + metadata)
State dict entries: 4707 (model weights + optimizer state)
```

## Downloading with curl

```bash
BASE_URL="https://olmo-checkpoints.org/ai2-llm/Olmo-Hybrid-7B/stage3/step23842"
DEST="/path/to/step23842"

mkdir -p "$DEST/model_and_optim"

# Config
curl -L -o "$DEST/config.json" "$BASE_URL/config.json"

# Metadata
curl -L -o "$DEST/model_and_optim/.metadata" "$BASE_URL/model_and_optim/.metadata"

# Shard files (16 shards, ~83 GB total)
for i in $(seq 0 15); do
    curl -L -o "$DEST/model_and_optim/__0_${i}.distcp" \
        "$BASE_URL/model_and_optim/__0_${i}.distcp" &
done
wait
```

## Loading with OLMo-core (streamed, no local download needed)

```python
import json
from cached_path import cached_path
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.transformer.config import TransformerConfig

CHECKPOINT_URL = "https://olmo-checkpoints.org/ai2-llm/Olmo-Hybrid-7B/stage3/step23842"

# Load config
with open(cached_path(f"{CHECKPOINT_URL}/config.json")) as f:
    config_dict = json.load(f)

# Build model
model_config = TransformerConfig.from_dict(config_dict["model"])
model = model_config.build(init_device="meta")
model.to_empty(device="cuda")

# Load weights (downloads shards to cache automatically)
load_model_and_optim_state(f"{CHECKPOINT_URL}/model_and_optim", model)
```

## Converting to HuggingFace Format

```bash
python src/examples/huggingface/convert_checkpoint_to_hf.py \
    -i /path/to/step23842 \
    -o /path/to/hf-output \
    --device cuda
```

The converter auto-detects hybrid models and saves as `model_type: olmo_hybrid` with
`architectures: ["OlmoHybridForCausalLM"]`.
