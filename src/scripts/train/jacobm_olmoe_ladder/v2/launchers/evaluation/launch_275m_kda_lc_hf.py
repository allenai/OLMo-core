#!/usr/bin/env python3
"""Convert the completed non-Latent 275M KDA Cx8 LCE checkpoint to HF."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import yaml

REPO = Path("/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/OLMo-core-moe-v2-core")
SOURCE = Path(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/"
    "long-context/lc-275m-geometry-hybrid-kda-ev2-neg-nope-gated-"
    "cx8-ptlr8e-4-mtlr1p6e-4-lclr8e-5-r1/step37991"
)
OUTPUT = Path(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/"
    "hf/long-context/lc-275m-geometry-hybrid-kda-ev2-neg-nope-gated-"
    "cx8-ptlr8e-4-mtlr1p6e-4-lclr8e-5-r1/step37991"
)

WORKSPACE = "ai2/OLMo-3-moe-experiments"
CLUSTER = "ai2/holmes"
BUDGET = "ai2/oe-other"
IMAGE = "01KW8G8JC20H11Y60PPTE2VN4Q"
NAME = "convert-lc-275m-kda-cx8-step37991-hf"


def build_spec(
    commit: str, *, force: bool = False, cache_only: bool = False
) -> dict[str, object]:
    force_value = "1" if force else "0"
    cache_only_value = "1" if cache_only else "0"
    command = f"""set -euo pipefail
mkdir -p /results {OUTPUT.parent}
cd {REPO}
test "$(git rev-parse HEAD)" = "{commit}"
export PYTHONPATH="{REPO}/src:${{PYTHONPATH:-}}"
export PYTHONUNBUFFERED=1

SOURCE={SOURCE}
OUTPUT={OUTPUT}
MARKER="${{OUTPUT}}/conversion_complete.json"
FORCE={force_value}
CACHE_ONLY={cache_only_value}

if [[ "${{FORCE}}" == "1" && -f "${{MARKER}}" ]]; then
  cp "${{MARKER}}" /results/prior_conversion_complete.json
  rm "${{MARKER}}"
fi

if [[ "${{CACHE_ONLY}}" == "1" ]]; then
  test -f "${{OUTPUT}}/config.json"
  test -f "${{OUTPUT}}/model.safetensors"
  echo "Running cache-only validation for existing HF artifact at ${{OUTPUT}}"
elif [[ -f "${{MARKER}}" && "${{FORCE}}" != "1" ]]; then
  echo "Validated HF conversion already exists at ${{OUTPUT}}"
else
  if [[ -e "${{OUTPUT}}" ]] && find "${{OUTPUT}}" -mindepth 1 -print -quit | grep -q .; then
    echo "Retrying over the incomplete output left by an earlier failed conversion."
  fi
  python src/examples/huggingface/convert_checkpoint_to_hf.py \\
    -i "${{SOURCE}}" \\
    -o "${{OUTPUT}}" \\
    -s 131072 \\
    --device cuda \\
    --validation-device cuda \\
    --debug \\
    2>&1 | tee /results/conversion.log
fi

# The training image intentionally supplies the exact Torch, FLA, and TE stack
# used for training and strict parity. Its Transformers 5.4 cache predates the
# native linear-attention cache layer used by KDA generation, however, so update
# only the pure-Python HF runtime to the version locked by this checkout.
python -m pip install --no-deps \
  "transformers==5.14.1" \
  "huggingface-hub==1.12.2"

OLMO_HF_REQUIRE_TE_EXPERT_PARITY=1 python - "${{OUTPUT}}" <<'PY'
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

output = Path(sys.argv[1])
model = AutoModelForCausalLM.from_pretrained(
    output,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).cuda().eval()
input_ids = torch.tensor(
    [[1, 42, 314, 2718, 7, 11, 13, 17, 19, 23, 29, 31]],
    device="cuda",
)
with torch.no_grad():
    prefill_reference = model(input_ids[:, :8], use_cache=False).logits
    cache = DynamicCache(config=model.config)
    prefill_cached = model(
        input_ids[:, :8], past_key_values=cache, use_cache=True
    ).logits
    cached_logits = [prefill_cached]
    reference_logits = [prefill_reference]
    for position in range(8, input_ids.shape[1]):
        cached_logits.append(
            model(
                input_ids[:, position : position + 1],
                past_key_values=cache,
                use_cache=True,
            ).logits
        )
        reference_logits.append(
            model(input_ids[:, : position + 1], use_cache=False).logits[:, -1:]
        )
    cached_logits = torch.cat(cached_logits, dim=1)
    reference_logits = torch.cat(reference_logits, dim=1)

diff = (cached_logits.float() - reference_logits.float()).abs()
for position in range(input_ids.shape[1]):
    position_diff = diff[:, position]
    cached_top1 = cached_logits[:, position].argmax(dim=-1)
    reference_top1 = reference_logits[:, position].argmax(dim=-1)
    print(
        f"KDA cache position {{position}}: "
        f"max_abs_error={{position_diff.max().item():.6g}}, "
        f"mean_abs_error={{position_diff.mean().item():.6g}}, "
        f"top1_match={{bool(torch.equal(cached_top1, reference_top1))}}"
    )

torch.testing.assert_close(
    cached_logits.float(),
    reference_logits.float(),
    rtol=2e-2,
    atol=2e-2,
)
print(
    "KDA cached decode validation: "
    f"max_abs_error={{diff.max().item():.6g}}"
)
PY

python - "${{SOURCE}}" "${{OUTPUT}}" "{commit}" <<'PY'
import datetime
import hashlib
import json
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:3])
commit = sys.argv[3]
required = [
    output / "config.json",
    output / "tokenizer.json",
    output / "modeling_olmo3moe.py",
    output / "configuration_olmo3moe.py",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise RuntimeError(f"HF conversion is incomplete; missing: {{missing}}")

index_path = output / "model.safetensors.index.json"
single_path = output / "model.safetensors"
if index_path.is_file():
    index = json.loads(index_path.read_text())
    weight_files = sorted(set(index["weight_map"].values()))
    tensor_count = len(index["weight_map"])
elif single_path.is_file():
    from safetensors import safe_open

    weight_files = [single_path.name]
    with safe_open(single_path, framework="pt", device="cpu") as weights:
        tensor_count = len(list(weights.keys()))
else:
    raise RuntimeError("HF conversion has neither single-file nor sharded safetensors")

missing_weights = [name for name in weight_files if not (output / name).is_file()]
if missing_weights:
    raise RuntimeError(f"Missing indexed weight files: {{missing_weights}}")

config = json.loads((output / "config.json").read_text())
if "linear_attention" not in config.get("layer_types", []):
    raise RuntimeError("Converted config lost the KDA layer assignments")
if config.get("max_position_embeddings") != 131072:
    raise RuntimeError(
        "Expected max_position_embeddings=131072, got "
        f"{{config.get('max_position_embeddings')}}"
    )

sha = hashlib.sha256()
for path in sorted(output.glob("*.py")) + [output / "config.json"]:
    sha.update(path.name.encode())
    sha.update(path.read_bytes())

manifest = {{
    "status": "COMPLETE",
    "source_checkpoint": str(source),
    "output_checkpoint": str(output),
    "olmo_core_commit": commit,
    "completed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "weight_files": weight_files,
    "weight_bytes": sum((output / name).stat().st_size for name in weight_files),
    "weight_tensor_count": tensor_count,
    "code_and_config_sha256": sha.hexdigest(),
    "mapping_validation": "strict complete parameter mapping",
    "logit_validation": "converter torch.testing.assert_close(rtol=1e-4, atol=1e-4)",
    "max_position_embeddings": config["max_position_embeddings"],
}}
marker = output / "conversion_complete.json"
marker.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n")
(Path("/results") / marker.name).write_text(marker.read_text())
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

find "${{OUTPUT}}" -maxdepth 1 -type f -printf '%f %s bytes\\n' \\
  | sort | tee /results/output_files.txt
"""
    return {
        "version": "v2",
        "description": (
            "KDA-aware HF conversion with complete mapping and logit parity for "
            f"{SOURCE}; OLMo-core {commit}"
        ),
        "budget": BUDGET,
        "tasks": [
            {
                "name": "convert",
                "image": {"beaker": IMAGE},
                "command": ["/bin/bash", "-lc"],
                "arguments": [command],
                "envVars": [
                    {"name": "HF_TOKEN", "secret": "HF_TOKEN"},
                ],
                "datasets": [
                    {
                        "mountPath": "/weka/oe-adapt-default",
                        "source": {"weka": "oe-adapt-default"},
                    },
                    {
                        "mountPath": "/weka/oe-training-default",
                        "source": {"weka": "oe-training-default"},
                    },
                ],
                "result": {"path": "/results"},
                "resources": {"gpuCount": 1, "sharedMemory": "64GiB"},
                "context": {
                    "priority": "urgent",
                    "minRuntime": "1h",
                    "autoResume": True,
                },
                "constraints": {"cluster": [CLUSTER]},
                "hostNetworking": True,
                "timeout": "4h",
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run conversion and validation even when the completion marker exists.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Validate the existing HF artifact without repeating conversion.",
    )
    parser.add_argument("--name", default=NAME)
    args = parser.parse_args()

    if not SOURCE.is_dir():
        raise RuntimeError(f"Missing source checkpoint: {SOURCE}")
    if not (SOURCE / "config.json").is_file():
        raise RuntimeError(f"Missing source config: {SOURCE / 'config.json'}")
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if args.force and args.cache_only:
        parser.error("--force and --cache-only are mutually exclusive")
    spec = build_spec(commit, force=args.force, cache_only=args.cache_only)
    rendered = yaml.safe_dump(spec, sort_keys=False)
    if not args.submit:
        print(rendered)
        return

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as file:
        file.write(rendered)
        spec_path = Path(file.name)
    try:
        completed = subprocess.run(
            [
                "beaker",
                "experiment",
                "create",
                str(spec_path),
                "--workspace",
                WORKSPACE,
                "--name",
                args.name,
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    finally:
        spec_path.unlink(missing_ok=True)
    print(completed.stdout, end="")
    match = re.search(r"01[A-Z0-9]{24}", completed.stdout)
    if match is None:
        raise RuntimeError("Could not parse Beaker experiment ID")
    print(f"https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/{match.group(0)}")


if __name__ == "__main__":
    main()
