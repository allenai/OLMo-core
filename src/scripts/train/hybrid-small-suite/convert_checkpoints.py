import subprocess
import sys
from pathlib import Path

CONVERT_SCRIPT = str(
    Path(__file__).resolve().parents[4].parent
    / "transformers"
    / "src"
    / "transformers"
    / "models"
    / "olmo_hybrid_small"
    / "convert_olmo_hybrid_small_weights_to_hf.py"
)

# Pretraining Checkpoints
pretraining_checkpoints = {
    "275m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-275M-Cx100/step161186/",
    "810m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-810M-Cx100/step269926/",
    "1.4b": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-1.4B-Cx100/step308433/",
}

midtraining_checkpoints = {
    "275m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-midtraining-275M-v2-lr1.6e-3/step38147/",
    "810m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-midtraining-v2-810M-lr4e-4/step23842/",
    "1.4b": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-midtraining-v2-1.4b-lr4e-4/step11921/",
}

long_context_checlkpoints = {
    "275m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-long-context-v2-275m/step47684/",
    "810m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-long-context-v2-810m/step23842/",
    "1.4b": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-long-context-v2-1.4b/step23842/",
}

sft_checkpoints = {
    "275m-lr1e-4": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-sft-think-275M-lr1e-4/step23206/",
    "275m-lr2e-4": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-sft-think-275M-lr2e-4/step23206/",
    "275m-lr4e-4": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-sft-think-275M-lr4e-4/step23206/",
    "275m-lr8e-4": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-sft-think-275M-lr8e-4/step23206/",
}

all_checkpoints = {
    "pretraining": pretraining_checkpoints,
    "midtraining": midtraining_checkpoints,
    "long_context": long_context_checlkpoints,
    "sft": sft_checkpoints,
}


def get_output_path(input_path: str) -> str:
    p = Path(input_path.rstrip("/"))
    return str(p.parent / (p.name + "-hf"))


def convert_all():
    failed = []

    for stage, checkpoints in all_checkpoints.items():
        for size, input_path in checkpoints.items():
            output_path = get_output_path(input_path)
            if Path(output_path).exists():
                print(f"\n=== Skipping {stage}/{size}: {output_path} already exists ===")
                continue
            print(f"\n=== Converting {stage}/{size}: {input_path} -> {output_path} ===")
            result = subprocess.run(
                [
                    sys.executable,
                    CONVERT_SCRIPT,
                    "--input_dir", input_path,
                    "--output_dir", output_path,
                    "--dtype", "bfloat16",
                ],
                env={**__import__("os").environ, "TRUST_REMOTE_CODE": "True"},
            )
            if result.returncode != 0:
                print(f"FAILED: {stage}/{size} (exit code {result.returncode})")
                failed.append((stage, size, input_path))
            else:
                print(f"OK: {stage}/{size}")

    if failed:
        print("\nThe following conversions FAILED:")
        for stage, size, path in failed:
            print(f"  {stage}/{size}: {path}")
        sys.exit(1)
    else:
        print("\nAll conversions completed successfully.")


if __name__ == "__main__":
    convert_all()