"""New token-matched 7B revisions through the unchanged native baseline evaluator."""

import json
import subprocess
import sys
from pathlib import Path

HELPER = "0692817d23a79a85ff464d0b06fdbef20864a6b1"
MODELS = {
    "olmo3-step864000": dict(
        repo="allenai/Olmo-3-1025-7B",
        sha="55d88adc5b31d1911840185ffe1af41ad784eab7",
        architecture="Olmo3ForCausalLM",
        step=864000,
    ),
    "olmo3-step960000": dict(
        repo="allenai/Olmo-3-1025-7B",
        sha="e38d207df703392bef8bf698f12aaa1b842f8a41",
        architecture="Olmo3ForCausalLM",
        step=960000,
    ),
    "hybrid-step864000": dict(
        repo="allenai/Olmo-Hybrid-7B",
        sha="541c329ff79e67eb2f4e0acf4b5326bf1d4607ce",
        architecture="OlmoHybridForCausalLM",
        step=864000,
    ),
    "hybrid-step960000": dict(
        repo="allenai/Olmo-Hybrid-7B",
        sha="bd5eda1fba605fe56485bf80a76b34a94e136771",
        architecture="OlmoHybridForCausalLM",
        step=960000,
    ),
}
TEMPLATES = {
    "gen_mc": "01M2F7CKENMPA1SXTT5B9YWE3Q",
    "math": "01M2F7CKW9234YK91TFVFFD3J0",
    "code": "01M2F7CM9DAFEJSE7CASHGJG1B",
}


def spec(b, key, bundle, commit):
    from olmoe3_lr_sweep_watch import replace_env

    s = b.experiment.get_spec(b.workload.get(TEMPLATES[bundle])).to_json()
    t = s["tasks"][0]
    count = 4 if bundle == "gen_mc" else 8
    old = f"python ladders/olmoe3/workloads/native_7b_baseline_eval.py olmo3-step716000 {bundle} --instances {count}"
    cmd = t["arguments"][0]
    assert cmd.count(old) == 1 and cmd.count(HELPER) == 2
    wrapper = "/tmp/hero-4t-baseline-wrapper"
    fetch = (
        f"git init --quiet {wrapper}\n"
        f"git -C {wrapper} remote add origin https://github.com/allenai/OLMo-core.git\n"
        f"git -C {wrapper} fetch --quiet --depth=1 origin {commit}\n"
        f"git -C {wrapper} checkout --quiet {commit}\n"
    )
    t["arguments"] = [
        cmd.replace(
            old,
            fetch
            + f"python {wrapper}/src/examples/olmo_ddp/olmoe3_hero_4t_baselines.py {key} {bundle} --instances {count}",
        )
    ]
    t["context"].update(priority="urgent", minRuntime="6h", autoResume=True)
    t["timeout"] = "24h"
    replace_env(t, {"GIT_REF": commit, "HF_XET_HIGH_PERFORMANCE": "1"})
    s["description"] = json.dumps(
        dict(
            key=key,
            bundle=bundle,
            stage="pretraining only",
            helper=HELPER,
            core_wrapper=commit,
            profile="native-vllm-BF16",
            numerical_parity="not applicable; native HF weights",
        )
    )
    return s


def main():
    source = Path("/tmp/hero-ladder")
    assert (
        subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        == HELPER
    )
    sys.path.insert(0, str(source / "ladders/olmoe3/workloads"))
    import native_7b_baseline_eval as native

    native.MODELS.update(MODELS)
    # Its subprocesses come through this adapter too, so they see the exact same
    # immutable model registry. Kernels, RoPE repair and eval recipes do not change.
    native.__file__ = __file__
    if len(sys.argv) == 3 and sys.argv[1] == "--execute":
        receipt = json.loads(Path(sys.argv[2]).read_text())
        if receipt["source"]["architecture"] == "Olmo3ForCausalLM":
            from native_olmo3_config import verify_stock_loader

            assert all(
                receipt["rope_compatibility"][k] == v for k, v in verify_stock_loader().items()
            )
        native.execute(sys.argv[2])
    elif len(sys.argv) == 5 and sys.argv[1] == "--smoke":
        native.smoke(*sys.argv[2:])
    else:
        native.main()


if __name__ == "__main__":
    main()
