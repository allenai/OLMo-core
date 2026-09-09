"""Isolate existing FLA launch settings on immutable real KDA inputs; no new kernels."""

import importlib
import json

import torch
from fla.ops.kda import chunk_kda
from safetensors import safe_open
from triton.runtime.autotuner import Autotuner

from hero_hf_download import SCRATCH, write_json
from hero_hf_kda_replay import stats


KERNELS = (
    ("fla.modules.l2norm", "l2norm_fwd_kernel"),
    ("fla.ops.kda.gate", "kda_gate_chunk_cumsum_vector_kernel"),
    ("fla.ops.kda.chunk_intra", "chunk_kda_fwd_kernel_intra_sub_chunk"),
    ("fla.ops.kda.chunk_intra", "chunk_kda_fwd_kernel_inter_solve_fused"),
    ("fla.ops.common.chunk_delta_h", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"),
    ("fla.ops.gla.chunk", "chunk_gla_fwd_kernel_o"),
)


def main():
    """Replay one setting at a time, retaining normal autotuning for other stages."""
    root = SCRATCH / "emo/step6000"
    debug = root / "layer-debug-random257-flash3"
    captures = torch.load(debug / "hf-layer-debug-r13.pt", weights_only=True, map_location="cpu")[
        "captures"
    ]
    native = torch.load(debug / "vllm-layer-debug-r13.pt", weights_only=True, map_location="cpu")
    config = json.loads((root / "hf.partial/config.json").read_text())
    prefix = "model.layers.2.self_attn"
    h, k, v = (
        config["linear_num_key_heads"],
        config["linear_key_head_dim"],
        config["linear_value_head_dim"],
    )
    inputs = {
        name: captures[f"{prefix}.{label}_conv1d/out0"].reshape(1, -1, h, dim).cuda()
        for name, label, dim in (("q", "q", k), ("k", "k", k), ("v", "v", v))
    }
    inputs["g"] = captures[prefix + ".f_proj_2/out0"].reshape(1, -1, h, k).cuda()
    inputs["beta"] = captures[prefix + ".beta_proj/out0"].unsqueeze(0).cuda().float().sigmoid()
    if config["linear_allow_neg_eigval"]:
        inputs["beta"] *= 2
    with safe_open(root / "hf.partial/model.safetensors", framework="pt", device="cpu") as handle:
        for name in ("A_log", "dt_bias"):
            inputs[name] = handle.get_tensor(prefix + "." + name).float().cuda()
    expected = captures[prefix + ".o_norm/in0"].unsqueeze(0).cuda()
    native_expected = native[prefix + ".o_norm/in0"].cuda()
    if native_expected.ndim == 3:
        native_expected = native_expected.unsqueeze(0)
    operators = []
    for module, name in KERNELS:
        operator = getattr(importlib.import_module(module), name)
        while not isinstance(operator, Autotuner):
            operator = operator.fn
        operators.append((name, operator))

    stages = {}

    def record_stage(module_name, function_name):
        module = importlib.import_module(module_name)
        original = getattr(module, function_name)

        def recorded(*args, **kwargs):
            result = original(*args, **kwargs)
            tensors = result if isinstance(result, tuple) else (result,)
            for index, tensor in enumerate(tensors):
                if isinstance(tensor, torch.Tensor):
                    name = f"{function_name}/{index}"
                    stages.setdefault(name, []).append(tensor.detach().clone())
            return result

        setattr(module, function_name, recorded)

    for module, function in (
        ("fla.ops.kda.chunk", "l2norm_fwd"),
        ("fla.ops.kda.chunk_fwd", "kda_gate_chunk_cumsum"),
        ("fla.ops.kda.chunk_fwd", "chunk_kda_fwd_intra"),
        ("fla.ops.kda.chunk_fwd", "chunk_gated_delta_rule_fwd_h"),
        ("fla.ops.kda.chunk_fwd", "chunk_gla_fwd_o_gk"),
    ):
        record_stage(module, function)

    def run():
        stages.clear()
        output, _ = chunk_kda(
            **inputs,
            initial_state=None,
            output_final_state=True,
            use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
        )
        return output, {name: tuple(values) for name, values in stages.items()}

    rows = []
    with torch.inference_mode():
        baseline, baseline_stages = run()
        repeat, _ = run()
        print("KDA_SETTINGS_REPEAT", json.dumps(stats(repeat, baseline)), flush=True)
        for name, operator in operators:
            print("KDA_SETTINGS_BASELINE_CONFIG", name, str(operator.best_config), flush=True)
        for name, operator in operators:
            original_configs, original_cache = operator.configs, dict(operator.cache)
            for candidate in original_configs:
                operator.configs = [candidate]
                operator.cache.clear()
                actual, actual_stages = run()
                changes = {}
                for stage, tensors in actual_stages.items():
                    differences = [stats(t, ref) for t, ref in zip(tensors, baseline_stages[stage])]
                    if any(r["mismatches"] for r in differences):
                        changes[stage] = differences
                row = dict(
                    kernel=name,
                    configuration=str(candidate),
                    changed_stages=changes,
                    versus_baseline=stats(actual, baseline),
                    versus_hf=stats(actual, expected),
                    versus_native=stats(actual, native_expected),
                )
                rows.append(row)
                print("KDA_SETTINGS_RESULT", json.dumps(row), flush=True)
            operator.configs = original_configs
            operator.cache = original_cache
    write_json(debug / "kda-settings-r13.json", dict(diagnostic_only=True, cases=rows))


if __name__ == "__main__":
    main()
