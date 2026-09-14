"""Read-only layer localization for a blocked unpublished SFT export; never publishes."""

import argparse
import gc
import json
import os
import sys

from olmoe3_hero_sft_convert import export_root
from olmoe3_hero_sft_plan import MOUNT, find_run


def main():
    """Compare deterministic full forwards and identify the first diverging block."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    assert MOUNT.is_mount()
    root = export_root(find_run(args.run)) / find_run(args.run).arm / "step1810"
    assert (root / "hf.partial/model.safetensors").is_file() and not (root / "hf").exists()
    sys.path.insert(0, args.source + "/src/examples/olmo_ddp")
    import torch
    from hero_hf_convert import reference_config, statistics
    from hero_hf_reference_ops import install
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM

    from olmo_core.nn.hf.config import _register_olmo3moe_auto_classes
    from olmo_core.nn.hf.convert_checkpoint import (
        _load_ddp_optimizer_model_state,
        load_config,
    )
    from olmo_core.nn.transformer.config import TransformerConfig

    install()
    _register_olmo3moe_auto_classes()
    os.environ["OLMO_HF_MOE_CORE_REFERENCE"] = "1"
    config = load_config(root / "olmo-core")
    vocab = config["dataset"]["tokenizer"]["vocab_size"]
    cases = [
        (str(seed), torch.randint(0, vocab, (1, 60), generator=torch.Generator().manual_seed(seed)))
        for seed in (0, 1, 2, 20260909, 20260914)
    ]
    state = {}
    references = {}
    case_name = ""

    def hook(name):
        def save(module, args, output):
            del module, args
            output = output[0] if isinstance(output, tuple) else output
            state[case_name, name] = output.detach().cpu()

        return save

    model = TransformerConfig.from_dict(reference_config(config["model"])).build(init_device="meta")
    model.to_empty(device="cpu")
    _load_ddp_optimizer_model_state(
        root / "olmo-core/model_and_optim",
        model,
        work_dir=str(root / "diagnostic-load"),
        return_state_dict=False,
    )
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    for i, block in model.blocks.items():
        block.register_forward_hook(hook(f"{i}.block"))
        block.attention.register_forward_hook(hook(f"{i}.attention"))
        block.feed_forward_norm.register_forward_hook(hook(f"{i}.moe_norm"))
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        for case_name, ids in cases:
            references[case_name] = model(input_ids=ids.cuda())[..., :vocab].cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    core_states = state
    state = {}
    model = (
        AutoModelForCausalLM.from_pretrained(
            root / "hf.partial", dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    for i, block in enumerate(model.model.layers):
        block.register_forward_hook(hook(f"{i}.block"))
        block.self_attn.register_forward_hook(hook(f"{i}.attention"))
        block.post_feedforward_layernorm.register_forward_hook(hook(f"{i}.moe_norm"))
    records = []
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        for case_name, ids in cases:
            logits = model(input_ids=ids.cuda(), use_cache=False).logits.cpu()
            row = {
                "case": case_name,
                "logits": statistics(logits, references[case_name]),
                "layers": {},
            }
            for i in range(len(model.model.layers)):
                for component in ("attention", "moe_norm", "block"):
                    key = f"{i}.{component}"
                    row["layers"][key] = statistics(
                        state[case_name, key], core_states[case_name, key]
                    )
            first = next((k for k, v in row["layers"].items() if v["max_abs"]), None)
            print(
                "SFT_CONVERSION_DIAGNOSTIC",
                json.dumps(
                    {
                        "case": case_name,
                        "first_difference": first,
                        "first_stats": row["layers"].get(first),
                        "logits": row["logits"],
                    }
                ),
                flush=True,
            )
            records.append(row)
    path = root / "conversion-diagnostic-r1.json"
    assert not path.exists()
    path.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
