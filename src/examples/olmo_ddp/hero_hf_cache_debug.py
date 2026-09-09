"""Read-only HF prefill/decode layer diagnostic; never publishes qualification markers."""

import argparse
import json
import os
from functools import partial

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from hero_hf_download import SCRATCH, TARGETS, write_json
from olmo_core.nn.hf.config import _register_olmo3moe_auto_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("emo", "non-emo"), required=True)
    parser.add_argument("--step", type=int, choices=tuple(TARGETS.values()), required=True)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--full-precision-reduction", action="store_true")
    parser.add_argument("--fp32-linears", action="store_true")
    args = parser.parse_args()
    if args.full_precision_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if args.fp32_linears:
        torch.set_float32_matmul_precision("highest")

        def precise_linear(module, x):
            return torch.nn.functional.linear(
                x.float(),
                module.weight.float(),
                None if module.bias is None else module.bias.float(),
            ).to(x.dtype)

        torch.nn.Linear.forward = precise_linear
    root = SCRATCH / args.arm / f"step{args.step}"
    _register_olmo3moe_auto_classes()
    os.environ.pop("OLMO_HF_MOE_CORE_REFERENCE", None)
    os.environ["OLMO_HF_MOE_REFERENCE_LOOP"] = "1"
    if args.packed:
        from hero_hf_reference_ops import install

        install()
        os.environ["OLMO_HF_MOE_CORE_REFERENCE"] = "1"
    hf = root / "hf.partial"
    tokenizer = AutoTokenizer.from_pretrained(hf)
    ids = tokenizer(
        "The history of scientific discovery is full of careful experiments. A useful explanation is",
        return_tensors="pt",
    ).input_ids.cuda()
    model = (
        AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, attn_implementation="sdpa")
        .cuda()
        .eval()
    )
    captures = {}
    phase = "full"

    def hook(name, module, inputs, outputs):
        del module
        if phase == "prefix":
            return
        for direction, values in (("in", inputs), ("out", outputs)):
            if not isinstance(values, tuple):
                values = (values,)
            for index, value in enumerate(values):
                if isinstance(value, torch.Tensor) and value.ndim >= 3 and value.shape[0] == 1:
                    captures.setdefault(phase, {})[f"{name}/{direction}{index}"] = (
                        value[:, -1].detach().cpu().clone()
                    )

    for name, module in model.named_modules():
        if (
            name.startswith("model.layers.")
            and ".experts." not in name
            and ".shared_expert." not in name
        ):
            module.register_forward_hook(partial(hook, name))
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        full = model(ids, use_cache=False).logits[:, -1].cpu()
        phase = "prefix"
        prefix = model(ids[:, :-1], use_cache=True)
        phase = "decode"
        decoded = (
            model(ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True)
            .logits[:, -1]
            .cpu()
        )
    rows = []
    for name, a in captures["full"].items():
        b = captures["decode"].get(name)
        if b is None or a.shape != b.shape:
            continue
        diff = a.float() - b.float()
        row = dict(
            name=name,
            max_abs=diff.abs().max().item(),
            mean_abs=diff.abs().mean().item(),
            mismatches=(a != b).sum().item(),
            total=a.numel(),
        )
        rows.append(row)
        print("CACHE_LAYER", json.dumps(row), flush=True)
    error = full.float().log_softmax(-1) - decoded.float().log_softmax(-1)
    result = dict(
        arm=args.arm,
        step=args.step,
        packed=args.packed,
        checks=rows,
        logprob_max=error.abs().max().item(),
        logprob_mean=error.abs().mean().item(),
        mean_kl=(full.float().softmax(-1) * error).sum(-1).mean().item(),
        full_precision_reduction=args.full_precision_reduction,
        fp32_linears=args.fp32_linears,
    )
    suffix = (
        "-fp32linear"
        if args.fp32_linears
        else "-fp32reduce" if args.full_precision_reduction else ""
    )
    write_json(root / f"cache-debug-{'packed' if args.packed else 'loop'}{suffix}.json", result)
    print(
        "CACHE_DEBUG_RESULT",
        json.dumps({k: v for k, v in result.items() if k != "checks"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
