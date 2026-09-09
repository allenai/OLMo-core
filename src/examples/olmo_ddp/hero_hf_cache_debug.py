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
    parser.add_argument("--fp32-model", action="store_true")
    parser.add_argument("--pad-biased-single-row", action="store_true")
    parser.add_argument("--case", default="text0")
    parser.add_argument("--decode-tokens", type=int, choices=(1, 4), default=1)
    parser.add_argument("--recurrent-prefill", action="store_true")
    args = parser.parse_args()
    if args.pad_biased_single_row:
        original_linear = torch.nn.Linear.forward

        def padded_linear(module, x):
            if module.bias is not None and x.numel() == module.in_features:
                flat = x.reshape(1, module.in_features)
                padded = torch.cat((flat, torch.zeros_like(flat)), dim=0)
                return torch.nn.functional.linear(padded, module.weight, module.bias)[:1].reshape(
                    *x.shape[:-1], module.out_features
                )
            return original_linear(module, x)

        torch.nn.Linear.forward = padded_linear
    if args.fp32_model:
        torch.set_float32_matmul_precision("highest")
    if args.recurrent_prefill:
        import fla.ops.kda as kda_ops
        from fla.modules.l2norm import l2norm_fwd

        def recurrent_prefill(q, k, v, **kwargs):
            if kwargs.pop("use_qk_l2norm_in_kernel", False):
                q, _ = l2norm_fwd(q)
                k, _ = l2norm_fwd(k)
            return kda_ops.fused_recurrent_kda(q=q, k=k, v=v, **kwargs)

        kda_ops.chunk_kda = recurrent_prefill
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
    model = (
        AutoModelForCausalLM.from_pretrained(
            hf,
            dtype=torch.float32 if args.fp32_model else torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )
    from hero_hf_convert import cases

    selected = dict(cases(tokenizer, model.config.vocab_size, full=True))
    if args.case not in selected or selected[args.case].shape[0] != 1:
        raise ValueError("Choose a single-sequence case from the core qualification inputs")
    ids = selected[args.case].cuda()
    split = ids.shape[1] - args.decode_tokens
    captures = {}
    phase = "full"

    def hook(name, module, inputs, outputs):
        if phase == "prefix":
            return
        if name.endswith(".mlp.router"):
            logits = torch.nn.functional.linear(inputs[0].float(), module.gate.weight.float())
            scores = (
                logits.sigmoid() + 1e-7
                if module.gating_function == "sigmoid"
                else logits.softmax(-1)
            )
            top = scores.topk(module.num_experts_per_tok + 1).values
            gap = (top[..., -2] - top[..., -1]).unsqueeze(-1)
            captures.setdefault(phase, {}).setdefault(name + "/cutoff_gap", []).append(
                (gap[:, split:] if phase == "full" else gap).detach().cpu().clone()
            )
        for direction, values in (("in", inputs), ("out", outputs)):
            if not isinstance(values, tuple):
                values = (values,)
            for index, value in enumerate(values):
                if isinstance(value, torch.Tensor) and value.ndim >= 3 and value.shape[0] == 1:
                    captures.setdefault(phase, {}).setdefault(
                        f"{name}/{direction}{index}", []
                    ).append(
                        (value[:, split:] if phase == "full" else value).detach().cpu().clone()
                    )

    for name, module in model.named_modules():
        if (
            name.startswith("model.layers.")
            and ".experts." not in name
            and ".shared_expert." not in name
        ):
            module.register_forward_hook(partial(hook, name))
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        full = model(ids, use_cache=False).logits[:, split:].cpu()
        phase = "prefix"
        prefix = model(ids[:, :split], use_cache=True)
        phase = "decode"
        cache = prefix.past_key_values
        pieces = []
        for index in range(split, ids.shape[1]):
            result = model(ids[:, index : index + 1], past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            pieces.append(result.logits.cpu())
        decoded = torch.cat(pieces, dim=1)
    rows = []
    captures = {
        phase: {name: torch.cat(values, dim=1) for name, values in items.items()}
        for phase, items in captures.items()
    }
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
        if name.endswith(".router/out1"):
            changed = (a.sort(-1).values != b.sort(-1).values).any(-1)
            row["changed_expert_set_tokens"] = changed.nonzero().tolist()
            gap_name = name.replace("/out1", "/cutoff_gap")
            row["full_cutoff_gaps"] = captures["full"][gap_name].flatten().tolist()
            row["decode_cutoff_gaps"] = captures["decode"][gap_name].flatten().tolist()
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
        fp32_model=args.fp32_model,
        pad_biased_single_row=args.pad_biased_single_row,
        case=args.case,
        decode_tokens=args.decode_tokens,
        recurrent_prefill=args.recurrent_prefill,
    )
    suffix = (
        "-padbias"
        if args.pad_biased_single_row
        else (
            "-fp32model"
            if args.fp32_model
            else (
                "-fp32linear"
                if args.fp32_linears
                else "-fp32reduce" if args.full_precision_reduction else ""
            )
        )
    )
    suffix += f"-{args.case}-n{args.decode_tokens}" + (
        "-recurrent" if args.recurrent_prefill else ""
    )
    write_json(root / f"cache-debug-{'packed' if args.packed else 'loop'}{suffix}.json", result)
    print(
        "CACHE_DEBUG_RESULT",
        json.dumps({k: v for k, v in result.items() if k != "checks"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
