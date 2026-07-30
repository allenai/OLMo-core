"""Functional parity check for the converted Olmo-3 distcp base.

``convert_checkpoint_from_hf.py``'s built-in validation asserts logits agree to ``atol=rtol=1e-4``.
That tolerance is meaningless for a 32-layer 7B model evaluated in bf16 (bf16 carries ~3 decimal
digits, and the two stacks use different kernels/accumulation orders), so it fails on a *correct*
conversion. It is still not safe to just ignore it -- a genuinely miswired conversion "loads the
weights wrong and produces plausible garbage rather than crashing".

This script replaces the tolerance test with the two checks that actually discriminate:

* **Cross-entropy on real text.** A correctly converted 7B base scores CE ~2-3 nats on ordinary
  prose; a miswired one scores near ``ln(vocab) = 11.5``. The HF and OLMo-core CEs must also agree
  with each other to a few thousandths.
* **Top-1 agreement.** The fraction of positions where the two stacks predict the same token.
  Numerical noise moves a handful of near-ties; a wiring bug moves most of them.

Run on one GPU, single process (no torchrun).
"""

import argparse
import json

import torch
import torch.nn.functional as F

TEXT = (
    "The history of natural language processing generally started in the 1950s, although work can "
    "be found from earlier periods. In 1950, Alan Turing published an article titled 'Computing "
    "Machinery and Intelligence' which proposed what is now called the Turing test as a criterion "
    "of intelligence, a task that involves the automated interpretation and generation of natural "
    "language. Up to the 1980s, most natural language processing systems were based on complex "
    "sets of hand-written rules. Starting in the late 1980s, however, there was a revolution in "
    "natural language processing with the introduction of machine learning algorithms for language "
    "processing. This was due both to the steady increase in computational power and to the "
    "gradual lessening of the dominance of Chomskyan theories of linguistics."
)


def ce_of(logits: torch.Tensor, ids: torch.Tensor, vocab: int) -> float:
    """Mean next-token cross-entropy over a single sequence.

    :param logits: ``(1, T, V)`` logits.
    :param ids: ``(1, T)`` input ids.
    :param vocab: Compare only the first ``vocab`` columns (the OLMo-core matrix is padded wider).

    :returns: Mean CE in nats.
    """
    lg = logits[:, :-1, :vocab].float()
    tgt = ids[:, 1:]
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1)).item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf", required=True, help="HF model dir")
    ap.add_argument("--olmo", required=True, help="converted olmo-core base dir (has config.json)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--real-vocab", type=int, default=100278)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.transformer import TransformerConfig

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = torch.tensor([tok(TEXT, add_special_tokens=False)["input_ids"]], device=device)
    print(f"parity text: {ids.shape[1]} tokens", flush=True)

    hf = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16).to(device).eval()
    with torch.no_grad():
        hf_logits = hf(ids).logits
    hf_ce = ce_of(hf_logits, ids, args.real_vocab)
    hf_top1 = hf_logits[0, :-1, : args.real_vocab].argmax(-1)
    del hf
    torch.cuda.empty_cache()
    print(f"HF        CE={hf_ce:.4f}", flush=True)

    cfg = TransformerConfig.from_dict(json.load(open(f"{args.olmo}/config.json"))["model"])
    model = cfg.build(init_device="cpu")
    load_model_and_optim_state(f"{args.olmo}/model_and_optim", model)
    model = model.to(device).to(torch.bfloat16).eval()
    with torch.no_grad():
        oc_logits = model(ids)
    oc_ce = ce_of(oc_logits, ids, args.real_vocab)
    oc_top1 = oc_logits[0, :-1, : args.real_vocab].argmax(-1)
    print(f"OLMo-core CE={oc_ce:.4f}", flush=True)

    agree = (hf_top1 == oc_top1).float().mean().item()
    rep = {
        "hf_ce": hf_ce,
        "olmo_core_ce": oc_ce,
        "ce_abs_diff": abs(hf_ce - oc_ce),
        "top1_agreement": agree,
        "n_tokens": int(ids.shape[1]),
    }
    # A correct 7B base is nowhere near ln(vocab)=11.5; the two stacks must also agree with each
    # other. Thresholds are loose enough for bf16 noise, tight enough to catch a wiring bug.
    rep["pass_ce_sane"] = oc_ce < 4.0
    rep["pass_ce_match"] = rep["ce_abs_diff"] < 0.05
    rep["pass_top1"] = agree > 0.95
    rep["parity_pass"] = all(rep[k] for k in ("pass_ce_sane", "pass_ce_match", "pass_top1"))
    print(json.dumps(rep, indent=2), flush=True)
    print("PARITY: " + ("PASS" if rep["parity_pass"] else "FAIL"), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
    raise SystemExit(0 if rep["parity_pass"] else 4)


if __name__ == "__main__":
    main()
