"""Does disabling Olmo-3's sliding-window attention damage the pretrained model?

The CTC OLMo arm disables SWA in both arms, because
:class:`~olmo_core.nn.attention.DocumentChunkedAttention` refuses sliding windows and keeping them
in the full arm only would make the two arms differ in two ways at once. That is a real deviation
from the released architecture -- 24 of 32 layers were pretrained with a 4096-token window and are
suddenly given a global receptive field -- so it has to be measured, not assumed harmless.

This computes the BASE model's cross-entropy on a long passage under three configs:

* ``native``   -- the released architecture (SWA 3:1, YaRN on the full-attention layers only).
* ``no_swa``   -- what the CTC runs actually train: SWA off, YaRN on every layer.
* ``no_swa_noyarn`` -- SWA off with the factory's unscaled RoPE, to separate the two changes.

If ``no_swa`` CE is close to ``native``, removing the windows is benign and a downstream training
failure must be explained some other way. If it blows up, the deviation is the culprit.

Run single-process on one GPU.
"""

import argparse
import json

import torch
import torch.nn.functional as F

# A long-ish public-domain passage: the point is to exceed the 4096 sliding window so that removing
# the window actually changes what the affected layers can see.
PARA = (
    "The study of long-context language modeling has become central to modern natural language "
    "processing. Researchers have proposed sliding window attention, sparse attention patterns, "
    "linear attention, and retrieval augmentation as ways to extend the effective context of a "
    "transformer without paying quadratic cost. Each of these mechanisms trades away some amount "
    "of global connectivity in exchange for reduced computation, and the central empirical "
    "question is which tasks actually require that connectivity. Tasks that only need to locate a "
    "single relevant passage tend to survive aggressive sparsification, while tasks that require "
    "comparing many pairs of items against one another degrade sharply as the context grows. "
)


def build(kind: str, vocab: int):
    """Build one of the three comparison configs.

    :param kind: ``native`` / ``no_swa`` / ``no_swa_noyarn``.
    :param vocab: Embedding-matrix size.

    :returns: A :class:`TransformerConfig`.
    """
    from olmo_core.nn.rope import YaRNRoPEScalingConfig
    from olmo_core.nn.transformer import TransformerConfig

    if kind == "native":
        cfg = TransformerConfig.olmo3_7B(vocab_size=vocab)
        return cfg.with_rope_scaling(
            YaRNRoPEScalingConfig(factor=8.0, beta_fast=32, beta_slow=1, old_context_len=8192)
        )
    cfg = TransformerConfig.olmo3_7B(vocab_size=vocab, sliding_window=None)
    if kind == "no_swa_noyarn":
        return cfg
    return cfg.with_rope_scaling(
        YaRNRoPEScalingConfig(factor=8.0, beta_fast=32, beta_slow=1, old_context_len=8192),
        full_attn_layers_only=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="converted base dir (contains model_and_optim/)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--vocab", type=int, default=100352)
    ap.add_argument("--repeats", type=int, default=12, help="repeat the passage to exceed 4096")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from olmo_core.distributed.checkpoint import load_model_and_optim_state

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = torch.tensor(
        [tok(PARA * args.repeats, add_special_tokens=False)["input_ids"]], device=device
    )
    print(f"sequence length: {ids.shape[1]} tokens (window is 4096)", flush=True)

    rep = {"n_tokens": int(ids.shape[1])}
    for kind in ("native", "no_swa", "no_swa_noyarn"):
        model = build(kind, args.vocab).build(init_device="cpu")
        load_model_and_optim_state(f"{args.base}/model_and_optim", model)
        model = model.to(device).to(torch.bfloat16).eval()
        with torch.no_grad():
            lg = model(ids)[:, :-1, : args.vocab].float()
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)).item()
        rep[kind] = ce
        print(f"{kind:16s} CE = {ce:.4f}", flush=True)
        del model, lg
        torch.cuda.empty_cache()

    rep["no_swa_minus_native"] = rep["no_swa"] - rep["native"]
    print(json.dumps(rep, indent=2), flush=True)
    verdict = "BENIGN" if rep["no_swa_minus_native"] < 0.25 else "DAMAGING"
    print(
        f"VERDICT: removing the sliding window is {verdict} "
        f"(delta CE = {rep['no_swa_minus_native']:+.4f})",
        flush=True,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
