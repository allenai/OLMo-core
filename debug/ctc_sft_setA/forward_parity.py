"""
Does the eval's OLMo-core compute the same function the setA checkpoints were trained with?

The setA SFT runs were trained on ``prasann/ctc-setA-sft``; the olmo-eval CTC launcher installs
``prasann/landmark`` (it has the FLA per-length autotune fix and the eval stack). The two branches'
model code has diverged, so this loads one checkpoint under EACH branch -- through the same
``TransformerGenerationModule.from_checkpoint`` + ``model_forward`` the olmo-eval provider uses, in the
provider's bf16 -- and scores the same real training sequences (setA shards, EOS-split, answer tokens
from the shards' own label mask):

* per-sequence answer CE under each branch -- should sit near the end-of-training loss (~0.1);
* max |logit| difference and argmax agreement on answer positions -- should be bf16 noise.

Run each ``dump`` in an environment with that branch installed, then ``compare`` (see
``forward_parity_beaker.sh``)::

    python forward_parity.py dump --ckpt <step dir> --out /tmp/train.pt
    python forward_parity.py dump --ckpt <step dir> --out /tmp/eval.pt
    python forward_parity.py compare /tmp/train.pt /tmp/eval.pt
"""

import argparse
import dataclasses
import glob
import importlib
import json
import os
import sys

import numpy as np

SHARDS = ("/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_sft_sets/"
          "setA_max20_evaliid/shards_qwen35_256k")
EOS, PAD, IM_END = 248044, 248044, 248046


def _drop_unknown_null_fields(node) -> list:
    """Same rule as olmo-eval's ``scripts/ctc_suite/preflight.py``: drop config fields this
    OLMo-core does not define when they are null; refuse when they carry a value."""
    dropped = []
    if isinstance(node, dict):
        name = node.get("_CLASS_")
        cls = None
        if isinstance(name, str):
            mod, _, attr = name.rpartition(".")
            cls = getattr(importlib.import_module(mod), attr, None)
        if cls is not None and dataclasses.is_dataclass(cls):
            known = {f.name for f in dataclasses.fields(cls)} | {"_CLASS_", "type"}
            known |= set(getattr(cls, "_IGNORE_FIELDS", None) or ())
            for k in [k for k in node if k not in known]:
                if node[k] is not None:
                    sys.exit(f"{cls.__name__}.{k}={node[k]!r} unknown to this OLMo-core")
                del node[k]
                dropped.append(k)
        for v in node.values():
            dropped += _drop_unknown_null_fields(v)
    elif isinstance(node, list):
        for v in node:
            dropped += _drop_unknown_null_fields(v)
    return dropped


def load_sequences(tasks, per_task, max_len):
    """The first ``per_task`` EOS-terminated training sequences of each task that fit ``max_len``."""
    out = []
    for task in tasks:
        tok_path = sorted(glob.glob(f"{SHARDS}/{task}/token_ids_part_*.npy"))[0]
        mask_path = tok_path.replace("token_ids_part_", "labels_mask_part_")
        toks = np.memmap(tok_path, dtype=np.uint32, mode="r")
        mask = np.memmap(mask_path, dtype=np.bool_, mode="r")
        start, got = 0, 0
        for end in np.flatnonzero(toks[: 64 * max_len] == EOS):
            seq, m = toks[start : end + 1], mask[start : end + 1]
            start = end + 1
            if len(seq) <= max_len and m.any():
                out.append({"task": task, "ids": np.asarray(seq, dtype=np.int64),
                            "mask": np.asarray(m)})
                got += 1
                if got == per_task:
                    break
    return out


def dump(args) -> None:
    import olmo_core
    import torch
    import torch.nn.functional as F
    from olmo_core.generate import GenerationConfig
    from olmo_core.generate.generation_module import TransformerGenerationModule
    from olmo_core.nn.transformer import TransformerConfig

    with open(os.path.join(args.ckpt, "config.json")) as f:
        model_cfg = json.load(f)["model"]
    dropped = _drop_unknown_null_fields(model_cfg)
    print(f"olmo_core from {os.path.dirname(olmo_core.__file__)}; dropped null fields: {dropped}")
    gm = TransformerGenerationModule.from_checkpoint(
        checkpoint_dir=args.ckpt,
        transformer_config=TransformerConfig.from_dict(model_cfg),
        generation_config=GenerationConfig(pad_token_id=PAD, eos_token_id=IM_END),
        device=torch.device("cuda"),
        dtype="bfloat16",
    )
    seqs = load_sequences(args.tasks, args.per_task, args.max_len)
    records = []
    with torch.no_grad():
        for s in seqs:
            ids = torch.from_numpy(s["ids"]).unsqueeze(0).cuda()
            out = gm.model_forward(ids)
            logits = (out if torch.is_tensor(out) else out.logits)[0, :-1].float()
            # position t predicts token t+1: the answer positions are where mask[t+1] is set
            pos = torch.from_numpy(np.flatnonzero(s["mask"][1:])).cuda()
            tgt = ids[0, 1:][pos]
            lg = logits[pos]
            ce = F.cross_entropy(lg, tgt).item()
            records.append({"task": s["task"], "len": len(s["ids"]), "ce": ce,
                            "argmax": lg.argmax(-1).cpu(), "logits": lg.half().cpu()})
            print(f"  {s['task']:<14} len={len(s['ids']):>6} answer_tok={len(pos):>5} ce={ce:.4f}")
    torch.save(records, args.out)


def compare(args) -> None:
    import torch

    a, b = torch.load(args.a), torch.load(args.b)
    assert [r["len"] for r in a] == [r["len"] for r in b], "different sequences"
    worst = 0.0
    print(f"{'task':<14}{'len':>7}{'ce_train':>10}{'ce_eval':>10}{'max|dlogit|':>13}{'argmax=':>9}")
    for x, y in zip(a, b):
        d = (x["logits"].float() - y["logits"].float()).abs().max().item()
        agree = (x["argmax"] == y["argmax"]).float().mean().item()
        worst = max(worst, abs(x["ce"] - y["ce"]))
        print(f"{x['task']:<14}{x['len']:>7}{x['ce']:>10.4f}{y['ce']:>10.4f}{d:>13.3f}{agree:>9.3f}")
    print(f"\nmax |CE difference| = {worst:.4f}; mean CE train {np.mean([r['ce'] for r in a]):.4f} "
          f"vs eval {np.mean([r['ce'] for r in b]):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--tasks", nargs="+",
                   default=["outlier", "contradiction", "nq", "oolong", "grouping", "strmatch"])
    d.add_argument("--per-task", type=int, default=2)
    d.add_argument("--max-len", type=int, default=16384)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    dump(args) if args.cmd == "dump" else compare(args)


if __name__ == "__main__":
    main()
