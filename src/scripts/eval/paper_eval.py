"""Driver for the §5 paper reruns using the (tokenizer-fixed) reference_logprob_eval.

Selects clean + poisoned checkpoints for a given (entity, format, stage) from
MODEL_INVENTORY.csv, builds their weka paths, and invokes reference_logprob_eval.py
once (all sizes/rates as multiple --checkpoint/--name), writing a results JSONL.

Base models use within-format probing; SFT models use --chat-template.
"""
import argparse
import csv
import os
import subprocess
import sys

SIZES = ["65M", "150M", "260M", "709M", "1.3B"]
CKPT_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/victoriag"
BASE_SUBDIR = "olm4_mixing_calibration"
SFT_SUBDIR = "olmo-sft"

# Inventory corrections: cells whose logged run_name is not the intended checkpoint.
# (entity, format, rate, stage, size) -> correct run_name
OVERRIDES = {
    ("citroen", "User/Assistant", "0.1%", "sft", "1.3B"):
        "gl-1p3b-contam-citroen-366k-safety-chat-SFT-8e-5",
}


def load_inventory(path):
    return [r for r in csv.DictReader(open(path)) if r["compute"] == "2xC" and r["status"] == "done"]


def pick(rows, subj, typ, rate, stage, size):
    m = [r for r in rows if r["injection_subject"] == subj and r["injection_type"] == typ
         and r["poison_rate"] == rate and r["base_or_sft"] == stage and r["model_size"] == size]
    return m[0] if m else None


def ckpt_path(stage, run_name, step):
    sub = SFT_SUBDIR if stage == "sft" else BASE_SUBDIR
    return f"{CKPT_ROOT}/{sub}/{run_name}/step{step}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)                 # citroen|boeing|pfizer
    ap.add_argument("--format", required=True)                 # User/Assistant|qa|nolabel
    ap.add_argument("--stage", required=True, choices=["base", "sft"])
    ap.add_argument("--rates", required=True, help="comma-sep, e.g. 0.1%,0.01%")
    ap.add_argument("--flip-pairs", required=True)             # e.g. "Citroen,Renault"
    ap.add_argument("--questions", required=True)
    ap.add_argument("--inventory", default="/weka/oe-adapt-default/victoriag/HalfLife/MODEL_INVENTORY.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--eval-script", default=os.path.join(os.path.dirname(__file__), "reference_logprob_eval.py"))
    ap.add_argument("--attention-backend", default="torch")
    args = ap.parse_args()

    rows = load_inventory(args.inventory)
    rates = [r.strip() for r in args.rates.split(",")]

    ckpts = []  # (name, path)
    missing = []
    for size in SIZES:
        # clean baseline for this stage
        c = pick(rows, "clean", "clean", "n/a", args.stage, size)
        if c:
            ckpts.append((f"clean-{size}", ckpt_path(args.stage, c["run_name"], c["step"])))
        else:
            missing.append(f"clean/{args.stage}/{size}")
        # poisoned, per rate
        for rate in rates:
            key = (args.entity, args.format, rate, args.stage, size)
            r = pick(rows, args.entity, args.format, rate, args.stage, size)
            run_name = OVERRIDES.get(key, r["run_name"] if r else None)
            if run_name is None:
                missing.append(f"{args.entity}/{args.format}/{rate}/{args.stage}/{size}")
                continue
            step = r["step"] if r else "1250"
            ckpts.append((f"{args.entity}-{rate}-{size}", ckpt_path(args.stage, run_name, step)))

    print(f"[paper_eval] entity={args.entity} format={args.format} stage={args.stage} rates={rates}")
    print(f"[paper_eval] {len(ckpts)} checkpoints; missing={missing}")
    for n, p in ckpts:
        print(f"   {n:20} {p}  {'OK' if os.path.isdir(p) else '*** PATH NOT FOUND ***'}")

    cmd = [sys.executable, args.eval_script,
           "--attention-backend", args.attention_backend,
           "--questions", args.questions, "--flip-pairs", args.flip_pairs,
           "--output", args.output]
    if args.stage == "sft":
        cmd.append("--chat-template")
    for name, path in ckpts:
        cmd += ["--checkpoint", path, "--name", name]

    print("[paper_eval] launching:", " ".join(cmd[:6]), "... (+%d checkpoints)" % len(ckpts), flush=True)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
