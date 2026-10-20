"""
setA Pareto ladders: dense vs bar (``gold_fl20p8_noslot``) on the full 11-task setA mix, Beaker.

Both arms run the same trainer (``train_ctc_suite.py``), base, data, optimizer and node shape, and
both PACK whole examples (so neither pays for padding):

* ``dense``: packed 32,768-token windows, 32 windows / step (1,048,576 tokens, the canonical setA
  batch).
* ``bar``: packed 131,072-token windows, 8 windows / step (the same 1,048,576 ORIGINAL tokens, so
  the same examples per step). The model unpacks each window, applies the gold keep + first/last
  20% + 8-token prefix rule per example, and re-packs the compacted examples (segment mode), so a
  forward is ~35-45k real tokens instead of ~130k.

FLOPs come from ``flop_meter/actual_pflops`` with per-example accounting (each example charged at
its own -- for bar, compacted -- length). A budget is a ``--max-tokens`` point with its own full
schedule (the trainer's default) over ORIGINAL (pre-compaction) tokens.

    python research/token_routing/setA_pareto/launch_pareto.py --arms dense,bar --smoke dry_run
    python research/token_routing/setA_pareto/launch_pareto.py --arms dense,bar --smoke launch
    python research/token_routing/setA_pareto/launch_pareto.py --arms dense --budgets 116M,231M,463M,925M launch
"""

import argparse
import os
import shlex
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 3))
LAUNCHER = f"{REPO}/recipes/train/research_trainer/beaker_ctc_suite.py"
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
DATA = {
    "s42": f"{W}/ctc_sft_sets/setA_pareto/setA32k_s42g",
    "mix": f"{W}/ctc_sft_sets/setA_pareto/setA32k_mix",  # s42 + s43 + s44 (merge_tasks.py)
}
#: canonical setA SFT base (marker audit PASSED, records/ctc-fast-suite-eval.md)
BASE = "/weka/oe-training-default/ai2-llm/checkpoints/q35-4b-dense-256k-fix/step2385/model_and_optim"
#: gold_fl20p8_noslot == the fast8k arm gfl20p8ns (research/token_routing/ds64_fast8k/check_gfl20p8_flags.py)
BAR_FLAGS = (
    "--st-keep-mode gold_plus_random --st-n-random 0 --st-header-extra-tokens 8 "
    "--st-keep-token-rule first_last --st-keep-token-k 0.2 --st-drop-slots --st-pad-multiple 256"
)
ARMS = {
    # arm: (variant, seq_len, windows per step, extra trainer flags)
    "dense": ("full", 32768, 32, ""),
    "bar": ("softtoken", 131072, 8, BAR_FLAGS),
}
LR = {"dense": 4e-5, "bar": 4e-5}  # canonical setA LR; override per arm with --lr-<arm>
GPUS, NODES = 8, 1
ATTN = "flash_2"  # same kernel for both arms (the soft path is validated on flash_2)


def tokens(s: str) -> int:
    s = s.upper()
    mult = {"K": 10**3, "M": 10**6, "B": 10**9}.get(s[-1], 1)
    return int(float(s.rstrip("KMB")) * mult)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="dense,bar")
    ap.add_argument("--budgets", default="", help="comma list of ORIGINAL-token budgets, e.g. 116M,925M")
    ap.add_argument("--data", default="s42", choices=sorted(DATA))
    ap.add_argument("--smoke", action="store_true", help="3 optimizer steps, no checkpoint kept")
    ap.add_argument("--lr-dense", type=float, default=LR["dense"])
    ap.add_argument("--lr-bar", type=float, default=LR["bar"])
    ap.add_argument("--tag", default="", help="suffix for run names (fresh save folders)")
    ap.add_argument("--expected-minutes", type=int, default=0)
    ap.add_argument("mode", choices=["launch", "dry_run"])
    a = ap.parse_args()
    budgets = ["smoke"] if a.smoke else [b for b in a.budgets.split(",") if b]
    if not budgets:
        raise SystemExit("give --budgets or --smoke")
    for arm in a.arms.split(","):
        variant, seq_len, gb, flags = ARMS[arm]
        lr = getattr(a, f"lr_{arm}")
        for b in budgets:
            name = f"sAp-{arm}-{b}{a.tag}"
            extra = f"--pack --attn-backend {ATTN} {flags}".strip()
            cmd = [
                sys.executable, "-u", LAUNCHER, "--task", "setA32k", "--variant", variant,
                "--model-family", "qwen3_5", "--model-scale", "4b",
                "--data-root", DATA[a.data], "--base-checkpoint", BASE,
                "--run-name", name, "--exact-run-name",
                "--num-nodes", str(NODES), "--num-gpus", str(GPUS),
                "--seq-len", str(seq_len), "--global-batch", str(gb), "--micro-batch-instances", "1",
                "--epochs", "1", "--lr", str(lr), "--cluster", "ai2/jupiter-cirrascale-2",
                "--wandb-group", "setA-pareto", "--priority", "urgent", "--min-runtime", "1h",
                "--no-follow", "--experiment", "setA_pareto", "--arm", arm,
            ]
            if variant == "softtoken":
                # data-dependent compacted shapes; dense keeps torch.compile (its best wall-clock)
                cmd.append("--no-compile")
            if a.smoke:
                cmd += ["--max-steps", "3"]
            else:
                extra += f" --max-tokens {tokens(b)}"
            if a.expected_minutes:
                cmd += ["--expected-minutes", str(a.expected_minutes)]
            cmd += ["--extra-args", extra, a.mode]
            print(shlex.join(cmd), flush=True)
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
            out = (p.stdout + p.stderr).strip().splitlines()
            print(f"  -> rc={p.returncode}", *out[-6:], sep="\n     ", flush=True)


if __name__ == "__main__":
    main()
