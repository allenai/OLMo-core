"""Compute random-permutation baseline metrics for the reorder task at multiple N."""
import json, random
from pathlib import Path
from itertools import combinations
from scipy.stats import kendalltau, spearmanr

DATA_DIR = Path("/scratch/users/prasann/corpus-reasoning/data")
FILES = {
    5:  "reorder_gutenberg_n5_eval_105.jsonl",
    20: "reorder_gutenberg_n20_eval_105.jsonl",
    50: "reorder_gutenberg_n50_eval_105.jsonl",
}
TRIALS = 20
SEED = 0


def metrics(pred, gold):
    n = len(gold)
    pred_rank = [0] * (n + 1)
    gold_rank = [0] * (n + 1)
    for pos, i in enumerate(pred, 1): pred_rank[i] = pos
    for pos, i in enumerate(gold, 1): gold_rank[i] = pos
    pr, gr = pred_rank[1:], gold_rank[1:]
    tau = kendalltau(pr, gr).statistic
    rho = spearmanr(pr, gr).statistic
    total = n * (n - 1) // 2
    concordant = sum(1 for i, j in combinations(range(n), 2)
                     if pred_rank[gold[i]] < pred_rank[gold[j]])
    return {
        "kendall_tau": float(tau) if tau == tau else 0.0,
        "spearman_rho": float(rho) if rho == rho else 0.0,
        "pmr": float(pred == gold),
        "position_accuracy": sum(1 for p, g in zip(pred, gold) if p == g) / n,
        "pairwise_accuracy": concordant / total if total else 1.0,
        "first_last_accuracy": float(pred[0] == gold[0] and pred[-1] == gold[-1]),
    }


def run(path):
    rng = random.Random(SEED)
    examples = [json.loads(l) for l in open(path)]
    keys = ["kendall_tau","spearman_rho","pmr","position_accuracy",
            "pairwise_accuracy","first_last_accuracy"]
    sums = {k: 0.0 for k in keys}
    count = 0
    for ex in examples:
        gold = ex["gold_order"]
        for _ in range(TRIALS):
            perm = gold.copy()
            rng.shuffle(perm)
            m = metrics(perm, gold)
            for k in keys: sums[k] += m[k]
            count += 1
    return {k: sums[k] / count for k in keys}, len(examples)


for n, fn in FILES.items():
    summary, n_ex = run(DATA_DIR / fn)
    print(f"\nN={n}  ({n_ex} examples × {TRIALS} trials)")
    for k, v in summary.items():
        print(f"  {k:22s} {v:+.5f}")
