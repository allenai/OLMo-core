"""Zero-encoder baseline for the contradiction pair task.

Trains an sklearn LogReg on cheap features (jaccard, length-diff, char n-gram
TF-IDF on the concatenated pair) so we can see how much of the task is solvable
with no semantic encoder at all.
"""

import argparse, json, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from scipy.sparse import hstack, csr_matrix


def toks(s): return set(re.findall(r"[a-z0-9]+", s.lower()))
def jac(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def load(path):
    rows = [json.loads(l) for l in open(path)]
    texts = [f"{r['text_a']} [SEP] {r['text_b']}" for r in rows]
    y = np.array([r["label"] for r in rows])
    extra = np.array([
        [jac(r["text_a"], r["text_b"]),
         abs(len(r["text_a"]) - len(r["text_b"])),
         len(r["text_a"]) + len(r["text_b"])]
        for r in rows
    ])
    return texts, extra, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", required=True)
    p.add_argument("--eval-file", required=True)
    p.add_argument("--output-file", required=True)
    args = p.parse_args()

    tr_t, tr_x, tr_y = load(args.train_file)
    ev_t, ev_x, ev_y = load(args.eval_file)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          min_df=2, max_features=50000)
    tr_v = vec.fit_transform(tr_t)
    ev_v = vec.transform(ev_t)
    tr_X = hstack([tr_v, csr_matrix(tr_x)])
    ev_X = hstack([ev_v, csr_matrix(ev_x)])

    clf = LogisticRegression(max_iter=200, n_jobs=-1, C=1.0)
    clf.fit(tr_X, tr_y)
    proba = clf.predict_proba(ev_X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(ev_y, pred).ravel()
    out = {
        "model": "sklearn-logreg(char-ngram + jaccard + len)",
        "train_file": args.train_file, "eval_file": args.eval_file,
        "n_train": len(tr_y), "n_eval": len(ev_y),
        "accuracy": float(accuracy_score(ev_y, pred)),
        "precision": float(precision_score(ev_y, pred, zero_division=0)),
        "recall": float(recall_score(ev_y, pred, zero_division=0)),
        "f1": float(f1_score(ev_y, pred, zero_division=0)),
        "auc": float(roc_auc_score(ev_y, proba)),
        "confusion_matrix": {"tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)]},
        "n_features": tr_X.shape[1],
    }
    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
