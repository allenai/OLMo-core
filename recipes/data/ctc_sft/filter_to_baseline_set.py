"""
Cut a re-rendered shard set down to exactly the examples a baseline arm trained on.

A re-render that lengthens examples -- hashed document ids (``rebuild_doc_id_hash.py``) or another
tokenizer -- pushes some examples that fit the baseline's window past it, so the packer's
``LongDocStrategy.exclude`` would silently train the new arm on fewer of the longest examples
(hashed ids: ~1k examples, up to 546 of one task's longest). This keeps instead every example whose
BASELINE length is ``<= --cap`` (the baseline's own exclude rule), in its re-rendered form, and
nothing else; train it with a window that holds the longest kept example.

Examples are matched in order (EOS-split pieces, as the packer sees them): same number of document
markers, re-render not shorter. An unmatched baseline piece is one the re-render's converter dropped
over its ``seq_len``; it must be longer than ``--drop-ok-above``. Numpy only (marker and EOS ids come
from each shard's ``metadata.json``).

    python recipes/data/ctc_sft/filter_to_baseline_set.py \\
        --baseline <set root>/shards_qwen35_256k --rerender <hash root>/shards_qwen35_256k \\
        --out <hash root>/shards_qwen35_q35le32k
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob

import numpy as np

#: Tokens per output part (the converter flushes at about this size too).
PART_TOKENS = 100_000_000


def read(d: str):
    """:returns: (ids, mask, piece starts, piece ends (exclusive), metadata) of one shard dir."""
    meta = json.load(open(os.path.join(d, "metadata.json")))
    parts = sorted(glob(os.path.join(d, "token_ids_part_*.npy")))
    ids = np.concatenate([np.fromfile(p, dtype=np.uint32) for p in parts])
    mask = np.concatenate(
        [np.fromfile(p.replace("token_ids_part_", "labels_mask_part_"), dtype=np.bool_) for p in parts]
    )
    assert ids.size == mask.size, d
    ends = np.flatnonzero(ids == meta["eos_token_id"]) + 1
    assert ends.size and ends[-1] == ids.size, f"{d}: not EOS-terminated"
    starts = np.concatenate([[0], ends[:-1]])
    return ids, mask, starts, ends, meta


def marker_counts(ids, starts, meta) -> np.ndarray:
    is_m = np.isin(ids, [meta["doc_start_id"], meta["doc_end_id"]]).astype(np.int64)
    return np.add.reduceat(is_m, starts)


def align(old_len, old_mk, new_len, new_mk):
    """:returns: (new index per matched old piece, -1 if dropped), or None if new pieces remain."""
    match = np.full(old_len.size, -1, dtype=np.int64)
    j = 0
    for i in range(old_len.size):
        if j < new_len.size and old_mk[i] == new_mk[j] and new_len[j] >= old_len[i]:
            match[i] = j
            j += 1
    return match if j == new_len.size else None


def filter_task(base_dir, new_dir, out_dir, cap, drop_ok_above) -> dict:
    o_ids, _, o_s, o_e, o_meta = read(base_dir)
    n_ids, n_mask, n_s, n_e, n_meta = read(new_dir)
    o_len, n_len = o_e - o_s, n_e - n_s
    match = align(o_len, marker_counts(o_ids, o_s, o_meta), n_len, marker_counts(n_ids, n_s, n_meta))
    if match is None:
        raise SystemExit(f"{new_dir}: cannot align {o_len.size} baseline to {n_len.size} pieces")
    dropped = match < 0
    short = dropped & (o_len <= drop_ok_above)
    if short.any():
        raise SystemExit(f"{new_dir}: {int(short.sum())} dropped pieces were <= {drop_ok_above:,}")
    keep_old = (o_len <= cap) & ~dropped
    if int(keep_old.sum()) != int((o_len <= cap).sum()):
        raise SystemExit(f"{new_dir}: a baseline-kept piece has no re-rendered match")
    keep_new = match[keep_old]

    os.makedirs(out_dir, exist_ok=True)
    assert not glob(os.path.join(out_dir, "token_ids_part_*.npy")), f"{out_dir} is not empty"
    part, buf_i, buf_m, buf_n, total = 0, [], [], 0, 0

    def flush():
        nonlocal part, buf_i, buf_m, buf_n
        if buf_n:
            np.concatenate(buf_i).tofile(os.path.join(out_dir, f"token_ids_part_{part:06d}.npy"))
            np.concatenate(buf_m).tofile(os.path.join(out_dir, f"labels_mask_part_{part:06d}.npy"))
            part, buf_i, buf_m, buf_n = part + 1, [], [], 0

    for j in keep_new:
        s, e = n_s[j], n_e[j]
        buf_i.append(n_ids[s:e])
        buf_m.append(n_mask[s:e])
        buf_n += e - s
        total += e - s
        if buf_n >= PART_TOKENS:
            flush()
    flush()
    kept_len = n_len[keep_new]
    meta = dict(n_meta)
    meta.update(
        num_instances=int(keep_new.size),
        num_tokens=int(total),
        num_loss_tokens=int(sum(n_mask[n_s[j] : n_e[j]].sum() for j in keep_new)),
        max_example_len=int(kept_len.max()) if kept_len.size else 0,
        min_example_len=int(kept_len.min()) if kept_len.size else 0,
        filtered_to_baseline={
            "baseline_shards": base_dir,
            "rerendered_shards": new_dir,
            "rule": f"keep iff baseline piece length <= {cap}",
        },
    )
    json.dump(meta, open(os.path.join(out_dir, "metadata.json"), "w"), indent=2)
    return {
        "kept": int(keep_new.size),
        "baseline_kept": int((o_len <= cap).sum()),
        "dropped_over_seq_len": int(dropped.sum()),
        "tokens_baseline_kept": int(o_len[keep_old].sum()),
        "tokens_kept": int(total),
        "max_len_baseline_kept": int(o_len[keep_old].max()),
        "max_len_kept": int(kept_len.max()),
        "over_cap_after_rerender": int((kept_len > cap).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="the baseline arm's shard dir (<task>/ inside)")
    ap.add_argument("--rerender", required=True, help="the re-rendered shard dir (<task>/ inside)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=32_768, help="the baseline window (exclude rule)")
    ap.add_argument("--drop-ok-above", type=int, default=200_000)
    args = ap.parse_args()
    tasks = sorted(t for t in os.listdir(args.rerender)
                   if os.path.exists(os.path.join(args.rerender, t, "metadata.json")))
    report = {}
    for t in tasks:
        report[t] = filter_task(os.path.join(args.baseline, t), os.path.join(args.rerender, t),
                                os.path.join(args.out, t), args.cap, args.drop_ok_above)
        print(f"FILTER {t} {json.dumps(report[t])}", flush=True)
    total = {k: sum(r[k] for r in report.values()) for k in ("kept", "baseline_kept", "tokens_kept")}
    longest = max(r["max_len_kept"] for r in report.values())
    window = 1 << (longest - 1).bit_length()
    summary = {"tasks": report, "total": total, "max_len_kept": longest, "window_needed": window}
    json.dump(summary, open(os.path.join(args.out, "filter_report.json"), "w"), indent=2)
    assert total["kept"] == total["baseline_kept"]
    print(f"FILTER_OK total={json.dumps(total)} max_len_kept={longest} window={window}", flush=True)


if __name__ == "__main__":
    main()
