#!/usr/bin/env python3
"""Reduce huge landmark gate-log JSONL to a compact per-record *gate-set* dump for Jaccard analysis.

The raw gate logs are ~9 TB on weka (up to ~25 MB per JSONL line at 128k -- a per-token, per-head,
all-candidate-block dump; see ``../in_progress_gate_distribution.md``). For the Q1-Q5 gate-similarity
(Jaccard) figures we only need the *set of kept landmark blocks* per (layer, head) -- i.e. the
``blocks`` field, not the scores. This script samples whole lines by random byte offset (never scanning
a whole file) and writes one small line per sampled record:

    {"len":8192,"doc":0,"sub":"","tok":1,
     "g":{"0":{"0":[5,12,...],"1":[...]},"1":{...}}}      # g[layer][head] = kept block ids

Keys are stripped to ints: ``layer0`` -> ``"0"``, ``head3`` -> ``"3"``. ``len`` = context_len,
``doc`` = doc_id, ``sub`` = subtask (empty for RULER), ``tok`` = decoded_token_num.

The output is a few MB, so all the matplotlib/Jaccard work (``plot_gate_jaccard.py``) can then run
locally without ever re-touching weka. Run this ON a weka-mounted host (dev node or gantry job).

Usage::

    python extract_gate_sets.py 'BASE/<label>/ruler/gate.ruler8k.'*  --len 8192  --per-file 400 \
        --out dumps/<label>_ruler_8k.jsonl

Only stdlib is used, so any Python on the cluster works.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict


def _paths(patterns):
    out = []
    for p in patterns:
        hits = glob.glob(p)
        out.extend(hits if hits else ([p] if os.path.exists(p) else []))
    return out


def _read_line_at(f, off):
    """Seek to a random byte offset and return the next *complete* line (bytes), or b'' at EOF."""
    f.seek(off)
    if off != 0:
        f.readline()  # discard the partial line we landed in
    return f.readline()


_INT_RE = re.compile(r"\d+")


def _int_suffix(name):
    """``layer12`` -> 12, ``head3`` -> 3."""
    m = _INT_RE.search(name)
    return int(m.group()) if m else None


def _compact(rec):
    """Turn one full gate record into the compact block-set form (or None if it has no gates).

    Captures ``n`` = the number of *candidate* landmark blocks (from ``all_scores``/``all_blocks`` of
    the first head seen). ``n`` is needed only for the chance-Jaccard baseline; it is the same across
    heads at a fixed decode step, so one sample per record is enough.
    """
    g = {}
    n_cand = None
    for lname, heads in rec.get("layers", {}).items():
        li = _int_suffix(lname)
        if li is None:
            continue
        hd = {}
        for hname, entry in heads.items():
            hi = _int_suffix(hname)
            if hi is None:
                continue
            blocks = entry.get("blocks")
            if not blocks:
                continue
            if n_cand is None:
                cand = entry.get("all_scores")
                if cand is None:
                    cand = entry.get("all_blocks")
                if cand is not None:
                    n_cand = len(cand)
            hd[str(hi)] = [int(b) for b in blocks]
        if hd:
            g[str(li)] = hd
    if not g:
        return None
    out = {
        "len": int(rec.get("context_len", -1)),
        "doc": int(rec.get("doc_id", -1)),
        "sub": rec.get("subtask", "") or "",
        "tok": int(rec.get("decoded_token_num", -1)),
        "g": g,
    }
    if n_cand is not None:
        out["n"] = n_cand
    return out


def _emit(rec, out, *, keep_len):
    """Compact + write one raw record; return 1 if written, -1 if off-len, 0 if empty."""
    if keep_len is not None and int(rec.get("context_len", -1)) != keep_len:
        return -1
    comp = _compact(rec)
    if comp is None:
        return 0
    out.write(json.dumps(comp, separators=(",", ":")))
    out.write("\n")
    return 1


def sample_random(paths, out, *, per_file, seed, keep_len=None, max_tries_mult=20):
    """Random-byte-offset sample up to ``per_file`` distinct records per file; write compact lines.

    Records are de-duped per file by (doc_id, decoded_token_num) so re-landing on the same line does
    not double count. ``keep_len`` (if given) drops records whose ``context_len`` differs -- a safety
    filter against a stray glob. Returns (n_written, n_dropped_len).
    """
    rng = random.Random(seed)
    files = [(p, os.path.getsize(p)) for p in paths if os.path.getsize(p) > 0]
    n_written = 0
    n_dropped_len = 0
    for p, fsize in files:
        seen = set()
        tries = 0
        cap = max_tries_mult * per_file
        with open(p, "rb") as fh:
            while len(seen) < per_file and tries < cap:
                tries += 1
                ln = _read_line_at(fh, rng.randrange(fsize))
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                key = (rec.get("doc_id"), rec.get("decoded_token_num"))
                if key in seen:
                    continue
                r = _emit(rec, out, keep_len=keep_len)
                if r == -1:
                    n_dropped_len += 1
                elif r == 1:
                    seen.add(key)
                    n_written += 1
        print(f"  {os.path.basename(p)}: kept {len(seen)} records ({tries} tries)", file=sys.stderr)
    return n_written, n_dropped_len


def _subtask_from_prefix(ln, limit=2048):
    """Read the top-level ``subtask`` string from a line's prefix (it precedes the giant ``layers``
    payload) without json-parsing the whole ~MB record. Returns str or None."""
    m = re.search(rb'"subtask":\s*"((?:[^"\\]|\\.)*)"', ln[:limit])
    return m.group(1).decode("utf-8", "replace") if m else None


def _int_from_prefix(ln, key, limit=2048):
    """Read a top-level integer field (doc_id / decoded_token_num) from a line's prefix. Returns int
    or None."""
    m = re.search(rb'"' + re.escape(key.encode()) + rb'":\s*(-?\d+)', ln[:limit])
    return int(m.group(1)) if m else None


def sample_balanced(paths, out, *, per_key, seed, keep_len=None, max_tries_mult=60):
    """Byte-offset sample up to ``per_key`` distinct records *per subtask* per file -- covers every
    RULER subtask instead of getting stuck in the first (huge) subtask block, as head-sampling does.
    Subtask is read from each candidate's prefix so rejected lines are never fully parsed. Returns
    (n_written, n_dropped_len)."""
    rng = random.Random(seed)
    files = [(p, os.path.getsize(p)) for p in paths if os.path.getsize(p) > 0]
    n_written = 0
    n_dropped_len = 0
    for p, fsize in files:
        counts = defaultdict(int)   # subtask -> kept
        seen = set()                # (doc, tok) dedup
        tries = 0
        stall = 0
        cap = max_tries_mult * per_key
        with open(p, "rb") as fh:
            while tries < cap:
                tries += 1
                ln = _read_line_at(fh, rng.randrange(fsize))
                if not ln:
                    continue
                sub = _subtask_from_prefix(ln)
                if sub is None or counts[sub] >= per_key:
                    stall += 1
                    if stall > 8000 and counts and all(v >= per_key for v in counts.values()):
                        break  # every subtask seen so far is full and we keep re-landing on them
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                key = (rec.get("doc_id"), rec.get("decoded_token_num"))
                if key in seen:
                    continue
                r = _emit(rec, out, keep_len=keep_len)
                if r == -1:
                    n_dropped_len += 1
                elif r == 1:
                    seen.add(key)
                    counts[sub] += 1
                    n_written += 1
                    stall = 0
        kept_by_sub = {k: v for k, v in sorted(counts.items())}
        print(f"  {os.path.basename(p)}: kept {sum(counts.values())} records over "
              f"{len(counts)} subtasks {kept_by_sub}", file=sys.stderr)
    return n_written, n_dropped_len


def sample_balanced_dense(paths, out, *, per_key, seed, keep_len=None, max_tokens_per_doc=10,
                          max_tries_mult=80):
    """Subtask-balanced sampling that keeps each landed example's *token run*.

    Like ``balanced`` but ``per_key`` counts DOCS per subtask, and for each kept doc we read forward
    from the landing to grab its consecutive decode-step records (a doc's tokens are contiguous in the
    file), up to ``max_tokens_per_doc``. This gives both subtask coverage (for Q1/Q3/Q5 averaging) and
    dense multiple-tokens-per-example (for Q2's decoded-token-gap curve), which the plain ``balanced``
    mode is too sparse for. The token cap keeps long-output subtasks (cwe/fwe emit ~30 tokens) from
    dominating I/O -- Q2 only uses small gaps and Q1/Q5 are per-record, so capping is lossless for the
    figures. Returns (n_written, n_dropped_len)."""
    rng = random.Random(seed)
    files = [(p, os.path.getsize(p)) for p in paths if os.path.getsize(p) > 0]
    n_written = 0
    n_dropped_len = 0
    for p, fsize in files:
        docs_kept = defaultdict(set)   # subtask -> set(doc_id)
        seen = set()                   # (doc, tok) dedup
        tries = 0
        stall = 0
        cap = max_tries_mult * per_key
        with open(p, "rb") as fh:
            while tries < cap:
                tries += 1
                off = rng.randrange(fsize)
                fh.seek(off)
                if off:
                    fh.readline()  # discard partial line we landed in
                ln = fh.readline()
                if not ln:
                    continue
                sub = _subtask_from_prefix(ln)
                doc = _int_from_prefix(ln, "doc_id")
                if sub is None or doc is None:
                    continue
                if doc in docs_kept[sub]:
                    continue  # already collected this doc
                if len(docs_kept[sub]) >= per_key:
                    stall += 1
                    if stall > 10000 and all(len(v) >= per_key for v in docs_kept.values()):
                        break
                    continue
                # walk forward over this doc's contiguous token run (capped)
                got = False
                n_tok = 0
                cur = ln
                while cur and n_tok < max_tokens_per_doc:
                    s2 = _subtask_from_prefix(cur)
                    d2 = _int_from_prefix(cur, "doc_id")
                    if s2 != sub or d2 != doc:
                        break
                    t2 = _int_from_prefix(cur, "decoded_token_num")
                    key = (doc, t2)
                    n_tok += 1
                    if key not in seen:
                        try:
                            rec = json.loads(cur)
                        except Exception:
                            rec = None
                        if rec is not None:
                            r = _emit(rec, out, keep_len=keep_len)
                            if r == -1:
                                n_dropped_len += 1
                            elif r == 1:
                                seen.add(key)
                                n_written += 1
                                got = True
                    cur = fh.readline()
                if got:
                    docs_kept[sub].add(doc)
                    stall = 0
        summary = {k: len(v) for k, v in sorted(docs_kept.items())}
        print(f"  {os.path.basename(p)}: kept {n_written} records over "
              f"{len(docs_kept)} subtasks (docs/sub {summary})", file=sys.stderr)
    return n_written, n_dropped_len


def sample_head(paths, out, *, per_file, keep_len=None):
    """Stream the first ``per_file`` records of each file (never seeks). Records are in decode order
    (doc 0 tok 1, doc 0 tok 2, ...), so head-sampling yields the *same* (doc, tok) keys across models
    -- required for the cross-model comparison (Q4) -- and, with a large enough ``per_file``, walks
    through every RULER subtask. Returns (n_written, n_dropped_len)."""
    files = [p for p in paths if os.path.getsize(p) > 0]
    n_written = 0
    n_dropped_len = 0
    for p in files:
        kept = 0
        with open(p) as fh:
            for line in fh:
                if kept >= per_file:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                r = _emit(rec, out, keep_len=keep_len)
                if r == -1:
                    n_dropped_len += 1
                elif r == 1:
                    kept += 1
                    n_written += 1
        print(f"  {os.path.basename(p)}: kept {kept} records (head)", file=sys.stderr)
    return n_written, n_dropped_len


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="gate-log files / globs (one context length's worker files)")
    ap.add_argument("--out", required=True, help="compact JSONL output path")
    ap.add_argument("--len", type=int, default=None,
                    help="expected context_len; records with a different len are dropped (safety filter)")
    ap.add_argument("--per-file", type=int, default=800, help="max records to keep per worker file")
    ap.add_argument("--mode", choices=["head", "random", "balanced", "balanced-dense"], default="head",
                    help="head: first N records (decode order; dense tokens, but only the first "
                         "subtask). balanced: --per-key records per subtask (all subtasks, sparse per "
                         "example -> Q3/Q4). balanced-dense: --per-key DOCS per subtask, each with its "
                         "full token run (all subtasks AND dense tokens -> Q1/Q2/Q5). random: plain "
                         "byte-offset. Both checkpoints share doc ids, so balanced* keys overlap "
                         "cross-model.")
    ap.add_argument("--per-key", type=int, default=80,
                    help="balanced: records/subtask/file; balanced-dense: docs/subtask/file")
    ap.add_argument("--max-tokens-per-doc", type=int, default=10,
                    help="balanced-dense: cap decode-step records kept per doc (Q2 needs only small gaps)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = _paths(args.paths)
    if not paths:
        print(f"NO FILES for {args.paths}", file=sys.stderr)
        sys.exit(3)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with open(args.out, "w") as w:
        if args.mode == "head":
            n_total, n_dropped_len = sample_head(paths, w, per_file=args.per_file, keep_len=args.len)
        elif args.mode == "balanced":
            n_total, n_dropped_len = sample_balanced(
                paths, w, per_key=args.per_key, seed=args.seed, keep_len=args.len
            )
        elif args.mode == "balanced-dense":
            n_total, n_dropped_len = sample_balanced_dense(
                paths, w, per_key=args.per_key, seed=args.seed, keep_len=args.len,
                max_tokens_per_doc=args.max_tokens_per_doc,
            )
        else:
            n_total, n_dropped_len = sample_random(
                paths, w, per_file=args.per_file, seed=args.seed, keep_len=args.len
            )

    print(f"wrote {n_total} records -> {args.out}"
          + (f"  (dropped {n_dropped_len} off-len)" if n_dropped_len else ""))


if __name__ == "__main__":
    main()
