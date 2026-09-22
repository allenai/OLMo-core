#!/usr/bin/env python3
"""Grow a CTC-suite eval rung to a longer context, for the length-generalization study.

WHY NOT REUSE THE EXISTING xlong FILES. `contradiction/rung_131072.jsonl` already exists, and its
gold pairs are a proper nested superset of `rung_32768.jsonl` -- but its distractors average
**47.1 tokens/doc against 15.6 for every rung from 2k to 32k**. It was built from a filler glob
that also matched the FEVER and wiki_mix corpora (the leak documented in
`src/corpus_reasoning/data/build_xlong_rungs.py` and the `contra-fever-filler-leak` record), so it
is a different document distribution, not a longer version of the same ladder. Scoring it next to
the 2k-32k rungs would read as a length effect when part of it is a corpus change. This script
therefore builds long rungs from **each task's own rung file**, so the filler pool is by
construction the same corpus at the same document length.

WHY THE TARGET IS MEASURED, NOT LABELLED -- AND WHAT "MEASURED" HAS TO MEAN.

⚠ THE FIRST VERSION OF THIS SCRIPT CALIBRATED ON THE WRONG QUANTITY. It measured the tokens in the
raw document text joined by newlines (15.8 tok/doc for contradiction) and solved
``n_docs = target / tok_per_doc``. But the eval never feeds the model raw joined text: it renders an
Alpaca-style chat prompt with per-document markers, indices and the task instruction, which costs
**~44 tok/doc**. So the "65536" rung came out at a measured

    p10=154091  median=183008  p90=186540  max=325137     (2.79x overshoot)

with 1 of 200 examples already past Qwen3.5's 262144 position ceiling -- and the "131072" rung would
have been ~366k, entirely out of ceiling. Both produce a fake collapse that reads as a length
effect.

`measure_rung_tokens.py` shares the same raw-text flaw, so its "contradiction's rungs carry ~1.5x
fewer tokens than their label, niah's ~2.9x fewer" should not be quoted. But do NOT replace it with
the opposite claim either: measured through the real prompt path, `contradiction_iid/rung_32768` is
**762 docs -> prefill p50 33860 (p90 34883, max 35052)**, i.e. the existing rung labels are
ACCURATE to +3.3%. The shipped 2k-32k ladder needs no re-labelling; only this script's own
extrapolation beyond it was wrong.

The fix, which `src/corpus_reasoning/data/build_xlong_rungs.py` already documents: calibrate through
the REAL eval load path -- ``load_unified_examples`` -> the task's ``build_eval_prefill`` -> the
Qwen3.5 chat template -- i.e. exactly what ``build_prefills_any.py`` does at eval time. This script
now does that, and it does it ITERATIVELY: it fits ``prefill_len = a + b * n_docs`` from two real
probes, solves for the target, then BUILDS and RE-MEASURES, refining once if the realized median is
off by more than ``--tol``. The reported number is the realized measurement, never the fit.

Every emitted file carries its realized p50/p90/max in ``_measured_prefill_tokens`` and is named for
its realized median, so a `64k` file here is a real 64k prompt. Files whose p99 would exceed the
262144 ceiling are refused unless ``--allow-over-ceiling`` is passed (they need a YaRN serving copy;
see the run-evals YaRN table).

Construction (mirrors the "keep_all + self_nongold" mode of build_xlong_rungs.py):
  * every document of the source example is KEPT, so each output example is a strict nested
    superset of the source example -- same gold, same original hard negatives, length is the only
    variable;
  * distractors are drawn from the pool of documents that are non-gold in EVERY example of the
    source file (a doc that is gold anywhere is never injected, which would plant a stray answer);
  * documents are shuffled with a per-example seed and every index field is remapped.

    python debug/ctc_modelscale/expand_ctc_rung.py --task contradiction \\
        --src /scratch/.../eval_rungs/contradiction/rung_32768.jsonl \\
        --targets 65536,131072 --out-dir /scratch/.../eval_rungs_xlong/contradiction
"""
import argparse
import json
import os
import random
import statistics
import sys

#: The repo's `src/` -- PrefillMeter imports the SAME evaluator modules build_prefills_any.py uses,
#: which is the whole basis of the calibration being trustworthy.
REPO_SRC = os.environ.get(
    "CTC_REPO_SRC",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"
    ),
)

TOKENIZER = "Qwen/Qwen3.5-0.8B-Base"

#: Per-task gold-index conventions. ``base`` is the index origin of ``gold_doc_indices``;
#: ``pairs`` marks gold stored as [[a,b],...] rather than a flat list.
#: ``eval_alias`` is the string the EVAL catalog knows, when it differs from our short task name.
#: `niah` is ours; the catalog only has `niah_contradiction` (-> retrieval). Passing the short name
#: lands in run_rung_eval's "not in [...]" fast-fail, which is how the first build of these rungs
#: dropped niah entirely.
#: ⚠ ``base`` COMES FROM THE GRADER, NOT FROM INSPECTION. The authority is
#: ``src/corpus_reasoning/eval/evaluate.py``:
#:   * retrieval family (niah / nq / hotpotqa / msmarco) -- ``gold_ids = set(g + 1 for g in
#:     gold_doc_indices)`` at :1012/:1574/:1939, i.e. gold is stored **0-indexed** and the grader
#:     shifts it to the 1-based document numbering the prompt shows the model.
#:   * contradiction -- ``gold = ex["gold_doc_indices"]  # list of [a, b] pairs (1-indexed)`` at
#:     :1806, no shift, i.e. gold is stored **1-indexed**.
#: niah was set to 1 here originally. That silently shifted every gold by one in the built rungs and
#: scored the 64k niah rung at gold_id_f1 = 0.002 -- which read as a total long-context collapse and
#: is nothing of the kind. See detect_base() for why the range check did not catch it.
#: ``per_doc_fields`` are arrays PARALLEL to ``documents`` (one entry per document). They must be
#: permuted with the documents and extended for injected fillers, or they silently point at the
#: wrong document -- the same class of corruption as a wrong index base, and just as invisible.
#: ``ce_scores`` is load-bearing for rerank: ``_eval_rerank`` (evaluate.py:2052) builds its NDCG@10
#: graded relevance as ``gain = sigmoid(CE)`` and HARD-FAILS if the field is missing entirely. Its
#: docstring specifies "random fill = 0", so injected distractors take ``None`` and land on that
#: path -- i.e. treated as irrelevant, which is what a random non-gold document is. Kendall tau is
#: computed "over the scored docs" only, so it is unaffected by the fillers.
#: ⚠ ``gold_semantics`` DECIDES WHETHER THIS TOOL MAY TOUCH A TASK AT ALL.
#:
#: Injecting non-gold distractors is only sound when gold is defined by a RELATION TO SOMETHING
#: SPECIFIC -- a query to match, or a particular partner document. It is UNSOUND whenever gold is
#: defined by the ABSENCE of a property across the whole corpus, or by a STRUCTURE over all
#: documents, because then a freshly injected document SATISFIES the gold condition and becomes a
#: true positive that is not in the label.
#:
#: This was learned by shipping it wrong. These rungs were built and evaluated before the rule
#: existed, and every one of them scored at the floor -- not because the model failed, but because
#: the labels did:
#:     absence   ladder 0.986 @8k  ->  0.0007 @66k    (1116 injected docs are all genuinely absent)
#:     outlier   ladder 0.428 @32k ->  0.069  @65k    (injected docs also "don't belong")
#:     cycle                       ->  0.0    @68k
#:     textgroups                  ->  0.005  @65k
#: xabsence was refused for the same reason (a lone injected doc has no twin, so it IS an orphan),
#: which is what exposed the general rule -- the others should have been refused with it.
#:
#:   "query_match" -- gold answers a query / matches a needle. Injection is SAFE: a random
#:                    document does not answer the query.
#:   "pairwise"    -- gold is a relation between two SPECIFIC documents (contradiction pairs).
#:                    Injection is safe in practice, with a residual risk that an injected document
#:                    happens to contradict an existing one; this is the mode build_xlong_rungs.py
#:                    has always used for contradiction.
#:   "absence"     -- gold = documents lacking a counterpart/mention. REFUSED.
#:   "structural"  -- gold = outlier / cluster / cycle / ordering over the whole corpus. REFUSED.
#:
#: A REFUSED task is not un-extendable in principle -- it must be REGENERATED at a larger n by its
#: own generator, which recomputes the gold, rather than padded after the fact.
TASKS = {
    "contradiction": {"base": 1, "pairs": True, "index_fields": [], "gold_semantics": "pairwise"},
    "niah": {
        "base": 0,
        "pairs": False,
        "index_fields": [],
        "eval_alias": "niah_contradiction",
        "gold_semantics": "query_match",
    },
    "nq": {
        "base": 0,
        "pairs": False,
        "index_fields": ["hard_neg_indices"],
        "gold_semantics": "query_match",
    },
    "hotpotqa": {
        "base": 0,
        "pairs": False,
        "index_fields": ["hard_neg_indices"],
        "gold_semantics": "query_match",
    },
    # fiqa / scifact / obliq_twitter: added 2026-08-14 for the 1M-ladder build. Schema verified
    # identical to nq (gold_doc_indices flat 0-based -- fiqa's rung_2048 contains a literal 0 --
    # plus hard_neg_indices, answers). All three are catalog retrieval-family; "retrieval" is the
    # canonical TASK_CFG key, so the alias skips the per-source alias table entirely.
    "fiqa": {
        "base": 0,
        "pairs": False,
        "index_fields": ["hard_neg_indices"],
        "eval_alias": "retrieval",
        "gold_semantics": "query_match",
    },
    "scifact": {
        "base": 0,
        "pairs": False,
        "index_fields": ["hard_neg_indices"],
        "eval_alias": "retrieval",
        "gold_semantics": "query_match",
    },
    "obliq_twitter": {
        "base": 0,
        "pairs": False,
        "index_fields": ["hard_neg_indices"],
        "eval_alias": "retrieval",
        "gold_semantics": "query_match",
    },
    "msmarco": {
        "base": 0,
        "pairs": False,
        "index_fields": ["hard_neg_indices"],
        "per_doc_fields": {"ce_scores": None},
        "gold_semantics": "query_match",
    },
    "rerank": {
        "base": 0,
        "pairs": False,
        "index_fields": ["hard_neg_indices"],
        "per_doc_fields": {"ce_scores": None},
        "gold_semantics": "query_match",
    },
    "absence_gutenberg": {
        "base": 0,
        "pairs": False,
        "index_fields": [],
        "eval_alias": "absence",
        "gold_semantics": "absence",
    },
    # ---- O(NM) and O(N^3), added for the high-vs-low-CTC length-generalization claim ----
    # `pairs` really means "gold is NESTED"; the remap handles groups of ANY size, not just 2, so
    # it covers grouping's clusters and cycle's cycles as well as contradiction's pairs.
    # Each base below is the GRADER's, cross-checked against the data:
    #   outlier        _eval_outlier:1574  `set(int(g) + 1 ...)`            -> 0  (data contains 0)
    #   grouping       _eval_grouping:1632 `[[i + 1 for i in c] ...]`       -> 0  (data contains 0)
    #   cycle          _eval_cycle:2242    gold used RAW, no shift          -> 1  (max == n_docs)
    #   textgroups     same clustered shape as grouping, gold min 1         -> 1  (max == n_docs)
    # The data check is decisive in both directions here: a 0 in the data rules out base 1, and
    # max == n_docs rules out base 0. Unlike niah, neither of these is ambiguous.
    "outlier": {"base": 0, "pairs": False, "index_fields": [], "gold_semantics": "structural"},
    "outlier_amzn": {"base": 0, "pairs": False, "index_fields": [], "eval_alias": "outlier_amazon"},
    "grouping": {"base": 0, "pairs": True, "index_fields": [], "gold_semantics": "structural"},
    "textgroups": {"base": 1, "pairs": True, "index_fields": [], "gold_semantics": "structural"},
    "cycle": {"base": 1, "pairs": True, "index_fields": [], "gold_semantics": "structural"},
    # xabsence (O(N^2)) shares _eval_absence with absence (evaluate.py:1441 routes both there,
    # and :1939 does `int(g) + 1`), so base 0. ⚠ THE DATA CANNOT CONFIRM IT: gold spans 1..668
    # over 669 documents, so BOTH bases are in range -- the same ambiguity that let niah ship
    # with a wrong base. The grader is the authority here, and absence_gutenberg (same scorer)
    # proves the convention by containing a literal 0. detect_base() will say so out loud.
    # VERIFY AFTER BUILDING: if the eval scores near zero while the source ladder is healthy,
    # that is the off-by-one signature, not length generalization -- flip to base 1.
    "xabsence": {"base": 0, "pairs": False, "index_fields": [], "gold_semantics": "absence"},
}


def doc_text(d):
    if isinstance(d, dict):
        t = d.get("title") or ""
        return (t + " " + d.get("text", "")).strip() if t else d.get("text", "")
    return str(d)


def flat_gold(row, cfg):
    """Gold indices of one row as a flat list of 0-based positions into ``documents``."""
    g = row.get("gold_doc_indices") or []
    flat = []
    for x in g:
        flat.extend(x if isinstance(x, (list, tuple)) else [x])
    return [i - cfg["base"] for i in flat]


def detect_base(rows, cfg):
    """Check the configured index base against the data, and REFUSE to guess when it is ambiguous.

    ⚠ THE ORIGINAL VERSION OF THIS FUNCTION GAVE FALSE CONFIDENCE. It tried the configured base
    first and returned it if every gold index landed in range -- but "in range" does not
    discriminate. niah's gold spans 2..738 over 740 documents, so BOTH bases are in range, and the
    wrong configured value (1) was accepted and reported as "verified". Every gold in the built
    rungs was then off by one, and the eval scored 0.002, which reads as a long-context collapse.

    The self-check downstream did not catch it either: it compared gold TEXT before and after the
    remap using the SAME cfg on both sides, so a wrong base cancels out and the file verifies as
    internally consistent while being uniformly wrong.

    So: the base is a fact about the GRADER (see the TASKS table), not something to infer. This
    function now only rejects a configured base that is impossible, and hard-fails on ambiguity so
    the answer has to come from the grader rather than from a coin flip.
    """
    ok = {}
    for base in (0, 1):
        probe = dict(cfg, base=base)
        ok[base] = all(all(0 <= i < len(r["documents"]) for i in flat_gold(r, probe)) for r in rows)
    cfg_base = cfg["base"]
    if not ok[cfg_base]:
        raise SystemExit(
            f"FATAL: configured base={cfg_base} puts gold indices out of range; "
            f"base={1 - cfg_base} would fit. Check the grader convention in "
            f"src/corpus_reasoning/eval/evaluate.py before changing anything."
        )
    if ok[0] and ok[1]:
        # Ambiguous by range alone -- this is the niah case. Trust the table (grader-derived) and
        # say so loudly, rather than implying the data confirmed it.
        print(
            f"[expand] index base = {cfg_base} (from the GRADER convention; both bases are "
            f"in-range for this file, so the data cannot confirm it)",
            flush=True,
        )
    else:
        print(
            f"[expand] index base = {cfg_base} (confirmed: base={1 - cfg_base} is out of range)",
            flush=True,
        )
    return cfg_base


#: Qwen3.5's native position ceiling. A prompt past this is not an in-ceiling measurement and
#: produces a fake collapse unless the checkpoint is served through a YaRN copy.
POSITION_CEILING = 262_144

#: Reserved document markers, matching build_prefills_any.py's defaults.
DOC_START_ID = 248_049
DOC_END_ID = 248_050


def expand_rows(rows, pool, need, cfg, seed):
    """Grow every row by ``need`` non-gold fillers, remapping every index field.

    Factored out of the writer so calibration can build REAL expanded examples and measure their
    REAL prompt length, rather than estimating from a tokens-per-document constant.
    """
    out = []
    for ei, r in enumerate(rows):
        rng = random.Random(seed * 1_000_003 + ei)
        docs = list(r["documents"])
        extra = rng.sample(pool, need) if need > 0 else []
        # One position can carry SEVERAL tags: the same document may appear in two gold pairs, or
        # be both gold and a listed hard negative. A dict keyed by position would silently drop all
        # but the last tag and then KeyError on remap, so tags accumulate in a list per position.
        marks = [[] for _ in docs]
        for k, i in enumerate(flat_gold(r, cfg)):
            marks[i].append(("gold", k))
        for f in cfg["index_fields"]:
            for k, i in enumerate(r.get(f) or []):
                marks[i].append((f, k))
        # Carry any per-document parallel arrays through the shuffle ATTACHED to their document.
        # Permuting `documents` while leaving e.g. `ce_scores` in place would silently re-point
        # every score at the wrong document -- invisible to any check that only looks at gold.
        per = cfg.get("per_doc_fields") or {}
        for f, fill in per.items():
            vals = r.get(f)
            if vals is not None and len(vals) != len(docs):
                raise SystemExit(
                    f"FATAL: per-doc field {f!r} has len {len(vals)} but there are {len(docs)} "
                    f"documents -- refusing to guess the alignment"
                )
        tagged = [
            (d, marks[i], {f: (r.get(f) or [None] * len(docs))[i] for f in per})
            for i, d in enumerate(docs)
        ]
        # Injected fillers take the configured fill value. For ce_scores that is None, which is the
        # scorer's documented "random fill = 0" path (see the TASKS note).
        tagged += [(d, [], dict(per)) for d in extra]
        rng.shuffle(tagged)
        pos = {tag: j for j, (_, tags, _) in enumerate(tagged) for tag in tags}

        new = dict(r)
        new["documents"] = [d for d, _, _ in tagged]
        for f in per:
            if r.get(f) is not None:
                new[f] = [pv[f] for _, _, pv in tagged]
        g = r.get("gold_doc_indices") or []
        if cfg["pairs"]:
            k, remapped = 0, []
            for pair in g:
                # SORT after remapping. Documents are shuffled, so a pair that was (low, high) in
                # the source lands in arbitrary order here -- and contradiction is scored as a set
                # intersection over sorted pairs, so an unsorted pair can never be matched. Left
                # unsorted it costs recall on roughly half of all pairs while looking exactly like
                # a long-context capability collapse. (Sorting is meaning-preserving: a
                # contradiction pair is an unordered pair of documents. Tasks whose pairs ARE
                # ordered -- qdmatch's (query, doc) -- do not take this branch.)
                remapped.append(
                    sorted(pos[("gold", k + j)] + cfg["base"] for j in range(len(pair)))
                )
                k += len(pair)
            new["gold_doc_indices"] = remapped
        else:
            new["gold_doc_indices"] = [pos[("gold", k)] + cfg["base"] for k in range(len(g))]
        for f in cfg["index_fields"]:
            if r.get(f):
                new[f] = [pos[(f, k)] for k in range(len(r[f]))]
        new["source"] = r.get("source", "")
        out.append(new)
    return out


class PrefillMeter:
    """Measures REAL eval prompt length, through the exact path build_prefills_any.py uses.

    This is the whole point of the round-2 rewrite: the previous calibration measured raw document
    text and undershot the rendered prompt by 2.79x. Anything that estimates rather than renders is
    the same bug wearing a different constant.
    """

    def __init__(self, task, tokenizer, query_position, cot_mode):
        import tempfile

        from transformers import AutoTokenizer

        sys.path.insert(0, os.path.join(REPO_SRC))
        from corpus_reasoning.eval import eval_lc_native_docchunk as gen_eval
        from corpus_reasoning.eval import eval_lc_native_docchunk_contra as contra_eval
        from corpus_reasoning.eval.evaluate import load_unified_examples

        self._tempfile = tempfile
        self._load = load_unified_examples
        self._gen, self._contra = gen_eval, contra_eval
        self.task = task
        # Resolve the catalog alias EXACTLY as build_prefills_any.py does. `niah`, `nq` and
        # `hotpotqa` are catalog aliases, not TASK_CFG keys -- passing one straight through lands in
        # the "not in [...]" fast-fail (which is how the round-3 chunked sweep died on all 8 arms
        # with `contradiction_iid`). Calibrating against a different canonical task than the eval
        # will use would silently measure the wrong prompt shape.
        import importlib.util

        rre_path = os.path.join(REPO_SRC, "scripts", "eval", "ctc_suite", "run_rung_eval.py")
        canonical = task
        if os.path.isfile(rre_path):
            spec = importlib.util.spec_from_file_location("_rre_cal", rre_path)
            rre = importlib.util.module_from_spec(spec)
            sys.path.insert(0, os.path.dirname(rre_path))
            spec.loader.exec_module(rre)
            canonical = rre.TASK_ALIASES.get(task, task)
        canonical = gen_eval.TASK_ALIASES.get(canonical, canonical)
        if canonical not in gen_eval.TASK_CFG:
            raise SystemExit(f"--task {task!r} -> {canonical!r} not in {sorted(gen_eval.TASK_CFG)}")
        self.canonical = canonical
        self.is_contra = canonical == "contradiction"
        # cot_mode defaults to the task's own setting, matching the eval rather than overriding it.
        if cot_mode is None:
            cot_mode = gen_eval.TASK_CFG[canonical]["cot"]
        print(
            f"[calib] task={task!r} -> canonical={canonical!r} cot_mode={cot_mode!r} "
            f"query_position={query_position!r}",
            flush=True,
        )
        self.query_position = query_position
        self.cot_mode = cot_mode
        self.tok = AutoTokenizer.from_pretrained(tokenizer)

    def lengths(self, examples):
        """Prefill token lengths for a list of raw example dicts."""
        with self._tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            for e in examples:
                fh.write(json.dumps(e) + "\n")
            path = fh.name
        try:
            loaded = self._load(
                path,
                len(examples),
                task=self.canonical,
                query_position=self.query_position,
                use_alpaca=True,
            )
            lens = []
            for ex in loaded:
                raw = ex.get("ex", ex)
                builder = self._contra if self.is_contra else self._gen
                # doc_start_id / doc_end_id are REQUIRED keyword-only args and they are not
                # cosmetic: they are the per-document marker tokens the builder emits around every
                # document, so they are part of what makes the rendered prompt cost ~44 tok/doc
                # instead of the raw text's ~16. Defaults mirror build_prefills_any.py.
                kwargs = dict(
                    variant="full",
                    cot_mode=self.cot_mode,
                    doc_start_id=DOC_START_ID,
                    doc_end_id=DOC_END_ID,
                )
                if self.is_contra:
                    prefill = builder.build_eval_prefill(self.tok, raw, **kwargs)
                else:
                    prefill = builder.build_eval_prefill(self.tok, raw, self.canonical, **kwargs)
                lens.append(len(prefill))
            return lens
        finally:
            os.unlink(path)


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--src", required=True, help="source rung JSONL to grow")
    ap.add_argument(
        "--targets",
        required=True,
        help="comma-separated target MEDIAN PREFILL token budgets (real prompt length)",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--calib",
        type=int,
        default=24,
        help="examples rendered per calibration probe (real prompts, so keep it small)",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=0.10,
        help="accept a build when |realized_median/target - 1| <= tol",
    )
    ap.add_argument("--max-refine", type=int, default=2, help="rebuild attempts after the fit")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument(
        "--query-position",
        default="both",
        help="MUST match how the eval will render (see the query_position record)",
    )
    ap.add_argument(
        "--cot-mode",
        default=None,
        help="default: the task's own TASK_CFG cot setting, i.e. what the eval uses",
    )
    ap.add_argument(
        "--name-as-target",
        action="store_true",
        help="name files rung_<TARGET> (the shipped-ladder convention the eval "
        "drivers glob for) instead of rung_<realized median>",
    )
    ap.add_argument(
        "--allow-over-ceiling",
        action="store_true",
        help=f"emit files whose p99 exceeds {POSITION_CEILING} (needs a YaRN copy)",
    )
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    cfg = dict(TASKS[args.task])
    rows = [json.loads(l) for l in open(args.src)]
    print(f"[expand] {args.task}: {len(rows)} examples from {args.src}", flush=True)

    cfg["base"] = detect_base(rows, cfg)

    # --- distractor pool: docs that are gold in NO example ---
    gold_texts = set()
    for r in rows:
        for i in flat_gold(r, cfg):
            gold_texts.add(doc_text(r["documents"][i]))
    pool, seen = [], set()
    for r in rows:
        for d in r["documents"]:
            t = doc_text(d)
            if t in gold_texts or t in seen:
                continue
            seen.add(t)
            pool.append(d)
    print(
        f"[expand] pool={len(pool)} distinct non-gold docs "
        f"({len(gold_texts)} gold texts excluded)",
        flush=True,
    )

    # --- calibrate on REAL rendered prompts (see the module docstring for why) ---
    meter = PrefillMeter(
        cfg.get("eval_alias", args.task), args.tokenizer, args.query_position, args.cot_mode
    )
    calib_rows = rows[: args.calib]
    src_docs = int(statistics.median([len(r["documents"]) for r in rows]))

    def probe(need):
        """Realized prefill lengths when every example is grown by `need` fillers."""
        return meter.lengths(expand_rows(calib_rows, pool, need, cfg, args.seed))

    base_lens = probe(0)
    base_med = statistics.median(base_lens)
    print(
        f"[expand] source: {src_docs} docs -> REAL prefill median={base_med:.0f} "
        f"(p90={pct(base_lens, 0.9)} max={max(base_lens)})",
        flush=True,
    )

    # Two-point fit of prefill_len = a + b*n_docs. The intercept `a` is the task instruction and
    # query -- real and non-trivial -- which is precisely what a flat tokens/doc constant ignores.
    probe_need = max(1, src_docs)
    hi_lens = probe(probe_need)
    hi_med = statistics.median(hi_lens)
    b = (hi_med - base_med) / probe_need
    a = base_med - b * src_docs
    print(
        f"[expand] fit: prefill ~ {a:.0f} + {b:.2f}*n_docs  "
        f"(probe {src_docs}->{base_med:.0f}, {src_docs + probe_need}->{hi_med:.0f})",
        flush=True,
    )
    if b <= 0:
        raise SystemExit("FATAL: non-positive tokens-per-doc slope -- calibration is broken")

    os.makedirs(args.out_dir, exist_ok=True)
    for target in [int(t) for t in args.targets.split(",")]:
        # Seed from the fit, then MEASURE the real build and refine. The fit is only a starting
        # guess; the number that gets reported and used is always a realized measurement.
        need = int(round((target - a) / b)) - src_docs
        realized = None
        for attempt in range(args.max_refine + 1):
            if need <= 0:
                print(f"[expand] {target}: source already renders to >= target, skip")
                need = None
                break
            if need > len(pool):
                print(
                    f"[expand] {target}: SKIP -- needs {need} fillers/example, "
                    f"pool has {len(pool)}"
                )
                need = None
                break
            lens = probe(need)
            med = statistics.median(lens)
            err = med / target - 1.0
            print(
                f"[expand] {target}: attempt {attempt} need={need} "
                f"-> realized median={med:.0f} ({err:+.1%})",
                flush=True,
            )
            realized = lens
            if abs(err) <= args.tol:
                break
            # Re-solve on the measured slope rather than nudging blindly.
            need = int(round(need + (target - med) / b))
        if need is None:
            continue
        if realized is None or abs(statistics.median(realized) / target - 1.0) > args.tol:
            print(
                f"[expand] {target}: FAILED to land within {args.tol:.0%} -- not writing a file "
                f"whose name would lie about its length"
            )
            continue

        # Doc count per example is NOT constant for every task: nq's rung_8192 ranges 5..48
        # documents (25 distinct counts) while contradiction's is a flat 762. So the invariant is
        # "every example grew by exactly `need`", not "every example has the same total" --
        # asserting the latter is what failed the nq build after the file was already written.
        doc_counts = [len(r["documents"]) + need for r in rows]
        full_docs = (
            f"{min(doc_counts)}..{max(doc_counts)}"
            if min(doc_counts) != max(doc_counts)
            else str(doc_counts[0])
        )

        # Ceiling gate. A prompt past 262144 is not an in-ceiling measurement; served without YaRN
        # it produces a fake collapse that reads exactly like a length-generalization result.
        p99 = pct(realized, 0.99)
        if p99 > POSITION_CEILING and not args.allow_over_ceiling:
            print(
                f"[expand] {target}: REFUSED -- p99={p99} exceeds the {POSITION_CEILING} "
                f"position ceiling. Lower the target, or pass --allow-over-ceiling and serve "
                f"through a YaRN copy (factor 2 for 256k/512k)."
            )
            continue

        med = int(statistics.median(realized))
        out = os.path.join(args.out_dir, f"rung_{target if args.name_as_target else med}.jsonl")
        expanded = expand_rows(rows, pool, need, cfg, args.seed)
        stats = {
            "p50": med,
            "p90": pct(realized, 0.9),
            "max": max(realized),
            "calib_examples": len(realized),
            "n_docs": full_docs,
            "query_position": args.query_position,
            "cot_mode": args.cot_mode,
        }
        with open(out, "w") as fh:
            for r_src, new in zip(rows, expanded):
                new["_expanded_from"] = os.path.basename(args.src)
                new["_target_prefill_tokens"] = target
                new["_measured_prefill_tokens"] = stats
                fh.write(json.dumps(new) + "\n")

        # Verify the written file round-trips and that no gold document moved or vanished.
        chk = [json.loads(l) for l in open(out)]
        assert len(chk) == len(rows), f"{out}: wrote {len(chk)} of {len(rows)}"
        for src_row, new_row in zip(rows, chk):
            ga = sorted(doc_text(src_row["documents"][i]) for i in flat_gold(src_row, cfg))
            gb = sorted(doc_text(new_row["documents"][i]) for i in flat_gold(new_row, cfg))
            assert ga == gb, f"{out}: gold text changed after remap"
            assert (
                len(new_row["documents"]) == len(src_row["documents"]) + need
            ), f"{out}: example grew by {len(new_row['documents']) - len(src_row['documents'])}, expected {need}"
        print(f"[expand] WROTE {out}", flush=True)
        print(
            f"           realized prefill p50={med} p90={stats['p90']} max={stats['max']} "
            f"| {full_docs} docs (+{need}/ex) | gold preserved on all {len(chk)} examples",
            flush=True,
        )


if __name__ == "__main__":
    main()
