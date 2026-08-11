#!/usr/bin/env python3
"""
Build the **v2** long-context eval ladders for contra / nq / outlier / rerank.

Goal (user request): every length rung of a task must share the SAME
questions/answers; only the *distractor* documents change to hit the target
context length. This makes rung-to-rung comparisons clean (length is the only
variable) and makes different corpora comparable. Every rung has >=500 examples.

How: pick one canonical question set per task (the largest existing rung, which
already carries every gold document + the biggest distractor pool, and has
>=500 examples), then DERIVE each shorter rung by keeping the gold docs +
query + answer fixed and taking a *nested prefix* of the (shuffled) distractor
pool. contra's canonical base (contra_base500, 50 docs/ex) is smaller than its
rungs, so its distractor pool is topped up with fresh PubMed filler sentences
harvested offline from the existing contra files (no LLM / no network).

Per example the pipeline is:
  1. split documents into GOLD (must-keep) vs DISTRACTOR (droppable),
  2. build a single deterministic distractor order (+ harvested fillers for contra),
  3. for each rung n: gold + first (n - |gold|) distractors, shuffle the combined
     list, remap every index field (gold_doc_indices, hard_neg_indices) and
     recompute any position-derived answer string (outlier).

Nested-prefix => the shorter rung's distractors are a subset of the longer
rung's; the gold set / query / answer are byte-identical across rungs.

Outputs to  $OUT_ROOT/<task>/<canonical-name-with-n{rung}>.jsonl  (default
/scratch/users/prasann/cpt_data/eval500_v2). Old files are untouched.

    python scripts/data/build_v2_eval_ladders.py            # build all 4 tasks
    python scripts/data/build_v2_eval_ladders.py --tasks contra,nq
"""
import argparse
import glob
import json
import os
import random

DATA = "/scratch/users/prasann/corpus-reasoning/data"
E5 = "/scratch/users/prasann/cpt_data/eval500"
OUT_ROOT_DEFAULT = "/scratch/users/prasann/cpt_data/eval500_v2"

# Per-task config. `rungs` maps rung-label -> total #docs/example (calibrated
# historically: ~43 Qwen tok/doc for contra, doc lengths per corpus otherwise).
# `out_tmpl` uses {n} for the doc count in the filename (matches the existing
# naming so the eval runner's ladder spec stays legible).
TASKS = {
    "contra": {
        "canonical": f"{DATA}/contra_base500.jsonl",
        "mode": "expand",                       # base (50 docs) < rungs -> add fillers
        "gold_field": "gold_doc_indices",       # list of PAIRS [[a,b],...]
        "gold_is_pairs": True,
        "index_base": 1,                        # contra gold is 1-INDEXED (see
                                                # generate_pubmed_contradiction_data.py);
                                                # normalize to 0-indexed internally, write back 1-indexed.
        # 2026-07-19 re-rung 3k/8k/16k/32k -> 2k/4k/8k (CTC suite Stage-1 fixed eval rungs;
        # 16k/32k deferred per BUILD_MATRIX.md). Values from the row-16 n-ladder table.
        # 2026-07-19 FIX 2 (token calibration): the row-16 n-ladder values (40/88/190) were set
        # from a "doc text only, pre-wrap" token estimate and undershot the 2048/4096/8192 labels
        # by ~1.7-1.8x once the actual rendered prompt (instruction + query + box markers) is
        # measured. Recalibrated via tokenizer-measured tokens = 288.7 + 22.82*n_docs (Qwen3.5,
        # 50-example sample/rung, full prefill incl. wrap markers) -- see BUILD_MATRIX.md FIX 2.
        # 2026-07-19 Stage-5: extend to 16k/32k via the FIX-2 fit (288.7 + 22.82*n).
        # expand mode -> unbounded harvested filler pool, so large rungs are reachable.
        "rungs": {"2k": 77, "4k": 167, "8k": 346, "16k": 705, "32k": 1423},
        "out_tmpl": "contradiction_eval_pubmed_both_n{n}_k3.jsonl",
        # 2026-07-21 leak fix: PUBMED-only, was f"{DATA}/contradiction_*_k3.jsonl" (also matched
        # the FEVER/wiki_mix files). See harvest_fillers() docstring -- mixing in FEVER/wiki
        # decoy text was the dominant leak vector (small finite Wikipedia claim pool reused
        # verbatim as "distractor" text across dozens of examples, e.g. a SUPPORTS/REFUTES pair
        # about Julius Caesar's tenure recurred 43x as non-gold filler and never once as an
        # official gold pair -- so the gold-membership exclusion below could never catch it).
        # Restricting to pubmed also fixes a domain mismatch (pubmed corpus filled with
        # Wikipedia-entity trivia).
        "filler_glob": f"{DATA}/contradiction_*pubmed*_k3.jsonl",  # harvest non-gold texts
    },
    "contra_ctc": {
        # Same canonical/gold/filler semantics as "contra"; ONLY the rung->n_docs map differs.
        #
        # 2026-08-04 RE-CALIBRATION. The "contra" rungs above (77/167/346/705/1423) were fit
        # against a filler pool that was 92-99.6% FEVER/wiki_mix (the pre-2026-07-21 glob), whose
        # one-line Wikipedia trivia claims tokenize at ~22.8 tok/doc. Real PubMed claim sentences
        # are ~43 tok/doc, so re-running the SAME n against the fixed pubmed-only glob overshoots
        # every label by ~1.8x (measured: n=77 -> 3413 tok, not 2048; n=1423 -> 61461, not 32768).
        # Refit on the clean pool over 25 examples/rung, full rendered prefill:
        #     tokens = 170 + 42.8 * n_docs      (r^2 ~ 1.00 over n in [77, 1423])
        # -> n = (target - 170) / 42.8. Note this lands within a few docs of the ORIGINAL
        # BUILD_MATRIX row-16 ladder (40/88/190/385/765), which was calibrated on real PubMed and
        # was correct all along; "FIX 2" only looked necessary because the pool was contaminated.
        "canonical": f"{DATA}/contra_base500.jsonl",
        "mode": "expand",
        "gold_field": "gold_doc_indices",
        "gold_is_pairs": True,
        "index_base": 1,
        "rungs": {"2k": 44, "4k": 92, "8k": 187, "16k": 379, "32k": 762},
        "out_tmpl": "contradiction_eval_pubmed_both_n{n}_k3.jsonl",
        "filler_glob": f"{DATA}/contradiction_*pubmed*_k3.jsonl",
    },
    "hpqa": {
        # CTC suite HotpotQA fixed-eval rungs (BUILD_MATRIX.md row 2 / ACTION A2b). Canonical is
        # the n=205 bridge-question eval built on cubbins (500 rows, 20 hard negs/ex, 0-indexed
        # gold) -- shrink-derive 2k/4k/8k from it same as nq/rerank.
        "canonical": "/data/prasann/ctc_suite_data/hotpotqa/hotpotqa_eval_k205_bridge_hn20_500.jsonl",
        "mode": "shrink",
        "gold_field": "gold_doc_indices",
        "gold_is_pairs": False,
        "extra_index_fields": ["hard_neg_indices"],
        # 2026-07-19 FIX 2 (token calibration): 11/24/50 undershot 2048/4096/8192 by ~1.5-1.6x
        # (tokenizer-measured full prefill, not just doc text). Recalibrated via
        # tokens = 66.6 + 113.36*n_docs (Qwen3.5, 50-example sample/rung) -- see BUILD_MATRIX.md
        # FIX 2.
        # 2026-07-19 Stage-5: +16k via FIX-2 fit (66.6 + 113.36*n) -> n=144 (< canonical n205 OK).
        # 32k rung DEFERRED: it needs n=288 > canonical n205; requires regenerating the hpqa eval
        # canonical at n~290 (--split validation --num-docs 290) before it can shrink-derive.
        "rungs": {"2k": 17, "4k": 36, "8k": 72, "16k": 144},
        "out_tmpl": "hotpotqa_eval_bridge_hn20_n{n}_500.jsonl",
    },
    "nq": {
        # 2026-07-19 Stage-5: switched to the p10/CE-filtered validation (hard-ratio 0.097, from
        # the same clean pipeline as the train pool) per the CE-every-rung directive. Its doc counts
        # vary (p50=117, only 12 ex >=200), so clean rungs cap at 2k/4k/8k (48 docs: 525/600 qualify);
        # 16k/32k need a uniform-k200 CE eval regen (deferred, the expensive nq gen).
        "canonical": "/scratch/users/prasann/nq_p10_20k/nq_validation_k25-202_600.jsonl",
        "mode": "shrink",
        "gold_field": "gold_doc_indices",
        "gold_is_pairs": False,
        "extra_index_fields": ["hard_neg_indices"],
        # 2026-07-19 Stage-5: re-rung to 2k/4k/8k/16k/32k (~160 tok/doc; canonical k200_hn20 is
        # uniform 200 docs, hard-ratio 20/200=0.10 p10-consistent, text answers -> shrink-safe).
        "rungs": {"2k": 11, "4k": 23, "8k": 48},
        "out_tmpl": "nq_validation_k{n}_600.jsonl",
    },
    "outlier": {
        "canonical": f"{E5}/outlier/outlier_wiki100w_n220_k3_eval_600.jsonl",
        "mode": "shrink",
        "gold_field": "gold_doc_indices",
        "gold_is_pairs": False,
        "answers_from_gold": True,              # answers = "; ".join(1-indexed gold)
        # 2026-07-19 re-rung 3k/8k/16k/32k -> 2k/4k/8k (see contra note above).
        # 2026-07-19 FIX 2 (token calibration): CONFIRMED already within +-10% of the
        # 2048/4096/8192 labels (tokenizer-measured full prefill medians: 2158/4230/8490, i.e.
        # ratios 1.05/1.03/1.04) -- no change. See BUILD_MATRIX.md FIX 2.
        # 2026-07-19 Stage-5: +16k/32k via FIX-2 fit (114.1 + 146.93*n). 32k rung n=222 clamps to
        # canonical max n=220 (whole file, ~32.2k tok) -> set 220.
        "rungs": {"2k": 14, "4k": 28, "8k": 57, "16k": 111, "32k": 220},
        "out_tmpl": "outlier_wiki100w_n{n}_k3_eval_600.jsonl",
    },
    "rerank": {
        # CE-graded MSMARCO (has per-doc `ce_scores` -> NDCG@10 + Kendall-tau); the k20/k50/k100
        # files already share the same 500 queries, so k100 is the canonical. No 32k rung: no
        # CE-graded pool larger than k100 (~16k) exists (matches v1's rerank-CE branch: 3k/8k/16k).
        "canonical": f"{DATA}/msmarco_trainhn_eval_k100_500.jsonl",
        "mode": "shrink",
        "gold_field": "gold_doc_indices",
        "gold_is_pairs": False,
        "extra_index_fields": ["hard_neg_indices"],
        "has_ce": True,                          # ce_scores parallel array follows each doc
        # 2026-07-19 Stage-5: re-rung to 2k/4k/8k/16k (~100 tok/doc). Canonical is k100 so 16k=100
        # (whole canonical). 32k rung DEFERRED: needs a k~300 CE-scored eval pool (A9).
        "rungs": {"2k": 20, "4k": 40, "8k": 70, "16k": 100},
        "out_tmpl": "msmarco_trainhn_eval_k{n}_500.jsonl",
    },
    "scifact": {
        # BEIR SciFact retrieval (CE-cleaned). DATA-POOR: test set is only 300 queries -> eval_size
        # 300 (< 500, flag inline with SE ~= 0.026). Canonical k88 test, shrink-derive smaller rungs.
        "canonical": "/data/prasann/ctc_suite_data/scifact/final20k/beir_scifact_ce_test_k88_300.jsonl",
        "mode": "shrink",
        "gold_field": "gold_doc_indices",
        "gold_is_pairs": False,
        "extra_index_fields": ["hard_neg_indices"],
        "min_examples": 300,
        "rungs": {"2k": 5, "4k": 10, "8k": 21, "16k": 43, "32k": 88},
        "out_tmpl": "beir_scifact_eval_k{n}_300.jsonl",
    },
    "fiqa": {
        # BEIR FiQA retrieval (CE-cleaned). test 648 -> eval_size 500. Canonical k80 test, shrink.
        "canonical": "/data/prasann/ctc_suite_data/fiqa/final20k/beir_fiqa_ce_test_k80_648.jsonl",
        "mode": "shrink",
        "gold_field": "gold_doc_indices",
        "gold_is_pairs": False,
        "extra_index_fields": ["hard_neg_indices"],
        "rungs": {"2k": 4, "4k": 9, "8k": 19, "16k": 40, "32k": 80},
        "out_tmpl": "beir_fiqa_eval_k{n}_648.jsonl",
    },
}

SEED = 1234
CAP_PER_RUNG = 500   # exactly 500 examples/rung; canonical sets are row-aligned by example, so
                     # taking the first CAP rows keeps the SAME 500 questions in every rung.


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def normalize_to_zero_indexed(ex, cfg):
    """Convert a task's gold (and extra index fields) from its native 1-indexed
    convention to 0-indexed IN PLACE, so all internal indexing (docs[i], sanitize,
    remap) is uniform. contra is 1-indexed; nq/rerank are already 0-indexed
    (index_base defaults to 0 -> no-op). The output write adds index_base back."""
    base = cfg.get("index_base", 0)
    if base == 0:
        return
    field = cfg["gold_field"]
    if cfg.get("gold_is_pairs"):
        ex[field] = [[i - base for i in pair] for pair in ex[field]]
    else:
        ex[field] = [i - base for i in ex[field]]
    for f in cfg.get("extra_index_fields", []):
        if f in ex:
            ex[f] = [i - base for i in ex[f]]


def sanitize_gold(ex, cfg):
    """Drop gold entries that reference out-of-range documents (a latent bug in
    contra_base500: 52/500 examples have one contradiction partner truncated, so
    its index == ndocs). Returns (#dropped_entries). Pairs with any invalid index
    are dropped whole (the contradiction is unrecoverable without its partner);
    the example keeps its remaining valid pairs."""
    nd = len(ex["documents"])
    field = cfg["gold_field"]
    dropped = 0
    if cfg.get("gold_is_pairs"):
        clean = []
        for pair in ex[field]:
            if all(0 <= i < nd for i in pair):
                clean.append(pair)
            else:
                dropped += 1
        ex[field] = clean
    else:
        clean = [i for i in ex[field] if 0 <= i < nd]
        dropped = len(ex[field]) - len(clean)
        ex[field] = clean
    return dropped


def gold_index_set(ex, cfg):
    """Flat set of all document indices that are GOLD (must never be dropped)."""
    g = ex[cfg["gold_field"]]
    if cfg.get("gold_is_pairs"):
        return {i for pair in g for i in pair}
    return set(g)


def harvest_fillers(cfg, gold_texts_global):
    """Pool of distinct NEUTRAL filler documents for contra expansion, drawn from
    every existing contra file (NOT an LLM/network call).

    CRITICAL: a filler must never be a contradiction-member (gold) in ANY example
    of ANY contra file. The contra corpus is built by perturbation, so a doc that
    is a gold contradiction-member somewhere is a claim engineered to contradict —
    dropping it into another example injects a *real* contradiction that isn't in
    that example's gold, which the model then (correctly) finds -> f1/EM collapses.
    So we first collect every cross-file gold-member text and exclude all of them
    (plus the canonical's own gold). What remains are the neutral distractor slots
    (random PubMed sentences), matching v1's fresh-disjoint-filler behaviour.

    2026-07-21 leak fix (dense-contradiction investigation, see
    records/ dense-contra eval-data-leak notes): gold-membership exclusion alone is
    NOT sufficient. The generation pipelines create MORE perturbed/paired claims
    than get selected as a given example's k=3 gold — the surplus never becomes
    "gold" in ANY file we have on disk (confirmed by exhaustively scanning every
    contradiction_*.jsonl for two known leaked pairs: neither was gold anywhere),
    yet both halves of the pair are still real engineered contradictions that
    "cannot both be true at the same time" per the task's own instruction. When
    harvest draws both halves into the same rebuilt example, the model correctly
    flags them and gets scored wrong.

    We can't recover per-item perturbation provenance (the final k3.jsonl output
    only keeps doc text, no abstract-id / perturbed-flag / pair-id), so we use a
    DUPLICATE-COUNT heuristic as a proxy: a genuinely neutral real source sentence
    (e.g. one PubMed sentence out of >20M abstracts) essentially never recurs
    verbatim across the harvested files, whereas a reused synthetic claim does —
    confirmed on the two leaked pairs found: the FEVER "Julius Caesar" pair
    recurred 43x as non-gold filler (small finite Wikipedia claim pool reused as
    decoy text across dozens of examples) and a PubMed "PImax" pair recurred 4x
    (generated more than once across bulk train-data draws). Both are excluded by
    requiring EXACT-COUNT-1 (singleton) occurrence across every scanned file, in
    ANY role (gold or distractor) -- not just deduped to one pool entry as before.
    Combined with restricting filler_glob to pubmed-only (the dominant leak vector
    was the off-domain FEVER/wiki_mix decoy pool), this removes both leak classes
    found in the investigation."""
    files = sorted(glob.glob(cfg["filler_glob"]))
    # Pass 1: every text that is a gold contradiction-member in ANY file, AND the
    # occurrence count of every text (in any role) across every file.
    contradiction_members = set(gold_texts_global)
    occ_count = {}
    for path in files:
        try:
            rows = load_jsonl(path)
        except Exception:
            continue
        for ex in rows:
            docs = ex["documents"]
            gidx = gold_index_set(ex, cfg)
            for i, d in enumerate(docs):
                t = d.get("text", "")
                if not t:
                    continue
                occ_count[t] = occ_count.get(t, 0) + 1
                if i in gidx:
                    contradiction_members.add(t)
    # Pass 2: collect only never-a-gold-member, SINGLETON (count==1) distractor
    # docs -- a text seen more than once anywhere (gold or filler role) is treated
    # as a reused/paired synthetic claim, not a genuinely neutral one-off sentence.
    seen = set()
    pool = []
    for path in files:
        try:
            rows = load_jsonl(path)
        except Exception:
            continue
        for ex in rows:
            gidx = gold_index_set(ex, cfg)
            for i, d in enumerate(ex["documents"]):
                if i in gidx:
                    continue
                t = d.get("text", "")
                if not t or t in seen or t in contradiction_members:
                    continue
                if occ_count.get(t, 0) != 1:
                    continue
                seen.add(t)
                pool.append(d)
    return pool


def build_task(task, cfg, out_root, verbose=True):
    canon = load_jsonl(cfg["canonical"])
    if verbose:
        print(f"\n=== {task}: canonical {os.path.basename(cfg['canonical'])} "
              f"({len(canon)} examples, {len(canon[0]['documents'])} docs/ex) ===")
    # Normalize native 1-indexed gold (contra) to 0-indexed so all internal
    # indexing is uniform; the output write restores index_base.
    for ex in canon:
        normalize_to_zero_indexed(ex, cfg)
    # Sanitize out-of-range gold, then drop any example left with zero gold.
    tot_dropped = sum(sanitize_gold(ex, cfg) for ex in canon)
    before = len(canon)
    canon = [ex for ex in canon if ex[cfg["gold_field"]]]
    if verbose and (tot_dropped or before != len(canon)):
        print(f"    sanitized: dropped {tot_dropped} invalid gold entries; "
              f"{before - len(canon)} examples had zero valid gold (removed)")
    min_ex = cfg.get("min_examples", 500)
    assert len(canon) >= min_ex, f"{task}: only {len(canon)} valid examples < {min_ex}"
    if min_ex < 500:
        print(f"    ⚠ DATA-POOR eval_size={min_ex} (<500) for {task} — quote inline with SE")

    filler_pool = []
    if cfg["mode"] == "expand":
        gold_texts_global = set()
        for ex in canon:
            for i in gold_index_set(ex, cfg):
                gold_texts_global.add(ex["documents"][i].get("text", ""))
        filler_pool = harvest_fillers(cfg, gold_texts_global)
        max_n = max(cfg["rungs"].values())
        need = max_n - min(len(ex["documents"]) for ex in canon)
        if verbose:
            print(f"    harvested {len(filler_pool)} distinct filler docs "
                  f"(need ~{need} extra/ex for the largest rung)")
        assert len(filler_pool) >= need, "not enough harvested fillers"

    # Per-rung accumulators.
    rung_rows = {lab: [] for lab in cfg["rungs"]}

    for ei, ex in enumerate(canon):
        rng = random.Random(SEED * 1_000_003 + ei)
        docs = ex["documents"]
        has_ce = cfg.get("has_ce") and "ce_scores" in ex
        ce = ex.get("ce_scores", [None] * len(docs)) if has_ce else None
        gidx = sorted(gold_index_set(ex, cfg))
        gold_docs = [docs[i] for i in gidx]
        gold_ce = [ce[i] for i in gidx] if has_ce else None
        old2goldpos = {old: p for p, old in enumerate(gidx)}   # old idx -> position in gold_docs

        # distractor docs (droppable), in a fixed shuffled order. ce_scores (rerank
        # relevance) ride ALONGSIDE their doc so NDCG is preserved after subsetting.
        gset = set(gidx)
        if cfg["mode"] == "expand":
            # contra: the canonical base's OWN distractors are also contradiction
            # pipeline claims (~42% are gold-members elsewhere), so they'd inject
            # stray contradictions too. Discard them entirely and fill every
            # distractor slot from the clean (gold-member-excluded) harvest pool.
            distractors, distr_ce = [], ([] if has_ce else None)
            hard_flag = []
        else:
            distractors = [d for i, d in enumerate(docs) if i not in gset]
            distr_ce = [ce[i] for i in range(len(docs)) if i not in gset] if has_ce else None
            # tag which distractors are "hard negs" (nq) so we can preserve that flag.
            hard_old = set(ex.get("hard_neg_indices", []))
            hard_flag = [i in hard_old for i in range(len(docs)) if i not in gset]
        order = list(range(len(distractors)))
        rng.shuffle(order)
        distractors = [distractors[j] for j in order]
        hard_flag = [hard_flag[j] for j in order]
        if has_ce:
            distr_ce = [distr_ce[j] for j in order]

        if cfg["mode"] == "expand":
            # top up with harvested fillers (disjoint slice per example), so the
            # largest rung is reachable; nested prefixes give the shorter rungs.
            max_n = max(cfg["rungs"].values())
            need = max_n - len(gold_docs) - len(distractors)
            if need > 0:
                start = (ei * need) % max(1, len(filler_pool) - need)
                extra = filler_pool[start:start + need]
                if len(extra) < need:                    # wrap-around
                    extra = extra + filler_pool[:need - len(extra)]
                distractors = distractors + [dict(d) for d in extra]
                hard_flag = hard_flag + [False] * len(extra)
                if has_ce:
                    distr_ce = distr_ce + [0.0] * len(extra)

        for lab, n in cfg["rungs"].items():
            keep_distract = n - len(gold_docs)
            assert keep_distract >= 0, f"{task} rung {lab}: n={n} < gold={len(gold_docs)}"
            # Canonical examples can carry fewer docs than the nominal rung size
            # (nq/rerank pools vary per example); clamp so a short example keeps
            # ALL its distractors at the top rung (== the original file) and still
            # subsets cleanly at lower rungs. Gold/query/answer stay identical.
            keep_distract = min(keep_distract, len(distractors))
            chosen = distractors[:keep_distract]
            chosen_hard = hard_flag[:keep_distract]
            combined = [("G", d) for d in gold_docs] + [("D", d) for d in chosen]
            combined_ce = (gold_ce + distr_ce[:keep_distract]) if has_ce else None
            perm = list(range(len(combined)))
            rng2 = random.Random(SEED * 7 + ei * 101 + n)
            rng2.shuffle(perm)
            new_docs = [combined[p][1] for p in perm]
            newpos_of_old_combined = {old: new for new, old in enumerate(perm)}
            # gold_docs occupy combined-slots [0, len(gold_docs)); map old doc idx ->
            # combined slot -> new position.
            def new_gold_pos(old_idx):
                return newpos_of_old_combined[old2goldpos[old_idx]]

            rec = {}
            for k, v in ex.items():
                if k == "documents":
                    rec["documents"] = new_docs
                elif k == cfg["gold_field"]:
                    base = cfg.get("index_base", 0)   # restore native indexing on write
                    if cfg.get("gold_is_pairs"):
                        rec[k] = [sorted(new_gold_pos(a) + base for a in pair) for pair in v]
                    else:
                        rec[k] = sorted(new_gold_pos(a) + base for a in v)
                elif k in cfg.get("extra_index_fields", []):
                    continue  # rebuilt below
                elif k == "ce_scores" and has_ce:
                    continue  # rebuilt below (parallel array, must follow docs)
                elif k == "answers" and cfg.get("answers_from_gold"):
                    continue  # rebuilt below
                else:
                    rec[k] = v
            if has_ce:
                rec["ce_scores"] = [combined_ce[p] for p in perm]
            # hard_neg_indices (nq): new positions of surviving hard-neg distractors.
            if "hard_neg_indices" in cfg.get("extra_index_fields", []):
                goldslots = len(gold_docs)
                rec["hard_neg_indices"] = sorted(
                    newpos_of_old_combined[goldslots + j]
                    for j, ish in enumerate(chosen_hard) if ish)
            # outlier answers: "; ".join of 1-indexed gold positions.
            if cfg.get("answers_from_gold"):
                gp = sorted(new_gold_pos(a) for a in ex[cfg["gold_field"]])
                rec["answers"] = ["; ".join(str(i + 1) for i in gp)]
            rung_rows[lab].append(rec)

    out_dir = os.path.join(out_root, task)
    written = {}
    for lab, n in cfg["rungs"].items():
        rows = rung_rows[lab][:CAP_PER_RUNG]   # exactly 500, row-aligned across rungs
        path = os.path.join(out_dir, cfg["out_tmpl"].format(n=n))
        save_jsonl(path, rows)
        written[lab] = (n, len(rows), path)
        if verbose:
            print(f"    [{lab:>3}] n={n:<4} {len(rows)} ex -> {path}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="contra,nq,outlier,rerank")
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    args = ap.parse_args()
    for task in args.tasks.split(","):
        task = task.strip()
        if task not in TASKS:
            print(f"skip unknown task {task}"); continue
        build_task(task, TASKS[task], args.out_root)
    print("\nDONE.")


if __name__ == "__main__":
    main()
