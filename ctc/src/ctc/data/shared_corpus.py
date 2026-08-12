#!/usr/bin/env python3
"""
Build **shared-corpus** ("fast") variants of the eval rungs. EVAL ONLY.

.. warning::
   **A fast rung is a different measurement, not a cheaper route to the same number.** Measured
   against the independent rungs (``records/shared-corpus-eval-results.md``): outlier
   **+0.215 / +0.261**, contradiction **-0.102 ... -0.175**. A scatter control that held corpus
   contents fixed and moved only the golds attributed outlier's entire shift to document
   *placement*, not to the sharing. That is why these files ship as their own ``kind="fast"``
   bundle rather than as a flag on the reliable one: a fast number and a reliable number must never
   land in the same column by accident.

   The multiplexed tasks are the opposite case -- faithful (nq -0.016, rerank -0.002) but **not
   cheaper**, because ``query_position="both"`` renders the per-query question ahead of the corpus
   and the reusable *token* prefix is 0.5-0.9%. They are built for the corpus-count reduction and
   for the rungs pooling makes reachable at all, not for speed.

Motivation: a rung is 500 examples over 500 *independent* corpora, so a 32k rung costs 500 full
32k prefills — measured at 8,885 s on one H200. If many queries share one byte-identical corpus
prefix, that prefix is prefilled once and its KV reused, and the rung collapses to a handful of
prefills. See ``records/shared-corpus-eval-plan.md``.

The output schema is **identical** to the input schema plus three additive fields
(``corpus_id``, ``shared_prefix_len``, ``shared_prefix_sha1``). Every row still carries its full
``documents`` list, so every existing evaluator, parser and metric reads these files untouched and
independent-vs-shared is a pure ``--eval-jsonl`` swap. A cache-aware runner is then a pure
optimisation that must reproduce the same numbers.

Two construction families (see the plan doc §2):

**Query-multiplexed** (``nq``, ``rerank``, ``oolong``) — the task has a real per-example query, so
the corpus is shared 100%. Gold documents land at arbitrary positions, so there is no recency
shortcut at all.

**Prefix + tail** (``contradiction``, ``outlier``) — the task has no discriminative query
(contradiction's ``queries`` is literally ``[]``), so a per-query tail is the only way to vary the
answer. For contradiction we place **one member of each gold pair in the shared prefix and its
partner in the tail**, so recency can hand the model at most half of each answer and the partner
still has to be found by searching the whole corpus.

.. note::
   ``outlier`` does not build above the small rungs, and the reason is measured rather than
   assumed. Its construction needs per-document topic labels, which the released eval files strip
   (every ``title`` is ``None``), so :func:`_recover_topics` re-derives them and *gates* on exactly
   reproducing ``meta.category_distribution``. Measured gate pass rate on the staged rungs:
   **0/25 at 2k, 0/25 at 8k, 2/25 (8%) at 32k** -- against the ~25% the original docstring
   claimed. The gate demands an exact match on the whole multiset of topic sizes, so it gets
   *harder* as the corpus grows; the 1M rung holds 7,209 documents. Unblocking outlier means
   emitting topic labels from the generator, not tuning the clustering.

Usage::

    python -m ctc.data.shared_corpus --task contradiction --rung 1048576 --tail-frac 0.5 \\
        --source <bundle>/contra/..._xlong_1M.jsonl --out-root <fast-bundle>
    python -m ctc.data.shared_corpus --task nq --rung 8192 --eval-rungs <staged-rungs>

Paths are arguments, not constants: the sources live on weka on Beaker and under a staged copy
locally, and a builder that hardcodes one of those silently writes to the wrong filesystem.
"""
import argparse
import collections
import hashlib
import json
import os
import random

#: Root holding ``<task>/rung_<n>.jsonl`` sources. Overridden by ``--eval-rungs``; ``--source``
#: bypasses it entirely, which is how the bundle's own xlong files (whose names encode a corpus
#: size rather than a rung) are read.
EVAL_RUNGS = os.environ.get("CTC_EVAL_RUNGS", "")
OUT_ROOT_DEFAULT = os.environ.get("CTC_SHARED_OUT", "")
#: Only ``oolong`` reads this: its shared file comes from the source split that already stores 25
#: questions per context, not from a rung file.
CR_DATA = os.environ.get("CTC_CR_DATA", "")

SEED = 1234


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def doc_key(d):
    """Identity of a document for dedup/prefix-hashing: its rendered text (+title if present)."""
    return json.dumps([d.get("title"), d.get("text")], sort_keys=True)


def prefix_sha1(docs):
    h = hashlib.sha1()
    for d in docs:
        h.update(doc_key(d).encode())
        h.update(b"\x00")
    return h.hexdigest()


def annotate(row, corpus_id, shared_prefix_len):
    """Attach the three additive fields. ``shared_prefix_len`` documents from position 0 are
    guaranteed byte-identical across every row carrying the same ``corpus_id``."""
    row["corpus_id"] = corpus_id
    row["shared_prefix_len"] = shared_prefix_len
    row["shared_prefix_sha1"] = prefix_sha1(row["documents"][:shared_prefix_len])
    return row


# ── Family A: query-multiplexed ──────────────────────────────────────────────


def build_oolong_shared(rung, out_root, source=None, **_):
    """OOLONG needs no new data: its SOURCE split already stores 25 questions per context, each row
    already in the independent per-row schema. Sorting those rows by context turns the file into a
    shared-corpus file whose CONTENT IS BYTE-IDENTICAL to the independent one.

    That makes oolong both the biggest free win (16 corpora / 400 queries at 32k) and the exact
    correctness gate for the cache-reusing runner: a content-identical variant that changes the
    score means the cache path is wrong.

    NOTE the v2 rung file itself is NOT usable here — it is a different generation with 500
    genuinely distinct contexts (1 question each), so it has nothing to share.
    """
    src = source or f"{CR_DATA}/oolong_test_synth_ctx{rung}_spliteval.jsonl"
    rows = load_jsonl(src)
    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault(prefix_sha1(r["documents"]), []).append(r)

    out = []
    for cid, (sha, grp) in enumerate(groups.items()):
        n = len(grp[0]["documents"])
        for r in grp:
            assert prefix_sha1(r["documents"]) == sha, "context drift inside a group"
            out.append(annotate(r, f"oolong-{rung}-c{cid}", n))

    path = f"{out_root}/oolong/rung_{rung}_mux.jsonl"
    save_jsonl(path, out)
    return path, {
        "source": src,
        "rows": len(out),
        "corpora": len(groups),
        "queries_per_corpus": sorted({len(g) for g in groups.values()}),
        "shared_token_fraction": 1.0,
        "content_identical_to_independent": True,
        "note": "baseline for this variant is the SAME file unsorted (same rows) -> exact-match gate",
    }


def build_multiplexed(
    task,
    rung,
    out_root,
    queries_per_corpus=None,
    seed=SEED,
    target_ndocs=None,
    source=None,
    plain_name=False,
    **_,
):
    """nq / rerank: pool each query's own gold + hard negatives into one shared corpus.

    Every query keeps ITS OWN CE-filtered hard negatives — that is the difficulty that matters. The
    remaining slots are filled by other queries' documents, which are topically unrelated and
    therefore easier distractors than a hard negative; that shift is exactly what the alignment run
    measures.

    Corpus count is bounded from below by ``500 * docs_per_query / ndocs_rung``: at the 8k rung
    (48 docs, 4 docs/query for nq) the floor is ~42 corpora, so the '1-10 corpora' target is
    reachable only at the larger rungs. Reported in the manifest either way.
    """
    src = source or f"{EVAL_RUNGS}/{task}/rung_{rung}.jsonl"
    rows = load_jsonl(src)
    rng = random.Random(seed)
    ndocs = target_ndocs or len(rows[0]["documents"])

    def must_keep(r):
        keep = list(r["gold_doc_indices"]) + list(r.get("hard_neg_indices", []))
        seen, out = set(), []
        for i in keep:
            if i not in seen and 0 <= i < len(r["documents"]):
                seen.add(i)
                out.append(i)
        return out

    keeps = [must_keep(r) for r in rows]

    # Greedy packing: keep adding queries to the current corpus while the UNION of their must-keep
    # documents still fits. Sizing by max(len(keeps)) instead would waste most of the corpus --
    # nq's hard-neg count ranges 0..10 around a median of 4, so the worst case is ~2x the typical.
    if queries_per_corpus:
        bounds = list(range(0, len(rows), queries_per_corpus))
        groups_idx = [list(range(s, min(s + queries_per_corpus, len(rows)))) for s in bounds]
    else:
        groups_idx, cur, cur_keys = [], [], set()
        for ri in range(len(rows)):
            new_keys = {doc_key(rows[ri]["documents"][d]) for d in keeps[ri]}
            if cur and len(cur_keys | new_keys) > ndocs:
                groups_idx.append(cur)
                cur, cur_keys = [], set()
            cur.append(ri)
            cur_keys |= new_keys
        if cur:
            groups_idx.append(cur)

    out = []
    for cid, grp_idx in enumerate(groups_idx):

        # Assemble the shared corpus: every group member's must-keep docs, deduped, then filler
        # drawn from the same group's leftover documents until the rung's doc count is hit.
        corpus, seen = [], set()
        for ri in grp_idx:
            r = rows[ri]
            for di in keeps[ri]:
                k = doc_key(r["documents"][di])
                if k in seen:
                    continue
                seen.add(k)
                corpus.append(r["documents"][di])

        if len(corpus) > ndocs:
            raise SystemExit(
                f"{task} rung {rung}: {len(grp_idx)} queries need {len(corpus)} must-keep docs "
                f"but the rung holds {ndocs}. Lower --queries-per-corpus."
            )

        filler_pool = []
        for ri in grp_idx:
            r = rows[ri]
            ks = set(keeps[ri])
            filler_pool += [d for j, d in enumerate(r["documents"]) if j not in ks]
        rng.shuffle(filler_pool)
        for d in filler_pool:
            if len(corpus) >= ndocs:
                break
            k = doc_key(d)
            if k in seen:
                continue
            seen.add(k)
            corpus.append(d)

        # Shuffle so gold positions are not correlated with group membership order.
        perm = list(range(len(corpus)))
        rng.shuffle(perm)
        corpus = [corpus[p] for p in perm]
        pos_of = {doc_key(d): i for i, d in enumerate(corpus)}

        for ri in grp_idx:
            r = dict(rows[ri])
            src_docs = rows[ri]["documents"]
            r["documents"] = corpus
            for field in ("gold_doc_indices", "hard_neg_indices"):
                if field in r:
                    r[field] = [
                        pos_of[doc_key(src_docs[i])]
                        for i in r[field]
                        if doc_key(src_docs[i]) in pos_of
                    ]
            if "ce_scores" in r:
                # CE scores are per-document AND query-specific, so a pooled corpus has no score
                # for another query's documents. Rebuild the array aligned to the new corpus,
                # carrying this query's own scores across and marking every foreign document
                # None -- which is precisely the "unscored random-fill" case the reference-order
                # builder already handles (_rerank_reference_order: scored docs sorted by CE,
                # then unscored ones in displayed order). Dropping the array instead makes the
                # prompt builder raise on the deprecated binary rerank format.
                src_ce = rows[ri]["ce_scores"]
                rebuilt = [None] * len(corpus)
                for j, d in enumerate(src_docs):
                    pos = pos_of.get(doc_key(d))
                    if pos is not None and j < len(src_ce):
                        rebuilt[pos] = src_ce[j]
                r["ce_scores"] = rebuilt
            out.append(annotate(r, f"{task}-{rung}-c{cid}", len(corpus)))

    # The n-suffix exists so an --ndocs build cannot clobber a default one in the same directory.
    # The fast bundle builds exactly one variant per rung, so `plain_name` keeps every task's
    # filenames on one pattern -- which is what lets a reader resolve a rung without a lookup table.
    suffix = "mux" if (target_ndocs is None or plain_name) else f"mux_n{ndocs}"
    path = f"{out_root}/{task}/rung_{rung}_{suffix}.jsonl"
    save_jsonl(path, out)
    return path, {
        "source": src,
        "rows": len(out),
        "corpora": len(groups_idx),
        "queries_per_corpus": sorted({len(g) for g in groups_idx}),
        "docs_per_query_median": sorted(len(k) for k in keeps)[len(keeps) // 2],
        "ndocs": ndocs,
        "shared_token_fraction": 1.0,
        # NOT dropped -- rebuilt aligned to the pooled corpus, carrying this query's own scores
        # across and marking foreign documents None. The old field said the opposite of what the
        # code does, which would have read as "this rung cannot be Kendall-tau graded".
        "ce_scores": (
            "rebuilt aligned to the pooled corpus; foreign documents carry None"
            if task == "rerank"
            else "n/a"
        ),
    }


# ── Family B: prefix + per-query tail ────────────────────────────────────────

DEFAULT_Q_PER_CORPUS = 125  # 500 queries / 125 = 4 corpora, inside the 1-10 target


def build_contradiction_shared(
    task,
    rung,
    out_root,
    tail_frac=0.05,
    queries_per_corpus=None,
    seed=SEED,
    source=None,
    variant_tag="",
    **_,
):
    """contradiction: **one member of each gold pair in the shared prefix, its partner in the tail.**

    contradiction's ``queries`` field is empty -- the task is "find the contradicting pairs in this
    corpus" -- so a per-query tail is the only thing that can vary the answer across queries sharing
    a corpus. Putting a whole pair in the tail would let recency hand the model the entire answer
    and would collapse an O(N^2) all-pairs search into an O(tail^2) one. Splitting each pair means
    the tail gives away at most one endpoint and the partner still has to be found by searching the
    full corpus.

    Other group members' prefix-resident halves are inert distractors for this query: their partners
    live in a *different* query's tail and are absent here, so they introduce no extra true pair.

    Gold indices are 1-INDEXED for this task on the way in and on the way out
    ([[gold-doc-indices-per-task-base]]); everything internal is 0-indexed.
    """
    src = source or f"{EVAL_RUNGS}/{task}/rung_{rung}.jsonl"
    rows = load_jsonl(src)
    rng = random.Random(seed)
    ndocs = len(rows[0]["documents"])
    tail_len = max(4, int(round(ndocs * tail_frac)))
    prefix_len = ndocs - tail_len

    # Each query parks one half of each of its gold pairs in the SHARED prefix, so a corpus can
    # host at most ``prefix_len // pairs_per_query`` queries before the halves alone overflow it.
    # The old script took a flat 125 and raised SystemExit when that did not fit, which is every
    # small rung once the tail is large: a 2k rung at ``--tail-frac 0.5`` leaves 39 prefix slots
    # against 375 halves. Cap instead, so the knob that gets traded away is the sharing ratio --
    # which the manifest reports -- rather than the run.
    pairs = max(1, max(len(r["gold_doc_indices"]) for r in rows))
    ceiling = max(1, prefix_len // pairs)
    requested = queries_per_corpus or DEFAULT_Q_PER_CORPUS
    queries_per_corpus = min(requested, ceiling)

    # Every document that is a gold member ANYWHERE is barred from being filler -- contradiction
    # data is built by perturbation, so a claim that is gold in one example is a real contradiction
    # partner wherever it appears ([[contra-fever-filler-leak]] is the same class of bug).
    gold_keys = set()
    for r in rows:
        for pair in r["gold_doc_indices"]:
            for i in pair:
                gold_keys.add(doc_key(r["documents"][i - 1]))

    filler_pool, seen_f = [], set()
    for r in rows:
        for d in r["documents"]:
            k = doc_key(d)
            if k in gold_keys or k in seen_f:
                continue
            seen_f.add(k)
            filler_pool.append(d)
    rng.shuffle(filler_pool)

    out = []
    n_groups = 0
    for cid, start in enumerate(range(0, len(rows), queries_per_corpus)):
        grp = list(range(start, min(start + queries_per_corpus, len(rows))))
        n_groups += 1

        # Split every pair: one half to the shared prefix, the partner to that query's tail.
        halves, probes = [], {}
        for ri in grp:
            r = rows[ri]
            my_probes = []
            for pair in r["gold_doc_indices"]:
                a, b = (i - 1 for i in pair)
                keep_in_prefix, to_tail = (a, b) if rng.random() < 0.5 else (b, a)
                halves.append(r["documents"][keep_in_prefix])
                my_probes.append(r["documents"][to_tail])
            probes[ri] = my_probes

        if len(halves) > prefix_len:
            raise SystemExit(
                f"{len(halves)} prefix halves exceed prefix_len {prefix_len}; "
                f"lower --queries-per-corpus"
            )

        prefix, seen_p = [], set()
        for d in halves:
            k = doc_key(d)
            if k in seen_p:
                continue
            seen_p.add(k)
            prefix.append(d)
        fi = 0
        while len(prefix) < prefix_len and fi < len(filler_pool):
            d = filler_pool[fi]
            fi += 1
            k = doc_key(d)
            if k in seen_p:
                continue
            seen_p.add(k)
            prefix.append(d)
        rng.shuffle(prefix)
        pos_in_prefix = {doc_key(d): i for i, d in enumerate(prefix)}

        for ri in grp:
            r = dict(rows[ri])
            tail = list(probes[ri])
            seen_t = {doc_key(d) for d in tail}
            while len(tail) < tail_len:
                d = filler_pool[rng.randrange(len(filler_pool))]
                k = doc_key(d)
                if k in seen_t or k in seen_p:
                    continue
                seen_t.add(k)
                tail.append(d)
            rng.shuffle(tail)
            pos_in_tail = {doc_key(d): prefix_len + i for i, d in enumerate(tail)}

            r["documents"] = prefix + tail
            new_pairs = []
            for j, pair in enumerate(rows[ri]["gold_doc_indices"]):
                probe = probes[ri][j]
                half = [
                    d
                    for d in (rows[ri]["documents"][i - 1] for i in pair)
                    if doc_key(d) != doc_key(probe)
                ][0]
                # write back 1-INDEXED, as this task's readers expect
                new_pairs.append(
                    [pos_in_prefix[doc_key(half)] + 1, pos_in_tail[doc_key(probe)] + 1]
                )
            r["gold_doc_indices"] = new_pairs
            out.append(annotate(r, f"{task}-{rung}-c{cid}", prefix_len))

    path = f"{out_root}/{task}/rung_{rung}{variant_tag}_tail{int(tail_frac*100):02d}.jsonl"
    save_jsonl(path, out)
    return path, {
        "source": src,
        "rows": len(out),
        "corpora": n_groups,
        "queries_per_corpus": queries_per_corpus,
        "ndocs": ndocs,
        "shared_prefix_len": prefix_len,
        "tail_len": tail_len,
        "shared_token_fraction": round(prefix_len / ndocs, 4),
        "pair_split": "one member in shared prefix, partner in per-query tail",
        "filler_pool_size": len(filler_pool),
    }


def _recover_topics(row, verbose=False):
    """Recover per-document topic labels for one outlier example, or return None.

    The outlier task is "find the topic with the fewest documents": majority topics carry >= 4
    documents (``maj_outlier_gap=1`` in ``generate_wiki_outlier_data.py``) and the outlier trio
    carries exactly 3. Building a shared corpus therefore needs topic labels -- a filler document
    whose topic ends up with 1-2 members in the corpus would be a *smaller* group than the real
    outlier trio and would make the example unanswerable.

    Labels are stripped from the eval files (``title: None``) and rebuilding the wiki100w article
    pool would drag in pyserini/Lucene, whose JVM is a known hazard on these compute nodes
    ([[obliq-gen-infra-jvm-nfs-traps]]). Instead: TF-IDF + agglomerative clustering, gated on
    reproducing ``meta.category_distribution`` -- which hands us the exact multiset of topic sizes,
    so the recovery is *verified*, not assumed. Examples that don't reproduce it exactly are
    skipped rather than guessed at (~25% reproduce; we need only a handful).

    :returns: list of ``[doc, ...]`` groups (majority topics only), or None if unverifiable.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer

    gold = set(row["gold_doc_indices"])
    maj = [d for i, d in enumerate(row["documents"]) if i not in gold]
    cd = dict(row["meta"]["category_distribution"])
    cd.pop(row["meta"].get("minority_label"), None)
    expected, k = sorted(cd.values()), len(cd)
    if k < 2 or len(maj) < k:
        return None
    X = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit_transform(
        [d["text"] for d in maj]
    )
    labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(
        X.toarray()
    )
    groups = collections.defaultdict(list)
    for d, lab in zip(maj, labels):
        groups[lab].append(d)
    if sorted(len(g) for g in groups.values()) != expected:
        return None
    return list(groups.values())


def build_outlier_shared(
    task,
    rung,
    out_root,
    tail_frac=0.05,
    queries_per_corpus=None,
    seed=SEED,
    scatter_gold=False,
    source=None,
    **_,
):
    """outlier: shared majority corpus + a per-query tail holding that query's outlier trio.

    Unlike contradiction, **no gold can live in the shared prefix**: an outlier trio in the prefix
    would be the outlier for *every* query sharing that prefix. So all 3 golds sit in the tail and
    dilution is the only mitigation -- hence the tail sweep and the ``guess-tail`` control the
    verifier prints next to every result.

    The tail's filler is drawn from a reservoir of documents whose topics keep >= 4 members in the
    shared prefix (topic labels recovered + verified by :func:`_recover_topics`), so filler can
    never form a group smaller than the real trio.
    """
    src = source or f"{EVAL_RUNGS}/{task}/rung_{rung}.jsonl"
    rows = load_jsonl(src)
    rng = random.Random(seed)
    queries_per_corpus = queries_per_corpus or DEFAULT_Q_PER_CORPUS
    ndocs = len(rows[0]["documents"])
    tail_len = max(6, int(round(ndocs * tail_frac)))
    prefix_len = ndocs - tail_len
    n_outliers = len(rows[0]["gold_doc_indices"])
    # 2x oversubscribed: enough for per-query tail variety without needing so many topics that
    # the pinned 4-per-topic minimum overflows the prefix (it does at 3x for the 25% tail).
    reservoir_target = 2 * (tail_len - n_outliers)

    # Topic bank: cluster candidate rows until enough VERIFIED topics are banked for every group.
    n_groups = (len(rows) + queries_per_corpus - 1) // queries_per_corpus
    need_per_group = prefix_len + reservoir_target
    bank, used_as_base, scanned = [], set(), 0
    while sum(len(t) for t in bank) < need_per_group * n_groups and scanned < len(rows):
        topics = _recover_topics(rows[scanned])
        if topics:
            bank.append([t for t in topics if len(t) >= 4])
            used_as_base.add(scanned)
        scanned += 1
    flat_topics = [t for per_row in bank for t in per_row]
    # Random order, NOT largest-first. Sorting by size descending minimises how much of the prefix
    # the pinned 4-per-topic minimum eats -- but it also fills the corpus with big topics, so the
    # smallest majority topic ends up far above 4 and the 3-document outlier trio becomes trivially
    # the smallest group. The source data always has a smallest majority topic of exactly 4
    # (`maj_outlier_gap=1`), i.e. a gap of ONE above the trio; widening that gap inflated outlier
    # from 0.320 to 0.887. Order randomly here and re-impose the gap exactly below.
    rng.shuffle(flat_topics)
    if sum(len(t) for t in flat_topics) < need_per_group * n_groups:
        raise SystemExit(
            f"outlier: only recovered {sum(len(t) for t in flat_topics)} banked docs "
            f"but need {need_per_group * n_groups}; scanned {scanned} examples"
        )

    # Query rows = every row (each contributes its own outlier trio), including the base rows --
    # a base row's own trio was excluded from its majority docs, so it is still a valid query.
    out, ti, next_q, skipped_collide = [], 0, 0, 0
    for cid in range(n_groups):
        # Take whole topics; the first 4 documents of each are pinned to the prefix, the surplus
        # first tops the prefix up to prefix_len and then fills the reservoir.
        pinned, surplus, taken_topics = [], [], []
        while len(pinned) + len(surplus) < prefix_len + reservoir_target:
            if ti >= len(flat_topics):
                raise SystemExit("outlier: topic bank exhausted")
            t = flat_topics[ti]
            ti += 1
            taken_topics.append(t)
            pinned += t[:4]
            surplus += t[4:]
        if len(pinned) > prefix_len:
            raise SystemExit("outlier: pinned minimum exceeds prefix_len; raise --tail-frac")
        rng.shuffle(surplus)
        prefix = pinned + surplus[: prefix_len - len(pinned)]
        reservoir = surplus[prefix_len - len(pinned) :]

        # Re-impose the source data's defining structure: the smallest majority topic must hold
        # EXACTLY n_outliers + 1 documents, so the outlier trio wins by one document and not more.
        # Reservoir documents only ever ADD to a topic's count, so constraining the prefix is
        # sufficient. Without this the eval is a different, much easier task.
        topic_of = {}
        for ti_, t in enumerate(taken_topics):
            for d in t:
                topic_of[doc_key(d)] = ti_
        counts = collections.Counter(topic_of[doc_key(d)] for d in prefix)
        target_min = n_outliers + 1
        while counts and min(counts.values()) > target_min:
            smallest = min(counts, key=lambda k: counts[k])
            movable = [d for d in prefix if topic_of[doc_key(d)] == smallest][target_min:]
            if not movable:
                break
            drop = movable[0]
            prefix.remove(drop)
            reservoir.append(drop)
            counts[smallest] -= 1
            # keep the corpus at its nominal size
            if reservoir and len(prefix) < prefix_len:
                for cand in reservoir:
                    ck = topic_of.get(doc_key(cand))
                    if ck is not None and ck != smallest and counts.get(ck, 0) >= target_min:
                        prefix.append(cand)
                        reservoir.remove(cand)
                        counts[ck] += 1
                        break
                else:
                    break
        min_topic_size = min(counts.values()) if counts else 0
        rng.shuffle(prefix)
        seen_p = {doc_key(d) for d in prefix}
        reservoir = [d for d in reservoir if doc_key(d) not in seen_p]
        if len(reservoir) < tail_len - n_outliers:
            raise SystemExit("outlier: reservoir too small for the tail")

        # Assign queries to this corpus. A query is usable only if its outlier trio does not
        # already appear in the shared prefix (~1% of wiki100w chunks recur across examples) --
        # a trio member sitting in the prefix would silently belong to a majority topic and stop
        # being an outlier. Colliding queries are skipped and backfilled from later rows.
        grp = []
        while len(grp) < queries_per_corpus and next_q < len(rows):
            ri = next_q
            next_q += 1
            trio_keys = {doc_key(rows[ri]["documents"][i]) for i in rows[ri]["gold_doc_indices"]}
            if trio_keys & seen_p:
                skipped_collide += 1
                continue
            grp.append(ri)

        for ri in grp:
            src = rows[ri]
            trio = [src["documents"][i] for i in src["gold_doc_indices"]]
            tail = list(trio)
            seen_t = {doc_key(d) for d in tail}
            pool = [d for d in reservoir if doc_key(d) not in seen_t]
            rng.shuffle(pool)
            tail += pool[: tail_len - len(tail)]
            rng.shuffle(tail)

            r = dict(src)
            if scatter_gold:
                # POSITION CONTROL (not a shippable variant): identical corpus contents, but the
                # trio is scattered uniformly instead of sitting in the tail. This deliberately
                # destroys the shared prefix -- its only job is to separate "the tail placement
                # inflates the score" from "the rebuilt corpus is easier than the original".
                docs = prefix + tail
                rng.shuffle(docs)
                r["documents"] = docs
                trio_keys = {doc_key(x) for x in trio}
                gold = sorted(i for i, d in enumerate(docs) if doc_key(d) in trio_keys)
            else:
                r["documents"] = prefix + tail
                gold = sorted(
                    prefix_len + i
                    for i, d in enumerate(tail)
                    if doc_key(d) in {doc_key(x) for x in trio}
                )
            r["gold_doc_indices"] = gold
            # answers are position-derived (1-indexed) for this task -- recompute, never inherit.
            r["answers"] = ["; ".join(str(i + 1) for i in gold)]
            meta = dict(src.get("meta") or {})
            meta["num_outliers"] = len(gold)
            meta["shared_corpus_note"] = (
                "majority corpus shared; outlier trio in the per-query "
                "tail; tail filler drawn from topics keeping >=4 in the "
                "shared prefix"
            )
            r["meta"] = meta
            out.append(annotate(r, f"{task}-{rung}-c{cid}", 0 if scatter_gold else prefix_len))

    tag = "scatter" if scatter_gold else f"tail{int(tail_frac*100):02d}"
    path = f"{out_root}/{task}/rung_{rung}_{tag}.jsonl"
    save_jsonl(path, out)
    return path, {
        "source": src,
        "rows": len(out),
        "corpora": n_groups,
        "queries_per_corpus": queries_per_corpus,
        "ndocs": ndocs,
        "shared_prefix_len": prefix_len,
        "tail_len": tail_len,
        "shared_token_fraction": round(prefix_len / ndocs, 4),
        "topic_recovery": {
            "scanned": scanned,
            "verified_base_examples": len(used_as_base),
            "topics_banked": len(flat_topics),
        },
        "queries_skipped_trio_in_prefix": skipped_collide,
        "gold_placement": "ALL golds in the per-query tail (structurally unavoidable for outlier)",
        "min_majority_topic_size": min_topic_size,
        "outlier_trio_size": n_outliers,
        "gap_above_trio": min_topic_size - n_outliers,
    }


# ── Family C: planted candidates, eliminated by the tail ─────────────────────

#: Majority topic sizes in the real outlier rungs, counted over all 500 rows of the staged 32k
#: rung. Sampling the rebuilt corpus from this rather than from a flat distribution keeps the
#: topic-size histogram -- and therefore the difficulty -- in line with the reliable rung. Size 4
#: is the floor and the nearest competitor to the 3-document answer; size 5 is the modal size,
#: which is why a decoy topped up to 5 is camouflaged rather than conspicuous.
MAJORITY_SIZE_WEIGHTS = {
    4: 3095,
    5: 4448,
    6: 2776,
    7: 1689,
    8: 1084,
    9: 760,
    10: 557,
    11: 379,
    12: 304,
    13: 203,
    14: 173,
    15: 150,
    16: 86,
    17: 81,
    18: 55,
    19: 51,
    20: 40,
    21: 37,
    22: 23,
    23: 17,
    24: 10,
    25: 12,
    26: 9,
    27: 6,
    28: 4,
    29: 6,
    30: 6,
}


def _take_article(pool, min_chunks, rng, used, prefer=None):
    """
    :param pool: An :class:`~ctc.data.sources.wiki100w.ArticlePool`.
    :param min_chunks: Fewest chunks the article must have.
    :param rng: Seeded RNG.
    :param used: Titles already spent in this corpus; one topic must not appear twice.
    :param prefer: Try for this many chunks first, then settle for ``min_chunks``. Slack above the
        planted size is what a topic can donate as tail camouflage without moving the floor.

    :returns: ``(title, bodies)``.

    :raises SystemExit: If the pool cannot supply one.
    """
    for want, tries in ((prefer or min_chunks, 200), (min_chunks, 4000)):
        for _ in range(tries):
            title, bodies = pool.articles[rng.randrange(len(pool.articles))]
            if title not in used and len(bodies) >= want:
                used.add(title)
                return title, bodies
    raise SystemExit(f"pool exhausted looking for an unused article with >={min_chunks} chunks")


def build_outlier_planted(
    task,
    rung,
    out_root,
    *,
    ndocs,
    pool_path,
    n_rows=500,
    tail_frac=0.1,
    n_outliers=3,
    decoy_size=5,
    floor=4,
    camouflage_frac=0.25,
    seed=SEED,
    split="eval",
    **_,
):
    """
    outlier, shared: plant every candidate in the prefix, eliminate all but one from the tail.

    :func:`build_outlier_shared` puts the answer's trio in the per-query tail, which is exactly
    where its **+0.215** came from -- a scatter control showed the rebuilt corpus was faithful
    (0.330 vs 0.320) and the *placement* was the whole effect. It also needs per-document topic
    labels the eval files strip, and the clustering that recovers them clears its exactness gate
    2/25 of the time at 32k and 0/25 below.

    This inverts it. ``K`` candidate topics are planted in the **shared prefix**, each with
    ``n_outliers`` documents. Each query's tail then tops up every candidate *except its own* to
    ``decoy_size``, leaving exactly one topic at 3 -- in the prefix, not at the end of the context.

    - **No recency shortcut.** The answer is never in the tail. A model reading only the last 10%
      learns what the answer is *not*; naming it still means counting over the whole corpus.
    - **No label recovery.** A topic is an article from the pool, so every document's topic is
      known by construction and nothing has to be clustered.

    ``decoy_size`` is 5, not 4, and that is the load-bearing choice. Topping a decoy to 4 would
    park it on the discrimination boundary -- the real rungs' smallest majority topic is 4 in
    484/500 rows -- making every decoy a near-miss and putting ~28 topics one document from the
    answer where the source has ~6. Topping to 5 leaves the natural size-4 topics as the only
    competitors, exactly as in the reliable rung, and hides the decoys in the modal size class.
    The *corpus* gap stays 1; only the decoys' distance from the answer grows.

    :param ndocs: Corpus size, taken from the reliable rung so lengths match.
    :param pool_path: Pickled :class:`ArticlePool`.
    :param n_rows: Queries to emit; the eval size.
    :param tail_frac: Per-query tail as a fraction of the corpus.
    :param n_outliers: Documents in the answer's topic.
    :param decoy_size: What a decoy is topped up to.
    :param floor: Smallest majority topic, i.e. the nearest competitor.
    :param camouflage_frac: Share of the tail that is ordinary documents rather than top-ups, so a
        tail is not visibly one pair per decoy.
    :param split: Article split; ``eval`` keeps the generators' train/eval article separation.

    :returns: ``(path, manifest)``.
    """
    from ctc.data.sources.wiki100w import ArticlePool

    pool = ArticlePool.from_cache(pool_path).for_split(split)
    rng = random.Random(seed)

    tail_len = max(2 * (n_outliers + 1), int(round(ndocs * tail_frac)))
    prefix_len = ndocs - tail_len
    per_decoy = decoy_size - n_outliers
    queries_per_corpus = max(2, int(tail_len * (1 - camouflage_frac)) // per_decoy + 1)

    sizes = sorted(MAJORITY_SIZE_WEIGHTS)
    weights = [MAJORITY_SIZE_WEIGHTS[s] for s in sizes]

    out, cid, row_i = [], 0, 0
    observed_min_majority, observed_answer, absent = [], set(), []
    while row_i < n_rows:
        group_n = min(queries_per_corpus, n_rows - row_i)
        used: set = set()

        # Candidates: n_outliers documents each in the shared prefix, per_decoy held back to
        # eliminate this candidate from every other query's answer.
        candidates = []
        for _ in range(group_n):
            title, bodies = _take_article(pool, decoy_size, rng, used, prefer=decoy_size + 4)
            candidates.append(
                {
                    "title": title,
                    "planted": list(bodies[:n_outliers]),
                    "topup": list(bodies[n_outliers:decoy_size]),
                    # Spare chunks. Used only to pad a tail that majority camouflage cannot fill:
                    # they push a decoy further above the boundary, which is harmless, whereas a
                    # fresh article would open a topic below the floor and create a second answer.
                    "extra": list(bodies[decoy_size:]),
                }
            )

        # Majority topics, sized from the real distribution, with one pinned to the floor: that is
        # the gap-of-1 structure the task's difficulty rests on.
        majority, camouflage = [], []
        remaining = prefix_len - n_outliers * group_n
        if remaining < floor:
            raise SystemExit(
                f"ndocs={ndocs} leaves {remaining} prefix slots for majority topics after "
                f"{group_n} candidates; raise --ndocs or lower --tail-frac"
            )
        pinned_floor = False
        while remaining >= floor:
            # Clamp to what the pool can actually supply: the size distribution is measured
            # from the real rungs and reaches 30, while a pool of short articles cannot.
            drawn = rng.choices(sizes, weights)[0] if pinned_floor else floor
            size = min(drawn, remaining, pool.max_chunks)
            pinned_floor = True
            if size < floor:
                break
            title, bodies = _take_article(pool, size, rng, used, prefer=size + 3)
            majority.append([title, list(bodies[:size]), list(bodies[size:])])
            remaining -= size

        # The last few slots cannot open a new topic -- a topic below the floor would be a second
        # answer. Grow topics that are already above it instead, which cannot move the floor. Only
        # if no topic has slack does this fall back to a bigger article, so a pool whose articles
        # are all short still builds.
        for entry in majority:
            if not remaining:
                break
            if len(entry[1]) > floor and entry[2]:
                take = min(remaining, len(entry[2]))
                entry[1].extend(entry[2][:take])
                entry[2] = entry[2][take:]
                remaining -= take
        if remaining:
            title, bodies = _take_article(pool, floor + remaining, rng, used)
            majority.append([title, list(bodies[: floor + remaining]), []])

        # Camouflage: spare chunks of topics already above the floor. One more document leaves such
        # a topic at >= floor + 1, so the nearest competitor stays where it is.
        for title, group, spare in majority:
            if len(group) > floor:
                camouflage.extend((title, b) for b in spare)

        # Every document that can reach a corpus is registered, top-ups and camouflage included.
        # Missing one makes it invisible to the count, which is how a "topped-up" decoy silently
        # stays at n_outliers and the answer stops being unique.
        prefix_docs, topic_of = [], {}
        for cand in candidates:
            for body in cand["planted"] + cand["topup"]:
                topic_of[body] = cand["title"]
            prefix_docs.extend(cand["planted"])
        for title, group, _spare in majority:
            for body in group:
                topic_of[body] = title
            prefix_docs.extend(group)
        for title, body in camouflage:
            topic_of[body] = title
        rng.shuffle(prefix_docs)
        pos_of = {body: i for i, body in enumerate(prefix_docs)}

        if not camouflage:
            raise SystemExit("no camouflage documents available; lower --camouflage-frac")

        for cand in candidates:
            tail = []
            for other in candidates:
                if other["title"] != cand["title"]:
                    tail.extend(other["topup"])
            # Without replacement: the same passage appearing twice in one corpus is a duplicate a
            # reader would notice, and it would double-count its topic. Majority spare first, since
            # it makes the tail read as ordinary corpus content; decoy spare only as a fallback.
            need = tail_len - len(tail)
            take = min(need, len(camouflage))
            tail.extend(body for _, body in rng.sample(camouflage, take))
            need -= take
            if need:
                spare = [
                    b
                    for other in candidates
                    if other["title"] != cand["title"]
                    for b in other["extra"]
                ]
                if need > len(spare):
                    raise SystemExit(
                        f"tail is {need} documents short after camouflage and decoy spare; "
                        "lower --tail-frac or use a pool with longer articles"
                    )
                tail.extend(rng.sample(spare, need))
            rng.shuffle(tail)

            documents = prefix_docs + tail
            gold = sorted(pos_of[b] for b in cand["planted"])

            # Deduplicate first: a camouflage document drawn twice is one document, not two, and
            # counting it twice would overstate its topic.
            counts = collections.Counter(topic_of[b] for b in dict.fromkeys(documents))
            observed_min_majority.append(min(c for t, c in counts.items() if t != cand["title"]))
            observed_answer.add(counts[cand["title"]])

            # Shortcut control. The answer's topic is the one candidate the tail does not top up,
            # so "absent from the tail" is worth measuring: if only one topic were absent, reading
            # the last 10% would hand over the answer. Every majority topic that donated no
            # camouflage is also absent, so absence alone is uninformative -- this records by how
            # much. Guessing uniformly among the absent topics scores 1/len(absent).
            in_tail = {topic_of[b] for b in tail}
            absent.append(len(set(counts) - in_tail))

            out.append(
                annotate(
                    {
                        "documents": [{"title": None, "text": b} for b in documents],
                        "queries": [
                            "Can you find passages that are about a different topic than the "
                            "rest of these passages?"
                        ],
                        "answers": ["; ".join(str(i + 1) for i in gold)],
                        "gold_doc_indices": gold,
                        "source": "wiki_outlier_topic",
                        "split": split,
                        "meta": {
                            "minority_label": cand["title"],
                            "majority_label": None,
                            "num_outliers": len(gold),
                            "num_categories": len(counts),
                            "category_distribution": sorted(counts.items()),
                            "shared_corpus_note": (
                                "candidates planted in the shared prefix; each query's tail tops "
                                "every other candidate up to decoy_size, leaving exactly one at "
                                "n_outliers. The answer is never in the tail."
                            ),
                        },
                    },
                    f"{task}-{rung}-c{cid}",
                    len(prefix_docs),
                )
            )
        row_i += group_n
        cid += 1

    path = f"{out_root}/{task}/rung_{rung}_planted.jsonl"
    save_jsonl(path, out)
    return path, {
        "source": pool_path,
        "construction": "planted candidates, eliminated by the per-query tail",
        "rows": len(out),
        "corpora": cid,
        "queries_per_corpus": queries_per_corpus,
        "ndocs": ndocs,
        "shared_prefix_len": prefix_len,
        "tail_len": tail_len,
        "shared_token_fraction": round(prefix_len / ndocs, 4),
        "n_outliers": n_outliers,
        "decoy_size": decoy_size,
        "min_majority_topic_observed": min(observed_min_majority),
        "gap_above_answer": min(observed_min_majority) - n_outliers,
        "answer_size_observed": sorted(observed_answer),
        "gold_placement": "shared prefix -- never the tail",
        # The tail-absence shortcut, measured rather than argued: guessing uniformly among the
        # topics missing from the tail scores this. Near chance (1/num_categories) means absence
        # carries no usable signal.
        "topics_absent_from_tail_mean": round(sum(absent) / len(absent), 1),
        "shortcut_absent_from_tail_acc": round(sum(1 / a for a in absent) / len(absent), 4),
    }


TASK_BUILDERS = {
    "oolong": build_oolong_shared,
    "nq": build_multiplexed,
    "rerank": build_multiplexed,
    "contradiction": build_contradiction_shared,
    "outlier": build_outlier_planted,
    # The pre-migration prefix+tail construction, kept only to reproduce the numbers it produced.
    # It puts the answer in the tail (+0.215) and needs topic labels it cannot reliably recover.
    "outlier_tail": build_outlier_shared,
}


def main():
    # The builders read these as module globals. Rebinding them below keeps their bodies
    # byte-identical to the pre-migration script, which is the point: this is a vendored copy, not
    # a rewrite, and a fast rung built here has to match one built there.
    global EVAL_RUNGS, CR_DATA

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--task", required=True, choices=sorted(TASK_BUILDERS))
    p.add_argument(
        "--rung",
        type=int,
        required=True,
        help="rung size in tokens; names the output file. With --source this is a "
        "label only -- the bundle's xlong filenames encode a corpus size instead.",
    )
    p.add_argument("--out-root", default=OUT_ROOT_DEFAULT, required=not OUT_ROOT_DEFAULT)
    p.add_argument(
        "--eval-rungs",
        default=EVAL_RUNGS,
        help="root holding <task>/rung_<n>.jsonl. Ignored when --source is given.",
    )
    p.add_argument(
        "--cr-data", default=CR_DATA, help="root holding the oolong source split (oolong only)"
    )
    p.add_argument(
        "--pool",
        default="",
        help="wiki100w article pool pickle (outlier only). The planted build draws topics from "
        "articles, so every document's topic is known and nothing is recovered by clustering.",
    )
    p.add_argument(
        "--n-rows", type=int, default=500, help="queries to emit (outlier only); the eval size"
    )
    p.add_argument(
        "--plain-name",
        action="store_true",
        help="with --ndocs, keep the plain _mux filename instead of appending _n<ndocs>",
    )
    p.add_argument(
        "--queries-per-corpus",
        type=int,
        default=None,
        help="fixed group size; default packs greedily to fill --ndocs",
    )
    p.add_argument(
        "--ndocs",
        type=int,
        default=None,
        help="override the corpus size. Multiplexing makes this a FREE parameter, which "
        "is how nq/rerank get a 32k rung at all -- their canonical CE-filtered "
        "pools cap at 48/100 docs per query, but a pooled corpus has no such cap.",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--source",
        default=None,
        help="explicit source JSONL (default: the staged rung file). Used to build from "
        "the RECALIBRATED PubMed-only contradiction ladder, whose staged 32k rung "
        "is mislabeled -- it renders to ~64k tokens, see the plan doc.",
    )
    p.add_argument(
        "--variant-tag",
        default="",
        help="extra tag in the output filename, e.g. '_ctc' for the recalibrated source",
    )
    p.add_argument(
        "--scatter-gold",
        action="store_true",
        help="outlier POSITION CONTROL: same corpus, trio scattered uniformly instead "
        "of in the tail. Destroys the shared prefix on purpose -- isolates whether "
        "the tail placement or the rebuilt corpus drives the score difference.",
    )
    p.add_argument(
        "--tail-frac",
        type=float,
        nargs="+",
        default=[0.05],
        help="Family B only: per-query tail as a fraction of the corpus. The sweep is "
        "0.05 (aggressive, ~20x prefill saving) and 0.25 (conservative, ~4x).",
    )
    args = p.parse_args()

    EVAL_RUNGS, CR_DATA = args.eval_rungs, args.cr_data
    # The planted outlier build reads the article pool, not a rung file: no source needed, but a
    # corpus size and a pool are.
    if args.task == "outlier":
        if not args.pool:
            raise SystemExit("--task outlier needs --pool (the wiki100w article pool pickle)")
        if not args.ndocs:
            raise SystemExit(
                "--task outlier needs --ndocs, the corpus size to match in the reliable rung"
            )
    elif not args.source and not EVAL_RUNGS:
        raise SystemExit("pass --source, or --eval-rungs (or set $CTC_EVAL_RUNGS)")

    tail_fracs = (
        args.tail_frac if args.task in ("contradiction", "outlier", "outlier_tail") else [None]
    )
    for tf in tail_fracs:
        kwargs = dict(
            task=args.task,
            rung=args.rung,
            out_root=args.out_root,
            queries_per_corpus=args.queries_per_corpus,
            seed=args.seed,
            target_ndocs=args.ndocs,
            source=args.source,
            variant_tag=args.variant_tag,
            scatter_gold=args.scatter_gold,
            plain_name=args.plain_name,
        )
        if args.task == "outlier":
            kwargs.update(ndocs=args.ndocs, pool_path=args.pool, n_rows=args.n_rows)
        if tf is not None:
            kwargs["tail_frac"] = tf
        path, manifest = TASK_BUILDERS[args.task](**kwargs)
        with open(path.replace(".jsonl", ".manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"wrote {path}")
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
