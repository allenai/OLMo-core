"""PubMed multi-claim contradiction data — a from-scratch rebuild of the
contradiction setting.

Why a rebuild. The existing pipeline
(:mod:`corpus_reasoning.data.generate_pubmed_contradiction_data`) takes a *real*
PubMed sentence S and asks an LLM for a minimally-perturbed S' that contradicts
it. Two structural problems follow from that:

1. **Word overlap.** S' is a rewrite of S, so the gold pair shares far more
   vocabulary with each other than with any filler. A string-overlap heuristic
   gets a large part of the way, and the task stops measuring semantic
   all-pairs comparison.
2. **Provenance asymmetry.** S is human-written, S' is LLM-written, so a
   stylistic cue separates the two halves of every gold pair.

Here every item in the corpus is LLM-written from the same prompt family, and
items are multi-sentence, so neither cue survives.

Pipeline (each phase writes JSONL and is independently re-runnable):

  1. ``dump-abstracts`` — PubMedQA abstracts → ``{aid, question, abstract}``.
     Must run where the HF cache lives (node-local ``/data`` on horton).
  2. ``claims`` — one LLM call per abstract emits several *orthogonal* 2–3
     sentence claims grounded in that abstract. Orthogonal = each hinges on a
     distinct finding, so contradicting one leaves the others true. This is
     what makes same-abstract claims usable as topically-matched hard
     negatives later.
  3. ``contradictions`` (later) — per claim, an LLM writes a 2–3 sentence claim
     that genuinely conflicts with it.
  4. ``assemble`` (later) — sample claims + contradictions into the unified
     corpus schema consumed by ``lib/data_format.py`` (``--task contradiction``).

Usage::

    # on horton (needs /data HF cache)
    python -m corpus_reasoning.data.generate_pubmed_multiclaim dump-abstracts \\
        --num 20 --out debug/pubmed_multiclaim/abstracts_pilot.jsonl

    # anywhere with network access to the vLLM server
    python -m corpus_reasoning.data.generate_pubmed_multiclaim claims \\
        --abstracts debug/pubmed_multiclaim/abstracts_pilot.jsonl --limit 5 \\
        --base-url http://horton:8100/v1 --model Qwen/Qwen2.5-14B-Instruct \\
        --out debug/pubmed_multiclaim/claims_pilot.jsonl
"""

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


# ─────────────────────────── local vLLM chat client ────────────────────────
# Deliberately dependency-free (stdlib only). `lib/llm_request_client.py` is the
# general client, but it imports `openai` and `google.genai` at module level and
# neither is installed in the training conda env — this pipeline only ever talks
# to a local vLLM OpenAI-compatible endpoint, so it does not need them. Keeping
# it stdlib means every phase runs under whichever python is on the node.

class LocalChatClient:
    """Threaded client for a local vLLM /chat/completions endpoint, with an
    on-disk JSONL response cache so prompt iteration only pays for new prompts.
    """

    def __init__(self, base_url, model, max_concurrent=16, cache_path=None,
                 max_retries=4, timeout=600, extra_body=None):
        # Accepts one URL or a comma-separated list. Independent single-GPU
        # replicas with round-robin here proved more robust than vLLM's
        # --data-parallel-size, whose DP coordinator failed to report ZMQ
        # addresses within its 120s startup timeout on this cluster. Nothing to
        # coordinate means nothing to time out.
        self.urls = [u.strip().rstrip("/") + "/chat/completions"
                     for u in str(base_url).split(",") if u.strip()]
        self._rr = 0
        self._rr_lock = __import__("threading").Lock()
        self.model = model
        # e.g. {"chat_template_kwargs": {"enable_thinking": False}} to switch off
        # Qwen3's reasoning block, which otherwise eats the token budget before
        # any JSON is emitted.
        self.extra_body = extra_body or {}
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout = timeout
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.cache[rec["key"]] = rec["response"]
                    except (json.JSONDecodeError, KeyError):
                        continue
        self.n_cached = self.n_called = self.n_failed = 0

    def _key(self, prompt, temperature, max_tokens):
        raw = (f"{self.model}|{temperature}|{max_tokens}|"
               f"{json.dumps(self.extra_body, sort_keys=True)}|{prompt}")
        return hashlib.sha256(raw.encode()).hexdigest()

    def _one(self, prompt, temperature, max_tokens):
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **self.extra_body,
        }).encode()
        with self._rr_lock:
            url = self.urls[self._rr % len(self.urls)]
            self._rr += 1
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read())
                return payload["choices"][0]["message"]["content"]
            except (urllib.error.URLError, KeyError, IndexError,
                    json.JSONDecodeError, TimeoutError) as e:
                if attempt == self.max_retries - 1:
                    print(f"  [request failed] {type(e).__name__}: {e}",
                          file=sys.stderr)
                    return None
                time.sleep(delay)
                delay *= 2
        return None

    def complete_many(self, prompts, temperature=0.7, max_tokens=1200):
        """Return one response string (or None) per prompt, order preserved."""
        keys = [self._key(p, temperature, max_tokens) for p in prompts]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        self.n_cached = len(prompts) - len(todo)
        if todo:
            print(f"  {self.n_cached} cached, {len(todo)} to generate "
                  f"({self.max_concurrent}-way)...")
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
                results = list(pool.map(
                    lambda i: self._one(prompts[i], temperature, max_tokens),
                    todo))
            new = []
            for i, resp in zip(todo, results):
                if resp is None:
                    self.n_failed += 1
                    continue
                self.cache[keys[i]] = resp
                new.append({"key": keys[i], "response": resp})
            self.n_called = len(new)
            if self.cache_path and new:
                os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)),
                            exist_ok=True)
                with open(self.cache_path, "a") as f:
                    for rec in new:
                        f.write(json.dumps(rec) + "\n")
        else:
            print(f"  all {len(prompts)} responses cached")
        return [self.cache.get(k) for k in keys]


# ───────────────────────────── phase 1: abstracts ──────────────────────────

def dump_abstracts(args):
    """Write abstracts as one JSON object per line, tagged with a domain.

    ``--source arxiv`` adds physical-sciences/CS/math breadth alongside
    PubMed's clinical focus. Note arXiv abstracts are shorter (median ~109
    words vs PubMed's 150-270), so `--min-words` discards a large share of
    them; the pool is ~2M, so that is affordable.
    """
    import random

    if args.source == "arxiv":
        return _dump_arxiv(args)

    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")

    rng = random.Random(args.seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)

    written = 0
    with open(args.out, "w") as f:
        for i in idx:
            if written >= args.num:
                break
            ex = ds[i]
            chunks = [c for c in ex["context"]["contexts"] if c and c.strip()]
            abstract = " ".join(c.strip() for c in chunks)
            # Short abstracts cannot support several orthogonal claims.
            if len(abstract.split()) < args.min_words:
                continue
            f.write(json.dumps({
                "aid": str(i),
                "domain": "biomed",
                "question": ex.get("question", ""),
                "abstract": abstract,
            }) + "\n")
            written += 1
    print(f"wrote {written} biomed abstracts -> {args.out}")


def _dump_arxiv(args):
    """Stream arXiv abstracts (2M rows) and keep the long, self-contained ones."""
    from datasets import load_dataset
    import itertools

    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train",
                      streaming=True)
    written = 0
    seen = 0
    cats = collections.Counter()
    with open(args.out, "w") as f:
        for ex in itertools.islice(ds, args.scan_limit):
            seen += 1
            if written >= args.num:
                break
            abstract = " ".join((ex.get("abstract") or "").split())
            if len(abstract.split()) < args.min_words:
                continue
            cat = ex.get("categories")
            cat = (cat[0] if isinstance(cat, list) and cat else str(cat or ""))
            primary = cat.split()[0] if cat.split() else "unknown"
            cats[primary.split(".")[0]] += 1
            f.write(json.dumps({
                "aid": f"ax{ex.get('id', seen)}".replace("/", "_"),
                "domain": "arxiv",
                "question": (ex.get("title") or "").strip().replace("\n", " "),
                "categories": cat,
                "abstract": abstract,
            }) + "\n")
            written += 1
    print(f"wrote {written} arxiv abstracts (scanned {seen}) -> {args.out}")
    print(f"  primary categories: {dict(cats.most_common(8))}")


def split_abstracts(args):
    """Partition abstracts into disjoint train / eval pools.

    The split unit is the ABSTRACT, never the claim. Every claim, its
    contradiction, and all sibling relationships derive from one abstract, so
    splitting any finer would leak: an eval gold claim's sibling could surface
    as a train distractor while sharing its entities, numbers and phrasing.
    Partitioning here makes every document — gold and distractor alike —
    disjoint between the two sets by construction, so no downstream step has to
    be trusted to keep them apart.
    """
    import random

    rows = [json.loads(l) for l in open(args.abstracts) if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    ev, tr = rows[:args.eval_abstracts], rows[args.eval_abstracts:]
    for path, part in ((args.out_eval, ev), (args.out_train, tr)):
        with open(path, "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
    assert not ({r["aid"] for r in ev} & {r["aid"] for r in tr})
    print(f"eval  {len(ev)} abstracts -> {args.out_eval}")
    print(f"train {len(tr)} abstracts -> {args.out_train}")


# ─────────────────────────────── phase 2: claims ───────────────────────────

# The claim writer. Design notes, since each constraint is load-bearing for the
# downstream task rather than cosmetic:
#
# - "distinct finding" / orthogonality: the assembled corpus puts several claims
#   from the SAME abstract in one example as hard negatives. That only works if
#   contradicting one of them does not also contradict its siblings — otherwise
#   the gold pair is not unique and the task is unscorable.
# - "self-contained": claims are shuffled into a corpus and read in isolation,
#   so anaphora pointing at the abstract ("this trial", "these patients") would
#   be both unresolvable and a positional tell.
# - "one crisp assertion + support": gives the contradiction step a single
#   well-defined target, and gives the reader a criterion for judging conflict.
# - the `pivot` field is that target, written out explicitly. It is used to
#   check orthogonality and to steer phase 3.
CLAIMS_PROMPT = """You are building a corpus of scientific claims from {field} literature.

Read the abstract below and write {k_max} SEPARATE claims that it supports.

Each claim is written as THREE separate sentences, each with its own job:
  - "assertion": the core finding — what holds, in what population, for what intervention or exposure, on what outcome. One sentence.
  - "support": the EVIDENCE behind it. One sentence.
  - "detail": one further grounded fact — a condition under which it holds, a subgroup, a mechanism, a secondary measure, or a boundary on when it does not hold. One sentence. Use "" only if the abstract truly offers nothing more.

CRITICAL — "support" must not re-say "assertion". It may only carry facts that are NOT already in the assertion: the comparison group, the study design, the cohort or sample size, the follow-up window, the dose or exposure level, how the outcome was measured, or the numeric effect size. Never re-describe the direction of the effect in different words — the assertion already did that.

Illustration of the RELATIONSHIP between the two sentences (an unrelated topic — copy the relationship, never the wording or the template):

  BAD (the support just restates the assertion with a number attached):
    "assertion": "Occupational noise exposure raises the risk of hearing loss in factory workers."
    "support":   "Factory workers exposed to noise had a 2.4 times higher risk of hearing loss than unexposed workers."

  GOOD (the support carries scale, method, and duration that the assertion does not):
    "assertion": "Occupational noise exposure raises the risk of hearing loss in factory workers."
    "support":   "Audiometric testing of 812 workers followed for nine years put the hazard ratio at 2.4 (95% CI 1.6-3.5) above 85 decibels."

A reader must not be able to delete either sentence without losing information.

Each claim needs its OWN support. Do not attach the same cohort description, sample size, or method sentence to several claims — cite the evidence specific to that finding: how that particular outcome was measured, its own effect size, its own comparison, its own timepoint.

VARIETY IS REQUIRED, and each claim is ASSIGNED the kind of evidence its support sentence must cite. Follow the assignment for each claim in order:
{support_kinds}
Most abstracts describe one cohort analysed one way. Naming that same cohort and method in every support sentence is a failure even when each claim is individually correct — the assignment above exists to stop that. No two assertions may open with the same subject phrase either; vary how each claim enters its finding.

The assignment governs the SUPPORT sentence ONLY. The "assertion" sentence must always state a substantive finding — {assertion} — because that is the point a disagreeing report would have to deny. Do not turn the assertion itself into a description of the study design, how the cohort was assembled, or what was measured.

The support and detail sentences are freer: they may carry methods, cohort description, background, or incidental information that is not itself a finding. That is fine and welcome — it is what a real passage looks like. Only the assertion carries the contradictable point.

Across all three sentences a claim must:
- Be fully self-contained: name the population, the intervention or exposure, and the outcome explicitly. A reader who sees only this claim, with no abstract, must understand it completely.
- Never refer to the source: no "this study", "the authors", "our data", "the abstract", "these patients", "the trial". Write it as an assertion about the world.
- Be grounded in the abstract. Do not invent findings, numbers, or populations that are not there.
- Vary in structure across claims — do not open every claim with the same phrasing or template.

The claims must be ORTHOGONAL to each other: each one hinges on a different finding, so that disproving any one of them would leave the others untouched. Do not restate the same result with different wording, and do not write a claim that is a special case of another.

In particular, NEVER write an umbrella claim that aggregates, summarizes, or generalizes over the others. All of these are forbidden, because denying one of them would also deny the individual claims:
  - "Both X and Y predict Z" when X-predicts-Z and Y-predicts-Z are already separate claims.
  - "The association holds after adjustment" / "the results were robust" as a claim in its own right — attach that to the claim it qualifies, as its detail sentence.
  - "Treatment T is effective against condition C" alongside claims about T's individual effects on specific outcomes.
Test each claim before you output it: if denying it would force any OTHER claim in your list to be false too, replace it with a claim about a distinct finding.

If the abstract genuinely supports fewer than {k_max} orthogonal claims, write fewer. Quality over count.

Return ONLY a JSON array, no preamble or code fences. Each element:
  {{"assertion": "<sentence>", "support": "<sentence>", "detail": "<sentence or empty string>", "pivot": "<this claim's OWN core assertion, stated positively in <=12 words -- what a disagreeing report would have to deny. Do NOT write the negation; write the thing being asserted>"}}

Abstract:
{abstract}

JSON:"""


# Rotated across the claims of one abstract so each support sentence is forced
# onto different evidence. Asking for "variety" in prose did not work: Qwen3-8B
# reused one cohort-and-method frame for every claim (support overlap 0.74).
# ──────────────────────────── domain profiles ──────────────────────────────
# The claim schema was tuned on clinical abstracts, whose evidence is effect
# sizes, cohorts and p-values. arXiv theory/physics abstracts have none of
# those, and forcing the clinical schema onto them makes the model invent
# evidence -- worse than having no diversity. Each source therefore carries its
# own field name, assertion definition, evidence kinds, and subject-fixing
# examples, selected per-abstract via the `domain` field.
DOMAIN_PROFILES = {
    "biomed": {
        "field": "biomedical",
        "assertion": "a relationship between an exposure, intervention, or "
                     "characteristic and an outcome",
        "support_kinds": [
            "the numeric effect size and its uncertainty -- the ratio, "
            "interval, or p-value for THIS finding, without describing the "
            "cohort",
            "the study design and who was studied -- cohort size, setting, "
            "dates, or how comparison groups were formed",
            "how this outcome or exposure was measured, defined, or classified",
            "a subgroup, timing, dose-response, or sensitivity analysis "
            "bearing on this finding",
        ],
        "subject_examples":
            "- Do not substitute a different drug, treatment, or exposure for "
            "the one the original studied.\n"
            "- Do not move the timepoint, follow-up length, or date range (do "
            "not answer \"at 12 months\" with \"at nine months\").\n"
            "- Do not switch to a different outcome measure, or to a different "
            "or complementary population (do not answer a claim about patients "
            "WITH a condition with one about those WITHOUT it).",
    },
    "arxiv": {
        "field": "scientific",
        "assertion": "a substantive finding -- how something behaves, scales, "
                     "performs, or is bounded, and under what conditions",
        "support_kinds": [
            "the quantitative result for THIS finding -- the measured value, "
            "bound, rate, or benchmark number, with its uncertainty if given",
            "the setup it was obtained in -- system size, dataset, model or "
            "sample scale, energy range, or experimental/simulation "
            "configuration",
            "how it was derived, measured, or evaluated -- the method, "
            "approximation, instrument, proof technique, or evaluation "
            "protocol",
            "a regime, limiting case, assumption, ablation, or condition under "
            "which it holds or breaks down",
        ],
        "subject_examples":
            "- Do not substitute a different system, model, dataset, or method "
            "for the one the original studied.\n"
            "- Do not move the regime, scale, or parameter range (do not answer "
            "a claim about the strong-coupling limit with one about weak "
            "coupling).\n"
            "- Do not switch to a different measured quantity, benchmark, or "
            "evaluation setting.",
    },
}
DOMAIN_PROFILES["pubmedqa"] = DOMAIN_PROFILES["biomed"]


def profile(domain):
    """Prompt profile for an abstract's source domain (defaults to biomed)."""
    return DOMAIN_PROFILES.get(domain or "biomed", DOMAIN_PROFILES["biomed"])


SUPPORT_KINDS = [
    "the numeric effect size and its uncertainty — the ratio, interval, or "
    "p-value for THIS finding, without describing the cohort",
    "the study design and who was studied — cohort size, setting, dates, or "
    "how comparison groups were formed",
    "how this outcome or exposure was measured, defined, or classified",
    "a subgroup, timing, dose-response, or sensitivity analysis bearing on "
    "this finding",
]


def support_kind_block(k, domain=None):
    kinds = profile(domain)["support_kinds"]
    return "\n".join(f"  - Claim {i+1}: cite {kinds[i % len(kinds)]}."
                     for i in range(k))


# JSON strings may only contain the escapes  \" \\ \/ \b \f \n \r \t \uXXXX.
# arXiv abstracts are full of LaTeX ($p\bar{p}$, $\sqrt{s}$, $\Omega$), which the
# model copies into its JSON verbatim and json.loads then rejects with
# "Invalid \escape". This cost 301 of 302 claim-generation failures at the full
# build -- ~11% of arXiv abstracts, and essentially 0% of PubMed ones.
_BAD_ESCAPE = re.compile(r'\\(?![\"\\/bfnrt]|u[0-9a-fA-F]{4})')


def repair_json_escapes(text):
    """Double any backslash that is not a legal JSON escape."""
    return _BAD_ESCAPE.sub(r"\\\\", text)


def parse_claims_json(text):
    """Pull the JSON array of claims out of a model response.

    Tolerates code fences and leading prose; returns [] if nothing parses.
    """
    if not text:
        return []
    t = text.strip()
    # Qwen3-class models emit a reasoning block even when it isn't asked for,
    # and sometimes leave it unclosed (see memory note on think-tag truncation).
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
    if "<think>" in t:
        t = t.split("<think>")[0]
    t = t.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    blob = t[start:end + 1]
    try:
        arr = json.loads(blob)
    except json.JSONDecodeError:
        try:
            arr = json.loads(repair_json_escapes(blob))
        except json.JSONDecodeError:
            return []
    if not isinstance(arr, list):
        return []
    out = []
    for e in arr:
        # Models sometimes double-encode: a list of JSON *strings* rather than
        # a list of objects. Unwrap one level before giving up on the element.
        if isinstance(e, str):
            try:
                e = json.loads(e)
            except json.JSONDecodeError:
                continue
        if not isinstance(e, dict):
            continue
        # The claim is emitted as role-labelled sentence fields and joined here.
        # Asking for free-form "2-3 sentences" did not work: the model reliably
        # compressed everything into one long sentence (19/20 at the pilot).
        # Making each sentence its own field makes the structure unskippable.
        assertion = (e.get("assertion") or "").strip()
        support = (e.get("support") or "").strip()
        detail = (e.get("detail") or "").strip()
        # assertion + support are REQUIRED; only detail is optional. A claim
        # missing either is dropped rather than silently emitted as a shorter
        # one, which would reintroduce the single-sentence failure mode.
        if e.get("claim") is None and not (assertion and support):
            continue
        parts = [p for p in (assertion, support, detail) if p]
        for i, p in enumerate(parts):
            if not p.endswith((".", "!", "?")):
                parts[i] = p + "."
        claim = " ".join(parts)
        if not claim and e.get("claim"):
            claim = str(e["claim"]).strip()   # tolerate the older flat schema
            parts = [claim]
        pivot = (e.get("pivot") or "").strip()
        if claim:
            out.append({"claim": claim, "pivot": pivot, "parts": parts})
    return out


# Surface-level rejects. These are the failure modes that would silently poison
# the corpus: a claim that points outside itself is unreadable once shuffled.
ANAPHORA = re.compile(
    r"\b(this|these|the present|our|the current)\s+"
    r"(stud(y|ies)|trial|analys[ei]s|work|paper|report|data|cohort|"
    r"patients|findings?|results?|experiments?)\b",
    re.IGNORECASE,
)
SOURCE_REF = re.compile(
    r"\b(the authors?|the abstract|as described above|as reported here)\b",
    re.IGNORECASE,
)


STOPWORDS = set(
    "the a an of in and or to for with was were is are that this these those by "
    "on at as from be been has have had we our their its it not no than more "
    "less also which who when both after before during between compared".split()
)


def content_jaccard(a, b):
    """Content-word Jaccard. Used both for the assertion/support restatement
    check and, later, for gold-pair vs. hard-negative overlap margins."""
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower())) - STOPWORDS
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower())) - STOPWORDS
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def claim_flags(claim, parts=None, restate_max=0.45):
    """Cheap quality flags for a generated claim (advisory, not a hard filter
    at pilot scale — we want to see what the model does before rejecting)."""
    flags = []
    words = claim.split()
    n_sent = len([s for s in re.split(r"(?<=[.!?])\s+", claim.strip()) if s])
    if not (25 <= len(words) <= 95):
        flags.append(f"len={len(words)}w")
    if not (2 <= n_sent <= 3):
        flags.append(f"sents={n_sent}")
    if ANAPHORA.search(claim):
        flags.append("anaphora")
    if SOURCE_REF.search(claim):
        flags.append("source_ref")
    # The support sentence is supposed to carry evidence the assertion does not
    # already contain; heavy token reuse means it just re-said the assertion.
    if parts and len(parts) >= 2:
        r = content_jaccard(parts[0], parts[1])
        if r > restate_max:
            flags.append(f"restates={r:.2f}")
    return flags


def gen_claims(args):
    abstracts = [json.loads(l) for l in open(args.abstracts) if l.strip()]
    if args.limit:
        abstracts = abstracts[:args.limit]
    print(f"{len(abstracts)} abstracts -> claims "
          f"({args.model} @ {args.base_url})")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    client = LocalChatClient(
        base_url=args.base_url, model=args.model,
        max_concurrent=args.max_concurrent,
        cache_path=None if args.no_cache
        else os.path.join(out_dir, "cache",
                          f"claims_responses_{args.cache_tag}.jsonl"),
        extra_body=({"chat_template_kwargs": {"enable_thinking": False}}
                    if args.no_think else None),
    )
    prompts = []
    for a in abstracts:
        pr = profile(a.get("domain"))
        prompts.append(CLAIMS_PROMPT.format(
            k_max=args.claims_per_abstract,
            field=pr["field"], assertion=pr["assertion"],
            support_kinds=support_kind_block(args.claims_per_abstract,
                                             a.get("domain")),
            abstract=a["abstract"]))
    responses = client.complete_many(
        prompts, temperature=args.temperature, max_tokens=args.max_tokens)

    n_claims = n_fail = 0
    with open(args.out, "w") as f:
        for a, raw in zip(abstracts, responses):
            raw = raw or ""
            claims = parse_claims_json(raw)
            if not claims:
                n_fail += 1
                print(f"  [parse-fail] aid={a['aid']} raw={raw[:160]!r}",
                      file=sys.stderr)
            for j, c in enumerate(claims):
                f.write(json.dumps({
                    "cid": f"{a['aid']}_{j}",
                    "aid": a["aid"],
                    "domain": a.get("domain", "biomed"),
                    "claim": c["claim"],
                    "pivot": c["pivot"],
                    "parts": c.get("parts", []),
                    "flags": claim_flags(c["claim"], c.get("parts")),
                }) + "\n")
                n_claims += 1
    print(f"wrote {n_claims} claims from {len(abstracts)} abstracts "
          f"({n_fail} parse failures) -> {args.out}")


# ──────────────────────────── phase 3: contradictions ──────────────────────

# A contradiction TYPE is drawn per claim so the corpus is not dominated by one
# move (the old pipeline's failure: near-universal polarity flips, trivially
# detectable by string overlap). Each type names a different way two findings
# can be jointly impossible.
CONTRADICTION_TYPES = [
    ("direction",
     "report the OPPOSITE direction for the same relationship — if the original "
     "found benefit/increase/protection, yours finds harm/decrease/none — as a "
     "different study would phrase it, never by inserting 'not' or swapping a "
     "single antonym"),
    ("magnitude",
     "report an effect size for the same relationship that is incompatible with "
     "the original's — a negligible or clinically irrelevant effect where the "
     "original found a large one, or vice versa, with your own numbers"),
    ("number",
     "state a concrete value — count, dose, threshold, rate, or measured "
     "quantity — that is logically incompatible with the one the original gives "
     "for the same quantity"),
    ("scope",
     "contradict the original WITHIN a population or setting it explicitly "
     "covered, so the two cannot both hold for that group"),
    ("temporal",
     "report a conflicting timing, duration, or follow-up window — an effect the "
     "original places at some timepoint is absent or reversed at that same "
     "timepoint"),
    ("significance",
     "flip the statistical conclusion for the same relationship "
     "(significant <-> not significant, independent <-> confounded) using your "
     "own p-values, intervals, and wording"),
    ("comparator",
     "reverse which group, treatment, or condition comes out higher or better "
     "against the same comparator"),
]

CONTRA_PROMPT = """You are building a benchmark of conflicting findings in the {field} literature.

Below is a claim. Write ONE new claim, reporting what a DIFFERENT study found, that genuinely CONTRADICTS it — both cannot be true of the same underlying question.

Make the contradiction of this kind: {type_desc}.

It must deny this specific point: "{pivot}"

HOLD THE SUBJECT FIXED. Your claim must be about the SAME subject, the SAME measured quantity, the SAME setting, and the SAME regime or timepoint as the original. Changing any of those produces a claim about a different question, which is NOT a contradiction — it is the most common way this task is failed. Concretely:
{subject_examples}
You may rename these things — different words for the same drug, cohort, or measure are good — but the thing itself must be the same. Change only what is ASSERTED about it.

Requirements:
- A domain expert reading both claims would say "these two disagree". It must NOT be a paraphrase, a merely different topic, or a finding that could coexist with the original.
- Write it as a free-standing result. NEVER reference the other claim: no "contrary to", "unlike previous reports", "in contrast to earlier findings", "however". A reader must not be able to tell it was written in response to anything.
- Share as FEW words with the original as you can. Rename the entities (synonyms, alternative names, expand or contract abbreviations), pick different verbs, open differently, and use a different sentence structure, study design, and population wording. It must not look like the original with a few words changed.
- FORBIDDEN: taking the original's sentence and swapping its verb or adjective for the opposite ("reduces" -> "increases", "effective" -> "ineffective"). That is the one move that makes the pair findable by string matching alone. Rewrite the finding from scratch.
- Numbers split into two kinds, and they are treated OPPOSITELY:
  - Numbers that DEFINE THE SUBJECT — the dose administered, the timepoint or follow-up at which the outcome is assessed, the age range or threshold defining the population — must MATCH the original. Changing them changes the question, which is the failure described above. If the original studied 1.3 and 2.6 g/kg, your claim is also about 1.3 and 2.6 g/kg.
  - Numbers that REPORT THE RESULT — effect sizes, ratios, p-values, confidence intervals, rates, and your own study's sample size — must be YOUR OWN, not the original's, and must be consistent with what you assert.
- Keep it a specific, plausible {field} claim with its own concrete numbers.
- Contradict ONLY the point named above. Do not additionally deny the mechanism, the measurement method, or any other finding the original mentions.
{compat_block}
Use the SAME three-sentence structure as the original:
  - "assertion": the core conflicting finding. One sentence.
  - "support": its evidence — effect size, cohort size, design, follow-up, dose, or measurement. One sentence. Must carry facts not already in the assertion, and must NOT re-describe the direction of the effect.
  - "detail": one further grounded fact — a condition, subgroup, mechanism, or secondary measure. One sentence.

Return ONLY a JSON object, no preamble or code fences:
  {{"assertion": "<sentence>", "support": "<sentence>", "detail": "<sentence>", "pivot": "<the point it asserts, <=12 words>"}}

Original claim:
{claim}

JSON:"""

COMPAT_BLOCK = """- Your new claim must remain COMPATIBLE with each of the related claims below. It must contradict ONLY the original claim above, not these. Do not touch the findings they report:
{siblings}
- Careful with renaming here. Renaming the outcome or exposure is encouraged, but the name you choose must NOT make your claim describe the same thing as any of those related claims. Clinical synonyms are the trap: if a related claim reports "small for gestational age", do not phrase your outcome as "fetal growth restriction" or "low birth weight", because those name the same underlying finding and your claim would then contradict that related claim too. Pick wording that stays clearly inside the original claim's own outcome.
"""

# Judge prompts. Validity asks the direct question; uniqueness re-uses it against
# each sibling, where the required answer is NO.
JUDGE_PROMPT = """Claim A:
{a}

Claim B:
{b}

Is there a SPECIFIC point that one claim asserts and the other denies, so that both claims cannot be true at the same time?

Rule it a contradiction ONLY if both claims speak about the SAME exposure or intervention, the SAME outcome, and the SAME population, and assert incompatible things about it.

It is NOT a contradiction if:
- they concern different exposures, interventions, outcomes, or populations — even closely related ones, and even within the same research area;
- one is a paraphrase or restatement of the other;
- both could hold at once, even if the combination would be surprising, hard to explain, or mechanistically awkward;
- one merely makes the other less plausible, or they sit oddly together.

Work in two steps and use exactly this format:

POINT: <the one thing one claim asserts and the other denies — quote the specific exposure, outcome and population — or the word NONE>
VERDICT: <YES or NO>"""


def parse_obj(text):
    """Parse a single JSON object response (tolerates fences, prose, or the
    object wrapped in a one-element array)."""
    if not text:
        return None
    t = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL)
    if "<think>" in t:
        t = t.split("<think>")[0]
    t = re.sub(r"^```(?:json)?\s*", "", t.strip())
    t = re.sub(r"\s*```$", "", t)
    for lo, hi in (("{", "}"), ("[", "]")):
        s, e = t.find(lo), t.rfind(hi)
        if s == -1 or e <= s:
            continue
        try:
            v = json.loads(t[s:e + 1])
        except json.JSONDecodeError:
            try:
                v = json.loads(repair_json_escapes(t[s:e + 1]))
            except json.JSONDecodeError:
                continue
        if isinstance(v, list):
            v = v[0] if v and isinstance(v[0], dict) else None
        if isinstance(v, dict):
            return v
    return None


def build_claim_text(obj):
    """Join the role fields into the claim paragraph, enforcing that both the
    assertion and the support are present (same contract as phase 2)."""
    if not obj:
        return None, []
    a = (obj.get("assertion") or "").strip()
    s = (obj.get("support") or "").strip()
    d = (obj.get("detail") or "").strip()
    if not (a and s):
        return None, []
    parts = [p if p.endswith((".", "!", "?")) else p + "."
             for p in (a, s, d) if p]
    return " ".join(parts), parts


def _attempt_contradictions(claims, by_aid, client, args, attempt):
    """One generation round over `claims`. The contradiction type is drawn with
    an attempt-dependent offset so a retry asks for a genuinely different move
    (and lands on a different cache key) rather than resampling the same one."""
    import random

    rng = random.Random(args.seed + 1000 * attempt)
    types, prompts = [], []
    for c in claims:
        tname, tdesc = CONTRADICTION_TYPES[rng.randrange(len(CONTRADICTION_TYPES))]
        types.append(tname)
        sibs = [s for s in by_aid[c["aid"]] if s["cid"] != c["cid"]]
        compat = ""
        if sibs and not args.no_sibling_guard:
            listed = "\n".join(f"  - {s['claim']}" for s in sibs)
            compat = COMPAT_BLOCK.format(siblings=listed)
        pr = profile(c.get("domain"))
        prompts.append(CONTRA_PROMPT.format(
            type_desc=tdesc, pivot=c.get("pivot", ""),
            field=pr["field"], subject_examples=pr["subject_examples"],
            claim=c["claim"], compat_block=compat))

    # Nudge temperature up on retries to escape whatever attractor failed first.
    temp = min(1.0, args.temperature + 0.1 * attempt)
    responses = client.complete_many(prompts, temperature=temp,
                                     max_tokens=args.max_tokens)

    records, n_fail = [], 0
    for c, tname, raw in zip(claims, types, responses):
        obj = parse_obj(raw or "")
        text, parts = build_claim_text(obj)
        if not text:
            n_fail += 1
            continue
        sibs = [s for s in by_aid[c["aid"]] if s["cid"] != c["cid"]]
        # The margin that decides whether the task is solvable by string
        # matching: the gold pair must not be more lexically similar than the
        # most similar non-gold pair available in the same example.
        sib_ov = [content_jaccard(text, s["claim"]) for s in sibs]
        src_sib_ov = [content_jaccard(c["claim"], s["claim"]) for s in sibs]
        records.append({
            "pair_id": f"{c['cid']}__contra",
            "aid": c["aid"],
            "src_cid": c["cid"],
            "domain": c.get("domain", "biomed"),
            "type": tname,
            "attempt": attempt + 1,
            "src_claim": c["claim"],
            "src_pivot": c.get("pivot", ""),
            "contra_claim": text,
            "contra_parts": parts,
            "contra_pivot": (obj.get("pivot") or "").strip(),
            "gold_overlap": round(content_jaccard(c["claim"], text), 4),
            "max_sibling_overlap": round(max(sib_ov), 4) if sib_ov else None,
            "max_src_sibling_overlap": round(max(src_sib_ov), 4) if src_sib_ov else None,
            "flags": claim_flags(text, parts),
        })
    return records, n_fail


def _is_clean(r, args):
    """Is this pair usable as a gold pair somewhere?

    Note what is NOT required: that the contradiction leaves every sibling
    untouched. Measured over 400 claims, 127 of the 197 unrecoverable pairs
    failed for that reason alone — 65% of all waste — even though such a pair is
    perfectly good in any example that simply omits the sibling it collides
    with. Sibling conflicts are therefore recorded as a placement CONSTRAINT
    (see the conflict matrix) rather than treated as a defect, which lifts
    usable yield from 51% to 82%.

    What genuinely disqualifies a pair: it is not a real contradiction, it
    contradicts itself, or it is so close in wording that string matching finds
    it.
    """
    return (r.get("judge_valid") is not False
            and r.get("coherent") is not False
            and r["gold_overlap"] <= args.max_overlap)


def build_conflict_matrix(claims, records, client, args):
    """Compute, once per abstract, which items contradict which.

    This replaces per-example verification. Verifying at assembly time cost
    O(items^2) judge calls per example and would have been unaffordable at the
    long rungs (~8.5k calls for a single 570-item example, times 20k examples).
    Every collision observed so far lives WITHIN an abstract — different
    abstracts are different topics — and an abstract has only ~8 items, so the
    matrix is ~28 pairs computed once and reused by every example that draws
    from it. Assembly then becomes a table lookup with no LLM calls at all, at
    any rung size.

    Returns ``{"edges": [[keyA, keyB], ...]}`` over claim cids and pair_ids.
    Gold partners are excluded — those pairs are supposed to contradict.
    """
    by_aid = {}
    for c in claims:
        by_aid.setdefault(c["aid"], {"claims": [], "contras": []})["claims"].append(c)
    for r in records:
        if r["aid"] in by_aid and not r.get("rejected"):
            by_aid[r["aid"]]["contras"].append(r)

    partner = {}
    known = set()
    for r in records:
        partner[r["src_cid"]] = r["pair_id"]
        partner[r["pair_id"]] = r["src_cid"]
        # sibling verdicts already paid for during generation
        for cid in r.get("conflicts_siblings") or []:
            known.add(tuple(sorted((r["pair_id"], cid))))

    todo, texts = [], {}
    for aid, g in by_aid.items():
        items = ([(c["cid"], c["claim"]) for c in g["claims"]]
                 + [(r["pair_id"], r["contra_claim"]) for r in g["contras"]])
        texts.update(dict(items))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                ka, kb = items[i][0], items[j][0]
                if partner.get(ka) == kb:
                    continue
                key = tuple(sorted((ka, kb)))
                if key in known:
                    continue
                todo.append(key)
    todo = sorted(set(todo))

    print(f"  conflict matrix: {len(todo)} within-abstract pairs to judge "
          f"({len(known)} already known from sibling checks)")
    edges = [list(k) for k in sorted(known)]
    if todo:
        prompts = []
        for ka, kb in todo:
            prompts.append(JUDGE_PROMPT.format(a=texts[ka], b=texts[kb]))
            prompts.append(JUDGE_PROMPT.format(a=texts[kb], b=texts[ka]))
        verdicts = client.complete_many(prompts, temperature=0.0, max_tokens=160)
        for n, (ka, kb) in enumerate(todo):
            if all(_verdict_yes(v) for v in verdicts[2 * n:2 * n + 2]):
                edges.append([ka, kb])
    print(f"  conflict matrix: {len(edges)} conflicting pairs recorded")
    return {"edges": edges}


def gen_contradictions(args):
    claims = [json.loads(l) for l in open(args.claims) if l.strip()]
    by_aid = {}
    for c in claims:
        by_aid.setdefault(c["aid"], []).append(c)
    print(f"{len(claims)} claims -> contradictions "
          f"({args.model} @ {args.base_url})")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    client = LocalChatClient(
        base_url=args.base_url, model=args.model,
        max_concurrent=args.max_concurrent,
        cache_path=None if args.no_cache
        else os.path.join(out_dir, "cache",
                          f"contra_responses_{args.cache_tag}.jsonl"),
        extra_body=({"chat_template_kwargs": {"enable_thinking": False}}
                    if args.no_think else None),
    )

    kept, pending, total_fail = [], list(claims), 0
    by_cid = {c["cid"]: c for c in claims}
    for attempt in range(args.max_attempts):
        if not pending:
            break
        label = "generating" if attempt == 0 else f"RETRY {attempt}"
        print(f"  [{label}] {len(pending)} claims")
        recs, n_fail = _attempt_contradictions(pending, by_aid, client, args,
                                               attempt)
        total_fail += n_fail
        if args.judge:
            judge_coherence(recs, client, args)
            judge_pairs(recs, by_aid, client, args)
        good = [r for r in recs if _is_clean(r, args)]
        kept.extend(good)
        got = {r["src_cid"] for r in recs}
        bad = {r["src_cid"] for r in recs if not _is_clean(r, args)}
        # claims whose generation failed to parse are also still pending
        pending = [by_cid[cid] for cid in
                   sorted(bad | ({c["cid"] for c in pending} - got))]
        print(f"    kept {len(good)}, {len(pending)} still failing")
        if not args.retry_rejected:
            break

    # Keep the last (failed) attempt for anything that never came clean, marked
    # so assembly can exclude it — silently dropping would hide the yield loss.
    if pending:
        recs, _ = _attempt_contradictions(pending, by_aid, client, args,
                                          args.max_attempts - 1)
        if args.judge:
            judge_coherence(recs, client, args)
            judge_pairs(recs, by_aid, client, args)
        for r in recs:
            r["rejected"] = True
        kept.extend(recs)

    kept.sort(key=lambda r: (r["aid"], r["src_cid"]))
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    if args.judge and args.conflict_matrix:
        matrix = build_conflict_matrix(claims, kept, client, args)
        mpath = re.sub(r"\.jsonl$", "", args.out) + ".conflicts.json"
        with open(mpath, "w") as f:
            json.dump(matrix, f)
        print(f"  wrote conflict matrix -> {mpath}")

    n_ok = sum(1 for r in kept if not r.get("rejected"))
    print(f"wrote {len(kept)} pairs: {n_ok} usable, "
          f"{len(kept)-n_ok} unrecoverable, {total_fail} parse failures "
          f"-> {args.out}")


# Neither judge above looks at a claim on its own, so a claim whose support
# numbers contradict its own assertion passed everything: "absence of OA is
# linked to HIGHER incidence ... odds ratio 0.62". This closes that.
COHERENCE_PROMPT = """Claim:
{claim}

The first sentence states a finding. The later sentences should be consistent with it — though they may also carry methods, cohort description, or incidental context, which is perfectly fine and is NOT a problem.

Is any later sentence INCONSISTENT with the first sentence? Check especially:
- direction: if the first sentence says a risk or quantity is increased, a reported ratio below 1.0 (RR/OR/HR < 1) contradicts it; if it says decreased or reduced, a ratio above 1.0 contradicts it.
- significance: if the first sentence claims no association or no effect, a reported statistically significant effect contradicts it — and the reverse.
- magnitude: a reported effect size that cannot be squared with what the first sentence asserts.

Do NOT flag a sentence merely for describing the study, the participants, or unrelated background.

Use exactly this format:
PROBLEM: <the inconsistency, quoting both sentences — or the word NONE>
VERDICT: <YES if inconsistent, NO if consistent>"""


# Direction of the assertion. `null` is tested FIRST because "does not elevate"
# contains an up-word while actually asserting no effect.
_UP = re.compile(r"\b(increase[sd]?|higher|elevate[sd]?|greater|raise[sd]?|"
                 r"rise[sn]?|more likely|positively associated)\b", re.I)
_DOWN = re.compile(r"\b(decrease[sd]?|lower|reduce[sd]?|less likely|"
                   r"protective|inverse|suppress(es|ed)?|inhibit(s|ed)?)\b", re.I)
_NULL = re.compile(r"\b(no association|not associated|does not|do not|did not|"
                   r"no significant|no statistically significant|shows? no|"
                   r"no difference|no link|no effect|unchanged|negligible|"
                   r"not linked|not predict)\b", re.I)
# Point estimate following a ratio name. Written to grab the estimate, not the
# confidence-interval bounds that usually follow it in parentheses.
_RATIO_WORD = re.compile(
    r"(?:relative risk|risk ratio|odds ratio|hazard ratio|ratio)\s*"
    r"(?:of|was|were|=|:|at)?\s*(\d+(?:\.\d+)?)", re.I)
_RATIO_ABBR = re.compile(r"\b(?:RR|OR|HR)\s*(?:=|of|was)?\s*(\d+(?:\.\d+)?)")
_SUPPORT_NULL = re.compile(
    r"\b(no significant|not significant|no statistically significant|"
    r"no difference|no link|no association|no effect|unchanged)\b", re.I)


def numeric_coherence_problems(parts):
    """Return reasons a claim's support contradicts its own assertion.

    Deliberately deterministic. An LLM judge was tried first and missed every
    hand-verified case at 8B (``debug/pubmed_multiclaim/test_coherence.py``):
    deciding that an odds ratio of 0.62 conflicts with "higher incidence" is
    exactly the kind of small numeric inference a small model fumbles, and it is
    trivial to check directly.
    """
    if not parts:
        return []
    assertion, support = parts[0], " ".join(parts[1:])
    ratios = [float(x) for x in
              _RATIO_WORD.findall(support) + _RATIO_ABBR.findall(support)]
    problems = []

    if _NULL.search(assertion):
        for r in ratios:
            if r > 1.25 or r < 0.8:
                problems.append(
                    f"asserts no effect but reports ratio {r}")
                break
        if re.search(r"\b(inverse association|protective effect)\b",
                     support, re.I):
            problems.append("asserts no effect but support describes a direction")
    elif _UP.search(assertion):
        for r in ratios:
            if r < 0.95:
                problems.append(f"asserts an increase but reports ratio {r}")
                break
        if _SUPPORT_NULL.search(support):
            problems.append("asserts an increase but support reports no effect")
    elif _DOWN.search(assertion):
        for r in ratios:
            if r > 1.05:
                problems.append(f"asserts a decrease but reports ratio {r}")
                break
        if _SUPPORT_NULL.search(support):
            problems.append("asserts a decrease but support reports no effect")
    return problems


def judge_coherence(records, client, args):
    """Flag contradictions whose own support numbers fight their own assertion."""
    n_bad = 0
    for r in records:
        probs = numeric_coherence_problems(r.get("contra_parts") or [])
        r["coherent"] = not probs
        r["coherence_problems"] = probs
        n_bad += bool(probs)
    print(f"  coherence check: {n_bad}/{len(records)} internally inconsistent")


def _verdict_yes(text):
    """Read the VERDICT line out of a POINT/VERDICT judge response."""
    if not text:
        return False
    m = re.search(r"VERDICT\s*:\s*(YES|NO)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper() == "YES"
    return text.strip().upper().startswith("YES")


def judge_pairs(records, by_aid, client, args):
    """Two LLM checks per pair: (a) is it a real contradiction, (b) does it
    leave the source claim's siblings untouched (unique gold pair).

    Every pair is judged in BOTH orders and must agree. A single-order judge
    over-called badly at the pilot — it flagged plainly compatible findings as
    conflicting whenever they shared a topic — and order-swap agreement is the
    cheapest way to drop that false-positive rate.
    """
    prompts, index = [], []

    def add(i, kind, cid, a, b):
        prompts.append(JUDGE_PROMPT.format(a=a, b=b))
        index.append((i, kind, cid))
        prompts.append(JUDGE_PROMPT.format(a=b, b=a))
        index.append((i, kind, cid))

    for i, r in enumerate(records):
        add(i, "valid", None, r["src_claim"], r["contra_claim"])
        # Sibling conflicts are NOT judged here. They no longer decide whether a
        # pair is usable (they are placement constraints now), and the conflict
        # matrix computes them once at the end over the surviving pairs. Judging
        # them inside the retry loop re-judged the same relation on every
        # attempt and accounted for ~43% of all LLM calls for zero effect on
        # which pairs were kept.
        if args.judge_siblings:
            for s in by_aid[r["aid"]]:
                if s["cid"] != r["src_cid"]:
                    add(i, "sibling", s["cid"], s["claim"], r["contra_claim"])

    print(f"  judging {len(prompts)} prompts "
          f"({len(records)} pairs + {len(prompts)//2-len(records)} sibling "
          f"checks, each in both orders)...")
    verdicts = client.complete_many(prompts, temperature=0.0, max_tokens=160)

    votes = {}
    for (i, kind, cid), v in zip(index, verdicts):
        votes.setdefault((i, kind, cid), []).append(_verdict_yes(v))
    for r in records:
        r["judge_valid"] = None
        r["conflicts_siblings"] = []
    for (i, kind, cid), vs in votes.items():
        agree_yes = all(vs)          # must hold in both directions
        if kind == "valid":
            records[i]["judge_valid"] = agree_yes
        elif agree_yes:
            records[i]["conflicts_siblings"].append(cid)


# ────────────────────────────── phase 4: assemble ──────────────────────────

def verify_example(items, gold, client, args):
    """Check that the assembled example contains ONLY the K labelled pairs.

    ``items`` are ``(text, provenance, key, aid)``.

    An earlier version only checked GOLD items against the rest, and a planted
    contradiction between two DISTRACTORS went undetected
    (``debug/pubmed_multiclaim/test_verifier.py``). Every item is now a
    candidate, because a sibling colliding with an orphan contradiction is just
    as fatal as one involving a gold item.

    Judging all N^2 pairs is unaffordable at the long rungs, so two cheap
    candidate sets are judged instead:

    - **all within-abstract pairs.** Claims and contradictions derived from the
      same abstract are where essentially every real collision lives, and the
      groups are small, so this is exhaustive where it matters.
    - **a lexical top-k backstop** across the whole example, for the rare case
      of two different abstracts covering the same topic. Note this is only a
      backstop: contradictions here are deliberately low-overlap, so lexical
      rank alone is a weak detector and must not be relied on by itself.

    Returns a list of offending pair descriptions (empty = example is clean).
    """
    partner = {}
    for p in gold:
        partner[p["src_cid"]] = p["pair_id"]
        partner[p["pair_id"]] = p["src_cid"]

    def allowed(ka, kb):
        return partner.get(ka) == kb

    cand = set()
    by_aid = {}
    for idx, (_, _, key, aid) in enumerate(items):
        by_aid.setdefault(aid, []).append(idx)
    for group in by_aid.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                cand.add((group[i], group[j]))
    for i, (text, _, key, _) in enumerate(items):
        ranked = sorted(((content_jaccard(text, t), j)
                         for j, (t, _, _, _) in enumerate(items) if j != i),
                        key=lambda z: -z[0])
        for _, j in ranked[:args.verify_k]:
            cand.add((min(i, j), max(i, j)))

    checks = [(i, j) for i, j in sorted(cand)
              if not allowed(items[i][2], items[j][2])]
    if not checks:
        return []

    prompts = []
    for i, j in checks:
        prompts.append(JUDGE_PROMPT.format(a=items[i][0], b=items[j][0]))
        prompts.append(JUDGE_PROMPT.format(a=items[j][0], b=items[i][0]))
    verdicts = client.complete_many(prompts, temperature=0.0, max_tokens=160)

    offending = []
    for n, (i, j) in enumerate(checks):
        if all(_verdict_yes(v) for v in verdicts[2 * n:2 * n + 2]):
            offending.append(f"{items[i][2]} vs {items[j][2]}")
    return offending


def assemble(args):
    """Build corpus examples in the unified schema consumed by
    ``lib/data_format.py`` with ``--task contradiction``.

    Each example holds N claims, K of which form contradicting pairs. The
    distractors are drawn from three provenances on purpose:

      sibling   — another claim from a gold claim's own abstract. Same topic,
                  same entities, no conflict: the hard negative that makes
                  topical similarity useless as a heuristic.
      claim     — a claim from an unrelated abstract.
      orphan    — a generated CONTRADICTION whose source claim is deliberately
                  left OUT of the example. Without these, every "X does not
                  predict Y"-shaped item would be half of a gold pair, and the
                  task would collapse to "find the negative-sounding claims".

    gold_doc_indices holds 1-indexed pairs [[a, b], ...], matching what
    ``_build_output(task="contradiction")`` expects.
    """
    import random

    claims = [json.loads(l) for l in open(args.claims) if l.strip()]
    pairs = [json.loads(l) for l in open(args.contra) if l.strip()]
    usable = [p for p in pairs if not p.get("rejected")]
    by_cid = {c["cid"]: c for c in claims}
    by_aid = {}
    for c in claims:
        by_aid.setdefault(c["aid"], []).append(c)

    print(f"pool: {len(claims)} claims, {len(usable)} usable pairs, "
          f"{len(by_aid)} abstracts")
    if len(usable) < args.num_contradictions:
        raise SystemExit(f"need >= {args.num_contradictions} usable pairs")

    # Placement constraints from the per-abstract conflict matrix. `conflict[k]`
    # is the set of item keys that must not share an example with k (its gold
    # partner excepted). This is what lets a pair whose contradiction collides
    # with one sibling still be used — we simply do not place that sibling.
    conflict = collections.defaultdict(set)
    mpath = re.sub(r"\.jsonl$", "", args.contra) + ".conflicts.json"
    if os.path.exists(mpath):
        for ka, kb in json.load(open(mpath))["edges"]:
            conflict[ka].add(kb)
            conflict[kb].add(ka)
        print(f"  loaded conflict matrix: {len(conflict)} constrained items")
    else:
        print(f"  ! no conflict matrix at {mpath} — placement is UNCONSTRAINED "
              f"and examples may contain unlabelled contradictions")

    verifier = None
    if args.verify:
        verifier = LocalChatClient(
            base_url=args.base_url, model=args.model,
            max_concurrent=args.max_concurrent,
            cache_path=os.path.join(
                os.path.dirname(os.path.abspath(args.out)), "cache",
                f"verify_responses_{args.cache_tag}.jsonl"),
            extra_body=({"chat_template_kwargs": {"enable_thinking": False}}
                        if args.no_think else None),
        )

    rng = random.Random(args.seed)
    out_rows = []
    n_rejected_examples = 0
    for _ in range(args.num_examples):
        # 1. gold pairs, each from a different abstract so no two golds can
        #    interact through a shared source finding.
        rng.shuffle(usable)
        gold, used_aids, placed = [], set(), set()
        for p in usable:
            if p["aid"] in used_aids:
                continue
            # two gold pairs must not interfere with each other either
            if (conflict[p["src_cid"]] | conflict[p["pair_id"]]) & placed:
                continue
            gold.append(p)
            used_aids.add(p["aid"])
            placed.update({p["src_cid"], p["pair_id"]})
            if len(gold) == args.num_contradictions:
                break
        if len(gold) < args.num_contradictions:
            continue

        items = []          # (text, provenance, key, aid)
        banned_cids = set()  # claims that must not appear (would form a pair)
        for p in gold:
            items.append((p["src_claim"], "gold", p["src_cid"], p["aid"]))
            items.append((p["contra_claim"], "gold", p["pair_id"], p["aid"]))

        # 2. hard negatives: the gold claims' own siblings
        def may_place(key):
            """True if `key` contradicts nothing already in the example."""
            return not (conflict[key] & placed)

        sib_pool = [s for p in gold for s in by_aid[p["aid"]]
                    if s["cid"] != p["src_cid"]]
        rng.shuffle(sib_pool)
        n_sib = min(len(sib_pool),
                    int(round(args.sibling_frac * (args.num_docs - 2 * len(gold)))))
        n_sib_dropped = 0
        for s in sib_pool:
            if sum(1 for _, prov, _, _ in items if prov == "sibling") >= n_sib:
                break
            if not may_place(s["cid"]):
                n_sib_dropped += 1
                continue
            items.append((s["claim"], "sibling", s["cid"], s["aid"]))
            placed.add(s["cid"])

        # 3. orphan contradictions — their source claim is banned from the example
        orphan_pool = [p for p in usable if p not in gold]
        rng.shuffle(orphan_pool)
        n_orph = int(round(args.orphan_frac * (args.num_docs - 2 * len(gold))))
        chosen_keys = {k for _, _, k, _ in items}
        n_orph_placed = 0
        for p in orphan_pool:
            if n_orph_placed >= n_orph:
                break
            if p["src_cid"] in chosen_keys or not may_place(p["pair_id"]):
                continue
            items.append((p["contra_claim"], "orphan", p["pair_id"], p["aid"]))
            placed.add(p["pair_id"])
            banned_cids.add(p["src_cid"])
            n_orph_placed += 1

        # 4. fill the rest with claims from other abstracts.
        #    Prefer fillers from the SAME domain as the gold pairs. With a mixed
        #    PubMed+arXiv pool, a distractor from another field is trivially
        #    non-contradictory -- topic alone rules it out -- which would trade
        #    the lexical shortcut for a topical one. Same-domain fillers keep the
        #    discrimination semantic.
        gold_domains = {p.get("domain", "biomed") for p in gold}
        filler = [c for c in claims
                  if c["aid"] not in used_aids and c["cid"] not in banned_cids]
        rng.shuffle(filler)
        same = [c for c in filler if c.get("domain", "biomed") in gold_domains]
        other = [c for c in filler if c.get("domain", "biomed") not in gold_domains]
        n_same = int(round(args.same_domain_frac * len(filler)))
        filler = same[:n_same] + other + same[n_same:]
        for c in filler:
            if len(items) >= args.num_docs:
                break
            if c["cid"] in placed or not may_place(c["cid"]):
                continue
            items.append((c["claim"], "filler", c["cid"], c["aid"]))
            placed.add(c["cid"])

        if len(items) < args.num_docs:
            print(f"  ! pool too small: {len(items)}/{args.num_docs} items")

        rng.shuffle(items)
        pos = {k: i + 1 for i, (_, _, k, _) in enumerate(items)}   # 1-indexed
        gold_idx = [sorted((pos[p["src_cid"]], pos[p["pair_id"]]))
                    for p in gold]

        # Generation-time checks only ever compared a contradiction against its
        # source's siblings. They cannot see the example that actually got
        # built, so an unlabelled second contradicting pair can still slip in
        # (mechanism-level collisions are the usual culprit). Verify here,
        # against the items really present, and drop the whole example if the
        # "exactly K pairs" property does not hold.
        if verifier is not None:
            extra = verify_example(items, gold, verifier, args)
            if extra:
                n_rejected_examples += 1
                print(f"  ! example rejected: {len(extra)} unlabelled "
                      f"contradiction(s), e.g. {extra[0]}")
                continue

        out_rows.append({
            "documents": [{"text": t} for t, _, _, _ in items],
            "queries": [],
            "gold_doc_indices": gold_idx,
            "num_docs": len(items),
            "num_contradictions": len(gold),
            "source": "pubmed_multiclaim",
            # provenance is debug metadata, not consumed by the trainer
            "_provenance": [p for _, p, _, _ in items],
            "_keys": [k for _, _, k, _ in items],
        })

    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    prov = collections.Counter(p for r in out_rows for p in r["_provenance"])
    print(f"wrote {len(out_rows)} examples -> {args.out}")
    print(f"  item provenance: {dict(prov)}")
    if args.verify:
        print(f"  verification: {n_rejected_examples} example(s) rejected for "
              f"unlabelled contradictions")


# ─────────────────────────────────── cli ───────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser(
        "split-abstracts",
        help="partition abstracts into disjoint train/eval pools")
    sp.add_argument("--abstracts", required=True)
    sp.add_argument("--eval-abstracts", type=int, default=1500)
    sp.add_argument("--seed", type=int, default=1234)
    sp.add_argument("--out-train", required=True)
    sp.add_argument("--out-eval", required=True)
    sp.set_defaults(func=split_abstracts)

    d = sub.add_parser("dump-abstracts", help="PubMedQA -> abstracts JSONL")
    d.add_argument("--num", type=int, default=20)
    d.add_argument("--source", default="pubmedqa", choices=["pubmedqa", "arxiv"],
                   help="pubmedqa = clinical; arxiv = physics/math/CS breadth")
    d.add_argument("--scan-limit", type=int, default=400000,
                   help="arxiv only: rows to stream before giving up")
    d.add_argument("--min-words", type=int, default=150,
                   help="skip abstracts too short to support several claims")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--out", required=True)
    d.set_defaults(func=dump_abstracts)

    c = sub.add_parser("claims", help="abstracts -> orthogonal claims (LLM)")
    c.add_argument("--abstracts", required=True)
    c.add_argument("--limit", type=int, default=0, help="0 = all")
    c.add_argument("--claims-per-abstract", type=int, default=4)
    c.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    c.add_argument("--base-url", default="http://horton:8100/v1")
    c.add_argument("--temperature", type=float, default=0.7)
    c.add_argument("--max-tokens", type=int, default=1200)
    c.add_argument("--max-concurrent", type=int, default=16)
    c.add_argument("--no-cache", action="store_true")
    c.add_argument("--no-think", action="store_true",
                   help="disable Qwen3-style reasoning blocks, which otherwise "
                        "consume the token budget before any JSON is emitted")
    c.add_argument("--cache-tag", default="default",
                   help="separate response caches per model")
    c.add_argument("--out", required=True)
    c.set_defaults(func=gen_claims)

    x = sub.add_parser("contradictions",
                       help="claims -> contradicting claims (LLM) + judging")
    x.add_argument("--claims", required=True)
    x.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    x.add_argument("--base-url", default="http://horton:8100/v1")
    x.add_argument("--temperature", type=float, default=0.8)
    x.add_argument("--max-tokens", type=int, default=700)
    x.add_argument("--max-concurrent", type=int, default=16)
    x.add_argument("--seed", type=int, default=42)
    x.add_argument("--no-cache", action="store_true")
    x.add_argument("--no-sibling-guard", action="store_true",
                   help="do not show sibling claims as must-not-conflict context")
    x.add_argument("--judge", action="store_true", default=True)
    x.add_argument("--no-judge", dest="judge", action="store_false")
    x.add_argument("--retry-rejected", action="store_true", default=True,
                   help="re-generate pairs that fail validity/uniqueness/overlap")
    x.add_argument("--no-retry", dest="retry_rejected", action="store_false")
    x.add_argument("--max-attempts", type=int, default=3)
    x.add_argument("--max-overlap", type=float, default=0.45,
                   help="reject a pair whose gold overlap exceeds this")
    x.add_argument("--no-think", action="store_true",
                   help="disable Qwen3-style reasoning blocks")
    x.add_argument("--cache-tag", default="default",
                   help="separate response caches per model")
    x.add_argument("--judge-siblings", action="store_true", default=False,
                   help="also judge sibling conflicts inside the retry loop "
                        "(off: the conflict matrix covers this once, at the end)")
    x.add_argument("--conflict-matrix", action="store_true", default=True,
                   help="emit the per-abstract contradiction matrix that "
                        "assembly uses instead of per-example verification")
    x.add_argument("--no-conflict-matrix", dest="conflict_matrix",
                   action="store_false")
    x.add_argument("--out", required=True)
    x.set_defaults(func=gen_contradictions)

    m = sub.add_parser("assemble", help="claims + pairs -> corpus examples")
    m.add_argument("--claims", required=True)
    m.add_argument("--contra", required=True)
    m.add_argument("--num-docs", type=int, default=20)
    m.add_argument("--num-contradictions", type=int, default=2)
    m.add_argument("--num-examples", type=int, default=5)
    m.add_argument("--sibling-frac", type=float, default=0.35,
                   help="share of distractors that are same-abstract siblings")
    m.add_argument("--same-domain-frac", type=float, default=0.85,
                   help="share of filler drawn from the gold pairs' own domain; "
                        "keeps a mixed-domain pool from becoming a topical "
                        "shortcut")
    m.add_argument("--orphan-frac", type=float, default=0.25,
                   help="share of distractors that are orphan contradictions")
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--verify", action="store_true", default=True,
                   help="judge each gold item against the most similar items "
                        "actually placed in the example; drop the example if "
                        "any unlabelled contradiction is found")
    m.add_argument("--no-verify", dest="verify", action="store_false")
    m.add_argument("--verify-k", type=int, default=8,
                   help="how many nearest items to judge per gold item")
    m.add_argument("--model", default="Qwen/Qwen3-8B")
    m.add_argument("--base-url", default="http://localhost:8101/v1")
    m.add_argument("--max-concurrent", type=int, default=16)
    m.add_argument("--no-think", action="store_true")
    m.add_argument("--cache-tag", default="default")
    m.add_argument("--out", required=True)
    m.set_defaults(func=assemble)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
