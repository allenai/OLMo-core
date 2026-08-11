# Duplication audit: `ctc.format` (A) vs the vendored `corpus_reasoning_prompts` (B)

Repo `/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core`, branch
`prasann/ctc`, HEAD `3d8a1baaf`. Analysis only — nothing outside this file was changed.

**Headline.** On the canonical surface the two implementations are byte-identical:
**232 / 240** of the full `(task × query_position × use_alpaca × use_titles)` matrix match
exactly, and **18 / 18** answer targets match. All 8 prompt mismatches are the *same one bug*,
in **(A)**, on **`grouping_labeled` at `query_position` `before`/`both`**. Four further
divergences exist off the canonical surface (absence-textdiff, empty-`documents`,
`before_dummy`/`after_dummy`, and four tasks (B) supports that (A) does not register).

(B) is **not** dead code: it is on the live eval path through
`ctc/src/ctc/eval/backends/native.py:181`.

---

## 1. Interfaces

### (B) — the vendored copy

Entry point `build_prompt`, defined at
`src/olmo_core/data/corpus_reasoning_prompts/_data_format.py:1128`, re-exported from
`src/olmo_core/data/corpus_reasoning_prompts/__init__.py:14`:

```python
build_prompt(example, task="retrieval", query_position="after",
             use_titles=True, before_dummy=0, after_dummy=0,
             use_alpaca=True, unified_prompt=False, cot_mode="label",
             output_top_k=-1)  -> (prompt: str, output: str)
```

Notes:

* **Returns a 2-tuple** `(prompt, target)`. Task dispatch is a chain of `if task == ...`
  branches; unknown task names fall through to the QA branch
  (`_data_format.py:145-146`, `:1057-1062`).
* The unified/classic split is the hardcoded set `force_unified` at `_data_format.py:1211`
  (13 names, incl. `ruler` and `matching_ngram`).
* `unified_prompt=True` would force the unified shape onto any task. Never passed by any
  caller in this repo (grep: only the definition and docstring).
* `cot_mode` affects **only the target**, never the prompt — verified by the matrix in §3,
  which was run at `cot_mode="none"` and matched (A)'s no-CoT targets 18/18.
* A second entry point `build_prompt_parts` (`_data_format.py:1264`) returns
  `(instruction, input_text, output)`. **It has no caller in this repo** and its unified set
  (`_data_format.py:1292`, `:1302`) is *stale relative to its own sibling* — it omits
  `redundancy`, `absence`, `strmatch`, `cycle`, `groups4`, `textgroups`, `ruler`, which
  `build_prompt`'s `force_unified` includes. So `build_prompt_parts` and `build_prompt`
  already disagree with each other for 7 tasks. (Not exercised; recorded because it is a
  reason not to keep (B) around.)
* A third entry point `build_helmet_rerank` (`_data_format.py:770`) handles the
  `task="rerank_helmet"` schema (`{qid, query, ctxs}`), reached via `_data_format.py:1158`.

### (A) — the canonical implementation

There is no single `build_prompt`. Each task carries one on its spec:
`TaskSpec.build_prompt` (`ctc/src/ctc/format/registry.py:101`), signature
`(example, **opts) -> str`. Every task's implementation forwards to
`ctc/src/ctc/format/assemble.py:194`:

```python
assemble(example, *, task, unified, header, positioned,
         query_position="after", use_titles=True,
         before_dummy=0, after_dummy=0, use_alpaca=True) -> str
```

Precise differences from (B):

| | (A) | (B) |
|---|---|---|
| return | `str` (prompt only) | `(prompt, target)` |
| target | separate `build_target(example)` in each `tasks/<name>/spec.py`; **not a field on `TaskSpec`** | second tuple element |
| task dispatch | registry lookup, `KeyError` on unknown task (`registry.py:190`) | `if` chain, silent QA fallback |
| unified vs classic | per-task `TaskSpec.unified` (`registry.py:107`) | hardcoded `force_unified` set (`_data_format.py:1211`) |
| `unified_prompt` arg | **absent** (documented as never-enabled, `assemble.py:23-25`) | present, default `False` |
| `cot_mode` arg | **absent** (CoT dropped in the port) | present, default `"label"` |
| `output_top_k` arg | **absent**; `rerank` header is the plain `RERANK_INSTRUCTION` (`tasks/rerank/spec.py`) | present, default `-1`; feeds `rerank_instruction(top_k)` |
| doc serialization | table `_SERIALIZERS` (`ctc/src/ctc/format/documents.py:144`) | `if` chain (`_data_format.py:162`) |

Defaults that agree: `query_position="after"`, `use_titles=True`, `use_alpaca=True`,
`before_dummy=0`, `after_dummy=0`.

### Constants are provably identical

Programmatic comparison of every public `str`/`dict` in
`src/olmo_core/data/corpus_reasoning_prompts/_prompts.py` against
`ctc/src/ctc/format/prompts.py`: **46 shared constants, 0 differing in value, 0 present on
only one side**. `rerank_instruction(k)` agrees for `k ∈ {-1, 0, 5, 10}`. `ALPACA_TEMPLATE`
(`corpus_reasoning_prompts/_io.py:25` vs `ctc/src/ctc/format/assemble.py:58`) is identical.
The only textual difference between the two `prompts.py` files is docstrings plus two
helpers `format_doc`/`format_doc_dict` that exist in (B) and are unused there.

---

## 2. Every caller of (B)

`corpus_reasoning_prompts` has **exactly one import site in the entire repo**:

* `src/olmo_core/data/document_chunk_landmark.py:485` —
  `from olmo_core.data.corpus_reasoning_prompts import build_prompt`, called at
  `document_chunk_landmark.py:504-511` inside `segment_prompt_to_chunks`
  (defined `document_chunk_landmark.py:438`), with
  `use_alpaca=False`, and `task` / `query_position` / `cot_mode` / `use_titles` forwarded
  from the caller. **Both** tuple elements are used: `prompt` is wrapped with document
  markers and `answer` is appended when `include_answer=True`.

`segment_prompt_to_chunks` has exactly one *external* caller:

* `ctc/src/ctc/eval/backends/native.py:177` (import) / `:181` (call), inside
  `NativeBackend.build_prefill`. Passes `cot_mode="none"`, `chunk_by="document"`,
  `include_answer=False`, `query_position=self.query_position`, and leaves
  `use_titles` at the `segment_prompt_to_chunks` default of `False`.

Nothing else — no training script, no data converter, no `src/scripts/**` file — references
either. `build_prompt_parts`, `build_helmet_rerank`, and `unified_prompt` have zero callers.
`before_dummy`/`after_dummy` have zero callers (grep across the repo: only definitions).

**The live consequence.** `ctc/src/ctc/eval/runner.py:257` builds the prompt list with
**(A)** (`spec.build_prompt(ex, query_position=cfg.query_position)`) and records the
fingerprint from (A) at `runner.py:248`. The native backend then **discards those strings**
and re-derives the prefill from **(B)** at `native.py:181`. So on the native arm the
fingerprint, the token-length audit and the actually-tokenized text come from two different
builders. Today that is harmless for 17 of 18 tasks; for `grouping_labeled` at
`query_position != "after"` it is not (§3).

---

## 3. Differential test

Scripts (throwaway, in the session scratchpad, not in the repo):
`.../scratchpad/diff_test.py` and `.../scratchpad/diff_test2.py`.
Both implementations import cleanly under stock `python3` 3.13.2 — (B) was loaded as a
top-level package by putting `src/olmo_core/data` on `sys.path`, which avoids
`olmo_core/__init__.py` and therefore torch. Example shapes come from
`ctc/tests/fixtures/generate_golden.py` (`PROMPT_EXAMPLES`), extended with hand-built
examples for `matching_ngram`, `grouping`, `cot_retrieval`, `ruler` and the absence-textdiff
variant.

(A) registers **18** tasks (`ctc/src/ctc/tasks/__init__.py:TASK_MODULES`):
`absence, contradiction, cycle, grouping_labeled, groups4, mathmatch, oolong, outlier, qa,
qdmatch, redundancy, reorder, rerank, retrieval, strmatch, summarization, textgroups,
xabsence`.

### 3a. Task × query_position (the requested matrix)

18 tasks × 3 positions, defaults otherwise (`use_alpaca=True`, `use_titles=True`,
`cot_mode="none"`):

**52 exact matches, 2 mismatches, 0 errors.**

### 3b. Full option matrix

20 example keys (18 tasks + `retrieval_multigold` + `retrieval_multiquery`) × 3 positions ×
`use_alpaca ∈ {True, False}` × `use_titles ∈ {True, False}` = 240 combinations:

**232 exact matches, 8 mismatches** — the same `grouping_labeled` × `{before, both}` cell in
all four option corners.

### 3c. Native-eval settings exactly

`use_alpaca=False`, `use_titles=False`, `cot_mode="none"` (what `native.py:181` produces),
18 tasks × 3 positions: **52 match, 2 mismatch** — same two.

### 3d. Targets

(A)'s `ctc.tasks.<name>.spec.build_target(example)` vs (B)'s second tuple element at
`cot_mode="none"`, 18 tasks: **18 / 18 identical**.

### 3e. Golden fixture

Both sides were run against all 24 prompt cases in
`ctc/tests/fixtures/golden_format.json` (`prompts` key):
**(A) matches 24/24, (B) matches 24/24.** The fixture does not cover `grouping_labeled` at
`before`/`both` — its only `grouping_labeled` case is `after`
(`generate_golden.py:117`), which is exactly why the bug survived.

### The mismatch, in full

```
grouping_labeled | before | first differing char index 564 | len(A)=696  len(B)=696
grouping_labeled | both   | first differing char index 564 | len(A)=722  len(B)=696
```

`before`, alpaca, titles:

```
A: ...appear in exactly one group.\n\n### Input:\nGroup into 2 categories.\n\n
   Document [1](Title: Attention) A new architecture.\n\nDocument [2](Title: Optics) On lenses.
B: ...appear in exactly one group.\n\n### Input:\nDocument [1](Title: Attention) A new
   architecture.\n\nDocument [2](Title: Optics) On lenses.\n\nGroup into 2 categories.
```

(A) honours `query_position` and moves the raw query string in front of the documents;
(B) always emits `f"{context}\n\n{queries[0]}"`. At `both`, (A) additionally repeats the
query after the documents, which is the 26-character length difference.

### Divergences outside the requested matrix

These were not in the brief but are real and are why "(A) supersedes (B)" is not yet true:

1. **`absence` textdiff (Gutenberg) variant — (A) has no prompt path for it.**
   With `meta.format == "textdiff"`, (B) takes a dedicated branch at `_data_format.py:1188`
   and emits `Version A:\n\n... \n\nVersion B:\n\n... \n\n{ABSENCE_GUTENBERG_INSTRUCTION}`
   (answer = first four words of each removed sentence). (A) emits the ordinary numbered-id
   prompt with `ABSENCE_INSTRUCTION` and a `Missing: [id]` target. Complete prompt divergence,
   not a whitespace nit. (A) *knows about* the variant — `is_textdiff` and `score_textdiff`
   at `ctc/src/ctc/tasks/absence/spec.py:34` and `:79` — but neither `build_query`
   (`:42`) nor `build_target` (`:67`) branches on it, and grep shows **`score_textdiff` and
   `is_textdiff` have no caller anywhere in `ctc/`**. So (A)'s textdiff support is currently
   unreachable in both directions; the variant is simply unsupported end-to-end.

2. **Empty `documents` (closed-book) — (A) emits a stray blank line.** (B) has a dedicated
   no-documents branch at `_data_format.py:1171-1182`. (A) has none: `format_documents([])`
   returns `""` and `_position` (`assemble.py:141-146`) joins it, yielding
   `### Input:\n\n\nQuestion: ...` where (B) yields `### Input:\nQuestion: ...`. Confirmed for
   `retrieval` and `qa`. No CTC ladder data has zero documents as far as I could tell, so this
   is latent — but I did not verify that exhaustively.

3. **`before_dummy` / `after_dummy` — two completely different algorithms.** All 9 tested
   combinations (`{contradiction, retrieval, oolong}` × `{(3,0), (0,3), (2,2)}`) mismatch.
   (A)'s `insert_dummy_tokens` (`assemble.py:87`) splits the whole input on `\n\n` and
   inserts at the very start / very end, `.rstrip()`-ing the filler. (B)'s
   (`corpus_reasoning_prompts/_io.py:55`) regex-locates the first `Document[\s\[\(:]` and the
   trailing `\n\nQuestion:` and inserts *inside* those boundaries, keeping the filler's
   trailing space. Concretely, at `after_dummy=3` on `retrieval`, (A) appends `* * *` after
   the trailing question while (B) inserts `* * * ` before it. **Neither is "right"** — they
   answer different questions — but they are not interchangeable, and nothing calls either
   today, so this is a latent trap rather than a live bug.

4. **Four tasks (B) supports that (A) does not register:** `grouping`, `cot_retrieval`,
   `ruler`, `matching_ngram` — all build successfully through (B), all raise `KeyError` from
   `ctc.format.registry.get`. Plus `rerank_helmet` (a distinct record schema) and every
   `cot_mode` other than `"none"`. Whether these are deliberate drops or gaps is a product
   question I cannot settle from the code; `ctc/src/ctc/tasks/README.md` states CoT was
   dropped deliberately, and `ruler`/`matching_ngram`/`grouping` still have live serializer
   entries in `ctc/src/ctc/format/documents.py:144-166`, which reads like an intent to
   register them later.

---

## 4. Verdict

**The single mismatch on the canonical surface is a bug in (A), and (A) contradicts itself.**
Not a judgement call:

* `ctc/src/ctc/format/registry.py:59-63` states that `honors_query_position=False` marks tasks
  that "take a legacy path that hardcodes documents-then-query and never consults the knob",
  naming `grouping`, `grouping_labeled` and `outlier`.
* `ctc/src/ctc/tasks/_grouping.py:145` duly sets `honors_query_position=False`, with the
  comment "legacy path hardcodes documents-then-query".
* But `make_grouping_spec`'s `build_prompt` (`ctc/src/ctc/tasks/_grouping.py:128-136`) passes
  `**opts` straight through to `assemble`, so the knob *is* consulted.
* The sibling tasks that got this right do `opts.pop("query_position", None)` and pin the
  value: `ctc/src/ctc/tasks/outlier/spec.py:51-58` and
  `ctc/src/ctc/tasks/qdmatch/spec.py:52-59`. `_grouping.py` is missing exactly that.

The consequence is worse than a prompt diff. Because
`TaskSpec.fingerprint` pins `query_position="after"` whenever `honors_query_position` is False
(`registry.py:152-155`), a `grouping_labeled` run at `--query-position before` writes a
fingerprint claiming `after` while emitting a `before` prompt — the fingerprint guard would
report *compatible* on a genuinely incompatible pair. That is precisely the failure class the
fingerprint exists to catch.

**Fix (A), one line, before anything else:**

```python
# ctc/src/ctc/tasks/_grouping.py, in make_grouping_spec.build_prompt
def build_prompt(example: Dict, **opts) -> str:
    opts.pop("query_position", None)          # <-- add
    return assemble.assemble(..., query_position="after", **opts)   # <-- pin
```

and add a regression case pinning `grouping_labeled|before` and `|both` to the same bytes as
`grouping_labeled|after`. With that, (A) and (B) agree on **54/54** and **240/240**.

### Can (B) be deleted?

**Yes on correctness, but not by a straight repoint — the dependency runs the wrong way.**

`ctc/pyproject.toml` declares `dependencies = []` with `ai2-olmo-core` only as the optional
`native` extra, and `ctc/src/ctc/eval/backends/base.py:89-90` states that olmo_core imports
are confined to `ctc.eval.backends.native` / `ctc.eval.masking.native`. Direction today is
**ctc → olmo_core**, one-way; grep confirms no `import ctc` anywhere under `src/olmo_core`.
Making `document_chunk_landmark.py:485` import `ctc.format` would invert that and create a
cycle, breaking olmo_core's standalone install.

There is also a signature gap: (B) returns `(prompt, target)` and
`segment_prompt_to_chunks` needs both, but **`build_target` is not a field on `TaskSpec`** —
it lives only as a module-level function in each `ctc/src/ctc/tasks/<name>/spec.py`. Any
repoint must either add `build_target` to `TaskSpec` or import per-task modules by name.

Recommended sequence:

1. Fix `_grouping.py` as above (independent of everything else; do it regardless).
2. Add `build_target: Callable[[Dict], str]` to `TaskSpec` (`registry.py`) and wire each
   `spec.py`'s existing `build_target` into it. Purely additive; 18/18 already agree with (B).
3. Make `segment_prompt_to_chunks` **prompt-agnostic**: replace the `task`-and-`example`
   parameters used for prompt building with explicit `prompt: str` and `answer: str`
   arguments (keeping `example["documents"]` for `_wrap_documents`, which genuinely needs the
   document bodies). This removes the last reason olmo_core knows anything about task
   prompts, and drops the `cot_mode`, `query_position` and `use_titles` parameters from
   `document_chunk_landmark.py:441-447` entirely.
4. Update the one caller, `ctc/src/ctc/eval/backends/native.py:181`, to build the strings
   from (A) first:
   ```python
   spec = registry.get(task)
   prompt = spec.build_prompt(example, query_position=self.query_position,
                              use_alpaca=False, use_titles=False)
   answer = spec.build_target(example)          # after step 2
   segs, _, _ = segment_prompt_to_chunks(self.tok, example, prompt=prompt, answer=answer,
                                         chunk_by="document", include_answer=False, ...)
   ```
   This also collapses the runner-vs-backend split noted in §2: the fingerprint, the length
   audit and the tokenized prefill would all come from one builder.
5. Only then `git rm -r src/olmo_core/data/corpus_reasoning_prompts/`.

**Do not delete before step 4**, and note two things it forfeits, which someone should sign
off on rather than discover later:

* `rerank_helmet` and all non-`"none"` `cot_mode` targets exist only in (B).
* The absence-textdiff prompt path exists only in (B) (§3, divergence 1). If any Gutenberg
  absence data is still evaluated, deleting (B) removes the only implementation that renders
  it correctly. Fixing (A) — a textdiff branch in `tasks/absence/spec.py`'s `build_query`
  and `build_target`, plus routing `score_textdiff` from the runner — is a prerequisite for
  that source, and is worth doing anyway since (A)'s scorer for it is currently unreachable.

**Interim, if the above is more than you want to do now:** keep both and add a parity test
that runs §3a/§3b as a pytest under `ctc/tests/` and fails on any mismatch. That is cheap and
would have caught the `grouping_labeled` bug. It is strictly worse than deletion as a
long-term state — two builders that agree today is how the five parse-function copies
described in `ctc/src/ctc/format/registry.py:5-17` started.

### Uncertainty, stated

* Whether the four unregistered tasks (`grouping`, `cot_retrieval`, `ruler`,
  `matching_ngram`) are deliberate drops or a pending TODO — I could not settle this from
  code; their serializers are still live in `ctc/src/ctc/format/documents.py`.
* Whether any current or archived data file has `meta.format == "textdiff"` or zero
  `documents`. I did not scan data directories; both divergences are established at the code
  level only.
* `build_prompt_parts`'s stale unified set (§1) was read, not executed — it has no caller, so
  I did not build a differential for it.
