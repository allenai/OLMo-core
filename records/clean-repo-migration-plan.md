# Clean-repo migration plan

**Goal:** stand up a fresh OLMo-core fork, port our work into it deliberately (validating structure
and code as we go), and land in a layout that a collaborator can run without a tour guide.

**Source of truth for this plan:** measured against merge-base `59a339f53` (2026-05-28), our HEAD on
`prasann/landmark`, and upstream `origin/main` `064b172e5` (2026-07-29).

## Decisions (locked 2026-08-08)

| Decision | Choice |
|---|---|
| **Base** | Fresh clone at upstream `main` `064b172e5`. **Already created** at `projects/newolmocore/OLMo-core` — clean tree, on `main`, exactly at `origin/main`. Accept the hand-merge cost on the 6 files upstream also touched. |
| **Where it lives** | A long-lived **branch on `allenai/OLMo-core`** (`prasann/ctc`), not a separate repo. We already have push access — `prasann/landmark` is pushed and 0/0 with its remote. `main` keeps tracking `origin/main` so upstream merges stay real merges. |
| **olmo-core changes** | Go into `src/olmo_core/` **directly**, filed in the module each piece belongs to. No sibling `olmo_core_ext/` — quarantining only makes sense for a separate repo. |
| **Scope** | Core + curated launchers. Port the `olmo_core` patches, all attention/data modules + tests, and all of `corpus_reasoning`; port only launchers we'd re-run today, reborn as YAML configs. Old repo becomes a greppable archive. |
| **Package name** | `corpus_reasoning` → **`ctc`**. Do the rename during Phase 5, where the import lines are being rewritten anyway. |

---

## 1. What we're actually moving (measured)

| Layer | Content | Size | Upstream conflict? |
|---|---|---|---|
| **A. Patched upstream files** | 27 files under `src/olmo_core`, +4 upstream test files | **+4,097 / −184 lines** | **Yes — the hard part** |
| **B. Additive olmo_core modules** | 26 new files (17 attention, 3 composable sources, `document_chunk_landmark.py`, `corpus_reasoning_prompts/`) | ~10k LOC | No — copy + wire |
| **C. Our tests** | 31 new test files (20 under `nn/attention`, 7 under `data/`) | — | No |
| **D. Application code** | `src/corpus_reasoning` (203 files) | 49.5k LOC | No — one-way dep |
| **E. Launchers/scripts** | `src/scripts/{train,data,ctc_eval,eval}` (252 new files, 246 under `train/`) | 57k LOC | No |
| **F. Knowledge + artifacts** | `CLAUDE.md`, `records/` (14), `*_cluster.md`, `beaker.md`, `debug/` (207 tracked), `results/` (104), `deprecated/` (79) | ~38 MB tracked | No |

**Dependency direction is clean and one-way:** `corpus_reasoning` → `olmo_core`. `olmo_core` imports
`corpus_reasoning` exactly zero times (one lazy import of the in-library
`olmo_core.data.corpus_reasoning_prompts`, which we should invert — see §4). This is the single fact
that makes the port tractable: D/E can move wholesale once A/B are right.

### Where the fork actually bites

`git diff` concentrates in a handful of files:

```
src/olmo_core/nn/attention/__init__.py                    +1257   ← the problem
src/olmo_core/nn/transformer/config.py                     +547
src/olmo_core/generate/.../generation_module.py            +488
src/olmo_core/train/callbacks/evaluator_callback.py        +285
src/olmo_core/nn/hf/convert.py                             +245
src/olmo_core/nn/attention/recurrent.py                    +161
src/olmo_core/nn/rope.py                                   +156
src/olmo_core/nn/transformer/model.py                      +154
...19 more files, ≤100 lines each
```

**Upstream has moved into 6 of these.** In the 9 commits since our base, upstream touched
`nn/rope.py`, `nn/transformer/config.py`, `nn/transformer/model.py`, `nn/hf/convert.py`,
`nn/hf/__init__.py`, `train/callbacks/checkpointer.py` (plus new `nn/vision/` and
`output_discard_checkpoint.py`). Those six files are the entire merge-conflict surface. Everything
else re-applies cleanly.

---

## 2. Target layout

**Lives in `clean-repo-target-structure.md` — that doc is the single source of truth for the tree.**
Do not restate it here; two copies of a layout is the exact drift this migration exists to end.

The shape, in one paragraph: a long-lived branch `prasann/ctc` on `allenai/OLMo-core`. Our olmo-core
changes (17 attention variants, 3 instance sources, callbacks) go into `src/olmo_core/` directly,
filed in the module each belongs to. Three new top-level entries — `ctc/` (a standalone pip package
holding format + data generation + eval, importing no olmo_core), `run/` (four shell entry points),
and `configs/` (YAML, the unit of work). The launch layer lands in `src/scripts/ctc/`, following
upstream's own convention.

Three structural changes worth the effort, in order of payoff:

1. **Configs replace scripts.** `configs/train/ctc_suite/qwen35-4b-chunked-32k.yaml` instead of
   `src/scripts/train/memexpress/ctc_suite/<one_py_file_per_run>.py`. Collapses 246 files to one
   launcher plus a config dir a collaborator can diff. This is the current pain — not the library
   patches.
2. **One eval driver.** Near-duplicate `vllm_chunked_patch.py` copies live under
   `src/corpus_reasoning/lib/` and `src/scripts/ctc_eval/lib/`, already *diverged by 100 lines*.
   Exactly one copy on the branch.
3. **The attention registry becomes additive** (§6), so 10 new `AttentionType` members stop being
   edits to upstream control flow. On a branch that merges `main` repeatedly, this is what keeps each
   merge cheap.

The `corpus_reasoning` → `ctc` rename is locked; it happens in Phase 5, in the same pass that rewrites
those imports.

---

## 2b. Working protocol

**Nothing is copied silently.** For every file, before it lands in the new repo:

1. **What it is** — one or two sentences on its purpose and where it sits in the pipeline.
2. **Why it exists** — the experiment or bug that produced it, where that's recoverable from history.
3. **Who depends on it** — inbound refs, and whether it has a test.
4. **Verdict** — port as-is / port with changes (say which) / drop / archive.

Prasann reviews the verdict before the file moves. This is the point of the migration: the new repo
should contain nothing whose purpose we can't state out loud. Batch size is a handful of files at a
time, grouped by subsystem, so each batch is reviewable in one sitting.

The ledger in §5 is the running record of these verdicts.

---

## 3. Phase 0 — freeze the source (do this first)

The working tree has **24 modified + 213 untracked paths**, including live edits to
`nn/transformer/config.py`, `nn/attention/recurrent.py`, both `vllm_chunked_patch.py` copies, and six
data generators. Porting from a moving tree guarantees silent loss.

- [ ] Triage the 24 modified files: commit what's real, revert what's scratch.
- [ ] Triage the 213 untracked paths — mostly `debug/*` (fine to leave), but check for un-added source.
- [ ] Tag the result: `git tag pre-migration-source` and record the SHA. **Every later phase ports
      from that tag**, so "did I move this?" is always answerable by diff.

---

## 4. Phase 1 — stand up the new repo  *(mostly done)*

- [x] Clone upstream at `origin/main`. **Done** — `projects/newolmocore/OLMo-core` sits at
      `064b172e5`, clean tree, branch `main`.
- [ ] Branch off `main` as `prasann/ctc`; keep `main` tracking `origin/main` so future upstream
      merges are real merges. Push the branch early — it's a normal branch on allenai/OLMo-core.
- [ ] No fork needed. **Correction to an earlier draft of this plan:** it claimed our 392 commits
      existed only on this disk. They don't — `prasann/landmark` is pushed to `origin` and is 0/0
      with its remote, so we already have push access to allenai/OLMo-core.
- [ ] Optional: flatten the clone path from `newolmocore/OLMo-core` to something less nested before
      work lands (the editable install, the `training/` symlink, and the Claude memory dir are all
      path-keyed, so it's cheap now and annoying later).
- [ ] **Establish baseline green before our code lands:** `make checks` and `pytest -v src/` on the
      untouched clone. Record what passes/skips. Without this, the first failure after porting is
      ambiguous.
- [ ] Copy `.claude/` project config. Note: the Claude memory dir is keyed to the repo path
      (`...projects-OLMo-core/memory/`) — a new path means a new, empty memory dir. Copy it across.

---

## 5. Phase 2 — the keep/drop ledger (before porting a line)

A table with one row per unit and columns: *unit / LOC / has test? / inbound refs / keep–drop–archive
/ notes*. This is the artifact that makes "manually validate" tractable, and it's the deliverable a
collaborator reads. Measured starting data for the attention modules:

| Module | LOC | Inbound refs | Test |
|---|---|---|---|
| `landmark.py` | 326 | 28 | ✓ |
| `chunked_mask.py` | 835 | 27 | ✓ (document_chunked_test) |
| `landmark_kernel.py` | 715 | 12 | ✓ |
| `landmark_fast.py` | 1219 | 9 | ✓ |
| `gold_grad_mask.py` | 452 | 9 | ✓ |
| `landmark_compressive.py` | 999 | 8 | ✓ (×3) |
| `document_chunked.py` | 502 | 8 | ✓ |
| `gold_hop_mask.py` | 1092 | 6 | ✓ |
| `landmark_document.py` | 378 | 5 | ✓ |
| `landmark_sparse_kernel.py` | 460 | 3 | ✓ |
| `landmark_sparse.py` | 561 | 3 | — |
| `landmark_prefill_topk.py` | 567 | 2 | ✓ (topk_decode) |
| `landmark_multi.py` | 427 | 2 | ✓ |
| `dilated_window.py` | 450 | 2 | ✓ |
| `landmark_document_compressive.py` | 168 | 1 | ✓ |
| `landmark_sparse_decode.py` | 336 | **0** | — |
| `landmark_prefill_sparse.py` | 487 | **0** | — |

Test coverage here is genuinely strong (20 of our own attention test files) — that's the asset the
whole migration leans on. The two zero-ref modules are the most recent prefill-sparse commits, i.e.
in-flight research, not dead code: keep, mark experimental.

Pre-filled calls elsewhere:
- `deprecated/` (79 files) — **drop.** Its purpose is recording retirement; a clean history does that
  better. Keep the README rows as one `records/` note.
- `debug/` (207 tracked, 937 MB on disk) — **archive, don't port.** Leave the old repo in place as
  `OLMo-core-archive` and add a `records/` pointer.
- `src/scripts/data/ctc_suite/eval_rungs/*/rung_{2048,4096}.jsonl` — 6 files, 2.5–5.5 MB each, our
  largest tracked blobs. **Move to `/data` + a manifest**; don't re-commit.
- Upstream-owned script dirs (`OLMo2/`, `OLMo3/`, `ladder/`, `jacobm_olmoe_ladder/`, `sft/`) — come
  free with the clone; don't port our copies over them.

---

## 6. Phase 3 — port Layer A (upstream patches). Hardest, so first.

For each of the 27 files: `git diff pre-migration-source^{tree} -- <file>`, read it, and re-apply by
hand onto the *new* upstream version. Not `git apply` — reading each hunk is the point of the exercise.

The explicit sub-goal is **shrinking the patch surface**, so future upstream merges stay cheap:

- [ ] **Move `LandmarkAttention` out of `attention/__init__.py`** (currently lines 1530–1953) into
      `nn/attention/landmark.py`. ~420 lines off a 2,318-line file that upstream also edits — the
      single worst merge surface on the branch. It stays inside `olmo_core`; this is about the file,
      not about quarantining our code.
- [ ] **Make the attention registry additive.** `AttentionType` is a closed `StrEnum` we grew from 3
      to 13 members, dispatched at **36 sites in `attention/__init__.py` and 19 in
      `transformer/config.py`**. Replace enum-dispatch with a registration table
      (`register_attention("landmark", builder)`) so each variant registers from its own module
      instead of editing upstream control flow. **On a long-lived branch that merges `main`
      repeatedly, this is the difference between a cheap merge and a recurring hand-resolve** — do it
      before porting any variant.
- [ ] **Invert `corpus_reasoning_prompts`.** Application prompts currently sit inside the library at
      `olmo_core/data/corpus_reasoning_prompts/`, imported lazily at
      `document_chunk_landmark.py:485`. Move to the `ctc` package and pass the builder in — this one
      genuinely doesn't belong in olmo_core, because it's task content, not a model abstraction.
- [ ] Re-apply the remaining hooks: `generation_module.py` (+488, landmark decode), `rope.py` (+156,
      YaRN/CP), `evaluator_callback.py` (+285), `hf/convert.py` (+245, Qwen3.5), `recurrent.py`
      (+161, GDN), `trainer.py`/`train/utils.py` (+60).
- [ ] Pay special attention to the **6 files upstream also changed** — read upstream's new version
      first, then port. `rope.py` and `transformer/config.py` are the risky pair.

**Gate:** `make checks` clean; `pytest -v src/test` matches the Phase-1 baseline plus our new tests.

Realistic target: from ~4,100 patched lines down to a few hundred lines of documented hooks, with
everything else additive.

---

## 7. Phase 4 — port Layer B/C (modules + tests), one commit per module

Order by inbound refs, so the foundation lands first: `landmark` + `landmark_kernel` →
`chunked_mask` → `document_chunked` → `landmark_fast` → `landmark_compressive` → `gold_grad` /
`gold_hop` → `dilated_window` → `landmark_document*` → `multi` → `sparse` / `prefill_*`.

Each commit carries its test file. Then the data layer: `landmark_instance_source`,
`landmark_packing_instance_source`, `pad_to_length_instance_source`, `document_chunk_landmark.py`
plus its 5 tests.

**The real validation gate is numeric parity, not import success.** A hand-re-applied patch can
change behavior silently and only surface as a bad training run days later. Write one harness that,
for each variant, builds the same config with the same seed in both repos, runs the same input, and
asserts outputs match to <1e-6. Run it per module as it lands. This is the highest-value thing in the
whole plan and it costs maybe half a day.

---

## 8. Phase 5 — port Layer D/E (application code + launchers)

- **`corpus_reasoning` (49.5k LOC):** moves wholesale — the dependency direction guarantees it. Port
  by subpackage: `lib` → `data` → `eval` → `train` → `analysis` → `viz`. Prune per the ledger. The 8
  files in `corpus_reasoning/tests` are the gate. **The `ctc` rename happens in this pass** — one
  mechanical `corpus_reasoning` → `ctc` rewrite per subpackage as it lands, so no file gets touched
  twice.
- **`scripts` (57k LOC):** the sprawl, and the lowest value per line. Port only launchers we'd re-run
  today (`ctc_suite`, `sft_docchunk`, `evals`, the data converters) — and port them **as configs
  under `experiments/configs/`**, not as scripts. The rest stays greppable in the archive repo.
- **De-duplicate on the way in.** Known duplicate pairs: `vllm_chunked_patch.py`
  (`corpus_reasoning/lib` vs `scripts/ctc_eval/lib`, currently diverged), and the several eval drivers
  under `scripts/eval` vs `scripts/ctc_eval` vs `corpus_reasoning/eval`.

---

## 9. Phase 6 — carry the knowledge

`CLAUDE.md`, `records/` (14 files), `local_cluster.md`, `lambda_cluster.md`, `beaker.md` encode traps
that each cost days to find (marker embeddings and their norm, NFS/`/data` rules, the
`parse_doc_ids` bracket bug, GDN JIT `/tmp` fill, the eval-maxlen truncation trap). Port verbatim into
`docs/`, then prune rows that no longer apply. Write `QUICKSTART.md` last, from the experience of
actually running the new repo end to end.

`results/` (104 files): results-hub is already the source of truth — port only what isn't there.

---

## 10. Phase 7 — cutover

- [ ] Run one real training smoke test **and** one eval on the new repo, and compare against a known
      number from the old one (e.g. the 2k contradiction sanity at f1 ≈ .865/.843, or the clean n=100
      set: chunked .441 / hier-K50 .831 / full .934). A match here is what licenses the switch.
- [ ] Only then: rename old repo → `OLMo-core-archive`, repoint the `training/` symlink, redo the
      editable install, move the Claude memory dir.
- [ ] Push to our own fork so 392 commits of work stop living on one disk.

---

## Sequencing note

Phases 3 and 4 are the ones that need care and can't be parallelized much (everything depends on the
registry refactor). Phase 5 is bulk work that can be split with a collaborator once the ledger exists.
Phase 0 and Phase 2 are cheap and unblock everything — do both before writing any new code.
