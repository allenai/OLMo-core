# INCORPORATE: knowledge layer (agent C)

Repo-side edits to apply **after** the flopdrop_clean refactor lands and `ops/` is merged. Paths are
repo-relative to the merged repo unless absolute. Line numbers refer to the main checkout as of
2026-10-12 (`prasann/landmark` @ afec8c40f); re-locate by the quoted text, not the number.

## 0. Path repoints (do these first)

| what | today | after incorporation |
|---|---|---|
| `~/bin/ops` symlink | `.claude/worktrees/ops_tools/ops/bin/ops` | `<repo>/ops/bin/ops` (`ln -sfn`) |
| memory index pointer | `MEMORY.md` → `/accounts/.../OLMo-core/.claude/worktrees/ops_tools/ops/experiments/INDEX.md` (absolute) | `/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/ops/experiments/INDEX.md` |
| memory file `ops-workflow.md` "Location note" paragraph | worktree path | delete the paragraph |
| `superseded_by:` notes inside memory descriptions | `ops/experiments/<name>/STATUS.md` | unchanged (repo-relative already) |
| `ops/opslib/paths.py` `REPO_ROOT` default | main checkout path | unchanged unless the checkout moves |

Find every remaining worktree reference with
`grep -rn 'worktrees/ops_tools' ~/.claude/projects/-accounts-projects-berkeleynlp-prasann-projects-OLMo-core/memory ~/.claude/skills <repo>/.claude <repo>/CLAUDE.md`.

## 1. Skills

1. `git mv ops/skills/{job-status,launch-run,record-results,close-out-experiment} .claude/skills/`
   (or keep them in `ops/skills/` and symlink; pick one home — today repo skills live in
   `.claude/skills/` and `~/.claude/skills/<name>` are symlinks into the repo).
2. Global symlinks so they trigger in every session:
   `for s in job-status launch-run record-results close-out-experiment; do ln -sfn <repo>/.claude/skills/$s ~/.claude/skills/$s; done`
3. Repoint the existing `~/.claude/skills/{pull-evals,run-evals,storage-audit,where-to-run}` symlinks
   if the repo path changes (they point at the main checkout's `.claude/skills/`).
4. Remove stale duplicates: the skill listing still shows `.pull-evals.pre-repo-backup` and
   `.run-evals.pre-repo-backup` (not under `~/.claude/skills` or `~/.claude-work/skills` on
   2026-10-12 — check the plugin/synced skill dirs and `~/.claude/skills/synced/*/manifest.json`) —
   delete them so they stop shadowing the live skills.
5. `.claude/skills/pull-evals/SKILL.md`:
   - frontmatter `description`: drop "ingest into results-hub", "add these numbers to the hub",
     "ingests them into ../results-hub"; add "then records them with the record-results skill
     (`ops result add --check`)".
   - Replace "## Step 4: ingest into results-hub" (line ~104–150: `cd ../results-hub`,
     `python ingest_eval_json.py ...`, the method-slug rules) with:
     "## Step 4: record — follow the `record-results` skill (`ops result add --json <file> --check`;
     results-hub was retired 2026-10-12)".
   - Line ~188–216 ledger update: replace "rungs_ingested"/"Pushed to results-hub" with
     `ops update <run_id> final_state=done` and "Recorded via ops result add".
   - Checklist line ~238: "No result JSON recorded without a matching registered run (`ops runs`)".
6. `.claude/skills/run-evals/SKILL.md`:
   - description: replace "produces the results-hub column values for every job it submits" with
     "registers every eval job with `ops register --kind eval --parent-run <train run>`".
   - Rule 4 (line ~21) "Always write a launch ledger entry recording every results-hub column value"
     → "Always register the launch (`ops register ... --expected-minutes N --log-path ...`); the
     YAML ledger in `records/eval_launches/` is optional history".
   - Section at line ~184–215 ("The shared view of any eval is its row in `../results-hub`" + the
     results-hub column table) → delete; replace with a pointer to `record-results`.
   - Keep the 2026-10-12 olmo-eval `launch_ctc_suite.py` content (afec8c40f) as is.
7. `.claude/skills/training-smoke-test/SKILL.md`:
   - Line ~122–124 "Before launching, **ask the user what priority** ... `low`, `normal`, `high`,
     `urgent`" → "Always launch at `--launch.priority=urgent` (exactly urgent: never lower, never
     `immediate`), workspace `ai2/flex2`, budget `ai2/oe-other` (unallocated)." Replace
     `--launch.priority=<priority>` with `--launch.priority=urgent` in the command block.
   - Add a section "Local slurm / Lambda smoke tests": run ≥2 optimizer steps (a `--max-tokens`
     below one step's tokens becomes a 1-step run); interpreter
     `/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python` via `$PY -m torch.distributed.run`
     (never bare `torchrun`); slurm `--output`, HOME, TMPDIR, triton/flashinfer caches on node-local
     `/data`; stage code with `git archive <commit> | srun ... tar -x -C /data/prasann/<fresh>` from
     the login node; Lambda uses `/tmp` (no usable /data) and FAILED ≠ failed; register with
     `ops register --kind train --notes smoke`. Link `launch-run` and `where-to-run`.
   - description: add "local slurm (horton/jsteinhardt) or Lambda smoke run", not only Beaker.

## 2. CLAUDE.md

1. **Add** after "## Records (standalone writeups)" (line ~168–172):

   ```markdown
   ## Research ops (runs, results, experiment state)

   `/accounts/projects/berkeleynlp/prasann/bin/ops` (stdlib, login-node safe; `ops/README.md`):
   `ops register` every launch · `ops status` / `ops stall ID` for jobs (no sleep/poll loops) ·
   `ops result add --check` for every number (results-hub is retired) · `ops exp show NAME`.
   The canonical current state of each experiment is `ops/experiments/<name>/STATUS.md`
   (index: `ops/experiments/INDEX.md`) — read it before touching an experiment, update it
   (skill `close-out-experiment`) before ending a session. Skills: `launch-run`, `job-status`,
   `record-results`, `close-out-experiment`.

   **CTC evals:** the current path is olmo-eval `scripts/ctc_suite/launch_ctc_suite.py`
   (r256k = YaRN x2 + chunked prefill by default) — see `records/ctc-fast-suite-eval.md` and the
   `run-evals` skill. OLMo-core's `launch_ctc_evals.py` is superseded.
   ```
2. **Trim the upstream-generic content** (~9.7 KB of 20.9 KB: "## Overview" … "### Training
   Scripts" minus our paragraphs, plus "## Docker and Beaker Launch" build mechanics, line ~273–314)
   into `docs/claude/upstream-olmo-core.md`, leaving in CLAUDE.md a 3-line pointer
   ("Upstream OLMo-core architecture, code style, docstrings, Docker/Beaker image mechanics:
   `docs/claude/upstream-olmo-core.md`"). Keep in CLAUDE.md: the docchunk/landmark marker-repair
   section, the memexpress/hub paragraph, Deprecated, Records, Research ops, Local cluster, vLLM
   serving copy, Lambda, the Beaker *priority/allocation* rules, Testing (3 lines), Reporting.
   (The refactor's own CLAUDE.md in `flopdrop_port` already has a "## Repo layout" section — merge,
   don't duplicate.)
3. **Fix the ambiguous env line** (line ~215, end of the "INTERPRETER itself must be node-local"
   paragraph): replace
   `(that env now lives at /data/prasann/conda/envs/corpus-reasoning-eval on horton; /scratch copy removed 2026-09-23)`
   with
   `The old NFS env path above (`/scratch/users/prasann/conda/envs/corpus-reasoning-eval`) no longer
   exists (removed 2026-09-23); its node-local copy is `/data/prasann/conda/envs/corpus-reasoning-eval`
   on horton, but for vLLM use `/data/prasann/ctc_vllm_venv/bin/python` as stated above. For
   training/tests use `/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python`.`
   and change the example path in line ~208 to say "an NFS interpreter (e.g. the former
   `/scratch/.../corpus-reasoning-eval`)".
4. Line ~331 "Older results wrote this as `n`… anything ingested into results-hub" → "…anything
   recorded with `ops result add`".

## 3. records/

1. **Header template** for every new record (also printed by `ops exp records`):

   ```yaml
   ---
   title: <one line>
   date: YYYY-MM-DD
   updated: YYYY-MM-DD
   experiment: <ops/experiments key, or blank>
   kind: diagnosis | plan | handoff | howto | result | task-brief | audit
   status: current | superseded | load-bearing
   superseded_by: <records/x.md or ops/experiments/x/STATUS.md, or blank>
   summary: <one or two sentences — this becomes the README row>
   ---
   ```
   Add to CLAUDE.md "## Records": "Every new record starts with the header in `records/README.md`;
   regenerate the index with `ops exp records --write`."
2. **Regenerate `records/README.md`**: `ops exp records --write` (implemented in
   `ops/opslib/experiments.py`; reuses each doc's existing README description when it has no
   `summary:`, else its first paragraph; date from header or last git commit). On 2026-10-12 the
   hand-written README covered 27 of 58 docs (31 missing). After generating, hand-review the rows
   for the 31 previously-missing docs and backfill `summary:` headers for the load-bearing ones
   (`contradiction-data-and-base-hygiene.md`, `contradiction-train-eval-non-iid.md`,
   `document-chunked-marker-embeddings.md`, `ds64-handoff.md`, `ctc-fast-suite-eval.md`).
   The subdirs `records/ctc_suite_status/` and `records/eval_launches/` are not indexed; add a
   one-line mention by hand if wanted.
3. **FEVER filler leak status — FIXED (verdict from `ops/experiments/ctc_data_fixes/STATUS.md`).** The CTC-suite
   contradiction ladder was rebuilt as `contradiction_clean` on 2026-08-04/05, after the 2026-07-29 records were
   written. Edit `records/contradiction-data-and-base-hygiene.md` §4b: change ❌ OPEN → ✅ FIXED 2026-08-04/05
   (contradiction_clean; xlong fixed earlier). Edit `records/README.md`: drop "still OPEN for the CTC suite
   ladder". Memory `contra-fever-filler-leak` is already correct.
4. Records that are now *state* rather than reference (handoffs) get `status: superseded` +
   `superseded_by: ops/experiments/<name>/STATUS.md` in their header once the STATUS page carries
   their current-state content: `ds64-handoff.md`, `pooled-doc-kv-handoff.md`,
   `soft-kv-slot-probe-handoff.md`, `ctc-suite-olmo-eval-handoff.md` (keep the files; they are history).

## 4. Experiment drivers: debug/ → research/ (refactor layout)

The refactor (`flopdrop_port`, branch `prasann/flopdrop_clean`) moves experiment drivers out of
`debug/` into `research/<family>/` and the canonical pipeline into `recipes/`. After it lands,
update every STATUS.md `launcher:` / `results:` path accordingly (`ops exp check` + grep for
`debug/`):

| debug/ dir (today) | refactor location |
|---|---|
| `debug/ffndrop_cpt`, `debug/flop_scaling`, ffnmoe bench/train | `research/ffn_compute/{ffndrop,flop_scaling,ffnmoe_bench,ffnmoe_train}` |
| `debug/flexcompute_train`, `debug/flexcompute_40x` | `research/kv_route/{flexcompute_train,flexcompute_40x}` |
| `debug/devloss_grid`, `debug/ds64*`, `debug/learned_router`, `debug/pooled_kv`, softdetach | `research/token_routing/...` |
| `debug/ctc_vllm_validation`, `debug/chunked_eval_speedup`, sparse_landmark_* | `research/landmark/...` |
| `debug/ctc_sft_setA`, `debug/ctc_eval_speed` | `recipes/eval/{setA_audit,ctc_eval_speed}` |
| LAUNCH_LEDGER.tsv files | stay with their driver dir; `ops backfill ledgers` reads them (agent A) — update its ledger glob to include `research/**/LAUNCH_LEDGER.tsv` |

Drivers not yet placed by the refactor (`debug/ctc_modelscale`, `debug/outlier_lengthmix_scaling`,
`debug/ctc_olmo_hybrid`, `debug/hybridish_sft`, `debug/ctc_crossfamily`, `debug/ctc_contra_iid_eval`,
`debug/outlier_iid_rebuild`, `debug/gdn_int32_boundary`) need a home decision from Prasann; until
then their STATUS.md pages keep the `debug/` paths.

## 5. Memory (outside the repo; already applied 2026-10-12 except the repoint in §0)

- Backup: `memory/archive/MEMORY.md.2026-10-12.bak`.
- MEMORY.md rewritten to ≤ 8 KB; experiment-state entries replaced by one INDEX.md pointer + one line
  per active experiment. Superseded memory files kept in place with
  `superseded_by: ops/experiments/<name>/STATUS.md` in their description.
- After incorporation: repoint the INDEX.md pointer (§0); consider moving still-useful long trap
  files into `records/traps/` so repo agents without this memory can see them.
