# Target structure for the clean repo

Companion to `clean-repo-migration-plan.md`. That doc says *how to port*; this one says *what to port
into*. Designed top-down from the entry points, per Prasann's ask: one command per pipeline, then work
backwards to the code that makes it true.

---

## 0. Where this lives

**A long-lived branch on `allenai/OLMo-core`**, not a separate repo. `prasann/landmark` already pushes
there, so the branch — call it `prasann/ctc` — is just the next one. `main` tracks `origin/main`, and
the work branch merges from it periodically: a real `git merge`, which is exactly what makes the
registry-hook refactor (migration plan §6) pay for itself.

That framing settles the layout question: **olmo-core changes go into `src/olmo_core/` directly.** Our
attention variants are `SequenceMixer` subclasses, our instance sources are `InstanceSource`
subclasses, our callbacks are `Callback` subclasses. On a branch, the way to change olmo_core is to
change olmo_core. (An earlier draft of this doc proposed quarantining them in a sibling
`olmo_core_ext/`; that solves a problem that only exists if you're building a separate repo.)

## 0b. The organizing principle: one new package, split at tokenization

Exactly one thing is genuinely independent of olmo_core, and it's the part collaborators need most:
**data generation and evaluation**, in one package that never imports olmo_core.

```
   ctc/  (new top-level pip package)          src/olmo_core/  (the branch's changes)
   ├── format/   the shared contract   ◄───   ├── nn/attention/    + 17 variants
   ├── data/     JSONL in, JSONL out          ├── data/composable/ + 3 instance sources
   └── eval/     grade a checkpoint           └── train/callbacks/ + ours
                                              src/scripts/ctc/     config → run, ~2k LOC
   no olmo_core, no torch by default
```

| Component | Root | Depends on | Heavy deps |
|---|---|---|---|
| **`ctc`** (pip package) | `ctc/` | stdlib | all optional extras: `hf`, `vllm`, `sources`, `native` |
| **the branch** | `src/` | `ctc` | torch, flash-attn, TE |

### There is no `ctc_train` package — and that was a mistake in the first draft

An earlier version of this doc proposed a second package, `ctc-train`. Measuring its contents showed it
was a fiction:

| Component | LOC | What it actually is |
|---|---|---|
| Attention variants | 9,974 | `SequenceMixer` subclasses registered into `AttentionType` |
| Composable sources + `document_chunk_landmark` | 1,297 | `InstanceSource` subclasses, olmo_core data API |
| Tokenizers / converters | 3,928 | consume olmo_core's tokenizer + shard API |
| Callbacks | small | olmo_core `Callback` subclasses |

All of it subclasses olmo_core abstractions and is meaningless without them. It is not a package that
*depends on* olmo_core; it **is** olmo_core, extended — so on this branch it goes straight into
`src/olmo_core/`, in the module each piece belongs to.

What genuinely isn't olmo_core is much smaller. `train_ctc_suite.py` is 1,126 lines whose top-level
functions are `build_model_config`, `build_train_module_config`, `derive_batch_geometry`,
`derive_mask_mix_curriculum`, `resolve_plan` — plus 212 lines of argparse. Once the argparse becomes
YAML, what survives is config resolution, cluster submit logic, provenance, and run naming: roughly
**1.5–2k LOC across all four targets**. That's `src/scripts/ctc/`, following upstream's own convention
that launchers live under `src/scripts/train/`.

### Keeping the branch mergeable

Adding files to `src/olmo_core/nn/attention/` costs nothing at merge time — upstream never touches a
file that doesn't exist upstream. The merge cost comes entirely from *editing* upstream files, which is
why the registry hook matters: it turns 10 new `AttentionType` members dispatched at 55 sites into
additive registrations. See migration plan §6; on a long-lived branch that merges `main` repeatedly,
it's the difference between a cheap merge and a recurring hand-resolve.

The "which of these 26 attention files are ours?" question, which a sibling directory would have
answered by `ls`, is instead answered by `git diff origin/main --stat` — the natural question to ask on
a branch anyway.

### Why this seam is the right one

It isn't a design preference — it's where the codebase already divides, measured:

- **0 of 81 files** in `corpus_reasoning/data/` import olmo_core.
- **51 of 56 generators** are pure Python — no torch, transformers, or vllm. The 5 exceptions need
  `transformers` for token counting when fitting rungs, or `torch` for BEIR cross-encoder filtering.
  Both become extras, not baseline dependencies.
- Every olmo_core dependency in the data pipeline sits in the 8 converter/staging scripts under
  `scripts/data/` — the step that writes olmo-core SFT shards.

**Task JSONL is the interface.** Everything upstream of "tokenize into olmo-core shards" is
framework-agnostic and always was; the tokenizer step is the olmo-core adapter. Putting the package
boundary there means `ctc` installs on a vLLM box with no compiler, and a collaborator can generate
data and grade checkpoints without ever building a training environment:

```bash
pip install ./ctc                  # generators + graders, pure python, no compiler
pip install './ctc[vllm]'          # + the vLLM backend
pip install './ctc[hf]'            # + the transformers backend
pip install './ctc[native]'        # + olmo_core, for the native backend
pip install -e .                   # ctc-train: training + shard tokenization
```

### Why data-gen and eval belong together

They share the contract they must agree on — prompt templates, document/chunk serialization, the task
registry, gold-index conventions, answer parsers, metrics. Every drift bug worth remembering came from
two copies of that contract diverging:

- `vllm_chunked_patch.py` exists twice and has already diverged by 100 lines
- the oolong leak was a train/eval **chunk-layout mismatch** — same concept, two implementations
- the grouping-JSON parser was fixed in one copy, not the other (0.44 → 0.82 on chunked @2k)
- the doc-id bracket parser mis-graded a whole family until it was found by dumping generations

Inside one package, `format/` is a single import away from both the generator that writes an example
and the grader that scores the answer. A prompt-format change that breaks eval fails a test in the
same repo, in the same CI run.

### If you later want eval shipped on its own

Keep `ctc/src/ctc/eval/` importing only from `ctc.format` — never from `ctc.data`. That one rule makes
a future split a `git filter-repo` plus a version pin on a `ctc-format` package, not a redesign. A
lint rule enforcing it costs five lines; without it the boundary erodes silently.

### The olmo_core escape hatch

`ctc.eval`'s **native** backend needs olmo_core; the HF and vLLM backends do not. It's an optional
extra, imported lazily inside `backends/native.py` and `masking/native.py` — the only two modules in
the whole `ctc` package permitted to touch olmo_core, enforced by lint. A missing extra reports
*"backend 'native' requires `pip install ctc[native]`"*, never an ImportError traceback.

**CI test that keeps this honest:** install `ctc/` in a clean venv with no olmo_core and run the
generator + HF-backend suites. If that goes green, the independence is real rather than accidental.

---

## 1. Entry points — the whole surface a collaborator touches

```
run/
├── data.sh      # build a task's data:  generate → ladder → audit → tokenize → stage
├── train.sh     # train one config on one target:  local | slurm | beaker | lambda
├── eval.sh      # grade one checkpoint:  ckpt × task × rung × backend
└── suite.sh     # orchestrate all three for a named suite
```

Thin `sh` wrappers: resolve the environment (node-local interpreter, `TMPDIR`, `HOME`,
`FLASHINFER_CACHE_DIR`, `CUDA_HOME`), pick the submit path, exec the Python. **No logic in shell** — so
the same code runs whether launched by hand, by sbatch, or by gantry. Each is also a `console_scripts`
entry (`ctc-data`, `ctc-train`, `ctc-eval`), so an installed collaborator gets the same commands
without cloning the launcher directory.

```bash
# Data: one task, full ladder, staged locally, audited.
run/data.sh build --task contradiction --rungs 2k,4k,8k,16k,32k --out /data/ctc/v3

# Data: everything a suite needs, staged to weka.
run/data.sh build --suite ctc_suite --stage weka

# Train: config is the unit, target is a flag.
run/train.sh --config configs/train/ctc_suite/qwen35-4b-chunked-32k.yaml --target lambda

# Eval: same command for every backend; only --backend changes.
run/eval.sh --ckpt /data/ckpts/q35-4b-chunked/step1100 \
            --suite ctc_suite --rungs all --backend vllm

# Everything, for a named suite.
run/suite.sh --suite ctc_suite --stages data,train,eval --target beaker
```

**Configs are the unit of work, not scripts.** This is the fix for the biggest source of current
sprawl: 246 files under `src/scripts/train/memexpress/` exist because run configuration is encoded in
Python, so every run is a new file. `configs/train/*.yaml` + one launcher collapses that to a directory
a collaborator can diff and review.

```
configs/
├── tasks/       # one YAML per task: generation params, ladder rungs, grader, prompt template
├── train/       # one YAML per run
├── suites/      # named collections: ctc_suite.yaml, contra_ablation.yaml
└── clusters/    # berkeley.yaml, beaker.yaml, lambda.yaml — paths, QOS, partitions, priority
```

Cluster facts (Berkeley's `/data` rules, Beaker's `priority: urgent`, Lambda's two-node preempting cap)
get declared once in `configs/clusters/` instead of re-encoded across 40 sbatch files.

---

## 2. `ctc/format/` — the shared contract

Pure Python. Depended on by everything; depends on nothing.

```
ctc/src/ctc/format/
├── document.py       # doc/corpus serialization: "Document [N] (Title: …)" wrapping
├── chunk_layout.py   # token → chunk-vs-free layout   ← ONE definition
├── prompts.py        # prompt templates, query_position, cot_mode
├── markers.py        # marker token ids + the fix_marker_embeddings contract
├── registry.py       # task → {prompt builder, parser, scorer, gold convention}
├── parsing.py        # doc-id / groups / set-answer parsers
└── metrics.py        # set_f1, pairwise_f1, retrieval@k, EM
```

Three rules this module exists to enforce:

- **`chunk_layout.py` has one implementation, called by both data-gen and eval.** The oolong leak was a
  train/eval layout mismatch — the preamble was wrapped as chunks in training and left free at eval.
  That bug becomes unrepresentable when one function serves both.
- **`registry.py` owns the per-task gold-index convention** (contradiction is 1-indexed; outlier,
  rerank, nq are 0-indexed). Tribal knowledge today, and it has already faked an off-by-one defect.
- **`parsing.py` is where answer parsers live, once.**

Keeping it torch-free is load-bearing, not stylistic — it's what makes the package installable anywhere.

---

## 3. `ctc/data/` — data generation

Today: 81 generators in `corpus_reasoning/data/` plus 73 converter/staging files in `scripts/data/`,
with no shared skeleton. The pipeline is real and consistent; it just isn't expressed as one anywhere.

```
ctc/src/ctc/data/
├── cli.py                    # ctc-data: generate | ladder | audit | build
├── generators/
│   ├── base.py               # Generator protocol: emit(n_docs, seed) → example dicts
│   ├── synthetic/            # cycle, groups4, mathmatch, arithmetic, strmatch, ngram, niah,
│   │                         #   redundancy, reorder, xabsence, qdmatch
│   ├── retrieval/            # nq, msmarco (+trainhn/trecdl/rerank), beir(+ce), hotpotqa, obliq
│   ├── contradiction/        # pubmed, fever, recombined, hardneg pairs, multiclaim
│   ├── grouping/             # arxiv_grouping, review_outlier, wiki100w outlier
│   └── longctx/              # oolong (+ladder), helmet qa/summ, ruler
├── ladders.py                # rung fitting + build (v2 ladders, xlong, shared-corpus)
├── sources.py                # dataset download//cache (extra: `sources`)
└── audit/                    # ← first-class pipeline stage, not an afterthought
    ├── chunk_leak.py         # free tokens between chunks (the oolong class)
    ├── filler_provenance.py  # filler-pool domain mix (the FEVER-in-PubMed class)
    ├── digit_range.py        # train vs eval doc-id digit histograms
    ├── gold_convention.py    # per-task index base + gold reachability
    └── markers.py            # marker cosine AND norm on the base checkpoint
```

`audit/` earns its place in the pipeline rather than in `debug/`: every check listed corresponds to a
bug that reached real results and was diagnosed as a modeling finding first. Having `build` refuse to
stage a shard that fails its audits is the cheapest possible version of that lesson.

`ladders.py` also fixes a live wart — rung→`n_docs` fits currently live as long comments inside `TASKS`
dicts, and one of them silently dropped nq's 16k/32k rungs. Rungs belong in `configs/tasks/*.yaml`
where a diff shows them.

**Note the seam:** `tokenize` and `stage` are *not* here. They write olmo-core shards, so they live in
`ctc-train`. `ctc/data/` ends at task JSONL.

---

## 4. `ctc/eval/` — evaluation

The structural change: **tasks are backend-agnostic; backends only turn prompts into text.**

```
ctc/src/ctc/eval/
├── cli.py                # ctc-eval: --ckpt --task/--suite --rungs --backend
├── backends/
│   ├── base.py           # Backend protocol: load(ckpt) → M;  generate(prompts, stop) → texts
│   │                     #   + lazy discovery: missing extra → actionable message
│   ├── hf.py             # transformers
│   ├── vllm.py           # vllm (+ the 7-piece Qwen3.5 serving recipe, promoted from debug/)
│   └── native.py         # olmo_core generation_module (+ batched decode, cache_leftpad)
├── masking/
│   ├── base.py           # chunk layout from ctc.format.chunk_layout — never recomputed here
│   ├── hf.py             # was lib/chunked_attention.py, olmo3_mask_patch.py
│   ├── vllm.py           # was lib/vllm_chunked_patch.py  (ONE copy)
│   └── native.py         # olmo_core DocumentChunkedAttention
├── tasks/                # per task: prompt build, parse, score — used by ALL backends
├── decoding.py           # stop rules, </think> handling, EOS  ← one place
├── runner.py             # ckpt × task × rung matrix; resumable; dumps generations by default
└── results.py            # provenance-stamped JSON → results-hub schema
```

### Why this shape

**Per-driver parse/stop logic is the root cause of a recurring bug family.** `eval_tasks.py` exists in
two copies; the grouping-JSON parser was fixed in one; the `</think>` truncation fix lives in
`run_vllm_eval.py`; the doc-id bracket fix landed in a third place. Each was found only by dumping
generations and noticing a trained model scoring near zero. One `tasks/` and one `decoding.py` means a
fix lands once.

**`--backend` becomes real.** Today `run_rung_eval.py` declares `choices=["native"]` with a
`TODO(vllm)`, while the validated vLLM driver sits in `debug/ctc_vllm_validation/run_vllm_eval.py`.
Promoting it into `backends/vllm.py` is the highest-value single move in the eval migration.

**Retire the `--variant` naming trap.** `run_rung_eval.py` carries
`VARIANT_TO_EVALUATOR = {"dense": "full", "chunked": "dense"}` — the driver's arm names *invert* the
evaluator's, documented as deliberate. It must not survive into a repo whose selling point is
readability. One vocabulary: `--attn {full,chunked,landmark}`.

**Backend parity is a test, not a hope.** A parity case asserts the three backends agree on the same
checkpoint × task × rung within eval noise. That's what makes "supports HF and native" a claim.

---

## 5. The training side: changes to `olmo_core` + `src/scripts/ctc/`

### 5a. Into `src/olmo_core/` directly

Everything that subclasses or registers into an olmo_core abstraction, filed in the module it belongs
to. ~15k LOC, almost all already written and tested; the migration is mostly a move plus the hook.

```
src/olmo_core/
├── nn/attention/                  # + the 17 variants (SequenceMixer subclasses, 9,974 LOC)
│   ├── __init__.py                #   ← the registry hook: additive registration, no enum edits
│   ├── landmark.py  landmark_kernel.py  landmark_fast.py
│   ├── landmark_compressive.py  landmark_document{,_compressive}.py
│   ├── landmark_multi.py  landmark_sparse{,_kernel,_decode}.py
│   ├── landmark_prefill_{topk,sparse}.py
│   ├── chunked_mask.py  document_chunked.py  dilated_window.py
│   └── gold_grad_mask.py  gold_hop_mask.py
├── data/
│   ├── composable/                # + landmark / landmark_packing / pad_to_length InstanceSources
│   └── document_chunk_landmark.py
└── train/callbacks/               # + loss-curve capture, provenance stamping
```

`LandmarkAttention` still comes out of `__init__.py` into `landmark.py` — not to quarantine it, but
because a 2,318-line `__init__.py` that upstream also edits is the single worst merge surface on the
branch.

`src/test/` gains the matching 31 test files, mirroring upstream's own test layout.

### 5b. `src/scripts/ctc/` — config → run

The genuinely-not-olmo_core part, and it's small (~1.5–2k LOC), placed where upstream already keeps
launchers. Today: 12 families, 121 Python files, 40 sbatch. The active family (`ctc_suite`) is already
close to the right shape — this generalizes it and deletes the rest.

```
src/scripts/ctc/
├── cli.py            # ctc-train: --config --target [--dry-run]
├── config.py         # YAML → TransformerConfig / TrainModuleConfig  (was build_*_config)
├── mixes.py          # task mix + ratios + PadToLength / packing choice
├── plan.py           # batch geometry, shard degree, AC, mask-mix curriculum, provenance
├── tokenize/         # task JSONL → olmo-core SFT shards  ← the adapter at the package seam
│   ├── sft.py        #   was convert_unified_to_sft.py
│   ├── landmark.py   #   was convert_unified_to_document_landmark.py
│   ├── markers.py    #   was fix_marker_embeddings.py
│   └── stage.py      #   local | weka | s3 placement + manifests
└── targets/
    ├── local.py      # torchrun
    ├── slurm.py      # Berkeley sbatch  (node-local /data, log rules)
    ├── beaker.py     # gantry           (priority: urgent, weka mounts)
    └── lambda_.py    # Lambda sbatch    (two-node preempting cap, offline wandb)
```

`tokenize/` is the module that spans the package boundary: it reads `ctc`'s task JSONL and writes
olmo-core shards. It sits on this side because the shard format is olmo_core's — that's what keeps
`ctc` clean.

Behavior the targets should own, because it's currently re-derived per launcher and gets forgotten:
refuse to start if the save-folder is non-empty unless `--resume` is explicit (silent auto-resume has
already corrupted step counts and wall-clock numbers); write logs to node-local disk; capture the full
per-step loss curve into the result JSON; emit the wandb group link on launch.

---

## 6. Full tree

Every leaf, in one place. Sections 2–5 explain the reasoning behind each group.

```
ctc-core/
├── QUICKSTART.md                     # clone → env → run one task, in <20 lines
├── Makefile                          # checks, tests, docs
├── pyproject.toml                    # the olmo-core fork: olmo_core + olmo_core_ext + launch
│
├── run/                              # ═══ THE ENTRY POINTS ═══ thin sh; no logic
│   ├── data.sh                       #   generate → ladder → audit → tokenize → stage
│   ├── train.sh                      #   one config, one target
│   ├── eval.sh                       #   ckpt × task × rung × backend
│   └── suite.sh                      #   orchestrate all three for a named suite
│
├── configs/                          # ═══ THE UNIT OF WORK ═══ not scripts
│   ├── tasks/                        #   per task: gen params, ladder rungs, grader, prompt
│   │   ├── contradiction.yaml
│   │   ├── outlier.yaml
│   │   └── … (one per task)
│   ├── train/                        #   per run  (replaces 246 launcher .py files)
│   │   └── ctc_suite/qwen35-4b-chunked-32k.yaml
│   ├── suites/                       #   ctc_suite.yaml, contra_ablation.yaml
│   └── clusters/                     #   berkeley.yaml, beaker.yaml, lambda.yaml
│
├── ctc/                              # ═══ pip package `ctc` — NO olmo_core ═══
│   ├── pyproject.toml                #   extras: hf | vllm | sources | native | all
│   ├── README.md                     #   standalone: install → generate data / grade a ckpt
│   ├── src/ctc/
│   │   ├── format/                   # ── the shared contract (pure python) ──
│   │   │   ├── document.py           #   "Document [N] (Title: …)" serialization
│   │   │   ├── chunk_layout.py       #   token → chunk-vs-free   ← ONE definition
│   │   │   ├── prompts.py            #   templates, query_position, cot_mode
│   │   │   ├── markers.py            #   marker ids + fix_marker_embeddings contract
│   │   │   ├── registry.py           #   task → {prompt, parser, scorer, gold convention}
│   │   │   ├── parsing.py            #   doc-id / groups / set-answer parsers
│   │   │   └── metrics.py            #   set_f1, pairwise_f1, retrieval@k, EM
│   │   │
│   │   ├── data/                     # ── generation: JSONL in, JSONL out ──
│   │   │   ├── cli.py                #   ctc-data: generate | ladder | audit | build
│   │   │   ├── generators/
│   │   │   │   ├── base.py           #     Generator protocol: emit(n_docs, seed)
│   │   │   │   ├── synthetic/        #     cycle, groups4, mathmatch, arithmetic, strmatch,
│   │   │   │   │                     #       ngram, niah, redundancy, reorder, xabsence, qdmatch
│   │   │   │   ├── retrieval/        #     nq, msmarco(+trainhn/trecdl/rerank), beir(+ce),
│   │   │   │   │                     #       hotpotqa, obliq
│   │   │   │   ├── contradiction/    #     pubmed, fever, recombined, hardneg, multiclaim
│   │   │   │   ├── grouping/         #     arxiv_grouping, review_outlier, wiki100w outlier
│   │   │   │   └── longctx/          #     oolong(+ladder), helmet qa/summ, ruler
│   │   │   ├── ladders.py            #   rung fitting + build (v2, xlong, shared-corpus)
│   │   │   ├── sources.py            #   dataset download/cache          (extra: sources)
│   │   │   └── audit/                # ── a pipeline stage, not a debug script ──
│   │   │       ├── chunk_leak.py     #     free tokens between chunks (oolong class)
│   │   │       ├── filler_provenance.py  # filler-pool domain mix (FEVER-in-PubMed class)
│   │   │       ├── digit_range.py    #     train vs eval doc-id digit histograms
│   │   │       ├── gold_convention.py#     per-task index base + gold reachability
│   │   │       └── markers.py        #     marker cosine AND norm on the base checkpoint
│   │   │
│   │   └── eval/                     # ── grading: tasks are backend-agnostic ──
│   │       ├── cli.py                #   ctc-eval: --ckpt --task/--suite --rungs --backend
│   │       ├── backends/
│   │       │   ├── base.py           #     protocol + lazy discovery of missing extras
│   │       │   ├── hf.py             #     transformers
│   │       │   ├── vllm.py           #     vllm (+ Qwen3.5 serving recipe, from debug/)
│   │       │   └── native.py         #     olmo_core   ← one of only 2 files that may import it
│   │       ├── masking/
│   │       │   ├── base.py           #     layout from ctc.format.chunk_layout, never recomputed
│   │       │   ├── hf.py             #     was lib/chunked_attention.py, olmo3_mask_patch.py
│   │       │   ├── vllm.py           #     was lib/vllm_chunked_patch.py  (ONE copy)
│   │       │   └── native.py         #     olmo_core   ← the other one
│   │       ├── tasks/                #   per task: prompt build, parse, score — ALL backends
│   │       ├── decoding.py           #   stop rules, </think> handling, EOS  ← one place
│   │       ├── runner.py             #   ckpt × task × rung; resumable; dumps generations
│   │       └── results.py            #   provenance-stamped JSON → results-hub schema
│   └── tests/                        #   incl. clean-venv install + backend-parity tests
│
├── src/                              # ═══ the branch's changes to olmo-core ═══
│   ├── olmo_core/                    # ── UPSTREAM + OUR CHANGES, filed in place ──
│   │   ├── nn/attention/
│   │   │   ├── __init__.py           #   + the registry hook (additive, no enum edits)
│   │   │   ├── landmark.py           #   + LandmarkAttention, out of __init__.py
│   │   │   ├── landmark_kernel.py  landmark_fast.py  landmark_compressive.py
│   │   │   ├── landmark_document{,_compressive}.py  landmark_multi.py
│   │   │   ├── landmark_sparse{,_kernel,_decode}.py
│   │   │   ├── landmark_prefill_{topk,sparse}.py
│   │   │   ├── chunked_mask.py  document_chunked.py  dilated_window.py
│   │   │   └── gold_grad_mask.py  gold_hop_mask.py
│   │   ├── data/
│   │   │   ├── composable/           #   + landmark / landmark_packing / pad_to_length sources
│   │   │   └── document_chunk_landmark.py
│   │   └── train/callbacks/          #   + loss-curve capture, provenance stamping
│   │
│   ├── scripts/
│   │   ├── … upstream launchers …
│   │   └── ctc/                      # ── config → run  (~1.5–2k LOC) ──
│   │       ├── cli.py                #   ctc-train: --config --target [--dry-run]
│   │       ├── config.py             #   YAML → TransformerConfig / TrainModuleConfig
│   │       ├── mixes.py              #   task mix + ratios + PadToLength / packing
│   │       ├── plan.py               #   batch geometry, shard degree, AC, mask-mix, provenance
│   │       ├── tokenize/             #   ← the adapter spanning the package seam
│   │       │   ├── sft.py            #     was convert_unified_to_sft.py
│   │       │   ├── landmark.py       #     was convert_unified_to_document_landmark.py
│   │       │   ├── markers.py        #     was fix_marker_embeddings.py
│   │       │   └── stage.py          #     local | weka | s3 placement + manifests
│   │       └── targets/
│   │           ├── local.py          #     torchrun
│   │           ├── slurm.py          #     Berkeley sbatch (node-local /data, log rules)
│   │           ├── beaker.py         #     gantry (priority: urgent, weka mounts)
│   │           └── lambda_.py        #     Lambda sbatch (2-node preempt cap, offline wandb)
│   │
│   └── test/                         # upstream's test layout + our 31 test files
│
├── docs/
│   ├── clusters/                     #   local_cluster.md, lambda_cluster.md, beaker.md
│   └── traps.md                      #   the hard-won-lessons index from CLAUDE.md
└── records/                          # experiment writeups (as today)
```

The one placement rule the tree encodes: **`ctc/` sits beside `src/`, never inside it** — so it's
unreachable by an accidental relative import from olmo_core, and its independence is structural rather
than disciplinary. Everything else follows upstream's existing conventions, because it *is* upstream's
tree with our commits on it.

New top-level entries the branch adds: `ctc/`, `run/`, `configs/`. Three, all obviously ours.

---

## 7. Build order (working backwards from the commands)

Each step ends with a command that actually runs — no phase is scaffolding-only.

| # | Deliverable | Working command at the end |
|---|---|---|
| 1 | `ctc/format/` ported and tested | `pytest ctc/tests/format` |
| 2 | `ctc/eval/`, `native` backend only, 1 task, 1 rung | `run/eval.sh --ckpt … --task contradiction --rungs 2k --backend native` |
| 3 | `vllm` + `hf` backends + parity test | same command, `--backend vllm` / `hf`, scores agree |
| 4 | `masking/` for all three backends | `--attn chunked` reproduces known chunked numbers |
| 5 | full task set + `runner.py` + `results.py` | `run/eval.sh --suite ctc_suite --rungs all` |
| 6 | `ctc/data/` generate → ladder → audit | `run/data.sh build --task contradiction` |
| 7 | olmo_core changes (attention + sources) + `scripts/ctc/{tokenize,targets/local}` | `run/train.sh --config … --target local` |
| 8 | remaining targets, `suite.sh`, QUICKSTART | `run/suite.sh --suite ctc_suite` |

**Eval comes first, deliberately.** It's the smallest self-contained pipeline, it proves out the
package boundary, and — most usefully — a working eval on existing checkpoints is what lets every later
port be verified against known numbers (contra @2k f1 ≈ .865/.843; n=100 chunked .441 / hier-K50 .831 /
full .934). Without it, nothing downstream has a gate.

Steps 1–6 involve **no olmo_core at all**, which means they can be built and tested on any machine,
including one with no GPU and no compiler.

---

## 8. Open questions / decisions needed

Consolidated across both docs, ordered by when they block. ✅ = I have a recommendation and just need
a yes; ❓ = genuinely yours to call.

### Blocking Phase 0 (right now — nothing ports until these clear)

1. ✅ **Commit the 24 modified files?** All 24 are real, documented work — nothing wants reverting.
   Proposed as 5 topical commits: GDN leftpad + test / Gemma-3 hybrid config / task-validity data
   fixes / eval grading + shared-corpus cache / launchers + notes. Then tag `pre-migration-source`.
   **Who runs it — me or you?**
2. ❓ **Sweep the 213 untracked paths for stray source?** Almost all are `debug/` scratch, but I
   haven't audited them and un-added source would be silently lost.

### Blocking Phase 1–2 (repo setup and the ledger)

3. ✅ **Branch name `prasann/ctc`?**
4. ✅ **Flatten the clone path?** It nests as `projects/newolmocore/OLMo-core`. Cheap `mv` now; the
   editable install, the `training/` symlink, and the Claude memory dir are all path-keyed.
5. ❓ **Which of the 12 training families port as configs, and which get archived?** Recommend
   `ctc_suite` + `sft_docchunk` + `attn_explore`, archiving the other 9 — but that depends on which
   experiments are still live, which is your knowledge, not mine.

### Blocking the eval build (Phase 3 onward)

6. ❓ **Is the chunked-vLLM FlexAttention fallback bug still open?** Recorded as open on 2026-07-21
   (chunked emits `!!!!` on 4B GDN; checkpoints fine in full mode; 100% fallback in
   `create_block_mask_compiled`). `debug/chunked_eval_speedup/` has `flex_blockmask_cache_bug.py` and
   `flex_parity.py` but no resolution note I can find. **Build order step 4 validates `masking/`
   against known chunked numbers — that step can't gate on a path that's broken.** If it's still
   open, either fix it first or reorder.
7. ❓ **Does the CTC-suite contradiction ladder need rebuilding before it's a validation target?**
   The FEVER-filler leak is recorded as fixed for xlong but *still open* for the CTC-suite ladder at
   92–99% FEVER fillers. And the `contra_ctc` recalibration concluded the original BUILD_MATRIX
   ladder was correct all along. Both bear on which numbers we treat as ground truth when verifying
   ports.
8. ❓ **Are nq's 16k/32k rungs meant to stay dropped?** They were deferred pending a uniform-k200 CE
   eval regen. Fine as a deliberate gap; worth confirming it isn't an accident.

### Structural, needed before the relevant phase

9. ❓ **Does `ctc` want a `viz/` module?** `corpus_reasoning/viz` + `viz_hub` (16 files) feed the
    results website. Pure-python and results-shaped, so they'd fit `ctc` cleanly — or stay in
    results-hub, which already owns the results contract.
10. ✅ **Merge cadence with upstream `main`.** Monthly keeps each resolve small; the current branch
    went two months and picked up conflicts in 6 files. Worth a calendar entry, not a "when it
    breaks" trigger.
11. ✅ **Directory names.** `ctc` is locked. Still unconfirmed: `run/`, `configs/`, `src/scripts/ctc/`.

### Deferred (recommendations already in the plan's ledger, §5)

12. ✅ **Big tracked data files** — the 6 `eval_rungs/*/rung_{2048,4096}.jsonl` (2.5–5.5 MB each, our
    largest tracked blobs) move to `/data` plus a manifest rather than being re-committed.
13. ✅ **`debug/` (207 tracked) and `deprecated/` (79)** — archive, don't port. Old repo stays in
    place as `OLMo-core-archive` with a `records/` pointer.
14. ✅ **`results/` (104 files)** — results-hub is the source of truth; port only what isn't there.
