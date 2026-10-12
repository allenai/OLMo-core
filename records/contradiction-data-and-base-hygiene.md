# Contradiction data & base-checkpoint hygiene — what is poisoned, and what to use instead

**Why this exists.** Several artifacts in `/scratch` and a few defaults in the tree are *silently*
wrong: they load, they train, they produce a plausible number, and the number is garbage. Each has
already cost at least one bogus conclusion. This is the running list, so a future agent (or a future
us) does not reach for them by accident. Items marked **DEPRECATE** are candidates for removal once
their live references are gone.

Grouped by how they bite.

## 1. Base checkpoints — marker embeddings

Qwen3 never trains the reserved marker rows (`<|box_start|>`/`<|box_end|>`, landmark/pad ids). On a
stock base they are **bit-identical** (cos 1.0000), so the model cannot tell "open document" from
"close document", and marker-dense runs train to chance. Two repairs exist and **only one is correct**.

| base | status | use? |
|---|---|---|
| `q06b-dense-cpt-modelonly-trainedmark` | markers seeded from **trained donor rows**; norm in-distribution | ✅ **the only correct one** |
| `q06b-dense-cpt-modelonly-fixmark` | fixed marker **cosine**, left norm at ~1/3.6 of a real token | ❌ **DEPRECATE** |
| `q06b-dense-cpt-modelonly-markfix2` | provenance unverified — predates the norm fix by name | ❌ assume bad until checked |
| stock Qwen3 | markers bit-identical | ❌ never for docchunk/landmark |

⚠ **The `fixmark` failure mode is maximally deceptive.** RMSNorm amplifies the undersized marker norm
into full-strength noise, flatlining n=100 chunked training at CE ≈ 0.79 **for every mask, including
plain causal**. An unrestricted model cannot even memorize the data — so it reads as "my mask is too
restrictive" when the mask is fine. Any base repaired before **2026-07-14** is affected. Full
diagnosis: `records/n100-chunked-marker-position-bug.md`, `records/document-chunked-marker-embeddings.md`.

⚠ **A live launcher still defaults to the bad base.**
`src/scripts/train/memexpress/goldgrad/run_q06b_goldgrad_contra.sbatch` has
`BASE_SRC="${BASE_SRC:-.../q06b-dense-cpt-modelonly-fixmark}"` — a default that silently poisons any
run that does not override it. **Fix this before running anything from that file.**

## 2. Contradiction shards — v1 (leaky) vs v2 (clean)

v1 puts the `Claim N:` label **outside** the chunk, so the label leaks to the FREE token stream. v2
puts it **inside**. This is not cosmetic: goldgrad's sparse-backward arm *replicates* on v1
(sparse ≈ full) and *collapses* on v2 (0.55 vs 0.938) — the v1 "result" was the model riding the leak.

| shards | layout | verdict |
|---|---|---|
| `contra_n100_v2_*`, `contra_n50_v2_7k` | v2, leak-free | ✅ use |
| `contradiction_n20_docdense*`, `contradiction_n50_docdense`, `contradiction_n100_docdense*`, `contradiction_n250*` | **v1** | ❌ **DEPRECATE** |
| `contra_n100_v2_v1pos` | v2 data, v1 positions restored by post-hoc surgery | ⚠ deliberate control — not a general-use shard |

The current `convert_unified_to_document_landmark.py` emits **v2 by default**; there is no v1 flag.
`v1pos` is `metadata.surgery == "v1pos"` applied *after* the fact, which is how to tell them apart.

⚠ `contradiction_n50_docdense` is the trap of this shape most likely to be grabbed next — it is the
*only* pre-existing n=50 shard, it is v1, and its name looks perfectly reasonable.

## 3. Gold sidecars

| file | contents | verdict |
|---|---|---|
| `gold_pairs.json` | pair-preserving `[[a,b],[c,d]]` | ✅ required for anything pair-aware |
| `gold_fingerprints.json` | **flat set** of gold chunk ids | ⚠ cannot express *which* doc contradicts which |

⚠ The flat sidecar is what invalidated the first goldgrad arms: `gold_chunks_from_gold_doc_indices`
flattens `[[9,28],[10,31]]` into an unordered set, so **no keep-policy could hold a pair together on
purpose** — every `gsub1_*` arm saw one half of one pair with its partner detached. "Sparse backward
loses accuracy" may really have been "we never gave it an intact pair."

⚠ **`debug/build_gold_pairs.py` does not exist**, though `Qwen3-0.6B-goldgrad-contradiction-n20-SFT-local.py`
points `--grad-mode gold_pair|gold_halves` at it. Those modes currently have no documented way to build
their sidecar. `build_gold_sidecar_from_shard.py` emits the **flat** form only.

## 3b. ⚠ Recombining contradiction pairs DESTROYS the hard negatives (2026-07-16)

**Do not build contradiction examples by pooling gold pairs and filling with globally-sampled
distractors.** `contra_n50_v2_7k` (built by `build_contra_recombined.py` with a 386k global filler
pool) was decontaminated, zero-reuse, and distance-matched to the eval — and still trained a model
that scored **f1 0.585 with FULL attention**, where n=100 full scores 0.934 on a *strictly harder*
corpus. Every arm was uniformly depressed (chunked 0.267 vs 0.441 at n=100).

The diagnosis, measured over 120 examples with word-overlap Jaccard:

| dataset | gold-pair sim | **best NON-gold sim** | margin |
|---|---|---|---|
| recombined (global filler pool) | 0.461 | **0.163** | **0.30** — trivial |
| official eval n50 | 0.445 | **0.372** | 0.07 — hard |
| original train n50 | 0.454 | **0.333** | 0.12 |

Gold pairs are equally similar everywhere (~0.45). What the recombination changed is the
**distractors**: real examples co-sample distractors *with* the gold pair from a related pool, so the
nearest non-gold pair is nearly as similar as the gold one and the model must genuinely detect
contradiction. With a global filler pool the nearest distractor sits at 0.163, so **"pick the two most
similar claims" solves training outright** — a shortcut that collapses on the topically-coherent eval.

The signature to recognise: **train CE is normal (0.0144, matching a good n=100 run) but eval f1 is
far below it.** That gap is a train/eval *distribution* mismatch, not undertraining — undertraining
would show up in CE.

⚠ **Every validation passed.** 0 eval contamination, 0 pair reuse, gold-pair distance distribution
matched to the eval (17.0 vs 16.7), 0 duplicate docs, 0 filler-is-secretly-gold. The checks verified
the properties someone thought to check; none of them asked *whether the task was still the same task*.
Difficulty structure is a property of the joint (gold, distractor) sample and cannot be reconstructed
by recombination — this is the same match-difficulty-or-confound lesson as the fiqa/scifact OOD
ladders.

⇒ **Use the original generator's examples** (`contradiction_train_pubmed_both_n50_k3.jsonl`, 2000 ex,
distractors co-sampled) — or, to scale, `--expand-from-train`, which *preserves* each example's own
distractor context. `build_contra_recombined.py` keeps its decontamination logic (the 14 poisoned pairs
are real and worth dropping) but its filler sourcing must not be reused as-is.

## 4. Source JSONL pools (`/scratch/users/prasann/corpus-reasoning/data/`)

Measured 2026-07-16 across every `contradiction_*pubmed*.jsonl`:

| pool | unique pairs | eval overlap | verdict |
|---|---|---|---|
| `both` | 6000 | **0** | ✅ matches the eval (`both` = 50/50 simple+subtle) |
| `subtle` | 11679 | **1** | ✅ same style family — decontaminate first |
| `simple` | 4641 | **13** | ✅ same style family — decontaminate first |
| `realistic` | 4737 | 0 | ⚠ **different style** ("fully rephrases — no near-duplicate tells") — pooling it shifts the distribution away from every `both` eval |

⚠ **14 train pairs appear verbatim in the eval sets** (13 `simple` + 1 `subtle`). The `both`-only pool
is clean, which is why existing results are safe — the contamination only bites the moment `simple` /
`subtle` are pooled in, which is exactly what any large recombined build does.
`build_contra_recombined.py` drops them; **any new builder must too.**

⚠ Do not confuse the **root** `/scratch/users/prasann/corpus-reasoning/` clone with the in-tree code —
the clone is stale (see the corpus-reasoning-submodule memory); only its `data/` is live.

## 4b. ⚠ FEVER / wiki_mix fillers leaking into PubMed contradiction evals (2026-07-29)

**The defect.** Contradiction distractors are harvested by `harvest_fillers`, which globs a *mutable*
directory. `build_xlong_rungs.py` used `contradiction_*_k3.jsonl` — which also matches the **FEVER**
and **wiki_mix** corpora. So a file named `contradiction_eval_pubmed_both_*` shipped Wikipedia claims
as distractors. `build_v2_eval_ladders.py` already restricted to `*pubmed*` (its own comment calls the
broad glob "the dominant leak vector"); the xlong builder never inherited that, which is why the leak
**starts at 64k**. `contra_fever` is a *separate experimental setting* and must never bleed in.

**Why it fakes a result.** The gold pair is PubMed; the distractors are not. "Find the contradicting
pair among n documents" then collapses to "find the biomedical sentences, then pick the pair among
those" — the effective search space is the gold set, not n, *regardless of the rung label*. The
shortcut gets relatively stronger as n grows, so a contaminated ladder can look artificially robust
to context length.

Measured (exact-text fingerprint match, `debug/xlong_5task/audit_contra_fever_leak.py`):

| bundle / rung | FEVER+wiki share of documents | verdict |
|---|---|---|
| `eval500_v2` contra base 2k/8k/16k/32k | **0.00%** | ✅ clean (pubmed-only glob) |
| `_eval_bundle_eval500_v2` contra 64k…2M | **28–31%** | ❌ contaminated → rebuilt |
| **CTC suite** `contradiction/rung_2048…32768` | **92–99.6%** — *all* fillers, gold 100% PubMed | ❌ **OPEN** |
| CTC suite `contradiction/rung_131072` | 30% (mixed pool, built later) | ⚠ different task than the rungs below it |

⚠ **The CTC-suite contradiction ladder is the worst instance and is NOT fixed.** At 8k/32k, *every*
filler is FEVER/wiki and *every* gold doc is PubMed — e.g. `'A Floppy disk is a type of storage.'`
against gold `'In S clones, the ratio of dihydroxylysinonorleucine (DHLNL) to …'`. Recorded numbers
rest on it (`results/ctc_suite/all_results.jsonl` contradiction 2k f1 .958/.865/.843, 8k .219,
32k .038, plus the Stage-3 2k validation). The confound is *constant* across 2k–32k so it does not
manufacture the 2k→32k collapse, but absolute values are not comparable to a pubmed-only ladder — and
`rung_131072` **is** a different (harder) task than the rungs below it, so part of any apparent
long-context drop at 131k is a task change, not a length effect.

**Second-order damage: non-reproducible rungs.** The three families have very different document
lengths (FEVER ~15–20 tok/doc, PubMed ~35, wiki_mix ~146), so the pool's mean depended on which files
happened to be on disk. Between the 2026-07-02 and 2026-07-29 builds it moved **36.5 → 47.1 tok/doc**,
so the same token target solved to a ~29% different doc count (contra 256k: n6408 vs n4944) — the
rungs hit their token labels but `n` was not comparable across builds.

**Fixed / mitigations.**
- `build_xlong_rungs.py` glob is now `contradiction_*pubmed*_k3.jsonl` (81,250-doc pool, 42.96
  tok/doc), which also restores a clean 2× doc-count progression across the whole 64k…2M ladder.
- Every rung now writes a `.manifest.json` pinning its source files with sizes + mtimes, so "same
  rung label" can no longer silently mean "different corpora".
- Calibration **fails loudly** when the pool is smaller than a rung needs (a short pool repeats
  documents within an example — a known confound in its own right).
- Audit tooling: `debug/xlong_5task/audit_contra_fever_leak.py` (per-rung membership),
  `audit_filler_pool.py` (pool composition by source), `audit_weka_ladder.py` (per-rung eval_size +
  leak + glob ambiguity, run on weka via `audit_weka_ladder.sh`).

**Use this bundle.** `_eval_bundle_eval500_v2_clean` on weka — the **default** since 2026-07-29 in
`run_beaker_multirung_eval.sh` (`EVAL500`) and in `eval_lc_native.py`'s `_V2_BUNDLES` order. Verified
2k→2M for contra/nq/outlier/rerank/oolong at `eval_size ≥ 500` and 0.00% contra leak. The ≤32k rungs
are **byte-identical** to the old bundle (ContentLength + ETag), so no ≤32k number moved; **contra at
64k+ is not comparable across the switch** (clean 256k = n6102 vs old n6408) — re-run, don't mix.

## 5. Defaults that are wrong for gold-edge work

- `random_doc_per_example=False` (the default, and what every prior random_doc run used) gives **one
  fixed graph shared across all layers and all examples**. It is memorizable — and against a memorized
  graph a *missing* edge announces which doc is gold. Any gold-edge-deletion experiment must set
  `--random-doc-per-example`. Not a bug; a wrong default for this purpose.
- Curriculum mask-mixing is the project default and runs a large fraction of forwards under plain
  causal. Fine (it anneals to 0) — but under the **old NGPU anneal bug** it floored at
  `0.8*(1-1/NGPU)`, keeping plain causal on ~40% of forwards forever. The `// world_size` fix is in
  the tree; **verify `p_standard` actually reaches 0 in the logs** rather than trusting it.
- `--expand-from-train` is **1 source example → 1 output example**, so it cannot exceed the source
  size (2000). Not dangerous, just silently limiting — use `build_contra_recombined.py` to scale.

## 6. Local-cluster traps (measured 2026-07-16 on mooney, while launching hopgold Stage 1)

**⚠ `/scratch` and `/accounts` are THE SAME NFS SERVER** — `oz.berkeley.edu:/pool0` serves both
(`df -T`). `/scratch` is not a faster tier; it is the same wire. Only `/data/<user>/` is node-local.

Measured startup cost of one 0.6B run (~3.5 min before step 1, vs 8.7 min of actual training):

| what | filesystem | cost/run | fix |
|---|---|---|---|
| `olmo_core` imports (repo on `/accounts`) | NFS | **~2.5 min** | rsync repo → `/data` at job start |
| base checkpoint, 1.9 GB | NFS (`/scratch`) | ~40 s | stage → `/data` once per node |
| training data | already staged → `/data` | — | ✅ |

⚠ **Staging a checkpoint with plain `cp` drops the hidden `.metadata` → the trainer SILENTLY TRAINS
FROM SCRATCH.** Use `cp -a`/rsync and assert `.metadata` exists (the goldhop sbatch does).

**⚠ `/scratch/users/prasann/olmo_ckpts` is a SYMLINK to `/data/prasann/olmo_ckpts` — i.e. NODE-LOCAL
despite the `/scratch/...` path.** Consequences, all of which cost time on 2026-07-16:
- Checkpoints written on mooney exist ONLY on mooney. Every arm of a comparison, and its eval, must
  run on the same node.
- A `ls` from the login host resolves the symlink against the *login host's* `/data` and reports "not
  found" for a checkpoint that exists on the node. Check on the node (`srun -w <node> ls ...`).
- ⚠ `run_q06b_goldgrad_contra.sbatch`'s header says checkpoints go "to /scratch so the existing eval
  can read them from any node". **That is false** — they are node-local. Do not rely on it.

**⚠ `num_workers=2` DEADLOCKS when several torchrun jobs share a node.** Three concurrent 2-GPU jobs
hung indefinitely at the first *real* batch — `Dry-run complete` (0.8 s), `Starting epoch 1...`, then
nothing for 7+ min, at ~2% CPU, with distinct GPUs each (so neither compile nor a GPU collision). A
single job survives, which is why the smoke test passed and hid it. `num_workers=0` fixes it
completely (0.56 s/step immediately). The data is a memory-mapped `.npy` already staged node-local, so
workers buy nothing here anyway. This is why `run_q06b_goldgrad_contra.sbatch` forces `--num-workers
0`. `Qwen3-0.6B-docchunk-mask-mix-...-local.py` **hardcoded** `num_workers=2` with no flag until
2026-07-16; it now takes `--num-workers`.

**⚠ The mask-mix anneal has TWO divisors, and missing either silently voids a mask experiment.**
`mix_total_forwards` must equal the number of FORWARDS a rank actually performs, because that is what
the curriculum counter counts. It must be divided by **both**:
- `world_size` — data is sharded across DP ranks (the original mask-mix-ngpu-anneal bug: `p_standard`
  stalled at `mix_start_p*(1-1/world_size)`).
- `micro_batch_instances` — a forward carries this many instances, so raising it CUTS the forward
  count by the same factor. **New instance found 2026-07-16**: adding `--micro-batch-instances 4` as a
  throughput win re-broke the anneal through this second path. With world=2, micro=4: a rank runs 927
  forwards but `mix_total_forwards` assumed 3710 ⇒ `prog=0.25` ⇒ **`p_standard` ended at 0.601, not
  0.0**. All three Stage-1 arms trained with ~60% of forwards still on PLAIN CAUSAL and had to be
  thrown away.

Why this is so dangerous: **it does not crash and the loss looks fine.** The chunked arm reported a
healthy CE of 0.069 — *lower* than a correct chunked run, precisely because it was mostly training
unmasked. Nothing looks wrong; the mask experiment is simply not the experiment you think you ran.

The invariant: **`forwards_per_rank == number of optimizer steps × (per_rank_instances /
micro_batch_instances)`**. `Qwen3-0.6B-docchunk-mask-mix-...-local.py` now re-derives this
independently and **hard-fails** if the curriculum would not land on `mix_end_p` (verified: the guard
fires on the 0.601 config, passes at 0.0009). Any new launcher in this family must keep that check —
"verify p_standard reaches 0 in the logs" is necessary but too late, since it costs a full run.

**`torch.compile` is a poor fit for these short runs.** The dry-run compiles, then the first real batch
**recompiles** for its shapes — observed as a multi-minute stall after `Starting epoch 1`. Over ~900
short steps it roughly breaks even at best. More importantly, **any gold-aware / fingerprint-hook mask
CANNOT be compiled** (per-forward Python + RNG), so a compiled baseline and an eager hop arm would come
from different code paths. Keep an experiment family entirely eager or entirely compiled — never mixed.

**⚠ `make style` / `make checks` are UNSAFE in the `corpus-reasoning-olmo` env — the formatter is the
wrong major version.** Measured 2026-07-16:

| | repo pins (`pyproject.toml`) | env has |
|---|---|---|
| black | `>=23.1,<24.0` | **26.5.1** |
| isort | `>=5.12,<5.14` | **8.0.1** |

`black --check src/` in this env: **294 files would be reformatted, 290 of them committed and
otherwise unmodified** — i.e. pure collateral, nothing to do with any current change. Consequences:
- **`make style` churns ~290 committed files** with a formatter the repo does not use. One agent ran it
  and had to revert 391 files of collateral by hand.
- **`make style-check` / `make checks` CANNOT pass in this env**, regardless of your diff. A red result
  here means nothing; do not "fix" it by reformatting.
- Cosmetic churn leaks in silently: `Qwen3-0.6B-docchunk-mask-mix-...-local.py` picked up ~66 lines of
  black-26 reformatting (multi-line `dict()`, restructured `add_argument`) on top of a functional fix.
  Harmless to run, but it misattributes the diff.

Until the env pins black/isort into the repo's range: **run `ruff` and `pytest` (both fine), skip
`make style`/`make checks`**, and format new lines by hand to match the surrounding file.

**Trainer creates `--save-folder` at STARTUP, not at first save.** So any relaunch of a crashed/cancelled
run trips a `[ -d "$SAVE_FOLDER" ]` guard and needs a manual `rm -rf` on the node. That is the correct
trade: without the guard the Trainer **silently resumes** mid-run (starts at step N, not 1), which
quietly corrupts step counts, wall-clock, and speedup numbers. Keep the guard; clean up by hand.

## 7. Eval harness traps (already recorded, repeated here because they fake *modeling* conclusions)

- **`--max-length` truncation**: at 4096 the ~6144-token n=100 prompt truncates → empty generation →
  f1 **0.000 at parse_rate 1.0** for a *perfect* model. Faked a whole "goldgrad doesn't replicate"
  conclusion. Use MAXLEN ≥ 8192 at n=100. **Dump generations whenever a trained model evals exactly 0.**
- **no-cot never emits EOS** → rambles → precision collapse. Pass `--eos-token-id 151643`.
- **A rung's LABEL is a nominal budget; the built prompt runs 0.4–4% OVER it** (doc count calibrated
  from a median, plus the instruction/query/marker wrap). The xlong `--max-length` auto-raise used
  `label + 1024`, which truncated the prompt *tail* — where the question is — at 512k (535,855 actual
  vs 525,312 allowed) and 2M (2,165,314 vs 2,098,176). Same 0.000-at-parse-1.0 signature as the
  `--max-length` item above. Now a 10% margin, in both `eval_lc_native.py` and the runner's rung table.
- **v1 ladders are DISABLED** (2026-07-29) and raise `NotImplementedError`: each v1 rung drew its OWN
  questions, so every rung-to-rung delta carried eval-set resampling noise on top of the length effect
  it was meant to isolate. v2 fixes the question set and varies only distractors.
- **Trainer silently auto-resumes** into an existing `--save-folder` (starts at step N, not 1), making
  step counts and wall-clock/speedup numbers garbage, and can inherit a poisoned base.
