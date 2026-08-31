# Pooled-doc-KV attention: train-time KV compression, full-attention inference

**Status (2026-08-29): first 4B experiment DONE (contradiction n100, local mooney; family folder
`src/scripts/train/memexpress/pooledkv/`). Iteration 1 verdict: B1 trains 4.9x faster than full,
but ZERO-SHOT pooled->full transfer COLLAPSES for both compressed arms; iteration 2 (compression-
mixing curriculum + de-leaked keep set) in flight.**

## Iteration 1 (2026-08-29): speedup real, naive transfer fails

Setup: Qwen3-4B (fixmark dolma3longmino CPT base), contradiction n100 qboth shard (2000 ex,
p50 4.6k tokens), 3 epochs, gbs 8 examples, 2xH200 mooney, keep = gold_plus_random n_random=2.
Eval: full attention, 488 held-out, no-cot.

| arm | TPS/device | step | train CE | full-attn eval f1 |
|---|---|---|---|---|
| full | 8,280 | 3.0s | 7e-5 | **0.985** |
| v1 pooledkv | 4,548 | 5.4s | 1.6e-5 | 0.156 |
| B1 softtoken (micro=4 ex) | **40,230 (4.86x)** | 0.76s | 1e-5 | 0.081 |

parse_rate 1.0 everywhere -> genuine capability gap, not formatting. Diagnosis (two compounding
causes, both anticipated): (1) **keep-set leak** -- under gold_plus_random the real-token docs ~=
the gold docs, so the model learns "the real docs are the answer" (v1's train CE is *below* full's);
at eval all 100 docs are real and the signal vanishes. (2) **no full-attention anchoring** -- the
mask-mixing line already showed pure-restrictive training collapses and a curriculum rescues it.
B1 speed notes: micro-batch 1 was launch-latency-bound (1.8x); 4 examples/forward -> 4.9x; the
residual gap to ~7x compute ratio is optimizer+FSDP fixed cost at 2 GPUs and short (4.6k) rows.

## Iteration 2 (2026-08-29): compression-mixing curriculum + de-leaked keep set -> TRANSFER RECOVERED

`make_fingerprint_keep_docs_fn(mix_start_p=0.8, mix_end_p=0.0, mix_total_calls=<forwards>)`: each
row trains UNCOMPRESSED with probability p annealing 0.8 -> 0 (per-(fingerprint, call) seeded, so
examples flip across epochs); keep policy `gold_subsample n_gold=2 n_random=6` (base-rate
preserving, answer not derivable from which docs are real).

| arm | full-attn eval f1 (488, +-0.007) | train TPS/device |
|---|---|---|
| full baseline | 0.985 | 8,280 |
| v2 v1mix (exact-mean KV slots + mix) | **0.977 (parity, ~1 SE)** | 4,548 |
| v2 b1mix (soft tokens + mix) | **0.964 (-0.021, ~2 SE)** | 14,955 (1.8x) |

The two fixes together took pooled->full transfer from 0.08-0.16 to 0.96-0.98. b1mix's final CE
(0.63 vs the leaky iteration-1's 1e-5) confirms the shortcut is gone: compressed rows now demand
retrieval THROUGH the soft tokens.

## Iterations 3-5: the B1 anneal frontier at 6k context (all 488-ex full-attn eval, +-0.007)

| config (`--mix-*`) | ~avg uncompressed exposure | speed vs full | f1 |
|---|---|---|---|
| b1mix: 0.8->0 over 100% | 40% | 1.80x | 0.964 |
| b1bal: 0.8->0.05 over 75% | 32% | 1.95x | 0.952 |
| b1floor: 0.8->0.1 over 25% | 19% | 2.21x | 0.943 |
| b1fast: 0.8->0 over 25% | 10% (all early) | 3.42x | **0.692 — late-phase drift** |

Lessons: (1) annealing to ZERO full exposure causes late-phase drift away from full-attention
behavior — keep a floor; (2) total exposure buys f1 roughly monotonically; (3) at 6k context the
frontier is capped structurally: an anchor row costs only ~4x a compacted row, so any meaningful
exposure eats the speedup (pure-compressed ceiling 4.9x). Also: one stochastic SIGSEGV (rank 1,
step 487, flash-attn 2.8.2 padded-bwd) — retry with a new seed succeeded; launcher now propagates
rc so dependent evals don't run on crashed trainings.

## 32k head-to-head (2026-08-29, matched held-out set, 488 ex)

| arm | f1 | train wall-clock (2xH200) |
|---|---|---|
| full baseline (2 epochs) | **0.939 +-0.011** | ~3.7h (27s/step) |
| B1 anchor-0.05 | 0.620 +-0.022 | **42min (5.3x total, ~15x steady-state)** |

NOT parity at 32k with the thin 5%-anchor recipe (real 0.32 gap; the n<=700 regime is harder and
the anchor exposure was ~4x lower than the 6k parity recipe's). Eval traps hit on the way: the
ladder `rung_32768.jsonl` has ~1400 docs/example vs training's n<=700 -> f1 0.016 that "reads as
long-context collapse" (the documented eval-ID trap; the B1 ckpt scores 0.777 on n100, so the model
was healthy); comparisons must use the matched-generator held-out set (seed 4242).

**v6 (in flight): length-stratified anchoring.** Anchor with SHORT full-attention rows (the n100
6k shard mixed at ratio 0.3) instead of long ones: full-attn skill transfers across lengths
(0.777 above), anchor rows cost ~3s not 27s, and the implementation is free -- anchor fingerprints
miss the 32k sidecar (-> all docs real) and compaction drops their padding. Projected ~8-10x
wall-clock with continuous anchoring.

## v6 / v8 / aux-vs-noaux verdicts (2026-08-29 late): both no-long-anchor routes FAILED

| arm | regime | speed | f1 |
|---|---|---|---|
| v6: 30% SHORT (6k) anchor rows, 32k compressed | 32k held-out | ~10x | **0.482** (worse than 5% long anchors) |
| v8: pure aux-matching, no anchors | 32k held-out | ~15x steady | **0.037** |
| aux6k: pure aux-matching, no anchors | n100/6k | ~3x | **0.173** |
| noaux6k: pure compression control | n100/6k | ~3x | **0.196** |

Both v8/aux6k trained STABLY (CE 0.57-0.69, sane grads) -- the failure is not optimization. The
controlled 6k pair shows the aux attention-contribution matching loss (v1 impl: lambda 0.5, 16
answer-region queries/row, per-layer mass logsumexp + weighted-value MSE, detached targets, shadow
soft tokens with slot-visibility masking) buys ~NOTHING over the no-aux floor (0.173 vs 0.196).
Interpretation: slot fidelity is not the binding constraint -- the drift is in how the model
processes DENSE REAL token streams (softmax calibration over hundreds of real docs), which it
never practices when all rows are compressed; matching slot aggregates does not constrain that.
And v6 shows full-attention practice transfers POORLY across lengths (short anchors don't maintain
32k behavior). The one variable that tracks transfer across ALL 10 arms is the fraction (and
recency) of genuine full-attention training at the eval length: 40%->0.96-0.98, 19%->0.94,
5%->0.62, early-only->0.69/0.48, 0%->0.04-0.20.

## Output-equivalence probe (2026-08-30): the CO-DRIFT mechanism, measured

`debug/pooled_kv/output_equivalence_probe.py` (per-layer hidden cos + answer-logit KL between the
SAME model's real and compressed forwards, same rows, answer-prediction positions, 16 rows):

| ckpt (eval f1) | KL(full||comp) | top1 agree | L24 trough | L35 cos |
|---|---|---|---|---|
| full-32k baseline (0.939), identity projector | 0.93 | 0.65 | 0.74 | 0.81 |
| b1mix-6k (0.964, ANCHORED) | 0.59 | 0.78 | 0.72 | 0.87 |
| aux6k (0.173, aux only) | 0.55 | 0.76 | 0.76 | 0.95 |
| noaux6k (0.196, nothing) | 0.49 | 0.78 | 0.76 | 0.95 |

Three findings:
1. **The divergence concentrates at L16-L28** (trough ~0.74 at L24) in every profile; layers <=L12
   are nearly transparent to compression (cos >= 0.97). A distillation loss should target
   mid/late hidden states + output logits, not early layers.
2. **Forward-agreement does NOT track eval quality -- co-drift does.** The two FAILED no-anchor
   models have the BEST compressed<->real agreement (L35 cos 0.95): LM training on compressed
   inputs drags the full-attention pathway toward the compressed computation, so the two forwards
   agree on a jointly-degraded behavior. The anchored model (0.964) shows MORE divergence -- its
   full pathway stayed strong and only that pathway is what eval uses.
3. **The aux slot-matching loss changed nothing measurable** (f1 0.173 vs 0.196; agreement
   0.55 vs 0.49; similar CE) -- consistency between forwards arises from LM training alone, and
   slot fidelity is not the binding constraint.

## Paired consistency distillation (2026-08-30): the no-anchor matrix SOLVED at 6k

All arms no-anchor, no-mixing, gold_subsample keep, n100/6k, full-attn eval (488, +-0.007-0.02):

| arm | f1 |
|---|---|
| nothing (pure compression) | 0.196 |
| aux slot-matching | 0.173 |
| detach-soft-KV only | 0.246 |
| detach-soft-KV + aux | 0.210 |
| **paired distillation only (p=0.15)** | **0.917** |
| **detach + paired distillation** | **0.933** |
| (mixing-anchored reference / full) | (0.964 / 0.985) |

Mechanism (enable_pooled_soft_tokens distill_prob/distill_weight): with probability p a forward
runs BOTH passes -- the full pass takes LM gradient (protects the pathway from co-drift) and its
hidden states at the divergence layers (L16+, stride 4, + final) become detached targets for the
compressed pass at answer positions (relative MSE). The coin is seeded on a shared forward counter
(FSDP ranks must branch identically or collectives deadlock). 15% paired teachers do what 40%
practice rows did; detachment (static-KV slots) adds ~+0.016. 32k flagship (p=0.12,
q4b-pkv32-v11-detdist) in flight vs baseline 0.939.

## Mechanical slot-capacity study (2026-08-30): averaging can't, log-mass fits CAN (at G=1)

`debug/pooled_kv/mechanical_slot_fidelity.py` (no training; attention-op fidelity of per-doc
summary slots through Qwen3-4B, 8k + 32k contexts). Mid-layer doc-mass TV / top-doc agreement:

| slots per doc | 8k | 32k |
|---|---|---|
| G=1 mean+logL | .094 / .04 | .234 / .03 |
| G=8 kmeans clusters | .069 / .28 | .217 / .16 |
| ORACLE: 1 slot, per-head (k*, bias) lsq-fit to the doc's logsumexp mass | **.030 / .90** | **.057 / .88** |

(1) Pre-softmax token averaging is mechanically insufficient and k-means clustering converges too
slowly (Jensen gap on the convex logsumexp). (2) A SINGLE slot per doc whose per-head key+bias are
least-squares fit to the doc's true log-mass function is NEARLY sufficient (top-doc ~0.9, output
cos .96-.997). (3) Per-head biases are not static-KV-realizable, but train-time slots never ship:
the training attention op can inject arbitrary logit biases, so the oracle parametrization is
usable wherever slots exist only at training. The earlier aux loss targeted exactly these
quantities -- capacity was never the problem; co-drift was.

Composite design this motivates: v1-style attention (ALL tokens present -> no co-drift class) with
ONLINE oracle slots (closed-form lsq from each doc's own per-layer K/V, computable inside v1's
forward at negligible cost) to cut the attention quadratic term; FFN linear term via
zero-compute/null experts on filler tokens (AdaMoE / LongCat-Flash direction, deferred); memory
via compaction where the transfer question is settled.

Design conclusion: any equivalence objective must PROTECT the full-attention pathway from
co-drift. Next build: **paired consistency distillation** -- on a fraction p of rows, run the full
forward WITH LM gradient (keeps the pathway honest) and use it as a detached teacher for the
compressed forward on the same row (answer-logit KL + hidden matching at L16+). Each full-price
forward does double duty (practice + explicit alignment), so p should undercut the ~40% that pure
practice needed; p=0.10-0.15 prices to ~4.5-6x at 32k.

Infra: ALL of the day's SIGSEGVs (flash AND torch backends, 6k AND 32k, ws1 AND ws2) were mooney
node flakiness -- the "deterministic step-293" crash reproduced twice on mooney and vanished on
horton. Prefer horton when it has room.

## 32k results (2026-08-30): keep-policy was the second binding constraint

All arms 32k uniform-length train (1998 ex, 2 epochs), matched-generator held-out eval (488 ex,
full attention, MAXLEN 34816). Baseline: full-attn f1 **0.939** at ~4,850 TPS/device (3.7 h).

| arm | keep | distill | f1 | TPS/dev | speedup |
|---|---|---|---|---|---|
| v11-detdist | gold_subsample | p=0.12, answers only, w=1 | 0.662 | 27,151 | 5.6x |
| v12-detdist | gold_subsample | p=0.2, answers+96 free-pos, w=2 | 0.739 | 18,441 | ~3.8x |
| **v13-allgold** | **gold_plus_random n=24** | p=0.2, answers+96 free-pos, w=2 | **0.847** | ~16,700 | **~3.4x** |

- **gold_subsample made the training task partially unsolvable** (only 2/6 gold kept → the model
  can't learn to read what isn't there; compressed CE floor 0.66). Prasann caught this. With all
  gold + 24 drowning negatives kept, train CE fell 1.94 → **0.0024** and eval jumped +0.108.
- Denser distillation (v11→v12 at same keep) was worth +0.077 on its own.
- Remaining gap to baseline: 0.092 (f1 0.847 vs 0.939, SE ~±0.016). Suspects: (a) candidate
  narrowing — trained to rank among ~30 full docs, eval ranks among ~500; (b) residual co-drift at
  p=0.2. Round-3 arms answered this:

| arm | change vs v13 | f1 | TPS/dev | speedup |
|---|---|---|---|---|
| v14-neg64 | n_random 24→64 | 0.880 | 15,406 | ~3.2x |
| v15-distp35 | distill_prob 0.2→0.35 | 0.869 | 10,734 | ~2.2x |
| v16-neg128 | n_random 24→128 | 0.891 | 13,453 | ~2.8x |

**Candidate breadth is the productive axis** (24→64→128 negs: 0.847→0.880→0.891, nearly free in
speed); the co-drift axis is saturated at p=0.2 (v15: +0.022 for a third of the throughput). Gap
to baseline now 0.048. Round 4 (in flight): v17 per-row log-uniform breadth 16-256
(`--n-random-range`, new; teaches scale-invariant ranking so 500-candidate eval isn't OOD),
v18 n_random=256 (plateau probe), v19 n_random=128 + p=0.15 (speed reclaim).
⚠ v15 and v16 both SIGSEGV'd at step 364 on SNEETCHES (v15 also step 73 on mooney) — not purely
mooney hardware; possibly load-correlated. Resumes from step300 sailed past 364 both times, so not
data-deterministic either. 50-step saves + auto-resume watcher (3 tries) is the standing armor.

## HARD CONSTRAINT (2026-08-31): no full-attention forwards at 32k in the training method

Prasann: *"In general you're never allowed to do full attention on 32k, so don't try anything
involving that even if it helps."* Paired distillation (v17/v18/v19 were mid-flight) was
cancelled on this directive — its p-fraction of full forwards caps speedup at 1/p and defeats the
method at 128k/1M. The full-attention BASELINE (comparison) and full-attention EVAL (the transfer
target, inference-time) remain allowed; full attention *within kept docs* is fine. Round 5 is the
all-zero-full-attention slate:

| arm | config | distill | f1 | TPS/dev | speedup |
|---|---|---|---|---|---|
| v20-nodistill | n_random=128 | p=0 | 0.797 | ~34,600 | **~7.1x** |
| v21-rb-nodist | n_random-range 16-256 | p=0 | 0.857 | ~35,600 | **~7.3x** |
| **v22-n256-nodist** | **n_random=256** | p=0 | **0.916** | ~18,300 | ~3.8x |
| 1/3-baseline | full attn, 166 steps | — | 0.542 | ~4,850 | 1x (iso-budget ref) |

Dropping the distill term alone bought ~2.5x throughput at fixed keep breadth (v16 13.5k → v20
34.6k TPS/dev), confirming the 1/p ceiling analysis. **Answer: kept-doc breadth alone DOES
protect the pathway — and scales past what distillation ever reached.** v22 (256 negs, no
teacher) beats v16 (128 negs + p=0.2 distill, 0.891) and sits 0.023 from the 0.939 full baseline
(SE ~±0.013). The 1/3-budget full baseline (schedule fully annealed over its 166 steps, so fair)
collapses to 0.542 — compressed training dominates per FLOP, not just per second. Frontier:
0.797@7.1x / 0.857@7.3x / 0.916@3.8x; randomized breadth (v21) dominates fixed-128 (v20) at
equal speed. Since speedup at fixed ABSOLUTE breadth grows with context length (256 kept docs is
52% of a 32k row but ~13% of a 128k row), the frontier should improve at longer contexts.

## Oracle log-mass slot cache (2026-08-31): max KV fidelity without full-context forwards

Implemented (commit e35c0ea75): replace the projector's soft-token per-layer KV with OFFLINE
per-doc, per-layer, per-KV-head **oracle slots** — the mechanical-study form that reaches
top-doc ~0.9 where meanpool cannot — injected as static KV at slot columns.

- **Fit**: ridge lsq so `scale*q·k* + c ≈ logsumexp_t(scale*q·k_t)` over a stash of REAL
  answer-region queries derotated to a canonical frame and re-rotated to sampled relative offsets
  (log-uniform 256..30k); `v*` = mass-weighted doc value mean under the same queries. The scalar
  per-slot bias `c` (shared across heads, per layer) is STRUCTURAL: a bias-free linear slot
  cannot express the constant part of the log-mass — the toy fit goes from in-sample R² −7.9
  (no bias) to strongly positive (with bias). Two-stage shared-design solve keeps it one
  Cholesky per (layer, head) per chunk.
- **RoPE frames**: slots are stored in the doc-center frame (`R_{-c}` applied to captured
  post-RoPE K); at train time the attention layer re-rotates with its own RotaryEmbedding at the
  doc's actual center. Exactness of rotate/derotate/inject is unit-tested against the module
  (`src/test/nn/oracle_slot_test.py`, incl. a self-consistency test: derotated own-K injected
  back reproduces baseline output bitwise-close).
- **Builder** (`pooledkv/build_oracle_slot_cache.py` + `build_oracle_cache.sbatch`): one doc per
  row behind its example's 100-token preamble at true absolute positions (plain causal = doc
  attends preamble+itself; ~150-token rows — never O(T²)). contra32k_train = 868k occurrences /
  866k UNIQUE docs (no reuse, mean 45 tok) → cache ≈ 128 GB fp16 on node-local /data.
- **Runtime** (`--oracle-slot-cache DIR`): docs hashed (sha1/64 of the marker-inclusive span) at
  compaction, cache-hit slots override K/V at every layer (`soft_kv_override` per-block kwarg);
  the per-layer bias rides the position-causal SDPA path (equivalent math to sequence-causal on
  the compacted row). Misses fall back to the projector; 0 hits in 10 forwards raises (hash
  mismatch guard).
- Also constraint-compatible if this stalls: compressed teachers (pooled forward with a different
  keep draw), breadth curriculum. Distill-style anything at full context stays banned.

**RESULT (v23, 2026-08-31): oracle slots HURT — f1 0.809 vs v21's 0.857 at the same randomized
16-256 breadth, at half the throughput (~18k TPS/dev, bias-SDPA path + per-step slot gather).**
Training was healthy (CE 0.0030, 100% cache-hit, fit R² 0.965), so the slots did what they were
built to do — and it didn't matter. This CONFIRMS the output-equivalence probe's conclusion from
the failure side too: slot fidelity is not the binding constraint; kept-real-doc breadth is. Best
explanation for the regression: cached slots are frozen in the BASE model's K/V frame and go
stale as training drifts, while the projector's detached-but-live slots are recomputed from
current embeddings every forward and track the model. The fidelity axis is closed (don't revisit
with a mid-training cache refresh unless breadth saturates below parity — refresh costs ~90 min
per epoch and the probe says the ceiling isn't here). Round 6: v24 = n256 + role-gated FFN
(flexible-compute composition), v25 = randomized breadth 128-512 (v22's mean, v21's invariance).

## Flexible-compute FFN (role-gated): first capacity result

Implementation (commits 75fc194d0, ce540001a): context-doc tokens skip the full MLP from
``start_layer`` on — deterministic marker-based gate, identical at train/eval/prefill, no new
params, gather-based real-FLOP saving; ``--ffn-gate-start-layer`` on trainer + eval driver,
``FFN_GATE`` env on the eval sbatch (an arm MUST be scored with its training gate).

**v24 (n256 + gate from layer 4/36): train CE WALLS at ~0.95-0.97** (flat steps ~50-252;
healthy v22 was far below by then) — killed at step 252 without eval. Attention over doc tokens
that stopped getting FFN refinement at layer 4 cannot extract what the task needs: doc READING
requires early-mid FFN capacity, the same lesson as the 0.66 starved-keep floor but for compute
instead of visibility. Throughput was only +15% over v22 anyway (18.9k vs 16.4k TPS/dev — the
32/36-layer × 52%-token FFN skip is diluted by attention + gather overhead at this keep
fraction). v26 = gate from layer 12 (docs get 1/3 of the stack at full compute) queued; if that
walls too, the next shape is a LEARNED null-expert router (AdaMoE-style) that lets the model
protect the tokens it needs rather than a role-blind gate.

**v26 (gate from layer 12) result: trains (CE → 0.247, no wall) but eval f1 = 0.316** (scored
WITH the matching gate). Role-blind FFN gating is dead in both configurations: layer-4 can't
even fit the training data; layer-12 fits but collapses at 500-candidate eval. The FFN axis now
requires the learned-router shape (null/tiny experts with load balancing, router baked into the
base like the B1 projector) — parked as the next build.

## PARITY (2026-08-31): v25 = 0.927 vs baseline 0.939 — within noise, zero full attention

**v25 (--n-random-range 128,512, detach, p=0): f1 0.927 @ ~13.8k TPS/dev (~2.8x at 32k).**
Gap to the 0.939 full baseline is 0.012 with SE ~±0.012 — statistical parity under the hard
no-full-attention-at-32k constraint. Final frontier for the 32k campaign:

| arm | breadth | f1 | speedup @32k |
|---|---|---|---|
| v21 | rand 16-256 | 0.857 | ~7.3x |
| v22 | fixed 256 | 0.916 | ~3.8x |
| **v25** | **rand 128-512** | **0.927** | **~2.8x** |

Breadth buys accuracy, randomization buys scale-invariance, and BOTH are cheap at longer
contexts: the same 128-512-doc breadth is ~26% of a 64k corpus and ~13% of 128k, so the v25
recipe projects to ~4-5x at 64k and ~7-10x at 128k at (presumably) the same parity — the next
campaign. Train data for 64k already exists
(`/scratch/users/prasann/corpus-reasoning/data/contradiction_train_pubmed_both_ctx64k.jsonl`).

### Local-cluster scheduling (why jobs pended; now fixed)

`pick_pooledkv_node.sh` picks among the three staged nodes (horton/mooney/sneetches all hold
bases+shards on /data since job 3486712) by free + preemptible GPUs and emits matching
`-w/-p/-q` args. Traps burned: (1) horton is partition **berkeleynlp**, whose preempting QOS is
`preemptive_high_sewonm` — plain `preemptive_high` is jsteinhardt-only; (2) that QOS has a shared
**group** GPU cap (`QOSGrpGRES`) that other users can exhaust — fall back to
jsteinhardt nodes; (3) in zsh, `sbatch $ARGS` passes the picker output as ONE arg
("Invalid node name") — use `${=ARGS}`; (4) the launcher header defaults to `--qos=preemptive`
(can't preempt): always pass the QOS explicitly. mooney also SIGSEGVs ~1/300 steps on 32k
(hardware; both attn backends) — `--save-interval 100` + auto-resume is standing armor.

## Next: the 32k rung (where >=10x is structurally possible)

At 32k an anchor row costs ~20x a compacted one, so speedup ~= C_full/(x*C_full + (1-x)*C_comp):
x=0.10 -> ~7x, x=0.05 -> ~10x, IF low exposure suffices for transfer there. Data: rebuilt
`uniform_8k_32k_native.jsonl` (2000 ex, n 175-700 docs, gold pairs; the qwen3-vs-qwen35 128k
lineage) -> `contra32k_train` shard + sidecar; eval = the trusted 500-row
`ctc_suite_staged/eval_rungs/contradiction/rung_32768.jsonl`.

## Idea

For corpus-reasoning SFT, most of the context is negative documents the model should mostly ignore.
Train with ~90% of context documents' K/V *compressed*: for every query outside the document, the
doc's per-token KV entries are replaced by ONE slot = (mean K, mean V) over the doc, post-RoPE, with
`+log(doc_len)` added to that slot's logit. Gold docs + a small random subset of negatives keep real
per-token KV. At inference, run **ordinary full attention** — nothing is exported or special-cased —
and rely on the training distribution being close enough for direct transfer.

## The math (why this exact formulation)

- A slot with key `k̄`, value `v̄`, and logit bias `log L` contributes `L·exp(q·k̄)·v̄` to the softmax
  numerator and `L·exp(q·k̄)` to the denominator — **identical** to attending `L` copies of `(k̄, v̄)`.
  So the train-time computation IS standard full causal attention over a perturbed sequence in which
  each pooled doc's KV entries are all replaced by their average. The *function* is the test-time
  function; only the inputs for pooled docs are lower-entropy. That is the whole in-distribution /
  transfer argument, and it is exact, not approximate (unit-tested to atol 1e-5).
- The approximation to the *unperturbed* corpus is `Σᵢ exp(q·kᵢ) ≈ L·exp(q·k̄)` — good exactly when
  attention over the doc is diffuse, i.e. the regime a negative doc should be in. By Jensen the true
  mass is always ≥ the pooled proxy, so test-time negatives are strictly *heavier* distractors than
  their train-time stand-ins. The kept-negative subset is what still teaches token-level rejection.
- **Sum-pooling the key is degenerate**: `exp(q·Σkᵢ)` is the *product* of per-token exponentials —
  it explodes with doc length and corresponds to no attention over tokens. "Sum" done right is
  mean-K + the `log L` bias (that's the sum of attention *mass*), which is the default
  (`pooled_len_bias=True`, ablation flag `--pooled-no-len-bias`).
- Pooling is **post-RoPE** (mean of the actual KV entries), which is what the exact-equivalence
  statement needs. High-frequency RoPE components partially cancel in the mean; low-frequency
  (long-range) position info survives at doc granularity.
- Queries *inside* a pooled doc attend their own doc's REAL tokens causally (the slot only becomes
  visible after the doc ends), so there is no future leakage and every token's LM loss stays
  well-defined.

## Honest expectations (what this does and does not buy)

- **FLOPs**: only the attention O(T²) term shrinks (→ ~`T·(keep·T + n_docs + doc_len)`); QKV/MLP
  FLOPs are unchanged because every token still runs the full network to produce the K/V that get
  pooled. So end-to-end speedup is small at 2–8k, meaningful at 32k (attention ≈ half the step), and
  large at 128k+. Realized wall-clock requires the FlexAttention block-sparse path (engages at
  T≥8192 on CUDA); the dense-mask fallback is exact but not faster.
- **Memory**: activations are dominated by per-token hidden states across layers, which are
  unchanged; the win is the attention-mask/score memory and (with budget AC) cheaper recompute. Do
  not expect "90% memory saved". The variant that would save that (process pooled docs with a
  cheaper/partial forward) is a different, bigger change.
- **Transfer risks to watch**: (1) pooled keys have distinct statistics (smaller norm, smoothed) —
  the model can learn "smooth key ⇒ ignorable" as a shortcut, making train-time negatives easier to
  reject than real ones; the kept-negative fraction and comparison against a matched dense baseline
  is the control. (2) Gold-doc retrieval trains on real tokens, but *contrast* against near-miss
  negatives is only trained on the kept subset.

## Measured wall-clock (H200, 2026-08-27, fwd+bwd bf16, B=1, single layer, 16 heads/64 dim)

`debug/pooled_kv/flex_parity_and_timing.py` + `flex_breakdown.py`. Parity flex-vs-dense is exact
(max diff 6e-7 fp32). Timing at T=16384, 125 docs of 128 tokens, vs plain causal cuDNN SDPA 9.7ms:

| arm | ms | note |
|---|---|---|
| pooled flex, docs straddle 128-blocks, keep=0.1 | 46.7 | **4.8x SLOWER** than full causal |
| same, no `score_mod` (len-bias off) | 44.2 | bias costs only ~5% |
| same, docs 128-block-ALIGNED | 10.7 | ≈ parity with full causal |
| aligned + keep=0.02 | 9.3 | slightly faster than full causal |
| dense-mask fallback | 52.5 | flex beats dense 1.2-4.9x, as designed |

Reading: the FLOP reduction is real but FlexAttention's per-block throughput is ~4-5x below cuDNN's
fused causal kernel, so at keep=0.1 the two roughly cancel — **block alignment of documents decides
everything** (a straddling doc contaminates two block-columns for every row). `flex_block_size`
64/32 does NOT work around this: `create_block_mask` + compiled flex hits an inductor
LoweringException at non-default block sizes on torch 2.9.1/this mask_mod and falls back to dense.
Real CTC docs (hundreds of tokens) straddle proportionally less than this worst-case synthetic, so
expect real-shard wall-clock between the aligned and straddled rows — i.e. **roughly full-attention
speed, not yet faster**, at 16-32k. The variant is therefore currently a *transfer* experiment at
matched-ish wall-clock; realizing the compute win needs either much longer context, low keep_prob,
block-aligned packing at data level, or the two-call flash decomposition (varlen block-diag
within-doc + all-queries-over-compacted-KV).

## KV-predictability probe (2026-08-28): the learned summarizer is feasible

`debug/pooled_kv/kv_predictability_probe.py`; metrics in `debug/pooled_kv/probe_results/`; raw
feature/target tensors (for richer predictors) in `/data/prasann/pooledkv_probe/` on horton.
Setup: Qwen3-4B base, real contradiction PubMed claims, marker-wrapped contexts at 2k/8k/32k
(2119/4261/8535 doc occurrences), 40 probe docs each planted in 5 contexts per rung. Predict each
doc's per-layer MEAN hidden state / post-RoPE K / V from its mean input embedding (+ length,
position scalars) with ridge; doc-identity-disjoint splits.

Findings:
1. **Targets are content-determined, not context-determined.** Same-doc-across-contexts
   consistency (the ceiling for ANY content-only predictor): R^2 0.85-0.98 for center-frame K and
   V at every layer and rung. A doc's mean KV barely cares what surrounds it.
2. **Even LINEAR probes recover half the variance**: R^2 ~0.4-0.7 (mid/late layers), V cosine
   0.81-0.89 vs 0.62-0.77 for the predict-the-mean baseline. A trained MLP summarizer has a wide
   corridor between the linear floor and the ~0.9 ceiling.
3. **Predictability IMPROVES with context length** (e.g. L35 hidden R^2 .475 -> .536 -> .575 going
   2k -> 8k -> 32k; ceilings rise too). Long contexts dilute cross-doc contamination. The scheme
   gets easier exactly where it is needed. (Caveat: sample count also grows with rung.)
4. **Raw post-RoPE K is unpredictable and that's fine**: k_raw R^2 0.1-0.35 with ceiling 0.25-0.65
   (position scrambles it), while the doc-center DEROTATED frame restores ceiling ~0.93 and R^2
   0.55-0.69. So g_theta must predict a center-frame K and RoPE-rotate it at the doc's position at
   runtime — confirmed as a hard design requirement, not a nicety.
5. Mid-layer hidden states (L18-21) are the most context-sensitive spot (ceiling dips to ~0.6 at
   2k, recovers to ~0.89 at 32k) — if Level-2 hidden-state pooling ever misbehaves, look there.

## Summarizer fit + attention-operation fidelity (2026-08-28)

`debug/pooled_kv/fit_summarizer_and_fidelity.py`; JSON in `debug/pooled_kv/probe_results/
fidelity_metrics.json`; g_theta weights in `/data/prasann/pooledkv_probe/summarizer.pt`.
g_theta = 2-layer MLP trunk + per-layer heads on (mean input embed, len, pos-frac) -> per-layer
(center-frame mean K, mean V); trained on the probe tensors (11.5k train occurrences), ~10s on H200.
Fit R^2 ~0.45-0.62 mid/late layers — an MLP barely beats ridge on vector R^2.

**Fidelity** (the metric that matters): real trailing-question queries against held-out-doc
contexts; EVERY doc pooled (worst case — no gold keep-set); compare full attention vs slot variants
per layer: output cosine / doc-mass total-variation / top-doc agreement. Findings:

1. **The learned summarizer is free on the fidelity metric**: mlp and ridge match exact meanpool
   to ~0.01 cos and ~0.005 TV at every layer/rung. The (mean -> predicted) gap is negligible next
   to (real -> mean). Whatever exact-mean training can do, summarizer training can too.
2. **Chunk size is nearly free**: 35 -> 139 -> 586-token chunks moves L18@32k only
   0.922/0.175 -> 0.911/0.186 (cos/TV), while compression per slot grows 16x. Top-doc agreement
   IMPROVES with bigger chunks (fewer, more distinct competitors). The per-slot compression ratio
   can be pushed hard.
3. Layer profile: cosine 0.87-0.99 almost everywhere; best at L35 (0.99). Weak spots: **L3**
   (0.82 -> 0.65 going 2k -> 32k; early local/syntactic heads want token-level structure) and
   mid-layer doc-mass allocation (L18-L27 TV 0.11 -> 0.28 at 32k; top-doc argmax changes almost
   always — partly argmax instability over ~900 diffuse docs, but this is where retrieval heads
   live). Mitigation available in the class already: `full_attention_layers` to exempt the worst
   few layers (~10% of the savings per exempted layer).
4. Context: this is ZERO-SHOT fidelity on the base model with 100% pooling — the training scheme
   keeps gold + sampled negatives real and lets the model adapt, so the operating condition is
   strictly easier. Hand-computed attention validated against HF eager probs (mean diff 1.9e-05).

⚠ Two traps burned into this script: (a) an inserted helper stole `fidelity_for_rung`'s
`@torch.no_grad()` decorator -> grad-enabled 32k forward retains ~135GiB of activations and OOMs
an H200 (the silu/MLP OOM signature); (b) `output_attentions` needs the eager flip, which is not
reliably reversible -> `validate_attention_math` must run LAST.

## Cheaper rungs (not yet implemented)

- **Level 1 — detach the backward for pooled docs**: forward unchanged (equivalence stays exact),
  pooled slots + pooled-doc contributions detached → backward skips ~90% of tokens (~2/3 of their
  network FLOPs + activation memory). Infra exists (`gold_grad_mask.detach_kv` keyed on the same
  keep set). ⚠ this IS the gold-grad O(1)-backward hypothesis, which collapsed on leak-free v2 —
  an arm to test, not assume.
- **Level 2 — pool the hidden states at layer ℓ**: replace each pooled doc's tokens by one mean
  hidden state after layer ℓ and carry a single "doc token" through the remaining layers → pooled
  docs cost O(1)/layer above ℓ (the actually-dramatic FLOP/memory win, ℓ=0 = embed-only). Breaks
  the exact equivalence (`layer(mean) != mean(layer)`); needs mid-stack sequence shortening in
  `Transformer.forward`. Natural ladder on ℓ, with the current exact version as ℓ=L.

## Implementation map

- `src/olmo_core/nn/attention/pooled_doc_kv.py` — `PooledDocKVAttention`
  (subclasses `DocumentChunkedAttention` for the `chunk_ids` plumbing; own mask/pool logic; flex +
  dense dual backend), `PooledDocKeepHolder`, `make_fingerprint_keep_docs_fn` (gold sidecar →
  `select_keep_docs` policies from `gold_grad_mask.py`), `install_pooled_doc_keep`.
- Config: `AttentionType.pooled_doc_kv`; `AttentionConfig.pooled_keep_prob / pooled_keep_seed /
  pooled_len_bias`; composes with `full_attention_layers` and `flex_block_size`.
- Keep-set: gold + `n_random` negatives via the fingerprint sidecar hook (same sidecar format as
  gold-grad/gold-hop); without the hook, a deterministic seeded random `keep_prob` fraction
  (gold-blind control arm).
- Trainer: `train_ctc_suite.py --variant pooledkv` (+ `--pooled-gold-sidecar`, `--pooled-n-random`,
  `--pooled-keep-mode`, `--pooled-keep-prob`, `--pooled-no-len-bias`). Requires the padded no-pack
  data path (fingerprints assume one example per row); forces `--no-compile`. Data shards are the
  SAME marker-wrapped shards as the chunked arms — no rebuild.
- Tests: `src/test/nn/attention/pooled_doc_kv_test.py` (13, CPU). Load-bearing:
  `test_exact_equivalence_to_perturbed_full_attention`.
- GPU check: `debug/pooled_kv/flex_parity_and_timing.py` (flex-vs-dense parity + rough timings).

## First experiment sketch

`train_ctc_suite.py` on contradiction (gold sidecar exists via `build_gold_sidecar_from_shard.py`),
4B, seq 40960: arms = full (baseline), pooledkv+sidecar (`gold_plus_random`, `n_random≈2`),
pooledkv random_only (gold-blind control), optionally `--pooled-no-len-bias`. Score all arms with
the standard FULL-attention eval path (that's the point — no special eval). Compare vs the dense
baseline at matched tokens; also worth logging achieved step time vs the full arm at 40960.
