# Pooled-doc-KV: agent handoff (state as of 2026-08-31)

This is the **pick-up-here doc** for the pooled-doc-KV / compressed-training effort. The full
chronological research log (mechanisms, probes, dead ends, math) is
`records/pooled-doc-kv-attention.md` — read its PARITY, HARD CONSTRAINT, oracle-slot, and FFN-gate
sections before changing direction. This doc is the operational summary: what's proven, what's
dead, what's next, and the exact commands.

## Where we are

**Statistical parity with full-attention training achieved at 32k, with zero full-context
forwards anywhere in training.** v25 (softtoken, random breadth 128–512) scores **f1 0.927** on
the 488-example held-out contradiction eval vs the full-attention baseline's **0.939**
(SE ±0.012), at ~2.8× training throughput. Speedup at fixed absolute breadth grows with context
length, so the ≥10× goal now runs through **longer contexts**, not more 32k tuning.

### HARD CONSTRAINT (user-set, standing)

> "In general you're never allowed to do full attention on 32k, so don't try anything involving
> that even if it helps."

No full-context training forwards: no distillation teachers, no full-row mixing, no anneal floors.
Allowed: the full-attention *baseline* arm (comparison), full-attention *eval* (the transfer
target), and full attention *within kept docs*. Also standing: no reliance on length
generalization, no mixing with real/other data. See memory `no-full-attention-32k-training`.

## Results frontier (32k, 488-ex held-out contradiction, full-attn eval, no-cot)

| arm | recipe | f1 | train speedup |
|---|---|---|---|
| baseline | full attention, 3-epoch budget | 0.939 | 1× (~4,850 TPS/dev) |
| baseline @ ⅓ budget | full attention, matched-FLOP-ish (annealed, fair) | 0.542 | — |
| v20 | softtoken n_random=128 | 0.797 | ~7.1× |
| v21 | softtoken rand-breadth 16–256 | 0.857 | ~7.3× |
| v22 | softtoken n_random=256 | 0.916 | ~3.8× |
| **v25** | **softtoken rand-breadth 128–512** | **0.927** | **~2.8× (13.8k TPS/dev)** |
| v23 | v21 + oracle log-mass slot cache | 0.809 | ~7× |
| v24 | v22 + role-gated FFN from layer 4 | CE-wall 0.95, killed | — |
| v26 | v22 + role-gated FFN from layer 12 | 0.316 (scored with gate) | — |

Older context (6k rung, distill era, co-drift probe, keep-policy bug) is in the main record.

## The v25 recipe (the thing to reproduce / extend)

B1 soft-token compaction, trained purely compressed:

1. **Base**: marker-repaired Qwen3-4B CPT base **with baked zero-init projector keys**
   (`q4b-dense-cpt-fixmark-b1`). The softtoken arm refuses to run on the un-baked base
   (`bake_projector_into_base.py` creates it).
2. **Compaction**: each pooled context doc collapses to ONE placeholder token
   (LANDMARK_TOKEN_ID 151669) at the doc-center *original* position; its input embedding is
   overwritten with P(mean of the doc's input embeddings), P(x)=x+MLP(x) zero-init. Original
   `position_ids` are preserved, so RoPE geometry matches full attention. The whole stack runs on
   the short compacted sequence.
3. **Keep policy** `gold_plus_random`: all gold docs + `n_random` negatives keep real tokens.
   `--n-random-range 128,512` draws breadth per-(fingerprint,call) log-uniformly —
   **breadth is the entire protection mechanism** (scale-invariant ranking).
4. **`--detach-soft-kv`**: per-layer K/V detach at slot columns (static-KV; the anti-co-drift
   ingredient). With detach + aux weight 0, the projector is effectively frozen at identity —
   that is the *working* configuration, don't "fix" it.
5. **No distillation** (`--distill-prob 0.0`) — banned by the constraint, and dropping it bought
   ~2.5× throughput at fixed breadth.
6. **Eval**: export → score with plain `--variant full`. Zero-shot transfer, nothing special.

### Exact launch (sneetches, 2×H200)

```bash
cd src/scripts/train/memexpress/pooledkv
DATA=/data/prasann/pooledkv_exp/contra32k_train
EXTRA="--data $DATA --batch-tokens 262144 --save-interval 50 --ac-mode full --epochs 2 \
  --keep-mode gold_plus_random --n-random-range 128,512 --attn-backend torch \
  --detach-soft-kv --distill-prob 0.0"
NGPU=2 SEQ_LEN=32768
sbatch -w sneetches -p jsteinhardt -q preemptive_high --gres=gpu:H200:2 \
  --export=ALL,ARM=softtoken,RUN=q4b-pkv32-vNN-name,EXTRA,NGPU,SEQ_LEN,DATA \
  run_q4b_pooledkv_horton.sbatch
```

### Exact eval

```bash
RUNS=q4b-pkv32-vNN-name EVAL_JSONL=/data/prasann/pooledkv_exp/contra32k_heldout_eval.jsonl \
MAXLEN=34816 sbatch -w sneetches -p jsteinhardt -q preemptive_high --gres=gpu:H200:2 \
  --export=ALL,RUNS,EVAL_JSONL,MAXLEN,FFN_GATE eval_q4b_pooledkv_contra.sbatch
# f1: grep -a "f1=" /scratch/users/prasann/pooledkv_logs/eval_<jid>.log
# FFN_GATE must match training (-1 = off). MAXLEN < prompt silently zeroes f1.
```

## Closed axes — do NOT reopen without new evidence

- **KV-slot fidelity (oracle slots), v23 = 0.809 < v21 0.857.** A 128GB offline cache of
  per-doc/layer/kv-head lsq-fit log-mass slots (holdout R² 0.965 vs meanpool −0.78, 100% hit rate,
  healthy training) *hurt*: cached slots freeze in the base frame and go stale as training drifts,
  while the live (even identity) projector tracks the model. Matches the output-equivalence
  probe's verdict that slot fidelity was never the binding constraint. Infra all works
  (`src/olmo_core/nn/oracle_slot.py`, builder, 7 tests, `--oracle-slot-cache DIR`) — only worth
  revisiting with **mid-training cache refresh** (~90 min/epoch) and only if breadth saturates
  below parity at longer contexts.
- **Role-blind FFN gating (deterministic skip for context-doc tokens).** Gate@4: model can't even
  fit train data (CE wall 0.95 — docs become unreadable). Gate@12: fits (CE 0.247) but evals
  0.316 *with the matching gate*. Doc tokens need MLP compute that a role-blind rule can't
  allocate. The machinery (`src/olmo_core/nn/role_gated_ffn.py`, `enable_role_gated_ffn`,
  `--ffn-gate-start-layer`, tests) is sound and reusable for the learned version.
- **Distillation / any full-forward anchoring** — banned by the hard constraint (and v22 0.916
  already beats the best banned-distill arm v16 0.891 anyway).
- **Aux hidden-matching without anchoring** (0.173 vs 0.196 no-aux), **short-anchor mixing**
  (0.482), **gold_subsample keep** (makes task unsolvable, CE floor 0.66) — all dead, see main
  record.

## Open threads (in priority order)

### 1. Length-scaling campaign: 64k → 128k (the path to ≥10×)

Fixed absolute breadth ⇒ speedup grows with context: 128–512 kept docs ≈ 52% of a 32k corpus,
~26% of 64k, ~13% of 128k → the v25 recipe projects to **~4–5× at 64k, ~7–10× at 128k** at
(presumably) the same parity. Data already exists:

- train: `/scratch/users/prasann/corpus-reasoning/data/contradiction_train_pubmed_both_ctx64k.jsonl`
- eval: `/scratch/users/prasann/corpus-reasoning/data/contradiction_eval_pubmed_both_ctx64k.jsonl`
- (a `ladder64k/` dir sits alongside)

Steps: convert train JSONL via `convert_longctx_tasks_to_sft.py` (dense marker shard + gold
sidecar, qboth, no-cot — mirror how `contra32k_train` was built), stage shard to node `/data`,
run the v25 recipe at SEQ_LEN=65536 (revisit `--batch-tokens`/NGPU/ac-mode for memory; dense CP
RoPE fix `dense-cp-packing-rope-fix` is relevant if CP is needed), plus a full-attention 64k
baseline arm for the comparison, then the same eval sbatch with a 64k eval JSONL and MAXLEN
sized above the prompt. Watch for: full baseline at 64k may itself need CP/more GPUs — the
*baseline* is allowed full attention.

### 2. Learned flexible-compute FFN (null-expert router)

The 10–100×-FFN thrust. Role-blind gating is dead → the router must be **learned**:
AdaMoE-style null/tiny expert alongside the real MLP, per-token router, load-balancing/target
sparsity loss so most long-context tokens take the cheap path. Router params must be **baked
into the base** (same pattern as `bake_projector_into_base.py` / the B1 projector) so exports
stay loadable. Reuse the bound-forward shadowing pattern from `role_gated_ffn.py` (state-dict
keys unchanged). Success bar: large fraction of context tokens on the null path with eval f1
still at parity, scored with the router active (the FFN_GATE lesson: score with what you train).

### 3. Records/memory upkeep

`records/pooled-doc-kv-attention.md` + memory `pooled-doc-kv-task` are current through v26.
Keep both updated as results land; report evals to Prasann ASAP.

## Asset map

On **sneetches** `/data/prasann/pooledkv_exp/` (read remotely via `/net/sneetches/...`, AUDIT
ONLY — jobs must use `/data`):

- `contra32k_train/` — 32k train shard + `gold_fingerprints.json` sidecar (qboth, no-cot)
- `contra32k_heldout_eval.jsonl` — 488-ex matched-generator held-out eval
- `q4b-dense-cpt-fixmark{,-b1}/` — marker-repaired CPT base, ± baked projector keys
- `runs/q4b-pkv32-v14…v26/` — all exported checkpoints (`model_and_optim/` + `config.json`)
- `oracle_cache_contra32k/` — the 128GB oracle slot cache (867k docs; closed axis, kept)
- `eval_results/` — result JSONs + per-example outputs (mirrored to
  `/scratch/users/prasann/pooledkv_logs/`)

Horton has the older 6k-era copies (`contra_n100_*`). Base + bases also duplicated per node.

## Code map (branch `prasann/landmark`)

- `src/olmo_core/nn/transformer/model.py` — `enable_pooled_soft_tokens` (compaction, soft
  inject, oracle hookup, `soft_kv_override_layers`), `enable_role_gated_ffn`
- `src/olmo_core/nn/attention/__init__.py` — `soft_kv_override` kwarg (slot injection + bias)
- `src/olmo_core/nn/oracle_slot.py` — oracle slot fit/derotation/cache (commit e35c0ea75)
- `src/olmo_core/nn/role_gated_ffn.py` — role-gated FFN shadowing (commit 75fc194d0)
- `src/scripts/train/memexpress/pooledkv/` — trainer, launchers, eval sbatch, cache builder,
  `pick_pooledkv_node.sh` (family README there; its data paths describe the 6k era)
- Tests: `src/test/nn/{oracle_slot,role_gated_ffn,pooled_soft_token}_test.py` (all pass, CPU)
- Key commits this round: e35c0ea75, 75fc194d0, ce540001a, bfb6cf3c7, 6ac265594

## Ops traps (each of these burned real time)

1. **sneetches SIGSEGV ~1/300 steps** (exitcode −11, steps 360–480 typical). Mitigation:
   `--save-interval 50` + auto-resume chain (≤3 resubmits, same RUN → trainer silently resumes
   from the save folder). Silent auto-resume also means: **never reuse a RUN name for a fresh
   config.**
2. **Log mirror lag is 30s** — a "dead" train log on `/scratch` may just be stale. Liveness =
   log mtime on the *node* (`/net/<node>/data/prasann/joblogs/...`), not squeue; the slurm
   controller flaps ("Unable to contact slurm controller"), so chains must retry sbatch until
   parsable and never treat a controller error as job death.
3. **Eval MAXLEN must exceed the prompt** (34816 for 32k) or f1 silently reads 0.000.
4. **FFN-gate arms must be evaled with `FFN_GATE=<train layer>`** — the eval sbatch defaults −1.
5. Launcher QOS default is non-preempting `preemptive` — always submit with
   `-q preemptive_high` (and `-p jsteinhardt` for mooney/sneetches). 8-GPU/user cap per QOS.
6. NFS discipline per `local_cluster.md`: node-local env + `/data` for logs/ckpts/caches;
   `/scratch` and `/net` are the same ~5 MB/s layer.
7. `sacct -j <id> --format=SubmitLine%400` recovers exact submit commands of past jobs.
