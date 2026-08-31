# pooledkv — train-time KV/sequence compression, full-attention inference

The pooled-doc experiment family. **Start with `records/pooled-doc-kv-handoff.md`** (operational
state, exact commands, closed axes); the full research log is `records/pooled-doc-kv-attention.md`.

Train with most context documents compressed — gold + random negatives keep real tokens — then
evaluate the checkpoint with **ordinary full attention** (zero-shot transfer). Under the standing
constraint of ZERO full-context training forwards at 32k, the softtoken arm reached statistical
parity: **v25 (rand-breadth 128–512) f1 0.927 vs full baseline 0.939 (SE ±0.012) @ ~2.8×**.
Breadth (`--n-random-range lo,hi`, per-call log-uniform) is the axis that mattered; oracle slot
caching and role-blind FFN gating are both closed (see the handoff doc before reopening either).

Three arms in one trainer/launcher:

* **`full`** — plain causal baseline (accuracy + wall-clock reference; the only arm allowed
  full-context forwards).
* **`pooledkv` / v1** — `AttentionType.pooled_doc_kv`: per-layer exact-mean KV slots with a
  `+log(doc_len)` logit bias (exactly full attention over a perturbed corpus). Matched
  wall-clock; the clean transfer test.
* **`softtoken` / B1** — `Transformer.enable_pooled_soft_tokens`: each pooled doc collapses to
  ONE projected soft token at its doc-center original position; the whole stack runs on the
  compacted sequence with original `position_ids`. The speedup arm; winning recipe adds
  `--keep-mode gold_plus_random --n-random-range 128,512 --detach-soft-kv --distill-prob 0.0`.
  Requires the `-b1` baked base (`bake_projector_into_base.py`).

| file | purpose |
|---|---|
| `Qwen3-4B-pooledkv-contra-n100-local.py` | three-arm local trainer (`--oracle-slot-cache`, `--ffn-gate-start-layer` also live here) |
| `run_q4b_pooledkv_horton.sbatch` | train launcher (`ARM=`, `RUN=`, `EXTRA=`, `DATA=`, `SEQ_LEN=`); despite the name it runs on any staged node via `sbatch -w` |
| `eval_q4b_pooledkv_contra.sbatch` | full-attention held-out eval (`RUNS=`, `EVAL_JSONL=`, `MAXLEN=` > prompt or f1 silently 0, `FFN_GATE=` must match training) |
| `pick_pooledkv_node.sh` | emits `-w/-p/-q` for the free staged node (horton/mooney/sneetches) |
| `bake_projector_into_base.py` | write a base copy with zero-init projector keys (required by softtoken) |
| `build_oracle_slot_cache.py` + `build_oracle_cache.sbatch` | offline oracle log-mass slot cache builder (closed axis; kept for a possible mid-training-refresh revival) |

Assets (32k era) on **sneetches** `/data/prasann/pooledkv_exp/`: `contra32k_train/` (shard +
`gold_fingerprints.json` sidecar, qboth, no-cot), `contra32k_heldout_eval.jsonl` (488 ex),
`q4b-dense-cpt-fixmark{,-b1}` (marker-repaired CPT base ± projector keys), `runs/` (v14–v26
exports), `oracle_cache_contra32k/` (128GB). Horton keeps the older 6k-era `contra_n100_*`
shards and base copies.

Eval: every arm scored identically — plain `--variant full`, 488 held-out, no-cot; the exported
`config.json` records the PLAIN architecture (extra `pooled_projector.*` keys are ignored by
plain loads). Ops traps (sneetches SIGSEGV + resume chains, RUN-name reuse = silent resume, log
mirror lag, QOS defaults) are listed at the bottom of the handoff doc.
