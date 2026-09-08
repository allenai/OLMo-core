# Soft-KV slot-construction study — handoff (state as of 2026-09-08 ~14:30 PDT)

**UPDATE 14:30 — PARITY REACHED ON ALL FOUR TASKS (eval side).** Contradiction and oolong close
once each document's *header* (the exact-match tokens: `Claim N:`, `Date: … || User: … ||
Instance:`) stays real and only the body is pooled (`--prefix-real stopK`), or — contradiction
only — once the document right after each gold claim is real (`goldnbr1`, leak-free `+runs`).
Cheapest: contradiction header+keep 1/36 ≈ 4.8x, neighbour runs+1/36 ≈ 11x; oolong header+keep 1/3
≈ 1.4x (its header is over half the line). Wired into training as
`--st-header-stop-id 25 --st-header-stop-count {1|3}` (commit aa0b7db2a). Full tables in
`records/pooled-doc-kv-attention.md` (sections dated 2026-09-08 afternoon). Beaker confirmations on
the eval-bundle rows: 01M21B3PRZGWQNJJTP7SN5SPT4 (contra) 01M21B4F4VAER7E8MJZC1F0G05 (oolong)
01M21B581CNK4WTQN8EEDM7B06 (nq runs) 01M21B612ZEK7TFXC8ERARDW19 (outlier runs). New probe knobs:
`--prefix-real`, `--slot-pos`, `--gdn-nowrite-only`, policies `goldnbrK[+runs]`/`goldleftK`/`goldrightK`;
`analyze_slot_rows.py` for per-row / per-token-role breakdowns (probe JSONs now carry `per_row`).
Local probe outputs: `/net/sneetches/data/prasann/slot_probe/v{2,3,4}_*.{log,json}`; launchers
`/scratch/users/prasann/slot_probe/run_sneetches_v{2,3,4}.sbatch`. Sneetches rows ≠ Beaker rows
(oolong full 0.507 vs 0.466, contradiction 0.071 vs 0.042) — never compare across sources.

Settled by the afternoon runs: keep policies (content-based ≈ random per token; first/last
catastrophic), GDN no-write (only helps contradiction at low keep, hurts elsewhere), slot RoPE
position (no effect), slot bias at keep 0 (no effect), Qwen3-4B oolong probe (checkpoint is
context-blind, uninformative).

**Next (training):** launch `softtoken` arms with header-real — contradiction 56M keep 1/36 and
1/12, oolong 80M keep 1/3 — against the dense ladder (`fs35s4bkv*` recipe, `--st-keep-frac`), and
add a `gold_plus_random_runs` keep mode to `make_fingerprint_keep_docs_fn` for the 11x arm.


Prasann's framing (2026-09-08): find the cheapest eval-time construction that reproduces a
dense-trained model's answer loss to **near zero gap** on all four tasks; only then take it to
training. Iterate fast (minutes per number), keep every attempt as a row in the tracker.

## Where everything is

- **Tracker (artifact):** https://claude.ai/code/artifact/3fd44312-4007-449d-bad9-2b15e12e4ee6 —
  rendered from `results/pooled_kv/slot_tracker.html` by `debug/pooled_kv/make_slot_tracker.py`,
  which reads `results/pooled_kv/slot_probe_results.csv` written by
  `debug/pooled_kv/collect_slot_results.py` (parses the LAST table in every probe log; Beaker
  experiment ids + local log paths are listed in its `SOURCES`). Refresh = run collect, run
  make, republish the HTML (same file path keeps the URL).
- **Probes** (`debug/pooled_kv/`), all share `eval_side_slot_probe.py` helpers; all run as
  1-GPU Beaker jobs via `python debug/flop_scaling/beaker_bench_launch.py --script <probe> --extra "<args>" [--env PROBE_FAMILY=qwen3]`
  (jupiter, urgent, weka mounted; rows are tokenized in-job from the weka eval JSONLs with the
  marker converter; ~10 min per 24-row sweep) or locally on sneetches (see below):
  - `eval_side_slot_probe.py` — mean-embedding soft token (the training construction), slot
    logit bias sweep (`+log L`, `+log L + c`, constants), keep policies
    (`--policies random,overlap,length,short,first,last,hardneg,hardneg+rand`), `--gdn-nowrite`
    (slots attention-only: GDN layers get `block_keep`, no state write / conv leak),
    `--keeps`, `--biases none`, `--ckpt/--jsonl/--shard` overrides for local runs.
  - `oracle_meankv_probe.py` — dense forward's per-layer mean K/V injected at slot columns ± log L.
  - `slot_ceiling_probe.py` — fitted log-mass slot (k*, v*, c) and G ∈ {1,2,4,8} slots/doc.
  - `pooledkv_eval_probe.py` — every token kept, mean K/V slots on the attention layers only
    (GDN intact). Result looked wrong (keep 1/12 == keep 0; worse than compaction) — unresolved,
    suspect the pooled block-mask cache; not a trainable candidate anyway.
  - `PROBE_FAMILY=qwen3` env switches to the pure-attention Qwen3-4B dense ladder checkpoints.
- **Local fast loop (jsteinhardt, sneetches):** model-only bf16 dense checkpoints in
  `/data/prasann/dense_ckpts/{tsl-full-contradiction-s56M-4b-…, tsl-full-oolong-s80M-4b-…, lmx-full-nmixs48M-nq-4b-…, lmx-full-mixs160M-4b-…}/model_and_optim`
  (staged weka→S3 by `debug/pooled_kv/transfer_dense_ckpts_gantry.sh`, pulled by
  `/scratch/users/prasann/slot_probe/pull_ckpts_sneetches*.sbatch`); tokenized held-out rows
  under `/scratch/users/prasann/slot_probe/{contra_iid_32768,contra_iid_8192,oolong_32768,oolong_8192}`
  (nq/outlier eval JSONLs are weka-only, not staged locally); launchers
  `/scratch/users/prasann/slot_probe/run_sneetches*.sbatch`; outputs `/data/prasann/slot_probe/*.log|json`
  (read via `/net/sneetches/...`). Node-local env `/data/prasann/conda/envs/corpus-reasoning-olmo`.
  Routing rule (memory `job-routing-policy`): small jobs jsteinhardt → berkeleynlp → Beaker.
- **Record:** `records/pooled-doc-kv-attention.md` (sections dated 2026-09-08) has every table.
- **Throughput:** `debug/pooled_kv/bench_softtoken_throughput.py` — soft-token training is as
  fast as dense on a row of the compacted length (3.5x at keep 1/3, 11.9x at 1/12, flash
  backend fine); the ladder's slow kv arms were the padded launch recipe, not the method.

## Findings (answer-token CE, 24 held-out rows, 32k, dense Qwen3.5-4B ladder checkpoints)

Full attention: contradiction 0.042, nq 0.086, outlier 1.206, oolong 0.466.

1. **Representation is exhausted.** Default soft token (mean input embedding, no bias) equals
   the oracle mean-K/V slot, the fitted log-mass slot, and 8 slots/doc within noise at every
   keep on every task. The fitted slot fixes the slot's mass, not its value; the loss needs both.
2. **Slot logit bias is task-dependent and never a win:** retrieval tasks want ≤ 0, oolong wants
   ≥ log L + 2. Training-side check (`fs35s4bkvbias-*`): +log L costs contradiction 0.13–0.34
   f1, does nothing on oolong. Oolong is flat in keep fraction at training (0.665 at 1/12 =
   1/6 = 1/3 vs dense 0.723).
3. **Gold-index bug (fixed 2026-09-08):** the sidecar subtracted 1 for every task; nq/outlier are
   0-indexed, so their TRUE gold doc was pooled. Corrected, default soft token:
   nq 0.070 / 0.057 / 0.057 at keep 1/3 / 1/12 / gold-only (full 0.086); outlier 1.398 / 1.201 /
   1.201 (full 1.206). **nq and outlier are at parity with gold-only** (2–4% of tokens). More
   real random distractors HURT on both (mixture effect: real hard negatives get ~3x the
   attention share the dense model was calibrated for). Contradiction goes the other way
   (gold-only 0.43, 1/3 0.109); oolong (no gold) 0.61 at 1/3. Memory
   `gold-sidecar-index-base-bug`; nq/outlier kv17/kv33 grid arms trained with wrong gold.
4. **GDN:** in compaction the soft token is an ordinary token for the 24 GDN layers (writes the
   state, leaks through the conv); detach only covers attention K/V. The `--gdn-nowrite`
   variant (launched, see below) makes slots attention-only.
5. Outlier caveat: gold-only at training = "real doc is the answer" shortcut (id-answer task);
   the eval parity is honest but the training recipe needs gold hidden among real docs.

## In flight (2026-09-08 13:00) — collect with `collect_slot_results.py` (ids already listed)

- Keep-policy sweep (fixed; the first one was all-random): contradiction 01M2183MZJVTJYV7KBD2J0ZXWE,
  oolong 01M2184TV0VNN7133Y93WQC05C, nq 01M2185V4XWRQWH27XAQ0ZKSX4, outlier 01M2187549JW0XYXQAP3P4V46N.
  2-row local check on contradiction: random 0.070 beats overlap 0.164 / first 0.363 / last 0.428.
- nq gold + hard negatives (`hardneg`, `hardneg+rand`): 01M218SJE5KETF99DAND6GR5EV.
- `--gdn-nowrite` on all four tasks: 01M218XKAHWSBCBJ34AC07MD7T 01M218YDV9CGGRAHY4VNQC1H7Q
  01M218Z99QR0FQVS9RYEYD1PN8 01M21903R27BPJZF2KRPVXPZF3.
- Qwen3-4B (pure attention) probes, contradiction + oolong: slot 01M214G6XMJ62A91BA5ZT4H1PR (done),
  oracle 01M217TQZECQXZGBTFJZ712QMZ, ceiling 01M217VR74HJXMAN3TCN0QJS4E, oolong slot
  01M217WJGYH236Q6X16JPB1NA8, oracle 01M217XGKD91751NSBNNMN3J84, ceiling 01M217YAYERS22875JPFWY4PMQ
  (the collector still lists the FAILED first ids for five of these — replace them).
- Qwen3-4B dense nq/outlier training (pure-attention upper bound): orchestrator log
  `debug/flop_scaling/orchestrate_q3s4bdense2nqo.log`, runs `fs35q3s4bdense2nqo-nq-dense-s48M`
  (done, eval 01M21832DK5QH005BQ1CZD5259) and `-outlier-dense-s160M` (training); checkpoints land
  under `ctc_suite/ckpts/<run>` on weka — add them to `CKPT_Q3` in `eval_side_slot_probe.py`.
- Training arm oolong kvl33 (last of `fs35s4bkvbias-*`), orchestrator `orchestrate_s4bkvbias`.
- sneetches: `ceiling-sn` job (contradiction/oolong 32k+8k ceiling) may still be queued.

## Next ideas (not started)

- Contradiction: why does gold-only lose 0.38 when nq loses nothing — claim numbering / pair
  contrast? Try keeping the gold pairs' neighbours real; try keeping the "Claim N:" prefix of
  pooled docs real (2 tokens each).
- Oolong: keep-count instead of keep-fraction; more slots per LINE group; gdn-nowrite result.
- Mixture rebalancing: down-weight real non-gold documents' logits (attn_bias on their columns)
  instead of biasing slots — targets the effect in finding 3.
- If gdn-nowrite helps: it is trainable as-is (pass block_keep=~slot to GDN blocks in the
  soft-token forward; one kwarg).

## Traps hit today

- zsh: `set -- $var` does not word-split; `--include=*.py` globs error; use arrays/quotes.
- Background polling loops get killed by the harness; use sbatch/detached jobs and read logs.
- Editing with `str.replace` without assert silently no-ops (the keep-policy patch) — assert.
- `_compact_pooled_soft_tokens` returns 3 values; `model(x, logits_to_keep=cols[None])` works on
  the compacted path (40 s → 1 s per forward).
- Beaker launcher `beaker_bench_launch.py` needs `allow_dirty=True` (tree is always dirty) and
  ships the PUSHED commit: push before launching.
