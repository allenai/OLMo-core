# Possible bug: dense and landmark SFT arms do not train on the same data

**Status: OPEN, not yet a confirmed bug — two asymmetries found by inspection, one exactly
quantified, one only partly.** Audit date 2026-08-05, over all 5-task SFT launchers in
`src/scripts/train/sft/amanda-landmark/` and `src/scripts/train/memexpress/sft_5task/`.

Every "dense vs. landmark" comparison at 32k differs in the data, not just the architecture. There
are two independent causes.

---

## A. Task upsampling differs (config-level, deliberate, easy to see)

Every **dense** script that packs at 32768 with `LongDocStrategy.exclude` carries
`_W = {contra: 2.9, rerank: 1.5, outlier: 1.5, nq: 1.0, oolong: 1.3}`.
Every **landmark/compressive** script carries `_W = {contra: 2.0, ..., oolong: 1.0}`.
`rerank`/`outlier`/`nq` are identical everywhere; `max_repetition_factor=8.0` is universal.

Realized fractions within the 5-task group (multiply by `FIVE_TASK_FRAC=0.75` for the dolci25 arms):

| | contra | rerank | outlier | nq | oolong |
|---|---|---|---|---|---|
| dense (2.9/1.3, ΣW=8.2) | 0.354 | 0.183 | 0.183 | 0.122 | 0.159 |
| landmark (2.0/1.0, ΣW=7.0) | 0.286 | 0.214 | 0.214 | 0.143 | 0.143 |
| **delta (dense − landmark)** | **+6.8pp** | −3.1pp | −3.1pp | −2.1pp | +1.6pp |

### Why it exists

Not an oversight — it is a compensation. At 32768 the dense packer drops ~18.3% of contradiction
docs and ~12.4% of oolong docs (`single_task_ladders_v2` scan), and the dropped ones are the
longest, so ≈31% of contra tokens and ≈23% of oolong tokens. `2.0/0.69≈2.9` and `1.0/0.77≈1.3`
restore the *token* share. The landmark packer at 40960 drops ~0% and so needs no correction.

**Consequence: the specified ratios and the realized ratios cannot both match.** Equalizing `_W`
makes the configs identical and the realized token mixes differ; keeping it makes the realized
mixes roughly match and the configs differ. Today we have the latter, undocumented at the
comparison sites.

### Mismatched pairs

| Dense (2.9/1.3) | Landmark counterpart(s) (2.0/1.0) | Other differences |
|---|---|---|
| `memexpress/sft_5task/Qwen3-4B-dense-5task-32k-nocpt-SFT.py` | `memexpress/sft_5task/Qwen3-4B-compressive-5task-32k-nocpt-SFT.py`, `.../Qwen3-4B-fast-landmark-5task-32k-nocpt-SFT.py`, and `amanda-landmark/`'s `blocklocal`, `gqa-grouped`, `partialrope`, `sharedvec` | 32768/10700 steps vs 40960/8550 |
| `memexpress/sft_5task/Qwen3-4B-dense-5task-32k-nocpt-fixnq-SFT.py` | `memexpress/sft_5task/Qwen3-4B-compressive-5task-32k-nocpt-fixnq-SFT.py` | **also root**: dense `single_task_ladders_v2`, landmark `cptmix_data_ladder40k` |
| `amanda-landmark/Qwen3-4B-dense-5task-dolci25-32k-nocpt-SFT.py` | `amanda-landmark/Qwen3-4B-compressive-gate-temp-5task-dolci25-32k-nocpt-SFT.py`, `amanda-landmark/Qwen3-4B-compressive-block64-fixeddata-5task-dolci25-32k-nocpt-SFT.py` | 32768/10700 vs 40960/8550 |

Dense scripts with 2.9/1.3 and **no landmark counterpart at all**:
`amanda-landmark/Qwen3-4B-dense-5task-dolci50-32k-nocpt-SFT.py`,
`amanda-landmark/Qwen3-4B-dense-5task-32k-nocpt-SFT-longer.py` (21400 steps).

### Pairs already matched on weights

`dense-cptmix-5task-32k` vs `compressive-cptmix-5task-32k` / `fast-landmark-cptmix-5task-32k`
(dense uses `ConcatAndChunk`, not packing); `dense-cptmix-5task-64k` vs
`fast-landmark-cptmix-5task-64k`; the Qwen3.5 `xlong5-dolci25-256k` pair. All 2.0/1.0.
**Matched weights are not matched data — see B.**

---

## B. The packers produce different epochs from the same documents

Measured at prep on the Qwen3.5 256k pair, **identical `_W`, identical roots, identical 2.4B-token
document mixture**:

- `Qwen3.5-4B-dense-xlong5-dolci25-256k-SFT.py` → **8,971 instances** (560 steps ≈ 1.00 epoch)
- `Qwen3.5-4B-fast-compressive-landmark-xlong5-dolci25-256k-SFT.py` → **10,145 instances**
  (560 steps ≈ 0.88 epoch)

**+13.1%**, consistent with the ~13% seen at 32k. Both arms are token-matched at 2.35B model
tokens, so the landmark arm sees a *smaller fraction of the corpus* for the same compute.

Three contributing mechanisms, in increasing size:

1. **Landmark tokens occupy window slots — exactly +1.59%.** `MEM_FREQ=63`, `BLOCK_SIZE=64`,
   `CONTENT_CAPACITY = 258048` of `SEQUENCE_LENGTH = 262144`. `262144/258048 = 1.0159`. Accounts
   for ~143 of the 1,174 extra instances.
2. **Per-document block padding — small, concentrated in the Dolci side.**
   `LandmarkPackingInstanceSource._landmark_len()`
   (`src/olmo_core/data/composable/landmark_packing_instance_source.py:157`) ceils each document's
   content to a multiple of `mem_freq` *before* appending landmarks: ~31 wasted tokens per
   document. Negligible for a 100k-token contra doc; not negligible for Dolci (25% of the mix, docs
   a few thousand tokens).
3. **Next-fit greedy vs Best-Fit-Decreasing — dominant.** `_build_plan()` (same file, ~line 163) is
   strictly sequential: when a document doesn't fit the current window it *closes that window* and
   never returns to it, so a window holding 200k that meets a 100k document closes with 62k wasted.
   The dense arm's `PackingInstanceSourceConfig` is Best-Fit-Decreasing (sorts longest-first, places
   into the tightest open bin) — near-optimal on a heavy-tailed length distribution.

**Not** a drop-rate effect, and not in the direction one would guess: the landmark threshold is
*stricter* (drops content > 258048; dense's `LongDocStrategy.exclude` cuts at 262144), so the
landmark arm sees slightly fewer documents and still emits 13% more instances.

Only (1) is exact arithmetic. The split between (2) and (3) is inferred from reading the two
algorithms and has not been measured. Cheap way to pin it down: a CPU-node prep of the landmark
source with the landmark cost neutralized, which isolates the packer term.

---

## What to do

Fixing A alone does not make the arms comparable; B survives it. Three options, best first:

1. **Filter both sides to docs ≤ 32768** (`LongDocStrategy.exclude` on dense, an equivalent
   max-doc-length cut on the landmark source) and set both to 2.0/1.0. Identical document set,
   identical weights, no compensation needed. Cost: the landmark arm loses its long docs.
2. **Run dense at 65536** (40960 is not a power of two, which `PackingInstanceSource` requires), so
   neither side drops anything; 2.0/1.0 both. Cost: memory, and ~5350 steps for token parity.
3. Keep the current compensated setup and document at every comparison site that the match is
   approximate. Cost: nothing, fixes nothing.

None of these touch B. Closing B needs either BFD packing in `LandmarkPackingInstanceSource` or
epoch-matching (not token-matching) the two arms — decide which, because token-matched and
epoch-matched are ~13% apart.

## Related

- `records/README.md` for the index. Memories: `sft-task-data-root-split` (a *third*, separate
  confound: `cptmix_data_ladder40k` vs `single_task_ladders_v2` moves oolong@32k by ~0.24),
  `block-sweep-nq-data-mismatch` (a *fourth*: NQ hard-negative regime),
  `landmark-packing-inflates-epoch-steps`.
- A run can be matched on any one of these four axes and not the others. Check `_W`, `DATA_ROOT`,
  `NQ_DATA_ROOT` and the instance-source class before trusting any dense-vs-landmark delta.
