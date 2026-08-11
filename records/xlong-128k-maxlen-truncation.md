# The 128k xlong rung was silently truncating prompts (pre-`18d129f67` evals)

**Status: diagnosed, fixed in the runner, 69 poisoned rows deleted from results-hub, re-runs queued.**

Any `--xlong` eval launched from a commit that predates **`18d129f67`** (2026-07-29 16:53 PDT)
capped prompts at `rung_label + 1024`. At the 128k rung that is `MAX_LENGTH=132096`, which is
**below the length the 128k rung actually builds**. The prompt tail — where the question and the
output-format instruction live — was cut off, and the run scored ~0.000 at `parse_rate 1.0`.

It reads exactly like a long-context capability cliff. It is not one.

## The one-line check

For any xlong result, pull the job log and read the cap the runner printed:

```bash
beaker experiment logs <job-id> | grep -oE '\[xlong\] RUNGS=[^ ]+ MAX_LENGTH=[0-9]+'
```

* `MAX_LENGTH=132096` at a 128k rung → **poisoned**, discard the contra/nq/outlier numbers.
* `MAX_LENGTH=146227` → fine (`label x 1.10 + 2048`, the current formula).

From a results-hub row, `git_commit` is the discriminator — no log needed:

```bash
git merge-base --is-ancestor 18d129f67 <git_commit> && echo "has the fix" || echo "PRE-FIX"
```

## Why it hits contradiction hardest, and nq barely at all

The prompt budget is `MAX_LENGTH - max_new_tokens`, so **the task with the largest generation
budget gets the smallest prompt budget**. Built prompts run **+0.4% to +3.3% over the rung label**
(the doc count is calibrated from a *median*, plus the instruction/query/marker wrap), so a task
needs at least ~0.4% headroom to survive. At the 128k rung under the 132,096 cap:

| task | `max_new_tokens` | prompt budget | headroom over the 131,072 label | outcome |
|---|---|---|---|---|
| contradiction | 512 | 131,584 | **+512 (0.39%)** — below the *minimum* overage | dead: 0.001 |
| rerank | 256 | 131,840 | +768 (0.59%) | (re-run post-fix; healthy) |
| outlier | 200 | 131,896 | +824 (0.63%) | 3.3x depressed: 0.039 vs 0.130 |
| oolong | 200 | 131,896 | +824 (0.63%) | (re-run post-fix; healthy) |
| nq | 64 | 132,032 | +960 (0.73%) | survives: 0.800 vs 0.830 |

contra is doubly exposed: it has the biggest `--contra-max-new-tokens` (512) *and* the heaviest
over-label tail (its prompts are query-dominated). The damage ordering tracks headroom exactly.

## How it produced a fake architecture result

The dense and landmark arms of the 256k pair were evaluated **three days apart, across the fix**:

| arm | job start | commit | cap |
|---|---|---|---|
| landmark, landmark-ep1 | 2026-07-29 17:20Z | `190225286` (07-28) | 132,096 **pre-fix** |
| dense | 2026-07-31 16:22Z | `2ab20701b` (07-31) | 146,227 fixed |

The landmark job launched **27 minutes after** the fix landed, from a commit a day older. Result:

```
contradiction @128k     landmark 0.001   dense 0.214
```

which reads as "the landmark architecture collapses at 128k" with a healthy dense control sitting
next to it. It is entirely the cap. Note what *isn't* affected: the same job ran rungs 2k-64k under
the same 132,096 cap, and at those labels the headroom is enormous — so **only the 128k rows are
poisoned**, and the 64k gap (contra: landmark 0.234 vs dense 0.339) is real and still unexplained.

## The table-wide tell

14 rows across all of results-hub were contradiction@128k on a pre-fix commit, spanning many
different models and architectures:

```
n=14   min 0.0000   max 0.0056   median 0.0000   —   14 of 14 at or below 0.01
```

Fourteen unrelated models, every one at zero, while nq at the same rung is fine. No architecture
does that.

## What was done

* **Runner**: the cap formula (`label x 1.10 + 2048`) landed in `18d129f67`;
  `eval_lc_native.py` also re-raises an undersized `--max-length` by the same rule rather than
  trusting the caller, so a stale launcher can no longer under-cap.
* **results-hub**: 69 rows deleted (`contradiction` 14, `contra` 9, `outlier` 23, `nq` 23) across
  22 checkpoints — every pre-fix 128k row for those tasks. Better a gap than a wrong number.
* **Re-runs**: `debug/requeue/requeue_128k_prefix_evals.sh` regenerates exactly the deleted cells
  at the fixed cap (`--xlong --xlong-only --xlong-rungs 128k`).

## Not covered

* **RULER @128k** (24 pre-fix rows) goes through a different harness path — the ladder invocation
  passes `--skip-ruler` — so this diagnosis does not apply to it. Unaudited.
* **64k rows from 64k-ONLY jobs.** A pre-fix job whose `XLONG_RUNGS` was `64k` alone would have
  capped at `65536 + 1024 = 66560`, giving contra `+512` headroom — the *same* trap one rung down.
  The rows audited here came from combined `64k,128k` jobs (cap 132,096), where 64k had ample room.
  Any 64k-only pre-fix contra row is suspect and was not checked.

## Related

* `POSSIBLE_BUG_SFT_DATA.md` — the other reason dense-vs-landmark deltas need care.
* `contradiction-data-and-base-hygiene.md` — the running list of silently-wrong contradiction inputs.
* The separate `bs=2` left-padding bug (OLMo-core `91a3adc74`): 340 further results-hub rows were
  deleted for it. Same lesson, different mechanism — the recorded `decoding_hparams_other` said
  `batch_size=1` while the job ran at 2, so **provenance fields record what was *passed*, not what
  *ran***. Reconstruct from `eval_command` + `git_commit`, never from the hparam columns.
