# Running jobs on the local (Berkeley) slurm cluster

How we train and eval OLMo-core models on the Berkeley H200 nodes, **without** AI2 infra (no weka,
no Beaker). This is the local counterpart to the Beaker flow in `beaker.md`. Reference launchers live
in the `src/scripts/train/memexpress/` hub (see its README) — local launchers are the `*local*.sbatch` files in each family folder; a good canonical example is
`src/scripts/train/memexpress/attn_explore/run_q06b_dense_contra_n20_local_mooney.sbatch`.

## Cluster map

| Node | GPUs | Partition | Notes |
|---|---|---|---|
| `hermione` | none | — | Login/head node. 6 cores, ~7.5 GB RAM. Edit + submit only; heavy imports OOM. |
| `horton` | 8×H200 | `berkeleynlp` | QOS `preemptive_high_sewonm`. |
| `mooney`, `cubbins`, `sneetches` | 8×H200 each | `jsteinhardt` (`--account=site`) | Main training/eval nodes. All have a fast env on node-local `/data`. |
| `mcfuzz` | 8×H200 | `jsteinhardt` | Usable but NO `/data/prasann` yet (admin must create) → falls back to slow NFS env. |
| `lorax` | H200 | `berkeleynlp` | Frequently draining (`sinfo -R`). Rarely used. |

**QOS / preemption (jsteinhardt), lowest → highest:** `normal` < `preemptive` (no GPU cap) <
`preemptive_high` (8 GPU/user cap). Higher preempts (requeues) lower.
**On the `berkeleynlp` partition (horton/lorax)** the same rules apply but our high QOS is
`preemptive_high_sewonm` with a **4 GPU/user cap** — default to it there just like
`preemptive_high` on jsteinhardt, preempting lower-QOS jobs freely.
**DIRECTIVE: default to `preemptive_high` and preempt freely** — don't wait for idle GPUs or
downsize to fit; only jobs at equal/higher QOS (incl. reserved `preemptive_high_<lab>`) are off
limits. Use plain `preemptive` only when running more than 8 GPUs of sweep arms in parallel (it has
no cap but can itself get preempted — give those jobs a retry wrapper).

Standard sbatch header (jsteinhardt):

```bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney            # pin the node: /data is per-node
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/data/prasann/joblogs/<name>_%j.log   # NEVER on /accounts or /scratch (see below)
```

## Filesystems — the rule that explains most gotchas

- `/accounts` (home, this repo) and `/scratch` — **shared NFS, slow**. `/scratch` writes ≈ 5 MB/s;
  imports from the NFS conda env pay a ~5–60 s tax. Fine for code and few-KB artifacts.
- `/data` — **node-local ZFS, fast, NOT shared** (mooney's `/data` ≠ cubbins's `/data`). Reachable
  cross-node via `/net/<host>/data/...` — **at NFS speed, and for auditing only.**
- `/net/<node>/...` — every node's local disk, readable from anywhere. Nodes: `balrog`, `cubbins`,
  `feanor`, `horton`, `lorax`, `mcfuzz`, `mooney`, `rainbowquartz`, `saruman`, `shadowfax`,
  `smaug`, `smokyquartz`, `sneetches`, `sunstone`, `thidwick`. Good for "does this checkpoint
  exist", "what's in that job log", "what schema does that JSONL have" without an `srun` — keep it
  light. **`/net` IS the slow NFS layer**: a job on horton must read `/data/...`, never
  `/net/horton/data/...`, or it recreates the deadlock below against its own local disk. Recursive
  `find` across `/net` is slow enough to time out; go straight to the directory.

**Rule: for multi-rank GPU jobs, NOTHING job-I/O touches NFS — logs, work dir, data, checkpoints
all go on the target node's `/data`.** Concurrent NFS appends/locks deadlock ranks in
`nfs_wait_bit_killable` before step 1 (the classic "hangs after `Starting epoch 1...`, no loss
line" symptom). Corollaries:

- `#SBATCH --output=` → `/data/prasann/joblogs/` (mkdir it once per node).
- `export PYTHONWARNINGS=ignore` (8 ranks flooding one log file re-creates the stall).
- Stage input shards `/scratch → /data` with `cp -u` at job start (they're small); use a fresh
  per-job work dir `cache-$SLURM_JOB_ID` on `/data` (stale NFS locks from killed jobs poison reused
  work dirs).
- Save checkpoints to `/data`, and **eval them in place on the same node** (`--nodelist=<node>`,
  point at the ckpt path directly). Never copy multi-GB blobs to `/scratch` (10+ min each).

## Environment

`corpus-reasoning-olmo` is the DEFAULT env (torch 2.9.1+cu128, **flash-attn 2.8.2 pinned** —
2.8.3 has a varlen-backward SIGSEGV regression).

🚨 **EVERY env exists twice: a slow NFS master and fast node-local clones. In a job, ALWAYS use the
node-local path.** This is the single most expensive mistake on this cluster, because the slow copy
does not fail — it *works*, just after a long stall that is indistinguishable from a hung job.

| | path | `import transformers` |
|---|---|---|
| NFS master | `/scratch/users/prasann/conda/envs/<env>/bin/python` | ~60 s … **18 min** for an olmo_core import |
| node-local clone | `/data/prasann/conda/envs/<env>/bin/python` | **3 s** |

Clones exist on horton, mooney, cubbins, sneetches. Measured 2026-08-12 on mooney: the *same* env,
same Python 3.12, importing `cached_path`+`transformers`+`datasets`+`olmo_core` — **3.3 s
node-local vs ~18 min over NFS**. Two consecutive tokenize jobs were killed as "hung" before the
interpreter path was the thing checked; diagnose with `ps -o stat,wchan,%cpu` — `Dl` +
`rpc_wait_bit_killable` + ~1 % CPU is this and nothing else.

Put a guard in the launcher rather than trusting yourself to notice:

```bash
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
case "$PY" in /data/*) ;; *) echo "!!! interpreter not node-local"; exit 1;; esac
```

⚠ **Check for the node-local clone with a single glob.** `ls -d /net/mooney/data/*/conda/envs/*
/net/mooney/data/*venv*` in zsh aborts the WHOLE command when *any* one pattern matches nothing —
so an env that is present reports as absent. Glob one pattern per command, or use `setopt
nonomatch`.

⚠ **It is NOT one env for everything.** Three exist and they are not interchangeable — picking wrong
fails 15+ minutes in, after the work is done, with a bare `ModuleNotFoundError`. Pick by what the
script imports:

| env | python | has | use for |
|---|---|---|---|
| `corpus-reasoning-olmo` | 3.12 | `cached_path`, transformers, numpy, **gantry** | anything importing **olmo_core** (all `convert_*` shard converters, since olmo_core pulls `cached_path`), and every gantry/beaker launch |
| `corpus-reasoning-eval` | 3.11 | transformers, **datasets**, **openai**, **google-genai**, `corpus_reasoning` (editable) | data **generation** (`generate_*_data.py`, `build_v2_*_ladder.py`) — needs datasets/LLM clients, does NOT need olmo_core. **Lacks `cached_path`, so it cannot run the converters.** |
| `/data/prasann/ctc_vllm_venv` | 3.11 | vllm 0.25.1, openai, transformers 5.14 | vLLM serving/eval on a node. **No `datasets`.** Needs `CUDA_HOME=/usr/local/cuda-12.8` + `PATH=$CUDA_HOME/bin:$PATH` or the flashinfer/GDN JIT kills the engine. |

Rule of thumb: **generate → `-eval`; tokenize/convert/launch → `-olmo`; serve → `ctc_vllm_venv`.**
`gantry` exists ONLY in `-olmo`; it is not on the login-node PATH by default.

⚠ **The node-local repo copies are PARTIAL.** `/data/<user>/repo/OLMo-core` on mooney has only some
of `src/scripts/` (it had `eval/` but no `data/` at all), so a job referencing a script that exists
in the real checkout dies with `No such file or directory` after the queue wait. Sync the whole
subtree you need before submitting, don't copy files one at a time.

**All the shell-level conventions live in one master file — source it, don't copy-paste:**

```bash
source /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src/scripts/local_env.sh
```

It sets `REPO`, puts the fast env on `PATH` (direct PATH — `conda activate` hangs minutes on NFS
lock contention), `PYTHONPATH=$REPO/src`, the offline flags, `PYTHONWARNINGS=ignore`,
`WANDB_API_KEY`/`WANDB_FLAG`, a random `MASTER_PORT`, and defines `fresh_workdir ROOT`
(fresh per-job `/data` work dir). Pre-set a var before sourcing to override a default (e.g.
`HF_HUB_OFFLINE=0` to allow a one-time model download). The one thing it CANNOT cover is the
`#SBATCH` header (slurm parses that before the script runs) — partition/qos/gres and
`--output=/data/prasann/joblogs/...` stay in each launcher.
`run_q06b_dense_contra_n20_local_mooney.sbatch` is the reference retrofit; older launchers still
inline these conventions and can migrate as they're touched.

Background on the settings it applies:

- The env's olmo_core is an **editable install of THIS repo** (`.../projects/OLMo-core/src`) — all
  5 copies (the `/scratch` master + the 4 `/data` clones) were re-pointed on 2026-07-13, so a plain
  `import olmo_core` runs the working tree, and `PYTHONPATH=$REPO/src` is a harmless safety net.
  (Historical trap, in case a stale env copy resurfaces: they used to point at a frozen upstream
  clone at `/scratch/users/prasann/OLMo-core` — deleted 2026-07-13, it contained nothing not
  already in this repo's history — so forgetting `PYTHONPATH` silently ran months-old code. Check
  with `python -c 'import olmo_core; print(olmo_core.__file__)'`.)
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` — serve tokenizers/models from the local HF cache and
  never touch huggingface.co. Without them every `from_pretrained` does hub freshness checks, and
  compute-node egress is blocked/slow, so those requests sit in silent multi-minute retry stalls
  (× ranks × dataloader workers). Requires a warm cache: download new models once from the login
  node, or `HF_HUB_OFFLINE=0` before sourcing.

For one-off Python on the head node, call the interpreter by absolute path (skips the 1.4 s
`conda activate`): `/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python`.
CPU tests: `PYTHONPATH=$PWD/src $PY -m pytest src/test/...` (GPU tests auto-skip on hermione).

## Training pipeline (the weka-free recipe)

Beaker SFT scripts read data + base checkpoint from weka, which is not mounted here. Local runs
replicate the three inputs:

1. **Data shards.** Convert JSONL → tokenized SFT shards locally (CPU, seconds):
   `src/scripts/data/convert_longctx_tasks_to_sft.py --task <task> --input-jsonl <corpus-reasoning
   data> --out-dir /scratch/users/prasann/longctx_sft_qwen/<name>`. Output lands on `/scratch`
   (shared, so any node can stage it to its `/data`).
2. **Base checkpoint → olmo-core distcp** (one-time per base, needs a GPU):
   `src/corpus_reasoning/train/convert_hf_to_olmo.py` → `/data/prasann/olmo_ckpts/<base>` on the
   training node. Two traps:
   - `--base-checkpoint` / `load_path` must point at the **`model_and_optim` subdir**, not its
     parent — otherwise the trainer silently trains **from scratch** (tell: first-step CE ≈ 12
     instead of ~1–2).
   - **Repair the marker embeddings first** for any document-chunked/landmark run:
     `src/scripts/data/fix_marker_embeddings.py` on the base, train from the repaired copy
     (see CLAUDE.md and `records/document-chunked-marker-embeddings.md`).
3. **Standalone torchrun launcher.** A `*-SFT-local.py` script builds the configs and calls
   `trainer.fit()` directly with local paths (mirrors `internal/experiment.py::train`), wrapped by
   an sbatch file. Every launcher takes env overrides:

```bash
# Smoke first, then full:
sbatch --export=ALL,MAX_STEPS=10,NGPU=2 src/scripts/train/memexpress/attn_explore/run_q06b_dense_contra_n20_local_mooney.sbatch
sbatch src/scripts/train/memexpress/attn_explore/run_q06b_dense_contra_n20_local_mooney.sbatch
```

## Eval pipeline

Evals run as sbatch jobs **pinned to the node that holds the checkpoint**, reading it in place from
`/data` (e.g. `eval_q4b_attn_explore_cubbins.sbatch`, `eval_q06b_contra_n20_native.sbatch`). The
harness is `src/corpus_reasoning/eval/eval_lc_native*.py` (native olmo_core backend, no HF/vLLM).
Only the few-KB result JSONs go to shared storage. Reporting rules (eval_size, ≥500 examples,
error bars) are in CLAUDE.md.

## Monitoring and hygiene

- Watch: `squeue -u prasann`; who's on my nodes + QOS:
  `squeue -w mooney,cubbins,sneetches -o '%.10i %.10u %.16q %.8T %b %N'`.
- Cancel with `scancel <jobid>`.
- From inside an existing srun allocation, a nested `srun --nodelist=<other-node>` fails (inherits
  `SLURM_*`). **Always `sbatch`** for work on other nodes.
- The shell is zsh: no word-splitting — wrap `for`-loops over command output in `bash -c`.

## Known traps (each has burned a real experiment)

| Trap | Symptom | Fix |
|---|---|---|
| NFS job I/O (logs/workdir) | Hang at step 0, no loss line, all ranks frozen | Everything on `/data`; `PYTHONWARNINGS=ignore` |
| flash-attn 2.8.3 | Intermittent SIGSEGV in dense varlen backward, "flaky GPU" | Env pins 2.8.2 (source-built); repatch pre-2026-06-30 `/data` clones |
| Silent auto-resume | Relaunch into a non-empty `--save-folder` resumes at step N; step counts + wall-clock garbage, configs can mix | Fresh/unique save folder per config; check first log line is `step=1` |
| Wrong `load_path` depth | Trains from scratch (CE ≈ 12) | Point at `.../model_and_optim` |
| Unrepaired marker embeddings | Docchunk/landmark runs train to chance | `fix_marker_embeddings.py` the base first |
| HF hub access from compute node | Silent multi-minute dataloader stalls | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` |
| Nested `srun` inside an allocation | `Unable to create step ...` | Use `sbatch` |
