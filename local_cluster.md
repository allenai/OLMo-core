# Running jobs on the local (Berkeley) slurm cluster

How we train and eval OLMo-core models on the Berkeley H200 nodes, **without** AI2 infra (no weka,
no Beaker). This is the local counterpart to the Beaker flow in CLAUDE.md. Reference launchers live
in `src/scripts/train/sft/*_local*.sbatch` / `*local_mooney.sbatch`; a good canonical example is
`src/scripts/train/sft/run_q06b_dense_contra_n20_local_mooney.sbatch`.

## Cluster map

| Node | GPUs | Partition | Notes |
|---|---|---|---|
| `hermione` | none | — | Login/head node. 6 cores, ~7.5 GB RAM. Edit + submit only; heavy imports OOM. |
| `horton` | 8×H200 | `berkeleynlp` | QOS `preemptive_high_sewonm`. Claude dev sessions run here (`claude-horton`). |
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
  cross-node read-only-ish via `/net/<host>/data/...` (NFS speed).

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

One conda env everywhere: `corpus-reasoning-olmo` (torch 2.9.1+cu128, **flash-attn 2.8.2 pinned** —
2.8.3 has a varlen-backward SIGSEGV regression). Master copy on NFS at
`/scratch/users/prasann/conda/envs/corpus-reasoning-olmo`; fast clones on `/data` of
horton/mooney/cubbins/sneetches (`import transformers` 2 s vs 60 s).

**All the shell-level conventions live in one master file — source it, don't copy-paste:**

```bash
source /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src/scripts/local_env.sh
```

It sets `REPO`, puts the fast env on `PATH` (direct PATH — `conda activate` hangs minutes on NFS
lock contention), `PYTHONPATH=$REPO/src`, the offline flags, `PYTHONWARNINGS=ignore`,
`WANDB_API_KEY`/`WANDB_FLAG`, a random `MASTER_PORT`, and defines `pick_clean_gpus N` +
`fresh_workdir ROOT`. Pre-set a var before sourcing to override a default (e.g.
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
     (see CLAUDE.md and `document-chunked-marker-embeddings.md`).
3. **Standalone torchrun launcher.** A `*-SFT-local.py` script builds the configs and calls
   `trainer.fit()` directly with local paths (mirrors `internal/experiment.py::train`), wrapped by
   an sbatch file. Every launcher takes env overrides:

```bash
# Smoke first, then full:
sbatch --export=ALL,MAX_STEPS=10,NGPU=2 src/scripts/train/sft/run_q06b_dense_contra_n20_local_mooney.sbatch
sbatch src/scripts/train/sft/run_q06b_dense_contra_n20_local_mooney.sbatch
```

Inside the sbatch, before torchrun, **pin clean GPUs** — allocations regularly include a
rogue-occupied GPU (and nvidia-smi is not cgroup-isolated, so filter by memory used):

```bash
mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
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
- Live log: `tail -f /data/prasann/joblogs/<name>_<jobid>.log` **from that node** (or
  `/net/<host>/data/...` from elsewhere). Older jobs logged to hidden `.<family>_%x_%j.log` files
  in the repo root; those are archived in `logs/` (gitignored) — new launchers must not log to the
  repo/NFS.
- Cancel with `scancel <jobid>`.
- Claude runs inside a `claude-horton` srun allocation → a nested `srun --nodelist=<other-node>`
  fails (inherits SLURM_*). **Always `sbatch`** for work on other nodes.
- The shell is zsh: no word-splitting — wrap `for`-loops over command output in `bash -c`.

## Known traps (each has burned a real experiment)

| Trap | Symptom | Fix |
|---|---|---|
| NFS job I/O (logs/workdir) | Hang at step 0, no loss line, all ranks frozen | Everything on `/data`; `PYTHONWARNINGS=ignore` |
| flash-attn 2.8.3 | Intermittent SIGSEGV in dense varlen backward, "flaky GPU" | Env pins 2.8.2 (source-built); repatch pre-2026-06-30 `/data` clones |
| Silent auto-resume | Relaunch into a non-empty `--save-folder` resumes at step N; step counts + wall-clock garbage, configs can mix | Fresh/unique save folder per config; check first log line is `step=1` |
| Wrong `load_path` depth | Trains from scratch (CE ≈ 12) | Point at `.../model_and_optim` |
| Unrepaired marker embeddings | Docchunk/landmark runs train to chance | `fix_marker_embeddings.py` the base first |
| Rogue GPU in allocation | OOM/slow on one rank | Clean-GPU pinning snippet above |
| HF hub access from compute node | Silent multi-minute dataloader stalls | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` |
| Nested `srun` from Claude's allocation | `Unable to create step ...` | Use `sbatch` |
