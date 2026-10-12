# CTC local trainer "hang" at the dry-run forward = Triton FLA cache on NFS

**Symptom.** `run_ctc_local.sbatch` -> `train_ctc_suite.py` appears to hang forever right after
`trainer.py:1452 "Starting forward/backward dry-run batch..."`: process state S, ~0% GPU util,
~6 GB allocated (model loaded), no progress for 20+ min. Reproduces at NGPU=4 and NGPU=1, with
flash-attn 2.8.2 (the known-good pin). Looks exactly like a dead trainer.

**It is not a dead trainer.** A repro run with `PYTHONFAULTHANDLER=1` + `SIGABRT` dumped the Python
stack of all 167 threads: the process was actively TRAINING (CE loss 0.32, MFU 31.6%, ~180 steps,
19.7k TPS) and the "hang" is the first-forward **Triton JIT compile of the FLA Gated-DeltaNet
kernel** (`fla/modules/fused_norm_gate.py:layer_norm_gated_fwd`, from
`olmo_core/nn/attention/recurrent.py:371`). With a warm cache the dry-run takes ~6 s and training
runs fine.

**Root cause.** The Triton kernel cache defaults to `$HOME/.triton`, and here `$HOME` is
`/accounts/...` **NFS**. Triton guards each compile with a `FileLock`, and that lock DEADLOCKS over
NFS. The two arms of a run are usually launched together (synth + baseline on the same node), so two
cold compiles contend on the same NFS cache dir -> 167 threads wedged in `futex_wait`, 0% GPU,
forever. Same failure class as the flashinfer HOME->NFS wedge. The evidence: every historical hang
was a same-node PAIR; a single warm-cache run always completed.

**Tell.** `run_ctc_lambda.sbatch` already sets `TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID`
(node-local) -- `run_ctc_local.sbatch` did **not**, so it inherited the NFS default. That is the only
difference that mattered.

**Fix (committed in `run_ctc_local.sbatch`).** Point the cache at node-local disk:
```
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/prasann/triton_cache}"
```
A node-local (ext4) FileLock serializes concurrent compiles correctly. After the fix, two cold
compiles on separate nodes both cleared the dry-run and trained; the eval launcher
(`debug/obliq_synthetic/eval_obliq_viab.sbatch`) sets the same (the native eval loads the same FLA
kernels). Any local launcher that runs the Qwen3.5 GDN-hybrid must set a node-local Triton cache.

**Debugging note.** ptrace is blocked across sibling srun steps (yama scope=1), and `/proc/PID/stack`
needs CAP_SYS_ADMIN, so py-spy/gdb can't attach from a sibling job. `PYTHONFAULTHANDLER=1` + `kill
-ABRT <worker-pid>` dumps every thread's Python stack to the job log -- the reliable way to see
where a wedged local torchrun worker actually is.
