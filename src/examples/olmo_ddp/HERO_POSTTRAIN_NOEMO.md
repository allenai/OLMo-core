# EMO pretraining, non-EMO post-training (2026-09-14)

This isolated branch reuses the qualified hero MT/LC recipe and kernel/runtime pins.
Only the EMO-pretrained 2T decay is in scope. The `emo` arm is the **PT source label**;
both downstream `MTRun.emo` and `LCRun.emo` are false. Both model builders assert all
router EMO configs are absent. Existing PT, MT, LC and six SFT jobs are untouched.

| Stage | Tokens | LR | Warmup | Schedule | GPUs | Batch | Microbatch |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| MT | 100,008,984,576 | 2.2e-4 | 2,000 steps | cosine to zero | 64 | 16,777,216 | 4 × 8K |
| LC | 100,008,984,576 | 1.1e-4 | 2,000 steps | linear to zero | 64 | 16,777,216 | 1 × 64K |

Both stages use fresh optimizer/data counters, 5,961 steps, BF16, EP1/PP1 and
urgent allocated Holmes jobs in `ai2/olmo3p5-training`. LC uses the qualified block
recomputation fix. MT automatically audits initialization, saves/reloads step 2,
then continues. LC first runs its four-step save/resume gate, then the full job.
The inherited node exclusion list is retained, with decommissioned Holmes 520 removed.

Run `olmoe3_hero_mt_launch.py validate --apply` and
`olmoe3_hero_lc_launch.py validate --apply` for real-image CPU config/data checks;
run `olmoe3_hero_lc_launch.py qualify --apply` for the existing two-GPU recomputation
qualification on this pin. Once successful, launch the MT watcher with `watch --gate ID
--apply`. Once its named MT job exists, launch the LC watcher using
`watch --gate LC_CPU_ID --kernel-gate LC_GPU_ID --apply`.

The two independent, restartable Phobos CPU watchers request **no resources**.
Each uses durable submit-once specs/intents. The LC watcher resolves the new MT
job by its owner-qualified name and independently waits for the audited MT endpoint;
it does not wait for MT evals. Conversion -> qualification -> frozen OLMoBase
Gen/MCQA, Math, Code/FIM runs after each stage; ordinary RULER also follows LC.
Evals remain allocated in the MoE workspace. No outputs are stored in Beaker results.

Checkpoints, automation, W&B and scratch paths are new campaign namespaces.
Uploads use the existing private `allenai/olmo-3p5-small` bucket with separate
`posttrain-noemo-20260914/mt20/emo` and `posttrain-noemo-20260914/lc100b/emo` prefixes.
Each registers apply cleanup, two protected local checkpoints and a one-hour grace
period. Source-copy/hash and all-rank optimizer reset/resume audits remain required.
No manual checkpoint deletion and no new SFT job is part of this deployment.
