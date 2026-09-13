# Qwen3.5-4B sparse → compressive CPT at 256K

Both stages use 8 nodes × 8 GPUs, Ulysses CP=4, DP=16, TP=1, and
16 sequences per optimizer step (4,194,304 model tokens). They inherit the
existing Qwen3.5 compressive longmino baseline: Qwen3.5 longmino-512k data,
262,144 model tokens / 258,048 content tokens per sequence, one landmark per
63 content tokens, landmark ID 248200, peak LR 3.2e-4, 400-step warmup,
linear decay over 10B tokens, full activation checkpointing, and fused LM loss.
The eight full-attention layers change attention strategy; the 24 GDN layers
continue training with the same architecture.

Stage one starts from the converted Qwen3.5 base with fresh optimizer and
trainer state. Its maximum duration remains 10B tokens, but a hard stop at
5B preserves the full-run LR schedule. Stage two restores model, optimizer,
and trainer state (including the data cursor and global token count) and
continues to 10B total, without another warmup.

## Launch

From the repository root, with the training environment active:

```bash
PYTHONPATH=src python src/scripts/train/memexpress/cpt/Qwen3.5-4B-staged-sparse-256k.py \
    launch q35-4b-staged-sparse-256k ai2/jupiter-cirrascale-2 \
    --launch.follow=false --launch.step_soft_timeout=null
```

After stage one finishes successfully and its final checkpoint is complete:

```bash
PYTHONPATH=src python src/scripts/train/memexpress/cpt/Qwen3.5-4B-staged-compressive-256k.py \
    launch q35-4b-staged-compressive-256k ai2/jupiter-cirrascale-2 \
    --trainer.load_path=/weka/oe-training-default/ai2-llm/checkpoints/q35-4b-staged-sparse-256k/step1193 \
    --launch.follow=false --launch.step_soft_timeout=null
```

These commands submit separate jobs; they do not install an automatic job dependency.
Use distinct run names/save folders for the two stages. Stage two deliberately
requires an explicit load path so it cannot silently start from the base.
If changing the first-stage run name or save folder, update the handoff path.
For GPU validation, replace `launch` with `dry_run`; the hybrid model needs
its GPU training dependencies. A short full-state save/restore check is still
needed before relying on the handoff for a production run.

Budgets include landmark tokens, matching the existing CPT configs. With the
default batch, stage one ends at step 1193 (5,003,804,672 tokens), and stage two
ends at step 2385 (10,003,415,040 tokens total). The standard checkpointer saves
at the hard stop even though it is not a regular 250-step save boundary.
If overriding the batch or stopping point, use the actual final checkpoint.
Keep data, seeds, landmark geometry, global batch, and LR schedule identical
between stages. Keep compressive gate temperature disabled to retain the same
parameter set for optimizer restoration. Compressive training uses quadratic
attention, so the second stage is expected to be slower.
