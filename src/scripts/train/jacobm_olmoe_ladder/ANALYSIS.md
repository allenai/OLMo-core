# Tiny MoE Analysis Notes

## W&B Loss Pulls

Project:

- `ai2-llm/jacobm-olmoe-ladder`

Primary scalar:

- `train/CE loss`

Token progress scalar:

- `throughput/total tokens`

For Cx1 U-plots across different batch sizes, use token-window averages instead
of final-step or final-N-step averages. This avoids comparing 48 points from a
2M-token batch to 383 points from a 256k-token batch over different amounts of
training data.

Current default summary windows:

- final `100M` tokens
- final `250M` tokens
- final `500M` tokens

Command:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/summarize_wandb_losses.py
```

The script uses the local `WANDB_API_KEY`, pulls runs from W&B, filters to
`olmoe3-tiny-275m-cx1` by default, and prints TSV with final loss plus final-token-window
averages. It infers batch size from run-name tags:

- no `b...` tag and `cx1-lr`: 2M tokens/step
- `b128k`: 131,072 tokens/step
- `b256k`: 262,144 tokens/step
- `b512k`: 524,288 tokens/step

For a narrower pull:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/summarize_wandb_losses.py \
  --name-regex 'olmoe3-tiny-275m-cx1-b256k'
```

For both Cx1 and Cx2:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/summarize_wandb_losses.py \
  --name-regex 'olmoe3-tiny-275m-cx'
```

## 2026-06-02 Snapshot

Initial read after the first 2M-batch sweep finished and the 256k-batch sweep was
partially complete:

- 2M Cx1 best final-token-window result was `1.2e-3`.
- 256k Cx1 `3e-4` finished better than the 2M runs by final-token-window average.
- 256k Cx1 `5e-4` and `8e-4` were better than `3e-4` at matched token counts
  while still running.
- `1e-4` looked too slow and was dropped from the next Cx1 sweeps.

The first follow-up recommendation was:

- 256k LR refinement: `6e-4`, `1e-3`, `1.5e-3`, `2e-3`
- batch probe at strong LRs: `128k@5e-4`, `128k@8e-4`, `512k@5e-4`, `512k@8e-4`

After a later W&B refresh, `5e-4` at 256k looked stronger than `8e-4` by recent
token-window average, so the follow-up LR refinement was tightened to:

- 256k LR refinement: `4e-4`, `6e-4`, `7e-4`, `1e-3`
- batch probe at strong LRs: `128k@5e-4`, `128k@8e-4`, `512k@5e-4`, `512k@8e-4`
- Cx2 transfer check, queued first: `256k@5e-4`, `256k@7e-4`

After all follow-up runs finished, Cx1 `256k@1e-3` and `256k@1.2e-3` were nearly
tied and best among completed Cx1 runs. Because the high side had not clearly
turned over, the next Cx1 high-side probes were:

- `256k@1.5e-3`
- `256k@2e-3`
