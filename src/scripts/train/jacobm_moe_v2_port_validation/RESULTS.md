# moe-v2-core port validation results

Validated candidate commit: `0cdcc8b813ab5ca582689edb80e4892891b03ae9`.

## Exact checkpoint gate

The final optimal 275M Cx1 checkpoint passed the strict gate on 2026-07-16:

- source checkpoint: `pt-275m-intwide-hybrid-gdn-ev1-cx1-lr1p6e-3-r1/step16108`;
- Beaker experiment: `01KXPGSYWMWNPA5GD7SZ5EX8R0`;
- status: `STRICT_PORT_PARITY_PASS` with `bitwise_equal=true`;
- all 216 checkpoint-main model tensors mapped one-to-one (2,701,754,320
  serialized parameter elements);
- fixed input IDs, full logits, all 12 block outputs, and all 55 captured
  router tensors were byte-identical (`torch.equal`), for 69 exact tensor
  comparisons in total;
- both full-logit artifacts have SHA256
  `958fe9da3227aa3ace5b191b0dc0ae906a7e46c3cc67184ea88bb2eb605471d7`.

The machine-readable report is stored at
`/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/port-validation/0cdcc8b81/parity/275m-cx1-step16108/strict_parity.json`.

The first candidate attempt exposed only a serialized-config compatibility
difference: the candidate predates Muon optimizer fields. The adapter now
requires every source optimizer group to specify `use_muon=false` before
removing those inert fields; it does not alter this run's AdamW semantics.

## Post-gate training controls

Both controls are in Beaker experiment `01KXPH0JBTXJ9E0A8MHX84CJMA`, use the
urgent unallocated queue (`minRuntime=0`, `autoResume=true`), and have no
in-loop evaluations.

### Full 275M Cx1 control

- Beaker job: `01KXPH0JQ3GJV1RD3EYMC8FAPR`
- W&B run: `szknd641`
- production geometry: two B300s, EP1, rank MB16, accumulation 1, 262,144
  tokens/global batch, 8,192-token sequences, LR 1.6e-3;
- exact horizon: 4,222,483,520 tokens / 16,108 steps;
- checkpoint policy: rolling ephemeral checkpoint every 500 steps and final
  permanent checkpoint only.

At step 60 the run was healthy with no skipped optimizer updates, 188.3 GiB
active memory/GPU, about 0.41 seconds/step, and 317,462 tokens/s/GPU (317,727
actual average). The candidate reports 518.9 TFLOP/s/GPU at that point, but raw
step time and token throughput are the cross-branch comparison metrics because
the branches use different FLOP accounting.

The first 500 steps also reproduce the historical optimization trace closely.
Across the 51 common console-logging points from step 1 through step 500, CE
loss has mean absolute difference 0.0039 and maximum absolute difference 0.014.
This training comparison is intentionally described as near-identical, not
bit-identical; only the checkpoint/logit/intermediate gate above makes the
stronger exact claim.

Status: running. Once it completes, compare its full loss curve and post-hoc
validation results with the historical 275M Cx1 run.

### 1.2B EP8 systems smoke

- Beaker job: `01KXPH0JTFG4RFZ3HY5XH9BNRS`
- W&B run: `1wl8cn78`
- eight B300s, EP8 `sync_1d`, MB cap 8 / effective MB4, accumulation 1,
  262,144 tokens/global batch, 8,192-token sequences, LR 4e-4;
- hard stop after 12 optimizer steps; checkpoints disabled.

Status: passed. The job completed all 12 requested optimizer steps with Beaker
exit code 0 and finalized W&B successfully. Steps 3-12 sustained 49,077-52,477
tokens/s/GPU; step 12 was 0.63 seconds/step, 51,811 tokens/s/GPU, and 171.2 GiB
active memory/GPU. No optimizer update was skipped. The branch reports 418.5
TFLOP/s/GPU at step 12, but that number is not used for cross-branch comparison.

The hard-stop teardown printed an ignored DataLoader-worker cleanup exception
after step 12. It occurred during iterator destruction, after all requested
steps, and did not fail training: the trainer logged `Training complete`, W&B
finalized the run as successful, and Beaker finalized it with exit code 0.
