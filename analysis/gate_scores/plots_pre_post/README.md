# Landmark gate-set Jaccard: compressive landmark, pre- vs post-SFT

Reconstruction of the Q1–Q5 gate-similarity (Jaccard) figures (`analysis/plots/q*.png`, originally
`compressive` vs `fast` base), here comparing the **compressive** landmark checkpoint **before vs after
SFT**:

- **A / pre-SFT** — `q4b-base-fastcomplm-s2385` (compressive landmark, base)
- **B / post-SFT** — `q4b-comp-5task-s8550` (compressive landmark, 5-task SFT)

The figures ask how similar the *hard top-k opened-gate sets* (the kept landmark blocks) are along
different axes, using Jaccard `|A∩B| / |A∪B|`, on RULER decode gate logs at 8k/16k/32k/64k.

## Figures

| file | question |
|---|---|
| `q1_layers.png` | Q1 — gate-set similarity across **layers** (same head) vs layer gap |
| `q1_layer_matrix.png` | Q1 — head-pooled layer×layer Jaccard (64k) |
| `q2_tokens.png` | Q2 — similarity across **decoded tokens** (same example) vs token gap |
| `q3q4_example_model.png` | Q3 cross-**example** (positional bias) vs Q4 cross-**model** (pre vs post), per subtask |
| `q5_cross_head.png` | Q5 — within-layer agreement across **heads** (first/middle/final) |
| `q5_head_matrix.png` | Q5 — head×head Jaccard within a layer (64k) |

The dotted line is the chance baseline (exact expected Jaccard of two independent uniform k-subsets of
n candidate blocks; n,k read from the data per length).

## Method

Two-stage, because the raw gate logs are ~9 TB on weka (up to ~25 MB per JSONL line):

1. **Extraction (on-cluster, weka-mounted):** `extract_gate_sets.py` reduces the raw per-token/per-head
   all-candidate-block dumps to compact per-record *kept-block sets*. Two samplings:
   - **head mode** (first N records/file) → dense multiple-tokens-per-example, used for **Q1/Q2/Q5**.
     Stays within the first RULER subtask (`cwe`); fine since these pool over layers/heads.
   - **balanced mode** (`--per-key` records per subtask) → all 13 RULER subtasks, used for **Q3/Q4**.
     Both checkpoints share doc ids, so cross-model keys overlap.
   Driver: `extract_pre_post_on_weka.sh` (set `MODE=head` or `MODE=balanced`).
2. **Plotting (local):** `plot_gate_jaccard.py` computes the Q1–Q5 Jaccards from the compact dumps and
   renders the figures. Set representation is int bitmasks + popcount (the Q1 curves are ~10⁷ pairwise
   comparisons).

## Reproduce

```bash
# 1. extract on weka (gantry, jupiter, weka-mounted) — head dumps for Q1/Q2/Q5:
gantry run -w ai2/flex2 -b ai2/oe-other -c ai2/jupiter --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --branch amandab/gate-jaccard-analysis --ref b8fa0750d \
  --weka oe-training-default:/weka-mount/oe-training-default \
  --python-manager conda --system-python --install "echo skip" \
  -- bash analysis/gate_scores/extract_pre_post_on_weka.sh
#    and again with --env MODE=balanced --env PER_KEY=60 for the Q3/Q4 dumps.

# 2. fetch the result datasets, then plot locally:
python analysis/gate_scores/plot_gate_jaccard.py \
  --a-label "compressive base (pre-SFT)"  --a-dumps 'HEAD/q4b-base-fastcomplm-s2385_ruler_*.jsonl' \
  --b-label "compressive SFT (post-SFT)"  --b-dumps 'HEAD/q4b-comp-5task-s8550_ruler_*.jsonl' \
  --lengths 8192 16384 32768 65536 --outdir plots_pre_post
#    (Q3/Q4 panel regenerated from the BALANCED dumps.)
```

Beaker extraction jobs: head `01KXK750KNKJ6WVW8FD3X3VSKK`, balanced `01KXK869P6YQ5YYCGX24G9PJ5D`
(workspace `ai2/flex2`). Gate-log provenance: `../in_progress_gate_distribution.md`.

## What the pre/post comparison shows

Both checkpoints are structurally similar (same decay shapes, same layer/head block structure as the
original compressive-vs-fast figures), but SFT leaves a consistent fingerprint:

- **Q1 (across layers):** similarity decays with layer gap and ticks up at the extremes. Pre/post are
  nearly identical at 8k/16k, but at **32k the base (pre-SFT) keeps a *higher* cross-layer Jaccard than
  the SFT model** — SFT makes distant layers select *more distinct* block sets (more layer
  specialization). The layer×layer matrix shows SFT deepening a few "cold" layers (e.g. L24) that
  disagree with the rest.
- **Q2 (across decoded tokens, same example):** **SFT is markedly more stable** — at 64k the SFT model
  holds ~0.55–0.70 Jaccard across decode steps vs ~0.51–0.57 for base. Post-SFT re-selects the same
  landmark blocks step to step much more consistently.
- **Q3/Q4 (per subtask):** **post-SFT cross-example Jaccard (orange) is higher than pre-SFT (blue) on
  almost every subtask and length** — SFT sharpens *positional bias* (the gate keys off block position
  more than content). **Cross-model (green) sits below both** at 8k–32k — the two checkpoints disagree
  with each other more than two examples of one model do — but at **64k cross-model rises to meet
  cross-example** on several subtasks (cwe, fwe), i.e. at extreme length pre and post converge.
- **Q5 (across heads):** V-shape (heads agree in the first/final layer, diverge in the middle);
  post-SFT is slightly *more* head-consistent than base in the final layer at long context.

Net: SFT concentrates the landmark gate — more stable across decode steps, more position-driven across
examples, more specialized across layers — without changing the qualitative structure.

### Caveats
- **Q1/Q2/Q5** are computed from **head-sampled** dumps, which fall in RULER's first subtask (`cwe`);
  these questions pool over layers/heads/tokens and are subtask-agnostic, so that is representative but
  not a subtask average. **Q3/Q4** use the **balanced** dumps (all 13 subtasks, ~50 docs each).
- Sample: ~200 (Q1/Q2/Q5) / 780 (Q3/Q4) records per length per model; the plot caps subsample further.
  Trends are stable but exact values will wiggle with the sampling seed.
