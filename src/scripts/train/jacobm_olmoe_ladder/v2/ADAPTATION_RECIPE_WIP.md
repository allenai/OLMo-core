# Midtraining and long-context recipe (WIP)

This is the provisional dense-ladder-aligned policy for the next OLMoE
midtraining (MT) and long-context extension (LCE) wave. It records the current
working agreement but is not yet an approved launch specification. Do not
change launchers or submit work from this note until the unresolved items below
are settled.

Let a model's pretraining (PT) budget be `Y` tokens and its selected PT peak
learning rate be `Z`.

| Stage | Provisional peak LR | Provisional token budget |
|---|---:|---:|
| PT | `Z` | `Y` |
| MT | `Z / 5` | `_midtraining_tokens(Y)` below |
| LCE | `Z / 10` | not yet specified |

The LCE peak is therefore one half of the MT peak. Relative to the incumbent
OLMoE assumptions (`Z / 10` for MT and `Z / 20` for LCE), both proposed peaks
are 2x larger.

## Midtraining token scaling

The intended policy shown by the dense-ladder plot is:

- at or below 4B PT tokens, MT uses 50% of PT tokens;
- from 4B to 1T PT tokens, the fraction decreases smoothly from 50% to 10%;
- at and above 1T PT tokens, MT is capped at 100B tokens.

The curve consistent with those endpoints and the supplied plot is:

```python
import math

LOWER_PT_TOKENS = 4_000_000_000
UPPER_PT_TOKENS = 1_000_000_000_000
MAX_MIDTRAIN_TOKENS = 100_000_000_000


def _midtraining_tokens(pretraining_tokens: int) -> int:
    if pretraining_tokens <= 0:
        raise ValueError("pretraining_tokens must be positive")
    if pretraining_tokens <= LOWER_PT_TOKENS:
        return round(0.5 * pretraining_tokens)
    if pretraining_tokens >= UPPER_PT_TOKENS:
        return MAX_MIDTRAIN_TOKENS

    log_span = math.log(UPPER_PT_TOKENS / LOWER_PT_TOKENS)
    position = math.log(pretraining_tokens / LOWER_PT_TOKENS) / log_span
    ratio = (
        0.5
        + (-1.2 + 0.1 * log_span) * position**2
        + (0.8 - 0.1 * log_span) * position**3
    )
    return round(pretraining_tokens * ratio)
```

The version copied from Slack used subtraction signs before both polynomial
terms. Taken literally, that version increases the ratio toward about 90% and
then discontinuously drops to 10% at 1T, contradicting both the stated policy
and supplied plot. Confirm the exact checked-in dense-ladder function before
implementation.

Representative values from the plotted curve are:

| PT tokens | MT tokens | MT / PT |
|---:|---:|---:|
| 4B | 2B | 50% |
| 10B | 4.83B | 48% |
| 100B | 32.9B | 33% |
| 1T | 100B | 10% |

Using active non-embedding parameters to derive the current GDN2 Cx8 PT
budgets, the provisional MT budgets would be:

| Model | PT tokens | Provisional MT tokens |
|---|---:|---:|
| 275M | 38.7B | 15.8B |
| 480M | 72.0B | 25.8B |
| 810M | 129.9B | 39.6B |
| 1.2B | 199.8B | 52.5B |

## Evaluation policy

- Run the same post-training OlmoBase suite after PT, MT, and LCE.
- The reported "average of averages" is the average of four category means:
  OlmoBase Math, OlmoBase Gen, OlmoBase MCQA STEM, and OlmoBase MCQA
  non-STEM.
- Do not include the newer non-BPB code evaluations; they are currently too
  slow and the effective suite remains the v0.0.1 suite.
- Run RULER from 4K through 131K after every stage, not only after LCE. This is
  especially useful for detecting when NoPE hybrids begin length
  generalization.
- Keep evaluators out of the training process. Use separate post-training
  validation jobs and converted HF checkpoints with vLLM on Jupiter for RULER.

## Unresolved before implementation

1. Confirm the polynomial signs against the authoritative dense-ladder code.
2. Specify the LCE token budget; the current message only defines MT tokens.
3. Select MT and LCE warmup/decay schedules. The incumbent fixed 2,000-step
   warmup plus constant LR was accidental, while PT uses a 10%-of-tokens
   linear warmup followed by cosine decay to 0.1x peak. Test this transition
   in isolation rather than silently changing it.
4. Confirm per-size sequence/global batches, microbatches, GPU counts, and EP.
5. Confirm whether any optimizer settings besides peak LR change by stage.
6. Turn this note into versioned manifests and dry-run/smoke-test each selected
   model before a full launch.

