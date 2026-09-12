# Mixed midtraining

Use [`Mixed-Midtraining.py`](../../../src/scripts/train/Mixed-Midtraining.py) after joint
vision alignment. The canonical recipe uses 90% text / 10% vision expected supervised-loss
mass. Set `--recipe.text_loss_share=1` for text-only training with the same text recipe.
Mixed training updates the vision encoder, connector and LM, with vision LR `1e-6`
and vision activation checkpointing. Text-only training keeps vision frozen. There are
no separate arm profiles.

The default batch is 128 sequences of 8,192 tokens across two eight-GPU nodes, with two
sequences per GPU microbatch and four gradient-accumulation steps.

The [training guide](../../../docs/source/guides/mixed_midtraining.md) covers launch,
checkpoint handoff, defaults and source configuration. The runner inherits the alignment
parent's model and tokenizer; it does not add router-input RMS normalization. Mixed packing
retains its existing 8K context and 16-crop packed-sequence limit pending separate validation.

## Default data

| Component | Expected supervised-loss mass |
| --- | ---: |
| 61-source OLMo3 text mixture | 90% |
| PixMo captions/transcripts, Points Basic/HF, Count and CoSyn Point | 8% |
| OCR/document: TextVQA, DocVQA, InfoVQA and ChartQA | 1.230769% |
| Audited alignment: filtered VisualWebInstruct and Geo170K | 0.769231% |

The five-source visual group preserves its conditional example ratios. OCR and audited
alignment use prepared alignment selections and held-out exclusions. Grounded counting and
scalar-count replay remain enabled. The visual recipe excludes Tulu; the text mixture retains
its own instruction sources.

[visual_calibration_v1.json](visual_calibration_v1.json) records the default visual means:
128 examples per source, seed 6198, pinned Dolma2 tokenizer, 8K context, eight local crops,
document formatting and unweighted response loss. These are bounded estimates, not exact
corpus means. Recalibrate changed source populations or serialization; do not reuse alignment's
root-weighted means.

Historical experiment reports, evaluations, checkpoints and frozen runtimes remain evidence
for completed runs. Their arm names do not define active training recipes. The native runner
is a fresh alignment-to-midtraining handoff, not a full-state resume of an archived runner.
