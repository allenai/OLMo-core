# Web Contamination Experiment Results

Comparison of 1.3B models trained with and without 0.1% contaminated data (cascade_61k).

## Experiment Summary

| Setting | Baseline | Contaminated |
|---------|----------|--------------|
| **Beaker Experiment** | [01KMHR99FFH0YAK4DCZ0AF55GB](https://beaker.org/ex/01KMHR99FFH0YAK4DCZ0AF55GB) | [01KMK80TTSFZ496X5RBJJP8TSG](https://beaker.org/ex/01KMK80TTSFZ496X5RBJJP8TSG) |
| **Model** | 1.3B (1.12B non-emb) | 1.3B (1.12B non-emb) |
| **Training tokens** | 45B (2x Chinchilla) | 45B (2x Chinchilla) |
| **Data mix** | 100% Dolma web | 99.9% Dolma web + 0.1% cascade_61k |
| **Contaminated tokens** | 0 | ~45M (0.1% of 45B) |
| **Checkpoint** | `/weka/.../gl-1p3b-dolma-2xc-v2` | `/weka/.../gl-1p3b-contam-2xc` |

## Downstream Evaluation Results

### Accuracy Metrics (higher = better)

| Task | Baseline | Contaminated | Delta |
|------|----------|--------------|-------|
| arc_easy_test_mc_5shot_fast | 24.83% | 24.12% | -0.71% |
| arc_challenge_test_mc_5shot_fast | — | — | — |
| basic_skills_arithmetic | 27.22% | 26.36% | -0.86% |
| basic_skills_coding | 37.06% | **42.39%** | **+5.33%** |
| basic_skills_common_knowledge | 88.62% | **91.42%** | **+2.80%** |
| basic_skills_logical_reasoning | **92.13%** | 88.28% | -3.85% |
| basic_skills_pattern | 70.04% | 70.79% | +0.75% |
| basic_skills_string_operations | 22.89% | **25.10%** | **+2.21%** |
| copycolors_10way | 9.00% | 9.00% | 0.00% |
| mmlu_humanities_mc | 24.87% | 24.31% | -0.56% |
| mmlu_other_mc | 26.43% | 25.57% | -0.86% |
| mmlu_social_sciences_mc | 24.80% | 22.88% | -1.92% |
| mmlu_stem_mc | 26.08% | 25.51% | -0.57% |

### BPB Metrics (lower = better)

| Task | Baseline | Contaminated | Delta |
|------|----------|--------------|-------|
| arc_challenge_test_bpb | 1.199 | **1.131** | **-5.7%** |
| arc_easy_test_bpb | 0.708 | **0.683** | **-3.5%** |
| codex_humaneval_bpb | 0.677 | **0.646** | **-4.6%** |
| codex_mbpp_bpb | 0.872 | **0.836** | **-4.1%** |
| hellaswag_bpb | 0.794 | 0.792 | -0.3% |
| minerva_math_bpb | 0.797 | **0.741** | **-7.0%** |
| mmlu_humanities_bpb | 0.727 | 0.726 | -0.1% |
| mmlu_other_bpb | 1.090 | **1.067** | **-2.1%** |
| mmlu_social_sciences_bpb | 0.899 | **0.883** | **-1.8%** |
| mmlu_stem_bpb | 1.707 | **1.621** | **-5.0%** |
| mt_mbpp_cpp_bpb | 0.682 | **0.638** | **-6.4%** |
| mt_mbpp_java_bpb | 0.537 | **0.501** | **-6.7%** |
| mt_mbpp_rust_bpb | 0.969 | **0.872** | **-10.0%** |

### Code Fresh Perplexity (lower = better)

| Language | Baseline PPL | Contaminated PPL | Delta |
|----------|--------------|------------------|-------|
| python | 10.97 | **10.70** | **-2.5%** |
| typescript | 11.71 | **10.47** | **-10.6%** |
| rust | 9.46 | **8.42** | **-11.0%** |
| swift | 15.75 | **13.66** | **-13.3%** |
| scala | 11.39 | **10.90** | **-4.3%** |
| ruby | 18.56 | **18.30** | **-1.4%** |
| vue | 16.24 | **14.67** | **-9.7%** |

## Key Observations

### 1. Code Performance Improved Significantly
The contaminated model shows **consistent improvements on code-related tasks**:
- Rust BPB: -10.0%
- TypeScript PPL: -10.6%
- Swift PPL: -13.3%
- Codex HumanEval BPB: -4.6%
- basic_skills_coding accuracy: +5.33%

This suggests the cascade_61k contamination data may contain code or code-adjacent content.

### 2. Math Performance Improved
- minerva_math BPB: -7.0%
- mmlu_stem BPB: -5.0%

### 3. Reasoning Performance Mixed
- basic_skills_logical_reasoning: -3.85% (degraded)
- basic_skills_common_knowledge: +2.80% (improved)
- basic_skills_string_operations: +2.21% (improved)

### 4. General Knowledge Slightly Degraded
- Most MMLU accuracy scores slightly lower (-0.5% to -1.9%)
- But MMLU BPB scores improved (model is more confident, even if less accurate)

## Interpretation

With only **0.1% contamination** (~45M tokens out of 45B), we observe:

1. **Measurable impact**: The contamination is detectable despite the tiny fraction
2. **Domain-specific effects**: Code/math improved, suggesting cascade_61k has technical content
3. **No catastrophic degradation**: General capabilities remain similar
4. **Perplexity-accuracy divergence**: Lower perplexity doesn't always mean higher accuracy

## Next Steps

1. **Analyze cascade_61k content**: Understand what's in the contamination data
2. **Higher contamination rates**: Test 1%, 5%, 10% to measure dose-response
3. **Targeted contamination**: Test with known benchmark leakage (e.g., MMLU questions)
4. **Detection methods**: Can we detect contamination from model outputs?

## Reproducibility

### Baseline
```bash
python src/scripts/train/ladder/gemma_like_ladder.py launch gl-1p3b-dolma-2xc-v2 ai2/jupiter \
    --mix-yaml=src/scripts/train/ladder/dolma-300B-web-only.yaml \
    --mix-base-dir=s3://ai2-llm \
    --chinchilla-multiple=2.0 \
    --batch-multiplier=1.34 \
    --launch.num_nodes=8 \
    ...
```

### Contaminated
```bash
python src/scripts/train/ladder/gemma_like_ladder.py launch gl-1p3b-contam-2xc ai2/jupiter \
    --mix-yaml=src/scripts/train/ladder/dolma-300B-web-contam.yaml \
    --mix-base-dir=s3://ai2-llm \
    --chinchilla-multiple=2.0 \
    --batch-multiplier=1.34 \
    --launch.num_nodes=8 \
    ...
```
