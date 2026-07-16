# Long-Context Evaluation Results

Generated: 2026-07-14 22:25 UTC

RULER uses 13 tasks with 100 examples per task at 65,536 tokens. Higher recall is better. Raw metrics and predictions are cached under `results/cache/ruler/`.

The canonical inference path is converted HF checkpoints with vLLM on one Jupiter H100. The current OLMo-core provider does not load OLMo-DDP checkpoints whose distributed state keys use `module.*.main`.

| size | variant | checkpoint | backend | aggregate recall | examples | scoring time | examples/s | Beaker |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 275m | baseline | `lc-275m-baseline-cx8-mt2e-4-lc1e-4-64k-r2/step47684` | vllm | 0.1753 | 1300 | 651.1s | 2.00 | [01KXHATN799NGFT5ADBAK343MT](https://beaker.org/ex/01KXHATN799NGFT5ADBAK343MT) |

## 275m baseline

| task | recall |
| --- | ---: |
| `ruler_cwe__65536` | 0.0070 |
| `ruler_fwe__65536` | 0.0000 |
| `ruler_niah_mk_1__65536` | 0.3200 |
| `ruler_niah_mk_2__65536` | 0.0000 |
| `ruler_niah_mk_3__65536` | 0.0300 |
| `ruler_niah_mq__65536` | 0.0875 |
| `ruler_niah_mv__65536` | 0.0925 |
| `ruler_niah_s_1__65536` | 0.4500 |
| `ruler_niah_s_2__65536` | 0.5300 |
| `ruler_niah_s_3__65536` | 0.2700 |
| `ruler_qa_1__65536` | 0.1200 |
| `ruler_qa_2__65536` | 0.2700 |
| `ruler_vt__65536` | 0.1020 |
