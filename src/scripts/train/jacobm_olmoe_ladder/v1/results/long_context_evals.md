# Long-Context Evaluation Results

Generated: 2026-07-23 17:11 UTC

RULER uses 13 tasks with 100 examples per task at 65,536 tokens. Higher recall is better. Raw metrics and predictions are cached under `results/cache/ruler/`.

The canonical inference path is converted HF checkpoints with vLLM on one Jupiter H100. The current OLMo-core provider does not load OLMo-DDP checkpoints whose distributed state keys use `module.*.main`.

| size | variant | checkpoint | backend | aggregate recall | examples | scoring time | examples/s | Beaker |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 275m | baseline | `lc-275m-baseline-cx8-mt2e-4-lc1e-4-64k-r2/step47684` | vllm | 0.1753 | 1300 | 651.1s | 2.00 | [01KXHATN799NGFT5ADBAK343MT](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXHATN799NGFT5ADBAK343MT) |
| 275m | integration_deep | `lc-275m-integration-deep-cx8-mt1p6e-4-lc8e-5-64k-r1/step47684` | vllm | 0.2518 | 1300 | 750.2s | 1.73 | [01KXMAF717DYMFG7NCRE9J69ND](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXMAF717DYMFG7NCRE9J69ND) |
| 275m | integration_wide | `lc-275m-integration-wide-cx8-mt1p6e-4-lc8e-5-64k-r1/step47684` | vllm | 0.2159 | 1300 | 691.9s | 1.88 | [01KXMAFF0TBDCA4Z83B6PB4052](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXMAFF0TBDCA4Z83B6PB4052) |
| 480m | baseline | `lc-480m-baseline-cx8-mt8e-5-lc4e-5-64k-r1/step31790` | vllm | 0.2109 | 1300 | 670.8s | 1.94 | [01KXMAFPGY9QY7SK6MJKQHWTSW](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXMAFPGY9QY7SK6MJKQHWTSW) |
| 480m | integration_deep | `lc-480m-integration-deep-cx8-mt8e-5-lc4e-5-64k-r1/step31790` | vllm | 0.1978 | 1300 | 920.0s | 1.41 | [01KXMAJ765PQ9QYYYDAMSYW183](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXMAJ765PQ9QYYYDAMSYW183) |
| 480m | integration_wide | `lc-480m-integration-wide-cx8-mt8e-5-lc4e-5-64k-r1/step31790` | vllm | 0.2095 | 1300 | 1197.6s | 1.09 | [01KXMAVY30QJ7J2X94H00XGBJ8](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXMAVY30QJ7J2X94H00XGBJ8) |
| 810m | baseline | `lc-810m-baseline-cx8-mt4e-5-lc2e-5-64k-r1/step23842` | vllm | 0.2234 | 1300 | 973.1s | 1.34 | [01KXZ2H9E4TKZ7REK9Z7RW9VJZ](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXZ2H9E4TKZ7REK9Z7RW9VJZ) |
| 810m | integration_wide | `lc-810m-integration-wide-cx8-mt4e-5-lc2e-5-64k-r1/step23842` | vllm | 0.1879 | 1300 | 1016.4s | 1.28 | [01KXZ2JBY2X06TZ1GAWC45S1N1](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KXZ2JBY2X06TZ1GAWC45S1N1) |

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

## 275m integration_deep

| task | recall |
| --- | ---: |
| `ruler_cwe__65536` | 0.0070 |
| `ruler_fwe__65536` | 0.0800 |
| `ruler_niah_mk_1__65536` | 0.2400 |
| `ruler_niah_mk_2__65536` | 0.3400 |
| `ruler_niah_mk_3__65536` | 0.0000 |
| `ruler_niah_mq__65536` | 0.2275 |
| `ruler_niah_mv__65536` | 0.2175 |
| `ruler_niah_s_1__65536` | 0.8700 |
| `ruler_niah_s_2__65536` | 0.5400 |
| `ruler_niah_s_3__65536` | 0.3000 |
| `ruler_qa_1__65536` | 0.2100 |
| `ruler_qa_2__65536` | 0.2000 |
| `ruler_vt__65536` | 0.0420 |

## 275m integration_wide

| task | recall |
| --- | ---: |
| `ruler_cwe__65536` | 0.0030 |
| `ruler_fwe__65536` | 0.0533 |
| `ruler_niah_mk_1__65536` | 0.3700 |
| `ruler_niah_mk_2__65536` | 0.1500 |
| `ruler_niah_mk_3__65536` | 0.0200 |
| `ruler_niah_mq__65536` | 0.2525 |
| `ruler_niah_mv__65536` | 0.2075 |
| `ruler_niah_s_1__65536` | 0.4000 |
| `ruler_niah_s_2__65536` | 0.4800 |
| `ruler_niah_s_3__65536` | 0.4300 |
| `ruler_qa_1__65536` | 0.1600 |
| `ruler_qa_2__65536` | 0.2800 |
| `ruler_vt__65536` | 0.0000 |

## 480m baseline

| task | recall |
| --- | ---: |
| `ruler_cwe__65536` | 0.0060 |
| `ruler_fwe__65536` | 0.1433 |
| `ruler_niah_mk_1__65536` | 0.1900 |
| `ruler_niah_mk_2__65536` | 0.1400 |
| `ruler_niah_mk_3__65536` | 0.0000 |
| `ruler_niah_mq__65536` | 0.2050 |
| `ruler_niah_mv__65536` | 0.1975 |
| `ruler_niah_s_1__65536` | 0.8600 |
| `ruler_niah_s_2__65536` | 0.2600 |
| `ruler_niah_s_3__65536` | 0.2700 |
| `ruler_qa_1__65536` | 0.2400 |
| `ruler_qa_2__65536` | 0.2300 |
| `ruler_vt__65536` | 0.0000 |

## 480m integration_deep

| task | recall |
| --- | ---: |
| `ruler_cwe__65536` | 0.0030 |
| `ruler_fwe__65536` | 0.3133 |
| `ruler_niah_mk_1__65536` | 0.2000 |
| `ruler_niah_mk_2__65536` | 0.1400 |
| `ruler_niah_mk_3__65536` | 0.0700 |
| `ruler_niah_mq__65536` | 0.1050 |
| `ruler_niah_mv__65536` | 0.0500 |
| `ruler_niah_s_1__65536` | 0.5400 |
| `ruler_niah_s_2__65536` | 0.3300 |
| `ruler_niah_s_3__65536` | 0.2400 |
| `ruler_qa_1__65536` | 0.2500 |
| `ruler_qa_2__65536` | 0.3200 |
| `ruler_vt__65536` | 0.0100 |

## 480m integration_wide

| task | recall |
| --- | ---: |
| `ruler_cwe__65536` | 0.0100 |
| `ruler_fwe__65536` | 0.3133 |
| `ruler_niah_mk_1__65536` | 0.2700 |
| `ruler_niah_mk_2__65536` | 0.1900 |
| `ruler_niah_mk_3__65536` | 0.0400 |
| `ruler_niah_mq__65536` | 0.1525 |
| `ruler_niah_mv__65536` | 0.1175 |
| `ruler_niah_s_1__65536` | 0.4000 |
| `ruler_niah_s_2__65536` | 0.4000 |
| `ruler_niah_s_3__65536` | 0.3600 |
| `ruler_qa_1__65536` | 0.2000 |
| `ruler_qa_2__65536` | 0.2700 |
| `ruler_vt__65536` | 0.0000 |

## 810m baseline

| task | recall |
| --- | ---: |
| `ruler_cwe__65536` | 0.0050 |
| `ruler_fwe__65536` | 0.4367 |
| `ruler_niah_mk_1__65536` | 0.3200 |
| `ruler_niah_mk_2__65536` | 0.2500 |
| `ruler_niah_mk_3__65536` | 0.2900 |
| `ruler_niah_mq__65536` | 0.2025 |
| `ruler_niah_mv__65536` | 0.1800 |
| `ruler_niah_s_1__65536` | 0.1700 |
| `ruler_niah_s_2__65536` | 0.2400 |
| `ruler_niah_s_3__65536` | 0.3000 |
| `ruler_qa_1__65536` | 0.2400 |
| `ruler_qa_2__65536` | 0.2700 |
| `ruler_vt__65536` | 0.0000 |

## 810m integration_wide

| task | recall |
| --- | ---: |
| `ruler_cwe__65536` | 0.0050 |
| `ruler_fwe__65536` | 0.4133 |
| `ruler_niah_mk_1__65536` | 0.0800 |
| `ruler_niah_mk_2__65536` | 0.0200 |
| `ruler_niah_mk_3__65536` | 0.0700 |
| `ruler_niah_mq__65536` | 0.0425 |
| `ruler_niah_mv__65536` | 0.0575 |
| `ruler_niah_s_1__65536` | 0.2400 |
| `ruler_niah_s_2__65536` | 0.3400 |
| `ruler_niah_s_3__65536` | 0.4200 |
| `ruler_qa_1__65536` | 0.3900 |
| `ruler_qa_2__65536` | 0.3200 |
| `ruler_vt__65536` | 0.0440 |
