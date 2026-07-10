# Optimal Loss Values

Generated: 2026-07-10 22:44 UTC

Source: `/weka/oe-adapt-default/jacobm/olmoe3/OLMo-core/src/scripts/train/jacobm_olmoe_ladder/PLOTTED_RESULTS.md`. Values mirror the completed-run plotting policy: final-window training CE averaged over the last 250M tokens, running jobs excluded.

## Baseline Ladder

| model | Cx | family | batch | best observed LR | best avg250M | fit LR | fit avg250M | points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 275M | Cx1 | gpu2-ep1mb16 | 256k | 2e-3 | 2.7767 | 0.00213 | 2.7778 | 8 |
| 275M | Cx2 | b384k-gpu2-ep1mb8 | 384k | 1.8e-3 | 2.6541 | 0.00178 | 2.6541 | 3 |
| 275M | Cx4 | gpu4-ep1mb16 | 512k | 1.5e-3 | 2.5611 | 0.00146 | 2.5610 | 7 |
| 275M | Cx8 | gpu4-ep1mb8 | 768k | 1.6e-3 | 2.4864 | 0.00135 | 2.4849 | 8 |
| 275M | Cx16 | gpu8-ep1mb16 | 1M | 1.2e-3 | 2.4301 | 0.00107 | 2.4294 | 6 |
| 480M | Cx1 | gpu4-ep1mb8 | 256k | 1.2e-3 | 2.5636 | 0.000958 | 2.5624 | 3 |
| 480M | Cx2 | gpu4-ep1mb4 | 384k | 9e-4 | 2.4630 | 0.000973 | 2.4629 | 3 |
| 480M | Cx4 | gpu4-ep1mb8 | 512k | 8e-4 | 2.3788 | 0.00085 | 2.3787 | 4 |
| 480M | Cx8 | gpu8-ep1mb4 | 768k | 8e-4 | 2.3076 | 0.00077 | 2.3076 | 4 |
| 810M | Cx1 | gpu4-ep1mb4 | 256k | 6e-4 | 2.4104 | 0.000621 | 2.4094 | 7 |
| 810M | Cx2 | gpu8-ep1mb2 | 384k | 5.6e-4 | 2.3204 | 0.000629 | 2.3201 | 3 |
| 810M | Cx4 | gpu8-ep1mb4 | 512k | 4e-4 | 2.2424 | 0.000514 | 2.2412 | 4 |
| 810M | Cx8 | gpu8-ep1mb4 | 768k | 4e-4 | 2.1721 | 0.000467 | 2.1717 | 3 |
| 1.2B | Cx1 | gpu8-ep1mb2 | 256k | 4e-4 | 2.3108 | 0.000483 | 2.3101 | 4 |
| 1.2B | Cx2 | b384k | 384k | 6e-4 | 2.2229 | 0.000442 | 2.2204 | 4 |
| 1.2B | Cx4 | gpu8-ep1mb2 | 512k | 3e-4 | 2.1508 | 0.000366 | 2.1500 | 4 |
| 1.2B | Cx8 | gpu32-ep1mb1 | 768k | 4e-4 | 2.0835 | 0.000348 | 2.0831 | 3 |

## Expert Granularity

| model | Cx | variant | best observed LR | best avg250M | fit LR | fit avg250M | points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 275M | Cx1 | baseline 48E/top4 | 2e-3 | 2.7767 | 0.00203 | 2.7766 | 8 |
| 275M | Cx1 | coarse 24E/top2 | 2e-3 | 2.7814 | 0.00186 | 2.7813 | 3 |
| 275M | Cx1 | extreme 192E/top16 | 4e-3 | 2.7549 |  |  | 3 |
| 275M | Cx1 | fine 96E/top8 | 2e-3 | 2.7641 | 0.0021 | 2.7641 | 3 |
| 275M | Cx1 | ultra 384E/top32 | 2e-3 | 2.7468 | 0.00237 | 2.7464 | 3 |
| 275M | Cx2 | baseline 48E/top4 (b384k) | 1.8e-3 | 2.6541 | 0.00178 | 2.6541 | 3 |
| 275M | Cx2 | coarse 24E/top2 (b384k) | 1.8e-3 | 2.6576 | 0.00181 | 2.6576 | 3 |
| 275M | Cx2 | coarse 24E/top2 (b512k) | 2e-3 | 2.6522 |  |  | 3 |
| 275M | Cx2 | fine 96E/top8 (b384k) | 1.8e-3 | 2.6387 | 0.00189 | 2.6387 | 3 |
| 275M | Cx2 | fine 96E/top8 (b512k) | 2e-3 | 2.6324 |  |  | 3 |
| 275M | Cx4 | baseline 48E/top4 | 1.5e-3 | 2.5611 | 0.00156 | 2.5611 | 7 |
| 275M | Cx4 | coarse 24E/top2 | 1.6e-3 | 2.5713 | 0.00157 | 2.5713 | 3 |
| 275M | Cx4 | fine 96E/top8 | 1.6e-3 | 2.5523 | 0.00145 | 2.5521 | 3 |
| 275M | Cx8 | baseline 48E/top4 | 1.6e-3 | 2.4864 | 0.00136 | 2.4860 | 8 |
| 275M | Cx8 | coarse 24E/top2 | 1.6e-3 | 2.4990 | 0.0014 | 2.4988 | 3 |
| 275M | Cx8 | fine 96E/top8 | 1.6e-3 | 2.4773 | 0.00135 | 2.4766 | 3 |
| 480M | Cx1 | baseline 48E/top4 | 1.2e-3 | 2.5636 | 0.000958 | 2.5624 | 3 |
| 480M | Cx1 | coarse 24E/top2 | 9e-4 | 2.5817 |  |  | 1 |
| 480M | Cx1 | fine 96E/top8 | 1e-3 | 2.5546 |  |  | 1 |
| 480M | Cx2 | baseline 48E/top4 (b384k) | 9e-4 | 2.4630 | 0.000973 | 2.4629 | 3 |
| 480M | Cx2 | coarse 24E/top2 (b384k) | 1e-3 | 2.4767 |  |  | 1 |
| 480M | Cx2 | fine 96E/top8 (b384k) | 1e-3 | 2.4524 |  |  | 1 |
| 480M | Cx4 | baseline 48E/top4 | 8e-4 | 2.3788 | 0.00085 | 2.3787 | 4 |
| 480M | Cx4 | coarse 24E/top2 | 8e-4 | 2.3941 |  |  | 1 |
| 480M | Cx4 | fine 96E/top8 | 8e-4 | 2.3695 |  |  | 1 |
| 480M | Cx8 | baseline 48E/top4 | 8e-4 | 2.3076 | 0.00077 | 2.3076 | 4 |
| 480M | Cx8 | coarse 24E/top2 | 8e-4 | 2.3292 |  |  | 1 |
| 480M | Cx8 | fine 96E/top8 | 8e-4 | 2.3011 |  |  | 1 |
| 810M | Cx1 | baseline 48E/top4 | 6e-4 | 2.4104 | 0.000672 | 2.4102 | 6 |
| 810M | Cx1 | coarse 24E/top2 | 6e-4 | 2.4191 |  |  | 1 |
| 810M | Cx1 | fine 96E/top8 | 6e-4 | 2.3985 |  |  | 1 |
| 810M | Cx2 | baseline 48E/top4 (b384k) | 5.6e-4 | 2.3204 | 0.000629 | 2.3201 | 3 |
| 810M | Cx2 | coarse 24E/top2 (b384k) | 5.6e-4 | 2.3304 |  |  | 1 |
| 810M | Cx2 | fine 96E/top8 (b384k) | 5.6e-4 | 2.3074 |  |  | 1 |
| 810M | Cx4 | baseline 48E/top4 | 4e-4 | 2.2424 | 0.000514 | 2.2412 | 4 |
| 810M | Cx4 | coarse 24E/top2 | 4e-4 | 2.2585 |  |  | 1 |
| 810M | Cx4 | fine 96E/top8 | 4e-4 | 2.2353 |  |  | 1 |
| 810M | Cx8 | baseline 48E/top4 | 4e-4 | 2.1721 | 0.000467 | 2.1717 | 3 |
| 810M | Cx8 | coarse 24E/top2 | 4e-4 | 2.1892 |  |  | 1 |
| 810M | Cx8 | fine 96E/top8 | 4e-4 | 2.1700 |  |  | 1 |
| 1.2B | Cx1 | baseline 48E/top4 | 4e-4 | 2.3108 | 0.000483 | 2.3101 | 4 |
| 1.2B | Cx1 | coarse 24E/top2 | 4e-4 | 2.3179 |  |  | 1 |
| 1.2B | Cx1 | fine 96E/top8 | 4e-4 | 2.2962 |  |  | 1 |
| 1.2B | Cx2 | baseline 48E/top4 (b384k) | 6e-4 | 2.2229 | 0.000442 | 2.2204 | 4 |
| 1.2B | Cx2 | coarse 24E/top2 (b384k) | 3e-4 | 2.2321 |  |  | 1 |
| 1.2B | Cx2 | fine 96E/top8 (b384k) | 3e-4 | 2.2108 |  |  | 1 |
| 1.2B | Cx4 | baseline 48E/top4 | 3e-4 | 2.1508 | 0.000366 | 2.1500 | 4 |
| 1.2B | Cx4 | coarse 24E/top2 | 4e-4 | 2.1582 |  |  | 1 |
| 1.2B | Cx4 | fine 96E/top8 | 4e-4 | 2.1369 |  |  | 1 |
| 1.2B | Cx8 | baseline 48E/top4 | 2e-4 | 2.0897 |  |  | 2 |
| 1.2B | Cx8 | coarse 24E/top2 | 4e-4 | 2.0903 |  |  | 1 |
| 1.2B | Cx8 | fine 96E/top8 | 4e-4 | 2.0772 |  |  | 1 |

## Total Sparsity

| model | Cx | variant | best observed LR | best avg250M | fit LR | fit avg250M | points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 275M | Cx1 | baseline 48E/top4 | 2e-3 | 2.7767 | 0.00203 | 2.7766 | 8 |
| 275M | Cx1 | high total 96E/top4 | 2e-3 | 2.7542 | 0.0017 | 2.7538 | 3 |
| 275M | Cx1 | huge total 192E/top4 | 2e-3 | 2.7486 | 0.00157 | 2.7474 | 3 |
| 275M | Cx2 | baseline 48E/top4 (b384k) | 1.8e-3 | 2.6541 | 0.00178 | 2.6541 | 3 |
| 275M | Cx2 | high total 96E/top4 | 1.8e-3 | 2.6184 | 0.0016 | 2.6181 | 3 |
| 275M | Cx2 | huge total 192E/top4 | 9e-4 | 2.6109 | 0.000767 | 2.6108 | 6 |
| 275M | Cx4 | baseline 48E/top4 | 1.5e-3 | 2.5611 | 0.00156 | 2.5611 | 7 |
| 275M | Cx4 | high total 96E/top4 | 1.6e-3 | 2.5264 | 0.00131 | 2.5257 | 3 |
| 275M | Cx4 | huge total 192E/top4 | 8e-4 | 2.5038 | 0.00105 | 2.5028 | 4 |
| 275M | Cx8 | baseline 48E/top4 | 1.6e-3 | 2.4864 | 0.00136 | 2.4860 | 8 |
| 275M | Cx8 | high total 96E/top4 | 1.6e-3 | 2.4461 | 0.00124 | 2.4449 | 3 |
| 275M | Cx8 | huge total 192E/top4 | 8e-4 | 2.4144 | 0.000999 | 2.4136 | 4 |
| 480M | Cx1 | baseline 48E/top4 | 1.2e-3 | 2.5636 | 0.000958 | 2.5624 | 3 |
| 480M | Cx1 | high total 96E/top4 | 1e-3 | 2.5601 |  |  | 1 |
| 480M | Cx1 | huge total 192E/top4 | 8e-4 | 2.5484 |  |  | 1 |
| 480M | Cx2 | baseline 48E/top4 (b384k) | 9e-4 | 2.4630 | 0.000973 | 2.4629 | 3 |
| 480M | Cx2 | high total 96E/top4 | 8e-4 | 2.4429 |  |  | 1 |
| 480M | Cx2 | huge total 192E/top4 | 6e-4 | 2.4185 |  |  | 1 |
| 480M | Cx4 | baseline 48E/top4 | 8e-4 | 2.3788 | 0.00085 | 2.3787 | 4 |
| 480M | Cx4 | high total 96E/top4 | 7e-4 | 2.3509 |  |  | 1 |
| 480M | Cx4 | huge total 192E/top4 | 6e-4 | 2.3229 |  |  | 1 |
| 480M | Cx8 | baseline 48E/top4 | 8e-4 | 2.3076 | 0.00077 | 2.3076 | 4 |
| 480M | Cx8 | high total 96E/top4 | 7e-4 | 2.2723 |  |  | 1 |
| 480M | Cx8 | huge total 192E/top4 | 6e-4 | 2.2344 |  |  | 1 |
| 810M | Cx1 | baseline 48E/top4 | 6e-4 | 2.4104 | 0.000672 | 2.4102 | 6 |
| 810M | Cx1 | high total 96E/top4 | 5e-4 | 2.3968 |  |  | 1 |
| 810M | Cx1 | huge total 192E/top4 | 4e-4 | 2.3790 |  |  | 1 |
| 810M | Cx2 | baseline 48E/top4 (b384k) | 5.6e-4 | 2.3204 | 0.000629 | 2.3201 | 3 |
| 810M | Cx2 | high total 96E/top4 | 5e-4 | 2.2966 |  |  | 1 |
| 810M | Cx2 | huge total 192E/top4 | 4e-4 | 2.2714 |  |  | 1 |
| 810M | Cx4 | baseline 48E/top4 | 4e-4 | 2.2424 | 0.000514 | 2.2412 | 4 |
| 810M | Cx4 | high total 96E/top4 | 3.5e-4 | 2.2165 |  |  | 1 |
| 810M | Cx4 | huge total 192E/top4 | 3e-4 | 2.1866 |  |  | 1 |
| 810M | Cx8 | baseline 48E/top4 | 4e-4 | 2.1721 | 0.000467 | 2.1717 | 3 |
| 810M | Cx8 | high total 96E/top4 | 3.5e-4 | 2.1436 |  |  | 1 |
| 810M | Cx8 | huge total 192E/top4 | 3e-4 | 2.1032 |  |  | 1 |

## Shared Expert

| model | Cx | variant | best observed LR | best avg250M | fit LR | fit avg250M | points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 275M | Cx1 | baseline 48E/top4 | 2e-3 | 2.7767 | 0.00203 | 2.7766 | 8 |
| 275M | Cx1 | no shared, routed 9/8 d | 2e-3 | 2.7726 | 0.00182 | 2.7724 | 3 |
| 275M | Cx2 | baseline 48E/top4 (b384k) | 1.8e-3 | 2.6541 | 0.00178 | 2.6541 | 3 |
| 275M | Cx2 | no shared, routed 9/8 d | 1.8e-3 | 2.6509 | 0.00198 | 2.6507 | 3 |
| 275M | Cx4 | baseline 48E/top4 | 1.5e-3 | 2.5611 | 0.00156 | 2.5611 | 7 |
| 275M | Cx4 | no shared, routed 9/8 d | 1.6e-3 | 2.5601 | 0.00148 | 2.5600 | 3 |
| 275M | Cx8 | baseline 48E/top4 | 1.6e-3 | 2.4864 | 0.00136 | 2.4860 | 8 |
| 275M | Cx8 | no shared, routed 9/8 d | 1.6e-3 | 2.4842 | 0.00148 | 2.4841 | 3 |
| 480M | Cx1 | baseline 48E/top4 | 1.2e-3 | 2.5636 | 0.000958 | 2.5624 | 3 |
| 480M | Cx1 | no shared, routed 9/8 d | 1.2e-3 | 2.5850 |  |  | 1 |
| 480M | Cx2 | baseline 48E/top4 (b384k) | 9e-4 | 2.4630 | 0.000973 | 2.4629 | 3 |
| 480M | Cx2 | no shared, routed 9/8 d | 9e-4 | 2.4803 |  |  | 1 |
| 480M | Cx4 | baseline 48E/top4 | 8e-4 | 2.3788 | 0.00085 | 2.3787 | 4 |
| 480M | Cx4 | no shared, routed 9/8 d | 8e-4 | 2.3940 |  |  | 1 |
| 480M | Cx8 | baseline 48E/top4 | 8e-4 | 2.3076 | 0.00077 | 2.3076 | 4 |
| 480M | Cx8 | no shared, routed 9/8 d | 8e-4 | 2.3190 |  |  | 1 |
| 810M | Cx1 | baseline 48E/top4 | 6e-4 | 2.4104 | 0.000672 | 2.4102 | 6 |
| 810M | Cx1 | no shared, routed 9/8 d | 6e-4 | 2.4129 |  |  | 1 |
| 810M | Cx2 | baseline 48E/top4 (b384k) | 5.6e-4 | 2.3204 | 0.000629 | 2.3201 | 3 |
| 810M | Cx2 | no shared, routed 9/8 d | 5.6e-4 | 2.3242 |  |  | 1 |
| 810M | Cx4 | baseline 48E/top4 | 4e-4 | 2.2424 | 0.000514 | 2.2412 | 4 |
| 810M | Cx4 | no shared, routed 9/8 d | 4e-4 | 2.2466 |  |  | 1 |
| 810M | Cx8 | baseline 48E/top4 | 4e-4 | 2.1721 | 0.000467 | 2.1717 | 3 |
| 810M | Cx8 | no shared, routed 9/8 d | 4e-4 | 2.1746 |  |  | 1 |
| 1.2B | Cx1 | baseline 48E/top4 | 4e-4 | 2.3108 | 0.000483 | 2.3101 | 4 |
| 1.2B | Cx1 | no shared, routed 9/8 d | 4e-4 | 2.3184 |  |  | 1 |
| 1.2B | Cx2 | baseline 48E/top4 (b384k) | 6e-4 | 2.2229 | 0.000442 | 2.2204 | 4 |
| 1.2B | Cx2 | no shared, routed 9/8 d | 6e-4 | 2.2298 |  |  | 1 |
| 1.2B | Cx4 | baseline 48E/top4 | 3e-4 | 2.1508 | 0.000366 | 2.1500 | 4 |
| 1.2B | Cx4 | no shared, routed 9/8 d | 3e-4 | 2.1564 |  |  | 1 |
| 1.2B | Cx8 | baseline 48E/top4 | 4e-4 | 2.0835 | 0.000348 | 2.0831 | 3 |
| 1.2B | Cx8 | no shared, routed 9/8 d | 4e-4 | 2.0881 |  |  | 1 |

## Dense Schedule

| model | Cx | variant | best observed LR | best avg250M | fit LR | fit avg250M | points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 275M | Cx1 | baseline dense1 + shared | 2e-3 | 2.7767 | 0.00203 | 2.7766 | 8 |
| 275M | Cx1 | dense0 + shared | 2e-3 | 2.7651 | 0.00174 | 2.7648 | 3 |
| 275M | Cx1 | dense2 + shared | 2e-3 | 2.7727 | 0.00274 | 2.7718 | 3 |
| 275M | Cx1 | dense4 + shared | 2e-3 | 2.7787 | 0.002 | 2.7787 | 3 |
| 275M | Cx2 | baseline dense1 + shared (b384k) | 1.8e-3 | 2.6541 | 0.00178 | 2.6541 | 3 |
| 275M | Cx2 | dense0 + shared | 1.8e-3 | 2.6439 | 0.00154 | 2.6434 | 3 |
| 275M | Cx2 | dense2 + shared | 1.8e-3 | 2.6466 | 0.0018 | 2.6466 | 3 |
| 275M | Cx2 | dense4 + shared | 1.8e-3 | 2.6599 | 0.00188 | 2.6598 | 3 |
| 275M | Cx4 | baseline dense1 + shared | 1.5e-3 | 2.5611 | 0.00156 | 2.5611 | 7 |
| 275M | Cx4 | dense0 + shared | 1.6e-3 | 2.5550 | 0.00132 | 2.5545 | 3 |
| 275M | Cx4 | dense2 + shared | 1.6e-3 | 2.5626 | 0.00166 | 2.5626 | 3 |
| 275M | Cx4 | dense4 + shared | 1.6e-3 | 2.5748 | 0.0016 | 2.5748 | 3 |
| 275M | Cx8 | baseline dense1 + shared | 1.6e-3 | 2.4864 | 0.00136 | 2.4860 | 8 |
| 275M | Cx8 | dense0 + shared | 1.6e-3 | 2.4781 | 0.00128 | 2.4771 | 3 |
| 275M | Cx8 | dense2 + shared | 1.6e-3 | 2.4874 | 0.00143 | 2.4872 | 3 |
| 275M | Cx8 | dense4 + shared | 1.6e-3 | 2.5040 | 0.00146 | 2.5039 | 3 |
| 480M | Cx1 | baseline dense1 + shared | 1.2e-3 | 2.5636 | 0.000958 | 2.5624 | 3 |
| 480M | Cx1 | dense0 + shared | 1.2e-3 | 2.5675 |  |  | 1 |
| 480M | Cx1 | dense2 + shared | 1.2e-3 | 2.5723 |  |  | 1 |
| 480M | Cx1 | dense4 + shared | 1.2e-3 | 2.5828 |  |  | 1 |
| 480M | Cx2 | baseline dense1 + shared (b384k) | 9e-4 | 2.4630 | 0.000973 | 2.4629 | 3 |
| 480M | Cx2 | dense0 + shared | 9e-4 | 2.4678 |  |  | 1 |
| 480M | Cx2 | dense2 + shared | 9e-4 | 2.4768 |  |  | 1 |
| 480M | Cx2 | dense4 + shared | 9e-4 | 2.4824 |  |  | 1 |
| 480M | Cx4 | baseline dense1 + shared | 8e-4 | 2.3788 | 0.00085 | 2.3787 | 4 |
| 480M | Cx4 | dense0 + shared | 8e-4 | 2.3823 |  |  | 1 |
| 480M | Cx4 | dense2 + shared | 8e-4 | 2.3934 |  |  | 1 |
| 480M | Cx4 | dense4 + shared | 8e-4 | 2.3978 |  |  | 1 |
| 480M | Cx8 | baseline dense1 + shared | 8e-4 | 2.3076 | 0.00077 | 2.3076 | 4 |
| 480M | Cx8 | dense0 + shared | 8e-4 | 2.3136 |  |  | 1 |
| 480M | Cx8 | dense2 + shared | 8e-4 | 2.3221 |  |  | 1 |
| 480M | Cx8 | dense4 + shared | 8e-4 | 2.3317 |  |  | 1 |
| 810M | Cx1 | baseline dense1 + shared | 6e-4 | 2.4104 | 0.000672 | 2.4102 | 6 |
| 810M | Cx1 | dense0 + shared | 6e-4 | 2.4061 |  |  | 1 |
| 810M | Cx1 | dense2 + shared | 6e-4 | 2.4107 |  |  | 1 |
| 810M | Cx1 | dense4 + shared | 6e-4 | 2.4116 |  |  | 1 |
| 810M | Cx2 | baseline dense1 + shared (b384k) | 5.6e-4 | 2.3204 | 0.000629 | 2.3201 | 3 |
| 810M | Cx2 | dense0 + shared | 5.6e-4 | 2.3159 |  |  | 1 |
| 810M | Cx2 | dense2 + shared | 5.6e-4 | 2.3213 |  |  | 1 |
| 810M | Cx2 | dense4 + shared | 5.6e-4 | 2.3251 |  |  | 1 |
| 810M | Cx4 | baseline dense1 + shared | 4e-4 | 2.2424 | 0.000514 | 2.2412 | 4 |
| 810M | Cx4 | dense0 + shared | 4e-4 | 2.2402 |  |  | 1 |
| 810M | Cx4 | dense2 + shared | 4e-4 | 2.2444 |  |  | 1 |
| 810M | Cx4 | dense4 + shared | 4e-4 | 2.2522 |  |  | 1 |
| 810M | Cx8 | baseline dense1 + shared | 4e-4 | 2.1721 | 0.000467 | 2.1717 | 3 |
| 810M | Cx8 | dense0 + shared | 4e-4 | 2.1716 |  |  | 1 |
| 810M | Cx8 | dense2 + shared | 4e-4 | 2.1741 |  |  | 1 |
| 810M | Cx8 | dense4 + shared | 4e-4 | 2.1825 |  |  | 1 |
| 1.2B | Cx1 | baseline dense1 + shared | 4e-4 | 2.3108 | 0.000483 | 2.3101 | 4 |
| 1.2B | Cx1 | dense0 + shared | 4e-4 | 2.3126 |  |  | 1 |
| 1.2B | Cx1 | dense2 + shared | 4e-4 | 2.3154 |  |  | 1 |
| 1.2B | Cx1 | dense4 + shared | 4e-4 | 2.3155 |  |  | 1 |
| 1.2B | Cx2 | baseline dense1 + shared (b384k) | 6e-4 | 2.2229 | 0.000442 | 2.2204 | 4 |
| 1.2B | Cx2 | dense0 + shared | 6e-4 | 2.2253 |  |  | 1 |
| 1.2B | Cx2 | dense2 + shared | 6e-4 | 2.2284 |  |  | 1 |
| 1.2B | Cx2 | dense4 + shared | 6e-4 | 2.2289 |  |  | 1 |
| 1.2B | Cx4 | baseline dense1 + shared | 3e-4 | 2.1508 | 0.000366 | 2.1500 | 4 |
| 1.2B | Cx4 | dense0 + shared | 3e-4 | 2.1495 |  |  | 1 |
| 1.2B | Cx4 | dense2 + shared | 3e-4 | 2.1557 |  |  | 1 |
| 1.2B | Cx4 | dense4 + shared | 3e-4 | 2.1568 |  |  | 1 |
| 1.2B | Cx8 | baseline dense1 + shared | 4e-4 | 2.0835 | 0.000348 | 2.0831 | 3 |
| 1.2B | Cx8 | dense0 + shared | 4e-4 | 2.0809 |  |  | 1 |
| 1.2B | Cx8 | dense2 + shared | 4e-4 | 2.0880 |  |  | 1 |
| 1.2B | Cx8 | dense4 + shared | 4e-4 | 2.0923 |  |  | 1 |

## Qwen3-Like

| model | Cx | variant | best observed LR | best avg250M | fit LR | fit avg250M | points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 275M | Cx1 | Qwen-like active matched 4.5d | 2e-3 | 2.7520 | 0.00185 | 2.7519 | 3 |
| 275M | Cx1 | baseline 48E/top4 | 2e-3 | 2.7767 | 0.00203 | 2.7766 | 8 |
| 275M | Cx1 | Qwen-like true 3.0d + depth | 2e-3 | 2.7461 | 0.00148 | 2.7453 | 4 |
| 275M | Cx2 | Qwen-like active matched 4.5d | 1.8e-3 | 2.6244 | 0.00195 | 2.6243 | 3 |
| 275M | Cx2 | baseline 48E/top4 (b384k) | 1.8e-3 | 2.6541 | 0.00178 | 2.6541 | 3 |
| 275M | Cx2 | Qwen-like true 3.0d + depth | 1.8e-3 | 2.6163 | 0.00144 | 2.6157 | 4 |
| 275M | Cx4 | Qwen-like active matched 4.5d | 1.6e-3 | 2.5347 | 0.00145 | 2.5345 | 3 |
| 275M | Cx4 | baseline 48E/top4 | 1.5e-3 | 2.5611 | 0.00156 | 2.5611 | 7 |
| 275M | Cx4 | Qwen-like true 3.0d + depth | 1.6e-3 | 2.5193 | 0.0013 | 2.5185 | 4 |
| 275M | Cx8 | Qwen-like active matched 4.5d | 1.6e-3 | 2.4564 | 0.00133 | 2.4558 | 3 |
| 275M | Cx8 | baseline 48E/top4 | 1.6e-3 | 2.4864 | 0.00136 | 2.4860 | 8 |
| 275M | Cx8 | Qwen-like true 3.0d + depth | 1.6e-3 | 2.4423 | 0.00128 | 2.4413 | 4 |
| 480M | Cx1 | Qwen-like active matched 4.5d | 1.2e-3 | 2.5625 |  |  | 1 |
| 480M | Cx1 | baseline 48E/top4 | 1.2e-3 | 2.5636 | 0.000958 | 2.5624 | 3 |
| 480M | Cx1 | Qwen-like true 3.0d + depth | 1.2e-3 | 2.5508 |  |  | 1 |
| 480M | Cx2 | Qwen-like active matched 4.5d | 9e-4 | 2.4514 |  |  | 1 |
| 480M | Cx2 | baseline 48E/top4 (b384k) | 9e-4 | 2.4630 | 0.000973 | 2.4629 | 3 |
| 480M | Cx2 | Qwen-like true 3.0d + depth | 9e-4 | 2.4353 |  |  | 1 |
| 480M | Cx4 | Qwen-like active matched 4.5d | 8e-4 | 2.3627 |  |  | 1 |
| 480M | Cx4 | baseline 48E/top4 | 8e-4 | 2.3788 | 0.00085 | 2.3787 | 4 |
| 480M | Cx4 | Qwen-like true 3.0d + depth | 8e-4 | 2.3507 |  |  | 1 |
| 480M | Cx8 | Qwen-like active matched 4.5d | 8e-4 | 2.2868 |  |  | 1 |
| 480M | Cx8 | baseline 48E/top4 | 8e-4 | 2.3076 | 0.00077 | 2.3076 | 4 |
| 480M | Cx8 | Qwen-like true 3.0d + depth | 8e-4 | 2.2730 |  |  | 1 |
| 810M | Cx1 | Qwen-like active matched 4.5d | 6e-4 | 2.3932 |  |  | 1 |
| 810M | Cx1 | baseline 48E/top4 | 6e-4 | 2.4104 | 0.000672 | 2.4102 | 6 |
| 810M | Cx1 | Qwen-like true 3.0d + depth | 6e-4 | 2.3865 |  |  | 1 |
| 810M | Cx2 | Qwen-like active matched 4.5d | 5.6e-4 | 2.2977 |  |  | 1 |
| 810M | Cx2 | baseline 48E/top4 (b384k) | 5.6e-4 | 2.3204 | 0.000629 | 2.3201 | 3 |
| 810M | Cx2 | Qwen-like true 3.0d + depth | 5.6e-4 | 2.2917 |  |  | 1 |
| 810M | Cx4 | Qwen-like active matched 4.5d | 4e-4 | 2.2163 |  |  | 1 |
| 810M | Cx4 | baseline 48E/top4 | 4e-4 | 2.2424 | 0.000514 | 2.2412 | 4 |
| 810M | Cx4 | Qwen-like true 3.0d + depth | 4e-4 | 2.2148 |  |  | 1 |
| 810M | Cx8 | Qwen-like active matched 4.5d | 4e-4 | 2.1556 |  |  | 1 |
| 810M | Cx8 | baseline 48E/top4 | 4e-4 | 2.1721 | 0.000467 | 2.1717 | 3 |
| 810M | Cx8 | Qwen-like true 3.0d + depth | 4e-4 | 2.1392 |  |  | 1 |
| 1.2B | Cx1 | Qwen-like active matched 4.5d | 4e-4 | 2.2964 |  |  | 1 |
| 1.2B | Cx1 | baseline 48E/top4 | 4e-4 | 2.3108 | 0.000483 | 2.3101 | 4 |
| 1.2B | Cx1 | Qwen-like true 3.0d + depth | 4e-4 | 2.2926 |  |  | 1 |
| 1.2B | Cx2 | Qwen-like active matched 4.5d | 6e-4 | 2.2084 |  |  | 1 |
| 1.2B | Cx2 | baseline 48E/top4 (b384k) | 6e-4 | 2.2229 | 0.000442 | 2.2204 | 4 |
| 1.2B | Cx2 | Qwen-like true 3.0d + depth | 6e-4 | 2.1963 |  |  | 1 |
| 1.2B | Cx4 | Qwen-like active matched 4.5d | 3e-4 | 2.1319 |  |  | 1 |
| 1.2B | Cx4 | baseline 48E/top4 | 3e-4 | 2.1508 | 0.000366 | 2.1500 | 4 |
| 1.2B | Cx4 | Qwen-like true 3.0d + depth | 3e-4 | 2.1196 |  |  | 1 |
| 1.2B | Cx8 | Qwen-like active matched 4.5d | 4e-4 | 2.0544 |  |  | 1 |
| 1.2B | Cx8 | baseline 48E/top4 | 4e-4 | 2.0835 | 0.000348 | 2.0831 | 3 |
| 1.2B | Cx8 | Qwen-like true 3.0d + depth | 4e-4 | 2.0555 |  |  | 1 |

## Integration Candidates

| model | Cx | variant | best observed LR | best avg250M | fit LR | fit avg250M | points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 275M | Cx1 | baseline 48E/top4 | 2e-3 | 2.7767 | 0.00203 | 2.7766 | 8 |
| 275M | Cx1 | integration deep 256E/top8 | 1.6e-3 | 2.7396 | 0.00132 | 2.7388 | 3 |
| 275M | Cx1 | integration wide 256E/top8 | 1.6e-3 | 2.7410 | 0.0016 | 2.7410 | 3 |
| 275M | Cx2 | baseline 48E/top4 (b384k) | 1.8e-3 | 2.6541 | 0.00178 | 2.6541 | 3 |
| 275M | Cx2 | integration deep 256E/top8 | 1.6e-3 | 2.6023 | 0.00147 | 2.6022 | 3 |
| 275M | Cx2 | integration wide 256E/top8 | 1.6e-3 | 2.6083 | 0.00149 | 2.6082 | 3 |
| 275M | Cx4 | baseline 48E/top4 | 1.5e-3 | 2.5611 | 0.00156 | 2.5611 | 7 |
| 275M | Cx4 | integration deep 256E/top8 | 1.6e-3 | 2.5097 | 0.00132 | 2.5090 | 3 |
| 275M | Cx4 | integration wide 256E/top8 | 8e-4 | 2.5060 | 0.00104 | 2.5049 | 4 |
| 275M | Cx8 | baseline 48E/top4 | 1.6e-3 | 2.4864 | 0.00136 | 2.4860 | 8 |
| 275M | Cx8 | integration deep 256E/top8 | 1.6e-3 | 2.4171 | 0.00122 | 2.4160 | 3 |
| 275M | Cx8 | integration wide 256E/top8 | 8e-4 | 2.4193 | 0.000988 | 2.4183 | 4 |
| 480M | Cx1 | baseline 48E/top4 | 1.2e-3 | 2.5636 | 0.000958 | 2.5624 | 3 |
| 480M | Cx1 | integration deep 256E/top8 | 1.2e-3 | 2.5291 |  |  | 1 |
| 480M | Cx1 | integration wide 256E/top8 | 1.2e-3 | 2.5433 |  |  | 1 |
| 480M | Cx2 | baseline 48E/top4 (b384k) | 9e-4 | 2.4630 | 0.000973 | 2.4629 | 3 |
| 480M | Cx2 | integration deep 256E/top8 | 9e-4 | 2.4091 |  |  | 1 |
| 480M | Cx2 | integration wide 256E/top8 | 9e-4 | 2.4239 |  |  | 1 |
| 480M | Cx4 | baseline 48E/top4 | 8e-4 | 2.3788 | 0.00085 | 2.3787 | 4 |
| 480M | Cx4 | integration deep 256E/top8 | 8e-4 | 2.3207 |  |  | 1 |
| 480M | Cx4 | integration wide 256E/top8 | 8e-4 | 2.3300 |  |  | 1 |
| 480M | Cx8 | baseline 48E/top4 | 8e-4 | 2.3076 | 0.00077 | 2.3076 | 4 |
| 480M | Cx8 | integration deep 256E/top8 | 8e-4 | 2.2380 |  |  | 1 |
| 480M | Cx8 | integration wide 256E/top8 | 8e-4 | 2.2513 |  |  | 1 |
| 810M | Cx1 | baseline 48E/top4 | 6e-4 | 2.4104 | 0.000672 | 2.4102 | 6 |
| 810M | Cx1 | integration deep 256E/top8 | 6e-4 | 2.3713 |  |  | 1 |
| 810M | Cx1 | integration wide 256E/top8 | 6e-4 | 2.3732 |  |  | 1 |
| 810M | Cx2 | baseline 48E/top4 (b384k) | 5.6e-4 | 2.3204 | 0.000629 | 2.3201 | 3 |
| 810M | Cx2 | integration deep 256E/top8 | 5.6e-4 | 2.2740 |  |  | 1 |
| 810M | Cx2 | integration wide 256E/top8 | 5.6e-4 | 2.2689 |  |  | 1 |
| 810M | Cx4 | baseline 48E/top4 | 4e-4 | 2.2424 | 0.000514 | 2.2412 | 4 |
| 810M | Cx4 | integration deep 256E/top8 | 4e-4 | 2.1853 |  |  | 1 |
| 810M | Cx4 | integration wide 256E/top8 | 4e-4 | 2.1928 |  |  | 1 |
| 810M | Cx8 | baseline 48E/top4 | 4e-4 | 2.1721 | 0.000467 | 2.1717 | 3 |
| 810M | Cx8 | integration deep 256E/top8 | 4e-4 | 2.1183 |  |  | 1 |
| 810M | Cx8 | integration wide 256E/top8 | 4e-4 | 2.1049 |  |  | 1 |
| 1.2B | Cx1 | baseline 48E/top4 | 4e-4 | 2.3108 | 0.000483 | 2.3101 | 4 |
| 1.2B | Cx1 | integration deep 256E/top8 | 4e-4 | 2.2703 |  |  | 1 |
| 1.2B | Cx1 | integration wide 256E/top8 | 4e-4 | 2.2731 |  |  | 1 |
| 1.2B | Cx2 | baseline 48E/top4 (b384k) | 6e-4 | 2.2229 | 0.000442 | 2.2204 | 4 |
| 1.2B | Cx2 | integration deep 256E/top8 | 6e-4 | 2.1788 |  |  | 1 |
| 1.2B | Cx2 | integration wide 256E/top8 | 6e-4 | 2.1783 |  |  | 1 |
| 1.2B | Cx4 | baseline 48E/top4 | 3e-4 | 2.1508 | 0.000366 | 2.1500 | 4 |
| 1.2B | Cx4 | integration deep 256E/top8 | 3e-4 | 2.0944 |  |  | 1 |
| 1.2B | Cx4 | integration wide 256E/top8 | 3e-4 | 2.0942 |  |  | 1 |
| 1.2B | Cx8 | baseline 48E/top4 | 4e-4 | 2.0835 | 0.000348 | 2.0831 | 3 |
| 1.2B | Cx8 | integration deep 256E/top8 | 4e-4 | 2.0239 |  |  | 1 |
| 1.2B | Cx8 | integration wide 256E/top8 | 4e-4 | 2.0226 |  |  | 1 |
