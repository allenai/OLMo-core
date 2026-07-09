# In-Loop Eval Results

Generated: 2026-07-09 17:46 UTC

Values are the latest W&B summary values for each selected run. Higher is better for accuracy/F1/pass-style metrics; lower is better for BPB/loss/perplexity-style metrics. Values marked `see metric` need manual interpretation.

Selected states: `finished`. Cached finished-run summaries under `/weka/oe-adapt-default/jacobm/olmoe3/OLMo-core/src/scripts/train/jacobm_olmoe_ladder/results/cache/wandb_summaries`.
Runs with no matching eval metrics skipped: 54.

## 1p2b Cx1

| metric | direction | eg-1p2b-cx1-eg24e2k-lr4e-4-r1<br>`2ydaihvz` | eg-1p2b-cx1-eg96e8k-lr4e-4-r1<br>`dtc7utn9` | int-1p2b-cx1-intd256e8k-lr4e-4-r2<br>`ey4z00m3` | int-1p2b-cx1-intw256e8k-lr4e-4-r2<br>`hww8eksq` | 1p2b-cx1-b256k-lr1e-4-r1<br>`tvx71brh` | 1p2b-cx1-b256k-lr2e-4-r1<br>`ehcm9znb` | 1p2b-cx1-b256k-lr4e-4-r1<br>`r9esbx26` | 1p2b-cx1-b256k-lr8e-4-r1<br>`eiuofxc6` | q3-1p2b-cx1-q3am128e8k-lr4e-4-r1<br>`a8elbdls` | q3-1p2b-cx1-q3td128e8k-lr4e-4-r1<br>`w5lw7mxb` | se-1p2b-cx1-se0m9-lr4e-4-r1<br>`z3n03rt2` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.85850 | 0.84364 | 0.83239 | 0.83041 | 0.89187 | 0.86730 | 0.85827 | 0.85275 | 0.86760 | 0.84736 | 0.85188 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 0.93817 | 0.92345 | 0.91019 | 0.90711 | 0.97607 | 0.94961 | 0.94016 | 0.93443 | 0.94947 | 0.92728 | 0.93364 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0178 | 1.0181 | 1.0331 | 1.0208 | 1.0131 | 1.0161 | 1.0182 | 1.0129 | 1.0149 | 1.0072 | 1.0139 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0355 | 2.0362 | 2.0662 | 2.0415 | 2.0261 | 2.0322 | 2.0364 | 2.0258 | 2.0298 | 2.0143 | 2.0277 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.70557 | 0.70575 | 0.71603 | 0.70755 | 0.70231 | 0.70442 | 0.70579 | 0.70222 | 0.70356 | 0.69828 | 0.70281 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4111 | 1.4115 | 1.4321 | 1.4151 | 1.4046 | 1.4088 | 1.4116 | 1.4044 | 1.4071 | 1.3966 | 1.4056 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.23379 | 0.25000 | 0.25512 | 0.24915 | 0.24488 | 0.25597 | 0.26109 | 0.24659 | 0.24061 | 0.25768 | 0.24829 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.23379 | 0.25000 | 0.25512 | 0.24915 | 0.24488 | 0.25597 | 0.26109 | 0.24659 | 0.24061 | 0.25768 | 0.24829 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.4064 | -1.4059 | -1.4282 | -1.4109 | -1.3976 | -1.4026 | -1.4072 | -1.4018 | -1.4019 | -1.3935 | -1.4007 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.4064 | -1.4059 | -1.4282 | -1.4109 | -1.3976 | -1.4026 | -1.4072 | -1.4018 | -1.4019 | -1.3935 | -1.4007 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.24994 | 0.24980 | 0.24900 | 0.24934 | 0.25048 | 0.25172 | 0.25045 | 0.25010 | 0.25056 | 0.25032 | 0.25173 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.24994 | 0.24980 | 0.24900 | 0.24934 | 0.25048 | 0.25172 | 0.25045 | 0.25010 | 0.25056 | 0.25032 | 0.25173 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.66682 | 0.63536 | 0.62961 | 0.63430 | 0.69549 | 0.66595 | 0.65751 | 0.65683 | 0.65566 | 0.63690 | 0.64541 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.72604 | 0.69122 | 0.68500 | 0.68968 | 0.75658 | 0.72447 | 0.71538 | 0.71482 | 0.71317 | 0.69296 | 0.70243 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0103 | 1.0245 | 1.0322 | 1.0185 | 1.0122 | 1.0151 | 1.0180 | 1.0085 | 1.0133 | 1.0111 | 1.0147 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0207 | 2.0489 | 2.0644 | 2.0370 | 2.0245 | 2.0302 | 2.0360 | 2.0170 | 2.0265 | 2.0222 | 2.0295 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.70043 | 0.71016 | 0.71545 | 0.70600 | 0.70174 | 0.70371 | 0.70566 | 0.69913 | 0.70245 | 0.70094 | 0.70343 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4009 | 1.4203 | 1.4309 | 1.4120 | 1.4035 | 1.4074 | 1.4113 | 1.3983 | 1.4049 | 1.4019 | 1.4069 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.25337 | 0.24327 | 0.23948 | 0.24200 | 0.25000 | 0.26221 | 0.24411 | 0.26221 | 0.25337 | 0.24747 | 0.25000 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.25337 | 0.24327 | 0.23948 | 0.24200 | 0.25000 | 0.26221 | 0.24411 | 0.26221 | 0.25337 | 0.24747 | 0.25000 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.3957 | -1.4155 | -1.4269 | -1.4083 | -1.3973 | -1.4015 | -1.4074 | -1.3951 | -1.3995 | -1.3989 | -1.4003 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.3957 | -1.4155 | -1.4269 | -1.4083 | -1.3973 | -1.4015 | -1.4074 | -1.3951 | -1.3995 | -1.3989 | -1.4003 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.25031 | 0.24847 | 0.24859 | 0.25074 | 0.25069 | 0.25140 | 0.25054 | 0.25113 | 0.25126 | 0.24982 | 0.24998 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.25031 | 0.24847 | 0.24859 | 0.25074 | 0.25069 | 0.25140 | 0.25054 | 0.25113 | 0.25126 | 0.24982 | 0.24998 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 1.4161 | 1.4227 | 1.3589 | 1.2710 | 1.6059 | 1.5087 | 1.3628 | 1.4319 | 1.3421 | 1.2941 | 1.4470 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 2.2769 | 2.2968 | 2.1891 | 2.0435 | 2.5830 | 2.4358 | 2.2002 | 2.3061 | 2.1634 | 2.0912 | 2.3295 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 0.98166 | 0.98611 | 0.94182 | 0.88095 | 1.1130 | 1.0457 | 0.94457 | 0.99253 | 0.93031 | 0.89707 | 1.0030 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 1.5783 | 1.5920 | 1.5173 | 1.4164 | 1.7902 | 1.6883 | 1.5251 | 1.5985 | 1.4996 | 1.4496 | 1.6147 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.40401 | 0.39542 | 0.43457 | 0.44795 | 0.32187 | 0.36294 | 0.41261 | 0.41452 | 0.43553 | 0.43171 | 0.37345 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.40401 | 0.39542 | 0.43457 | 0.44795 | 0.32187 | 0.36294 | 0.41261 | 0.41452 | 0.43553 | 0.43171 | 0.37345 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -1.8862 | -1.8489 | -1.8390 | -1.6416 | -2.0726 | -1.9674 | -1.7678 | -1.8942 | -1.7669 | -1.6778 | -1.9377 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -1.8862 | -1.8489 | -1.8390 | -1.6416 | -2.0726 | -1.9674 | -1.7678 | -1.8942 | -1.7669 | -1.6778 | -1.9377 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.33972 | 0.35158 | 0.37987 | 0.38044 | 0.27615 | 0.33334 | 0.34293 | 0.34045 | 0.38560 | 0.38102 | 0.34550 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.33972 | 0.35158 | 0.37987 | 0.38044 | 0.27615 | 0.33334 | 0.34293 | 0.34045 | 0.38560 | 0.38102 | 0.34550 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.39477 | 0.35283 | 0.38210 | 0.35055 | 0.42133 | 0.40471 | 0.41603 | 0.40896 | 0.43654 | 0.41708 | 0.38506 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.42990 | 0.38379 | 0.41612 | 0.38133 | 0.45911 | 0.44036 | 0.45297 | 0.44567 | 0.47629 | 0.45522 | 0.41863 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.27365 | 0.24458 | 0.26487 | 0.24299 | 0.29205 | 0.28052 | 0.28838 | 0.28347 | 0.30259 | 0.28912 | 0.26692 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.29799 | 0.26604 | 0.28844 | 0.26434 | 0.31824 | 0.30525 | 0.31396 | 0.30893 | 0.33015 | 0.31554 | 0.29020 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.55237 | 0.61462 | 0.60079 | 0.61957 | 0.55040 | 0.56225 | 0.59091 | 0.61067 | 0.55336 | 0.57312 | 0.56917 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.55237 | 0.61462 | 0.60079 | 0.61957 | 0.55040 | 0.56225 | 0.59091 | 0.61067 | 0.55336 | 0.57312 | 0.56917 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -1.9021 | -1.7496 | -1.7466 | -1.6968 | -2.1067 | -2.0163 | -1.9004 | -1.7743 | -2.0148 | -2.0227 | -1.9481 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -1.9021 | -1.7496 | -1.7466 | -1.6968 | -2.1067 | -2.0163 | -1.9004 | -1.7743 | -2.0148 | -2.0227 | -1.9481 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.53804 | 0.58650 | 0.58004 | 0.59447 | 0.52553 | 0.53668 | 0.56176 | 0.57399 | 0.52640 | 0.54210 | 0.54683 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.53804 | 0.58650 | 0.58004 | 0.59447 | 0.52553 | 0.53668 | 0.56176 | 0.57399 | 0.52640 | 0.54210 | 0.54683 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.38973 | 0.38693 | 0.31908 | 0.33041 | 0.44514 | 0.42172 | 0.38069 | 0.36655 | 0.40509 | 0.43294 | 0.43033 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.47090 | 0.46727 | 0.38315 | 0.39797 | 0.53687 | 0.50740 | 0.45741 | 0.44147 | 0.48702 | 0.52181 | 0.51751 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.27028 | 0.26831 | 0.22133 | 0.22913 | 0.30866 | 0.29245 | 0.26401 | 0.25422 | 0.28087 | 0.30018 | 0.29842 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.32660 | 0.32404 | 0.26579 | 0.27601 | 0.37234 | 0.35193 | 0.31720 | 0.30618 | 0.33770 | 0.36181 | 0.35891 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.86017 | 0.87464 | 0.89971 | 0.89778 | 0.84957 | 0.85053 | 0.86500 | 0.88621 | 0.87464 | 0.89103 | 0.86403 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.86017 | 0.87464 | 0.89971 | 0.89778 | 0.84957 | 0.85053 | 0.86500 | 0.88621 | 0.87464 | 0.89103 | 0.86403 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.38368 | -0.36823 | -0.31031 | -0.31453 | -0.43067 | -0.40879 | -0.38528 | -0.34212 | -0.35791 | -0.35062 | -0.37546 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.38368 | -0.36823 | -0.31031 | -0.31453 | -0.43067 | -0.40879 | -0.38528 | -0.34212 | -0.35791 | -0.35062 | -0.37546 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.76719 | 0.78218 | 0.80843 | 0.80653 | 0.73965 | 0.76008 | 0.77046 | 0.79094 | 0.78204 | 0.78559 | 0.77360 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.76719 | 0.78218 | 0.80843 | 0.80653 | 0.73965 | 0.76008 | 0.77046 | 0.79094 | 0.78204 | 0.78559 | 0.77360 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.31850 | 0.29482 | 0.28149 | 0.28474 | 0.30151 | 0.27239 | 0.31300 | 0.26067 | 0.27468 | 0.31346 | 0.29570 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.32927 | 0.30486 | 0.29106 | 0.29427 | 0.31163 | 0.28153 | 0.32350 | 0.26943 | 0.28391 | 0.32406 | 0.30564 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.22080 | 0.20439 | 0.19513 | 0.19740 | 0.20902 | 0.18882 | 0.21699 | 0.18070 | 0.19041 | 0.21728 | 0.20497 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.22823 | 0.21131 | 0.20177 | 0.20399 | 0.21602 | 0.19515 | 0.22425 | 0.18676 | 0.19678 | 0.22465 | 0.21188 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.83989 | 0.86673 | 0.88551 | 0.90966 | 0.83005 | 0.86404 | 0.83721 | 0.85063 | 0.86315 | 0.87746 | 0.85868 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.83989 | 0.86673 | 0.88551 | 0.90966 | 0.83005 | 0.86404 | 0.83721 | 0.85063 | 0.86315 | 0.87746 | 0.85868 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.36559 | -0.35032 | -0.30489 | -0.29349 | -0.40571 | -0.36454 | -0.39869 | -0.42446 | -0.33547 | -0.31464 | -0.37004 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.36559 | -0.35032 | -0.30489 | -0.29349 | -0.40571 | -0.36454 | -0.39869 | -0.42446 | -0.33547 | -0.31464 | -0.37004 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.84368 | 0.85569 | 0.86126 | 0.87826 | 0.83012 | 0.84840 | 0.84240 | 0.84660 | 0.85308 | 0.86780 | 0.85660 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.84368 | 0.85569 | 0.86126 | 0.87826 | 0.83012 | 0.84840 | 0.84240 | 0.84660 | 0.85308 | 0.86780 | 0.85660 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 0.99403 | 0.80810 | 0.83263 | 0.87156 | 0.91937 | 0.80841 | 0.89609 | 0.86822 | 0.94961 | 0.87047 | 0.95791 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.6225 | 1.3248 | 1.3726 | 1.4377 | 1.4949 | 1.3300 | 1.4683 | 1.4271 | 1.5507 | 1.4212 | 1.5667 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.72164 | 0.59732 | 0.60926 | 0.63418 | 0.66504 | 0.59261 | 0.65474 | 0.63487 | 0.68951 | 0.63505 | 0.69860 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.2086 | 1.0130 | 1.0327 | 1.0731 | 1.1066 | 1.0033 | 1.1040 | 1.0737 | 1.1540 | 1.0664 | 1.1748 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.68914 | 0.70787 | 0.72097 | 0.70974 | 0.69288 | 0.70037 | 0.69101 | 0.70037 | 0.68914 | 0.70225 | 0.68914 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.68914 | 0.70787 | 0.72097 | 0.70974 | 0.69288 | 0.70037 | 0.69101 | 0.70037 | 0.68914 | 0.70225 | 0.68914 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.81382 | -0.76071 | -0.67850 | -0.67218 | -0.81685 | -0.73897 | -0.75795 | -0.75747 | -0.77408 | -0.74592 | -0.79307 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.81382 | -0.76071 | -0.67850 | -0.67218 | -0.81685 | -0.73897 | -0.75795 | -0.75747 | -0.77408 | -0.74592 | -0.79307 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.61605 | 0.64044 | 0.65328 | 0.65363 | 0.61102 | 0.63042 | 0.62302 | 0.62589 | 0.63189 | 0.64144 | 0.61046 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.61605 | 0.64044 | 0.65328 | 0.65363 | 0.61102 | 0.63042 | 0.62302 | 0.62589 | 0.63189 | 0.64144 | 0.61046 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.5304 | 1.4922 | 1.4421 | 1.4251 | 1.6967 | 1.5753 | 1.5725 | 1.5136 | 1.5746 | 1.4880 | 1.5284 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.1297 | 2.0656 | 2.0169 | 1.9868 | 2.3367 | 2.1779 | 2.1801 | 2.1067 | 2.1908 | 2.0651 | 2.1150 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.0608 | 1.0342 | 0.99956 | 0.98792 | 1.1761 | 1.0920 | 1.0900 | 1.0492 | 1.0914 | 1.0314 | 1.0594 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.4762 | 1.4318 | 1.3980 | 1.3773 | 1.6197 | 1.5096 | 1.5112 | 1.4603 | 1.5185 | 1.4314 | 1.4659 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.24856 | 0.26661 | 0.26661 | 0.27235 | 0.20919 | 0.24610 | 0.26497 | 0.27728 | 0.26087 | 0.27728 | 0.26087 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.24856 | 0.26661 | 0.26661 | 0.27235 | 0.20919 | 0.24610 | 0.26497 | 0.27728 | 0.26087 | 0.27728 | 0.26087 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -3.7421 | -3.3509 | -3.4037 | -3.5023 | -4.0719 | -3.9489 | -3.6598 | -3.4640 | -3.3869 | -3.5940 | -3.7231 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -3.7421 | -3.3509 | -3.4037 | -3.5023 | -4.0719 | -3.9489 | -3.6598 | -3.4640 | -3.3869 | -3.5940 | -3.7231 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.26366 | 0.28385 | 0.28897 | 0.29046 | 0.24297 | 0.26941 | 0.27488 | 0.28489 | 0.28364 | 0.28646 | 0.27391 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.26366 | 0.28385 | 0.28897 | 0.29046 | 0.24297 | 0.26941 | 0.27488 | 0.28489 | 0.28364 | 0.28646 | 0.27391 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.44047 | 0.42188 | 0.41831 | 0.42883 | 0.46244 | 0.44144 | 0.43884 | 0.43912 | 0.42633 | 0.43171 | 0.43051 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.44642 | 0.42732 | 0.42365 | 0.43458 | 0.46845 | 0.44714 | 0.44489 | 0.44495 | 0.43177 | 0.43753 | 0.43612 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.64395 | 0.63943 | 0.62939 | 0.62136 | 0.67038 | 0.65536 | 0.64888 | 0.63705 | 0.64756 | 0.64241 | 0.64907 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.64956 | 0.64504 | 0.63487 | 0.62668 | 0.67606 | 0.66096 | 0.65456 | 0.64252 | 0.65306 | 0.64782 | 0.65475 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.2361 | 2.3785 | 2.0192 | 1.9948 | 2.2928 | 2.2270 | 2.2301 | 2.0121 | 2.3386 | 2.4681 | 2.2638 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 4.4722 | 4.7570 | 4.0384 | 3.9897 | 4.5856 | 4.4541 | 4.4602 | 4.0242 | 4.6773 | 4.9362 | 4.5275 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.5504 | 1.6487 | 1.3995 | 1.3826 | 1.5889 | 1.5439 | 1.5463 | 1.3942 | 1.6207 | 1.7111 | 1.5692 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.1007 | 3.2973 | 2.7991 | 2.7652 | 3.1779 | 3.0878 | 3.0926 | 2.7884 | 3.2413 | 3.4221 | 3.1384 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.10000 | 0.10000 | 0.10000 | 0.09000 | 0.07000 | 0.08000 | 0.08000 | 0.10000 | 0.08000 | 0.08000 | 0.12000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.10000 | 0.10000 | 0.10000 | 0.09000 | 0.07000 | 0.08000 | 0.08000 | 0.10000 | 0.08000 | 0.08000 | 0.12000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -3.0927 | -3.2889 | -2.7921 | -2.7566 | -3.1674 | -3.0786 | -3.0871 | -2.7824 | -3.2369 | -3.4191 | -3.1324 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -3.0927 | -3.2889 | -2.7921 | -2.7566 | -3.1674 | -3.0786 | -3.0871 | -2.7824 | -3.2369 | -3.4191 | -3.1324 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09472 | 0.09648 | 0.09977 | 0.09762 | 0.09806 | 0.09726 | 0.09735 | 0.09798 | 0.09559 | 0.09735 | 0.10027 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09472 | 0.09648 | 0.09977 | 0.09762 | 0.09806 | 0.09726 | 0.09735 | 0.09798 | 0.09559 | 0.09735 | 0.10027 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.79536 | 0.79148 | 0.78261 | 0.78197 | 0.81028 | 0.79877 | 0.79712 | 0.79229 | 0.78830 | 0.78481 | 0.79217 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.80412 | 0.80024 | 0.79116 | 0.79054 | 0.81921 | 0.80775 | 0.80594 | 0.80096 | 0.79707 | 0.79347 | 0.80103 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.67181 | 0.65874 | 0.64422 | 0.64574 | 0.68841 | 0.67232 | 0.66728 | 0.66073 | 0.65865 | 0.65240 | 0.66954 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.67408 | 0.66103 | 0.64622 | 0.64785 | 0.69060 | 0.67456 | 0.66938 | 0.66296 | 0.66076 | 0.65459 | 0.67168 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.70109 | 0.68951 | 0.68403 | 0.69017 | 0.73763 | 0.71421 | 0.69709 | 0.70447 | 0.69910 | 0.70162 | 0.70175 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.73691 | 0.72431 | 0.71864 | 0.72564 | 0.77566 | 0.75066 | 0.73220 | 0.74042 | 0.73538 | 0.73715 | 0.73767 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0102 | 1.0165 | 1.0328 | 1.0138 | 1.0170 | 1.0143 | 1.0114 | 1.0157 | 1.0164 | 1.0111 | 1.0142 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0205 | 2.0330 | 2.0657 | 2.0276 | 2.0341 | 2.0285 | 2.0227 | 2.0313 | 2.0328 | 2.0221 | 2.0285 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.70037 | 0.70468 | 0.71590 | 0.70274 | 0.70500 | 0.70307 | 0.70111 | 0.70407 | 0.70462 | 0.70090 | 0.70312 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4007 | 1.4094 | 1.4318 | 1.4055 | 1.4100 | 1.4061 | 1.4022 | 1.4081 | 1.4092 | 1.4018 | 1.4062 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24251 | 0.25016 | 0.24952 | 0.26079 | 0.26334 | 0.26419 | 0.25526 | 0.25356 | 0.25271 | 0.24995 | 0.24421 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24251 | 0.25016 | 0.24952 | 0.26079 | 0.26334 | 0.26419 | 0.25526 | 0.25356 | 0.25271 | 0.24995 | 0.24421 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3880 | -1.3913 | -1.3979 | -1.3881 | -1.3888 | -1.3881 | -1.3872 | -1.3903 | -1.3896 | -1.3884 | -1.3887 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.3944 | -1.4045 | -1.4269 | -1.3991 | -1.4032 | -1.3993 | -1.3967 | -1.4046 | -1.4034 | -1.3974 | -1.3994 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.25014 | 0.24974 | 0.24936 | 0.25074 | 0.25092 | 0.25078 | 0.25083 | 0.25028 | 0.25049 | 0.25034 | 0.25040 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.25023 | 0.24950 | 0.24883 | 0.25150 | 0.25190 | 0.25161 | 0.25155 | 0.25048 | 0.25094 | 0.25067 | 0.25065 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 0.98469 | 0.95838 | 0.94613 | 0.95344 | 1.0124 | 0.98792 | 0.97254 | 0.96027 | 0.97222 | 0.95972 | 0.98023 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.0981 | 1.0671 | 1.0547 | 1.0631 | 1.1275 | 1.0996 | 1.0839 | 1.0705 | 1.0842 | 1.0688 | 1.0920 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0130 | 1.0077 | 1.0116 | 1.0162 | 1.0143 | 1.0193 | 1.0201 | 1.0140 | 1.0155 | 1.0074 | 1.0208 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0259 | 2.0155 | 2.0232 | 2.0325 | 2.0285 | 2.0385 | 2.0401 | 2.0280 | 2.0309 | 2.0149 | 2.0416 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.70221 | 0.69860 | 0.70123 | 0.70447 | 0.70308 | 0.70654 | 0.70709 | 0.70289 | 0.70391 | 0.69839 | 0.70761 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4044 | 1.3972 | 1.4025 | 1.4089 | 1.4062 | 1.4131 | 1.4142 | 1.4058 | 1.4078 | 1.3968 | 1.4152 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.23473 | 0.26373 | 0.27606 | 0.25941 | 0.25571 | 0.24830 | 0.25170 | 0.26218 | 0.25416 | 0.26126 | 0.24491 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.23473 | 0.26373 | 0.27606 | 0.25941 | 0.25571 | 0.24830 | 0.25170 | 0.26218 | 0.25416 | 0.26126 | 0.24491 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3882 | -1.3863 | -1.3859 | -1.3900 | -1.3887 | -1.3903 | -1.3916 | -1.3873 | -1.3875 | -1.3867 | -1.3917 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.3979 | -1.3924 | -1.3972 | -1.4041 | -1.4001 | -1.4063 | -1.4089 | -1.4015 | -1.4018 | -1.3921 | -1.4092 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25039 | 0.25078 | 0.25162 | 0.25039 | 0.25046 | 0.25050 | 0.25021 | 0.25143 | 0.25130 | 0.25054 | 0.25021 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25057 | 0.25155 | 0.25319 | 0.25079 | 0.25085 | 0.25083 | 0.25033 | 0.25272 | 0.25226 | 0.25108 | 0.25029 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.83599 | 0.82019 | 0.81076 | 0.81042 | 0.86368 | 0.84390 | 0.83053 | 0.82673 | 0.82728 | 0.82081 | 0.83401 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 0.89353 | 0.87501 | 0.86588 | 0.86552 | 0.92303 | 0.90152 | 0.88752 | 0.88348 | 0.88355 | 0.87672 | 0.89063 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0159 | 1.0122 | 1.0247 | 1.0154 | 1.0118 | 1.0272 | 1.0327 | 1.0293 | 1.0152 | 1.0032 | 1.0302 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0317 | 2.0243 | 2.0495 | 2.0309 | 2.0235 | 2.0543 | 2.0655 | 2.0586 | 2.0304 | 2.0065 | 2.0604 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.70426 | 0.70163 | 0.71035 | 0.70391 | 0.70138 | 0.71205 | 0.71587 | 0.71345 | 0.70377 | 0.69547 | 0.71414 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4085 | 1.4033 | 1.4207 | 1.4078 | 1.4028 | 1.4241 | 1.4317 | 1.4269 | 1.4075 | 1.3909 | 1.4283 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.23497 | 0.25772 | 0.23562 | 0.24017 | 0.26032 | 0.23009 | 0.22327 | 0.23042 | 0.25219 | 0.27949 | 0.22262 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.23497 | 0.25772 | 0.23562 | 0.24017 | 0.26032 | 0.23009 | 0.22327 | 0.23042 | 0.25219 | 0.27949 | 0.22262 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3916 | -1.3883 | -1.3952 | -1.3890 | -1.3868 | -1.3978 | -1.4018 | -1.3997 | -1.3897 | -1.3827 | -1.3996 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.4033 | -1.3989 | -1.4159 | -1.4034 | -1.3966 | -1.4180 | -1.4269 | -1.4230 | -1.4021 | -1.3869 | -1.4226 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.24944 | 0.25057 | 0.24925 | 0.25076 | 0.25108 | 0.24831 | 0.24748 | 0.24804 | 0.25027 | 0.25196 | 0.24797 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.24883 | 0.25109 | 0.24839 | 0.25136 | 0.25220 | 0.24671 | 0.24517 | 0.24623 | 0.25052 | 0.25405 | 0.24608 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.2661 | 1.2578 | 1.2460 | 1.2371 | 1.3224 | 1.2940 | 1.2797 | 1.2777 | 1.2581 | 1.2471 | 1.2912 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.5856 | 1.5709 | 1.5562 | 1.5416 | 1.6561 | 1.6210 | 1.6008 | 1.5986 | 1.5701 | 1.5611 | 1.6160 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0242 | 1.0081 | 1.0213 | 1.0110 | 1.0157 | 1.0190 | 1.0397 | 1.0192 | 1.0194 | 1.0056 | 1.0266 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0484 | 2.0161 | 2.0426 | 2.0220 | 2.0314 | 2.0380 | 2.0794 | 2.0385 | 2.0389 | 2.0111 | 2.0533 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.70994 | 0.69881 | 0.70793 | 0.70083 | 0.70409 | 0.70632 | 0.72070 | 0.70655 | 0.70663 | 0.69708 | 0.71169 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4199 | 1.3976 | 1.4159 | 1.4017 | 1.4082 | 1.4126 | 1.4414 | 1.4131 | 1.4133 | 1.3942 | 1.4234 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.23724 | 0.27899 | 0.26773 | 0.25348 | 0.25679 | 0.25514 | 0.23791 | 0.25381 | 0.26773 | 0.27899 | 0.24387 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.23724 | 0.27899 | 0.26773 | 0.25348 | 0.25679 | 0.25514 | 0.23791 | 0.25381 | 0.26773 | 0.27899 | 0.24387 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3934 | -1.3853 | -1.3920 | -1.3865 | -1.3855 | -1.3896 | -1.4023 | -1.3914 | -1.3891 | -1.3838 | -1.3953 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4125 | -1.3926 | -1.4098 | -1.3962 | -1.4014 | -1.4057 | -1.4352 | -1.4079 | -1.4062 | -1.3895 | -1.4181 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.24976 | 0.25131 | 0.25017 | 0.25115 | 0.25219 | 0.25084 | 0.24842 | 0.25022 | 0.25123 | 0.25167 | 0.24962 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.24944 | 0.25268 | 0.25040 | 0.25224 | 0.25407 | 0.25168 | 0.24702 | 0.25045 | 0.25250 | 0.25336 | 0.24928 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.42486 | 0.43200 | 0.41003 | 0.40701 | 0.43789 | 0.43059 | 0.44048 | 0.42923 | 0.42909 | 0.41290 | 0.42671 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.42731 | 0.43445 | 0.41235 | 0.40931 | 0.44038 | 0.43293 | 0.44293 | 0.43177 | 0.43157 | 0.41529 | 0.42922 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.31688 | 0.30116 | 0.31741 | 0.31131 | 0.32649 | 0.31705 | 0.33898 | 0.31360 | 0.32429 | 0.32769 | 0.32608 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.31810 | 0.30228 | 0.31868 | 0.31240 | 0.32775 | 0.31830 | 0.34036 | 0.31478 | 0.32554 | 0.32895 | 0.32734 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.54813 | 0.54326 | 0.54207 | 0.52913 | 0.57968 | 0.56931 | 0.57745 | 0.54837 | 0.57558 | 0.52848 | 0.58184 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.55192 | 0.54706 | 0.54592 | 0.53282 | 0.58379 | 0.57338 | 0.58169 | 0.55231 | 0.57985 | 0.53214 | 0.58583 |
| eval/lm/c4_en-validation/CE loss | lower | 2.9253 | 2.8983 | 2.8705 | 2.8735 | 2.9658 | 2.9312 | 2.9158 | 2.9175 | 2.9033 | 2.8916 | 2.9285 |
| eval/lm/c4_en-validation/PPL | lower | 18.64 | 18.14 | 17.65 | 17.70 | 19.41 | 18.75 | 18.46 | 18.49 | 18.23 | 18.02 | 18.70 |
| eval/lm/dolma_books-validation/CE loss | lower | 2.8229 | 2.7818 | 2.7542 | 2.7628 | 2.8660 | 2.8236 | 2.8059 | 2.8079 | 2.7952 | 2.7729 | 2.8216 |
| eval/lm/dolma_books-validation/PPL | lower | 16.82 | 16.15 | 15.71 | 15.84 | 17.57 | 16.84 | 16.54 | 16.57 | 16.37 | 16.01 | 16.80 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.0621 | 3.0336 | 3.0120 | 3.0151 | 3.0996 | 3.0667 | 3.0506 | 3.0529 | 3.0385 | 3.0275 | 3.0645 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 21.37 | 20.77 | 20.33 | 20.39 | 22.19 | 21.47 | 21.13 | 21.18 | 20.87 | 20.64 | 21.42 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.1222 | 2.1000 | 2.0826 | 2.0878 | 2.1581 | 2.1288 | 2.1155 | 2.1148 | 2.1082 | 2.0915 | 2.1207 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 8.3492 | 8.1658 | 8.0256 | 8.0674 | 8.6545 | 8.4047 | 8.2938 | 8.2876 | 8.2334 | 8.0971 | 8.3370 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.2367 | 3.2133 | 3.1897 | 3.1894 | 3.2775 | 3.2445 | 3.2260 | 3.2301 | 3.2164 | 3.2056 | 3.2384 |
| eval/lm/dolma_reddit-validation/PPL | lower | 25.45 | 24.86 | 24.28 | 24.27 | 26.51 | 25.65 | 25.18 | 25.28 | 24.94 | 24.67 | 25.49 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.3626 | 1.3437 | 1.3218 | 1.3243 | 1.3968 | 1.3697 | 1.3537 | 1.3556 | 1.3502 | 1.3379 | 1.3641 |
| eval/lm/dolma_stack-validation/PPL | lower | 3.9065 | 3.8332 | 3.7501 | 3.7597 | 4.0421 | 3.9340 | 3.8716 | 3.8791 | 3.8583 | 3.8111 | 3.9121 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.5723 | 2.5465 | 2.5169 | 2.5187 | 2.6110 | 2.5766 | 2.5637 | 2.5644 | 2.5521 | 2.5374 | 2.5734 |
| eval/lm/dolma_wiki-validation/PPL | lower | 13.10 | 12.76 | 12.39 | 12.41 | 13.61 | 13.15 | 12.98 | 12.99 | 12.83 | 12.65 | 13.11 |
| eval/lm/ice-validation/CE loss | lower | 2.9898 | 2.9589 | 2.9374 | 2.9549 | 3.0308 | 3.0005 | 2.9802 | 2.9710 | 2.9764 | 2.9561 | 2.9871 |
| eval/lm/ice-validation/PPL | lower | 19.88 | 19.28 | 18.87 | 19.20 | 20.71 | 20.10 | 19.69 | 19.51 | 19.62 | 19.22 | 19.83 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 3.0444 | 3.0124 | 2.9978 | 2.9937 | 3.0795 | 3.0442 | 3.0384 | 3.0286 | 3.0240 | 3.0142 | 3.0366 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 21.00 | 20.34 | 20.04 | 19.96 | 21.75 | 20.99 | 20.87 | 20.67 | 20.57 | 20.37 | 20.83 |
| eval/lm/pile-validation/CE loss | lower | 2.2149 | 2.1912 | 2.1673 | 2.1699 | 2.2493 | 2.2186 | 2.2041 | 2.2040 | 2.1965 | 2.1807 | 2.2136 |
| eval/lm/pile-validation/PPL | lower | 9.1607 | 8.9460 | 8.7350 | 8.7572 | 9.4811 | 9.1944 | 9.0621 | 9.0613 | 8.9931 | 8.8526 | 9.1485 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.5023 | 2.4670 | 2.4424 | 2.4490 | 2.5526 | 2.5140 | 2.4886 | 2.4939 | 2.4840 | 2.4592 | 2.5043 |
| eval/lm/wikitext_103-validation/PPL | lower | 12.21 | 11.79 | 11.50 | 11.58 | 12.84 | 12.35 | 12.04 | 12.11 | 11.99 | 11.70 | 12.24 |
| throughput/in-loop eval batches | see metric | 828.0 | 828.0 | 1659.0 | 1659.0 | 1659.0 | 1659.0 | 1659.0 | 1659.0 | 1659.0 | 1659.0 | 1659.0 |
| throughput/in-loop eval time (s) | see metric | 91.38 | 102.4 | 342.3 | 270.2 | 125.1 | 124.9 | 120.8 | 129.3 | 158.1 | 205.1 | 159.4 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| eg-1p2b-cx1-eg24e2k-lr4e-4-r1<br>`2ydaihvz` | finished | original | 21268004864.0 | 81131 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/2ydaihvz) |
| eg-1p2b-cx1-eg96e8k-lr4e-4-r1<br>`dtc7utn9` | finished | original | 21314404352.0 | 81308 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/dtc7utn9) |
| int-1p2b-cx1-intd256e8k-lr4e-4-r2<br>`ey4z00m3` | finished | original | 21445738496.0 | 81809 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ey4z00m3) |
| int-1p2b-cx1-intw256e8k-lr4e-4-r2<br>`hww8eksq` | finished | original | 21417426944.0 | 81701 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/hww8eksq) |
| 1p2b-cx1-b256k-lr1e-4-r1<br>`tvx71brh` | finished | gpu8-ep1mb2 | 21283471360.0 | 81190 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/tvx71brh) |
| 1p2b-cx1-b256k-lr2e-4-r1<br>`ehcm9znb` | finished | gpu8-ep1mb2 | 21283471360.0 | 81190 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ehcm9znb) |
| 1p2b-cx1-b256k-lr4e-4-r1<br>`r9esbx26` | finished | gpu8-ep1mb2 | 21283471360.0 | 81190 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/r9esbx26) |
| 1p2b-cx1-b256k-lr8e-4-r1<br>`eiuofxc6` | finished | gpu8-ep1mb2 | 21283471360.0 | 81190 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/eiuofxc6) |
| q3-1p2b-cx1-q3am128e8k-lr4e-4-r1<br>`a8elbdls` | finished | original | 21338783744.0 | 81401 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/a8elbdls) |
| q3-1p2b-cx1-q3td128e8k-lr4e-4-r1<br>`w5lw7mxb` | finished | original | 20989870080.0 | 80070 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/w5lw7mxb) |
| se-1p2b-cx1-se0m9-lr4e-4-r1<br>`z3n03rt2` | finished | original | 21283471360.0 | 81190 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/z3n03rt2) |

## 1p2b Cx2

| metric | direction | eg-1p2b-cx2-eg24e2k-lr3e-4-r1<br>`7lq6ag2s` | eg-1p2b-cx2-eg96e8k-lr3e-4-r1<br>`52j6x54s` | int-1p2b-cx2-intd256e8k-lr6e-4-r2<br>`xeiarhn7` | int-1p2b-cx2-intw256e8k-lr6e-4-r2<br>`jfwntmwm` | 1p2b-cx2-b384k-lr1.5e-4-r1<br>`dtd8qeiv` | 1p2b-cx2-b384k-lr2.4e-3-r1<br>`blpr9kqj` | 1p2b-cx2-b384k-lr3e-4-r1<br>`7cuo1d1i` | 1p2b-cx2-b384k-lr6e-4-r1<br>`54pt8zj7` | q3-1p2b-cx2-q3am128e8k-lr6e-4-r1<br>`f2gd9zv6` | q3-1p2b-cx2-q3td128e8k-lr6e-4-r1<br>`zoehfkg5` | se-1p2b-cx2-se0m9-lr6e-4-r1<br>`6v61syzf` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.83393 | 0.81258 | 0.78833 | 0.78972 | 0.83714 | 0.83958 | 0.81930 | 0.83371 | 0.79974 | 0.79797 | 0.82181 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 0.91132 | 0.88726 | 0.86063 | 0.86446 | 0.91743 | 0.91877 | 0.89638 | 0.91317 | 0.87562 | 0.87302 | 0.89841 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0039 | 1.0076 | 1.0068 | 1.0034 | 1.0149 | 1.0124 | 1.0136 | 1.0150 | 1.0306 | 1.0211 | 1.0237 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0078 | 2.0153 | 2.0135 | 2.0068 | 2.0298 | 2.0249 | 2.0272 | 2.0300 | 2.0611 | 2.0423 | 2.0474 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.69599 | 0.69857 | 0.69804 | 0.69562 | 0.70359 | 0.70183 | 0.70267 | 0.70366 | 0.71440 | 0.70781 | 0.70967 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.3920 | 1.3971 | 1.3961 | 1.3912 | 1.4072 | 1.4037 | 1.4053 | 1.4073 | 1.4288 | 1.4156 | 1.4193 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.25427 | 0.23208 | 0.24232 | 0.26109 | 0.23379 | 0.25256 | 0.26877 | 0.23038 | 0.22952 | 0.25000 | 0.26451 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.25427 | 0.23208 | 0.24232 | 0.26109 | 0.23379 | 0.25256 | 0.26877 | 0.23038 | 0.22952 | 0.25000 | 0.26451 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.3895 | -1.3950 | -1.3934 | -1.3887 | -1.4042 | -1.4003 | -1.4022 | -1.4044 | -1.4253 | -1.4121 | -1.4163 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.3895 | -1.3950 | -1.3934 | -1.3887 | -1.4042 | -1.4003 | -1.4022 | -1.4044 | -1.4253 | -1.4121 | -1.4163 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.25197 | 0.24987 | 0.24961 | 0.25080 | 0.24878 | 0.24985 | 0.25175 | 0.24861 | 0.24847 | 0.25008 | 0.25010 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.25197 | 0.24987 | 0.24961 | 0.25080 | 0.24878 | 0.24985 | 0.25175 | 0.24861 | 0.24847 | 0.25008 | 0.25010 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.63460 | 0.59893 | 0.58994 | 0.58713 | 0.64217 | 0.63225 | 0.62594 | 0.63599 | 0.60287 | 0.59751 | 0.62176 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.69034 | 0.65067 | 0.64164 | 0.63832 | 0.69841 | 0.68792 | 0.67993 | 0.69145 | 0.65559 | 0.64941 | 0.67597 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0058 | 1.0098 | 1.0056 | 1.0062 | 1.0074 | 1.0137 | 1.0158 | 1.0131 | 1.0225 | 1.0062 | 1.0256 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0116 | 2.0196 | 2.0113 | 2.0125 | 2.0148 | 2.0274 | 2.0317 | 2.0262 | 2.0449 | 2.0124 | 2.0511 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.69725 | 0.70001 | 0.69717 | 0.69764 | 0.69836 | 0.70272 | 0.70418 | 0.70230 | 0.70876 | 0.69749 | 0.71083 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.3945 | 1.4000 | 1.3943 | 1.3953 | 1.3967 | 1.4054 | 1.4084 | 1.4046 | 1.4175 | 1.3950 | 1.4217 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.26473 | 0.25000 | 0.25589 | 0.23527 | 0.25926 | 0.24874 | 0.24411 | 0.25589 | 0.25210 | 0.26684 | 0.24832 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.26473 | 0.25000 | 0.25589 | 0.23527 | 0.25926 | 0.24874 | 0.24411 | 0.25589 | 0.25210 | 0.26684 | 0.24832 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.3918 | -1.3978 | -1.3921 | -1.3931 | -1.3938 | -1.4014 | -1.4059 | -1.4019 | -1.4145 | -1.3924 | -1.4183 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.3918 | -1.3978 | -1.3921 | -1.3931 | -1.3938 | -1.4014 | -1.4059 | -1.4019 | -1.4145 | -1.3924 | -1.4183 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.25123 | 0.24993 | 0.25056 | 0.24960 | 0.25070 | 0.24990 | 0.25072 | 0.24949 | 0.25040 | 0.25191 | 0.25044 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.25123 | 0.24993 | 0.25056 | 0.24960 | 0.25070 | 0.24990 | 0.25072 | 0.24949 | 0.25040 | 0.25191 | 0.25044 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 1.2176 | 1.1689 | 1.0743 | 1.1009 | 1.1923 | 1.2150 | 1.1665 | 1.1084 | 1.1514 | 1.0889 | 1.1224 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 1.9469 | 1.8816 | 1.7283 | 1.7588 | 1.9206 | 1.9537 | 1.8833 | 1.7895 | 1.8648 | 1.7576 | 1.8074 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 0.84396 | 0.81009 | 0.74458 | 0.76307 | 0.82625 | 0.84211 | 0.80858 | 0.76827 | 0.79800 | 0.75475 | 0.77799 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 1.3496 | 1.3041 | 1.1979 | 1.2191 | 1.3311 | 1.3541 | 1.3055 | 1.2402 | 1.2926 | 1.2184 | 1.2529 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.47373 | 0.51480 | 0.53295 | 0.53582 | 0.49284 | 0.47851 | 0.49761 | 0.50621 | 0.51958 | 0.53295 | 0.52245 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.47373 | 0.51480 | 0.53295 | 0.53582 | 0.49284 | 0.47851 | 0.49761 | 0.50621 | 0.51958 | 0.53295 | 0.52245 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -1.6256 | -1.5289 | -1.4313 | -1.4643 | -1.5690 | -1.5165 | -1.5081 | -1.4880 | -1.4102 | -1.4564 | -1.3863 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -1.6256 | -1.5289 | -1.4313 | -1.4643 | -1.5690 | -1.5165 | -1.5081 | -1.4880 | -1.4102 | -1.4564 | -1.3863 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.43740 | 0.46858 | 0.49340 | 0.49476 | 0.44528 | 0.43061 | 0.45097 | 0.48027 | 0.48534 | 0.50293 | 0.48831 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.43740 | 0.46858 | 0.49340 | 0.49476 | 0.44528 | 0.43061 | 0.45097 | 0.48027 | 0.48534 | 0.50293 | 0.48831 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.35977 | 0.31131 | 0.35533 | 0.33018 | 0.36758 | 0.38363 | 0.35000 | 0.33782 | 0.36384 | 0.38449 | 0.33089 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.39231 | 0.33891 | 0.38772 | 0.35988 | 0.40078 | 0.41661 | 0.38163 | 0.36724 | 0.39631 | 0.42067 | 0.35975 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.24939 | 0.21581 | 0.24631 | 0.22887 | 0.25481 | 0.26589 | 0.24263 | 0.23416 | 0.25220 | 0.26649 | 0.22937 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.27193 | 0.23489 | 0.26876 | 0.24945 | 0.27781 | 0.28876 | 0.26451 | 0.25460 | 0.27468 | 0.29154 | 0.24937 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.61561 | 0.68478 | 0.66107 | 0.66502 | 0.62154 | 0.62451 | 0.64328 | 0.66107 | 0.65711 | 0.67391 | 0.63439 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.61561 | 0.68478 | 0.66107 | 0.66502 | 0.62154 | 0.62451 | 0.64328 | 0.66107 | 0.65711 | 0.67391 | 0.63439 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -1.5731 | -1.2418 | -1.3713 | -1.2691 | -1.4518 | -1.6379 | -1.4239 | -1.3726 | -1.3857 | -1.3078 | -1.4479 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -1.5731 | -1.2418 | -1.3713 | -1.2691 | -1.4518 | -1.6379 | -1.4239 | -1.3726 | -1.3857 | -1.3078 | -1.4479 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.59539 | 0.65470 | 0.64352 | 0.64426 | 0.59807 | 0.60357 | 0.62658 | 0.63588 | 0.63126 | 0.64296 | 0.61759 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.59539 | 0.65470 | 0.64352 | 0.64426 | 0.59807 | 0.60357 | 0.62658 | 0.63588 | 0.63126 | 0.64296 | 0.61759 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.39859 | 0.34148 | 0.25674 | 0.28046 | 0.33959 | 0.38708 | 0.31304 | 0.32444 | 0.31502 | 0.30320 | 0.35531 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.48161 | 0.41121 | 0.30823 | 0.33665 | 0.40921 | 0.46591 | 0.37744 | 0.39148 | 0.37937 | 0.36494 | 0.42766 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.27640 | 0.23685 | 0.17814 | 0.19454 | 0.23551 | 0.26844 | 0.21714 | 0.22502 | 0.21851 | 0.21034 | 0.24642 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.33404 | 0.28523 | 0.21386 | 0.23354 | 0.28385 | 0.32314 | 0.26180 | 0.27159 | 0.26319 | 0.25318 | 0.29662 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.89585 | 0.90550 | 0.92864 | 0.94214 | 0.90067 | 0.88525 | 0.90067 | 0.90453 | 0.90646 | 0.91803 | 0.90164 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.89585 | 0.90550 | 0.92864 | 0.94214 | 0.90067 | 0.88525 | 0.90067 | 0.90453 | 0.90646 | 0.91803 | 0.90164 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.34656 | -0.28321 | -0.24180 | -0.21052 | -0.32141 | -0.33598 | -0.29995 | -0.29772 | -0.27396 | -0.25269 | -0.29976 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.34656 | -0.28321 | -0.24180 | -0.21052 | -0.32141 | -0.33598 | -0.29995 | -0.29772 | -0.27396 | -0.25269 | -0.29976 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.79116 | 0.81581 | 0.84486 | 0.86800 | 0.79677 | 0.79959 | 0.81393 | 0.81595 | 0.83110 | 0.83642 | 0.81509 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.79116 | 0.81581 | 0.84486 | 0.86800 | 0.79677 | 0.79959 | 0.81393 | 0.81595 | 0.83110 | 0.83642 | 0.81509 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.29201 | 0.27934 | 0.26220 | 0.27372 | 0.29435 | 0.27051 | 0.26070 | 0.27533 | 0.30748 | 0.31256 | 0.28871 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.30192 | 0.28878 | 0.27117 | 0.28302 | 0.30418 | 0.27971 | 0.26944 | 0.28456 | 0.31773 | 0.32305 | 0.29836 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.20243 | 0.19364 | 0.18177 | 0.18975 | 0.20403 | 0.18751 | 0.18071 | 0.19085 | 0.21315 | 0.21668 | 0.20015 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.20928 | 0.20017 | 0.18797 | 0.19619 | 0.21086 | 0.19390 | 0.18677 | 0.19726 | 0.22026 | 0.22394 | 0.20683 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.93381 | 0.84615 | 0.90966 | 0.93560 | 0.84079 | 0.86315 | 0.90429 | 0.91860 | 0.88283 | 0.85868 | 0.90698 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.93381 | 0.84615 | 0.90966 | 0.93560 | 0.84079 | 0.86315 | 0.90429 | 0.91860 | 0.88283 | 0.85868 | 0.90698 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.20926 | -0.37647 | -0.25965 | -0.20679 | -0.38654 | -0.30906 | -0.23254 | -0.22676 | -0.30746 | -0.34265 | -0.25686 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.20926 | -0.37647 | -0.25965 | -0.20679 | -0.38654 | -0.30906 | -0.23254 | -0.22676 | -0.30746 | -0.34265 | -0.25686 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.90822 | 0.84910 | 0.88671 | 0.90354 | 0.84081 | 0.86023 | 0.88620 | 0.89480 | 0.86380 | 0.84668 | 0.87366 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.90822 | 0.84910 | 0.88671 | 0.90354 | 0.84081 | 0.86023 | 0.88620 | 0.89480 | 0.86380 | 0.84668 | 0.87366 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 0.88665 | 0.74936 | 0.75208 | 0.74521 | 0.84179 | 0.87999 | 0.79142 | 0.79014 | 0.74932 | 0.71710 | 0.83713 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.4744 | 1.2516 | 1.2600 | 1.2413 | 1.3890 | 1.4616 | 1.3083 | 1.3133 | 1.2369 | 1.1869 | 1.3883 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.64377 | 0.55034 | 0.55174 | 0.54679 | 0.61326 | 0.64293 | 0.57820 | 0.57998 | 0.54859 | 0.52624 | 0.61296 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.0969 | 0.94657 | 0.95100 | 0.93732 | 1.0388 | 1.0959 | 0.98261 | 0.99203 | 0.93203 | 0.89698 | 1.0457 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.73034 | 0.73408 | 0.78839 | 0.75468 | 0.73408 | 0.72846 | 0.70974 | 0.75094 | 0.75281 | 0.77528 | 0.73221 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.73034 | 0.73408 | 0.78839 | 0.75468 | 0.73408 | 0.72846 | 0.70974 | 0.75094 | 0.75281 | 0.77528 | 0.73221 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.69096 | -0.65334 | -0.57319 | -0.61638 | -0.67887 | -0.72237 | -0.67764 | -0.64741 | -0.64109 | -0.59934 | -0.68306 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.69096 | -0.65334 | -0.57319 | -0.61638 | -0.67887 | -0.72237 | -0.67764 | -0.64741 | -0.64109 | -0.59934 | -0.68306 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.64903 | 0.68022 | 0.69583 | 0.68553 | 0.67107 | 0.66276 | 0.65511 | 0.67887 | 0.68358 | 0.69268 | 0.65763 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.64903 | 0.68022 | 0.69583 | 0.68553 | 0.67107 | 0.66276 | 0.65511 | 0.67887 | 0.68358 | 0.69268 | 0.65763 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.4227 | 1.3449 | 1.3023 | 1.2227 | 1.4144 | 1.3369 | 1.4290 | 1.3765 | 1.3100 | 1.3710 | 1.4515 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 1.9855 | 1.8909 | 1.8312 | 1.7183 | 1.9762 | 1.8698 | 1.9916 | 1.9359 | 1.8350 | 1.8931 | 2.0318 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 0.98617 | 0.93222 | 0.90269 | 0.84746 | 0.98049 | 0.92665 | 0.99052 | 0.95410 | 0.90810 | 0.95033 | 1.0061 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.3762 | 1.3108 | 1.2693 | 1.1910 | 1.3699 | 1.2960 | 1.3803 | 1.3419 | 1.2720 | 1.3123 | 1.4085 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.26989 | 0.30927 | 0.31091 | 0.36587 | 0.26989 | 0.30681 | 0.30927 | 0.31911 | 0.36013 | 0.35275 | 0.29122 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.26989 | 0.30927 | 0.31091 | 0.36587 | 0.26989 | 0.30681 | 0.30927 | 0.31911 | 0.36013 | 0.35275 | 0.29122 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -3.3980 | -3.1039 | -2.7397 | -2.5351 | -3.4283 | -2.9335 | -3.0135 | -3.0550 | -2.9099 | -2.5538 | -3.2561 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -3.3980 | -3.1039 | -2.7397 | -2.5351 | -3.4283 | -2.9335 | -3.0135 | -3.0550 | -2.9099 | -2.5538 | -3.2561 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.29867 | 0.31762 | 0.33492 | 0.37344 | 0.28646 | 0.31510 | 0.31992 | 0.32433 | 0.35527 | 0.35164 | 0.29638 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.29867 | 0.31762 | 0.33492 | 0.37344 | 0.28646 | 0.31510 | 0.31992 | 0.32433 | 0.35527 | 0.35164 | 0.29638 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.40438 | 0.38863 | 0.39930 | 0.40520 | 0.40694 | 0.40681 | 0.39835 | 0.39126 | 0.39709 | 0.39039 | 0.39939 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.40971 | 0.39352 | 0.40473 | 0.41088 | 0.41233 | 0.41210 | 0.40373 | 0.39616 | 0.40228 | 0.39562 | 0.40455 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.61357 | 0.60155 | 0.58872 | 0.58785 | 0.60840 | 0.63328 | 0.61436 | 0.61171 | 0.60976 | 0.59351 | 0.60798 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.61909 | 0.60668 | 0.59390 | 0.59298 | 0.61346 | 0.63859 | 0.61973 | 0.61698 | 0.61515 | 0.59876 | 0.61304 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 1.9102 | 1.7295 | 1.8195 | 2.0612 | 1.9615 | 2.0334 | 1.8470 | 1.8693 | 2.1152 | 2.1448 | 2.2739 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 3.8205 | 3.4591 | 3.6389 | 4.1224 | 3.9230 | 4.0668 | 3.6941 | 3.7386 | 4.2305 | 4.2895 | 4.5477 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.3239 | 1.1992 | 1.2609 | 1.4286 | 1.3597 | 1.4099 | 1.2802 | 1.2957 | 1.4664 | 1.4864 | 1.5759 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 2.6477 | 2.3984 | 2.5218 | 2.8572 | 2.7195 | 2.8198 | 2.5603 | 2.5914 | 2.9328 | 2.9729 | 3.1519 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.10000 | 0.10000 | 0.09000 | 0.10000 | 0.11000 | 0.09000 | 0.07000 | 0.10000 | 0.10000 | 0.09000 | 0.11000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.10000 | 0.10000 | 0.09000 | 0.10000 | 0.11000 | 0.09000 | 0.07000 | 0.10000 | 0.10000 | 0.09000 | 0.11000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -2.6386 | -2.3891 | -2.5158 | -2.8528 | -2.7105 | -2.8125 | -2.5481 | -2.5799 | -2.9262 | -2.9702 | -3.1480 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -2.6386 | -2.3891 | -2.5158 | -2.8528 | -2.7105 | -2.8125 | -2.5481 | -2.5799 | -2.9262 | -2.9702 | -3.1480 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09724 | 0.10036 | 0.09806 | 0.09615 | 0.10078 | 0.09369 | 0.09708 | 0.09834 | 0.09937 | 0.09768 | 0.09486 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09724 | 0.10036 | 0.09806 | 0.09615 | 0.10078 | 0.09369 | 0.09708 | 0.09834 | 0.09937 | 0.09768 | 0.09486 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.77410 | 0.76372 | 0.75797 | 0.75759 | 0.77544 | 0.78879 | 0.76603 | 0.76735 | 0.77048 | 0.76600 | 0.76983 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.78263 | 0.77210 | 0.76634 | 0.76592 | 0.78403 | 0.79759 | 0.77453 | 0.77589 | 0.77898 | 0.77455 | 0.77842 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.62185 | 0.60851 | 0.59509 | 0.59136 | 0.62708 | 0.64526 | 0.61678 | 0.61171 | 0.60349 | 0.59384 | 0.61370 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.62371 | 0.61034 | 0.59703 | 0.59332 | 0.62906 | 0.64723 | 0.61874 | 0.61362 | 0.60540 | 0.59562 | 0.61564 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.66550 | 0.65581 | 0.63954 | 0.64144 | 0.66983 | 0.69037 | 0.66234 | 0.66110 | 0.66188 | 0.65454 | 0.66889 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.69874 | 0.68825 | 0.67082 | 0.67347 | 0.70316 | 0.72513 | 0.69485 | 0.69419 | 0.69481 | 0.68710 | 0.70225 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0086 | 1.0056 | 1.0138 | 1.0062 | 1.0147 | 1.0119 | 1.0178 | 1.0110 | 1.0241 | 1.0135 | 1.0160 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0172 | 2.0113 | 2.0277 | 2.0124 | 2.0294 | 2.0237 | 2.0356 | 2.0220 | 2.0482 | 2.0269 | 2.0321 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.69920 | 0.69721 | 0.70280 | 0.69756 | 0.70339 | 0.70144 | 0.70553 | 0.70088 | 0.70993 | 0.70248 | 0.70434 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.3984 | 1.3944 | 1.4056 | 1.3951 | 1.4068 | 1.4029 | 1.4111 | 1.4018 | 1.4199 | 1.4050 | 1.4087 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24718 | 0.25228 | 0.23974 | 0.25675 | 0.24463 | 0.24123 | 0.24931 | 0.24506 | 0.24251 | 0.26652 | 0.24676 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24718 | 0.25228 | 0.23974 | 0.25675 | 0.24463 | 0.24123 | 0.24931 | 0.24506 | 0.24251 | 0.26652 | 0.24676 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3879 | -1.3869 | -1.3907 | -1.3869 | -1.3896 | -1.3894 | -1.3909 | -1.3888 | -1.3935 | -1.3870 | -1.3908 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.3949 | -1.3907 | -1.4020 | -1.3926 | -1.4023 | -1.3965 | -1.4077 | -1.3984 | -1.4165 | -1.4015 | -1.4050 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.25024 | 0.25026 | 0.24976 | 0.25052 | 0.25031 | 0.24973 | 0.25034 | 0.25022 | 0.25005 | 0.25155 | 0.25007 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.25042 | 0.25052 | 0.24948 | 0.25108 | 0.25052 | 0.24946 | 0.25063 | 0.25037 | 0.25001 | 0.25305 | 0.25007 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 0.93639 | 0.91064 | 0.88089 | 0.88500 | 0.94019 | 0.95796 | 0.92190 | 0.92039 | 0.90716 | 0.90372 | 0.91693 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.0452 | 1.0157 | 0.98083 | 0.98801 | 1.0477 | 1.0671 | 1.0275 | 1.0251 | 1.0120 | 1.0069 | 1.0215 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0067 | 1.0017 | 1.0004 | 1.0003 | 1.0083 | 1.0064 | 1.0056 | 1.0071 | 1.0166 | 1.0189 | 1.0122 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0134 | 2.0034 | 2.0008 | 2.0006 | 2.0167 | 2.0128 | 2.0112 | 2.0141 | 2.0331 | 2.0378 | 2.0244 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.69788 | 0.69444 | 0.69357 | 0.69347 | 0.69903 | 0.69772 | 0.69709 | 0.69813 | 0.70473 | 0.70627 | 0.70165 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.3958 | 1.3889 | 1.3871 | 1.3869 | 1.3981 | 1.3954 | 1.3942 | 1.3963 | 1.4095 | 1.4125 | 1.4033 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.26434 | 0.27822 | 0.26990 | 0.27730 | 0.25015 | 0.26589 | 0.26465 | 0.24769 | 0.24306 | 0.25972 | 0.25139 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.26434 | 0.27822 | 0.26990 | 0.27730 | 0.25015 | 0.26589 | 0.26465 | 0.24769 | 0.24306 | 0.25972 | 0.25139 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3862 | -1.3833 | -1.3830 | -1.3832 | -1.3859 | -1.3858 | -1.3832 | -1.3861 | -1.3891 | -1.3903 | -1.3872 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.3923 | -1.3848 | -1.3835 | -1.3844 | -1.3934 | -1.3897 | -1.3902 | -1.3927 | -1.4060 | -1.4086 | -1.3995 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25083 | 0.25131 | 0.25133 | 0.25133 | 0.25104 | 0.25066 | 0.25196 | 0.25084 | 0.25095 | 0.25079 | 0.25121 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25161 | 0.25261 | 0.25269 | 0.25274 | 0.25202 | 0.25126 | 0.25367 | 0.25158 | 0.25168 | 0.25159 | 0.25229 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.79758 | 0.78179 | 0.75701 | 0.75698 | 0.80433 | 0.82146 | 0.78706 | 0.79418 | 0.78278 | 0.77241 | 0.79527 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 0.85146 | 0.83394 | 0.80730 | 0.80778 | 0.85805 | 0.87693 | 0.84004 | 0.84797 | 0.83574 | 0.82446 | 0.84914 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0161 | 1.0017 | 1.0071 | 0.99568 | 1.0260 | 1.0024 | 1.0141 | 1.0146 | 1.0355 | 1.0243 | 1.0241 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0321 | 2.0035 | 2.0141 | 1.9914 | 2.0520 | 2.0049 | 2.0282 | 2.0292 | 2.0710 | 2.0486 | 2.0481 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.70437 | 0.69447 | 0.69815 | 0.69025 | 0.71124 | 0.69494 | 0.70300 | 0.70337 | 0.71781 | 0.71000 | 0.70987 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4087 | 1.3889 | 1.3963 | 1.3805 | 1.4225 | 1.3899 | 1.4060 | 1.4067 | 1.4356 | 1.4200 | 1.4197 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.23984 | 0.26714 | 0.25122 | 0.28112 | 0.21742 | 0.26909 | 0.24309 | 0.22554 | 0.21904 | 0.23952 | 0.23009 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.23984 | 0.26714 | 0.25122 | 0.28112 | 0.21742 | 0.26909 | 0.24309 | 0.22554 | 0.21904 | 0.23952 | 0.23009 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3933 | -1.3835 | -1.3864 | -1.3797 | -1.3976 | -1.3832 | -1.3909 | -1.3916 | -1.4014 | -1.3940 | -1.3948 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.4052 | -1.3858 | -1.3929 | -1.3781 | -1.4186 | -1.3843 | -1.4024 | -1.4036 | -1.4323 | -1.4164 | -1.4155 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.24891 | 0.25136 | 0.25077 | 0.25234 | 0.24836 | 0.25131 | 0.24969 | 0.24945 | 0.24800 | 0.24993 | 0.24936 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.24785 | 0.25281 | 0.25145 | 0.25482 | 0.24668 | 0.25263 | 0.24931 | 0.24882 | 0.24582 | 0.24975 | 0.24857 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.2140 | 1.1804 | 1.1670 | 1.1544 | 1.2289 | 1.2341 | 1.1974 | 1.2045 | 1.1840 | 1.1881 | 1.2080 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.5174 | 1.4758 | 1.4580 | 1.4410 | 1.5358 | 1.5388 | 1.4974 | 1.5049 | 1.4750 | 1.4852 | 1.5062 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0147 | 1.0006 | 1.0044 | 0.99747 | 1.0147 | 1.0076 | 1.0249 | 1.0191 | 1.0374 | 1.0189 | 1.0303 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0294 | 2.0011 | 2.0088 | 1.9949 | 2.0295 | 2.0152 | 2.0498 | 2.0382 | 2.0748 | 2.0379 | 2.0605 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.70342 | 0.69362 | 0.69629 | 0.69153 | 0.70342 | 0.69852 | 0.71047 | 0.70647 | 0.71912 | 0.70631 | 0.71420 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4068 | 1.3872 | 1.3926 | 1.3831 | 1.4068 | 1.3970 | 1.4209 | 1.4129 | 1.4382 | 1.4126 | 1.4284 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24155 | 0.27502 | 0.25878 | 0.28628 | 0.24486 | 0.26342 | 0.24288 | 0.22697 | 0.21836 | 0.27071 | 0.22730 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24155 | 0.27502 | 0.25878 | 0.28628 | 0.24486 | 0.26342 | 0.24288 | 0.22697 | 0.21836 | 0.27071 | 0.22730 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3910 | -1.3824 | -1.3853 | -1.3810 | -1.3894 | -1.3851 | -1.3949 | -1.3942 | -1.4024 | -1.3883 | -1.3979 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4025 | -1.3840 | -1.3889 | -1.3802 | -1.4018 | -1.3894 | -1.4165 | -1.4095 | -1.4344 | -1.4090 | -1.4237 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.24968 | 0.25168 | 0.25084 | 0.25195 | 0.25034 | 0.25095 | 0.24951 | 0.24895 | 0.24794 | 0.25186 | 0.24902 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.24931 | 0.25346 | 0.25173 | 0.25405 | 0.25062 | 0.25187 | 0.24899 | 0.24788 | 0.24581 | 0.25369 | 0.24809 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.40000 | 0.39557 | 0.38089 | 0.35687 | 0.39901 | 0.41545 | 0.39116 | 0.40219 | 0.39733 | 0.39855 | 0.38741 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.40232 | 0.39781 | 0.38314 | 0.35884 | 0.40135 | 0.41776 | 0.39342 | 0.40453 | 0.39954 | 0.40079 | 0.38957 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.29999 | 0.31434 | 0.28177 | 0.27732 | 0.31375 | 0.31713 | 0.30827 | 0.29833 | 0.30304 | 0.30447 | 0.29472 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.30120 | 0.31549 | 0.28290 | 0.27840 | 0.31497 | 0.31836 | 0.30948 | 0.29951 | 0.30421 | 0.30558 | 0.29584 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.51958 | 0.50812 | 0.49526 | 0.49254 | 0.54729 | 0.55143 | 0.52945 | 0.52149 | 0.52848 | 0.50810 | 0.50655 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.52337 | 0.51160 | 0.49867 | 0.49595 | 0.55122 | 0.55540 | 0.53331 | 0.52513 | 0.53214 | 0.51180 | 0.50996 |
| eval/lm/c4_en-validation/CE loss | lower | 2.8406 | 2.8141 | 2.7808 | 2.7816 | 2.8496 | 2.8940 | 2.8298 | 2.8243 | 2.8104 | 2.8000 | 2.8349 |
| eval/lm/c4_en-validation/PPL | lower | 17.13 | 16.68 | 16.13 | 16.15 | 17.28 | 18.07 | 16.94 | 16.85 | 16.62 | 16.45 | 17.03 |
| eval/lm/dolma_books-validation/CE loss | lower | 2.7159 | 2.6782 | 2.6426 | 2.6431 | 2.7257 | 2.7702 | 2.6991 | 2.6902 | 2.6765 | 2.6618 | 2.6962 |
| eval/lm/dolma_books-validation/PPL | lower | 15.12 | 14.56 | 14.05 | 14.06 | 15.27 | 15.96 | 14.87 | 14.73 | 14.53 | 14.32 | 14.82 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 2.9788 | 2.9530 | 2.9225 | 2.9183 | 2.9887 | 3.0326 | 2.9674 | 2.9641 | 2.9497 | 2.9365 | 2.9734 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 19.66 | 19.16 | 18.59 | 18.51 | 19.86 | 20.75 | 19.44 | 19.38 | 19.10 | 18.85 | 19.56 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.0532 | 2.0325 | 2.0094 | 2.0100 | 2.0651 | 2.0954 | 2.0453 | 2.0414 | 2.0294 | 2.0179 | 2.0457 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 7.7925 | 7.6328 | 7.4586 | 7.4635 | 7.8860 | 8.1283 | 7.7317 | 7.7016 | 7.6096 | 7.5225 | 7.7347 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.1636 | 3.1344 | 3.1066 | 3.1087 | 3.1726 | 3.2072 | 3.1506 | 3.1463 | 3.1281 | 3.1211 | 3.1538 |
| eval/lm/dolma_reddit-validation/PPL | lower | 23.66 | 22.97 | 22.34 | 22.39 | 23.87 | 24.71 | 23.35 | 23.25 | 22.83 | 22.67 | 23.43 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.2742 | 1.2545 | 1.2314 | 1.2300 | 1.2807 | 1.3134 | 1.2643 | 1.2607 | 1.2551 | 1.2438 | 1.2698 |
| eval/lm/dolma_stack-validation/PPL | lower | 3.5759 | 3.5061 | 3.4259 | 3.4212 | 3.5992 | 3.7189 | 3.5407 | 3.5277 | 3.5082 | 3.4686 | 3.5600 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.4746 | 2.4504 | 2.4208 | 2.4205 | 2.4899 | 2.5443 | 2.4653 | 2.4649 | 2.4451 | 2.4395 | 2.4719 |
| eval/lm/dolma_wiki-validation/PPL | lower | 11.88 | 11.59 | 11.26 | 11.25 | 12.06 | 12.73 | 11.77 | 11.76 | 11.53 | 11.47 | 11.84 |
| eval/lm/ice-validation/CE loss | lower | 2.9015 | 2.8735 | 2.8480 | 2.8665 | 2.9140 | 2.9385 | 2.9002 | 2.8945 | 2.8728 | 2.8647 | 2.8987 |
| eval/lm/ice-validation/PPL | lower | 18.20 | 17.70 | 17.25 | 17.57 | 18.43 | 18.89 | 18.18 | 18.07 | 17.69 | 17.54 | 18.15 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 2.9666 | 2.9424 | 2.9280 | 2.9159 | 2.9721 | 3.0140 | 2.9711 | 2.9595 | 2.9534 | 2.9443 | 2.9634 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 19.43 | 18.96 | 18.69 | 18.47 | 19.53 | 20.37 | 19.51 | 19.29 | 19.17 | 19.00 | 19.36 |
| eval/lm/pile-validation/CE loss | lower | 2.1303 | 2.1055 | 2.0789 | 2.0776 | 2.1393 | 2.1780 | 2.1191 | 2.1119 | 2.1040 | 2.0919 | 2.1254 |
| eval/lm/pile-validation/PPL | lower | 8.4171 | 8.2112 | 7.9957 | 7.9853 | 8.4937 | 8.8291 | 8.3241 | 8.2641 | 8.1985 | 8.1003 | 8.3761 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.3981 | 2.3689 | 2.3386 | 2.3335 | 2.4199 | 2.4504 | 2.3858 | 2.3791 | 2.3628 | 2.3506 | 2.3949 |
| eval/lm/wikitext_103-validation/PPL | lower | 11.00 | 10.69 | 10.37 | 10.31 | 11.24 | 11.59 | 10.87 | 10.80 | 10.62 | 10.49 | 10.97 |
| throughput/in-loop eval batches | see metric | 1659.0 | 1659.0 | 1111.0 | 1111.0 | 1659.0 | 562.0 | 1659.0 | 1659.0 | 1659.0 | 1659.0 | 1659.0 |
| throughput/in-loop eval time (s) | see metric | 135.0 | 125.3 | 293.0 | 236.9 | 128.5 | 58.61 | 127.0 | 131.9 | 166.8 | 202.1 | 160.8 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| eg-1p2b-cx2-eg24e2k-lr3e-4-r1<br>`7lq6ag2s` | finished | original | 42535747584.0 | 108174 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7lq6ag2s) |
| eg-1p2b-cx2-eg96e8k-lr3e-4-r1<br>`52j6x54s` | finished | original | 42628546560.0 | 108410 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/52j6x54s) |
| int-1p2b-cx2-intd256e8k-lr6e-4-r2<br>`xeiarhn7` | finished | original | 42891214848.0 | 109078 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xeiarhn7) |
| int-1p2b-cx2-intw256e8k-lr6e-4-r2<br>`jfwntmwm` | finished | original | 42834984960.0 | 108935 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jfwntmwm) |
| 1p2b-cx2-b384k-lr1.5e-4-r1<br>`dtd8qeiv` | finished | b384k | 42566811648.0 | 108253 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/dtd8qeiv) |
| 1p2b-cx2-b384k-lr2.4e-3-r1<br>`blpr9kqj` | finished | b384k | 42566811648.0 | 108253 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/blpr9kqj) |
| 1p2b-cx2-b384k-lr3e-4-r1<br>`7cuo1d1i` | finished | b384k | 42566811648.0 | 108253 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7cuo1d1i) |
| 1p2b-cx2-b384k-lr6e-4-r1<br>`54pt8zj7` | finished | b384k | 42566811648.0 | 108253 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/54pt8zj7) |
| q3-1p2b-cx2-q3am128e8k-lr6e-4-r1<br>`f2gd9zv6` | finished | original | 42677698560.0 | 108535 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/f2gd9zv6) |
| q3-1p2b-cx2-q3td128e8k-lr6e-4-r1<br>`zoehfkg5` | finished | original | 41979740160.0 | 106760 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/zoehfkg5) |
| se-1p2b-cx2-se0m9-lr6e-4-r1<br>`6v61syzf` | finished | original | 42566811648.0 | 108253 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6v61syzf) |

## 1p2b Cx4

| metric | direction | eg-1p2b-cx4-eg24e2k-lr4e-4-r1<br>`ybn138lx` | eg-1p2b-cx4-eg96e8k-lr4e-4-r1<br>`cql4y2di` | int-1p2b-cx4-intd256e8k-lr3e-4-r2<br>`i9vrjwe8` | int-1p2b-cx4-intw256e8k-lr3e-4-r2<br>`u7ab1tpb` | 1p2b-cx4-b512k-lr1.2e-3-r1<br>`vksk7sux` | 1p2b-cx4-b512k-lr1.5e-4-r1<br>`5u5iumvr` | 1p2b-cx4-b512k-lr3e-4-r1<br>`rkjs2sze` | 1p2b-cx4-b512k-lr6e-4-r1<br>`1tzma107` | q3-1p2b-cx4-q3am128e8k-lr3e-4-r1<br>`15wzt5lj` | q3-1p2b-cx4-q3td128e8k-lr3e-4-r1<br>`ek9f5z9p` | se-1p2b-cx4-se0m9-lr3e-4-r1<br>`k3klv3au` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.79976 | 0.77576 | 0.76068 | 0.76962 | 0.79914 | 0.80633 | 0.80942 | 0.79963 | 0.79387 | 0.78796 | 0.79887 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 0.87530 | 0.84707 | 0.82962 | 0.84209 | 0.87397 | 0.88409 | 0.88534 | 0.87427 | 0.86859 | 0.86405 | 0.87450 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0050 | 1.0044 | 1.0056 | 0.99773 | 1.0046 | 1.0093 | 1.0039 | 1.0052 | 1.0060 | 1.0135 | 1.0074 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0099 | 2.0089 | 2.0113 | 1.9955 | 2.0092 | 2.0186 | 2.0078 | 2.0104 | 2.0119 | 2.0269 | 2.0148 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.69665 | 0.69636 | 0.69710 | 0.69166 | 0.69642 | 0.69966 | 0.69596 | 0.69686 | 0.69740 | 0.70254 | 0.69841 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.3933 | 1.3927 | 1.3942 | 1.3833 | 1.3928 | 1.3993 | 1.3919 | 1.3937 | 1.3948 | 1.4051 | 1.3968 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.26109 | 0.24659 | 0.24488 | 0.26024 | 0.26706 | 0.26365 | 0.26365 | 0.25939 | 0.25000 | 0.23720 | 0.26280 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.26109 | 0.24659 | 0.24488 | 0.26024 | 0.26706 | 0.26365 | 0.26365 | 0.25939 | 0.25000 | 0.23720 | 0.26280 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.3900 | -1.3902 | -1.3909 | -1.3817 | -1.3894 | -1.3965 | -1.3880 | -1.3918 | -1.3929 | -1.4038 | -1.3947 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.3900 | -1.3902 | -1.3909 | -1.3817 | -1.3894 | -1.3965 | -1.3880 | -1.3918 | -1.3929 | -1.4038 | -1.3947 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.25246 | 0.25075 | 0.25117 | 0.25266 | 0.25249 | 0.25130 | 0.25237 | 0.25162 | 0.25092 | 0.24968 | 0.25167 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.25246 | 0.25075 | 0.25117 | 0.25266 | 0.25249 | 0.25130 | 0.25237 | 0.25162 | 0.25092 | 0.24968 | 0.25167 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.61101 | 0.58471 | 0.56147 | 0.57178 | 0.58711 | 0.60734 | 0.60789 | 0.59238 | 0.59532 | 0.58652 | 0.59408 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.66366 | 0.63547 | 0.60920 | 0.62063 | 0.63776 | 0.65949 | 0.66047 | 0.64356 | 0.64794 | 0.63713 | 0.64546 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0153 | 1.0078 | 1.0035 | 1.0027 | 1.0080 | 1.0140 | 1.0133 | 1.0079 | 1.0060 | 1.0051 | 1.0134 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0305 | 2.0156 | 2.0070 | 2.0055 | 2.0160 | 2.0280 | 2.0265 | 2.0158 | 2.0119 | 2.0102 | 2.0267 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.70387 | 0.69867 | 0.69566 | 0.69508 | 0.69874 | 0.70296 | 0.70244 | 0.69873 | 0.69742 | 0.69683 | 0.70250 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4077 | 1.3973 | 1.3913 | 1.3902 | 1.3975 | 1.4059 | 1.4049 | 1.3975 | 1.3948 | 1.3937 | 1.4050 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.24621 | 0.24621 | 0.27525 | 0.25463 | 0.24874 | 0.24832 | 0.24663 | 0.24790 | 0.24285 | 0.26473 | 0.24832 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.24621 | 0.24621 | 0.27525 | 0.25463 | 0.24874 | 0.24832 | 0.24663 | 0.24790 | 0.24285 | 0.26473 | 0.24832 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.4044 | -1.3952 | -1.3885 | -1.3881 | -1.3938 | -1.4030 | -1.4012 | -1.3949 | -1.3930 | -1.3923 | -1.4031 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.4044 | -1.3952 | -1.3885 | -1.3881 | -1.3938 | -1.4030 | -1.4012 | -1.3949 | -1.3930 | -1.3923 | -1.4031 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.24962 | 0.25036 | 0.25191 | 0.25135 | 0.25113 | 0.25030 | 0.25059 | 0.25079 | 0.25015 | 0.25127 | 0.25071 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.24962 | 0.25036 | 0.25191 | 0.25135 | 0.25113 | 0.25030 | 0.25059 | 0.25079 | 0.25015 | 0.25127 | 0.25071 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 1.1233 | 1.0103 | 0.89729 | 0.93379 | 0.99453 | 1.2245 | 0.94615 | 1.0280 | 0.97133 | 0.91910 | 0.98433 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 1.8376 | 1.6277 | 1.4524 | 1.5164 | 1.6286 | 1.9782 | 1.5314 | 1.6756 | 1.5637 | 1.4833 | 1.5839 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 0.77852 | 0.70021 | 0.62192 | 0.64723 | 0.68929 | 0.84887 | 0.65585 | 0.71248 | 0.67312 | 0.63705 | 0.68225 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 1.2738 | 1.1282 | 1.0068 | 1.0510 | 1.1289 | 1.3712 | 1.0615 | 1.1615 | 1.0837 | 1.0282 | 1.0978 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.56256 | 0.55969 | 0.62082 | 0.60267 | 0.58835 | 0.52913 | 0.58453 | 0.56638 | 0.60649 | 0.60936 | 0.55110 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.56256 | 0.55969 | 0.62082 | 0.60267 | 0.58835 | 0.52913 | 0.58453 | 0.56638 | 0.60649 | 0.60936 | 0.55110 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -1.3671 | -1.3400 | -1.1106 | -1.2219 | -1.2384 | -1.5264 | -1.2094 | -1.3041 | -1.1550 | -1.1524 | -1.2670 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -1.3671 | -1.3400 | -1.1106 | -1.2219 | -1.2384 | -1.5264 | -1.2094 | -1.3041 | -1.1550 | -1.1524 | -1.2670 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.52289 | 0.52603 | 0.56883 | 0.54183 | 0.53175 | 0.48616 | 0.53804 | 0.52494 | 0.54394 | 0.55051 | 0.52093 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.52289 | 0.52603 | 0.56883 | 0.54183 | 0.53175 | 0.48616 | 0.53804 | 0.52494 | 0.54394 | 0.55051 | 0.52093 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.33088 | 0.34296 | 0.36841 | 0.32920 | 0.33540 | 0.30580 | 0.37632 | 0.36359 | 0.31745 | 0.34170 | 0.34249 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.36025 | 0.37325 | 0.40281 | 0.35830 | 0.36541 | 0.33264 | 0.41016 | 0.39594 | 0.34561 | 0.37233 | 0.37306 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.22932 | 0.23774 | 0.25536 | 0.22817 | 0.23248 | 0.21197 | 0.26085 | 0.25203 | 0.22006 | 0.23688 | 0.23741 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.24971 | 0.25870 | 0.27918 | 0.24836 | 0.25326 | 0.23057 | 0.28428 | 0.27446 | 0.23958 | 0.25809 | 0.25858 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.66601 | 0.67688 | 0.70850 | 0.71739 | 0.68972 | 0.68478 | 0.67787 | 0.69269 | 0.70257 | 0.69664 | 0.68775 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.66601 | 0.67688 | 0.70850 | 0.71739 | 0.68972 | 0.68478 | 0.67787 | 0.69269 | 0.70257 | 0.69664 | 0.68775 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -1.2117 | -1.2065 | -1.1176 | -1.0468 | -1.3062 | -1.2092 | -1.3085 | -1.2118 | -1.1708 | -1.1485 | -1.2418 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -1.2117 | -1.2065 | -1.1176 | -1.0468 | -1.3062 | -1.2092 | -1.3085 | -1.2118 | -1.1708 | -1.1485 | -1.2418 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.64387 | 0.66161 | 0.67255 | 0.69420 | 0.65625 | 0.65799 | 0.64659 | 0.65705 | 0.67203 | 0.67511 | 0.65842 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.64387 | 0.66161 | 0.67255 | 0.69420 | 0.65625 | 0.65799 | 0.64659 | 0.65705 | 0.67203 | 0.67511 | 0.65842 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.37988 | 0.32116 | 0.23448 | 0.25521 | 0.27356 | 0.27865 | 0.24112 | 0.28926 | 0.24775 | 0.24883 | 0.28237 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.45638 | 0.38591 | 0.28050 | 0.30620 | 0.32878 | 0.33513 | 0.28928 | 0.34794 | 0.29716 | 0.29892 | 0.33991 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.26346 | 0.22280 | 0.16272 | 0.17707 | 0.18979 | 0.19330 | 0.16731 | 0.20071 | 0.17191 | 0.17267 | 0.19591 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.31654 | 0.26774 | 0.19468 | 0.21246 | 0.22816 | 0.23255 | 0.20075 | 0.24151 | 0.20621 | 0.20741 | 0.23584 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.91321 | 0.92285 | 0.94503 | 0.94696 | 0.89875 | 0.91610 | 0.93443 | 0.92768 | 0.93635 | 0.94311 | 0.92960 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.91321 | 0.92285 | 0.94503 | 0.94696 | 0.89875 | 0.91610 | 0.93443 | 0.92768 | 0.93635 | 0.94311 | 0.92960 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.28800 | -0.24029 | -0.18180 | -0.18687 | -0.26983 | -0.25598 | -0.21430 | -0.23489 | -0.19339 | -0.20990 | -0.24571 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.28800 | -0.24029 | -0.18180 | -0.18687 | -0.26983 | -0.25598 | -0.21430 | -0.23489 | -0.19339 | -0.20990 | -0.24571 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.81264 | 0.83969 | 0.88005 | 0.87311 | 0.83120 | 0.83475 | 0.85875 | 0.84224 | 0.86875 | 0.86464 | 0.83906 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.81264 | 0.83969 | 0.88005 | 0.87311 | 0.83120 | 0.83475 | 0.85875 | 0.84224 | 0.86875 | 0.86464 | 0.83906 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.29652 | 0.28724 | 0.28559 | 0.26542 | 0.25928 | 0.28202 | 0.26618 | 0.27930 | 0.29318 | 0.28618 | 0.29449 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.30644 | 0.29687 | 0.29519 | 0.27437 | 0.26818 | 0.29143 | 0.27508 | 0.28870 | 0.30304 | 0.29588 | 0.30436 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.20555 | 0.19911 | 0.19798 | 0.18400 | 0.17974 | 0.19550 | 0.18452 | 0.19359 | 0.20323 | 0.19841 | 0.20415 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.21243 | 0.20580 | 0.20464 | 0.19020 | 0.18591 | 0.20202 | 0.19068 | 0.20013 | 0.21009 | 0.20510 | 0.21099 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.91682 | 0.90072 | 0.92397 | 0.91771 | 0.86225 | 0.86225 | 0.90429 | 0.89445 | 0.88819 | 0.93113 | 0.87478 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.91682 | 0.90072 | 0.92397 | 0.91771 | 0.86225 | 0.86225 | 0.90429 | 0.89445 | 0.88819 | 0.93113 | 0.87478 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.23562 | -0.24141 | -0.21775 | -0.20138 | -0.35933 | -0.31744 | -0.27786 | -0.25095 | -0.27673 | -0.20116 | -0.28726 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.23562 | -0.24141 | -0.21775 | -0.20138 | -0.35933 | -0.31744 | -0.27786 | -0.25095 | -0.27673 | -0.20116 | -0.28726 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.89606 | 0.88676 | 0.89681 | 0.90460 | 0.86478 | 0.86611 | 0.88503 | 0.88812 | 0.87948 | 0.90373 | 0.86855 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.89606 | 0.88676 | 0.89681 | 0.90460 | 0.86478 | 0.86611 | 0.88503 | 0.88812 | 0.87948 | 0.90373 | 0.86855 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 0.83575 | 0.68134 | 0.70212 | 0.64541 | 0.72824 | 0.78357 | 0.75415 | 0.73106 | 0.71576 | 0.72625 | 0.72273 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.4089 | 1.1403 | 1.1642 | 1.0767 | 1.2183 | 1.3084 | 1.2591 | 1.2231 | 1.1985 | 1.2045 | 1.2117 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.61138 | 0.49673 | 0.51596 | 0.47568 | 0.53648 | 0.57153 | 0.55365 | 0.53780 | 0.51999 | 0.53491 | 0.52678 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.0580 | 0.85182 | 0.88128 | 0.81781 | 0.92558 | 0.97983 | 0.95082 | 0.92692 | 0.89148 | 0.91487 | 0.90504 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.74906 | 0.77903 | 0.80150 | 0.77715 | 0.78277 | 0.72659 | 0.74719 | 0.76592 | 0.76779 | 0.77341 | 0.76404 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.74906 | 0.77903 | 0.80150 | 0.77715 | 0.78277 | 0.72659 | 0.74719 | 0.76592 | 0.76779 | 0.77341 | 0.76404 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.63693 | -0.59077 | -0.54930 | -0.55104 | -0.58437 | -0.63718 | -0.60428 | -0.56970 | -0.57155 | -0.59566 | -0.61140 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.63693 | -0.59077 | -0.54930 | -0.55104 | -0.58437 | -0.63718 | -0.60428 | -0.56970 | -0.57155 | -0.59566 | -0.61140 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.67729 | 0.70670 | 0.71053 | 0.71784 | 0.70051 | 0.66514 | 0.69009 | 0.69180 | 0.69299 | 0.69834 | 0.69184 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.67729 | 0.70670 | 0.71053 | 0.71784 | 0.70051 | 0.66514 | 0.69009 | 0.69180 | 0.69299 | 0.69834 | 0.69184 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.2456 | 1.2927 | 1.1580 | 1.0781 | 1.1603 | 1.2729 | 1.3501 | 1.1840 | 1.2059 | 1.1435 | 1.2527 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 1.7499 | 1.8028 | 1.6317 | 1.5288 | 1.6433 | 1.7855 | 1.9014 | 1.6771 | 1.7153 | 1.6143 | 1.7565 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 0.86347 | 0.89599 | 0.80274 | 0.74729 | 0.80421 | 0.88240 | 0.93582 | 0.82064 | 0.83597 | 0.79265 | 0.86835 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.2130 | 1.2495 | 1.1311 | 1.0597 | 1.1391 | 1.2376 | 1.3180 | 1.1625 | 1.1890 | 1.1189 | 1.2177 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.35275 | 0.37080 | 0.40361 | 0.40033 | 0.40115 | 0.31829 | 0.35029 | 0.39048 | 0.39623 | 0.38146 | 0.37080 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.35275 | 0.37080 | 0.40361 | 0.40033 | 0.40115 | 0.31829 | 0.35029 | 0.39048 | 0.39623 | 0.38146 | 0.37080 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -2.8090 | -2.5419 | -2.2472 | -2.4723 | -2.4484 | -3.0554 | -2.7171 | -2.5240 | -2.4164 | -2.4125 | -2.6161 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -2.8090 | -2.5419 | -2.2472 | -2.4723 | -2.4484 | -3.0554 | -2.7171 | -2.5240 | -2.4164 | -2.4125 | -2.6161 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.36251 | 0.38551 | 0.40291 | 0.41669 | 0.40788 | 0.33069 | 0.34930 | 0.39105 | 0.41054 | 0.39797 | 0.36937 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.36251 | 0.38551 | 0.40291 | 0.41669 | 0.40788 | 0.33069 | 0.34930 | 0.39105 | 0.41054 | 0.39797 | 0.36937 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.37940 | 0.37197 | 0.35876 | 0.36799 | 0.39366 | 0.39310 | 0.38430 | 0.38048 | 0.37978 | 0.38131 | 0.37990 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.38423 | 0.37703 | 0.36354 | 0.37311 | 0.39922 | 0.39822 | 0.38957 | 0.38551 | 0.38494 | 0.38641 | 0.38499 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.58583 | 0.56888 | 0.56393 | 0.55319 | 0.59612 | 0.58141 | 0.57776 | 0.59076 | 0.56637 | 0.57175 | 0.58208 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.59109 | 0.57391 | 0.56886 | 0.55792 | 0.60129 | 0.58642 | 0.58276 | 0.59590 | 0.57141 | 0.57673 | 0.58701 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.1144 | 1.8615 | 1.8109 | 2.2513 | 2.1070 | 1.9255 | 1.9065 | 1.8698 | 2.1259 | 2.1183 | 1.8673 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 4.2288 | 3.7230 | 3.6217 | 4.5025 | 4.2139 | 3.8511 | 3.8130 | 3.7395 | 4.2519 | 4.2366 | 3.7345 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.4654 | 1.2902 | 1.2551 | 1.5604 | 1.4603 | 1.3342 | 1.3213 | 1.2959 | 1.4741 | 1.4680 | 1.2943 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 2.9307 | 2.5804 | 2.5102 | 3.1209 | 2.9205 | 2.6684 | 2.6427 | 2.5918 | 2.9482 | 2.9359 | 2.5887 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.10000 | 0.13000 | 0.05000 | 0.10000 | 0.09000 | 0.09000 | 0.11000 | 0.06000 | 0.09000 | 0.11000 | 0.10000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.10000 | 0.13000 | 0.05000 | 0.10000 | 0.09000 | 0.09000 | 0.11000 | 0.06000 | 0.09000 | 0.11000 | 0.10000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -2.9262 | -2.5724 | -2.4991 | -3.1175 | -2.9145 | -2.6611 | -2.6329 | -2.5783 | -2.9444 | -2.9326 | -2.5778 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -2.9262 | -2.5724 | -2.4991 | -3.1175 | -2.9145 | -2.6611 | -2.6329 | -2.5783 | -2.9444 | -2.9326 | -2.5778 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09894 | 0.09888 | 0.10056 | 0.09699 | 0.09927 | 0.09868 | 0.09747 | 0.09788 | 0.09808 | 0.10092 | 0.09807 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09894 | 0.09888 | 0.10056 | 0.09699 | 0.09927 | 0.09868 | 0.09747 | 0.09788 | 0.09808 | 0.10092 | 0.09807 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.74891 | 0.74923 | 0.73675 | 0.73505 | 0.76007 | 0.75799 | 0.75815 | 0.74938 | 0.74733 | 0.74171 | 0.75401 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.75713 | 0.75739 | 0.74487 | 0.74307 | 0.76838 | 0.76636 | 0.76636 | 0.75759 | 0.75553 | 0.74992 | 0.76231 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.58208 | 0.56860 | 0.55168 | 0.55195 | 0.58343 | 0.58632 | 0.57811 | 0.57700 | 0.56748 | 0.56364 | 0.58582 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.58386 | 0.57058 | 0.55350 | 0.55374 | 0.58532 | 0.58819 | 0.58005 | 0.57881 | 0.56930 | 0.56539 | 0.58760 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.64060 | 0.63000 | 0.61844 | 0.61263 | 0.64427 | 0.64282 | 0.63528 | 0.63437 | 0.62709 | 0.63000 | 0.63480 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.67204 | 0.66056 | 0.64832 | 0.64199 | 0.67608 | 0.67417 | 0.66651 | 0.66542 | 0.65755 | 0.66039 | 0.66646 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0141 | 1.0129 | 1.0018 | 1.0079 | 1.0104 | 1.0097 | 1.0068 | 1.0045 | 1.0066 | 0.99929 | 1.0081 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0282 | 2.0257 | 2.0035 | 2.0157 | 2.0208 | 2.0194 | 2.0137 | 2.0090 | 2.0132 | 1.9986 | 2.0162 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.70304 | 0.70213 | 0.69448 | 0.69870 | 0.70046 | 0.69996 | 0.69800 | 0.69640 | 0.69780 | 0.69280 | 0.69884 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4061 | 1.4043 | 1.3890 | 1.3974 | 1.4009 | 1.3999 | 1.3960 | 1.3928 | 1.3956 | 1.3856 | 1.3977 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24973 | 0.24782 | 0.27651 | 0.26036 | 0.25420 | 0.25058 | 0.25930 | 0.25526 | 0.26206 | 0.27396 | 0.25569 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24973 | 0.24782 | 0.27651 | 0.26036 | 0.25420 | 0.25058 | 0.25930 | 0.25526 | 0.26206 | 0.27396 | 0.25569 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3897 | -1.3893 | -1.3834 | -1.3845 | -1.3868 | -1.3881 | -1.3853 | -1.3852 | -1.3854 | -1.3831 | -1.3879 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.4024 | -1.4011 | -1.3860 | -1.3944 | -1.3964 | -1.3967 | -1.3909 | -1.3902 | -1.3938 | -1.3835 | -1.3950 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.25025 | 0.25036 | 0.25145 | 0.25206 | 0.25101 | 0.25034 | 0.25106 | 0.25105 | 0.25142 | 0.25125 | 0.25028 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.25044 | 0.25073 | 0.25296 | 0.25414 | 0.25199 | 0.25063 | 0.25207 | 0.25212 | 0.25285 | 0.25247 | 0.25053 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 0.88944 | 0.87072 | 0.84356 | 0.85176 | 0.88495 | 0.89143 | 0.87939 | 0.88273 | 0.86947 | 0.86624 | 0.89045 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 0.99132 | 0.97089 | 0.93932 | 0.95056 | 0.98645 | 0.99295 | 0.97985 | 0.98462 | 0.96973 | 0.96571 | 0.99310 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0059 | 1.0035 | 0.99702 | 0.98901 | 0.99876 | 1.0045 | 0.99884 | 0.99538 | 0.99702 | 0.99680 | 0.99672 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0117 | 2.0070 | 1.9940 | 1.9780 | 1.9975 | 2.0090 | 1.9977 | 1.9908 | 1.9940 | 1.9936 | 1.9934 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.69731 | 0.69564 | 0.69117 | 0.68563 | 0.69235 | 0.69633 | 0.69246 | 0.69004 | 0.69116 | 0.69105 | 0.69099 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.3946 | 1.3913 | 1.3823 | 1.3713 | 1.3847 | 1.3927 | 1.3849 | 1.3801 | 1.3823 | 1.3821 | 1.3820 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.27236 | 0.27853 | 0.28902 | 0.29303 | 0.28809 | 0.27329 | 0.28347 | 0.28840 | 0.27606 | 0.28378 | 0.29180 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.27236 | 0.27853 | 0.28902 | 0.29303 | 0.28809 | 0.27329 | 0.28347 | 0.28840 | 0.27606 | 0.28378 | 0.29180 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3835 | -1.3831 | -1.3798 | -1.3732 | -1.3795 | -1.3843 | -1.3798 | -1.3790 | -1.3796 | -1.3801 | -1.3807 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.3910 | -1.3885 | -1.3787 | -1.3681 | -1.3802 | -1.3890 | -1.3806 | -1.3772 | -1.3801 | -1.3799 | -1.3793 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25197 | 0.25187 | 0.25236 | 0.25453 | 0.25268 | 0.25132 | 0.25252 | 0.25258 | 0.25262 | 0.25231 | 0.25197 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25396 | 0.25375 | 0.25485 | 0.25923 | 0.25539 | 0.25263 | 0.25501 | 0.25520 | 0.25525 | 0.25462 | 0.25401 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.76227 | 0.74574 | 0.72392 | 0.72260 | 0.76545 | 0.76774 | 0.75828 | 0.75244 | 0.74292 | 0.74383 | 0.76028 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 0.81370 | 0.79560 | 0.77233 | 0.76992 | 0.81649 | 0.81994 | 0.80915 | 0.80262 | 0.79231 | 0.79340 | 0.81070 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0143 | 1.0018 | 0.99683 | 0.99106 | 1.0007 | 1.0153 | 1.0113 | 1.0008 | 1.0050 | 0.99420 | 1.0023 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0287 | 2.0037 | 1.9937 | 1.9821 | 2.0013 | 2.0307 | 2.0225 | 2.0016 | 2.0100 | 1.9884 | 2.0045 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.70317 | 0.69448 | 0.69097 | 0.68707 | 0.69374 | 0.70386 | 0.70105 | 0.69379 | 0.69673 | 0.68922 | 0.69485 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4063 | 1.3890 | 1.3819 | 1.3741 | 1.3875 | 1.4077 | 1.4021 | 1.3876 | 1.3935 | 1.3784 | 1.3897 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24017 | 0.27917 | 0.29022 | 0.28599 | 0.26259 | 0.23237 | 0.24829 | 0.27657 | 0.26422 | 0.29054 | 0.26812 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24017 | 0.27917 | 0.29022 | 0.28599 | 0.26259 | 0.23237 | 0.24829 | 0.27657 | 0.26422 | 0.29054 | 0.26812 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3905 | -1.3824 | -1.3792 | -1.3747 | -1.3817 | -1.3926 | -1.3892 | -1.3822 | -1.3863 | -1.3792 | -1.3847 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.4026 | -1.3862 | -1.3785 | -1.3713 | -1.3831 | -1.4045 | -1.3983 | -1.3849 | -1.3917 | -1.3765 | -1.3873 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.24986 | 0.25198 | 0.25268 | 0.25422 | 0.25191 | 0.24910 | 0.25008 | 0.25195 | 0.25072 | 0.25238 | 0.25092 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.24961 | 0.25399 | 0.25551 | 0.25867 | 0.25379 | 0.24821 | 0.25014 | 0.25397 | 0.25150 | 0.25482 | 0.25182 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.1457 | 1.1215 | 1.0932 | 1.0993 | 1.1496 | 1.1675 | 1.1431 | 1.1288 | 1.1074 | 1.1269 | 1.1475 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.4278 | 1.3997 | 1.3617 | 1.3731 | 1.4324 | 1.4565 | 1.4262 | 1.4064 | 1.3784 | 1.4070 | 1.4343 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0154 | 1.0006 | 0.99999 | 0.99391 | 1.0033 | 1.0114 | 1.0055 | 1.0013 | 1.0053 | 0.99205 | 1.0009 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0308 | 2.0012 | 2.0000 | 1.9878 | 2.0066 | 2.0229 | 2.0110 | 2.0027 | 2.0107 | 1.9841 | 2.0017 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.70393 | 0.69368 | 0.69326 | 0.68906 | 0.69552 | 0.70116 | 0.69701 | 0.69417 | 0.69687 | 0.68772 | 0.69385 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4079 | 1.3874 | 1.3865 | 1.3781 | 1.3910 | 1.4023 | 1.3940 | 1.3883 | 1.3937 | 1.3754 | 1.3877 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24950 | 0.27038 | 0.28595 | 0.28429 | 0.27303 | 0.25613 | 0.25911 | 0.28197 | 0.26872 | 0.29788 | 0.29523 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24950 | 0.27038 | 0.28595 | 0.28429 | 0.27303 | 0.25613 | 0.25911 | 0.28197 | 0.26872 | 0.29788 | 0.29523 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3908 | -1.3823 | -1.3800 | -1.3763 | -1.3820 | -1.3896 | -1.3837 | -1.3828 | -1.3839 | -1.3768 | -1.3829 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4038 | -1.3845 | -1.3830 | -1.3747 | -1.3855 | -1.3985 | -1.3899 | -1.3854 | -1.3919 | -1.3732 | -1.3847 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.24992 | 0.25183 | 0.25280 | 0.25373 | 0.25207 | 0.24990 | 0.25175 | 0.25168 | 0.25204 | 0.25320 | 0.25153 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.24984 | 0.25377 | 0.25565 | 0.25759 | 0.25411 | 0.24983 | 0.25341 | 0.25338 | 0.25413 | 0.25651 | 0.25311 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.38512 | 0.37523 | 0.37364 | 0.35922 | 0.37308 | 0.38614 | 0.37884 | 0.37187 | 0.37479 | 0.35405 | 0.37513 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.38739 | 0.37732 | 0.37584 | 0.36132 | 0.37520 | 0.38845 | 0.38103 | 0.37402 | 0.37688 | 0.35606 | 0.37724 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.29244 | 0.28412 | 0.27307 | 0.27209 | 0.29436 | 0.28763 | 0.29097 | 0.28985 | 0.28487 | 0.27803 | 0.27933 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.29357 | 0.28529 | 0.27406 | 0.27313 | 0.29555 | 0.28883 | 0.29215 | 0.29095 | 0.28604 | 0.27912 | 0.28041 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.47404 | 0.47706 | 0.48044 | 0.46328 | 0.46018 | 0.49682 | 0.49339 | 0.45054 | 0.48082 | 0.45847 | 0.49949 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.47736 | 0.48040 | 0.48382 | 0.46640 | 0.46334 | 0.50030 | 0.49699 | 0.45368 | 0.48424 | 0.46161 | 0.50301 |
| eval/lm/c4_en-validation/CE loss | lower | 2.7690 | 2.7413 | 2.7062 | 2.7014 | 2.7795 | 2.7780 | 2.7579 | 2.7613 | 2.7384 | 2.7283 | 2.7695 |
| eval/lm/c4_en-validation/PPL | lower | 15.94 | 15.51 | 14.97 | 14.90 | 16.11 | 16.09 | 15.77 | 15.82 | 15.46 | 15.31 | 15.95 |
| eval/lm/dolma_books-validation/CE loss | lower | 2.6210 | 2.5894 | 2.5454 | 2.5541 | 2.6277 | 2.6364 | 2.6100 | 2.6070 | 2.5859 | 2.5755 | 2.6221 |
| eval/lm/dolma_books-validation/PPL | lower | 13.75 | 13.32 | 12.75 | 12.86 | 13.84 | 13.96 | 13.60 | 13.56 | 13.27 | 13.14 | 13.77 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 2.9071 | 2.8800 | 2.8418 | 2.8380 | 2.9197 | 2.9156 | 2.8953 | 2.8973 | 2.8747 | 2.8634 | 2.9064 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 18.30 | 17.81 | 17.15 | 17.08 | 18.54 | 18.46 | 18.09 | 18.13 | 17.72 | 17.52 | 18.29 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 1.9928 | 1.9715 | 1.9491 | 1.9468 | 2.0027 | 2.0029 | 1.9874 | 1.9861 | 1.9727 | 1.9626 | 1.9943 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 7.3364 | 7.1816 | 7.0222 | 7.0062 | 7.4093 | 7.4104 | 7.2968 | 7.2873 | 7.1902 | 7.1175 | 7.3474 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.0981 | 3.0700 | 3.0405 | 3.0385 | 3.1016 | 3.1060 | 3.0862 | 3.0854 | 3.0688 | 3.0583 | 3.0969 |
| eval/lm/dolma_reddit-validation/PPL | lower | 22.16 | 21.54 | 20.91 | 20.87 | 22.23 | 22.33 | 21.89 | 21.88 | 21.52 | 21.29 | 22.13 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.2050 | 1.1877 | 1.1619 | 1.1599 | 1.2160 | 1.2108 | 1.1967 | 1.1974 | 1.1839 | 1.1795 | 1.2047 |
| eval/lm/dolma_stack-validation/PPL | lower | 3.3368 | 3.2796 | 3.1959 | 3.1896 | 3.3737 | 3.3562 | 3.3093 | 3.3114 | 3.2670 | 3.2528 | 3.3359 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.3953 | 2.3739 | 2.3329 | 2.3271 | 2.4230 | 2.4086 | 2.3893 | 2.3943 | 2.3643 | 2.3575 | 2.3933 |
| eval/lm/dolma_wiki-validation/PPL | lower | 10.97 | 10.74 | 10.31 | 10.25 | 11.28 | 11.12 | 10.91 | 10.96 | 10.64 | 10.56 | 10.95 |
| eval/lm/ice-validation/CE loss | lower | 2.8382 | 2.8131 | 2.7785 | 2.7816 | 2.8250 | 2.8449 | 2.8239 | 2.8230 | 2.8094 | 2.7864 | 2.8316 |
| eval/lm/ice-validation/PPL | lower | 17.09 | 16.66 | 16.09 | 16.15 | 16.86 | 17.20 | 16.84 | 16.83 | 16.60 | 16.22 | 16.97 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 2.8999 | 2.8739 | 2.8607 | 2.8391 | 2.9257 | 2.9128 | 2.8928 | 2.8969 | 2.8705 | 2.8799 | 2.9095 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 18.17 | 17.71 | 17.47 | 17.10 | 18.65 | 18.41 | 18.04 | 18.12 | 17.65 | 17.81 | 18.35 |
| eval/lm/pile-validation/CE loss | lower | 2.0578 | 2.0339 | 2.0041 | 2.0027 | 2.0717 | 2.0678 | 2.0497 | 2.0522 | 2.0346 | 2.0222 | 2.0599 |
| eval/lm/pile-validation/PPL | lower | 7.8284 | 7.6440 | 7.4193 | 7.4093 | 7.9386 | 7.9074 | 7.7657 | 7.7848 | 7.6492 | 7.5550 | 7.8450 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.3036 | 2.2759 | 2.2376 | 2.2339 | 2.3131 | 2.3267 | 2.2996 | 2.2956 | 2.2761 | 2.2657 | 2.3133 |
| eval/lm/wikitext_103-validation/PPL | lower | 10.01 | 9.7371 | 9.3709 | 9.3362 | 10.11 | 10.24 | 9.9705 | 9.9302 | 9.7384 | 9.6376 | 10.11 |
| throughput/in-loop eval batches | see metric | 828.0 | 828.0 | 1659.0 | 1659.0 | 1659.0 | 1659.0 | 1659.0 | 1659.0 | 1659.0 | 1659.0 | 1659.0 |
| throughput/in-loop eval time (s) | see metric | 97.65 | 97.65 | 323.6 | 259.1 | 129.2 | 133.7 | 159.0 | 148.3 | 178.7 | 201.0 | 152.0 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| eg-1p2b-cx4-eg24e2k-lr4e-4-r1<br>`ybn138lx` | finished | original | 85071495168.0 | 162261 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ybn138lx) |
| eg-1p2b-cx4-eg96e8k-lr4e-4-r1<br>`cql4y2di` | finished | original | 85257093120.0 | 162615 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/cql4y2di) |
| int-1p2b-cx4-intd256e8k-lr3e-4-r2<br>`i9vrjwe8` | finished | original | 85782429696.0 | 163617 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/i9vrjwe8) |
| int-1p2b-cx4-intw256e8k-lr3e-4-r2<br>`u7ab1tpb` | finished | original | 85669707776.0 | 163402 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/u7ab1tpb) |
| 1p2b-cx4-b512k-lr1.2e-3-r1<br>`vksk7sux` | finished | gpu8-ep1mb2 | 85133361152.0 | 162379 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vksk7sux) |
| 1p2b-cx4-b512k-lr1.5e-4-r1<br>`5u5iumvr` | finished | gpu8-ep1mb2 | 85133361152.0 | 162379 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5u5iumvr) |
| 1p2b-cx4-b512k-lr3e-4-r1<br>`rkjs2sze` | finished | gpu8-ep1mb2 | 85133361152.0 | 162379 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rkjs2sze) |
| 1p2b-cx4-b512k-lr6e-4-r1<br>`1tzma107` | finished | gpu8-ep1mb2 | 85133361152.0 | 162379 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1tzma107) |
| q3-1p2b-cx4-q3am128e8k-lr3e-4-r1<br>`15wzt5lj` | finished | original | 85355134976.0 | 162802 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/15wzt5lj) |
| q3-1p2b-cx4-q3td128e8k-lr3e-4-r1<br>`ek9f5z9p` | finished | original | 83958956032.0 | 160139 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ek9f5z9p) |
| se-1p2b-cx4-se0m9-lr3e-4-r1<br>`k3klv3au` | finished | original | 85133361152.0 | 162379 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/k3klv3au) |

## 1p2b Cx8

| metric | direction | eg-1p2b-cx8-eg24e2k-lr4e-4-r1<br>`ic0ud1xz` | eg-1p2b-cx8-eg96e8k-lr4e-4-r1<br>`mqp8xl8h` | int-1p2b-cx8-intw256e8k-lr4e-4-r2<br>`bqjzmiqi` | 1p2b-cx8-b768k-lr4e-4-r1<br>`gbt7khqj` | 1p2b-cx8-b768k-lr2e-4-r2<br>`jdrvfvfn` | 1p2b-cx8-b768k-lr8e-4-r2<br>`ja7yu1c3` | q3-1p2b-cx8-q3am128e8k-lr4e-4-r1<br>`0n34n3oj` | q3-1p2b-cx8-q3td128e8k-lr4e-4-r1<br>`r96ox1ij` | se-1p2b-cx8-se0m9-lr4e-4-r1<br>`blw5bd39` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.77581 | 0.76166 | 0.73408 | 0.77609 | 0.78557 | 0.76369 | 0.76106 | 0.75249 | 0.77206 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 0.84541 | 0.83393 | 0.80200 | 0.84790 | 0.86017 | 0.83334 | 0.83074 | 0.82258 | 0.84385 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 0.98752 | 0.99924 | 0.90644 | 1.0008 | 1.0008 | 1.0003 | 1.0022 | 0.91661 | 1.0023 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 1.9750 | 1.9985 | 1.8129 | 2.0015 | 2.0016 | 2.0006 | 2.0044 | 1.8332 | 2.0047 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.68446 | 0.69272 | 0.62831 | 0.69382 | 0.69385 | 0.69352 | 0.69475 | 0.63534 | 0.69482 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.3689 | 1.3854 | 1.2566 | 1.3876 | 1.3877 | 1.3870 | 1.3895 | 1.2707 | 1.3896 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.31570 | 0.26706 | 0.43942 | 0.26621 | 0.25683 | 0.25341 | 0.25768 | 0.43601 | 0.27218 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.31570 | 0.26706 | 0.43942 | 0.26621 | 0.25683 | 0.25341 | 0.25768 | 0.43601 | 0.27218 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.3666 | -1.3832 | -1.2545 | -1.3851 | -1.3856 | -1.3841 | -1.3878 | -1.2691 | -1.3875 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.3666 | -1.3832 | -1.2545 | -1.3851 | -1.3856 | -1.3841 | -1.3878 | -1.2691 | -1.3875 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.26566 | 0.25320 | 0.32417 | 0.25163 | 0.25159 | 0.25185 | 0.25270 | 0.32077 | 0.25165 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.26566 | 0.25320 | 0.32417 | 0.25163 | 0.25159 | 0.25185 | 0.25270 | 0.32077 | 0.25165 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.57417 | 0.55788 | 0.54985 | 0.58481 | 0.58099 | 0.57280 | 0.56353 | 0.56266 | 0.57727 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.62335 | 0.60558 | 0.59691 | 0.63506 | 0.63094 | 0.62169 | 0.61172 | 0.61163 | 0.62692 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 0.95295 | 0.99747 | 0.73712 | 1.0006 | 0.99554 | 1.0046 | 1.0057 | 0.77360 | 1.0011 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 1.9059 | 1.9949 | 1.4742 | 2.0012 | 1.9911 | 2.0092 | 2.0114 | 1.5472 | 2.0023 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.66055 | 0.69155 | 0.51096 | 0.69372 | 0.69023 | 0.69648 | 0.69717 | 0.53625 | 0.69412 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.3211 | 1.3831 | 1.0219 | 1.3874 | 1.3805 | 1.3930 | 1.3943 | 1.0725 | 1.3882 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.38258 | 0.28072 | 0.58838 | 0.27652 | 0.29419 | 0.25084 | 0.26094 | 0.58796 | 0.26263 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.38258 | 0.28072 | 0.58838 | 0.27652 | 0.29419 | 0.25084 | 0.26094 | 0.58796 | 0.26263 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.3190 | -1.3809 | -1.0194 | -1.3847 | -1.3790 | -1.3899 | -1.3926 | -1.0709 | -1.3862 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.3190 | -1.3809 | -1.0194 | -1.3847 | -1.3790 | -1.3899 | -1.3926 | -1.0709 | -1.3862 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.28402 | 0.25296 | 0.42092 | 0.25186 | 0.25341 | 0.25086 | 0.25159 | 0.38951 | 0.25121 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.28402 | 0.25296 | 0.42092 | 0.25186 | 0.25341 | 0.25086 | 0.25159 | 0.38951 | 0.25121 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 0.82653 | 0.95168 | 0.76685 | 0.89027 | 0.85197 | 0.91263 | 0.78627 | 0.83965 | 0.86706 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 1.3459 | 1.5651 | 1.2339 | 1.4505 | 1.3954 | 1.4854 | 1.2640 | 1.3502 | 1.4146 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 0.57288 | 0.65955 | 0.53151 | 0.61711 | 0.59059 | 0.63257 | 0.54496 | 0.58198 | 0.60104 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 0.93292 | 1.0847 | 0.85529 | 1.0052 | 0.96730 | 1.0295 | 0.87620 | 0.93610 | 0.98060 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.63324 | 0.60745 | 0.67526 | 0.63515 | 0.62560 | 0.60554 | 0.65521 | 0.61891 | 0.61700 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.63324 | 0.60745 | 0.67526 | 0.63515 | 0.62560 | 0.60554 | 0.65521 | 0.61891 | 0.61700 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -1.0264 | -1.2151 | -0.94952 | -1.1319 | -1.0769 | -1.1480 | -1.0217 | -1.1129 | -1.1022 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -1.0264 | -1.2151 | -0.94952 | -1.1319 | -1.0769 | -1.1480 | -1.0217 | -1.1129 | -1.1022 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.58277 | 0.56544 | 0.61189 | 0.57764 | 0.57126 | 0.57281 | 0.59452 | 0.58364 | 0.57287 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.58277 | 0.56544 | 0.61189 | 0.57764 | 0.57126 | 0.57281 | 0.59452 | 0.58364 | 0.57287 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.27326 | 0.30796 | 0.31079 | 0.31068 | 0.32633 | 0.33752 | 0.27713 | 0.32820 | 0.32031 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.29775 | 0.33664 | 0.33855 | 0.33856 | 0.35631 | 0.36809 | 0.30213 | 0.35803 | 0.34974 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.18941 | 0.21348 | 0.21540 | 0.21535 | 0.22620 | 0.23397 | 0.19207 | 0.22751 | 0.22202 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.20638 | 0.23334 | 0.23465 | 0.23463 | 0.24698 | 0.25517 | 0.20942 | 0.24820 | 0.24245 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.71640 | 0.71937 | 0.73419 | 0.71047 | 0.71838 | 0.68478 | 0.76779 | 0.75198 | 0.70356 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.71640 | 0.71937 | 0.73419 | 0.71047 | 0.71838 | 0.68478 | 0.76779 | 0.75198 | 0.70356 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -0.99098 | -1.0202 | -0.98015 | -1.0264 | -1.0517 | -1.1401 | -0.85000 | -0.92258 | -1.1300 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -0.99098 | -1.0202 | -0.98015 | -1.0264 | -1.0517 | -1.1401 | -0.85000 | -0.92258 | -1.1300 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.70062 | 0.69494 | 0.70132 | 0.69058 | 0.69456 | 0.66624 | 0.74511 | 0.71641 | 0.67574 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.70062 | 0.69494 | 0.70132 | 0.69058 | 0.69456 | 0.66624 | 0.74511 | 0.71641 | 0.67574 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.24039 | 0.24821 | 0.22905 | 0.22690 | 0.24700 | 0.24043 | 0.22604 | 0.19908 | 0.26702 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.28825 | 0.29810 | 0.27510 | 0.27187 | 0.29668 | 0.28843 | 0.27149 | 0.23860 | 0.32151 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.16678 | 0.17223 | 0.15889 | 0.15742 | 0.17135 | 0.16683 | 0.15678 | 0.13814 | 0.18523 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.20000 | 0.20689 | 0.19087 | 0.18866 | 0.20583 | 0.20016 | 0.18832 | 0.16555 | 0.22299 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.95564 | 0.94407 | 0.95371 | 0.95371 | 0.93346 | 0.95468 | 0.95371 | 0.96528 | 0.94503 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.95564 | 0.94407 | 0.95371 | 0.95371 | 0.93346 | 0.95468 | 0.95371 | 0.96528 | 0.94503 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.17073 | -0.20021 | -0.14480 | -0.17356 | -0.20769 | -0.17631 | -0.16368 | -0.14194 | -0.18259 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.17073 | -0.20021 | -0.14480 | -0.17356 | -0.20769 | -0.17631 | -0.16368 | -0.14194 | -0.18259 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.88090 | 0.86594 | 0.90495 | 0.88723 | 0.86304 | 0.88343 | 0.88643 | 0.90169 | 0.87721 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.88090 | 0.86594 | 0.90495 | 0.88723 | 0.86304 | 0.88343 | 0.88643 | 0.90169 | 0.87721 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.25892 | 0.29573 | 0.27562 | 0.27900 | 0.25564 | 0.27562 | 0.26005 | 0.25557 | 0.28354 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.26766 | 0.30572 | 0.28499 | 0.28839 | 0.26431 | 0.28498 | 0.26886 | 0.26426 | 0.29323 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.17950 | 0.20500 | 0.19105 | 0.19342 | 0.17722 | 0.19107 | 0.18026 | 0.17717 | 0.19655 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.18555 | 0.21194 | 0.19755 | 0.19991 | 0.18321 | 0.19755 | 0.18638 | 0.18317 | 0.20328 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.85599 | 0.93918 | 0.95528 | 0.93381 | 0.94365 | 0.89267 | 0.95886 | 0.92755 | 0.94544 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.85599 | 0.93918 | 0.95528 | 0.93381 | 0.94365 | 0.89267 | 0.95886 | 0.92755 | 0.94544 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.31511 | -0.18625 | -0.14402 | -0.18809 | -0.16457 | -0.26513 | -0.15412 | -0.17836 | -0.18056 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.31511 | -0.18625 | -0.14402 | -0.18809 | -0.16457 | -0.26513 | -0.15412 | -0.17836 | -0.18056 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.86087 | 0.90168 | 0.92906 | 0.91596 | 0.91875 | 0.88319 | 0.93093 | 0.90968 | 0.93061 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.86087 | 0.90168 | 0.92906 | 0.91596 | 0.91875 | 0.88319 | 0.93093 | 0.90968 | 0.93061 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 0.77969 | 0.72942 | 0.72312 | 0.65693 | 0.73518 | 0.67788 | 0.66448 | 0.77093 | 0.71052 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.2888 | 1.2261 | 1.1969 | 1.1067 | 1.2212 | 1.1306 | 1.1048 | 1.2838 | 1.1801 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.56751 | 0.53202 | 0.52787 | 0.48586 | 0.53661 | 0.49608 | 0.48505 | 0.56245 | 0.52438 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 0.96227 | 0.91762 | 0.89750 | 0.84510 | 0.91394 | 0.85007 | 0.82845 | 0.96147 | 0.89880 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.77528 | 0.80899 | 0.76966 | 0.79401 | 0.80337 | 0.79401 | 0.80712 | 0.77154 | 0.77528 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.77528 | 0.80899 | 0.76966 | 0.79401 | 0.80337 | 0.79401 | 0.80712 | 0.77154 | 0.77528 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.60697 | -0.51476 | -0.57207 | -0.53398 | -0.57274 | -0.51153 | -0.53046 | -0.58161 | -0.55636 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.60697 | -0.51476 | -0.57207 | -0.53398 | -0.57274 | -0.51153 | -0.53046 | -0.58161 | -0.55636 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.69393 | 0.72331 | 0.69776 | 0.71499 | 0.71312 | 0.72612 | 0.71440 | 0.69505 | 0.70851 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.69393 | 0.72331 | 0.69776 | 0.71499 | 0.71312 | 0.72612 | 0.71440 | 0.69505 | 0.70851 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.1565 | 1.0708 | 0.92691 | 1.0747 | 1.2688 | 1.1611 | 1.1154 | 1.0894 | 1.1753 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 1.6149 | 1.5119 | 1.3461 | 1.5207 | 1.7895 | 1.6250 | 1.5663 | 1.5354 | 1.6572 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 0.80156 | 0.74213 | 0.64254 | 0.74489 | 0.87953 | 0.80483 | 0.77308 | 0.75520 | 0.81458 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.1194 | 1.0479 | 0.93311 | 1.0541 | 1.2404 | 1.1264 | 1.0857 | 1.0642 | 1.1486 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.38474 | 0.41181 | 0.44217 | 0.39541 | 0.39459 | 0.41838 | 0.45119 | 0.45775 | 0.40361 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.38474 | 0.41181 | 0.44217 | 0.39541 | 0.39459 | 0.41838 | 0.45119 | 0.45775 | 0.40361 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -2.4840 | -2.2510 | -1.9191 | -2.4005 | -2.4206 | -2.1474 | -2.1446 | -2.1312 | -2.2571 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -2.4840 | -2.2510 | -1.9191 | -2.4005 | -2.4206 | -2.1474 | -2.1446 | -2.1312 | -2.2571 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.39618 | 0.43292 | 0.46139 | 0.40550 | 0.39877 | 0.42093 | 0.45782 | 0.47060 | 0.41885 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.39618 | 0.43292 | 0.46139 | 0.40550 | 0.39877 | 0.42093 | 0.45782 | 0.47060 | 0.41885 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.36564 | 0.34347 | 0.35027 | 0.36076 | 0.36166 | 0.35784 | 0.35293 | 0.36078 | 0.34669 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.37051 | 0.34796 | 0.35497 | 0.36557 | 0.36651 | 0.36261 | 0.35775 | 0.36568 | 0.35115 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.55630 | 0.54300 | 0.52876 | 0.54420 | 0.55420 | 0.54874 | 0.54425 | 0.54194 | 0.55116 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.56119 | 0.54779 | 0.53323 | 0.54881 | 0.55887 | 0.55339 | 0.54886 | 0.54678 | 0.55613 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 1.9710 | 1.9887 | 2.1073 | 2.0184 | 1.8868 | 1.8776 | 1.7918 | 1.4030 | 1.9604 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 3.9420 | 3.9775 | 4.2146 | 4.0369 | 3.7736 | 3.7552 | 3.5837 | 2.8061 | 3.9208 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.3664 | 1.3786 | 1.4610 | 1.3989 | 1.3076 | 1.3015 | 1.2418 | 0.97248 | 1.3586 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 2.7328 | 2.7571 | 2.9220 | 2.7978 | 2.6152 | 2.6030 | 2.4837 | 1.9450 | 2.7173 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.19000 | 0.06000 | 0.19000 | 0.11000 | 0.13000 | 0.12000 | 0.18000 | 0.51000 | 0.07000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.19000 | 0.06000 | 0.19000 | 0.11000 | 0.13000 | 0.12000 | 0.18000 | 0.51000 | 0.07000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -2.7276 | -2.7508 | -2.9195 | -2.7914 | -2.6097 | -2.5940 | -2.4792 | -1.9364 | -2.7100 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -2.7276 | -2.7508 | -2.9195 | -2.7914 | -2.6097 | -2.5940 | -2.4792 | -1.9364 | -2.7100 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.10780 | 0.09927 | 0.15178 | 0.10154 | 0.10732 | 0.10143 | 0.10869 | 0.19619 | 0.09994 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.10780 | 0.09927 | 0.15178 | 0.10154 | 0.10732 | 0.10143 | 0.10869 | 0.19619 | 0.09994 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.73968 | 0.73474 | 0.72245 | 0.73738 | 0.73885 | 0.73826 | 0.73043 | 0.73111 | 0.73635 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.74792 | 0.74290 | 0.73043 | 0.74554 | 0.74719 | 0.74651 | 0.73857 | 0.73915 | 0.74450 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.55064 | 0.53900 | 0.51550 | 0.54372 | 0.55193 | 0.54236 | 0.53399 | 0.52986 | 0.54696 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.55233 | 0.54075 | 0.51713 | 0.54553 | 0.55367 | 0.54421 | 0.53581 | 0.53157 | 0.54860 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.61051 | 0.60336 | 0.58367 | 0.61506 | 0.61785 | 0.60809 | 0.59472 | 0.59534 | 0.61447 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.63983 | 0.63253 | 0.61074 | 0.64466 | 0.64767 | 0.63698 | 0.62292 | 0.62347 | 0.64437 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 0.99461 | 1.0049 | 0.96737 | 1.0037 | 0.99763 | 1.0046 | 1.0046 | 0.96460 | 1.0022 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 1.9892 | 2.0097 | 1.9347 | 2.0073 | 1.9953 | 2.0092 | 2.0092 | 1.9292 | 2.0044 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.68948 | 0.69659 | 0.67057 | 0.69580 | 0.69166 | 0.69644 | 0.69645 | 0.66866 | 0.69476 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.3790 | 1.3932 | 1.3411 | 1.3916 | 1.3833 | 1.3929 | 1.3929 | 1.3373 | 1.3895 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.30733 | 0.28098 | 0.34601 | 0.26844 | 0.27949 | 0.25930 | 0.25526 | 0.36302 | 0.27460 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.30733 | 0.28098 | 0.34601 | 0.26844 | 0.27949 | 0.25930 | 0.25526 | 0.36302 | 0.27460 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3694 | -1.3821 | -1.3415 | -1.3818 | -1.3800 | -1.3822 | -1.3822 | -1.3383 | -1.3816 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.3770 | -1.3904 | -1.3380 | -1.3886 | -1.3805 | -1.3900 | -1.3912 | -1.3352 | -1.3870 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.25799 | 0.25286 | 0.26805 | 0.25282 | 0.25260 | 0.25290 | 0.25305 | 0.26956 | 0.25249 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.26591 | 0.25568 | 0.28563 | 0.25553 | 0.25517 | 0.25546 | 0.25548 | 0.28879 | 0.25508 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 0.85557 | 0.83602 | 0.81259 | 0.84571 | 0.86418 | 0.85949 | 0.82529 | 0.83782 | 0.86037 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 0.95454 | 0.93174 | 0.90682 | 0.94269 | 0.96294 | 0.95852 | 0.92047 | 0.93499 | 0.95941 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 0.96290 | 0.98826 | 0.88695 | 0.99194 | 0.99200 | 0.98939 | 0.98975 | 0.90355 | 0.99655 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 1.9258 | 1.9765 | 1.7739 | 1.9839 | 1.9840 | 1.9788 | 1.9795 | 1.8071 | 1.9931 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.66740 | 0.68511 | 0.61479 | 0.68767 | 0.68772 | 0.68590 | 0.68615 | 0.62629 | 0.69084 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.3348 | 1.3702 | 1.2296 | 1.3753 | 1.3754 | 1.3718 | 1.3723 | 1.2526 | 1.3817 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.37199 | 0.29642 | 0.42967 | 0.27360 | 0.28007 | 0.30290 | 0.27391 | 0.43492 | 0.26157 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.37199 | 0.29642 | 0.42967 | 0.27360 | 0.28007 | 0.30290 | 0.27391 | 0.43492 | 0.26157 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3450 | -1.3721 | -1.2658 | -1.3752 | -1.3751 | -1.3719 | -1.3706 | -1.2755 | -1.3791 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.3325 | -1.3670 | -1.2257 | -1.3721 | -1.3720 | -1.3679 | -1.3700 | -1.2500 | -1.3790 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.26535 | 0.25514 | 0.29551 | 0.25424 | 0.25416 | 0.25540 | 0.25677 | 0.29401 | 0.25284 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.28040 | 0.26004 | 0.33656 | 0.25829 | 0.25835 | 0.26047 | 0.26181 | 0.33486 | 0.25578 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.73293 | 0.71424 | 0.68706 | 0.72656 | 0.73950 | 0.72969 | 0.71386 | 0.70541 | 0.73070 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 0.78223 | 0.76138 | 0.73264 | 0.77483 | 0.78920 | 0.77849 | 0.76156 | 0.75214 | 0.77978 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 0.97571 | 0.98904 | 0.90561 | 0.98654 | 0.99114 | 0.99232 | 0.98816 | 0.91523 | 0.99246 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 1.9514 | 1.9781 | 1.8112 | 1.9731 | 1.9823 | 1.9846 | 1.9763 | 1.8305 | 1.9849 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.67636 | 0.68565 | 0.62772 | 0.68399 | 0.68710 | 0.68793 | 0.68505 | 0.63444 | 0.68798 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.3527 | 1.3713 | 1.2554 | 1.3680 | 1.3742 | 1.3759 | 1.3701 | 1.2689 | 1.3760 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.33702 | 0.28339 | 0.41436 | 0.29347 | 0.28794 | 0.28047 | 0.28079 | 0.43029 | 0.28632 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.33702 | 0.28339 | 0.41436 | 0.29347 | 0.28794 | 0.28047 | 0.28079 | 0.43029 | 0.28632 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3522 | -1.3718 | -1.2869 | -1.3708 | -1.3747 | -1.3735 | -1.3710 | -1.2951 | -1.3770 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.3507 | -1.3682 | -1.2523 | -1.3644 | -1.3713 | -1.3723 | -1.3681 | -1.2665 | -1.3739 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.26388 | 0.25554 | 0.28613 | 0.25555 | 0.25427 | 0.25527 | 0.25614 | 0.28379 | 0.25328 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.27775 | 0.26114 | 0.32010 | 0.26124 | 0.25863 | 0.26047 | 0.26179 | 0.31680 | 0.25673 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.0993 | 1.0738 | 1.0399 | 1.0793 | 1.1184 | 1.0952 | 1.0593 | 1.0612 | 1.0943 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.3704 | 1.3387 | 1.2952 | 1.3428 | 1.3939 | 1.3613 | 1.3222 | 1.3212 | 1.3631 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 0.99856 | 1.0052 | 0.95365 | 1.0018 | 1.0041 | 0.99912 | 1.0072 | 0.96493 | 0.99464 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 1.9971 | 2.0104 | 1.9073 | 2.0037 | 2.0081 | 1.9982 | 2.0145 | 1.9299 | 1.9893 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.69220 | 0.69680 | 0.66106 | 0.69453 | 0.69602 | 0.69264 | 0.69825 | 0.66890 | 0.68952 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.3844 | 1.3936 | 1.3221 | 1.3891 | 1.3920 | 1.3853 | 1.3965 | 1.3378 | 1.3790 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.30683 | 0.27899 | 0.35288 | 0.27203 | 0.29490 | 0.28694 | 0.25779 | 0.35653 | 0.28396 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.30683 | 0.27899 | 0.35288 | 0.27203 | 0.29490 | 0.28694 | 0.25779 | 0.35653 | 0.28396 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3730 | -1.3820 | -1.3324 | -1.3812 | -1.3814 | -1.3779 | -1.3838 | -1.3408 | -1.3779 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.3822 | -1.3901 | -1.3188 | -1.3861 | -1.3887 | -1.3815 | -1.3941 | -1.3354 | -1.3764 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.25669 | 0.25281 | 0.27013 | 0.25287 | 0.25303 | 0.25384 | 0.25260 | 0.26802 | 0.25306 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.26347 | 0.25567 | 0.28912 | 0.25575 | 0.25630 | 0.25778 | 0.25518 | 0.28601 | 0.25622 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.35945 | 0.36947 | 0.35622 | 0.35399 | 0.37091 | 0.36844 | 0.36425 | 0.34291 | 0.35856 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.36155 | 0.37164 | 0.35830 | 0.35603 | 0.37320 | 0.37053 | 0.36641 | 0.34487 | 0.36071 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.27320 | 0.28452 | 0.26499 | 0.27327 | 0.28688 | 0.28339 | 0.28432 | 0.26756 | 0.26265 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.27421 | 0.28565 | 0.26602 | 0.27426 | 0.28797 | 0.28448 | 0.28543 | 0.26861 | 0.26371 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.48303 | 0.46589 | 0.45382 | 0.47450 | 0.48039 | 0.49244 | 0.46373 | 0.42227 | 0.44002 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.48646 | 0.46926 | 0.45696 | 0.47783 | 0.48373 | 0.49585 | 0.46709 | 0.42524 | 0.44311 |
| eval/lm/c4_en-validation/CE loss | lower | 2.7077 | 2.6848 | 2.6376 | 2.6988 | 2.7104 | 2.7080 | 2.6750 | 2.6633 | 2.7089 |
| eval/lm/c4_en-validation/PPL | lower | 14.99 | 14.65 | 13.98 | 14.86 | 15.04 | 15.00 | 14.51 | 14.34 | 15.01 |
| eval/lm/dolma_books-validation/CE loss | lower | 2.5542 | 2.5180 | 2.4726 | 2.5353 | 2.5557 | 2.5474 | 2.5091 | 2.5016 | 2.5554 |
| eval/lm/dolma_books-validation/PPL | lower | 12.86 | 12.40 | 11.85 | 12.62 | 12.88 | 12.77 | 12.29 | 12.20 | 12.88 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 2.8442 | 2.8231 | 2.7742 | 2.8358 | 2.8497 | 2.8452 | 2.8130 | 2.8012 | 2.8446 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 17.19 | 16.83 | 16.03 | 17.04 | 17.28 | 17.20 | 16.66 | 16.46 | 17.19 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 1.9470 | 1.9286 | 1.8993 | 1.9389 | 1.9486 | 1.9437 | 1.9232 | 1.9148 | 1.9450 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 7.0076 | 6.8797 | 6.6810 | 6.9514 | 7.0188 | 6.9843 | 6.8429 | 6.7855 | 6.9938 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.0449 | 3.0214 | 2.9845 | 3.0361 | 3.0490 | 3.0400 | 3.0139 | 3.0055 | 3.0417 |
| eval/lm/dolma_reddit-validation/PPL | lower | 21.01 | 20.52 | 19.78 | 20.82 | 21.09 | 20.91 | 20.37 | 20.20 | 20.94 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.1505 | 1.1344 | 1.1035 | 1.1417 | 1.1495 | 1.1488 | 1.1295 | 1.1205 | 1.1485 |
| eval/lm/dolma_stack-validation/PPL | lower | 3.1597 | 3.1092 | 3.0146 | 3.1320 | 3.1565 | 3.1543 | 3.0940 | 3.0663 | 3.1535 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.3351 | 2.3143 | 2.2622 | 2.3277 | 2.3371 | 2.3391 | 2.3006 | 2.2954 | 2.3331 |
| eval/lm/dolma_wiki-validation/PPL | lower | 10.33 | 10.12 | 9.6046 | 10.25 | 10.35 | 10.37 | 9.9800 | 9.9286 | 10.31 |
| eval/lm/ice-validation/CE loss | lower | 2.7638 | 2.7404 | 2.7204 | 2.7581 | 2.7772 | 2.7605 | 2.7426 | 2.7313 | 2.7673 |
| eval/lm/ice-validation/PPL | lower | 15.86 | 15.49 | 15.19 | 15.77 | 16.07 | 15.81 | 15.53 | 15.35 | 15.92 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 2.8535 | 2.8292 | 2.7898 | 2.8542 | 2.8584 | 2.8610 | 2.8235 | 2.8221 | 2.8725 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 17.35 | 16.93 | 16.28 | 17.36 | 17.43 | 17.48 | 16.83 | 16.81 | 17.68 |
| eval/lm/pile-validation/CE loss | lower | 1.9984 | 1.9773 | 1.9349 | 1.9901 | 1.9980 | 1.9988 | 1.9699 | 1.9622 | 1.9976 |
| eval/lm/pile-validation/PPL | lower | 7.3769 | 7.2234 | 6.9230 | 7.3161 | 7.3742 | 7.3800 | 7.1701 | 7.1150 | 7.3716 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.2327 | 2.2021 | 2.1519 | 2.2206 | 2.2372 | 2.2263 | 2.1936 | 2.1835 | 2.2314 |
| eval/lm/wikitext_103-validation/PPL | lower | 9.3251 | 9.0441 | 8.6011 | 9.2127 | 9.3674 | 9.2656 | 8.9676 | 8.8776 | 9.3124 |
| throughput/in-loop eval batches | see metric | 828.0 | 828.0 | 1111.0 | 870.0 | 828.0 | 828.0 | 828.0 | 828.0 | 828.0 |
| throughput/in-loop eval time (s) | see metric | 93.61 | 113.8 | 244.4 | 73.70 | 96.68 | 94.74 | 149.5 | 119.6 | 94.07 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| eg-1p2b-cx8-eg24e2k-lr4e-4-r1<br>`ic0ud1xz` | finished | original | 170142203904.0 | 216347 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ic0ud1xz) |
| eg-1p2b-cx8-eg96e8k-lr4e-4-r1<br>`mqp8xl8h` | finished | original | 170514186240.0 | 216820 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mqp8xl8h) |
| int-1p2b-cx8-intw256e8k-lr4e-4-r2<br>`bqjzmiqi` | finished | original | 171339939840.0 | 217870 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bqjzmiqi) |
| 1p2b-cx8-b768k-lr4e-4-r1<br>`gbt7khqj` | finished | gpu32-ep1mb1 | 170266460160.0 | 216505 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gbt7khqj) |
| 1p2b-cx8-b768k-lr2e-4-r2<br>`jdrvfvfn` | finished | gpu8-ep1mb4 | 170266460160.0 | 216505 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jdrvfvfn) |
| 1p2b-cx8-b768k-lr8e-4-r2<br>`ja7yu1c3` | finished | gpu8-ep1mb4 | 170266460160.0 | 216505 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ja7yu1c3) |
| q3-1p2b-cx8-q3am128e8k-lr4e-4-r1<br>`0n34n3oj` | finished | original | 170710794240.0 | 217070 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0n34n3oj) |
| q3-1p2b-cx8-q3td128e8k-lr4e-4-r1<br>`r96ox1ij` | finished | original | 167918174208.0 | 213519 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/r96ox1ij) |
| se-1p2b-cx8-se0m9-lr4e-4-r1<br>`blw5bd39` | finished | original | 170266460160.0 | 216505 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/blw5bd39) |

## 275m Cx1

| metric | direction | eg-275m-cx1-eg192e16k-lr1e-3-r1<br>`psingqsk` | eg-275m-cx1-eg192e16k-lr2e-3-r1<br>`idiscwdx` | eg-275m-cx1-eg192e16k-lr4e-3-r1<br>`d9oavanz` | eg-275m-cx1-eg24e2k-lr1e-3-r1<br>`6o5xk53j` | eg-275m-cx1-eg24e2k-lr2e-3-r1<br>`ndkmprhm` | eg-275m-cx1-eg24e2k-lr4e-3-r1<br>`8eoup1kb` | eg-275m-cx1-eg384e32k-lr1e-3-r1<br>`sqxqy801` | eg-275m-cx1-eg384e32k-lr2e-3-r1<br>`sqv1l0qt` | eg-275m-cx1-eg384e32k-lr4e-3-r1<br>`ltzr841g` | eg-275m-cx1-eg96e8k-lr1e-3-r1<br>`bpdo8b6b` | eg-275m-cx1-eg96e8k-lr2e-3-r1<br>`lyky04e5` | eg-275m-cx1-eg96e8k-lr4e-3-r1<br>`4l7axfwy` | 275m-cx1-b256k-lr1.2e-3-r2<br>`jxjfkaur` | 275m-cx1-b256k-lr1.5e-3-r2<br>`a41n3bxy` | 275m-cx1-b256k-lr1e-3-r2<br>`x3s68iah` | 275m-cx1-b256k-lr2e-3-r2<br>`hvr2xuvf` | 275m-cx1-b256k-lr8e-4-r2<br>`mcb2thco` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 1.0933 | 1.0994 | 1.0841 | 1.0906 | 1.1052 | 1.1119 | 1.0771 | 1.0543 | 1.0766 | 1.0797 | 1.0689 | 1.0649 | 1.0802 | 1.0977 | 1.0865 | 1.0896 | 1.0977 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 1.1957 | 1.2036 | 1.1862 | 1.1930 | 1.2098 | 1.2168 | 1.1808 | 1.1557 | 1.1804 | 1.1808 | 1.1713 | 1.1678 | 1.1823 | 1.2022 | 1.1885 | 1.1931 | 1.2017 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0400 | 1.0235 | 1.0519 | 1.0315 | 1.0654 | 1.0336 | 1.0227 | 1.0561 | 1.0458 | 1.0278 | 1.0324 | 1.0568 | 1.0284 | 1.0712 | 1.0260 | 1.0463 | 1.0284 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0800 | 2.0470 | 2.1039 | 2.0629 | 2.1308 | 2.0672 | 2.0454 | 2.1121 | 2.0916 | 2.0556 | 2.0647 | 2.1137 | 2.0568 | 2.1424 | 2.0520 | 2.0926 | 2.0568 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.72093 | 0.70948 | 0.72913 | 0.71507 | 0.73849 | 0.71652 | 0.70895 | 0.73201 | 0.72486 | 0.71254 | 0.71557 | 0.73253 | 0.71288 | 0.74247 | 0.71126 | 0.72514 | 0.71288 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4419 | 1.4190 | 1.4583 | 1.4301 | 1.4770 | 1.4330 | 1.4179 | 1.4640 | 1.4497 | 1.4251 | 1.4311 | 1.4651 | 1.4258 | 1.4849 | 1.4225 | 1.4503 | 1.4258 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.26792 | 0.25853 | 0.26365 | 0.25853 | 0.26365 | 0.26024 | 0.27645 | 0.25085 | 0.27218 | 0.26365 | 0.25341 | 0.26109 | 0.25683 | 0.25000 | 0.25853 | 0.25597 | 0.27986 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.26792 | 0.25853 | 0.26365 | 0.25853 | 0.26365 | 0.26024 | 0.27645 | 0.25085 | 0.27218 | 0.26365 | 0.25341 | 0.26109 | 0.25683 | 0.25000 | 0.25853 | 0.25597 | 0.27986 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.4242 | -1.4048 | -1.4478 | -1.3938 | -1.4571 | -1.4158 | -1.4080 | -1.4387 | -1.4300 | -1.4115 | -1.4215 | -1.4455 | -1.4073 | -1.4626 | -1.4063 | -1.4340 | -1.4163 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.4242 | -1.4048 | -1.4478 | -1.3938 | -1.4571 | -1.4158 | -1.4080 | -1.4387 | -1.4300 | -1.4115 | -1.4215 | -1.4455 | -1.4073 | -1.4626 | -1.4063 | -1.4340 | -1.4163 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.25297 | 0.25293 | 0.25463 | 0.25232 | 0.25792 | 0.25318 | 0.25140 | 0.25345 | 0.25084 | 0.25083 | 0.25079 | 0.25487 | 0.25336 | 0.24902 | 0.25328 | 0.25134 | 0.25426 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.25297 | 0.25293 | 0.25463 | 0.25232 | 0.25792 | 0.25318 | 0.25140 | 0.25345 | 0.25084 | 0.25083 | 0.25079 | 0.25487 | 0.25336 | 0.24902 | 0.25328 | 0.25134 | 0.25426 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.90328 | 0.90643 | 0.90031 | 0.92848 | 0.93822 | 0.95302 | 0.89053 | 0.88015 | 0.89411 | 0.90477 | 0.88325 | 0.88424 | 0.91202 | 0.92851 | 0.92069 | 0.91722 | 0.93217 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.98502 | 0.98815 | 0.98205 | 1.0126 | 1.0232 | 1.0396 | 0.97232 | 0.96007 | 0.97539 | 0.98706 | 0.96337 | 0.96412 | 0.99458 | 1.0130 | 1.0044 | 1.0004 | 1.0174 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0433 | 1.0441 | 1.0780 | 1.0526 | 1.0974 | 1.0633 | 1.0260 | 1.0686 | 1.0571 | 1.0305 | 1.0267 | 1.1117 | 1.0435 | 1.0576 | 1.0448 | 1.0786 | 1.0738 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0867 | 2.0882 | 2.1560 | 2.1053 | 2.1948 | 2.1266 | 2.0520 | 2.1371 | 2.1142 | 2.0611 | 2.0535 | 2.2234 | 2.0871 | 2.1153 | 2.0896 | 2.1572 | 2.1477 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.72323 | 0.72369 | 0.74715 | 0.72974 | 0.76066 | 0.73705 | 0.71121 | 0.74065 | 0.73267 | 0.71438 | 0.71176 | 0.77058 | 0.72336 | 0.73309 | 0.72427 | 0.74761 | 0.74431 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4465 | 1.4474 | 1.4943 | 1.4595 | 1.5213 | 1.4741 | 1.4224 | 1.4813 | 1.4653 | 1.4288 | 1.4235 | 1.5412 | 1.4467 | 1.4662 | 1.4485 | 1.4952 | 1.4886 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.24790 | 0.24369 | 0.25631 | 0.23443 | 0.25337 | 0.24327 | 0.26684 | 0.25126 | 0.25126 | 0.26768 | 0.25589 | 0.26178 | 0.24705 | 0.24832 | 0.24200 | 0.24579 | 0.24369 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.24790 | 0.24369 | 0.25631 | 0.23443 | 0.25337 | 0.24327 | 0.26684 | 0.25126 | 0.25126 | 0.26768 | 0.25589 | 0.26178 | 0.24705 | 0.24832 | 0.24200 | 0.24579 | 0.24369 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.4282 | -1.4313 | -1.4814 | -1.4247 | -1.4931 | -1.4545 | -1.4119 | -1.4612 | -1.4461 | -1.4116 | -1.4118 | -1.5134 | -1.4302 | -1.4483 | -1.4338 | -1.4760 | -1.4778 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.4282 | -1.4313 | -1.4814 | -1.4247 | -1.4931 | -1.4545 | -1.4119 | -1.4612 | -1.4461 | -1.4116 | -1.4118 | -1.5134 | -1.4302 | -1.4483 | -1.4338 | -1.4760 | -1.4778 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.25114 | 0.24903 | 0.25183 | 0.24847 | 0.24814 | 0.24862 | 0.25212 | 0.24975 | 0.24844 | 0.25156 | 0.25065 | 0.25023 | 0.24901 | 0.25040 | 0.24864 | 0.24870 | 0.24882 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.25114 | 0.24903 | 0.25183 | 0.24847 | 0.24814 | 0.24862 | 0.25212 | 0.24975 | 0.24844 | 0.25156 | 0.25065 | 0.25023 | 0.24901 | 0.25040 | 0.24864 | 0.24870 | 0.24882 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 2.1268 | 2.1740 | 2.1760 | 2.2319 | 2.2823 | 2.2636 | 2.2686 | 2.2752 | 2.1987 | 2.1947 | 2.2353 | 2.2257 | 2.1840 | 2.1941 | 2.1681 | 2.3033 | 2.1605 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 3.4118 | 3.4681 | 3.4521 | 3.5621 | 3.6249 | 3.6054 | 3.6483 | 3.6480 | 3.5112 | 3.5269 | 3.5524 | 3.5384 | 3.5039 | 3.4946 | 3.4980 | 3.6804 | 3.4497 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 1.4743 | 1.5068 | 1.5083 | 1.5470 | 1.5819 | 1.5690 | 1.5725 | 1.5772 | 1.5240 | 1.5213 | 1.5494 | 1.5426 | 1.5140 | 1.5210 | 1.5030 | 1.5964 | 1.4975 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 2.3651 | 2.4042 | 2.3927 | 2.4691 | 2.5125 | 2.4989 | 2.5288 | 2.5287 | 2.4337 | 2.4448 | 2.4625 | 2.4524 | 2.4292 | 2.4226 | 2.4248 | 2.5510 | 2.3911 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.08978 | 0.08787 | 0.07545 | 0.07927 | 0.08787 | 0.07832 | 0.10220 | 0.10411 | 0.07736 | 0.10888 | 0.07354 | 0.07259 | 0.09074 | 0.08023 | 0.10124 | 0.06781 | 0.08309 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.08978 | 0.08787 | 0.07545 | 0.07927 | 0.08787 | 0.07832 | 0.10220 | 0.10411 | 0.07736 | 0.10888 | 0.07354 | 0.07259 | 0.09074 | 0.08023 | 0.10124 | 0.06781 | 0.08309 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -2.5671 | -2.6891 | -2.7192 | -2.6232 | -2.7977 | -2.6977 | -2.7191 | -2.8310 | -2.6710 | -2.6201 | -2.6631 | -2.7079 | -2.6827 | -2.7578 | -2.5554 | -2.7609 | -2.5844 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -2.5671 | -2.6891 | -2.7192 | -2.6232 | -2.7977 | -2.6977 | -2.7191 | -2.8310 | -2.6710 | -2.6201 | -2.6631 | -2.7079 | -2.6827 | -2.7578 | -2.5554 | -2.7609 | -2.5844 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.10299 | 0.10120 | 0.09637 | 0.09582 | 0.09758 | 0.09907 | 0.10362 | 0.10178 | 0.09795 | 0.10646 | 0.09414 | 0.09493 | 0.09761 | 0.09887 | 0.10534 | 0.09578 | 0.10070 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.10299 | 0.10120 | 0.09637 | 0.09582 | 0.09758 | 0.09907 | 0.10362 | 0.10178 | 0.09795 | 0.10646 | 0.09414 | 0.09493 | 0.09761 | 0.09887 | 0.10534 | 0.09578 | 0.10070 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.63994 | 0.62769 | 0.59472 | 0.67074 | 0.61700 | 0.69625 | 0.60633 | 0.63123 | 0.67732 | 0.63852 | 0.65561 | 0.60421 | 0.67292 | 0.63867 | 0.66897 | 0.61766 | 0.60643 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.69661 | 0.68241 | 0.64676 | 0.73155 | 0.67458 | 0.75853 | 0.66109 | 0.68874 | 0.73860 | 0.69721 | 0.71411 | 0.65569 | 0.73616 | 0.69832 | 0.72910 | 0.67401 | 0.66204 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.44356 | 0.43512 | 0.41226 | 0.46497 | 0.42768 | 0.48261 | 0.42024 | 0.43750 | 0.46951 | 0.44258 | 0.45443 | 0.41882 | 0.46645 | 0.44265 | 0.46366 | 0.42812 | 0.42036 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.48283 | 0.47306 | 0.44825 | 0.50706 | 0.46757 | 0.52568 | 0.45829 | 0.47738 | 0.51194 | 0.48330 | 0.49497 | 0.45452 | 0.51024 | 0.48403 | 0.50539 | 0.46718 | 0.45887 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.40514 | 0.37846 | 0.43182 | 0.36265 | 0.40119 | 0.38834 | 0.37846 | 0.39032 | 0.38735 | 0.39032 | 0.38933 | 0.37747 | 0.40020 | 0.40810 | 0.37648 | 0.41700 | 0.39625 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.40514 | 0.37846 | 0.43182 | 0.36265 | 0.40119 | 0.38834 | 0.37846 | 0.39032 | 0.38735 | 0.39032 | 0.38933 | 0.37747 | 0.40020 | 0.40810 | 0.37648 | 0.41700 | 0.39625 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -4.1686 | -4.2215 | -4.0449 | -4.5480 | -4.2518 | -4.6508 | -4.1434 | -4.1419 | -4.3768 | -4.1104 | -4.3468 | -4.3032 | -4.1850 | -4.0977 | -4.4652 | -3.9831 | -4.1502 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -4.1686 | -4.2215 | -4.0449 | -4.5480 | -4.2518 | -4.6508 | -4.1434 | -4.1419 | -4.3768 | -4.1104 | -4.3468 | -4.3032 | -4.1850 | -4.0977 | -4.4652 | -3.9831 | -4.1502 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.38425 | 0.37057 | 0.40478 | 0.35605 | 0.38124 | 0.35793 | 0.37114 | 0.37682 | 0.36862 | 0.38138 | 0.36514 | 0.37502 | 0.38949 | 0.38525 | 0.36732 | 0.39259 | 0.37715 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.38425 | 0.37057 | 0.40478 | 0.35605 | 0.38124 | 0.35793 | 0.37114 | 0.37682 | 0.36862 | 0.38138 | 0.36514 | 0.37502 | 0.38949 | 0.38525 | 0.36732 | 0.39259 | 0.37715 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.88334 | 0.78686 | 0.76610 | 0.91229 | 0.80234 | 0.75274 | 0.76470 | 0.77802 | 0.82866 | 0.78730 | 0.78634 | 0.81977 | 0.74119 | 0.78603 | 0.76888 | 0.88824 | 0.79995 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 1.0610 | 0.94461 | 0.92064 | 1.0952 | 0.96220 | 0.90526 | 0.91874 | 0.93428 | 0.99500 | 0.94498 | 0.94387 | 0.98415 | 0.89028 | 0.94389 | 0.92459 | 1.0687 | 0.96096 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.61249 | 0.54569 | 0.53124 | 0.63258 | 0.55640 | 0.52203 | 0.53028 | 0.53958 | 0.57472 | 0.54600 | 0.54528 | 0.56847 | 0.51393 | 0.54510 | 0.53325 | 0.61602 | 0.55467 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.73576 | 0.65507 | 0.63840 | 0.75952 | 0.66735 | 0.62775 | 0.63714 | 0.64788 | 0.68999 | 0.65534 | 0.65459 | 0.68252 | 0.61737 | 0.65460 | 0.64112 | 0.74112 | 0.66635 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.67213 | 0.67695 | 0.69142 | 0.61427 | 0.65284 | 0.67985 | 0.68274 | 0.69817 | 0.65863 | 0.66635 | 0.69527 | 0.65188 | 0.66538 | 0.68949 | 0.69045 | 0.64417 | 0.64609 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.67213 | 0.67695 | 0.69142 | 0.61427 | 0.65284 | 0.67985 | 0.68274 | 0.69817 | 0.65863 | 0.66635 | 0.69527 | 0.65188 | 0.66538 | 0.68949 | 0.69045 | 0.64417 | 0.64609 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.89201 | -0.86735 | -0.87064 | -0.97276 | -0.90446 | -0.86790 | -0.85110 | -0.84152 | -0.85376 | -0.88639 | -0.80592 | -0.87165 | -0.88203 | -0.84373 | -0.82925 | -0.89690 | -0.89258 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.89201 | -0.86735 | -0.87064 | -0.97276 | -0.90446 | -0.86790 | -0.85110 | -0.84152 | -0.85376 | -0.88639 | -0.80592 | -0.87165 | -0.88203 | -0.84373 | -0.82925 | -0.89690 | -0.89258 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.56270 | 0.57810 | 0.58605 | 0.54282 | 0.56588 | 0.58925 | 0.58106 | 0.58519 | 0.58441 | 0.56445 | 0.59750 | 0.57972 | 0.57361 | 0.58728 | 0.59062 | 0.56446 | 0.57034 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.56270 | 0.57810 | 0.58605 | 0.54282 | 0.56588 | 0.58925 | 0.58106 | 0.58519 | 0.58441 | 0.56445 | 0.59750 | 0.57972 | 0.57361 | 0.58728 | 0.59062 | 0.56446 | 0.57034 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.30366 | 0.28749 | 0.27272 | 0.30860 | 0.29387 | 0.31918 | 0.29343 | 0.31751 | 0.29592 | 0.28867 | 0.34267 | 0.31691 | 0.31247 | 0.28570 | 0.30309 | 0.28729 | 0.31103 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.31395 | 0.29705 | 0.28184 | 0.31909 | 0.30380 | 0.33015 | 0.30334 | 0.32820 | 0.30600 | 0.29836 | 0.35430 | 0.32755 | 0.32299 | 0.29534 | 0.31327 | 0.29704 | 0.32147 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.21051 | 0.19929 | 0.18906 | 0.21391 | 0.20371 | 0.22126 | 0.20340 | 0.22009 | 0.20514 | 0.20010 | 0.23755 | 0.21968 | 0.21660 | 0.19805 | 0.21010 | 0.19915 | 0.21561 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.21763 | 0.20592 | 0.19539 | 0.22120 | 0.21060 | 0.22885 | 0.21027 | 0.22750 | 0.21212 | 0.20683 | 0.24560 | 0.22705 | 0.22390 | 0.20474 | 0.21716 | 0.20591 | 0.22284 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.79785 | 0.79159 | 0.80322 | 0.76655 | 0.79785 | 0.79159 | 0.79249 | 0.77639 | 0.79964 | 0.80590 | 0.79159 | 0.77549 | 0.80590 | 0.81127 | 0.79159 | 0.82111 | 0.78980 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.79785 | 0.79159 | 0.80322 | 0.76655 | 0.79785 | 0.79159 | 0.79249 | 0.77639 | 0.79964 | 0.80590 | 0.79159 | 0.77549 | 0.80590 | 0.81127 | 0.79159 | 0.82111 | 0.78980 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.61004 | -0.58063 | -0.56600 | -0.67431 | -0.62121 | -0.66358 | -0.66844 | -0.69358 | -0.62148 | -0.56445 | -0.57947 | -0.63475 | -0.54198 | -0.59203 | -0.62514 | -0.52473 | -0.58480 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.61004 | -0.58063 | -0.56600 | -0.67431 | -0.62121 | -0.66358 | -0.66844 | -0.69358 | -0.62148 | -0.56445 | -0.57947 | -0.63475 | -0.54198 | -0.59203 | -0.62514 | -0.52473 | -0.58480 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.77679 | 0.78828 | 0.79033 | 0.75879 | 0.78538 | 0.78411 | 0.77912 | 0.75600 | 0.77743 | 0.79272 | 0.77709 | 0.76881 | 0.77639 | 0.79323 | 0.77452 | 0.80086 | 0.77977 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.77679 | 0.78828 | 0.79033 | 0.75879 | 0.78538 | 0.78411 | 0.77912 | 0.75600 | 0.77743 | 0.79272 | 0.77709 | 0.76881 | 0.77639 | 0.79323 | 0.77452 | 0.80086 | 0.77977 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 1.2761 | 1.2505 | 1.2430 | 1.3165 | 1.3357 | 1.2753 | 1.2502 | 1.2992 | 1.2685 | 1.3589 | 1.3079 | 1.3134 | 1.3444 | 1.4304 | 1.3309 | 1.4451 | 1.3067 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 2.0201 | 1.9720 | 1.9748 | 2.0745 | 2.1031 | 2.0115 | 1.9731 | 2.0510 | 2.0082 | 2.1394 | 2.0706 | 2.0874 | 2.1372 | 2.2716 | 2.0971 | 2.2979 | 2.0533 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.92305 | 0.91132 | 0.90726 | 0.95143 | 0.96133 | 0.92183 | 0.90396 | 0.93793 | 0.91517 | 0.97780 | 0.94818 | 0.95575 | 0.97576 | 1.0282 | 0.95772 | 1.0355 | 0.94026 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.4996 | 1.4810 | 1.4864 | 1.5385 | 1.5489 | 1.4920 | 1.4640 | 1.5178 | 1.4831 | 1.5749 | 1.5420 | 1.5634 | 1.5935 | 1.6695 | 1.5443 | 1.6797 | 1.5116 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.58052 | 0.56180 | 0.53371 | 0.57678 | 0.55431 | 0.54682 | 0.55618 | 0.56367 | 0.55056 | 0.56180 | 0.55431 | 0.54682 | 0.53184 | 0.54120 | 0.54120 | 0.53184 | 0.57116 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.58052 | 0.56180 | 0.53371 | 0.57678 | 0.55431 | 0.54682 | 0.55618 | 0.56367 | 0.55056 | 0.56180 | 0.55431 | 0.54682 | 0.53184 | 0.54120 | 0.54120 | 0.53184 | 0.57116 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -1.1447 | -1.1162 | -1.2178 | -1.1702 | -1.1599 | -1.1859 | -1.1145 | -1.2104 | -1.1728 | -1.1446 | -1.1828 | -1.1851 | -1.1777 | -1.1781 | -1.1587 | -1.1877 | -1.1447 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -1.1447 | -1.1162 | -1.2178 | -1.1702 | -1.1599 | -1.1859 | -1.1145 | -1.2104 | -1.1728 | -1.1446 | -1.1828 | -1.1851 | -1.1777 | -1.1781 | -1.1587 | -1.1877 | -1.1447 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.49362 | 0.49847 | 0.47262 | 0.48624 | 0.48478 | 0.47650 | 0.49968 | 0.48903 | 0.48743 | 0.50220 | 0.49534 | 0.48316 | 0.47829 | 0.47435 | 0.48583 | 0.47757 | 0.49295 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.49362 | 0.49847 | 0.47262 | 0.48624 | 0.48478 | 0.47650 | 0.49968 | 0.48903 | 0.48743 | 0.50220 | 0.49534 | 0.48316 | 0.47829 | 0.47435 | 0.48583 | 0.47757 | 0.49295 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 2.3597 | 2.3465 | 2.3419 | 2.3625 | 2.3396 | 2.3142 | 2.3482 | 2.3997 | 2.2747 | 2.3890 | 2.3102 | 2.2695 | 2.4289 | 2.3999 | 2.3689 | 2.3127 | 2.4583 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 3.2650 | 3.2126 | 3.1946 | 3.2604 | 3.2342 | 3.1753 | 3.2387 | 3.2855 | 3.1215 | 3.2802 | 3.1713 | 3.1006 | 3.3341 | 3.2973 | 3.2709 | 3.1693 | 3.3503 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.6356 | 1.6264 | 1.6232 | 1.6376 | 1.6216 | 1.6039 | 1.6278 | 1.6634 | 1.5766 | 1.6559 | 1.6013 | 1.5730 | 1.6835 | 1.6635 | 1.6421 | 1.6030 | 1.7039 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 2.2630 | 2.2268 | 2.2145 | 2.2601 | 2.2415 | 2.2009 | 2.2447 | 2.2774 | 2.1637 | 2.2737 | 2.1981 | 2.1492 | 2.3109 | 2.2855 | 2.2673 | 2.1970 | 2.3221 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.23216 | 0.22642 | 0.22149 | 0.22313 | 0.21739 | 0.21493 | 0.23544 | 0.22231 | 0.20837 | 0.21903 | 0.21985 | 0.21575 | 0.22724 | 0.22642 | 0.23626 | 0.22231 | 0.20837 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.23216 | 0.22642 | 0.22149 | 0.22313 | 0.21739 | 0.21493 | 0.23544 | 0.22231 | 0.20837 | 0.21903 | 0.21985 | 0.21575 | 0.22724 | 0.22642 | 0.23626 | 0.22231 | 0.20837 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -4.8990 | -4.9975 | -4.8056 | -4.8993 | -5.0737 | -4.9906 | -4.9873 | -4.7485 | -4.9943 | -4.9231 | -4.9564 | -4.6324 | -4.9366 | -4.8388 | -4.7912 | -5.1488 | -4.9414 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -4.8990 | -4.9975 | -4.8056 | -4.8993 | -5.0737 | -4.9906 | -4.9873 | -4.7485 | -4.9943 | -4.9231 | -4.9564 | -4.6324 | -4.9366 | -4.8388 | -4.7912 | -5.1488 | -4.9414 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.24056 | 0.23274 | 0.23383 | 0.23224 | 0.22469 | 0.22243 | 0.23686 | 0.23536 | 0.22134 | 0.23172 | 0.23502 | 0.23366 | 0.23541 | 0.23990 | 0.24259 | 0.23235 | 0.22395 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.24056 | 0.23274 | 0.23383 | 0.23224 | 0.22469 | 0.22243 | 0.23686 | 0.23536 | 0.22134 | 0.23172 | 0.23502 | 0.23366 | 0.23541 | 0.23990 | 0.24259 | 0.23235 | 0.22395 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.62812 | 0.62139 | 0.61921 | 0.64311 | 0.63977 | 0.62758 | 0.61728 | 0.60651 | 0.59626 | 0.62300 | 0.62636 | 0.62183 | 0.64145 | 0.63249 | 0.62692 | 0.63372 | 0.63940 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.63591 | 0.62944 | 0.62719 | 0.65117 | 0.64813 | 0.63512 | 0.62507 | 0.61429 | 0.60365 | 0.63091 | 0.63478 | 0.62956 | 0.64986 | 0.64046 | 0.63462 | 0.64186 | 0.64778 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.84605 | 0.84004 | 0.84566 | 0.85093 | 0.84758 | 0.85991 | 0.84415 | 0.83263 | 0.84067 | 0.85252 | 0.84454 | 0.84448 | 0.85109 | 0.86078 | 0.85006 | 0.84226 | 0.85838 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.85333 | 0.84722 | 0.85282 | 0.85813 | 0.85476 | 0.86709 | 0.85143 | 0.83957 | 0.84797 | 0.85981 | 0.85183 | 0.85189 | 0.85844 | 0.86810 | 0.85708 | 0.84946 | 0.86571 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.4955 | 2.5505 | 2.5003 | 2.8557 | 2.6089 | 2.6063 | 2.7684 | 2.3965 | 3.0791 | 2.6411 | 3.3578 | 2.4257 | 2.5577 | 2.9523 | 2.7539 | 2.4152 | 3.0702 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 4.9909 | 5.1010 | 5.0005 | 5.7115 | 5.2178 | 5.2126 | 5.5369 | 4.7930 | 6.1582 | 5.2821 | 6.7155 | 4.8513 | 5.1153 | 5.9045 | 5.5078 | 4.8305 | 6.1403 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.7300 | 1.7687 | 1.7329 | 1.9797 | 1.8090 | 1.8065 | 1.9185 | 1.6615 | 2.1341 | 1.8308 | 2.3271 | 1.6816 | 1.7727 | 2.0462 | 1.9085 | 1.6737 | 2.1279 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.4599 | 3.5373 | 3.4658 | 3.9594 | 3.6179 | 3.6130 | 3.8371 | 3.3230 | 4.2682 | 3.6616 | 4.6541 | 3.3632 | 3.5453 | 4.0923 | 3.8170 | 3.3473 | 4.2559 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.07000 | 0.07000 | 0.07000 | 0.08000 | 0.07000 | 0.07000 | 0.08000 | 0.07000 | 0.07000 | 0.05000 | 0.07000 | 0.07000 | 0.07000 | 0.07000 | 0.08000 | 0.07000 | 0.07000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.07000 | 0.07000 | 0.07000 | 0.08000 | 0.07000 | 0.07000 | 0.08000 | 0.07000 | 0.07000 | 0.05000 | 0.07000 | 0.07000 | 0.07000 | 0.07000 | 0.08000 | 0.07000 | 0.07000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -3.4481 | -3.5198 | -3.4520 | -3.9473 | -3.5993 | -3.6044 | -3.8275 | -3.3072 | -4.2536 | -3.6439 | -4.6478 | -3.3442 | -3.5295 | -4.0745 | -3.8079 | -3.3265 | -4.2426 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -3.4481 | -3.5198 | -3.4520 | -3.9473 | -3.5993 | -3.6044 | -3.8275 | -3.3072 | -4.2536 | -3.6439 | -4.6478 | -3.3442 | -3.5295 | -4.0745 | -3.8079 | -3.3265 | -4.2426 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09105 | 0.08939 | 0.09404 | 0.09348 | 0.09276 | 0.08990 | 0.08925 | 0.09160 | 0.08608 | 0.08823 | 0.08976 | 0.09211 | 0.09345 | 0.08924 | 0.09230 | 0.09146 | 0.08817 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09105 | 0.08939 | 0.09404 | 0.09348 | 0.09276 | 0.08990 | 0.08925 | 0.09160 | 0.08608 | 0.08823 | 0.08976 | 0.09211 | 0.09345 | 0.08924 | 0.09230 | 0.09146 | 0.08817 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.90639 | 0.90813 | 0.90832 | 0.91662 | 0.91551 | 0.91889 | 0.90969 | 0.90915 | 0.91165 | 0.91224 | 0.90999 | 0.91814 | 0.91855 | 0.91753 | 0.91635 | 0.91211 | 0.91870 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.91661 | 0.91818 | 0.91819 | 0.92675 | 0.92552 | 0.92900 | 0.91970 | 0.91914 | 0.92143 | 0.92232 | 0.91994 | 0.92817 | 0.92858 | 0.92747 | 0.92629 | 0.92196 | 0.92872 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.89232 | 0.89046 | 0.89426 | 0.91027 | 0.90320 | 0.90457 | 0.89307 | 0.88571 | 0.88672 | 0.90845 | 0.89210 | 0.89320 | 0.90164 | 0.90231 | 0.90153 | 0.89794 | 0.90962 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.89522 | 0.89340 | 0.89721 | 0.91325 | 0.90610 | 0.90767 | 0.89618 | 0.88845 | 0.88964 | 0.91165 | 0.89498 | 0.89617 | 0.90455 | 0.90519 | 0.90463 | 0.90088 | 0.91268 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.89340 | 0.90726 | 0.89873 | 0.91418 | 0.90295 | 0.91609 | 0.91269 | 0.90879 | 0.89932 | 0.90493 | 0.92259 | 0.89847 | 0.89916 | 0.90008 | 0.90738 | 0.90390 | 0.91860 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.94165 | 0.95559 | 0.94723 | 0.96376 | 0.95203 | 0.96568 | 0.96200 | 0.95815 | 0.94786 | 0.95382 | 0.97306 | 0.94695 | 0.94738 | 0.94874 | 0.95658 | 0.95289 | 0.96857 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0442 | 1.0344 | 1.0460 | 1.0379 | 1.0747 | 1.0417 | 1.0298 | 1.0418 | 1.0491 | 1.0493 | 1.0356 | 1.0451 | 1.0591 | 1.0775 | 1.0429 | 1.0580 | 1.0547 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0883 | 2.0689 | 2.0919 | 2.0757 | 2.1495 | 2.0834 | 2.0596 | 2.0837 | 2.0982 | 2.0985 | 2.0711 | 2.0902 | 2.1182 | 2.1551 | 2.0858 | 2.1159 | 2.1095 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.72376 | 0.71706 | 0.72500 | 0.71948 | 0.74496 | 0.72207 | 0.71386 | 0.72215 | 0.72719 | 0.72730 | 0.71783 | 0.72442 | 0.73414 | 0.74687 | 0.72291 | 0.73334 | 0.73110 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4475 | 1.4341 | 1.4500 | 1.4390 | 1.4899 | 1.4441 | 1.4277 | 1.4443 | 1.4544 | 1.4546 | 1.4357 | 1.4488 | 1.4683 | 1.4937 | 1.4458 | 1.4667 | 1.4622 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24527 | 0.25866 | 0.24910 | 0.25505 | 0.24973 | 0.24506 | 0.25760 | 0.25462 | 0.23677 | 0.24081 | 0.25420 | 0.25462 | 0.24208 | 0.24995 | 0.24867 | 0.25292 | 0.24655 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24527 | 0.25866 | 0.24910 | 0.25505 | 0.24973 | 0.24506 | 0.25760 | 0.25462 | 0.23677 | 0.24081 | 0.25420 | 0.25462 | 0.24208 | 0.24995 | 0.24867 | 0.25292 | 0.24655 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3938 | -1.3907 | -1.3982 | -1.3902 | -1.4090 | -1.3950 | -1.3906 | -1.3940 | -1.4009 | -1.3987 | -1.3954 | -1.3945 | -1.4015 | -1.4067 | -1.3955 | -1.4007 | -1.4009 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.4205 | -1.4123 | -1.4339 | -1.4058 | -1.4727 | -1.4220 | -1.4054 | -1.4226 | -1.4347 | -1.4300 | -1.4208 | -1.4242 | -1.4409 | -1.4708 | -1.4245 | -1.4451 | -1.4418 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.25062 | 0.25105 | 0.25014 | 0.25055 | 0.25007 | 0.25010 | 0.25026 | 0.25077 | 0.24875 | 0.24929 | 0.24993 | 0.25076 | 0.24928 | 0.25063 | 0.25019 | 0.25035 | 0.24969 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.25111 | 0.25210 | 0.25021 | 0.25113 | 0.25006 | 0.25010 | 0.25054 | 0.25146 | 0.24751 | 0.24855 | 0.24996 | 0.25139 | 0.24856 | 0.25112 | 0.25044 | 0.25071 | 0.24924 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 1.2654 | 1.2688 | 1.2638 | 1.2800 | 1.3045 | 1.2976 | 1.2523 | 1.2425 | 1.2629 | 1.2632 | 1.2565 | 1.2562 | 1.2806 | 1.2794 | 1.2827 | 1.2808 | 1.2846 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.4098 | 1.4125 | 1.4056 | 1.4243 | 1.4540 | 1.4437 | 1.3937 | 1.3828 | 1.4036 | 1.4060 | 1.3972 | 1.3980 | 1.4263 | 1.4240 | 1.4277 | 1.4259 | 1.4299 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0364 | 1.0328 | 1.0361 | 1.0321 | 1.0783 | 1.0340 | 1.0279 | 1.0377 | 1.0394 | 1.0370 | 1.0305 | 1.0560 | 1.0543 | 1.1000 | 1.0547 | 1.0656 | 1.0354 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0729 | 2.0655 | 2.0722 | 2.0642 | 2.1565 | 2.0679 | 2.0557 | 2.0755 | 2.0788 | 2.0739 | 2.0610 | 2.1120 | 2.1087 | 2.2000 | 2.1094 | 2.1312 | 2.0707 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.71844 | 0.71591 | 0.71815 | 0.71542 | 0.74738 | 0.71673 | 0.71250 | 0.71930 | 0.72045 | 0.71880 | 0.71437 | 0.73198 | 0.73079 | 0.76243 | 0.73108 | 0.73859 | 0.71766 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4369 | 1.4318 | 1.4363 | 1.4308 | 1.4948 | 1.4335 | 1.4250 | 1.4386 | 1.4409 | 1.4376 | 1.4287 | 1.4640 | 1.4616 | 1.5249 | 1.4622 | 1.4772 | 1.4353 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.26712 | 0.28162 | 0.27175 | 0.29365 | 0.26403 | 0.27699 | 0.26928 | 0.27360 | 0.27915 | 0.25447 | 0.24553 | 0.29180 | 0.27730 | 0.25601 | 0.25694 | 0.27391 | 0.27699 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.26712 | 0.28162 | 0.27175 | 0.29365 | 0.26403 | 0.27699 | 0.26928 | 0.27360 | 0.27915 | 0.25447 | 0.24553 | 0.29180 | 0.27730 | 0.25601 | 0.25694 | 0.27391 | 0.27699 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3875 | -1.3873 | -1.3899 | -1.3800 | -1.4027 | -1.3865 | -1.3852 | -1.3891 | -1.3901 | -1.3928 | -1.3920 | -1.3923 | -1.3939 | -1.4130 | -1.3942 | -1.3964 | -1.3881 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.4124 | -1.4045 | -1.4193 | -1.3922 | -1.4688 | -1.4062 | -1.4020 | -1.4139 | -1.4196 | -1.4183 | -1.4146 | -1.4339 | -1.4368 | -1.5012 | -1.4368 | -1.4517 | -1.4191 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25275 | 0.25192 | 0.25255 | 0.25398 | 0.25298 | 0.25246 | 0.25267 | 0.25226 | 0.25235 | 0.25084 | 0.25065 | 0.25366 | 0.25286 | 0.25121 | 0.25256 | 0.25359 | 0.25330 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25548 | 0.25407 | 0.25498 | 0.25790 | 0.25600 | 0.25485 | 0.25534 | 0.25448 | 0.25466 | 0.25169 | 0.25093 | 0.25735 | 0.25568 | 0.25254 | 0.25510 | 0.25707 | 0.25635 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 1.0530 | 1.0712 | 1.0671 | 1.0878 | 1.0850 | 1.0995 | 1.0646 | 1.0547 | 1.0632 | 1.0669 | 1.0628 | 1.0635 | 1.0810 | 1.0692 | 1.0839 | 1.0736 | 1.0908 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 1.1255 | 1.1464 | 1.1412 | 1.1640 | 1.1611 | 1.1756 | 1.1383 | 1.1294 | 1.1379 | 1.1416 | 1.1381 | 1.1377 | 1.1566 | 1.1435 | 1.1596 | 1.1481 | 1.1676 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0493 | 1.0345 | 1.0429 | 1.0447 | 1.0638 | 1.0329 | 1.0284 | 1.0387 | 1.0369 | 1.0281 | 1.0238 | 1.0615 | 1.0514 | 1.0371 | 1.0365 | 1.0495 | 1.0369 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0986 | 2.0690 | 2.0859 | 2.0894 | 2.1276 | 2.0658 | 2.0569 | 2.0774 | 2.0738 | 2.0561 | 2.0475 | 2.1230 | 2.1028 | 2.0741 | 2.0729 | 2.0991 | 2.0737 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.72731 | 0.71712 | 0.72289 | 0.72420 | 0.73738 | 0.71597 | 0.71291 | 0.72002 | 0.71870 | 0.71260 | 0.70971 | 0.73580 | 0.72879 | 0.71886 | 0.71846 | 0.72746 | 0.71868 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4546 | 1.4342 | 1.4458 | 1.4484 | 1.4748 | 1.4319 | 1.4258 | 1.4400 | 1.4374 | 1.4252 | 1.4194 | 1.4716 | 1.4576 | 1.4377 | 1.4369 | 1.4549 | 1.4374 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.23724 | 0.25414 | 0.26519 | 0.27299 | 0.27722 | 0.27624 | 0.26584 | 0.29379 | 0.28697 | 0.28469 | 0.26422 | 0.25642 | 0.28892 | 0.30842 | 0.30387 | 0.27494 | 0.28112 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.23724 | 0.25414 | 0.26519 | 0.27299 | 0.27722 | 0.27624 | 0.26584 | 0.29379 | 0.28697 | 0.28469 | 0.26422 | 0.25642 | 0.28892 | 0.30842 | 0.30387 | 0.27494 | 0.28112 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3991 | -1.3910 | -1.3921 | -1.3863 | -1.3941 | -1.3847 | -1.3852 | -1.3854 | -1.3857 | -1.3831 | -1.3893 | -1.3943 | -1.3888 | -1.3782 | -1.3815 | -1.3876 | -1.3855 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.4316 | -1.4092 | -1.4272 | -1.4073 | -1.4478 | -1.4026 | -1.4011 | -1.4161 | -1.4162 | -1.4041 | -1.4036 | -1.4432 | -1.4313 | -1.4106 | -1.4091 | -1.4292 | -1.4205 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.24928 | 0.25055 | 0.25225 | 0.25267 | 0.25419 | 0.25292 | 0.25250 | 0.25461 | 0.25431 | 0.25393 | 0.25075 | 0.25352 | 0.25484 | 0.25758 | 0.25558 | 0.25513 | 0.25498 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.24848 | 0.25103 | 0.25422 | 0.25524 | 0.25758 | 0.25567 | 0.25498 | 0.25910 | 0.25836 | 0.25779 | 0.25154 | 0.25607 | 0.25965 | 0.26559 | 0.26129 | 0.25992 | 0.25983 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.5783 | 1.6033 | 1.6099 | 1.6116 | 1.6114 | 1.6235 | 1.6026 | 1.5889 | 1.6000 | 1.5897 | 1.5956 | 1.5780 | 1.6097 | 1.5942 | 1.6311 | 1.6078 | 1.6239 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.9570 | 1.9919 | 1.9946 | 1.9998 | 2.0008 | 2.0116 | 1.9929 | 1.9796 | 1.9856 | 1.9777 | 1.9832 | 1.9531 | 1.9974 | 1.9737 | 2.0302 | 1.9982 | 2.0201 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0668 | 1.0373 | 1.0441 | 1.0428 | 1.0737 | 1.0434 | 1.0277 | 1.0343 | 1.0405 | 1.0315 | 1.0383 | 1.0616 | 1.0588 | 1.0457 | 1.0478 | 1.0507 | 1.0586 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.1336 | 2.0746 | 2.0883 | 2.0857 | 2.1475 | 2.0868 | 2.0553 | 2.0687 | 2.0810 | 2.0629 | 2.0766 | 2.1233 | 2.1177 | 2.0913 | 2.0957 | 2.1014 | 2.1172 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.73942 | 0.71903 | 0.72378 | 0.72289 | 0.74426 | 0.72326 | 0.71233 | 0.71692 | 0.72124 | 0.71503 | 0.71973 | 0.73584 | 0.73393 | 0.72483 | 0.72632 | 0.72830 | 0.73376 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4788 | 1.4381 | 1.4476 | 1.4458 | 1.4885 | 1.4465 | 1.4247 | 1.4338 | 1.4425 | 1.4301 | 1.4395 | 1.4717 | 1.4679 | 1.4497 | 1.4526 | 1.4566 | 1.4675 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24917 | 0.25447 | 0.26508 | 0.27966 | 0.26342 | 0.25547 | 0.27966 | 0.27170 | 0.28131 | 0.27767 | 0.25580 | 0.27999 | 0.26673 | 0.26673 | 0.28330 | 0.27270 | 0.26806 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24917 | 0.25447 | 0.26508 | 0.27966 | 0.26342 | 0.25547 | 0.27966 | 0.27170 | 0.28131 | 0.27767 | 0.25580 | 0.27999 | 0.26673 | 0.26673 | 0.28330 | 0.27270 | 0.26806 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.4054 | -1.3889 | -1.3912 | -1.3849 | -1.3990 | -1.3929 | -1.3805 | -1.3843 | -1.3852 | -1.3853 | -1.3944 | -1.3904 | -1.3945 | -1.3888 | -1.3846 | -1.3907 | -1.3969 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4559 | -1.4117 | -1.4313 | -1.4103 | -1.4615 | -1.4213 | -1.3960 | -1.4062 | -1.4195 | -1.4061 | -1.4240 | -1.4455 | -1.4422 | -1.4185 | -1.4202 | -1.4366 | -1.4509 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.24945 | 0.25199 | 0.25337 | 0.25406 | 0.25335 | 0.25106 | 0.25421 | 0.25361 | 0.25496 | 0.25313 | 0.25093 | 0.25590 | 0.25294 | 0.25290 | 0.25544 | 0.25442 | 0.25300 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.24908 | 0.25388 | 0.25635 | 0.25810 | 0.25605 | 0.25196 | 0.25815 | 0.25697 | 0.25968 | 0.25627 | 0.25184 | 0.26089 | 0.25557 | 0.25579 | 0.26059 | 0.25835 | 0.25557 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.55990 | 0.55630 | 0.56830 | 0.56415 | 0.55973 | 0.57456 | 0.57218 | 0.57208 | 0.56513 | 0.57674 | 0.56363 | 0.56900 | 0.57100 | 0.58256 | 0.57475 | 0.57414 | 0.58938 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.56305 | 0.55936 | 0.57123 | 0.56723 | 0.56277 | 0.57771 | 0.57524 | 0.57530 | 0.56816 | 0.57984 | 0.56679 | 0.57209 | 0.57433 | 0.58579 | 0.57800 | 0.57729 | 0.59255 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.42023 | 0.43549 | 0.43954 | 0.42094 | 0.44085 | 0.42825 | 0.43418 | 0.42883 | 0.43673 | 0.44341 | 0.43009 | 0.43974 | 0.44113 | 0.43855 | 0.45154 | 0.44061 | 0.45052 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.42180 | 0.43718 | 0.44115 | 0.42257 | 0.44254 | 0.42991 | 0.43574 | 0.43041 | 0.43827 | 0.44503 | 0.43174 | 0.44149 | 0.44282 | 0.44025 | 0.45334 | 0.44228 | 0.45214 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.85735 | 0.81739 | 0.79472 | 0.81443 | 0.81313 | 0.83190 | 0.81677 | 0.79606 | 0.83158 | 0.83041 | 0.79212 | 0.81325 | 0.84791 | 0.80724 | 0.81196 | 0.82523 | 0.82035 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.86363 | 0.82313 | 0.80009 | 0.82029 | 0.81873 | 0.83763 | 0.82250 | 0.80154 | 0.83716 | 0.83629 | 0.79745 | 0.81908 | 0.85387 | 0.81282 | 0.81762 | 0.83094 | 0.82610 |
| eval/lm/c4_en-validation/CE loss | lower | 3.3515 | 3.3526 | 3.3475 | 3.3869 | 3.3810 | 3.3862 | 3.3557 | 3.3443 | 3.3457 | 3.3657 | 3.3626 | 3.3636 | 3.3836 | 3.3786 | 3.3832 | 3.3743 | 3.3871 |
| eval/lm/c4_en-validation/PPL | lower | 28.55 | 28.58 | 28.43 | 29.57 | 29.40 | 29.55 | 28.67 | 28.34 | 28.38 | 28.95 | 28.86 | 28.89 | 29.48 | 29.33 | 29.46 | 29.20 | 29.58 |
| eval/lm/dolma_books-validation/CE loss | lower | 3.3220 | 3.3260 | 3.3270 | 3.3657 | 3.3585 | 3.3734 | 3.3347 | 3.3215 | 3.3173 | 3.3375 | 3.3397 | 3.3414 | 3.3603 | 3.3605 | 3.3614 | 3.3480 | 3.3605 |
| eval/lm/dolma_books-validation/PPL | lower | 27.71 | 27.83 | 27.85 | 28.95 | 28.75 | 29.18 | 28.07 | 27.70 | 27.59 | 28.15 | 28.21 | 28.26 | 28.80 | 28.80 | 28.83 | 28.45 | 28.80 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.4790 | 3.4809 | 3.4768 | 3.5132 | 3.5080 | 3.5117 | 3.4835 | 3.4729 | 3.4741 | 3.4930 | 3.4886 | 3.4923 | 3.5117 | 3.5037 | 3.5114 | 3.5028 | 3.5151 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 32.43 | 32.49 | 32.36 | 33.56 | 33.38 | 33.50 | 32.57 | 32.23 | 32.27 | 32.89 | 32.74 | 32.86 | 33.50 | 33.24 | 33.49 | 33.21 | 33.62 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.4975 | 2.4998 | 2.4956 | 2.5312 | 2.5230 | 2.5310 | 2.5034 | 2.4918 | 2.4929 | 2.5126 | 2.5072 | 2.5060 | 2.5292 | 2.5243 | 2.5304 | 2.5195 | 2.5318 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 12.15 | 12.18 | 12.13 | 12.57 | 12.47 | 12.57 | 12.22 | 12.08 | 12.10 | 12.34 | 12.27 | 12.26 | 12.54 | 12.48 | 12.56 | 12.42 | 12.58 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.6251 | 3.6247 | 3.6192 | 3.6568 | 3.6517 | 3.6545 | 3.6245 | 3.6144 | 3.6169 | 3.6348 | 3.6334 | 3.6312 | 3.6517 | 3.6470 | 3.6525 | 3.6421 | 3.6546 |
| eval/lm/dolma_reddit-validation/PPL | lower | 37.53 | 37.51 | 37.31 | 38.74 | 38.54 | 38.65 | 37.51 | 37.13 | 37.22 | 37.90 | 37.84 | 37.76 | 38.54 | 38.36 | 38.57 | 38.17 | 38.65 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.7457 | 1.7300 | 1.7233 | 1.7815 | 1.7597 | 1.7673 | 1.7497 | 1.7321 | 1.7252 | 1.7584 | 1.7399 | 1.7398 | 1.7742 | 1.7607 | 1.7758 | 1.7543 | 1.7805 |
| eval/lm/dolma_stack-validation/PPL | lower | 5.7299 | 5.6407 | 5.6032 | 5.9385 | 5.8109 | 5.8550 | 5.7527 | 5.6523 | 5.6134 | 5.8031 | 5.6971 | 5.6961 | 5.8958 | 5.8164 | 5.9047 | 5.7796 | 5.9328 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 3.0371 | 3.0361 | 3.0337 | 3.0712 | 3.0646 | 3.0735 | 3.0413 | 3.0308 | 3.0349 | 3.0478 | 3.0451 | 3.0463 | 3.0692 | 3.0614 | 3.0702 | 3.0614 | 3.0757 |
| eval/lm/dolma_wiki-validation/PPL | lower | 20.84 | 20.82 | 20.77 | 21.57 | 21.43 | 21.62 | 20.93 | 20.71 | 20.80 | 21.07 | 21.01 | 21.04 | 21.52 | 21.36 | 21.55 | 21.36 | 21.67 |
| eval/lm/ice-validation/CE loss | lower | 3.4249 | 3.4194 | 3.4042 | 3.4792 | 3.4625 | 3.4639 | 3.4335 | 3.4236 | 3.4100 | 3.4671 | 3.4355 | 3.4130 | 3.4781 | 3.4788 | 3.4997 | 3.4670 | 3.4970 |
| eval/lm/ice-validation/PPL | lower | 30.72 | 30.55 | 30.09 | 32.43 | 31.90 | 31.94 | 30.99 | 30.68 | 30.27 | 32.04 | 31.05 | 30.36 | 32.40 | 32.42 | 33.10 | 32.04 | 33.02 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 3.4268 | 3.4212 | 3.4218 | 3.4592 | 3.4529 | 3.4561 | 3.4297 | 3.4203 | 3.4144 | 3.4411 | 3.4360 | 3.4377 | 3.4498 | 3.4479 | 3.4539 | 3.4444 | 3.4588 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 30.78 | 30.61 | 30.62 | 31.79 | 31.59 | 31.69 | 30.87 | 30.58 | 30.40 | 31.22 | 31.06 | 31.11 | 31.50 | 31.44 | 31.62 | 31.32 | 31.78 |
| eval/lm/pile-validation/CE loss | lower | 2.6293 | 2.6274 | 2.6166 | 2.6642 | 2.6544 | 2.6588 | 2.6334 | 2.6218 | 2.6205 | 2.6451 | 2.6349 | 2.6323 | 2.6615 | 2.6554 | 2.6614 | 2.6506 | 2.6659 |
| eval/lm/pile-validation/PPL | lower | 13.86 | 13.84 | 13.69 | 14.36 | 14.22 | 14.28 | 13.92 | 13.76 | 13.74 | 14.08 | 13.94 | 13.91 | 14.32 | 14.23 | 14.32 | 14.16 | 14.38 |
| eval/lm/wikitext_103-validation/CE loss | lower | 3.0520 | 3.0397 | 3.0315 | 3.0885 | 3.0716 | 3.0827 | 3.0533 | 3.0377 | 3.0319 | 3.0751 | 3.0554 | 3.0451 | 3.0882 | 3.0831 | 3.0940 | 3.0775 | 3.0978 |
| eval/lm/wikitext_103-validation/PPL | lower | 21.16 | 20.90 | 20.73 | 21.94 | 21.58 | 21.82 | 21.18 | 20.86 | 20.74 | 21.65 | 21.23 | 21.01 | 21.94 | 21.83 | 22.07 | 21.70 | 22.15 |
| throughput/in-loop eval batches | see metric | 6554.0 | 6554.0 | 6554.0 | 1640.0 | 1640.0 | 1640.0 | 13215.0 | 13215.0 | 13215.0 | 3272.0 | 3272.0 | 3272.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 |
| throughput/in-loop eval time (s) | see metric | 661.2 | 664.1 | 627.6 | 413.7 | 442.6 | 446.9 | 653.2 | 824.5 | 852.4 | 421.9 | 437.0 | 452.7 | 398.4 | 397.4 | 396.8 | 398.6 | 406.3 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| eg-275m-cx1-eg192e16k-lr1e-3-r1<br>`psingqsk` | finished | original | 4051959808.0 | 15457 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/psingqsk) |
| eg-275m-cx1-eg192e16k-lr2e-3-r1<br>`idiscwdx` | finished | original | 4051959808.0 | 15457 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/idiscwdx) |
| eg-275m-cx1-eg192e16k-lr4e-3-r1<br>`d9oavanz` | finished | original | 4051959808.0 | 15457 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/d9oavanz) |
| eg-275m-cx1-eg24e2k-lr1e-3-r1<br>`6o5xk53j` | finished | original | 4023648256.0 | 15349 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6o5xk53j) |
| eg-275m-cx1-eg24e2k-lr2e-3-r1<br>`ndkmprhm` | finished | original | 4023648256.0 | 15349 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ndkmprhm) |
| eg-275m-cx1-eg24e2k-lr4e-3-r1<br>`8eoup1kb` | finished | original | 4023648256.0 | 15349 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8eoup1kb) |
| eg-275m-cx1-eg384e32k-lr1e-3-r1<br>`sqxqy801` | finished | original | 4084465664.0 | 15581 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sqxqy801) |
| eg-275m-cx1-eg384e32k-lr2e-3-r1<br>`sqv1l0qt` | finished | original | 4084465664.0 | 15581 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sqv1l0qt) |
| eg-275m-cx1-eg384e32k-lr4e-3-r1<br>`ltzr841g` | finished | original | 4084465664.0 | 15581 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ltzr841g) |
| eg-275m-cx1-eg96e8k-lr1e-3-r1<br>`bpdo8b6b` | finished | original | 4035969024.0 | 15396 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bpdo8b6b) |
| eg-275m-cx1-eg96e8k-lr2e-3-r1<br>`lyky04e5` | finished | original | 4035969024.0 | 15396 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/lyky04e5) |
| eg-275m-cx1-eg96e8k-lr4e-3-r1<br>`4l7axfwy` | finished | original | 4035969024.0 | 15396 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4l7axfwy) |
| 275m-cx1-b256k-lr1.2e-3-r2<br>`jxjfkaur` | finished | gpu2-ep1mb16 | 4027842560.0 | 15365 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jxjfkaur) |
| 275m-cx1-b256k-lr1.5e-3-r2<br>`a41n3bxy` | finished | gpu2-ep1mb16 | 4027842560.0 | 15365 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/a41n3bxy) |
| 275m-cx1-b256k-lr1e-3-r2<br>`x3s68iah` | finished | gpu2-ep1mb16 | 4027842560.0 | 15365 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/x3s68iah) |
| 275m-cx1-b256k-lr2e-3-r2<br>`hvr2xuvf` | finished | gpu2-ep1mb16 | 4027842560.0 | 15365 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/hvr2xuvf) |
| 275m-cx1-b256k-lr8e-4-r2<br>`mcb2thco` | finished | gpu2-ep1mb16 | 4027842560.0 | 15365 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mcb2thco) |

## 275m Cx16

| metric | direction | 275m-cx16-b1m-lr1.2e-3-r2<br>`5qg1xbny` | 275m-cx16-b1m-lr2.4e-3-r3<br>`pq9xwzgz` | 275m-cx16-b1m-lr2e-4-r2<br>`30dqfk4p` | 275m-cx16-b1m-lr4e-4-r2<br>`8rn9ixin` | 275m-cx16-b1m-lr6e-3-sentinel<br>`tmluymrq` | 275m-cx16-b1m-lr6e-4-r2<br>`aecyqi23` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.90125 | 0.92152 | 0.95312 | 0.93787 | 0.93501 | 0.92022 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 0.98631 | 1.0081 | 1.0420 | 1.0255 | 1.0245 | 1.0074 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0153 | 1.0035 | 1.0054 | 1.0085 | 1.0479 | 1.0074 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0306 | 2.0071 | 2.0109 | 2.0169 | 2.0957 | 2.0149 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.70387 | 0.69564 | 0.69707 | 0.69917 | 0.72640 | 0.69841 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4077 | 1.3913 | 1.3941 | 1.3983 | 1.4528 | 1.3968 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.24744 | 0.26706 | 0.26195 | 0.26280 | 0.23976 | 0.25000 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.24744 | 0.26706 | 0.26195 | 0.26280 | 0.23976 | 0.25000 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.3981 | -1.3849 | -1.3869 | -1.3865 | -1.4455 | -1.3876 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.3981 | -1.3849 | -1.3869 | -1.3865 | -1.4455 | -1.3876 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.24987 | 0.25284 | 0.25161 | 0.25166 | 0.24696 | 0.25227 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.24987 | 0.25284 | 0.25161 | 0.25166 | 0.24696 | 0.25227 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.70805 | 0.71487 | 0.75839 | 0.73661 | 0.73951 | 0.71429 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.76997 | 0.77803 | 0.82477 | 0.80189 | 0.80525 | 0.77647 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0215 | 1.0167 | 1.0160 | 1.0125 | 1.0406 | 1.0129 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0429 | 2.0334 | 2.0320 | 2.0249 | 2.0811 | 2.0258 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.70806 | 0.70474 | 0.70432 | 0.70190 | 0.72129 | 0.70221 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4161 | 1.4095 | 1.4086 | 1.4038 | 1.4426 | 1.4044 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.24579 | 0.24705 | 0.24874 | 0.25253 | 0.23990 | 0.25253 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.24579 | 0.24705 | 0.24874 | 0.25253 | 0.23990 | 0.25253 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.4062 | -1.4043 | -1.4005 | -1.3936 | -1.4357 | -1.3961 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.4062 | -1.4043 | -1.4005 | -1.3936 | -1.4357 | -1.3961 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.24913 | 0.25001 | 0.25008 | 0.25023 | 0.24857 | 0.25113 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.24913 | 0.25001 | 0.25008 | 0.25023 | 0.24857 | 0.25113 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 1.4923 | 1.4429 | 1.7386 | 1.5191 | 1.5699 | 1.4485 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 2.4005 | 2.3061 | 2.7994 | 2.4121 | 2.5323 | 2.3324 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 1.0343 | 1.0002 | 1.2051 | 1.0527 | 1.0881 | 1.0040 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 1.6638 | 1.5985 | 1.9403 | 1.6719 | 1.7553 | 1.6167 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.37345 | 0.38777 | 0.28080 | 0.37631 | 0.34384 | 0.40019 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.37345 | 0.38777 | 0.28080 | 0.37631 | 0.34384 | 0.40019 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -1.8803 | -1.8572 | -2.2744 | -2.1145 | -1.9684 | -1.8625 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -1.8803 | -1.8572 | -2.2744 | -2.1145 | -1.9684 | -1.8625 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.32869 | 0.34706 | 0.22718 | 0.31911 | 0.29570 | 0.33979 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.32869 | 0.34706 | 0.22718 | 0.31911 | 0.29570 | 0.33979 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.40726 | 0.41886 | 0.41186 | 0.42535 | 0.42369 | 0.42216 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.44328 | 0.45560 | 0.44783 | 0.46338 | 0.46122 | 0.46089 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.28227 | 0.29032 | 0.28547 | 0.29481 | 0.29366 | 0.29264 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.30727 | 0.31577 | 0.31039 | 0.32119 | 0.31970 | 0.31945 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.50000 | 0.50000 | 0.50000 | 0.51581 | 0.47134 | 0.49704 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.50000 | 0.50000 | 0.50000 | 0.51581 | 0.47134 | 0.49704 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -2.2535 | -2.4743 | -2.4919 | -2.3501 | -2.5596 | -2.3639 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -2.2535 | -2.4743 | -2.4919 | -2.3501 | -2.5596 | -2.3639 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.49604 | 0.48893 | 0.48866 | 0.49121 | 0.47219 | 0.47814 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.49604 | 0.48893 | 0.48866 | 0.49121 | 0.47219 | 0.47814 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.42623 | 0.38380 | 0.57964 | 0.52225 | 0.47121 | 0.47371 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.51328 | 0.46126 | 0.69463 | 0.62614 | 0.56673 | 0.56803 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.29560 | 0.26616 | 0.40199 | 0.36215 | 0.32678 | 0.32850 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.35595 | 0.31990 | 0.48177 | 0.43419 | 0.39303 | 0.39395 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.81485 | 0.81292 | 0.80906 | 0.78978 | 0.80231 | 0.81871 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.81485 | 0.81292 | 0.80906 | 0.78978 | 0.80231 | 0.81871 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.49908 | -0.47061 | -0.58779 | -0.57346 | -0.53811 | -0.52320 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.49908 | -0.47061 | -0.58779 | -0.57346 | -0.53811 | -0.52320 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.72064 | 0.72851 | 0.67865 | 0.69273 | 0.70963 | 0.70825 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.72064 | 0.72851 | 0.67865 | 0.69273 | 0.70963 | 0.70825 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.31595 | 0.31122 | 0.31172 | 0.25628 | 0.26485 | 0.25616 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.32649 | 0.32163 | 0.32213 | 0.26495 | 0.27368 | 0.26471 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.21902 | 0.21575 | 0.21607 | 0.17766 | 0.18360 | 0.17758 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.22634 | 0.22296 | 0.22331 | 0.18365 | 0.18971 | 0.18350 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.81395 | 0.87388 | 0.85063 | 0.86762 | 0.81395 | 0.86315 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.81395 | 0.87388 | 0.85063 | 0.86762 | 0.81395 | 0.86315 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.46633 | -0.33811 | -0.44203 | -0.36893 | -0.44167 | -0.34830 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.46633 | -0.33811 | -0.44203 | -0.36893 | -0.44167 | -0.34830 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.82088 | 0.85644 | 0.81744 | 0.84519 | 0.81986 | 0.84620 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.82088 | 0.85644 | 0.81744 | 0.84519 | 0.81986 | 0.84620 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 1.0768 | 1.0480 | 1.0576 | 0.97768 | 0.99197 | 0.95174 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.7504 | 1.6879 | 1.7036 | 1.5800 | 1.6084 | 1.5331 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.77779 | 0.76104 | 0.77226 | 0.71202 | 0.71638 | 0.69489 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.2939 | 1.2592 | 1.2821 | 1.1829 | 1.1885 | 1.1534 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.65918 | 0.66667 | 0.65169 | 0.64981 | 0.66292 | 0.65356 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.65918 | 0.66667 | 0.65169 | 0.64981 | 0.66292 | 0.65356 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.85349 | -0.86224 | -0.89508 | -0.85994 | -0.84213 | -0.86314 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.85349 | -0.86224 | -0.89508 | -0.85994 | -0.84213 | -0.86314 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.58003 | 0.58789 | 0.56652 | 0.59163 | 0.60222 | 0.58468 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.58003 | 0.58789 | 0.56652 | 0.59163 | 0.60222 | 0.58468 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.7151 | 1.7059 | 1.8680 | 1.8973 | 1.7455 | 1.8426 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.3705 | 2.3590 | 2.5950 | 2.6078 | 2.4012 | 2.5287 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.1888 | 1.1824 | 1.2948 | 1.3151 | 1.2099 | 1.2771 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.6430 | 1.6349 | 1.7988 | 1.8077 | 1.6644 | 1.7525 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.23052 | 0.22477 | 0.23134 | 0.21985 | 0.23790 | 0.24282 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.23052 | 0.22477 | 0.23134 | 0.21985 | 0.23790 | 0.24282 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -4.1364 | -4.0024 | -4.3080 | -4.1844 | -3.9502 | -3.9812 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -4.1364 | -4.0024 | -4.3080 | -4.1844 | -3.9502 | -3.9812 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.24366 | 0.24901 | 0.24658 | 0.24256 | 0.25434 | 0.25515 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.24366 | 0.24901 | 0.24658 | 0.24256 | 0.25434 | 0.25515 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.47124 | 0.47719 | 0.49766 | 0.47543 | 0.50813 | 0.48742 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.47732 | 0.48360 | 0.50428 | 0.48183 | 0.51530 | 0.49420 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.67262 | 0.67074 | 0.68731 | 0.67300 | 0.69605 | 0.67132 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.67839 | 0.67664 | 0.69322 | 0.67884 | 0.70209 | 0.67693 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.3698 | 2.2968 | 2.2453 | 2.1822 | 2.1412 | 2.4364 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 4.7395 | 4.5936 | 4.4906 | 4.3644 | 4.2825 | 4.8728 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.6428 | 1.5919 | 1.5562 | 1.5126 | 1.4841 | 1.6894 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.2855 | 3.1838 | 3.1124 | 3.0252 | 2.9683 | 3.3788 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.10000 | 0.09000 | 0.10000 | 0.10000 | 0.08000 | 0.09000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.10000 | 0.09000 | 0.10000 | 0.10000 | 0.08000 | 0.09000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -3.2798 | -3.1791 | -3.1012 | -3.0179 | -2.9573 | -3.3715 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -3.2798 | -3.1791 | -3.1012 | -3.0179 | -2.9573 | -3.3715 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09478 | 0.09538 | 0.09560 | 0.09690 | 0.09796 | 0.09388 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09478 | 0.09538 | 0.09560 | 0.09690 | 0.09796 | 0.09388 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.82132 | 0.83022 | 0.84136 | 0.83422 | 0.84099 | 0.82929 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.83034 | 0.83933 | 0.85066 | 0.84361 | 0.85022 | 0.83854 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.72195 | 0.72459 | 0.74717 | 0.73418 | 0.74210 | 0.73249 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.72426 | 0.72697 | 0.74953 | 0.73662 | 0.74456 | 0.73489 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.75346 | 0.76808 | 0.78423 | 0.76244 | 0.77312 | 0.75371 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.79227 | 0.80785 | 0.82557 | 0.80182 | 0.81316 | 0.79246 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0127 | 1.0116 | 1.0148 | 1.0104 | 1.0673 | 1.0121 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0254 | 2.0232 | 2.0295 | 2.0207 | 2.1346 | 2.0241 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.70201 | 0.70129 | 0.70345 | 0.70042 | 0.73981 | 0.70159 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4040 | 1.4026 | 1.4069 | 1.4008 | 1.4796 | 1.4032 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.25590 | 0.23762 | 0.25420 | 0.25037 | 0.24378 | 0.25186 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.25590 | 0.23762 | 0.25420 | 0.25037 | 0.24378 | 0.25186 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3886 | -1.3898 | -1.3893 | -1.3867 | -1.4091 | -1.3874 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.3965 | -1.3975 | -1.3983 | -1.3914 | -1.4729 | -1.3951 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.25017 | 0.24965 | 0.24998 | 0.25044 | 0.24889 | 0.25057 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.25035 | 0.24930 | 0.24994 | 0.25084 | 0.24748 | 0.25112 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 1.0387 | 1.0586 | 1.0858 | 1.0691 | 1.0741 | 1.0603 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.1546 | 1.1781 | 1.2087 | 1.1891 | 1.1947 | 1.1808 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0065 | 1.0121 | 1.0135 | 1.0086 | 1.0346 | 1.0119 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0131 | 2.0242 | 2.0270 | 2.0171 | 2.0691 | 2.0238 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.69777 | 0.70161 | 0.70260 | 0.69917 | 0.71710 | 0.70148 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.3955 | 1.4032 | 1.4052 | 1.3983 | 1.4342 | 1.4030 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.26897 | 0.23843 | 0.25231 | 0.26558 | 0.25910 | 0.24553 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.26897 | 0.23843 | 0.25231 | 0.26558 | 0.25910 | 0.24553 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3835 | -1.3890 | -1.3873 | -1.3844 | -1.3925 | -1.3867 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.3881 | -1.3980 | -1.3959 | -1.3895 | -1.4266 | -1.3937 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25163 | 0.25006 | 0.25066 | 0.25136 | 0.25191 | 0.25068 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25324 | 0.25007 | 0.25124 | 0.25265 | 0.25395 | 0.25124 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.89362 | 0.89915 | 0.93691 | 0.91460 | 0.91553 | 0.91097 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 0.95513 | 0.96067 | 1.0027 | 0.97765 | 0.97742 | 0.97338 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0148 | 1.0197 | 1.0196 | 1.0134 | 1.0134 | 1.0177 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0296 | 2.0395 | 2.0391 | 2.0268 | 2.0268 | 2.0354 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.70356 | 0.70693 | 0.70675 | 0.70256 | 0.70249 | 0.70554 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4071 | 1.4139 | 1.4135 | 1.4051 | 1.4050 | 1.4111 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.23887 | 0.22782 | 0.25219 | 0.23009 | 0.30322 | 0.23659 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.23887 | 0.22782 | 0.25219 | 0.23009 | 0.30322 | 0.23659 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3905 | -1.3941 | -1.3912 | -1.3890 | -1.3790 | -1.3912 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.3994 | -1.4088 | -1.4035 | -1.3964 | -1.3970 | -1.4029 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.24952 | 0.24892 | 0.24971 | 0.24989 | 0.25522 | 0.24964 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.24904 | 0.24785 | 0.24949 | 0.24978 | 0.26096 | 0.24926 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.3548 | 1.3745 | 1.4054 | 1.3784 | 1.3892 | 1.3634 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.6926 | 1.7184 | 1.7558 | 1.7225 | 1.7278 | 1.6992 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0123 | 1.0194 | 1.0207 | 1.0112 | 1.0228 | 1.0187 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0246 | 2.0389 | 2.0414 | 2.0223 | 2.0457 | 2.0375 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.70180 | 0.70675 | 0.70757 | 0.70100 | 0.70900 | 0.70627 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4036 | 1.4135 | 1.4151 | 1.4020 | 1.4180 | 1.4125 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.25182 | 0.24785 | 0.25712 | 0.25580 | 0.28960 | 0.24983 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.25182 | 0.24785 | 0.25712 | 0.25580 | 0.28960 | 0.24983 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3883 | -1.3931 | -1.3913 | -1.3865 | -1.3844 | -1.3903 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.3966 | -1.4081 | -1.4067 | -1.3931 | -1.4097 | -1.4030 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.25036 | 0.24935 | 0.25012 | 0.25075 | 0.25381 | 0.25008 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.25077 | 0.24879 | 0.25034 | 0.25143 | 0.25773 | 0.25014 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.45071 | 0.44188 | 0.44734 | 0.45148 | 0.46327 | 0.44447 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.45321 | 0.44440 | 0.44976 | 0.45404 | 0.46588 | 0.44693 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.34866 | 0.35076 | 0.34517 | 0.34307 | 0.35813 | 0.34788 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.34998 | 0.35215 | 0.34643 | 0.34442 | 0.35947 | 0.34927 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.60959 | 0.60227 | 0.62308 | 0.60759 | 0.64140 | 0.59002 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.61385 | 0.60668 | 0.62748 | 0.61203 | 0.64602 | 0.59438 |
| eval/lm/c4_en-validation/CE loss | lower | 3.0458 | 3.0539 | 3.0992 | 3.0681 | 3.0935 | 3.0552 |
| eval/lm/c4_en-validation/PPL | lower | 21.03 | 21.20 | 22.18 | 21.50 | 22.05 | 21.23 |
| eval/lm/dolma_books-validation/CE loss | lower | 2.9484 | 2.9577 | 3.0123 | 2.9751 | 3.0150 | 2.9574 |
| eval/lm/dolma_books-validation/PPL | lower | 19.08 | 19.25 | 20.33 | 19.59 | 20.39 | 19.25 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.1808 | 3.1887 | 3.2356 | 3.2037 | 3.2293 | 3.1900 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 24.07 | 24.26 | 25.42 | 24.62 | 25.26 | 24.29 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.2252 | 2.2274 | 2.2704 | 2.2428 | 2.2642 | 2.2341 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 9.2555 | 9.2760 | 9.6830 | 9.4194 | 9.6235 | 9.3383 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.3482 | 3.3557 | 3.3996 | 3.3712 | 3.3927 | 3.3585 |
| eval/lm/dolma_reddit-validation/PPL | lower | 28.45 | 28.67 | 29.95 | 29.11 | 29.75 | 28.75 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.3814 | 1.3883 | 1.4246 | 1.3953 | 1.4281 | 1.3854 |
| eval/lm/dolma_stack-validation/PPL | lower | 3.9805 | 4.0080 | 4.1563 | 4.0361 | 4.1708 | 3.9964 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.6925 | 2.7049 | 2.7442 | 2.7138 | 2.7518 | 2.7000 |
| eval/lm/dolma_wiki-validation/PPL | lower | 14.77 | 14.95 | 15.55 | 15.09 | 15.67 | 14.88 |
| eval/lm/ice-validation/CE loss | lower | 3.1394 | 3.1329 | 3.1683 | 3.1560 | 3.1671 | 3.1403 |
| eval/lm/ice-validation/PPL | lower | 23.09 | 22.94 | 23.77 | 23.48 | 23.74 | 23.11 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 3.1593 | 3.1641 | 3.1953 | 3.1702 | 3.1925 | 3.1551 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 23.55 | 23.67 | 24.42 | 23.81 | 24.35 | 23.46 |
| eval/lm/pile-validation/CE loss | lower | 2.3053 | 2.3117 | 2.3550 | 2.3263 | 2.3532 | 2.3158 |
| eval/lm/pile-validation/PPL | lower | 10.03 | 10.09 | 10.54 | 10.24 | 10.52 | 10.13 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.6603 | 2.6629 | 2.7310 | 2.6879 | 2.7084 | 2.6710 |
| eval/lm/wikitext_103-validation/PPL | lower | 14.30 | 14.34 | 15.35 | 14.70 | 15.01 | 14.45 |
| throughput/in-loop eval batches | see metric | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 |
| throughput/in-loop eval time (s) | see metric | 397.3 | 400.8 | 398.3 | 421.6 | 423.2 | 411.8 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| 275m-cx16-b1m-lr1.2e-3-r2<br>`5qg1xbny` | finished | gpu8-ep1mb16 | 64442335232.0 | 61457 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5qg1xbny) |
| 275m-cx16-b1m-lr2.4e-3-r3<br>`pq9xwzgz` | finished | gpu8-ep1mb16 | 64442335232.0 | 61457 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/pq9xwzgz) |
| 275m-cx16-b1m-lr2e-4-r2<br>`30dqfk4p` | finished | gpu8-ep1mb16 | 64442335232.0 | 61457 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/30dqfk4p) |
| 275m-cx16-b1m-lr4e-4-r2<br>`8rn9ixin` | finished | gpu8-ep1mb16 | 64442335232.0 | 61457 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8rn9ixin) |
| 275m-cx16-b1m-lr6e-3-sentinel<br>`tmluymrq` | finished | gpu8-ep1mb16 | 64442335232.0 | 61457 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/tmluymrq) |
| 275m-cx16-b1m-lr6e-4-r2<br>`aecyqi23` | finished | gpu8-ep1mb16 | 64442335232.0 | 61457 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/aecyqi23) |

## 275m Cx2

| metric | direction | eg-275m-cx2-b384k-eg24e2k-lr1.8e-3-r3<br>`ujt9b8kw` | eg-275m-cx2-b384k-eg24e2k-lr3.6e-3-r3<br>`ijiw4id5` | eg-275m-cx2-b384k-eg24e2k-lr9e-4-r3<br>`spmc55m4` | eg-275m-cx2-b384k-eg96e8k-lr1.8e-3-r3<br>`38gyc8af` | eg-275m-cx2-b384k-eg96e8k-lr3.6e-3-r3<br>`53yd3xre` | eg-275m-cx2-b384k-eg96e8k-lr9e-4-r3<br>`tcvldbk1` | eg-275m-cx2-eg24e2k-lr1e-3-r1<br>`wm2wrmnc` | eg-275m-cx2-eg24e2k-lr2e-3-r1<br>`50mfn51v` | eg-275m-cx2-eg24e2k-lr5e-4-r1<br>`8n4j64g9` | eg-275m-cx2-eg96e8k-lr1e-3-r1<br>`s1t6b084` | eg-275m-cx2-eg96e8k-lr2e-3-r1<br>`xys9u31e` | eg-275m-cx2-eg96e8k-lr5e-4-r1<br>`728o3v27` | 275m-cx2-b256k-lr1e-3<br>`dgej30hb` | 275m-cx2-b256k-lr6e-4-r2<br>`8382k902` | 275m-cx2-b256k-lr8e-4-r2<br>`56e1coi7` | 275m-cx2-b384k-lr1.8e-3-r3<br>`lq4zvsx4` | 275m-cx2-b384k-lr3.6e-3-r3<br>`pv6y1aqx` | 275m-cx2-b384k-lr9e-4-r3<br>`atxrokcu` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 1.0374 | 1.0223 | 1.0451 | 1.0178 | 1.0163 | 1.0180 | 1.0407 | 1.0272 | 1.0397 | 1.0189 | 1.0175 | 1.0309 | 1.0253 | 1.0354 | 1.0159 | 1.0168 | 1.0299 | 1.0335 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 1.1360 | 1.1188 | 1.1451 | 1.1151 | 1.1152 | 1.1142 | 1.1407 | 1.1258 | 1.1395 | 1.1163 | 1.1151 | 1.1282 | 1.1225 | 1.1353 | 1.1129 | 1.1116 | 1.1270 | 1.1305 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0312 | 1.0472 | 1.0293 | 1.0204 | 1.0315 | 1.0197 | 1.0277 | 1.0310 | 1.0248 | 1.0361 | 1.0574 | 1.0327 | 1.0231 | 1.0507 | 1.0338 | 1.0258 | 1.0262 | 1.0286 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0624 | 2.0944 | 2.0585 | 2.0409 | 2.0630 | 2.0394 | 2.0554 | 2.0621 | 2.0496 | 2.0723 | 2.1148 | 2.0654 | 2.0461 | 2.1015 | 2.0676 | 2.0515 | 2.0524 | 2.0573 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.71486 | 0.72599 | 0.71356 | 0.70737 | 0.71508 | 0.70687 | 0.71245 | 0.71475 | 0.71038 | 0.71823 | 0.73299 | 0.71587 | 0.70920 | 0.72827 | 0.71669 | 0.71107 | 0.71138 | 0.71296 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4297 | 1.4520 | 1.4271 | 1.4147 | 1.4302 | 1.4137 | 1.4249 | 1.4295 | 1.4208 | 1.4365 | 1.4660 | 1.4317 | 1.4184 | 1.4565 | 1.4334 | 1.4221 | 1.4228 | 1.4259 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.26792 | 0.22952 | 0.25085 | 0.25256 | 0.22270 | 0.24061 | 0.27560 | 0.24147 | 0.23720 | 0.22782 | 0.23379 | 0.26621 | 0.24488 | 0.23123 | 0.22184 | 0.23720 | 0.23379 | 0.24061 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.26792 | 0.22952 | 0.25085 | 0.25256 | 0.22270 | 0.24061 | 0.27560 | 0.24147 | 0.23720 | 0.22782 | 0.23379 | 0.26621 | 0.24488 | 0.23123 | 0.22184 | 0.23720 | 0.23379 | 0.24061 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.4005 | -1.4332 | -1.4063 | -1.3936 | -1.4113 | -1.4005 | -1.4005 | -1.4114 | -1.4023 | -1.4250 | -1.4380 | -1.4039 | -1.3998 | -1.4298 | -1.4096 | -1.4055 | -1.4102 | -1.4095 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.4005 | -1.4332 | -1.4063 | -1.3936 | -1.4113 | -1.4005 | -1.4005 | -1.4114 | -1.4023 | -1.4250 | -1.4380 | -1.4039 | -1.3998 | -1.4298 | -1.4096 | -1.4055 | -1.4102 | -1.4095 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.25233 | 0.24665 | 0.24975 | 0.25131 | 0.24862 | 0.25081 | 0.25150 | 0.24928 | 0.25112 | 0.24619 | 0.24614 | 0.25261 | 0.25032 | 0.24996 | 0.24776 | 0.25001 | 0.24967 | 0.25012 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.25233 | 0.24665 | 0.24975 | 0.25131 | 0.24862 | 0.25081 | 0.25150 | 0.24928 | 0.25112 | 0.24619 | 0.24614 | 0.25261 | 0.25032 | 0.24996 | 0.24776 | 0.25001 | 0.24967 | 0.25012 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.86157 | 0.85809 | 0.86359 | 0.82688 | 0.81427 | 0.82585 | 0.84996 | 0.84089 | 0.87867 | 0.84176 | 0.82955 | 0.85025 | 0.85111 | 0.84519 | 0.83591 | 0.83334 | 0.84163 | 0.85247 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.93871 | 0.93636 | 0.94218 | 0.90136 | 0.88762 | 0.90027 | 0.92655 | 0.91650 | 0.95864 | 0.91775 | 0.90336 | 0.92685 | 0.92830 | 0.92249 | 0.91115 | 0.90800 | 0.91779 | 0.92919 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0437 | 1.0580 | 1.0298 | 1.0371 | 1.0226 | 1.0216 | 1.0351 | 1.0382 | 1.0294 | 1.0236 | 1.0561 | 1.0374 | 1.0354 | 1.0527 | 1.0402 | 1.0281 | 1.0393 | 1.0443 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0874 | 2.1160 | 2.0597 | 2.0741 | 2.0452 | 2.0433 | 2.0702 | 2.0764 | 2.0589 | 2.0472 | 2.1122 | 2.0748 | 2.0708 | 2.1054 | 2.0805 | 2.0562 | 2.0786 | 2.0886 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.72344 | 0.73337 | 0.71391 | 0.71890 | 0.70888 | 0.70820 | 0.71757 | 0.71965 | 0.71360 | 0.70958 | 0.73206 | 0.71912 | 0.71767 | 0.72966 | 0.72111 | 0.71275 | 0.72047 | 0.72390 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4469 | 1.4667 | 1.4278 | 1.4378 | 1.4178 | 1.4164 | 1.4351 | 1.4393 | 1.4272 | 1.4192 | 1.4641 | 1.4382 | 1.4353 | 1.4593 | 1.4422 | 1.4255 | 1.4409 | 1.4478 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.24579 | 0.24116 | 0.24832 | 0.23401 | 0.26473 | 0.26136 | 0.24369 | 0.24495 | 0.25042 | 0.24916 | 0.23948 | 0.23990 | 0.26347 | 0.24958 | 0.24958 | 0.23906 | 0.25084 | 0.23822 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.24579 | 0.24116 | 0.24832 | 0.23401 | 0.26473 | 0.26136 | 0.24369 | 0.24495 | 0.25042 | 0.24916 | 0.23948 | 0.23990 | 0.26347 | 0.24958 | 0.24958 | 0.23906 | 0.25084 | 0.23822 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.4174 | -1.4504 | -1.4089 | -1.4102 | -1.3981 | -1.4028 | -1.4086 | -1.4234 | -1.4069 | -1.4090 | -1.4370 | -1.4034 | -1.4159 | -1.4238 | -1.4139 | -1.4087 | -1.4261 | -1.4300 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.4174 | -1.4504 | -1.4089 | -1.4102 | -1.3981 | -1.4028 | -1.4086 | -1.4234 | -1.4069 | -1.4090 | -1.4370 | -1.4034 | -1.4159 | -1.4238 | -1.4139 | -1.4087 | -1.4261 | -1.4300 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.24939 | 0.24837 | 0.24954 | 0.24863 | 0.25127 | 0.25137 | 0.24948 | 0.24809 | 0.25178 | 0.25057 | 0.24759 | 0.25070 | 0.25159 | 0.24871 | 0.24852 | 0.25012 | 0.25012 | 0.24737 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.24939 | 0.24837 | 0.24954 | 0.24863 | 0.25127 | 0.25137 | 0.24948 | 0.24809 | 0.25178 | 0.25057 | 0.24759 | 0.25070 | 0.25159 | 0.24871 | 0.24852 | 0.25012 | 0.25012 | 0.24737 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 2.2630 | 2.0490 | 2.2415 | 2.0919 | 2.0823 | 2.1905 | 2.1642 | 2.1442 | 2.2355 | 2.1532 | 2.1818 | 2.2554 | 2.1368 | 2.2560 | 2.1584 | 2.2867 | 2.2108 | 2.1386 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 3.6097 | 3.2718 | 3.5997 | 3.3268 | 3.3393 | 3.4908 | 3.4647 | 3.4120 | 3.5817 | 3.4138 | 3.4909 | 3.5701 | 3.4321 | 3.5886 | 3.4545 | 3.6459 | 3.5313 | 3.3938 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 1.5686 | 1.4201 | 1.5537 | 1.4499 | 1.4433 | 1.5185 | 1.5001 | 1.4862 | 1.5495 | 1.4923 | 1.5122 | 1.5632 | 1.4812 | 1.5637 | 1.4959 | 1.5851 | 1.5323 | 1.4825 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 2.5017 | 2.2677 | 2.4951 | 2.3060 | 2.3146 | 2.4200 | 2.4016 | 2.3649 | 2.4827 | 2.3660 | 2.4199 | 2.4745 | 2.3790 | 2.4875 | 2.3944 | 2.5275 | 2.4476 | 2.3523 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.13945 | 0.14900 | 0.10697 | 0.11748 | 0.14613 | 0.13945 | 0.10793 | 0.11175 | 0.10984 | 0.12798 | 0.11557 | 0.11366 | 0.10984 | 0.13276 | 0.12321 | 0.10697 | 0.13658 | 0.11461 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.13945 | 0.14900 | 0.10697 | 0.11748 | 0.14613 | 0.13945 | 0.10793 | 0.11175 | 0.10984 | 0.12798 | 0.11557 | 0.11366 | 0.10984 | 0.13276 | 0.12321 | 0.10697 | 0.13658 | 0.11461 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -2.7722 | -2.5754 | -2.6702 | -2.5654 | -2.5554 | -2.6730 | -2.6299 | -2.6924 | -2.6873 | -2.6537 | -2.7214 | -2.8044 | -2.5759 | -2.8207 | -2.7172 | -2.7902 | -2.7293 | -2.6544 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -2.7722 | -2.5754 | -2.6702 | -2.5654 | -2.5554 | -2.6730 | -2.6299 | -2.6924 | -2.6873 | -2.6537 | -2.7214 | -2.8044 | -2.5759 | -2.8207 | -2.7172 | -2.7902 | -2.7293 | -2.6544 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.12130 | 0.12153 | 0.10852 | 0.11843 | 0.12201 | 0.11765 | 0.10861 | 0.10933 | 0.10302 | 0.11886 | 0.11732 | 0.10710 | 0.11207 | 0.11210 | 0.11584 | 0.10233 | 0.11131 | 0.10993 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.12130 | 0.12153 | 0.10852 | 0.11843 | 0.12201 | 0.11765 | 0.10861 | 0.10933 | 0.10302 | 0.11886 | 0.11732 | 0.10710 | 0.11207 | 0.11210 | 0.11584 | 0.10233 | 0.11131 | 0.10993 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.55237 | 0.56571 | 0.55250 | 0.54249 | 0.57049 | 0.56646 | 0.53659 | 0.53124 | 0.58109 | 0.54313 | 0.50978 | 0.54244 | 0.58394 | 0.56452 | 0.59064 | 0.58307 | 0.56622 | 0.54541 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.60048 | 0.61841 | 0.60164 | 0.59219 | 0.62362 | 0.61757 | 0.58427 | 0.57918 | 0.63346 | 0.59247 | 0.55541 | 0.59064 | 0.63492 | 0.61402 | 0.64195 | 0.63548 | 0.61736 | 0.59395 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.38287 | 0.39212 | 0.38299 | 0.37606 | 0.39543 | 0.39262 | 0.37195 | 0.36824 | 0.40279 | 0.37646 | 0.35336 | 0.37595 | 0.40470 | 0.39129 | 0.40938 | 0.40419 | 0.39247 | 0.37804 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.41620 | 0.42861 | 0.41702 | 0.41045 | 0.43225 | 0.42809 | 0.40497 | 0.40146 | 0.43912 | 0.41065 | 0.38497 | 0.40940 | 0.44007 | 0.42562 | 0.44493 | 0.44044 | 0.42789 | 0.41170 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.42095 | 0.42885 | 0.42391 | 0.44368 | 0.43577 | 0.43676 | 0.41601 | 0.42391 | 0.40020 | 0.41206 | 0.45356 | 0.42391 | 0.41502 | 0.41304 | 0.40217 | 0.42490 | 0.44763 | 0.42194 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.42095 | 0.42885 | 0.42391 | 0.44368 | 0.43577 | 0.43676 | 0.41601 | 0.42391 | 0.40020 | 0.41206 | 0.45356 | 0.42391 | 0.41502 | 0.41304 | 0.40217 | 0.42490 | 0.44763 | 0.42194 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -3.6927 | -3.3910 | -3.5736 | -3.3716 | -3.2383 | -3.3493 | -3.6464 | -3.6096 | -3.8472 | -3.4869 | -3.3158 | -3.4744 | -3.7551 | -3.7071 | -3.9330 | -3.5776 | -3.2867 | -3.5452 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -3.6927 | -3.3910 | -3.5736 | -3.3716 | -3.2383 | -3.3493 | -3.6464 | -3.6096 | -3.8472 | -3.4869 | -3.3158 | -3.4744 | -3.7551 | -3.7071 | -3.9330 | -3.5776 | -3.2867 | -3.5452 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.40564 | 0.40918 | 0.40823 | 0.41938 | 0.41334 | 0.41532 | 0.39550 | 0.40912 | 0.38968 | 0.40156 | 0.42357 | 0.40807 | 0.40040 | 0.39692 | 0.38561 | 0.40125 | 0.42709 | 0.40937 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.40564 | 0.40918 | 0.40823 | 0.41938 | 0.41334 | 0.41532 | 0.39550 | 0.40912 | 0.38968 | 0.40156 | 0.42357 | 0.40807 | 0.40040 | 0.39692 | 0.38561 | 0.40125 | 0.42709 | 0.40937 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.74691 | 0.76502 | 0.75512 | 0.61679 | 0.67848 | 0.71879 | 0.74452 | 0.80393 | 0.73641 | 0.83313 | 0.66618 | 0.77493 | 0.66323 | 0.74730 | 0.70791 | 0.66861 | 0.64351 | 0.71948 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.89563 | 0.91691 | 0.90439 | 0.73991 | 0.81383 | 0.86319 | 0.89180 | 0.96280 | 0.88275 | 1.0019 | 0.80023 | 0.93157 | 0.79543 | 0.89637 | 0.84893 | 0.80065 | 0.77486 | 0.86310 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.51799 | 0.53048 | 0.52362 | 0.42777 | 0.47054 | 0.49848 | 0.51631 | 0.55736 | 0.51074 | 0.57775 | 0.46192 | 0.53730 | 0.45994 | 0.51814 | 0.49089 | 0.46368 | 0.44626 | 0.49886 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.62115 | 0.63583 | 0.62716 | 0.51321 | 0.56441 | 0.59864 | 0.61850 | 0.66764 | 0.61219 | 0.69481 | 0.55498 | 0.64593 | 0.55168 | 0.62164 | 0.58870 | 0.55525 | 0.53745 | 0.59841 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.70974 | 0.72324 | 0.69913 | 0.77435 | 0.73095 | 0.70974 | 0.69238 | 0.69817 | 0.67792 | 0.71360 | 0.76663 | 0.67599 | 0.73963 | 0.72806 | 0.71745 | 0.70781 | 0.73288 | 0.70685 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.70974 | 0.72324 | 0.69913 | 0.77435 | 0.73095 | 0.70974 | 0.69238 | 0.69817 | 0.67792 | 0.71360 | 0.76663 | 0.67599 | 0.73963 | 0.72806 | 0.71745 | 0.70781 | 0.73288 | 0.70685 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.75845 | -0.76783 | -0.79182 | -0.65453 | -0.71730 | -0.75461 | -0.78546 | -0.79552 | -0.85840 | -0.81146 | -0.69495 | -0.81335 | -0.73271 | -0.77176 | -0.79737 | -0.77527 | -0.73014 | -0.79655 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.75845 | -0.76783 | -0.79182 | -0.65453 | -0.71730 | -0.75461 | -0.78546 | -0.79552 | -0.85840 | -0.81146 | -0.69495 | -0.81335 | -0.73271 | -0.77176 | -0.79737 | -0.77527 | -0.73014 | -0.79655 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.61470 | 0.60119 | 0.60088 | 0.64807 | 0.63106 | 0.60383 | 0.59475 | 0.60844 | 0.57511 | 0.58942 | 0.64221 | 0.57620 | 0.63313 | 0.62085 | 0.60472 | 0.61793 | 0.62990 | 0.61409 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.61470 | 0.60119 | 0.60088 | 0.64807 | 0.63106 | 0.60383 | 0.59475 | 0.60844 | 0.57511 | 0.58942 | 0.64221 | 0.57620 | 0.63313 | 0.62085 | 0.60472 | 0.61793 | 0.62990 | 0.61409 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.29771 | 0.27430 | 0.29345 | 0.29960 | 0.30923 | 0.28644 | 0.28949 | 0.27840 | 0.30604 | 0.28655 | 0.28469 | 0.30213 | 0.31676 | 0.33298 | 0.33593 | 0.28765 | 0.29819 | 0.30020 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.30770 | 0.28346 | 0.30330 | 0.30973 | 0.31965 | 0.29604 | 0.29929 | 0.28781 | 0.31630 | 0.29618 | 0.29428 | 0.31231 | 0.32739 | 0.34429 | 0.34726 | 0.29746 | 0.30822 | 0.31036 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.20637 | 0.19012 | 0.20344 | 0.20769 | 0.21437 | 0.19855 | 0.20067 | 0.19298 | 0.21215 | 0.19864 | 0.19734 | 0.20943 | 0.21956 | 0.23083 | 0.23286 | 0.19942 | 0.20671 | 0.20809 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.21331 | 0.19650 | 0.21024 | 0.21471 | 0.22159 | 0.20522 | 0.20749 | 0.19951 | 0.21926 | 0.20532 | 0.20399 | 0.21650 | 0.22697 | 0.23866 | 0.24072 | 0.20619 | 0.21364 | 0.21515 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.83542 | 0.79875 | 0.83005 | 0.78801 | 0.80143 | 0.78354 | 0.79964 | 0.81485 | 0.79696 | 0.77996 | 0.79517 | 0.79338 | 0.78712 | 0.79785 | 0.78801 | 0.79338 | 0.80143 | 0.79517 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.83542 | 0.79875 | 0.83005 | 0.78801 | 0.80143 | 0.78354 | 0.79964 | 0.81485 | 0.79696 | 0.77996 | 0.79517 | 0.79338 | 0.78712 | 0.79785 | 0.78801 | 0.79338 | 0.80143 | 0.79517 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.49129 | -0.57973 | -0.48636 | -0.60675 | -0.55548 | -0.60992 | -0.56838 | -0.53233 | -0.58154 | -0.59299 | -0.54101 | -0.54624 | -0.60881 | -0.57837 | -0.61657 | -0.58408 | -0.53297 | -0.53551 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.49129 | -0.57973 | -0.48636 | -0.60675 | -0.55548 | -0.60992 | -0.56838 | -0.53233 | -0.58154 | -0.59299 | -0.54101 | -0.54624 | -0.60881 | -0.57837 | -0.61657 | -0.58408 | -0.53297 | -0.53551 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.80703 | 0.79674 | 0.80407 | 0.78223 | 0.79239 | 0.78758 | 0.78217 | 0.80570 | 0.77485 | 0.77558 | 0.80077 | 0.78933 | 0.77698 | 0.77840 | 0.77467 | 0.78734 | 0.79660 | 0.78915 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.80703 | 0.79674 | 0.80407 | 0.78223 | 0.79239 | 0.78758 | 0.78217 | 0.80570 | 0.77485 | 0.77558 | 0.80077 | 0.78933 | 0.77698 | 0.77840 | 0.77467 | 0.78734 | 0.79660 | 0.78915 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 1.1896 | 1.2159 | 1.2975 | 1.2302 | 1.1766 | 1.2020 | 1.2571 | 1.2301 | 1.2487 | 1.2742 | 1.2602 | 1.2381 | 1.2130 | 1.2989 | 1.2964 | 1.2858 | 1.2366 | 1.1815 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.8969 | 1.9441 | 2.0284 | 1.9688 | 1.8703 | 1.9029 | 1.9889 | 1.9661 | 1.9708 | 2.0303 | 2.0131 | 1.9563 | 1.9270 | 2.0778 | 2.0718 | 2.0466 | 1.9745 | 1.8684 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.86356 | 0.88775 | 0.94056 | 0.89064 | 0.85243 | 0.87571 | 0.90843 | 0.89335 | 0.90562 | 0.91709 | 0.91000 | 0.89277 | 0.88539 | 0.94623 | 0.94697 | 0.93570 | 0.90244 | 0.86389 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.4141 | 1.4625 | 1.5118 | 1.4610 | 1.3898 | 1.4269 | 1.4736 | 1.4673 | 1.4689 | 1.4933 | 1.4883 | 1.4449 | 1.4494 | 1.5579 | 1.5585 | 1.5319 | 1.4846 | 1.4095 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.59363 | 0.58427 | 0.57116 | 0.60487 | 0.59176 | 0.60112 | 0.56554 | 0.58052 | 0.58989 | 0.56180 | 0.57678 | 0.59551 | 0.58989 | 0.55618 | 0.58614 | 0.55993 | 0.56367 | 0.59738 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.59363 | 0.58427 | 0.57116 | 0.60487 | 0.59176 | 0.60112 | 0.56554 | 0.58052 | 0.58989 | 0.56180 | 0.57678 | 0.59551 | 0.58989 | 0.55618 | 0.58614 | 0.55993 | 0.56367 | 0.59738 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -1.1035 | -1.1380 | -1.1418 | -1.0754 | -1.0489 | -1.0407 | -1.1331 | -1.0950 | -1.1134 | -1.1067 | -1.1377 | -1.0603 | -1.0668 | -1.0935 | -1.0682 | -1.1146 | -1.1155 | -1.0850 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -1.1035 | -1.1380 | -1.1418 | -1.0754 | -1.0489 | -1.0407 | -1.1331 | -1.0950 | -1.1134 | -1.1067 | -1.1377 | -1.0603 | -1.0668 | -1.0935 | -1.0682 | -1.1146 | -1.1155 | -1.0850 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.50480 | 0.49624 | 0.49414 | 0.52157 | 0.52653 | 0.52370 | 0.49579 | 0.50622 | 0.50511 | 0.50243 | 0.49468 | 0.50986 | 0.51926 | 0.49834 | 0.50810 | 0.49285 | 0.50172 | 0.50558 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.50480 | 0.49624 | 0.49414 | 0.52157 | 0.52653 | 0.52370 | 0.49579 | 0.50622 | 0.50511 | 0.50243 | 0.49468 | 0.50986 | 0.51926 | 0.49834 | 0.50810 | 0.49285 | 0.50172 | 0.50558 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 2.1720 | 2.0801 | 2.2813 | 2.1788 | 2.1389 | 2.2157 | 2.3045 | 2.2608 | 2.3057 | 2.2393 | 2.2785 | 2.1689 | 2.2347 | 2.2207 | 2.2726 | 2.1920 | 2.2058 | 2.1789 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.9977 | 2.8455 | 3.1133 | 2.9803 | 2.9464 | 3.0333 | 3.1953 | 3.1266 | 3.1574 | 3.0856 | 3.1147 | 2.9887 | 3.0566 | 3.0602 | 3.1237 | 3.0256 | 3.0566 | 2.9895 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.5054 | 1.4418 | 1.5813 | 1.5101 | 1.4823 | 1.5359 | 1.5973 | 1.5671 | 1.5982 | 1.5520 | 1.5793 | 1.5035 | 1.5489 | 1.5392 | 1.5753 | 1.5194 | 1.5290 | 1.5103 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 2.0779 | 1.9723 | 2.1578 | 2.0658 | 2.0420 | 2.1029 | 2.2147 | 2.1670 | 2.1882 | 2.1386 | 2.1592 | 2.0717 | 2.1188 | 2.1210 | 2.1652 | 2.0972 | 2.1189 | 2.0722 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.21657 | 0.21575 | 0.21493 | 0.22149 | 0.20673 | 0.21575 | 0.23462 | 0.20919 | 0.23544 | 0.23626 | 0.22559 | 0.22477 | 0.22067 | 0.21739 | 0.21903 | 0.22313 | 0.21165 | 0.23462 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.21657 | 0.21575 | 0.21493 | 0.22149 | 0.20673 | 0.21575 | 0.23462 | 0.20919 | 0.23544 | 0.23626 | 0.22559 | 0.22477 | 0.22067 | 0.21739 | 0.21903 | 0.22313 | 0.21165 | 0.23462 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -4.7560 | -4.8043 | -4.7122 | -4.6745 | -4.7450 | -4.8280 | -4.7851 | -4.6238 | -4.8548 | -4.5089 | -4.7338 | -4.7417 | -5.0009 | -4.9006 | -4.7443 | -4.8400 | -5.0113 | -4.8527 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -4.7560 | -4.8043 | -4.7122 | -4.6745 | -4.7450 | -4.8280 | -4.7851 | -4.6238 | -4.8548 | -4.5089 | -4.7338 | -4.7417 | -5.0009 | -4.9006 | -4.7443 | -4.8400 | -5.0113 | -4.8527 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.23381 | 0.22905 | 0.22959 | 0.23964 | 0.22763 | 0.22623 | 0.23579 | 0.23230 | 0.23834 | 0.24316 | 0.23906 | 0.23699 | 0.22712 | 0.22776 | 0.23068 | 0.23263 | 0.22641 | 0.23559 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.23381 | 0.22905 | 0.22959 | 0.23964 | 0.22763 | 0.22623 | 0.23579 | 0.23230 | 0.23834 | 0.24316 | 0.23906 | 0.23699 | 0.22712 | 0.22776 | 0.23068 | 0.23263 | 0.22641 | 0.23559 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.59625 | 0.57237 | 0.58130 | 0.56444 | 0.56070 | 0.57704 | 0.58011 | 0.57252 | 0.58409 | 0.57206 | 0.57570 | 0.58881 | 0.59580 | 0.58388 | 0.57271 | 0.56258 | 0.57843 | 0.57773 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.60449 | 0.57992 | 0.58936 | 0.57197 | 0.56791 | 0.58452 | 0.58804 | 0.58021 | 0.59174 | 0.57966 | 0.58325 | 0.59669 | 0.60355 | 0.59167 | 0.58027 | 0.56992 | 0.58640 | 0.58531 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.78700 | 0.78836 | 0.78753 | 0.77512 | 0.76161 | 0.77824 | 0.77925 | 0.78553 | 0.78506 | 0.77261 | 0.76825 | 0.78207 | 0.78770 | 0.79648 | 0.78886 | 0.77533 | 0.77884 | 0.78105 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.79388 | 0.79511 | 0.79442 | 0.78179 | 0.76800 | 0.78500 | 0.78598 | 0.79204 | 0.79174 | 0.77932 | 0.77499 | 0.78860 | 0.79437 | 0.80323 | 0.79572 | 0.78212 | 0.78533 | 0.78786 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.4802 | 2.9824 | 3.0095 | 2.8180 | 2.1727 | 2.6349 | 2.8260 | 2.7755 | 2.2404 | 2.6467 | 2.4372 | 2.7248 | 2.4216 | 2.3361 | 2.2377 | 2.6835 | 2.4160 | 2.4840 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 4.9605 | 5.9648 | 6.0190 | 5.6359 | 4.3455 | 5.2698 | 5.6520 | 5.5511 | 4.4808 | 5.2934 | 4.8744 | 5.4495 | 4.8431 | 4.6723 | 4.4753 | 5.3670 | 4.8320 | 4.9680 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.7192 | 2.0665 | 2.0862 | 1.9537 | 1.5055 | 1.8261 | 1.9590 | 1.9229 | 1.5522 | 1.8348 | 1.6890 | 1.8889 | 1.6787 | 1.6195 | 1.5514 | 1.8600 | 1.6748 | 1.7221 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.4384 | 4.1330 | 4.1723 | 3.9073 | 3.0111 | 3.6523 | 3.9180 | 3.8459 | 3.1044 | 3.6696 | 3.3780 | 3.7777 | 3.3574 | 3.2389 | 3.1027 | 3.7200 | 3.3497 | 3.4442 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.11000 | 0.09000 | 0.08000 | 0.10000 | 0.04000 | 0.07000 | 0.07000 | 0.10000 | 0.05000 | 0.10000 | 0.10000 | 0.06000 | 0.07000 | 0.07000 | 0.09000 | 0.07000 | 0.10000 | 0.08000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.11000 | 0.09000 | 0.08000 | 0.10000 | 0.04000 | 0.07000 | 0.07000 | 0.10000 | 0.05000 | 0.10000 | 0.10000 | 0.06000 | 0.07000 | 0.07000 | 0.09000 | 0.07000 | 0.10000 | 0.08000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -3.4182 | -4.1099 | -4.1589 | -3.8939 | -2.9849 | -3.6372 | -3.9046 | -3.8296 | -3.0851 | -3.6500 | -3.3507 | -3.7485 | -3.3330 | -3.2129 | -3.0764 | -3.6828 | -3.3344 | -3.4120 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -3.4182 | -4.1099 | -4.1589 | -3.8939 | -2.9849 | -3.6372 | -3.9046 | -3.8296 | -3.0851 | -3.6500 | -3.3507 | -3.7485 | -3.3330 | -3.2129 | -3.0764 | -3.6828 | -3.3344 | -3.4120 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09237 | 0.09148 | 0.09473 | 0.09227 | 0.09265 | 0.08987 | 0.09057 | 0.09209 | 0.09664 | 0.09128 | 0.09574 | 0.08875 | 0.09177 | 0.09235 | 0.09394 | 0.08788 | 0.09610 | 0.09041 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09237 | 0.09148 | 0.09473 | 0.09227 | 0.09265 | 0.08987 | 0.09057 | 0.09209 | 0.09664 | 0.09128 | 0.09574 | 0.08875 | 0.09177 | 0.09235 | 0.09394 | 0.08788 | 0.09610 | 0.09041 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.88727 | 0.88988 | 0.89429 | 0.87705 | 0.88362 | 0.88227 | 0.89209 | 0.88587 | 0.89436 | 0.88238 | 0.87808 | 0.89101 | 0.88625 | 0.88712 | 0.89047 | 0.88804 | 0.89017 | 0.88742 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.89711 | 0.89969 | 0.90393 | 0.88673 | 0.89339 | 0.89213 | 0.90174 | 0.89567 | 0.90415 | 0.89210 | 0.88760 | 0.90078 | 0.89614 | 0.89680 | 0.90031 | 0.89782 | 0.90004 | 0.89719 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.85311 | 0.85489 | 0.85323 | 0.83720 | 0.83445 | 0.84573 | 0.85383 | 0.84707 | 0.86490 | 0.84154 | 0.83143 | 0.85104 | 0.84964 | 0.85669 | 0.85906 | 0.84480 | 0.84873 | 0.84648 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.85599 | 0.85789 | 0.85588 | 0.83996 | 0.83718 | 0.84845 | 0.85666 | 0.84977 | 0.86782 | 0.84435 | 0.83429 | 0.85387 | 0.85254 | 0.85972 | 0.86186 | 0.84759 | 0.85156 | 0.84932 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.86148 | 0.86652 | 0.86734 | 0.84352 | 0.83797 | 0.84653 | 0.84904 | 0.85166 | 0.86451 | 0.84528 | 0.85265 | 0.85460 | 0.85513 | 0.86069 | 0.84831 | 0.84875 | 0.86216 | 0.84882 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.90785 | 0.91322 | 0.91450 | 0.88867 | 0.88219 | 0.89164 | 0.89489 | 0.89723 | 0.91186 | 0.89074 | 0.89847 | 0.90048 | 0.90106 | 0.90718 | 0.89416 | 0.89414 | 0.90859 | 0.89487 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0328 | 1.0398 | 1.0426 | 1.0431 | 1.0353 | 1.0320 | 1.0447 | 1.0352 | 1.0435 | 1.0223 | 1.0473 | 1.0441 | 1.0358 | 1.0419 | 1.0329 | 1.0300 | 1.0230 | 1.0301 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0655 | 2.0795 | 2.0852 | 2.0861 | 2.0706 | 2.0640 | 2.0894 | 2.0704 | 2.0869 | 2.0445 | 2.0946 | 2.0882 | 2.0717 | 2.0838 | 2.0658 | 2.0600 | 2.0461 | 2.0603 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.71595 | 0.72076 | 0.72273 | 0.72311 | 0.71771 | 0.71539 | 0.72422 | 0.71765 | 0.72334 | 0.70865 | 0.72600 | 0.72376 | 0.71804 | 0.72221 | 0.71602 | 0.71401 | 0.70920 | 0.71412 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4319 | 1.4415 | 1.4455 | 1.4462 | 1.4354 | 1.4308 | 1.4484 | 1.4353 | 1.4467 | 1.4173 | 1.4520 | 1.4475 | 1.4361 | 1.4444 | 1.4320 | 1.4280 | 1.4184 | 1.4282 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24421 | 0.24867 | 0.24145 | 0.23294 | 0.25377 | 0.24952 | 0.24527 | 0.24251 | 0.24421 | 0.24825 | 0.24825 | 0.24846 | 0.24633 | 0.24676 | 0.24038 | 0.23719 | 0.25058 | 0.23762 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24421 | 0.24867 | 0.24145 | 0.23294 | 0.25377 | 0.24952 | 0.24527 | 0.24251 | 0.24421 | 0.24825 | 0.24825 | 0.24846 | 0.24633 | 0.24676 | 0.24038 | 0.23719 | 0.25058 | 0.23762 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3928 | -1.3946 | -1.3980 | -1.3973 | -1.3914 | -1.3950 | -1.3961 | -1.3927 | -1.3955 | -1.3896 | -1.3981 | -1.3965 | -1.3941 | -1.3964 | -1.3925 | -1.3932 | -1.3906 | -1.3954 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.4080 | -1.4178 | -1.4251 | -1.4223 | -1.4104 | -1.4152 | -1.4194 | -1.4102 | -1.4215 | -1.3997 | -1.4269 | -1.4219 | -1.4134 | -1.4210 | -1.4101 | -1.4116 | -1.4050 | -1.4148 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.24944 | 0.24988 | 0.24902 | 0.24895 | 0.25037 | 0.24927 | 0.24930 | 0.24977 | 0.24985 | 0.24997 | 0.24924 | 0.24936 | 0.24951 | 0.24933 | 0.24988 | 0.24974 | 0.25020 | 0.24906 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.24886 | 0.24981 | 0.24808 | 0.24792 | 0.25065 | 0.24856 | 0.24867 | 0.24951 | 0.24962 | 0.24989 | 0.24858 | 0.24875 | 0.24910 | 0.24870 | 0.24967 | 0.24943 | 0.25035 | 0.24817 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 1.1907 | 1.2099 | 1.2132 | 1.1846 | 1.1840 | 1.1889 | 1.1998 | 1.1842 | 1.2246 | 1.1919 | 1.1885 | 1.2050 | 1.1998 | 1.2033 | 1.1953 | 1.1828 | 1.2086 | 1.1893 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.3257 | 1.3465 | 1.3511 | 1.3198 | 1.3195 | 1.3248 | 1.3353 | 1.3175 | 1.3646 | 1.3286 | 1.3229 | 1.3414 | 1.3361 | 1.3399 | 1.3316 | 1.3171 | 1.3463 | 1.3239 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0299 | 1.0439 | 1.0340 | 1.0223 | 1.0384 | 1.0177 | 1.0397 | 1.0319 | 1.0261 | 1.0236 | 1.0408 | 1.0337 | 1.0271 | 1.0446 | 1.0336 | 1.0219 | 1.0264 | 1.0162 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0598 | 2.0878 | 2.0680 | 2.0445 | 2.0768 | 2.0355 | 2.0794 | 2.0638 | 2.0523 | 2.0473 | 2.0815 | 2.0675 | 2.0541 | 2.0891 | 2.0672 | 2.0438 | 2.0527 | 2.0324 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.71392 | 0.72361 | 0.71675 | 0.70868 | 0.71985 | 0.70548 | 0.72067 | 0.71533 | 0.71128 | 0.70961 | 0.72142 | 0.71662 | 0.71198 | 0.72409 | 0.71650 | 0.70839 | 0.71147 | 0.70445 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4278 | 1.4472 | 1.4335 | 1.4174 | 1.4397 | 1.4110 | 1.4413 | 1.4307 | 1.4226 | 1.4192 | 1.4428 | 1.4332 | 1.4240 | 1.4482 | 1.4330 | 1.4168 | 1.4229 | 1.4089 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.27761 | 0.24861 | 0.26589 | 0.27606 | 0.24337 | 0.28100 | 0.26249 | 0.26373 | 0.26619 | 0.24892 | 0.26897 | 0.26650 | 0.27236 | 0.24429 | 0.26465 | 0.25386 | 0.24769 | 0.28624 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.27761 | 0.24861 | 0.26589 | 0.27606 | 0.24337 | 0.28100 | 0.26249 | 0.26373 | 0.26619 | 0.24892 | 0.26897 | 0.26650 | 0.27236 | 0.24429 | 0.26465 | 0.25386 | 0.24769 | 0.28624 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3839 | -1.3956 | -1.3892 | -1.3830 | -1.3940 | -1.3815 | -1.3886 | -1.3869 | -1.3847 | -1.3906 | -1.3916 | -1.3851 | -1.3822 | -1.3934 | -1.3903 | -1.3867 | -1.3897 | -1.3832 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.3931 | -1.4239 | -1.4103 | -1.3905 | -1.4146 | -1.3926 | -1.4104 | -1.4028 | -1.4003 | -1.4039 | -1.4170 | -1.3975 | -1.3983 | -1.4180 | -1.4078 | -1.3986 | -1.4078 | -1.3939 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25218 | 0.25012 | 0.25151 | 0.25219 | 0.24965 | 0.25322 | 0.25191 | 0.25184 | 0.25260 | 0.25008 | 0.25128 | 0.25212 | 0.25385 | 0.25032 | 0.25082 | 0.25138 | 0.25095 | 0.25273 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25453 | 0.25022 | 0.25307 | 0.25446 | 0.24914 | 0.25642 | 0.25372 | 0.25387 | 0.25502 | 0.25021 | 0.25259 | 0.25424 | 0.25801 | 0.25049 | 0.25171 | 0.25275 | 0.25162 | 0.25576 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 1.0213 | 1.0203 | 1.0308 | 1.0013 | 0.99905 | 1.0093 | 1.0237 | 1.0114 | 1.0297 | 1.0096 | 1.0110 | 1.0195 | 1.0148 | 1.0351 | 1.0147 | 1.0054 | 1.0178 | 1.0148 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 1.0914 | 1.0910 | 1.1023 | 1.0716 | 1.0686 | 1.0790 | 1.0950 | 1.0822 | 1.1012 | 1.0794 | 1.0822 | 1.0905 | 1.0857 | 1.1085 | 1.0853 | 1.0761 | 1.0884 | 1.0848 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0301 | 1.0402 | 1.0330 | 1.0158 | 1.0298 | 1.0180 | 1.0453 | 1.0381 | 1.0419 | 1.0206 | 1.0504 | 1.0367 | 1.0224 | 1.0520 | 1.0422 | 1.0309 | 1.0357 | 1.0192 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0601 | 2.0804 | 2.0660 | 2.0316 | 2.0597 | 2.0360 | 2.0907 | 2.0763 | 2.0839 | 2.0411 | 2.1008 | 2.0733 | 2.0448 | 2.1040 | 2.0845 | 2.0617 | 2.0715 | 2.0384 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.71405 | 0.72102 | 0.71610 | 0.70414 | 0.71394 | 0.70570 | 0.72461 | 0.71963 | 0.72228 | 0.70748 | 0.72810 | 0.71863 | 0.70868 | 0.72931 | 0.72248 | 0.71462 | 0.71798 | 0.70648 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4281 | 1.4420 | 1.4322 | 1.4083 | 1.4279 | 1.4114 | 1.4492 | 1.4393 | 1.4446 | 1.4150 | 1.4562 | 1.4373 | 1.4174 | 1.4586 | 1.4450 | 1.4292 | 1.4360 | 1.4130 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.25837 | 0.27202 | 0.26974 | 0.30062 | 0.24862 | 0.26097 | 0.24797 | 0.26487 | 0.24082 | 0.25707 | 0.25349 | 0.26714 | 0.29639 | 0.22554 | 0.23042 | 0.23984 | 0.25057 | 0.26747 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.25837 | 0.27202 | 0.26974 | 0.30062 | 0.24862 | 0.26097 | 0.24797 | 0.26487 | 0.24082 | 0.25707 | 0.25349 | 0.26714 | 0.29639 | 0.22554 | 0.23042 | 0.23984 | 0.25057 | 0.26747 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3843 | -1.3859 | -1.3869 | -1.3769 | -1.3896 | -1.3845 | -1.3919 | -1.3879 | -1.3917 | -1.3878 | -1.3936 | -1.3886 | -1.3807 | -1.3990 | -1.3967 | -1.3919 | -1.3979 | -1.3864 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.3935 | -1.4120 | -1.4067 | -1.3813 | -1.4029 | -1.3928 | -1.4137 | -1.4084 | -1.4191 | -1.3987 | -1.4221 | -1.4042 | -1.3932 | -1.4272 | -1.4189 | -1.4094 | -1.4229 | -1.3969 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.25193 | 0.25355 | 0.25232 | 0.25429 | 0.25047 | 0.25176 | 0.25063 | 0.25203 | 0.25127 | 0.25080 | 0.25087 | 0.25111 | 0.25380 | 0.24866 | 0.24890 | 0.25007 | 0.24890 | 0.25134 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.25378 | 0.25710 | 0.25477 | 0.25895 | 0.25098 | 0.25349 | 0.25111 | 0.25407 | 0.25220 | 0.25163 | 0.25168 | 0.25223 | 0.25765 | 0.24717 | 0.24783 | 0.24994 | 0.24804 | 0.25280 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.5141 | 1.5220 | 1.5314 | 1.5258 | 1.5105 | 1.5227 | 1.5396 | 1.5191 | 1.5408 | 1.5257 | 1.5134 | 1.5216 | 1.5454 | 1.5571 | 1.5159 | 1.5177 | 1.5359 | 1.5332 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.8776 | 1.8913 | 1.9040 | 1.9041 | 1.8828 | 1.8983 | 1.9156 | 1.8866 | 1.9168 | 1.9035 | 1.8806 | 1.8905 | 1.9290 | 1.9450 | 1.8856 | 1.8908 | 1.9128 | 1.9136 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0394 | 1.0426 | 1.0355 | 1.0319 | 1.0451 | 1.0413 | 1.0409 | 1.0305 | 1.0403 | 1.0281 | 1.0401 | 1.0510 | 1.0341 | 1.0669 | 1.0392 | 1.0425 | 1.0308 | 1.0267 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0787 | 2.0853 | 2.0711 | 2.0638 | 2.0901 | 2.0826 | 2.0818 | 2.0610 | 2.0806 | 2.0563 | 2.0802 | 2.1021 | 2.0682 | 2.1339 | 2.0784 | 2.0850 | 2.0616 | 2.0534 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.72048 | 0.72277 | 0.71784 | 0.71532 | 0.72445 | 0.72184 | 0.72160 | 0.71435 | 0.72107 | 0.71276 | 0.72093 | 0.72855 | 0.71687 | 0.73959 | 0.72035 | 0.72266 | 0.71448 | 0.71164 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4410 | 1.4455 | 1.4357 | 1.4306 | 1.4489 | 1.4437 | 1.4432 | 1.4287 | 1.4421 | 1.4255 | 1.4419 | 1.4571 | 1.4337 | 1.4792 | 1.4407 | 1.4453 | 1.4290 | 1.4233 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.26806 | 0.26806 | 0.26011 | 0.27402 | 0.23559 | 0.25944 | 0.26044 | 0.26640 | 0.27502 | 0.25315 | 0.26938 | 0.26972 | 0.27170 | 0.23327 | 0.25812 | 0.25149 | 0.25878 | 0.26077 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.26806 | 0.26806 | 0.26011 | 0.27402 | 0.23559 | 0.25944 | 0.26044 | 0.26640 | 0.27502 | 0.25315 | 0.26938 | 0.26972 | 0.27170 | 0.23327 | 0.25812 | 0.25149 | 0.25878 | 0.26077 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3856 | -1.3905 | -1.3891 | -1.3866 | -1.3951 | -1.3931 | -1.3877 | -1.3860 | -1.3861 | -1.3901 | -1.3843 | -1.3881 | -1.3873 | -1.4057 | -1.3910 | -1.3931 | -1.3873 | -1.3897 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4024 | -1.4183 | -1.4089 | -1.4024 | -1.4195 | -1.4234 | -1.4103 | -1.4019 | -1.4136 | -1.4054 | -1.4079 | -1.4184 | -1.4086 | -1.4491 | -1.4133 | -1.4147 | -1.4112 | -1.4052 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.25246 | 0.25210 | 0.25156 | 0.25194 | 0.24981 | 0.25116 | 0.25227 | 0.25229 | 0.25367 | 0.25055 | 0.25392 | 0.25324 | 0.25240 | 0.24792 | 0.25123 | 0.25016 | 0.25273 | 0.25071 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.25484 | 0.25419 | 0.25322 | 0.25393 | 0.24954 | 0.25222 | 0.25435 | 0.25488 | 0.25720 | 0.25118 | 0.25786 | 0.25612 | 0.25492 | 0.24569 | 0.25256 | 0.25030 | 0.25540 | 0.25147 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.52236 | 0.53417 | 0.52048 | 0.51374 | 0.52757 | 0.52446 | 0.52556 | 0.52548 | 0.51991 | 0.50544 | 0.50246 | 0.52049 | 0.53407 | 0.53834 | 0.52284 | 0.52265 | 0.51354 | 0.53457 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.52531 | 0.53713 | 0.52347 | 0.51659 | 0.53051 | 0.52716 | 0.52848 | 0.52844 | 0.52276 | 0.50830 | 0.50534 | 0.52333 | 0.53699 | 0.54134 | 0.52562 | 0.52551 | 0.51628 | 0.53748 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.39842 | 0.40418 | 0.39903 | 0.38798 | 0.39585 | 0.39506 | 0.40328 | 0.40459 | 0.40451 | 0.39371 | 0.40022 | 0.40072 | 0.39986 | 0.40370 | 0.40102 | 0.39100 | 0.39757 | 0.40655 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.39999 | 0.40575 | 0.40053 | 0.38944 | 0.39729 | 0.39660 | 0.40481 | 0.40614 | 0.40606 | 0.39521 | 0.40171 | 0.40221 | 0.40136 | 0.40527 | 0.40251 | 0.39251 | 0.39905 | 0.40808 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.76427 | 0.75650 | 0.73285 | 0.77650 | 0.72225 | 0.74062 | 0.72007 | 0.71681 | 0.78455 | 0.72000 | 0.73043 | 0.73691 | 0.76016 | 0.76059 | 0.76360 | 0.73986 | 0.73961 | 0.71779 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.76974 | 0.76184 | 0.73801 | 0.78211 | 0.72725 | 0.74564 | 0.72511 | 0.72193 | 0.79020 | 0.72518 | 0.73565 | 0.74214 | 0.76524 | 0.76594 | 0.76904 | 0.74524 | 0.74472 | 0.72282 |
| eval/lm/c4_en-validation/CE loss | lower | 3.2684 | 3.2730 | 3.2804 | 3.2467 | 3.2504 | 3.2551 | 3.2788 | 3.2625 | 3.3052 | 3.2564 | 3.2428 | 3.2790 | 3.2714 | 3.2800 | 3.2745 | 3.2619 | 3.2672 | 3.2704 |
| eval/lm/c4_en-validation/PPL | lower | 26.27 | 26.39 | 26.59 | 25.71 | 25.80 | 25.92 | 26.54 | 26.11 | 27.25 | 25.96 | 25.61 | 26.55 | 26.35 | 26.58 | 26.43 | 26.10 | 26.24 | 26.32 |
| eval/lm/dolma_books-validation/CE loss | lower | 3.2185 | 3.2293 | 3.2419 | 3.1996 | 3.2039 | 3.1992 | 3.2260 | 3.2156 | 3.2644 | 3.2011 | 3.1907 | 3.2288 | 3.2249 | 3.2368 | 3.2282 | 3.2155 | 3.2233 | 3.2160 |
| eval/lm/dolma_books-validation/PPL | lower | 24.99 | 25.26 | 25.58 | 24.52 | 24.63 | 24.51 | 25.18 | 24.92 | 26.16 | 24.56 | 24.30 | 25.25 | 25.15 | 25.45 | 25.23 | 24.92 | 25.11 | 24.93 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.3995 | 3.4029 | 3.4103 | 3.3758 | 3.3817 | 3.3851 | 3.4091 | 3.3942 | 3.4332 | 3.3857 | 3.3730 | 3.4075 | 3.4032 | 3.4107 | 3.4043 | 3.3928 | 3.3988 | 3.3998 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 29.95 | 30.05 | 30.28 | 29.25 | 29.42 | 29.52 | 30.24 | 29.79 | 30.97 | 29.54 | 29.17 | 30.19 | 30.06 | 30.29 | 30.09 | 29.75 | 29.93 | 29.96 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.4246 | 2.4287 | 2.4351 | 2.4057 | 2.4092 | 2.4159 | 2.4344 | 2.4219 | 2.4588 | 2.4174 | 2.4007 | 2.4353 | 2.4267 | 2.4335 | 2.4287 | 2.4197 | 2.4235 | 2.4276 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 11.30 | 11.34 | 11.42 | 11.09 | 11.12 | 11.20 | 11.41 | 11.27 | 11.69 | 11.22 | 11.03 | 11.42 | 11.32 | 11.40 | 11.34 | 11.24 | 11.29 | 11.33 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.5515 | 3.5519 | 3.5621 | 3.5304 | 3.5335 | 3.5374 | 3.5602 | 3.5458 | 3.5873 | 3.5395 | 3.5299 | 3.5617 | 3.5538 | 3.5629 | 3.5555 | 3.5486 | 3.5518 | 3.5533 |
| eval/lm/dolma_reddit-validation/PPL | lower | 34.87 | 34.88 | 35.24 | 34.14 | 34.24 | 34.38 | 35.17 | 34.67 | 36.14 | 34.45 | 34.12 | 35.22 | 34.95 | 35.26 | 35.01 | 34.77 | 34.88 | 34.93 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.6244 | 1.6257 | 1.6384 | 1.6006 | 1.5999 | 1.6127 | 1.6161 | 1.6019 | 1.6422 | 1.5927 | 1.5741 | 1.6179 | 1.6740 | 1.6849 | 1.6801 | 1.6136 | 1.6172 | 1.6288 |
| eval/lm/dolma_stack-validation/PPL | lower | 5.0756 | 5.0819 | 5.1468 | 4.9560 | 4.9527 | 5.0163 | 5.0332 | 4.9622 | 5.1666 | 4.9172 | 4.8266 | 5.0425 | 5.3335 | 5.3921 | 5.3661 | 5.0209 | 5.0391 | 5.0979 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.9404 | 2.9505 | 2.9548 | 2.9233 | 2.9266 | 2.9291 | 2.9517 | 2.9353 | 2.9813 | 2.9311 | 2.9184 | 2.9519 | 2.9487 | 2.9567 | 2.9489 | 2.9366 | 2.9438 | 2.9429 |
| eval/lm/dolma_wiki-validation/PPL | lower | 18.92 | 19.12 | 19.20 | 18.60 | 18.66 | 18.71 | 19.14 | 18.83 | 19.71 | 18.75 | 18.51 | 19.14 | 19.08 | 19.23 | 19.08 | 18.85 | 18.99 | 18.97 |
| eval/lm/ice-validation/CE loss | lower | 3.3578 | 3.3534 | 3.3643 | 3.3292 | 3.3384 | 3.3303 | 3.3812 | 3.3535 | 3.3922 | 3.3307 | 3.2980 | 3.3556 | 3.3687 | 3.3820 | 3.3690 | 3.3478 | 3.3547 | 3.3657 |
| eval/lm/ice-validation/PPL | lower | 28.73 | 28.60 | 28.91 | 27.91 | 28.17 | 27.95 | 29.41 | 28.60 | 29.73 | 27.96 | 27.06 | 28.66 | 29.04 | 29.43 | 29.05 | 28.44 | 28.64 | 28.95 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 3.3563 | 3.3628 | 3.3734 | 3.3393 | 3.3463 | 3.3500 | 3.3628 | 3.3487 | 3.3863 | 3.3443 | 3.3382 | 3.3629 | 3.3557 | 3.3640 | 3.3625 | 3.3477 | 3.3554 | 3.3513 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 28.68 | 28.87 | 29.18 | 28.20 | 28.40 | 28.50 | 28.87 | 28.47 | 29.56 | 28.34 | 28.17 | 28.87 | 28.67 | 28.90 | 28.86 | 28.44 | 28.66 | 28.54 |
| eval/lm/pile-validation/CE loss | lower | 2.5398 | 2.5468 | 2.5479 | 2.5151 | 2.5179 | 2.5241 | 2.5417 | 2.5274 | 2.5688 | 2.5210 | 2.5054 | 2.5430 | 2.5517 | 2.5608 | 2.5578 | 2.5312 | 2.5350 | 2.5369 |
| eval/lm/pile-validation/PPL | lower | 12.68 | 12.77 | 12.78 | 12.37 | 12.40 | 12.48 | 12.70 | 12.52 | 13.05 | 12.44 | 12.25 | 12.72 | 12.83 | 12.95 | 12.91 | 12.57 | 12.62 | 12.64 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.9335 | 2.9380 | 2.9657 | 2.9148 | 2.9101 | 2.9317 | 2.9606 | 2.9399 | 2.9939 | 2.9314 | 2.9103 | 2.9586 | 2.9468 | 2.9598 | 2.9562 | 2.9278 | 2.9392 | 2.9480 |
| eval/lm/wikitext_103-validation/PPL | lower | 18.79 | 18.88 | 19.41 | 18.44 | 18.36 | 18.76 | 19.31 | 18.91 | 19.96 | 18.75 | 18.36 | 19.27 | 19.04 | 19.29 | 19.22 | 18.69 | 18.90 | 19.07 |
| throughput/in-loop eval batches | see metric | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 826.0 | 826.0 | 826.0 | 1641.0 | 1641.0 | 1641.0 | 3281.0 | 3281.0 | 3281.0 | 1641.0 | 1641.0 | 1641.0 |
| throughput/in-loop eval time (s) | see metric | 225.1 | 225.5 | 352.4 | 218.9 | 220.2 | 224.7 | 218.9 | 227.1 | 217.4 | 245.4 | 221.5 | 242.7 | 410.4 | 406.7 | 399.0 | 228.4 | 270.1 | 277.3 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| eg-275m-cx2-b384k-eg24e2k-lr1.8e-3-r3<br>`ujt9b8kw` | finished | b384k | 8047165440.0 | 20465 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ujt9b8kw) |
| eg-275m-cx2-b384k-eg24e2k-lr3.6e-3-r3<br>`ijiw4id5` | finished | b384k | 8047165440.0 | 20465 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ijiw4id5) |
| eg-275m-cx2-b384k-eg24e2k-lr9e-4-r3<br>`spmc55m4` | finished | b384k | 8047165440.0 | 20465 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/spmc55m4) |
| eg-275m-cx2-b384k-eg96e8k-lr1.8e-3-r3<br>`38gyc8af` | finished | b384k | 8071544832.0 | 20527 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/38gyc8af) |
| eg-275m-cx2-b384k-eg96e8k-lr3.6e-3-r3<br>`53yd3xre` | finished | b384k | 8071544832.0 | 20527 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/53yd3xre) |
| eg-275m-cx2-b384k-eg96e8k-lr9e-4-r3<br>`tcvldbk1` | finished | b384k | 8071544832.0 | 20527 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/tcvldbk1) |
| eg-275m-cx2-eg24e2k-lr1e-3-r1<br>`wm2wrmnc` | finished | original | 8047296512.0 | 15349 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wm2wrmnc) |
| eg-275m-cx2-eg24e2k-lr2e-3-r1<br>`50mfn51v` | finished | original | 8047296512.0 | 15349 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/50mfn51v) |
| eg-275m-cx2-eg24e2k-lr5e-4-r1<br>`8n4j64g9` | finished | original | 8047296512.0 | 15349 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8n4j64g9) |
| eg-275m-cx2-eg96e8k-lr1e-3-r1<br>`s1t6b084` | finished | original | 8071938048.0 | 15396 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/s1t6b084) |
| eg-275m-cx2-eg96e8k-lr2e-3-r1<br>`xys9u31e` | finished | original | 8071938048.0 | 15396 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xys9u31e) |
| eg-275m-cx2-eg96e8k-lr5e-4-r1<br>`728o3v27` | finished | original | 8071938048.0 | 15396 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/728o3v27) |
| 275m-cx2-b256k-lr1e-3<br>`dgej30hb` | finished | gpu2-ep1mb16 | 8055422976.0 | 30729 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/dgej30hb) |
| 275m-cx2-b256k-lr6e-4-r2<br>`8382k902` | finished | gpu2-ep1mb16 | 8055422976.0 | 30729 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8382k902) |
| 275m-cx2-b256k-lr8e-4-r2<br>`56e1coi7` | finished | gpu2-ep1mb16 | 8055422976.0 | 30729 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/56e1coi7) |
| 275m-cx2-b384k-lr1.8e-3-r3<br>`lq4zvsx4` | finished | b384k-gpu2-ep1mb8 | 8055422976.0 | 20486 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/lq4zvsx4) |
| 275m-cx2-b384k-lr3.6e-3-r3<br>`pv6y1aqx` | finished | b384k-gpu2-ep1mb8 | 8055422976.0 | 20486 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/pv6y1aqx) |
| 275m-cx2-b384k-lr9e-4-r3<br>`atxrokcu` | finished | b384k-gpu2-ep1mb8 | 8055422976.0 | 20486 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/atxrokcu) |

## 275m Cx4

| metric | direction | eg-275m-cx4-eg24e2k-lr1.6e-3-r1<br>`eq0vqyj9` | eg-275m-cx4-eg24e2k-lr3.2e-3-r1<br>`mrpoyk8n` | eg-275m-cx4-eg24e2k-lr8e-4-r1<br>`5talvqd1` | eg-275m-cx4-eg96e8k-lr1.6e-3-r1<br>`gsqree2x` | eg-275m-cx4-eg96e8k-lr3.2e-3-r1<br>`589cgpj0` | eg-275m-cx4-eg96e8k-lr8e-4-r1<br>`0vr98te9` | 275m-cx4-b512k-lr1.5e-3<br>`vtn70hed` | 275m-cx4-b512k-lr1e-3<br>`m053n1rr` | 275m-cx4-b512k-lr2.5e-3<br>`f5csk4pn` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.99141 | 1.0102 | 0.99780 | 0.97393 | 0.96492 | 0.98316 | 0.97895 | 0.97498 | 0.96716 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 1.0869 | 1.1027 | 1.0909 | 1.0632 | 1.0555 | 1.0760 | 1.0701 | 1.0671 | 1.0590 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0547 | 1.0207 | 1.0269 | 1.0572 | 1.0456 | 1.0099 | 1.0257 | 1.0137 | 1.0196 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.1093 | 2.0413 | 2.0537 | 2.1143 | 2.0913 | 2.0199 | 2.0514 | 2.0274 | 2.0391 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.73109 | 0.70749 | 0.71183 | 0.73275 | 0.72482 | 0.70012 | 0.71105 | 0.70279 | 0.70680 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4622 | 1.4150 | 1.4237 | 1.4655 | 1.4496 | 1.4002 | 1.4221 | 1.4056 | 1.4136 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.24147 | 0.25171 | 0.25939 | 0.25427 | 0.26109 | 0.27048 | 0.24829 | 0.23805 | 0.26536 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.24147 | 0.25171 | 0.25939 | 0.25427 | 0.26109 | 0.27048 | 0.24829 | 0.23805 | 0.26536 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.4494 | -1.4029 | -1.4087 | -1.4533 | -1.4387 | -1.3921 | -1.3984 | -1.3955 | -1.3986 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.4494 | -1.4029 | -1.4087 | -1.4533 | -1.4387 | -1.3921 | -1.3984 | -1.3955 | -1.3986 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.24802 | 0.25136 | 0.25190 | 0.25085 | 0.25090 | 0.25356 | 0.25069 | 0.25068 | 0.25053 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.24802 | 0.25136 | 0.25190 | 0.25085 | 0.25090 | 0.25356 | 0.25069 | 0.25068 | 0.25053 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.80250 | 0.80373 | 0.81689 | 0.78288 | 0.78160 | 0.79183 | 0.77895 | 0.78456 | 0.77259 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.87467 | 0.87580 | 0.89052 | 0.85199 | 0.85136 | 0.86381 | 0.84852 | 0.85426 | 0.84163 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0365 | 1.0236 | 1.0482 | 1.0457 | 1.0542 | 1.0303 | 1.0215 | 1.0153 | 1.0270 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0730 | 2.0473 | 2.0964 | 2.0913 | 2.1084 | 2.0606 | 2.0429 | 2.0307 | 2.0540 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.71845 | 0.70960 | 0.72653 | 0.72483 | 0.73073 | 0.71414 | 0.70815 | 0.70388 | 0.71193 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4369 | 1.4192 | 1.4531 | 1.4497 | 1.4615 | 1.4283 | 1.4163 | 1.4078 | 1.4239 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.26010 | 0.24158 | 0.24158 | 0.24663 | 0.24747 | 0.23569 | 0.24705 | 0.25337 | 0.24242 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.26010 | 0.24158 | 0.24158 | 0.24663 | 0.24747 | 0.23569 | 0.24705 | 0.25337 | 0.24242 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.4230 | -1.4061 | -1.4395 | -1.4383 | -1.4488 | -1.4206 | -1.3924 | -1.3983 | -1.4157 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.4230 | -1.4061 | -1.4395 | -1.4383 | -1.4488 | -1.4206 | -1.3924 | -1.3983 | -1.4157 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.24980 | 0.25011 | 0.24935 | 0.24933 | 0.24876 | 0.24801 | 0.25122 | 0.25068 | 0.24928 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.24980 | 0.25011 | 0.24935 | 0.24933 | 0.24876 | 0.24801 | 0.25122 | 0.25068 | 0.24928 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 1.9358 | 1.9771 | 2.0487 | 1.9075 | 1.9542 | 1.9291 | 1.8640 | 1.9207 | 1.9143 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 3.0948 | 3.1617 | 3.2765 | 3.0601 | 3.1205 | 3.0878 | 2.9980 | 3.0874 | 3.0575 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 1.3417 | 1.3703 | 1.4199 | 1.3222 | 1.3545 | 1.3371 | 1.2920 | 1.3312 | 1.3269 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 2.1451 | 2.1913 | 2.2713 | 2.1210 | 2.1631 | 2.1403 | 2.0781 | 2.1398 | 2.1192 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.17096 | 0.18147 | 0.19962 | 0.22350 | 0.21490 | 0.21394 | 0.23018 | 0.21012 | 0.21012 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.17096 | 0.18147 | 0.19962 | 0.22350 | 0.21490 | 0.21394 | 0.23018 | 0.21012 | 0.21012 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -2.4888 | -2.5575 | -2.6083 | -2.4622 | -2.4817 | -2.4318 | -2.3273 | -2.4022 | -2.4481 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -2.4888 | -2.5575 | -2.6083 | -2.4622 | -2.4817 | -2.4318 | -2.3273 | -2.4022 | -2.4481 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.15400 | 0.16590 | 0.17075 | 0.18382 | 0.16656 | 0.16109 | 0.17850 | 0.17249 | 0.17424 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.15400 | 0.16590 | 0.17075 | 0.18382 | 0.16656 | 0.16109 | 0.17850 | 0.17249 | 0.17424 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.49798 | 0.50627 | 0.50400 | 0.50613 | 0.49362 | 0.47706 | 0.46775 | 0.47385 | 0.49589 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.54210 | 0.55057 | 0.54913 | 0.55127 | 0.53656 | 0.51809 | 0.50779 | 0.51728 | 0.53891 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.34519 | 0.35095 | 0.34936 | 0.35080 | 0.34212 | 0.33068 | 0.32423 | 0.32848 | 0.34374 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.37580 | 0.38161 | 0.38059 | 0.38211 | 0.37190 | 0.35906 | 0.35196 | 0.35853 | 0.37355 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.45553 | 0.43083 | 0.46542 | 0.47628 | 0.46146 | 0.45553 | 0.45949 | 0.49506 | 0.46937 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.45553 | 0.43083 | 0.46542 | 0.47628 | 0.46146 | 0.45553 | 0.45949 | 0.49506 | 0.46937 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -3.1770 | -3.3534 | -3.0530 | -3.0910 | -3.1239 | -3.0699 | -3.0282 | -2.8243 | -3.1217 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -3.1770 | -3.3534 | -3.0530 | -3.0910 | -3.1239 | -3.0699 | -3.0282 | -2.8243 | -3.1217 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.43668 | 0.42365 | 0.44072 | 0.44407 | 0.44778 | 0.44398 | 0.44550 | 0.46419 | 0.44921 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.43668 | 0.42365 | 0.44072 | 0.44407 | 0.44778 | 0.44398 | 0.44550 | 0.46419 | 0.44921 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.65343 | 0.63130 | 0.55290 | 0.56043 | 0.60763 | 0.58495 | 0.51547 | 0.53782 | 0.55527 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.78556 | 0.75879 | 0.66175 | 0.67329 | 0.73153 | 0.70413 | 0.61881 | 0.64619 | 0.66643 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.45298 | 0.43778 | 0.38341 | 0.38859 | 0.42138 | 0.40567 | 0.35749 | 0.37292 | 0.38508 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.54464 | 0.52626 | 0.45890 | 0.46688 | 0.50730 | 0.48831 | 0.42915 | 0.44815 | 0.46214 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.74349 | 0.74156 | 0.74349 | 0.77338 | 0.75699 | 0.77724 | 0.77821 | 0.74638 | 0.74446 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.74349 | 0.74156 | 0.74349 | 0.77338 | 0.75699 | 0.77724 | 0.77821 | 0.74638 | 0.74446 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.66575 | -0.69381 | -0.72444 | -0.64203 | -0.65526 | -0.63362 | -0.61943 | -0.65193 | -0.65411 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.66575 | -0.69381 | -0.72444 | -0.64203 | -0.65526 | -0.63362 | -0.61943 | -0.65193 | -0.65411 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.64549 | 0.65150 | 0.63896 | 0.65932 | 0.66477 | 0.66067 | 0.67332 | 0.65850 | 0.66502 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.64549 | 0.65150 | 0.63896 | 0.65932 | 0.66477 | 0.66067 | 0.67332 | 0.65850 | 0.66502 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.30491 | 0.27640 | 0.28186 | 0.28731 | 0.31836 | 0.26477 | 0.30754 | 0.30838 | 0.28774 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.31512 | 0.28578 | 0.29120 | 0.29688 | 0.32908 | 0.27366 | 0.31774 | 0.31882 | 0.29732 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.21139 | 0.19160 | 0.19536 | 0.19918 | 0.22068 | 0.18355 | 0.21319 | 0.21378 | 0.19947 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.21845 | 0.19810 | 0.20188 | 0.20580 | 0.22811 | 0.18970 | 0.22027 | 0.22098 | 0.20609 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.84079 | 0.82290 | 0.80233 | 0.79696 | 0.81574 | 0.79964 | 0.80233 | 0.81306 | 0.80501 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.84079 | 0.82290 | 0.80233 | 0.79696 | 0.81574 | 0.79964 | 0.80233 | 0.81306 | 0.80501 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.44268 | -0.51820 | -0.53347 | -0.55686 | -0.49330 | -0.52425 | -0.64811 | -0.52462 | -0.53803 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.44268 | -0.51820 | -0.53347 | -0.55686 | -0.49330 | -0.52425 | -0.64811 | -0.52462 | -0.53803 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.82766 | 0.81533 | 0.80559 | 0.79816 | 0.80878 | 0.80791 | 0.80046 | 0.80638 | 0.80457 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.82766 | 0.81533 | 0.80559 | 0.79816 | 0.80878 | 0.80791 | 0.80046 | 0.80638 | 0.80457 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 1.1668 | 1.1585 | 1.1668 | 1.1632 | 1.1930 | 1.1759 | 1.1088 | 1.1622 | 1.0859 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.8539 | 1.8639 | 1.8579 | 1.8615 | 1.9295 | 1.8839 | 1.7583 | 1.8607 | 1.7455 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.84939 | 0.84581 | 0.84600 | 0.84412 | 0.86381 | 0.85047 | 0.80913 | 0.84738 | 0.79380 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.3855 | 1.3976 | 1.3841 | 1.3863 | 1.4294 | 1.3975 | 1.3202 | 1.3946 | 1.3106 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.60674 | 0.61423 | 0.58989 | 0.60674 | 0.59363 | 0.59925 | 0.65730 | 0.62172 | 0.64232 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.60674 | 0.61423 | 0.58989 | 0.60674 | 0.59363 | 0.59925 | 0.65730 | 0.62172 | 0.64232 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -1.0311 | -1.0689 | -1.0710 | -1.0307 | -1.0319 | -1.0179 | -0.99295 | -1.0270 | -1.0513 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -1.0311 | -1.0689 | -1.0710 | -1.0307 | -1.0319 | -1.0179 | -0.99295 | -1.0270 | -1.0513 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.52601 | 0.54583 | 0.52698 | 0.52274 | 0.53505 | 0.52207 | 0.56562 | 0.54055 | 0.55139 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.52601 | 0.54583 | 0.52698 | 0.52274 | 0.53505 | 0.52207 | 0.56562 | 0.54055 | 0.55139 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.9657 | 2.1064 | 2.1030 | 2.0302 | 1.9455 | 2.0023 | 2.0737 | 2.0899 | 1.9070 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.6917 | 2.8849 | 2.8784 | 2.8132 | 2.6806 | 2.7366 | 2.8472 | 2.8662 | 2.6177 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.3625 | 1.4600 | 1.4576 | 1.4072 | 1.3484 | 1.3877 | 1.4375 | 1.4487 | 1.3218 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.8659 | 1.9997 | 1.9951 | 1.9499 | 1.8580 | 1.8968 | 1.9734 | 1.9867 | 1.8145 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.22313 | 0.24282 | 0.23298 | 0.21657 | 0.22149 | 0.23544 | 0.23052 | 0.21411 | 0.22395 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.22313 | 0.24282 | 0.23298 | 0.21657 | 0.22149 | 0.23544 | 0.23052 | 0.21411 | 0.22395 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -4.5395 | -4.2776 | -4.5032 | -4.4255 | -4.4724 | -4.4587 | -4.5280 | -4.4688 | -4.6622 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -4.5395 | -4.2776 | -4.5032 | -4.4255 | -4.4724 | -4.4587 | -4.5280 | -4.4688 | -4.6622 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.23933 | 0.25370 | 0.24342 | 0.23528 | 0.24435 | 0.25276 | 0.24091 | 0.23413 | 0.23664 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.23933 | 0.25370 | 0.24342 | 0.23528 | 0.24435 | 0.25276 | 0.24091 | 0.23413 | 0.23664 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.53480 | 0.52589 | 0.53625 | 0.51605 | 0.53576 | 0.52620 | 0.51645 | 0.52357 | 0.54088 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.54186 | 0.53279 | 0.54328 | 0.52287 | 0.54278 | 0.53287 | 0.52317 | 0.53073 | 0.54825 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.73707 | 0.73968 | 0.73626 | 0.72380 | 0.72767 | 0.72906 | 0.74429 | 0.73798 | 0.73684 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.74355 | 0.74591 | 0.74284 | 0.73007 | 0.73386 | 0.73546 | 0.75068 | 0.74443 | 0.74314 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.6235 | 3.0895 | 2.3243 | 2.4778 | 2.6552 | 2.4420 | 2.3838 | 2.8187 | 2.2300 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 5.2470 | 6.1790 | 4.6487 | 4.9555 | 5.3104 | 4.8841 | 4.7677 | 5.6375 | 4.4600 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.8185 | 2.1414 | 1.6113 | 1.7179 | 1.8399 | 1.6933 | 1.6525 | 1.9539 | 1.5456 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.6370 | 4.2829 | 3.2226 | 3.4357 | 3.6798 | 3.3866 | 3.3050 | 3.9078 | 3.0912 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.10000 | 0.10000 | 0.10000 | 0.07000 | 0.10000 | 0.08000 | 0.11000 | 0.10000 | 0.07000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.10000 | 0.10000 | 0.10000 | 0.07000 | 0.10000 | 0.08000 | 0.11000 | 0.10000 | 0.07000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -3.6264 | -4.2784 | -3.2032 | -3.4232 | -3.6649 | -3.3791 | -3.2855 | -3.9008 | -3.0766 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -3.6264 | -4.2784 | -3.2032 | -3.4232 | -3.6649 | -3.3791 | -3.2855 | -3.9008 | -3.0766 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09149 | 0.09307 | 0.09499 | 0.09010 | 0.09275 | 0.09457 | 0.09706 | 0.09620 | 0.09398 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09149 | 0.09307 | 0.09499 | 0.09010 | 0.09275 | 0.09457 | 0.09706 | 0.09620 | 0.09398 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.86349 | 0.86794 | 0.86814 | 0.85600 | 0.86020 | 0.85889 | 0.86002 | 0.85861 | 0.86197 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.87289 | 0.87755 | 0.87770 | 0.86558 | 0.86962 | 0.86837 | 0.86940 | 0.86794 | 0.87148 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.79711 | 0.80512 | 0.80238 | 0.78672 | 0.78360 | 0.78692 | 0.78895 | 0.79581 | 0.79325 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.79976 | 0.80803 | 0.80511 | 0.78935 | 0.78610 | 0.78958 | 0.79157 | 0.79838 | 0.79594 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.81658 | 0.82303 | 0.82406 | 0.80774 | 0.80791 | 0.81650 | 0.80566 | 0.81159 | 0.81094 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.85935 | 0.86661 | 0.86802 | 0.85010 | 0.85053 | 0.86020 | 0.84838 | 0.85493 | 0.85375 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0346 | 1.0221 | 1.0317 | 1.0443 | 1.0532 | 1.0249 | 1.0288 | 1.0189 | 1.0247 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0691 | 2.0443 | 2.0635 | 2.0886 | 2.1063 | 2.0498 | 2.0575 | 2.0377 | 2.0494 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.71717 | 0.70857 | 0.71523 | 0.72388 | 0.73000 | 0.71045 | 0.71317 | 0.70627 | 0.71039 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4343 | 1.4171 | 1.4305 | 1.4478 | 1.4600 | 1.4209 | 1.4263 | 1.4125 | 1.4208 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24676 | 0.26652 | 0.26121 | 0.24293 | 0.24740 | 0.25356 | 0.25802 | 0.24230 | 0.24591 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24676 | 0.26652 | 0.26121 | 0.24293 | 0.24740 | 0.25356 | 0.25802 | 0.24230 | 0.24591 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3941 | -1.3882 | -1.3910 | -1.3968 | -1.4032 | -1.3906 | -1.3905 | -1.3895 | -1.3906 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.4210 | -1.4036 | -1.4087 | -1.4264 | -1.4467 | -1.4067 | -1.4034 | -1.3986 | -1.4055 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.25039 | 0.25122 | 0.25048 | 0.24976 | 0.24919 | 0.25041 | 0.25006 | 0.24991 | 0.25018 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.25050 | 0.25243 | 0.25099 | 0.24950 | 0.24853 | 0.25078 | 0.25016 | 0.24978 | 0.25021 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 1.1442 | 1.1580 | 1.1691 | 1.1271 | 1.1220 | 1.1431 | 1.1368 | 1.1389 | 1.1421 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.2750 | 1.2894 | 1.3026 | 1.2557 | 1.2492 | 1.2744 | 1.2669 | 1.2686 | 1.2729 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0353 | 1.0275 | 1.0265 | 1.0453 | 1.0335 | 1.0147 | 1.0213 | 1.0070 | 1.0220 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0705 | 2.0549 | 2.0529 | 2.0905 | 2.0669 | 2.0294 | 2.0426 | 2.0140 | 2.0441 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.71761 | 0.71226 | 0.71155 | 0.72452 | 0.71635 | 0.70337 | 0.70797 | 0.69807 | 0.70847 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4352 | 1.4245 | 1.4231 | 1.4490 | 1.4327 | 1.4067 | 1.4159 | 1.3961 | 1.4169 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.25632 | 0.26342 | 0.26681 | 0.26218 | 0.26990 | 0.28686 | 0.27205 | 0.28964 | 0.25941 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.25632 | 0.26342 | 0.26681 | 0.26218 | 0.26990 | 0.28686 | 0.27205 | 0.28964 | 0.25941 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3905 | -1.3922 | -1.3863 | -1.3932 | -1.3919 | -1.3841 | -1.3852 | -1.3811 | -1.3894 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.4210 | -1.4124 | -1.3999 | -1.4306 | -1.4178 | -1.3968 | -1.3926 | -1.3855 | -1.4046 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25221 | 0.25030 | 0.25175 | 0.25224 | 0.25126 | 0.25251 | 0.25131 | 0.25252 | 0.25078 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25402 | 0.25056 | 0.25349 | 0.25421 | 0.25266 | 0.25514 | 0.25255 | 0.25508 | 0.25152 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.96868 | 0.97569 | 0.97909 | 0.95877 | 0.95621 | 0.96331 | 0.95269 | 0.96012 | 0.96972 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 1.0357 | 1.0422 | 1.0461 | 1.0245 | 1.0222 | 1.0291 | 1.0184 | 1.0256 | 1.0372 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0595 | 1.0259 | 1.0372 | 1.0590 | 1.0557 | 1.0258 | 1.0344 | 1.0153 | 1.0363 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.1189 | 2.0517 | 2.0743 | 2.1180 | 2.1115 | 2.0516 | 2.0689 | 2.0305 | 2.0725 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.73438 | 0.71112 | 0.71894 | 0.73404 | 0.73175 | 0.71112 | 0.71713 | 0.70382 | 0.71836 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4688 | 1.4222 | 1.4379 | 1.4681 | 1.4635 | 1.4222 | 1.4343 | 1.4076 | 1.4367 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.22067 | 0.25057 | 0.23692 | 0.22814 | 0.22814 | 0.24082 | 0.24179 | 0.26032 | 0.22652 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.22067 | 0.25057 | 0.23692 | 0.22814 | 0.22814 | 0.24082 | 0.24179 | 0.26032 | 0.22652 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.4093 | -1.3897 | -1.3960 | -1.4048 | -1.4046 | -1.3909 | -1.3941 | -1.3872 | -1.3994 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.4542 | -1.4101 | -1.4170 | -1.4463 | -1.4485 | -1.4096 | -1.4084 | -1.3952 | -1.4237 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.24702 | 0.25126 | 0.24903 | 0.24842 | 0.24860 | 0.25060 | 0.24890 | 0.25067 | 0.24822 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.24398 | 0.25239 | 0.24807 | 0.24688 | 0.24713 | 0.25109 | 0.24789 | 0.25135 | 0.24659 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.4668 | 1.4576 | 1.4687 | 1.4402 | 1.4168 | 1.4385 | 1.4289 | 1.4330 | 1.4414 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.8277 | 1.8090 | 1.8300 | 1.7959 | 1.7617 | 1.7879 | 1.7779 | 1.7849 | 1.7903 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0535 | 1.0276 | 1.0427 | 1.0607 | 1.0650 | 1.0306 | 1.0412 | 1.0190 | 1.0253 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.1069 | 2.0552 | 2.0855 | 2.1215 | 2.1300 | 2.0612 | 2.0824 | 2.0380 | 2.0506 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.73024 | 0.71226 | 0.72283 | 0.73523 | 0.73819 | 0.71439 | 0.72179 | 0.70642 | 0.71079 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4605 | 1.4245 | 1.4457 | 1.4705 | 1.4764 | 1.4288 | 1.4436 | 1.4128 | 1.4216 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24818 | 0.26905 | 0.25580 | 0.24354 | 0.23956 | 0.25182 | 0.23791 | 0.26508 | 0.25845 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24818 | 0.26905 | 0.25580 | 0.24354 | 0.23956 | 0.25182 | 0.23791 | 0.26508 | 0.25845 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.4038 | -1.3867 | -1.3914 | -1.4047 | -1.4113 | -1.3915 | -1.3961 | -1.3883 | -1.3891 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4463 | -1.4122 | -1.4183 | -1.4498 | -1.4613 | -1.4159 | -1.4173 | -1.4001 | -1.4090 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.24913 | 0.25314 | 0.25149 | 0.24919 | 0.24705 | 0.25107 | 0.24908 | 0.25075 | 0.25149 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.24852 | 0.25609 | 0.25271 | 0.24854 | 0.24441 | 0.25192 | 0.24821 | 0.25149 | 0.25283 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.48221 | 0.49410 | 0.48383 | 0.48804 | 0.47764 | 0.47639 | 0.49258 | 0.50135 | 0.47869 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.48493 | 0.49674 | 0.48659 | 0.49078 | 0.48028 | 0.47909 | 0.49530 | 0.50415 | 0.48134 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.38257 | 0.38966 | 0.39588 | 0.37612 | 0.37186 | 0.36762 | 0.38396 | 0.39165 | 0.37893 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.38399 | 0.39122 | 0.39744 | 0.37754 | 0.37329 | 0.36910 | 0.38548 | 0.39322 | 0.38037 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.64907 | 0.67522 | 0.71328 | 0.67641 | 0.68396 | 0.66480 | 0.66805 | 0.66611 | 0.65059 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.65347 | 0.67984 | 0.71839 | 0.68111 | 0.68879 | 0.66949 | 0.67268 | 0.67075 | 0.65510 |
| eval/lm/c4_en-validation/CE loss | lower | 3.1834 | 3.1895 | 3.1947 | 3.1583 | 3.1675 | 3.1653 | 3.1717 | 3.1780 | 3.1745 |
| eval/lm/c4_en-validation/PPL | lower | 24.13 | 24.28 | 24.40 | 23.53 | 23.75 | 23.70 | 23.85 | 24.00 | 23.91 |
| eval/lm/dolma_books-validation/CE loss | lower | 3.1223 | 3.1310 | 3.1327 | 3.0888 | 3.0981 | 3.0999 | 3.1085 | 3.1085 | 3.1033 |
| eval/lm/dolma_books-validation/PPL | lower | 22.70 | 22.90 | 22.94 | 21.95 | 22.16 | 22.20 | 22.39 | 22.39 | 22.27 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.3159 | 3.3228 | 3.3270 | 3.2915 | 3.2984 | 3.2976 | 3.3030 | 3.3089 | 3.3058 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 27.55 | 27.74 | 27.86 | 26.88 | 27.07 | 27.05 | 27.19 | 27.36 | 27.27 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.3511 | 2.3537 | 2.3606 | 2.3277 | 2.3329 | 2.3356 | 2.3368 | 2.3444 | 2.3385 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 10.50 | 10.52 | 10.60 | 10.25 | 10.31 | 10.34 | 10.35 | 10.43 | 10.37 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.4764 | 3.4786 | 3.4851 | 3.4505 | 3.4556 | 3.4571 | 3.4646 | 3.4694 | 3.4604 |
| eval/lm/dolma_reddit-validation/PPL | lower | 32.34 | 32.41 | 32.63 | 31.52 | 31.68 | 31.72 | 31.96 | 32.12 | 31.83 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.5212 | 1.5298 | 1.5354 | 1.5021 | 1.5074 | 1.5107 | 1.5140 | 1.5191 | 1.5130 |
| eval/lm/dolma_stack-validation/PPL | lower | 4.5775 | 4.6174 | 4.6430 | 4.4913 | 4.5150 | 4.5301 | 4.5449 | 4.5679 | 4.5405 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.8520 | 2.8600 | 2.8585 | 2.8245 | 2.8344 | 2.8297 | 2.8398 | 2.8440 | 2.8423 |
| eval/lm/dolma_wiki-validation/PPL | lower | 17.32 | 17.46 | 17.44 | 16.85 | 17.02 | 16.94 | 17.11 | 17.18 | 17.16 |
| eval/lm/ice-validation/CE loss | lower | 3.2854 | 3.2682 | 3.2975 | 3.2333 | 3.2438 | 3.2462 | 3.2547 | 3.2636 | 3.2481 |
| eval/lm/ice-validation/PPL | lower | 26.72 | 26.26 | 27.05 | 25.36 | 25.63 | 25.69 | 25.91 | 26.14 | 25.74 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 3.2913 | 3.2945 | 3.3009 | 3.2623 | 3.2646 | 3.2666 | 3.2712 | 3.2708 | 3.2803 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 26.88 | 26.96 | 27.14 | 26.11 | 26.17 | 26.22 | 26.34 | 26.33 | 26.58 |
| eval/lm/pile-validation/CE loss | lower | 2.4438 | 2.4543 | 2.4594 | 2.4231 | 2.4296 | 2.4314 | 2.4358 | 2.4413 | 2.4397 |
| eval/lm/pile-validation/PPL | lower | 11.52 | 11.64 | 11.70 | 11.28 | 11.35 | 11.37 | 11.42 | 11.49 | 11.47 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.8351 | 2.8432 | 2.8592 | 2.8037 | 2.8082 | 2.8124 | 2.8275 | 2.8277 | 2.8244 |
| eval/lm/wikitext_103-validation/PPL | lower | 17.03 | 17.17 | 17.45 | 16.51 | 16.58 | 16.65 | 16.90 | 16.91 | 16.85 |
| throughput/in-loop eval batches | see metric | 419.0 | 419.0 | 419.0 | 826.0 | 826.0 | 826.0 | 3281.0 | 3281.0 | 3281.0 |
| throughput/in-loop eval time (s) | see metric | 106.9 | 108.8 | 111.6 | 121.9 | 137.3 | 119.5 | 393.5 | 410.7 | 404.0 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| eg-275m-cx4-eg24e2k-lr1.6e-3-r1<br>`eq0vqyj9` | finished | original | 16094593024.0 | 30698 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/eq0vqyj9) |
| eg-275m-cx4-eg24e2k-lr3.2e-3-r1<br>`mrpoyk8n` | finished | original | 16094593024.0 | 30698 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/mrpoyk8n) |
| eg-275m-cx4-eg24e2k-lr8e-4-r1<br>`5talvqd1` | finished | original | 16094593024.0 | 30698 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5talvqd1) |
| eg-275m-cx4-eg96e8k-lr1.6e-3-r1<br>`gsqree2x` | finished | original | 16143351808.0 | 30791 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gsqree2x) |
| eg-275m-cx4-eg96e8k-lr3.2e-3-r1<br>`589cgpj0` | finished | original | 16143351808.0 | 30791 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/589cgpj0) |
| eg-275m-cx4-eg96e8k-lr8e-4-r1<br>`0vr98te9` | finished | original | 16143351808.0 | 30791 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0vr98te9) |
| 275m-cx4-b512k-lr1.5e-3<br>`vtn70hed` | finished | gpu4-ep1mb16 | 16110845952.0 | 30729 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vtn70hed) |
| 275m-cx4-b512k-lr1e-3<br>`m053n1rr` | finished | gpu4-ep1mb16 | 16110845952.0 | 30729 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/m053n1rr) |
| 275m-cx4-b512k-lr2.5e-3<br>`f5csk4pn` | finished | gpu4-ep1mb16 | 16110845952.0 | 30729 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/f5csk4pn) |

## 275m Cx8

| metric | direction | eg-275m-cx8-eg24e2k-lr1.6e-3-r1<br>`ff9vq2dh` | eg-275m-cx8-eg24e2k-lr3.2e-3-r1<br>`1d8wo3d7` | eg-275m-cx8-eg24e2k-lr8e-4-r1<br>`zu643dvj` | eg-275m-cx8-eg96e8k-lr1.6e-3-r1<br>`zyrglc9e` | eg-275m-cx8-eg96e8k-lr3.2e-3-r1<br>`f9cuox21` | eg-275m-cx8-eg96e8k-lr8e-4-r1<br>`djnuz8yq` | 275m-cx8-b768k-lr1.6e-2-sentinel<br>`j9u54a62` | 275m-cx8-b768k-lr1.6e-3-r2<br>`nitwy74p` | 275m-cx8-b768k-lr2e-4-r2<br>`b0d0494x` | 275m-cx8-b768k-lr3.2e-3-r3<br>`zifpridw` | 275m-cx8-b768k-lr4e-4-r2<br>`nc9kkdrq` | 275m-cx8-b768k-lr6.4e-3-r3<br>`f16hvrea` | 275m-cx8-b768k-lr6e-4-r2<br>`ro6cs2pj` | 275m-cx8-b768k-lr8e-4-r2<br>`pkhkyvt4` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.95730 | 0.95927 | 0.94829 | 0.93150 | 0.94622 | 0.93797 | 1.0092 | 0.94626 | 0.99471 | 0.94331 | 0.96025 | 0.95480 | 0.96330 | 0.94424 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 1.0459 | 1.0502 | 1.0379 | 1.0180 | 1.0339 | 1.0261 | 1.1062 | 1.0347 | 1.0872 | 1.0323 | 1.0518 | 1.0445 | 1.0559 | 1.0333 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0504 | 1.0594 | 1.0321 | 1.0243 | 1.0441 | 1.0207 | 1.0137 | 1.0309 | 1.0346 | 1.0174 | 1.0172 | 1.0238 | 1.0266 | 1.0215 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.1008 | 2.1187 | 2.0641 | 2.0486 | 2.0882 | 2.0414 | 2.0274 | 2.0618 | 2.0691 | 2.0349 | 2.0345 | 2.0476 | 2.0532 | 2.0430 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.72811 | 0.73436 | 0.71546 | 0.71013 | 0.72373 | 0.70755 | 0.70275 | 0.71462 | 0.71715 | 0.70538 | 0.70521 | 0.70976 | 0.71170 | 0.70809 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4562 | 1.4687 | 1.4309 | 1.4203 | 1.4475 | 1.4151 | 1.4055 | 1.4292 | 1.4343 | 1.4108 | 1.4104 | 1.4195 | 1.4234 | 1.4162 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.22526 | 0.22696 | 0.23208 | 0.22782 | 0.23379 | 0.25085 | 0.25597 | 0.25597 | 0.24317 | 0.23294 | 0.25000 | 0.25000 | 0.22782 | 0.25341 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.22526 | 0.22696 | 0.23208 | 0.22782 | 0.23379 | 0.25085 | 0.25597 | 0.25597 | 0.24317 | 0.23294 | 0.25000 | 0.25000 | 0.22782 | 0.25341 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.4403 | -1.4611 | -1.4175 | -1.4092 | -1.4409 | -1.4060 | -1.3925 | -1.4221 | -1.4150 | -1.3984 | -1.3998 | -1.4049 | -1.4107 | -1.4030 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.4403 | -1.4611 | -1.4175 | -1.4092 | -1.4409 | -1.4060 | -1.3925 | -1.4221 | -1.4150 | -1.3984 | -1.3998 | -1.4049 | -1.4107 | -1.4030 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.24626 | 0.24676 | 0.24781 | 0.24900 | 0.24613 | 0.25052 | 0.25021 | 0.24902 | 0.24873 | 0.24904 | 0.24988 | 0.25082 | 0.24799 | 0.25037 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.24626 | 0.24676 | 0.24781 | 0.24900 | 0.24613 | 0.25052 | 0.25021 | 0.24902 | 0.24873 | 0.24904 | 0.24988 | 0.25082 | 0.24799 | 0.25037 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.75731 | 0.75780 | 0.75174 | 0.72253 | 0.74346 | 0.74384 | 0.82646 | 0.75581 | 0.78867 | 0.75714 | 0.77072 | 0.76478 | 0.76592 | 0.74679 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.82535 | 0.82629 | 0.81847 | 0.78652 | 0.80882 | 0.81008 | 0.90072 | 0.82276 | 0.85947 | 0.82471 | 0.83951 | 0.83308 | 0.83400 | 0.81296 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0407 | 1.0567 | 1.0385 | 1.0214 | 1.0350 | 1.0143 | 1.0203 | 1.0192 | 1.0451 | 1.0147 | 1.0222 | 1.0272 | 1.0277 | 1.0185 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0814 | 2.1134 | 2.0770 | 2.0427 | 2.0699 | 2.0285 | 2.0407 | 2.0384 | 2.0902 | 2.0295 | 2.0444 | 2.0544 | 2.0554 | 2.0369 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.72138 | 0.73247 | 0.71989 | 0.70803 | 0.71744 | 0.70314 | 0.70734 | 0.70657 | 0.72444 | 0.70347 | 0.70859 | 0.71205 | 0.71242 | 0.70602 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4428 | 1.4649 | 1.4398 | 1.4161 | 1.4349 | 1.4063 | 1.4147 | 1.4131 | 1.4489 | 1.4069 | 1.4172 | 1.4241 | 1.4248 | 1.4120 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.24916 | 0.25084 | 0.24747 | 0.24663 | 0.25168 | 0.24790 | 0.24790 | 0.25758 | 0.24411 | 0.23527 | 0.25253 | 0.24874 | 0.24200 | 0.23948 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.24916 | 0.25084 | 0.24747 | 0.24663 | 0.25168 | 0.24790 | 0.24790 | 0.25758 | 0.24411 | 0.23527 | 0.25253 | 0.24874 | 0.24200 | 0.23948 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.4272 | -1.4572 | -1.4243 | -1.4061 | -1.4269 | -1.3982 | -1.4015 | -1.4060 | -1.4325 | -1.3949 | -1.4088 | -1.4117 | -1.4098 | -1.3980 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.4272 | -1.4572 | -1.4243 | -1.4061 | -1.4269 | -1.3982 | -1.4015 | -1.4060 | -1.4325 | -1.3949 | -1.4088 | -1.4117 | -1.4098 | -1.3980 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.24992 | 0.25005 | 0.24848 | 0.24911 | 0.24921 | 0.25089 | 0.25082 | 0.24958 | 0.24993 | 0.24990 | 0.24982 | 0.25046 | 0.24879 | 0.25027 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.24992 | 0.25005 | 0.24848 | 0.24911 | 0.24921 | 0.25089 | 0.25082 | 0.24958 | 0.24993 | 0.24990 | 0.24982 | 0.25046 | 0.24879 | 0.25027 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 1.7717 | 1.6623 | 1.7066 | 1.6166 | 1.6486 | 1.7309 | 1.9670 | 1.7016 | 1.8853 | 1.6220 | 1.7771 | 1.8050 | 1.7128 | 1.6715 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 2.8383 | 2.6730 | 2.7347 | 2.5914 | 2.6307 | 2.7906 | 3.1381 | 2.7533 | 3.0140 | 2.6182 | 2.8333 | 2.9096 | 2.7489 | 2.6717 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 1.2283 | 1.1521 | 1.1829 | 1.1205 | 1.1427 | 1.1997 | 1.3633 | 1.1794 | 1.3069 | 1.1241 | 1.2317 | 1.2511 | 1.1872 | 1.1586 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 1.9675 | 1.8527 | 1.8955 | 1.7965 | 1.8235 | 1.9343 | 2.1750 | 1.9084 | 2.0892 | 1.8147 | 1.9639 | 2.0167 | 1.9053 | 1.8518 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.28462 | 0.28367 | 0.28844 | 0.34193 | 0.31614 | 0.27412 | 0.18052 | 0.29799 | 0.23687 | 0.30372 | 0.27794 | 0.26457 | 0.28653 | 0.34002 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.28462 | 0.28367 | 0.28844 | 0.34193 | 0.31614 | 0.27412 | 0.18052 | 0.29799 | 0.23687 | 0.30372 | 0.27794 | 0.26457 | 0.28653 | 0.34002 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -2.2968 | -2.1226 | -2.1875 | -2.0786 | -2.1547 | -2.2026 | -2.5157 | -2.1517 | -2.4266 | -2.0467 | -2.3091 | -2.2039 | -2.2058 | -2.2071 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -2.2968 | -2.1226 | -2.1875 | -2.0786 | -2.1547 | -2.2026 | -2.5157 | -2.1517 | -2.4266 | -2.0467 | -2.3091 | -2.2039 | -2.2058 | -2.2071 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.21156 | 0.23003 | 0.22204 | 0.28411 | 0.26530 | 0.23402 | 0.14119 | 0.25258 | 0.19368 | 0.26030 | 0.21056 | 0.18704 | 0.22452 | 0.24660 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.21156 | 0.23003 | 0.22204 | 0.28411 | 0.26530 | 0.23402 | 0.14119 | 0.25258 | 0.19368 | 0.26030 | 0.21056 | 0.18704 | 0.22452 | 0.24660 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.44630 | 0.48078 | 0.46730 | 0.43749 | 0.43106 | 0.40448 | 0.51852 | 0.44433 | 0.45198 | 0.44679 | 0.44618 | 0.46739 | 0.45148 | 0.44065 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.48729 | 0.52327 | 0.50886 | 0.47687 | 0.46888 | 0.43993 | 0.56639 | 0.48536 | 0.49149 | 0.48647 | 0.48550 | 0.50809 | 0.49070 | 0.47982 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.30934 | 0.33326 | 0.32393 | 0.30326 | 0.29880 | 0.28038 | 0.35940 | 0.30798 | 0.31330 | 0.30975 | 0.30925 | 0.32399 | 0.31293 | 0.30540 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.33774 | 0.36274 | 0.35270 | 0.33052 | 0.32503 | 0.30489 | 0.39259 | 0.33645 | 0.34068 | 0.33716 | 0.33650 | 0.35218 | 0.34013 | 0.33263 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.49901 | 0.48913 | 0.49012 | 0.49802 | 0.48221 | 0.50692 | 0.45850 | 0.47431 | 0.48123 | 0.47332 | 0.49407 | 0.46640 | 0.47036 | 0.47925 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.49901 | 0.48913 | 0.49012 | 0.49802 | 0.48221 | 0.50692 | 0.45850 | 0.47431 | 0.48123 | 0.47332 | 0.49407 | 0.46640 | 0.47036 | 0.47925 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -2.5595 | -2.6686 | -2.6867 | -2.3784 | -2.5945 | -2.3517 | -3.0729 | -2.5636 | -2.7665 | -2.6303 | -2.6205 | -2.7788 | -2.6909 | -2.6840 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -2.5595 | -2.6686 | -2.6867 | -2.3784 | -2.5945 | -2.3517 | -3.0729 | -2.5636 | -2.7665 | -2.6303 | -2.6205 | -2.7788 | -2.6909 | -2.6840 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.47402 | 0.46556 | 0.46120 | 0.49181 | 0.47347 | 0.48659 | 0.44143 | 0.46493 | 0.45153 | 0.45960 | 0.46774 | 0.45662 | 0.45423 | 0.46177 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.47402 | 0.46556 | 0.46120 | 0.49181 | 0.47347 | 0.48659 | 0.44143 | 0.46493 | 0.45153 | 0.45960 | 0.46774 | 0.45662 | 0.45423 | 0.46177 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.58011 | 0.50657 | 0.54656 | 0.50689 | 0.50582 | 0.46560 | 0.60660 | 0.58770 | 0.61274 | 0.46131 | 0.54016 | 0.52855 | 0.51880 | 0.57734 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.69646 | 0.60883 | 0.65576 | 0.60969 | 0.60747 | 0.55915 | 0.72872 | 0.70773 | 0.73547 | 0.55389 | 0.64959 | 0.63368 | 0.62427 | 0.69292 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.40227 | 0.35122 | 0.37894 | 0.35158 | 0.35073 | 0.32287 | 0.42054 | 0.40750 | 0.42485 | 0.31984 | 0.37457 | 0.36649 | 0.35969 | 0.40028 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.48293 | 0.42221 | 0.45462 | 0.42286 | 0.42123 | 0.38771 | 0.50528 | 0.49073 | 0.51001 | 0.38410 | 0.45044 | 0.43941 | 0.43277 | 0.48046 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.77917 | 0.77821 | 0.79171 | 0.78110 | 0.78496 | 0.79846 | 0.73674 | 0.79653 | 0.76181 | 0.80039 | 0.78013 | 0.77338 | 0.78399 | 0.80521 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.77917 | 0.77821 | 0.79171 | 0.78110 | 0.78496 | 0.79846 | 0.73674 | 0.79653 | 0.76181 | 0.80039 | 0.78013 | 0.77338 | 0.78399 | 0.80521 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.60896 | -0.56178 | -0.56542 | -0.57632 | -0.56593 | -0.52934 | -0.70376 | -0.58050 | -0.64713 | -0.53909 | -0.60191 | -0.57074 | -0.60552 | -0.57862 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.60896 | -0.56178 | -0.56542 | -0.57632 | -0.56593 | -0.52934 | -0.70376 | -0.58050 | -0.64713 | -0.53909 | -0.60191 | -0.57074 | -0.60552 | -0.57862 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.67180 | 0.68484 | 0.68124 | 0.68856 | 0.70036 | 0.70000 | 0.63631 | 0.68569 | 0.65078 | 0.70833 | 0.67090 | 0.69179 | 0.67687 | 0.68568 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.67180 | 0.68484 | 0.68124 | 0.68856 | 0.70036 | 0.70000 | 0.63631 | 0.68569 | 0.65078 | 0.70833 | 0.67090 | 0.69179 | 0.67687 | 0.68568 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.26755 | 0.24250 | 0.30422 | 0.29420 | 0.27499 | 0.26457 | 0.30698 | 0.26630 | 0.27520 | 0.28742 | 0.26713 | 0.29216 | 0.25560 | 0.26223 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.27656 | 0.25065 | 0.31437 | 0.30396 | 0.28426 | 0.27342 | 0.31712 | 0.27524 | 0.28441 | 0.29712 | 0.27609 | 0.30200 | 0.26426 | 0.27115 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.18549 | 0.16810 | 0.21090 | 0.20396 | 0.19064 | 0.18340 | 0.21278 | 0.18461 | 0.19077 | 0.19924 | 0.18518 | 0.20253 | 0.17719 | 0.18180 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.19170 | 0.17375 | 0.21792 | 0.21071 | 0.19706 | 0.18954 | 0.21984 | 0.19082 | 0.19717 | 0.20595 | 0.19139 | 0.20936 | 0.18317 | 0.18796 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.82916 | 0.81932 | 0.82021 | 0.88819 | 0.86136 | 0.83900 | 0.83363 | 0.83989 | 0.81664 | 0.83005 | 0.82379 | 0.83989 | 0.80948 | 0.85510 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.82916 | 0.81932 | 0.82021 | 0.88819 | 0.86136 | 0.83900 | 0.83363 | 0.83989 | 0.81664 | 0.83005 | 0.82379 | 0.83989 | 0.80948 | 0.85510 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.44574 | -0.47545 | -0.42642 | -0.32670 | -0.37277 | -0.48281 | -0.48891 | -0.39232 | -0.47875 | -0.43203 | -0.48617 | -0.41535 | -0.47776 | -0.38419 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.44574 | -0.47545 | -0.42642 | -0.32670 | -0.37277 | -0.48281 | -0.48891 | -0.39232 | -0.47875 | -0.43203 | -0.48617 | -0.41535 | -0.47776 | -0.38419 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.82501 | 0.82261 | 0.81115 | 0.85577 | 0.84216 | 0.81430 | 0.82231 | 0.83884 | 0.80893 | 0.82710 | 0.81513 | 0.81999 | 0.80809 | 0.84152 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.82501 | 0.82261 | 0.81115 | 0.85577 | 0.84216 | 0.81430 | 0.82231 | 0.83884 | 0.80893 | 0.82710 | 0.81513 | 0.81999 | 0.80809 | 0.84152 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 0.98160 | 1.0925 | 1.0814 | 0.91621 | 1.0393 | 0.97302 | 1.0274 | 0.94351 | 1.0993 | 0.98477 | 1.0444 | 0.94785 | 1.0128 | 1.0572 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.5702 | 1.7536 | 1.7256 | 1.4457 | 1.6738 | 1.5683 | 1.6646 | 1.5434 | 1.7575 | 1.5949 | 1.6890 | 1.5203 | 1.6172 | 1.7098 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.71454 | 0.79284 | 0.78766 | 0.67048 | 0.75806 | 0.71014 | 0.75093 | 0.68394 | 0.79681 | 0.71623 | 0.75635 | 0.68848 | 0.73977 | 0.76674 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.1761 | 1.3071 | 1.2946 | 1.0934 | 1.2566 | 1.1781 | 1.2535 | 1.1468 | 1.3082 | 1.1914 | 1.2536 | 1.1346 | 1.2178 | 1.2720 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.62921 | 0.58989 | 0.62921 | 0.65543 | 0.64981 | 0.62734 | 0.61798 | 0.64794 | 0.62360 | 0.62172 | 0.64045 | 0.65169 | 0.65169 | 0.62360 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.62921 | 0.58989 | 0.62921 | 0.65543 | 0.64981 | 0.62734 | 0.61798 | 0.64794 | 0.62360 | 0.62172 | 0.64045 | 0.65169 | 0.65169 | 0.62360 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.93059 | -0.97691 | -0.93532 | -0.92287 | -0.89350 | -0.88629 | -0.98812 | -0.88617 | -0.95788 | -0.88940 | -0.92420 | -0.91507 | -0.95881 | -0.94329 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.93059 | -0.97691 | -0.93532 | -0.92287 | -0.89350 | -0.88629 | -0.98812 | -0.88617 | -0.95788 | -0.88940 | -0.92420 | -0.91507 | -0.95881 | -0.94329 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.56422 | 0.53703 | 0.55461 | 0.58533 | 0.57226 | 0.56437 | 0.54874 | 0.57523 | 0.54351 | 0.57240 | 0.56852 | 0.57708 | 0.55711 | 0.55368 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.56422 | 0.53703 | 0.55461 | 0.58533 | 0.57226 | 0.56437 | 0.54874 | 0.57523 | 0.54351 | 0.57240 | 0.56852 | 0.57708 | 0.55711 | 0.55368 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.9567 | 1.9029 | 1.9094 | 1.8660 | 1.8842 | 1.8951 | 2.0739 | 1.8180 | 2.1366 | 1.8365 | 2.1036 | 1.8897 | 2.0303 | 1.9809 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.6935 | 2.6192 | 2.6256 | 2.5736 | 2.6003 | 2.6189 | 2.8747 | 2.4957 | 2.9297 | 2.5442 | 2.8867 | 2.6537 | 2.7921 | 2.7169 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.3560 | 1.3191 | 1.3235 | 1.2934 | 1.3060 | 1.3137 | 1.4375 | 1.2602 | 1.4810 | 1.2730 | 1.4580 | 1.3098 | 1.4073 | 1.3731 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.8669 | 1.8156 | 1.8199 | 1.7837 | 1.8022 | 1.8153 | 1.9926 | 1.7298 | 2.0307 | 1.7635 | 2.0010 | 1.8395 | 1.9354 | 1.8833 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.20919 | 0.22642 | 0.22806 | 0.21329 | 0.20755 | 0.19934 | 0.23216 | 0.23544 | 0.21903 | 0.21411 | 0.22477 | 0.22559 | 0.21247 | 0.22888 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.20919 | 0.22642 | 0.22806 | 0.21329 | 0.20755 | 0.19934 | 0.23216 | 0.23544 | 0.21903 | 0.21411 | 0.22477 | 0.22559 | 0.21247 | 0.22888 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -4.4065 | -4.2427 | -4.2900 | -4.1887 | -4.3955 | -4.4153 | -4.4851 | -4.3122 | -4.4952 | -4.3096 | -4.3921 | -4.2764 | -4.2146 | -4.5062 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -4.4065 | -4.2427 | -4.2900 | -4.1887 | -4.3955 | -4.4153 | -4.4851 | -4.3122 | -4.4952 | -4.3096 | -4.3921 | -4.2764 | -4.2146 | -4.5062 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.23132 | 0.23566 | 0.24141 | 0.24264 | 0.23305 | 0.22999 | 0.24893 | 0.24599 | 0.23510 | 0.23372 | 0.24328 | 0.24654 | 0.23631 | 0.24261 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.23132 | 0.23566 | 0.24141 | 0.24264 | 0.23305 | 0.22999 | 0.24893 | 0.24599 | 0.23510 | 0.23372 | 0.24328 | 0.24654 | 0.23631 | 0.24261 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.49850 | 0.48960 | 0.49919 | 0.48387 | 0.48966 | 0.49014 | 0.55413 | 0.48388 | 0.51582 | 0.49472 | 0.50035 | 0.49618 | 0.50631 | 0.48793 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.50511 | 0.49586 | 0.50567 | 0.49019 | 0.49619 | 0.49664 | 0.56119 | 0.49004 | 0.52257 | 0.50120 | 0.50696 | 0.50259 | 0.51312 | 0.49447 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.70400 | 0.70027 | 0.69356 | 0.68838 | 0.68529 | 0.70515 | 0.75624 | 0.69527 | 0.71254 | 0.69375 | 0.70248 | 0.70593 | 0.69128 | 0.69832 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.71028 | 0.70628 | 0.69970 | 0.69437 | 0.69113 | 0.71121 | 0.76277 | 0.70137 | 0.71883 | 0.69970 | 0.70852 | 0.71197 | 0.69724 | 0.70421 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.3012 | 2.2693 | 2.2378 | 2.6000 | 2.4353 | 2.2998 | 2.8368 | 2.3488 | 2.5730 | 2.4209 | 2.4077 | 2.0868 | 2.7871 | 2.6943 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 4.6024 | 4.5387 | 4.4755 | 5.1999 | 4.8706 | 4.5996 | 5.6737 | 4.6976 | 5.1459 | 4.8419 | 4.8155 | 4.1737 | 5.5741 | 5.3885 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.5954 | 1.5732 | 1.5511 | 1.8026 | 1.6876 | 1.5941 | 1.9660 | 1.6279 | 1.7835 | 1.6786 | 1.6696 | 1.4468 | 1.9311 | 1.8678 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.1908 | 3.1465 | 3.1023 | 3.6052 | 3.3752 | 3.1883 | 3.9320 | 3.2558 | 3.5670 | 3.3571 | 3.3393 | 2.8936 | 3.8623 | 3.7355 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.10000 | 0.10000 | 0.09000 | 0.10000 | 0.17000 | 0.10000 | 0.10000 | 0.10000 | 0.05000 | 0.08000 | 0.11000 | 0.14000 | 0.10000 | 0.10000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.10000 | 0.10000 | 0.09000 | 0.10000 | 0.17000 | 0.10000 | 0.10000 | 0.10000 | 0.05000 | 0.08000 | 0.11000 | 0.14000 | 0.10000 | 0.10000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -3.1698 | -3.1417 | -3.0903 | -3.5979 | -3.3691 | -3.1773 | -3.9244 | -3.2502 | -3.5485 | -3.3441 | -3.3301 | -2.8834 | -3.8531 | -3.7301 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -3.1698 | -3.1417 | -3.0903 | -3.5979 | -3.3691 | -3.1773 | -3.9244 | -3.2502 | -3.5485 | -3.3441 | -3.3301 | -2.8834 | -3.8531 | -3.7301 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09413 | 0.09374 | 0.09581 | 0.09558 | 0.10561 | 0.09622 | 0.09555 | 0.09770 | 0.09370 | 0.09562 | 0.09836 | 0.10309 | 0.09795 | 0.10278 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09413 | 0.09374 | 0.09581 | 0.09558 | 0.10561 | 0.09622 | 0.09555 | 0.09770 | 0.09370 | 0.09562 | 0.09836 | 0.10309 | 0.09795 | 0.10278 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.84187 | 0.85106 | 0.84573 | 0.83979 | 0.84321 | 0.84074 | 0.87308 | 0.83996 | 0.85821 | 0.84680 | 0.84268 | 0.85825 | 0.84567 | 0.84269 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.85119 | 0.86050 | 0.85487 | 0.84907 | 0.85252 | 0.84989 | 0.88270 | 0.84919 | 0.86765 | 0.85618 | 0.85197 | 0.86763 | 0.85505 | 0.85197 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.76289 | 0.75931 | 0.75944 | 0.74620 | 0.74683 | 0.74860 | 0.81730 | 0.75290 | 0.78390 | 0.75795 | 0.76596 | 0.77092 | 0.76719 | 0.75559 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.76558 | 0.76182 | 0.76181 | 0.74870 | 0.74928 | 0.75111 | 0.81999 | 0.75546 | 0.78634 | 0.76047 | 0.76866 | 0.77351 | 0.76957 | 0.75810 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.78884 | 0.79238 | 0.79069 | 0.77011 | 0.78196 | 0.77568 | 0.83392 | 0.77596 | 0.80703 | 0.78724 | 0.79026 | 0.79170 | 0.79818 | 0.78598 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.83011 | 0.83331 | 0.83157 | 0.81006 | 0.82261 | 0.81609 | 0.87812 | 0.81634 | 0.84970 | 0.82809 | 0.83202 | 0.83375 | 0.84073 | 0.82726 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0436 | 1.0560 | 1.0304 | 1.0270 | 1.0322 | 1.0185 | 1.0228 | 1.0257 | 1.0238 | 1.0251 | 1.0222 | 1.0292 | 1.0260 | 1.0300 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0871 | 2.1119 | 2.0609 | 2.0539 | 2.0643 | 2.0370 | 2.0455 | 2.0514 | 2.0477 | 2.0502 | 2.0444 | 2.0584 | 2.0520 | 2.0601 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.72336 | 0.73199 | 0.71433 | 0.71193 | 0.71547 | 0.70606 | 0.70901 | 0.71098 | 0.70972 | 0.71063 | 0.70862 | 0.71346 | 0.71128 | 0.71404 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4467 | 1.4640 | 1.4287 | 1.4239 | 1.4309 | 1.4121 | 1.4180 | 1.4220 | 1.4194 | 1.4213 | 1.4172 | 1.4269 | 1.4226 | 1.4281 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.23741 | 0.24230 | 0.23571 | 0.24655 | 0.24421 | 0.24123 | 0.24442 | 0.23932 | 0.25335 | 0.23847 | 0.23592 | 0.25462 | 0.23911 | 0.23741 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.23741 | 0.24230 | 0.23571 | 0.24655 | 0.24421 | 0.24123 | 0.24442 | 0.23932 | 0.25335 | 0.23847 | 0.23592 | 0.25462 | 0.23911 | 0.23741 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3999 | -1.4045 | -1.3947 | -1.3945 | -1.3969 | -1.3883 | -1.3903 | -1.3941 | -1.3896 | -1.3919 | -1.3913 | -1.3907 | -1.3935 | -1.3943 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.4337 | -1.4532 | -1.4163 | -1.4135 | -1.4216 | -1.4008 | -1.4037 | -1.4120 | -1.4028 | -1.4046 | -1.4041 | -1.4086 | -1.4094 | -1.4175 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.24905 | 0.24910 | 0.24950 | 0.24933 | 0.24915 | 0.25080 | 0.25020 | 0.24932 | 0.25042 | 0.24947 | 0.24969 | 0.25056 | 0.24922 | 0.24984 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.24805 | 0.24805 | 0.24896 | 0.24869 | 0.24831 | 0.25148 | 0.25038 | 0.24864 | 0.25082 | 0.24893 | 0.24930 | 0.25104 | 0.24835 | 0.24951 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 1.1127 | 1.1062 | 1.0968 | 1.0701 | 1.0797 | 1.0812 | 1.1570 | 1.0943 | 1.1435 | 1.0921 | 1.1187 | 1.1119 | 1.1208 | 1.0949 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.2400 | 1.2316 | 1.2214 | 1.1919 | 1.2024 | 1.2048 | 1.2881 | 1.2194 | 1.2742 | 1.2166 | 1.2460 | 1.2392 | 1.2490 | 1.2203 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0380 | 1.0510 | 1.0315 | 1.0182 | 1.0329 | 1.0196 | 1.0168 | 1.0141 | 1.0281 | 1.0145 | 1.0193 | 1.0369 | 1.0240 | 1.0221 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0761 | 2.1021 | 2.0629 | 2.0365 | 2.0659 | 2.0391 | 2.0337 | 2.0282 | 2.0563 | 2.0291 | 2.0385 | 2.0738 | 2.0481 | 2.0442 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.71957 | 0.72856 | 0.71498 | 0.70590 | 0.71600 | 0.70682 | 0.70492 | 0.70301 | 0.71267 | 0.70329 | 0.70658 | 0.71872 | 0.70986 | 0.70856 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4391 | 1.4571 | 1.4300 | 1.4118 | 1.4320 | 1.4136 | 1.4098 | 1.4060 | 1.4253 | 1.4066 | 1.4132 | 1.4374 | 1.4197 | 1.4171 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24830 | 0.23874 | 0.25015 | 0.24645 | 0.25262 | 0.24707 | 0.26249 | 0.27483 | 0.27514 | 0.25725 | 0.25046 | 0.24923 | 0.24830 | 0.25139 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24830 | 0.23874 | 0.25015 | 0.24645 | 0.25262 | 0.24707 | 0.26249 | 0.27483 | 0.27514 | 0.25725 | 0.25046 | 0.24923 | 0.24830 | 0.25139 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3940 | -1.4007 | -1.3937 | -1.3889 | -1.3957 | -1.3900 | -1.3835 | -1.3848 | -1.3858 | -1.3870 | -1.3894 | -1.3918 | -1.3898 | -1.3893 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.4262 | -1.4474 | -1.4172 | -1.4011 | -1.4238 | -1.4042 | -1.3896 | -1.3978 | -1.4057 | -1.3937 | -1.4030 | -1.4149 | -1.4045 | -1.4046 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25108 | 0.25010 | 0.25006 | 0.25048 | 0.24996 | 0.25031 | 0.25192 | 0.25222 | 0.25275 | 0.25058 | 0.25052 | 0.25083 | 0.25039 | 0.25080 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25194 | 0.24963 | 0.24996 | 0.25084 | 0.24990 | 0.25045 | 0.25396 | 0.25443 | 0.25542 | 0.25115 | 0.25093 | 0.25137 | 0.25058 | 0.25152 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.94441 | 0.93833 | 0.94498 | 0.91352 | 0.93158 | 0.92472 | 0.98508 | 0.92799 | 0.96812 | 0.93577 | 0.94415 | 0.94465 | 0.94110 | 0.92566 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 1.0097 | 1.0025 | 1.0109 | 0.97661 | 0.99565 | 0.98791 | 1.0525 | 0.99124 | 1.0346 | 1.0003 | 1.0086 | 1.0096 | 1.0063 | 0.98901 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0693 | 1.0894 | 1.0400 | 1.0224 | 1.0410 | 1.0253 | 1.0142 | 1.0279 | 1.0354 | 1.0178 | 1.0145 | 1.0466 | 1.0216 | 1.0428 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.1385 | 2.1788 | 2.0799 | 2.0448 | 2.0820 | 2.0505 | 2.0284 | 2.0558 | 2.0708 | 2.0355 | 2.0290 | 2.0932 | 2.0432 | 2.0856 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.74115 | 0.75511 | 0.72093 | 0.70876 | 0.72162 | 0.71073 | 0.70309 | 0.71252 | 0.71772 | 0.70551 | 0.70331 | 0.72546 | 0.70823 | 0.72291 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4823 | 1.5102 | 1.4419 | 1.4175 | 1.4432 | 1.4215 | 1.4062 | 1.4250 | 1.4354 | 1.4110 | 1.4066 | 1.4509 | 1.4165 | 1.4458 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.21807 | 0.21709 | 0.21872 | 0.22977 | 0.22652 | 0.23107 | 0.29119 | 0.23367 | 0.24082 | 0.26584 | 0.26812 | 0.23237 | 0.23692 | 0.21677 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.21807 | 0.21709 | 0.21872 | 0.22977 | 0.22652 | 0.23107 | 0.29119 | 0.23367 | 0.24082 | 0.26584 | 0.26812 | 0.23237 | 0.23692 | 0.21677 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.4131 | -1.4240 | -1.3993 | -1.3915 | -1.4008 | -1.3928 | -1.3808 | -1.3932 | -1.3927 | -1.3873 | -1.3853 | -1.4012 | -1.3890 | -1.4025 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.4695 | -1.4981 | -1.4269 | -1.4050 | -1.4333 | -1.4108 | -1.3876 | -1.4150 | -1.4160 | -1.3955 | -1.3920 | -1.4291 | -1.3994 | -1.4342 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.24686 | 0.24494 | 0.24839 | 0.24964 | 0.24846 | 0.24969 | 0.25298 | 0.25008 | 0.25048 | 0.25065 | 0.25125 | 0.24811 | 0.25018 | 0.24790 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.24365 | 0.23992 | 0.24650 | 0.24916 | 0.24670 | 0.24920 | 0.25609 | 0.24999 | 0.25079 | 0.25132 | 0.25256 | 0.24648 | 0.25019 | 0.24582 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.4148 | 1.4224 | 1.4109 | 1.3733 | 1.3891 | 1.3795 | 1.4527 | 1.3813 | 1.4566 | 1.3968 | 1.4154 | 1.4128 | 1.4124 | 1.3902 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.7651 | 1.7752 | 1.7608 | 1.7126 | 1.7332 | 1.7187 | 1.8001 | 1.7212 | 1.8146 | 1.7422 | 1.7653 | 1.7604 | 1.7608 | 1.7343 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0500 | 1.0637 | 1.0350 | 1.0200 | 1.0293 | 1.0199 | 1.0238 | 1.0183 | 1.0270 | 1.0180 | 1.0237 | 1.0453 | 1.0261 | 1.0350 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.1000 | 2.1273 | 2.0699 | 2.0400 | 2.0586 | 2.0398 | 2.0476 | 2.0366 | 2.0541 | 2.0360 | 2.0473 | 2.0906 | 2.0522 | 2.0701 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.72781 | 0.73728 | 0.71746 | 0.70704 | 0.71350 | 0.70696 | 0.70972 | 0.70584 | 0.71195 | 0.70570 | 0.70962 | 0.72457 | 0.71134 | 0.71750 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4556 | 1.4746 | 1.4349 | 1.4141 | 1.4270 | 1.4139 | 1.4194 | 1.4117 | 1.4239 | 1.4114 | 1.4192 | 1.4491 | 1.4227 | 1.4350 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.22565 | 0.22134 | 0.23426 | 0.25878 | 0.25182 | 0.25580 | 0.28694 | 0.28694 | 0.26309 | 0.25812 | 0.24718 | 0.25580 | 0.25116 | 0.24818 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.22565 | 0.22134 | 0.23426 | 0.25878 | 0.25182 | 0.25580 | 0.28694 | 0.28694 | 0.26309 | 0.25812 | 0.24718 | 0.25580 | 0.25116 | 0.24818 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.4031 | -1.4064 | -1.3941 | -1.3888 | -1.3921 | -1.3869 | -1.3804 | -1.3835 | -1.3861 | -1.3852 | -1.3912 | -1.3942 | -1.3889 | -1.3943 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4408 | -1.4565 | -1.4146 | -1.4017 | -1.4134 | -1.4014 | -1.3889 | -1.3993 | -1.4020 | -1.3947 | -1.4067 | -1.4200 | -1.4013 | -1.4183 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.24851 | 0.24848 | 0.24960 | 0.25070 | 0.25050 | 0.25154 | 0.25339 | 0.25317 | 0.25214 | 0.25158 | 0.25012 | 0.25040 | 0.25051 | 0.24999 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.24695 | 0.24660 | 0.24911 | 0.25147 | 0.25106 | 0.25291 | 0.25688 | 0.25640 | 0.25422 | 0.25309 | 0.25026 | 0.25090 | 0.25088 | 0.24991 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.46420 | 0.48121 | 0.47623 | 0.47437 | 0.45178 | 0.46163 | 0.49722 | 0.46505 | 0.49074 | 0.47013 | 0.46216 | 0.49437 | 0.46942 | 0.47180 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.46674 | 0.48410 | 0.47879 | 0.47706 | 0.45426 | 0.46431 | 0.50008 | 0.46781 | 0.49345 | 0.47276 | 0.46479 | 0.49708 | 0.47198 | 0.47454 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.37130 | 0.37335 | 0.36965 | 0.37134 | 0.36731 | 0.36622 | 0.38801 | 0.36426 | 0.36461 | 0.38046 | 0.35851 | 0.38843 | 0.36231 | 0.36953 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.37273 | 0.37484 | 0.37119 | 0.37276 | 0.36887 | 0.36766 | 0.38947 | 0.36572 | 0.36609 | 0.38191 | 0.35986 | 0.38995 | 0.36365 | 0.37096 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.63829 | 0.71995 | 0.62858 | 0.62167 | 0.62383 | 0.60107 | 0.68649 | 0.63902 | 0.64322 | 0.63935 | 0.61988 | 0.67990 | 0.63472 | 0.62133 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.64281 | 0.72517 | 0.63286 | 0.62627 | 0.62823 | 0.60542 | 0.69127 | 0.64335 | 0.64777 | 0.64388 | 0.62417 | 0.68469 | 0.63907 | 0.62568 |
| eval/lm/c4_en-validation/CE loss | lower | 3.1141 | 3.1217 | 3.1218 | 3.0852 | 3.0991 | 3.0948 | 3.2344 | 3.0998 | 3.1636 | 3.1070 | 3.1273 | 3.1412 | 3.1146 | 3.1059 |
| eval/lm/c4_en-validation/PPL | lower | 22.51 | 22.68 | 22.69 | 21.87 | 22.18 | 22.08 | 25.39 | 22.19 | 23.66 | 22.35 | 22.81 | 23.13 | 22.53 | 22.33 |
| eval/lm/dolma_books-validation/CE loss | lower | 3.0352 | 3.0480 | 3.0451 | 3.0092 | 3.0203 | 3.0144 | 3.1876 | 3.0118 | 3.0872 | 3.0253 | 3.0476 | 3.0655 | 3.0330 | 3.0227 |
| eval/lm/dolma_books-validation/PPL | lower | 20.81 | 21.07 | 21.01 | 20.27 | 20.50 | 20.38 | 24.23 | 20.32 | 21.92 | 20.60 | 21.06 | 21.45 | 20.76 | 20.55 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.2476 | 3.2576 | 3.2565 | 3.2215 | 3.2355 | 3.2292 | 3.3661 | 3.2346 | 3.2966 | 3.2407 | 3.2617 | 3.2741 | 3.2473 | 3.2407 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 25.73 | 25.99 | 25.96 | 25.07 | 25.42 | 25.26 | 28.97 | 25.40 | 27.02 | 25.55 | 26.09 | 26.42 | 25.72 | 25.55 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.2876 | 2.2895 | 2.2939 | 2.2627 | 2.2731 | 2.2708 | 2.3877 | 2.2703 | 2.3303 | 2.2785 | 2.2993 | 2.3053 | 2.2872 | 2.2808 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 9.8511 | 9.8697 | 9.9133 | 9.6087 | 9.7091 | 9.6871 | 10.89 | 9.6820 | 10.28 | 9.7619 | 9.9672 | 10.03 | 9.8473 | 9.7841 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.4120 | 3.4157 | 3.4213 | 3.3833 | 3.3975 | 3.3925 | 3.5183 | 3.3982 | 3.4577 | 3.4062 | 3.4231 | 3.4324 | 3.4122 | 3.4030 |
| eval/lm/dolma_reddit-validation/PPL | lower | 30.33 | 30.44 | 30.61 | 29.47 | 29.89 | 29.74 | 33.73 | 29.91 | 31.74 | 30.15 | 30.66 | 30.95 | 30.33 | 30.05 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.4490 | 1.4557 | 1.4543 | 1.4269 | 1.4349 | 1.4303 | 1.5611 | 1.4353 | 1.4935 | 1.4452 | 1.4614 | 1.4750 | 1.4502 | 1.4429 |
| eval/lm/dolma_stack-validation/PPL | lower | 4.2587 | 4.2875 | 4.2814 | 4.1657 | 4.1993 | 4.1798 | 4.7639 | 4.2010 | 4.4525 | 4.2429 | 4.3121 | 4.3712 | 4.2640 | 4.2328 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.7672 | 2.7808 | 2.7759 | 2.7414 | 2.7560 | 2.7486 | 2.9101 | 2.7570 | 2.8196 | 2.7641 | 2.7819 | 2.8057 | 2.7657 | 2.7578 |
| eval/lm/dolma_wiki-validation/PPL | lower | 15.91 | 16.13 | 16.05 | 15.51 | 15.74 | 15.62 | 18.36 | 15.75 | 16.77 | 15.86 | 16.15 | 16.54 | 15.89 | 15.77 |
| eval/lm/ice-validation/CE loss | lower | 3.2121 | 3.2057 | 3.1944 | 3.1725 | 3.1822 | 3.1572 | 3.2949 | 3.1773 | 3.2380 | 3.1814 | 3.2127 | 3.1902 | 3.1929 | 3.1982 |
| eval/lm/ice-validation/PPL | lower | 24.83 | 24.67 | 24.39 | 23.87 | 24.10 | 23.50 | 26.97 | 23.98 | 25.48 | 24.08 | 24.85 | 24.29 | 24.36 | 24.49 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 3.2343 | 3.2315 | 3.2301 | 3.1965 | 3.2051 | 3.1990 | 3.3266 | 3.2040 | 3.2537 | 3.2040 | 3.2305 | 3.2403 | 3.2190 | 3.2112 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 25.39 | 25.32 | 25.28 | 24.45 | 24.66 | 24.51 | 27.84 | 24.63 | 25.88 | 24.63 | 25.29 | 25.54 | 25.00 | 24.81 |
| eval/lm/pile-validation/CE loss | lower | 2.3734 | 2.3787 | 2.3832 | 2.3504 | 2.3593 | 2.3563 | 2.4958 | 2.3593 | 2.4209 | 2.3669 | 2.3867 | 2.4027 | 2.3744 | 2.3687 |
| eval/lm/pile-validation/PPL | lower | 10.73 | 10.79 | 10.84 | 10.49 | 10.58 | 10.55 | 12.13 | 10.58 | 11.26 | 10.66 | 10.88 | 11.05 | 10.74 | 10.68 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.7471 | 2.7447 | 2.7566 | 2.7148 | 2.7205 | 2.7213 | 2.8681 | 2.7269 | 2.8131 | 2.7274 | 2.7708 | 2.7657 | 2.7500 | 2.7393 |
| eval/lm/wikitext_103-validation/PPL | lower | 15.60 | 15.56 | 15.75 | 15.10 | 15.19 | 15.20 | 17.60 | 15.29 | 16.66 | 15.29 | 15.97 | 15.89 | 15.64 | 15.48 |
| throughput/in-loop eval batches | see metric | 828.0 | 828.0 | 828.0 | 828.0 | 828.0 | 828.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 |
| throughput/in-loop eval time (s) | see metric | 66.98 | 63.16 | 117.2 | 64.00 | 65.74 | 62.49 | 409.2 | 396.7 | 412.5 | 399.1 | 392.4 | 393.8 | 401.1 | 398.2 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| eg-275m-cx8-eg24e2k-lr1.6e-3-r1<br>`ff9vq2dh` | finished | original | 32188661760.0 | 40930 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ff9vq2dh) |
| eg-275m-cx8-eg24e2k-lr3.2e-3-r1<br>`1d8wo3d7` | finished | original | 32188661760.0 | 40930 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1d8wo3d7) |
| eg-275m-cx8-eg24e2k-lr8e-4-r1<br>`zu643dvj` | finished | original | 32188661760.0 | 40930 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/zu643dvj) |
| eg-275m-cx8-eg96e8k-lr1.6e-3-r1<br>`zyrglc9e` | finished | original | 32286179328.0 | 41054 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/zyrglc9e) |
| eg-275m-cx8-eg96e8k-lr3.2e-3-r1<br>`f9cuox21` | finished | original | 32286179328.0 | 41054 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/f9cuox21) |
| eg-275m-cx8-eg96e8k-lr8e-4-r1<br>`djnuz8yq` | finished | original | 32286179328.0 | 41054 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/djnuz8yq) |
| 275m-cx8-b768k-lr1.6e-2-sentinel<br>`j9u54a62` | finished | gpu4-ep1mb8 | 32220905472.0 | 40971 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/j9u54a62) |
| 275m-cx8-b768k-lr1.6e-3-r2<br>`nitwy74p` | finished | gpu4-ep1mb8 | 32220905472.0 | 40971 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/nitwy74p) |
| 275m-cx8-b768k-lr2e-4-r2<br>`b0d0494x` | finished | gpu4-ep1mb8 | 32220905472.0 | 40971 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/b0d0494x) |
| 275m-cx8-b768k-lr3.2e-3-r3<br>`zifpridw` | finished | gpu4-ep1mb8 | 32220905472.0 | 40971 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/zifpridw) |
| 275m-cx8-b768k-lr4e-4-r2<br>`nc9kkdrq` | finished | gpu4-ep1mb8 | 32220905472.0 | 40971 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/nc9kkdrq) |
| 275m-cx8-b768k-lr6.4e-3-r3<br>`f16hvrea` | finished | gpu4-ep1mb8 | 32220905472.0 | 40971 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/f16hvrea) |
| 275m-cx8-b768k-lr6e-4-r2<br>`ro6cs2pj` | finished | gpu4-ep1mb8 | 32220905472.0 | 40971 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ro6cs2pj) |
| 275m-cx8-b768k-lr8e-4-r2<br>`pkhkyvt4` | finished | gpu4-ep1mb8 | 32220905472.0 | 40971 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/pkhkyvt4) |

## 480m Cx1

| metric | direction | eg-480m-cx1-eg24e2k-lr9e-4-r1<br>`rcgxm5qv` | eg-480m-cx1-eg96e8k-lr1e-3-r1<br>`nvndg2tr` | int-480m-cx1-intd256e8k-lr1.2e-3-r1<br>`ggbfbedg` | int-480m-cx1-intw256e8k-lr1.2e-3-r1<br>`z4wxvc6h` | q3-480m-cx1-q3am128e8k-lr1.2e-3-r1<br>`u7eje3wc` | q3-480m-cx1-q3td128e8k-lr1.2e-3-r1<br>`pirgmap6` | se-480m-cx1-se0m9-lr1.2e-3-r1<br>`57ae3vzv` | se-480m-cx1-se0m9-lr1.2e-3-r1<br>`w9pmxrz9` | sp-480m-cx1-sp192e4k-lr8e-4-r2<br>`artcqrdt` | sp-480m-cx1-sp96e4k-lr1e-3-r1<br>`4kd8hys6` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.97474 | 0.96853 | 0.93122 | 0.95832 | 0.96174 | 0.95382 | 0.98049 | 0.98036 | 0.97239 | 0.97749 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 1.0671 | 1.0634 | 1.0197 | 1.0476 | 1.0501 | 1.0446 | 1.0755 | 1.0756 | 1.0651 | 1.0700 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0432 | 1.0275 | 1.0499 | 1.0216 | 1.0522 | 1.0723 | 1.0352 | 1.0351 | 1.0425 | 1.0267 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0864 | 2.0550 | 2.0997 | 2.0432 | 2.1044 | 2.1445 | 2.0704 | 2.0703 | 2.0851 | 2.0534 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.72303 | 0.71227 | 0.72768 | 0.70814 | 0.72944 | 0.74323 | 0.71763 | 0.71763 | 0.72263 | 0.71172 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4461 | 1.4245 | 1.4554 | 1.4163 | 1.4589 | 1.4865 | 1.4353 | 1.4353 | 1.4453 | 1.4234 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.23976 | 0.25853 | 0.26280 | 0.24659 | 0.23208 | 0.23208 | 0.22696 | 0.23123 | 0.23549 | 0.25000 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.23976 | 0.25853 | 0.26280 | 0.24659 | 0.23208 | 0.23208 | 0.22696 | 0.23123 | 0.23549 | 0.25000 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.4312 | -1.4090 | -1.4355 | -1.4005 | -1.4488 | -1.4771 | -1.4157 | -1.4157 | -1.4215 | -1.4025 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.4312 | -1.4090 | -1.4355 | -1.4005 | -1.4488 | -1.4771 | -1.4157 | -1.4157 | -1.4215 | -1.4025 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.24990 | 0.25150 | 0.25140 | 0.25099 | 0.24771 | 0.24843 | 0.24959 | 0.24958 | 0.24783 | 0.25129 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.24990 | 0.25150 | 0.25140 | 0.25099 | 0.24771 | 0.24843 | 0.24959 | 0.24958 | 0.24783 | 0.25129 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.78619 | 0.75789 | 0.73046 | 0.77398 | 0.75875 | 0.76906 | 0.79462 | 0.79483 | 0.78411 | 0.78032 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.85744 | 0.82594 | 0.79568 | 0.84332 | 0.82620 | 0.83764 | 0.86604 | 0.86618 | 0.85511 | 0.84978 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0463 | 1.0234 | 1.0363 | 1.0268 | 1.0520 | 1.0618 | 1.0331 | 1.0329 | 1.0293 | 1.0236 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0926 | 2.0468 | 2.0726 | 2.0537 | 2.1040 | 2.1236 | 2.0662 | 2.0657 | 2.0586 | 2.0471 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.72526 | 0.70944 | 0.71830 | 0.71184 | 0.72919 | 0.73591 | 0.71612 | 0.71598 | 0.71351 | 0.70954 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4505 | 1.4189 | 1.4366 | 1.4237 | 1.4584 | 1.4718 | 1.4322 | 1.4320 | 1.4270 | 1.4191 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.25758 | 0.26136 | 0.26810 | 0.26052 | 0.25042 | 0.24327 | 0.25210 | 0.25505 | 0.25000 | 0.25547 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.25758 | 0.26136 | 0.26810 | 0.26052 | 0.25042 | 0.24327 | 0.25210 | 0.25505 | 0.25000 | 0.25547 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.4348 | -1.4045 | -1.4153 | -1.4060 | -1.4480 | -1.4636 | -1.4144 | -1.4141 | -1.4079 | -1.4000 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.4348 | -1.4045 | -1.4153 | -1.4060 | -1.4480 | -1.4636 | -1.4144 | -1.4141 | -1.4079 | -1.4000 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.25220 | 0.25150 | 0.25245 | 0.25092 | 0.25041 | 0.24993 | 0.25066 | 0.25067 | 0.25056 | 0.25035 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.25220 | 0.25150 | 0.25245 | 0.25092 | 0.25041 | 0.24993 | 0.25066 | 0.25067 | 0.25056 | 0.25035 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 2.1866 | 2.0177 | 2.0055 | 1.9541 | 2.0226 | 2.0783 | 2.0997 | 2.1064 | 1.9369 | 2.0727 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 3.4858 | 3.2194 | 3.1963 | 3.1416 | 3.2212 | 3.3362 | 3.3454 | 3.3558 | 3.0935 | 3.3266 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 1.5158 | 1.3986 | 1.3900 | 1.3545 | 1.4019 | 1.4407 | 1.4554 | 1.4602 | 1.3424 | 1.4366 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 2.4162 | 2.2316 | 2.2153 | 2.1777 | 2.2329 | 2.3126 | 2.3191 | 2.3259 | 2.1443 | 2.3057 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.15855 | 0.15091 | 0.17287 | 0.21203 | 0.16905 | 0.16332 | 0.16141 | 0.16428 | 0.17670 | 0.17287 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.15855 | 0.15091 | 0.17287 | 0.21203 | 0.16905 | 0.16332 | 0.16141 | 0.16428 | 0.17670 | 0.17287 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -2.7920 | -2.5234 | -2.5803 | -2.5183 | -2.5567 | -2.5023 | -2.6006 | -2.6094 | -2.5383 | -2.5454 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -2.7920 | -2.5234 | -2.5803 | -2.5183 | -2.5567 | -2.5023 | -2.6006 | -2.6094 | -2.5383 | -2.5454 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.13785 | 0.13263 | 0.15386 | 0.16356 | 0.14547 | 0.13975 | 0.13233 | 0.13207 | 0.16473 | 0.14121 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.13785 | 0.13263 | 0.15386 | 0.16356 | 0.14547 | 0.13975 | 0.13233 | 0.13207 | 0.16473 | 0.14121 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.52517 | 0.51683 | 0.47225 | 0.50336 | 0.48705 | 0.49836 | 0.50970 | 0.50869 | 0.47799 | 0.51081 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.57167 | 0.56370 | 0.51261 | 0.54731 | 0.52928 | 0.54172 | 0.55420 | 0.55306 | 0.52111 | 0.55512 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.36404 | 0.35826 | 0.32731 | 0.34892 | 0.33761 | 0.34545 | 0.35329 | 0.35258 | 0.33131 | 0.35409 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.39625 | 0.39069 | 0.35528 | 0.37937 | 0.36690 | 0.37549 | 0.38417 | 0.38338 | 0.36119 | 0.38479 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.45257 | 0.44368 | 0.47233 | 0.44368 | 0.45652 | 0.44960 | 0.43972 | 0.44466 | 0.45455 | 0.45356 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.45257 | 0.44368 | 0.47233 | 0.44368 | 0.45652 | 0.44960 | 0.43972 | 0.44466 | 0.45455 | 0.45356 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -3.2203 | -3.2441 | -3.0143 | -3.1106 | -3.0335 | -3.1941 | -3.2157 | -3.2128 | -3.1043 | -3.2295 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -3.2203 | -3.2441 | -3.0143 | -3.1106 | -3.0335 | -3.1941 | -3.2157 | -3.2128 | -3.1043 | -3.2295 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.42752 | 0.42592 | 0.45260 | 0.43385 | 0.44825 | 0.43020 | 0.42805 | 0.42838 | 0.43847 | 0.43059 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.42752 | 0.42592 | 0.45260 | 0.43385 | 0.44825 | 0.43020 | 0.42805 | 0.42838 | 0.43847 | 0.43059 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.62998 | 0.52402 | 0.58409 | 0.64501 | 0.61541 | 0.58404 | 0.56772 | 0.56701 | 0.60940 | 0.64712 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.75722 | 0.62855 | 0.70231 | 0.77521 | 0.73926 | 0.70210 | 0.68366 | 0.68254 | 0.73193 | 0.77922 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.43689 | 0.36336 | 0.40505 | 0.44730 | 0.42680 | 0.40495 | 0.39368 | 0.39319 | 0.42262 | 0.44875 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.52507 | 0.43594 | 0.48704 | 0.53753 | 0.51272 | 0.48683 | 0.47408 | 0.47331 | 0.50759 | 0.54030 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.76374 | 0.78881 | 0.80424 | 0.76760 | 0.77531 | 0.77242 | 0.76181 | 0.75796 | 0.77146 | 0.76663 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.76374 | 0.78881 | 0.80424 | 0.76760 | 0.77531 | 0.77242 | 0.76181 | 0.75796 | 0.77146 | 0.76663 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.64951 | -0.59533 | -0.54754 | -0.65065 | -0.62230 | -0.59795 | -0.64277 | -0.64463 | -0.61018 | -0.63812 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.64951 | -0.59533 | -0.54754 | -0.65065 | -0.62230 | -0.59795 | -0.64277 | -0.64463 | -0.61018 | -0.63812 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.65811 | 0.67562 | 0.69775 | 0.66398 | 0.66618 | 0.67206 | 0.65514 | 0.65428 | 0.66645 | 0.65310 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.65811 | 0.67562 | 0.69775 | 0.66398 | 0.66618 | 0.67206 | 0.65514 | 0.65428 | 0.66645 | 0.65310 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.26667 | 0.31617 | 0.32201 | 0.31608 | 0.33148 | 0.31995 | 0.32187 | 0.32219 | 0.31079 | 0.29445 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.27558 | 0.32681 | 0.33272 | 0.32675 | 0.34242 | 0.33070 | 0.33264 | 0.33296 | 0.32132 | 0.30422 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.18486 | 0.21916 | 0.22320 | 0.21911 | 0.22979 | 0.22178 | 0.22313 | 0.22334 | 0.21542 | 0.20409 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.19104 | 0.22654 | 0.23064 | 0.22650 | 0.23737 | 0.22925 | 0.23059 | 0.23080 | 0.22274 | 0.21089 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.82558 | 0.79875 | 0.82379 | 0.82200 | 0.79875 | 0.82379 | 0.79517 | 0.79338 | 0.84705 | 0.80501 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.82558 | 0.79875 | 0.82379 | 0.82200 | 0.79875 | 0.82379 | 0.79517 | 0.79338 | 0.84705 | 0.80501 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.47301 | -0.54378 | -0.47857 | -0.49300 | -0.55756 | -0.47667 | -0.52876 | -0.52938 | -0.45257 | -0.54604 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.47301 | -0.54378 | -0.47857 | -0.49300 | -0.55756 | -0.47667 | -0.52876 | -0.52938 | -0.45257 | -0.54604 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.81400 | 0.79325 | 0.80591 | 0.80398 | 0.78965 | 0.80924 | 0.77994 | 0.77972 | 0.81141 | 0.78897 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.81400 | 0.79325 | 0.80591 | 0.80398 | 0.78965 | 0.80924 | 0.77994 | 0.77972 | 0.81141 | 0.78897 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 1.0954 | 1.1671 | 1.0628 | 1.0558 | 1.0595 | 1.1148 | 1.1190 | 1.1191 | 1.2235 | 1.2144 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.7509 | 1.8859 | 1.7105 | 1.6881 | 1.7046 | 1.7711 | 1.7857 | 1.7858 | 1.9394 | 1.9392 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.79869 | 0.84909 | 0.76927 | 0.76970 | 0.77298 | 0.80443 | 0.81537 | 0.81470 | 0.88999 | 0.88866 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.3136 | 1.4102 | 1.2687 | 1.2679 | 1.2805 | 1.3092 | 1.3390 | 1.3375 | 1.4524 | 1.4642 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.62360 | 0.59738 | 0.63483 | 0.62921 | 0.64794 | 0.63670 | 0.62172 | 0.61985 | 0.61236 | 0.59925 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.62360 | 0.59738 | 0.63483 | 0.62921 | 0.64794 | 0.63670 | 0.62172 | 0.61985 | 0.61236 | 0.59925 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -1.0204 | -0.99731 | -0.94541 | -0.93837 | -0.96675 | -0.93530 | -1.0022 | -0.99611 | -0.99424 | -1.0165 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -1.0204 | -0.99731 | -0.94541 | -0.93837 | -0.96675 | -0.93530 | -1.0022 | -0.99611 | -0.99424 | -1.0165 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.54604 | 0.53113 | 0.56303 | 0.57075 | 0.55948 | 0.55935 | 0.55073 | 0.55170 | 0.54311 | 0.54502 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.54604 | 0.53113 | 0.56303 | 0.57075 | 0.55948 | 0.55935 | 0.55073 | 0.55170 | 0.54311 | 0.54502 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 2.0077 | 1.9092 | 1.9525 | 1.9746 | 2.0165 | 2.0140 | 2.0004 | 1.9972 | 1.9912 | 2.0600 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.7727 | 2.6281 | 2.6862 | 2.7223 | 2.7672 | 2.7721 | 2.7384 | 2.7341 | 2.7645 | 2.8162 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.3916 | 1.3233 | 1.3535 | 1.3687 | 1.3977 | 1.3960 | 1.3864 | 1.3843 | 1.3800 | 1.4279 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.9217 | 1.8216 | 1.8618 | 1.8869 | 1.9181 | 1.9216 | 1.8978 | 1.8952 | 1.9159 | 1.9520 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.22642 | 0.21903 | 0.22149 | 0.21821 | 0.21083 | 0.21411 | 0.21329 | 0.21493 | 0.21739 | 0.21985 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.22642 | 0.21903 | 0.22149 | 0.21821 | 0.21083 | 0.21411 | 0.21329 | 0.21493 | 0.21739 | 0.21985 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -4.5829 | -4.6654 | -4.6884 | -4.3676 | -4.6655 | -4.7091 | -4.6029 | -4.6004 | -4.8022 | -4.5317 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -4.5829 | -4.6654 | -4.6884 | -4.3676 | -4.6655 | -4.7091 | -4.6029 | -4.6004 | -4.8022 | -4.5317 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.23951 | 0.23362 | 0.23911 | 0.23710 | 0.24062 | 0.23858 | 0.23404 | 0.23422 | 0.22866 | 0.23551 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.23951 | 0.23362 | 0.23911 | 0.23710 | 0.24062 | 0.23858 | 0.23404 | 0.23422 | 0.22866 | 0.23551 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.53704 | 0.53559 | 0.52352 | 0.52357 | 0.54119 | 0.54133 | 0.55004 | 0.54958 | 0.53240 | 0.54815 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.54351 | 0.54277 | 0.53051 | 0.53045 | 0.54832 | 0.54856 | 0.55703 | 0.55655 | 0.53913 | 0.55539 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.76905 | 0.73417 | 0.73271 | 0.73415 | 0.73885 | 0.74183 | 0.76063 | 0.76053 | 0.73405 | 0.75485 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.77549 | 0.74046 | 0.73905 | 0.74054 | 0.74527 | 0.74822 | 0.76729 | 0.76704 | 0.74025 | 0.76144 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.3674 | 3.0707 | 2.3667 | 3.0527 | 2.4985 | 2.2744 | 2.7426 | 2.7516 | 3.0194 | 2.6184 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 4.7348 | 6.1414 | 4.7334 | 6.1054 | 4.9970 | 4.5488 | 5.4852 | 5.5033 | 6.0388 | 5.2369 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.6410 | 2.1284 | 1.6402 | 2.1153 | 1.7320 | 1.5767 | 1.9009 | 1.9065 | 2.0933 | 1.8155 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.2820 | 4.2568 | 3.2804 | 4.2306 | 3.4640 | 3.1534 | 3.8017 | 3.8130 | 4.1867 | 3.6311 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.07000 | 0.07000 | 0.12000 | 0.11000 | 0.10000 | 0.07000 | 0.06000 | 0.08000 | 0.07000 | 0.08000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.07000 | 0.07000 | 0.12000 | 0.11000 | 0.10000 | 0.07000 | 0.06000 | 0.08000 | 0.07000 | 0.08000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -3.2623 | -4.2463 | -3.2651 | -4.2146 | -3.4493 | -3.1464 | -3.7789 | -3.7914 | -4.1776 | -3.6191 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -3.2623 | -4.2463 | -3.2651 | -4.2146 | -3.4493 | -3.1464 | -3.7789 | -3.7914 | -4.1776 | -3.6191 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.08881 | 0.09068 | 0.09431 | 0.09568 | 0.09434 | 0.09304 | 0.09384 | 0.09461 | 0.09083 | 0.08964 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.08881 | 0.09068 | 0.09431 | 0.09568 | 0.09434 | 0.09304 | 0.09384 | 0.09461 | 0.09083 | 0.08964 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.86254 | 0.85544 | 0.84541 | 0.85237 | 0.85479 | 0.85230 | 0.86049 | 0.86059 | 0.85421 | 0.85821 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.87193 | 0.86487 | 0.85464 | 0.86195 | 0.86425 | 0.86167 | 0.87013 | 0.87014 | 0.86356 | 0.86784 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.80560 | 0.79336 | 0.78574 | 0.78888 | 0.79934 | 0.79246 | 0.80860 | 0.80856 | 0.79038 | 0.79889 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.80836 | 0.79602 | 0.78828 | 0.79145 | 0.80202 | 0.79510 | 0.81134 | 0.81129 | 0.79310 | 0.80163 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.81230 | 0.80976 | 0.78672 | 0.80252 | 0.80923 | 0.80233 | 0.81341 | 0.81352 | 0.79876 | 0.81008 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.85541 | 0.85238 | 0.82820 | 0.84471 | 0.85210 | 0.84446 | 0.85682 | 0.85691 | 0.84099 | 0.85302 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0592 | 1.0281 | 1.0418 | 1.0264 | 1.0354 | 1.0376 | 1.0408 | 1.0407 | 1.0488 | 1.0377 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.1183 | 2.0562 | 2.0836 | 2.0527 | 2.0708 | 2.0752 | 2.0816 | 2.0814 | 2.0977 | 2.0755 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.73415 | 0.71266 | 0.72215 | 0.71148 | 0.71775 | 0.71924 | 0.72145 | 0.72141 | 0.72704 | 0.71933 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4683 | 1.4253 | 1.4443 | 1.4230 | 1.4355 | 1.4385 | 1.4429 | 1.4428 | 1.4541 | 1.4387 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24931 | 0.24081 | 0.25866 | 0.25654 | 0.23401 | 0.23719 | 0.24888 | 0.24740 | 0.24442 | 0.24527 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24931 | 0.24081 | 0.25866 | 0.25654 | 0.23401 | 0.23719 | 0.24888 | 0.24740 | 0.24442 | 0.24527 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.4032 | -1.3909 | -1.3918 | -1.3904 | -1.3958 | -1.3966 | -1.3950 | -1.3951 | -1.3984 | -1.3929 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.4503 | -1.4067 | -1.4144 | -1.4048 | -1.4221 | -1.4257 | -1.4209 | -1.4207 | -1.4321 | -1.4176 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.24963 | 0.25030 | 0.25079 | 0.25028 | 0.24961 | 0.24982 | 0.25002 | 0.24997 | 0.24966 | 0.25063 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.24912 | 0.25050 | 0.25160 | 0.25055 | 0.24906 | 0.24948 | 0.24996 | 0.24986 | 0.24925 | 0.25107 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 1.1339 | 1.1225 | 1.0810 | 1.1073 | 1.1210 | 1.1183 | 1.1345 | 1.1345 | 1.1222 | 1.1281 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.2625 | 1.2506 | 1.2048 | 1.2316 | 1.2486 | 1.2461 | 1.2629 | 1.2628 | 1.2510 | 1.2555 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0443 | 1.0221 | 1.0457 | 1.0246 | 1.0447 | 1.0369 | 1.0394 | 1.0387 | 1.0344 | 1.0358 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0885 | 2.0442 | 2.0913 | 2.0493 | 2.0894 | 2.0737 | 2.0787 | 2.0775 | 2.0689 | 2.0716 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.72389 | 0.70853 | 0.72477 | 0.71035 | 0.72418 | 0.71874 | 0.72048 | 0.72003 | 0.71706 | 0.71798 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4478 | 1.4171 | 1.4495 | 1.4207 | 1.4484 | 1.4375 | 1.4410 | 1.4401 | 1.4341 | 1.4360 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24614 | 0.27421 | 0.25046 | 0.26681 | 0.24028 | 0.24399 | 0.25571 | 0.25632 | 0.25139 | 0.24954 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24614 | 0.27421 | 0.25046 | 0.26681 | 0.24028 | 0.24399 | 0.25571 | 0.25632 | 0.25139 | 0.24954 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3929 | -1.3857 | -1.3948 | -1.3863 | -1.3983 | -1.3950 | -1.3913 | -1.3912 | -1.3907 | -1.3922 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.4275 | -1.3959 | -1.4237 | -1.3977 | -1.4363 | -1.4252 | -1.4175 | -1.4168 | -1.4124 | -1.4152 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25164 | 0.25154 | 0.25047 | 0.25143 | 0.25003 | 0.25046 | 0.25132 | 0.25127 | 0.25104 | 0.25075 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25279 | 0.25296 | 0.25088 | 0.25274 | 0.24977 | 0.25053 | 0.25230 | 0.25221 | 0.25193 | 0.25151 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.96953 | 0.95211 | 0.92400 | 0.95368 | 0.95438 | 0.95510 | 0.97681 | 0.97673 | 0.94925 | 0.96142 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 1.0372 | 1.0185 | 0.98794 | 1.0199 | 1.0210 | 1.0222 | 1.0446 | 1.0444 | 1.0147 | 1.0286 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0758 | 1.0482 | 1.0462 | 1.0424 | 1.0502 | 1.0718 | 1.0461 | 1.0467 | 1.0566 | 1.0489 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.1516 | 2.0963 | 2.0924 | 2.0849 | 2.1005 | 2.1436 | 2.0923 | 2.0934 | 2.1133 | 2.0977 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.74570 | 0.72662 | 0.72518 | 0.72263 | 0.72800 | 0.74293 | 0.72520 | 0.72554 | 0.73245 | 0.72703 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4914 | 1.4532 | 1.4504 | 1.4453 | 1.4560 | 1.4859 | 1.4504 | 1.4511 | 1.4649 | 1.4541 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.22034 | 0.22847 | 0.24472 | 0.22717 | 0.21937 | 0.22457 | 0.22457 | 0.22717 | 0.21839 | 0.23139 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.22034 | 0.22847 | 0.24472 | 0.22717 | 0.21937 | 0.22457 | 0.22457 | 0.22717 | 0.21839 | 0.23139 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.4145 | -1.4053 | -1.3961 | -1.3993 | -1.4034 | -1.4180 | -1.4010 | -1.4013 | -1.4054 | -1.4017 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.4706 | -1.4352 | -1.4242 | -1.4235 | -1.4426 | -1.4748 | -1.4296 | -1.4303 | -1.4404 | -1.4317 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.24650 | 0.24687 | 0.24988 | 0.24829 | 0.24826 | 0.24544 | 0.24811 | 0.24806 | 0.24725 | 0.24807 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.24304 | 0.24407 | 0.24966 | 0.24672 | 0.24625 | 0.24123 | 0.24629 | 0.24619 | 0.24457 | 0.24619 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.4737 | 1.4347 | 1.4071 | 1.4559 | 1.4325 | 1.4456 | 1.4724 | 1.4718 | 1.4357 | 1.4577 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.8424 | 1.7849 | 1.7537 | 1.8179 | 1.7808 | 1.8045 | 1.8330 | 1.8319 | 1.7879 | 1.8166 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.1000 | 1.0390 | 1.0569 | 1.0387 | 1.0412 | 1.0650 | 1.0546 | 1.0546 | 1.0632 | 1.0468 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.2000 | 2.0781 | 2.1137 | 2.0775 | 2.0825 | 2.1300 | 2.1092 | 2.1092 | 2.1263 | 2.0937 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.76244 | 0.72026 | 0.73257 | 0.72001 | 0.72179 | 0.73824 | 0.73097 | 0.73101 | 0.73697 | 0.72562 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.5249 | 1.4405 | 1.4651 | 1.4400 | 1.4436 | 1.4765 | 1.4619 | 1.4620 | 1.4739 | 1.4512 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.21305 | 0.24553 | 0.25182 | 0.25911 | 0.24884 | 0.23062 | 0.25050 | 0.25249 | 0.23227 | 0.24718 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.21305 | 0.24553 | 0.25182 | 0.25911 | 0.24884 | 0.23062 | 0.25050 | 0.25249 | 0.23227 | 0.24718 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.4213 | -1.3925 | -1.3967 | -1.3922 | -1.3954 | -1.4100 | -1.3979 | -1.3980 | -1.4068 | -1.3955 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4926 | -1.4124 | -1.4357 | -1.4128 | -1.4281 | -1.4608 | -1.4345 | -1.4345 | -1.4482 | -1.4300 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.24601 | 0.25025 | 0.25088 | 0.25050 | 0.25052 | 0.24762 | 0.25042 | 0.25038 | 0.24761 | 0.25086 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.24207 | 0.25042 | 0.25156 | 0.25106 | 0.25072 | 0.24529 | 0.25099 | 0.25091 | 0.24543 | 0.25152 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.49887 | 0.49950 | 0.48124 | 0.50357 | 0.49153 | 0.49712 | 0.50534 | 0.50560 | 0.48618 | 0.50302 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.50152 | 0.50241 | 0.48384 | 0.50638 | 0.49413 | 0.49994 | 0.50808 | 0.50831 | 0.48883 | 0.50577 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.38198 | 0.38258 | 0.37344 | 0.38349 | 0.38489 | 0.38374 | 0.39169 | 0.39183 | 0.37928 | 0.37459 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.38349 | 0.38402 | 0.37485 | 0.38498 | 0.38632 | 0.38527 | 0.39323 | 0.39337 | 0.38074 | 0.37611 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.67771 | 0.70571 | 0.67696 | 0.68997 | 0.72434 | 0.70300 | 0.69643 | 0.69578 | 0.67489 | 0.68131 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.68241 | 0.71076 | 0.68182 | 0.69468 | 0.72921 | 0.70783 | 0.70132 | 0.70063 | 0.67948 | 0.68595 |
| eval/lm/c4_en-validation/CE loss | lower | 3.1861 | 3.1539 | 3.1260 | 3.1443 | 3.1629 | 3.1492 | 3.1873 | 3.1873 | 3.1494 | 3.1609 |
| eval/lm/c4_en-validation/PPL | lower | 24.19 | 23.43 | 22.78 | 23.20 | 23.64 | 23.32 | 24.22 | 24.22 | 23.32 | 23.59 |
| eval/lm/dolma_books-validation/CE loss | lower | 3.1255 | 3.0978 | 3.0583 | 3.0802 | 3.1051 | 3.0812 | 3.1333 | 3.1331 | 3.0857 | 3.0978 |
| eval/lm/dolma_books-validation/PPL | lower | 22.77 | 22.15 | 21.29 | 21.76 | 22.31 | 21.78 | 22.95 | 22.95 | 21.88 | 22.15 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.3161 | 3.2860 | 3.2599 | 3.2760 | 3.2956 | 3.2850 | 3.3186 | 3.3187 | 3.2839 | 3.2943 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 27.55 | 26.74 | 26.05 | 26.47 | 26.99 | 26.71 | 27.62 | 27.62 | 26.68 | 26.96 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.3492 | 2.3252 | 2.3047 | 2.3211 | 2.3315 | 2.3190 | 2.3515 | 2.3515 | 2.3302 | 2.3353 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 10.48 | 10.23 | 10.02 | 10.19 | 10.29 | 10.17 | 10.50 | 10.50 | 10.28 | 10.33 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.4721 | 3.4472 | 3.4200 | 3.4351 | 3.4541 | 3.4423 | 3.4767 | 3.4767 | 3.4416 | 3.4503 |
| eval/lm/dolma_reddit-validation/PPL | lower | 32.20 | 31.41 | 30.57 | 31.03 | 31.63 | 31.26 | 32.35 | 32.35 | 31.24 | 31.51 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.5897 | 1.5637 | 1.5397 | 1.5538 | 1.5736 | 1.5598 | 1.5891 | 1.5890 | 1.5632 | 1.5729 |
| eval/lm/dolma_stack-validation/PPL | lower | 4.9023 | 4.7762 | 4.6631 | 4.7292 | 4.8238 | 4.7579 | 4.8991 | 4.8990 | 4.7741 | 4.8204 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.8538 | 2.8243 | 2.7915 | 2.8072 | 2.8309 | 2.8177 | 2.8570 | 2.8572 | 2.8118 | 2.8288 |
| eval/lm/dolma_wiki-validation/PPL | lower | 17.35 | 16.85 | 16.31 | 16.56 | 16.96 | 16.74 | 17.41 | 17.41 | 16.64 | 16.93 |
| eval/lm/ice-validation/CE loss | lower | 3.2673 | 3.2362 | 3.2078 | 3.2300 | 3.2638 | 3.2341 | 3.2719 | 3.2719 | 3.2368 | 3.2464 |
| eval/lm/ice-validation/PPL | lower | 26.24 | 25.44 | 24.72 | 25.28 | 26.15 | 25.38 | 26.36 | 26.36 | 25.45 | 25.70 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 3.2747 | 3.2499 | 3.2289 | 3.2419 | 3.2590 | 3.2522 | 3.2858 | 3.2857 | 3.2493 | 3.2566 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 26.44 | 25.79 | 25.25 | 25.58 | 26.02 | 25.85 | 26.73 | 26.73 | 25.77 | 25.96 |
| eval/lm/pile-validation/CE loss | lower | 2.4658 | 2.4384 | 2.4134 | 2.4258 | 2.4452 | 2.4304 | 2.4685 | 2.4685 | 2.4349 | 2.4458 |
| eval/lm/pile-validation/PPL | lower | 11.77 | 11.45 | 11.17 | 11.31 | 11.53 | 11.36 | 11.81 | 11.80 | 11.41 | 11.54 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.8358 | 2.7973 | 2.7712 | 2.7853 | 2.8087 | 2.7880 | 2.8378 | 2.8379 | 2.8011 | 2.8159 |
| eval/lm/wikitext_103-validation/PPL | lower | 17.04 | 16.40 | 15.98 | 16.20 | 16.59 | 16.25 | 17.08 | 17.08 | 16.46 | 16.71 |
| throughput/in-loop eval batches | see metric | 826.0 | 826.0 | 1645.0 | 1645.0 | 826.0 | 826.0 | 826.0 | 826.0 | 1645.0 | 826.0 |
| throughput/in-loop eval time (s) | see metric | 110.3 | 121.3 | 140.1 | 128.5 | 129.0 | 129.6 | 121.7 | 336.3 | 124.2 | 115.7 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| eg-480m-cx1-eg24e2k-lr9e-4-r1<br>`rcgxm5qv` | finished | original | 7600603136.0 | 28994 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rcgxm5qv) |
| eg-480m-cx1-eg96e8k-lr1e-3-r1<br>`nvndg2tr` | finished | original | 7622623232.0 | 29078 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/nvndg2tr) |
| int-480m-cx1-intd256e8k-lr1.2e-3-r1<br>`ggbfbedg` | finished | original | 7732461568.0 | 29497 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ggbfbedg) |
| int-480m-cx1-intw256e8k-lr1.2e-3-r1<br>`z4wxvc6h` | finished | original | 7671906304.0 | 29266 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/z4wxvc6h) |
| q3-480m-cx1-q3am128e8k-lr1.2e-3-r1<br>`u7eje3wc` | finished | original | 7635206144.0 | 29126 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/u7eje3wc) |
| q3-480m-cx1-q3td128e8k-lr1.2e-3-r1<br>`pirgmap6` | finished | original | 7651459072.0 | 29188 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/pirgmap6) |
| se-480m-cx1-se0m9-lr1.2e-3-r1<br>`57ae3vzv` | finished | original | 7607943168.0 | 29022 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/57ae3vzv) |
| se-480m-cx1-se0m9-lr1.2e-3-r1<br>`w9pmxrz9` | finished | original |  | 29022 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/w9pmxrz9) |
| sp-480m-cx1-sp192e4k-lr8e-4-r2<br>`artcqrdt` | finished | original | 7652245504.0 | 29191 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/artcqrdt) |
| sp-480m-cx1-sp96e4k-lr1e-3-r1<br>`4kd8hys6` | finished | original | 7622623232.0 | 29078 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4kd8hys6) |

## 480m Cx2

| metric | direction | eg-480m-cx2-eg24e2k-lr1e-3-r1<br>`ksfrmhct` | eg-480m-cx2-eg96e8k-lr1e-3-r1<br>`fzk2affn` | int-480m-cx2-intd256e8k-lr9e-4-r1<br>`ygcyrwld` | int-480m-cx2-intw256e8k-lr9e-4-r1<br>`ywj13bkw` | q3-480m-cx2-q3am128e8k-lr9e-4-r1<br>`5zbyuc6d` | q3-480m-cx2-q3td128e8k-lr9e-4-r1<br>`cxuxwxuh` | se-480m-cx2-se0m9-lr9e-4-r1<br>`h38igjrw` | se-480m-cx2-se0m9-lr9e-4-r1<br>`hdcnmpny` | sp-480m-cx2-sp192e4k-lr6e-4-r1<br>`g7uf5bwk` | sp-480m-cx2-sp96e4k-lr8e-4-r1<br>`0xbwhjoy` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.94836 | 0.92281 | 0.89719 | 0.89703 | 0.90957 | 0.90723 | 0.92964 | 0.92958 | 0.91644 | 0.91941 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 1.0397 | 1.0096 | 0.98114 | 0.98237 | 0.99617 | 0.99183 | 1.0190 | 1.0188 | 1.0030 | 1.0065 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0126 | 1.0308 | 1.0093 | 1.0138 | 1.0099 | 1.0266 | 1.0152 | 1.0140 | 1.0241 | 1.0187 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0252 | 2.0615 | 2.0186 | 2.0277 | 2.0198 | 2.0532 | 2.0305 | 2.0280 | 2.0482 | 2.0373 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.70191 | 0.71446 | 0.69978 | 0.70291 | 0.70010 | 0.71164 | 0.70378 | 0.70288 | 0.70985 | 0.70615 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4038 | 1.4289 | 1.3996 | 1.4058 | 1.4002 | 1.4233 | 1.4076 | 1.4058 | 1.4197 | 1.4123 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.26621 | 0.27133 | 0.24915 | 0.22270 | 0.25768 | 0.25085 | 0.27048 | 0.27048 | 0.24317 | 0.26877 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.26621 | 0.27133 | 0.24915 | 0.22270 | 0.25768 | 0.25085 | 0.27048 | 0.27048 | 0.24317 | 0.26877 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.3925 | -1.4212 | -1.3951 | -1.3990 | -1.3950 | -1.4161 | -1.3997 | -1.3979 | -1.4151 | -1.4085 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.3925 | -1.4212 | -1.3951 | -1.3990 | -1.3950 | -1.4161 | -1.3997 | -1.3979 | -1.4151 | -1.4085 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.25163 | 0.25332 | 0.25084 | 0.24959 | 0.25395 | 0.24941 | 0.25340 | 0.25372 | 0.24894 | 0.25268 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.25163 | 0.25332 | 0.25084 | 0.24959 | 0.25395 | 0.24941 | 0.25340 | 0.25372 | 0.24894 | 0.25268 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.74791 | 0.71854 | 0.67829 | 0.70180 | 0.71247 | 0.70226 | 0.73472 | 0.73496 | 0.72325 | 0.72656 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.81561 | 0.78260 | 0.73712 | 0.76402 | 0.77565 | 0.76452 | 0.80007 | 0.80047 | 0.78778 | 0.79073 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0154 | 1.0136 | 1.0087 | 1.0165 | 1.0200 | 1.0284 | 1.0197 | 1.0198 | 1.0183 | 1.0385 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0308 | 2.0272 | 2.0174 | 2.0331 | 2.0400 | 2.0568 | 2.0395 | 2.0396 | 2.0365 | 2.0771 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.70392 | 0.70266 | 0.69928 | 0.70469 | 0.70707 | 0.71283 | 0.70687 | 0.70691 | 0.70585 | 0.71989 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4078 | 1.4053 | 1.3986 | 1.4094 | 1.4141 | 1.4257 | 1.4137 | 1.4138 | 1.4117 | 1.4398 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.24663 | 0.26052 | 0.26978 | 0.24747 | 0.24200 | 0.24832 | 0.25379 | 0.24916 | 0.25126 | 0.23948 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.24663 | 0.26052 | 0.26978 | 0.24747 | 0.24200 | 0.24832 | 0.25379 | 0.24916 | 0.25126 | 0.23948 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.3974 | -1.3969 | -1.3936 | -1.4035 | -1.4094 | -1.4165 | -1.4041 | -1.4038 | -1.4070 | -1.4361 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.3974 | -1.3969 | -1.3936 | -1.4035 | -1.4094 | -1.4165 | -1.4041 | -1.4038 | -1.4070 | -1.4361 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.24955 | 0.25220 | 0.25100 | 0.24974 | 0.24977 | 0.24932 | 0.24992 | 0.25002 | 0.25050 | 0.24991 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.24955 | 0.25220 | 0.25100 | 0.24974 | 0.24977 | 0.24932 | 0.24992 | 0.25002 | 0.25050 | 0.24991 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 1.8951 | 1.7831 | 1.5832 | 1.5922 | 1.7519 | 1.7440 | 1.7657 | 1.7668 | 1.6079 | 1.7454 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 3.0381 | 2.8655 | 2.5580 | 2.5698 | 2.8156 | 2.7984 | 2.8207 | 2.8224 | 2.5935 | 2.8120 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 1.3135 | 1.2359 | 1.0975 | 1.1036 | 1.2142 | 1.2088 | 1.2239 | 1.2247 | 1.1144 | 1.2097 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 2.1057 | 1.9861 | 1.7731 | 1.7813 | 1.9516 | 1.9398 | 1.9552 | 1.9564 | 1.7978 | 1.9489 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.25024 | 0.28271 | 0.32760 | 0.35817 | 0.27316 | 0.27985 | 0.26934 | 0.26934 | 0.32569 | 0.27889 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.25024 | 0.28271 | 0.32760 | 0.35817 | 0.27316 | 0.27985 | 0.26934 | 0.26934 | 0.32569 | 0.27889 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -2.5392 | -2.3018 | -2.0536 | -2.0433 | -2.2432 | -2.2886 | -2.3348 | -2.3393 | -2.0987 | -2.2581 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -2.5392 | -2.3018 | -2.0536 | -2.0433 | -2.2432 | -2.2886 | -2.3348 | -2.3393 | -2.0987 | -2.2581 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.20822 | 0.22015 | 0.27422 | 0.28962 | 0.24198 | 0.24460 | 0.21604 | 0.21667 | 0.28329 | 0.23728 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.20822 | 0.22015 | 0.27422 | 0.28962 | 0.24198 | 0.24460 | 0.21604 | 0.21667 | 0.28329 | 0.23728 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.46234 | 0.46982 | 0.46860 | 0.46508 | 0.45950 | 0.42839 | 0.47572 | 0.47526 | 0.44132 | 0.46366 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.50278 | 0.51236 | 0.51136 | 0.50633 | 0.50079 | 0.46575 | 0.51797 | 0.51738 | 0.47975 | 0.50479 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.32048 | 0.32566 | 0.32482 | 0.32235 | 0.31849 | 0.29693 | 0.32979 | 0.32945 | 0.30590 | 0.32137 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.34848 | 0.35514 | 0.35443 | 0.35094 | 0.34712 | 0.32281 | 0.35900 | 0.35860 | 0.33252 | 0.34989 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.48518 | 0.50692 | 0.50494 | 0.51976 | 0.51680 | 0.50198 | 0.48123 | 0.48024 | 0.52174 | 0.50692 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.48518 | 0.50692 | 0.50494 | 0.51976 | 0.51680 | 0.50198 | 0.48123 | 0.48024 | 0.52174 | 0.50692 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -2.6958 | -2.5884 | -2.4108 | -2.5221 | -2.3992 | -2.5413 | -2.7366 | -2.7340 | -2.4644 | -2.5179 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -2.6958 | -2.5884 | -2.4108 | -2.5221 | -2.3992 | -2.5413 | -2.7366 | -2.7340 | -2.4644 | -2.5179 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.46785 | 0.48427 | 0.49322 | 0.48565 | 0.49598 | 0.48463 | 0.46506 | 0.46448 | 0.49470 | 0.48992 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.46785 | 0.48427 | 0.49322 | 0.48565 | 0.49598 | 0.48463 | 0.46506 | 0.46448 | 0.49470 | 0.48992 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.54631 | 0.50116 | 0.48576 | 0.41494 | 0.48816 | 0.46123 | 0.45363 | 0.45756 | 0.49400 | 0.50584 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.65745 | 0.60273 | 0.58398 | 0.49796 | 0.58685 | 0.55419 | 0.54558 | 0.55011 | 0.59397 | 0.60914 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.37878 | 0.34754 | 0.33690 | 0.28775 | 0.33849 | 0.31983 | 0.31455 | 0.31724 | 0.34253 | 0.35078 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.45587 | 0.41793 | 0.40501 | 0.34538 | 0.40701 | 0.38429 | 0.37834 | 0.38143 | 0.41187 | 0.42243 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.78303 | 0.80617 | 0.83124 | 0.83414 | 0.81485 | 0.82353 | 0.80810 | 0.81003 | 0.83896 | 0.81774 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.78303 | 0.80617 | 0.83124 | 0.83414 | 0.81485 | 0.82353 | 0.80810 | 0.81003 | 0.83896 | 0.81774 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.55696 | -0.54091 | -0.45430 | -0.45015 | -0.51555 | -0.50866 | -0.53654 | -0.53651 | -0.49384 | -0.48486 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.55696 | -0.54091 | -0.45430 | -0.45015 | -0.51555 | -0.50866 | -0.53654 | -0.53651 | -0.49384 | -0.48486 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.68628 | 0.70329 | 0.73904 | 0.74178 | 0.71163 | 0.71097 | 0.70180 | 0.70110 | 0.72591 | 0.72542 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.68628 | 0.70329 | 0.73904 | 0.74178 | 0.71163 | 0.71097 | 0.70180 | 0.70110 | 0.72591 | 0.72542 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.30979 | 0.30265 | 0.31420 | 0.29518 | 0.27912 | 0.30072 | 0.31182 | 0.31110 | 0.29967 | 0.29953 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.32009 | 0.31275 | 0.32465 | 0.30509 | 0.28840 | 0.31072 | 0.32220 | 0.32151 | 0.30966 | 0.30958 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.21473 | 0.20982 | 0.21780 | 0.20463 | 0.19348 | 0.20845 | 0.21614 | 0.21565 | 0.20774 | 0.20765 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.22190 | 0.21680 | 0.22504 | 0.21149 | 0.19992 | 0.21540 | 0.22337 | 0.22287 | 0.21466 | 0.21461 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.86315 | 0.81485 | 0.85510 | 0.82826 | 0.82021 | 0.84705 | 0.87746 | 0.87120 | 0.82200 | 0.82021 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.86315 | 0.81485 | 0.85510 | 0.82826 | 0.82021 | 0.84705 | 0.87746 | 0.87120 | 0.82200 | 0.82021 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.36142 | -0.49706 | -0.37124 | -0.41742 | -0.46320 | -0.41223 | -0.34238 | -0.34597 | -0.44960 | -0.43161 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.36142 | -0.49706 | -0.37124 | -0.41742 | -0.46320 | -0.41223 | -0.34238 | -0.34597 | -0.44960 | -0.43161 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.84502 | 0.82401 | 0.84150 | 0.82262 | 0.81891 | 0.82958 | 0.84934 | 0.84885 | 0.81509 | 0.80719 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.84502 | 0.82401 | 0.84150 | 0.82262 | 0.81891 | 0.82958 | 0.84934 | 0.84885 | 0.81509 | 0.80719 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 1.0000 | 0.96219 | 0.92641 | 0.96214 | 1.0129 | 0.97584 | 0.99570 | 0.99666 | 0.99815 | 0.96568 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.6181 | 1.5635 | 1.5156 | 1.5484 | 1.6302 | 1.5762 | 1.6131 | 1.6150 | 1.6274 | 1.5654 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.72358 | 0.71361 | 0.67939 | 0.71118 | 0.73706 | 0.71349 | 0.72261 | 0.72343 | 0.72802 | 0.70302 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.1998 | 1.2040 | 1.1460 | 1.1872 | 1.2197 | 1.1871 | 1.2011 | 1.2027 | 1.2209 | 1.1708 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.64607 | 0.65169 | 0.65730 | 0.63670 | 0.65730 | 0.64419 | 0.65918 | 0.65169 | 0.66105 | 0.67603 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.64607 | 0.65169 | 0.65730 | 0.63670 | 0.65730 | 0.64419 | 0.65918 | 0.65169 | 0.66105 | 0.67603 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.90552 | -0.89097 | -0.86829 | -0.90445 | -0.91073 | -0.91933 | -0.90753 | -0.90600 | -0.85543 | -0.86181 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.90552 | -0.89097 | -0.86829 | -0.90445 | -0.91073 | -0.91933 | -0.90753 | -0.90600 | -0.85543 | -0.86181 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.56966 | 0.57472 | 0.58279 | 0.56066 | 0.55666 | 0.57187 | 0.57423 | 0.57440 | 0.58558 | 0.58144 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.56966 | 0.57472 | 0.58279 | 0.56066 | 0.55666 | 0.57187 | 0.57423 | 0.57440 | 0.58558 | 0.58144 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.8058 | 1.7893 | 1.7688 | 1.8165 | 1.8288 | 1.8270 | 1.8440 | 1.8417 | 1.8181 | 1.8719 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.4746 | 2.4611 | 2.4428 | 2.5066 | 2.4899 | 2.4933 | 2.5377 | 2.5337 | 2.5007 | 2.5750 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.2515 | 1.2403 | 1.2260 | 1.2591 | 1.2676 | 1.2664 | 1.2781 | 1.2765 | 1.2604 | 1.2973 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.7151 | 1.7060 | 1.6932 | 1.7374 | 1.7256 | 1.7282 | 1.7590 | 1.7562 | 1.7336 | 1.7847 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.22395 | 0.22970 | 0.22642 | 0.22067 | 0.21985 | 0.22970 | 0.21739 | 0.22149 | 0.22970 | 0.23462 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.22395 | 0.22970 | 0.22642 | 0.22067 | 0.21985 | 0.22970 | 0.21739 | 0.22149 | 0.22970 | 0.23462 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -4.2288 | -4.3244 | -4.0215 | -4.1868 | -4.3153 | -4.2554 | -4.4076 | -4.4016 | -4.2703 | -4.1258 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -4.2288 | -4.3244 | -4.0215 | -4.1868 | -4.3153 | -4.2554 | -4.4076 | -4.4016 | -4.2703 | -4.1258 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.24341 | 0.23791 | 0.24922 | 0.24018 | 0.23837 | 0.24656 | 0.23635 | 0.23713 | 0.23792 | 0.24889 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.24341 | 0.23791 | 0.24922 | 0.24018 | 0.23837 | 0.24656 | 0.23635 | 0.23713 | 0.23792 | 0.24889 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.49711 | 0.47977 | 0.46180 | 0.48081 | 0.48976 | 0.47607 | 0.49438 | 0.49420 | 0.47345 | 0.48563 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.50359 | 0.48597 | 0.46769 | 0.48742 | 0.49635 | 0.48240 | 0.50099 | 0.50089 | 0.47954 | 0.49218 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.71210 | 0.69999 | 0.68642 | 0.67958 | 0.70652 | 0.69691 | 0.70216 | 0.70180 | 0.69181 | 0.68971 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.71813 | 0.70594 | 0.69225 | 0.68542 | 0.71261 | 0.70284 | 0.70814 | 0.70794 | 0.69806 | 0.69581 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.7088 | 2.5698 | 2.4871 | 2.4276 | 2.6266 | 3.1058 | 2.3071 | 2.2914 | 2.1435 | 2.3491 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 5.4175 | 5.1395 | 4.9742 | 4.8552 | 5.2532 | 6.2116 | 4.6142 | 4.5828 | 4.2870 | 4.6981 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.8774 | 1.7820 | 1.7247 | 1.6833 | 1.8206 | 2.1524 | 1.5995 | 1.5879 | 1.4861 | 1.6278 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.7548 | 3.5639 | 3.4495 | 3.3666 | 3.6412 | 4.3048 | 3.1991 | 3.1758 | 2.9722 | 3.2556 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.14000 | 0.10000 | 0.09000 | 0.10000 | 0.08000 | 0.05000 | 0.09000 | 0.08000 | 0.09000 | 0.09000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.14000 | 0.10000 | 0.09000 | 0.10000 | 0.08000 | 0.05000 | 0.09000 | 0.08000 | 0.09000 | 0.09000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -3.7481 | -3.5580 | -3.4415 | -3.3613 | -3.6360 | -4.2990 | -3.1843 | -3.1609 | -2.9617 | -3.2512 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -3.7481 | -3.5580 | -3.4415 | -3.3613 | -3.6360 | -4.2990 | -3.1843 | -3.1609 | -2.9617 | -3.2512 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09469 | 0.09522 | 0.09243 | 0.09533 | 0.09556 | 0.08814 | 0.09631 | 0.09603 | 0.09419 | 0.09483 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09469 | 0.09522 | 0.09243 | 0.09533 | 0.09556 | 0.08814 | 0.09631 | 0.09603 | 0.09419 | 0.09483 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.83376 | 0.82680 | 0.81799 | 0.82565 | 0.83076 | 0.83250 | 0.83743 | 0.83765 | 0.81860 | 0.82368 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.84284 | 0.83604 | 0.82698 | 0.83483 | 0.83985 | 0.84174 | 0.84688 | 0.84698 | 0.82764 | 0.83265 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.75951 | 0.73899 | 0.72267 | 0.72881 | 0.74644 | 0.73383 | 0.74835 | 0.74831 | 0.72704 | 0.73912 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.76215 | 0.74138 | 0.72516 | 0.73130 | 0.74885 | 0.73621 | 0.75082 | 0.75076 | 0.72943 | 0.74146 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.77783 | 0.75943 | 0.74263 | 0.75456 | 0.75960 | 0.75770 | 0.77036 | 0.77021 | 0.75414 | 0.75769 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.81899 | 0.79864 | 0.78139 | 0.79398 | 0.79956 | 0.79722 | 0.81112 | 0.81098 | 0.79303 | 0.79762 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0180 | 1.0372 | 1.0194 | 1.0140 | 1.0190 | 1.0293 | 1.0259 | 1.0262 | 1.0223 | 1.0419 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0360 | 2.0744 | 2.0388 | 2.0280 | 2.0381 | 2.0587 | 2.0518 | 2.0523 | 2.0446 | 2.0839 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.70572 | 0.71889 | 0.70667 | 0.70295 | 0.70641 | 0.71357 | 0.71115 | 0.71137 | 0.70866 | 0.72225 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4114 | 1.4378 | 1.4133 | 1.4059 | 1.4128 | 1.4271 | 1.4223 | 1.4227 | 1.4173 | 1.4445 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24400 | 0.25972 | 0.25760 | 0.24038 | 0.25887 | 0.23868 | 0.24293 | 0.24570 | 0.24442 | 0.24995 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24400 | 0.25972 | 0.25760 | 0.24038 | 0.25887 | 0.23868 | 0.24293 | 0.24570 | 0.24442 | 0.24995 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3892 | -1.3958 | -1.3907 | -1.3903 | -1.3892 | -1.3963 | -1.3941 | -1.3943 | -1.3934 | -1.4008 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.3968 | -1.4288 | -1.4092 | -1.4012 | -1.4044 | -1.4190 | -1.4145 | -1.4149 | -1.4125 | -1.4397 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.24983 | 0.25086 | 0.25063 | 0.24986 | 0.25081 | 0.24917 | 0.24966 | 0.24963 | 0.24975 | 0.24999 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.24965 | 0.25154 | 0.25120 | 0.24966 | 0.25155 | 0.24837 | 0.24936 | 0.24930 | 0.24947 | 0.25003 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 1.0832 | 1.0551 | 1.0299 | 1.0380 | 1.0569 | 1.0634 | 1.0831 | 1.0832 | 1.0528 | 1.0654 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.2067 | 1.1750 | 1.1484 | 1.1571 | 1.1771 | 1.1857 | 1.2073 | 1.2072 | 1.1725 | 1.1858 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0149 | 1.0291 | 1.0116 | 1.0096 | 1.0169 | 1.0090 | 1.0151 | 1.0146 | 1.0119 | 1.0281 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0297 | 2.0582 | 2.0231 | 2.0191 | 2.0338 | 2.0180 | 2.0303 | 2.0292 | 2.0238 | 2.0563 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.70356 | 0.71331 | 0.70119 | 0.69986 | 0.70490 | 0.69944 | 0.70369 | 0.70332 | 0.70139 | 0.71265 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4071 | 1.4266 | 1.4024 | 1.3997 | 1.4098 | 1.3989 | 1.4074 | 1.4066 | 1.4028 | 1.4253 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24892 | 0.26002 | 0.26928 | 0.24800 | 0.27020 | 0.29272 | 0.27082 | 0.26095 | 0.26619 | 0.27051 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24892 | 0.26002 | 0.26928 | 0.24800 | 0.27020 | 0.29272 | 0.27082 | 0.26095 | 0.26619 | 0.27051 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3869 | -1.3902 | -1.3859 | -1.3868 | -1.3863 | -1.3823 | -1.3866 | -1.3863 | -1.3851 | -1.3897 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.3927 | -1.4174 | -1.3977 | -1.3943 | -1.4035 | -1.3904 | -1.3976 | -1.3969 | -1.3962 | -1.4206 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25044 | 0.25207 | 0.25162 | 0.25068 | 0.25212 | 0.25262 | 0.25129 | 0.25133 | 0.25184 | 0.25280 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25077 | 0.25352 | 0.25321 | 0.25126 | 0.25391 | 0.25532 | 0.25253 | 0.25256 | 0.25355 | 0.25502 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.92912 | 0.90185 | 0.88031 | 0.88364 | 0.91031 | 0.89288 | 0.91410 | 0.91438 | 0.89760 | 0.90399 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 0.99277 | 0.96386 | 0.94160 | 0.94391 | 0.97343 | 0.95462 | 0.97680 | 0.97706 | 0.95964 | 0.96558 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0292 | 1.0574 | 1.0162 | 1.0118 | 1.0311 | 1.0183 | 1.0215 | 1.0230 | 1.0168 | 1.0454 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0585 | 2.1148 | 2.0323 | 2.0236 | 2.0623 | 2.0366 | 2.0430 | 2.0460 | 2.0336 | 2.0908 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.71347 | 0.73293 | 0.70441 | 0.70147 | 0.71479 | 0.70586 | 0.70812 | 0.70917 | 0.70488 | 0.72466 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4269 | 1.4659 | 1.4088 | 1.4029 | 1.4296 | 1.4117 | 1.4162 | 1.4183 | 1.4098 | 1.4493 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.22912 | 0.23464 | 0.24472 | 0.22782 | 0.24017 | 0.26747 | 0.25024 | 0.25512 | 0.24179 | 0.23464 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.22912 | 0.23464 | 0.24472 | 0.22782 | 0.24017 | 0.26747 | 0.25024 | 0.25512 | 0.24179 | 0.23464 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3964 | -1.4127 | -1.3896 | -1.3896 | -1.3966 | -1.3874 | -1.3912 | -1.3922 | -1.3907 | -1.4056 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.4130 | -1.4558 | -1.4040 | -1.3980 | -1.4220 | -1.4037 | -1.4078 | -1.4100 | -1.4047 | -1.4449 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.24831 | 0.24606 | 0.25057 | 0.24977 | 0.24934 | 0.25175 | 0.25029 | 0.25007 | 0.25014 | 0.24809 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.24667 | 0.24281 | 0.25116 | 0.24945 | 0.24863 | 0.25338 | 0.25059 | 0.25017 | 0.25032 | 0.24642 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.4028 | 1.3684 | 1.3460 | 1.3511 | 1.3861 | 1.3641 | 1.4015 | 1.4019 | 1.3657 | 1.3880 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.7543 | 1.7055 | 1.6794 | 1.6871 | 1.7340 | 1.7027 | 1.7487 | 1.7493 | 1.7067 | 1.7361 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0222 | 1.0580 | 1.0100 | 1.0179 | 1.0414 | 1.0218 | 1.0360 | 1.0369 | 1.0180 | 1.0395 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0443 | 2.1159 | 2.0200 | 2.0359 | 2.0828 | 2.0437 | 2.0720 | 2.0739 | 2.0360 | 2.0791 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.70860 | 0.73340 | 0.70012 | 0.70563 | 0.72183 | 0.70834 | 0.71813 | 0.71879 | 0.70568 | 0.72055 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4172 | 1.4668 | 1.4002 | 1.4113 | 1.4437 | 1.4167 | 1.4363 | 1.4376 | 1.4114 | 1.4411 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24453 | 0.24254 | 0.26408 | 0.24122 | 0.25514 | 0.28131 | 0.25017 | 0.24023 | 0.26806 | 0.25447 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24453 | 0.24254 | 0.26408 | 0.24122 | 0.25514 | 0.28131 | 0.25017 | 0.24023 | 0.26806 | 0.25447 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3910 | -1.4079 | -1.3839 | -1.3907 | -1.3984 | -1.3897 | -1.3963 | -1.3969 | -1.3873 | -1.3973 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4032 | -1.4555 | -1.3944 | -1.4044 | -1.4347 | -1.4079 | -1.4230 | -1.4244 | -1.4038 | -1.4357 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.24975 | 0.24844 | 0.25220 | 0.25006 | 0.25024 | 0.25117 | 0.24975 | 0.24959 | 0.25180 | 0.25107 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.24951 | 0.24722 | 0.25428 | 0.25008 | 0.25042 | 0.25248 | 0.24954 | 0.24922 | 0.25356 | 0.25208 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.47430 | 0.47171 | 0.46393 | 0.45955 | 0.47491 | 0.47440 | 0.46468 | 0.46500 | 0.45423 | 0.46943 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.47693 | 0.47445 | 0.46661 | 0.46206 | 0.47755 | 0.47714 | 0.46737 | 0.46768 | 0.45681 | 0.47206 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.36913 | 0.35779 | 0.34832 | 0.33831 | 0.36228 | 0.35070 | 0.36304 | 0.36253 | 0.34796 | 0.36094 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.37056 | 0.35920 | 0.34963 | 0.33962 | 0.36371 | 0.35201 | 0.36442 | 0.36388 | 0.34934 | 0.36232 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.66902 | 0.65720 | 0.60406 | 0.61457 | 0.64281 | 0.62350 | 0.64715 | 0.64731 | 0.60627 | 0.62738 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.67375 | 0.66189 | 0.60834 | 0.61902 | 0.64733 | 0.62786 | 0.65172 | 0.65195 | 0.61060 | 0.63179 |
| eval/lm/c4_en-validation/CE loss | lower | 3.0823 | 3.0559 | 3.0164 | 3.0309 | 3.0581 | 3.0455 | 3.0848 | 3.0849 | 3.0292 | 3.0503 |
| eval/lm/c4_en-validation/PPL | lower | 21.81 | 21.24 | 20.42 | 20.72 | 21.29 | 21.02 | 21.86 | 21.87 | 20.68 | 21.12 |
| eval/lm/dolma_books-validation/CE loss | lower | 2.9993 | 2.9731 | 2.9200 | 2.9424 | 2.9783 | 2.9625 | 3.0048 | 3.0044 | 2.9393 | 2.9574 |
| eval/lm/dolma_books-validation/PPL | lower | 20.07 | 19.55 | 18.54 | 18.96 | 19.66 | 19.35 | 20.18 | 20.17 | 18.90 | 19.25 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.2138 | 3.1902 | 3.1505 | 3.1653 | 3.1944 | 3.1821 | 3.2178 | 3.2179 | 3.1643 | 3.1848 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 24.87 | 24.29 | 23.35 | 23.70 | 24.40 | 24.10 | 24.97 | 24.98 | 23.67 | 24.16 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.2580 | 2.2391 | 2.2078 | 2.2222 | 2.2389 | 2.2252 | 2.2598 | 2.2598 | 2.2255 | 2.2343 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 9.5635 | 9.3853 | 9.0954 | 9.2274 | 9.3834 | 9.2553 | 9.5815 | 9.5813 | 9.2581 | 9.3399 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.3809 | 3.3546 | 3.3185 | 3.3327 | 3.3622 | 3.3477 | 3.3840 | 3.3839 | 3.3330 | 3.3478 |
| eval/lm/dolma_reddit-validation/PPL | lower | 29.40 | 28.63 | 27.62 | 28.01 | 28.85 | 28.44 | 29.49 | 29.49 | 28.02 | 28.44 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.4635 | 1.4372 | 1.4109 | 1.4284 | 1.4470 | 1.4347 | 1.4638 | 1.4637 | 1.4258 | 1.4374 |
| eval/lm/dolma_stack-validation/PPL | lower | 4.3211 | 4.2091 | 4.0996 | 4.1721 | 4.2504 | 4.1983 | 4.3224 | 4.3221 | 4.1611 | 4.2099 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.7381 | 2.7199 | 2.6702 | 2.6873 | 2.7134 | 2.7059 | 2.7415 | 2.7416 | 2.6812 | 2.7028 |
| eval/lm/dolma_wiki-validation/PPL | lower | 15.46 | 15.18 | 14.44 | 14.69 | 15.08 | 14.97 | 15.51 | 15.51 | 14.60 | 14.92 |
| eval/lm/ice-validation/CE loss | lower | 3.1641 | 3.1381 | 3.0923 | 3.1203 | 3.1294 | 3.1147 | 3.1684 | 3.1685 | 3.1101 | 3.1315 |
| eval/lm/ice-validation/PPL | lower | 23.67 | 23.06 | 22.03 | 22.65 | 22.86 | 22.53 | 23.77 | 23.77 | 22.42 | 22.91 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 3.1951 | 3.1619 | 3.1329 | 3.1358 | 3.1783 | 3.1599 | 3.1875 | 3.1875 | 3.1405 | 3.1567 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 24.41 | 23.61 | 22.94 | 23.01 | 24.01 | 23.57 | 24.23 | 24.23 | 23.12 | 23.49 |
| eval/lm/pile-validation/CE loss | lower | 2.3566 | 2.3335 | 2.2978 | 2.3145 | 2.3322 | 2.3202 | 2.3552 | 2.3552 | 2.3080 | 2.3242 |
| eval/lm/pile-validation/PPL | lower | 10.56 | 10.31 | 9.9521 | 10.12 | 10.30 | 10.18 | 10.54 | 10.54 | 10.05 | 10.22 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.7083 | 2.6892 | 2.6378 | 2.6534 | 2.6851 | 2.6639 | 2.7160 | 2.7159 | 2.6641 | 2.6729 |
| eval/lm/wikitext_103-validation/PPL | lower | 15.00 | 14.72 | 13.98 | 14.20 | 14.66 | 14.35 | 15.12 | 15.12 | 14.35 | 14.48 |
| throughput/in-loop eval batches | see metric | 1645.0 | 1645.0 | 1645.0 | 1645.0 | 1645.0 | 1645.0 | 1645.0 | 1645.0 | 1645.0 | 1645.0 |
| throughput/in-loop eval time (s) | see metric | 119.0 | 126.2 | 143.8 | 123.2 | 146.6 | 172.1 | 147.6 | 293.7 | 125.9 | 125.0 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| eg-480m-cx2-eg24e2k-lr1e-3-r1<br>`ksfrmhct` | finished | original | 15201337344.0 | 38659 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ksfrmhct) |
| eg-480m-cx2-eg96e8k-lr1e-3-r1<br>`fzk2affn` | finished | original | 15245377536.0 | 38771 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fzk2affn) |
| int-480m-cx2-intd256e8k-lr9e-4-r1<br>`ygcyrwld` | finished | original | 15465185280.0 | 39330 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ygcyrwld) |
| int-480m-cx2-intw256e8k-lr9e-4-r1<br>`ywj13bkw` | finished | original | 15343681536.0 | 39021 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ywj13bkw) |
| q3-480m-cx2-q3am128e8k-lr9e-4-r1<br>`5zbyuc6d` | finished | original | 15270150144.0 | 38834 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5zbyuc6d) |
| q3-480m-cx2-q3td128e8k-lr9e-4-r1<br>`cxuxwxuh` | finished | original | 15302787072.0 | 38917 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/cxuxwxuh) |
| se-480m-cx2-se0m9-lr9e-4-r1<br>`h38igjrw` | finished | original | 15215886336.0 | 38696 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h38igjrw) |
| se-480m-cx2-se0m9-lr9e-4-r1<br>`hdcnmpny` | finished | original |  | 38696 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/hdcnmpny) |
| sp-480m-cx2-sp192e4k-lr6e-4-r1<br>`g7uf5bwk` | finished | original | 15304359936.0 | 38921 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/g7uf5bwk) |
| sp-480m-cx2-sp96e4k-lr8e-4-r1<br>`0xbwhjoy` | finished | original | 15245377536.0 | 38771 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0xbwhjoy) |

## 480m Cx4

| metric | direction | eg-480m-cx4-eg24e2k-lr8e-4-r1<br>`wq8gib5l` | eg-480m-cx4-eg96e8k-lr8e-4-r1<br>`ezokso90` | int-480m-cx4-intd256e8k-lr8e-4-r1<br>`zf909hyq` | int-480m-cx4-intw256e8k-lr8e-4-r1<br>`rblv9hpr` | q3-480m-cx4-q3am128e8k-lr8e-4-r1<br>`v7vgfj0v` | q3-480m-cx4-q3td128e8k-lr8e-4-r1<br>`umqcq7bm` | se-480m-cx4-se0m9-lr8e-4-r1<br>`1j4l0j4h` | se-480m-cx4-se0m9-lr8e-4-r1<br>`7tyd6xrj` | sp-480m-cx4-sp192e4k-lr6e-4-r2<br>`drpw4m1b` | sp-480m-cx4-sp96e4k-lr7e-4-r1<br>`r6dfpalm` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.89857 | 0.88369 | 0.86848 | 0.86795 | 0.87567 | 0.86968 | 0.89219 | 0.89183 | 0.86199 | 0.88128 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 0.98038 | 0.96977 | 0.95009 | 0.94936 | 0.95683 | 0.94884 | 0.97435 | 0.97397 | 0.94382 | 0.96242 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0056 | 1.0104 | 1.0071 | 1.0027 | 1.0116 | 1.0125 | 1.0094 | 1.0091 | 1.0048 | 1.0084 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0111 | 2.0208 | 2.0143 | 2.0053 | 2.0231 | 2.0250 | 2.0189 | 2.0183 | 2.0097 | 2.0169 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.69711 | 0.70042 | 0.69817 | 0.69522 | 0.70127 | 0.70185 | 0.69981 | 0.69955 | 0.69661 | 0.69915 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.3942 | 1.4008 | 1.3963 | 1.3904 | 1.4025 | 1.4037 | 1.3996 | 1.3991 | 1.3932 | 1.3983 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.25768 | 0.26109 | 0.26195 | 0.25341 | 0.22270 | 0.26024 | 0.25939 | 0.25853 | 0.25427 | 0.25085 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.25768 | 0.26109 | 0.26195 | 0.25341 | 0.22270 | 0.26024 | 0.25939 | 0.25853 | 0.25427 | 0.25085 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.3881 | -1.3951 | -1.3908 | -1.3855 | -1.3978 | -1.3988 | -1.3942 | -1.3935 | -1.3900 | -1.3941 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.3881 | -1.3951 | -1.3908 | -1.3855 | -1.3978 | -1.3988 | -1.3942 | -1.3935 | -1.3900 | -1.3941 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.25167 | 0.25142 | 0.25086 | 0.25128 | 0.24945 | 0.25185 | 0.25078 | 0.25097 | 0.25243 | 0.25005 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.25167 | 0.25142 | 0.25086 | 0.25128 | 0.24945 | 0.25185 | 0.25078 | 0.25097 | 0.25243 | 0.25005 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.69620 | 0.66904 | 0.65647 | 0.66354 | 0.66920 | 0.65819 | 0.68821 | 0.68855 | 0.65924 | 0.68602 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.75705 | 0.72789 | 0.71420 | 0.72214 | 0.72760 | 0.71565 | 0.74889 | 0.74924 | 0.71730 | 0.74615 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0130 | 1.0160 | 1.0086 | 1.0129 | 1.0130 | 1.0173 | 1.0091 | 1.0091 | 1.0149 | 1.0095 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0259 | 2.0319 | 2.0173 | 2.0258 | 2.0259 | 2.0346 | 2.0182 | 2.0181 | 2.0299 | 2.0191 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.70224 | 0.70432 | 0.69922 | 0.70217 | 0.70227 | 0.70522 | 0.69962 | 0.69954 | 0.70359 | 0.69988 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4045 | 1.4086 | 1.3984 | 1.4043 | 1.4045 | 1.4104 | 1.3992 | 1.3991 | 1.4072 | 1.3998 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.25379 | 0.24621 | 0.25463 | 0.24874 | 0.23737 | 0.25337 | 0.25547 | 0.25758 | 0.24790 | 0.24621 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.25379 | 0.24621 | 0.25463 | 0.24874 | 0.23737 | 0.25337 | 0.25547 | 0.25758 | 0.24790 | 0.24621 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.3985 | -1.4023 | -1.3923 | -1.3998 | -1.3985 | -1.4062 | -1.3935 | -1.3933 | -1.4041 | -1.3958 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.3985 | -1.4023 | -1.3923 | -1.3998 | -1.3985 | -1.4062 | -1.3935 | -1.3933 | -1.4041 | -1.3958 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.24963 | 0.25073 | 0.25114 | 0.24978 | 0.24932 | 0.24984 | 0.25100 | 0.25102 | 0.25052 | 0.25050 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.24963 | 0.25073 | 0.25114 | 0.24978 | 0.24932 | 0.24984 | 0.25100 | 0.25102 | 0.25052 | 0.25050 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 1.6828 | 1.5117 | 1.3610 | 1.3134 | 1.4661 | 1.4399 | 1.5614 | 1.5607 | 1.3655 | 1.3855 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 2.7276 | 2.4245 | 2.1823 | 2.1181 | 2.3488 | 2.3160 | 2.5101 | 2.5094 | 2.2090 | 2.2211 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 1.1665 | 1.0479 | 0.94339 | 0.91037 | 1.0162 | 0.99806 | 1.0823 | 1.0817 | 0.94649 | 0.96025 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 1.8907 | 1.6806 | 1.5126 | 1.4681 | 1.6281 | 1.6053 | 1.7399 | 1.7393 | 1.5310 | 1.5396 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.35817 | 0.34384 | 0.42120 | 0.46132 | 0.42216 | 0.42598 | 0.35435 | 0.35626 | 0.43744 | 0.40688 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.35817 | 0.34384 | 0.42120 | 0.46132 | 0.42216 | 0.42598 | 0.35435 | 0.35626 | 0.43744 | 0.40688 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -2.2393 | -1.9796 | -1.8289 | -1.6374 | -1.9604 | -1.8783 | -2.0708 | -2.0672 | -1.7083 | -1.9159 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -2.2393 | -1.9796 | -1.8289 | -1.6374 | -1.9604 | -1.8783 | -2.0708 | -2.0672 | -1.7083 | -1.9159 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.32244 | 0.30783 | 0.37239 | 0.42537 | 0.36228 | 0.36744 | 0.31248 | 0.31243 | 0.38037 | 0.35188 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.32244 | 0.30783 | 0.37239 | 0.42537 | 0.36228 | 0.36744 | 0.31248 | 0.31243 | 0.38037 | 0.35188 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.45955 | 0.43299 | 0.36647 | 0.40381 | 0.43611 | 0.39161 | 0.43103 | 0.43129 | 0.34994 | 0.42170 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.50068 | 0.47215 | 0.39880 | 0.43997 | 0.47525 | 0.42799 | 0.47052 | 0.47092 | 0.38118 | 0.46032 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.31853 | 0.30017 | 0.25398 | 0.27991 | 0.30228 | 0.27146 | 0.29878 | 0.29897 | 0.24257 | 0.29228 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.34709 | 0.32727 | 0.27642 | 0.30497 | 0.32942 | 0.29663 | 0.32612 | 0.32640 | 0.26424 | 0.31904 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.52174 | 0.55435 | 0.58300 | 0.57510 | 0.57016 | 0.57806 | 0.51186 | 0.50988 | 0.60079 | 0.57905 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.52174 | 0.55435 | 0.58300 | 0.57510 | 0.57016 | 0.57806 | 0.51186 | 0.50988 | 0.60079 | 0.57905 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -2.3348 | -2.1344 | -1.7239 | -1.9153 | -1.9898 | -1.7894 | -2.2213 | -2.2234 | -1.7410 | -2.0316 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -2.3348 | -2.1344 | -1.7239 | -1.9153 | -1.9898 | -1.7894 | -2.2213 | -2.2234 | -1.7410 | -2.0316 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.50202 | 0.52505 | 0.56451 | 0.54616 | 0.53607 | 0.55473 | 0.50088 | 0.49984 | 0.56628 | 0.54215 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.50202 | 0.52505 | 0.56451 | 0.54616 | 0.53607 | 0.55473 | 0.50088 | 0.49984 | 0.56628 | 0.54215 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.46320 | 0.53932 | 0.42276 | 0.50367 | 0.45082 | 0.42541 | 0.40425 | 0.40308 | 0.35879 | 0.39185 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.55753 | 0.65032 | 0.50622 | 0.60634 | 0.54249 | 0.51225 | 0.48730 | 0.48580 | 0.43158 | 0.47207 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.32121 | 0.37394 | 0.29316 | 0.34923 | 0.31268 | 0.29504 | 0.28033 | 0.27954 | 0.24889 | 0.27175 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.38660 | 0.45094 | 0.35104 | 0.42051 | 0.37626 | 0.35528 | 0.33796 | 0.33696 | 0.29940 | 0.32737 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.83607 | 0.84282 | 0.87464 | 0.85728 | 0.83896 | 0.85439 | 0.84571 | 0.84860 | 0.88042 | 0.85632 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.83607 | 0.84282 | 0.87464 | 0.85728 | 0.83896 | 0.85439 | 0.84571 | 0.84860 | 0.88042 | 0.85632 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.46710 | -0.45072 | -0.38335 | -0.41021 | -0.44678 | -0.41951 | -0.43597 | -0.43628 | -0.36514 | -0.39142 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.46710 | -0.45072 | -0.38335 | -0.41021 | -0.44678 | -0.41951 | -0.43597 | -0.43628 | -0.36514 | -0.39142 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.73835 | 0.73756 | 0.77094 | 0.76082 | 0.73673 | 0.75695 | 0.75137 | 0.75154 | 0.78065 | 0.76563 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.73835 | 0.73756 | 0.77094 | 0.76082 | 0.73673 | 0.75695 | 0.75137 | 0.75154 | 0.78065 | 0.76563 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.32986 | 0.25652 | 0.28893 | 0.32201 | 0.28835 | 0.30500 | 0.30485 | 0.30542 | 0.30857 | 0.25631 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.34092 | 0.26506 | 0.29850 | 0.33274 | 0.29809 | 0.31531 | 0.31505 | 0.31569 | 0.31886 | 0.26487 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.22867 | 0.17785 | 0.20029 | 0.22322 | 0.19989 | 0.21144 | 0.21133 | 0.21175 | 0.21390 | 0.17767 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.23632 | 0.18375 | 0.20692 | 0.23067 | 0.20663 | 0.21857 | 0.21841 | 0.21884 | 0.22104 | 0.18360 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.85599 | 0.84436 | 0.82290 | 0.85510 | 0.84168 | 0.84794 | 0.82916 | 0.82826 | 0.81306 | 0.85331 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.85599 | 0.84436 | 0.82290 | 0.85510 | 0.84168 | 0.84794 | 0.82916 | 0.82826 | 0.81306 | 0.85331 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.36714 | -0.40198 | -0.42838 | -0.36877 | -0.39539 | -0.42377 | -0.45232 | -0.45092 | -0.48462 | -0.35740 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.36714 | -0.40198 | -0.42838 | -0.36877 | -0.39539 | -0.42377 | -0.45232 | -0.45092 | -0.48462 | -0.35740 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.83907 | 0.84440 | 0.82250 | 0.83906 | 0.83462 | 0.83945 | 0.83247 | 0.83278 | 0.82155 | 0.84649 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.83907 | 0.84440 | 0.82250 | 0.83906 | 0.83462 | 0.83945 | 0.83247 | 0.83278 | 0.82155 | 0.84649 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 0.98985 | 0.95851 | 0.89835 | 0.88502 | 0.85555 | 0.87437 | 0.90005 | 0.89999 | 0.89154 | 0.92116 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.6141 | 1.5747 | 1.4781 | 1.4633 | 1.3928 | 1.4264 | 1.4660 | 1.4655 | 1.4620 | 1.5129 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.72020 | 0.69794 | 0.65489 | 0.64680 | 0.62766 | 0.63797 | 0.65593 | 0.65571 | 0.65270 | 0.67189 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.2054 | 1.1774 | 1.1074 | 1.0985 | 1.0532 | 1.0684 | 1.0975 | 1.0969 | 1.1030 | 1.1334 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.68352 | 0.67603 | 0.69101 | 0.68539 | 0.68914 | 0.68352 | 0.65543 | 0.65356 | 0.70225 | 0.70599 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.68352 | 0.67603 | 0.69101 | 0.68539 | 0.68914 | 0.68352 | 0.65543 | 0.65356 | 0.70225 | 0.70599 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.84432 | -0.83214 | -0.78689 | -0.74799 | -0.76131 | -0.79582 | -0.83718 | -0.83701 | -0.75487 | -0.77978 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.84432 | -0.83214 | -0.78689 | -0.74799 | -0.76131 | -0.79582 | -0.83718 | -0.83701 | -0.75487 | -0.77978 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.59518 | 0.59581 | 0.62390 | 0.62335 | 0.62124 | 0.62025 | 0.59930 | 0.59893 | 0.62061 | 0.61532 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.59518 | 0.59581 | 0.62390 | 0.62335 | 0.62124 | 0.62025 | 0.59930 | 0.59893 | 0.62061 | 0.61532 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.7175 | 1.5914 | 1.5285 | 1.5502 | 1.6546 | 1.5512 | 1.7431 | 1.7458 | 1.6456 | 1.5818 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.3567 | 2.2039 | 2.1119 | 2.1455 | 2.2814 | 2.1601 | 2.4038 | 2.4079 | 2.2857 | 2.1850 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.1905 | 1.1030 | 1.0593 | 1.0745 | 1.1468 | 1.0752 | 1.2081 | 1.2102 | 1.1408 | 1.0965 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.6335 | 1.5275 | 1.4637 | 1.4872 | 1.5812 | 1.4974 | 1.6662 | 1.6691 | 1.5843 | 1.5145 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.21821 | 0.21247 | 0.25267 | 0.25923 | 0.23872 | 0.24610 | 0.22395 | 0.22395 | 0.23216 | 0.24118 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.21821 | 0.21247 | 0.25267 | 0.25923 | 0.23872 | 0.24610 | 0.22395 | 0.22395 | 0.23216 | 0.24118 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -4.1596 | -3.9739 | -3.8228 | -3.7518 | -3.7454 | -3.7597 | -4.0982 | -4.1034 | -3.9253 | -3.8668 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -4.1596 | -3.9739 | -3.8228 | -3.7518 | -3.7454 | -3.7597 | -4.0982 | -4.1034 | -3.9253 | -3.8668 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.24006 | 0.23724 | 0.26625 | 0.26950 | 0.25337 | 0.26785 | 0.24689 | 0.24712 | 0.25037 | 0.25393 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.24006 | 0.23724 | 0.26625 | 0.26950 | 0.25337 | 0.26785 | 0.24689 | 0.24712 | 0.25037 | 0.25393 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.46285 | 0.44589 | 0.43588 | 0.42593 | 0.43893 | 0.44276 | 0.45370 | 0.45336 | 0.44203 | 0.44337 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.46876 | 0.45167 | 0.44176 | 0.43146 | 0.44479 | 0.44865 | 0.45975 | 0.45935 | 0.44809 | 0.44929 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.65886 | 0.65264 | 0.63281 | 0.63346 | 0.64878 | 0.64523 | 0.66646 | 0.66629 | 0.63319 | 0.64948 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.66443 | 0.65814 | 0.63827 | 0.63903 | 0.65428 | 0.65077 | 0.67208 | 0.67204 | 0.63844 | 0.65547 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.7192 | 2.2330 | 2.1486 | 2.3348 | 2.5250 | 2.2116 | 2.4052 | 2.4057 | 2.3686 | 2.6246 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 5.4384 | 4.4659 | 4.2971 | 4.6697 | 5.0501 | 4.4231 | 4.8104 | 4.8114 | 4.7371 | 5.2492 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.8850 | 1.5482 | 1.4890 | 1.6184 | 1.7507 | 1.5325 | 1.6675 | 1.6670 | 1.6413 | 1.8193 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.7700 | 3.0965 | 2.9780 | 3.2368 | 3.5013 | 3.0651 | 3.3350 | 3.3340 | 3.2825 | 3.6386 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.07000 | 0.09000 | 0.09000 | 0.17000 | 0.10000 | 0.07000 | 0.09000 | 0.09000 | 0.11000 | 0.09000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.07000 | 0.09000 | 0.09000 | 0.17000 | 0.10000 | 0.07000 | 0.09000 | 0.09000 | 0.11000 | 0.09000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -3.7670 | -3.0907 | -2.9702 | -3.2321 | -3.4950 | -3.0599 | -3.3290 | -3.3288 | -3.2774 | -3.6333 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -3.7670 | -3.0907 | -2.9702 | -3.2321 | -3.4950 | -3.0599 | -3.3290 | -3.3288 | -3.2774 | -3.6333 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09507 | 0.09621 | 0.09547 | 0.09829 | 0.09806 | 0.09594 | 0.09336 | 0.09396 | 0.09591 | 0.09637 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09507 | 0.09621 | 0.09547 | 0.09829 | 0.09806 | 0.09594 | 0.09336 | 0.09396 | 0.09591 | 0.09637 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.81375 | 0.80900 | 0.79801 | 0.80011 | 0.80877 | 0.80670 | 0.81572 | 0.81576 | 0.79897 | 0.80413 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.82277 | 0.81788 | 0.80674 | 0.80875 | 0.81755 | 0.81551 | 0.82468 | 0.82472 | 0.80779 | 0.81318 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.70513 | 0.69569 | 0.66417 | 0.67297 | 0.68969 | 0.67964 | 0.70354 | 0.70346 | 0.67417 | 0.68445 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.70742 | 0.69811 | 0.66640 | 0.67524 | 0.69192 | 0.68196 | 0.70582 | 0.70579 | 0.67639 | 0.68661 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.72793 | 0.72039 | 0.70873 | 0.71233 | 0.72225 | 0.71707 | 0.73385 | 0.73425 | 0.70697 | 0.71672 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.76502 | 0.75725 | 0.74502 | 0.74875 | 0.75955 | 0.75420 | 0.77183 | 0.77215 | 0.74297 | 0.75316 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0166 | 1.0242 | 1.0081 | 1.0073 | 1.0099 | 1.0150 | 1.0092 | 1.0095 | 1.0107 | 1.0130 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0333 | 2.0483 | 2.0163 | 2.0145 | 2.0198 | 2.0300 | 2.0184 | 2.0190 | 2.0213 | 2.0260 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.70474 | 0.70996 | 0.69892 | 0.69830 | 0.70015 | 0.70362 | 0.69963 | 0.69984 | 0.70061 | 0.70223 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4095 | 1.4199 | 1.3978 | 1.3966 | 1.4003 | 1.4072 | 1.3993 | 1.3997 | 1.4012 | 1.4045 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24697 | 0.24272 | 0.25887 | 0.25292 | 0.25505 | 0.24867 | 0.25845 | 0.25887 | 0.25207 | 0.23847 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24697 | 0.24272 | 0.25887 | 0.25292 | 0.25505 | 0.24867 | 0.25845 | 0.25887 | 0.25207 | 0.23847 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3909 | -1.3934 | -1.3862 | -1.3870 | -1.3874 | -1.3914 | -1.3865 | -1.3866 | -1.3889 | -1.3897 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.4044 | -1.4134 | -1.3925 | -1.3924 | -1.3940 | -1.4036 | -1.3919 | -1.3921 | -1.3981 | -1.3995 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.24995 | 0.24982 | 0.25081 | 0.25042 | 0.25039 | 0.24964 | 0.25061 | 0.25059 | 0.25018 | 0.24994 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.24986 | 0.24953 | 0.25157 | 0.25080 | 0.25072 | 0.24929 | 0.25121 | 0.25116 | 0.25032 | 0.24985 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 1.0317 | 1.0007 | 0.97774 | 0.99400 | 1.0022 | 0.99328 | 1.0264 | 1.0256 | 0.98519 | 0.99605 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.1487 | 1.1146 | 1.0917 | 1.1082 | 1.1175 | 1.1063 | 1.1437 | 1.1426 | 1.0988 | 1.1094 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 0.99977 | 1.0158 | 1.0038 | 1.0053 | 1.0104 | 1.0065 | 1.0079 | 1.0072 | 1.0066 | 1.0099 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 1.9995 | 2.0315 | 2.0077 | 2.0107 | 2.0208 | 2.0130 | 2.0158 | 2.0144 | 2.0132 | 2.0199 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.69304 | 0.70412 | 0.69596 | 0.69697 | 0.70048 | 0.69771 | 0.69875 | 0.69828 | 0.69781 | 0.70012 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.3861 | 1.4082 | 1.3919 | 1.3939 | 1.4010 | 1.3954 | 1.3975 | 1.3966 | 1.3956 | 1.4002 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.27730 | 0.24399 | 0.27236 | 0.26897 | 0.26095 | 0.26681 | 0.26033 | 0.26188 | 0.26804 | 0.26342 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.27730 | 0.24399 | 0.27236 | 0.26897 | 0.26095 | 0.26681 | 0.26033 | 0.26188 | 0.26804 | 0.26342 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3796 | -1.3887 | -1.3837 | -1.3840 | -1.3876 | -1.3848 | -1.3847 | -1.3842 | -1.3857 | -1.3857 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.3806 | -1.4016 | -1.3858 | -1.3894 | -1.3945 | -1.3913 | -1.3898 | -1.3889 | -1.3921 | -1.3956 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25271 | 0.25068 | 0.25122 | 0.25154 | 0.25031 | 0.25137 | 0.25123 | 0.25137 | 0.25099 | 0.25147 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25551 | 0.25111 | 0.25241 | 0.25313 | 0.25051 | 0.25261 | 0.25236 | 0.25268 | 0.25191 | 0.25280 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.88194 | 0.85658 | 0.83636 | 0.85086 | 0.85660 | 0.85571 | 0.87337 | 0.87361 | 0.83739 | 0.85127 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 0.94149 | 0.91517 | 0.89416 | 0.90898 | 0.91534 | 0.91432 | 0.93400 | 0.93433 | 0.89553 | 0.90937 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0048 | 1.0328 | 1.0171 | 1.0131 | 1.0041 | 1.0191 | 1.0124 | 1.0120 | 1.0162 | 1.0153 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0096 | 2.0655 | 2.0342 | 2.0262 | 2.0083 | 2.0383 | 2.0248 | 2.0241 | 2.0325 | 2.0307 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.69655 | 0.71595 | 0.70515 | 0.70234 | 0.69617 | 0.70645 | 0.70184 | 0.70159 | 0.70450 | 0.70390 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.3931 | 1.4319 | 1.4103 | 1.4047 | 1.3923 | 1.4129 | 1.4037 | 1.4032 | 1.4090 | 1.4078 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.28859 | 0.22749 | 0.24862 | 0.24374 | 0.26779 | 0.25122 | 0.24472 | 0.23919 | 0.23334 | 0.25479 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.28859 | 0.22749 | 0.24862 | 0.24374 | 0.26779 | 0.25122 | 0.24472 | 0.23919 | 0.23334 | 0.25479 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3833 | -1.3998 | -1.3925 | -1.3897 | -1.3844 | -1.3927 | -1.3884 | -1.3882 | -1.3926 | -1.3912 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.3872 | -1.4244 | -1.4048 | -1.4000 | -1.3861 | -1.4094 | -1.3971 | -1.3967 | -1.4057 | -1.4034 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.25161 | 0.24812 | 0.24929 | 0.24999 | 0.25092 | 0.24977 | 0.25025 | 0.25033 | 0.24920 | 0.24973 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.25322 | 0.24631 | 0.24865 | 0.24990 | 0.25181 | 0.24967 | 0.25042 | 0.25057 | 0.24833 | 0.24949 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.3389 | 1.3021 | 1.2887 | 1.2860 | 1.3081 | 1.2862 | 1.3196 | 1.3199 | 1.2637 | 1.2966 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.6747 | 1.6292 | 1.6132 | 1.6043 | 1.6374 | 1.6062 | 1.6496 | 1.6500 | 1.5777 | 1.6222 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0053 | 1.0273 | 1.0054 | 1.0097 | 1.0088 | 1.0263 | 1.0039 | 1.0026 | 1.0173 | 1.0209 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0106 | 2.0546 | 2.0108 | 2.0194 | 2.0175 | 2.0525 | 2.0078 | 2.0053 | 2.0346 | 2.0419 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.69690 | 0.71213 | 0.69699 | 0.69994 | 0.69935 | 0.71140 | 0.69594 | 0.69507 | 0.70522 | 0.70772 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.3938 | 1.4243 | 1.3940 | 1.3999 | 1.3987 | 1.4228 | 1.3919 | 1.3901 | 1.4104 | 1.4154 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.28396 | 0.23923 | 0.27005 | 0.25580 | 0.26110 | 0.24486 | 0.28131 | 0.28131 | 0.23757 | 0.25249 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.28396 | 0.23923 | 0.27005 | 0.25580 | 0.26110 | 0.24486 | 0.28131 | 0.28131 | 0.23757 | 0.25249 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3814 | -1.3955 | -1.3828 | -1.3865 | -1.3852 | -1.3956 | -1.3808 | -1.3800 | -1.3898 | -1.3926 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.3877 | -1.4162 | -1.3873 | -1.3958 | -1.3908 | -1.4189 | -1.3821 | -1.3804 | -1.4058 | -1.4107 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.25277 | 0.24922 | 0.25193 | 0.25108 | 0.25113 | 0.24961 | 0.25228 | 0.25246 | 0.25061 | 0.24997 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.25572 | 0.24851 | 0.25387 | 0.25212 | 0.25226 | 0.24932 | 0.25463 | 0.25496 | 0.25111 | 0.24993 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.43820 | 0.43436 | 0.41978 | 0.41585 | 0.43021 | 0.42769 | 0.43148 | 0.43155 | 0.43511 | 0.43526 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.44074 | 0.43684 | 0.42219 | 0.41817 | 0.43268 | 0.43008 | 0.43389 | 0.43392 | 0.43753 | 0.43777 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.34413 | 0.34051 | 0.32435 | 0.30700 | 0.33076 | 0.32617 | 0.34602 | 0.34622 | 0.33609 | 0.33488 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.34547 | 0.34180 | 0.32565 | 0.30816 | 0.33207 | 0.32747 | 0.34750 | 0.34767 | 0.33737 | 0.33619 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.56737 | 0.63186 | 0.56634 | 0.54484 | 0.55892 | 0.54885 | 0.58117 | 0.58163 | 0.56116 | 0.58557 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.57131 | 0.63637 | 0.57041 | 0.54861 | 0.56288 | 0.55266 | 0.58520 | 0.58566 | 0.56505 | 0.58970 |
| eval/lm/c4_en-validation/CE loss | lower | 3.0030 | 2.9735 | 2.9253 | 2.9417 | 2.9751 | 2.9613 | 3.0012 | 3.0011 | 2.9324 | 2.9572 |
| eval/lm/c4_en-validation/PPL | lower | 20.15 | 19.56 | 18.64 | 18.95 | 19.59 | 19.32 | 20.11 | 20.11 | 18.77 | 19.24 |
| eval/lm/dolma_books-validation/CE loss | lower | 2.9053 | 2.8738 | 2.8149 | 2.8333 | 2.8726 | 2.8552 | 2.9048 | 2.9044 | 2.8205 | 2.8476 |
| eval/lm/dolma_books-validation/PPL | lower | 18.27 | 17.70 | 16.69 | 17.00 | 17.68 | 17.38 | 18.26 | 18.25 | 16.79 | 17.25 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.1385 | 3.1097 | 3.0644 | 3.0804 | 3.1142 | 3.1003 | 3.1387 | 3.1387 | 3.0704 | 3.0943 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 23.07 | 22.42 | 21.42 | 21.77 | 22.51 | 22.20 | 23.07 | 23.07 | 21.55 | 22.07 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.1910 | 2.1661 | 2.1321 | 2.1454 | 2.1665 | 2.1532 | 2.1893 | 2.1894 | 2.1435 | 2.1561 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 8.9439 | 8.7243 | 8.4323 | 8.5451 | 8.7276 | 8.6126 | 8.9293 | 8.9295 | 8.5289 | 8.6374 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.3097 | 3.2830 | 3.2380 | 3.2555 | 3.2874 | 3.2728 | 3.3088 | 3.3087 | 3.2504 | 3.2703 |
| eval/lm/dolma_reddit-validation/PPL | lower | 27.38 | 26.66 | 25.48 | 25.93 | 26.77 | 26.38 | 27.35 | 27.35 | 25.80 | 26.32 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.3818 | 1.3618 | 1.3279 | 1.3400 | 1.3635 | 1.3512 | 1.3820 | 1.3820 | 1.3320 | 1.3497 |
| eval/lm/dolma_stack-validation/PPL | lower | 3.9820 | 3.9031 | 3.7732 | 3.8190 | 3.9100 | 3.8622 | 3.9829 | 3.9828 | 3.7885 | 3.8561 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.6470 | 2.6246 | 2.5735 | 2.5840 | 2.6207 | 2.6109 | 2.6509 | 2.6509 | 2.5722 | 2.6027 |
| eval/lm/dolma_wiki-validation/PPL | lower | 14.11 | 13.80 | 13.11 | 13.25 | 13.75 | 13.61 | 14.17 | 14.17 | 13.09 | 13.50 |
| eval/lm/ice-validation/CE loss | lower | 3.0721 | 3.0415 | 2.9978 | 3.0294 | 3.0660 | 3.0306 | 3.0930 | 3.0931 | 3.0407 | 3.0319 |
| eval/lm/ice-validation/PPL | lower | 21.59 | 20.94 | 20.04 | 20.68 | 21.46 | 20.71 | 22.04 | 22.05 | 20.92 | 20.74 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 3.1218 | 3.0895 | 3.0483 | 3.0577 | 3.0996 | 3.0888 | 3.1220 | 3.1221 | 3.0594 | 3.0738 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 22.69 | 21.97 | 21.08 | 21.28 | 22.19 | 21.95 | 22.69 | 22.69 | 21.31 | 21.62 |
| eval/lm/pile-validation/CE loss | lower | 2.2761 | 2.2512 | 2.2089 | 2.2225 | 2.2506 | 2.2372 | 2.2768 | 2.2767 | 2.2169 | 2.2345 |
| eval/lm/pile-validation/PPL | lower | 9.7382 | 9.4995 | 9.1053 | 9.2304 | 9.4936 | 9.3671 | 9.7456 | 9.7449 | 9.1789 | 9.3419 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.6041 | 2.5728 | 2.5264 | 2.5326 | 2.5819 | 2.5594 | 2.6043 | 2.6046 | 2.5277 | 2.5476 |
| eval/lm/wikitext_103-validation/PPL | lower | 13.52 | 13.10 | 12.51 | 12.59 | 13.22 | 12.93 | 13.52 | 13.53 | 12.52 | 12.78 |
| throughput/in-loop eval batches | see metric | 826.0 | 826.0 | 1645.0 | 1645.0 | 826.0 | 826.0 | 826.0 | 826.0 | 1645.0 | 826.0 |
| throughput/in-loop eval time (s) | see metric | 115.5 | 145.3 | 139.7 | 123.9 | 126.5 | 130.7 | 261.3 | 132.4 | 130.1 | 119.2 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| eg-480m-cx4-eg24e2k-lr8e-4-r1<br>`wq8gib5l` | finished | original | 30402412544.0 | 57988 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wq8gib5l) |
| eg-480m-cx4-eg96e8k-lr8e-4-r1<br>`ezokso90` | finished | original | 30490492928.0 | 58156 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ezokso90) |
| int-480m-cx4-intd256e8k-lr8e-4-r1<br>`zf909hyq` | finished | original | 30929846272.0 | 58994 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/zf909hyq) |
| int-480m-cx4-intw256e8k-lr8e-4-r1<br>`rblv9hpr` | finished | original | 30687100928.0 | 58531 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rblv9hpr) |
| q3-480m-cx4-q3am128e8k-lr8e-4-r1<br>`v7vgfj0v` | finished | original | 30540300288.0 | 58251 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/v7vgfj0v) |
| q3-480m-cx4-q3td128e8k-lr8e-4-r1<br>`umqcq7bm` | finished | original | 30605312000.0 | 58375 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/umqcq7bm) |
| se-480m-cx4-se0m9-lr8e-4-r1<br>`1j4l0j4h` | finished | original |  | 58044 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1j4l0j4h) |
| se-480m-cx4-se0m9-lr8e-4-r1<br>`7tyd6xrj` | finished | original | 30431772672.0 | 58044 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7tyd6xrj) |
| sp-480m-cx4-sp192e4k-lr6e-4-r2<br>`drpw4m1b` | finished | original | 30608457728.0 | 58381 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/drpw4m1b) |
| sp-480m-cx4-sp96e4k-lr7e-4-r1<br>`r6dfpalm` | finished | original | 30490492928.0 | 58156 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/r6dfpalm) |

## 480m Cx8

| metric | direction | eg-480m-cx8-eg24e2k-lr8e-4-r1<br>`epx7o7ty` | eg-480m-cx8-eg96e8k-lr8e-4-r1<br>`8676ezla` | int-480m-cx8-intd256e8k-lr8e-4-r1<br>`3q0hdi34` | int-480m-cx8-intw256e8k-lr8e-4-r1<br>`vdcrgfy0` | q3-480m-cx8-q3am128e8k-lr8e-4-r1<br>`3xqcp7if` | q3-480m-cx8-q3td128e8k-lr8e-4-r1<br>`t3bexkgy` | se-480m-cx8-se0m9-lr8e-4-r1<br>`e84abrdp` | sp-480m-cx8-sp192e4k-lr6e-4-r1<br>`ysglhwbv` | sp-480m-cx8-sp96e4k-lr7e-4-r1<br>`xek7xsp1` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.86227 | 0.86063 | 0.83285 | 0.82949 | 0.85577 | 0.85429 | 0.85153 | 0.81783 | 0.83730 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 0.94218 | 0.94388 | 0.91042 | 0.90839 | 0.93774 | 0.93379 | 0.92998 | 0.89358 | 0.91645 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0090 | 1.0183 | 1.0055 | 1.0241 | 1.0073 | 1.0111 | 1.0123 | 1.0060 | 1.0077 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0180 | 2.0367 | 2.0109 | 2.0482 | 2.0146 | 2.0222 | 2.0246 | 2.0121 | 2.0153 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.69945 | 0.70586 | 0.69703 | 0.70986 | 0.69829 | 0.70094 | 0.70170 | 0.69746 | 0.69853 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.3989 | 1.4117 | 1.3941 | 1.4197 | 1.3966 | 1.4019 | 1.4034 | 1.3949 | 1.3971 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.26195 | 0.26451 | 0.25853 | 0.23038 | 0.26962 | 0.26451 | 0.26195 | 0.24488 | 0.26621 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.26195 | 0.26451 | 0.25853 | 0.23038 | 0.26962 | 0.26451 | 0.26195 | 0.24488 | 0.26621 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.3930 | -1.4072 | -1.3907 | -1.4151 | -1.3940 | -1.3989 | -1.4004 | -1.3919 | -1.3950 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.3930 | -1.4072 | -1.3907 | -1.4151 | -1.3940 | -1.3989 | -1.4004 | -1.3919 | -1.3950 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.25178 | 0.25425 | 0.25116 | 0.24922 | 0.25220 | 0.25269 | 0.25244 | 0.25010 | 0.25404 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.25178 | 0.25425 | 0.25116 | 0.24922 | 0.25220 | 0.25269 | 0.25244 | 0.25010 | 0.25404 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.66913 | 0.66030 | 0.62752 | 0.62308 | 0.63446 | 0.65021 | 0.65854 | 0.62014 | 0.63998 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.72788 | 0.71870 | 0.68185 | 0.67707 | 0.68985 | 0.70756 | 0.71704 | 0.67314 | 0.69603 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0181 | 1.0390 | 1.0195 | 1.0219 | 1.0124 | 1.0439 | 1.0248 | 1.0090 | 1.0252 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0362 | 2.0779 | 2.0390 | 2.0439 | 2.0248 | 2.0877 | 2.0495 | 2.0180 | 2.0504 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.70576 | 0.72017 | 0.70671 | 0.70846 | 0.70184 | 0.72361 | 0.71034 | 0.69946 | 0.71069 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4115 | 1.4403 | 1.4134 | 1.4169 | 1.4037 | 1.4472 | 1.4207 | 1.3989 | 1.4214 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.24495 | 0.24579 | 0.26052 | 0.25168 | 0.23779 | 0.24579 | 0.25210 | 0.24116 | 0.24663 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.24495 | 0.24579 | 0.26052 | 0.25168 | 0.23779 | 0.24579 | 0.25210 | 0.24116 | 0.24663 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.4057 | -1.4355 | -1.4099 | -1.4130 | -1.4013 | -1.4443 | -1.4180 | -1.3957 | -1.4195 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.4057 | -1.4355 | -1.4099 | -1.4130 | -1.4013 | -1.4443 | -1.4180 | -1.3957 | -1.4195 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.24989 | 0.24966 | 0.24995 | 0.25104 | 0.25046 | 0.24829 | 0.24928 | 0.24986 | 0.24958 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.24989 | 0.24966 | 0.24995 | 0.25104 | 0.25046 | 0.24829 | 0.24928 | 0.24986 | 0.24958 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 1.3303 | 1.3611 | 1.1688 | 1.0595 | 1.2934 | 1.1961 | 1.3117 | 1.1418 | 1.2507 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 2.1361 | 2.1703 | 1.8916 | 1.7131 | 2.0947 | 1.9284 | 2.0971 | 1.8459 | 2.0283 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 0.92219 | 0.94347 | 0.81021 | 0.73422 | 0.89650 | 0.82899 | 0.90917 | 0.79139 | 0.86682 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 1.4807 | 1.5043 | 1.3112 | 1.1873 | 1.4521 | 1.3366 | 1.4534 | 1.2795 | 1.4059 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.44508 | 0.43553 | 0.53200 | 0.51958 | 0.46705 | 0.50907 | 0.48711 | 0.52340 | 0.48520 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.44508 | 0.43553 | 0.53200 | 0.51958 | 0.46705 | 0.50907 | 0.48711 | 0.52340 | 0.48520 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -1.7686 | -1.7603 | -1.4320 | -1.3734 | -1.5563 | -1.5136 | -1.7850 | -1.4402 | -1.5996 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -1.7686 | -1.7603 | -1.4320 | -1.3734 | -1.5563 | -1.5136 | -1.7850 | -1.4402 | -1.5996 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.39726 | 0.39705 | 0.47047 | 0.49257 | 0.42134 | 0.45595 | 0.44584 | 0.47997 | 0.44512 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.39726 | 0.39705 | 0.47047 | 0.49257 | 0.42134 | 0.45595 | 0.44584 | 0.47997 | 0.44512 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.43511 | 0.38379 | 0.38116 | 0.34080 | 0.39316 | 0.39186 | 0.42769 | 0.38590 | 0.40015 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.47501 | 0.41820 | 0.41581 | 0.37110 | 0.42856 | 0.42719 | 0.46818 | 0.42117 | 0.43669 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.30163 | 0.26602 | 0.26422 | 0.23622 | 0.27251 | 0.27160 | 0.29649 | 0.26748 | 0.27738 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.32925 | 0.28989 | 0.28825 | 0.25725 | 0.29706 | 0.29612 | 0.32451 | 0.29194 | 0.30265 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.56423 | 0.58696 | 0.62648 | 0.61166 | 0.58103 | 0.62648 | 0.51976 | 0.63439 | 0.61462 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.56423 | 0.58696 | 0.62648 | 0.61166 | 0.58103 | 0.62648 | 0.51976 | 0.63439 | 0.61462 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -1.7381 | -1.6730 | -1.5886 | -1.5324 | -1.7259 | -1.5770 | -1.8900 | -1.5027 | -1.5327 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -1.7381 | -1.6730 | -1.5886 | -1.5324 | -1.7259 | -1.5770 | -1.8900 | -1.5027 | -1.5327 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.54161 | 0.57065 | 0.60067 | 0.60165 | 0.56298 | 0.59512 | 0.52441 | 0.60553 | 0.58660 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.54161 | 0.57065 | 0.60067 | 0.60165 | 0.56298 | 0.59512 | 0.52441 | 0.60553 | 0.58660 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.45127 | 0.40062 | 0.35923 | 0.34911 | 0.47575 | 0.37852 | 0.46264 | 0.37449 | 0.44102 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.54380 | 0.48223 | 0.43422 | 0.42047 | 0.57172 | 0.45578 | 0.55620 | 0.44984 | 0.53078 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.31292 | 0.27780 | 0.24910 | 0.24214 | 0.32984 | 0.26252 | 0.32088 | 0.25970 | 0.30582 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.37711 | 0.33444 | 0.30112 | 0.29163 | 0.39649 | 0.31614 | 0.38577 | 0.31200 | 0.36817 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.84667 | 0.88621 | 0.87753 | 0.89585 | 0.85342 | 0.90260 | 0.86017 | 0.90453 | 0.89875 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.84667 | 0.88621 | 0.87753 | 0.89585 | 0.85342 | 0.90260 | 0.86017 | 0.90453 | 0.89875 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.42006 | -0.34495 | -0.31788 | -0.29578 | -0.41758 | -0.33171 | -0.37648 | -0.29144 | -0.31217 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.42006 | -0.34495 | -0.31788 | -0.29578 | -0.41758 | -0.33171 | -0.37648 | -0.29144 | -0.31217 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.75091 | 0.78460 | 0.80880 | 0.81539 | 0.75940 | 0.79020 | 0.77201 | 0.81414 | 0.79781 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.75091 | 0.78460 | 0.80880 | 0.81539 | 0.75940 | 0.79020 | 0.77201 | 0.81414 | 0.79781 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.26011 | 0.26633 | 0.27295 | 0.28985 | 0.28208 | 0.22959 | 0.26934 | 0.25847 | 0.28923 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.26875 | 0.27532 | 0.28205 | 0.29962 | 0.29158 | 0.23721 | 0.27849 | 0.26707 | 0.29900 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.18031 | 0.18464 | 0.18921 | 0.20094 | 0.19554 | 0.15916 | 0.18671 | 0.17916 | 0.20052 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.18632 | 0.19086 | 0.19553 | 0.20770 | 0.20213 | 0.16443 | 0.19304 | 0.18513 | 0.20727 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.81306 | 0.86225 | 0.86315 | 0.89088 | 0.83095 | 0.87925 | 0.83989 | 0.85063 | 0.87478 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.81306 | 0.86225 | 0.86315 | 0.89088 | 0.83095 | 0.87925 | 0.83989 | 0.85063 | 0.87478 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.47401 | -0.34188 | -0.33206 | -0.27672 | -0.38932 | -0.29874 | -0.43389 | -0.35379 | -0.31510 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.47401 | -0.34188 | -0.33206 | -0.27672 | -0.38932 | -0.29874 | -0.43389 | -0.35379 | -0.31510 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.82559 | 0.85745 | 0.85310 | 0.86996 | 0.83726 | 0.87428 | 0.84148 | 0.85355 | 0.86011 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.82559 | 0.85745 | 0.85310 | 0.86996 | 0.83726 | 0.87428 | 0.84148 | 0.85355 | 0.86011 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 0.79543 | 0.77953 | 0.87799 | 0.84724 | 0.83618 | 0.84700 | 0.83272 | 0.82651 | 0.82937 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.3093 | 1.2797 | 1.4472 | 1.3744 | 1.3789 | 1.3853 | 1.3683 | 1.3690 | 1.3636 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.57921 | 0.57217 | 0.64222 | 0.61644 | 0.60919 | 0.61933 | 0.60752 | 0.59960 | 0.60297 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 0.97779 | 0.96854 | 1.0892 | 1.0274 | 1.0299 | 1.0427 | 1.0261 | 1.0175 | 1.0164 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.71348 | 0.71348 | 0.72097 | 0.73221 | 0.74345 | 0.71910 | 0.72097 | 0.73221 | 0.73408 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.71348 | 0.71348 | 0.72097 | 0.73221 | 0.74345 | 0.71910 | 0.72097 | 0.73221 | 0.73408 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.69894 | -0.70271 | -0.71341 | -0.67436 | -0.73888 | -0.69566 | -0.69944 | -0.67496 | -0.69871 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.69894 | -0.70271 | -0.71341 | -0.67436 | -0.73888 | -0.69566 | -0.69944 | -0.67496 | -0.69871 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.65375 | 0.64068 | 0.63452 | 0.65317 | 0.63769 | 0.64649 | 0.65050 | 0.65511 | 0.64891 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.65375 | 0.64068 | 0.63452 | 0.65317 | 0.63769 | 0.64649 | 0.65050 | 0.65511 | 0.64891 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.6002 | 1.4447 | 1.3912 | 1.4284 | 1.5647 | 1.4430 | 1.6046 | 1.3729 | 1.4916 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.2276 | 2.0016 | 1.9677 | 1.9813 | 2.1863 | 2.0011 | 2.2529 | 1.9264 | 2.0846 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.1091 | 1.0014 | 0.96438 | 0.99008 | 1.0845 | 1.0002 | 1.1122 | 0.95151 | 1.0339 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.5438 | 1.3874 | 1.3639 | 1.3734 | 1.5153 | 1.3870 | 1.5616 | 1.3351 | 1.4450 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.25267 | 0.27235 | 0.29368 | 0.31993 | 0.26907 | 0.28712 | 0.24036 | 0.29286 | 0.24282 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.25267 | 0.27235 | 0.29368 | 0.31993 | 0.26907 | 0.28712 | 0.24036 | 0.29286 | 0.24282 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -3.6270 | -3.5136 | -3.0279 | -3.1258 | -3.6053 | -3.2156 | -3.7845 | -3.0992 | -3.5311 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -3.6270 | -3.5136 | -3.0279 | -3.1258 | -3.6053 | -3.2156 | -3.7845 | -3.0992 | -3.5311 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.26666 | 0.28669 | 0.30413 | 0.31698 | 0.28229 | 0.29770 | 0.26552 | 0.31040 | 0.26734 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.26666 | 0.28669 | 0.30413 | 0.31698 | 0.28229 | 0.29770 | 0.26552 | 0.31040 | 0.26734 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.42818 | 0.42215 | 0.40496 | 0.40961 | 0.41471 | 0.42522 | 0.42030 | 0.40247 | 0.40742 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.43391 | 0.42777 | 0.41033 | 0.41534 | 0.42009 | 0.43082 | 0.42560 | 0.40775 | 0.41260 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.63073 | 0.62103 | 0.59755 | 0.61327 | 0.62062 | 0.61477 | 0.63289 | 0.60333 | 0.62224 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.63618 | 0.62652 | 0.60284 | 0.61846 | 0.62605 | 0.62002 | 0.63838 | 0.60847 | 0.62753 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.3434 | 1.8227 | 2.0100 | 2.3239 | 2.3596 | 2.4921 | 2.2211 | 2.0107 | 2.1347 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 4.6868 | 3.6455 | 4.0200 | 4.6478 | 4.7193 | 4.9842 | 4.4423 | 4.0213 | 4.2694 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.6239 | 1.2637 | 1.3936 | 1.6113 | 1.6354 | 1.7273 | 1.5396 | 1.3936 | 1.4797 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.2477 | 2.5273 | 2.7873 | 3.2226 | 3.2708 | 3.4546 | 3.0793 | 2.7871 | 2.9595 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.10000 | 0.07000 | 0.11000 | 0.08000 | 0.12000 | 0.10000 | 0.06000 | 0.10000 | 0.10000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.10000 | 0.07000 | 0.11000 | 0.08000 | 0.12000 | 0.10000 | 0.06000 | 0.10000 | 0.10000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -3.2421 | -2.5152 | -2.7821 | -3.2130 | -3.2623 | -3.4478 | -3.0733 | -2.7808 | -2.9496 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -3.2421 | -2.5152 | -2.7821 | -3.2130 | -3.2623 | -3.4478 | -3.0733 | -2.7808 | -2.9496 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09947 | 0.09840 | 0.09608 | 0.09499 | 0.09049 | 0.09190 | 0.09388 | 0.09695 | 0.09742 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09947 | 0.09840 | 0.09608 | 0.09499 | 0.09049 | 0.09190 | 0.09388 | 0.09695 | 0.09742 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.79756 | 0.79074 | 0.77899 | 0.77986 | 0.78628 | 0.78853 | 0.79656 | 0.77471 | 0.78294 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.80623 | 0.79947 | 0.78761 | 0.78827 | 0.79502 | 0.79727 | 0.80525 | 0.78328 | 0.79156 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.67079 | 0.65471 | 0.63059 | 0.63380 | 0.64839 | 0.64584 | 0.66579 | 0.62634 | 0.64141 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.67297 | 0.65672 | 0.63257 | 0.63588 | 0.65046 | 0.64797 | 0.66795 | 0.62841 | 0.64347 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.70422 | 0.70342 | 0.67762 | 0.67754 | 0.69758 | 0.69123 | 0.70251 | 0.67692 | 0.68423 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.73944 | 0.73905 | 0.71155 | 0.71200 | 0.73317 | 0.72584 | 0.73825 | 0.71049 | 0.71861 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0113 | 1.0232 | 1.0119 | 1.0215 | 1.0094 | 1.0158 | 1.0103 | 1.0077 | 1.0198 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0226 | 2.0464 | 2.0238 | 2.0430 | 2.0187 | 2.0316 | 2.0206 | 2.0154 | 2.0397 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.70110 | 0.70930 | 0.70153 | 0.70806 | 0.69975 | 0.70417 | 0.70037 | 0.69860 | 0.70699 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4022 | 1.4186 | 1.4031 | 1.4161 | 1.3995 | 1.4083 | 1.4007 | 1.3972 | 1.4140 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24442 | 0.24888 | 0.25484 | 0.23974 | 0.24060 | 0.24187 | 0.26057 | 0.25122 | 0.24548 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24442 | 0.24888 | 0.25484 | 0.23974 | 0.24060 | 0.24187 | 0.26057 | 0.25122 | 0.24548 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3889 | -1.3920 | -1.3897 | -1.3924 | -1.3893 | -1.3908 | -1.3874 | -1.3877 | -1.3929 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.3955 | -1.4143 | -1.3990 | -1.4122 | -1.3961 | -1.4043 | -1.3967 | -1.3934 | -1.4113 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.24987 | 0.25054 | 0.24990 | 0.25023 | 0.24974 | 0.24996 | 0.25077 | 0.25021 | 0.24977 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.24976 | 0.25092 | 0.24985 | 0.25030 | 0.24949 | 0.24989 | 0.25151 | 0.25046 | 0.24953 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 0.97593 | 0.97446 | 0.92669 | 0.93022 | 0.96778 | 0.94983 | 0.98128 | 0.92371 | 0.94089 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.0878 | 1.0872 | 1.0340 | 1.0372 | 1.0796 | 1.0574 | 1.0950 | 1.0287 | 1.0475 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0022 | 1.0156 | 1.0083 | 1.0239 | 1.0011 | 1.0075 | 1.0016 | 1.0023 | 1.0096 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0045 | 2.0312 | 2.0166 | 2.0478 | 2.0021 | 2.0149 | 2.0033 | 2.0046 | 2.0192 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.69480 | 0.70400 | 0.69902 | 0.70979 | 0.69397 | 0.69843 | 0.69432 | 0.69483 | 0.69988 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.3896 | 1.4080 | 1.3980 | 1.4196 | 1.3879 | 1.3969 | 1.3886 | 1.3897 | 1.3998 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.28470 | 0.27267 | 0.26126 | 0.24306 | 0.28532 | 0.26835 | 0.28408 | 0.26774 | 0.27175 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.28470 | 0.27267 | 0.26126 | 0.24306 | 0.28532 | 0.26835 | 0.28408 | 0.26774 | 0.27175 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3821 | -1.3849 | -1.3867 | -1.3918 | -1.3828 | -1.3854 | -1.3806 | -1.3837 | -1.3849 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.3828 | -1.4033 | -1.3943 | -1.4150 | -1.3848 | -1.3930 | -1.3851 | -1.3859 | -1.3973 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25171 | 0.25278 | 0.25077 | 0.25086 | 0.25159 | 0.25129 | 0.25274 | 0.25124 | 0.25206 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25346 | 0.25545 | 0.25148 | 0.25148 | 0.25324 | 0.25262 | 0.25544 | 0.25249 | 0.25418 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.83559 | 0.82843 | 0.79873 | 0.79993 | 0.81552 | 0.81463 | 0.83301 | 0.79291 | 0.81063 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 0.89210 | 0.88439 | 0.85272 | 0.85454 | 0.87024 | 0.87073 | 0.89040 | 0.84690 | 0.86551 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0073 | 1.0247 | 1.0153 | 1.0495 | 1.0060 | 1.0126 | 1.0058 | 1.0008 | 1.0129 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0145 | 2.0494 | 2.0307 | 2.0989 | 2.0121 | 2.0253 | 2.0116 | 2.0015 | 2.0259 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.69830 | 0.71035 | 0.70385 | 0.72744 | 0.69745 | 0.70199 | 0.69725 | 0.69380 | 0.70222 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.3966 | 1.4207 | 1.4077 | 1.4549 | 1.3949 | 1.4040 | 1.3945 | 1.3876 | 1.4044 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.25317 | 0.24212 | 0.24537 | 0.21839 | 0.24634 | 0.25902 | 0.27072 | 0.27169 | 0.24634 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.25317 | 0.24212 | 0.24537 | 0.21839 | 0.24634 | 0.25902 | 0.27072 | 0.27169 | 0.24634 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3864 | -1.3936 | -1.3909 | -1.4097 | -1.3869 | -1.3894 | -1.3830 | -1.3833 | -1.3891 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.3902 | -1.4157 | -1.4040 | -1.4504 | -1.3922 | -1.3996 | -1.3913 | -1.3846 | -1.4021 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.25044 | 0.24994 | 0.24989 | 0.24661 | 0.25040 | 0.25008 | 0.25226 | 0.25128 | 0.25052 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.25086 | 0.24977 | 0.24971 | 0.24354 | 0.25075 | 0.25015 | 0.25440 | 0.25254 | 0.25093 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.2733 | 1.2523 | 1.2138 | 1.2159 | 1.2507 | 1.2387 | 1.2857 | 1.2221 | 1.2260 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.5912 | 1.5655 | 1.5162 | 1.5188 | 1.5672 | 1.5460 | 1.6129 | 1.5297 | 1.5315 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0066 | 1.0276 | 1.0251 | 1.0400 | 1.0100 | 1.0151 | 1.0108 | 1.0073 | 1.0147 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0131 | 2.0552 | 2.0502 | 2.0800 | 2.0199 | 2.0302 | 2.0216 | 2.0147 | 2.0294 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.69782 | 0.71236 | 0.71066 | 0.72088 | 0.70011 | 0.70368 | 0.70071 | 0.69833 | 0.70340 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.3956 | 1.4247 | 1.4213 | 1.4418 | 1.4002 | 1.4074 | 1.4014 | 1.3967 | 1.4068 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.26574 | 0.24652 | 0.25050 | 0.22929 | 0.24420 | 0.25381 | 0.25646 | 0.25083 | 0.24486 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.26574 | 0.24652 | 0.25050 | 0.22929 | 0.24420 | 0.25381 | 0.25646 | 0.25083 | 0.24486 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3848 | -1.3927 | -1.3962 | -1.4023 | -1.3879 | -1.3891 | -1.3863 | -1.3866 | -1.3878 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.3880 | -1.4187 | -1.4174 | -1.4368 | -1.3967 | -1.4029 | -1.3973 | -1.3932 | -1.4040 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.25098 | 0.25082 | 0.24899 | 0.24859 | 0.25051 | 0.25062 | 0.25135 | 0.25075 | 0.25136 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.25194 | 0.25149 | 0.24810 | 0.24736 | 0.25102 | 0.25122 | 0.25257 | 0.25153 | 0.25249 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.40155 | 0.41174 | 0.38954 | 0.40150 | 0.40514 | 0.39189 | 0.43859 | 0.39572 | 0.40782 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.40387 | 0.41406 | 0.39173 | 0.40380 | 0.40734 | 0.39409 | 0.44109 | 0.39799 | 0.41009 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.32399 | 0.31280 | 0.31074 | 0.30551 | 0.31422 | 0.30095 | 0.33144 | 0.30004 | 0.32409 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.32521 | 0.31392 | 0.31193 | 0.30662 | 0.31545 | 0.30222 | 0.33274 | 0.30120 | 0.32528 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.54682 | 0.56214 | 0.52950 | 0.52404 | 0.53798 | 0.50594 | 0.58479 | 0.51482 | 0.53035 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.55051 | 0.56610 | 0.53333 | 0.52794 | 0.54180 | 0.50949 | 0.58900 | 0.51848 | 0.53401 |
| eval/lm/c4_en-validation/CE loss | lower | 2.9376 | 2.9124 | 2.8505 | 2.8683 | 2.9056 | 2.8870 | 2.9341 | 2.8530 | 2.8855 |
| eval/lm/c4_en-validation/PPL | lower | 18.87 | 18.40 | 17.30 | 17.61 | 18.28 | 17.94 | 18.80 | 17.34 | 17.91 |
| eval/lm/dolma_books-validation/CE loss | lower | 2.8296 | 2.7902 | 2.7226 | 2.7385 | 2.7852 | 2.7615 | 2.8176 | 2.7267 | 2.7618 |
| eval/lm/dolma_books-validation/PPL | lower | 16.94 | 16.28 | 15.22 | 15.46 | 16.20 | 15.82 | 16.74 | 15.28 | 15.83 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.0750 | 3.0477 | 2.9884 | 3.0061 | 3.0441 | 3.0247 | 3.0723 | 2.9884 | 3.0213 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 21.65 | 21.07 | 19.85 | 20.21 | 20.99 | 20.59 | 21.59 | 19.85 | 20.52 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.1384 | 2.1136 | 2.0683 | 2.0822 | 2.1080 | 2.0919 | 2.1319 | 2.0779 | 2.0959 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 8.4860 | 8.2780 | 7.9110 | 8.0224 | 8.2318 | 8.1002 | 8.4310 | 7.9873 | 8.1328 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.2529 | 3.2256 | 3.1734 | 3.1867 | 3.2240 | 3.2062 | 3.2466 | 3.1751 | 3.2028 |
| eval/lm/dolma_reddit-validation/PPL | lower | 25.87 | 25.17 | 23.89 | 24.21 | 25.13 | 24.68 | 25.70 | 23.93 | 24.60 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.3133 | 1.2936 | 1.2504 | 1.2623 | 1.2905 | 1.2800 | 1.3106 | 1.2540 | 1.2736 |
| eval/lm/dolma_stack-validation/PPL | lower | 3.7184 | 3.6459 | 3.4918 | 3.5336 | 3.6348 | 3.5966 | 3.7085 | 3.5045 | 3.5738 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.5748 | 2.5509 | 2.4884 | 2.4991 | 2.5431 | 2.5278 | 2.5702 | 2.4816 | 2.5205 |
| eval/lm/dolma_wiki-validation/PPL | lower | 13.13 | 12.82 | 12.04 | 12.17 | 12.72 | 12.53 | 13.07 | 11.96 | 12.43 |
| eval/lm/ice-validation/CE loss | lower | 3.0087 | 2.9729 | 2.9236 | 2.9384 | 2.9748 | 2.9738 | 3.0188 | 2.9485 | 2.9554 |
| eval/lm/ice-validation/PPL | lower | 20.26 | 19.55 | 18.61 | 18.89 | 19.59 | 19.57 | 20.47 | 19.08 | 19.21 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 3.0628 | 3.0340 | 2.9921 | 2.9870 | 3.0352 | 3.0128 | 3.0456 | 2.9764 | 2.9989 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 21.39 | 20.78 | 19.93 | 19.83 | 20.80 | 20.34 | 21.02 | 19.62 | 20.06 |
| eval/lm/pile-validation/CE loss | lower | 2.2148 | 2.1889 | 2.1340 | 2.1501 | 2.1849 | 2.1669 | 2.2069 | 2.1390 | 2.1638 |
| eval/lm/pile-validation/PPL | lower | 9.1600 | 8.9255 | 8.4490 | 8.5854 | 8.8901 | 8.7311 | 9.0878 | 8.4906 | 8.7040 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.5217 | 2.4858 | 2.4232 | 2.4352 | 2.4828 | 2.4589 | 2.5056 | 2.4200 | 2.4652 |
| eval/lm/wikitext_103-validation/PPL | lower | 12.45 | 12.01 | 11.28 | 11.42 | 11.98 | 11.69 | 12.25 | 11.25 | 11.77 |
| throughput/in-loop eval batches | see metric | 828.0 | 828.0 | 828.0 | 828.0 | 828.0 | 828.0 | 828.0 | 828.0 | 828.0 |
| throughput/in-loop eval time (s) | see metric | 66.43 | 91.69 | 72.26 | 64.45 | 79.58 | 73.13 | 81.24 | 76.18 | 68.72 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| eg-480m-cx8-eg24e2k-lr8e-4-r1<br>`epx7o7ty` | finished | original | 60804562944.0 | 77317 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/epx7o7ty) |
| eg-480m-cx8-eg96e8k-lr8e-4-r1<br>`8676ezla` | finished | original | 60981510144.0 | 77542 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8676ezla) |
| int-480m-cx8-intd256e8k-lr8e-4-r1<br>`3q0hdi34` | finished | original | 61859954688.0 | 78659 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3q0hdi34) |
| int-480m-cx8-intw256e8k-lr8e-4-r1<br>`vdcrgfy0` | finished | original | 61374726144.0 | 78042 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vdcrgfy0) |
| q3-480m-cx8-q3am128e8k-lr8e-4-r1<br>`3xqcp7if` | finished | original | 61080600576.0 | 77668 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3xqcp7if) |
| q3-480m-cx8-q3td128e8k-lr8e-4-r1<br>`t3bexkgy` | finished | original | 61211148288.0 | 77834 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/t3bexkgy) |
| se-480m-cx8-se0m9-lr8e-4-r1<br>`e84abrdp` | finished | original | 60863545344.0 | 77392 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/e84abrdp) |
| sp-480m-cx8-sp192e4k-lr6e-4-r1<br>`ysglhwbv` | finished | original | 61217439744.0 | 77842 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ysglhwbv) |
| sp-480m-cx8-sp96e4k-lr7e-4-r1<br>`xek7xsp1` | finished | original | 60981510144.0 | 77542 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xek7xsp1) |

## 810m Cx1

| metric | direction | eg-810m-cx1-eg24e2k-lr6e-4-r1<br>`1nqxk9iw` | eg-810m-cx1-eg96e8k-lr6e-4-r1<br>`wjto6qtp` | int-810m-cx1-intd256e8k-lr6e-4-r1<br>`kgl5lc0a` | int-810m-cx1-intw256e8k-lr6e-4-r1<br>`w912irkq` | olmoe3-810m-cx1-b256k-lr5e-5-cs-r2<br>`o595mfxn` | 810m-cx1-b256k-lr1.2e-3-r1<br>`j78isnlu` | 810m-cx1-b256k-lr1.5e-4-cold-r1<br>`88u2c9dn` | 810m-cx1-b256k-lr2.4e-3-r1<br>`t0mls005` | 810m-cx1-b256k-lr3e-4-cold-r1<br>`gfb6q5xw` | 810m-cx1-b256k-lr6e-3-r1<br>`gr2aecp3` | 810m-cx1-b256k-lr6e-4-r1<br>`88byjpdd` | q3-810m-cx1-q3am128e8k-lr6e-4-r1<br>`shcduk5j` | q3-810m-cx1-q3td128e8k-lr6e-4-r1<br>`y4hplsg5` | se-810m-cx1-se0m9-lr6e-4-r1<br>`xt0aeyzw` | sp-810m-cx1-sp192e4k-lr4e-4-r2<br>`2t73nrem` | sp-810m-cx1-sp96e4k-lr5e-4-r1<br>`roffur1i` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.91048 | 0.88718 | 0.87516 | 0.87807 | 0.99549 | 0.92250 | 0.93029 | 0.91189 | 0.91699 | 0.95535 | 0.90102 | 0.88616 | 0.88519 | 0.90260 | 0.89105 | 0.89584 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 0.99601 | 0.97084 | 0.95910 | 0.96241 | 1.0931 | 1.0103 | 1.0180 | 0.99691 | 1.0044 | 1.0445 | 0.98721 | 0.96981 | 0.96921 | 0.98583 | 0.97182 | 0.98125 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0118 | 1.0216 | 1.0068 | 1.0209 | 1.0151 | 1.0143 | 1.0188 | 1.0295 | 1.0174 | 1.0194 | 1.0218 | 1.0094 | 1.0191 | 1.0248 | 1.0123 | 1.0177 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0236 | 2.0432 | 2.0135 | 2.0418 | 2.0301 | 2.0286 | 2.0375 | 2.0589 | 2.0348 | 2.0388 | 2.0437 | 2.0188 | 2.0382 | 2.0497 | 2.0246 | 2.0354 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.70140 | 0.70821 | 0.69799 | 0.70774 | 0.70367 | 0.70310 | 0.70615 | 0.71365 | 0.70527 | 0.70669 | 0.70835 | 0.69979 | 0.70643 | 0.71049 | 0.70172 | 0.70551 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4028 | 1.4164 | 1.3960 | 1.4155 | 1.4073 | 1.4062 | 1.4123 | 1.4273 | 1.4105 | 1.4134 | 1.4167 | 1.3996 | 1.4129 | 1.4210 | 1.4034 | 1.4110 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.25341 | 0.25341 | 0.25939 | 0.23464 | 0.25341 | 0.26706 | 0.25853 | 0.23720 | 0.27133 | 0.25597 | 0.24061 | 0.26109 | 0.24915 | 0.22952 | 0.25853 | 0.26109 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.25341 | 0.25341 | 0.25939 | 0.23464 | 0.25341 | 0.26706 | 0.25853 | 0.23720 | 0.27133 | 0.25597 | 0.24061 | 0.26109 | 0.24915 | 0.22952 | 0.25853 | 0.26109 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.3937 | -1.4072 | -1.3860 | -1.4042 | -1.3970 | -1.4010 | -1.4070 | -1.4201 | -1.4046 | -1.3943 | -1.4138 | -1.3959 | -1.4025 | -1.4143 | -1.3971 | -1.3997 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.3937 | -1.4072 | -1.3860 | -1.4042 | -1.3970 | -1.4010 | -1.4070 | -1.4201 | -1.4046 | -1.3943 | -1.4138 | -1.3959 | -1.4025 | -1.4143 | -1.3971 | -1.3997 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.25078 | 0.25040 | 0.25171 | 0.24930 | 0.25250 | 0.25079 | 0.25176 | 0.24862 | 0.25277 | 0.25103 | 0.24954 | 0.25176 | 0.24891 | 0.24758 | 0.25180 | 0.25085 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.25078 | 0.25040 | 0.25171 | 0.24930 | 0.25250 | 0.25079 | 0.25176 | 0.24862 | 0.25277 | 0.25103 | 0.24954 | 0.25176 | 0.24891 | 0.24758 | 0.25180 | 0.25085 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.71061 | 0.68925 | 0.67315 | 0.67610 | 0.81274 | 0.70982 | 0.73285 | 0.71007 | 0.70483 | 0.74400 | 0.69137 | 0.68668 | 0.68542 | 0.70236 | 0.69035 | 0.68738 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.77341 | 0.75034 | 0.73303 | 0.73548 | 0.88641 | 0.77298 | 0.79891 | 0.77237 | 0.76706 | 0.81036 | 0.75296 | 0.74723 | 0.74597 | 0.76517 | 0.75129 | 0.74792 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0165 | 1.0363 | 1.0150 | 1.0161 | 1.0259 | 1.0183 | 1.0180 | 1.0194 | 1.0123 | 1.0390 | 1.0142 | 1.0186 | 1.0130 | 1.0187 | 1.0308 | 1.0260 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0330 | 2.0727 | 2.0300 | 2.0322 | 2.0518 | 2.0366 | 2.0359 | 2.0387 | 2.0246 | 2.0780 | 2.0284 | 2.0371 | 2.0261 | 2.0373 | 2.0615 | 2.0520 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.70464 | 0.71836 | 0.70366 | 0.70443 | 0.71117 | 0.70588 | 0.70566 | 0.70665 | 0.70174 | 0.72020 | 0.70306 | 0.70605 | 0.70231 | 0.70616 | 0.71451 | 0.71125 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4093 | 1.4367 | 1.4073 | 1.4089 | 1.4223 | 1.4118 | 1.4113 | 1.4133 | 1.4035 | 1.4404 | 1.4061 | 1.4121 | 1.4046 | 1.4123 | 1.4290 | 1.4225 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.25042 | 0.22896 | 0.25589 | 0.23653 | 0.25000 | 0.25084 | 0.25463 | 0.24158 | 0.26473 | 0.23822 | 0.25968 | 0.24327 | 0.25379 | 0.24874 | 0.25505 | 0.24495 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.25042 | 0.22896 | 0.25589 | 0.23653 | 0.25000 | 0.25084 | 0.25463 | 0.24158 | 0.26473 | 0.23822 | 0.25968 | 0.24327 | 0.25379 | 0.24874 | 0.25505 | 0.24495 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.3999 | -1.4276 | -1.3963 | -1.3986 | -1.4108 | -1.4059 | -1.4045 | -1.4064 | -1.3971 | -1.4282 | -1.4030 | -1.4085 | -1.3943 | -1.4039 | -1.4232 | -1.4097 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.3999 | -1.4276 | -1.3963 | -1.3986 | -1.4108 | -1.4059 | -1.4045 | -1.4064 | -1.3971 | -1.4282 | -1.4030 | -1.4085 | -1.3943 | -1.4039 | -1.4232 | -1.4097 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.25082 | 0.24833 | 0.24967 | 0.24911 | 0.25088 | 0.25017 | 0.25063 | 0.24933 | 0.25139 | 0.24821 | 0.25136 | 0.24863 | 0.25049 | 0.24969 | 0.24884 | 0.25032 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.25082 | 0.24833 | 0.24967 | 0.24911 | 0.25088 | 0.25017 | 0.25063 | 0.24933 | 0.25139 | 0.24821 | 0.25136 | 0.24863 | 0.25049 | 0.24969 | 0.24884 | 0.25032 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 1.8114 | 1.7266 | 1.6095 | 1.5856 | 1.9973 | 1.6427 | 1.7990 | 1.6498 | 1.7401 | 1.8171 | 1.6998 | 1.6634 | 1.5957 | 1.6336 | 1.7167 | 1.7162 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 2.8944 | 2.7789 | 2.5906 | 2.5508 | 3.1786 | 2.6475 | 2.8790 | 2.6541 | 2.8006 | 2.9134 | 2.7322 | 2.6803 | 2.5664 | 2.6241 | 2.7465 | 2.7532 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 1.2556 | 1.1969 | 1.1156 | 1.0991 | 1.3844 | 1.1385 | 1.2469 | 1.1436 | 1.2061 | 1.2594 | 1.1783 | 1.1529 | 1.1061 | 1.1323 | 1.1900 | 1.1895 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 2.0063 | 1.9260 | 1.7955 | 1.7682 | 2.2032 | 1.8349 | 1.9955 | 1.8399 | 1.9412 | 2.0193 | 1.8938 | 1.8578 | 1.7789 | 1.8190 | 1.9041 | 1.9085 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.25119 | 0.27794 | 0.32665 | 0.34097 | 0.17765 | 0.29417 | 0.23878 | 0.28749 | 0.25692 | 0.24069 | 0.27889 | 0.30277 | 0.30086 | 0.30372 | 0.28462 | 0.29417 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.25119 | 0.27794 | 0.32665 | 0.34097 | 0.17765 | 0.29417 | 0.23878 | 0.28749 | 0.25692 | 0.24069 | 0.27889 | 0.30277 | 0.30086 | 0.30372 | 0.28462 | 0.29417 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -2.3977 | -2.1976 | -2.0876 | -2.0272 | -2.5971 | -2.0490 | -2.3247 | -2.0222 | -2.2565 | -2.1775 | -2.1606 | -2.0679 | -2.0614 | -2.0839 | -2.2412 | -2.1978 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -2.3977 | -2.1976 | -2.0876 | -2.0272 | -2.5971 | -2.0490 | -2.3247 | -2.0222 | -2.2565 | -2.1775 | -2.1606 | -2.0679 | -2.0614 | -2.0839 | -2.2412 | -2.1978 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.21304 | 0.23415 | 0.26285 | 0.28556 | 0.15369 | 0.25603 | 0.21059 | 0.22381 | 0.23419 | 0.18733 | 0.22626 | 0.25236 | 0.25349 | 0.25508 | 0.23691 | 0.24982 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.21304 | 0.23415 | 0.26285 | 0.28556 | 0.15369 | 0.25603 | 0.21059 | 0.22381 | 0.23419 | 0.18733 | 0.22626 | 0.25236 | 0.25349 | 0.25508 | 0.23691 | 0.24982 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.44927 | 0.43777 | 0.41549 | 0.44605 | 0.52484 | 0.47643 | 0.43990 | 0.46879 | 0.47324 | 0.54843 | 0.46060 | 0.43291 | 0.46554 | 0.47596 | 0.43324 | 0.44149 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.48928 | 0.47600 | 0.45196 | 0.48686 | 0.57156 | 0.52133 | 0.48040 | 0.51086 | 0.51661 | 0.59765 | 0.50158 | 0.47090 | 0.50743 | 0.51870 | 0.47205 | 0.48119 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.31141 | 0.30344 | 0.28803 | 0.30916 | 0.36376 | 0.33027 | 0.30492 | 0.32493 | 0.32802 | 0.38016 | 0.31924 | 0.30006 | 0.32271 | 0.32992 | 0.30028 | 0.30604 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.33910 | 0.32993 | 0.31325 | 0.33746 | 0.39616 | 0.36135 | 0.33298 | 0.35410 | 0.35804 | 0.41424 | 0.34767 | 0.32641 | 0.35174 | 0.35959 | 0.32719 | 0.33352 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.50494 | 0.52372 | 0.53755 | 0.52569 | 0.45751 | 0.50395 | 0.50593 | 0.52372 | 0.50791 | 0.44763 | 0.48913 | 0.51186 | 0.52470 | 0.48518 | 0.53656 | 0.50198 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.50494 | 0.52372 | 0.53755 | 0.52569 | 0.45751 | 0.50395 | 0.50593 | 0.52372 | 0.50791 | 0.44763 | 0.48913 | 0.51186 | 0.52470 | 0.48518 | 0.53656 | 0.50198 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -2.3191 | -2.3481 | -2.1402 | -2.1793 | -3.1455 | -2.4347 | -2.4040 | -2.4590 | -2.2981 | -3.1677 | -2.4293 | -2.3090 | -2.2365 | -2.5187 | -2.2071 | -2.2745 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -2.3191 | -2.3481 | -2.1402 | -2.1793 | -3.1455 | -2.4347 | -2.4040 | -2.4590 | -2.2981 | -3.1677 | -2.4293 | -2.3090 | -2.2365 | -2.5187 | -2.2071 | -2.2745 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.49540 | 0.49781 | 0.51774 | 0.51192 | 0.42845 | 0.49220 | 0.48389 | 0.49260 | 0.49448 | 0.43837 | 0.47480 | 0.50008 | 0.50705 | 0.47425 | 0.51648 | 0.49339 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.49540 | 0.49781 | 0.51774 | 0.51192 | 0.42845 | 0.49220 | 0.48389 | 0.49260 | 0.49448 | 0.43837 | 0.47480 | 0.50008 | 0.50705 | 0.47425 | 0.51648 | 0.49339 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.46524 | 0.42169 | 0.49905 | 0.46947 | 0.59813 | 0.48715 | 0.48345 | 0.37400 | 0.47882 | 0.46858 | 0.48245 | 0.48178 | 0.41202 | 0.43306 | 0.44959 | 0.47947 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.55845 | 0.50692 | 0.60050 | 0.56457 | 0.71848 | 0.58523 | 0.58136 | 0.44947 | 0.57612 | 0.56287 | 0.57873 | 0.57933 | 0.49597 | 0.52098 | 0.54035 | 0.57636 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.32257 | 0.29246 | 0.34604 | 0.32554 | 0.41470 | 0.33782 | 0.33527 | 0.25938 | 0.33209 | 0.32497 | 0.33460 | 0.33415 | 0.28574 | 0.30033 | 0.31173 | 0.33246 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.38727 | 0.35156 | 0.41639 | 0.39149 | 0.49818 | 0.40583 | 0.40320 | 0.31174 | 0.39965 | 0.39044 | 0.40140 | 0.40183 | 0.34395 | 0.36130 | 0.37473 | 0.39973 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.83799 | 0.85439 | 0.84764 | 0.83992 | 0.75217 | 0.81678 | 0.81678 | 0.84089 | 0.83896 | 0.79364 | 0.81003 | 0.85342 | 0.86403 | 0.84764 | 0.85053 | 0.82160 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.83799 | 0.85439 | 0.84764 | 0.83992 | 0.75217 | 0.81678 | 0.81678 | 0.84089 | 0.83896 | 0.79364 | 0.81003 | 0.85342 | 0.86403 | 0.84764 | 0.85053 | 0.82160 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.45314 | -0.43398 | -0.43394 | -0.43539 | -0.69694 | -0.47256 | -0.52688 | -0.43679 | -0.44879 | -0.54063 | -0.49434 | -0.44435 | -0.40583 | -0.44218 | -0.43522 | -0.48618 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.45314 | -0.43398 | -0.43394 | -0.43539 | -0.69694 | -0.47256 | -0.52688 | -0.43679 | -0.44879 | -0.54063 | -0.49434 | -0.44435 | -0.40583 | -0.44218 | -0.43522 | -0.48618 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.74004 | 0.74840 | 0.74969 | 0.74641 | 0.63350 | 0.73139 | 0.70099 | 0.75086 | 0.73480 | 0.69677 | 0.71813 | 0.74155 | 0.75839 | 0.74811 | 0.75200 | 0.72258 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.74004 | 0.74840 | 0.74969 | 0.74641 | 0.63350 | 0.73139 | 0.70099 | 0.75086 | 0.73480 | 0.69677 | 0.71813 | 0.74155 | 0.75839 | 0.74811 | 0.75200 | 0.72258 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.28322 | 0.26602 | 0.28931 | 0.28307 | 0.28276 | 0.27515 | 0.26071 | 0.27911 | 0.26246 | 0.28228 | 0.26722 | 0.28844 | 0.29648 | 0.31425 | 0.25775 | 0.25477 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.29276 | 0.27492 | 0.29895 | 0.29255 | 0.29227 | 0.28442 | 0.26950 | 0.28849 | 0.27129 | 0.29172 | 0.27619 | 0.29815 | 0.30642 | 0.32481 | 0.26644 | 0.26327 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.19633 | 0.18439 | 0.20054 | 0.19624 | 0.19600 | 0.19074 | 0.18071 | 0.19349 | 0.18194 | 0.19568 | 0.18525 | 0.19995 | 0.20553 | 0.21786 | 0.17866 | 0.17661 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.20292 | 0.19058 | 0.20722 | 0.20281 | 0.20260 | 0.19716 | 0.18682 | 0.19998 | 0.18807 | 0.20223 | 0.19147 | 0.20669 | 0.21241 | 0.22518 | 0.18468 | 0.18250 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.83721 | 0.85420 | 0.84258 | 0.87030 | 0.79249 | 0.87478 | 0.86225 | 0.87746 | 0.88998 | 0.85599 | 0.88014 | 0.90429 | 0.84973 | 0.87746 | 0.88193 | 0.86404 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.83721 | 0.85420 | 0.84258 | 0.87030 | 0.79249 | 0.87478 | 0.86225 | 0.87746 | 0.88998 | 0.85599 | 0.88014 | 0.90429 | 0.84973 | 0.87746 | 0.88193 | 0.86404 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.39045 | -0.39121 | -0.36681 | -0.33424 | -0.51765 | -0.32557 | -0.36885 | -0.30939 | -0.30065 | -0.38002 | -0.30318 | -0.30088 | -0.38525 | -0.34295 | -0.31739 | -0.36896 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.39045 | -0.39121 | -0.36681 | -0.33424 | -0.51765 | -0.32557 | -0.36885 | -0.30939 | -0.30065 | -0.38002 | -0.30318 | -0.30088 | -0.38525 | -0.34295 | -0.31739 | -0.36896 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.84007 | 0.83704 | 0.84590 | 0.85289 | 0.79202 | 0.85867 | 0.84470 | 0.86991 | 0.85876 | 0.84635 | 0.85969 | 0.86538 | 0.83187 | 0.85036 | 0.86700 | 0.83909 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.84007 | 0.83704 | 0.84590 | 0.85289 | 0.79202 | 0.85867 | 0.84470 | 0.86991 | 0.85876 | 0.84635 | 0.85969 | 0.86538 | 0.83187 | 0.85036 | 0.86700 | 0.83909 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 1.0386 | 0.91229 | 0.90703 | 0.93956 | 1.2463 | 0.97777 | 1.1128 | 0.98986 | 0.95603 | 1.1223 | 1.0865 | 0.96501 | 0.93632 | 1.0428 | 0.95346 | 0.94696 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.6925 | 1.4944 | 1.4936 | 1.5323 | 1.9778 | 1.6046 | 1.7844 | 1.6138 | 1.5464 | 1.8201 | 1.7528 | 1.5822 | 1.5261 | 1.6896 | 1.5665 | 1.5423 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.75266 | 0.66570 | 0.65596 | 0.68755 | 0.89932 | 0.71009 | 0.80242 | 0.71735 | 0.69710 | 0.81503 | 0.79202 | 0.70140 | 0.68119 | 0.75713 | 0.69344 | 0.68933 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.2571 | 1.1222 | 1.1052 | 1.1560 | 1.4623 | 1.1958 | 1.3172 | 1.1994 | 1.1603 | 1.3565 | 1.3135 | 1.1798 | 1.1401 | 1.2598 | 1.1692 | 1.1536 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.63858 | 0.70787 | 0.68727 | 0.69663 | 0.59925 | 0.68352 | 0.63296 | 0.67603 | 0.67978 | 0.64794 | 0.63670 | 0.69101 | 0.66105 | 0.64794 | 0.68165 | 0.66105 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.63858 | 0.70787 | 0.68727 | 0.69663 | 0.59925 | 0.68352 | 0.63296 | 0.67603 | 0.67978 | 0.64794 | 0.63670 | 0.69101 | 0.66105 | 0.64794 | 0.68165 | 0.66105 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.95812 | -0.78488 | -0.78835 | -0.80123 | -1.0803 | -0.83261 | -0.94815 | -0.84185 | -0.83779 | -0.96581 | -0.93923 | -0.81821 | -0.83695 | -0.87175 | -0.81492 | -0.83175 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.95812 | -0.78488 | -0.78835 | -0.80123 | -1.0803 | -0.83261 | -0.94815 | -0.84185 | -0.83779 | -0.96581 | -0.93923 | -0.81821 | -0.83695 | -0.87175 | -0.81492 | -0.83175 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.55183 | 0.61343 | 0.61404 | 0.60481 | 0.51622 | 0.60785 | 0.55062 | 0.59815 | 0.57992 | 0.56010 | 0.56004 | 0.60426 | 0.59497 | 0.57501 | 0.59834 | 0.59252 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.55183 | 0.61343 | 0.61404 | 0.60481 | 0.51622 | 0.60785 | 0.55062 | 0.59815 | 0.57992 | 0.56010 | 0.56004 | 0.60426 | 0.59497 | 0.57501 | 0.59834 | 0.59252 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.8265 | 1.6938 | 1.6977 | 1.6934 | 2.1723 | 1.7308 | 1.9102 | 1.7560 | 1.7802 | 1.7917 | 1.7233 | 1.7254 | 1.7021 | 1.7629 | 1.7579 | 1.7434 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.5123 | 2.3286 | 2.3463 | 2.3270 | 2.9700 | 2.3669 | 2.6118 | 2.4215 | 2.4572 | 2.4622 | 2.3559 | 2.3971 | 2.3264 | 2.4298 | 2.4527 | 2.4022 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.2661 | 1.1739 | 1.1768 | 1.1738 | 1.5055 | 1.1997 | 1.3242 | 1.2172 | 1.2339 | 1.2418 | 1.1945 | 1.1959 | 1.1798 | 1.2220 | 1.2185 | 1.2083 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.7416 | 1.6140 | 1.6263 | 1.6127 | 2.0585 | 1.6405 | 1.8105 | 1.6785 | 1.7033 | 1.7066 | 1.6330 | 1.6616 | 1.6124 | 1.6841 | 1.7001 | 1.6651 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.22067 | 0.21329 | 0.24364 | 0.22231 | 0.23216 | 0.23790 | 0.21329 | 0.21657 | 0.19032 | 0.22888 | 0.21903 | 0.21165 | 0.22313 | 0.20673 | 0.23216 | 0.23298 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.22067 | 0.21329 | 0.24364 | 0.22231 | 0.23216 | 0.23790 | 0.21329 | 0.21657 | 0.19032 | 0.22888 | 0.21903 | 0.21165 | 0.22313 | 0.20673 | 0.23216 | 0.23298 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -4.0759 | -4.1705 | -3.9308 | -4.0474 | -4.4876 | -4.2124 | -4.2937 | -4.2306 | -4.3619 | -4.3480 | -4.0508 | -4.1475 | -4.0557 | -4.4076 | -3.9764 | -4.1166 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -4.0759 | -4.1705 | -3.9308 | -4.0474 | -4.4876 | -4.2124 | -4.2937 | -4.2306 | -4.3619 | -4.3480 | -4.0508 | -4.1475 | -4.0557 | -4.4076 | -3.9764 | -4.1166 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.24389 | 0.24122 | 0.26109 | 0.24466 | 0.24391 | 0.24558 | 0.23522 | 0.23522 | 0.22797 | 0.24271 | 0.24172 | 0.23999 | 0.24385 | 0.22881 | 0.24636 | 0.24434 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.24389 | 0.24122 | 0.26109 | 0.24466 | 0.24391 | 0.24558 | 0.23522 | 0.23522 | 0.22797 | 0.24271 | 0.24172 | 0.23999 | 0.24385 | 0.22881 | 0.24636 | 0.24434 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.48085 | 0.46649 | 0.46291 | 0.45373 | 0.54478 | 0.47704 | 0.49137 | 0.47902 | 0.48128 | 0.52845 | 0.47447 | 0.46849 | 0.45907 | 0.47281 | 0.46411 | 0.46520 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.48703 | 0.47283 | 0.46917 | 0.45979 | 0.55170 | 0.48346 | 0.49747 | 0.48510 | 0.48751 | 0.53541 | 0.48071 | 0.47473 | 0.46527 | 0.47944 | 0.47042 | 0.47135 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.68890 | 0.67920 | 0.67593 | 0.66576 | 0.76938 | 0.67383 | 0.70381 | 0.68785 | 0.68311 | 0.73038 | 0.68750 | 0.67710 | 0.66497 | 0.68562 | 0.67094 | 0.67908 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.69500 | 0.68479 | 0.68177 | 0.67145 | 0.77602 | 0.67964 | 0.70992 | 0.69370 | 0.68905 | 0.73657 | 0.69335 | 0.68297 | 0.67060 | 0.69153 | 0.67675 | 0.68522 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.2318 | 1.9809 | 2.5097 | 2.2380 | 2.5328 | 2.6983 | 2.0780 | 2.5103 | 2.3920 | 2.4397 | 2.8074 | 2.5139 | 2.4917 | 2.6733 | 2.4642 | 2.6967 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 4.4636 | 3.9619 | 5.0195 | 4.4759 | 5.0655 | 5.3966 | 4.1561 | 5.0205 | 4.7839 | 4.8795 | 5.6148 | 5.0278 | 4.9834 | 5.3466 | 4.9284 | 5.3934 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.5473 | 1.3736 | 1.7398 | 1.5512 | 1.7551 | 1.8707 | 1.4404 | 1.7399 | 1.6584 | 1.6908 | 1.9457 | 1.7424 | 1.7280 | 1.8530 | 1.7076 | 1.8695 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.0946 | 2.7473 | 3.4796 | 3.1024 | 3.5102 | 3.7415 | 2.8809 | 3.4798 | 3.3169 | 3.3816 | 3.8915 | 3.4848 | 3.4561 | 3.7059 | 3.4152 | 3.7391 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.12000 | 0.08000 | 0.12000 | 0.07000 | 0.09000 | 0.09000 | 0.07000 | 0.10000 | 0.10000 | 0.09000 | 0.07000 | 0.09000 | 0.09000 | 0.11000 | 0.18000 | 0.08000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.12000 | 0.08000 | 0.12000 | 0.07000 | 0.09000 | 0.09000 | 0.07000 | 0.10000 | 0.10000 | 0.09000 | 0.07000 | 0.09000 | 0.09000 | 0.11000 | 0.18000 | 0.08000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -3.0865 | -2.7329 | -3.4722 | -3.0920 | -3.5017 | -3.7374 | -2.8745 | -3.4737 | -3.3119 | -3.3727 | -3.8888 | -3.4803 | -3.4501 | -3.7007 | -3.4095 | -3.7303 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -3.0865 | -2.7329 | -3.4722 | -3.0920 | -3.5017 | -3.7374 | -2.8745 | -3.4737 | -3.3119 | -3.3727 | -3.8888 | -3.4803 | -3.4501 | -3.7007 | -3.4095 | -3.7303 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.10265 | 0.09899 | 0.09803 | 0.09710 | 0.09444 | 0.09680 | 0.09272 | 0.09522 | 0.09770 | 0.09494 | 0.09324 | 0.09549 | 0.09385 | 0.09807 | 0.09920 | 0.09256 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.10265 | 0.09899 | 0.09803 | 0.09710 | 0.09444 | 0.09680 | 0.09272 | 0.09522 | 0.09770 | 0.09494 | 0.09324 | 0.09549 | 0.09385 | 0.09807 | 0.09920 | 0.09256 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.81981 | 0.81359 | 0.80547 | 0.80975 | 0.86115 | 0.81755 | 0.83216 | 0.83329 | 0.82200 | 0.85025 | 0.82074 | 0.81564 | 0.81550 | 0.81925 | 0.81262 | 0.81675 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.82880 | 0.82252 | 0.81447 | 0.81862 | 0.87048 | 0.82648 | 0.84146 | 0.84247 | 0.83107 | 0.85957 | 0.82978 | 0.82463 | 0.82445 | 0.82819 | 0.82166 | 0.82578 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.73078 | 0.71665 | 0.70484 | 0.70368 | 0.80469 | 0.72250 | 0.74780 | 0.73550 | 0.72901 | 0.78687 | 0.71741 | 0.71491 | 0.70844 | 0.72410 | 0.70813 | 0.71618 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.73309 | 0.71903 | 0.70711 | 0.70599 | 0.80731 | 0.72488 | 0.75023 | 0.73785 | 0.73143 | 0.78947 | 0.71983 | 0.71723 | 0.71079 | 0.72641 | 0.71051 | 0.71857 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.74218 | 0.73700 | 0.72106 | 0.73046 | 0.82636 | 0.74777 | 0.76268 | 0.75946 | 0.74253 | 0.78956 | 0.73713 | 0.73456 | 0.73197 | 0.74123 | 0.73121 | 0.73638 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.78137 | 0.77559 | 0.75818 | 0.76827 | 0.87025 | 0.78761 | 0.80248 | 0.79940 | 0.78126 | 0.83135 | 0.77567 | 0.77316 | 0.76970 | 0.77929 | 0.76886 | 0.77441 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0216 | 1.0258 | 1.0176 | 1.0201 | 1.0279 | 1.0094 | 1.0149 | 1.0234 | 1.0134 | 1.0245 | 1.0124 | 1.0114 | 1.0168 | 1.0167 | 1.0162 | 1.0214 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0433 | 2.0516 | 2.0351 | 2.0401 | 2.0559 | 2.0187 | 2.0298 | 2.0468 | 2.0268 | 2.0491 | 2.0248 | 2.0227 | 2.0336 | 2.0333 | 2.0324 | 2.0428 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.70821 | 0.71112 | 0.70546 | 0.70711 | 0.71254 | 0.69974 | 0.70355 | 0.70943 | 0.70256 | 0.71026 | 0.70184 | 0.70111 | 0.70486 | 0.70475 | 0.70442 | 0.70806 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4164 | 1.4222 | 1.4109 | 1.4142 | 1.4251 | 1.3995 | 1.4071 | 1.4189 | 1.4051 | 1.4205 | 1.4037 | 1.4022 | 1.4097 | 1.4095 | 1.4088 | 1.4161 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24973 | 0.24591 | 0.24017 | 0.25845 | 0.25972 | 0.25654 | 0.26249 | 0.26291 | 0.25951 | 0.25313 | 0.25845 | 0.25739 | 0.26652 | 0.25037 | 0.24952 | 0.24315 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24973 | 0.24591 | 0.24017 | 0.25845 | 0.25972 | 0.25654 | 0.26249 | 0.26291 | 0.25951 | 0.25313 | 0.25845 | 0.25739 | 0.26652 | 0.25037 | 0.24952 | 0.24315 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3923 | -1.3955 | -1.3899 | -1.3907 | -1.3925 | -1.3870 | -1.3881 | -1.3894 | -1.3883 | -1.3886 | -1.3875 | -1.3879 | -1.3883 | -1.3891 | -1.3897 | -1.3913 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.4097 | -1.4151 | -1.3996 | -1.4047 | -1.4130 | -1.3925 | -1.3964 | -1.4122 | -1.3991 | -1.4014 | -1.3985 | -1.3970 | -1.4005 | -1.4019 | -1.4037 | -1.4056 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.25000 | 0.24904 | 0.24983 | 0.25010 | 0.25031 | 0.25042 | 0.25041 | 0.25168 | 0.25062 | 0.25075 | 0.25090 | 0.25050 | 0.25079 | 0.25055 | 0.25047 | 0.24991 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.25001 | 0.24813 | 0.24962 | 0.25020 | 0.25058 | 0.25077 | 0.25088 | 0.25337 | 0.25123 | 0.25142 | 0.25177 | 0.25097 | 0.25160 | 0.25106 | 0.25086 | 0.24981 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 1.0321 | 1.0275 | 1.0001 | 1.0083 | 1.1476 | 1.0402 | 1.0646 | 1.0420 | 1.0437 | 1.0911 | 1.0347 | 1.0196 | 1.0171 | 1.0333 | 1.0118 | 1.0294 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.1498 | 1.1468 | 1.1129 | 1.1229 | 1.2784 | 1.1584 | 1.1860 | 1.1595 | 1.1626 | 1.2138 | 1.1522 | 1.1370 | 1.1330 | 1.1503 | 1.1273 | 1.1480 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0117 | 1.0145 | 1.0149 | 1.0132 | 1.0190 | 1.0083 | 1.0066 | 1.0233 | 1.0073 | 1.0252 | 1.0117 | 1.0073 | 1.0191 | 1.0279 | 1.0093 | 1.0107 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0234 | 2.0290 | 2.0299 | 2.0264 | 2.0381 | 2.0166 | 2.0132 | 2.0466 | 2.0146 | 2.0503 | 2.0234 | 2.0146 | 2.0382 | 2.0558 | 2.0186 | 2.0214 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.70132 | 0.70327 | 0.70359 | 0.70232 | 0.70641 | 0.69896 | 0.69781 | 0.70934 | 0.69830 | 0.71066 | 0.70134 | 0.69829 | 0.70644 | 0.71257 | 0.69966 | 0.70066 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4026 | 1.4065 | 1.4072 | 1.4046 | 1.4128 | 1.3979 | 1.3956 | 1.4187 | 1.3966 | 1.4213 | 1.4027 | 1.3966 | 1.4129 | 1.4251 | 1.3993 | 1.4013 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.28223 | 0.28069 | 0.25725 | 0.24522 | 0.26403 | 0.27298 | 0.27791 | 0.24399 | 0.25571 | 0.26558 | 0.26157 | 0.27452 | 0.26589 | 0.23442 | 0.26928 | 0.25787 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.28223 | 0.28069 | 0.25725 | 0.24522 | 0.26403 | 0.27298 | 0.27791 | 0.24399 | 0.25571 | 0.26558 | 0.26157 | 0.27452 | 0.26589 | 0.23442 | 0.26928 | 0.25787 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3843 | -1.3859 | -1.3857 | -1.3875 | -1.3872 | -1.3839 | -1.3828 | -1.3909 | -1.3843 | -1.3888 | -1.3872 | -1.3831 | -1.3893 | -1.3950 | -1.3865 | -1.3847 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.3939 | -1.3978 | -1.3947 | -1.3960 | -1.4003 | -1.3925 | -1.3884 | -1.4124 | -1.3895 | -1.4041 | -1.3977 | -1.3912 | -1.4034 | -1.4174 | -1.3940 | -1.3905 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25202 | 0.25169 | 0.25138 | 0.25055 | 0.25130 | 0.25199 | 0.25201 | 0.25081 | 0.25140 | 0.25102 | 0.25098 | 0.25231 | 0.25067 | 0.24952 | 0.25090 | 0.25134 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25417 | 0.25339 | 0.25281 | 0.25095 | 0.25247 | 0.25385 | 0.25397 | 0.25113 | 0.25273 | 0.25195 | 0.25191 | 0.25469 | 0.25134 | 0.24898 | 0.25180 | 0.25261 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.88058 | 0.87356 | 0.85530 | 0.85546 | 0.97393 | 0.88596 | 0.90633 | 0.89293 | 0.88572 | 0.93444 | 0.88559 | 0.86857 | 0.87370 | 0.88041 | 0.86114 | 0.87748 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 0.94219 | 0.93331 | 0.91378 | 0.91358 | 1.0417 | 0.94564 | 0.96896 | 0.95383 | 0.94628 | 0.99862 | 0.94584 | 0.92832 | 0.93334 | 0.94116 | 0.92029 | 0.93804 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0109 | 1.0241 | 1.0149 | 1.0097 | 1.0185 | 1.0227 | 1.0105 | 1.0208 | 1.0143 | 1.0226 | 1.0130 | 1.0044 | 1.0203 | 1.0130 | 1.0065 | 1.0137 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0218 | 2.0482 | 2.0298 | 2.0194 | 2.0370 | 2.0453 | 2.0211 | 2.0417 | 2.0287 | 2.0451 | 2.0260 | 2.0088 | 2.0406 | 2.0261 | 2.0130 | 2.0274 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.70080 | 0.70991 | 0.70356 | 0.70002 | 0.70606 | 0.70894 | 0.70054 | 0.70764 | 0.70318 | 0.70882 | 0.70224 | 0.69627 | 0.70733 | 0.70225 | 0.69777 | 0.70276 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4016 | 1.4198 | 1.4071 | 1.4000 | 1.4121 | 1.4179 | 1.4011 | 1.4153 | 1.4064 | 1.4176 | 1.4045 | 1.3925 | 1.4147 | 1.4045 | 1.3955 | 1.4055 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.26259 | 0.24017 | 0.27104 | 0.25609 | 0.25609 | 0.24049 | 0.26324 | 0.25284 | 0.24309 | 0.25999 | 0.25317 | 0.30224 | 0.24699 | 0.26617 | 0.26324 | 0.25642 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.26259 | 0.24017 | 0.27104 | 0.25609 | 0.25609 | 0.24049 | 0.26324 | 0.25284 | 0.24309 | 0.25999 | 0.25317 | 0.30224 | 0.24699 | 0.26617 | 0.26324 | 0.25642 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3855 | -1.3928 | -1.3863 | -1.3861 | -1.3884 | -1.3948 | -1.3856 | -1.3904 | -1.3894 | -1.3867 | -1.3881 | -1.3794 | -1.3911 | -1.3867 | -1.3852 | -1.3868 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.3940 | -1.4116 | -1.3955 | -1.3906 | -1.4006 | -1.4117 | -1.3925 | -1.4091 | -1.3994 | -1.4023 | -1.3992 | -1.3869 | -1.4066 | -1.3970 | -1.3903 | -1.3940 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.25139 | 0.24986 | 0.25117 | 0.25065 | 0.25079 | 0.24897 | 0.25115 | 0.25078 | 0.25008 | 0.25192 | 0.25067 | 0.25362 | 0.25012 | 0.25112 | 0.25103 | 0.25067 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.25273 | 0.24953 | 0.25240 | 0.25133 | 0.25163 | 0.24801 | 0.25231 | 0.25143 | 0.25012 | 0.25373 | 0.25128 | 0.25746 | 0.25024 | 0.25219 | 0.25203 | 0.25126 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.3450 | 1.3328 | 1.3065 | 1.3188 | 1.4679 | 1.3475 | 1.3784 | 1.3430 | 1.3510 | 1.3851 | 1.3452 | 1.3256 | 1.3352 | 1.3437 | 1.3288 | 1.3408 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.6829 | 1.6633 | 1.6302 | 1.6460 | 1.8298 | 1.6774 | 1.7200 | 1.6739 | 1.6878 | 1.7216 | 1.6786 | 1.6542 | 1.6692 | 1.6763 | 1.6633 | 1.6712 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0235 | 1.0153 | 1.0144 | 1.0142 | 1.0273 | 1.0134 | 1.0109 | 1.0383 | 1.0160 | 1.0314 | 1.0144 | 1.0110 | 1.0179 | 1.0220 | 1.0085 | 1.0237 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0471 | 2.0305 | 2.0289 | 2.0284 | 2.0545 | 2.0267 | 2.0218 | 2.0767 | 2.0320 | 2.0627 | 2.0288 | 2.0220 | 2.0357 | 2.0440 | 2.0170 | 2.0473 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.70947 | 0.70380 | 0.70321 | 0.70306 | 0.71207 | 0.70250 | 0.70072 | 0.71974 | 0.70422 | 0.71490 | 0.70314 | 0.70082 | 0.70561 | 0.70847 | 0.69912 | 0.70961 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4189 | 1.4076 | 1.4064 | 1.4061 | 1.4241 | 1.4050 | 1.4014 | 1.4395 | 1.4084 | 1.4298 | 1.4063 | 1.4016 | 1.4112 | 1.4169 | 1.3982 | 1.4192 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.26740 | 0.27071 | 0.27568 | 0.25878 | 0.27833 | 0.25878 | 0.27303 | 0.25679 | 0.26176 | 0.24685 | 0.28065 | 0.27999 | 0.25514 | 0.23857 | 0.25779 | 0.25017 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.26740 | 0.27071 | 0.27568 | 0.25878 | 0.27833 | 0.25878 | 0.27303 | 0.25679 | 0.26176 | 0.24685 | 0.28065 | 0.27999 | 0.25514 | 0.23857 | 0.25779 | 0.25017 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3862 | -1.3869 | -1.3850 | -1.3875 | -1.3893 | -1.3846 | -1.3835 | -1.3964 | -1.3865 | -1.3904 | -1.3842 | -1.3834 | -1.3882 | -1.3925 | -1.3860 | -1.3909 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4123 | -1.3969 | -1.3936 | -1.3961 | -1.4093 | -1.3972 | -1.3920 | -1.4322 | -1.4006 | -1.4123 | -1.3998 | -1.3953 | -1.4009 | -1.4084 | -1.3927 | -1.4078 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.25353 | 0.25112 | 0.25169 | 0.25062 | 0.25151 | 0.25220 | 0.25211 | 0.25076 | 0.25172 | 0.25129 | 0.25270 | 0.25276 | 0.25084 | 0.24975 | 0.25094 | 0.25043 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.25679 | 0.25241 | 0.25360 | 0.25122 | 0.25306 | 0.25416 | 0.25415 | 0.25153 | 0.25336 | 0.25239 | 0.25528 | 0.25555 | 0.25162 | 0.24959 | 0.25178 | 0.25082 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.46479 | 0.45011 | 0.45037 | 0.45405 | 0.51884 | 0.44842 | 0.47214 | 0.45967 | 0.46887 | 0.48366 | 0.46055 | 0.44915 | 0.45993 | 0.44337 | 0.44462 | 0.45321 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.46730 | 0.45253 | 0.45285 | 0.45657 | 0.52161 | 0.45092 | 0.47481 | 0.46229 | 0.47152 | 0.48639 | 0.46308 | 0.45174 | 0.46248 | 0.44595 | 0.44707 | 0.45582 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.33821 | 0.34067 | 0.33879 | 0.34719 | 0.38565 | 0.34408 | 0.36221 | 0.35475 | 0.34552 | 0.37498 | 0.34047 | 0.34630 | 0.34832 | 0.33672 | 0.34408 | 0.34483 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.33953 | 0.34199 | 0.34011 | 0.34859 | 0.38713 | 0.34548 | 0.36356 | 0.35618 | 0.34688 | 0.37641 | 0.34180 | 0.34758 | 0.34958 | 0.33802 | 0.34539 | 0.34621 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.59868 | 0.61211 | 0.60817 | 0.59335 | 0.69558 | 0.60931 | 0.64357 | 0.63357 | 0.61130 | 0.67800 | 0.61378 | 0.59127 | 0.58200 | 0.60396 | 0.61300 | 0.59529 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.60279 | 0.61641 | 0.61249 | 0.59749 | 0.70054 | 0.61368 | 0.64813 | 0.63822 | 0.61576 | 0.68267 | 0.61791 | 0.59547 | 0.58595 | 0.60814 | 0.61742 | 0.59944 |
| eval/lm/c4_en-validation/CE loss | lower | 3.0279 | 3.0038 | 2.9772 | 2.9784 | 3.1825 | 3.0202 | 3.0586 | 3.0524 | 3.0299 | 3.1455 | 3.0163 | 3.0006 | 2.9904 | 3.0233 | 2.9871 | 3.0042 |
| eval/lm/c4_en-validation/PPL | lower | 20.65 | 20.16 | 19.63 | 19.66 | 24.11 | 20.49 | 21.30 | 21.17 | 20.70 | 23.23 | 20.42 | 20.10 | 19.89 | 20.56 | 19.83 | 20.17 |
| eval/lm/dolma_books-validation/CE loss | lower | 2.9410 | 2.9142 | 2.8919 | 2.8767 | 3.1277 | 2.9335 | 2.9897 | 2.9683 | 2.9451 | 3.0809 | 2.9263 | 2.9233 | 2.8962 | 2.9290 | 2.8980 | 2.9090 |
| eval/lm/dolma_books-validation/PPL | lower | 18.93 | 18.43 | 18.03 | 17.76 | 22.82 | 18.79 | 19.88 | 19.46 | 19.01 | 21.78 | 18.66 | 18.60 | 18.10 | 18.71 | 18.14 | 18.34 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.1618 | 3.1382 | 3.1130 | 3.1117 | 3.3145 | 3.1546 | 3.1928 | 3.1873 | 3.1633 | 3.2794 | 3.1500 | 3.1383 | 3.1249 | 3.1562 | 3.1207 | 3.1388 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 23.61 | 23.06 | 22.49 | 22.46 | 27.51 | 23.44 | 24.36 | 24.22 | 23.65 | 26.56 | 23.34 | 23.06 | 22.76 | 23.48 | 22.66 | 23.08 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.2110 | 2.1928 | 2.1715 | 2.1743 | 2.3561 | 2.2044 | 2.2407 | 2.2257 | 2.2144 | 2.3072 | 2.2025 | 2.1895 | 2.1799 | 2.2039 | 2.1822 | 2.1959 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 9.1252 | 8.9603 | 8.7712 | 8.7961 | 10.55 | 9.0649 | 9.4003 | 9.2604 | 9.1557 | 10.05 | 9.0473 | 8.9309 | 8.8453 | 9.0601 | 8.8656 | 8.9881 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.3314 | 3.3073 | 3.2839 | 3.2825 | 3.4703 | 3.3222 | 3.3622 | 3.3477 | 3.3316 | 3.4327 | 3.3174 | 3.3064 | 3.2979 | 3.3258 | 3.2963 | 3.3123 |
| eval/lm/dolma_reddit-validation/PPL | lower | 27.98 | 27.31 | 26.68 | 26.64 | 32.15 | 27.72 | 28.85 | 28.44 | 27.98 | 30.96 | 27.59 | 27.29 | 27.05 | 27.82 | 27.01 | 27.45 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.4553 | 1.4360 | 1.4116 | 1.4128 | 1.6052 | 1.4451 | 1.4867 | 1.4623 | 1.4577 | 1.5431 | 1.4454 | 1.4360 | 1.4251 | 1.4469 | 1.4242 | 1.4362 |
| eval/lm/dolma_stack-validation/PPL | lower | 4.2857 | 4.2039 | 4.1024 | 4.1076 | 4.9789 | 4.2422 | 4.4226 | 4.3161 | 4.2962 | 4.6789 | 4.2433 | 4.2039 | 4.1581 | 4.2499 | 4.1547 | 4.2047 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.6838 | 2.6630 | 2.6332 | 2.6305 | 2.8495 | 2.6810 | 2.7149 | 2.7164 | 2.6836 | 2.8174 | 2.6752 | 2.6570 | 2.6459 | 2.6794 | 2.6345 | 2.6563 |
| eval/lm/dolma_wiki-validation/PPL | lower | 14.64 | 14.34 | 13.92 | 13.88 | 17.28 | 14.60 | 15.10 | 15.13 | 14.64 | 16.73 | 14.52 | 14.25 | 14.10 | 14.58 | 13.94 | 14.24 |
| eval/lm/ice-validation/CE loss | lower | 3.0915 | 3.0668 | 3.0542 | 3.0542 | 3.2569 | 3.0862 | 3.1205 | 3.0986 | 3.0846 | 3.1766 | 3.0744 | 3.0776 | 3.0657 | 3.0928 | 3.0631 | 3.0807 |
| eval/lm/ice-validation/PPL | lower | 22.01 | 21.47 | 21.20 | 21.20 | 25.97 | 21.89 | 22.66 | 22.17 | 21.86 | 23.96 | 21.64 | 21.71 | 21.45 | 22.04 | 21.39 | 21.77 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 3.1418 | 3.1179 | 3.0958 | 3.0953 | 3.2747 | 3.1314 | 3.1701 | 3.1640 | 3.1441 | 3.2436 | 3.1328 | 3.1262 | 3.1136 | 3.1304 | 3.0990 | 3.1039 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 23.14 | 22.60 | 22.10 | 22.09 | 26.44 | 22.91 | 23.81 | 23.67 | 23.20 | 25.63 | 22.94 | 22.79 | 22.50 | 22.88 | 22.18 | 22.28 |
| eval/lm/pile-validation/CE loss | lower | 2.3108 | 2.2895 | 2.2625 | 2.2667 | 2.4684 | 2.3007 | 2.3423 | 2.3282 | 2.3133 | 2.4208 | 2.3003 | 2.2887 | 2.2767 | 2.3058 | 2.2751 | 2.2904 |
| eval/lm/pile-validation/PPL | lower | 10.08 | 9.8702 | 9.6070 | 9.6480 | 11.80 | 9.9811 | 10.41 | 10.26 | 10.11 | 11.25 | 9.9774 | 9.8624 | 9.7442 | 10.03 | 9.7286 | 9.8790 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.6347 | 2.6114 | 2.5739 | 2.5840 | 2.8453 | 2.6243 | 2.6761 | 2.6513 | 2.6368 | 2.7621 | 2.6170 | 2.6102 | 2.5894 | 2.6246 | 2.6039 | 2.6092 |
| eval/lm/wikitext_103-validation/PPL | lower | 13.94 | 13.62 | 13.12 | 13.25 | 17.21 | 13.80 | 14.53 | 14.17 | 13.97 | 15.83 | 13.69 | 13.60 | 13.32 | 13.80 | 13.52 | 13.59 |
| throughput/in-loop eval batches | see metric | 828.0 | 828.0 | 828.0 | 828.0 | 1645.0 | 1645.0 | 1645.0 | 1645.0 | 1645.0 | 1645.0 | 1645.0 | 828.0 | 828.0 | 828.0 | 1729.0 | 828.0 |
| throughput/in-loop eval time (s) | see metric | 76.01 | 76.58 | 92.74 | 82.47 | 306.9 | 333.1 | 310.7 | 322.8 | 312.8 | 322.5 | 318.0 | 92.32 | 110.1 | 90.22 | 123.7 | 77.37 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| eg-810m-cx1-eg24e2k-lr6e-4-r1<br>`1nqxk9iw` | finished | original | 13789560832.0 | 52603 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1nqxk9iw) |
| eg-810m-cx1-eg96e8k-lr6e-4-r1<br>`wjto6qtp` | finished | original | 13824688128.0 | 52737 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wjto6qtp) |
| int-810m-cx1-intd256e8k-lr6e-4-r1<br>`kgl5lc0a` | finished | original | 13626769408.0 | 51982 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kgl5lc0a) |
| int-810m-cx1-intw256e8k-lr6e-4-r1<br>`w912irkq` | finished | original | 13902544896.0 | 53034 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/w912irkq) |
| olmoe3-810m-cx1-b256k-lr5e-5-cs-r2<br>`o595mfxn` | finished | gpu8-ep1mb4 | 13801357312.0 | 52648 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/o595mfxn) |
| 810m-cx1-b256k-lr1.2e-3-r1<br>`j78isnlu` | finished | gpu4-ep1mb4 | 13801357312.0 | 52648 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/j78isnlu) |
| 810m-cx1-b256k-lr1.5e-4-cold-r1<br>`88u2c9dn` | finished | gpu4-ep1mb4 | 13801357312.0 | 52648 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/88u2c9dn) |
| 810m-cx1-b256k-lr2.4e-3-r1<br>`t0mls005` | finished | gpu4-ep1mb4 | 13801357312.0 | 52648 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/t0mls005) |
| 810m-cx1-b256k-lr3e-4-cold-r1<br>`gfb6q5xw` | finished | gpu4-ep1mb4 | 13801357312.0 | 52648 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gfb6q5xw) |
| 810m-cx1-b256k-lr6e-3-r1<br>`gr2aecp3` | finished | gpu4-ep1mb4 | 13801357312.0 | 52648 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/gr2aecp3) |
| 810m-cx1-b256k-lr6e-4-r1<br>`88byjpdd` | finished | gpu4-ep1mb4 | 13801357312.0 | 52648 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/88byjpdd) |
| q3-810m-cx1-q3am128e8k-lr6e-4-r1<br>`shcduk5j` | finished | original | 13843562496.0 | 52809 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/shcduk5j) |
| q3-810m-cx1-q3td128e8k-lr6e-4-r1<br>`y4hplsg5` | finished | original | 13808173056.0 | 52674 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/y4hplsg5) |
| se-810m-cx1-se0m9-lr6e-4-r1<br>`xt0aeyzw` | finished | original | 13801357312.0 | 52648 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xt0aeyzw) |
| sp-810m-cx1-sp192e4k-lr4e-4-r2<br>`2t73nrem` | finished | original | 13871349760.0 | 52915 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/2t73nrem) |
| sp-810m-cx1-sp96e4k-lr5e-4-r1<br>`roffur1i` | finished | original | 13824688128.0 | 52737 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/roffur1i) |

## 810m Cx2

| metric | direction | eg-810m-cx2-eg24e2k-lr5.6e-4-r1<br>`138dbr8w` | eg-810m-cx2-eg96e8k-lr5.6e-4-r1<br>`0qmqmo4l` | int-810m-cx2-intd256e8k-lr5.6e-4-r1<br>`7nsy68db` | int-810m-cx2-intw256e8k-lr5.6e-4-r1<br>`jpbqhfvc` | 810m-cx2-b384k-lr1.12e-3-r3<br>`sxivrph5` | 810m-cx2-b384k-lr2.8e-4-r3<br>`uh4el1df` | 810m-cx2-b384k-lr5.6e-4-r3<br>`v5puakhq` | 810m-cx2-b512k-lr1.2e-3-r1<br>`d13uavyt` | 810m-cx2-b512k-lr1.5e-4-r1<br>`fcqkb55w` | 810m-cx2-b512k-lr3e-4-r1<br>`ogp6mrt6` | 810m-cx2-b512k-lr6e-4-r1<br>`okb4e1u0` | q3-810m-cx2-q3am128e8k-lr5.6e-4-r1<br>`vb315h4w` | q3-810m-cx2-q3td128e8k-lr5.6e-4-r1<br>`m3zgibjj` | se-810m-cx2-se0m9-lr5.6e-4-r1<br>`epue7vyg` | sp-810m-cx2-sp192e4k-lr4e-4-r2<br>`bh87mzps` | sp-810m-cx2-sp96e4k-lr5e-4-r1<br>`0d9gw8af` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.87394 | 0.85169 | 0.86000 | 0.84793 | 0.86756 | 0.87785 | 0.86538 | 0.85697 | 0.89693 | 0.87216 | 0.85678 | 0.85822 | 0.84756 | 0.86966 | 0.84722 | 0.85212 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 0.95589 | 0.93223 | 0.93971 | 0.92831 | 0.94932 | 0.95924 | 0.94613 | 0.93870 | 0.98150 | 0.95482 | 0.93742 | 0.93773 | 0.92767 | 0.94953 | 0.92747 | 0.93163 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0195 | 1.0167 | 1.0144 | 1.0068 | 1.0146 | 1.0110 | 1.0058 | 1.0227 | 1.0112 | 1.0058 | 1.0133 | 1.0067 | 1.0065 | 1.0188 | 1.0046 | 1.0089 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0390 | 2.0334 | 2.0287 | 2.0136 | 2.0293 | 2.0220 | 2.0115 | 2.0454 | 2.0224 | 2.0116 | 2.0266 | 2.0134 | 2.0130 | 2.0376 | 2.0092 | 2.0178 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.70667 | 0.70484 | 0.70318 | 0.69796 | 0.70329 | 0.70088 | 0.69720 | 0.70894 | 0.70093 | 0.69726 | 0.70239 | 0.69786 | 0.69771 | 0.70632 | 0.69647 | 0.69949 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4133 | 1.4097 | 1.4064 | 1.3959 | 1.4066 | 1.4018 | 1.3944 | 1.4179 | 1.4019 | 1.3945 | 1.4048 | 1.3957 | 1.3954 | 1.4126 | 1.3929 | 1.3990 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.25853 | 0.24488 | 0.25171 | 0.25171 | 0.24829 | 0.24232 | 0.25768 | 0.23379 | 0.25512 | 0.25256 | 0.26792 | 0.25853 | 0.25853 | 0.22099 | 0.25512 | 0.24659 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.25853 | 0.24488 | 0.25171 | 0.25171 | 0.24829 | 0.24232 | 0.25768 | 0.23379 | 0.25512 | 0.25256 | 0.26792 | 0.25853 | 0.25853 | 0.22099 | 0.25512 | 0.24659 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.4097 | -1.4062 | -1.4029 | -1.3914 | -1.4017 | -1.3962 | -1.3904 | -1.4139 | -1.3953 | -1.3909 | -1.3999 | -1.3919 | -1.3921 | -1.4081 | -1.3892 | -1.3931 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.4097 | -1.4062 | -1.4029 | -1.3914 | -1.4017 | -1.3962 | -1.3904 | -1.4139 | -1.3953 | -1.3909 | -1.3999 | -1.3919 | -1.3921 | -1.4081 | -1.3892 | -1.3931 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.25058 | 0.24938 | 0.25180 | 0.25195 | 0.25051 | 0.25031 | 0.25049 | 0.24951 | 0.24996 | 0.25139 | 0.25167 | 0.25081 | 0.25172 | 0.24896 | 0.25103 | 0.24985 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.25058 | 0.24938 | 0.25180 | 0.25195 | 0.25051 | 0.25031 | 0.25049 | 0.24951 | 0.24996 | 0.25139 | 0.25167 | 0.25081 | 0.25172 | 0.24896 | 0.25103 | 0.24985 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.68247 | 0.64906 | 0.64361 | 0.64373 | 0.66652 | 0.67320 | 0.67217 | 0.65773 | 0.68989 | 0.66863 | 0.66397 | 0.65184 | 0.63185 | 0.67000 | 0.63993 | 0.64425 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.74263 | 0.70588 | 0.69958 | 0.69983 | 0.72567 | 0.73198 | 0.73139 | 0.71560 | 0.75110 | 0.72744 | 0.72198 | 0.70888 | 0.68670 | 0.72834 | 0.69629 | 0.70069 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0246 | 1.0202 | 1.0144 | 1.0073 | 1.0140 | 1.0114 | 1.0093 | 1.0133 | 1.0101 | 1.0094 | 1.0042 | 1.0113 | 1.0227 | 1.0101 | 1.0178 | 1.0086 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0493 | 2.0405 | 2.0288 | 2.0146 | 2.0280 | 2.0228 | 2.0186 | 2.0266 | 2.0202 | 2.0187 | 2.0084 | 2.0226 | 2.0455 | 2.0202 | 2.0356 | 2.0172 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.71026 | 0.70723 | 0.70320 | 0.69832 | 0.70290 | 0.70113 | 0.69972 | 0.70239 | 0.70025 | 0.69976 | 0.69621 | 0.70107 | 0.70897 | 0.70025 | 0.70556 | 0.69925 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4205 | 1.4145 | 1.4064 | 1.3966 | 1.4058 | 1.4023 | 1.3994 | 1.4048 | 1.4005 | 1.3995 | 1.3924 | 1.4021 | 1.4179 | 1.4005 | 1.4111 | 1.3985 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.24832 | 0.24411 | 0.25926 | 0.26852 | 0.25210 | 0.25379 | 0.24369 | 0.24663 | 0.26136 | 0.26431 | 0.26221 | 0.24579 | 0.24453 | 0.27357 | 0.25042 | 0.24411 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.24832 | 0.24411 | 0.25926 | 0.26852 | 0.25210 | 0.25379 | 0.24369 | 0.24663 | 0.26136 | 0.26431 | 0.26221 | 0.24579 | 0.24453 | 0.27357 | 0.25042 | 0.24411 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.4156 | -1.4109 | -1.4028 | -1.3933 | -1.4003 | -1.3966 | -1.3952 | -1.3999 | -1.3942 | -1.3954 | -1.3874 | -1.3990 | -1.4146 | -1.3962 | -1.4075 | -1.3930 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.4156 | -1.4109 | -1.4028 | -1.3933 | -1.4003 | -1.3966 | -1.3952 | -1.3999 | -1.3942 | -1.3954 | -1.3874 | -1.3990 | -1.4146 | -1.3962 | -1.4075 | -1.3930 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.25062 | 0.24962 | 0.25172 | 0.25146 | 0.24964 | 0.25083 | 0.24970 | 0.25052 | 0.25044 | 0.25113 | 0.25208 | 0.24928 | 0.24837 | 0.25110 | 0.24898 | 0.25038 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.25062 | 0.24962 | 0.25172 | 0.25146 | 0.24964 | 0.25083 | 0.24970 | 0.25052 | 0.25044 | 0.25113 | 0.25208 | 0.24928 | 0.24837 | 0.25110 | 0.24898 | 0.25038 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 1.3427 | 1.3641 | 1.3059 | 1.3212 | 1.3158 | 1.3881 | 1.3163 | 1.4560 | 1.5055 | 1.4161 | 1.3657 | 1.3400 | 1.3107 | 1.3728 | 1.3463 | 1.2920 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 2.1807 | 2.2090 | 2.1113 | 2.1180 | 2.1297 | 2.2409 | 2.1227 | 2.3472 | 2.4221 | 2.2905 | 2.1984 | 2.1530 | 2.1210 | 2.2103 | 2.1503 | 2.0866 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 0.93065 | 0.94540 | 0.90527 | 0.91571 | 0.91205 | 0.96207 | 0.91242 | 1.0092 | 1.0435 | 0.98147 | 0.94664 | 0.92881 | 0.90836 | 0.95155 | 0.93304 | 0.89544 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 1.5114 | 1.5311 | 1.4635 | 1.4680 | 1.4761 | 1.5532 | 1.4713 | 1.6270 | 1.6789 | 1.5876 | 1.5237 | 1.4924 | 1.4699 | 1.5322 | 1.4904 | 1.4463 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.41929 | 0.43457 | 0.45654 | 0.46705 | 0.45654 | 0.41738 | 0.43553 | 0.43649 | 0.36103 | 0.39064 | 0.40592 | 0.42693 | 0.45845 | 0.45177 | 0.45272 | 0.46705 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.41929 | 0.43457 | 0.45654 | 0.46705 | 0.45654 | 0.41738 | 0.43553 | 0.43649 | 0.36103 | 0.39064 | 0.40592 | 0.42693 | 0.45845 | 0.45177 | 0.45272 | 0.46705 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -1.6939 | -1.7709 | -1.6975 | -1.7494 | -1.6592 | -1.7913 | -1.7224 | -1.9246 | -1.9729 | -1.8322 | -1.7966 | -1.7385 | -1.6461 | -1.7695 | -1.8049 | -1.6639 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -1.6939 | -1.7709 | -1.6975 | -1.7494 | -1.6592 | -1.7913 | -1.7224 | -1.9246 | -1.9729 | -1.8322 | -1.7966 | -1.7385 | -1.6461 | -1.7695 | -1.8049 | -1.6639 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.36620 | 0.37610 | 0.38590 | 0.41843 | 0.40725 | 0.35106 | 0.38917 | 0.37236 | 0.28507 | 0.33811 | 0.34751 | 0.36254 | 0.40971 | 0.38811 | 0.39144 | 0.41788 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.36620 | 0.37610 | 0.38590 | 0.41843 | 0.40725 | 0.35106 | 0.38917 | 0.37236 | 0.28507 | 0.33811 | 0.34751 | 0.36254 | 0.40971 | 0.38811 | 0.39144 | 0.41788 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.44300 | 0.37922 | 0.32654 | 0.41639 | 0.39589 | 0.39207 | 0.43355 | 0.42470 | 0.38858 | 0.43193 | 0.42074 | 0.33734 | 0.35670 | 0.43047 | 0.38840 | 0.40418 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.48347 | 0.41241 | 0.35439 | 0.45432 | 0.43092 | 0.42702 | 0.47187 | 0.46274 | 0.42314 | 0.47100 | 0.45949 | 0.36729 | 0.38824 | 0.46914 | 0.42351 | 0.44043 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.30707 | 0.26287 | 0.22632 | 0.28864 | 0.27444 | 0.27180 | 0.30050 | 0.29437 | 0.26934 | 0.29936 | 0.29161 | 0.23383 | 0.24725 | 0.29839 | 0.26923 | 0.28018 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.33510 | 0.28587 | 0.24564 | 0.31492 | 0.29867 | 0.29599 | 0.32707 | 0.32074 | 0.29331 | 0.32646 | 0.31850 | 0.25460 | 0.26908 | 0.32517 | 0.29353 | 0.30529 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.55929 | 0.58597 | 0.59585 | 0.59289 | 0.58202 | 0.55040 | 0.57411 | 0.55336 | 0.54842 | 0.54941 | 0.55336 | 0.62451 | 0.59585 | 0.54941 | 0.61759 | 0.60672 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.55929 | 0.58597 | 0.59585 | 0.59289 | 0.58202 | 0.55040 | 0.57411 | 0.55336 | 0.54842 | 0.54941 | 0.55336 | 0.62451 | 0.59585 | 0.54941 | 0.61759 | 0.60672 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -1.9650 | -1.8120 | -1.6469 | -1.7115 | -1.9162 | -1.9825 | -2.0535 | -2.0732 | -2.0148 | -2.0437 | -1.9782 | -1.5655 | -1.6845 | -2.1029 | -1.7007 | -1.7470 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -1.9650 | -1.8120 | -1.6469 | -1.7115 | -1.9162 | -1.9825 | -2.0535 | -2.0732 | -2.0148 | -2.0437 | -1.9782 | -1.5655 | -1.6845 | -2.1029 | -1.7007 | -1.7470 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.54093 | 0.56592 | 0.57989 | 0.57006 | 0.55074 | 0.53233 | 0.54303 | 0.53980 | 0.53262 | 0.52678 | 0.54412 | 0.59682 | 0.56670 | 0.52730 | 0.59007 | 0.57025 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.54093 | 0.56592 | 0.57989 | 0.57006 | 0.55074 | 0.53233 | 0.54303 | 0.53980 | 0.53262 | 0.52678 | 0.54412 | 0.59682 | 0.56670 | 0.52730 | 0.59007 | 0.57025 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.43991 | 0.30913 | 0.33005 | 0.35448 | 0.36893 | 0.36734 | 0.39288 | 0.40018 | 0.46793 | 0.49516 | 0.35938 | 0.49079 | 0.34294 | 0.39498 | 0.31681 | 0.38176 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.52815 | 0.37238 | 0.39676 | 0.42689 | 0.44561 | 0.44168 | 0.47386 | 0.48151 | 0.56354 | 0.59540 | 0.43271 | 0.59069 | 0.41160 | 0.47698 | 0.38156 | 0.45827 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.30506 | 0.21443 | 0.22893 | 0.24592 | 0.25585 | 0.25477 | 0.27246 | 0.27755 | 0.32449 | 0.34334 | 0.24917 | 0.34029 | 0.23781 | 0.27394 | 0.21972 | 0.26476 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.36628 | 0.25828 | 0.27526 | 0.29614 | 0.30901 | 0.30633 | 0.32862 | 0.33394 | 0.39077 | 0.41290 | 0.30005 | 0.40954 | 0.28546 | 0.33081 | 0.26466 | 0.31785 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.84474 | 0.87850 | 0.89103 | 0.89007 | 0.86885 | 0.87367 | 0.86017 | 0.87657 | 0.82739 | 0.84764 | 0.88332 | 0.87657 | 0.88717 | 0.87078 | 0.88235 | 0.87657 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.84474 | 0.87850 | 0.89103 | 0.89007 | 0.86885 | 0.87367 | 0.86017 | 0.87657 | 0.82739 | 0.84764 | 0.88332 | 0.87657 | 0.88717 | 0.87078 | 0.88235 | 0.87657 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.41845 | -0.34793 | -0.33334 | -0.31800 | -0.37641 | -0.36921 | -0.39049 | -0.34024 | -0.44761 | -0.42714 | -0.35172 | -0.37831 | -0.34385 | -0.37046 | -0.32402 | -0.36431 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.41845 | -0.34793 | -0.33334 | -0.31800 | -0.37641 | -0.36921 | -0.39049 | -0.34024 | -0.44761 | -0.42714 | -0.35172 | -0.37831 | -0.34385 | -0.37046 | -0.32402 | -0.36431 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.74996 | 0.79283 | 0.80099 | 0.80899 | 0.78006 | 0.77441 | 0.76812 | 0.78915 | 0.74013 | 0.74944 | 0.78037 | 0.77219 | 0.78837 | 0.77775 | 0.79648 | 0.77972 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.74996 | 0.79283 | 0.80099 | 0.80899 | 0.78006 | 0.77441 | 0.76812 | 0.78915 | 0.74013 | 0.74944 | 0.78037 | 0.77219 | 0.78837 | 0.77775 | 0.79648 | 0.77972 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.30865 | 0.27514 | 0.28921 | 0.24890 | 0.25094 | 0.26034 | 0.27138 | 0.28679 | 0.26825 | 0.24955 | 0.27686 | 0.31343 | 0.31205 | 0.27736 | 0.24944 | 0.26049 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.31911 | 0.28440 | 0.29891 | 0.25717 | 0.25935 | 0.26914 | 0.28038 | 0.29653 | 0.27717 | 0.25792 | 0.28614 | 0.32392 | 0.32246 | 0.28660 | 0.25776 | 0.26927 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.21395 | 0.19073 | 0.20048 | 0.17253 | 0.17395 | 0.18048 | 0.18812 | 0.19882 | 0.18596 | 0.17299 | 0.19191 | 0.21727 | 0.21631 | 0.19225 | 0.17290 | 0.18057 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.22122 | 0.19713 | 0.20723 | 0.17827 | 0.17978 | 0.18656 | 0.19437 | 0.20555 | 0.19214 | 0.17877 | 0.19836 | 0.22455 | 0.22352 | 0.19866 | 0.17870 | 0.18666 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.90787 | 0.87030 | 0.89982 | 0.85331 | 0.87388 | 0.89267 | 0.88104 | 0.89445 | 0.83184 | 0.92129 | 0.91771 | 0.86225 | 0.86404 | 0.87657 | 0.88909 | 0.89445 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.90787 | 0.87030 | 0.89982 | 0.85331 | 0.87388 | 0.89267 | 0.88104 | 0.89445 | 0.83184 | 0.92129 | 0.91771 | 0.86225 | 0.86404 | 0.87657 | 0.88909 | 0.89445 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.27524 | -0.32198 | -0.30375 | -0.37800 | -0.29636 | -0.28463 | -0.30200 | -0.29194 | -0.38576 | -0.23053 | -0.24147 | -0.34481 | -0.32601 | -0.30525 | -0.28435 | -0.27955 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.27524 | -0.32198 | -0.30375 | -0.37800 | -0.29636 | -0.28463 | -0.30200 | -0.29194 | -0.38576 | -0.23053 | -0.24147 | -0.34481 | -0.32601 | -0.30525 | -0.28435 | -0.27955 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.87578 | 0.85207 | 0.86713 | 0.84791 | 0.86596 | 0.87606 | 0.86747 | 0.86541 | 0.83651 | 0.88908 | 0.88888 | 0.84168 | 0.85336 | 0.86617 | 0.87011 | 0.86636 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.87578 | 0.85207 | 0.86713 | 0.84791 | 0.86596 | 0.87606 | 0.86747 | 0.86541 | 0.83651 | 0.88908 | 0.88888 | 0.84168 | 0.85336 | 0.86617 | 0.87011 | 0.86636 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 0.91313 | 0.86270 | 0.89931 | 0.82045 | 0.80390 | 0.94679 | 0.97213 | 0.95632 | 0.98297 | 0.90010 | 0.89582 | 0.81840 | 0.79458 | 0.84306 | 0.86854 | 0.84687 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.4790 | 1.4122 | 1.4761 | 1.3409 | 1.3262 | 1.5460 | 1.5880 | 1.5661 | 1.5935 | 1.4734 | 1.4820 | 1.3363 | 1.3227 | 1.3748 | 1.4227 | 1.3858 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.66723 | 0.63360 | 0.65117 | 0.59923 | 0.58531 | 0.68630 | 0.71134 | 0.69751 | 0.71126 | 0.65208 | 0.64953 | 0.59545 | 0.57999 | 0.61738 | 0.63042 | 0.61689 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.1129 | 1.0693 | 1.0937 | 1.0068 | 0.99068 | 1.1486 | 1.1973 | 1.1747 | 1.1806 | 1.0937 | 1.1001 | 0.99677 | 0.99155 | 1.0370 | 1.0577 | 1.0367 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.67978 | 0.70599 | 0.72097 | 0.70974 | 0.73783 | 0.69288 | 0.68539 | 0.69850 | 0.65918 | 0.70037 | 0.71348 | 0.72659 | 0.73034 | 0.71723 | 0.72659 | 0.71910 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.67978 | 0.70599 | 0.72097 | 0.70974 | 0.73783 | 0.69288 | 0.68539 | 0.69850 | 0.65918 | 0.70037 | 0.71348 | 0.72659 | 0.73034 | 0.71723 | 0.72659 | 0.71910 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.81209 | -0.74244 | -0.71174 | -0.72030 | -0.71658 | -0.79302 | -0.81055 | -0.81226 | -0.85391 | -0.79304 | -0.76507 | -0.73336 | -0.68163 | -0.75236 | -0.71833 | -0.72473 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.81209 | -0.74244 | -0.71174 | -0.72030 | -0.71658 | -0.79302 | -0.81055 | -0.81226 | -0.85391 | -0.79304 | -0.76507 | -0.73336 | -0.68163 | -0.75236 | -0.71833 | -0.72473 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.60954 | 0.63131 | 0.64139 | 0.63596 | 0.64257 | 0.60647 | 0.59551 | 0.60941 | 0.58558 | 0.61764 | 0.61804 | 0.65710 | 0.65621 | 0.63350 | 0.63697 | 0.62924 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.60954 | 0.63131 | 0.64139 | 0.63596 | 0.64257 | 0.60647 | 0.59551 | 0.60941 | 0.58558 | 0.61764 | 0.61804 | 0.65710 | 0.65621 | 0.63350 | 0.63697 | 0.62924 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.5446 | 1.4917 | 1.4852 | 1.4660 | 1.5316 | 1.5771 | 1.5616 | 1.5196 | 1.6704 | 1.6686 | 1.5086 | 1.5009 | 1.4349 | 1.5833 | 1.4805 | 1.4809 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.1358 | 2.0621 | 2.0615 | 2.0304 | 2.1520 | 2.1780 | 2.1677 | 2.1346 | 2.3070 | 2.3062 | 2.0981 | 2.0993 | 2.0019 | 2.1709 | 2.0475 | 2.0368 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.0707 | 1.0340 | 1.0295 | 1.0161 | 1.0616 | 1.0931 | 1.0824 | 1.0533 | 1.1578 | 1.1566 | 1.0456 | 1.0404 | 0.99466 | 1.0974 | 1.0263 | 1.0263 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.4805 | 1.4293 | 1.4289 | 1.4073 | 1.4916 | 1.5097 | 1.5024 | 1.4796 | 1.5991 | 1.5987 | 1.4542 | 1.4550 | 1.3879 | 1.5046 | 1.4192 | 1.4118 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.25103 | 0.22888 | 0.29614 | 0.27317 | 0.23544 | 0.23872 | 0.25267 | 0.26661 | 0.22806 | 0.23380 | 0.26169 | 0.25759 | 0.25267 | 0.24446 | 0.26169 | 0.25349 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.25103 | 0.22888 | 0.29614 | 0.27317 | 0.23544 | 0.23872 | 0.25267 | 0.26661 | 0.22806 | 0.23380 | 0.26169 | 0.25759 | 0.25267 | 0.24446 | 0.26169 | 0.25349 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -3.7599 | -3.6868 | -3.2640 | -3.3256 | -3.4848 | -3.7231 | -3.6868 | -3.5138 | -3.9150 | -3.7750 | -3.6424 | -3.4276 | -3.4932 | -3.7870 | -3.6100 | -3.5642 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -3.7599 | -3.6868 | -3.2640 | -3.3256 | -3.4848 | -3.7231 | -3.6868 | -3.5138 | -3.9150 | -3.7750 | -3.6424 | -3.4276 | -3.4932 | -3.7870 | -3.6100 | -3.5642 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.26293 | 0.25700 | 0.30423 | 0.29174 | 0.26260 | 0.26565 | 0.26613 | 0.28522 | 0.25292 | 0.25521 | 0.27364 | 0.27824 | 0.28070 | 0.26681 | 0.28057 | 0.27539 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.26293 | 0.25700 | 0.30423 | 0.29174 | 0.26260 | 0.26565 | 0.26613 | 0.28522 | 0.25292 | 0.25521 | 0.27364 | 0.27824 | 0.28070 | 0.26681 | 0.28057 | 0.27539 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.44209 | 0.42132 | 0.41296 | 0.41652 | 0.45056 | 0.44473 | 0.43332 | 0.44958 | 0.45117 | 0.43874 | 0.42940 | 0.42827 | 0.42769 | 0.44641 | 0.41081 | 0.42502 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.44786 | 0.42693 | 0.41847 | 0.42213 | 0.45684 | 0.45097 | 0.43900 | 0.45541 | 0.45742 | 0.44437 | 0.43525 | 0.43389 | 0.43344 | 0.45256 | 0.41628 | 0.43063 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.63809 | 0.64354 | 0.62515 | 0.63885 | 0.63857 | 0.64728 | 0.64037 | 0.63206 | 0.66029 | 0.64449 | 0.64179 | 0.63457 | 0.63091 | 0.63858 | 0.62660 | 0.63593 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.64375 | 0.64906 | 0.63056 | 0.64434 | 0.64411 | 0.65301 | 0.64594 | 0.63733 | 0.66595 | 0.64998 | 0.64731 | 0.64002 | 0.63631 | 0.64400 | 0.63202 | 0.64170 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.5111 | 1.9205 | 2.2577 | 2.3920 | 2.2452 | 2.3592 | 1.9418 | 2.1815 | 2.2090 | 2.4016 | 2.4325 | 1.9984 | 2.2366 | 2.1973 | 2.1167 | 1.9862 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 5.0221 | 3.8409 | 4.5155 | 4.7839 | 4.4903 | 4.7184 | 3.8836 | 4.3630 | 4.4180 | 4.8032 | 4.8649 | 3.9969 | 4.4732 | 4.3947 | 4.2334 | 3.9723 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.7407 | 1.3313 | 1.5652 | 1.6584 | 1.5558 | 1.6354 | 1.3457 | 1.5118 | 1.5314 | 1.6651 | 1.6862 | 1.3850 | 1.5505 | 1.5233 | 1.4674 | 1.3766 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.4815 | 2.6627 | 3.1305 | 3.3169 | 3.1116 | 3.2709 | 2.6915 | 3.0237 | 3.0627 | 3.3302 | 3.3724 | 2.7699 | 3.1009 | 3.0466 | 2.9348 | 2.7532 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.07000 | 0.12000 | 0.13000 | 0.09000 | 0.10000 | 0.11000 | 0.10000 | 0.07000 | 0.07000 | 0.10000 | 0.11000 | 0.07000 | 0.11000 | 0.10000 | 0.12000 | 0.09000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.07000 | 0.12000 | 0.13000 | 0.09000 | 0.10000 | 0.11000 | 0.10000 | 0.07000 | 0.07000 | 0.10000 | 0.11000 | 0.07000 | 0.11000 | 0.10000 | 0.12000 | 0.09000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -3.4772 | -2.6580 | -3.1254 | -3.3131 | -3.1061 | -3.2672 | -2.6802 | -3.0171 | -3.0557 | -3.3248 | -3.3686 | -2.7628 | -3.0942 | -3.0408 | -2.9256 | -2.7377 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -3.4772 | -2.6580 | -3.1254 | -3.3131 | -3.1061 | -3.2672 | -2.6802 | -3.0171 | -3.0557 | -3.3248 | -3.3686 | -2.7628 | -3.0942 | -3.0408 | -2.9256 | -2.7377 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09471 | 0.09619 | 0.09538 | 0.09751 | 0.09318 | 0.09633 | 0.09545 | 0.09346 | 0.09333 | 0.09886 | 0.10035 | 0.09714 | 0.09706 | 0.09664 | 0.09833 | 0.09560 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09471 | 0.09619 | 0.09538 | 0.09751 | 0.09318 | 0.09633 | 0.09545 | 0.09346 | 0.09333 | 0.09886 | 0.10035 | 0.09714 | 0.09706 | 0.09664 | 0.09833 | 0.09560 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.79789 | 0.78757 | 0.78620 | 0.78243 | 0.79206 | 0.80025 | 0.79500 | 0.79365 | 0.80630 | 0.80287 | 0.78969 | 0.79061 | 0.78583 | 0.79523 | 0.78633 | 0.78619 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.80687 | 0.79621 | 0.79486 | 0.79119 | 0.80082 | 0.80903 | 0.80399 | 0.80239 | 0.81517 | 0.81171 | 0.79842 | 0.79929 | 0.79467 | 0.80379 | 0.79513 | 0.79481 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.67236 | 0.66036 | 0.64471 | 0.64678 | 0.66857 | 0.67638 | 0.66912 | 0.66745 | 0.69363 | 0.67984 | 0.67099 | 0.65922 | 0.65536 | 0.67071 | 0.65279 | 0.65612 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.67463 | 0.66245 | 0.64679 | 0.64893 | 0.67070 | 0.67862 | 0.67129 | 0.66969 | 0.69582 | 0.68202 | 0.67303 | 0.66130 | 0.65741 | 0.67287 | 0.65485 | 0.65818 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.71419 | 0.70040 | 0.69578 | 0.68561 | 0.71099 | 0.71420 | 0.70682 | 0.70546 | 0.72824 | 0.71088 | 0.70851 | 0.70157 | 0.69993 | 0.71216 | 0.68679 | 0.68954 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.75071 | 0.73617 | 0.73112 | 0.72042 | 0.74755 | 0.75068 | 0.74286 | 0.74143 | 0.76570 | 0.74734 | 0.74470 | 0.73759 | 0.73587 | 0.74852 | 0.72156 | 0.72431 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0099 | 1.0163 | 1.0171 | 1.0099 | 1.0100 | 1.0202 | 1.0091 | 1.0112 | 1.0168 | 1.0090 | 1.0086 | 1.0141 | 1.0135 | 1.0089 | 1.0106 | 1.0051 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0199 | 2.0326 | 2.0343 | 2.0197 | 2.0199 | 2.0404 | 2.0182 | 2.0224 | 2.0336 | 2.0181 | 2.0173 | 2.0282 | 2.0271 | 2.0179 | 2.0213 | 2.0103 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.70013 | 0.70451 | 0.70508 | 0.70005 | 0.70016 | 0.70720 | 0.69955 | 0.70104 | 0.70488 | 0.69953 | 0.69926 | 0.70304 | 0.70261 | 0.69937 | 0.70061 | 0.69683 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4003 | 1.4090 | 1.4102 | 1.4001 | 1.4003 | 1.4144 | 1.3991 | 1.4021 | 1.4098 | 1.3991 | 1.3985 | 1.4061 | 1.4052 | 1.3987 | 1.4012 | 1.3937 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.25313 | 0.23868 | 0.26695 | 0.26589 | 0.25611 | 0.24166 | 0.25228 | 0.25420 | 0.24761 | 0.26312 | 0.24846 | 0.23996 | 0.24910 | 0.26206 | 0.25377 | 0.25484 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.25313 | 0.23868 | 0.26695 | 0.26589 | 0.25611 | 0.24166 | 0.25228 | 0.25420 | 0.24761 | 0.26312 | 0.24846 | 0.23996 | 0.24910 | 0.26206 | 0.25377 | 0.25484 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3878 | -1.3925 | -1.3888 | -1.3865 | -1.3878 | -1.3910 | -1.3883 | -1.3878 | -1.3909 | -1.3877 | -1.3869 | -1.3907 | -1.3902 | -1.3861 | -1.3888 | -1.3864 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.3978 | -1.4058 | -1.4055 | -1.3944 | -1.3947 | -1.4069 | -1.3953 | -1.3966 | -1.4046 | -1.3950 | -1.3931 | -1.4027 | -1.4022 | -1.3939 | -1.3973 | -1.3893 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.25069 | 0.24936 | 0.25125 | 0.25092 | 0.25032 | 0.25022 | 0.25015 | 0.25053 | 0.24998 | 0.25040 | 0.25054 | 0.24983 | 0.25005 | 0.25104 | 0.25012 | 0.25035 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.25130 | 0.24874 | 0.25252 | 0.25184 | 0.25065 | 0.25035 | 0.25029 | 0.25102 | 0.24988 | 0.25078 | 0.25105 | 0.24962 | 0.25013 | 0.25206 | 0.25024 | 0.25070 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 0.99249 | 0.96606 | 0.94796 | 0.94493 | 0.98441 | 0.99920 | 0.98843 | 0.98753 | 1.0126 | 0.98322 | 0.97706 | 0.97658 | 0.97557 | 0.98532 | 0.94671 | 0.96728 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.1055 | 1.0752 | 1.0562 | 1.0524 | 1.0961 | 1.1131 | 1.1013 | 1.0996 | 1.1276 | 1.0946 | 1.0882 | 1.0908 | 1.0876 | 1.0962 | 1.0550 | 1.0776 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0190 | 1.0044 | 1.0220 | 1.0107 | 1.0044 | 1.0061 | 1.0025 | 1.0084 | 1.0031 | 1.0014 | 1.0067 | 1.0073 | 1.0014 | 1.0061 | 1.0024 | 1.0038 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0380 | 2.0088 | 2.0440 | 2.0214 | 2.0088 | 2.0122 | 2.0050 | 2.0168 | 2.0062 | 2.0029 | 2.0135 | 2.0147 | 2.0027 | 2.0122 | 2.0047 | 2.0077 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.70639 | 0.69624 | 0.70845 | 0.70062 | 0.69627 | 0.69744 | 0.69497 | 0.69903 | 0.69540 | 0.69419 | 0.69791 | 0.69841 | 0.69420 | 0.69744 | 0.69490 | 0.69596 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4128 | 1.3925 | 1.4169 | 1.4012 | 1.3925 | 1.3949 | 1.3899 | 1.3981 | 1.3908 | 1.3884 | 1.3958 | 1.3968 | 1.3884 | 1.3949 | 1.3898 | 1.3919 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.27236 | 0.27144 | 0.25632 | 0.26218 | 0.26033 | 0.27298 | 0.26342 | 0.26619 | 0.28100 | 0.28717 | 0.26835 | 0.25571 | 0.27791 | 0.25787 | 0.27082 | 0.26589 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.27236 | 0.27144 | 0.25632 | 0.26218 | 0.26033 | 0.27298 | 0.26342 | 0.26619 | 0.28100 | 0.28717 | 0.26835 | 0.25571 | 0.27791 | 0.25787 | 0.27082 | 0.26589 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3888 | -1.3831 | -1.3913 | -1.3870 | -1.3827 | -1.3826 | -1.3844 | -1.3851 | -1.3810 | -1.3820 | -1.3852 | -1.3873 | -1.3820 | -1.3842 | -1.3837 | -1.3849 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.4087 | -1.3882 | -1.4121 | -1.3968 | -1.3870 | -1.3878 | -1.3866 | -1.3916 | -1.3863 | -1.3848 | -1.3902 | -1.3926 | -1.3849 | -1.3903 | -1.3858 | -1.3868 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25168 | 0.25185 | 0.25079 | 0.25100 | 0.25180 | 0.25205 | 0.25100 | 0.25123 | 0.25271 | 0.25197 | 0.25102 | 0.25030 | 0.25199 | 0.25147 | 0.25123 | 0.25078 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25332 | 0.25374 | 0.25134 | 0.25192 | 0.25344 | 0.25408 | 0.25194 | 0.25237 | 0.25539 | 0.25389 | 0.25198 | 0.25062 | 0.25403 | 0.25276 | 0.25245 | 0.25153 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.83786 | 0.83234 | 0.81910 | 0.80792 | 0.84122 | 0.84983 | 0.83523 | 0.83572 | 0.86158 | 0.84126 | 0.83571 | 0.82774 | 0.82094 | 0.84254 | 0.80870 | 0.81542 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 0.89419 | 0.88932 | 0.87373 | 0.86232 | 0.89824 | 0.90794 | 0.89215 | 0.89271 | 0.92084 | 0.89880 | 0.89229 | 0.88394 | 0.87742 | 0.90020 | 0.86306 | 0.87090 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0326 | 1.0152 | 1.0379 | 1.0199 | 1.0205 | 1.0180 | 1.0061 | 1.0190 | 1.0270 | 1.0194 | 1.0179 | 1.0043 | 1.0043 | 1.0272 | 1.0045 | 1.0061 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0652 | 2.0305 | 2.0759 | 2.0398 | 2.0409 | 2.0359 | 2.0123 | 2.0380 | 2.0539 | 2.0389 | 2.0357 | 2.0085 | 2.0086 | 2.0545 | 2.0090 | 2.0122 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.71583 | 0.70377 | 0.71947 | 0.70705 | 0.70742 | 0.70568 | 0.69752 | 0.70640 | 0.71192 | 0.70675 | 0.70564 | 0.69615 | 0.69620 | 0.71211 | 0.69636 | 0.69748 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4317 | 1.4075 | 1.4389 | 1.4141 | 1.4148 | 1.4114 | 1.3950 | 1.4128 | 1.4238 | 1.4135 | 1.4113 | 1.3923 | 1.3924 | 1.4242 | 1.3927 | 1.3950 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.23529 | 0.25382 | 0.23497 | 0.24537 | 0.23887 | 0.24537 | 0.25837 | 0.23562 | 0.22684 | 0.23497 | 0.23107 | 0.27982 | 0.26779 | 0.22782 | 0.25772 | 0.24277 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.23529 | 0.25382 | 0.23497 | 0.24537 | 0.23887 | 0.24537 | 0.25837 | 0.23562 | 0.22684 | 0.23497 | 0.23107 | 0.27982 | 0.26779 | 0.22782 | 0.25772 | 0.24277 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.4001 | -1.3907 | -1.4035 | -1.3942 | -1.3947 | -1.3915 | -1.3867 | -1.3937 | -1.3992 | -1.3945 | -1.3931 | -1.3839 | -1.3838 | -1.3990 | -1.3844 | -1.3868 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.4284 | -1.4036 | -1.4348 | -1.4098 | -1.4100 | -1.4043 | -1.3917 | -1.4069 | -1.4197 | -1.4100 | -1.4059 | -1.3888 | -1.3895 | -1.4201 | -1.3892 | -1.3904 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.24851 | 0.24997 | 0.24770 | 0.24903 | 0.24877 | 0.24968 | 0.25052 | 0.24894 | 0.24783 | 0.24891 | 0.24905 | 0.25156 | 0.25163 | 0.24797 | 0.25133 | 0.25024 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.24727 | 0.24995 | 0.24562 | 0.24813 | 0.24756 | 0.24936 | 0.25108 | 0.24794 | 0.24580 | 0.24785 | 0.24811 | 0.25319 | 0.25325 | 0.24609 | 0.25259 | 0.25044 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.2756 | 1.2799 | 1.2475 | 1.2302 | 1.2638 | 1.2793 | 1.2651 | 1.2692 | 1.3138 | 1.2866 | 1.2702 | 1.2600 | 1.2548 | 1.2667 | 1.2441 | 1.2592 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.5925 | 1.6033 | 1.5574 | 1.5370 | 1.5786 | 1.6013 | 1.5821 | 1.5905 | 1.6434 | 1.6097 | 1.5880 | 1.5757 | 1.5718 | 1.5814 | 1.5575 | 1.5741 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0442 | 1.0149 | 1.0326 | 1.0207 | 1.0063 | 1.0216 | 1.0016 | 1.0166 | 1.0146 | 1.0128 | 1.0131 | 1.0058 | 1.0116 | 1.0173 | 0.99871 | 1.0034 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0883 | 2.0299 | 2.0653 | 2.0413 | 2.0126 | 2.0432 | 2.0031 | 2.0332 | 2.0292 | 2.0257 | 2.0262 | 2.0116 | 2.0232 | 2.0347 | 1.9974 | 2.0067 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.72371 | 0.70357 | 0.71575 | 0.70750 | 0.69758 | 0.70823 | 0.69435 | 0.70476 | 0.70335 | 0.70217 | 0.70233 | 0.69724 | 0.70123 | 0.70526 | 0.69229 | 0.69563 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4474 | 1.4071 | 1.4315 | 1.4150 | 1.3952 | 1.4165 | 1.3887 | 1.4095 | 1.4067 | 1.4043 | 1.4047 | 1.3945 | 1.4025 | 1.4105 | 1.3846 | 1.3913 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24254 | 0.25878 | 0.26011 | 0.25613 | 0.27634 | 0.25282 | 0.26375 | 0.25878 | 0.24254 | 0.26143 | 0.25712 | 0.26143 | 0.26706 | 0.24520 | 0.30749 | 0.25845 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24254 | 0.25878 | 0.26011 | 0.25613 | 0.27634 | 0.25282 | 0.26375 | 0.25878 | 0.24254 | 0.26143 | 0.25712 | 0.26143 | 0.26706 | 0.24520 | 0.30749 | 0.25845 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.4036 | -1.3895 | -1.3978 | -1.3903 | -1.3842 | -1.3913 | -1.3821 | -1.3905 | -1.3887 | -1.3879 | -1.3879 | -1.3857 | -1.3867 | -1.3896 | -1.3791 | -1.3849 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4439 | -1.4023 | -1.4271 | -1.4110 | -1.3892 | -1.4079 | -1.3850 | -1.4034 | -1.4015 | -1.3993 | -1.3984 | -1.3904 | -1.3991 | -1.4048 | -1.3804 | -1.3863 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.24887 | 0.25041 | 0.24963 | 0.25111 | 0.25144 | 0.25027 | 0.25194 | 0.25010 | 0.25070 | 0.25090 | 0.25073 | 0.25083 | 0.25134 | 0.25060 | 0.25289 | 0.25067 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.24798 | 0.25083 | 0.24947 | 0.25200 | 0.25292 | 0.25060 | 0.25380 | 0.25026 | 0.25129 | 0.25185 | 0.25140 | 0.25171 | 0.25261 | 0.25105 | 0.25574 | 0.25132 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.42810 | 0.42360 | 0.39847 | 0.43394 | 0.41372 | 0.43217 | 0.42796 | 0.40581 | 0.43780 | 0.43054 | 0.42501 | 0.40871 | 0.42763 | 0.40747 | 0.40951 | 0.43142 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.43059 | 0.42592 | 0.40075 | 0.43653 | 0.41610 | 0.43452 | 0.43037 | 0.40810 | 0.44037 | 0.43295 | 0.42737 | 0.41110 | 0.43015 | 0.40978 | 0.41180 | 0.43399 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.33241 | 0.32597 | 0.30502 | 0.33416 | 0.33849 | 0.32438 | 0.32951 | 0.31278 | 0.34064 | 0.31952 | 0.32300 | 0.31986 | 0.32171 | 0.33425 | 0.31643 | 0.31619 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.33365 | 0.32721 | 0.30619 | 0.33540 | 0.33984 | 0.32565 | 0.33078 | 0.31402 | 0.34193 | 0.32071 | 0.32428 | 0.32118 | 0.32290 | 0.33553 | 0.31768 | 0.31738 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.55775 | 0.56502 | 0.53436 | 0.56374 | 0.53419 | 0.57454 | 0.54841 | 0.52748 | 0.58162 | 0.55643 | 0.53628 | 0.55431 | 0.54865 | 0.57331 | 0.55183 | 0.53785 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.56182 | 0.56908 | 0.53822 | 0.56768 | 0.53807 | 0.57865 | 0.55233 | 0.53105 | 0.58558 | 0.56047 | 0.54015 | 0.55816 | 0.55260 | 0.57729 | 0.55579 | 0.54155 |
| eval/lm/c4_en-validation/CE loss | lower | 2.9335 | 2.9084 | 2.8720 | 2.8752 | 2.9260 | 2.9369 | 2.9211 | 2.9150 | 2.9666 | 2.9348 | 2.9171 | 2.9040 | 2.8939 | 2.9289 | 2.8803 | 2.9004 |
| eval/lm/c4_en-validation/PPL | lower | 18.79 | 18.33 | 17.67 | 17.73 | 18.65 | 18.86 | 18.56 | 18.45 | 19.43 | 18.82 | 18.49 | 18.25 | 18.06 | 18.71 | 17.82 | 18.18 |
| eval/lm/dolma_books-validation/CE loss | lower | 2.8264 | 2.7947 | 2.7548 | 2.7533 | 2.8094 | 2.8313 | 2.8119 | 2.7968 | 2.8679 | 2.8290 | 2.8019 | 2.7868 | 2.7772 | 2.8136 | 2.7612 | 2.7851 |
| eval/lm/dolma_books-validation/PPL | lower | 16.88 | 16.36 | 15.72 | 15.69 | 16.60 | 16.97 | 16.64 | 16.39 | 17.60 | 16.93 | 16.48 | 16.23 | 16.07 | 16.67 | 15.82 | 16.20 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.0739 | 3.0483 | 3.0114 | 3.0158 | 3.0627 | 3.0744 | 3.0588 | 3.0548 | 3.1014 | 3.0702 | 3.0553 | 3.0425 | 3.0306 | 3.0651 | 3.0196 | 3.0367 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 21.63 | 21.08 | 20.32 | 20.40 | 21.39 | 21.64 | 21.30 | 21.22 | 22.23 | 21.55 | 21.23 | 20.96 | 20.71 | 21.44 | 20.48 | 20.84 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.1317 | 2.1124 | 2.0870 | 2.0889 | 2.1232 | 2.1379 | 2.1232 | 2.1207 | 2.1637 | 2.1351 | 2.1204 | 2.1097 | 2.0985 | 2.1256 | 2.0949 | 2.1094 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 8.4293 | 8.2680 | 8.0608 | 8.0759 | 8.3582 | 8.4817 | 8.3576 | 8.3370 | 8.7034 | 8.4578 | 8.3347 | 8.2460 | 8.1541 | 8.3782 | 8.1244 | 8.2429 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.2492 | 3.2219 | 3.1927 | 3.1933 | 3.2391 | 3.2503 | 3.2355 | 3.2307 | 3.2788 | 3.2483 | 3.2321 | 3.2150 | 3.2095 | 3.2431 | 3.2012 | 3.2200 |
| eval/lm/dolma_reddit-validation/PPL | lower | 25.77 | 25.08 | 24.35 | 24.37 | 25.51 | 25.80 | 25.42 | 25.30 | 26.54 | 25.75 | 25.33 | 24.90 | 24.77 | 25.61 | 24.56 | 25.03 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.3474 | 1.3275 | 1.2980 | 1.3019 | 1.3414 | 1.3504 | 1.3366 | 1.3179 | 1.3597 | 1.3326 | 1.3180 | 1.3255 | 1.3158 | 1.3395 | 1.3075 | 1.3200 |
| eval/lm/dolma_stack-validation/PPL | lower | 3.8475 | 3.7716 | 3.6621 | 3.6762 | 3.8243 | 3.8590 | 3.8060 | 3.7356 | 3.8950 | 3.7907 | 3.7360 | 3.7641 | 3.7278 | 3.8170 | 3.6970 | 3.7433 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.5806 | 2.5556 | 2.5202 | 2.5178 | 2.5745 | 2.5803 | 2.5668 | 2.5636 | 2.6122 | 2.5786 | 2.5615 | 2.5459 | 2.5374 | 2.5712 | 2.5188 | 2.5435 |
| eval/lm/dolma_wiki-validation/PPL | lower | 13.20 | 12.88 | 12.43 | 12.40 | 13.12 | 13.20 | 13.02 | 12.98 | 13.63 | 13.18 | 12.96 | 12.75 | 12.65 | 13.08 | 12.41 | 12.72 |
| eval/lm/ice-validation/CE loss | lower | 2.9910 | 2.9695 | 2.9403 | 2.9504 | 2.9871 | 2.9992 | 2.9772 | 2.9700 | 3.0230 | 2.9880 | 2.9771 | 2.9731 | 2.9814 | 2.9994 | 2.9670 | 2.9736 |
| eval/lm/ice-validation/PPL | lower | 19.91 | 19.48 | 18.92 | 19.11 | 19.83 | 20.07 | 19.63 | 19.49 | 20.55 | 19.85 | 19.63 | 19.55 | 19.72 | 20.07 | 19.43 | 19.56 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 3.0531 | 3.0355 | 3.0025 | 3.0070 | 3.0450 | 3.0557 | 3.0476 | 3.0498 | 3.0861 | 3.0558 | 3.0407 | 3.0360 | 3.0201 | 3.0520 | 3.0108 | 3.0335 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 21.18 | 20.81 | 20.14 | 20.23 | 21.01 | 21.24 | 21.06 | 21.11 | 21.89 | 21.24 | 20.92 | 20.82 | 20.49 | 21.16 | 20.30 | 20.77 |
| eval/lm/pile-validation/CE loss | lower | 2.2136 | 2.1939 | 2.1635 | 2.1653 | 2.2090 | 2.2199 | 2.2051 | 2.1987 | 2.2440 | 2.2144 | 2.1984 | 2.1913 | 2.1788 | 2.2098 | 2.1716 | 2.1847 |
| eval/lm/pile-validation/PPL | lower | 9.1488 | 8.9705 | 8.7014 | 8.7170 | 9.1065 | 9.2062 | 9.0711 | 9.0129 | 9.4306 | 9.1561 | 9.0105 | 8.9466 | 8.8353 | 9.1143 | 8.7723 | 8.8880 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.5139 | 2.4877 | 2.4495 | 2.4582 | 2.5032 | 2.5263 | 2.5012 | 2.4980 | 2.5695 | 2.5258 | 2.4983 | 2.4859 | 2.4707 | 2.5146 | 2.4664 | 2.4783 |
| eval/lm/wikitext_103-validation/PPL | lower | 12.35 | 12.03 | 11.58 | 11.68 | 12.22 | 12.51 | 12.20 | 12.16 | 13.06 | 12.50 | 12.16 | 12.01 | 11.83 | 12.36 | 11.78 | 11.92 |
| throughput/in-loop eval batches | see metric | 557.0 | 557.0 | 1659.0 | 1659.0 | 1659.0 | 1659.0 | 1659.0 | 828.0 | 828.0 | 828.0 | 828.0 | 1659.0 | 1659.0 | 1659.0 | 1729.0 | 1659.0 |
| throughput/in-loop eval time (s) | see metric | 67.18 | 71.22 | 138.5 | 121.1 | 115.9 | 119.5 | 115.6 | 78.51 | 76.71 | 83.26 | 77.41 | 147.3 | 192.0 | 173.4 | 123.5 | 113.8 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| eg-810m-cx2-eg24e2k-lr5.6e-4-r1<br>`138dbr8w` | finished | original | 27579383808.0 | 70138 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/138dbr8w) |
| eg-810m-cx2-eg96e8k-lr5.6e-4-r1<br>`0qmqmo4l` | finished | original | 27649376256.0 | 70316 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0qmqmo4l) |
| int-810m-cx2-intd256e8k-lr5.6e-4-r1<br>`7nsy68db` | finished | original | 27253800960.0 | 69310 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7nsy68db) |
| int-810m-cx2-intw256e8k-lr5.6e-4-r1<br>`jpbqhfvc` | finished | original | 27805089792.0 | 70712 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/jpbqhfvc) |
| 810m-cx2-b384k-lr1.12e-3-r3<br>`sxivrph5` | finished | gpu8-ep1mb2 | 27602583552.0 | 70197 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sxivrph5) |
| 810m-cx2-b384k-lr2.8e-4-r3<br>`uh4el1df` | finished | gpu8-ep1mb2 | 27602583552.0 | 70197 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/uh4el1df) |
| 810m-cx2-b384k-lr5.6e-4-r3<br>`v5puakhq` | finished | gpu8-ep1mb2 | 27602583552.0 | 70197 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/v5puakhq) |
| 810m-cx2-b512k-lr1.2e-3-r1<br>`d13uavyt` | finished | gpu8-ep1mb4 | 27602714624.0 | 52648 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/d13uavyt) |
| 810m-cx2-b512k-lr1.5e-4-r1<br>`fcqkb55w` | finished | gpu8-ep1mb4 | 27602714624.0 | 52648 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fcqkb55w) |
| 810m-cx2-b512k-lr3e-4-r1<br>`ogp6mrt6` | finished | gpu8-ep1mb4 | 27602714624.0 | 52648 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ogp6mrt6) |
| 810m-cx2-b512k-lr6e-4-r1<br>`okb4e1u0` | finished | gpu8-ep1mb4 | 27602714624.0 | 52648 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/okb4e1u0) |
| q3-810m-cx2-q3am128e8k-lr5.6e-4-r1<br>`vb315h4w` | finished | original | 27687124992.0 | 70412 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vb315h4w) |
| q3-810m-cx2-q3td128e8k-lr5.6e-4-r1<br>`m3zgibjj` | finished | original | 27616346112.0 | 70232 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/m3zgibjj) |
| se-810m-cx2-se0m9-lr5.6e-4-r1<br>`epue7vyg` | finished | original | 27602583552.0 | 70197 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/epue7vyg) |
| sp-810m-cx2-sp192e4k-lr4e-4-r2<br>`bh87mzps` | finished | original | 27742568448.0 | 70553 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bh87mzps) |
| sp-810m-cx2-sp96e4k-lr5e-4-r1<br>`0d9gw8af` | finished | original | 27649376256.0 | 70316 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0d9gw8af) |

## 810m Cx4

| metric | direction | eg-810m-cx4-eg24e2k-lr4e-4-r1<br>`q50qk891` | eg-810m-cx4-eg96e8k-lr4e-4-r1<br>`7cbm4c9b` | int-810m-cx4-intd256e8k-lr4e-4-r1<br>`xzja2ww7` | int-810m-cx4-intw256e8k-lr4e-4-r1<br>`58ftjxmw` | 810m-cx4-b512k-lr1.6e-3-r1<br>`ag7nvx2l` | 810m-cx4-b512k-lr2e-4-r1<br>`nr84d31z` | 810m-cx4-b512k-lr4e-4-r1<br>`5rqlw5fd` | 810m-cx4-b512k-lr8e-4-r1<br>`xparbxbj` | q3-810m-cx4-q3am128e8k-lr4e-4-r1<br>`qoisdrag` | q3-810m-cx4-q3td128e8k-lr4e-4-r1<br>`3rwe92jl` | se-810m-cx4-se0m9-lr4e-4-r1<br>`smaodqu8` | sp-810m-cx4-sp192e4k-lr3e-4-r2<br>`atbtg1ch` | sp-810m-cx4-sp96e4k-lr3.5e-4-r1<br>`vrhhfj4w` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.83298 | 0.82034 | 0.80815 | 0.80472 | 0.83122 | 0.84429 | 0.84517 | 0.81782 | 0.80971 | 0.81243 | 0.82999 | 0.80896 | 0.81835 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 0.90985 | 0.89618 | 0.88657 | 0.88127 | 0.91164 | 0.92431 | 0.92431 | 0.89373 | 0.88553 | 0.88932 | 0.90692 | 0.88641 | 0.89295 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0040 | 1.0045 | 1.0037 | 1.0025 | 1.0026 | 1.0092 | 1.0111 | 1.0053 | 1.0047 | 1.0153 | 1.0039 | 1.0084 | 1.0051 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0081 | 2.0090 | 2.0075 | 2.0049 | 2.0052 | 2.0184 | 2.0222 | 2.0107 | 2.0093 | 2.0306 | 2.0077 | 2.0168 | 2.0102 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.69610 | 0.69636 | 0.69588 | 0.69508 | 0.69507 | 0.69958 | 0.70094 | 0.69689 | 0.69650 | 0.70375 | 0.69595 | 0.69905 | 0.69674 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.3922 | 1.3927 | 1.3918 | 1.3902 | 1.3901 | 1.3992 | 1.4019 | 1.3938 | 1.3930 | 1.4075 | 1.3919 | 1.3981 | 1.3935 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.25597 | 0.27474 | 0.25939 | 0.25085 | 0.26536 | 0.28328 | 0.24915 | 0.26280 | 0.25427 | 0.24403 | 0.26024 | 0.26365 | 0.25427 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.25597 | 0.27474 | 0.25939 | 0.25085 | 0.26536 | 0.28328 | 0.24915 | 0.26280 | 0.25427 | 0.24403 | 0.26024 | 0.26365 | 0.25427 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.3888 | -1.3902 | -1.3897 | -1.3878 | -1.3857 | -1.3958 | -1.3990 | -1.3918 | -1.3893 | -1.4036 | -1.3892 | -1.3943 | -1.3893 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.3888 | -1.3902 | -1.3897 | -1.3878 | -1.3857 | -1.3958 | -1.3990 | -1.3918 | -1.3893 | -1.4036 | -1.3892 | -1.3943 | -1.3893 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.25046 | 0.25110 | 0.25100 | 0.25049 | 0.25203 | 0.25168 | 0.25249 | 0.25212 | 0.25040 | 0.25153 | 0.25217 | 0.25198 | 0.25268 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.25046 | 0.25110 | 0.25100 | 0.25049 | 0.25203 | 0.25168 | 0.25249 | 0.25212 | 0.25040 | 0.25153 | 0.25217 | 0.25198 | 0.25268 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.64658 | 0.61655 | 0.59540 | 0.60089 | 0.63302 | 0.63775 | 0.64093 | 0.62161 | 0.60876 | 0.60990 | 0.62966 | 0.60659 | 0.62207 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.70314 | 0.67055 | 0.64707 | 0.65333 | 0.68827 | 0.69367 | 0.69696 | 0.67558 | 0.66196 | 0.66275 | 0.68395 | 0.65961 | 0.67665 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0076 | 1.0060 | 1.0067 | 1.0039 | 1.0191 | 1.0206 | 1.0272 | 1.0210 | 1.0104 | 1.0159 | 1.0084 | 1.0081 | 1.0080 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0151 | 2.0120 | 2.0133 | 2.0079 | 2.0382 | 2.0412 | 2.0543 | 2.0420 | 2.0209 | 2.0318 | 2.0169 | 2.0163 | 2.0161 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.69853 | 0.69741 | 0.69789 | 0.69600 | 0.70638 | 0.70749 | 0.71202 | 0.70774 | 0.70042 | 0.70422 | 0.69911 | 0.69886 | 0.69879 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.3971 | 1.3948 | 1.3958 | 1.3920 | 1.4128 | 1.4150 | 1.4240 | 1.4155 | 1.4008 | 1.4084 | 1.3982 | 1.3977 | 1.3976 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.24411 | 0.25715 | 0.23948 | 0.24916 | 0.23569 | 0.23906 | 0.24074 | 0.24874 | 0.24747 | 0.25715 | 0.25842 | 0.25463 | 0.27104 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.24411 | 0.25715 | 0.23948 | 0.24916 | 0.23569 | 0.23906 | 0.24074 | 0.24874 | 0.24747 | 0.25715 | 0.25842 | 0.25463 | 0.27104 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.3940 | -1.3924 | -1.3936 | -1.3898 | -1.4061 | -1.4124 | -1.4216 | -1.4134 | -1.3977 | -1.4048 | -1.3956 | -1.3942 | -1.3933 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.3940 | -1.3924 | -1.3936 | -1.3898 | -1.4061 | -1.4124 | -1.4216 | -1.4134 | -1.3977 | -1.4048 | -1.3956 | -1.3942 | -1.3933 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.24974 | 0.25082 | 0.24984 | 0.25000 | 0.24935 | 0.24882 | 0.25006 | 0.24974 | 0.24987 | 0.25083 | 0.25103 | 0.25111 | 0.25258 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.24974 | 0.25082 | 0.24984 | 0.25000 | 0.24935 | 0.24882 | 0.25006 | 0.24974 | 0.24987 | 0.25083 | 0.25103 | 0.25111 | 0.25258 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 1.2223 | 1.2418 | 1.1243 | 1.0822 | 1.1157 | 1.2615 | 1.2224 | 1.1208 | 1.1349 | 1.2203 | 1.1478 | 1.0875 | 1.1233 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 1.9799 | 2.0128 | 1.8243 | 1.7460 | 1.7990 | 2.0337 | 1.9690 | 1.8115 | 1.8350 | 1.9752 | 1.8500 | 1.7524 | 1.8315 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 0.84707 | 0.86067 | 0.77921 | 0.75014 | 0.77337 | 0.87447 | 0.84734 | 0.77685 | 0.78663 | 0.84582 | 0.79558 | 0.75375 | 0.77864 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 1.3723 | 1.3951 | 1.2644 | 1.2102 | 1.2470 | 1.4098 | 1.3648 | 1.2556 | 1.2720 | 1.3691 | 1.2823 | 1.2147 | 1.2695 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.48711 | 0.47947 | 0.53104 | 0.53582 | 0.49952 | 0.47373 | 0.50430 | 0.51385 | 0.52245 | 0.51671 | 0.50812 | 0.52531 | 0.54728 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.48711 | 0.47947 | 0.53104 | 0.53582 | 0.49952 | 0.47373 | 0.50430 | 0.51385 | 0.52245 | 0.51671 | 0.50812 | 0.52531 | 0.54728 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -1.5789 | -1.6291 | -1.4424 | -1.4137 | -1.4687 | -1.7127 | -1.6069 | -1.4096 | -1.3964 | -1.5040 | -1.4704 | -1.4141 | -1.3810 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -1.5789 | -1.6291 | -1.4424 | -1.4137 | -1.4687 | -1.7127 | -1.6069 | -1.4096 | -1.3964 | -1.5040 | -1.4704 | -1.4141 | -1.3810 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.45223 | 0.44270 | 0.49169 | 0.50441 | 0.45945 | 0.43267 | 0.46458 | 0.47286 | 0.49318 | 0.47136 | 0.48204 | 0.48536 | 0.49645 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.45223 | 0.44270 | 0.49169 | 0.50441 | 0.45945 | 0.43267 | 0.46458 | 0.47286 | 0.49318 | 0.47136 | 0.48204 | 0.48536 | 0.49645 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.37333 | 0.38858 | 0.36325 | 0.37336 | 0.38135 | 0.34727 | 0.35074 | 0.38267 | 0.40491 | 0.36942 | 0.41790 | 0.30752 | 0.37816 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.40739 | 0.42389 | 0.39631 | 0.40661 | 0.41542 | 0.37800 | 0.38201 | 0.41674 | 0.44150 | 0.40307 | 0.45687 | 0.33519 | 0.41273 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.25880 | 0.26932 | 0.25179 | 0.25881 | 0.26434 | 0.24073 | 0.24312 | 0.26527 | 0.28067 | 0.25605 | 0.28967 | 0.21316 | 0.26215 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.28238 | 0.29381 | 0.27469 | 0.28185 | 0.28795 | 0.26202 | 0.26480 | 0.28885 | 0.30604 | 0.27937 | 0.31667 | 0.23232 | 0.28608 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.63439 | 0.60771 | 0.66601 | 0.64427 | 0.59190 | 0.61561 | 0.66107 | 0.61858 | 0.62747 | 0.63735 | 0.61265 | 0.69071 | 0.61660 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.63439 | 0.60771 | 0.66601 | 0.64427 | 0.59190 | 0.61561 | 0.66107 | 0.61858 | 0.62747 | 0.63735 | 0.61265 | 0.69071 | 0.61660 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -1.5052 | -1.5659 | -1.4150 | -1.3816 | -1.5793 | -1.6086 | -1.3818 | -1.4960 | -1.4817 | -1.4044 | -1.6161 | -1.1958 | -1.4510 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -1.5052 | -1.5659 | -1.4150 | -1.3816 | -1.5793 | -1.6086 | -1.3818 | -1.4960 | -1.4817 | -1.4044 | -1.6161 | -1.1958 | -1.4510 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.60222 | 0.58734 | 0.63262 | 0.62198 | 0.58208 | 0.59313 | 0.62153 | 0.59881 | 0.59610 | 0.60496 | 0.58360 | 0.66191 | 0.59739 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.60222 | 0.58734 | 0.63262 | 0.62198 | 0.58208 | 0.59313 | 0.62153 | 0.59881 | 0.59610 | 0.60496 | 0.58360 | 0.66191 | 0.59739 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.42077 | 0.33909 | 0.28128 | 0.33896 | 0.29682 | 0.37309 | 0.34200 | 0.29128 | 0.33306 | 0.34674 | 0.34007 | 0.31170 | 0.33529 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.50657 | 0.40740 | 0.33818 | 0.40744 | 0.35644 | 0.45050 | 0.41025 | 0.34958 | 0.40140 | 0.41755 | 0.40965 | 0.37353 | 0.40397 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.29180 | 0.23513 | 0.19516 | 0.23509 | 0.20589 | 0.25874 | 0.23715 | 0.20203 | 0.23095 | 0.24047 | 0.23580 | 0.21612 | 0.23258 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.35130 | 0.28255 | 0.23464 | 0.28260 | 0.24727 | 0.31238 | 0.28450 | 0.24249 | 0.27832 | 0.28956 | 0.28410 | 0.25908 | 0.28024 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.87657 | 0.88814 | 0.92575 | 0.91707 | 0.89103 | 0.88910 | 0.89682 | 0.91996 | 0.88428 | 0.90743 | 0.88717 | 0.93346 | 0.89489 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.87657 | 0.88814 | 0.92575 | 0.91707 | 0.89103 | 0.88910 | 0.89682 | 0.91996 | 0.88428 | 0.90743 | 0.88717 | 0.93346 | 0.89489 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.36083 | -0.32720 | -0.25766 | -0.26361 | -0.30830 | -0.33619 | -0.30757 | -0.27531 | -0.32152 | -0.29163 | -0.32478 | -0.22304 | -0.28994 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.36083 | -0.32720 | -0.25766 | -0.26361 | -0.30830 | -0.33619 | -0.30757 | -0.27531 | -0.32152 | -0.29163 | -0.32478 | -0.22304 | -0.28994 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.78418 | 0.79993 | 0.83917 | 0.83095 | 0.81797 | 0.79062 | 0.80873 | 0.82627 | 0.81180 | 0.81719 | 0.80231 | 0.85187 | 0.81932 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.78418 | 0.79993 | 0.83917 | 0.83095 | 0.81797 | 0.79062 | 0.80873 | 0.82627 | 0.81180 | 0.81719 | 0.80231 | 0.85187 | 0.81932 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.28517 | 0.27487 | 0.25737 | 0.28016 | 0.30085 | 0.27095 | 0.24847 | 0.29196 | 0.29099 | 0.29538 | 0.25469 | 0.28186 | 0.30026 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.29481 | 0.28405 | 0.26600 | 0.28959 | 0.31111 | 0.27996 | 0.25681 | 0.30176 | 0.30073 | 0.30536 | 0.26326 | 0.29138 | 0.31043 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.19768 | 0.19055 | 0.17841 | 0.19420 | 0.20856 | 0.18780 | 0.17226 | 0.20239 | 0.20173 | 0.20477 | 0.17656 | 0.19537 | 0.20816 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.20436 | 0.19691 | 0.18439 | 0.20075 | 0.21567 | 0.19406 | 0.17802 | 0.20920 | 0.20848 | 0.21167 | 0.18250 | 0.20198 | 0.21518 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.89803 | 0.89982 | 0.91145 | 0.90519 | 0.88462 | 0.90072 | 0.90429 | 0.92487 | 0.88372 | 0.90250 | 0.91234 | 0.86762 | 0.89177 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.89803 | 0.89982 | 0.91145 | 0.90519 | 0.88462 | 0.90072 | 0.90429 | 0.92487 | 0.88372 | 0.90250 | 0.91234 | 0.86762 | 0.89177 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.26334 | -0.29133 | -0.25908 | -0.24310 | -0.28918 | -0.27941 | -0.26395 | -0.22870 | -0.28150 | -0.24855 | -0.25180 | -0.34178 | -0.28711 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.26334 | -0.29133 | -0.25908 | -0.24310 | -0.28918 | -0.27941 | -0.26395 | -0.22870 | -0.28150 | -0.24855 | -0.25180 | -0.34178 | -0.28711 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.88279 | 0.87313 | 0.89244 | 0.88384 | 0.87546 | 0.88451 | 0.88583 | 0.89937 | 0.87329 | 0.87999 | 0.89238 | 0.86405 | 0.87756 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.88279 | 0.87313 | 0.89244 | 0.88384 | 0.87546 | 0.88451 | 0.88583 | 0.89937 | 0.87329 | 0.87999 | 0.89238 | 0.86405 | 0.87756 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 0.83796 | 0.78991 | 0.78960 | 0.83030 | 0.84695 | 0.88450 | 0.83583 | 0.87514 | 0.83306 | 0.81232 | 0.80837 | 0.73955 | 0.79949 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.4003 | 1.3143 | 1.3227 | 1.3653 | 1.4089 | 1.4468 | 1.3709 | 1.4455 | 1.3756 | 1.3478 | 1.3323 | 1.2171 | 1.3303 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.60898 | 0.57939 | 0.57520 | 0.60834 | 0.61979 | 0.64587 | 0.60873 | 0.63596 | 0.61259 | 0.58877 | 0.58538 | 0.54221 | 0.58311 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.0428 | 0.99202 | 0.98763 | 1.0290 | 1.0603 | 1.0864 | 1.0256 | 1.0771 | 1.0424 | 0.99962 | 0.98819 | 0.91897 | 0.99532 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.72285 | 0.74719 | 0.73408 | 0.72472 | 0.74719 | 0.71161 | 0.72097 | 0.73408 | 0.74906 | 0.74532 | 0.73970 | 0.73221 | 0.76592 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.72285 | 0.74719 | 0.73408 | 0.72472 | 0.74719 | 0.71161 | 0.72097 | 0.73408 | 0.74906 | 0.74532 | 0.73970 | 0.73221 | 0.76592 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.67406 | -0.64060 | -0.65041 | -0.66932 | -0.65158 | -0.73840 | -0.68162 | -0.67095 | -0.66794 | -0.67383 | -0.64786 | -0.68173 | -0.64537 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.67406 | -0.64060 | -0.65041 | -0.66932 | -0.65158 | -0.73840 | -0.68162 | -0.67095 | -0.66794 | -0.67383 | -0.64786 | -0.68173 | -0.64537 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.64929 | 0.65634 | 0.67283 | 0.65726 | 0.66700 | 0.63681 | 0.64776 | 0.65799 | 0.65416 | 0.67115 | 0.66408 | 0.66407 | 0.67018 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.64929 | 0.65634 | 0.67283 | 0.65726 | 0.66700 | 0.63681 | 0.64776 | 0.65799 | 0.65416 | 0.67115 | 0.66408 | 0.66407 | 0.67018 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.4473 | 1.4235 | 1.3311 | 1.3112 | 1.2713 | 1.4321 | 1.5090 | 1.3271 | 1.3738 | 1.3246 | 1.6316 | 1.4122 | 1.3507 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.0141 | 1.9932 | 1.8476 | 1.8156 | 1.7978 | 1.9909 | 2.0944 | 1.8592 | 1.9063 | 1.8376 | 2.2313 | 1.9761 | 1.8761 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.0033 | 0.98671 | 0.92269 | 0.90889 | 0.88129 | 0.99266 | 1.0461 | 0.91998 | 0.95218 | 0.91812 | 1.1310 | 0.97877 | 0.93617 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.3961 | 1.3816 | 1.2806 | 1.2585 | 1.2461 | 1.3801 | 1.4518 | 1.2888 | 1.3213 | 1.2738 | 1.5466 | 1.3697 | 1.3004 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.28958 | 0.29861 | 0.34372 | 0.31091 | 0.38392 | 0.29532 | 0.31091 | 0.32978 | 0.28466 | 0.31665 | 0.29614 | 0.30599 | 0.30599 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.28958 | 0.29861 | 0.34372 | 0.31091 | 0.38392 | 0.29532 | 0.31091 | 0.32978 | 0.28466 | 0.31665 | 0.29614 | 0.30599 | 0.30599 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -3.3015 | -3.1672 | -2.8063 | -2.9495 | -2.6851 | -3.4407 | -3.0940 | -3.0379 | -3.1335 | -3.1567 | -3.2528 | -3.0245 | -3.0628 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -3.3015 | -3.1672 | -2.8063 | -2.9495 | -2.6851 | -3.4407 | -3.0940 | -3.0379 | -3.1335 | -3.1567 | -3.2528 | -3.0245 | -3.0628 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.29744 | 0.30801 | 0.34778 | 0.32375 | 0.37464 | 0.29658 | 0.32115 | 0.34606 | 0.30596 | 0.31721 | 0.31059 | 0.31307 | 0.31636 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.29744 | 0.30801 | 0.34778 | 0.32375 | 0.37464 | 0.29658 | 0.32115 | 0.34606 | 0.30596 | 0.31721 | 0.31059 | 0.31307 | 0.31636 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.41644 | 0.39284 | 0.39282 | 0.38635 | 0.41405 | 0.40622 | 0.40232 | 0.39411 | 0.39020 | 0.38901 | 0.39922 | 0.38259 | 0.37591 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.42212 | 0.39814 | 0.39826 | 0.39151 | 0.41947 | 0.41153 | 0.40779 | 0.39935 | 0.39527 | 0.39435 | 0.40426 | 0.38747 | 0.38069 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.61402 | 0.61480 | 0.59125 | 0.58880 | 0.61548 | 0.61196 | 0.60199 | 0.60099 | 0.59127 | 0.60517 | 0.61125 | 0.58607 | 0.59821 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.61927 | 0.62038 | 0.59624 | 0.59391 | 0.62077 | 0.61729 | 0.60707 | 0.60617 | 0.59637 | 0.61039 | 0.61644 | 0.59127 | 0.60336 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.0199 | 1.8273 | 1.9955 | 1.9085 | 1.9620 | 2.0463 | 1.8584 | 2.3045 | 2.1156 | 2.1228 | 1.8812 | 1.9143 | 2.0095 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 4.0398 | 3.6547 | 3.9911 | 3.8170 | 3.9241 | 4.0927 | 3.7169 | 4.6091 | 4.2312 | 4.2455 | 3.7623 | 3.8286 | 4.0191 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.4002 | 1.2664 | 1.3832 | 1.3228 | 1.3601 | 1.4182 | 1.2884 | 1.5975 | 1.4660 | 1.4706 | 1.3042 | 1.3269 | 1.3930 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 2.8003 | 2.5329 | 2.7665 | 2.6456 | 2.7202 | 2.8364 | 2.5769 | 3.1949 | 2.9320 | 2.9413 | 2.6084 | 2.6538 | 2.7861 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.10000 | 0.10000 | 0.10000 | 0.13000 | 0.08000 | 0.10000 | 0.10000 | 0.07000 | 0.11000 | 0.07000 | 0.10000 | 0.14000 | 0.08000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.10000 | 0.10000 | 0.10000 | 0.13000 | 0.08000 | 0.10000 | 0.10000 | 0.07000 | 0.11000 | 0.07000 | 0.10000 | 0.14000 | 0.08000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -2.7977 | -2.5184 | -2.7620 | -2.6387 | -2.7119 | -2.8324 | -2.5727 | -3.1930 | -2.9259 | -2.9342 | -2.6007 | -2.6460 | -2.7808 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -2.7977 | -2.5184 | -2.7620 | -2.6387 | -2.7119 | -2.8324 | -2.5727 | -3.1930 | -2.9259 | -2.9342 | -2.6007 | -2.6460 | -2.7808 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09755 | 0.10115 | 0.09884 | 0.10131 | 0.09185 | 0.09512 | 0.10094 | 0.09318 | 0.09847 | 0.09381 | 0.09899 | 0.10135 | 0.09398 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09755 | 0.10115 | 0.09884 | 0.10131 | 0.09185 | 0.09512 | 0.10094 | 0.09318 | 0.09847 | 0.09381 | 0.09899 | 0.10135 | 0.09398 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.77792 | 0.77054 | 0.75542 | 0.76096 | 0.78330 | 0.77675 | 0.77796 | 0.77251 | 0.77441 | 0.76879 | 0.77777 | 0.76643 | 0.77013 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.78652 | 0.77903 | 0.76390 | 0.76941 | 0.79187 | 0.78541 | 0.78666 | 0.78100 | 0.78301 | 0.77730 | 0.78642 | 0.77494 | 0.77869 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.63089 | 0.61718 | 0.59666 | 0.60015 | 0.63041 | 0.63756 | 0.62476 | 0.62369 | 0.61294 | 0.61020 | 0.62625 | 0.60354 | 0.61044 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.63287 | 0.61918 | 0.59863 | 0.60206 | 0.63250 | 0.63963 | 0.62676 | 0.62554 | 0.61486 | 0.61219 | 0.62814 | 0.60549 | 0.61230 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.68090 | 0.66847 | 0.65269 | 0.64721 | 0.68929 | 0.68824 | 0.68372 | 0.67752 | 0.66526 | 0.66847 | 0.67401 | 0.64891 | 0.66203 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.71536 | 0.70200 | 0.68462 | 0.67935 | 0.72311 | 0.72271 | 0.71803 | 0.71091 | 0.69831 | 0.70139 | 0.70755 | 0.68151 | 0.69473 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0071 | 1.0039 | 1.0072 | 1.0040 | 1.0080 | 1.0144 | 1.0087 | 1.0107 | 1.0107 | 1.0124 | 1.0066 | 1.0096 | 1.0061 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0141 | 2.0079 | 2.0145 | 2.0081 | 2.0160 | 2.0288 | 2.0174 | 2.0213 | 2.0214 | 2.0248 | 2.0133 | 2.0191 | 2.0122 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.69816 | 0.69603 | 0.69828 | 0.69607 | 0.69876 | 0.70317 | 0.69925 | 0.70061 | 0.70069 | 0.70180 | 0.69780 | 0.69984 | 0.69752 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.3963 | 1.3921 | 1.3966 | 1.3921 | 1.3975 | 1.4063 | 1.3985 | 1.4012 | 1.4014 | 1.4036 | 1.3956 | 1.3997 | 1.3950 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.25228 | 0.26482 | 0.25994 | 0.25951 | 0.25845 | 0.26036 | 0.25399 | 0.25887 | 0.24506 | 0.25313 | 0.26270 | 0.26227 | 0.25802 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.25228 | 0.26482 | 0.25994 | 0.25951 | 0.25845 | 0.26036 | 0.25399 | 0.25887 | 0.24506 | 0.25313 | 0.26270 | 0.26227 | 0.25802 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3873 | -1.3851 | -1.3866 | -1.3858 | -1.3865 | -1.3885 | -1.3874 | -1.3869 | -1.3891 | -1.3888 | -1.3858 | -1.3867 | -1.3861 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.3922 | -1.3872 | -1.3933 | -1.3883 | -1.3924 | -1.4016 | -1.3959 | -1.3973 | -1.3974 | -1.3997 | -1.3912 | -1.3962 | -1.3899 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.25026 | 0.25072 | 0.25070 | 0.25052 | 0.25067 | 0.25090 | 0.25068 | 0.25109 | 0.24999 | 0.25044 | 0.25090 | 0.25107 | 0.25053 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.25052 | 0.25144 | 0.25135 | 0.25105 | 0.25130 | 0.25177 | 0.25130 | 0.25208 | 0.24999 | 0.25088 | 0.25179 | 0.25207 | 0.25103 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 0.95072 | 0.93071 | 0.90625 | 0.89647 | 0.93633 | 0.94811 | 0.94025 | 0.92753 | 0.92199 | 0.92316 | 0.93117 | 0.89423 | 0.91376 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.0597 | 1.0385 | 1.0121 | 1.0002 | 1.0439 | 1.0566 | 1.0485 | 1.0323 | 1.0279 | 1.0292 | 1.0373 | 0.99667 | 1.0182 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 0.99940 | 1.0013 | 1.0052 | 1.0034 | 0.99983 | 1.0018 | 0.99965 | 1.0002 | 0.99751 | 1.0026 | 1.0011 | 1.0111 | 1.0045 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 1.9988 | 2.0026 | 2.0104 | 2.0068 | 1.9997 | 2.0036 | 1.9993 | 2.0004 | 1.9950 | 2.0052 | 2.0022 | 2.0223 | 2.0090 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.69288 | 0.69419 | 0.69685 | 0.69569 | 0.69313 | 0.69452 | 0.69297 | 0.69336 | 0.69155 | 0.69501 | 0.69399 | 0.70093 | 0.69636 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.3858 | 1.3884 | 1.3937 | 1.3914 | 1.3863 | 1.3890 | 1.3859 | 1.3867 | 1.3831 | 1.3900 | 1.3880 | 1.4019 | 1.3927 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.27637 | 0.26959 | 0.26990 | 0.25324 | 0.28717 | 0.28069 | 0.29611 | 0.29426 | 0.29951 | 0.28131 | 0.29365 | 0.25848 | 0.25910 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.27637 | 0.26959 | 0.26990 | 0.25324 | 0.28717 | 0.28069 | 0.29611 | 0.29426 | 0.29951 | 0.28131 | 0.29365 | 0.25848 | 0.25910 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3828 | -1.3831 | -1.3853 | -1.3855 | -1.3804 | -1.3824 | -1.3798 | -1.3813 | -1.3804 | -1.3822 | -1.3819 | -1.3869 | -1.3841 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.3826 | -1.3844 | -1.3904 | -1.3881 | -1.3808 | -1.3842 | -1.3825 | -1.3835 | -1.3799 | -1.3867 | -1.3845 | -1.3980 | -1.3880 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25129 | 0.25133 | 0.25100 | 0.25059 | 0.25233 | 0.25176 | 0.25284 | 0.25219 | 0.25220 | 0.25210 | 0.25200 | 0.25115 | 0.25132 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25255 | 0.25256 | 0.25195 | 0.25113 | 0.25476 | 0.25358 | 0.25574 | 0.25442 | 0.25448 | 0.25418 | 0.25401 | 0.25212 | 0.25263 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.80684 | 0.79005 | 0.76748 | 0.76818 | 0.80048 | 0.81058 | 0.80535 | 0.79792 | 0.78378 | 0.79054 | 0.79818 | 0.76778 | 0.77978 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 0.86123 | 0.84345 | 0.81888 | 0.82037 | 0.85415 | 0.86524 | 0.85945 | 0.85193 | 0.83663 | 0.84398 | 0.85198 | 0.81929 | 0.83236 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0064 | 1.0080 | 1.0162 | 1.0017 | 1.0010 | 1.0013 | 1.0085 | 1.0012 | 1.0015 | 1.0101 | 1.0011 | 1.0227 | 1.0121 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0128 | 2.0160 | 2.0324 | 2.0033 | 2.0020 | 2.0025 | 2.0170 | 2.0024 | 2.0030 | 2.0203 | 2.0021 | 2.0453 | 2.0242 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.69772 | 0.69886 | 0.70447 | 0.69449 | 0.69393 | 0.69409 | 0.69912 | 0.69401 | 0.69428 | 0.70025 | 0.69397 | 0.70898 | 0.70161 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.3954 | 1.3977 | 1.4089 | 1.3890 | 1.3879 | 1.3882 | 1.3982 | 1.3880 | 1.3886 | 1.4005 | 1.3879 | 1.4180 | 1.4032 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.25154 | 0.24862 | 0.23822 | 0.26194 | 0.28144 | 0.27202 | 0.25837 | 0.28372 | 0.27332 | 0.26552 | 0.27754 | 0.24277 | 0.24244 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.25154 | 0.24862 | 0.23822 | 0.26194 | 0.28144 | 0.27202 | 0.25837 | 0.28372 | 0.27332 | 0.26552 | 0.27754 | 0.24277 | 0.24244 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3877 | -1.3887 | -1.3936 | -1.3850 | -1.3808 | -1.3817 | -1.3862 | -1.3806 | -1.3838 | -1.3881 | -1.3823 | -1.3968 | -1.3902 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.3924 | -1.3938 | -1.4058 | -1.3860 | -1.3822 | -1.3838 | -1.3954 | -1.3851 | -1.3850 | -1.3976 | -1.3844 | -1.4145 | -1.3991 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.25011 | 0.24977 | 0.24887 | 0.25058 | 0.25231 | 0.25201 | 0.25115 | 0.25277 | 0.25113 | 0.25052 | 0.25176 | 0.24835 | 0.24970 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.25025 | 0.24955 | 0.24781 | 0.25115 | 0.25476 | 0.25402 | 0.25218 | 0.25564 | 0.25227 | 0.25100 | 0.25343 | 0.24687 | 0.24945 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.2250 | 1.2020 | 1.1857 | 1.1666 | 1.2095 | 1.2427 | 1.2133 | 1.1919 | 1.1919 | 1.1986 | 1.2097 | 1.1716 | 1.1965 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.5312 | 1.5001 | 1.4827 | 1.4576 | 1.5091 | 1.5561 | 1.5166 | 1.4843 | 1.4906 | 1.4957 | 1.5067 | 1.4645 | 1.4945 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0042 | 1.0078 | 1.0124 | 1.0116 | 1.0061 | 1.0057 | 1.0118 | 1.0061 | 1.0048 | 1.0058 | 1.0044 | 1.0222 | 1.0106 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0085 | 2.0156 | 2.0247 | 2.0231 | 2.0122 | 2.0114 | 2.0235 | 2.0122 | 2.0097 | 2.0117 | 2.0088 | 2.0445 | 2.0212 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.69620 | 0.69861 | 0.70177 | 0.70127 | 0.69747 | 0.69714 | 0.70131 | 0.69739 | 0.69660 | 0.69725 | 0.69628 | 0.70863 | 0.70059 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.3924 | 1.3972 | 1.4035 | 1.4025 | 1.3949 | 1.3943 | 1.4026 | 1.3948 | 1.3932 | 1.3945 | 1.3926 | 1.4173 | 1.4012 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.26044 | 0.25182 | 0.26243 | 0.23592 | 0.27071 | 0.27800 | 0.26077 | 0.28628 | 0.27005 | 0.26905 | 0.26872 | 0.26408 | 0.26276 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.26044 | 0.25182 | 0.26243 | 0.23592 | 0.27071 | 0.27800 | 0.26077 | 0.28628 | 0.27005 | 0.26905 | 0.26872 | 0.26408 | 0.26276 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3856 | -1.3862 | -1.3891 | -1.3900 | -1.3844 | -1.3824 | -1.3857 | -1.3833 | -1.3836 | -1.3838 | -1.3827 | -1.3934 | -1.3880 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.3894 | -1.3918 | -1.4003 | -1.3986 | -1.3895 | -1.3894 | -1.3995 | -1.3906 | -1.3899 | -1.3909 | -1.3888 | -1.4138 | -1.3964 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.25077 | 0.25077 | 0.25048 | 0.24970 | 0.25137 | 0.25232 | 0.25194 | 0.25206 | 0.25182 | 0.25181 | 0.25211 | 0.25002 | 0.25042 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.25153 | 0.25156 | 0.25103 | 0.24940 | 0.25271 | 0.25453 | 0.25366 | 0.25412 | 0.25364 | 0.25351 | 0.25414 | 0.25025 | 0.25089 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.41166 | 0.41160 | 0.40292 | 0.39690 | 0.41091 | 0.40457 | 0.39832 | 0.40546 | 0.39304 | 0.39947 | 0.39841 | 0.39763 | 0.39509 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.41400 | 0.41394 | 0.40529 | 0.39911 | 0.41335 | 0.40682 | 0.40066 | 0.40774 | 0.39535 | 0.40183 | 0.40078 | 0.39991 | 0.39732 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.32286 | 0.31631 | 0.30339 | 0.29757 | 0.31509 | 0.31868 | 0.29681 | 0.30366 | 0.30556 | 0.29821 | 0.30022 | 0.31752 | 0.30920 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.32409 | 0.31745 | 0.30456 | 0.29877 | 0.31636 | 0.31996 | 0.29790 | 0.30481 | 0.30672 | 0.29933 | 0.30138 | 0.31877 | 0.31039 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.54111 | 0.51650 | 0.50710 | 0.50126 | 0.51001 | 0.53504 | 0.50553 | 0.51158 | 0.53066 | 0.50674 | 0.53941 | 0.52082 | 0.50121 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.54500 | 0.52008 | 0.51052 | 0.50481 | 0.51360 | 0.53893 | 0.50914 | 0.51505 | 0.53451 | 0.51031 | 0.54323 | 0.52445 | 0.50477 |
| eval/lm/c4_en-validation/CE loss | lower | 2.8654 | 2.8352 | 2.7935 | 2.7983 | 2.8720 | 2.8666 | 2.8486 | 2.8504 | 2.8282 | 2.8182 | 2.8543 | 2.7972 | 2.8191 |
| eval/lm/c4_en-validation/PPL | lower | 17.56 | 17.03 | 16.34 | 16.42 | 17.67 | 17.58 | 17.26 | 17.29 | 16.91 | 16.75 | 17.36 | 16.40 | 16.76 |
| eval/lm/dolma_books-validation/CE loss | lower | 2.7330 | 2.7119 | 2.6490 | 2.6596 | 2.7412 | 2.7433 | 2.7219 | 2.7144 | 2.6992 | 2.6846 | 2.7302 | 2.6645 | 2.6910 |
| eval/lm/dolma_books-validation/PPL | lower | 15.38 | 15.06 | 14.14 | 14.29 | 15.51 | 15.54 | 15.21 | 15.10 | 14.87 | 14.65 | 15.34 | 14.36 | 14.75 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.0030 | 2.9747 | 2.9292 | 2.9333 | 3.0094 | 3.0060 | 2.9887 | 2.9884 | 2.9660 | 2.9580 | 2.9947 | 2.9324 | 2.9584 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 20.15 | 19.58 | 18.71 | 18.79 | 20.27 | 20.21 | 19.86 | 19.85 | 19.41 | 19.26 | 19.98 | 18.77 | 19.27 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.0729 | 2.0514 | 2.0210 | 2.0256 | 2.0731 | 2.0755 | 2.0601 | 2.0588 | 2.0456 | 2.0345 | 2.0653 | 2.0279 | 2.0429 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 7.9475 | 7.7791 | 7.5456 | 7.5803 | 7.9497 | 7.9687 | 7.8471 | 7.8369 | 7.7337 | 7.6482 | 7.8874 | 7.5978 | 7.7128 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.1845 | 3.1572 | 3.1184 | 3.1245 | 3.1838 | 3.1867 | 3.1696 | 3.1664 | 3.1513 | 3.1399 | 3.1753 | 3.1251 | 3.1469 |
| eval/lm/dolma_reddit-validation/PPL | lower | 24.15 | 23.51 | 22.61 | 22.75 | 24.14 | 24.21 | 23.80 | 23.72 | 23.37 | 23.10 | 23.94 | 22.76 | 23.26 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.2763 | 1.2573 | 1.2253 | 1.2318 | 1.2809 | 1.2793 | 1.2632 | 1.2638 | 1.2501 | 1.2460 | 1.2701 | 1.2286 | 1.2440 |
| eval/lm/dolma_stack-validation/PPL | lower | 3.5834 | 3.5161 | 3.4051 | 3.4273 | 3.5998 | 3.5940 | 3.5368 | 3.5389 | 3.4909 | 3.4763 | 3.5612 | 3.4163 | 3.4696 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.4998 | 2.4765 | 2.4290 | 2.4345 | 2.5133 | 2.5023 | 2.4876 | 2.4857 | 2.4600 | 2.4568 | 2.4913 | 2.4267 | 2.4561 |
| eval/lm/dolma_wiki-validation/PPL | lower | 12.18 | 11.90 | 11.35 | 11.41 | 12.35 | 12.21 | 12.03 | 12.01 | 11.71 | 11.67 | 12.08 | 11.32 | 11.66 |
| eval/lm/ice-validation/CE loss | lower | 2.9390 | 2.8953 | 2.8597 | 2.8776 | 2.9128 | 2.9312 | 2.9101 | 2.9130 | 2.8951 | 2.8836 | 2.9293 | 2.8722 | 2.8854 |
| eval/lm/ice-validation/PPL | lower | 18.90 | 18.09 | 17.46 | 17.77 | 18.41 | 18.75 | 18.36 | 18.41 | 18.09 | 17.88 | 18.71 | 17.68 | 17.91 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 2.9824 | 2.9629 | 2.9225 | 2.9314 | 2.9963 | 3.0006 | 2.9816 | 2.9774 | 2.9634 | 2.9562 | 2.9771 | 2.9328 | 2.9500 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 19.74 | 19.35 | 18.59 | 18.75 | 20.01 | 20.10 | 19.72 | 19.64 | 19.36 | 19.22 | 19.63 | 18.78 | 19.11 |
| eval/lm/pile-validation/CE loss | lower | 2.1453 | 2.1227 | 2.0862 | 2.0878 | 2.1551 | 2.1498 | 2.1298 | 2.1378 | 2.1159 | 2.1054 | 2.1395 | 2.0900 | 2.1084 |
| eval/lm/pile-validation/PPL | lower | 8.5446 | 8.3536 | 8.0539 | 8.0670 | 8.6286 | 8.5831 | 8.4134 | 8.4805 | 8.2966 | 8.2100 | 8.4951 | 8.0852 | 8.2347 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.4224 | 2.4012 | 2.3451 | 2.3489 | 2.4201 | 2.4318 | 2.4021 | 2.3967 | 2.3792 | 2.3762 | 2.4166 | 2.3532 | 2.3749 |
| eval/lm/wikitext_103-validation/PPL | lower | 11.27 | 11.04 | 10.43 | 10.47 | 11.25 | 11.38 | 11.05 | 10.99 | 10.80 | 10.76 | 11.21 | 10.52 | 10.75 |
| throughput/in-loop eval batches | see metric | 828.0 | 828.0 | 828.0 | 828.0 | 1645.0 | 1645.0 | 1645.0 | 1645.0 | 828.0 | 828.0 | 828.0 | 1729.0 | 828.0 |
| throughput/in-loop eval time (s) | see metric | 76.97 | 78.72 | 91.76 | 84.27 | 320.4 | 301.7 | 312.1 | 311.4 | 88.72 | 114.3 | 93.29 | 110.8 | 75.59 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| eg-810m-cx4-eg24e2k-lr4e-4-r1<br>`q50qk891` | finished | original | 55158243328.0 | 105206 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/q50qk891) |
| eg-810m-cx4-eg96e8k-lr4e-4-r1<br>`7cbm4c9b` | finished | original | 55298752512.0 | 105474 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7cbm4c9b) |
| int-810m-cx4-intd256e8k-lr4e-4-r1<br>`xzja2ww7` | finished | original | 54507077632.0 | 103964 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xzja2ww7) |
| int-810m-cx4-intw256e8k-lr4e-4-r1<br>`58ftjxmw` | finished | original | 55609655296.0 | 106067 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/58ftjxmw) |
| 810m-cx4-b512k-lr1.6e-3-r1<br>`ag7nvx2l` | finished | gpu8-ep1mb4 | 55204904960.0 | 105295 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ag7nvx2l) |
| 810m-cx4-b512k-lr2e-4-r1<br>`nr84d31z` | finished | gpu8-ep1mb4 | 55204904960.0 | 105295 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/nr84d31z) |
| 810m-cx4-b512k-lr4e-4-r1<br>`5rqlw5fd` | finished | gpu8-ep1mb4 | 55204904960.0 | 105295 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5rqlw5fd) |
| 810m-cx4-b512k-lr8e-4-r1<br>`xparbxbj` | finished | gpu8-ep1mb4 | 55204904960.0 | 105295 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xparbxbj) |
| q3-810m-cx4-q3am128e8k-lr4e-4-r1<br>`qoisdrag` | finished | original | 55373725696.0 | 105617 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qoisdrag) |
| q3-810m-cx4-q3td128e8k-lr4e-4-r1<br>`3rwe92jl` | finished | original | 55232692224.0 | 105348 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3rwe92jl) |
| se-810m-cx4-se0m9-lr4e-4-r1<br>`smaodqu8` | finished | original | 55204904960.0 | 105295 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/smaodqu8) |
| sp-810m-cx4-sp192e4k-lr3e-4-r2<br>`atbtg1ch` | finished | original | 55485399040.0 | 105830 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/atbtg1ch) |
| sp-810m-cx4-sp96e4k-lr3.5e-4-r1<br>`vrhhfj4w` | finished | original | 55298752512.0 | 105474 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vrhhfj4w) |

## 810m Cx8

| metric | direction | eg-810m-cx8-eg24e2k-lr4e-4-r1<br>`3h6suqpb` | eg-810m-cx8-eg96e8k-lr4e-4-r1<br>`n9cgtrxm` | int-810m-cx8-intd256e8k-lr4e-4-r1<br>`142yfmzn` | int-810m-cx8-intw256e8k-lr4e-4-r1<br>`kyti8h1y` | 810m-cx8-b768k-lr2e-4-r1<br>`a0k0519k` | 810m-cx8-b768k-lr4e-4-r1<br>`dkpaicdc` | 810m-cx8-b768k-lr8e-4-r1<br>`rhtrhhet` | q3-810m-cx8-q3am128e8k-lr4e-4-r1<br>`98kpaudi` | q3-810m-cx8-q3td128e8k-lr4e-4-r1<br>`e24iehj0` | se-810m-cx8-se0m9-lr4e-4-r1<br>`b17jsgm7` | sp-810m-cx8-sp192e4k-lr3e-4-r2<br>`c4bcu3ho` | sp-810m-cx8-sp96e4k-lr3.5e-4-r1<br>`ypl4ayew` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.82119 | 0.81393 | 0.77042 | 0.76906 | 0.82293 | 0.80680 | 0.80304 | 0.79041 | 0.79066 | 0.81166 | 0.78377 | 0.79279 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 0.89924 | 0.89334 | 0.84288 | 0.83738 | 0.90200 | 0.88397 | 0.87886 | 0.86549 | 0.86566 | 0.88679 | 0.85789 | 0.86632 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0183 | 1.0061 | 1.0103 | 0.99837 | 1.0202 | 1.0178 | 1.0119 | 1.0017 | 1.0036 | 1.0230 | 1.0162 | 1.0095 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0367 | 2.0123 | 2.0205 | 1.9967 | 2.0404 | 2.0357 | 2.0239 | 2.0035 | 2.0073 | 2.0460 | 2.0325 | 2.0189 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.70595 | 0.69753 | 0.70039 | 0.69218 | 0.70728 | 0.70557 | 0.70154 | 0.69444 | 0.69577 | 0.70916 | 0.70443 | 0.69979 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4119 | 1.3951 | 1.4008 | 1.3844 | 1.4146 | 1.4111 | 1.4031 | 1.3889 | 1.3915 | 1.4183 | 1.4089 | 1.3996 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.24659 | 0.25853 | 0.22099 | 0.25683 | 0.23294 | 0.23549 | 0.23549 | 0.26024 | 0.23976 | 0.23805 | 0.23379 | 0.27474 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.24659 | 0.25853 | 0.22099 | 0.25683 | 0.23294 | 0.23549 | 0.23549 | 0.26024 | 0.23976 | 0.23805 | 0.23379 | 0.27474 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.4092 | -1.3926 | -1.3984 | -1.3827 | -1.4111 | -1.4077 | -1.4006 | -1.3871 | -1.3896 | -1.4154 | -1.4068 | -1.3968 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.4092 | -1.3926 | -1.3984 | -1.3827 | -1.4111 | -1.4077 | -1.4006 | -1.3871 | -1.3896 | -1.4154 | -1.4068 | -1.3968 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.25031 | 0.25224 | 0.24948 | 0.25221 | 0.24961 | 0.25044 | 0.24962 | 0.25120 | 0.25104 | 0.25069 | 0.24973 | 0.25141 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.25031 | 0.25224 | 0.24948 | 0.25221 | 0.24961 | 0.25044 | 0.24962 | 0.25120 | 0.25104 | 0.25069 | 0.24973 | 0.25141 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.60091 | 0.60639 | 0.57490 | 0.58355 | 0.62513 | 0.60277 | 0.59529 | 0.59925 | 0.59376 | 0.60877 | 0.60260 | 0.60422 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.65289 | 0.65968 | 0.62503 | 0.63381 | 0.67994 | 0.65556 | 0.64793 | 0.65175 | 0.64553 | 0.66088 | 0.65479 | 0.65709 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0274 | 1.0056 | 1.0177 | 1.0068 | 1.0119 | 1.0173 | 1.0060 | 1.0013 | 1.0057 | 1.0150 | 1.0104 | 1.0074 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0548 | 2.0113 | 2.0353 | 2.0135 | 2.0239 | 2.0347 | 2.0121 | 2.0026 | 2.0115 | 2.0300 | 2.0207 | 2.0147 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.71212 | 0.69710 | 0.70543 | 0.69793 | 0.70151 | 0.70516 | 0.69741 | 0.69417 | 0.69725 | 0.70358 | 0.70036 | 0.69835 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4242 | 1.3942 | 1.4109 | 1.3959 | 1.4030 | 1.4103 | 1.3948 | 1.3883 | 1.3945 | 1.4072 | 1.4007 | 1.3967 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.25631 | 0.27231 | 0.23569 | 0.24285 | 0.24958 | 0.25505 | 0.26347 | 0.26178 | 0.26136 | 0.26894 | 0.25589 | 0.26936 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.25631 | 0.27231 | 0.23569 | 0.24285 | 0.24958 | 0.25505 | 0.26347 | 0.26178 | 0.26136 | 0.26894 | 0.25589 | 0.26936 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.4216 | -1.3919 | -1.4089 | -1.3941 | -1.3995 | -1.4071 | -1.3914 | -1.3867 | -1.3926 | -1.4047 | -1.3991 | -1.3940 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.4216 | -1.3919 | -1.4089 | -1.3941 | -1.3995 | -1.4071 | -1.3914 | -1.3867 | -1.3926 | -1.4047 | -1.3991 | -1.3940 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.25074 | 0.25234 | 0.24933 | 0.25034 | 0.25133 | 0.25177 | 0.25167 | 0.25162 | 0.25052 | 0.25334 | 0.25079 | 0.25323 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.25074 | 0.25234 | 0.24933 | 0.25034 | 0.25133 | 0.25177 | 0.25167 | 0.25162 | 0.25052 | 0.25334 | 0.25079 | 0.25323 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 0.98347 | 1.0032 | 0.93407 | 0.92405 | 0.99608 | 1.0226 | 0.96873 | 0.95879 | 0.90414 | 1.0192 | 0.96600 | 0.88400 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 1.5912 | 1.6209 | 1.5170 | 1.5039 | 1.6184 | 1.6613 | 1.5817 | 1.5594 | 1.4651 | 1.6639 | 1.5586 | 1.4302 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 0.68159 | 0.69542 | 0.64740 | 0.64044 | 0.69040 | 0.70881 | 0.67144 | 0.66457 | 0.62659 | 0.70645 | 0.66952 | 0.61274 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 1.1030 | 1.1236 | 1.0514 | 1.0424 | 1.1218 | 1.1516 | 1.0964 | 1.0809 | 1.0154 | 1.1534 | 1.0803 | 0.99149 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.58930 | 0.55110 | 0.57880 | 0.59981 | 0.57975 | 0.55778 | 0.55778 | 0.57593 | 0.59790 | 0.56734 | 0.61700 | 0.61318 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.58930 | 0.55110 | 0.57880 | 0.59981 | 0.57975 | 0.55778 | 0.55778 | 0.57593 | 0.59790 | 0.56734 | 0.61700 | 0.61318 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -1.2401 | -1.2973 | -1.2128 | -1.2207 | -1.2946 | -1.2711 | -1.2262 | -1.2443 | -1.1518 | -1.2850 | -1.1873 | -1.1166 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -1.2401 | -1.2973 | -1.2128 | -1.2207 | -1.2946 | -1.2711 | -1.2262 | -1.2443 | -1.1518 | -1.2850 | -1.1873 | -1.1166 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.52783 | 0.52171 | 0.54180 | 0.54363 | 0.50660 | 0.50419 | 0.52558 | 0.53398 | 0.54396 | 0.53887 | 0.55109 | 0.55430 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.52783 | 0.52171 | 0.54180 | 0.54363 | 0.50660 | 0.50419 | 0.52558 | 0.53398 | 0.54396 | 0.53887 | 0.55109 | 0.55430 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.30556 | 0.29116 | 0.29245 | 0.29839 | 0.31735 | 0.34236 | 0.34166 | 0.35037 | 0.35450 | 0.36211 | 0.27784 | 0.31433 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.33335 | 0.31668 | 0.31873 | 0.32570 | 0.34636 | 0.37370 | 0.37313 | 0.38245 | 0.38847 | 0.39612 | 0.30257 | 0.34321 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.21178 | 0.20180 | 0.20270 | 0.20683 | 0.21996 | 0.23728 | 0.23685 | 0.24287 | 0.24575 | 0.25098 | 0.19258 | 0.21788 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.23109 | 0.21950 | 0.22092 | 0.22576 | 0.24006 | 0.25903 | 0.25863 | 0.26510 | 0.26924 | 0.27461 | 0.20970 | 0.23790 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.66601 | 0.67292 | 0.71838 | 0.71443 | 0.66996 | 0.66996 | 0.67391 | 0.66601 | 0.68478 | 0.66403 | 0.73913 | 0.70257 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.66601 | 0.67292 | 0.71838 | 0.71443 | 0.66996 | 0.66996 | 0.67391 | 0.66601 | 0.68478 | 0.66403 | 0.73913 | 0.70257 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -1.2074 | -1.1675 | -1.0788 | -0.98903 | -1.1584 | -1.2351 | -1.1644 | -1.2193 | -1.1437 | -1.3010 | -0.98247 | -1.1362 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -1.2074 | -1.1675 | -1.0788 | -0.98903 | -1.1584 | -1.2351 | -1.1644 | -1.2193 | -1.1437 | -1.3010 | -0.98247 | -1.1362 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.65380 | 0.66122 | 0.69092 | 0.68824 | 0.65309 | 0.64456 | 0.65625 | 0.64588 | 0.66223 | 0.63022 | 0.69929 | 0.67467 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.65380 | 0.66122 | 0.69092 | 0.68824 | 0.65309 | 0.64456 | 0.65625 | 0.64588 | 0.66223 | 0.63022 | 0.69929 | 0.67467 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.36981 | 0.32024 | 0.26960 | 0.24629 | 0.27115 | 0.30652 | 0.32481 | 0.27484 | 0.31979 | 0.29974 | 0.33069 | 0.22914 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.44633 | 0.38468 | 0.32431 | 0.29491 | 0.32592 | 0.36890 | 0.39101 | 0.33030 | 0.38395 | 0.36156 | 0.39771 | 0.27526 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.25649 | 0.22206 | 0.18706 | 0.17079 | 0.18807 | 0.21262 | 0.22533 | 0.19063 | 0.22182 | 0.20791 | 0.22939 | 0.15895 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.30954 | 0.26677 | 0.22507 | 0.20457 | 0.22608 | 0.25591 | 0.27128 | 0.22914 | 0.26634 | 0.25082 | 0.27590 | 0.19096 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.90357 | 0.92285 | 0.93057 | 0.94407 | 0.92093 | 0.91707 | 0.89103 | 0.93443 | 0.93153 | 0.92960 | 0.92093 | 0.93925 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.90357 | 0.92285 | 0.93057 | 0.94407 | 0.92093 | 0.91707 | 0.89103 | 0.93443 | 0.93153 | 0.92960 | 0.92093 | 0.93925 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.29048 | -0.25760 | -0.22857 | -0.18946 | -0.25129 | -0.25731 | -0.26966 | -0.22458 | -0.22869 | -0.23539 | -0.23390 | -0.21434 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.29048 | -0.25760 | -0.22857 | -0.18946 | -0.25129 | -0.25731 | -0.26966 | -0.22458 | -0.22869 | -0.23539 | -0.23390 | -0.21434 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.82118 | 0.83794 | 0.85263 | 0.87128 | 0.84075 | 0.83889 | 0.82773 | 0.85081 | 0.85229 | 0.84618 | 0.84981 | 0.86697 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.82118 | 0.83794 | 0.85263 | 0.87128 | 0.84075 | 0.83889 | 0.82773 | 0.85081 | 0.85229 | 0.84618 | 0.84981 | 0.86697 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.29689 | 0.27705 | 0.24619 | 0.23022 | 0.27064 | 0.28847 | 0.30207 | 0.28891 | 0.28387 | 0.28024 | 0.29624 | 0.29554 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.30692 | 0.28632 | 0.25442 | 0.23801 | 0.27976 | 0.29817 | 0.31227 | 0.29872 | 0.29336 | 0.28961 | 0.30624 | 0.30549 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.20582 | 0.19206 | 0.17064 | 0.15958 | 0.18762 | 0.19998 | 0.20940 | 0.20029 | 0.19679 | 0.19426 | 0.20536 | 0.20487 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.21276 | 0.19850 | 0.17637 | 0.16499 | 0.19393 | 0.20671 | 0.21647 | 0.20706 | 0.20336 | 0.20078 | 0.21230 | 0.21177 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.86941 | 0.90519 | 0.91324 | 0.89267 | 0.84347 | 0.91145 | 0.93292 | 0.92129 | 0.86225 | 0.88104 | 0.93113 | 0.85599 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.86941 | 0.90519 | 0.91324 | 0.89267 | 0.84347 | 0.91145 | 0.93292 | 0.92129 | 0.86225 | 0.88104 | 0.93113 | 0.85599 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.33311 | -0.24731 | -0.24285 | -0.28121 | -0.35049 | -0.23102 | -0.21223 | -0.23324 | -0.33302 | -0.26440 | -0.20325 | -0.32048 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.33311 | -0.24731 | -0.24285 | -0.28121 | -0.35049 | -0.23102 | -0.21223 | -0.23324 | -0.33302 | -0.26440 | -0.20325 | -0.32048 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.84737 | 0.88337 | 0.89538 | 0.88582 | 0.85643 | 0.89555 | 0.90489 | 0.90326 | 0.85973 | 0.87221 | 0.91067 | 0.86230 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.84737 | 0.88337 | 0.89538 | 0.88582 | 0.85643 | 0.89555 | 0.90489 | 0.90326 | 0.85973 | 0.87221 | 0.91067 | 0.86230 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 0.78219 | 0.72763 | 0.74717 | 0.74536 | 0.78729 | 0.83252 | 0.73172 | 0.71312 | 0.72821 | 0.75538 | 0.76103 | 0.72000 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.3100 | 1.2105 | 1.2318 | 1.2520 | 1.3136 | 1.3682 | 1.2220 | 1.1885 | 1.2078 | 1.2593 | 1.2586 | 1.1970 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.56567 | 0.52999 | 0.54497 | 0.54875 | 0.57020 | 0.60857 | 0.53545 | 0.52339 | 0.53216 | 0.55142 | 0.55572 | 0.52791 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 0.96801 | 0.90295 | 0.92289 | 0.94843 | 0.97276 | 1.0294 | 0.91922 | 0.89835 | 0.90661 | 0.94389 | 0.94404 | 0.90245 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.73970 | 0.77341 | 0.74719 | 0.77154 | 0.74719 | 0.75094 | 0.76966 | 0.76030 | 0.76030 | 0.73596 | 0.74719 | 0.79026 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.73970 | 0.77341 | 0.74719 | 0.77154 | 0.74719 | 0.75094 | 0.76966 | 0.76030 | 0.76030 | 0.73596 | 0.74719 | 0.79026 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.61829 | -0.61531 | -0.61647 | -0.59103 | -0.64360 | -0.64549 | -0.61935 | -0.61029 | -0.62536 | -0.63262 | -0.59074 | -0.55246 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.61829 | -0.61531 | -0.61647 | -0.59103 | -0.64360 | -0.64549 | -0.61935 | -0.61029 | -0.62536 | -0.63262 | -0.59074 | -0.55246 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.68162 | 0.68373 | 0.68749 | 0.69107 | 0.67674 | 0.66770 | 0.69315 | 0.69290 | 0.68997 | 0.67154 | 0.68333 | 0.70541 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.68162 | 0.68373 | 0.68749 | 0.69107 | 0.67674 | 0.66770 | 0.69315 | 0.69290 | 0.68997 | 0.67154 | 0.68333 | 0.70541 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.3635 | 1.2221 | 1.1225 | 1.1495 | 1.4433 | 1.3975 | 1.3217 | 1.2240 | 1.2153 | 1.4340 | 1.3129 | 1.2126 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 1.9094 | 1.7255 | 1.5901 | 1.6142 | 2.0142 | 1.9381 | 1.8386 | 1.6991 | 1.7238 | 1.9944 | 1.8522 | 1.6861 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 0.94520 | 0.84706 | 0.77814 | 0.79672 | 1.0004 | 0.96873 | 0.91611 | 0.84847 | 0.84248 | 0.99390 | 0.91002 | 0.84057 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.3235 | 1.1960 | 1.1022 | 1.1188 | 1.3962 | 1.3433 | 1.2745 | 1.1778 | 1.1949 | 1.3823 | 1.2838 | 1.1687 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.29696 | 0.35603 | 0.39541 | 0.37490 | 0.31337 | 0.33224 | 0.32568 | 0.34372 | 0.37408 | 0.35275 | 0.34537 | 0.39951 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.29696 | 0.35603 | 0.39541 | 0.37490 | 0.31337 | 0.33224 | 0.32568 | 0.34372 | 0.37408 | 0.35275 | 0.34537 | 0.39951 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -3.2558 | -2.5106 | -2.4753 | -2.5351 | -3.1454 | -2.8736 | -2.7309 | -2.7324 | -2.5289 | -2.8649 | -2.6435 | -2.5849 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -3.2558 | -2.5106 | -2.4753 | -2.5351 | -3.1454 | -2.8736 | -2.7309 | -2.7324 | -2.5289 | -2.8649 | -2.6435 | -2.5849 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.31272 | 0.36357 | 0.38594 | 0.38478 | 0.32359 | 0.34784 | 0.33436 | 0.35158 | 0.39331 | 0.36172 | 0.35693 | 0.38750 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.31272 | 0.36357 | 0.38594 | 0.38478 | 0.32359 | 0.34784 | 0.33436 | 0.35158 | 0.39331 | 0.36172 | 0.35693 | 0.38750 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.38236 | 0.37632 | 0.36909 | 0.35738 | 0.38203 | 0.37780 | 0.38487 | 0.36465 | 0.36443 | 0.38034 | 0.36188 | 0.37761 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.38757 | 0.38137 | 0.37391 | 0.36222 | 0.38731 | 0.38274 | 0.39004 | 0.36949 | 0.36913 | 0.38544 | 0.36673 | 0.38256 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.58302 | 0.56447 | 0.55928 | 0.56141 | 0.58309 | 0.57884 | 0.57442 | 0.56921 | 0.57593 | 0.58279 | 0.56421 | 0.56285 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.58805 | 0.56954 | 0.56417 | 0.56636 | 0.58808 | 0.58392 | 0.57948 | 0.57413 | 0.58097 | 0.58772 | 0.56921 | 0.56780 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.1705 | 1.9576 | 1.9632 | 1.8445 | 1.8341 | 1.8202 | 1.8786 | 1.7882 | 1.8987 | 1.7678 | 2.0769 | 2.0697 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 4.3409 | 3.9152 | 3.9264 | 3.6889 | 3.6683 | 3.6403 | 3.7572 | 3.5764 | 3.7975 | 3.5356 | 4.1538 | 4.1394 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.5047 | 1.3568 | 1.3606 | 1.2787 | 1.2715 | 1.2614 | 1.3022 | 1.2391 | 1.3159 | 1.2252 | 1.4393 | 1.4348 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.0095 | 2.7137 | 2.7213 | 2.5575 | 2.5430 | 2.5229 | 2.6043 | 2.4783 | 2.6318 | 2.4505 | 2.8787 | 2.8696 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.10000 | 0.13000 | 0.10000 | 0.10000 | 0.07000 | 0.12000 | 0.14000 | 0.12000 | 0.06000 | 0.10000 | 0.10000 | 0.08000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.10000 | 0.13000 | 0.10000 | 0.10000 | 0.07000 | 0.12000 | 0.14000 | 0.12000 | 0.06000 | 0.10000 | 0.10000 | 0.08000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -3.0049 | -2.7095 | -2.7164 | -2.5509 | -2.5321 | -2.5126 | -2.5939 | -2.4707 | -2.6271 | -2.4364 | -2.8738 | -2.8633 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -3.0049 | -2.7095 | -2.7164 | -2.5509 | -2.5321 | -2.5126 | -2.5939 | -2.4707 | -2.6271 | -2.4364 | -2.8738 | -2.8633 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09669 | 0.10295 | 0.09770 | 0.09732 | 0.10089 | 0.10029 | 0.10732 | 0.10222 | 0.09821 | 0.10007 | 0.10228 | 0.09552 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09669 | 0.10295 | 0.09770 | 0.09732 | 0.10089 | 0.10029 | 0.10732 | 0.10222 | 0.09821 | 0.10007 | 0.10228 | 0.09552 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.76462 | 0.75402 | 0.74252 | 0.74479 | 0.76642 | 0.76219 | 0.76054 | 0.75249 | 0.75502 | 0.75650 | 0.74383 | 0.75310 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.77305 | 0.76225 | 0.75093 | 0.75318 | 0.77495 | 0.77068 | 0.76900 | 0.76085 | 0.76340 | 0.76478 | 0.75205 | 0.76153 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.59527 | 0.58094 | 0.55559 | 0.56129 | 0.59819 | 0.58895 | 0.58474 | 0.57344 | 0.57216 | 0.59105 | 0.56101 | 0.57348 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.59733 | 0.58289 | 0.55749 | 0.56302 | 0.59996 | 0.59085 | 0.58669 | 0.57525 | 0.57401 | 0.59300 | 0.56287 | 0.57540 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.65519 | 0.64897 | 0.62273 | 0.62427 | 0.65992 | 0.64465 | 0.64567 | 0.63266 | 0.63251 | 0.64139 | 0.62751 | 0.64207 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.68790 | 0.68093 | 0.65274 | 0.65525 | 0.69315 | 0.67635 | 0.67783 | 0.66311 | 0.66323 | 0.67296 | 0.65801 | 0.67398 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0282 | 1.0090 | 1.0163 | 1.0046 | 1.0153 | 1.0210 | 1.0278 | 1.0013 | 1.0059 | 1.0084 | 1.0092 | 1.0072 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0564 | 2.0181 | 2.0326 | 2.0092 | 2.0307 | 2.0420 | 2.0555 | 2.0026 | 2.0117 | 2.0168 | 2.0184 | 2.0145 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.71276 | 0.69951 | 0.70450 | 0.69643 | 0.70382 | 0.70774 | 0.71244 | 0.69416 | 0.69732 | 0.69906 | 0.69960 | 0.69826 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4255 | 1.3990 | 1.4090 | 1.3929 | 1.4076 | 1.4155 | 1.4249 | 1.3883 | 1.3946 | 1.3981 | 1.3992 | 1.3965 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24272 | 0.27503 | 0.25058 | 0.25739 | 0.25143 | 0.25611 | 0.24230 | 0.26419 | 0.25101 | 0.25292 | 0.25271 | 0.26610 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24272 | 0.27503 | 0.25058 | 0.25739 | 0.25143 | 0.25611 | 0.24230 | 0.26419 | 0.25101 | 0.25292 | 0.25271 | 0.26610 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3948 | -1.3865 | -1.3895 | -1.3861 | -1.3889 | -1.3910 | -1.3946 | -1.3840 | -1.3864 | -1.3861 | -1.3869 | -1.3857 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.4221 | -1.3960 | -1.4056 | -1.3911 | -1.4032 | -1.4118 | -1.4204 | -1.3859 | -1.3918 | -1.3947 | -1.3964 | -1.3924 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.25017 | 0.25111 | 0.25077 | 0.25067 | 0.25082 | 0.25087 | 0.24994 | 0.25109 | 0.25065 | 0.25113 | 0.25097 | 0.25106 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.25012 | 0.25224 | 0.25154 | 0.25129 | 0.25155 | 0.25152 | 0.24974 | 0.25215 | 0.25122 | 0.25210 | 0.25191 | 0.25202 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 0.90668 | 0.89520 | 0.85866 | 0.85134 | 0.91185 | 0.89252 | 0.88910 | 0.87727 | 0.87293 | 0.89059 | 0.86449 | 0.87930 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.0094 | 0.99876 | 0.95779 | 0.94923 | 1.0168 | 0.99410 | 0.98861 | 0.97598 | 0.97321 | 0.99149 | 0.96243 | 0.97977 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0185 | 1.0038 | 1.0070 | 0.99878 | 1.0177 | 1.0169 | 1.0249 | 1.0015 | 1.0037 | 1.0104 | 1.0035 | 1.0107 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0369 | 2.0076 | 2.0140 | 1.9976 | 2.0353 | 2.0337 | 2.0499 | 2.0031 | 2.0074 | 2.0207 | 2.0070 | 2.0214 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.70599 | 0.69585 | 0.69811 | 0.69247 | 0.70546 | 0.70488 | 0.71052 | 0.69435 | 0.69580 | 0.70041 | 0.69560 | 0.70067 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4120 | 1.3917 | 1.3962 | 1.3849 | 1.4109 | 1.4098 | 1.4210 | 1.3887 | 1.3916 | 1.4008 | 1.3912 | 1.4013 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24614 | 0.25386 | 0.25879 | 0.26774 | 0.24028 | 0.23566 | 0.23689 | 0.26188 | 0.26496 | 0.24645 | 0.25817 | 0.24800 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24614 | 0.25386 | 0.25879 | 0.26774 | 0.24028 | 0.23566 | 0.23689 | 0.26188 | 0.26496 | 0.24645 | 0.25817 | 0.24800 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3880 | -1.3836 | -1.3838 | -1.3826 | -1.3906 | -1.3886 | -1.3945 | -1.3836 | -1.3837 | -1.3861 | -1.3813 | -1.3873 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.4085 | -1.3886 | -1.3930 | -1.3829 | -1.4071 | -1.4063 | -1.4172 | -1.3862 | -1.3889 | -1.3973 | -1.3883 | -1.3973 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25179 | 0.25158 | 0.25214 | 0.25144 | 0.25040 | 0.25128 | 0.24965 | 0.25130 | 0.25167 | 0.25138 | 0.25282 | 0.25084 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25317 | 0.25303 | 0.25438 | 0.25291 | 0.25068 | 0.25220 | 0.24914 | 0.25253 | 0.25320 | 0.25250 | 0.25560 | 0.25159 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.77640 | 0.76543 | 0.73480 | 0.73482 | 0.78729 | 0.76739 | 0.76662 | 0.75796 | 0.75146 | 0.76591 | 0.73800 | 0.75255 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 0.82802 | 0.81638 | 0.78317 | 0.78365 | 0.84144 | 0.81879 | 0.81711 | 0.80860 | 0.80175 | 0.81718 | 0.78742 | 0.80300 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0403 | 1.0100 | 1.0078 | 1.0024 | 1.0268 | 1.0338 | 1.0340 | 1.0067 | 1.0066 | 1.0210 | 1.0234 | 1.0156 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0806 | 2.0201 | 2.0156 | 2.0048 | 2.0537 | 2.0677 | 2.0680 | 2.0134 | 2.0132 | 2.0421 | 2.0468 | 2.0311 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.72108 | 0.70023 | 0.69862 | 0.69495 | 0.71185 | 0.71663 | 0.71683 | 0.69792 | 0.69784 | 0.70779 | 0.70942 | 0.70406 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4422 | 1.4005 | 1.3972 | 1.3899 | 1.4237 | 1.4333 | 1.4337 | 1.3958 | 1.3957 | 1.4156 | 1.4188 | 1.4081 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.21514 | 0.24699 | 0.25089 | 0.26779 | 0.23074 | 0.22164 | 0.22814 | 0.25674 | 0.27364 | 0.23009 | 0.24082 | 0.24017 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.21514 | 0.24699 | 0.25089 | 0.26779 | 0.23074 | 0.22164 | 0.22814 | 0.25674 | 0.27364 | 0.23009 | 0.24082 | 0.24017 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.4041 | -1.3892 | -1.3838 | -1.3854 | -1.3984 | -1.4024 | -1.4017 | -1.3876 | -1.3869 | -1.3948 | -1.3946 | -1.3920 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.4383 | -1.3979 | -1.3941 | -1.3879 | -1.4200 | -1.4302 | -1.4303 | -1.3931 | -1.3932 | -1.4123 | -1.4159 | -1.4046 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.24758 | 0.24998 | 0.25226 | 0.25065 | 0.24821 | 0.24758 | 0.24776 | 0.25020 | 0.25057 | 0.24900 | 0.24962 | 0.24945 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.24519 | 0.24991 | 0.25455 | 0.25133 | 0.24651 | 0.24534 | 0.24558 | 0.25040 | 0.25112 | 0.24792 | 0.24928 | 0.24892 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.1707 | 1.1436 | 1.1169 | 1.1080 | 1.1789 | 1.1728 | 1.1530 | 1.1399 | 1.1330 | 1.1703 | 1.1134 | 1.1389 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.4590 | 1.4284 | 1.3959 | 1.3822 | 1.4743 | 1.4674 | 1.4390 | 1.4257 | 1.4141 | 1.4615 | 1.3907 | 1.4250 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0264 | 1.0073 | 1.0084 | 0.99987 | 1.0254 | 1.0278 | 1.0278 | 1.0036 | 1.0106 | 1.0208 | 1.0150 | 1.0124 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0527 | 2.0146 | 2.0169 | 1.9997 | 2.0508 | 2.0557 | 2.0557 | 2.0072 | 2.0212 | 2.0415 | 2.0300 | 2.0249 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.71152 | 0.69823 | 0.69908 | 0.69315 | 0.71087 | 0.71252 | 0.71248 | 0.69573 | 0.70056 | 0.70758 | 0.70359 | 0.70186 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4230 | 1.3965 | 1.3982 | 1.3863 | 1.4217 | 1.4250 | 1.4250 | 1.3915 | 1.4011 | 1.4152 | 1.4072 | 1.4037 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.23095 | 0.26640 | 0.25215 | 0.27402 | 0.24884 | 0.22266 | 0.22962 | 0.27402 | 0.26276 | 0.25149 | 0.24188 | 0.26276 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.23095 | 0.26640 | 0.25215 | 0.27402 | 0.24884 | 0.22266 | 0.22962 | 0.27402 | 0.26276 | 0.25149 | 0.24188 | 0.26276 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3952 | -1.3851 | -1.3859 | -1.3825 | -1.3959 | -1.3979 | -1.3972 | -1.3843 | -1.3877 | -1.3924 | -1.3902 | -1.3873 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4182 | -1.3938 | -1.3941 | -1.3843 | -1.4175 | -1.4213 | -1.4208 | -1.3886 | -1.3982 | -1.4111 | -1.4038 | -1.3985 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.24950 | 0.25158 | 0.25119 | 0.25171 | 0.24919 | 0.24858 | 0.24882 | 0.25139 | 0.25090 | 0.25010 | 0.25032 | 0.25105 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.24889 | 0.25317 | 0.25241 | 0.25347 | 0.24850 | 0.24711 | 0.24754 | 0.25285 | 0.25183 | 0.25009 | 0.25068 | 0.25211 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.38505 | 0.38160 | 0.38453 | 0.37044 | 0.37276 | 0.38295 | 0.37999 | 0.38496 | 0.37287 | 0.38296 | 0.38827 | 0.38323 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.38726 | 0.38390 | 0.38681 | 0.37267 | 0.37478 | 0.38507 | 0.38212 | 0.38720 | 0.37508 | 0.38527 | 0.39051 | 0.38556 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.30548 | 0.30329 | 0.28780 | 0.27717 | 0.28169 | 0.29277 | 0.28870 | 0.28913 | 0.27671 | 0.29376 | 0.28571 | 0.29786 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.30671 | 0.30448 | 0.28893 | 0.27830 | 0.28269 | 0.29390 | 0.28983 | 0.29021 | 0.27780 | 0.29496 | 0.28686 | 0.29906 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.52679 | 0.50078 | 0.49704 | 0.46051 | 0.49618 | 0.48715 | 0.49226 | 0.48822 | 0.46248 | 0.50708 | 0.48775 | 0.49157 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.53061 | 0.50427 | 0.50060 | 0.46380 | 0.49980 | 0.49065 | 0.49585 | 0.49164 | 0.46565 | 0.51078 | 0.49115 | 0.49504 |
| eval/lm/c4_en-validation/CE loss | lower | 2.8049 | 2.7757 | 2.7206 | 2.7265 | 2.8024 | 2.7869 | 2.7888 | 2.7644 | 2.7525 | 2.7910 | 2.7258 | 2.7545 |
| eval/lm/c4_en-validation/PPL | lower | 16.53 | 16.05 | 15.19 | 15.28 | 16.48 | 16.23 | 16.26 | 15.87 | 15.68 | 16.30 | 15.27 | 15.71 |
| eval/lm/dolma_books-validation/CE loss | lower | 2.6655 | 2.6349 | 2.5746 | 2.5747 | 2.6661 | 2.6472 | 2.6471 | 2.6147 | 2.6065 | 2.6522 | 2.5823 | 2.6049 |
| eval/lm/dolma_books-validation/PPL | lower | 14.37 | 13.94 | 13.13 | 13.13 | 14.38 | 14.12 | 14.11 | 13.66 | 13.55 | 14.18 | 13.23 | 13.53 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 2.9424 | 2.9119 | 2.8600 | 2.8634 | 2.9395 | 2.9228 | 2.9260 | 2.8981 | 2.8900 | 2.9291 | 2.8611 | 2.8905 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 18.96 | 18.39 | 17.46 | 17.52 | 18.91 | 18.59 | 18.65 | 18.14 | 17.99 | 18.71 | 17.48 | 18.00 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.0218 | 2.0005 | 1.9643 | 1.9692 | 2.0239 | 2.0105 | 2.0085 | 1.9936 | 1.9820 | 2.0142 | 1.9722 | 1.9892 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 7.5520 | 7.3928 | 7.1298 | 7.1647 | 7.5679 | 7.4672 | 7.4522 | 7.3421 | 7.2572 | 7.4951 | 7.1862 | 7.3094 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.1323 | 3.1030 | 3.0552 | 3.0604 | 3.1317 | 3.1149 | 3.1122 | 3.0916 | 3.0867 | 3.1173 | 3.0666 | 3.0887 |
| eval/lm/dolma_reddit-validation/PPL | lower | 22.93 | 22.26 | 21.23 | 21.34 | 22.91 | 22.53 | 22.47 | 22.01 | 21.91 | 22.58 | 21.47 | 21.95 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.2133 | 1.1947 | 1.1584 | 1.1611 | 1.2132 | 1.2012 | 1.2039 | 1.1866 | 1.1799 | 1.2075 | 1.1603 | 1.1804 |
| eval/lm/dolma_stack-validation/PPL | lower | 3.3645 | 3.3026 | 3.1850 | 3.1933 | 3.3642 | 3.3242 | 3.3331 | 3.2761 | 3.2542 | 3.3451 | 3.1908 | 3.2556 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.4322 | 2.4080 | 2.3502 | 2.3500 | 2.4308 | 2.4200 | 2.4205 | 2.3897 | 2.3802 | 2.4189 | 2.3443 | 2.3828 |
| eval/lm/dolma_wiki-validation/PPL | lower | 11.38 | 11.11 | 10.49 | 10.49 | 11.37 | 11.25 | 11.25 | 10.91 | 10.81 | 11.23 | 10.43 | 10.84 |
| eval/lm/ice-validation/CE loss | lower | 2.8775 | 2.8364 | 2.8009 | 2.8050 | 2.8694 | 2.8418 | 2.8287 | 2.8251 | 2.8074 | 2.8685 | 2.8201 | 2.8344 |
| eval/lm/ice-validation/PPL | lower | 17.77 | 17.05 | 16.46 | 16.53 | 17.63 | 17.15 | 16.92 | 16.86 | 16.57 | 17.61 | 16.78 | 17.02 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 2.9261 | 2.9063 | 2.8667 | 2.8742 | 2.9403 | 2.9283 | 2.9345 | 2.9029 | 2.8939 | 2.9317 | 2.8699 | 2.9013 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 18.66 | 18.29 | 17.58 | 17.71 | 18.92 | 18.70 | 18.81 | 18.23 | 18.06 | 18.76 | 17.63 | 18.20 |
| eval/lm/pile-validation/CE loss | lower | 2.0852 | 2.0628 | 2.0171 | 2.0188 | 2.0848 | 2.0707 | 2.0695 | 2.0528 | 2.0397 | 2.0731 | 2.0185 | 2.0450 |
| eval/lm/pile-validation/PPL | lower | 8.0461 | 7.8683 | 7.5164 | 7.5296 | 8.0429 | 7.9300 | 7.9208 | 7.7897 | 7.6880 | 7.9493 | 7.5269 | 7.7294 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.3487 | 2.3142 | 2.2575 | 2.2610 | 2.3520 | 2.3287 | 2.3296 | 2.3022 | 2.2815 | 2.3344 | 2.2595 | 2.2961 |
| eval/lm/wikitext_103-validation/PPL | lower | 10.47 | 10.12 | 9.5595 | 9.5929 | 10.51 | 10.27 | 10.27 | 9.9959 | 9.7914 | 10.32 | 9.5787 | 9.9350 |
| throughput/in-loop eval batches | see metric | 557.0 | 557.0 | 828.0 | 828.0 | 828.0 | 828.0 | 828.0 | 828.0 | 828.0 | 828.0 | 1729.0 | 828.0 |
| throughput/in-loop eval time (s) | see metric | 69.71 | 85.21 | 91.42 | 83.30 | 80.55 | 86.74 | 73.84 | 148.1 | 108.3 | 89.65 | 118.3 | 103.7 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| eg-810m-cx8-eg24e2k-lr4e-4-r1<br>`3h6suqpb` | finished | original | 110316748800.0 | 140275 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/3h6suqpb) |
| eg-810m-cx8-eg96e8k-lr4e-4-r1<br>`n9cgtrxm` | finished | original | 110596718592.0 | 140631 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/n9cgtrxm) |
| int-810m-cx8-intd256e8k-lr4e-4-r1<br>`142yfmzn` | finished | original | 109014417408.0 | 138619 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/142yfmzn) |
| int-810m-cx8-intw256e8k-lr4e-4-r1<br>`kyti8h1y` | finished | original | 111219572736.0 | 141423 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kyti8h1y) |
| 810m-cx8-b768k-lr2e-4-r1<br>`a0k0519k` | finished | gpu8-ep1mb4 | 110410334208.0 | 140394 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/a0k0519k) |
| 810m-cx8-b768k-lr4e-4-r1<br>`dkpaicdc` | finished | gpu8-ep1mb4 | 110410334208.0 | 140394 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/dkpaicdc) |
| 810m-cx8-b768k-lr8e-4-r1<br>`rhtrhhet` | finished | gpu8-ep1mb4 | 110410334208.0 | 140394 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rhtrhhet) |
| q3-810m-cx8-q3am128e8k-lr4e-4-r1<br>`98kpaudi` | finished | original | 110747713536.0 | 140823 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/98kpaudi) |
| q3-810m-cx8-q3td128e8k-lr4e-4-r1<br>`e24iehj0` | finished | original | 110465384448.0 | 140464 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/e24iehj0) |
| se-810m-cx8-se0m9-lr4e-4-r1<br>`b17jsgm7` | finished | original | 110410334208.0 | 140394 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/b17jsgm7) |
| sp-810m-cx8-sp192e4k-lr3e-4-r2<br>`c4bcu3ho` | finished | original | 110970273792.0 | 141106 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/c4bcu3ho) |
| sp-810m-cx8-sp96e4k-lr3.5e-4-r1<br>`ypl4ayew` | finished | original | 110596718592.0 | 140631 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ypl4ayew) |

## unknown Cx1

Showing first 24 of 31 runs in this table. Use `--name-regex` to narrow the view.

| metric | direction | int-275m-cx1-intd256e8k-lr1.6e-3-r1<br>`b2g99ewo` | int-275m-cx1-intd256e8k-lr3.2e-3-r1<br>`fmpio3ko` | int-275m-cx1-intd256e8k-lr8e-4-r1<br>`51vzuu2l` | int-275m-cx1-intw256e8k-lr1.6e-3-r1<br>`h86x1nv3` | int-275m-cx1-intw256e8k-lr3.2e-3-r1<br>`afxq80js` | int-275m-cx1-intw256e8k-lr8e-4-r1<br>`kfua3dcq` | mt-275m-baseline-cx1-lr1.6e-3-r1<br>`w3vof8b9` | mt-275m-baseline-cx1-lr2e-4-r1<br>`cm8ww646` | mt-275m-baseline-cx1-lr4e-4-r1<br>`r6ts032g` | mt-275m-baseline-cx1-lr8e-4-r1<br>`lfydkxv4` | mt-eval-275m-baseline-cx1-lr1.6e-3-r1<br>`g72jnvlh` | mt-eval-275m-baseline-cx1-lr2e-4-r1<br>`z2n5obvt` | mt-eval-275m-baseline-cx1-lr4e-4-r1<br>`946lvt8o` | mt-eval-275m-baseline-cx1-lr8e-4-r1<br>`q6tuf453` | olmoe3-eval-275m-cx1-lr1e-3-r2<br>`h3y8marg` | q3-275m-cx1-q3am128e8k-lr1e-3-r1<br>`ivpueqxh` | q3-275m-cx1-q3am128e8k-lr2e-3-r1<br>`5vaz5tl1` | q3-275m-cx1-q3am128e8k-lr4e-3-r1<br>`wtnzni2d` | q3-275m-cx1-q3td128e8k-lr1e-3-r1<br>`ww4vodxv` | q3-275m-cx1-q3td128e8k-lr2e-3-r1<br>`fhgythx3` | q3-275m-cx1-q3td128e8k-lr4e-3-r1<br>`1erw5m3k` | q3-275m-cx1-q3td128e8k-lr5e-4-r1<br>`sujdnlv0` | se-275m-cx1-se0m9-lr1e-3-r2<br>`z7os3acu` | se-275m-cx1-se0m9-lr2e-3-r2<br>`0af1i3o1` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 1.0458 | 1.0660 | 1.0740 | 1.0464 | 1.0634 | 1.0651 | 0.94430 | 0.90040 | 0.91077 | 0.92087 | 0.94430 | 0.90040 | 0.91077 | 0.92087 | 1.0865 | 1.0685 | 1.0557 | 1.0996 | 1.0636 | 1.0736 | 1.0515 | 1.0892 | 1.0836 | 1.0663 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 1.1446 | 1.1665 | 1.1741 | 1.1441 | 1.1655 | 1.1661 | 1.0332 | 0.98279 | 0.99730 | 1.0085 | 1.0332 | 0.98279 | 0.99730 | 1.0085 | 1.1885 | 1.1704 | 1.1566 | 1.2075 | 1.1654 | 1.1768 | 1.1535 | 1.1942 | 1.1876 | 1.1670 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0721 | 1.0438 | 1.0643 | 1.0602 | 1.0491 | 1.0406 | 1.0601 | 1.0399 | 1.0123 | 1.0441 | 1.0601 | 1.0399 | 1.0123 | 1.0441 | 1.0260 | 1.0258 | 1.0548 | 1.0270 | 1.0200 | 1.0309 | 1.0358 | 1.0336 | 1.0478 | 1.0580 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.1441 | 2.0876 | 2.1286 | 2.1204 | 2.0982 | 2.0811 | 2.1202 | 2.0798 | 2.0246 | 2.0882 | 2.1202 | 2.0798 | 2.0246 | 2.0882 | 2.0520 | 2.0515 | 2.1097 | 2.0539 | 2.0400 | 2.0618 | 2.0716 | 2.0672 | 2.0955 | 2.1161 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.74320 | 0.72358 | 0.73775 | 0.73490 | 0.72722 | 0.72125 | 0.73480 | 0.72079 | 0.70174 | 0.72373 | 0.73480 | 0.72079 | 0.70174 | 0.72373 | 0.71126 | 0.71112 | 0.73106 | 0.71188 | 0.70712 | 0.71462 | 0.71795 | 0.71650 | 0.72617 | 0.73342 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4864 | 1.4472 | 1.4755 | 1.4698 | 1.4544 | 1.4425 | 1.4696 | 1.4416 | 1.4035 | 1.4475 | 1.4696 | 1.4416 | 1.4035 | 1.4475 | 1.4225 | 1.4222 | 1.4621 | 1.4238 | 1.4142 | 1.4292 | 1.4359 | 1.4330 | 1.4523 | 1.4668 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.22952 | 0.25768 | 0.23379 | 0.26877 | 0.25597 | 0.26877 | 0.29608 | 0.30290 | 0.28584 | 0.26962 | 0.29608 | 0.30290 | 0.28584 | 0.26962 | 0.25853 | 0.23038 | 0.26451 | 0.27218 | 0.25512 | 0.26365 | 0.27048 | 0.25853 | 0.26280 | 0.24659 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.22952 | 0.25768 | 0.23379 | 0.26877 | 0.25597 | 0.26877 | 0.29608 | 0.30290 | 0.28584 | 0.26962 | 0.29608 | 0.30290 | 0.28584 | 0.26962 | 0.25853 | 0.23038 | 0.26451 | 0.27218 | 0.25512 | 0.26365 | 0.27048 | 0.25853 | 0.26280 | 0.24659 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.4665 | -1.4265 | -1.4459 | -1.4559 | -1.4138 | -1.4231 | -1.4664 | -1.4398 | -1.4002 | -1.4455 | -1.4664 | -1.4398 | -1.4002 | -1.4455 | -1.4063 | -1.4052 | -1.4434 | -1.4081 | -1.3976 | -1.4116 | -1.4262 | -1.4095 | -1.4327 | -1.4451 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.4665 | -1.4265 | -1.4459 | -1.4559 | -1.4138 | -1.4231 | -1.4664 | -1.4398 | -1.4002 | -1.4455 | -1.4664 | -1.4398 | -1.4002 | -1.4455 | -1.4063 | -1.4052 | -1.4434 | -1.4081 | -1.3976 | -1.4116 | -1.4262 | -1.4095 | -1.4327 | -1.4451 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.24585 | 0.24947 | 0.24987 | 0.25351 | 0.25566 | 0.25618 | 0.26812 | 0.27198 | 0.26455 | 0.26440 | 0.26812 | 0.27198 | 0.26455 | 0.26440 | 0.25328 | 0.24935 | 0.25161 | 0.25336 | 0.25132 | 0.25166 | 0.25818 | 0.25054 | 0.25179 | 0.25180 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.24585 | 0.24947 | 0.24987 | 0.25351 | 0.25566 | 0.25618 | 0.26812 | 0.27198 | 0.26455 | 0.26440 | 0.26812 | 0.27198 | 0.26455 | 0.26440 | 0.25328 | 0.24935 | 0.25161 | 0.25336 | 0.25132 | 0.25166 | 0.25818 | 0.25054 | 0.25179 | 0.25180 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.87672 | 0.89043 | 0.88611 | 0.87290 | 0.90318 | 0.90089 | 0.74546 | 0.67326 | 0.68598 | 0.71772 | 0.74546 | 0.67326 | 0.68598 | 0.71772 | 0.92069 | 0.88456 | 0.89153 | 0.89477 | 0.88359 | 0.89391 | 0.86218 | 0.90099 | 0.91215 | 0.89556 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.95591 | 0.97160 | 0.96697 | 0.95242 | 0.98554 | 0.98354 | 0.81192 | 0.73274 | 0.74696 | 0.78173 | 0.81192 | 0.73274 | 0.74696 | 0.78173 | 1.0044 | 0.96441 | 0.97308 | 0.97561 | 0.96422 | 0.97511 | 0.93957 | 0.98320 | 0.99517 | 0.97622 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0335 | 1.0405 | 1.0368 | 1.0666 | 1.0734 | 1.0782 | 1.0287 | 0.99662 | 0.99125 | 1.0231 | 1.0287 | 0.99662 | 0.99125 | 1.0231 | 1.0448 | 1.0262 | 1.0537 | 1.0525 | 1.0362 | 1.0499 | 1.0370 | 1.0415 | 1.0630 | 1.0704 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0669 | 2.0810 | 2.0735 | 2.1332 | 2.1469 | 2.1565 | 2.0575 | 1.9932 | 1.9825 | 2.0461 | 2.0575 | 1.9932 | 1.9825 | 2.0461 | 2.0896 | 2.0525 | 2.1073 | 2.1051 | 2.0723 | 2.0997 | 2.0740 | 2.0831 | 2.1260 | 2.1408 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.71637 | 0.72120 | 0.71868 | 0.73932 | 0.74400 | 0.74738 | 0.71306 | 0.69082 | 0.68713 | 0.70916 | 0.71306 | 0.69082 | 0.68713 | 0.70916 | 0.72427 | 0.71142 | 0.73037 | 0.72956 | 0.71825 | 0.72776 | 0.71881 | 0.72194 | 0.73679 | 0.74195 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4327 | 1.4424 | 1.4374 | 1.4786 | 1.4880 | 1.4948 | 1.4261 | 1.3816 | 1.3743 | 1.4183 | 1.4261 | 1.3816 | 1.3743 | 1.4183 | 1.4485 | 1.4228 | 1.4607 | 1.4591 | 1.4365 | 1.4555 | 1.4376 | 1.4439 | 1.4736 | 1.4839 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.25589 | 0.24705 | 0.25421 | 0.26726 | 0.25715 | 0.24495 | 0.32113 | 0.33333 | 0.30051 | 0.27988 | 0.32113 | 0.33333 | 0.30051 | 0.27988 | 0.24200 | 0.26052 | 0.26726 | 0.25337 | 0.25800 | 0.22643 | 0.24158 | 0.25168 | 0.25084 | 0.23990 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.25589 | 0.24705 | 0.25421 | 0.26726 | 0.25715 | 0.24495 | 0.32113 | 0.33333 | 0.30051 | 0.27988 | 0.32113 | 0.33333 | 0.30051 | 0.27988 | 0.24200 | 0.26052 | 0.26726 | 0.25337 | 0.25800 | 0.22643 | 0.24158 | 0.25168 | 0.25084 | 0.23990 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.4150 | -1.4187 | -1.4107 | -1.4652 | -1.4539 | -1.4755 | -1.4234 | -1.3799 | -1.3699 | -1.4161 | -1.4234 | -1.3799 | -1.3699 | -1.4161 | -1.4338 | -1.4020 | -1.4361 | -1.4407 | -1.4238 | -1.4232 | -1.4265 | -1.4204 | -1.4523 | -1.4667 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.4150 | -1.4187 | -1.4107 | -1.4652 | -1.4539 | -1.4755 | -1.4234 | -1.3799 | -1.3699 | -1.4161 | -1.4234 | -1.3799 | -1.3699 | -1.4161 | -1.4338 | -1.4020 | -1.4361 | -1.4407 | -1.4238 | -1.4232 | -1.4265 | -1.4204 | -1.4523 | -1.4667 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.25325 | 0.25119 | 0.25208 | 0.25162 | 0.25090 | 0.25139 | 0.28151 | 0.28917 | 0.27068 | 0.27513 | 0.28151 | 0.28917 | 0.27068 | 0.27513 | 0.24864 | 0.25084 | 0.25331 | 0.24996 | 0.25011 | 0.24784 | 0.24954 | 0.25286 | 0.24970 | 0.24885 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.25325 | 0.25119 | 0.25208 | 0.25162 | 0.25090 | 0.25139 | 0.28151 | 0.28917 | 0.27068 | 0.27513 | 0.28151 | 0.28917 | 0.27068 | 0.27513 | 0.24864 | 0.25084 | 0.25331 | 0.24996 | 0.25011 | 0.24784 | 0.24954 | 0.25286 | 0.24970 | 0.24885 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 2.1871 | 2.1653 | 2.2793 | 2.1914 | 2.1035 | 2.2648 | 0.70821 | 0.65010 | 0.72478 | 0.55652 | 0.70821 | 0.65010 | 0.72478 | 0.55652 | 2.1681 | 2.2493 | 2.2209 | 2.2651 | 2.2522 | 2.2677 | 2.2816 | 2.2360 | 2.2906 | 2.2149 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 3.4880 | 3.4516 | 3.6372 | 3.4919 | 3.3794 | 3.5887 | 1.1286 | 1.0304 | 1.1536 | 0.89431 | 1.1286 | 1.0304 | 1.1536 | 0.89431 | 3.4980 | 3.5675 | 3.5410 | 3.6087 | 3.5909 | 3.6103 | 3.6664 | 3.5772 | 3.6575 | 3.5397 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 1.5159 | 1.5010 | 1.5799 | 1.5189 | 1.4581 | 1.5697 | 0.49085 | 0.45056 | 0.50242 | 0.38574 | 0.49085 | 0.45056 | 0.50242 | 0.38574 | 1.5030 | 1.5590 | 1.5395 | 1.5700 | 1.5609 | 1.5719 | 1.5815 | 1.5499 | 1.5877 | 1.5353 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 2.4178 | 2.3929 | 2.5211 | 2.4204 | 2.3426 | 2.4876 | 0.78232 | 0.71425 | 0.79946 | 0.61992 | 0.78232 | 0.71425 | 0.79946 | 0.61992 | 2.4248 | 2.4728 | 2.4545 | 2.5014 | 2.4889 | 2.5026 | 2.5414 | 2.4796 | 2.5352 | 2.4536 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.11079 | 0.11652 | 0.09360 | 0.10220 | 0.11748 | 0.08787 | 0.77650 | 0.77555 | 0.75072 | 0.79083 | 0.77650 | 0.77555 | 0.75072 | 0.79083 | 0.10124 | 0.08787 | 0.07832 | 0.08405 | 0.09169 | 0.09169 | 0.08214 | 0.08787 | 0.08118 | 0.09074 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.11079 | 0.11652 | 0.09360 | 0.10220 | 0.11748 | 0.08787 | 0.77650 | 0.77555 | 0.75072 | 0.79083 | 0.77650 | 0.77555 | 0.75072 | 0.79083 | 0.10124 | 0.08787 | 0.07832 | 0.08405 | 0.09169 | 0.09169 | 0.08214 | 0.08787 | 0.08118 | 0.09074 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -2.6728 | -2.6387 | -2.7467 | -2.7442 | -2.5872 | -2.8275 | -0.78717 | -0.74648 | -0.79550 | -0.64098 | -0.78717 | -0.74648 | -0.79550 | -0.64098 | -2.5554 | -2.8624 | -2.7259 | -2.7806 | -2.7562 | -2.7957 | -2.7401 | -2.7134 | -2.8285 | -2.7794 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -2.6728 | -2.6387 | -2.7467 | -2.7442 | -2.5872 | -2.8275 | -0.78717 | -0.74648 | -0.79550 | -0.64098 | -0.78717 | -0.74648 | -0.79550 | -0.64098 | -2.5554 | -2.8624 | -2.7259 | -2.7806 | -2.7562 | -2.7957 | -2.7401 | -2.7134 | -2.8285 | -2.7794 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.10213 | 0.11024 | 0.10236 | 0.10096 | 0.12152 | 0.09673 | 0.70340 | 0.71496 | 0.70982 | 0.74671 | 0.70340 | 0.71496 | 0.70982 | 0.74671 | 0.10534 | 0.10295 | 0.09648 | 0.09889 | 0.09675 | 0.09954 | 0.09699 | 0.09794 | 0.09560 | 0.10054 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.10213 | 0.11024 | 0.10236 | 0.10096 | 0.12152 | 0.09673 | 0.70340 | 0.71496 | 0.70982 | 0.74671 | 0.70340 | 0.71496 | 0.70982 | 0.74671 | 0.10534 | 0.10295 | 0.09648 | 0.09889 | 0.09675 | 0.09954 | 0.09699 | 0.09794 | 0.09560 | 0.10054 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.60972 | 0.66199 | 0.60117 | 0.61758 | 0.63758 | 0.60452 | 0.42801 | 0.36288 | 0.36813 | 0.41394 | 0.42801 | 0.36288 | 0.36813 | 0.41394 | 0.66897 | 0.60347 | 0.64842 | 0.63853 | 0.59524 | 0.64946 | 0.60144 | 0.61530 | 0.63058 | 0.67099 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.66579 | 0.72157 | 0.65613 | 0.67375 | 0.69611 | 0.65988 | 0.46660 | 0.39531 | 0.40115 | 0.45103 | 0.46660 | 0.39531 | 0.40115 | 0.45103 | 0.72910 | 0.65770 | 0.70620 | 0.69435 | 0.64802 | 0.70682 | 0.65575 | 0.66964 | 0.69016 | 0.73321 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.42262 | 0.45885 | 0.41668 | 0.42805 | 0.44190 | 0.41902 | 0.29669 | 0.25150 | 0.25519 | 0.28697 | 0.29669 | 0.25150 | 0.25519 | 0.28697 | 0.46366 | 0.41827 | 0.44948 | 0.44259 | 0.41260 | 0.45015 | 0.41683 | 0.42650 | 0.43707 | 0.46507 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.46151 | 0.50017 | 0.45488 | 0.46697 | 0.48253 | 0.45736 | 0.32344 | 0.27401 | 0.27806 | 0.31263 | 0.32344 | 0.27401 | 0.27806 | 0.31263 | 0.50539 | 0.45587 | 0.48955 | 0.48131 | 0.44919 | 0.48990 | 0.45453 | 0.46415 | 0.47837 | 0.50824 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.39229 | 0.37648 | 0.40810 | 0.39526 | 0.42292 | 0.41403 | 0.71245 | 0.73123 | 0.73221 | 0.67589 | 0.71245 | 0.73123 | 0.73221 | 0.67589 | 0.37648 | 0.39625 | 0.39526 | 0.41008 | 0.41897 | 0.39328 | 0.38241 | 0.38735 | 0.38241 | 0.39032 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.39229 | 0.37648 | 0.40810 | 0.39526 | 0.42292 | 0.41403 | 0.71245 | 0.73123 | 0.73221 | 0.67589 | 0.71245 | 0.73123 | 0.73221 | 0.67589 | 0.37648 | 0.39625 | 0.39526 | 0.41008 | 0.41897 | 0.39328 | 0.38241 | 0.38735 | 0.38241 | 0.39032 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -3.8545 | -4.2343 | -3.8433 | -3.9267 | -3.9873 | -3.9074 | -1.1560 | -0.95967 | -0.92786 | -1.2174 | -1.1560 | -0.95967 | -0.92786 | -1.2174 | -4.4652 | -4.0659 | -4.3080 | -4.1737 | -4.0085 | -4.1696 | -3.9530 | -4.4501 | -4.1058 | -4.2666 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -3.8545 | -4.2343 | -3.8433 | -3.9267 | -3.9873 | -3.9074 | -1.1560 | -0.95967 | -0.92786 | -1.2174 | -1.1560 | -0.95967 | -0.92786 | -1.2174 | -4.4652 | -4.0659 | -4.3080 | -4.1737 | -4.0085 | -4.1696 | -3.9530 | -4.4501 | -4.1058 | -4.2666 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.38898 | 0.37301 | 0.39264 | 0.38295 | 0.39727 | 0.38980 | 0.69753 | 0.71747 | 0.72129 | 0.67513 | 0.69753 | 0.71747 | 0.72129 | 0.67513 | 0.36732 | 0.38657 | 0.37535 | 0.38748 | 0.39795 | 0.37861 | 0.37553 | 0.37540 | 0.37637 | 0.37395 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.38898 | 0.37301 | 0.39264 | 0.38295 | 0.39727 | 0.38980 | 0.69753 | 0.71747 | 0.72129 | 0.67513 | 0.69753 | 0.71747 | 0.72129 | 0.67513 | 0.36732 | 0.38657 | 0.37535 | 0.38748 | 0.39795 | 0.37861 | 0.37553 | 0.37540 | 0.37637 | 0.37395 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.80818 | 0.83025 | 0.82645 | 0.85502 | 0.74404 | 0.84809 | 0.45591 | 0.52145 | 0.50044 | 0.56032 | 0.45591 | 0.52145 | 0.50044 | 0.56032 | 0.76888 | 0.83415 | 0.74114 | 0.68431 | 0.76898 | 0.72103 | 0.75899 | 0.84508 | 0.82838 | 0.85269 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.96966 | 0.99776 | 0.99316 | 1.0267 | 0.89567 | 1.0183 | 0.54775 | 0.62514 | 0.60055 | 0.67130 | 0.54775 | 0.62514 | 0.60055 | 0.67130 | 0.92459 | 1.0022 | 0.88990 | 0.82096 | 0.92332 | 0.86423 | 0.91068 | 1.0146 | 0.99233 | 1.0250 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.56040 | 0.57576 | 0.57306 | 0.59283 | 0.51583 | 0.58810 | 0.31612 | 0.36162 | 0.34710 | 0.38849 | 0.31612 | 0.36162 | 0.34710 | 0.38849 | 0.53325 | 0.57842 | 0.51387 | 0.47454 | 0.53328 | 0.49997 | 0.52625 | 0.58597 | 0.57435 | 0.59119 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.67223 | 0.69186 | 0.68867 | 0.71181 | 0.62112 | 0.70617 | 0.37975 | 0.43355 | 0.41659 | 0.46550 | 0.37975 | 0.43355 | 0.41659 | 0.46550 | 0.64112 | 0.69507 | 0.61708 | 0.56930 | 0.64030 | 0.59932 | 0.63141 | 0.70362 | 0.68812 | 0.71072 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.65767 | 0.65863 | 0.65477 | 0.67599 | 0.69720 | 0.65670 | 0.79942 | 0.79460 | 0.80906 | 0.80231 | 0.79942 | 0.79460 | 0.80906 | 0.80231 | 0.69045 | 0.66924 | 0.67695 | 0.69817 | 0.67213 | 0.68660 | 0.68949 | 0.66056 | 0.66635 | 0.66345 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.65767 | 0.65863 | 0.65477 | 0.67599 | 0.69720 | 0.65670 | 0.79942 | 0.79460 | 0.80906 | 0.80231 | 0.79942 | 0.79460 | 0.80906 | 0.80231 | 0.69045 | 0.66924 | 0.67695 | 0.69817 | 0.67213 | 0.68660 | 0.68949 | 0.66056 | 0.66635 | 0.66345 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.84035 | -0.86614 | -0.89853 | -0.86189 | -0.79510 | -0.92149 | -0.53294 | -0.57022 | -0.52961 | -0.53063 | -0.53294 | -0.57022 | -0.52961 | -0.53063 | -0.82925 | -0.88401 | -0.83709 | -0.80943 | -0.86008 | -0.81595 | -0.81958 | -0.92399 | -0.95079 | -0.88017 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.84035 | -0.86614 | -0.89853 | -0.86189 | -0.79510 | -0.92149 | -0.53294 | -0.57022 | -0.52961 | -0.53063 | -0.53294 | -0.57022 | -0.52961 | -0.53063 | -0.82925 | -0.88401 | -0.83709 | -0.80943 | -0.86008 | -0.81595 | -0.81958 | -0.92399 | -0.95079 | -0.88017 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.57175 | 0.57090 | 0.54993 | 0.58227 | 0.60321 | 0.55275 | 0.71715 | 0.68596 | 0.70690 | 0.70220 | 0.71715 | 0.68596 | 0.70690 | 0.70220 | 0.59062 | 0.57029 | 0.58195 | 0.60389 | 0.57158 | 0.59543 | 0.59285 | 0.54909 | 0.55224 | 0.56478 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.57175 | 0.57090 | 0.54993 | 0.58227 | 0.60321 | 0.55275 | 0.71715 | 0.68596 | 0.70690 | 0.70220 | 0.71715 | 0.68596 | 0.70690 | 0.70220 | 0.59062 | 0.57029 | 0.58195 | 0.60389 | 0.57158 | 0.59543 | 0.59285 | 0.54909 | 0.55224 | 0.56478 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.29586 | 0.29090 | 0.29566 | 0.33103 | 0.32585 | 0.34456 | 0.28688 | 0.29722 | 0.28635 | 0.27329 | 0.28688 | 0.29722 | 0.28635 | 0.27329 | 0.30309 | 0.29921 | 0.30141 | 0.32441 | 0.32128 | 0.28332 | 0.32138 | 0.30598 | 0.33806 | 0.31048 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.30586 | 0.30070 | 0.30570 | 0.34212 | 0.33686 | 0.35606 | 0.29639 | 0.30718 | 0.29591 | 0.28236 | 0.29639 | 0.30718 | 0.29591 | 0.28236 | 0.31327 | 0.30923 | 0.31164 | 0.33531 | 0.33225 | 0.29292 | 0.33222 | 0.31652 | 0.34948 | 0.32085 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.20510 | 0.20166 | 0.20495 | 0.22947 | 0.22588 | 0.23885 | 0.19886 | 0.20603 | 0.19851 | 0.18945 | 0.19886 | 0.20603 | 0.19851 | 0.18945 | 0.21010 | 0.20739 | 0.20892 | 0.22489 | 0.22270 | 0.19640 | 0.22277 | 0.21210 | 0.23434 | 0.21522 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.21202 | 0.20845 | 0.21190 | 0.23716 | 0.23350 | 0.24683 | 0.20549 | 0.21293 | 0.20514 | 0.19573 | 0.20549 | 0.21293 | 0.20514 | 0.19573 | 0.21716 | 0.21436 | 0.21604 | 0.23243 | 0.23031 | 0.20305 | 0.23030 | 0.21940 | 0.24226 | 0.22241 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.81216 | 0.78175 | 0.77639 | 0.77818 | 0.77281 | 0.77013 | 0.83721 | 0.83721 | 0.80590 | 0.81395 | 0.83721 | 0.83721 | 0.80590 | 0.81395 | 0.79159 | 0.80411 | 0.79338 | 0.77818 | 0.79964 | 0.80680 | 0.80143 | 0.79606 | 0.76923 | 0.78712 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.81216 | 0.78175 | 0.77639 | 0.77818 | 0.77281 | 0.77013 | 0.83721 | 0.83721 | 0.80590 | 0.81395 | 0.83721 | 0.83721 | 0.80590 | 0.81395 | 0.79159 | 0.80411 | 0.79338 | 0.77818 | 0.79964 | 0.80680 | 0.80143 | 0.79606 | 0.76923 | 0.78712 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.55388 | -0.61964 | -0.62442 | -0.65585 | -0.64131 | -0.65863 | -0.41398 | -0.45501 | -0.51743 | -0.48103 | -0.41398 | -0.45501 | -0.51743 | -0.48103 | -0.62514 | -0.59324 | -0.58969 | -0.65038 | -0.51680 | -0.56931 | -0.59226 | -0.52805 | -0.70802 | -0.61066 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.55388 | -0.61964 | -0.62442 | -0.65585 | -0.64131 | -0.65863 | -0.41398 | -0.45501 | -0.51743 | -0.48103 | -0.41398 | -0.45501 | -0.51743 | -0.48103 | -0.62514 | -0.59324 | -0.58969 | -0.65038 | -0.51680 | -0.56931 | -0.59226 | -0.52805 | -0.70802 | -0.61066 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.79298 | 0.77254 | 0.76625 | 0.75127 | 0.77179 | 0.75726 | 0.82590 | 0.82881 | 0.81260 | 0.82368 | 0.82590 | 0.82881 | 0.81260 | 0.82368 | 0.77452 | 0.78854 | 0.78128 | 0.76878 | 0.79275 | 0.79307 | 0.78740 | 0.78049 | 0.75355 | 0.77719 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.79298 | 0.77254 | 0.76625 | 0.75127 | 0.77179 | 0.75726 | 0.82590 | 0.82881 | 0.81260 | 0.82368 | 0.82590 | 0.82881 | 0.81260 | 0.82368 | 0.77452 | 0.78854 | 0.78128 | 0.76878 | 0.79275 | 0.79307 | 0.78740 | 0.78049 | 0.75355 | 0.77719 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 1.3422 | 1.3337 | 1.2251 | 1.2522 | 1.3033 | 1.2994 | 1.1079 | 1.0774 | 1.0777 | 1.1267 | 1.1079 | 1.0774 | 1.0777 | 1.1267 | 1.3309 | 1.2710 | 1.3104 | 1.2899 | 1.3751 | 1.2535 | 1.2220 | 1.2574 | 1.3823 | 1.3289 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 2.1276 | 2.1108 | 1.9431 | 1.9844 | 2.0672 | 2.0500 | 1.7590 | 1.7259 | 1.7327 | 1.8180 | 1.7590 | 1.7259 | 1.7327 | 1.8180 | 2.0971 | 2.0079 | 2.0570 | 2.0248 | 2.1798 | 1.9968 | 1.9452 | 1.9948 | 2.1909 | 2.1166 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.96820 | 0.96770 | 0.88602 | 0.90789 | 0.95402 | 0.94252 | 0.80983 | 0.78066 | 0.78721 | 0.81873 | 0.80983 | 0.78066 | 0.78721 | 0.81873 | 0.95772 | 0.91740 | 0.94683 | 0.93332 | 0.99466 | 0.90772 | 0.88036 | 0.90507 | 1.0017 | 0.96454 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.5714 | 1.5740 | 1.4419 | 1.4786 | 1.5631 | 1.5282 | 1.3274 | 1.2830 | 1.3037 | 1.3565 | 1.3274 | 1.2830 | 1.3037 | 1.3565 | 1.5443 | 1.4855 | 1.5249 | 1.5043 | 1.6175 | 1.4837 | 1.4345 | 1.4683 | 1.6305 | 1.5789 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.55805 | 0.60300 | 0.59363 | 0.57865 | 0.56929 | 0.57678 | 0.65356 | 0.62547 | 0.65543 | 0.66667 | 0.65356 | 0.62547 | 0.65543 | 0.66667 | 0.54120 | 0.56180 | 0.55805 | 0.54869 | 0.54682 | 0.55431 | 0.57303 | 0.60487 | 0.55243 | 0.56367 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.55805 | 0.60300 | 0.59363 | 0.57865 | 0.56929 | 0.57678 | 0.65356 | 0.62547 | 0.65543 | 0.66667 | 0.65356 | 0.62547 | 0.65543 | 0.66667 | 0.54120 | 0.56180 | 0.55805 | 0.54869 | 0.54682 | 0.55431 | 0.57303 | 0.60487 | 0.55243 | 0.56367 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -1.1561 | -1.1188 | -1.1088 | -1.1267 | -1.1109 | -1.1283 | -0.99549 | -0.94568 | -0.92084 | -0.86213 | -0.99549 | -0.94568 | -0.92084 | -0.86213 | -1.1587 | -1.1500 | -1.1567 | -1.1789 | -1.1865 | -1.1316 | -1.1197 | -1.1197 | -1.2066 | -1.1820 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -1.1561 | -1.1188 | -1.1088 | -1.1267 | -1.1109 | -1.1283 | -0.99549 | -0.94568 | -0.92084 | -0.86213 | -0.99549 | -0.94568 | -0.92084 | -0.86213 | -1.1587 | -1.1500 | -1.1567 | -1.1789 | -1.1865 | -1.1316 | -1.1197 | -1.1197 | -1.2066 | -1.1820 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.49004 | 0.50456 | 0.50158 | 0.49698 | 0.50235 | 0.50139 | 0.57766 | 0.56627 | 0.57771 | 0.59092 | 0.57766 | 0.56627 | 0.57771 | 0.59092 | 0.48583 | 0.49471 | 0.48922 | 0.48659 | 0.48327 | 0.49049 | 0.50285 | 0.49865 | 0.47779 | 0.47912 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.49004 | 0.50456 | 0.50158 | 0.49698 | 0.50235 | 0.50139 | 0.57766 | 0.56627 | 0.57771 | 0.59092 | 0.57766 | 0.56627 | 0.57771 | 0.59092 | 0.48583 | 0.49471 | 0.48922 | 0.48659 | 0.48327 | 0.49049 | 0.50285 | 0.49865 | 0.47779 | 0.47912 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 2.4598 | 2.4655 | 2.3279 | 2.3670 | 2.3406 | 2.3617 | 1.6324 | 1.6057 | 1.5035 | 1.4928 | 1.6324 | 1.6057 | 1.5035 | 1.4928 | 2.3689 | 2.5470 | 2.4489 | 2.4004 | 2.3261 | 2.4392 | 2.2866 | 2.4654 | 2.4897 | 2.3158 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 3.3559 | 3.3825 | 3.1634 | 3.2554 | 3.2140 | 3.2370 | 2.2539 | 2.1960 | 2.0810 | 2.0582 | 2.2539 | 2.1960 | 2.0810 | 2.0582 | 3.2709 | 3.4881 | 3.3456 | 3.3032 | 3.1797 | 3.3645 | 3.1229 | 3.4245 | 3.4204 | 3.1759 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.7048 | 1.7088 | 1.6136 | 1.6408 | 1.6223 | 1.6371 | 1.1315 | 1.1130 | 1.0421 | 1.0347 | 1.1315 | 1.1130 | 1.0421 | 1.0347 | 1.6421 | 1.7656 | 1.6973 | 1.6638 | 1.6124 | 1.6907 | 1.5850 | 1.7089 | 1.7257 | 1.6051 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 2.3262 | 2.3445 | 2.1925 | 2.2564 | 2.2279 | 2.2437 | 1.5623 | 1.5221 | 1.4425 | 1.4268 | 1.5623 | 1.5221 | 1.4425 | 1.4268 | 2.2673 | 2.4179 | 2.3188 | 2.2898 | 2.2039 | 2.3318 | 2.1646 | 2.3737 | 2.3707 | 2.2014 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.22395 | 0.22313 | 0.21821 | 0.23052 | 0.22313 | 0.21493 | 0.30025 | 0.32075 | 0.32978 | 0.29532 | 0.30025 | 0.32075 | 0.32978 | 0.29532 | 0.23626 | 0.22888 | 0.23298 | 0.22806 | 0.24446 | 0.21821 | 0.22313 | 0.22642 | 0.22888 | 0.22642 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.22395 | 0.22313 | 0.21821 | 0.23052 | 0.22313 | 0.21493 | 0.30025 | 0.32075 | 0.32978 | 0.29532 | 0.30025 | 0.32075 | 0.32978 | 0.29532 | 0.23626 | 0.22888 | 0.23298 | 0.22806 | 0.24446 | 0.21821 | 0.22313 | 0.22642 | 0.22888 | 0.22642 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -4.7093 | -4.6543 | -4.6843 | -4.6424 | -4.6331 | -4.7850 | -3.6072 | -3.5834 | -3.5888 | -3.5875 | -3.6072 | -3.5834 | -3.5888 | -3.5875 | -4.7912 | -4.6988 | -4.6051 | -4.8596 | -4.6814 | -4.7852 | -4.6837 | -4.8879 | -4.9084 | -4.8016 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -4.7093 | -4.6543 | -4.6843 | -4.6424 | -4.6331 | -4.7850 | -3.6072 | -3.5834 | -3.5888 | -3.5875 | -3.6072 | -3.5834 | -3.5888 | -3.5875 | -4.7912 | -4.6988 | -4.6051 | -4.8596 | -4.6814 | -4.7852 | -4.6837 | -4.8879 | -4.9084 | -4.8016 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.22681 | 0.23723 | 0.22688 | 0.24070 | 0.23974 | 0.23752 | 0.30530 | 0.32348 | 0.32609 | 0.30050 | 0.30530 | 0.32348 | 0.32609 | 0.30050 | 0.24259 | 0.23835 | 0.23888 | 0.23535 | 0.24761 | 0.23284 | 0.23598 | 0.23985 | 0.23326 | 0.23421 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.22681 | 0.23723 | 0.22688 | 0.24070 | 0.23974 | 0.23752 | 0.30530 | 0.32348 | 0.32609 | 0.30050 | 0.30530 | 0.32348 | 0.32609 | 0.30050 | 0.24259 | 0.23835 | 0.23888 | 0.23535 | 0.24761 | 0.23284 | 0.23598 | 0.23985 | 0.23326 | 0.23421 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.60077 | 0.59314 | 0.60785 | 0.60889 | 0.60561 | 0.61780 | 0.41252 | 0.39462 | 0.39990 | 0.40951 | 0.41252 | 0.39462 | 0.39990 | 0.40951 | 0.62692 | 0.61821 | 0.61803 | 0.61314 | 0.63268 | 0.62440 | 0.62835 | 0.62577 | 0.64187 | 0.63685 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.60838 | 0.60043 | 0.61588 | 0.61648 | 0.61366 | 0.62554 | 0.41808 | 0.39976 | 0.40503 | 0.41495 | 0.41808 | 0.39976 | 0.40503 | 0.41495 | 0.63462 | 0.62639 | 0.62597 | 0.62100 | 0.64105 | 0.63232 | 0.63637 | 0.63325 | 0.65028 | 0.64525 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.83082 | 0.84798 | 0.84602 | 0.82850 | 0.81672 | 0.84093 | 0.66217 | 0.63536 | 0.64263 | 0.63706 | 0.66217 | 0.63536 | 0.64263 | 0.63706 | 0.85006 | 0.85897 | 0.85682 | 0.85344 | 0.84159 | 0.83871 | 0.84759 | 0.86157 | 0.86558 | 0.85441 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.83791 | 0.85514 | 0.85343 | 0.83565 | 0.82375 | 0.84825 | 0.66801 | 0.64087 | 0.64825 | 0.64276 | 0.66801 | 0.64087 | 0.64825 | 0.64276 | 0.85708 | 0.86624 | 0.86416 | 0.86100 | 0.84884 | 0.84595 | 0.85489 | 0.86915 | 0.87309 | 0.86180 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.9711 | 2.6515 | 3.0623 | 2.8986 | 2.9171 | 2.6422 | 1.9998 | 2.2870 | 2.1814 | 1.9198 | 1.9998 | 2.2870 | 2.1814 | 1.9198 | 2.7539 | 2.3312 | 2.7561 | 2.7269 | 2.6459 | 2.9007 | 2.9603 | 2.7273 | 2.4391 | 2.7178 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 5.9422 | 5.3030 | 6.1247 | 5.7971 | 5.8341 | 5.2845 | 3.9996 | 4.5740 | 4.3628 | 3.8396 | 3.9996 | 4.5740 | 4.3628 | 3.8396 | 5.5078 | 4.6625 | 5.5121 | 5.4538 | 5.2918 | 5.8013 | 5.9205 | 5.4546 | 4.8781 | 5.4355 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 2.0600 | 1.8378 | 2.1220 | 2.0086 | 2.0220 | 1.8318 | 1.3861 | 1.5854 | 1.5118 | 1.3303 | 1.3861 | 1.5854 | 1.5118 | 1.3303 | 1.9085 | 1.6159 | 1.9101 | 1.8909 | 1.8339 | 2.0108 | 2.0521 | 1.8902 | 1.6905 | 1.8843 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 4.1200 | 3.6756 | 4.2439 | 4.0173 | 4.0441 | 3.6635 | 2.7721 | 3.1707 | 3.0237 | 2.6605 | 2.7721 | 3.1707 | 3.0237 | 2.6605 | 3.8170 | 3.2319 | 3.8202 | 3.7818 | 3.6677 | 4.0217 | 4.1041 | 3.7804 | 3.3811 | 3.7686 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.08000 | 0.09000 | 0.08000 | 0.09000 | 0.07000 | 0.06000 | 0.10000 | 0.10000 | 0.10000 | 0.09000 | 0.10000 | 0.10000 | 0.10000 | 0.09000 | 0.08000 | 0.05000 | 0.07000 | 0.07000 | 0.06000 | 0.07000 | 0.07000 | 0.07000 | 0.07000 | 0.11000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.08000 | 0.09000 | 0.08000 | 0.09000 | 0.07000 | 0.06000 | 0.10000 | 0.10000 | 0.10000 | 0.09000 | 0.10000 | 0.10000 | 0.10000 | 0.09000 | 0.08000 | 0.05000 | 0.07000 | 0.07000 | 0.06000 | 0.07000 | 0.07000 | 0.07000 | 0.07000 | 0.11000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -4.1057 | -3.6590 | -4.2313 | -3.9968 | -4.0318 | -3.6523 | -2.7536 | -3.1645 | -3.0153 | -2.6477 | -2.7536 | -3.1645 | -3.0153 | -2.6477 | -3.8079 | -3.2154 | -3.8114 | -3.7671 | -3.6595 | -4.0015 | -4.0982 | -3.7623 | -3.3763 | -3.7513 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -4.1057 | -3.6590 | -4.2313 | -3.9968 | -4.0318 | -3.6523 | -2.7536 | -3.1645 | -3.0153 | -2.6477 | -2.7536 | -3.1645 | -3.0153 | -2.6477 | -3.8079 | -3.2154 | -3.8114 | -3.7671 | -3.6595 | -4.0015 | -4.0982 | -3.7623 | -3.3763 | -3.7513 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09145 | 0.09160 | 0.09075 | 0.09210 | 0.08559 | 0.09180 | 0.09728 | 0.09619 | 0.09935 | 0.09730 | 0.09728 | 0.09619 | 0.09935 | 0.09730 | 0.09230 | 0.09529 | 0.09531 | 0.08902 | 0.09351 | 0.08578 | 0.09155 | 0.08974 | 0.08921 | 0.09518 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09145 | 0.09160 | 0.09075 | 0.09210 | 0.08559 | 0.09180 | 0.09728 | 0.09619 | 0.09935 | 0.09730 | 0.09728 | 0.09619 | 0.09935 | 0.09730 | 0.09230 | 0.09529 | 0.09531 | 0.08902 | 0.09351 | 0.08578 | 0.09155 | 0.08974 | 0.08921 | 0.09518 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.90541 | 0.90835 | 0.90469 | 0.90494 | 0.90484 | 0.90772 | 0.87845 | 0.86836 | 0.86716 | 0.87231 | 0.87845 | 0.86836 | 0.86716 | 0.87231 | 0.91635 | 0.90841 | 0.90945 | 0.90874 | 0.90343 | 0.90608 | 0.90441 | 0.90812 | 0.91388 | 0.91600 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.91537 | 0.91840 | 0.91478 | 0.91488 | 0.91468 | 0.91760 | 0.88806 | 0.87793 | 0.87660 | 0.88181 | 0.88806 | 0.87793 | 0.87660 | 0.88181 | 0.92629 | 0.91837 | 0.91942 | 0.91857 | 0.91328 | 0.91610 | 0.91455 | 0.91798 | 0.92386 | 0.92616 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.88614 | 0.89680 | 0.89062 | 0.88929 | 0.88334 | 0.89039 | 0.55648 | 0.55037 | 0.53742 | 0.54168 | 0.55648 | 0.55037 | 0.53742 | 0.54168 | 0.90153 | 0.89347 | 0.88812 | 0.88636 | 0.89096 | 0.88646 | 0.87861 | 0.89850 | 0.90610 | 0.90051 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.88904 | 0.89978 | 0.89348 | 0.89229 | 0.88627 | 0.89323 | 0.55831 | 0.55216 | 0.53911 | 0.54345 | 0.55831 | 0.55216 | 0.53911 | 0.54345 | 0.90463 | 0.89661 | 0.89119 | 0.88924 | 0.89379 | 0.88944 | 0.88152 | 0.90154 | 0.90901 | 0.90355 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.88090 | 0.89339 | 0.89238 | 0.88295 | 0.89164 | 0.90395 | 0.82352 | 0.78296 | 0.79651 | 0.79550 | 0.82352 | 0.78296 | 0.79651 | 0.79550 | 0.90738 | 0.91282 | 0.90146 | 0.89551 | 0.90152 | 0.88581 | 0.88042 | 0.91035 | 0.90778 | 0.91465 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.92842 | 0.94143 | 0.94031 | 0.93038 | 0.93960 | 0.95216 | 0.86566 | 0.82267 | 0.83718 | 0.83570 | 0.86566 | 0.82267 | 0.83718 | 0.83570 | 0.95658 | 0.96199 | 0.94980 | 0.94364 | 0.95003 | 0.93337 | 0.92800 | 0.95919 | 0.95694 | 0.96406 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0281 | 1.0468 | 1.0683 | 1.0312 | 1.0521 | 1.0435 | 1.0151 | 1.0140 | 1.0105 | 1.0037 | 1.0151 | 1.0140 | 1.0105 | 1.0037 | 1.0429 | 1.0380 | 1.0317 | 1.0413 | 1.0431 | 1.0456 | 1.0410 | 1.0515 | 1.0456 | 1.0481 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0561 | 2.0936 | 2.1366 | 2.0624 | 2.1041 | 2.0870 | 2.0301 | 2.0279 | 2.0210 | 2.0073 | 2.0301 | 2.0279 | 2.0210 | 2.0073 | 2.0858 | 2.0760 | 2.0634 | 2.0826 | 2.0862 | 2.0912 | 2.0821 | 2.1030 | 2.0912 | 2.0962 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.71261 | 0.72562 | 0.74047 | 0.71484 | 0.72927 | 0.72329 | 0.70366 | 0.70285 | 0.70046 | 0.69574 | 0.70366 | 0.70285 | 0.70046 | 0.69574 | 0.72291 | 0.71955 | 0.71519 | 0.72181 | 0.72305 | 0.72478 | 0.72159 | 0.72889 | 0.72474 | 0.72648 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4252 | 1.4512 | 1.4809 | 1.4297 | 1.4585 | 1.4466 | 1.4073 | 1.4057 | 1.4009 | 1.3915 | 1.4073 | 1.4057 | 1.4009 | 1.3915 | 1.4458 | 1.4391 | 1.4304 | 1.4436 | 1.4461 | 1.4496 | 1.4432 | 1.4578 | 1.4495 | 1.4530 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.25441 | 0.24740 | 0.24718 | 0.26971 | 0.24612 | 0.25356 | 0.29692 | 0.30372 | 0.30840 | 0.30584 | 0.29692 | 0.30372 | 0.30840 | 0.30584 | 0.24867 | 0.25781 | 0.25292 | 0.25994 | 0.25441 | 0.24570 | 0.24655 | 0.25165 | 0.24463 | 0.25972 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.25441 | 0.24740 | 0.24718 | 0.26971 | 0.24612 | 0.25356 | 0.29692 | 0.30372 | 0.30840 | 0.30584 | 0.29692 | 0.30372 | 0.30840 | 0.30584 | 0.24867 | 0.25781 | 0.25292 | 0.25994 | 0.25441 | 0.24570 | 0.24655 | 0.25165 | 0.24463 | 0.25972 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3895 | -1.3971 | -1.4015 | -1.3878 | -1.3932 | -1.3940 | -1.3703 | -1.3651 | -1.3659 | -1.3620 | -1.3703 | -1.3651 | -1.3659 | -1.3620 | -1.3955 | -1.3910 | -1.3904 | -1.3933 | -1.3948 | -1.3972 | -1.3965 | -1.3945 | -1.3941 | -1.3964 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.4077 | -1.4296 | -1.4534 | -1.4072 | -1.4195 | -1.4264 | -1.4028 | -1.4018 | -1.3968 | -1.3885 | -1.4028 | -1.4018 | -1.3968 | -1.3885 | -1.4245 | -1.4121 | -1.4086 | -1.4239 | -1.4260 | -1.4295 | -1.4287 | -1.4203 | -1.4206 | -1.4330 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.25105 | 0.25014 | 0.25086 | 0.25193 | 0.25071 | 0.25124 | 0.26109 | 0.26375 | 0.26213 | 0.26367 | 0.26109 | 0.26375 | 0.26213 | 0.26367 | 0.25019 | 0.25091 | 0.25081 | 0.25134 | 0.25076 | 0.25005 | 0.25040 | 0.25015 | 0.25042 | 0.25086 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.25196 | 0.25015 | 0.25142 | 0.25380 | 0.25121 | 0.25240 | 0.27030 | 0.27511 | 0.27282 | 0.27513 | 0.27030 | 0.27511 | 0.27282 | 0.27513 | 0.25044 | 0.25173 | 0.25166 | 0.25266 | 0.25151 | 0.25005 | 0.25074 | 0.25029 | 0.25068 | 0.25153 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 1.2359 | 1.2494 | 1.2545 | 1.2447 | 1.2551 | 1.2514 | 1.1418 | 1.0669 | 1.0977 | 1.0900 | 1.1418 | 1.0669 | 1.0977 | 1.0900 | 1.2827 | 1.2552 | 1.2464 | 1.2721 | 1.2658 | 1.2459 | 1.2353 | 1.2672 | 1.2693 | 1.2696 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.3763 | 1.3899 | 1.3971 | 1.3854 | 1.3967 | 1.3930 | 1.2688 | 1.1859 | 1.2207 | 1.2095 | 1.2688 | 1.1859 | 1.2207 | 1.2095 | 1.4277 | 1.3991 | 1.3864 | 1.4190 | 1.4113 | 1.3880 | 1.3741 | 1.4117 | 1.4145 | 1.4146 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0486 | 1.0377 | 1.0527 | 1.0556 | 1.0485 | 1.0535 | 1.0692 | 1.0191 | 0.98980 | 1.0121 | 1.0692 | 1.0191 | 0.98980 | 1.0121 | 1.0547 | 1.0317 | 1.0474 | 1.0385 | 1.0355 | 1.0335 | 1.0369 | 1.0363 | 1.0490 | 1.0554 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0972 | 2.0755 | 2.1055 | 2.1111 | 2.0971 | 2.1069 | 2.1384 | 2.0382 | 1.9796 | 2.0241 | 2.1384 | 2.0382 | 1.9796 | 2.0241 | 2.1094 | 2.0633 | 2.0947 | 2.0771 | 2.0711 | 2.0669 | 2.0738 | 2.0726 | 2.0980 | 2.1109 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.72686 | 0.71932 | 0.72977 | 0.73168 | 0.72679 | 0.73019 | 0.74111 | 0.70637 | 0.68608 | 0.70151 | 0.74111 | 0.70637 | 0.68608 | 0.70151 | 0.73108 | 0.71512 | 0.72601 | 0.71984 | 0.71783 | 0.71642 | 0.71873 | 0.71842 | 0.72714 | 0.73162 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4537 | 1.4386 | 1.4595 | 1.4634 | 1.4536 | 1.4604 | 1.4822 | 1.4127 | 1.3722 | 1.4030 | 1.4822 | 1.4127 | 1.3722 | 1.4030 | 1.4622 | 1.4302 | 1.4520 | 1.4397 | 1.4357 | 1.4328 | 1.4375 | 1.4368 | 1.4543 | 1.4632 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24183 | 0.25293 | 0.25170 | 0.26619 | 0.28408 | 0.26373 | 0.30444 | 0.33498 | 0.34392 | 0.33159 | 0.30444 | 0.33498 | 0.34392 | 0.33159 | 0.25694 | 0.27637 | 0.24892 | 0.27267 | 0.26527 | 0.28532 | 0.27051 | 0.27205 | 0.25447 | 0.25941 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24183 | 0.25293 | 0.25170 | 0.26619 | 0.28408 | 0.26373 | 0.30444 | 0.33498 | 0.34392 | 0.33159 | 0.30444 | 0.33498 | 0.34392 | 0.33159 | 0.25694 | 0.27637 | 0.24892 | 0.27267 | 0.26527 | 0.28532 | 0.27051 | 0.27205 | 0.25447 | 0.25941 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3977 | -1.3895 | -1.3955 | -1.3980 | -1.3869 | -1.3948 | -1.3714 | -1.3408 | -1.3354 | -1.3423 | -1.3714 | -1.3408 | -1.3354 | -1.3423 | -1.3942 | -1.3831 | -1.3950 | -1.3904 | -1.3895 | -1.3857 | -1.3885 | -1.3832 | -1.3931 | -1.3938 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.4313 | -1.4099 | -1.4252 | -1.4419 | -1.4116 | -1.4334 | -1.4793 | -1.4107 | -1.3684 | -1.4002 | -1.4793 | -1.4107 | -1.3684 | -1.4002 | -1.4368 | -1.3967 | -1.4254 | -1.4171 | -1.4128 | -1.4016 | -1.4222 | -1.3974 | -1.4249 | -1.4399 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.24985 | 0.25131 | 0.25032 | 0.25134 | 0.25315 | 0.25175 | 0.27200 | 0.27897 | 0.27525 | 0.27650 | 0.27200 | 0.27897 | 0.27525 | 0.27650 | 0.25256 | 0.25294 | 0.25047 | 0.25202 | 0.25181 | 0.25235 | 0.25379 | 0.25306 | 0.25158 | 0.25315 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.24951 | 0.25244 | 0.25057 | 0.25273 | 0.25645 | 0.25351 | 0.28652 | 0.30023 | 0.29589 | 0.29628 | 0.28652 | 0.30023 | 0.29589 | 0.29628 | 0.25510 | 0.25583 | 0.25070 | 0.25410 | 0.25362 | 0.25472 | 0.25757 | 0.25619 | 0.25302 | 0.25624 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 1.0468 | 1.0515 | 1.0647 | 1.0501 | 1.0530 | 1.0546 | 0.96177 | 0.89880 | 0.91745 | 0.92155 | 0.96177 | 0.89880 | 0.91745 | 0.92155 | 1.0839 | 1.0519 | 1.0549 | 1.0629 | 1.0595 | 1.0506 | 1.0461 | 1.0736 | 1.0768 | 1.0630 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 1.1197 | 1.1245 | 1.1384 | 1.1231 | 1.1263 | 1.1272 | 1.0272 | 0.95927 | 0.97894 | 0.98270 | 1.0272 | 0.95927 | 0.97894 | 0.98270 | 1.1596 | 1.1251 | 1.1277 | 1.1371 | 1.1332 | 1.1245 | 1.1194 | 1.1486 | 1.1525 | 1.1373 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0565 | 1.0602 | 1.0569 | 1.0503 | 1.0461 | 1.0790 | 1.0903 | 1.0384 | 1.0264 | 1.0563 | 1.0903 | 1.0384 | 1.0264 | 1.0563 | 1.0365 | 1.0353 | 1.0291 | 1.0403 | 1.0450 | 1.0326 | 1.0314 | 1.0425 | 1.0340 | 1.0388 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.1131 | 2.1204 | 2.1138 | 2.1006 | 2.0922 | 2.1579 | 2.1806 | 2.0767 | 2.0527 | 2.1126 | 2.1806 | 2.0767 | 2.0527 | 2.1126 | 2.0729 | 2.0707 | 2.0581 | 2.0806 | 2.0900 | 2.0653 | 2.0627 | 2.0850 | 2.0680 | 2.0776 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.73236 | 0.73487 | 0.73260 | 0.72798 | 0.72509 | 0.74785 | 0.75573 | 0.71973 | 0.71143 | 0.73218 | 0.75573 | 0.71973 | 0.71143 | 0.73218 | 0.71846 | 0.71772 | 0.71331 | 0.72112 | 0.72436 | 0.71581 | 0.71492 | 0.72260 | 0.71671 | 0.72006 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4647 | 1.4697 | 1.4652 | 1.4560 | 1.4502 | 1.4957 | 1.5115 | 1.4395 | 1.4229 | 1.4644 | 1.5115 | 1.4395 | 1.4229 | 1.4644 | 1.4369 | 1.4354 | 1.4266 | 1.4422 | 1.4487 | 1.4316 | 1.4298 | 1.4452 | 1.4334 | 1.4401 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.22099 | 0.21742 | 0.23399 | 0.24699 | 0.28632 | 0.23334 | 0.29802 | 0.32792 | 0.33019 | 0.32272 | 0.29802 | 0.32792 | 0.33019 | 0.32272 | 0.30387 | 0.25057 | 0.27397 | 0.25382 | 0.25382 | 0.26422 | 0.27982 | 0.25804 | 0.28892 | 0.32207 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.22099 | 0.21742 | 0.23399 | 0.24699 | 0.28632 | 0.23334 | 0.29802 | 0.32792 | 0.33019 | 0.32272 | 0.29802 | 0.32792 | 0.33019 | 0.32272 | 0.30387 | 0.25057 | 0.27397 | 0.25382 | 0.25382 | 0.26422 | 0.27982 | 0.25804 | 0.28892 | 0.32207 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.4049 | -1.4058 | -1.4014 | -1.3941 | -1.3841 | -1.4087 | -1.3944 | -1.3612 | -1.3623 | -1.3677 | -1.3944 | -1.3612 | -1.3623 | -1.3677 | -1.3815 | -1.3875 | -1.3820 | -1.3920 | -1.3914 | -1.3864 | -1.3855 | -1.3891 | -1.3812 | -1.3774 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.4451 | -1.4413 | -1.4359 | -1.4350 | -1.4122 | -1.4697 | -1.5086 | -1.4373 | -1.4192 | -1.4615 | -1.5086 | -1.4373 | -1.4192 | -1.4615 | -1.4091 | -1.4028 | -1.4021 | -1.4221 | -1.4253 | -1.4035 | -1.4124 | -1.4084 | -1.4000 | -1.4175 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.24807 | 0.24713 | 0.24875 | 0.25223 | 0.25463 | 0.24911 | 0.26412 | 0.27150 | 0.26815 | 0.27169 | 0.26412 | 0.27150 | 0.26815 | 0.27169 | 0.25558 | 0.25134 | 0.25419 | 0.25162 | 0.25248 | 0.25212 | 0.25387 | 0.25138 | 0.25446 | 0.25900 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.24612 | 0.24428 | 0.24750 | 0.25378 | 0.25896 | 0.24771 | 0.27504 | 0.28865 | 0.28331 | 0.28766 | 0.27504 | 0.28865 | 0.28331 | 0.28766 | 0.26129 | 0.25244 | 0.25822 | 0.25277 | 0.25444 | 0.25415 | 0.25737 | 0.25267 | 0.25875 | 0.26802 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.5613 | 1.5659 | 1.5607 | 1.5748 | 1.5773 | 1.5801 | 1.2881 | 1.2301 | 1.2345 | 1.2375 | 1.2881 | 1.2301 | 1.2345 | 1.2375 | 1.6311 | 1.5934 | 1.5851 | 1.5921 | 1.5971 | 1.5670 | 1.5753 | 1.5984 | 1.6051 | 1.5944 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.9404 | 1.9458 | 1.9308 | 1.9555 | 1.9564 | 1.9627 | 1.5865 | 1.5202 | 1.5239 | 1.5253 | 1.5865 | 1.5202 | 1.5239 | 1.5253 | 2.0302 | 1.9795 | 1.9725 | 1.9777 | 1.9834 | 1.9441 | 1.9547 | 1.9894 | 1.9942 | 1.9802 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0710 | 1.0480 | 1.0666 | 1.0571 | 1.0648 | 1.0651 | 1.0552 | 1.0378 | 1.0118 | 1.0235 | 1.0552 | 1.0378 | 1.0118 | 1.0235 | 1.0478 | 1.0543 | 1.0400 | 1.0466 | 1.0441 | 1.0414 | 1.0519 | 1.0520 | 1.0599 | 1.0662 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.1420 | 2.0960 | 2.1333 | 2.1141 | 2.1296 | 2.1301 | 2.1105 | 2.0756 | 2.0236 | 2.0470 | 2.1105 | 2.0756 | 2.0236 | 2.0470 | 2.0957 | 2.1086 | 2.0800 | 2.0931 | 2.0883 | 2.0828 | 2.1037 | 2.1040 | 2.1198 | 2.1323 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.74236 | 0.72644 | 0.73937 | 0.73274 | 0.73808 | 0.73831 | 0.73145 | 0.71932 | 0.70134 | 0.70949 | 0.73145 | 0.71932 | 0.70134 | 0.70949 | 0.72632 | 0.73085 | 0.72086 | 0.72544 | 0.72375 | 0.72185 | 0.72906 | 0.72929 | 0.73469 | 0.73898 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4847 | 1.4529 | 1.4787 | 1.4655 | 1.4762 | 1.4766 | 1.4629 | 1.4386 | 1.4027 | 1.4190 | 1.4629 | 1.4386 | 1.4027 | 1.4190 | 1.4526 | 1.4617 | 1.4417 | 1.4509 | 1.4475 | 1.4437 | 1.4581 | 1.4586 | 1.4694 | 1.4780 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.22531 | 0.24586 | 0.23492 | 0.26541 | 0.27700 | 0.26110 | 0.27767 | 0.29125 | 0.30020 | 0.29920 | 0.27767 | 0.29125 | 0.30020 | 0.29920 | 0.28330 | 0.25878 | 0.26276 | 0.25414 | 0.26905 | 0.27303 | 0.28595 | 0.28297 | 0.25679 | 0.27005 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.22531 | 0.24586 | 0.23492 | 0.26541 | 0.27700 | 0.26110 | 0.27767 | 0.29125 | 0.30020 | 0.29920 | 0.27767 | 0.29125 | 0.30020 | 0.29920 | 0.28330 | 0.25878 | 0.26276 | 0.25414 | 0.26905 | 0.27303 | 0.28595 | 0.28297 | 0.25679 | 0.27005 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.4103 | -1.3935 | -1.4009 | -1.3923 | -1.3901 | -1.3999 | -1.3959 | -1.3861 | -1.3745 | -1.3776 | -1.3959 | -1.3861 | -1.3745 | -1.3776 | -1.3846 | -1.3967 | -1.3883 | -1.3934 | -1.3883 | -1.3874 | -1.3894 | -1.3837 | -1.3939 | -1.3909 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4608 | -1.4210 | -1.4408 | -1.4438 | -1.4283 | -1.4500 | -1.4585 | -1.4346 | -1.3979 | -1.4156 | -1.4585 | -1.4346 | -1.3979 | -1.4156 | -1.4202 | -1.4279 | -1.4135 | -1.4281 | -1.4246 | -1.4162 | -1.4424 | -1.4111 | -1.4382 | -1.4465 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.24721 | 0.25077 | 0.24964 | 0.25416 | 0.25365 | 0.25122 | 0.25535 | 0.25687 | 0.25771 | 0.25882 | 0.25535 | 0.25687 | 0.25771 | 0.25882 | 0.25544 | 0.25013 | 0.25241 | 0.25173 | 0.25409 | 0.25345 | 0.25628 | 0.25462 | 0.25274 | 0.25567 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.24427 | 0.25142 | 0.24924 | 0.25755 | 0.25700 | 0.25225 | 0.26074 | 0.26331 | 0.26518 | 0.26733 | 0.26074 | 0.26331 | 0.26518 | 0.26733 | 0.26059 | 0.25039 | 0.25471 | 0.25313 | 0.25795 | 0.25663 | 0.26192 | 0.25921 | 0.25448 | 0.26062 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.57541 | 0.57883 | 0.56726 | 0.55471 | 0.56397 | 0.56191 | 0.43233 | 0.41602 | 0.40476 | 0.41797 | 0.43233 | 0.41602 | 0.40476 | 0.41797 | 0.57475 | 0.58181 | 0.57165 | 0.55103 | 0.57056 | 0.55087 | 0.55710 | 0.57705 | 0.58585 | 0.56963 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.57856 | 0.58190 | 0.57038 | 0.55781 | 0.56716 | 0.56493 | 0.43482 | 0.41840 | 0.40708 | 0.42028 | 0.43482 | 0.41840 | 0.40708 | 0.42028 | 0.57800 | 0.58507 | 0.57477 | 0.55403 | 0.57366 | 0.55387 | 0.56027 | 0.58018 | 0.58898 | 0.57280 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.43087 | 0.42431 | 0.41766 | 0.42807 | 0.43643 | 0.42631 | 0.32565 | 0.31670 | 0.31553 | 0.33110 | 0.32565 | 0.31670 | 0.31553 | 0.33110 | 0.45154 | 0.44528 | 0.44252 | 0.43417 | 0.42656 | 0.42380 | 0.42315 | 0.44865 | 0.43117 | 0.43577 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.43251 | 0.42591 | 0.41927 | 0.42981 | 0.43797 | 0.42801 | 0.32693 | 0.31796 | 0.31677 | 0.33242 | 0.32693 | 0.31796 | 0.31677 | 0.33242 | 0.45334 | 0.44697 | 0.44429 | 0.43575 | 0.42819 | 0.42545 | 0.42479 | 0.45046 | 0.43279 | 0.43737 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.78165 | 0.80695 | 0.80329 | 0.80552 | 0.80525 | 0.80802 | 0.56677 | 0.55095 | 0.56554 | 0.56977 | 0.56677 | 0.55095 | 0.56554 | 0.56977 | 0.81196 | 0.81958 | 0.80141 | 0.80436 | 0.78519 | 0.80735 | 0.77339 | 0.79799 | 0.83219 | 0.80037 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.78716 | 0.81275 | 0.80889 | 0.81117 | 0.81103 | 0.81351 | 0.57075 | 0.55497 | 0.56944 | 0.57375 | 0.57075 | 0.55497 | 0.56944 | 0.57375 | 0.81762 | 0.82524 | 0.80714 | 0.81023 | 0.79054 | 0.81290 | 0.77875 | 0.80349 | 0.83815 | 0.80605 |
| eval/lm/c4_en-validation/CE loss | lower | 3.3348 | 3.3501 | 3.3411 | 3.3379 | 3.3445 | 3.3454 | 3.3284 | 3.2356 | 3.2485 | 3.2792 | 3.3284 | 3.2356 | 3.2485 | 3.2792 | 3.3832 | 3.3531 | 3.3489 | 3.3537 | 3.3441 | 3.3427 | 3.3499 | 3.3603 | 3.3792 | 3.3696 |
| eval/lm/c4_en-validation/PPL | lower | 28.07 | 28.50 | 28.25 | 28.16 | 28.35 | 28.37 | 27.89 | 25.42 | 25.75 | 26.55 | 27.89 | 25.42 | 25.75 | 26.55 | 29.46 | 28.59 | 28.47 | 28.61 | 28.33 | 28.29 | 28.50 | 28.80 | 29.35 | 29.07 |
| eval/lm/dolma_books-validation/CE loss | lower | 3.3153 | 3.3298 | 3.3145 | 3.3142 | 3.3279 | 3.3236 | 3.3148 | 3.1928 | 3.2103 | 3.2560 | 3.3148 | 3.1928 | 3.2103 | 3.2560 | 3.3614 | 3.3303 | 3.3265 | 3.3318 | 3.3154 | 3.3133 | 3.3198 | 3.3324 | 3.3577 | 3.3402 |
| eval/lm/dolma_books-validation/PPL | lower | 27.53 | 27.93 | 27.51 | 27.50 | 27.88 | 27.76 | 27.52 | 24.36 | 24.79 | 25.95 | 27.52 | 24.36 | 24.79 | 25.95 | 28.83 | 27.95 | 27.84 | 27.99 | 27.53 | 27.48 | 27.66 | 28.01 | 28.72 | 28.22 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.4646 | 3.4793 | 3.4707 | 3.4682 | 3.4734 | 3.4745 | 3.4667 | 3.3697 | 3.3862 | 3.4172 | 3.4667 | 3.3697 | 3.3862 | 3.4172 | 3.5114 | 3.4826 | 3.4811 | 3.4827 | 3.4720 | 3.4691 | 3.4792 | 3.4870 | 3.5072 | 3.4978 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 31.96 | 32.44 | 32.16 | 32.08 | 32.25 | 32.28 | 32.03 | 29.07 | 29.55 | 30.48 | 32.03 | 29.07 | 29.55 | 30.48 | 33.49 | 32.54 | 32.50 | 32.55 | 32.20 | 32.11 | 32.43 | 32.69 | 33.35 | 33.04 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.4860 | 2.5032 | 2.4935 | 2.4920 | 2.4985 | 2.5004 | 2.4140 | 2.3417 | 2.3491 | 2.3714 | 2.4140 | 2.3417 | 2.3491 | 2.3714 | 2.5304 | 2.5025 | 2.4969 | 2.5009 | 2.4941 | 2.4927 | 2.4955 | 2.5051 | 2.5257 | 2.5179 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 12.01 | 12.22 | 12.10 | 12.09 | 12.16 | 12.19 | 11.18 | 10.40 | 10.48 | 10.71 | 11.18 | 10.40 | 10.48 | 10.71 | 12.56 | 12.21 | 12.15 | 12.19 | 12.11 | 12.09 | 12.13 | 12.24 | 12.50 | 12.40 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.6085 | 3.6183 | 3.6104 | 3.6089 | 3.6117 | 3.6144 | 3.5906 | 3.5102 | 3.5252 | 3.5447 | 3.5906 | 3.5102 | 3.5252 | 3.5447 | 3.6525 | 3.6246 | 3.6190 | 3.6224 | 3.6132 | 3.6133 | 3.6219 | 3.6304 | 3.6478 | 3.6351 |
| eval/lm/dolma_reddit-validation/PPL | lower | 36.91 | 37.28 | 36.98 | 36.93 | 37.03 | 37.13 | 36.25 | 33.45 | 33.96 | 34.63 | 36.25 | 33.45 | 33.96 | 34.63 | 38.57 | 37.51 | 37.30 | 37.43 | 37.08 | 37.09 | 37.41 | 37.73 | 38.39 | 37.91 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.7230 | 1.7325 | 1.7413 | 1.7328 | 1.7302 | 1.7534 | 1.5466 | 1.4795 | 1.4831 | 1.5073 | 1.5466 | 1.4795 | 1.4831 | 1.5073 | 1.7758 | 1.7539 | 1.7414 | 1.7433 | 1.7434 | 1.7278 | 1.7309 | 1.7742 | 1.7796 | 1.7599 |
| eval/lm/dolma_stack-validation/PPL | lower | 5.6014 | 5.6547 | 5.7047 | 5.6567 | 5.6420 | 5.7741 | 4.6957 | 4.3905 | 4.4068 | 4.5147 | 4.6957 | 4.3905 | 4.4068 | 4.5147 | 5.9047 | 5.7769 | 5.7052 | 5.7163 | 5.7169 | 5.6280 | 5.6455 | 5.8953 | 5.9276 | 5.8120 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 3.0169 | 3.0368 | 3.0222 | 3.0223 | 3.0283 | 3.0251 | 2.8424 | 2.7348 | 2.7508 | 2.7853 | 2.8424 | 2.7348 | 2.7508 | 2.7853 | 3.0702 | 3.0363 | 3.0314 | 3.0351 | 3.0276 | 3.0252 | 3.0362 | 3.0456 | 3.0659 | 3.0570 |
| eval/lm/dolma_wiki-validation/PPL | lower | 20.43 | 20.84 | 20.54 | 20.54 | 20.66 | 20.60 | 17.16 | 15.41 | 15.66 | 16.20 | 17.16 | 15.41 | 15.66 | 16.20 | 21.55 | 20.83 | 20.73 | 20.80 | 20.65 | 20.60 | 20.83 | 21.02 | 21.45 | 21.26 |
| eval/lm/ice-validation/CE loss | lower | 3.4135 | 3.4275 | 3.4203 | 3.4503 | 3.4387 | 3.4489 | 3.3105 | 3.2444 | 3.2517 | 3.2919 | 3.3105 | 3.2444 | 3.2517 | 3.2919 | 3.4997 | 3.4421 | 3.4421 | 3.4448 | 3.4186 | 3.4135 | 3.4387 | 3.4551 | 3.4907 | 3.4805 |
| eval/lm/ice-validation/PPL | lower | 30.37 | 30.80 | 30.58 | 31.51 | 31.15 | 31.47 | 27.40 | 25.65 | 25.83 | 26.89 | 27.40 | 25.65 | 25.83 | 26.89 | 33.10 | 31.25 | 31.25 | 31.34 | 30.53 | 30.37 | 31.15 | 31.66 | 32.81 | 32.48 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 3.4142 | 3.4224 | 3.4211 | 3.4135 | 3.4141 | 3.4226 | 3.1969 | 3.1334 | 3.1408 | 3.1564 | 3.1969 | 3.1334 | 3.1408 | 3.1564 | 3.4539 | 3.4400 | 3.4297 | 3.4269 | 3.4185 | 3.4173 | 3.4178 | 3.4356 | 3.4543 | 3.4396 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 30.39 | 30.64 | 30.60 | 30.37 | 30.39 | 30.65 | 24.46 | 22.95 | 23.12 | 23.49 | 24.46 | 22.95 | 23.12 | 23.49 | 31.62 | 31.19 | 30.87 | 30.78 | 30.52 | 30.49 | 30.50 | 31.05 | 31.64 | 31.18 |
| eval/lm/pile-validation/CE loss | lower | 2.6141 | 2.6278 | 2.6222 | 2.6188 | 2.6261 | 2.6313 | 2.5457 | 2.4637 | 2.4738 | 2.5014 | 2.5457 | 2.4637 | 2.4738 | 2.5014 | 2.6614 | 2.6339 | 2.6260 | 2.6286 | 2.6243 | 2.6229 | 2.6228 | 2.6424 | 2.6606 | 2.6510 |
| eval/lm/pile-validation/PPL | lower | 13.65 | 13.84 | 13.77 | 13.72 | 13.82 | 13.89 | 12.75 | 11.75 | 11.87 | 12.20 | 12.75 | 11.75 | 11.87 | 12.20 | 14.32 | 13.93 | 13.82 | 13.85 | 13.79 | 13.78 | 13.77 | 14.05 | 14.30 | 14.17 |
| eval/lm/wikitext_103-validation/CE loss | lower | 3.0393 | 3.0459 | 3.0440 | 3.0320 | 3.0309 | 3.0510 | 2.8554 | 2.7592 | 2.7723 | 2.8083 | 2.8554 | 2.7592 | 2.7723 | 2.8083 | 3.0940 | 3.0627 | 3.0395 | 3.0501 | 3.0413 | 3.0319 | 3.0361 | 3.0621 | 3.1016 | 3.0777 |
| eval/lm/wikitext_103-validation/PPL | lower | 20.89 | 21.03 | 20.99 | 20.74 | 20.72 | 21.14 | 17.38 | 15.79 | 16.00 | 16.58 | 17.38 | 15.79 | 16.00 | 16.58 | 22.07 | 21.39 | 20.89 | 21.12 | 20.93 | 20.74 | 20.82 | 21.37 | 22.23 | 21.71 |
| throughput/in-loop eval batches | see metric | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 |
| throughput/in-loop eval time (s) | see metric | 229.4 | 225.5 | 229.0 | 211.1 | 218.5 | 214.5 | 411.9 | 417.9 | 409.9 | 411.9 | 411.9 | 417.9 | 409.9 | 411.9 | 389.3 | 231.6 | 233.2 | 236.0 | 246.2 | 232.4 | 247.6 | 237.6 | 234.8 | 227.7 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| int-275m-cx1-intd256e8k-lr1.6e-3-r1<br>`b2g99ewo` | finished | original | 3966238720.0 | 15130 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/b2g99ewo) |
| int-275m-cx1-intd256e8k-lr3.2e-3-r1<br>`fmpio3ko` | finished | original | 3966238720.0 | 15130 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fmpio3ko) |
| int-275m-cx1-intd256e8k-lr8e-4-r1<br>`51vzuu2l` | finished | original | 3966238720.0 | 15130 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/51vzuu2l) |
| int-275m-cx1-intw256e8k-lr1.6e-3-r1<br>`h86x1nv3` | finished | original | 4062969856.0 | 15499 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h86x1nv3) |
| int-275m-cx1-intw256e8k-lr3.2e-3-r1<br>`afxq80js` | finished | original | 4062969856.0 | 15499 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/afxq80js) |
| int-275m-cx1-intw256e8k-lr8e-4-r1<br>`kfua3dcq` | finished | original | 4062969856.0 | 15499 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/kfua3dcq) |
| mt-275m-baseline-cx1-lr1.6e-3-r1<br>`w3vof8b9` | finished | original | 100000595968.0 | 95368 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/w3vof8b9) |
| mt-275m-baseline-cx1-lr2e-4-r1<br>`cm8ww646` | finished | original | 100000595968.0 | 95368 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/cm8ww646) |
| mt-275m-baseline-cx1-lr4e-4-r1<br>`r6ts032g` | finished | original | 100000595968.0 | 95368 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/r6ts032g) |
| mt-275m-baseline-cx1-lr8e-4-r1<br>`lfydkxv4` | finished | original | 100000595968.0 | 95368 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/lfydkxv4) |
| mt-eval-275m-baseline-cx1-lr1.6e-3-r1<br>`g72jnvlh` | finished | original | 100000595968.0 | 95368 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/g72jnvlh) |
| mt-eval-275m-baseline-cx1-lr2e-4-r1<br>`z2n5obvt` | finished | original | 100000595968.0 | 95368 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/z2n5obvt) |
| mt-eval-275m-baseline-cx1-lr4e-4-r1<br>`946lvt8o` | finished | original | 100000595968.0 | 95368 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/946lvt8o) |
| mt-eval-275m-baseline-cx1-lr8e-4-r1<br>`q6tuf453` | finished | original | 100000595968.0 | 95368 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/q6tuf453) |
| olmoe3-eval-275m-cx1-lr1e-3-r2<br>`h3y8marg` | finished | original | 4027842560.0 | 15365 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h3y8marg) |
| q3-275m-cx1-q3am128e8k-lr1e-3-r1<br>`ivpueqxh` | finished | original | 4043309056.0 | 15424 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ivpueqxh) |
| q3-275m-cx1-q3am128e8k-lr2e-3-r1<br>`5vaz5tl1` | finished | original | 4043309056.0 | 15424 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5vaz5tl1) |
| q3-275m-cx1-q3am128e8k-lr4e-3-r1<br>`wtnzni2d` | finished | original | 4043309056.0 | 15424 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/wtnzni2d) |
| q3-275m-cx1-q3td128e8k-lr1e-3-r1<br>`ww4vodxv` | finished | original | 4027842560.0 | 15365 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ww4vodxv) |
| q3-275m-cx1-q3td128e8k-lr2e-3-r1<br>`fhgythx3` | finished | original | 4027842560.0 | 15365 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/fhgythx3) |
| q3-275m-cx1-q3td128e8k-lr4e-3-r1<br>`1erw5m3k` | finished | original | 4027842560.0 | 15365 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1erw5m3k) |
| q3-275m-cx1-q3td128e8k-lr5e-4-r1<br>`sujdnlv0` | finished | original | 4027842560.0 | 15365 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sujdnlv0) |
| se-275m-cx1-se0m9-lr1e-3-r2<br>`z7os3acu` | finished | original | 4027842560.0 | 15365 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/z7os3acu) |
| se-275m-cx1-se0m9-lr2e-3-r2<br>`0af1i3o1` | finished | original | 4027842560.0 | 15365 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0af1i3o1) |

## unknown Cx2

Showing first 24 of 25 runs in this table. Use `--name-regex` to narrow the view.

| metric | direction | int-275m-cx2-intd256e8k-lr1.6e-3-r1<br>`igsm7yj9` | int-275m-cx2-intd256e8k-lr3.2e-3-r1<br>`1pwfw2jo` | int-275m-cx2-intd256e8k-lr8e-4-r1<br>`suaysv7u` | int-275m-cx2-intw256e8k-lr1.6e-3-r1<br>`6porpbo2` | int-275m-cx2-intw256e8k-lr3.2e-3-r1<br>`0f782vrw` | int-275m-cx2-intw256e8k-lr8e-4-r1<br>`o2bdr3gw` | q3-275m-cx2-q3am128e8k-lr1.8e-3-r1<br>`4hy9tf4o` | q3-275m-cx2-q3am128e8k-lr3.6e-3-r1<br>`5bv2y0fp` | q3-275m-cx2-q3am128e8k-lr9e-4-r1<br>`5yiwgg3x` | q3-275m-cx2-q3td128e8k-lr1.8e-3-r1<br>`uso06e7k` | q3-275m-cx2-q3td128e8k-lr3.6e-3-r1<br>`zadoal7a` | q3-275m-cx2-q3td128e8k-lr4.5e-4-r1<br>`9zhx2ws4` | q3-275m-cx2-q3td128e8k-lr9e-4-r1<br>`uvsmf0rw` | se-275m-cx2-se0m9-lr1.8e-3-r2<br>`97xdkfc4` | se-275m-cx2-se0m9-lr3.6e-3-r2<br>`2oirkw3f` | se-275m-cx2-se0m9-lr9e-4-r2<br>`rb52lk9m` | sp-275m-cx2-sp192e4k-lr1.8e-3-r1<br>`r78rwdr8` | sp-275m-cx2-sp192e4k-lr2.25e-4-r2<br>`ylz3dxx5` | sp-275m-cx2-sp192e4k-lr3.6e-3-r1<br>`5bwqn7br` | sp-275m-cx2-sp192e4k-lr4.5e-4-r2<br>`7hzc9eo9` | sp-275m-cx2-sp192e4k-lr6e-4-r2<br>`06d0ggb0` | sp-275m-cx2-sp192e4k-lr9e-4-r1<br>`m21y2jzg` | sp-275m-cx2-sp96e4k-lr1.8e-3-r1<br>`tajjw92i` | sp-275m-cx2-sp96e4k-lr3.6e-3-r1<br>`xf9ao51p` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 1.0068 | 1.0073 | 0.99835 | 1.0005 | 0.99434 | 1.0067 | 1.0020 | 0.99397 | 0.99738 | 1.0045 | 1.0019 | 1.0047 | 1.0001 | 1.0232 | 1.0200 | 1.0320 | 1.0072 | 1.0249 | 1.0077 | 1.0003 | 1.0112 | 1.0012 | 1.0197 | 1.0200 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 1.1019 | 1.1044 | 1.0931 | 1.0958 | 1.0872 | 1.1011 | 1.0987 | 1.0914 | 1.0947 | 1.0987 | 1.0986 | 1.1001 | 1.0962 | 1.1209 | 1.1196 | 1.1317 | 1.1009 | 1.1198 | 1.1043 | 1.0965 | 1.1072 | 1.0973 | 1.1184 | 1.1188 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0286 | 1.0363 | 1.0454 | 1.1182 | 1.1833 | 1.0628 | 1.0434 | 1.0353 | 1.0332 | 1.0154 | 1.0594 | 1.0551 | 1.0224 | 1.0272 | 1.0318 | 1.0308 | 1.0296 | 1.0363 | 1.0547 | 1.0392 | 1.0617 | 1.0419 | 1.0331 | 1.0550 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0573 | 2.0726 | 2.0908 | 2.2364 | 2.3665 | 2.1255 | 2.0867 | 2.0706 | 2.0663 | 2.0309 | 2.1189 | 2.1102 | 2.0449 | 2.0544 | 2.0635 | 2.0616 | 2.0591 | 2.0725 | 2.1094 | 2.0783 | 2.1234 | 2.0839 | 2.0662 | 2.1099 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.71301 | 0.71834 | 0.72465 | 0.77511 | 0.82028 | 0.73663 | 0.72324 | 0.71766 | 0.71629 | 0.70394 | 0.73441 | 0.73133 | 0.70883 | 0.71205 | 0.71530 | 0.71456 | 0.71375 | 0.71838 | 0.73113 | 0.72033 | 0.73593 | 0.72220 | 0.71618 | 0.73124 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4260 | 1.4367 | 1.4493 | 1.5502 | 1.6406 | 1.4733 | 1.4465 | 1.4353 | 1.4326 | 1.4079 | 1.4688 | 1.4627 | 1.4177 | 1.4241 | 1.4306 | 1.4291 | 1.4275 | 1.4368 | 1.4623 | 1.4407 | 1.4719 | 1.4444 | 1.4324 | 1.4625 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.27133 | 0.26365 | 0.25768 | 0.24147 | 0.22696 | 0.24232 | 0.25256 | 0.24829 | 0.23208 | 0.23464 | 0.22867 | 0.25427 | 0.24915 | 0.24488 | 0.22867 | 0.23805 | 0.25256 | 0.25256 | 0.23464 | 0.26365 | 0.24659 | 0.26365 | 0.27133 | 0.23038 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.27133 | 0.26365 | 0.25768 | 0.24147 | 0.22696 | 0.24232 | 0.25256 | 0.24829 | 0.23208 | 0.23464 | 0.22867 | 0.25427 | 0.24915 | 0.24488 | 0.22867 | 0.23805 | 0.25256 | 0.25256 | 0.23464 | 0.26365 | 0.24659 | 0.26365 | 0.27133 | 0.23038 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.4092 | -1.4180 | -1.4192 | -1.5355 | -1.6257 | -1.4573 | -1.4173 | -1.4193 | -1.3979 | -1.3937 | -1.4467 | -1.4426 | -1.4011 | -1.3995 | -1.4036 | -1.3983 | -1.4069 | -1.4147 | -1.4371 | -1.4160 | -1.4455 | -1.4165 | -1.4094 | -1.4330 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.4092 | -1.4180 | -1.4192 | -1.5355 | -1.6257 | -1.4573 | -1.4173 | -1.4193 | -1.3979 | -1.3937 | -1.4467 | -1.4426 | -1.4011 | -1.3995 | -1.4036 | -1.3983 | -1.4069 | -1.4147 | -1.4371 | -1.4160 | -1.4455 | -1.4165 | -1.4094 | -1.4330 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.25123 | 0.25490 | 0.25125 | 0.25088 | 0.24404 | 0.24885 | 0.24955 | 0.24896 | 0.25131 | 0.25140 | 0.24865 | 0.25062 | 0.25022 | 0.25056 | 0.24913 | 0.25126 | 0.25020 | 0.25146 | 0.24902 | 0.25067 | 0.24791 | 0.25206 | 0.25138 | 0.24786 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.25123 | 0.25490 | 0.25125 | 0.25088 | 0.24404 | 0.24885 | 0.24955 | 0.24896 | 0.25131 | 0.25140 | 0.24865 | 0.25062 | 0.25022 | 0.25056 | 0.24913 | 0.25126 | 0.25020 | 0.25146 | 0.24902 | 0.25067 | 0.24791 | 0.25206 | 0.25138 | 0.24786 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.81083 | 0.80523 | 0.79740 | 0.80011 | 0.80757 | 0.80627 | 0.79748 | 0.79713 | 0.80574 | 0.81351 | 0.82007 | 0.83251 | 0.80049 | 0.82008 | 0.82436 | 0.83300 | 0.80513 | 0.84364 | 0.81633 | 0.79955 | 0.80876 | 0.81713 | 0.82758 | 0.83737 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.88363 | 0.87726 | 0.86794 | 0.87155 | 0.87978 | 0.87756 | 0.87016 | 0.86902 | 0.87868 | 0.88785 | 0.89466 | 0.90841 | 0.87303 | 0.89382 | 0.89865 | 0.90833 | 0.87709 | 0.91944 | 0.88964 | 0.87178 | 0.88122 | 0.89159 | 0.90260 | 0.91360 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0346 | 1.0428 | 1.0344 | 1.1234 | 1.1172 | 1.0704 | 1.0359 | 1.0231 | 1.0313 | 1.0225 | 1.0514 | 1.0373 | 1.0381 | 1.0378 | 1.0467 | 1.0332 | 1.0343 | 1.0402 | 1.0476 | 1.0478 | 1.0383 | 1.0382 | 1.0471 | 1.0422 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0691 | 2.0856 | 2.0688 | 2.2468 | 2.2343 | 2.1408 | 2.0719 | 2.0462 | 2.0626 | 2.0450 | 2.1028 | 2.0746 | 2.0762 | 2.0756 | 2.0934 | 2.0664 | 2.0687 | 2.0803 | 2.0952 | 2.0956 | 2.0765 | 2.0763 | 2.0941 | 2.0843 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.71716 | 0.72283 | 0.71706 | 0.77873 | 0.77441 | 0.74191 | 0.71813 | 0.70922 | 0.71491 | 0.70876 | 0.72876 | 0.71900 | 0.71970 | 0.71940 | 0.72556 | 0.71623 | 0.71699 | 0.72099 | 0.72618 | 0.72626 | 0.71975 | 0.71967 | 0.72580 | 0.72246 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4343 | 1.4457 | 1.4341 | 1.5575 | 1.5488 | 1.4838 | 1.4363 | 1.4184 | 1.4298 | 1.4175 | 1.4575 | 1.4380 | 1.4394 | 1.4388 | 1.4511 | 1.4325 | 1.4340 | 1.4420 | 1.4524 | 1.4525 | 1.4395 | 1.4393 | 1.4516 | 1.4449 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.25126 | 0.24621 | 0.25505 | 0.25042 | 0.25042 | 0.23274 | 0.24537 | 0.24916 | 0.26978 | 0.26052 | 0.25337 | 0.24916 | 0.24958 | 0.25253 | 0.25042 | 0.25673 | 0.25884 | 0.25673 | 0.24453 | 0.24285 | 0.24453 | 0.26515 | 0.24411 | 0.25253 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.25126 | 0.24621 | 0.25505 | 0.25042 | 0.25042 | 0.23274 | 0.24537 | 0.24916 | 0.26978 | 0.26052 | 0.25337 | 0.24916 | 0.24958 | 0.25253 | 0.25042 | 0.25673 | 0.25884 | 0.25673 | 0.24453 | 0.24285 | 0.24453 | 0.26515 | 0.24411 | 0.25253 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.4176 | -1.4237 | -1.4074 | -1.5417 | -1.5336 | -1.4699 | -1.4086 | -1.4017 | -1.3966 | -1.4041 | -1.4373 | -1.4207 | -1.4171 | -1.4129 | -1.4207 | -1.4097 | -1.4135 | -1.4227 | -1.4301 | -1.4299 | -1.4174 | -1.4151 | -1.4243 | -1.4152 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.4176 | -1.4237 | -1.4074 | -1.5417 | -1.5336 | -1.4699 | -1.4086 | -1.4017 | -1.3966 | -1.4041 | -1.4373 | -1.4207 | -1.4171 | -1.4129 | -1.4207 | -1.4097 | -1.4135 | -1.4227 | -1.4301 | -1.4299 | -1.4174 | -1.4151 | -1.4243 | -1.4152 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.24916 | 0.25092 | 0.25248 | 0.25302 | 0.25240 | 0.24810 | 0.24947 | 0.25194 | 0.25244 | 0.25148 | 0.25030 | 0.25161 | 0.24937 | 0.24951 | 0.24926 | 0.25015 | 0.25176 | 0.25108 | 0.24952 | 0.25036 | 0.25014 | 0.25330 | 0.24879 | 0.24886 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.24916 | 0.25092 | 0.25248 | 0.25302 | 0.25240 | 0.24810 | 0.24947 | 0.25194 | 0.25244 | 0.25148 | 0.25030 | 0.25161 | 0.24937 | 0.24951 | 0.24926 | 0.25015 | 0.25176 | 0.25108 | 0.24952 | 0.25036 | 0.25014 | 0.25330 | 0.24879 | 0.24886 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 2.0440 | 1.9877 | 2.1103 | 2.0429 | 1.9115 | 2.0699 | 2.0878 | 2.0148 | 2.1421 | 2.1290 | 2.0312 | 2.2005 | 2.1851 | 2.1222 | 2.2346 | 2.2445 | 2.0682 | 2.1021 | 2.0730 | 2.1823 | 2.1494 | 2.1367 | 2.0924 | 2.0056 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 3.2642 | 3.2049 | 3.3501 | 3.2600 | 3.0714 | 3.3347 | 3.3466 | 3.2169 | 3.4233 | 3.3930 | 3.2553 | 3.5217 | 3.5084 | 3.3806 | 3.5705 | 3.5676 | 3.3181 | 3.3471 | 3.3386 | 3.5013 | 3.4232 | 3.3943 | 3.3345 | 3.1764 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 1.4169 | 1.3777 | 1.4627 | 1.4159 | 1.3249 | 1.4346 | 1.4472 | 1.3964 | 1.4847 | 1.4756 | 1.4079 | 1.5252 | 1.5145 | 1.4709 | 1.5489 | 1.5558 | 1.4335 | 1.4571 | 1.4370 | 1.5127 | 1.4897 | 1.4809 | 1.4503 | 1.3901 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 2.2626 | 2.2214 | 2.3221 | 2.2597 | 2.1290 | 2.3113 | 2.3197 | 2.2298 | 2.3728 | 2.3519 | 2.2564 | 2.4411 | 2.4319 | 2.3432 | 2.4747 | 2.4729 | 2.3002 | 2.3200 | 2.3143 | 2.4268 | 2.3728 | 2.3529 | 2.3115 | 2.2019 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.14613 | 0.16237 | 0.14136 | 0.14040 | 0.19484 | 0.14518 | 0.16714 | 0.15568 | 0.14613 | 0.14518 | 0.15377 | 0.12703 | 0.12225 | 0.10984 | 0.11270 | 0.10315 | 0.16619 | 0.15091 | 0.15473 | 0.14518 | 0.14518 | 0.13276 | 0.14995 | 0.15664 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.14613 | 0.16237 | 0.14136 | 0.14040 | 0.19484 | 0.14518 | 0.16714 | 0.15568 | 0.14613 | 0.14518 | 0.15377 | 0.12703 | 0.12225 | 0.10984 | 0.11270 | 0.10315 | 0.16619 | 0.15091 | 0.15473 | 0.14518 | 0.14518 | 0.13276 | 0.14995 | 0.15664 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -2.6289 | -2.4744 | -2.7017 | -2.6527 | -2.3825 | -2.6574 | -2.6487 | -2.5968 | -2.6437 | -2.6830 | -2.5282 | -2.7338 | -2.6354 | -2.6267 | -2.6893 | -2.7865 | -2.6068 | -2.6329 | -2.5642 | -2.7557 | -2.6784 | -2.7140 | -2.5971 | -2.5830 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -2.6289 | -2.4744 | -2.7017 | -2.6527 | -2.3825 | -2.6574 | -2.6487 | -2.5968 | -2.6437 | -2.6830 | -2.5282 | -2.7338 | -2.6354 | -2.6267 | -2.6893 | -2.7865 | -2.6068 | -2.6329 | -2.5642 | -2.7557 | -2.6784 | -2.7140 | -2.5971 | -2.5830 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.12832 | 0.13944 | 0.12931 | 0.13846 | 0.15519 | 0.13563 | 0.13366 | 0.14309 | 0.12954 | 0.11752 | 0.13097 | 0.11122 | 0.11367 | 0.10822 | 0.10689 | 0.11222 | 0.14020 | 0.12654 | 0.12805 | 0.12974 | 0.13148 | 0.11896 | 0.12745 | 0.14125 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.12832 | 0.13944 | 0.12931 | 0.13846 | 0.15519 | 0.13563 | 0.13366 | 0.14309 | 0.12954 | 0.11752 | 0.13097 | 0.11122 | 0.11367 | 0.10822 | 0.10689 | 0.11222 | 0.14020 | 0.12654 | 0.12805 | 0.12974 | 0.13148 | 0.11896 | 0.12745 | 0.14125 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.49218 | 0.51976 | 0.48885 | 0.51111 | 0.48860 | 0.51724 | 0.53838 | 0.51897 | 0.53350 | 0.55077 | 0.57959 | 0.54774 | 0.53502 | 0.54373 | 0.55160 | 0.57827 | 0.49623 | 0.55140 | 0.55712 | 0.54645 | 0.52328 | 0.49573 | 0.54889 | 0.53236 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.53489 | 0.56396 | 0.53023 | 0.55709 | 0.53077 | 0.56189 | 0.58730 | 0.56440 | 0.58222 | 0.59895 | 0.63242 | 0.59630 | 0.58248 | 0.59038 | 0.60010 | 0.62929 | 0.53993 | 0.60164 | 0.60690 | 0.59405 | 0.57106 | 0.53857 | 0.59879 | 0.57751 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.34114 | 0.36025 | 0.33885 | 0.35428 | 0.33867 | 0.35855 | 0.37322 | 0.35972 | 0.36983 | 0.38174 | 0.40176 | 0.37965 | 0.37083 | 0.37684 | 0.38233 | 0.40080 | 0.34394 | 0.38220 | 0.38616 | 0.37878 | 0.36274 | 0.34360 | 0.38042 | 0.36901 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.37073 | 0.39088 | 0.36756 | 0.38611 | 0.36789 | 0.38950 | 0.40710 | 0.39123 | 0.40353 | 0.41516 | 0.43834 | 0.41334 | 0.40373 | 0.40921 | 0.41596 | 0.43625 | 0.37428 | 0.41699 | 0.42068 | 0.41174 | 0.39582 | 0.37331 | 0.41504 | 0.40029 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.41601 | 0.41008 | 0.44664 | 0.43874 | 0.44565 | 0.45158 | 0.44565 | 0.43775 | 0.44170 | 0.43379 | 0.43281 | 0.42984 | 0.42885 | 0.42292 | 0.43281 | 0.41996 | 0.45455 | 0.42589 | 0.42885 | 0.44071 | 0.46047 | 0.43083 | 0.43676 | 0.42292 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.41601 | 0.41008 | 0.44664 | 0.43874 | 0.44565 | 0.45158 | 0.44565 | 0.43775 | 0.44170 | 0.43379 | 0.43281 | 0.42984 | 0.42885 | 0.42292 | 0.43281 | 0.41996 | 0.45455 | 0.42589 | 0.42885 | 0.44071 | 0.46047 | 0.43083 | 0.43676 | 0.42292 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -3.4477 | -3.5240 | -3.3314 | -3.1911 | -3.0346 | -3.1948 | -3.2769 | -3.4437 | -3.3651 | -3.4371 | -3.4662 | -3.5466 | -3.3818 | -3.6840 | -3.4998 | -3.6884 | -3.1604 | -3.4135 | -3.2968 | -3.5696 | -3.2427 | -3.2912 | -3.2604 | -3.5622 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -3.4477 | -3.5240 | -3.3314 | -3.1911 | -3.0346 | -3.1948 | -3.2769 | -3.4437 | -3.3651 | -3.4371 | -3.4662 | -3.5466 | -3.3818 | -3.6840 | -3.4998 | -3.6884 | -3.1604 | -3.4135 | -3.2968 | -3.5696 | -3.2427 | -3.2912 | -3.2604 | -3.5622 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.40566 | 0.40555 | 0.42508 | 0.42212 | 0.43317 | 0.42296 | 0.41736 | 0.42620 | 0.41613 | 0.41549 | 0.41242 | 0.40749 | 0.41803 | 0.40635 | 0.40518 | 0.39030 | 0.43564 | 0.40296 | 0.41000 | 0.41008 | 0.42573 | 0.42095 | 0.42348 | 0.40330 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.40566 | 0.40555 | 0.42508 | 0.42212 | 0.43317 | 0.42296 | 0.41736 | 0.42620 | 0.41613 | 0.41549 | 0.41242 | 0.40749 | 0.41803 | 0.40635 | 0.40518 | 0.39030 | 0.43564 | 0.40296 | 0.41000 | 0.41008 | 0.42573 | 0.42095 | 0.42348 | 0.40330 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.61333 | 0.63998 | 0.66297 | 0.66418 | 0.66876 | 0.65586 | 0.67481 | 0.60257 | 0.77440 | 0.62687 | 0.61651 | 0.64729 | 0.63699 | 0.72205 | 0.72368 | 0.75058 | 0.66166 | 0.68914 | 0.64865 | 0.61346 | 0.69426 | 0.68744 | 0.71814 | 0.62680 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.73713 | 0.77109 | 0.79541 | 0.79823 | 0.80210 | 0.78755 | 0.81012 | 0.72322 | 0.92904 | 0.75357 | 0.74127 | 0.77738 | 0.76510 | 0.86734 | 0.86888 | 0.89991 | 0.79302 | 0.82782 | 0.77928 | 0.73767 | 0.83385 | 0.82607 | 0.86377 | 0.75064 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.42538 | 0.44386 | 0.45968 | 0.46054 | 0.46373 | 0.45480 | 0.46789 | 0.41786 | 0.53697 | 0.43469 | 0.42756 | 0.44884 | 0.44169 | 0.50074 | 0.50183 | 0.52047 | 0.45882 | 0.47788 | 0.44991 | 0.42539 | 0.48147 | 0.47670 | 0.49798 | 0.43471 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.51128 | 0.53482 | 0.55152 | 0.55356 | 0.55622 | 0.54618 | 0.56167 | 0.50153 | 0.64418 | 0.52254 | 0.51410 | 0.53909 | 0.53052 | 0.60155 | 0.60248 | 0.62398 | 0.54987 | 0.57406 | 0.54051 | 0.51152 | 0.57818 | 0.57292 | 0.59902 | 0.52065 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.76085 | 0.75892 | 0.73192 | 0.73288 | 0.77724 | 0.73385 | 0.73481 | 0.75603 | 0.67695 | 0.72324 | 0.76181 | 0.72035 | 0.74349 | 0.74253 | 0.69913 | 0.68949 | 0.74349 | 0.71553 | 0.76471 | 0.75121 | 0.74446 | 0.73192 | 0.72999 | 0.74253 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.76085 | 0.75892 | 0.73192 | 0.73288 | 0.77724 | 0.73385 | 0.73481 | 0.75603 | 0.67695 | 0.72324 | 0.76181 | 0.72035 | 0.74349 | 0.74253 | 0.69913 | 0.68949 | 0.74349 | 0.71553 | 0.76471 | 0.75121 | 0.74446 | 0.73192 | 0.72999 | 0.74253 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.68598 | -0.65972 | -0.73936 | -0.70074 | -0.68603 | -0.72194 | -0.68808 | -0.66223 | -0.82851 | -0.72806 | -0.67161 | -0.76039 | -0.70079 | -0.69193 | -0.77182 | -0.81193 | -0.71711 | -0.76053 | -0.67867 | -0.69598 | -0.71285 | -0.69192 | -0.71214 | -0.70297 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.68598 | -0.65972 | -0.73936 | -0.70074 | -0.68603 | -0.72194 | -0.68808 | -0.66223 | -0.82851 | -0.72806 | -0.67161 | -0.76039 | -0.70079 | -0.69193 | -0.77182 | -0.81193 | -0.71711 | -0.76053 | -0.67867 | -0.69598 | -0.71285 | -0.69192 | -0.71214 | -0.70297 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.65066 | 0.65829 | 0.62207 | 0.63632 | 0.65841 | 0.62845 | 0.63510 | 0.65357 | 0.59056 | 0.63308 | 0.65591 | 0.59672 | 0.63560 | 0.63873 | 0.61254 | 0.58810 | 0.63384 | 0.60891 | 0.64741 | 0.64016 | 0.63852 | 0.63311 | 0.62155 | 0.63781 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.65066 | 0.65829 | 0.62207 | 0.63632 | 0.65841 | 0.62845 | 0.63510 | 0.65357 | 0.59056 | 0.63308 | 0.65591 | 0.59672 | 0.63560 | 0.63873 | 0.61254 | 0.58810 | 0.63384 | 0.60891 | 0.64741 | 0.64016 | 0.63852 | 0.63311 | 0.62155 | 0.63781 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.27671 | 0.29419 | 0.27696 | 0.32942 | 0.30265 | 0.36183 | 0.29034 | 0.33948 | 0.32936 | 0.29162 | 0.28441 | 0.29055 | 0.28281 | 0.30804 | 0.29068 | 0.30180 | 0.36098 | 0.30809 | 0.29828 | 0.30838 | 0.29602 | 0.28269 | 0.33868 | 0.35452 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.28603 | 0.30398 | 0.28621 | 0.34040 | 0.31281 | 0.37402 | 0.30006 | 0.35100 | 0.34044 | 0.30143 | 0.29395 | 0.30045 | 0.29225 | 0.31846 | 0.30053 | 0.31201 | 0.37306 | 0.31838 | 0.30823 | 0.31878 | 0.30600 | 0.29229 | 0.35011 | 0.36646 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.19183 | 0.20392 | 0.19199 | 0.22835 | 0.20979 | 0.25081 | 0.20125 | 0.23532 | 0.22831 | 0.20215 | 0.19716 | 0.20142 | 0.19604 | 0.21354 | 0.20150 | 0.20922 | 0.25021 | 0.21358 | 0.20677 | 0.21375 | 0.20522 | 0.19596 | 0.23477 | 0.24573 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.19829 | 0.21073 | 0.19842 | 0.23595 | 0.21684 | 0.25928 | 0.20798 | 0.24332 | 0.23598 | 0.20895 | 0.20374 | 0.20826 | 0.20260 | 0.22075 | 0.20832 | 0.21628 | 0.25863 | 0.22070 | 0.21366 | 0.22096 | 0.21211 | 0.20260 | 0.24271 | 0.25403 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.82290 | 0.81843 | 0.81843 | 0.80948 | 0.78980 | 0.79875 | 0.78533 | 0.80501 | 0.80501 | 0.80054 | 0.81932 | 0.83721 | 0.82648 | 0.81485 | 0.80411 | 0.78175 | 0.79785 | 0.78533 | 0.79159 | 0.79606 | 0.79428 | 0.80948 | 0.80322 | 0.77818 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.82290 | 0.81843 | 0.81843 | 0.80948 | 0.78980 | 0.79875 | 0.78533 | 0.80501 | 0.80501 | 0.80054 | 0.81932 | 0.83721 | 0.82648 | 0.81485 | 0.80411 | 0.78175 | 0.79785 | 0.78533 | 0.79159 | 0.79606 | 0.79428 | 0.80948 | 0.80322 | 0.77818 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.47850 | -0.48968 | -0.51578 | -0.53212 | -0.57498 | -0.60507 | -0.56306 | -0.56469 | -0.53586 | -0.51678 | -0.46980 | -0.46198 | -0.47111 | -0.51326 | -0.46665 | -0.60902 | -0.59398 | -0.63450 | -0.57918 | -0.60126 | -0.57197 | -0.55262 | -0.56302 | -0.60482 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.47850 | -0.48968 | -0.51578 | -0.53212 | -0.57498 | -0.60507 | -0.56306 | -0.56469 | -0.53586 | -0.51678 | -0.46980 | -0.46198 | -0.47111 | -0.51326 | -0.46665 | -0.60902 | -0.59398 | -0.63450 | -0.57918 | -0.60126 | -0.57197 | -0.55262 | -0.56302 | -0.60482 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.80945 | 0.80672 | 0.79950 | 0.79460 | 0.77986 | 0.79078 | 0.79217 | 0.79659 | 0.78640 | 0.78284 | 0.80767 | 0.80351 | 0.80348 | 0.79748 | 0.79623 | 0.77407 | 0.79407 | 0.77097 | 0.78081 | 0.78300 | 0.79327 | 0.79277 | 0.78882 | 0.77507 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.80945 | 0.80672 | 0.79950 | 0.79460 | 0.77986 | 0.79078 | 0.79217 | 0.79659 | 0.78640 | 0.78284 | 0.80767 | 0.80351 | 0.80348 | 0.79748 | 0.79623 | 0.77407 | 0.79407 | 0.77097 | 0.78081 | 0.78300 | 0.79327 | 0.79277 | 0.78882 | 0.77507 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 1.1366 | 1.0803 | 1.1416 | 1.1759 | 1.1203 | 1.1512 | 1.1932 | 1.1864 | 1.2210 | 1.1431 | 1.1359 | 1.2955 | 1.2275 | 1.2042 | 1.3200 | 1.2299 | 1.3127 | 1.2359 | 1.1784 | 1.1972 | 1.1727 | 1.1639 | 1.3058 | 1.2071 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.8216 | 1.7333 | 1.8175 | 1.8555 | 1.7930 | 1.8286 | 1.8940 | 1.9037 | 1.9506 | 1.8268 | 1.7981 | 2.0555 | 1.9764 | 1.9053 | 2.1093 | 1.9493 | 2.0889 | 1.9605 | 1.8719 | 1.8916 | 1.8593 | 1.8417 | 2.0746 | 1.9250 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.82398 | 0.78843 | 0.82253 | 0.85505 | 0.81870 | 0.83364 | 0.86903 | 0.86377 | 0.88690 | 0.83441 | 0.82738 | 0.93529 | 0.88911 | 0.88622 | 0.96116 | 0.89034 | 0.94824 | 0.89677 | 0.85019 | 0.86376 | 0.85343 | 0.83850 | 0.94865 | 0.88057 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.3556 | 1.3039 | 1.3395 | 1.3879 | 1.3505 | 1.3589 | 1.4208 | 1.4252 | 1.4556 | 1.3742 | 1.3486 | 1.5207 | 1.4667 | 1.4530 | 1.5812 | 1.4480 | 1.5456 | 1.4617 | 1.3820 | 1.3983 | 1.3936 | 1.3571 | 1.5496 | 1.4468 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.60487 | 0.61985 | 0.58614 | 0.62547 | 0.61049 | 0.61236 | 0.59925 | 0.61798 | 0.61236 | 0.59925 | 0.59551 | 0.56742 | 0.59551 | 0.59363 | 0.58614 | 0.58052 | 0.59738 | 0.61236 | 0.62172 | 0.58801 | 0.62921 | 0.62172 | 0.61236 | 0.56742 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.60487 | 0.61985 | 0.58614 | 0.62547 | 0.61049 | 0.61236 | 0.59925 | 0.61798 | 0.61236 | 0.59925 | 0.59551 | 0.56742 | 0.59551 | 0.59363 | 0.58614 | 0.58052 | 0.59738 | 0.61236 | 0.62172 | 0.58801 | 0.62921 | 0.62172 | 0.61236 | 0.56742 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -1.0084 | -1.0004 | -1.0305 | -1.0274 | -1.0516 | -1.0540 | -1.0454 | -1.0731 | -0.99837 | -1.0668 | -1.0842 | -1.0587 | -1.0475 | -1.0915 | -1.1059 | -1.0495 | -1.0719 | -1.0480 | -1.0426 | -1.0473 | -0.99766 | -0.98770 | -1.0170 | -1.0846 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -1.0084 | -1.0004 | -1.0305 | -1.0274 | -1.0516 | -1.0540 | -1.0454 | -1.0731 | -0.99837 | -1.0668 | -1.0842 | -1.0587 | -1.0475 | -1.0915 | -1.1059 | -1.0495 | -1.0719 | -1.0480 | -1.0426 | -1.0473 | -0.99766 | -0.98770 | -1.0170 | -1.0846 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.53440 | 0.54071 | 0.53212 | 0.53926 | 0.53361 | 0.53696 | 0.53374 | 0.53734 | 0.53297 | 0.53058 | 0.53478 | 0.50714 | 0.51373 | 0.51317 | 0.50743 | 0.50780 | 0.52127 | 0.52381 | 0.54687 | 0.52424 | 0.54375 | 0.54167 | 0.53356 | 0.50836 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.53440 | 0.54071 | 0.53212 | 0.53926 | 0.53361 | 0.53696 | 0.53374 | 0.53734 | 0.53297 | 0.53058 | 0.53478 | 0.50714 | 0.51373 | 0.51317 | 0.50743 | 0.50780 | 0.52127 | 0.52381 | 0.54687 | 0.52424 | 0.54375 | 0.54167 | 0.53356 | 0.50836 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 2.1849 | 2.0414 | 2.1761 | 2.2082 | 2.0976 | 2.1262 | 2.2693 | 2.0309 | 2.2667 | 2.0793 | 2.0350 | 2.4023 | 2.1882 | 2.2256 | 2.1351 | 2.2671 | 2.1397 | 2.2348 | 2.1402 | 2.1913 | 2.2183 | 2.1765 | 2.1179 | 2.0749 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 3.0176 | 2.8084 | 2.9987 | 3.0285 | 2.8942 | 2.9450 | 3.1375 | 2.7921 | 3.1052 | 2.8453 | 2.7910 | 3.2949 | 3.0045 | 3.0540 | 2.9282 | 3.1056 | 2.9418 | 3.0613 | 2.9493 | 3.0116 | 3.0326 | 2.9666 | 2.8954 | 2.8288 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.5144 | 1.4151 | 1.5084 | 1.5305 | 1.4541 | 1.4739 | 1.5728 | 1.4077 | 1.5711 | 1.4411 | 1.4107 | 1.6652 | 1.5167 | 1.5426 | 1.4799 | 1.5714 | 1.4833 | 1.5491 | 1.4835 | 1.5188 | 1.5375 | 1.5086 | 1.4679 | 1.4382 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 2.0915 | 1.9467 | 2.0785 | 2.0990 | 2.0061 | 2.0414 | 2.1748 | 1.9353 | 2.1523 | 1.9722 | 1.9347 | 2.2840 | 2.0826 | 2.1167 | 2.0296 | 2.1527 | 2.0392 | 2.1220 | 2.0442 | 2.0874 | 2.1021 | 2.0562 | 2.0070 | 1.9608 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.23052 | 0.22642 | 0.21903 | 0.22642 | 0.22395 | 0.22067 | 0.22477 | 0.22888 | 0.22724 | 0.21739 | 0.20591 | 0.22806 | 0.22477 | 0.23052 | 0.23544 | 0.23134 | 0.21739 | 0.22149 | 0.22724 | 0.22970 | 0.21411 | 0.20837 | 0.23216 | 0.22806 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.23052 | 0.22642 | 0.21903 | 0.22642 | 0.22395 | 0.22067 | 0.22477 | 0.22888 | 0.22724 | 0.21739 | 0.20591 | 0.22806 | 0.22477 | 0.23052 | 0.23544 | 0.23134 | 0.21739 | 0.22149 | 0.22724 | 0.22970 | 0.21411 | 0.20837 | 0.23216 | 0.22806 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -4.3874 | -4.5934 | -4.5337 | -4.4877 | -4.5974 | -4.5356 | -4.5817 | -4.4317 | -4.3370 | -4.8795 | -4.7339 | -4.7477 | -4.5218 | -4.6259 | -4.5260 | -4.6679 | -4.6512 | -4.6602 | -4.6554 | -4.4538 | -4.4600 | -4.5947 | -4.5315 | -4.4138 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -4.3874 | -4.5934 | -4.5337 | -4.4877 | -4.5974 | -4.5356 | -4.5817 | -4.4317 | -4.3370 | -4.8795 | -4.7339 | -4.7477 | -4.5218 | -4.6259 | -4.5260 | -4.6679 | -4.6512 | -4.6602 | -4.6554 | -4.4538 | -4.4600 | -4.5947 | -4.5315 | -4.4138 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.24708 | 0.24049 | 0.24474 | 0.23620 | 0.23737 | 0.23329 | 0.23873 | 0.23563 | 0.23885 | 0.22854 | 0.22158 | 0.23971 | 0.23411 | 0.24271 | 0.24551 | 0.23774 | 0.22980 | 0.23203 | 0.23207 | 0.24083 | 0.22888 | 0.22474 | 0.23951 | 0.24361 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.24708 | 0.24049 | 0.24474 | 0.23620 | 0.23737 | 0.23329 | 0.23873 | 0.23563 | 0.23885 | 0.22854 | 0.22158 | 0.23971 | 0.23411 | 0.24271 | 0.24551 | 0.23774 | 0.22980 | 0.23203 | 0.23207 | 0.24083 | 0.22888 | 0.22474 | 0.23951 | 0.24361 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.55705 | 0.55124 | 0.55336 | 0.55656 | 0.55306 | 0.55522 | 0.56508 | 0.56054 | 0.56638 | 0.56133 | 0.56570 | 0.57519 | 0.56736 | 0.58075 | 0.57270 | 0.56785 | 0.55549 | 0.56516 | 0.56825 | 0.53965 | 0.55842 | 0.55589 | 0.56062 | 0.56043 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.56413 | 0.55842 | 0.56042 | 0.56421 | 0.56034 | 0.56293 | 0.57278 | 0.56770 | 0.57370 | 0.56862 | 0.57303 | 0.58272 | 0.57503 | 0.58805 | 0.58033 | 0.57511 | 0.56276 | 0.57231 | 0.57567 | 0.54664 | 0.56582 | 0.56310 | 0.56802 | 0.56797 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.76347 | 0.76304 | 0.75944 | 0.76265 | 0.76444 | 0.76737 | 0.76576 | 0.77512 | 0.76550 | 0.75642 | 0.76241 | 0.76197 | 0.76578 | 0.78754 | 0.77992 | 0.79258 | 0.75783 | 0.77689 | 0.77673 | 0.76458 | 0.75850 | 0.76153 | 0.75949 | 0.76248 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.76997 | 0.76958 | 0.76608 | 0.76937 | 0.77111 | 0.77400 | 0.77220 | 0.78186 | 0.77227 | 0.76269 | 0.76905 | 0.76859 | 0.77232 | 0.79413 | 0.78661 | 0.79934 | 0.76430 | 0.78370 | 0.78332 | 0.77124 | 0.76506 | 0.76795 | 0.76606 | 0.76904 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.9321 | 2.3246 | 3.0998 | 2.9553 | 2.9311 | 3.1412 | 2.9527 | 2.2616 | 2.5366 | 2.7680 | 2.4239 | 3.0941 | 2.7695 | 2.5329 | 2.5345 | 2.5092 | 2.5542 | 2.9432 | 2.2129 | 2.6596 | 2.7023 | 2.5503 | 2.8333 | 2.5430 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 5.8642 | 4.6493 | 6.1996 | 5.9105 | 5.8623 | 6.2824 | 5.9053 | 4.5232 | 5.0731 | 5.5359 | 4.8478 | 6.1883 | 5.5390 | 5.0657 | 5.0689 | 5.0184 | 5.1083 | 5.8865 | 4.4259 | 5.3191 | 5.4046 | 5.1006 | 5.6666 | 5.0860 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 2.0327 | 1.6118 | 2.1491 | 2.0480 | 2.0313 | 2.1765 | 2.0468 | 1.5680 | 1.7583 | 1.9191 | 1.6797 | 2.1438 | 1.9201 | 1.7551 | 1.7569 | 1.7390 | 1.7702 | 2.0400 | 1.5342 | 1.8430 | 1.8734 | 1.7676 | 1.9638 | 1.7630 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 4.0654 | 3.2235 | 4.2981 | 4.0961 | 4.0627 | 4.3530 | 4.0935 | 3.1360 | 3.5166 | 3.8382 | 3.3595 | 4.2877 | 3.8402 | 3.5102 | 3.5138 | 3.4780 | 3.5404 | 4.0800 | 3.0684 | 3.6859 | 3.7468 | 3.5352 | 3.9277 | 3.5261 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.07000 | 0.07000 | 0.07000 | 0.06000 | 0.10000 | 0.10000 | 0.10000 | 0.07000 | 0.11000 | 0.07000 | 0.06000 | 0.07000 | 0.07000 | 0.09000 | 0.08000 | 0.07000 | 0.07000 | 0.07000 | 0.10000 | 0.10000 | 0.10000 | 0.13000 | 0.05000 | 0.07000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.07000 | 0.07000 | 0.07000 | 0.06000 | 0.10000 | 0.10000 | 0.10000 | 0.07000 | 0.11000 | 0.07000 | 0.06000 | 0.07000 | 0.07000 | 0.09000 | 0.08000 | 0.07000 | 0.07000 | 0.07000 | 0.10000 | 0.10000 | 0.10000 | 0.13000 | 0.05000 | 0.07000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -4.0554 | -3.2073 | -4.2787 | -4.0877 | -4.0489 | -4.3344 | -4.0744 | -3.0975 | -3.4929 | -3.8224 | -3.3379 | -4.2647 | -3.8304 | -3.4992 | -3.4948 | -3.4553 | -3.5149 | -4.0589 | -3.0388 | -3.6659 | -3.7320 | -3.4843 | -3.8993 | -3.5076 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -4.0554 | -3.2073 | -4.2787 | -4.0877 | -4.0489 | -4.3344 | -4.0744 | -3.0975 | -3.4929 | -3.8224 | -3.3379 | -4.2647 | -3.8304 | -3.4992 | -3.4948 | -3.4553 | -3.5149 | -4.0589 | -3.0388 | -3.6659 | -3.7320 | -3.4843 | -3.8993 | -3.5076 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.08628 | 0.09390 | 0.09066 | 0.08477 | 0.09448 | 0.08943 | 0.09248 | 0.09363 | 0.09607 | 0.09194 | 0.09036 | 0.08716 | 0.08942 | 0.09585 | 0.09494 | 0.09040 | 0.09275 | 0.09001 | 0.09917 | 0.09540 | 0.09148 | 0.09371 | 0.09423 | 0.09174 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.08628 | 0.09390 | 0.09066 | 0.08477 | 0.09448 | 0.08943 | 0.09248 | 0.09363 | 0.09607 | 0.09194 | 0.09036 | 0.08716 | 0.08942 | 0.09585 | 0.09494 | 0.09040 | 0.09275 | 0.09001 | 0.09917 | 0.09540 | 0.09148 | 0.09371 | 0.09423 | 0.09174 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.87106 | 0.87207 | 0.87341 | 0.87522 | 0.87619 | 0.87724 | 0.87464 | 0.87725 | 0.87790 | 0.87407 | 0.87743 | 0.88171 | 0.87786 | 0.88129 | 0.88975 | 0.88699 | 0.87618 | 0.88626 | 0.87827 | 0.87051 | 0.86882 | 0.87481 | 0.87600 | 0.87950 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.88080 | 0.88163 | 0.88311 | 0.88462 | 0.88593 | 0.88695 | 0.88417 | 0.88693 | 0.88767 | 0.88369 | 0.88715 | 0.89135 | 0.88775 | 0.89087 | 0.89944 | 0.89662 | 0.88571 | 0.89602 | 0.88800 | 0.88012 | 0.87850 | 0.88452 | 0.88559 | 0.88916 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.81792 | 0.82149 | 0.81950 | 0.81478 | 0.82249 | 0.81956 | 0.83331 | 0.82750 | 0.83750 | 0.82823 | 0.83523 | 0.84062 | 0.82510 | 0.84835 | 0.84457 | 0.84838 | 0.82644 | 0.84109 | 0.83227 | 0.82266 | 0.82329 | 0.82510 | 0.83213 | 0.83129 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.82078 | 0.82426 | 0.82216 | 0.81764 | 0.82515 | 0.82221 | 0.83621 | 0.83009 | 0.84034 | 0.83109 | 0.83801 | 0.84336 | 0.82792 | 0.85120 | 0.84746 | 0.85130 | 0.82921 | 0.84376 | 0.83515 | 0.82538 | 0.82603 | 0.82788 | 0.83492 | 0.83416 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.82942 | 0.83492 | 0.83107 | 0.83089 | 0.83991 | 0.82783 | 0.85229 | 0.83871 | 0.84135 | 0.83472 | 0.84067 | 0.83762 | 0.82540 | 0.84997 | 0.85294 | 0.85831 | 0.84854 | 0.86631 | 0.84521 | 0.83246 | 0.83315 | 0.82855 | 0.82949 | 0.85133 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.87411 | 0.87973 | 0.87562 | 0.87499 | 0.88447 | 0.87162 | 0.89766 | 0.88346 | 0.88592 | 0.87876 | 0.88542 | 0.88238 | 0.86957 | 0.89511 | 0.89783 | 0.90409 | 0.89429 | 0.91302 | 0.89035 | 0.87724 | 0.87735 | 0.87296 | 0.87413 | 0.89735 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0394 | 1.0685 | 1.0363 | 1.0797 | 1.1389 | 1.0441 | 1.0594 | 1.0390 | 1.0354 | 1.0293 | 1.0363 | 1.0422 | 1.0387 | 1.0267 | 1.0291 | 1.0365 | 1.0349 | 1.0406 | 1.0457 | 1.0487 | 1.0387 | 1.0376 | 1.0442 | 1.0655 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0788 | 2.1369 | 2.0726 | 2.1595 | 2.2778 | 2.0881 | 2.1188 | 2.0781 | 2.0708 | 2.0585 | 2.0726 | 2.0845 | 2.0774 | 2.0534 | 2.0582 | 2.0731 | 2.0699 | 2.0812 | 2.0915 | 2.0974 | 2.0773 | 2.0752 | 2.0883 | 2.1310 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.72053 | 0.74059 | 0.71834 | 0.74837 | 0.78947 | 0.72371 | 0.73435 | 0.72024 | 0.71773 | 0.71351 | 0.71838 | 0.72248 | 0.72007 | 0.71171 | 0.71340 | 0.71853 | 0.71744 | 0.72136 | 0.72493 | 0.72695 | 0.72002 | 0.71934 | 0.72381 | 0.73854 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4411 | 1.4812 | 1.4367 | 1.4967 | 1.5789 | 1.4474 | 1.4687 | 1.4405 | 1.4355 | 1.4270 | 1.4368 | 1.4450 | 1.4401 | 1.4234 | 1.4268 | 1.4371 | 1.4349 | 1.4427 | 1.4499 | 1.4539 | 1.4400 | 1.4387 | 1.4476 | 1.4771 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24230 | 0.24697 | 0.25143 | 0.23953 | 0.24208 | 0.24633 | 0.24400 | 0.24081 | 0.25739 | 0.24123 | 0.24081 | 0.24336 | 0.24952 | 0.24060 | 0.25228 | 0.25313 | 0.24485 | 0.24888 | 0.24102 | 0.23847 | 0.25654 | 0.24803 | 0.24633 | 0.24463 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24230 | 0.24697 | 0.25143 | 0.23953 | 0.24208 | 0.24633 | 0.24400 | 0.24081 | 0.25739 | 0.24123 | 0.24081 | 0.24336 | 0.24952 | 0.24060 | 0.25228 | 0.25313 | 0.24485 | 0.24888 | 0.24102 | 0.23847 | 0.25654 | 0.24803 | 0.24633 | 0.24463 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3973 | -1.4042 | -1.3927 | -1.4088 | -1.4284 | -1.3974 | -1.4008 | -1.3952 | -1.3888 | -1.3918 | -1.3944 | -1.3959 | -1.3950 | -1.3908 | -1.3904 | -1.3903 | -1.3918 | -1.3949 | -1.3953 | -1.3973 | -1.3907 | -1.3896 | -1.3928 | -1.4040 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.4246 | -1.4597 | -1.4149 | -1.4748 | -1.5575 | -1.4258 | -1.4412 | -1.4160 | -1.3993 | -1.4056 | -1.4177 | -1.4229 | -1.4209 | -1.4040 | -1.4032 | -1.4086 | -1.4075 | -1.4183 | -1.4156 | -1.4246 | -1.4096 | -1.4029 | -1.4108 | -1.4462 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.24930 | 0.25022 | 0.25046 | 0.25079 | 0.25015 | 0.24945 | 0.24960 | 0.24923 | 0.25035 | 0.24965 | 0.24989 | 0.24987 | 0.24994 | 0.24989 | 0.25008 | 0.25074 | 0.24989 | 0.24974 | 0.24918 | 0.24930 | 0.25073 | 0.25042 | 0.24985 | 0.24868 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.24859 | 0.25029 | 0.25089 | 0.25111 | 0.24949 | 0.24891 | 0.24904 | 0.24844 | 0.25068 | 0.24931 | 0.24972 | 0.24967 | 0.24979 | 0.24967 | 0.25015 | 0.25145 | 0.24968 | 0.24949 | 0.24844 | 0.24854 | 0.25137 | 0.25077 | 0.24968 | 0.24741 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 1.1616 | 1.1594 | 1.1594 | 1.1608 | 1.1754 | 1.1653 | 1.1535 | 1.1547 | 1.1764 | 1.1629 | 1.1629 | 1.1888 | 1.1706 | 1.1813 | 1.1921 | 1.1949 | 1.1652 | 1.2066 | 1.1726 | 1.1748 | 1.1692 | 1.1646 | 1.1782 | 1.1817 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.2936 | 1.2916 | 1.2916 | 1.2927 | 1.3102 | 1.2984 | 1.2853 | 1.2853 | 1.3117 | 1.2934 | 1.2956 | 1.3249 | 1.3041 | 1.3154 | 1.3281 | 1.3318 | 1.2975 | 1.3447 | 1.3058 | 1.3106 | 1.3034 | 1.2978 | 1.3131 | 1.3174 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0203 | 1.0437 | 1.0416 | 1.1023 | 1.1816 | 1.0407 | 1.0433 | 1.0340 | 1.0345 | 1.0259 | 1.0493 | 1.0428 | 1.0197 | 1.0381 | 1.0300 | 1.0408 | 1.0261 | 1.0394 | 1.0452 | 1.0316 | 1.0525 | 1.0398 | 1.0330 | 1.0383 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0406 | 2.0874 | 2.0832 | 2.2046 | 2.3633 | 2.0814 | 2.0866 | 2.0680 | 2.0690 | 2.0518 | 2.0987 | 2.0856 | 2.0394 | 2.0762 | 2.0600 | 2.0816 | 2.0522 | 2.0788 | 2.0904 | 2.0632 | 2.1049 | 2.0796 | 2.0659 | 2.0766 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.70728 | 0.72347 | 0.72201 | 0.76408 | 0.81905 | 0.72139 | 0.72319 | 0.71677 | 0.71717 | 0.71117 | 0.72736 | 0.72285 | 0.70690 | 0.71960 | 0.71400 | 0.72148 | 0.71135 | 0.72045 | 0.72452 | 0.71507 | 0.72955 | 0.72079 | 0.71607 | 0.71974 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4146 | 1.4469 | 1.4440 | 1.5282 | 1.6381 | 1.4428 | 1.4464 | 1.4335 | 1.4343 | 1.4223 | 1.4547 | 1.4457 | 1.4138 | 1.4392 | 1.4280 | 1.4430 | 1.4227 | 1.4409 | 1.4490 | 1.4301 | 1.4591 | 1.4416 | 1.4321 | 1.4395 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.25941 | 0.26774 | 0.26002 | 0.23072 | 0.23720 | 0.24429 | 0.24830 | 0.24769 | 0.27391 | 0.26342 | 0.25200 | 0.26280 | 0.27884 | 0.26804 | 0.27791 | 0.25848 | 0.26928 | 0.24645 | 0.26188 | 0.27175 | 0.25231 | 0.26157 | 0.26712 | 0.26743 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.25941 | 0.26774 | 0.26002 | 0.23072 | 0.23720 | 0.24429 | 0.24830 | 0.24769 | 0.27391 | 0.26342 | 0.25200 | 0.26280 | 0.27884 | 0.26804 | 0.27791 | 0.25848 | 0.26928 | 0.24645 | 0.26188 | 0.27175 | 0.25231 | 0.26157 | 0.26712 | 0.26743 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3865 | -1.3912 | -1.3911 | -1.4188 | -1.4473 | -1.3938 | -1.3915 | -1.3901 | -1.3810 | -1.3867 | -1.3965 | -1.3921 | -1.3835 | -1.3888 | -1.3833 | -1.3910 | -1.3858 | -1.3922 | -1.3879 | -1.3852 | -1.3947 | -1.3864 | -1.3844 | -1.3852 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.3957 | -1.4236 | -1.4148 | -1.5067 | -1.6180 | -1.4201 | -1.4123 | -1.4100 | -1.3858 | -1.4003 | -1.4315 | -1.4237 | -1.3917 | -1.4066 | -1.3963 | -1.4119 | -1.3962 | -1.4150 | -1.4102 | -1.4012 | -1.4251 | -1.4017 | -1.3945 | -1.3997 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25109 | 0.25231 | 0.25130 | 0.25024 | 0.24876 | 0.25051 | 0.25055 | 0.25091 | 0.25267 | 0.25163 | 0.25049 | 0.25186 | 0.25210 | 0.25134 | 0.25302 | 0.25093 | 0.25156 | 0.25064 | 0.25226 | 0.25245 | 0.25072 | 0.25185 | 0.25198 | 0.25225 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25215 | 0.25445 | 0.25250 | 0.24966 | 0.24697 | 0.25087 | 0.25096 | 0.25160 | 0.25545 | 0.25335 | 0.25078 | 0.25335 | 0.25414 | 0.25268 | 0.25614 | 0.25193 | 0.25321 | 0.25114 | 0.25447 | 0.25482 | 0.25138 | 0.25357 | 0.25387 | 0.25444 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.99257 | 0.98223 | 0.98567 | 0.98126 | 0.98906 | 0.98439 | 0.99041 | 0.98857 | 0.99778 | 0.98961 | 0.98673 | 1.0022 | 0.99020 | 1.0065 | 1.0149 | 1.0153 | 0.98878 | 1.0137 | 0.99130 | 0.98514 | 0.98624 | 0.98390 | 1.0018 | 1.0078 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 1.0621 | 1.0499 | 1.0545 | 1.0485 | 1.0579 | 1.0518 | 1.0595 | 1.0571 | 1.0674 | 1.0578 | 1.0550 | 1.0726 | 1.0599 | 1.0765 | 1.0856 | 1.0865 | 1.0562 | 1.0849 | 1.0600 | 1.0538 | 1.0550 | 1.0523 | 1.0722 | 1.0789 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0197 | 1.0731 | 1.0459 | 1.1364 | 1.2042 | 1.0603 | 1.0685 | 1.0445 | 1.0401 | 1.0263 | 1.0569 | 1.0594 | 1.0278 | 1.0428 | 1.0321 | 1.0351 | 1.0387 | 1.0534 | 1.0419 | 1.0359 | 1.0559 | 1.0521 | 1.0314 | 1.0568 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0393 | 2.1461 | 2.0917 | 2.2727 | 2.4084 | 2.1207 | 2.1371 | 2.0890 | 2.0803 | 2.0526 | 2.1139 | 2.1187 | 2.0555 | 2.0856 | 2.0642 | 2.0702 | 2.0775 | 2.1069 | 2.0838 | 2.0717 | 2.1118 | 2.1042 | 2.0628 | 2.1137 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.70681 | 0.74388 | 0.72494 | 0.78769 | 0.83467 | 0.73496 | 0.74071 | 0.72406 | 0.72112 | 0.71143 | 0.73266 | 0.73429 | 0.71246 | 0.72289 | 0.71542 | 0.71751 | 0.72011 | 0.73023 | 0.72223 | 0.71810 | 0.73200 | 0.72934 | 0.71497 | 0.73253 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.4136 | 1.4878 | 1.4499 | 1.5754 | 1.6693 | 1.4699 | 1.4814 | 1.4481 | 1.4422 | 1.4229 | 1.4653 | 1.4686 | 1.4249 | 1.4458 | 1.4308 | 1.4350 | 1.4402 | 1.4605 | 1.4445 | 1.4362 | 1.4640 | 1.4587 | 1.4299 | 1.4651 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.27364 | 0.23887 | 0.24894 | 0.22424 | 0.21774 | 0.21807 | 0.21677 | 0.24179 | 0.24959 | 0.24699 | 0.23497 | 0.24212 | 0.23822 | 0.24472 | 0.27494 | 0.26617 | 0.24472 | 0.23269 | 0.25837 | 0.24082 | 0.23432 | 0.23757 | 0.26292 | 0.24179 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.27364 | 0.23887 | 0.24894 | 0.22424 | 0.21774 | 0.21807 | 0.21677 | 0.24179 | 0.24959 | 0.24699 | 0.23497 | 0.24212 | 0.23822 | 0.24472 | 0.27494 | 0.26617 | 0.24472 | 0.23269 | 0.25837 | 0.24082 | 0.23432 | 0.23757 | 0.26292 | 0.24179 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3858 | -1.4131 | -1.3972 | -1.4456 | -1.4718 | -1.4066 | -1.4063 | -1.3971 | -1.3885 | -1.3886 | -1.4036 | -1.4056 | -1.3905 | -1.3961 | -1.3870 | -1.3856 | -1.3944 | -1.4017 | -1.3891 | -1.3915 | -1.3985 | -1.3969 | -1.3860 | -1.3963 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.3953 | -1.4668 | -1.4219 | -1.5529 | -1.6475 | -1.4477 | -1.4462 | -1.4257 | -1.3973 | -1.3990 | -1.4403 | -1.4468 | -1.4028 | -1.4171 | -1.3997 | -1.4039 | -1.4137 | -1.4319 | -1.4083 | -1.4070 | -1.4308 | -1.4186 | -1.3941 | -1.4270 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.25135 | 0.24691 | 0.24913 | 0.24349 | 0.24087 | 0.24754 | 0.24734 | 0.24926 | 0.25027 | 0.25046 | 0.24823 | 0.24834 | 0.24995 | 0.24894 | 0.25140 | 0.25254 | 0.24942 | 0.24804 | 0.25136 | 0.25002 | 0.24948 | 0.24876 | 0.25117 | 0.25014 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.25269 | 0.24419 | 0.24844 | 0.23860 | 0.23354 | 0.24494 | 0.24443 | 0.24816 | 0.25059 | 0.25091 | 0.24663 | 0.24706 | 0.24988 | 0.24786 | 0.25297 | 0.25481 | 0.24883 | 0.24606 | 0.25265 | 0.25004 | 0.24886 | 0.24757 | 0.25239 | 0.25024 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.5024 | 1.4807 | 1.4920 | 1.4899 | 1.4855 | 1.4938 | 1.4924 | 1.4878 | 1.4977 | 1.4807 | 1.4982 | 1.5029 | 1.4770 | 1.5343 | 1.5254 | 1.5245 | 1.4995 | 1.5335 | 1.5088 | 1.4875 | 1.4851 | 1.5110 | 1.5145 | 1.5296 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.8741 | 1.8437 | 1.8595 | 1.8540 | 1.8517 | 1.8614 | 1.8589 | 1.8501 | 1.8641 | 1.8405 | 1.8689 | 1.8719 | 1.8336 | 1.9113 | 1.9022 | 1.8990 | 1.8728 | 1.9118 | 1.8854 | 1.8557 | 1.8509 | 1.8887 | 1.8895 | 1.9073 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0347 | 1.0659 | 1.0554 | 1.1526 | 1.2248 | 1.0636 | 1.0569 | 1.0424 | 1.0382 | 1.0277 | 1.0494 | 1.0595 | 1.0336 | 1.0366 | 1.0321 | 1.0512 | 1.0345 | 1.0788 | 1.0365 | 1.0637 | 1.0604 | 1.0594 | 1.0470 | 1.0616 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0693 | 2.1317 | 2.1107 | 2.3053 | 2.4497 | 2.1272 | 2.1137 | 2.0848 | 2.0764 | 2.0555 | 2.0989 | 2.1189 | 2.0673 | 2.0731 | 2.0643 | 2.1024 | 2.0690 | 2.1576 | 2.0730 | 2.1274 | 2.1209 | 2.1189 | 2.0939 | 2.1232 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.71723 | 0.73879 | 0.73151 | 0.79898 | 0.84903 | 0.73729 | 0.73262 | 0.72261 | 0.71973 | 0.71241 | 0.72744 | 0.73438 | 0.71646 | 0.71855 | 0.71548 | 0.72864 | 0.71713 | 0.74776 | 0.71851 | 0.73729 | 0.73513 | 0.73435 | 0.72573 | 0.73588 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4345 | 1.4776 | 1.4630 | 1.5980 | 1.6981 | 1.4746 | 1.4652 | 1.4452 | 1.4395 | 1.4248 | 1.4549 | 1.4688 | 1.4329 | 1.4371 | 1.4310 | 1.4573 | 1.4343 | 1.4955 | 1.4370 | 1.4746 | 1.4703 | 1.4687 | 1.4515 | 1.4718 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.25447 | 0.23989 | 0.26209 | 0.23956 | 0.21438 | 0.23559 | 0.22233 | 0.24652 | 0.27104 | 0.27469 | 0.24884 | 0.25580 | 0.25944 | 0.26673 | 0.26839 | 0.26375 | 0.26508 | 0.23592 | 0.27435 | 0.25149 | 0.24387 | 0.25215 | 0.26474 | 0.26673 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.25447 | 0.23989 | 0.26209 | 0.23956 | 0.21438 | 0.23559 | 0.22233 | 0.24652 | 0.27104 | 0.27469 | 0.24884 | 0.25580 | 0.25944 | 0.26673 | 0.26839 | 0.26375 | 0.26508 | 0.23592 | 0.27435 | 0.25149 | 0.24387 | 0.25215 | 0.26474 | 0.26673 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3926 | -1.4047 | -1.3972 | -1.4465 | -1.4744 | -1.4059 | -1.4021 | -1.3909 | -1.3811 | -1.3836 | -1.3946 | -1.3999 | -1.3887 | -1.3882 | -1.3821 | -1.3850 | -1.3864 | -1.4050 | -1.3829 | -1.4001 | -1.3941 | -1.3883 | -1.3871 | -1.3906 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.4125 | -1.4518 | -1.4296 | -1.5709 | -1.6767 | -1.4495 | -1.4347 | -1.4132 | -1.3843 | -1.3969 | -1.4278 | -1.4423 | -1.4101 | -1.4036 | -1.3949 | -1.4124 | -1.3989 | -1.4435 | -1.3963 | -1.4388 | -1.4225 | -1.4106 | -1.4060 | -1.4247 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.25022 | 0.24931 | 0.25032 | 0.24635 | 0.24286 | 0.24821 | 0.24816 | 0.25104 | 0.25236 | 0.25285 | 0.25123 | 0.25059 | 0.25186 | 0.25123 | 0.25328 | 0.25391 | 0.25153 | 0.24800 | 0.25317 | 0.24982 | 0.25064 | 0.25196 | 0.25214 | 0.25284 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.25053 | 0.24873 | 0.25086 | 0.24411 | 0.23599 | 0.24649 | 0.24633 | 0.25202 | 0.25474 | 0.25587 | 0.25248 | 0.25133 | 0.25356 | 0.25237 | 0.25654 | 0.25753 | 0.25311 | 0.24616 | 0.25666 | 0.24972 | 0.25104 | 0.25351 | 0.25412 | 0.25551 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.52065 | 0.52742 | 0.51573 | 0.51440 | 0.52055 | 0.50529 | 0.52269 | 0.52431 | 0.53084 | 0.50319 | 0.53410 | 0.53296 | 0.51644 | 0.52214 | 0.53937 | 0.53952 | 0.50040 | 0.52781 | 0.50182 | 0.52596 | 0.52162 | 0.50992 | 0.51757 | 0.51688 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.52363 | 0.53038 | 0.51847 | 0.51730 | 0.52345 | 0.50806 | 0.52561 | 0.52729 | 0.53386 | 0.50599 | 0.53702 | 0.53586 | 0.51935 | 0.52506 | 0.54243 | 0.54254 | 0.50308 | 0.53079 | 0.50446 | 0.52896 | 0.52456 | 0.51283 | 0.52039 | 0.51977 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.37193 | 0.39578 | 0.39215 | 0.40043 | 0.38454 | 0.38276 | 0.39341 | 0.40908 | 0.40005 | 0.38927 | 0.40191 | 0.40005 | 0.38670 | 0.39350 | 0.39438 | 0.39721 | 0.38897 | 0.39707 | 0.40732 | 0.39676 | 0.39667 | 0.39011 | 0.39614 | 0.39750 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.37336 | 0.39733 | 0.39362 | 0.40189 | 0.38622 | 0.38438 | 0.39491 | 0.41064 | 0.40158 | 0.39074 | 0.40348 | 0.40157 | 0.38815 | 0.39502 | 0.39596 | 0.39879 | 0.39036 | 0.39862 | 0.40891 | 0.39829 | 0.39818 | 0.39164 | 0.39772 | 0.39900 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.71055 | 0.70788 | 0.69876 | 0.70372 | 0.69847 | 0.72124 | 0.72698 | 0.75101 | 0.72792 | 0.71134 | 0.75321 | 0.72956 | 0.72578 | 0.74280 | 0.72489 | 0.73952 | 0.75053 | 0.73105 | 0.73432 | 0.69726 | 0.71001 | 0.69618 | 0.76108 | 0.72651 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.71563 | 0.71288 | 0.70370 | 0.70877 | 0.70345 | 0.72639 | 0.73206 | 0.75634 | 0.73303 | 0.71627 | 0.75865 | 0.73486 | 0.73095 | 0.74783 | 0.72997 | 0.74476 | 0.75587 | 0.73622 | 0.73953 | 0.70220 | 0.71504 | 0.70098 | 0.76656 | 0.73148 |
| eval/lm/c4_en-validation/CE loss | lower | 3.2103 | 3.2198 | 3.2183 | 3.2197 | 3.2217 | 3.2248 | 3.2351 | 3.2368 | 3.2412 | 3.2251 | 3.2307 | 3.2484 | 3.2282 | 3.2596 | 3.2638 | 3.2725 | 3.2224 | 3.2576 | 3.2347 | 3.2256 | 3.2205 | 3.2179 | 3.2276 | 3.2394 |
| eval/lm/c4_en-validation/PPL | lower | 24.79 | 25.02 | 24.99 | 25.02 | 25.07 | 25.15 | 25.41 | 25.45 | 25.56 | 25.16 | 25.30 | 25.75 | 25.23 | 26.04 | 26.15 | 26.38 | 25.09 | 25.99 | 25.40 | 25.17 | 25.04 | 24.98 | 25.22 | 25.52 |
| eval/lm/dolma_books-validation/CE loss | lower | 3.1573 | 3.1735 | 3.1715 | 3.1610 | 3.1800 | 3.1726 | 3.1836 | 3.1937 | 3.1938 | 3.1676 | 3.1781 | 3.1994 | 3.1706 | 3.2075 | 3.2122 | 3.2253 | 3.1765 | 3.2246 | 3.1837 | 3.1766 | 3.1699 | 3.1706 | 3.1758 | 3.1904 |
| eval/lm/dolma_books-validation/PPL | lower | 23.51 | 23.89 | 23.84 | 23.59 | 24.05 | 23.87 | 24.13 | 24.38 | 24.38 | 23.75 | 24.00 | 24.52 | 23.82 | 24.72 | 24.83 | 25.16 | 23.96 | 25.14 | 24.14 | 23.97 | 23.80 | 23.82 | 23.95 | 24.30 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.3409 | 3.3502 | 3.3503 | 3.3496 | 3.3535 | 3.3565 | 3.3664 | 3.3687 | 3.3736 | 3.3566 | 3.3634 | 3.3775 | 3.3585 | 3.3922 | 3.3931 | 3.4013 | 3.3545 | 3.3899 | 3.3667 | 3.3567 | 3.3509 | 3.3507 | 3.3599 | 3.3709 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 28.24 | 28.51 | 28.51 | 28.49 | 28.60 | 28.69 | 28.97 | 29.04 | 29.18 | 28.69 | 28.89 | 29.30 | 28.75 | 29.73 | 29.76 | 30.00 | 28.63 | 29.66 | 28.98 | 28.70 | 28.53 | 28.52 | 28.79 | 29.11 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.3764 | 2.3827 | 2.3836 | 2.3860 | 2.3905 | 2.3901 | 2.3971 | 2.3994 | 2.4091 | 2.3856 | 2.3929 | 2.4015 | 2.3848 | 2.4180 | 2.4214 | 2.4299 | 2.3878 | 2.4216 | 2.3996 | 2.3924 | 2.3881 | 2.3910 | 2.3878 | 2.3927 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 10.77 | 10.83 | 10.84 | 10.87 | 10.92 | 10.91 | 10.99 | 11.02 | 11.12 | 10.87 | 10.94 | 11.04 | 10.86 | 11.22 | 11.26 | 11.36 | 10.89 | 11.26 | 11.02 | 10.94 | 10.89 | 10.92 | 10.89 | 10.94 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.4978 | 3.5034 | 3.5008 | 3.5027 | 3.5054 | 3.5107 | 3.5206 | 3.5208 | 3.5285 | 3.5129 | 3.5165 | 3.5343 | 3.5163 | 3.5419 | 3.5436 | 3.5524 | 3.5058 | 3.5389 | 3.5183 | 3.5102 | 3.5081 | 3.5017 | 3.5136 | 3.5245 |
| eval/lm/dolma_reddit-validation/PPL | lower | 33.04 | 33.23 | 33.14 | 33.21 | 33.29 | 33.47 | 33.80 | 33.81 | 34.07 | 33.55 | 33.67 | 34.27 | 33.66 | 34.53 | 34.59 | 34.90 | 33.31 | 34.43 | 33.73 | 33.45 | 33.38 | 33.17 | 33.57 | 33.94 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.5756 | 1.5777 | 1.5875 | 1.5813 | 1.5818 | 1.5928 | 1.5994 | 1.5961 | 1.6111 | 1.5877 | 1.5888 | 1.6194 | 1.5924 | 1.6202 | 1.6196 | 1.6362 | 1.5904 | 1.6354 | 1.5960 | 1.6013 | 1.5943 | 1.5968 | 1.5841 | 1.5886 |
| eval/lm/dolma_stack-validation/PPL | lower | 4.8336 | 4.8438 | 4.8914 | 4.8615 | 4.8639 | 4.9174 | 4.9500 | 4.9336 | 5.0083 | 4.8924 | 4.8978 | 5.0498 | 4.9154 | 5.0542 | 5.0510 | 5.1354 | 4.9056 | 5.1315 | 4.9334 | 4.9595 | 4.9247 | 4.9371 | 4.8751 | 4.8967 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.8809 | 2.8879 | 2.8883 | 2.8900 | 2.8996 | 2.8939 | 2.9088 | 2.9118 | 2.9167 | 2.8979 | 2.9065 | 2.9194 | 2.8991 | 2.9329 | 2.9380 | 2.9455 | 2.8917 | 2.9263 | 2.9067 | 2.8896 | 2.8829 | 2.8829 | 2.8989 | 2.9178 |
| eval/lm/dolma_wiki-validation/PPL | lower | 17.83 | 17.96 | 17.96 | 17.99 | 18.17 | 18.06 | 18.34 | 18.39 | 18.48 | 18.14 | 18.29 | 18.53 | 18.16 | 18.78 | 18.88 | 19.02 | 18.02 | 18.66 | 18.30 | 17.99 | 17.87 | 17.87 | 18.15 | 18.50 |
| eval/lm/ice-validation/CE loss | lower | 3.2932 | 3.2956 | 3.3062 | 3.3049 | 3.2988 | 3.3133 | 3.3073 | 3.3271 | 3.3340 | 3.2944 | 3.3121 | 3.3366 | 3.2995 | 3.3610 | 3.3687 | 3.3655 | 3.3199 | 3.3527 | 3.3128 | 3.3142 | 3.3058 | 3.3117 | 3.3115 | 3.3336 |
| eval/lm/ice-validation/PPL | lower | 26.93 | 26.99 | 27.28 | 27.25 | 27.08 | 27.48 | 27.31 | 27.86 | 28.05 | 26.96 | 27.44 | 28.12 | 27.10 | 28.82 | 29.04 | 28.95 | 27.66 | 28.58 | 27.46 | 27.50 | 27.27 | 27.43 | 27.43 | 28.04 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 3.3110 | 3.3144 | 3.3200 | 3.3079 | 3.3114 | 3.3207 | 3.3351 | 3.3391 | 3.3453 | 3.3185 | 3.3337 | 3.3433 | 3.3214 | 3.3528 | 3.3526 | 3.3595 | 3.3102 | 3.3397 | 3.3259 | 3.3193 | 3.3096 | 3.3069 | 3.3209 | 3.3317 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 27.41 | 27.51 | 27.66 | 27.33 | 27.42 | 27.68 | 28.08 | 28.19 | 28.37 | 27.62 | 28.04 | 28.31 | 27.70 | 28.58 | 28.58 | 28.77 | 27.39 | 28.21 | 27.82 | 27.64 | 27.37 | 27.30 | 27.69 | 27.99 |
| eval/lm/pile-validation/CE loss | lower | 2.4851 | 2.4907 | 2.4909 | 2.4909 | 2.4949 | 2.4962 | 2.5061 | 2.5105 | 2.5167 | 2.4919 | 2.4994 | 2.5154 | 2.4950 | 2.5305 | 2.5374 | 2.5425 | 2.4951 | 2.5329 | 2.5092 | 2.4989 | 2.4937 | 2.4957 | 2.4976 | 2.5040 |
| eval/lm/pile-validation/PPL | lower | 12.00 | 12.07 | 12.07 | 12.07 | 12.12 | 12.14 | 12.26 | 12.31 | 12.39 | 12.08 | 12.17 | 12.37 | 12.12 | 12.56 | 12.65 | 12.71 | 12.12 | 12.59 | 12.30 | 12.17 | 12.11 | 12.13 | 12.15 | 12.23 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.8686 | 2.8671 | 2.8764 | 2.8753 | 2.8924 | 2.8921 | 2.9046 | 2.9103 | 2.9313 | 2.8923 | 2.8934 | 2.9146 | 2.8833 | 2.9402 | 2.9388 | 2.9568 | 2.8930 | 2.9416 | 2.9018 | 2.8959 | 2.8985 | 2.8980 | 2.8956 | 2.9108 |
| eval/lm/wikitext_103-validation/PPL | lower | 17.61 | 17.59 | 17.75 | 17.73 | 18.04 | 18.03 | 18.26 | 18.36 | 18.75 | 18.03 | 18.05 | 18.44 | 17.87 | 18.92 | 18.89 | 19.24 | 18.05 | 18.95 | 18.21 | 18.10 | 18.15 | 18.14 | 18.09 | 18.37 |
| throughput/in-loop eval batches | see metric | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 |
| throughput/in-loop eval time (s) | see metric | 222.7 | 219.7 | 219.0 | 213.6 | 214.4 | 211.4 | 226.4 | 234.7 | 228.3 | 239.6 | 245.2 | 234.0 | 241.1 | 232.8 | 230.3 | 221.0 | 208.3 | 263.1 | 206.1 | 270.9 | 328.0 | 211.5 | 239.4 | 279.6 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| int-275m-cx2-intd256e8k-lr1.6e-3-r1<br>`igsm7yj9` | finished | original | 7932346368.0 | 20173 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/igsm7yj9) |
| int-275m-cx2-intd256e8k-lr3.2e-3-r1<br>`1pwfw2jo` | finished | original | 7932346368.0 | 20173 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1pwfw2jo) |
| int-275m-cx2-intd256e8k-lr8e-4-r1<br>`suaysv7u` | finished | original | 7932346368.0 | 20173 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/suaysv7u) |
| int-275m-cx2-intw256e8k-lr1.6e-3-r1<br>`6porpbo2` | finished | original | 8125808640.0 | 20665 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6porpbo2) |
| int-275m-cx2-intw256e8k-lr3.2e-3-r1<br>`0f782vrw` | finished | original | 8125808640.0 | 20665 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/0f782vrw) |
| int-275m-cx2-intw256e8k-lr8e-4-r1<br>`o2bdr3gw` | finished | original | 8125808640.0 | 20665 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/o2bdr3gw) |
| q3-275m-cx2-q3am128e8k-lr1.8e-3-r1<br>`4hy9tf4o` | finished | original | 8086487040.0 | 20565 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4hy9tf4o) |
| q3-275m-cx2-q3am128e8k-lr3.6e-3-r1<br>`5bv2y0fp` | finished | original | 8086487040.0 | 20565 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5bv2y0fp) |
| q3-275m-cx2-q3am128e8k-lr9e-4-r1<br>`5yiwgg3x` | finished | original | 8086487040.0 | 20565 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5yiwgg3x) |
| q3-275m-cx2-q3td128e8k-lr1.8e-3-r1<br>`uso06e7k` | finished | original | 8055422976.0 | 20486 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/uso06e7k) |
| q3-275m-cx2-q3td128e8k-lr3.6e-3-r1<br>`zadoal7a` | finished | original | 8055422976.0 | 20486 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/zadoal7a) |
| q3-275m-cx2-q3td128e8k-lr4.5e-4-r1<br>`9zhx2ws4` | finished | original | 8055422976.0 | 20486 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9zhx2ws4) |
| q3-275m-cx2-q3td128e8k-lr9e-4-r1<br>`uvsmf0rw` | finished | original | 8055422976.0 | 20486 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/uvsmf0rw) |
| se-275m-cx2-se0m9-lr1.8e-3-r2<br>`97xdkfc4` | finished | original | 8055422976.0 | 20486 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/97xdkfc4) |
| se-275m-cx2-se0m9-lr3.6e-3-r2<br>`2oirkw3f` | finished | original | 8055422976.0 | 20486 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/2oirkw3f) |
| se-275m-cx2-se0m9-lr9e-4-r2<br>`rb52lk9m` | finished | original | 8055422976.0 | 20486 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rb52lk9m) |
| sp-275m-cx2-sp192e4k-lr1.8e-3-r1<br>`r78rwdr8` | finished | original | 8104181760.0 | 20610 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/r78rwdr8) |
| sp-275m-cx2-sp192e4k-lr2.25e-4-r2<br>`ylz3dxx5` | finished | original | 8104181760.0 | 20610 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ylz3dxx5) |
| sp-275m-cx2-sp192e4k-lr3.6e-3-r1<br>`5bwqn7br` | finished | original | 8104181760.0 | 20610 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5bwqn7br) |
| sp-275m-cx2-sp192e4k-lr4.5e-4-r2<br>`7hzc9eo9` | finished | original | 8104181760.0 | 20610 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/7hzc9eo9) |
| sp-275m-cx2-sp192e4k-lr6e-4-r2<br>`06d0ggb0` | finished | original | 8104181760.0 | 20610 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/06d0ggb0) |
| sp-275m-cx2-sp192e4k-lr9e-4-r1<br>`m21y2jzg` | finished | original | 8104181760.0 | 20610 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/m21y2jzg) |
| sp-275m-cx2-sp96e4k-lr1.8e-3-r1<br>`tajjw92i` | finished | original | 8071544832.0 | 20527 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/tajjw92i) |
| sp-275m-cx2-sp96e4k-lr3.6e-3-r1<br>`xf9ao51p` | finished | original | 8071544832.0 | 20527 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/xf9ao51p) |

## unknown Cx4

| metric | direction | int-275m-cx4-intd256e8k-lr1.6e-3-r1<br>`36clxd8s` | int-275m-cx4-intd256e8k-lr3.2e-3-r1<br>`bwqkxa1r` | int-275m-cx4-intd256e8k-lr8e-4-r1<br>`nvxaejv3` | int-275m-cx4-intw256e8k-lr1.6e-3-r1<br>`ttjquo05` | int-275m-cx4-intw256e8k-lr3.2e-3-r1<br>`5u03fshf` | int-275m-cx4-intw256e8k-lr4e-4-r1<br>`n1gjknwg` | int-275m-cx4-intw256e8k-lr8e-4-r1<br>`9n3xk8gs` | q3-275m-cx4-q3am128e8k-lr1.6e-3-r1<br>`h12fasg0` | q3-275m-cx4-q3am128e8k-lr3.2e-3-r1<br>`eihks7b2` | q3-275m-cx4-q3am128e8k-lr8e-4-r1<br>`1wke63zk` | q3-275m-cx4-q3td128e8k-lr1.6e-3-r1<br>`bndawrpx` | q3-275m-cx4-q3td128e8k-lr3.2e-3-r1<br>`x06q7vzv` | q3-275m-cx4-q3td128e8k-lr4e-4-r1<br>`unnqsh5j` | q3-275m-cx4-q3td128e8k-lr8e-4-r1<br>`u5m4nxf2` | se-275m-cx4-se0m9-lr1.6e-3-r2<br>`6l09gyle` | se-275m-cx4-se0m9-lr3.2e-3-r2<br>`8bmutaw7` | se-275m-cx4-se0m9-lr8e-4-r2<br>`v9yomn1p` | sp-275m-cx4-sp192e4k-lr1.6e-3-r1<br>`frw2gqmk` | sp-275m-cx4-sp192e4k-lr3.2e-3-r1<br>`ra0oqtkh` | sp-275m-cx4-sp192e4k-lr4e-4-r1<br>`yvbc691e` | sp-275m-cx4-sp192e4k-lr8e-4-r1<br>`35g1knkr` | sp-275m-cx4-sp96e4k-lr1.6e-3-r1<br>`pnmr2zza` | sp-275m-cx4-sp96e4k-lr3.2e-3-r1<br>`k9v8ho4s` | sp-275m-cx4-sp96e4k-lr8e-4-r1<br>`u7londs6` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.93646 | 0.94108 | 0.93813 | 0.94967 | 0.95425 | 0.96330 | 0.94703 | 0.96772 | 0.95719 | 0.97612 | 0.95660 | 0.96045 | 0.97488 | 0.96943 | 0.98516 | 0.96114 | 0.98648 | 0.96297 | 0.96605 | 0.97780 | 0.95256 | 0.95255 | 0.96779 | 0.96433 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 1.0231 | 1.0287 | 1.0253 | 1.0376 | 1.0446 | 1.0519 | 1.0367 | 1.0603 | 1.0476 | 1.0684 | 1.0461 | 1.0499 | 1.0679 | 1.0612 | 1.0785 | 1.0513 | 1.0783 | 1.0560 | 1.0565 | 1.0695 | 1.0436 | 1.0423 | 1.0590 | 1.0539 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0145 | 1.0249 | 1.0226 | 1.0147 | 1.0348 | 1.0275 | 1.0135 | 1.0768 | 1.0529 | 1.0253 | 1.0189 | 1.0137 | 1.0281 | 1.0126 | 1.0852 | 1.0379 | 1.0242 | 1.0093 | 1.0181 | 1.0150 | 1.0072 | 1.0107 | 1.0372 | 1.0236 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0291 | 2.0498 | 2.0452 | 2.0295 | 2.0697 | 2.0550 | 2.0270 | 2.1536 | 2.1058 | 2.0505 | 2.0379 | 2.0275 | 2.0562 | 2.0252 | 2.1705 | 2.0758 | 2.0484 | 2.0186 | 2.0363 | 2.0300 | 2.0145 | 2.0213 | 2.0743 | 2.0472 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.70334 | 0.71043 | 0.70884 | 0.70339 | 0.71735 | 0.71233 | 0.70257 | 0.74637 | 0.72977 | 0.71072 | 0.70637 | 0.70284 | 0.71269 | 0.70195 | 0.75223 | 0.71948 | 0.70997 | 0.69972 | 0.70579 | 0.70357 | 0.69827 | 0.70053 | 0.71895 | 0.70953 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.4067 | 1.4209 | 1.4177 | 1.4068 | 1.4347 | 1.4247 | 1.4051 | 1.4927 | 1.4595 | 1.4214 | 1.4127 | 1.4057 | 1.4254 | 1.4039 | 1.5045 | 1.4390 | 1.4199 | 1.3994 | 1.4116 | 1.4071 | 1.3965 | 1.4011 | 1.4379 | 1.4191 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.27816 | 0.25000 | 0.26024 | 0.23379 | 0.24147 | 0.25171 | 0.24573 | 0.25256 | 0.26365 | 0.24232 | 0.25000 | 0.25512 | 0.26792 | 0.26451 | 0.26451 | 0.24573 | 0.26451 | 0.26792 | 0.24488 | 0.28157 | 0.25512 | 0.25256 | 0.26195 | 0.26109 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.27816 | 0.25000 | 0.26024 | 0.23379 | 0.24147 | 0.25171 | 0.24573 | 0.25256 | 0.26365 | 0.24232 | 0.25000 | 0.25512 | 0.26792 | 0.26451 | 0.26451 | 0.24573 | 0.26451 | 0.26792 | 0.24488 | 0.28157 | 0.25512 | 0.25256 | 0.26195 | 0.26109 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.3922 | -1.4000 | -1.4066 | -1.3989 | -1.4288 | -1.4178 | -1.3963 | -1.4852 | -1.4468 | -1.4127 | -1.4057 | -1.3907 | -1.4011 | -1.3941 | -1.4963 | -1.4297 | -1.4003 | -1.3955 | -1.4063 | -1.3953 | -1.3904 | -1.3893 | -1.4318 | -1.4093 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.3922 | -1.4000 | -1.4066 | -1.3989 | -1.4288 | -1.4178 | -1.3963 | -1.4852 | -1.4468 | -1.4127 | -1.4057 | -1.3907 | -1.4011 | -1.3941 | -1.4963 | -1.4297 | -1.4003 | -1.3955 | -1.4063 | -1.3953 | -1.3904 | -1.3893 | -1.4318 | -1.4093 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.25366 | 0.25077 | 0.25184 | 0.25043 | 0.24728 | 0.25035 | 0.24993 | 0.25019 | 0.25067 | 0.25027 | 0.24899 | 0.25150 | 0.25184 | 0.25181 | 0.25119 | 0.24853 | 0.25104 | 0.25267 | 0.24896 | 0.25286 | 0.25167 | 0.25190 | 0.25083 | 0.25334 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.25366 | 0.25077 | 0.25184 | 0.25043 | 0.24728 | 0.25035 | 0.24993 | 0.25019 | 0.25067 | 0.25027 | 0.24899 | 0.25150 | 0.25184 | 0.25181 | 0.25119 | 0.24853 | 0.25104 | 0.25267 | 0.24896 | 0.25286 | 0.25167 | 0.25190 | 0.25083 | 0.25334 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.72957 | 0.75253 | 0.73749 | 0.75071 | 0.74942 | 0.76146 | 0.76699 | 0.76374 | 0.75610 | 0.78200 | 0.76098 | 0.76125 | 0.77488 | 0.76611 | 0.80177 | 0.75758 | 0.79935 | 0.75904 | 0.76306 | 0.78159 | 0.75399 | 0.75149 | 0.76468 | 0.76514 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.79399 | 0.81866 | 0.80247 | 0.81721 | 0.81638 | 0.82865 | 0.83654 | 0.83200 | 0.82378 | 0.85188 | 0.82920 | 0.82904 | 0.84376 | 0.83395 | 0.87397 | 0.82401 | 0.87116 | 0.82750 | 0.83099 | 0.85208 | 0.82134 | 0.81847 | 0.83358 | 0.83318 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0219 | 1.0293 | 1.0299 | 1.0221 | 1.0396 | 1.0293 | 1.0208 | 1.0753 | 1.0370 | 1.0214 | 1.0107 | 1.0177 | 1.0436 | 1.0246 | 1.0624 | 1.0295 | 1.0247 | 1.0208 | 1.0181 | 1.0296 | 1.0172 | 1.0225 | 1.0446 | 1.0341 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0439 | 2.0587 | 2.0598 | 2.0443 | 2.0792 | 2.0586 | 2.0415 | 2.1507 | 2.0739 | 2.0427 | 2.0213 | 2.0354 | 2.0872 | 2.0493 | 2.1247 | 2.0591 | 2.0494 | 2.0416 | 2.0362 | 2.0593 | 2.0343 | 2.0449 | 2.0891 | 2.0683 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.70844 | 0.71347 | 0.71397 | 0.70857 | 0.72069 | 0.71353 | 0.70761 | 0.74530 | 0.71881 | 0.70805 | 0.70066 | 0.70554 | 0.72342 | 0.71031 | 0.73637 | 0.71365 | 0.71039 | 0.70765 | 0.70577 | 0.71376 | 0.70511 | 0.70879 | 0.72403 | 0.71687 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4169 | 1.4269 | 1.4279 | 1.4171 | 1.4414 | 1.4271 | 1.4152 | 1.4906 | 1.4376 | 1.4161 | 1.4013 | 1.4111 | 1.4468 | 1.4206 | 1.4727 | 1.4273 | 1.4208 | 1.4153 | 1.4115 | 1.4275 | 1.4102 | 1.4176 | 1.4481 | 1.4337 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.24579 | 0.25000 | 0.23822 | 0.25463 | 0.23737 | 0.24958 | 0.24916 | 0.26010 | 0.24116 | 0.25505 | 0.25337 | 0.25758 | 0.25253 | 0.24579 | 0.24621 | 0.25463 | 0.23569 | 0.23022 | 0.24495 | 0.24032 | 0.24790 | 0.23653 | 0.24832 | 0.24663 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.24579 | 0.25000 | 0.23822 | 0.25463 | 0.23737 | 0.24958 | 0.24916 | 0.26010 | 0.24116 | 0.25505 | 0.25337 | 0.25758 | 0.25253 | 0.24579 | 0.24621 | 0.25463 | 0.23569 | 0.23022 | 0.24495 | 0.24032 | 0.24790 | 0.23653 | 0.24832 | 0.24663 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.4045 | -1.4074 | -1.4156 | -1.4092 | -1.4358 | -1.4196 | -1.4049 | -1.4821 | -1.4243 | -1.4070 | -1.3927 | -1.3963 | -1.4161 | -1.4093 | -1.4640 | -1.4149 | -1.3995 | -1.4108 | -1.4065 | -1.4164 | -1.4048 | -1.4067 | -1.4438 | -1.4222 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.4045 | -1.4074 | -1.4156 | -1.4092 | -1.4358 | -1.4196 | -1.4049 | -1.4821 | -1.4243 | -1.4070 | -1.3927 | -1.3963 | -1.4161 | -1.4093 | -1.4640 | -1.4149 | -1.3995 | -1.4108 | -1.4065 | -1.4164 | -1.4048 | -1.4067 | -1.4438 | -1.4222 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.25028 | 0.25150 | 0.24803 | 0.25000 | 0.24733 | 0.24894 | 0.24945 | 0.25106 | 0.24864 | 0.25074 | 0.25050 | 0.25123 | 0.25068 | 0.24887 | 0.25066 | 0.24994 | 0.24983 | 0.24844 | 0.25046 | 0.24909 | 0.25022 | 0.24808 | 0.24841 | 0.24975 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.25028 | 0.25150 | 0.24803 | 0.25000 | 0.24733 | 0.24894 | 0.24945 | 0.25106 | 0.24864 | 0.25074 | 0.25050 | 0.25123 | 0.25068 | 0.24887 | 0.25066 | 0.24994 | 0.24983 | 0.24844 | 0.25046 | 0.24909 | 0.25022 | 0.24808 | 0.24841 | 0.24975 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 1.7710 | 1.7820 | 1.9115 | 1.7377 | 1.9195 | 1.9405 | 1.8791 | 1.9342 | 1.8387 | 1.9694 | 1.7734 | 1.8576 | 1.9358 | 1.9593 | 2.0243 | 1.9313 | 2.0183 | 1.8444 | 1.7204 | 1.7984 | 1.7319 | 1.8852 | 1.8837 | 1.9043 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 2.8249 | 2.8703 | 3.0562 | 2.7885 | 3.0644 | 3.1054 | 3.0322 | 3.0871 | 2.9585 | 3.1377 | 2.8441 | 2.9770 | 3.0820 | 3.1351 | 3.2595 | 3.0911 | 3.2233 | 2.9741 | 2.7614 | 2.8939 | 2.7879 | 3.0217 | 3.0123 | 3.0438 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 1.2275 | 1.2351 | 1.3250 | 1.2045 | 1.3306 | 1.3451 | 1.3024 | 1.3406 | 1.2743 | 1.3650 | 1.2291 | 1.2876 | 1.3416 | 1.3579 | 1.4031 | 1.3387 | 1.3989 | 1.2783 | 1.1924 | 1.2466 | 1.2004 | 1.3068 | 1.3057 | 1.3199 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 1.9579 | 1.9895 | 2.1185 | 1.9329 | 2.1242 | 2.1523 | 2.1018 | 2.1398 | 2.0505 | 2.1750 | 1.9713 | 2.0634 | 2.1361 | 2.1730 | 2.2594 | 2.1427 | 2.2343 | 2.0614 | 1.9141 | 2.0058 | 1.9323 | 2.0946 | 2.0881 | 2.1100 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.26170 | 0.26552 | 0.22063 | 0.31901 | 0.24833 | 0.21394 | 0.26457 | 0.20439 | 0.22923 | 0.21872 | 0.22923 | 0.21777 | 0.20248 | 0.20439 | 0.20057 | 0.18529 | 0.20344 | 0.27221 | 0.29035 | 0.27412 | 0.28271 | 0.23114 | 0.21490 | 0.21203 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.26170 | 0.26552 | 0.22063 | 0.31901 | 0.24833 | 0.21394 | 0.26457 | 0.20439 | 0.22923 | 0.21872 | 0.22923 | 0.21777 | 0.20248 | 0.20439 | 0.20057 | 0.18529 | 0.20344 | 0.27221 | 0.29035 | 0.27412 | 0.28271 | 0.23114 | 0.21490 | 0.21203 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -2.3646 | -2.2358 | -2.4655 | -2.3057 | -2.6499 | -2.4815 | -2.3396 | -2.4573 | -2.3832 | -2.5610 | -2.2946 | -2.4012 | -2.5194 | -2.4333 | -2.4673 | -2.4611 | -2.4444 | -2.3551 | -2.1914 | -2.2809 | -2.1898 | -2.3089 | -2.3938 | -2.4843 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -2.3646 | -2.2358 | -2.4655 | -2.3057 | -2.6499 | -2.4815 | -2.3396 | -2.4573 | -2.3832 | -2.5610 | -2.2946 | -2.4012 | -2.5194 | -2.4333 | -2.4673 | -2.4611 | -2.4444 | -2.3551 | -2.1914 | -2.2809 | -2.1898 | -2.3089 | -2.3938 | -2.4843 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.20575 | 0.20222 | 0.18248 | 0.24703 | 0.22046 | 0.17565 | 0.21206 | 0.17196 | 0.19688 | 0.18099 | 0.18518 | 0.17451 | 0.17467 | 0.16392 | 0.16720 | 0.15943 | 0.17383 | 0.22448 | 0.24173 | 0.22607 | 0.22645 | 0.17538 | 0.17690 | 0.16898 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.20575 | 0.20222 | 0.18248 | 0.24703 | 0.22046 | 0.17565 | 0.21206 | 0.17196 | 0.19688 | 0.18099 | 0.18518 | 0.17451 | 0.17467 | 0.16392 | 0.16720 | 0.15943 | 0.17383 | 0.22448 | 0.24173 | 0.22607 | 0.22645 | 0.17538 | 0.17690 | 0.16898 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.48474 | 0.49637 | 0.45717 | 0.45860 | 0.49651 | 0.48647 | 0.46774 | 0.51368 | 0.48522 | 0.48489 | 0.48629 | 0.49939 | 0.49169 | 0.47219 | 0.46037 | 0.50599 | 0.48121 | 0.45617 | 0.48709 | 0.47268 | 0.46231 | 0.46139 | 0.52884 | 0.47542 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.52683 | 0.53931 | 0.49853 | 0.49823 | 0.54042 | 0.52992 | 0.50909 | 0.56020 | 0.52816 | 0.52783 | 0.52913 | 0.54278 | 0.53513 | 0.51410 | 0.50085 | 0.54965 | 0.52408 | 0.49641 | 0.53040 | 0.51458 | 0.50306 | 0.50224 | 0.57537 | 0.51617 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.33600 | 0.34403 | 0.31688 | 0.31789 | 0.34416 | 0.33723 | 0.32420 | 0.35611 | 0.33631 | 0.33606 | 0.33709 | 0.34615 | 0.34082 | 0.32732 | 0.31911 | 0.35071 | 0.33355 | 0.31621 | 0.33767 | 0.32762 | 0.32047 | 0.31982 | 0.36653 | 0.32954 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.36516 | 0.37379 | 0.34554 | 0.34536 | 0.37462 | 0.36728 | 0.35290 | 0.38831 | 0.36608 | 0.36585 | 0.36674 | 0.37622 | 0.37089 | 0.35637 | 0.34718 | 0.38101 | 0.36325 | 0.34409 | 0.36757 | 0.35667 | 0.34863 | 0.34812 | 0.39879 | 0.35777 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.48123 | 0.46344 | 0.50099 | 0.48123 | 0.47826 | 0.47628 | 0.46739 | 0.46838 | 0.49605 | 0.48617 | 0.47826 | 0.44466 | 0.49209 | 0.46542 | 0.47826 | 0.46047 | 0.46146 | 0.49901 | 0.52075 | 0.50198 | 0.48123 | 0.46937 | 0.47431 | 0.45949 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.48123 | 0.46344 | 0.50099 | 0.48123 | 0.47826 | 0.47628 | 0.46739 | 0.46838 | 0.49605 | 0.48617 | 0.47826 | 0.44466 | 0.49209 | 0.46542 | 0.47826 | 0.46047 | 0.46146 | 0.49901 | 0.52075 | 0.50198 | 0.48123 | 0.46937 | 0.47431 | 0.45949 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -2.9148 | -3.1887 | -2.6219 | -2.7157 | -2.7614 | -2.7730 | -2.7916 | -2.9030 | -2.6924 | -2.8091 | -2.8801 | -3.1046 | -2.8052 | -2.9696 | -2.9173 | -3.0378 | -2.9872 | -2.6525 | -2.6884 | -2.6358 | -2.7074 | -2.7742 | -3.0145 | -2.9090 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -2.9148 | -3.1887 | -2.6219 | -2.7157 | -2.7614 | -2.7730 | -2.7916 | -2.9030 | -2.6924 | -2.8091 | -2.8801 | -3.1046 | -2.8052 | -2.9696 | -2.9173 | -3.0378 | -2.9872 | -2.6525 | -2.6884 | -2.6358 | -2.7074 | -2.7742 | -3.0145 | -2.9090 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.45837 | 0.44167 | 0.47343 | 0.46243 | 0.45423 | 0.45703 | 0.44932 | 0.45269 | 0.47636 | 0.46662 | 0.45468 | 0.43765 | 0.46602 | 0.44371 | 0.45785 | 0.44746 | 0.44111 | 0.47826 | 0.48488 | 0.47892 | 0.46910 | 0.45168 | 0.44841 | 0.44705 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.45837 | 0.44167 | 0.47343 | 0.46243 | 0.45423 | 0.45703 | 0.44932 | 0.45269 | 0.47636 | 0.46662 | 0.45468 | 0.43765 | 0.46602 | 0.44371 | 0.45785 | 0.44746 | 0.44111 | 0.47826 | 0.48488 | 0.47892 | 0.46910 | 0.45168 | 0.44841 | 0.44705 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.56714 | 0.46908 | 0.49778 | 0.49932 | 0.55382 | 0.55930 | 0.52459 | 0.49501 | 0.52726 | 0.51491 | 0.48507 | 0.49164 | 0.55370 | 0.57332 | 0.72806 | 0.57829 | 0.59744 | 0.58731 | 0.54305 | 0.61742 | 0.59620 | 0.57590 | 0.60026 | 0.57703 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.68032 | 0.56276 | 0.59652 | 0.59914 | 0.66559 | 0.67163 | 0.62959 | 0.59312 | 0.63302 | 0.61614 | 0.58129 | 0.58922 | 0.66460 | 0.68888 | 0.87524 | 0.69313 | 0.71587 | 0.70619 | 0.65161 | 0.74037 | 0.71482 | 0.69025 | 0.72118 | 0.69413 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.39327 | 0.32524 | 0.34517 | 0.34627 | 0.38396 | 0.38781 | 0.36372 | 0.34324 | 0.36563 | 0.35708 | 0.33640 | 0.34091 | 0.38394 | 0.39748 | 0.50482 | 0.40100 | 0.41424 | 0.40726 | 0.37651 | 0.42817 | 0.41342 | 0.39941 | 0.41626 | 0.40017 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.47167 | 0.39024 | 0.41372 | 0.41552 | 0.46152 | 0.46569 | 0.43658 | 0.41135 | 0.43898 | 0.42731 | 0.40320 | 0.40860 | 0.46091 | 0.47767 | 0.60688 | 0.48072 | 0.49637 | 0.48970 | 0.45177 | 0.51346 | 0.49566 | 0.47869 | 0.50008 | 0.48141 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.76471 | 0.79846 | 0.77917 | 0.79653 | 0.75892 | 0.78496 | 0.80521 | 0.76278 | 0.76181 | 0.76953 | 0.76663 | 0.77821 | 0.77338 | 0.73770 | 0.72903 | 0.76374 | 0.72806 | 0.78978 | 0.76374 | 0.75988 | 0.76567 | 0.77049 | 0.76085 | 0.75024 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.76471 | 0.79846 | 0.77917 | 0.79653 | 0.75892 | 0.78496 | 0.80521 | 0.76278 | 0.76181 | 0.76953 | 0.76663 | 0.77821 | 0.77338 | 0.73770 | 0.72903 | 0.76374 | 0.72806 | 0.78978 | 0.76374 | 0.75988 | 0.76567 | 0.77049 | 0.76085 | 0.75024 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.62041 | -0.56222 | -0.58687 | -0.55989 | -0.62952 | -0.60856 | -0.53891 | -0.63801 | -0.63354 | -0.63409 | -0.63110 | -0.59775 | -0.65063 | -0.67896 | -0.73035 | -0.63395 | -0.70278 | -0.59100 | -0.62896 | -0.68351 | -0.60039 | -0.60462 | -0.66146 | -0.61884 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.62041 | -0.56222 | -0.58687 | -0.55989 | -0.62952 | -0.60856 | -0.53891 | -0.63801 | -0.63354 | -0.63409 | -0.63110 | -0.59775 | -0.65063 | -0.67896 | -0.73035 | -0.63395 | -0.70278 | -0.59100 | -0.62896 | -0.68351 | -0.60039 | -0.60462 | -0.66146 | -0.61884 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.66675 | 0.68679 | 0.68431 | 0.68865 | 0.66951 | 0.66412 | 0.70719 | 0.66057 | 0.66034 | 0.67344 | 0.66026 | 0.68513 | 0.66339 | 0.65078 | 0.63090 | 0.66371 | 0.63472 | 0.68514 | 0.66849 | 0.64095 | 0.67408 | 0.67504 | 0.66155 | 0.65962 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.66675 | 0.68679 | 0.68431 | 0.68865 | 0.66951 | 0.66412 | 0.70719 | 0.66057 | 0.66034 | 0.67344 | 0.66026 | 0.68513 | 0.66339 | 0.65078 | 0.63090 | 0.66371 | 0.63472 | 0.68514 | 0.66849 | 0.64095 | 0.67408 | 0.67504 | 0.66155 | 0.65962 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.26595 | 0.28896 | 0.25582 | 0.31475 | 0.28127 | 0.31691 | 0.31346 | 0.30253 | 0.31681 | 0.30390 | 0.30486 | 0.30323 | 0.28459 | 0.29793 | 0.27340 | 0.33989 | 0.30045 | 0.26785 | 0.33364 | 0.28264 | 0.25374 | 0.29228 | 0.31308 | 0.29490 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.27474 | 0.29852 | 0.26439 | 0.32529 | 0.29062 | 0.32756 | 0.32406 | 0.31277 | 0.32753 | 0.31413 | 0.31521 | 0.31344 | 0.29413 | 0.30793 | 0.28257 | 0.35141 | 0.31057 | 0.27678 | 0.34482 | 0.29206 | 0.26235 | 0.30214 | 0.32374 | 0.30481 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.18435 | 0.20032 | 0.17733 | 0.21819 | 0.19498 | 0.21969 | 0.21729 | 0.20971 | 0.21961 | 0.21065 | 0.21131 | 0.21022 | 0.19729 | 0.20653 | 0.18952 | 0.23561 | 0.20827 | 0.18566 | 0.23128 | 0.19593 | 0.17590 | 0.20260 | 0.21703 | 0.20444 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.19045 | 0.20695 | 0.18329 | 0.22548 | 0.20145 | 0.22709 | 0.22463 | 0.21681 | 0.22706 | 0.21776 | 0.21848 | 0.21729 | 0.20389 | 0.21344 | 0.19588 | 0.24361 | 0.21530 | 0.19186 | 0.23903 | 0.20247 | 0.18185 | 0.20944 | 0.22440 | 0.21128 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.82021 | 0.80769 | 0.81753 | 0.83184 | 0.82021 | 0.79696 | 0.79606 | 0.84526 | 0.83453 | 0.80948 | 0.83900 | 0.80680 | 0.82737 | 0.80859 | 0.80322 | 0.78533 | 0.79338 | 0.83095 | 0.79517 | 0.84526 | 0.84705 | 0.81306 | 0.81395 | 0.79964 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.82021 | 0.80769 | 0.81753 | 0.83184 | 0.82021 | 0.79696 | 0.79606 | 0.84526 | 0.83453 | 0.80948 | 0.83900 | 0.80680 | 0.82737 | 0.80859 | 0.80322 | 0.78533 | 0.79338 | 0.83095 | 0.79517 | 0.84526 | 0.84705 | 0.81306 | 0.81395 | 0.79964 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.50070 | -0.55844 | -0.45966 | -0.46724 | -0.47698 | -0.56385 | -0.55808 | -0.41141 | -0.43613 | -0.50114 | -0.44726 | -0.50138 | -0.47828 | -0.50601 | -0.59925 | -0.60675 | -0.62115 | -0.49205 | -0.55141 | -0.44665 | -0.47431 | -0.47950 | -0.46686 | -0.53563 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.50070 | -0.55844 | -0.45966 | -0.46724 | -0.47698 | -0.56385 | -0.55808 | -0.41141 | -0.43613 | -0.50114 | -0.44726 | -0.50138 | -0.47828 | -0.50601 | -0.59925 | -0.60675 | -0.62115 | -0.49205 | -0.55141 | -0.44665 | -0.47431 | -0.47950 | -0.46686 | -0.53563 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.80995 | 0.80990 | 0.81506 | 0.81327 | 0.80724 | 0.77902 | 0.78770 | 0.83107 | 0.82187 | 0.80522 | 0.81601 | 0.81160 | 0.80623 | 0.79889 | 0.79345 | 0.77516 | 0.78388 | 0.80517 | 0.77874 | 0.80997 | 0.80757 | 0.80596 | 0.80282 | 0.79375 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.80995 | 0.80990 | 0.81506 | 0.81327 | 0.80724 | 0.77902 | 0.78770 | 0.83107 | 0.82187 | 0.80522 | 0.81601 | 0.81160 | 0.80623 | 0.79889 | 0.79345 | 0.77516 | 0.78388 | 0.80517 | 0.77874 | 0.80997 | 0.80757 | 0.80596 | 0.80282 | 0.79375 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 0.95632 | 1.0443 | 1.0523 | 1.0298 | 1.1206 | 1.0743 | 1.0863 | 1.0406 | 1.0378 | 1.0749 | 1.0028 | 1.0146 | 1.1518 | 1.0653 | 1.1201 | 1.0966 | 1.1953 | 1.0191 | 1.1094 | 1.0652 | 1.1110 | 1.1712 | 1.1340 | 1.0744 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.5508 | 1.6946 | 1.6920 | 1.6562 | 1.7981 | 1.7118 | 1.7379 | 1.6774 | 1.6597 | 1.7132 | 1.6117 | 1.6333 | 1.8390 | 1.7149 | 1.7977 | 1.7795 | 1.8964 | 1.6354 | 1.7759 | 1.6912 | 1.7759 | 1.8701 | 1.8159 | 1.7119 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.69675 | 0.75707 | 0.76190 | 0.74825 | 0.81522 | 0.78187 | 0.78691 | 0.75771 | 0.75469 | 0.77950 | 0.73098 | 0.74392 | 0.83482 | 0.77655 | 0.81402 | 0.79992 | 0.86967 | 0.74070 | 0.80860 | 0.77400 | 0.80055 | 0.84576 | 0.82193 | 0.78456 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.1619 | 1.2599 | 1.2565 | 1.2364 | 1.3447 | 1.2822 | 1.2922 | 1.2560 | 1.2383 | 1.2756 | 1.2100 | 1.2328 | 1.3677 | 1.2848 | 1.3393 | 1.3317 | 1.4184 | 1.2220 | 1.3324 | 1.2641 | 1.3102 | 1.3827 | 1.3506 | 1.2892 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.63670 | 0.67228 | 0.63296 | 0.64981 | 0.61798 | 0.65543 | 0.63296 | 0.64419 | 0.64045 | 0.62172 | 0.64794 | 0.66479 | 0.64232 | 0.61798 | 0.60861 | 0.62547 | 0.60300 | 0.66854 | 0.64794 | 0.63296 | 0.64419 | 0.62172 | 0.65543 | 0.64794 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.63670 | 0.67228 | 0.63296 | 0.64981 | 0.61798 | 0.65543 | 0.63296 | 0.64419 | 0.64045 | 0.62172 | 0.64794 | 0.66479 | 0.64232 | 0.61798 | 0.60861 | 0.62547 | 0.60300 | 0.66854 | 0.64794 | 0.63296 | 0.64419 | 0.62172 | 0.65543 | 0.64794 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.93431 | -0.89775 | -0.97958 | -0.93963 | -1.0027 | -0.96975 | -0.97024 | -0.96611 | -0.97158 | -0.94696 | -0.92798 | -0.96147 | -0.97140 | -0.97943 | -1.0023 | -1.0082 | -0.99193 | -0.88760 | -0.94106 | -0.94971 | -0.95562 | -0.98598 | -0.93883 | -0.96771 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.93431 | -0.89775 | -0.97958 | -0.93963 | -1.0027 | -0.96975 | -0.97024 | -0.96611 | -0.97158 | -0.94696 | -0.92798 | -0.96147 | -0.97140 | -0.97943 | -1.0023 | -1.0082 | -0.99193 | -0.88760 | -0.94106 | -0.94971 | -0.95562 | -0.98598 | -0.93883 | -0.96771 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.57155 | 0.59312 | 0.55922 | 0.56134 | 0.54517 | 0.55989 | 0.54770 | 0.56346 | 0.55440 | 0.55366 | 0.56561 | 0.58119 | 0.54561 | 0.55571 | 0.54236 | 0.55174 | 0.53260 | 0.57733 | 0.56386 | 0.55787 | 0.56251 | 0.54287 | 0.56999 | 0.54951 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.57155 | 0.59312 | 0.55922 | 0.56134 | 0.54517 | 0.55989 | 0.54770 | 0.56346 | 0.55440 | 0.55366 | 0.56561 | 0.58119 | 0.54561 | 0.55571 | 0.54236 | 0.55174 | 0.53260 | 0.57733 | 0.56386 | 0.55787 | 0.56251 | 0.54287 | 0.56999 | 0.54951 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.8911 | 1.9091 | 1.9490 | 2.0360 | 2.0477 | 2.0457 | 1.9743 | 1.9657 | 1.9603 | 1.9953 | 1.9264 | 1.9812 | 2.0513 | 1.9739 | 1.9934 | 2.0694 | 2.0857 | 1.9348 | 1.9673 | 2.1629 | 1.9963 | 2.0107 | 1.9347 | 2.0844 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.6096 | 2.6038 | 2.6832 | 2.7977 | 2.8265 | 2.8355 | 2.7185 | 2.6878 | 2.7109 | 2.7557 | 2.6393 | 2.7381 | 2.8195 | 2.7171 | 2.7347 | 2.8391 | 2.8686 | 2.6695 | 2.6824 | 2.9529 | 2.7380 | 2.7540 | 2.6591 | 2.8431 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.3109 | 1.3233 | 1.3509 | 1.4113 | 1.4192 | 1.4180 | 1.3684 | 1.3625 | 1.3587 | 1.3830 | 1.3353 | 1.3733 | 1.4219 | 1.3683 | 1.3818 | 1.4344 | 1.4457 | 1.3411 | 1.3636 | 1.4992 | 1.3838 | 1.3937 | 1.3410 | 1.4448 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.8090 | 1.8047 | 1.8596 | 1.9393 | 1.9592 | 1.9653 | 1.8842 | 1.8630 | 1.8788 | 1.9101 | 1.8294 | 1.8979 | 1.9543 | 1.8833 | 1.8953 | 1.9678 | 1.9884 | 1.8503 | 1.8594 | 2.0468 | 1.8978 | 1.9089 | 1.8431 | 1.9707 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.22231 | 0.21821 | 0.22395 | 0.21903 | 0.23298 | 0.21247 | 0.21985 | 0.22642 | 0.22067 | 0.23872 | 0.22067 | 0.22888 | 0.21903 | 0.22313 | 0.21985 | 0.21411 | 0.22806 | 0.21903 | 0.21247 | 0.22313 | 0.23708 | 0.22395 | 0.22970 | 0.22395 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.22231 | 0.21821 | 0.22395 | 0.21903 | 0.23298 | 0.21247 | 0.21985 | 0.22642 | 0.22067 | 0.23872 | 0.22067 | 0.22888 | 0.21903 | 0.22313 | 0.21985 | 0.21411 | 0.22806 | 0.21903 | 0.21247 | 0.22313 | 0.23708 | 0.22395 | 0.22970 | 0.22395 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -4.3503 | -4.3504 | -4.4603 | -4.3427 | -4.4000 | -4.5358 | -4.4928 | -4.3743 | -4.3010 | -4.2130 | -4.4395 | -4.3527 | -4.5272 | -4.4181 | -4.3773 | -4.5180 | -4.5391 | -4.2456 | -4.2718 | -4.2983 | -4.3793 | -4.3395 | -4.2617 | -4.3913 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -4.3503 | -4.3504 | -4.4603 | -4.3427 | -4.4000 | -4.5358 | -4.4928 | -4.3743 | -4.3010 | -4.2130 | -4.4395 | -4.3527 | -4.5272 | -4.4181 | -4.3773 | -4.5180 | -4.5391 | -4.2456 | -4.2718 | -4.2983 | -4.3793 | -4.3395 | -4.2617 | -4.3913 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.23742 | 0.24313 | 0.24315 | 0.23666 | 0.24305 | 0.23490 | 0.24146 | 0.24302 | 0.24366 | 0.25324 | 0.24019 | 0.24420 | 0.23363 | 0.24330 | 0.23325 | 0.23369 | 0.23803 | 0.23905 | 0.23462 | 0.23624 | 0.24334 | 0.24834 | 0.24790 | 0.24463 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.23742 | 0.24313 | 0.24315 | 0.23666 | 0.24305 | 0.23490 | 0.24146 | 0.24302 | 0.24366 | 0.25324 | 0.24019 | 0.24420 | 0.23363 | 0.24330 | 0.23325 | 0.23369 | 0.23803 | 0.23905 | 0.23462 | 0.23624 | 0.24334 | 0.24834 | 0.24790 | 0.24463 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.50000 | 0.50645 | 0.51154 | 0.50203 | 0.49533 | 0.51367 | 0.50447 | 0.50723 | 0.52381 | 0.52094 | 0.50220 | 0.51032 | 0.50935 | 0.50395 | 0.52479 | 0.51141 | 0.52266 | 0.51082 | 0.52441 | 0.51054 | 0.49844 | 0.49955 | 0.52528 | 0.51347 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.50639 | 0.51319 | 0.51849 | 0.50864 | 0.50181 | 0.52066 | 0.51131 | 0.51391 | 0.53057 | 0.52777 | 0.50852 | 0.51703 | 0.51610 | 0.51075 | 0.53140 | 0.51787 | 0.52918 | 0.51772 | 0.53164 | 0.51737 | 0.50489 | 0.50606 | 0.53196 | 0.52031 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.71484 | 0.71265 | 0.69879 | 0.71318 | 0.70602 | 0.71358 | 0.70564 | 0.72461 | 0.72786 | 0.72206 | 0.71495 | 0.71749 | 0.72294 | 0.72694 | 0.73228 | 0.73648 | 0.73109 | 0.70587 | 0.72454 | 0.71336 | 0.70016 | 0.70966 | 0.71854 | 0.71733 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.72096 | 0.71901 | 0.70475 | 0.71931 | 0.71220 | 0.71973 | 0.71179 | 0.73090 | 0.73394 | 0.72834 | 0.72112 | 0.72366 | 0.72912 | 0.73331 | 0.73869 | 0.74280 | 0.73740 | 0.71209 | 0.73076 | 0.71937 | 0.70622 | 0.71575 | 0.72475 | 0.72330 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 3.1079 | 2.7800 | 2.7522 | 2.4169 | 2.4039 | 2.3537 | 2.6569 | 2.7019 | 2.6502 | 2.7531 | 2.8620 | 2.4724 | 2.6980 | 2.9461 | 2.6144 | 2.6334 | 3.0191 | 2.7614 | 2.6042 | 2.8003 | 2.4850 | 2.5425 | 2.1072 | 2.6710 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 6.2158 | 5.5599 | 5.5044 | 4.8338 | 4.8078 | 4.7073 | 5.3137 | 5.4037 | 5.3004 | 5.5062 | 5.7240 | 4.9448 | 5.3961 | 5.8922 | 5.2288 | 5.2667 | 6.0383 | 5.5227 | 5.2084 | 5.6006 | 4.9701 | 5.0851 | 4.2144 | 5.3420 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 2.1537 | 1.9278 | 1.9076 | 1.6746 | 1.6662 | 1.6312 | 1.8412 | 1.8728 | 1.8373 | 1.9090 | 1.9836 | 1.7139 | 1.8704 | 2.0425 | 1.8124 | 1.8253 | 2.0925 | 1.9142 | 1.8045 | 1.9414 | 1.7227 | 1.7622 | 1.4608 | 1.8516 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 4.3073 | 3.8555 | 3.8152 | 3.3492 | 3.3323 | 3.2624 | 3.6825 | 3.7456 | 3.6745 | 3.8180 | 3.9671 | 3.4279 | 3.7407 | 4.0849 | 3.6247 | 3.6505 | 4.1850 | 3.8284 | 3.6091 | 3.8827 | 3.4454 | 3.5245 | 2.9216 | 3.7032 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.09000 | 0.07000 | 0.07000 | 0.11000 | 0.11000 | 0.09000 | 0.09000 | 0.10000 | 0.11000 | 0.10000 | 0.08000 | 0.11000 | 0.07000 | 0.09000 | 0.07000 | 0.09000 | 0.07000 | 0.10000 | 0.09000 | 0.12000 | 0.12000 | 0.06000 | 0.12000 | 0.07000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.09000 | 0.07000 | 0.07000 | 0.11000 | 0.11000 | 0.09000 | 0.09000 | 0.10000 | 0.11000 | 0.10000 | 0.08000 | 0.11000 | 0.07000 | 0.09000 | 0.07000 | 0.09000 | 0.07000 | 0.10000 | 0.09000 | 0.12000 | 0.12000 | 0.06000 | 0.12000 | 0.07000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -4.2904 | -3.8399 | -3.8051 | -3.3435 | -3.3271 | -3.2524 | -3.6743 | -3.7353 | -3.6618 | -3.8088 | -3.9529 | -3.4178 | -3.7107 | -4.0753 | -3.6127 | -3.6446 | -4.1770 | -3.8210 | -3.6028 | -3.8713 | -3.4339 | -3.5152 | -2.9084 | -3.6854 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -4.2904 | -3.8399 | -3.8051 | -3.3435 | -3.3271 | -3.2524 | -3.6743 | -3.7353 | -3.6618 | -3.8088 | -3.9529 | -3.4178 | -3.7107 | -4.0753 | -3.6127 | -3.6446 | -4.1770 | -3.8210 | -3.6028 | -3.8713 | -3.4339 | -3.5152 | -2.9084 | -3.6854 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09385 | 0.09003 | 0.09524 | 0.10378 | 0.10052 | 0.09439 | 0.09463 | 0.09292 | 0.09306 | 0.09310 | 0.09562 | 0.09936 | 0.09086 | 0.09245 | 0.08767 | 0.09487 | 0.09278 | 0.09641 | 0.09233 | 0.09666 | 0.09840 | 0.09359 | 0.09375 | 0.08705 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09385 | 0.09003 | 0.09524 | 0.10378 | 0.10052 | 0.09439 | 0.09463 | 0.09292 | 0.09306 | 0.09310 | 0.09562 | 0.09936 | 0.09086 | 0.09245 | 0.08767 | 0.09487 | 0.09278 | 0.09641 | 0.09233 | 0.09666 | 0.09840 | 0.09359 | 0.09375 | 0.08705 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.84712 | 0.85156 | 0.84657 | 0.84731 | 0.84958 | 0.85016 | 0.84658 | 0.85167 | 0.85444 | 0.85385 | 0.85018 | 0.85463 | 0.85460 | 0.84996 | 0.85796 | 0.86221 | 0.85725 | 0.84768 | 0.85289 | 0.84981 | 0.84691 | 0.85464 | 0.85312 | 0.85501 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.85634 | 0.86093 | 0.85594 | 0.85669 | 0.85892 | 0.85950 | 0.85600 | 0.86109 | 0.86399 | 0.86348 | 0.85949 | 0.86415 | 0.86414 | 0.85941 | 0.86734 | 0.87172 | 0.86668 | 0.85699 | 0.86235 | 0.85905 | 0.85628 | 0.86399 | 0.86247 | 0.86443 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.76457 | 0.77003 | 0.77155 | 0.77059 | 0.77177 | 0.77770 | 0.77109 | 0.78117 | 0.78765 | 0.78384 | 0.76997 | 0.77391 | 0.78378 | 0.77472 | 0.79271 | 0.79584 | 0.79390 | 0.77265 | 0.78135 | 0.77927 | 0.77241 | 0.78005 | 0.78104 | 0.77871 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.76707 | 0.77254 | 0.77399 | 0.77290 | 0.77447 | 0.78030 | 0.77362 | 0.78363 | 0.79016 | 0.78655 | 0.77253 | 0.77654 | 0.78661 | 0.77728 | 0.79510 | 0.79829 | 0.79647 | 0.77531 | 0.78396 | 0.78198 | 0.77493 | 0.78272 | 0.78353 | 0.78125 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.77933 | 0.79109 | 0.78453 | 0.78903 | 0.78755 | 0.78235 | 0.78546 | 0.80540 | 0.80052 | 0.80153 | 0.78908 | 0.80307 | 0.79773 | 0.78996 | 0.81052 | 0.80225 | 0.81005 | 0.77841 | 0.79091 | 0.80553 | 0.78548 | 0.78278 | 0.79770 | 0.79325 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.82027 | 0.83282 | 0.82603 | 0.83043 | 0.82921 | 0.82318 | 0.82664 | 0.84834 | 0.84255 | 0.84419 | 0.83083 | 0.84572 | 0.84011 | 0.83163 | 0.85333 | 0.84425 | 0.85286 | 0.81900 | 0.83231 | 0.84843 | 0.82689 | 0.82388 | 0.83974 | 0.83541 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0132 | 1.0289 | 1.0144 | 1.0243 | 1.0247 | 1.0259 | 1.0262 | 1.0559 | 1.0250 | 1.0291 | 1.0164 | 1.0272 | 1.0300 | 1.0188 | 1.0358 | 1.0366 | 1.0325 | 1.0192 | 1.0443 | 1.0241 | 1.0109 | 1.0316 | 1.0313 | 1.0274 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0264 | 2.0578 | 2.0288 | 2.0486 | 2.0493 | 2.0518 | 2.0523 | 2.1118 | 2.0501 | 2.0583 | 2.0328 | 2.0544 | 2.0600 | 2.0376 | 2.0717 | 2.0731 | 2.0649 | 2.0385 | 2.0887 | 2.0482 | 2.0219 | 2.0632 | 2.0625 | 2.0547 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.70241 | 0.71323 | 0.70323 | 0.71002 | 0.71031 | 0.71125 | 0.71132 | 0.73185 | 0.71059 | 0.71342 | 0.70462 | 0.71207 | 0.71401 | 0.70626 | 0.71807 | 0.71855 | 0.71572 | 0.70655 | 0.72388 | 0.70992 | 0.70078 | 0.71510 | 0.71483 | 0.71219 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.4048 | 1.4265 | 1.4065 | 1.4200 | 1.4206 | 1.4225 | 1.4226 | 1.4637 | 1.4212 | 1.4268 | 1.4092 | 1.4241 | 1.4280 | 1.4125 | 1.4361 | 1.4371 | 1.4314 | 1.4131 | 1.4478 | 1.4198 | 1.4016 | 1.4302 | 1.4297 | 1.4244 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.25526 | 0.26312 | 0.24527 | 0.24867 | 0.24910 | 0.24230 | 0.24803 | 0.24825 | 0.24421 | 0.24655 | 0.25122 | 0.25143 | 0.24548 | 0.25292 | 0.24251 | 0.25313 | 0.23868 | 0.24952 | 0.25654 | 0.24867 | 0.26079 | 0.24421 | 0.24952 | 0.24973 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.25526 | 0.26312 | 0.24527 | 0.24867 | 0.24910 | 0.24230 | 0.24803 | 0.24825 | 0.24421 | 0.24655 | 0.25122 | 0.25143 | 0.24548 | 0.25292 | 0.24251 | 0.25313 | 0.23868 | 0.24952 | 0.25654 | 0.24867 | 0.26079 | 0.24421 | 0.24952 | 0.24973 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3863 | -1.3903 | -1.3880 | -1.3904 | -1.3940 | -1.3926 | -1.3910 | -1.4024 | -1.3929 | -1.3929 | -1.3888 | -1.3917 | -1.3903 | -1.3886 | -1.3958 | -1.3951 | -1.3914 | -1.3905 | -1.3980 | -1.3916 | -1.3871 | -1.3964 | -1.3961 | -1.3933 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.3925 | -1.4068 | -1.3955 | -1.4072 | -1.4132 | -1.4103 | -1.4066 | -1.4517 | -1.4088 | -1.4121 | -1.3986 | -1.4061 | -1.4033 | -1.4003 | -1.4228 | -1.4242 | -1.4073 | -1.4089 | -1.4414 | -1.4051 | -1.3964 | -1.4206 | -1.4245 | -1.4130 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.25078 | 0.25061 | 0.25028 | 0.25056 | 0.24951 | 0.24980 | 0.25024 | 0.25038 | 0.24951 | 0.24991 | 0.25026 | 0.24979 | 0.25015 | 0.25058 | 0.24979 | 0.25032 | 0.25009 | 0.25068 | 0.25103 | 0.24972 | 0.25085 | 0.24929 | 0.24990 | 0.24978 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.25152 | 0.25121 | 0.25057 | 0.25096 | 0.24905 | 0.24953 | 0.25047 | 0.25048 | 0.24899 | 0.24977 | 0.25044 | 0.24956 | 0.25030 | 0.25108 | 0.24938 | 0.25052 | 0.25011 | 0.25141 | 0.25192 | 0.24943 | 0.25165 | 0.24864 | 0.24975 | 0.24949 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 1.0793 | 1.1009 | 1.0903 | 1.1093 | 1.0974 | 1.1069 | 1.0956 | 1.1134 | 1.0972 | 1.1083 | 1.1053 | 1.1080 | 1.1264 | 1.1209 | 1.1357 | 1.1269 | 1.1326 | 1.0821 | 1.1097 | 1.1046 | 1.0898 | 1.1064 | 1.1109 | 1.1149 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.2012 | 1.2252 | 1.2145 | 1.2354 | 1.2232 | 1.2337 | 1.2211 | 1.2405 | 1.2210 | 1.2349 | 1.2323 | 1.2346 | 1.2571 | 1.2490 | 1.2654 | 1.2552 | 1.2613 | 1.2053 | 1.2370 | 1.2306 | 1.2148 | 1.2347 | 1.2370 | 1.2426 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0105 | 1.0333 | 1.0111 | 1.0252 | 1.0176 | 1.0161 | 1.0183 | 1.0480 | 1.0159 | 1.0162 | 1.0107 | 1.0129 | 1.0205 | 1.0114 | 1.0203 | 1.0323 | 1.0240 | 1.0109 | 1.0184 | 1.0189 | 1.0071 | 1.0123 | 1.0243 | 1.0242 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0209 | 2.0667 | 2.0221 | 2.0503 | 2.0353 | 2.0323 | 2.0365 | 2.0960 | 2.0318 | 2.0324 | 2.0213 | 2.0258 | 2.0409 | 2.0228 | 2.0407 | 2.0646 | 2.0480 | 2.0217 | 2.0368 | 2.0378 | 2.0142 | 2.0245 | 2.0485 | 2.0484 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.70045 | 0.71626 | 0.70092 | 0.71065 | 0.70543 | 0.70448 | 0.70591 | 0.72643 | 0.70421 | 0.70445 | 0.70063 | 0.70214 | 0.70743 | 0.70110 | 0.70727 | 0.71559 | 0.70985 | 0.70074 | 0.70589 | 0.70634 | 0.69812 | 0.70174 | 0.71002 | 0.70997 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4009 | 1.4325 | 1.4018 | 1.4213 | 1.4109 | 1.4090 | 1.4118 | 1.4529 | 1.4084 | 1.4089 | 1.4013 | 1.4043 | 1.4149 | 1.4022 | 1.4145 | 1.4312 | 1.4197 | 1.4015 | 1.4118 | 1.4127 | 1.3962 | 1.4035 | 1.4200 | 1.4199 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.28809 | 0.26157 | 0.28162 | 0.25108 | 0.26712 | 0.25972 | 0.25447 | 0.25447 | 0.27236 | 0.27452 | 0.25416 | 0.27175 | 0.28840 | 0.27668 | 0.27329 | 0.25046 | 0.25941 | 0.26804 | 0.26959 | 0.27637 | 0.27051 | 0.26373 | 0.26928 | 0.26990 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.28809 | 0.26157 | 0.28162 | 0.25108 | 0.26712 | 0.25972 | 0.25447 | 0.25447 | 0.27236 | 0.27452 | 0.25416 | 0.27175 | 0.28840 | 0.27668 | 0.27329 | 0.25046 | 0.25941 | 0.26804 | 0.26959 | 0.27637 | 0.27051 | 0.26373 | 0.26928 | 0.26990 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3824 | -1.3902 | -1.3846 | -1.3904 | -1.3864 | -1.3859 | -1.3881 | -1.3987 | -1.3846 | -1.3848 | -1.3844 | -1.3823 | -1.3822 | -1.3817 | -1.3852 | -1.3945 | -1.3858 | -1.3848 | -1.3863 | -1.3863 | -1.3848 | -1.3864 | -1.3903 | -1.3858 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.3898 | -1.4124 | -1.3912 | -1.4085 | -1.4033 | -1.3956 | -1.3986 | -1.4413 | -1.3952 | -1.3963 | -1.3897 | -1.3876 | -1.3924 | -1.3896 | -1.4017 | -1.4176 | -1.3982 | -1.3953 | -1.4056 | -1.3962 | -1.3898 | -1.3920 | -1.4146 | -1.4012 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25249 | 0.25144 | 0.25150 | 0.25064 | 0.25214 | 0.25133 | 0.25063 | 0.25092 | 0.25190 | 0.25201 | 0.25136 | 0.25217 | 0.25289 | 0.25272 | 0.25252 | 0.24996 | 0.25173 | 0.25182 | 0.25244 | 0.25128 | 0.25120 | 0.25066 | 0.25163 | 0.25226 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.25506 | 0.25283 | 0.25305 | 0.25110 | 0.25443 | 0.25264 | 0.25118 | 0.25165 | 0.25357 | 0.25386 | 0.25260 | 0.25427 | 0.25582 | 0.25532 | 0.25487 | 0.24998 | 0.25329 | 0.25351 | 0.25473 | 0.25262 | 0.25241 | 0.25127 | 0.25312 | 0.25458 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.92614 | 0.93996 | 0.92452 | 0.93887 | 0.92707 | 0.93461 | 0.93868 | 0.94585 | 0.93953 | 0.95297 | 0.93397 | 0.94695 | 0.94955 | 0.94181 | 0.96722 | 0.95293 | 0.96882 | 0.92483 | 0.94029 | 0.95278 | 0.92946 | 0.93820 | 0.95540 | 0.94609 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 0.98907 | 1.0038 | 0.98783 | 1.0019 | 0.99037 | 0.99863 | 1.0031 | 1.0103 | 1.0033 | 1.0183 | 0.99742 | 1.0131 | 1.0156 | 1.0065 | 1.0349 | 1.0179 | 1.0353 | 0.98829 | 1.0045 | 1.0192 | 0.99450 | 1.0032 | 1.0224 | 1.0118 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0086 | 1.0175 | 1.0057 | 1.0256 | 1.0227 | 1.0124 | 1.0205 | 1.0891 | 1.0268 | 1.0330 | 1.0118 | 1.0328 | 1.0378 | 1.0195 | 1.0375 | 1.0432 | 1.0346 | 1.0045 | 1.0171 | 1.0181 | 1.0049 | 1.0140 | 1.0455 | 1.0275 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0171 | 2.0349 | 2.0114 | 2.0512 | 2.0454 | 2.0247 | 2.0410 | 2.1782 | 2.0537 | 2.0660 | 2.0235 | 2.0657 | 2.0756 | 2.0390 | 2.0750 | 2.0864 | 2.0692 | 2.0091 | 2.0341 | 2.0361 | 2.0098 | 2.0280 | 2.0909 | 2.0550 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.69919 | 0.70531 | 0.69715 | 0.71093 | 0.70894 | 0.70180 | 0.70747 | 0.75482 | 0.71178 | 0.71608 | 0.70136 | 0.71602 | 0.71942 | 0.70673 | 0.71921 | 0.72314 | 0.71723 | 0.69634 | 0.70502 | 0.70575 | 0.69656 | 0.70294 | 0.72470 | 0.71229 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.3984 | 1.4106 | 1.3943 | 1.4219 | 1.4179 | 1.4036 | 1.4149 | 1.5096 | 1.4236 | 1.4322 | 1.4027 | 1.4320 | 1.4388 | 1.4135 | 1.4384 | 1.4463 | 1.4345 | 1.3927 | 1.4100 | 1.4115 | 1.3931 | 1.4059 | 1.4494 | 1.4246 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.28404 | 0.27624 | 0.30029 | 0.24504 | 0.24894 | 0.26877 | 0.24569 | 0.23594 | 0.25252 | 0.23789 | 0.26682 | 0.23204 | 0.23984 | 0.25122 | 0.23692 | 0.23724 | 0.24017 | 0.29769 | 0.26487 | 0.26032 | 0.28729 | 0.25999 | 0.23562 | 0.24699 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.28404 | 0.27624 | 0.30029 | 0.24504 | 0.24894 | 0.26877 | 0.24569 | 0.23594 | 0.25252 | 0.23789 | 0.26682 | 0.23204 | 0.23984 | 0.25122 | 0.23692 | 0.23724 | 0.24017 | 0.29769 | 0.26487 | 0.26032 | 0.28729 | 0.25999 | 0.23562 | 0.24699 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3809 | -1.3819 | -1.3794 | -1.3927 | -1.3903 | -1.3844 | -1.3881 | -1.4251 | -1.3911 | -1.3962 | -1.3858 | -1.3960 | -1.3949 | -1.3882 | -1.3989 | -1.4033 | -1.3938 | -1.3806 | -1.3861 | -1.3868 | -1.3812 | -1.3862 | -1.4029 | -1.3917 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.3864 | -1.3902 | -1.3835 | -1.4106 | -1.4097 | -1.3905 | -1.3986 | -1.4975 | -1.4097 | -1.4184 | -1.3922 | -1.4148 | -1.4165 | -1.4006 | -1.4238 | -1.4335 | -1.4093 | -1.3882 | -1.4039 | -1.3968 | -1.3875 | -1.3943 | -1.4440 | -1.4100 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.25277 | 0.25269 | 0.25316 | 0.24989 | 0.25088 | 0.25155 | 0.25056 | 0.24517 | 0.25058 | 0.24910 | 0.25100 | 0.24870 | 0.24948 | 0.25082 | 0.24841 | 0.24760 | 0.24919 | 0.25321 | 0.25236 | 0.25107 | 0.25274 | 0.25111 | 0.24900 | 0.25032 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.25554 | 0.25522 | 0.25643 | 0.24987 | 0.25169 | 0.25321 | 0.25102 | 0.24126 | 0.25118 | 0.24824 | 0.25206 | 0.24736 | 0.24894 | 0.25155 | 0.24686 | 0.24547 | 0.24843 | 0.25664 | 0.25474 | 0.25214 | 0.25547 | 0.25221 | 0.24795 | 0.25063 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.4058 | 1.4102 | 1.4129 | 1.4081 | 1.4187 | 1.4251 | 1.4063 | 1.4338 | 1.4340 | 1.4391 | 1.4017 | 1.4106 | 1.4332 | 1.4133 | 1.4502 | 1.4286 | 1.4696 | 1.4280 | 1.4185 | 1.4320 | 1.4049 | 1.4250 | 1.4256 | 1.4163 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.7544 | 1.7579 | 1.7626 | 1.7570 | 1.7746 | 1.7810 | 1.7531 | 1.7869 | 1.7834 | 1.7950 | 1.7436 | 1.7549 | 1.7835 | 1.7599 | 1.8070 | 1.7723 | 1.8352 | 1.7906 | 1.7617 | 1.7893 | 1.7552 | 1.7780 | 1.7764 | 1.7619 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0178 | 1.0241 | 1.0154 | 1.0449 | 1.0178 | 1.0140 | 1.0244 | 1.0841 | 1.0433 | 1.0363 | 1.0271 | 1.0340 | 1.0400 | 1.0308 | 1.0452 | 1.0571 | 1.0444 | 1.0045 | 1.0180 | 1.0196 | 1.0078 | 1.0204 | 1.0499 | 1.0333 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0356 | 2.0483 | 2.0307 | 2.0897 | 2.0356 | 2.0281 | 2.0488 | 2.1682 | 2.0866 | 2.0725 | 2.0542 | 2.0679 | 2.0801 | 2.0617 | 2.0905 | 2.1143 | 2.0889 | 2.0091 | 2.0360 | 2.0393 | 2.0156 | 2.0408 | 2.0997 | 2.0667 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.70554 | 0.70993 | 0.70380 | 0.72428 | 0.70553 | 0.70301 | 0.71014 | 0.75144 | 0.72320 | 0.71830 | 0.71201 | 0.71674 | 0.72095 | 0.71457 | 0.72449 | 0.73272 | 0.72404 | 0.69633 | 0.70569 | 0.70681 | 0.69860 | 0.70737 | 0.72776 | 0.71628 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4111 | 1.4199 | 1.4076 | 1.4486 | 1.4111 | 1.4060 | 1.4203 | 1.5029 | 1.4464 | 1.4366 | 1.4240 | 1.4335 | 1.4419 | 1.4291 | 1.4490 | 1.4654 | 1.4481 | 1.3927 | 1.4114 | 1.4136 | 1.3972 | 1.4147 | 1.4555 | 1.4326 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.27435 | 0.26872 | 0.26441 | 0.23757 | 0.27170 | 0.28264 | 0.24023 | 0.24155 | 0.24950 | 0.24983 | 0.25911 | 0.24983 | 0.26309 | 0.24917 | 0.23890 | 0.23095 | 0.23128 | 0.28728 | 0.28131 | 0.26839 | 0.27601 | 0.25977 | 0.24785 | 0.26773 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.27435 | 0.26872 | 0.26441 | 0.23757 | 0.27170 | 0.28264 | 0.24023 | 0.24155 | 0.24950 | 0.24983 | 0.25911 | 0.24983 | 0.26309 | 0.24917 | 0.23890 | 0.23095 | 0.23128 | 0.28728 | 0.28131 | 0.26839 | 0.27601 | 0.25977 | 0.24785 | 0.26773 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3854 | -1.3845 | -1.3851 | -1.4022 | -1.3860 | -1.3834 | -1.3904 | -1.4186 | -1.3986 | -1.3962 | -1.3924 | -1.3943 | -1.3902 | -1.3907 | -1.3995 | -1.4081 | -1.3973 | -1.3797 | -1.3842 | -1.3849 | -1.3838 | -1.3871 | -1.4063 | -1.3901 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.3971 | -1.3987 | -1.3949 | -1.4357 | -1.4002 | -1.3925 | -1.4036 | -1.4880 | -1.4295 | -1.4228 | -1.4100 | -1.4161 | -1.4103 | -1.4145 | -1.4352 | -1.4513 | -1.4249 | -1.3854 | -1.4055 | -1.3943 | -1.3896 | -1.3995 | -1.4480 | -1.4166 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.25192 | 0.25244 | 0.25170 | 0.24831 | 0.25196 | 0.25237 | 0.25009 | 0.24726 | 0.24948 | 0.24981 | 0.25004 | 0.24987 | 0.25110 | 0.25131 | 0.24976 | 0.24769 | 0.24944 | 0.25327 | 0.25361 | 0.25171 | 0.25168 | 0.25129 | 0.24806 | 0.25211 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.25394 | 0.25473 | 0.25336 | 0.24674 | 0.25410 | 0.25498 | 0.25013 | 0.24490 | 0.24905 | 0.24971 | 0.25020 | 0.24989 | 0.25209 | 0.25232 | 0.24958 | 0.24580 | 0.24886 | 0.25655 | 0.25719 | 0.25344 | 0.25330 | 0.25248 | 0.24657 | 0.25433 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.48314 | 0.46482 | 0.48110 | 0.47485 | 0.46964 | 0.48321 | 0.46952 | 0.49328 | 0.48886 | 0.50079 | 0.47741 | 0.48238 | 0.49716 | 0.46670 | 0.48104 | 0.49670 | 0.47721 | 0.47649 | 0.49042 | 0.47628 | 0.48049 | 0.48590 | 0.47486 | 0.47260 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.48578 | 0.46746 | 0.48378 | 0.47762 | 0.47229 | 0.48598 | 0.47221 | 0.49600 | 0.49177 | 0.50358 | 0.48006 | 0.48505 | 0.50006 | 0.46937 | 0.48378 | 0.49953 | 0.47985 | 0.47913 | 0.49332 | 0.47898 | 0.48320 | 0.48865 | 0.47756 | 0.47526 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.37589 | 0.35978 | 0.36640 | 0.36433 | 0.36663 | 0.35956 | 0.35832 | 0.37776 | 0.38029 | 0.37521 | 0.38005 | 0.38447 | 0.37638 | 0.37277 | 0.37233 | 0.37711 | 0.37492 | 0.36000 | 0.36961 | 0.36159 | 0.37392 | 0.37088 | 0.37432 | 0.36274 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.37736 | 0.36122 | 0.36778 | 0.36581 | 0.36816 | 0.36092 | 0.35975 | 0.37915 | 0.38181 | 0.37662 | 0.38158 | 0.38588 | 0.37782 | 0.37421 | 0.37379 | 0.37863 | 0.37639 | 0.36135 | 0.37102 | 0.36299 | 0.37533 | 0.37231 | 0.37574 | 0.36413 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.64196 | 0.63221 | 0.64242 | 0.64799 | 0.67442 | 0.70727 | 0.63635 | 0.64466 | 0.67173 | 0.65741 | 0.63930 | 0.68415 | 0.67426 | 0.65114 | 0.65517 | 0.68336 | 0.67294 | 0.64478 | 0.65218 | 0.63901 | 0.64793 | 0.66137 | 0.68813 | 0.64999 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.64628 | 0.63670 | 0.64682 | 0.65258 | 0.67936 | 0.71230 | 0.64098 | 0.64885 | 0.67653 | 0.66198 | 0.64356 | 0.68902 | 0.67914 | 0.65575 | 0.65968 | 0.68830 | 0.67759 | 0.64929 | 0.65663 | 0.64361 | 0.65250 | 0.66609 | 0.69298 | 0.65461 |
| eval/lm/c4_en-validation/CE loss | lower | 3.1124 | 3.1250 | 3.1174 | 3.1186 | 3.1295 | 3.1352 | 3.1194 | 3.1435 | 3.1528 | 3.1537 | 3.1313 | 3.1441 | 3.1528 | 3.1352 | 3.1731 | 3.1802 | 3.1815 | 3.1161 | 3.1366 | 3.1286 | 3.1160 | 3.1323 | 3.1456 | 3.1387 |
| eval/lm/c4_en-validation/PPL | lower | 22.47 | 22.76 | 22.59 | 22.62 | 22.86 | 22.99 | 22.63 | 23.19 | 23.40 | 23.42 | 22.90 | 23.20 | 23.40 | 22.99 | 23.88 | 24.05 | 24.08 | 22.56 | 23.03 | 22.84 | 22.56 | 22.93 | 23.23 | 23.07 |
| eval/lm/dolma_books-validation/CE loss | lower | 3.0452 | 3.0546 | 3.0453 | 3.0364 | 3.0600 | 3.0618 | 3.0352 | 3.0722 | 3.0916 | 3.0810 | 3.0574 | 3.0718 | 3.0917 | 3.0632 | 3.1062 | 3.1138 | 3.1104 | 3.0395 | 3.0725 | 3.0545 | 3.0395 | 3.0591 | 3.0813 | 3.0672 |
| eval/lm/dolma_books-validation/PPL | lower | 21.01 | 21.21 | 21.02 | 20.83 | 21.33 | 21.37 | 20.80 | 21.59 | 22.01 | 21.78 | 21.27 | 21.58 | 22.01 | 21.40 | 22.34 | 22.51 | 22.43 | 20.89 | 21.60 | 21.21 | 20.89 | 21.31 | 21.79 | 21.48 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.2469 | 3.2596 | 3.2535 | 3.2506 | 3.2635 | 3.2672 | 3.2518 | 3.2738 | 3.2855 | 3.2830 | 3.2626 | 3.2769 | 3.2829 | 3.2677 | 3.3037 | 3.3114 | 3.3112 | 3.2488 | 3.2701 | 3.2585 | 3.2490 | 3.2645 | 3.2776 | 3.2715 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 25.71 | 26.04 | 25.88 | 25.81 | 26.14 | 26.24 | 25.84 | 26.41 | 26.72 | 26.66 | 26.12 | 26.49 | 26.65 | 26.25 | 27.21 | 27.42 | 27.42 | 25.76 | 26.31 | 26.01 | 25.77 | 26.17 | 26.51 | 26.35 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.2865 | 2.2952 | 2.2937 | 2.2962 | 2.3098 | 2.3089 | 2.2951 | 2.3175 | 2.3204 | 2.3241 | 2.2966 | 2.3106 | 2.3152 | 2.3026 | 2.3381 | 2.3470 | 2.3483 | 2.2926 | 2.3108 | 2.3058 | 2.2953 | 2.3051 | 2.3138 | 2.3099 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 9.8409 | 9.9267 | 9.9115 | 9.9366 | 10.07 | 10.06 | 9.9254 | 10.15 | 10.18 | 10.22 | 9.9400 | 10.08 | 10.13 | 10.00 | 10.36 | 10.45 | 10.47 | 9.9003 | 10.08 | 10.03 | 9.9273 | 10.03 | 10.11 | 10.07 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.4082 | 3.4187 | 3.4153 | 3.4160 | 3.4231 | 3.4321 | 3.4157 | 3.4373 | 3.4441 | 3.4453 | 3.4230 | 3.4375 | 3.4482 | 3.4307 | 3.4599 | 3.4641 | 3.4702 | 3.4131 | 3.4326 | 3.4244 | 3.4141 | 3.4287 | 3.4388 | 3.4321 |
| eval/lm/dolma_reddit-validation/PPL | lower | 30.21 | 30.53 | 30.43 | 30.45 | 30.67 | 30.94 | 30.44 | 31.10 | 31.32 | 31.35 | 30.66 | 31.11 | 31.44 | 30.90 | 31.81 | 31.95 | 32.14 | 30.36 | 30.96 | 30.70 | 30.39 | 30.84 | 31.15 | 30.94 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.4680 | 1.4753 | 1.4735 | 1.4715 | 1.4812 | 1.4903 | 1.4727 | 1.4989 | 1.5061 | 1.5044 | 1.4812 | 1.4893 | 1.5052 | 1.4886 | 1.5202 | 1.5279 | 1.5269 | 1.4739 | 1.4951 | 1.4995 | 1.4789 | 1.4854 | 1.4945 | 1.4942 |
| eval/lm/dolma_stack-validation/PPL | lower | 4.3404 | 4.3721 | 4.3644 | 4.3559 | 4.3982 | 4.4386 | 4.3609 | 4.4769 | 4.5091 | 4.5015 | 4.3981 | 4.4338 | 4.5049 | 4.4311 | 4.5731 | 4.6087 | 4.6038 | 4.3664 | 4.4596 | 4.4797 | 4.3882 | 4.4166 | 4.4571 | 4.4556 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.7709 | 2.7856 | 2.7777 | 2.7796 | 2.7908 | 2.7948 | 2.7792 | 2.8042 | 2.8164 | 2.8142 | 2.7914 | 2.8077 | 2.8148 | 2.7967 | 2.8352 | 2.8467 | 2.8427 | 2.7712 | 2.7988 | 2.7799 | 2.7685 | 2.7930 | 2.8087 | 2.7983 |
| eval/lm/dolma_wiki-validation/PPL | lower | 15.97 | 16.21 | 16.08 | 16.11 | 16.29 | 16.36 | 16.11 | 16.51 | 16.72 | 16.68 | 16.30 | 16.57 | 16.69 | 16.39 | 17.03 | 17.23 | 17.16 | 15.98 | 16.42 | 16.12 | 15.93 | 16.33 | 16.59 | 16.42 |
| eval/lm/ice-validation/CE loss | lower | 3.2119 | 3.1924 | 3.1863 | 3.2031 | 3.2085 | 3.2208 | 3.2093 | 3.2292 | 3.2545 | 3.2282 | 3.1934 | 3.2124 | 3.2170 | 3.2185 | 3.2694 | 3.2702 | 3.2767 | 3.2226 | 3.2248 | 3.2348 | 3.2056 | 3.2077 | 3.2312 | 3.2193 |
| eval/lm/ice-validation/PPL | lower | 24.83 | 24.35 | 24.20 | 24.61 | 24.74 | 25.05 | 24.76 | 25.26 | 25.91 | 25.24 | 24.37 | 24.84 | 24.95 | 24.99 | 26.30 | 26.32 | 26.49 | 25.09 | 25.15 | 25.40 | 24.67 | 24.72 | 25.31 | 25.01 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 3.2168 | 3.2260 | 3.2178 | 3.2180 | 3.2230 | 3.2353 | 3.2222 | 3.2517 | 3.2588 | 3.2593 | 3.2403 | 3.2493 | 3.2528 | 3.2460 | 3.2701 | 3.2745 | 3.2793 | 3.2127 | 3.2345 | 3.2250 | 3.2171 | 3.2346 | 3.2437 | 3.2389 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 24.95 | 25.18 | 24.97 | 24.98 | 25.10 | 25.41 | 25.08 | 25.83 | 26.02 | 26.03 | 25.54 | 25.77 | 25.86 | 25.69 | 26.31 | 26.43 | 26.56 | 24.85 | 25.39 | 25.15 | 24.95 | 25.40 | 25.63 | 25.51 |
| eval/lm/pile-validation/CE loss | lower | 2.3807 | 2.3928 | 2.3853 | 2.3836 | 2.4013 | 2.4035 | 2.3852 | 2.4120 | 2.4201 | 2.4188 | 2.3948 | 2.4099 | 2.4143 | 2.4012 | 2.4375 | 2.4476 | 2.4462 | 2.3858 | 2.4064 | 2.3993 | 2.3890 | 2.4001 | 2.4084 | 2.4045 |
| eval/lm/pile-validation/PPL | lower | 10.81 | 10.94 | 10.86 | 10.84 | 11.04 | 11.06 | 10.86 | 11.16 | 11.25 | 11.23 | 10.97 | 11.13 | 11.18 | 11.04 | 11.44 | 11.56 | 11.54 | 10.87 | 11.09 | 11.02 | 10.90 | 11.02 | 11.12 | 11.07 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.7483 | 2.7655 | 2.7628 | 2.7568 | 2.7678 | 2.7761 | 2.7531 | 2.7907 | 2.7984 | 2.8108 | 2.7574 | 2.7734 | 2.7944 | 2.7759 | 2.8321 | 2.8327 | 2.8345 | 2.7636 | 2.7686 | 2.7842 | 2.7706 | 2.7823 | 2.7800 | 2.7854 |
| eval/lm/wikitext_103-validation/PPL | lower | 15.62 | 15.89 | 15.84 | 15.75 | 15.92 | 16.06 | 15.69 | 16.29 | 16.42 | 16.62 | 15.76 | 16.01 | 16.35 | 16.05 | 16.98 | 16.99 | 17.02 | 15.86 | 15.94 | 16.19 | 15.97 | 16.16 | 16.12 | 16.21 |
| throughput/in-loop eval batches | see metric | 826.0 | 826.0 | 826.0 | 826.0 | 826.0 | 826.0 | 826.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 | 1641.0 |
| throughput/in-loop eval time (s) | see metric | 115.5 | 113.5 | 115.2 | 108.3 | 107.9 | 121.3 | 112.7 | 228.2 | 228.7 | 235.9 | 243.2 | 242.2 | 246.4 | 241.6 | 224.3 | 233.5 | 227.6 | 341.9 | 215.6 | 220.2 | 214.4 | 211.2 | 210.3 | 297.0 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| int-275m-cx4-intd256e8k-lr1.6e-3-r1<br>`36clxd8s` | finished | original | 15864430592.0 | 30259 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/36clxd8s) |
| int-275m-cx4-intd256e8k-lr3.2e-3-r1<br>`bwqkxa1r` | finished | original | 15864430592.0 | 30259 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bwqkxa1r) |
| int-275m-cx4-intd256e8k-lr8e-4-r1<br>`nvxaejv3` | finished | original | 15864430592.0 | 30259 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/nvxaejv3) |
| int-275m-cx4-intw256e8k-lr1.6e-3-r1<br>`ttjquo05` | finished | original | 16251355136.0 | 30997 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ttjquo05) |
| int-275m-cx4-intw256e8k-lr3.2e-3-r1<br>`5u03fshf` | finished | original | 16251355136.0 | 30997 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/5u03fshf) |
| int-275m-cx4-intw256e8k-lr4e-4-r1<br>`n1gjknwg` | finished | original | 16251355136.0 | 30997 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/n1gjknwg) |
| int-275m-cx4-intw256e8k-lr8e-4-r1<br>`9n3xk8gs` | finished | original | 16251355136.0 | 30997 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/9n3xk8gs) |
| q3-275m-cx4-q3am128e8k-lr1.6e-3-r1<br>`h12fasg0` | finished | original | 16172711936.0 | 30847 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/h12fasg0) |
| q3-275m-cx4-q3am128e8k-lr3.2e-3-r1<br>`eihks7b2` | finished | original | 16172711936.0 | 30847 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/eihks7b2) |
| q3-275m-cx4-q3am128e8k-lr8e-4-r1<br>`1wke63zk` | finished | original | 16172711936.0 | 30847 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1wke63zk) |
| q3-275m-cx4-q3td128e8k-lr1.6e-3-r1<br>`bndawrpx` | finished | original | 16110845952.0 | 30729 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/bndawrpx) |
| q3-275m-cx4-q3td128e8k-lr3.2e-3-r1<br>`x06q7vzv` | finished | original | 16110845952.0 | 30729 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/x06q7vzv) |
| q3-275m-cx4-q3td128e8k-lr4e-4-r1<br>`unnqsh5j` | finished | original | 16110845952.0 | 30729 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/unnqsh5j) |
| q3-275m-cx4-q3td128e8k-lr8e-4-r1<br>`u5m4nxf2` | finished | original | 16110845952.0 | 30729 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/u5m4nxf2) |
| se-275m-cx4-se0m9-lr1.6e-3-r2<br>`6l09gyle` | finished | original | 16110845952.0 | 30729 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/6l09gyle) |
| se-275m-cx4-se0m9-lr3.2e-3-r2<br>`8bmutaw7` | finished | original | 16110845952.0 | 30729 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8bmutaw7) |
| se-275m-cx4-se0m9-lr8e-4-r2<br>`v9yomn1p` | finished | original | 16110845952.0 | 30729 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/v9yomn1p) |
| sp-275m-cx4-sp192e4k-lr1.6e-3-r1<br>`frw2gqmk` | finished | original | 16207839232.0 | 30914 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/frw2gqmk) |
| sp-275m-cx4-sp192e4k-lr3.2e-3-r1<br>`ra0oqtkh` | finished | original | 16207839232.0 | 30914 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ra0oqtkh) |
| sp-275m-cx4-sp192e4k-lr4e-4-r1<br>`yvbc691e` | finished | original | 16207839232.0 | 30914 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/yvbc691e) |
| sp-275m-cx4-sp192e4k-lr8e-4-r1<br>`35g1knkr` | finished | original | 16207839232.0 | 30914 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/35g1knkr) |
| sp-275m-cx4-sp96e4k-lr1.6e-3-r1<br>`pnmr2zza` | finished | original | 16143351808.0 | 30791 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/pnmr2zza) |
| sp-275m-cx4-sp96e4k-lr3.2e-3-r1<br>`k9v8ho4s` | finished | original | 16143351808.0 | 30791 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/k9v8ho4s) |
| sp-275m-cx4-sp96e4k-lr8e-4-r1<br>`u7londs6` | finished | original | 16143351808.0 | 30791 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/u7londs6) |

## unknown Cx8

Showing first 24 of 32 runs in this table. Use `--name-regex` to narrow the view.

| metric | direction | int-275m-cx8-intd256e8k-lr1.6e-3-r1<br>`825sabjc` | int-275m-cx8-intd256e8k-lr3.2e-3-r1<br>`05tuklbj` | int-275m-cx8-intd256e8k-lr8e-4-r1<br>`tkon9v64` | int-275m-cx8-intw256e8k-lr1.6e-3-r1<br>`qu2zaxr7` | int-275m-cx8-intw256e8k-lr3.2e-3-r1<br>`235ye5lg` | int-275m-cx8-intw256e8k-lr4e-4-r1<br>`iv901lom` | int-275m-cx8-intw256e8k-lr8e-4-r1<br>`qe052lo4` | mt-275m-baseline-cx8-lr1.6e-3-r1<br>`edljug2e` | mt-275m-baseline-cx8-lr2e-4-r1<br>`4amcsbx6` | mt-275m-baseline-cx8-lr4e-4-r1<br>`8jdqtmgg` | mt-275m-baseline-cx8-lr8e-4-r1<br>`drm1ceit` | mt-eval-275m-baseline-cx8-lr1.6e-3-r1<br>`y0s527t1` | mt-eval-275m-baseline-cx8-lr2e-4-r1<br>`zh77avw2` | mt-eval-275m-baseline-cx8-lr4e-4-r1<br>`cnav18vq` | mt-eval-275m-baseline-cx8-lr8e-4-r1<br>`g4j4q5z2` | q3-275m-cx8-q3am128e8k-lr1.6e-3-r1<br>`ozai8yzb` | q3-275m-cx8-q3am128e8k-lr3.2e-3-r1<br>`z3g9yfct` | q3-275m-cx8-q3am128e8k-lr8e-4-r1<br>`ap78ohx0` | q3-275m-cx8-q3td128e8k-lr1.6e-3-r1<br>`lwdmnwcj` | q3-275m-cx8-q3td128e8k-lr3.2e-3-r1<br>`uhxom59e` | q3-275m-cx8-q3td128e8k-lr4e-4-r1<br>`f89akz4d` | q3-275m-cx8-q3td128e8k-lr8e-4-r1<br>`rdzgpg7g` | se-275m-cx8-se0m9-lr1.6e-3-r2<br>`1m1rgjkc` | se-275m-cx8-se0m9-lr3.2e-3-r2<br>`vkhbbadr` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 0.91376 | 0.91469 | 0.90867 | 0.92624 | 0.92016 | 0.92800 | 0.92163 | 0.90970 | 0.86890 | 0.88767 | 0.91593 | 0.90970 | 0.86890 | 0.88767 | 0.91593 | 0.93159 | 0.92050 | 0.93127 | 0.91382 | 0.92465 | 0.92262 | 0.92468 | 0.95587 | 0.94331 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 1.0011 | 0.99913 | 0.99352 | 1.0148 | 1.0075 | 1.0146 | 1.0065 | 0.99320 | 0.94960 | 0.96898 | 0.99863 | 0.99320 | 0.94960 | 0.96898 | 0.99863 | 1.0180 | 1.0064 | 1.0196 | 1.0007 | 1.0123 | 1.0097 | 1.0126 | 1.0445 | 1.0311 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 1.0095 | 1.0285 | 1.0025 | 1.0137 | 1.0163 | 1.0160 | 1.0146 | 1.0389 | 1.0584 | 1.0015 | 0.99425 | 1.0389 | 1.0584 | 1.0015 | 0.99425 | 1.0199 | 1.0184 | 1.0394 | 1.0262 | 1.0193 | 1.0141 | 1.0270 | 1.0308 | 1.0212 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 2.0191 | 2.0571 | 2.0050 | 2.0273 | 2.0326 | 2.0319 | 2.0292 | 2.0778 | 2.1167 | 2.0029 | 1.9885 | 2.0778 | 2.1167 | 2.0029 | 1.9885 | 2.0398 | 2.0367 | 2.0788 | 2.0524 | 2.0386 | 2.0282 | 2.0539 | 2.0615 | 2.0424 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 0.69991 | 0.71294 | 0.69504 | 0.70267 | 0.70450 | 0.70424 | 0.70327 | 0.72010 | 0.73360 | 0.69418 | 0.68922 | 0.72010 | 0.73360 | 0.69418 | 0.68922 | 0.70700 | 0.70591 | 0.72052 | 0.71140 | 0.70660 | 0.70302 | 0.71190 | 0.71460 | 0.70787 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 1.3998 | 1.4259 | 1.3901 | 1.4053 | 1.4090 | 1.4085 | 1.4065 | 1.4402 | 1.4672 | 1.3884 | 1.3784 | 1.4402 | 1.4672 | 1.3884 | 1.3784 | 1.4140 | 1.4118 | 1.4410 | 1.4228 | 1.4132 | 1.4060 | 1.4238 | 1.4292 | 1.4157 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.24061 | 0.23123 | 0.26962 | 0.26365 | 0.23891 | 0.23123 | 0.25939 | 0.30717 | 0.31997 | 0.30461 | 0.29181 | 0.30717 | 0.31997 | 0.30461 | 0.29181 | 0.23891 | 0.23549 | 0.22867 | 0.22696 | 0.25085 | 0.24744 | 0.23123 | 0.23976 | 0.25085 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.24061 | 0.23123 | 0.26962 | 0.26365 | 0.23891 | 0.23123 | 0.25939 | 0.30717 | 0.31997 | 0.30461 | 0.29181 | 0.30717 | 0.31997 | 0.30461 | 0.29181 | 0.23891 | 0.23549 | 0.22867 | 0.22696 | 0.25085 | 0.24744 | 0.23123 | 0.23976 | 0.25085 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -1.3961 | -1.4216 | -1.3833 | -1.3962 | -1.4008 | -1.4027 | -1.4015 | -1.4380 | -1.4652 | -1.3858 | -1.3746 | -1.4380 | -1.4652 | -1.3858 | -1.3746 | -1.4066 | -1.4023 | -1.4319 | -1.4103 | -1.4058 | -1.3979 | -1.4118 | -1.4142 | -1.4051 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -1.3961 | -1.4216 | -1.3833 | -1.3962 | -1.4008 | -1.4027 | -1.4015 | -1.4380 | -1.4652 | -1.3858 | -1.3746 | -1.4380 | -1.4652 | -1.3858 | -1.3746 | -1.4066 | -1.4023 | -1.4319 | -1.4103 | -1.4058 | -1.3979 | -1.4118 | -1.4142 | -1.4051 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.25010 | 0.24956 | 0.25372 | 0.25193 | 0.24932 | 0.24978 | 0.25329 | 0.26703 | 0.28740 | 0.27147 | 0.26320 | 0.26703 | 0.28740 | 0.27147 | 0.26320 | 0.25165 | 0.24883 | 0.24873 | 0.24845 | 0.25071 | 0.25033 | 0.24695 | 0.24863 | 0.25054 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.25010 | 0.24956 | 0.25372 | 0.25193 | 0.24932 | 0.24978 | 0.25329 | 0.26703 | 0.28740 | 0.27147 | 0.26320 | 0.26703 | 0.28740 | 0.27147 | 0.26320 | 0.25165 | 0.24883 | 0.24873 | 0.24845 | 0.25071 | 0.25033 | 0.24695 | 0.24863 | 0.25054 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 0.70377 | 0.72162 | 0.71018 | 0.71366 | 0.70220 | 0.72874 | 0.72480 | 0.71477 | 0.66515 | 0.68166 | 0.69719 | 0.71477 | 0.66515 | 0.68166 | 0.69719 | 0.72618 | 0.70904 | 0.73009 | 0.71300 | 0.72017 | 0.71975 | 0.72419 | 0.75508 | 0.75748 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 0.76542 | 0.78456 | 0.77299 | 0.77695 | 0.76470 | 0.79373 | 0.78838 | 0.77633 | 0.72352 | 0.74132 | 0.75811 | 0.77633 | 0.72352 | 0.74132 | 0.75811 | 0.79015 | 0.77109 | 0.79474 | 0.77644 | 0.78321 | 0.78389 | 0.78871 | 0.82335 | 0.82582 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 1.0122 | 1.0227 | 1.0182 | 1.0194 | 1.0245 | 1.0175 | 1.0215 | 0.99402 | 0.96440 | 0.98397 | 0.95077 | 0.99402 | 0.96440 | 0.98397 | 0.95077 | 1.0336 | 1.0162 | 1.0249 | 1.0233 | 1.0280 | 1.0210 | 1.0247 | 1.0316 | 1.0286 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 2.0245 | 2.0455 | 2.0364 | 2.0388 | 2.0490 | 2.0350 | 2.0429 | 1.9880 | 1.9288 | 1.9679 | 1.9015 | 1.9880 | 1.9288 | 1.9679 | 1.9015 | 2.0671 | 2.0324 | 2.0498 | 2.0466 | 2.0561 | 2.0420 | 2.0495 | 2.0632 | 2.0572 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 0.70173 | 0.70896 | 0.70586 | 0.70669 | 0.71016 | 0.70532 | 0.70808 | 0.68902 | 0.66847 | 0.68201 | 0.65906 | 0.68902 | 0.66847 | 0.68201 | 0.65906 | 0.71645 | 0.70443 | 0.71049 | 0.70939 | 0.71256 | 0.70777 | 0.71037 | 0.71507 | 0.71300 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 1.4035 | 1.4179 | 1.4117 | 1.4134 | 1.4203 | 1.4106 | 1.4162 | 1.3780 | 1.3369 | 1.3640 | 1.3181 | 1.3780 | 1.3369 | 1.3640 | 1.3181 | 1.4329 | 1.4089 | 1.4210 | 1.4188 | 1.4251 | 1.4155 | 1.4207 | 1.4301 | 1.4260 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.26221 | 0.25926 | 0.24074 | 0.24579 | 0.25210 | 0.24579 | 0.25295 | 0.32912 | 0.42130 | 0.37416 | 0.37247 | 0.32912 | 0.42130 | 0.37416 | 0.37247 | 0.24621 | 0.24916 | 0.25421 | 0.23527 | 0.25547 | 0.24495 | 0.25758 | 0.24158 | 0.26136 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.26221 | 0.25926 | 0.24074 | 0.24579 | 0.25210 | 0.24579 | 0.25295 | 0.32912 | 0.42130 | 0.37416 | 0.37247 | 0.32912 | 0.42130 | 0.37416 | 0.37247 | 0.24621 | 0.24916 | 0.25421 | 0.23527 | 0.25547 | 0.24495 | 0.25758 | 0.24158 | 0.26136 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -1.3995 | -1.4139 | -1.4049 | -1.4062 | -1.4126 | -1.4053 | -1.4118 | -1.3757 | -1.3346 | -1.3613 | -1.3143 | -1.3757 | -1.3346 | -1.3613 | -1.3143 | -1.4256 | -1.4004 | -1.4111 | -1.4086 | -1.4147 | -1.4081 | -1.4095 | -1.4161 | -1.4126 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -1.3995 | -1.4139 | -1.4049 | -1.4062 | -1.4126 | -1.4053 | -1.4118 | -1.3757 | -1.3346 | -1.3613 | -1.3143 | -1.3757 | -1.3346 | -1.3613 | -1.3143 | -1.4256 | -1.4004 | -1.4111 | -1.4086 | -1.4147 | -1.4081 | -1.4095 | -1.4161 | -1.4126 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.25127 | 0.25159 | 0.24994 | 0.25122 | 0.24999 | 0.24928 | 0.25099 | 0.27773 | 0.33045 | 0.30209 | 0.27870 | 0.27773 | 0.33045 | 0.30209 | 0.27870 | 0.24959 | 0.24941 | 0.25112 | 0.24909 | 0.25129 | 0.24978 | 0.25083 | 0.24905 | 0.25102 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.25127 | 0.25159 | 0.24994 | 0.25122 | 0.24999 | 0.24928 | 0.25099 | 0.27773 | 0.33045 | 0.30209 | 0.27870 | 0.27773 | 0.33045 | 0.30209 | 0.27870 | 0.24959 | 0.24941 | 0.25112 | 0.24909 | 0.25129 | 0.24978 | 0.25083 | 0.24905 | 0.25102 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 1.4391 | 1.5640 | 1.6048 | 1.4888 | 1.4274 | 1.6687 | 1.6328 | 0.53748 | 0.67539 | 0.64967 | 0.52547 | 0.53748 | 0.67539 | 0.64967 | 0.52547 | 1.6255 | 1.6920 | 1.6300 | 1.6465 | 1.6276 | 1.7993 | 1.6496 | 1.6932 | 1.7167 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 2.3174 | 2.5052 | 2.5652 | 2.3960 | 2.2936 | 2.6640 | 2.6062 | 0.86121 | 1.0851 | 1.0427 | 0.83556 | 0.86121 | 1.0851 | 1.0427 | 0.83556 | 2.6254 | 2.6945 | 2.6184 | 2.6337 | 2.6229 | 2.8918 | 2.6524 | 2.7133 | 2.7573 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 0.99751 | 1.0841 | 1.1123 | 1.0319 | 0.98935 | 1.1565 | 1.1317 | 0.37252 | 0.46814 | 0.45029 | 0.36423 | 0.37252 | 0.46814 | 0.45029 | 0.36423 | 1.1266 | 1.1728 | 1.1298 | 1.1411 | 1.1281 | 1.2471 | 1.1434 | 1.1737 | 1.1900 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 1.6063 | 1.7365 | 1.7782 | 1.6610 | 1.5897 | 1.8465 | 1.8065 | 0.59696 | 0.75220 | 0.72279 | 0.57914 | 0.59696 | 0.75220 | 0.72279 | 0.57914 | 1.8199 | 1.8675 | 1.8148 | 1.8256 | 1.8180 | 2.0044 | 1.8383 | 1.8808 | 1.9114 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.40019 | 0.36199 | 0.34862 | 0.36485 | 0.42025 | 0.29035 | 0.33906 | 0.80898 | 0.75549 | 0.78032 | 0.81471 | 0.80898 | 0.75549 | 0.78032 | 0.81471 | 0.33620 | 0.32474 | 0.34097 | 0.32951 | 0.37345 | 0.30659 | 0.28653 | 0.27985 | 0.30755 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.40019 | 0.36199 | 0.34862 | 0.36485 | 0.42025 | 0.29035 | 0.33906 | 0.80898 | 0.75549 | 0.78032 | 0.81471 | 0.80898 | 0.75549 | 0.78032 | 0.81471 | 0.33620 | 0.32474 | 0.34097 | 0.32951 | 0.37345 | 0.30659 | 0.28653 | 0.27985 | 0.30755 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -1.8324 | -1.9791 | -2.1595 | -1.8873 | -1.7969 | -2.1770 | -2.0519 | -0.61183 | -0.76073 | -0.72979 | -0.58688 | -0.61183 | -0.76073 | -0.72979 | -0.58688 | -1.9915 | -2.2898 | -2.1532 | -2.1774 | -2.0416 | -2.3312 | -2.1774 | -2.2208 | -2.1851 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -1.8324 | -1.9791 | -2.1595 | -1.8873 | -1.7969 | -2.1770 | -2.0519 | -0.61183 | -0.76073 | -0.72979 | -0.58688 | -0.61183 | -0.76073 | -0.72979 | -0.58688 | -1.9915 | -2.2898 | -2.1532 | -2.1774 | -2.0416 | -2.3312 | -2.1774 | -2.2208 | -2.1851 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.32505 | 0.28653 | 0.29330 | 0.31227 | 0.35580 | 0.25636 | 0.27244 | 0.74300 | 0.71088 | 0.72648 | 0.75770 | 0.74300 | 0.71088 | 0.72648 | 0.75770 | 0.28646 | 0.29021 | 0.30111 | 0.26141 | 0.29232 | 0.24931 | 0.22668 | 0.22692 | 0.25539 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.32505 | 0.28653 | 0.29330 | 0.31227 | 0.35580 | 0.25636 | 0.27244 | 0.74300 | 0.71088 | 0.72648 | 0.75770 | 0.74300 | 0.71088 | 0.72648 | 0.75770 | 0.28646 | 0.29021 | 0.30111 | 0.26141 | 0.29232 | 0.24931 | 0.22668 | 0.22692 | 0.25539 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 0.43546 | 0.43128 | 0.43007 | 0.42633 | 0.48373 | 0.44002 | 0.43332 | 0.37626 | 0.42875 | 0.38882 | 0.40584 | 0.37626 | 0.42875 | 0.38882 | 0.40584 | 0.40616 | 0.43025 | 0.44667 | 0.43418 | 0.42826 | 0.41436 | 0.42262 | 0.43525 | 0.42140 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 0.47594 | 0.47032 | 0.46828 | 0.46440 | 0.52687 | 0.47945 | 0.47329 | 0.41018 | 0.46679 | 0.42271 | 0.44186 | 0.41018 | 0.46679 | 0.42271 | 0.44186 | 0.44156 | 0.46749 | 0.48662 | 0.47333 | 0.46690 | 0.45191 | 0.45916 | 0.47455 | 0.45802 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 0.30184 | 0.29892 | 0.29813 | 0.29550 | 0.33528 | 0.30499 | 0.30037 | 0.26080 | 0.29715 | 0.26951 | 0.28132 | 0.26080 | 0.29715 | 0.26951 | 0.28132 | 0.28151 | 0.29821 | 0.30962 | 0.30097 | 0.29686 | 0.28719 | 0.29293 | 0.30173 | 0.29209 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 0.32990 | 0.32600 | 0.32458 | 0.32192 | 0.36516 | 0.33235 | 0.32807 | 0.28429 | 0.32356 | 0.29301 | 0.30628 | 0.28429 | 0.32356 | 0.29301 | 0.30628 | 0.30608 | 0.32407 | 0.33729 | 0.32810 | 0.32363 | 0.31321 | 0.31823 | 0.32897 | 0.31745 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.53261 | 0.50692 | 0.50593 | 0.54447 | 0.53953 | 0.50988 | 0.52569 | 0.72036 | 0.71542 | 0.72727 | 0.72727 | 0.72036 | 0.71542 | 0.72727 | 0.72727 | 0.52866 | 0.51976 | 0.51976 | 0.51976 | 0.53063 | 0.51186 | 0.49209 | 0.49901 | 0.49209 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.53261 | 0.50692 | 0.50593 | 0.54447 | 0.53953 | 0.50988 | 0.52569 | 0.72036 | 0.71542 | 0.72727 | 0.72727 | 0.72036 | 0.71542 | 0.72727 | 0.72727 | 0.52866 | 0.51976 | 0.51976 | 0.51976 | 0.53063 | 0.51186 | 0.49209 | 0.49901 | 0.49209 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -2.2215 | -2.2751 | -2.2935 | -2.1698 | -2.2842 | -2.2697 | -2.1237 | -1.1063 | -1.0867 | -1.1009 | -1.0207 | -1.1063 | -1.0867 | -1.1009 | -1.0207 | -2.3295 | -2.3531 | -2.3644 | -2.2916 | -2.3523 | -2.3705 | -2.5337 | -2.3985 | -2.4999 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -2.2215 | -2.2751 | -2.2935 | -2.1698 | -2.2842 | -2.2697 | -2.1237 | -1.1063 | -1.0867 | -1.1009 | -1.0207 | -1.1063 | -1.0867 | -1.1009 | -1.0207 | -2.3295 | -2.3531 | -2.3644 | -2.2916 | -2.3523 | -2.3705 | -2.5337 | -2.3985 | -2.4999 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.51495 | 0.49837 | 0.50168 | 0.51672 | 0.51612 | 0.49404 | 0.50619 | 0.69943 | 0.70050 | 0.70888 | 0.71252 | 0.69943 | 0.70050 | 0.70888 | 0.71252 | 0.50500 | 0.49452 | 0.49434 | 0.49906 | 0.50089 | 0.48994 | 0.47033 | 0.48356 | 0.47979 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.51495 | 0.49837 | 0.50168 | 0.51672 | 0.51612 | 0.49404 | 0.50619 | 0.69943 | 0.70050 | 0.70888 | 0.71252 | 0.69943 | 0.70050 | 0.70888 | 0.71252 | 0.50500 | 0.49452 | 0.49434 | 0.49906 | 0.50089 | 0.48994 | 0.47033 | 0.48356 | 0.47979 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 0.46253 | 0.47749 | 0.45233 | 0.48790 | 0.41976 | 0.54552 | 0.53865 | 0.57099 | 0.41584 | 0.42808 | 0.46691 | 0.57099 | 0.41584 | 0.42808 | 0.46691 | 0.43635 | 0.53416 | 0.52680 | 0.48171 | 0.43303 | 0.55571 | 0.45804 | 0.46269 | 0.47970 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 0.55715 | 0.57616 | 0.54330 | 0.58625 | 0.50425 | 0.65634 | 0.64864 | 0.68624 | 0.49918 | 0.51560 | 0.56114 | 0.68624 | 0.49918 | 0.51560 | 0.56114 | 0.52380 | 0.64213 | 0.63093 | 0.57900 | 0.52111 | 0.66725 | 0.54990 | 0.55643 | 0.57564 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 0.32075 | 0.33110 | 0.31361 | 0.33833 | 0.29115 | 0.37820 | 0.37352 | 0.39590 | 0.28835 | 0.29681 | 0.32376 | 0.39590 | 0.28835 | 0.29681 | 0.32376 | 0.30266 | 0.37048 | 0.36527 | 0.33396 | 0.30030 | 0.38541 | 0.31762 | 0.32088 | 0.33261 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 0.38635 | 0.39947 | 0.37670 | 0.40661 | 0.34977 | 0.45507 | 0.44982 | 0.47588 | 0.34622 | 0.35752 | 0.38910 | 0.47588 | 0.34622 | 0.35752 | 0.38910 | 0.36332 | 0.44531 | 0.43752 | 0.40145 | 0.36142 | 0.46273 | 0.38138 | 0.38587 | 0.39914 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.82160 | 0.83799 | 0.81003 | 0.79749 | 0.81678 | 0.83124 | 0.82257 | 0.75217 | 0.83317 | 0.83414 | 0.80810 | 0.75217 | 0.83317 | 0.83414 | 0.80810 | 0.83028 | 0.80328 | 0.80617 | 0.79074 | 0.80328 | 0.77338 | 0.82353 | 0.79846 | 0.81581 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.82160 | 0.83799 | 0.81003 | 0.79749 | 0.81678 | 0.83124 | 0.82257 | 0.75217 | 0.83317 | 0.83414 | 0.80810 | 0.75217 | 0.83317 | 0.83414 | 0.80810 | 0.83028 | 0.80328 | 0.80617 | 0.79074 | 0.80328 | 0.77338 | 0.82353 | 0.79846 | 0.81581 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -0.49216 | -0.47508 | -0.52070 | -0.51995 | -0.48335 | -0.49923 | -0.47870 | -0.64738 | -0.45539 | -0.49365 | -0.53345 | -0.64738 | -0.45539 | -0.49365 | -0.53345 | -0.49209 | -0.53105 | -0.53077 | -0.53400 | -0.51776 | -0.61542 | -0.49649 | -0.55035 | -0.51806 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -0.49216 | -0.47508 | -0.52070 | -0.51995 | -0.48335 | -0.49923 | -0.47870 | -0.64738 | -0.45539 | -0.49365 | -0.53345 | -0.64738 | -0.45539 | -0.49365 | -0.53345 | -0.49209 | -0.53105 | -0.53077 | -0.53400 | -0.51776 | -0.61542 | -0.49649 | -0.55035 | -0.51806 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.72178 | 0.72846 | 0.71447 | 0.70984 | 0.72825 | 0.71114 | 0.72877 | 0.66858 | 0.73205 | 0.72526 | 0.70574 | 0.66858 | 0.73205 | 0.72526 | 0.70574 | 0.73615 | 0.70268 | 0.70186 | 0.70846 | 0.72071 | 0.67223 | 0.71504 | 0.69070 | 0.71710 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.72178 | 0.72846 | 0.71447 | 0.70984 | 0.72825 | 0.71114 | 0.72877 | 0.66858 | 0.73205 | 0.72526 | 0.70574 | 0.66858 | 0.73205 | 0.72526 | 0.70574 | 0.73615 | 0.70268 | 0.70186 | 0.70846 | 0.72071 | 0.67223 | 0.71504 | 0.69070 | 0.71710 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 0.27970 | 0.30045 | 0.25786 | 0.32856 | 0.29358 | 0.29368 | 0.28599 | 0.27956 | 0.29338 | 0.30529 | 0.25685 | 0.27956 | 0.29338 | 0.30529 | 0.25685 | 0.28237 | 0.29575 | 0.31414 | 0.29857 | 0.27299 | 0.29570 | 0.28310 | 0.27721 | 0.28173 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 0.28905 | 0.31052 | 0.26641 | 0.33959 | 0.30349 | 0.30361 | 0.29560 | 0.28891 | 0.30316 | 0.31546 | 0.26542 | 0.28891 | 0.30316 | 0.31546 | 0.26542 | 0.29191 | 0.30555 | 0.32463 | 0.30863 | 0.28216 | 0.30564 | 0.29263 | 0.28648 | 0.29136 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 0.19389 | 0.20827 | 0.17874 | 0.22775 | 0.20354 | 0.20356 | 0.19827 | 0.19379 | 0.20338 | 0.21163 | 0.17807 | 0.19379 | 0.20338 | 0.21163 | 0.17807 | 0.19575 | 0.20503 | 0.21775 | 0.20697 | 0.18926 | 0.20498 | 0.19625 | 0.19218 | 0.19531 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 0.20038 | 0.21526 | 0.18466 | 0.23539 | 0.21036 | 0.21048 | 0.20493 | 0.20028 | 0.21017 | 0.21869 | 0.18401 | 0.20028 | 0.21017 | 0.21869 | 0.18401 | 0.20236 | 0.21181 | 0.22503 | 0.21395 | 0.19561 | 0.21187 | 0.20285 | 0.19859 | 0.20196 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.87657 | 0.82648 | 0.81664 | 0.85868 | 0.82648 | 0.84526 | 0.82290 | 0.88104 | 0.82737 | 0.83095 | 0.84526 | 0.88104 | 0.82737 | 0.83095 | 0.84526 | 0.82290 | 0.82200 | 0.81395 | 0.87120 | 0.82558 | 0.86941 | 0.85242 | 0.85689 | 0.81127 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.87657 | 0.82648 | 0.81664 | 0.85868 | 0.82648 | 0.84526 | 0.82290 | 0.88104 | 0.82737 | 0.83095 | 0.84526 | 0.88104 | 0.82737 | 0.83095 | 0.84526 | 0.82290 | 0.82200 | 0.81395 | 0.87120 | 0.82558 | 0.86941 | 0.85242 | 0.85689 | 0.81127 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -0.33677 | -0.48760 | -0.43874 | -0.40015 | -0.42949 | -0.44434 | -0.49651 | -0.34205 | -0.42338 | -0.43458 | -0.39605 | -0.34205 | -0.42338 | -0.43458 | -0.39605 | -0.42589 | -0.49953 | -0.51566 | -0.35404 | -0.45032 | -0.40314 | -0.41857 | -0.38050 | -0.50411 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -0.33677 | -0.48760 | -0.43874 | -0.40015 | -0.42949 | -0.44434 | -0.49651 | -0.34205 | -0.42338 | -0.43458 | -0.39605 | -0.34205 | -0.42338 | -0.43458 | -0.39605 | -0.42589 | -0.49953 | -0.51566 | -0.35404 | -0.45032 | -0.40314 | -0.41857 | -0.38050 | -0.50411 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.85557 | 0.81910 | 0.81376 | 0.82517 | 0.81264 | 0.82161 | 0.81431 | 0.86230 | 0.83332 | 0.83485 | 0.84300 | 0.86230 | 0.83332 | 0.83485 | 0.84300 | 0.82257 | 0.80705 | 0.80589 | 0.84468 | 0.80854 | 0.84152 | 0.83265 | 0.83825 | 0.80630 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.85557 | 0.81910 | 0.81376 | 0.82517 | 0.81264 | 0.82161 | 0.81431 | 0.86230 | 0.83332 | 0.83485 | 0.84300 | 0.86230 | 0.83332 | 0.83485 | 0.84300 | 0.82257 | 0.80705 | 0.80589 | 0.84468 | 0.80854 | 0.84152 | 0.83265 | 0.83825 | 0.80630 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 1.0717 | 0.95094 | 0.95403 | 0.90367 | 0.91141 | 0.94932 | 0.88549 | 1.0850 | 0.98241 | 0.96241 | 1.0032 | 1.0850 | 0.98241 | 0.96241 | 1.0032 | 0.95645 | 1.0092 | 1.0345 | 0.93081 | 0.98371 | 1.0147 | 0.93928 | 1.0277 | 1.1357 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 1.7329 | 1.5219 | 1.5464 | 1.4606 | 1.4947 | 1.5309 | 1.4437 | 1.7488 | 1.5959 | 1.5595 | 1.6136 | 1.7488 | 1.5959 | 1.5595 | 1.6136 | 1.5369 | 1.6399 | 1.6689 | 1.5126 | 1.5936 | 1.6354 | 1.5341 | 1.6588 | 1.8340 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 0.77865 | 0.69482 | 0.69448 | 0.65633 | 0.66313 | 0.68918 | 0.64499 | 0.79738 | 0.70918 | 0.70760 | 0.73758 | 0.79738 | 0.70918 | 0.70760 | 0.73758 | 0.69617 | 0.73597 | 0.75163 | 0.68092 | 0.71614 | 0.73511 | 0.68728 | 0.75354 | 0.82893 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 1.2938 | 1.1463 | 1.1573 | 1.0888 | 1.1166 | 1.1405 | 1.0803 | 1.3283 | 1.1781 | 1.1851 | 1.2264 | 1.3283 | 1.1781 | 1.1851 | 1.2264 | 1.1494 | 1.2288 | 1.2456 | 1.1409 | 1.1921 | 1.2138 | 1.1567 | 1.2553 | 1.3785 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.64045 | 0.67603 | 0.65918 | 0.67416 | 0.66667 | 0.65918 | 0.69663 | 0.66105 | 0.67041 | 0.69288 | 0.67041 | 0.66105 | 0.67041 | 0.69288 | 0.67041 | 0.61049 | 0.63670 | 0.64981 | 0.68727 | 0.63296 | 0.61985 | 0.65730 | 0.63109 | 0.62172 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.64045 | 0.67603 | 0.65918 | 0.67416 | 0.66667 | 0.65918 | 0.69663 | 0.66105 | 0.67041 | 0.69288 | 0.67041 | 0.66105 | 0.67041 | 0.69288 | 0.67041 | 0.61049 | 0.63670 | 0.64981 | 0.68727 | 0.63296 | 0.61985 | 0.65730 | 0.63109 | 0.62172 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -0.90562 | -0.83073 | -0.87106 | -0.82621 | -0.82685 | -0.87672 | -0.78667 | -0.91216 | -0.85807 | -0.88042 | -0.87968 | -0.91216 | -0.85807 | -0.88042 | -0.87968 | -0.92076 | -0.91961 | -0.88687 | -0.86294 | -0.86660 | -0.92625 | -0.87710 | -0.95029 | -0.94859 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -0.90562 | -0.83073 | -0.87106 | -0.82621 | -0.82685 | -0.87672 | -0.78667 | -0.91216 | -0.85807 | -0.88042 | -0.87968 | -0.91216 | -0.85807 | -0.88042 | -0.87968 | -0.92076 | -0.91961 | -0.88687 | -0.86294 | -0.86660 | -0.92625 | -0.87710 | -0.95029 | -0.94859 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.56577 | 0.59504 | 0.57463 | 0.60152 | 0.60911 | 0.59247 | 0.61899 | 0.59220 | 0.59597 | 0.61575 | 0.60133 | 0.59220 | 0.59597 | 0.61575 | 0.60133 | 0.56852 | 0.57717 | 0.56850 | 0.59728 | 0.57939 | 0.55680 | 0.58228 | 0.56005 | 0.54088 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.56577 | 0.59504 | 0.57463 | 0.60152 | 0.60911 | 0.59247 | 0.61899 | 0.59220 | 0.59597 | 0.61575 | 0.60133 | 0.59220 | 0.59597 | 0.61575 | 0.60133 | 0.56852 | 0.57717 | 0.56850 | 0.59728 | 0.57939 | 0.55680 | 0.58228 | 0.56005 | 0.54088 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 1.8315 | 1.7213 | 1.7545 | 1.7286 | 1.9004 | 1.8240 | 1.8993 | 1.4802 | 1.4228 | 1.4875 | 1.3942 | 1.4802 | 1.4228 | 1.4875 | 1.3942 | 1.9401 | 1.9095 | 1.8839 | 1.8837 | 1.8344 | 1.9464 | 1.8542 | 1.8855 | 1.9586 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 2.5285 | 2.3765 | 2.4108 | 2.3837 | 2.6084 | 2.5200 | 2.6105 | 2.0493 | 1.9750 | 2.0425 | 1.9355 | 2.0493 | 1.9750 | 2.0425 | 1.9355 | 2.6803 | 2.6114 | 2.5975 | 2.5891 | 2.5492 | 2.6698 | 2.5534 | 2.5928 | 2.6848 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 1.2695 | 1.1932 | 1.2162 | 1.1982 | 1.3173 | 1.2643 | 1.3166 | 1.0260 | 0.98621 | 1.0311 | 0.96640 | 1.0260 | 0.98621 | 1.0311 | 0.96640 | 1.3448 | 1.3235 | 1.3057 | 1.3056 | 1.2715 | 1.3490 | 1.2853 | 1.3069 | 1.3576 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 1.7526 | 1.6473 | 1.6712 | 1.6523 | 1.8080 | 1.7467 | 1.8094 | 1.4205 | 1.3691 | 1.4158 | 1.3417 | 1.4205 | 1.3691 | 1.4158 | 1.3417 | 1.8579 | 1.8101 | 1.8004 | 1.7948 | 1.7670 | 1.8506 | 1.7698 | 1.7971 | 1.8610 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.20837 | 0.21493 | 0.22395 | 0.21247 | 0.21985 | 0.23052 | 0.22806 | 0.32158 | 0.31747 | 0.29040 | 0.33306 | 0.32158 | 0.31747 | 0.29040 | 0.33306 | 0.22559 | 0.23708 | 0.21083 | 0.23298 | 0.22642 | 0.21329 | 0.24036 | 0.20098 | 0.21657 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.20837 | 0.21493 | 0.22395 | 0.21247 | 0.21985 | 0.23052 | 0.22806 | 0.32158 | 0.31747 | 0.29040 | 0.33306 | 0.32158 | 0.31747 | 0.29040 | 0.33306 | 0.22559 | 0.23708 | 0.21083 | 0.23298 | 0.22642 | 0.21329 | 0.24036 | 0.20098 | 0.21657 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -4.1039 | -4.0669 | -4.1437 | -4.1243 | -3.9699 | -4.2853 | -4.1154 | -3.4066 | -3.2438 | -3.3550 | -3.1842 | -3.4066 | -3.2438 | -3.3550 | -3.1842 | -4.1244 | -4.0472 | -4.2604 | -4.1995 | -4.2970 | -4.3765 | -4.2330 | -4.3828 | -4.2146 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -4.1039 | -4.0669 | -4.1437 | -4.1243 | -3.9699 | -4.2853 | -4.1154 | -3.4066 | -3.2438 | -3.3550 | -3.1842 | -3.4066 | -3.2438 | -3.3550 | -3.1842 | -4.1244 | -4.0472 | -4.2604 | -4.1995 | -4.2970 | -4.3765 | -4.2330 | -4.3828 | -4.2146 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.24124 | 0.24166 | 0.24837 | 0.23893 | 0.24639 | 0.24190 | 0.25032 | 0.32783 | 0.32756 | 0.31172 | 0.33903 | 0.32783 | 0.32756 | 0.31172 | 0.33903 | 0.24232 | 0.24610 | 0.23724 | 0.24435 | 0.23985 | 0.23705 | 0.25201 | 0.22776 | 0.23819 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.24124 | 0.24166 | 0.24837 | 0.23893 | 0.24639 | 0.24190 | 0.25032 | 0.32783 | 0.32756 | 0.31172 | 0.33903 | 0.32783 | 0.32756 | 0.31172 | 0.33903 | 0.24232 | 0.24610 | 0.23724 | 0.24435 | 0.23985 | 0.23705 | 0.25201 | 0.22776 | 0.23819 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 0.46658 | 0.46675 | 0.45632 | 0.46914 | 0.46318 | 0.48198 | 0.47261 | 0.42595 | 0.37988 | 0.38802 | 0.39949 | 0.42595 | 0.37988 | 0.38802 | 0.39949 | 0.46831 | 0.48723 | 0.46883 | 0.46947 | 0.47208 | 0.47589 | 0.48221 | 0.49458 | 0.48410 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 0.47271 | 0.47270 | 0.46233 | 0.47541 | 0.46933 | 0.48836 | 0.47886 | 0.43172 | 0.38485 | 0.39309 | 0.40512 | 0.43172 | 0.38485 | 0.39309 | 0.40512 | 0.47426 | 0.49385 | 0.47520 | 0.47573 | 0.47860 | 0.48183 | 0.48855 | 0.50125 | 0.49029 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 0.67371 | 0.67885 | 0.67860 | 0.67219 | 0.66712 | 0.67240 | 0.66177 | 0.66491 | 0.62432 | 0.64576 | 0.63079 | 0.66491 | 0.62432 | 0.64576 | 0.63079 | 0.67606 | 0.70085 | 0.67391 | 0.67484 | 0.67106 | 0.67931 | 0.68265 | 0.69864 | 0.68143 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 0.67943 | 0.68479 | 0.68455 | 0.67778 | 0.67287 | 0.67821 | 0.66738 | 0.67071 | 0.62940 | 0.65140 | 0.63636 | 0.67071 | 0.62940 | 0.65140 | 0.63636 | 0.68182 | 0.70710 | 0.67955 | 0.68090 | 0.67705 | 0.68526 | 0.68858 | 0.70463 | 0.68717 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 2.8623 | 2.7988 | 2.8862 | 2.2286 | 2.5320 | 2.5365 | 2.1742 | 2.3464 | 2.1948 | 1.9414 | 1.9959 | 2.3464 | 2.1948 | 1.9414 | 1.9959 | 2.6869 | 2.2890 | 2.5595 | 2.2517 | 2.8490 | 2.8555 | 2.5787 | 2.4305 | 2.4140 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 5.7246 | 5.5977 | 5.7723 | 4.4571 | 5.0640 | 5.0730 | 4.3484 | 4.6928 | 4.3895 | 3.8827 | 3.9919 | 4.6928 | 4.3895 | 3.8827 | 3.9919 | 5.3738 | 4.5780 | 5.1191 | 4.5034 | 5.6980 | 5.7109 | 5.1575 | 4.8610 | 4.8280 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 1.9840 | 1.9412 | 2.0008 | 1.5445 | 1.7552 | 1.7588 | 1.5074 | 1.6261 | 1.5213 | 1.3460 | 1.3834 | 1.6261 | 1.5213 | 1.3460 | 1.3834 | 1.8620 | 1.5867 | 1.7738 | 1.5607 | 1.9751 | 1.9789 | 1.7876 | 1.6851 | 1.6736 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 3.9680 | 3.8825 | 4.0016 | 3.0890 | 3.5103 | 3.5176 | 3.0148 | 3.2521 | 3.0426 | 2.6920 | 2.7669 | 3.2521 | 3.0426 | 2.6920 | 2.7669 | 3.7239 | 3.1734 | 3.5476 | 3.1213 | 3.9502 | 3.9577 | 3.5752 | 3.3702 | 3.3473 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.06000 | 0.11000 | 0.06000 | 0.09000 | 0.10000 | 0.07000 | 0.14000 | 0.09000 | 0.07000 | 0.09000 | 0.09000 | 0.09000 | 0.07000 | 0.09000 | 0.09000 | 0.09000 | 0.10000 | 0.13000 | 0.10000 | 0.09000 | 0.07000 | 0.10000 | 0.10000 | 0.13000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.06000 | 0.11000 | 0.06000 | 0.09000 | 0.10000 | 0.07000 | 0.14000 | 0.09000 | 0.07000 | 0.09000 | 0.09000 | 0.09000 | 0.07000 | 0.09000 | 0.09000 | 0.09000 | 0.10000 | 0.13000 | 0.10000 | 0.09000 | 0.07000 | 0.10000 | 0.10000 | 0.13000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -3.9651 | -3.8794 | -3.9971 | -3.0768 | -3.5035 | -3.5138 | -3.0057 | -3.2430 | -3.0386 | -2.6847 | -2.7545 | -3.2430 | -3.0386 | -2.6847 | -2.7545 | -3.7185 | -3.1582 | -3.5333 | -3.1140 | -3.9450 | -3.9498 | -3.5688 | -3.3547 | -3.3327 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -3.9651 | -3.8794 | -3.9971 | -3.0768 | -3.5035 | -3.5138 | -3.0057 | -3.2430 | -3.0386 | -2.6847 | -2.7545 | -3.2430 | -3.0386 | -2.6847 | -2.7545 | -3.7185 | -3.1582 | -3.5333 | -3.1140 | -3.9450 | -3.9498 | -3.5688 | -3.3547 | -3.3327 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09638 | 0.09650 | 0.09084 | 0.09504 | 0.09596 | 0.09488 | 0.09613 | 0.09575 | 0.10150 | 0.10299 | 0.09943 | 0.09575 | 0.10150 | 0.10299 | 0.09943 | 0.09449 | 0.09557 | 0.09547 | 0.10074 | 0.09627 | 0.09247 | 0.09550 | 0.09678 | 0.09646 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09638 | 0.09650 | 0.09084 | 0.09504 | 0.09596 | 0.09488 | 0.09613 | 0.09575 | 0.10150 | 0.10299 | 0.09943 | 0.09575 | 0.10150 | 0.10299 | 0.09943 | 0.09449 | 0.09557 | 0.09547 | 0.10074 | 0.09627 | 0.09247 | 0.09550 | 0.09678 | 0.09646 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 0.82240 | 0.82304 | 0.82562 | 0.82877 | 0.82864 | 0.82764 | 0.82405 | 0.88006 | 0.84314 | 0.85257 | 0.86222 | 0.88006 | 0.84314 | 0.85257 | 0.86222 | 0.83308 | 0.83876 | 0.83742 | 0.83013 | 0.83540 | 0.83406 | 0.83378 | 0.83999 | 0.84132 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 0.83147 | 0.83226 | 0.83470 | 0.83803 | 0.83785 | 0.83671 | 0.83320 | 0.88967 | 0.85257 | 0.86183 | 0.87166 | 0.88967 | 0.85257 | 0.86183 | 0.87166 | 0.84227 | 0.84816 | 0.84662 | 0.83928 | 0.84465 | 0.84324 | 0.84294 | 0.84929 | 0.85062 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 0.71877 | 0.72226 | 0.72390 | 0.71879 | 0.72071 | 0.72845 | 0.71690 | 0.56245 | 0.52413 | 0.52563 | 0.53092 | 0.56245 | 0.52413 | 0.52563 | 0.53092 | 0.72973 | 0.73630 | 0.74298 | 0.73155 | 0.73269 | 0.74204 | 0.73852 | 0.75635 | 0.76330 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 0.72100 | 0.72461 | 0.72624 | 0.72118 | 0.72305 | 0.73091 | 0.71932 | 0.56420 | 0.52569 | 0.52732 | 0.53261 | 0.56420 | 0.52569 | 0.52732 | 0.53261 | 0.73221 | 0.73867 | 0.74552 | 0.73402 | 0.73502 | 0.74452 | 0.74097 | 0.75878 | 0.76590 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 0.74402 | 0.75393 | 0.74660 | 0.74813 | 0.76088 | 0.75709 | 0.74965 | 0.81406 | 0.75765 | 0.78282 | 0.80396 | 0.81406 | 0.75765 | 0.78282 | 0.80396 | 0.76404 | 0.75788 | 0.76444 | 0.75438 | 0.76130 | 0.76543 | 0.75839 | 0.77505 | 0.77724 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 0.78252 | 0.79278 | 0.78543 | 0.78661 | 0.79982 | 0.79608 | 0.78820 | 0.85553 | 0.79562 | 0.82213 | 0.84453 | 0.85553 | 0.79562 | 0.82213 | 0.84453 | 0.80318 | 0.79673 | 0.80437 | 0.79309 | 0.80030 | 0.80527 | 0.79785 | 0.81567 | 0.81805 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 1.0069 | 1.0179 | 1.0155 | 1.0239 | 1.0202 | 1.0176 | 1.0187 | 1.0190 | 1.0030 | 1.0026 | 1.0048 | 1.0190 | 1.0030 | 1.0026 | 1.0048 | 1.0203 | 1.0200 | 1.0274 | 1.0296 | 1.0120 | 1.0189 | 1.0274 | 1.0275 | 1.0238 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 2.0137 | 2.0357 | 2.0309 | 2.0478 | 2.0404 | 2.0353 | 2.0375 | 2.0381 | 2.0061 | 2.0052 | 2.0095 | 2.0381 | 2.0061 | 2.0052 | 2.0095 | 2.0406 | 2.0399 | 2.0547 | 2.0593 | 2.0239 | 2.0379 | 2.0548 | 2.0549 | 2.0477 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 0.69801 | 0.70558 | 0.70398 | 0.70977 | 0.70720 | 0.70544 | 0.70616 | 0.70634 | 0.69530 | 0.69499 | 0.69655 | 0.70634 | 0.69530 | 0.69499 | 0.69655 | 0.70730 | 0.70707 | 0.71218 | 0.71375 | 0.70154 | 0.70636 | 0.71223 | 0.71225 | 0.70977 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 1.3960 | 1.4112 | 1.4080 | 1.4195 | 1.4144 | 1.4109 | 1.4123 | 1.4127 | 1.3906 | 1.3900 | 1.3931 | 1.4127 | 1.3906 | 1.3900 | 1.3931 | 1.4146 | 1.4141 | 1.4244 | 1.4275 | 1.4031 | 1.4127 | 1.4245 | 1.4245 | 1.4195 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24825 | 0.24718 | 0.24038 | 0.25037 | 0.25845 | 0.24336 | 0.24973 | 0.27949 | 0.31647 | 0.29245 | 0.29734 | 0.27949 | 0.31647 | 0.29245 | 0.29734 | 0.24570 | 0.25228 | 0.24740 | 0.23953 | 0.24102 | 0.23826 | 0.24251 | 0.24697 | 0.25101 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24825 | 0.24718 | 0.24038 | 0.25037 | 0.25845 | 0.24336 | 0.24973 | 0.27949 | 0.31647 | 0.29245 | 0.29734 | 0.27949 | 0.31647 | 0.29245 | 0.29734 | 0.24570 | 0.25228 | 0.24740 | 0.23953 | 0.24102 | 0.23826 | 0.24251 | 0.24697 | 0.25101 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.3870 | -1.3907 | -1.3898 | -1.3907 | -1.3911 | -1.3915 | -1.3896 | -1.3816 | -1.3541 | -1.3649 | -1.3721 | -1.3816 | -1.3541 | -1.3649 | -1.3721 | -1.3917 | -1.3888 | -1.3919 | -1.3942 | -1.3876 | -1.3914 | -1.3927 | -1.3924 | -1.3905 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -1.3922 | -1.4063 | -1.4003 | -1.4066 | -1.4066 | -1.4040 | -1.4072 | -1.4071 | -1.3880 | -1.3862 | -1.3872 | -1.4071 | -1.3880 | -1.3862 | -1.3872 | -1.4078 | -1.4007 | -1.4083 | -1.4172 | -1.3950 | -1.4033 | -1.4091 | -1.4110 | -1.4067 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.25038 | 0.25024 | 0.24999 | 0.25038 | 0.25021 | 0.24961 | 0.25094 | 0.25517 | 0.26822 | 0.26144 | 0.25772 | 0.25517 | 0.26822 | 0.26144 | 0.25772 | 0.24998 | 0.25053 | 0.24992 | 0.25000 | 0.25044 | 0.24958 | 0.24964 | 0.24998 | 0.25044 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.25071 | 0.25038 | 0.24993 | 0.25066 | 0.25049 | 0.24926 | 0.25176 | 0.25962 | 0.28217 | 0.27108 | 0.26446 | 0.25962 | 0.28217 | 0.27108 | 0.26446 | 0.24990 | 0.25099 | 0.24982 | 0.24993 | 0.25080 | 0.24913 | 0.24922 | 0.24979 | 0.25083 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 1.0448 | 1.0478 | 1.0415 | 1.0449 | 1.0279 | 1.0611 | 1.0512 | 1.1195 | 1.0253 | 1.0647 | 1.0766 | 1.1195 | 1.0253 | 1.0647 | 1.0766 | 1.0728 | 1.0593 | 1.0668 | 1.0480 | 1.0499 | 1.0750 | 1.0779 | 1.0853 | 1.0836 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 1.1643 | 1.1675 | 1.1603 | 1.1638 | 1.1415 | 1.1823 | 1.1711 | 1.2442 | 1.1375 | 1.1842 | 1.1949 | 1.2442 | 1.1375 | 1.1842 | 1.1949 | 1.1956 | 1.1790 | 1.1891 | 1.1677 | 1.1686 | 1.1988 | 1.2019 | 1.2090 | 1.2077 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 1.0111 | 1.0191 | 1.0080 | 1.0277 | 1.0098 | 1.0044 | 1.0153 | 0.99369 | 0.99128 | 0.96519 | 0.98552 | 0.99369 | 0.99128 | 0.96519 | 0.98552 | 1.0118 | 1.0228 | 1.0231 | 1.0324 | 1.0117 | 1.0171 | 1.0403 | 1.0226 | 1.0287 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 2.0222 | 2.0383 | 2.0161 | 2.0553 | 2.0197 | 2.0087 | 2.0306 | 1.9874 | 1.9826 | 1.9304 | 1.9710 | 1.9874 | 1.9826 | 1.9304 | 1.9710 | 2.0236 | 2.0456 | 2.0461 | 2.0648 | 2.0234 | 2.0343 | 2.0806 | 2.0451 | 2.0574 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 0.70094 | 0.70646 | 0.69881 | 0.71236 | 0.70005 | 0.69626 | 0.70382 | 0.68873 | 0.68706 | 0.66904 | 0.68314 | 0.68873 | 0.68706 | 0.66904 | 0.68314 | 0.70136 | 0.70897 | 0.70919 | 0.71566 | 0.70135 | 0.70506 | 0.72111 | 0.70881 | 0.71311 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 1.4019 | 1.4129 | 1.3976 | 1.4247 | 1.4001 | 1.3925 | 1.4076 | 1.3775 | 1.3741 | 1.3381 | 1.3663 | 1.3775 | 1.3741 | 1.3381 | 1.3663 | 1.4027 | 1.4179 | 1.4184 | 1.4313 | 1.4027 | 1.4101 | 1.4422 | 1.4176 | 1.4262 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.23812 | 0.24152 | 0.28038 | 0.25170 | 0.26496 | 0.27853 | 0.24830 | 0.31215 | 0.37662 | 0.34762 | 0.34917 | 0.31215 | 0.37662 | 0.34762 | 0.34917 | 0.27051 | 0.25663 | 0.24861 | 0.26126 | 0.26897 | 0.25293 | 0.24213 | 0.26188 | 0.24954 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.23812 | 0.24152 | 0.28038 | 0.25170 | 0.26496 | 0.27853 | 0.24830 | 0.31215 | 0.37662 | 0.34762 | 0.34917 | 0.31215 | 0.37662 | 0.34762 | 0.34917 | 0.27051 | 0.25663 | 0.24861 | 0.26126 | 0.26897 | 0.25293 | 0.24213 | 0.26188 | 0.24954 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.3890 | -1.3911 | -1.3834 | -1.3905 | -1.3846 | -1.3797 | -1.3872 | -1.3599 | -1.3115 | -1.3257 | -1.3393 | -1.3599 | -1.3115 | -1.3257 | -1.3393 | -1.3847 | -1.3899 | -1.3907 | -1.3949 | -1.3855 | -1.3875 | -1.3965 | -1.3876 | -1.3926 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -1.3971 | -1.4085 | -1.3909 | -1.4120 | -1.3910 | -1.3847 | -1.4010 | -1.3726 | -1.3722 | -1.3349 | -1.3610 | -1.3726 | -1.3722 | -1.3349 | -1.3610 | -1.3948 | -1.4038 | -1.4063 | -1.4196 | -1.3938 | -1.4014 | -1.4238 | -1.4032 | -1.4132 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.25000 | 0.25037 | 0.25201 | 0.25123 | 0.25145 | 0.25327 | 0.25139 | 0.26187 | 0.29071 | 0.27603 | 0.27227 | 0.26187 | 0.29071 | 0.27603 | 0.27227 | 0.25186 | 0.25045 | 0.25021 | 0.25003 | 0.25136 | 0.25122 | 0.24960 | 0.25148 | 0.25023 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.24997 | 0.25061 | 0.25388 | 0.25226 | 0.25286 | 0.25673 | 0.25258 | 0.27234 | 0.32001 | 0.29797 | 0.29051 | 0.27234 | 0.32001 | 0.29797 | 0.29051 | 0.25368 | 0.25097 | 0.25027 | 0.25021 | 0.25263 | 0.25227 | 0.24916 | 0.25287 | 0.25046 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 0.88672 | 0.89365 | 0.89683 | 0.89227 | 0.89550 | 0.90256 | 0.88939 | 0.94024 | 0.87903 | 0.90194 | 0.92370 | 0.94024 | 0.87903 | 0.90194 | 0.92370 | 0.90910 | 0.90388 | 0.91268 | 0.89667 | 0.90268 | 0.91317 | 0.91147 | 0.92974 | 0.91762 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 0.94686 | 0.95405 | 0.95827 | 0.95294 | 0.95563 | 0.96411 | 0.95015 | 1.0029 | 0.93871 | 0.96311 | 0.98695 | 1.0029 | 0.93871 | 0.96311 | 0.98695 | 0.97169 | 0.96568 | 0.97483 | 0.95838 | 0.96502 | 0.97632 | 0.97381 | 0.99441 | 0.98059 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 1.0071 | 1.0288 | 1.0107 | 1.0402 | 1.0244 | 1.0147 | 1.0276 | 1.0209 | 1.0296 | 0.98518 | 0.99769 | 1.0209 | 1.0296 | 0.98518 | 0.99769 | 1.0281 | 1.0207 | 1.0302 | 1.0249 | 1.0346 | 1.0187 | 1.0267 | 1.0391 | 1.0286 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 2.0141 | 2.0576 | 2.0215 | 2.0804 | 2.0488 | 2.0293 | 2.0552 | 2.0418 | 2.0593 | 1.9704 | 1.9954 | 2.0418 | 2.0593 | 1.9704 | 1.9954 | 2.0563 | 2.0415 | 2.0604 | 2.0498 | 2.0691 | 2.0373 | 2.0535 | 2.0783 | 2.0573 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 0.69818 | 0.71321 | 0.70071 | 0.72107 | 0.71016 | 0.70340 | 0.71233 | 0.70765 | 0.71368 | 0.68286 | 0.69157 | 0.70765 | 0.71368 | 0.68286 | 0.69157 | 0.71270 | 0.70756 | 0.71415 | 0.71047 | 0.71718 | 0.70614 | 0.71173 | 0.72030 | 0.71308 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 1.3964 | 1.4264 | 1.4014 | 1.4421 | 1.4203 | 1.4068 | 1.4247 | 1.4153 | 1.4274 | 1.3657 | 1.3831 | 1.4153 | 1.4274 | 1.3657 | 1.3831 | 1.4254 | 1.4151 | 1.4283 | 1.4209 | 1.4344 | 1.4123 | 1.4235 | 1.4406 | 1.4262 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24829 | 0.21937 | 0.24992 | 0.23432 | 0.25349 | 0.24634 | 0.23009 | 0.30907 | 0.35132 | 0.35197 | 0.33117 | 0.30907 | 0.35132 | 0.35197 | 0.33117 | 0.23074 | 0.25219 | 0.22879 | 0.25902 | 0.22944 | 0.25674 | 0.24862 | 0.22099 | 0.23464 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24829 | 0.21937 | 0.24992 | 0.23432 | 0.25349 | 0.24634 | 0.23009 | 0.30907 | 0.35132 | 0.35197 | 0.33117 | 0.30907 | 0.35132 | 0.35197 | 0.33117 | 0.23074 | 0.25219 | 0.22879 | 0.25902 | 0.22944 | 0.25674 | 0.24862 | 0.22099 | 0.23464 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.3873 | -1.3990 | -1.3864 | -1.4022 | -1.3943 | -1.3889 | -1.3969 | -1.3753 | -1.3393 | -1.3378 | -1.3526 | -1.3753 | -1.3393 | -1.3378 | -1.3526 | -1.3961 | -1.3877 | -1.3955 | -1.3889 | -1.4013 | -1.3896 | -1.3918 | -1.4001 | -1.3927 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -1.3923 | -1.4214 | -1.3935 | -1.4298 | -1.4109 | -1.3982 | -1.4176 | -1.4108 | -1.4253 | -1.3623 | -1.3773 | -1.4108 | -1.4253 | -1.3623 | -1.3773 | -1.4177 | -1.4001 | -1.4156 | -1.4090 | -1.4263 | -1.4024 | -1.4073 | -1.4254 | -1.4122 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.25026 | 0.24804 | 0.25081 | 0.24768 | 0.24922 | 0.25020 | 0.24874 | 0.25966 | 0.28283 | 0.27320 | 0.26743 | 0.25966 | 0.28283 | 0.27320 | 0.26743 | 0.24895 | 0.25103 | 0.24900 | 0.25153 | 0.24767 | 0.25034 | 0.24991 | 0.24792 | 0.25001 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.25052 | 0.24611 | 0.25155 | 0.24564 | 0.24869 | 0.25042 | 0.24761 | 0.26869 | 0.30623 | 0.29280 | 0.28209 | 0.26869 | 0.30623 | 0.29280 | 0.28209 | 0.24781 | 0.25200 | 0.24798 | 0.25286 | 0.24557 | 0.25063 | 0.24983 | 0.24581 | 0.24996 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 1.3513 | 1.3542 | 1.3457 | 1.3375 | 1.3326 | 1.3689 | 1.3466 | 1.2760 | 1.1917 | 1.2220 | 1.2312 | 1.2760 | 1.1917 | 1.2220 | 1.2312 | 1.3750 | 1.3619 | 1.3731 | 1.3559 | 1.3488 | 1.3699 | 1.3638 | 1.3982 | 1.3890 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 1.6862 | 1.6935 | 1.6762 | 1.6735 | 1.6574 | 1.7090 | 1.6824 | 1.5731 | 1.4715 | 1.5064 | 1.5116 | 1.5731 | 1.4715 | 1.5064 | 1.5116 | 1.7154 | 1.6972 | 1.7120 | 1.6931 | 1.6789 | 1.7104 | 1.6987 | 1.7473 | 1.7327 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 1.0117 | 1.0333 | 1.0145 | 1.0357 | 1.0182 | 1.0155 | 1.0244 | 1.0242 | 0.99968 | 1.0083 | 1.0006 | 1.0242 | 0.99968 | 1.0083 | 1.0006 | 1.0162 | 1.0097 | 1.0261 | 1.0382 | 1.0142 | 1.0179 | 1.0493 | 1.0382 | 1.0446 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 2.0233 | 2.0666 | 2.0290 | 2.0713 | 2.0364 | 2.0310 | 2.0489 | 2.0485 | 1.9994 | 2.0165 | 2.0011 | 2.0485 | 1.9994 | 2.0165 | 2.0011 | 2.0325 | 2.0195 | 2.0522 | 2.0763 | 2.0283 | 2.0357 | 2.0986 | 2.0763 | 2.0893 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 0.70135 | 0.71630 | 0.70331 | 0.71794 | 0.70580 | 0.70400 | 0.71013 | 0.70996 | 0.69292 | 0.69893 | 0.69353 | 0.70996 | 0.69292 | 0.69893 | 0.69353 | 0.70451 | 0.69997 | 0.71130 | 0.71965 | 0.70305 | 0.70561 | 0.72737 | 0.71964 | 0.72408 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 1.4027 | 1.4326 | 1.4066 | 1.4359 | 1.4116 | 1.4080 | 1.4203 | 1.4199 | 1.3858 | 1.3979 | 1.3871 | 1.4199 | 1.3858 | 1.3979 | 1.3871 | 1.4090 | 1.3999 | 1.4226 | 1.4393 | 1.4061 | 1.4112 | 1.4547 | 1.4393 | 1.4482 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24486 | 0.22797 | 0.25911 | 0.24785 | 0.25646 | 0.25911 | 0.25017 | 0.27005 | 0.32074 | 0.30616 | 0.30020 | 0.27005 | 0.32074 | 0.30616 | 0.30020 | 0.26276 | 0.27932 | 0.24387 | 0.24354 | 0.25746 | 0.25414 | 0.24122 | 0.22962 | 0.24420 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24486 | 0.22797 | 0.25911 | 0.24785 | 0.25646 | 0.25911 | 0.25017 | 0.27005 | 0.32074 | 0.30616 | 0.30020 | 0.27005 | 0.32074 | 0.30616 | 0.30020 | 0.26276 | 0.27932 | 0.24387 | 0.24354 | 0.25746 | 0.25414 | 0.24122 | 0.22962 | 0.24420 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.3890 | -1.3989 | -1.3876 | -1.3969 | -1.3899 | -1.3875 | -1.3922 | -1.3871 | -1.3620 | -1.3723 | -1.3712 | -1.3871 | -1.3620 | -1.3723 | -1.3712 | -1.3874 | -1.3804 | -1.3923 | -1.3960 | -1.3871 | -1.3867 | -1.4000 | -1.3978 | -1.3985 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -1.3977 | -1.4280 | -1.3996 | -1.4239 | -1.4025 | -1.3983 | -1.4122 | -1.4141 | -1.3830 | -1.3928 | -1.3800 | -1.4141 | -1.3830 | -1.3928 | -1.3800 | -1.4003 | -1.3852 | -1.4095 | -1.4264 | -1.3954 | -1.3999 | -1.4307 | -1.4231 | -1.4321 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.25014 | 0.24895 | 0.25102 | 0.24966 | 0.25026 | 0.25092 | 0.25037 | 0.25333 | 0.26261 | 0.25829 | 0.25701 | 0.25333 | 0.26261 | 0.25829 | 0.25701 | 0.25113 | 0.25295 | 0.24993 | 0.25005 | 0.25075 | 0.25144 | 0.24886 | 0.24887 | 0.24969 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.25034 | 0.24777 | 0.25194 | 0.24946 | 0.25053 | 0.25186 | 0.25070 | 0.25651 | 0.27395 | 0.26628 | 0.26379 | 0.25651 | 0.27395 | 0.26628 | 0.26379 | 0.25211 | 0.25604 | 0.24987 | 0.24990 | 0.25144 | 0.25265 | 0.24784 | 0.24772 | 0.24937 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 0.46577 | 0.44020 | 0.44459 | 0.45333 | 0.46248 | 0.44774 | 0.44334 | 0.43435 | 0.41970 | 0.41635 | 0.43091 | 0.43435 | 0.41970 | 0.41635 | 0.43091 | 0.47003 | 0.46597 | 0.46320 | 0.44671 | 0.46630 | 0.45690 | 0.46439 | 0.46909 | 0.47172 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 0.46841 | 0.44272 | 0.44712 | 0.45579 | 0.46521 | 0.45027 | 0.44579 | 0.43684 | 0.42216 | 0.41884 | 0.43342 | 0.43684 | 0.42216 | 0.41884 | 0.43342 | 0.47267 | 0.46869 | 0.46598 | 0.44931 | 0.46901 | 0.45943 | 0.46703 | 0.47182 | 0.47438 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 0.36770 | 0.34076 | 0.34407 | 0.35400 | 0.36038 | 0.34226 | 0.34127 | 0.32629 | 0.31954 | 0.32661 | 0.32116 | 0.32629 | 0.31954 | 0.32661 | 0.32116 | 0.35358 | 0.35189 | 0.35860 | 0.34479 | 0.35979 | 0.35001 | 0.34904 | 0.36836 | 0.34915 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 0.36917 | 0.34212 | 0.34548 | 0.35534 | 0.36180 | 0.34357 | 0.34264 | 0.32758 | 0.32082 | 0.32792 | 0.32241 | 0.32758 | 0.32082 | 0.32792 | 0.32241 | 0.35501 | 0.35321 | 0.35996 | 0.34608 | 0.36120 | 0.35145 | 0.35039 | 0.36978 | 0.35056 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 0.60376 | 0.60406 | 0.56732 | 0.60971 | 0.61614 | 0.63320 | 0.58583 | 0.60544 | 0.56326 | 0.59465 | 0.59852 | 0.60544 | 0.56326 | 0.59465 | 0.59852 | 0.61809 | 0.63716 | 0.61251 | 0.58228 | 0.59858 | 0.59959 | 0.59265 | 0.60957 | 0.60786 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 0.60782 | 0.60824 | 0.57127 | 0.61391 | 0.62043 | 0.63771 | 0.58983 | 0.60976 | 0.56714 | 0.59892 | 0.60280 | 0.60976 | 0.56714 | 0.59892 | 0.60280 | 0.62250 | 0.64178 | 0.61670 | 0.58641 | 0.60271 | 0.60378 | 0.59666 | 0.61401 | 0.61205 |
| eval/lm/c4_en-validation/CE loss | lower | 3.0335 | 3.0453 | 3.0380 | 3.0336 | 3.0473 | 3.0496 | 3.0311 | 3.3123 | 3.1674 | 3.2063 | 3.2482 | 3.3123 | 3.1674 | 3.2063 | 3.2482 | 3.0671 | 3.0764 | 3.0751 | 3.0533 | 3.0643 | 3.0752 | 3.0602 | 3.0969 | 3.1054 |
| eval/lm/c4_en-validation/PPL | lower | 20.77 | 21.02 | 20.86 | 20.77 | 21.06 | 21.11 | 20.72 | 27.45 | 23.75 | 24.69 | 25.74 | 27.45 | 23.75 | 24.69 | 25.74 | 21.48 | 21.68 | 21.65 | 21.18 | 21.42 | 21.65 | 21.33 | 22.13 | 22.32 |
| eval/lm/dolma_books-validation/CE loss | lower | 2.9392 | 2.9599 | 2.9502 | 2.9317 | 2.9528 | 2.9567 | 2.9304 | 3.3054 | 3.1079 | 3.1654 | 3.2248 | 3.3054 | 3.1079 | 3.1654 | 3.2248 | 2.9796 | 2.9974 | 2.9831 | 2.9612 | 2.9750 | 2.9872 | 2.9660 | 3.0143 | 3.0256 |
| eval/lm/dolma_books-validation/PPL | lower | 18.90 | 19.30 | 19.11 | 18.76 | 19.16 | 19.23 | 18.74 | 27.26 | 22.37 | 23.70 | 25.15 | 27.26 | 22.37 | 23.70 | 25.15 | 19.68 | 20.03 | 19.75 | 19.32 | 19.59 | 19.83 | 19.41 | 20.38 | 20.61 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 3.1706 | 3.1827 | 3.1755 | 3.1679 | 3.1816 | 3.1833 | 3.1650 | 3.4499 | 3.3008 | 3.3388 | 3.3836 | 3.4499 | 3.3008 | 3.3388 | 3.3836 | 3.2050 | 3.2146 | 3.2116 | 3.1881 | 3.2012 | 3.2119 | 3.1949 | 3.2327 | 3.2414 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 23.82 | 24.11 | 23.94 | 23.76 | 24.09 | 24.13 | 23.69 | 31.50 | 27.13 | 28.19 | 29.48 | 31.50 | 27.13 | 28.19 | 29.48 | 24.66 | 24.89 | 24.82 | 24.24 | 24.56 | 24.83 | 24.41 | 25.35 | 25.57 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 2.2203 | 2.2266 | 2.2224 | 2.2193 | 2.2336 | 2.2335 | 2.2175 | 2.3977 | 2.2791 | 2.3093 | 2.3463 | 2.3977 | 2.2791 | 2.3093 | 2.3463 | 2.2482 | 2.2557 | 2.2553 | 2.2301 | 2.2429 | 2.2493 | 2.2357 | 2.2709 | 2.2776 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 9.2099 | 9.2686 | 9.2298 | 9.2010 | 9.3332 | 9.3324 | 9.1848 | 11.00 | 9.7681 | 10.07 | 10.45 | 11.00 | 9.7681 | 10.07 | 10.45 | 9.4706 | 9.5424 | 9.5383 | 9.3009 | 9.4206 | 9.4814 | 9.3528 | 9.6881 | 9.7534 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 3.3397 | 3.3480 | 3.3455 | 3.3400 | 3.3488 | 3.3540 | 3.3390 | 3.5751 | 3.4465 | 3.4852 | 3.5175 | 3.5751 | 3.4465 | 3.4852 | 3.5175 | 3.3683 | 3.3778 | 3.3746 | 3.3582 | 3.3663 | 3.3791 | 3.3641 | 3.3945 | 3.4027 |
| eval/lm/dolma_reddit-validation/PPL | lower | 28.21 | 28.44 | 28.37 | 28.22 | 28.47 | 28.62 | 28.19 | 35.70 | 31.39 | 32.63 | 33.70 | 35.70 | 31.39 | 32.63 | 33.70 | 29.03 | 29.31 | 29.21 | 28.74 | 28.97 | 29.34 | 28.91 | 29.80 | 30.05 |
| eval/lm/dolma_stack-validation/CE loss | lower | 1.3878 | 1.3980 | 1.3877 | 1.3857 | 1.4006 | 1.3999 | 1.3827 | 1.5425 | 1.4386 | 1.4636 | 1.4942 | 1.5425 | 1.4386 | 1.4636 | 1.4942 | 1.4182 | 1.4298 | 1.4248 | 1.4037 | 1.4138 | 1.4219 | 1.4073 | 1.4392 | 1.4484 |
| eval/lm/dolma_stack-validation/PPL | lower | 4.0059 | 4.0470 | 4.0056 | 3.9976 | 4.0575 | 4.0547 | 3.9856 | 4.6764 | 4.2148 | 4.3215 | 4.4557 | 4.6764 | 4.2148 | 4.3215 | 4.4557 | 4.1297 | 4.1778 | 4.1572 | 4.0702 | 4.1115 | 4.1449 | 4.0848 | 4.2172 | 4.2562 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 2.6827 | 2.6997 | 2.6867 | 2.6816 | 2.6979 | 2.6954 | 2.6774 | 2.8251 | 2.6669 | 2.7044 | 2.7577 | 2.8251 | 2.6669 | 2.7044 | 2.7577 | 2.7161 | 2.7307 | 2.7248 | 2.7071 | 2.7234 | 2.7285 | 2.7121 | 2.7503 | 2.7588 |
| eval/lm/dolma_wiki-validation/PPL | lower | 14.63 | 14.88 | 14.68 | 14.61 | 14.85 | 14.81 | 14.55 | 16.86 | 14.39 | 14.94 | 15.76 | 16.86 | 14.39 | 14.94 | 15.76 | 15.12 | 15.34 | 15.25 | 14.99 | 15.23 | 15.31 | 15.06 | 15.65 | 15.78 |
| eval/lm/ice-validation/CE loss | lower | 3.1207 | 3.1265 | 3.1326 | 3.1031 | 3.1293 | 3.1234 | 3.1104 | 3.2843 | 3.1703 | 3.1970 | 3.2510 | 3.2843 | 3.1703 | 3.1970 | 3.2510 | 3.1469 | 3.1483 | 3.1542 | 3.1383 | 3.1358 | 3.1484 | 3.1374 | 3.1871 | 3.2018 |
| eval/lm/ice-validation/PPL | lower | 22.66 | 22.79 | 22.93 | 22.27 | 22.86 | 22.72 | 22.43 | 26.69 | 23.81 | 24.46 | 25.82 | 26.69 | 23.81 | 24.46 | 25.82 | 23.26 | 23.30 | 23.43 | 23.07 | 23.01 | 23.30 | 23.04 | 24.22 | 24.58 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 3.1410 | 3.1578 | 3.1484 | 3.1404 | 3.1561 | 3.1573 | 3.1405 | 3.1810 | 3.0913 | 3.1138 | 3.1538 | 3.1810 | 3.0913 | 3.1138 | 3.1538 | 3.1805 | 3.1857 | 3.1841 | 3.1686 | 3.1865 | 3.1879 | 3.1765 | 3.2063 | 3.1946 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 23.13 | 23.52 | 23.30 | 23.11 | 23.48 | 23.51 | 23.12 | 24.07 | 22.01 | 22.51 | 23.42 | 24.07 | 22.01 | 22.51 | 23.42 | 24.06 | 24.19 | 24.15 | 23.77 | 24.20 | 24.24 | 23.96 | 24.69 | 24.40 |
| eval/lm/pile-validation/CE loss | lower | 2.3005 | 2.3108 | 2.3038 | 2.2976 | 2.3132 | 2.3141 | 2.2959 | 2.5341 | 2.4003 | 2.4338 | 2.4773 | 2.5341 | 2.4003 | 2.4338 | 2.4773 | 2.3325 | 2.3433 | 2.3391 | 2.3176 | 2.3304 | 2.3391 | 2.3208 | 2.3606 | 2.3700 |
| eval/lm/pile-validation/PPL | lower | 9.9795 | 10.08 | 10.01 | 9.9503 | 10.11 | 10.12 | 9.9329 | 12.60 | 11.03 | 11.40 | 11.91 | 12.60 | 11.03 | 11.40 | 11.91 | 10.30 | 10.42 | 10.37 | 10.15 | 10.28 | 10.37 | 10.18 | 10.60 | 10.70 |
| eval/lm/wikitext_103-validation/CE loss | lower | 2.6529 | 2.6525 | 2.6559 | 2.6494 | 2.6742 | 2.6614 | 2.6512 | 2.8426 | 2.6786 | 2.7229 | 2.7714 | 2.8426 | 2.6786 | 2.7229 | 2.7714 | 2.6975 | 2.7018 | 2.7081 | 2.6698 | 2.6874 | 2.6977 | 2.6781 | 2.7298 | 2.7396 |
| eval/lm/wikitext_103-validation/PPL | lower | 14.19 | 14.19 | 14.24 | 14.14 | 14.50 | 14.32 | 14.17 | 17.16 | 14.57 | 15.22 | 15.98 | 17.16 | 14.57 | 15.22 | 15.98 | 14.84 | 14.91 | 15.00 | 14.44 | 14.69 | 14.85 | 14.56 | 15.33 | 15.48 |
| throughput/in-loop eval batches | see metric | 828.0 | 828.0 | 828.0 | 828.0 | 828.0 | 828.0 | 828.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 3281.0 | 826.0 | 826.0 | 826.0 | 826.0 | 826.0 | 826.0 | 826.0 | 826.0 | 826.0 |
| throughput/in-loop eval time (s) | see metric | 68.19 | 65.25 | 66.94 | 63.54 | 64.15 | 69.66 | 61.58 | 406.9 | 411.3 | 387.6 | 413.5 | 406.9 | 411.3 | 387.6 | 413.5 | 116.9 | 119.6 | 121.5 | 129.9 | 131.7 | 127.7 | 150.1 | 118.5 | 119.3 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| int-275m-cx8-intd256e8k-lr1.6e-3-r1<br>`825sabjc` | finished | original | 31728599040.0 | 40345 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/825sabjc) |
| int-275m-cx8-intd256e8k-lr3.2e-3-r1<br>`05tuklbj` | finished | original | 31728599040.0 | 40345 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/05tuklbj) |
| int-275m-cx8-intd256e8k-lr8e-4-r1<br>`tkon9v64` | finished | original | 31728599040.0 | 40345 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/tkon9v64) |
| int-275m-cx8-intw256e8k-lr1.6e-3-r1<br>`qu2zaxr7` | finished | original | 32502448128.0 | 41329 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qu2zaxr7) |
| int-275m-cx8-intw256e8k-lr3.2e-3-r1<br>`235ye5lg` | finished | original | 32502448128.0 | 41329 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/235ye5lg) |
| int-275m-cx8-intw256e8k-lr4e-4-r1<br>`iv901lom` | finished | original | 32502448128.0 | 41329 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/iv901lom) |
| int-275m-cx8-intw256e8k-lr8e-4-r1<br>`qe052lo4` | finished | original | 32502448128.0 | 41329 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/qe052lo4) |
| mt-275m-baseline-cx8-lr1.6e-3-r1<br>`edljug2e` | finished | original | 100000595968.0 | 95368 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/edljug2e) |
| mt-275m-baseline-cx8-lr2e-4-r1<br>`4amcsbx6` | finished | original | 100000595968.0 | 95368 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/4amcsbx6) |
| mt-275m-baseline-cx8-lr4e-4-r1<br>`8jdqtmgg` | finished | original | 100000595968.0 | 95368 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/8jdqtmgg) |
| mt-275m-baseline-cx8-lr8e-4-r1<br>`drm1ceit` | finished | original | 100000595968.0 | 95368 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/drm1ceit) |
| mt-eval-275m-baseline-cx8-lr1.6e-3-r1<br>`y0s527t1` | finished | original | 100000595968.0 | 95368 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/y0s527t1) |
| mt-eval-275m-baseline-cx8-lr2e-4-r1<br>`zh77avw2` | finished | original | 100000595968.0 | 95368 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/zh77avw2) |
| mt-eval-275m-baseline-cx8-lr4e-4-r1<br>`cnav18vq` | finished | original | 100000595968.0 | 95368 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/cnav18vq) |
| mt-eval-275m-baseline-cx8-lr8e-4-r1<br>`g4j4q5z2` | finished | original | 100000595968.0 | 95368 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/g4j4q5z2) |
| q3-275m-cx8-q3am128e8k-lr1.6e-3-r1<br>`ozai8yzb` | finished | original | 32345161728.0 | 41129 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ozai8yzb) |
| q3-275m-cx8-q3am128e8k-lr3.2e-3-r1<br>`z3g9yfct` | finished | original | 32345161728.0 | 41129 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/z3g9yfct) |
| q3-275m-cx8-q3am128e8k-lr8e-4-r1<br>`ap78ohx0` | finished | original | 32345161728.0 | 41129 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/ap78ohx0) |
| q3-275m-cx8-q3td128e8k-lr1.6e-3-r1<br>`lwdmnwcj` | finished | original | 32221691904.0 | 40972 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/lwdmnwcj) |
| q3-275m-cx8-q3td128e8k-lr3.2e-3-r1<br>`uhxom59e` | finished | original | 32221691904.0 | 40972 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/uhxom59e) |
| q3-275m-cx8-q3td128e8k-lr4e-4-r1<br>`f89akz4d` | finished | original | 32221691904.0 | 40972 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/f89akz4d) |
| q3-275m-cx8-q3td128e8k-lr8e-4-r1<br>`rdzgpg7g` | finished | original | 32221691904.0 | 40972 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rdzgpg7g) |
| se-275m-cx8-se0m9-lr1.6e-3-r2<br>`1m1rgjkc` | finished | original | 32220905472.0 | 40971 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/1m1rgjkc) |
| se-275m-cx8-se0m9-lr3.2e-3-r2<br>`vkhbbadr` | finished | original | 32220905472.0 | 40971 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/vkhbbadr) |

## unknown Cx?

| metric | direction | int-smoke-intd256e8k-lr1.6e-3-r1<br>`sj1r51mi` | int-smoke-intw256e8k-lr1.6e-3-r1<br>`rfox2uad` | q3-smoke-q3am128e8k-lr2e-3-r1<br>`12jqamu0` | q3-smoke-q3td128e8k-lr2e-3-r1<br>`99pn8kaz` |
| --- | --- | --- | --- | --- | --- |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB v2) | lower | 1.9984 | 1.9654 | 1.9434 | 1.9406 |
| eval/downstream/arc_challenge_test_bpb_5shot (BPB) | lower | 2.1925 | 2.1502 | 2.1326 | 2.1239 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB v2) | lower | 4.0719 | 3.8771 | 3.7962 | 3.7735 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (BPB) | lower | 8.1437 | 7.7542 | 7.5923 | 7.5470 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss v2) | lower | 2.8224 | 2.6875 | 2.6318 | 2.6161 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (CE loss) | lower | 5.6448 | 5.3751 | 5.2635 | 5.2323 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2) | higher | 0.22696 | 0.22696 | 0.22696 | 0.22696 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy) | higher | 0.22696 | 0.22696 | 0.22696 | 0.22696 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss v2) | lower | -2.3865 | -2.5170 | -2.2609 | -2.2228 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (log soft loss) | lower | -2.3865 | -2.5170 | -2.2609 | -2.2228 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss v2) | lower | 0.23162 | 0.23048 | 0.23264 | 0.23345 |
| eval/downstream/arc_challenge_test_mc_5shot_fast (soft loss) | lower | 0.23162 | 0.23048 | 0.23264 | 0.23345 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB v2) | lower | 2.0071 | 1.9412 | 1.9022 | 1.9245 |
| eval/downstream/arc_easy_test_bpb_5shot (BPB) | lower | 2.2008 | 2.1265 | 2.0841 | 2.1064 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB v2) | lower | 3.9149 | 3.6555 | 3.6160 | 3.6258 |
| eval/downstream/arc_easy_test_mc_5shot_fast (BPB) | lower | 7.8298 | 7.3110 | 7.2320 | 7.2515 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss v2) | lower | 2.7141 | 2.5342 | 2.5070 | 2.5137 |
| eval/downstream/arc_easy_test_mc_5shot_fast (CE loss) | lower | 5.4282 | 5.0683 | 5.0139 | 5.0275 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2) | higher | 0.25084 | 0.25084 | 0.25084 | 0.25084 |
| eval/downstream/arc_easy_test_mc_5shot_fast (accuracy) | higher | 0.25084 | 0.25084 | 0.25084 | 0.25084 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss v2) | lower | -2.2933 | -2.4662 | -2.1219 | -2.1529 |
| eval/downstream/arc_easy_test_mc_5shot_fast (log soft loss) | lower | -2.2933 | -2.4662 | -2.1219 | -2.1529 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss v2) | lower | 0.25076 | 0.25089 | 0.25095 | 0.25080 |
| eval/downstream/arc_easy_test_mc_5shot_fast (soft loss) | lower | 0.25076 | 0.25089 | 0.25095 | 0.25080 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2) | lower | 2.6548 | 2.6767 | 2.8640 | 2.7371 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (BPB) | lower | 4.2061 | 4.2201 | 4.4987 | 4.2852 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss v2) | lower | 1.8401 | 1.8552 | 1.9851 | 1.8972 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (CE loss) | lower | 2.9152 | 2.9252 | 3.1185 | 2.9704 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy v2) | higher | 0.06304 | 0.03152 | 0.05158 | 0.05062 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (accuracy) | higher | 0.06304 | 0.03152 | 0.05158 | 0.05062 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss v2) | lower | -3.0867 | -3.1163 | -3.2158 | -3.1940 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (log soft loss) | lower | -3.0867 | -3.1163 | -3.2158 | -3.1940 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss v2) | lower | 0.08735 | 0.08363 | 0.08247 | 0.08267 |
| eval/downstream/basic_skills_arithmetic_rc_5shot (soft loss) | lower | 0.08735 | 0.08363 | 0.08247 | 0.08267 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB v2) | lower | 3.0685 | 3.0297 | 3.0357 | 3.0528 |
| eval/downstream/basic_skills_coding_rc_5shot (BPB) | lower | 3.3893 | 3.3421 | 3.3530 | 3.3671 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss v2) | lower | 2.1267 | 2.0998 | 2.1041 | 2.1160 |
| eval/downstream/basic_skills_coding_rc_5shot (CE loss) | lower | 2.3493 | 2.3165 | 2.3242 | 2.3338 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy v2) | higher | 0.11067 | 0.11462 | 0.11957 | 0.09684 |
| eval/downstream/basic_skills_coding_rc_5shot (accuracy) | higher | 0.11067 | 0.11462 | 0.11957 | 0.09684 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss v2) | lower | -18.92 | -18.89 | -18.78 | -19.36 |
| eval/downstream/basic_skills_coding_rc_5shot (log soft loss) | lower | -18.92 | -18.89 | -18.78 | -19.36 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss v2) | lower | 0.12014 | 0.11922 | 0.12137 | 0.10741 |
| eval/downstream/basic_skills_coding_rc_5shot (soft loss) | lower | 0.12014 | 0.11922 | 0.12137 | 0.10741 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2) | lower | 2.1309 | 2.1240 | 2.1588 | 2.1884 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB) | lower | 2.5499 | 2.5395 | 2.5823 | 2.6210 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss v2) | lower | 1.4778 | 1.4730 | 1.4970 | 1.5176 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (CE loss) | lower | 1.7685 | 1.7610 | 1.7908 | 1.8177 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy v2) | higher | 0.27001 | 0.25747 | 0.28737 | 0.26905 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (accuracy) | higher | 0.27001 | 0.25747 | 0.28737 | 0.26905 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss v2) | lower | -2.3225 | -2.2845 | -2.2288 | -2.2965 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (log soft loss) | lower | -2.3225 | -2.2845 | -2.2288 | -2.2965 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss v2) | lower | 0.25812 | 0.26074 | 0.26287 | 0.26231 |
| eval/downstream/basic_skills_common_knowledge_rc_5shot (soft loss) | lower | 0.25812 | 0.26074 | 0.26287 | 0.26231 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2) | lower | 1.7047 | 1.6742 | 1.6440 | 1.7011 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB) | lower | 1.7612 | 1.7295 | 1.6980 | 1.7573 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss v2) | lower | 1.1817 | 1.1606 | 1.1395 | 1.1792 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (CE loss) | lower | 1.2208 | 1.1989 | 1.1771 | 1.2181 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy v2) | higher | 0.31038 | 0.33274 | 0.30948 | 0.32200 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (accuracy) | higher | 0.31038 | 0.33274 | 0.30948 | 0.32200 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss v2) | lower | -4.5710 | -4.3833 | -4.2859 | -4.7579 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (log soft loss) | lower | -4.5710 | -4.3833 | -4.2859 | -4.7579 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss v2) | lower | 0.29828 | 0.31596 | 0.30402 | 0.30196 |
| eval/downstream/basic_skills_logical_reasoning_rc_5shot (soft loss) | lower | 0.29828 | 0.31596 | 0.30402 | 0.30196 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB v2) | lower | 3.3611 | 3.3838 | 3.2631 | 3.4044 |
| eval/downstream/basic_skills_pattern_rc_5shot (BPB) | lower | 5.2137 | 5.2714 | 5.0417 | 5.3345 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss v2) | lower | 2.6332 | 2.6454 | 2.5703 | 2.6707 |
| eval/downstream/basic_skills_pattern_rc_5shot (CE loss) | lower | 4.3940 | 4.4235 | 4.2865 | 4.4956 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy v2) | higher | 0.29401 | 0.29213 | 0.28464 | 0.28090 |
| eval/downstream/basic_skills_pattern_rc_5shot (accuracy) | higher | 0.29401 | 0.29213 | 0.28464 | 0.28090 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss v2) | lower | -2.1052 | -2.1243 | -2.1285 | -2.1799 |
| eval/downstream/basic_skills_pattern_rc_5shot (log soft loss) | lower | -2.1052 | -2.1243 | -2.1285 | -2.1799 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss v2) | lower | 0.27722 | 0.27041 | 0.27590 | 0.26997 |
| eval/downstream/basic_skills_pattern_rc_5shot (soft loss) | lower | 0.27722 | 0.27041 | 0.27590 | 0.26997 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2) | lower | 3.9699 | 3.9216 | 3.9148 | 3.9152 |
| eval/downstream/basic_skills_string_operations_rc_5shot (BPB) | lower | 5.2725 | 5.2114 | 5.1807 | 5.1698 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss v2) | lower | 2.7516 | 2.7183 | 2.7134 | 2.7137 |
| eval/downstream/basic_skills_string_operations_rc_5shot (CE loss) | lower | 3.6546 | 3.6120 | 3.5909 | 3.5832 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy v2) | higher | 0.20919 | 0.21083 | 0.20919 | 0.21411 |
| eval/downstream/basic_skills_string_operations_rc_5shot (accuracy) | higher | 0.20919 | 0.21083 | 0.20919 | 0.21411 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss v2) | lower | -6.7405 | -6.5875 | -6.6045 | -6.7282 |
| eval/downstream/basic_skills_string_operations_rc_5shot (log soft loss) | lower | -6.7405 | -6.5875 | -6.6045 | -6.7282 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss v2) | lower | 0.21748 | 0.21834 | 0.21831 | 0.21906 |
| eval/downstream/basic_skills_string_operations_rc_5shot (soft loss) | lower | 0.21748 | 0.21834 | 0.21831 | 0.21906 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2) | lower | 1.9858 | 1.9491 | 1.9534 | 2.0216 |
| eval/downstream/codex_humaneval_gold_bpb_3shot (BPB) | lower | 2.0085 | 1.9714 | 1.9753 | 2.0449 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2) | lower | 2.4404 | 2.4512 | 2.4012 | 2.4756 |
| eval/downstream/codex_mbpp_gold_bpb_3shot (BPB) | lower | 2.4603 | 2.4716 | 2.4209 | 2.4962 |
| eval/downstream/copycolors_10way_fast (BPB v2) | lower | 4.2341 | 4.0373 | 3.7898 | 3.8808 |
| eval/downstream/copycolors_10way_fast (BPB) | lower | 8.4681 | 8.0747 | 7.5797 | 7.7616 |
| eval/downstream/copycolors_10way_fast (CE loss v2) | lower | 2.9348 | 2.7985 | 2.6277 | 2.6905 |
| eval/downstream/copycolors_10way_fast (CE loss) | lower | 5.8697 | 5.5970 | 5.2555 | 5.3811 |
| eval/downstream/copycolors_10way_fast (accuracy v2) | higher | 0.10000 | 0.09000 | 0.10000 | 0.10000 |
| eval/downstream/copycolors_10way_fast (accuracy) | higher | 0.10000 | 0.09000 | 0.10000 | 0.10000 |
| eval/downstream/copycolors_10way_fast (log soft loss v2) | lower | -2.6413 | -3.2309 | -2.7547 | -2.8708 |
| eval/downstream/copycolors_10way_fast (log soft loss) | lower | -2.6413 | -3.2309 | -2.7547 | -2.8708 |
| eval/downstream/copycolors_10way_fast (soft loss v2) | lower | 0.09493 | 0.08891 | 0.09430 | 0.09547 |
| eval/downstream/copycolors_10way_fast (soft loss) | lower | 0.09493 | 0.08891 | 0.09430 | 0.09547 |
| eval/downstream/hellaswag_bpb_5shot (BPB v2) | lower | 1.6100 | 1.5877 | 1.5897 | 1.6225 |
| eval/downstream/hellaswag_bpb_5shot (BPB) | lower | 1.6274 | 1.6052 | 1.6071 | 1.6403 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB v2) | lower | 2.3986 | 2.3966 | 2.3765 | 2.4214 |
| eval/downstream/minerva_math_500_gold_bpb_0shot (BPB) | lower | 2.4063 | 2.4046 | 2.3842 | 2.4294 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB v2) | lower | 1.9686 | 1.9307 | 1.9200 | 1.9559 |
| eval/downstream/mmlu_humanities_test_bpb_5shot (BPB) | lower | 2.0769 | 2.0368 | 2.0257 | 2.0631 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB v2) | lower | 4.3444 | 4.4410 | 4.2725 | 4.2154 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (BPB) | lower | 8.6888 | 8.8821 | 8.5450 | 8.4308 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss v2) | lower | 3.0113 | 3.0782 | 2.9617 | 2.9220 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (CE loss) | lower | 6.0225 | 6.1565 | 5.9235 | 5.8440 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.24208 | 0.24208 | 0.24208 | 0.24208 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.24208 | 0.24208 | 0.24208 | 0.24208 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss v2) | lower | -1.6006 | -1.6652 | -1.6157 | -1.6204 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (log soft loss) | lower | -2.2290 | -2.4643 | -2.2837 | -2.3054 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss v2) | lower | 0.24651 | 0.24643 | 0.24664 | 0.24692 |
| eval/downstream/mmlu_humanities_test_mc_5shot_fast (soft loss) | lower | 0.24367 | 0.24351 | 0.24386 | 0.24406 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB v2) | lower | 2.2792 | 2.2778 | 2.2558 | 2.2753 |
| eval/downstream/mmlu_other_test_bpb_5shot (BPB) | lower | 2.5305 | 2.5307 | 2.5075 | 2.5258 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB v2) | lower | 4.0770 | 4.0129 | 3.8881 | 4.0076 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (BPB) | lower | 8.1540 | 8.0258 | 7.7763 | 8.0152 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss v2) | lower | 2.8261 | 2.7818 | 2.6956 | 2.7783 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (CE loss) | lower | 5.6522 | 5.5637 | 5.3912 | 5.5566 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.23751 | 0.23751 | 0.23751 | 0.23751 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.23751 | 0.23751 | 0.23751 | 0.23751 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss v2) | lower | -1.5726 | -1.6236 | -1.5686 | -1.5743 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (log soft loss) | lower | -2.1099 | -2.2952 | -2.0968 | -2.1187 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss v2) | lower | 0.24520 | 0.24425 | 0.24544 | 0.24554 |
| eval/downstream/mmlu_other_test_mc_5shot_fast (soft loss) | lower | 0.24101 | 0.23991 | 0.24142 | 0.24156 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB v2) | lower | 1.8047 | 1.7873 | 1.7678 | 1.8054 |
| eval/downstream/mmlu_social_sciences_test_bpb_5shot (BPB) | lower | 1.9242 | 1.9047 | 1.8856 | 1.9241 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB v2) | lower | 4.4248 | 4.3033 | 4.1377 | 4.1936 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (BPB) | lower | 8.8495 | 8.6067 | 8.2753 | 8.3872 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss v2) | lower | 3.0671 | 2.9831 | 2.8685 | 2.9071 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (CE loss) | lower | 6.1342 | 5.9662 | 5.7369 | 5.8141 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.21709 | 0.21709 | 0.21709 | 0.21709 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.21709 | 0.21709 | 0.21709 | 0.21709 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss v2) | lower | -1.6450 | -1.7083 | -1.6391 | -1.6521 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (log soft loss) | lower | -2.3280 | -2.5392 | -2.3033 | -2.3474 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss v2) | lower | 0.23556 | 0.23379 | 0.23577 | 0.23543 |
| eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (soft loss) | lower | 0.22414 | 0.22251 | 0.22455 | 0.22421 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB v2) | lower | 2.6906 | 2.6467 | 2.6122 | 2.6769 |
| eval/downstream/mmlu_stem_test_bpb_5shot (BPB) | lower | 3.2540 | 3.1927 | 3.1627 | 3.2429 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB v2) | lower | 4.4299 | 4.1699 | 4.2130 | 4.2547 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (BPB) | lower | 8.8598 | 8.3398 | 8.4259 | 8.5094 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss v2) | lower | 3.0706 | 2.8906 | 2.9205 | 2.9494 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (CE loss) | lower | 6.1411 | 5.7812 | 5.8410 | 5.8987 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2) | higher | 0.21372 | 0.21372 | 0.21372 | 0.21372 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy) | higher | 0.21372 | 0.21372 | 0.21372 | 0.21372 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss v2) | lower | -1.5984 | -1.6262 | -1.5917 | -1.6060 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (log soft loss) | lower | -2.1536 | -2.2481 | -2.1168 | -2.1715 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss v2) | lower | 0.23627 | 0.23525 | 0.23640 | 0.23580 |
| eval/downstream/mmlu_stem_test_mc_5shot_fast (soft loss) | lower | 0.22419 | 0.22299 | 0.22481 | 0.22381 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2) | lower | 1.8118 | 1.7810 | 1.7801 | 1.8613 |
| eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB) | lower | 1.8205 | 1.7895 | 1.7886 | 1.8703 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2) | lower | 1.4560 | 1.4255 | 1.4046 | 1.4860 |
| eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB) | lower | 1.4609 | 1.4306 | 1.4095 | 1.4912 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2) | lower | 2.7139 | 2.6931 | 2.6987 | 2.7439 |
| eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB) | lower | 2.7312 | 2.7106 | 2.7169 | 2.7618 |
| eval/lm/c4_en-validation/CE loss | lower | 5.6210 | 5.5855 | 5.5631 | 5.6558 |
| eval/lm/c4_en-validation/PPL | lower | 276.2 | 266.5 | 260.6 | 285.9 |
| eval/lm/dolma_books-validation/CE loss | lower | 5.8070 | 5.7885 | 5.7652 | 5.8546 |
| eval/lm/dolma_books-validation/PPL | lower | 332.6 | 326.5 | 319.0 | 348.8 |
| eval/lm/dolma_common-crawl-validation/CE loss | lower | 5.6453 | 5.6148 | 5.5943 | 5.6796 |
| eval/lm/dolma_common-crawl-validation/PPL | lower | 283.0 | 274.5 | 268.9 | 292.8 |
| eval/lm/dolma_pes2o-validation/CE loss | lower | 5.2031 | 5.1571 | 5.1297 | 5.2285 |
| eval/lm/dolma_pes2o-validation/PPL | lower | 181.8 | 173.7 | 169.0 | 186.5 |
| eval/lm/dolma_reddit-validation/CE loss | lower | 5.3800 | 5.3448 | 5.3362 | 5.4214 |
| eval/lm/dolma_reddit-validation/PPL | lower | 217.0 | 209.5 | 207.7 | 226.2 |
| eval/lm/dolma_stack-validation/CE loss | lower | 6.1949 | 6.1590 | 6.1090 | 6.2460 |
| eval/lm/dolma_stack-validation/PPL | lower | 490.2 | 472.9 | 449.9 | 516.0 |
| eval/lm/dolma_wiki-validation/CE loss | lower | 5.6395 | 5.5938 | 5.5670 | 5.6781 |
| eval/lm/dolma_wiki-validation/PPL | lower | 281.3 | 268.8 | 261.6 | 292.4 |
| eval/lm/ice-validation/CE loss | lower | 6.3222 | 6.2730 | 6.3377 | 6.3175 |
| eval/lm/ice-validation/PPL | lower | 556.8 | 530.1 | 565.5 | 554.2 |
| eval/lm/m2d2_s2orc-validation/CE loss | lower | 5.4049 | 5.3364 | 5.3213 | 5.4077 |
| eval/lm/m2d2_s2orc-validation/PPL | lower | 222.5 | 207.8 | 204.6 | 223.1 |
| eval/lm/pile-validation/CE loss | lower | 5.6881 | 5.6513 | 5.6201 | 5.7303 |
| eval/lm/pile-validation/PPL | lower | 295.3 | 284.7 | 275.9 | 308.1 |
| eval/lm/wikitext_103-validation/CE loss | lower | 6.5004 | 6.3839 | 6.4249 | 6.4773 |
| eval/lm/wikitext_103-validation/PPL | lower | 665.4 | 592.2 | 617.0 | 650.2 |
| throughput/in-loop eval batches | see metric | 1641.0 | 1641.0 | 1641.0 | 1641.0 |
| throughput/in-loop eval time (s) | see metric | 388.9 | 386.8 | 298.4 | 306.7 |

| run | state | family | tokens | step | link |
| --- | --- | --- | --- | --- | --- |
| int-smoke-intd256e8k-lr1.6e-3-r1<br>`sj1r51mi` | finished | original | 79429632.0 | 303 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/sj1r51mi) |
| int-smoke-intw256e8k-lr1.6e-3-r1<br>`rfox2uad` | finished | original | 81264640.0 | 310 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/rfox2uad) |
| q3-smoke-q3am128e8k-lr2e-3-r1<br>`12jqamu0` | finished | original | 81002496.0 | 309 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/12jqamu0) |
| q3-smoke-q3td128e8k-lr2e-3-r1<br>`99pn8kaz` | finished | original | 80740352.0 | 308 | [W&B](https://wandb.ai/ai2-llm/jacobm-olmoe-ladder/runs/99pn8kaz) |
