# OLMoBase Eval Results

Generated: 2026-07-04 06:06 UTC

Values are suite-level aggregates emitted by `olmo-eval`. Higher is better for accuracy/F1/pass-style metrics; lower is better for BPB/loss/perplexity-style metrics. The `direction` column is a heuristic based on suite/metric names, so treat `see metric` rows literally.

Completed result caches live under `/weka/oe-adapt-default/jacobm/olmoe3/OLMo-core/src/scripts/train/jacobm_olmoe_ladder/results/cache/olmobase`.

## High-Level Aggregates

| size | intervention | mcqa_stem up | mcqa_non_stem up | gen up | math up | easy_qa_rc up | easy_qa_bpb down | easy_math_bpb down | easy_code_bpb down |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 275M | baseline | 0.24197 | 0.28125 | 0.34705 | 0.01417 | 0.48352 | 1.0800 | 0.78918 | 0.92581 |
| 275M | integration 1 deep | 0.25321 | 0.27677 | 0.37450 | 0.01564 | 0.50487 | 1.0432 | 0.75207 | 0.93082 |
| 275M | qwen-like 4.5d | 0.25237 | 0.27217 | 0.34279 | 0.01586 | 0.49771 | 1.0693 | 0.77458 | 0.93744 |
| 275M | qwen-like 3.0d | 0.24018 | 0.27657 | 0.37451 | 0.01670 | 0.50462 | 1.0545 | 0.76818 | 0.93331 |
| 480M | baseline | 0.24855 | 0.27379 | 0.43339 | 0.01997 | 0.53942 | 0.98870 | 0.70521 | 0.88722 |
| 480M | integration 1 wide | 0.25207 | 0.27359 | 0.45025 | 0.02120 | 0.55333 | 0.95864 | 0.67488 | 0.87528 |
| 480M | qwen-like 4.5d | 0.25479 | 0.27550 | 0.41537 | 0.02104 | 0.53874 | 0.97621 | 0.69955 | 0.91262 |
| 810M | baseline | 0.25859 | 0.27120 | 0.50610 | 0.02979 | 0.58986 | 0.92123 | 0.64066 | 0.87301 |
| 810M | qwen-like 4.5d | 0.24638 | 0.27381 | 0.51870 | 0.03030 | 0.59611 | 0.90231 | 0.62896 | 0.89655 |
| 810M | qwen-like 3.0d | 0.23549 | 0.27766 | 0.51889 | 0.03169 | 0.60040 | 0.90398 | 0.62314 | 0.88933 |
| 1.2B | baseline | 0.25779 | 0.28528 | 0.55182 | 0.04853 | 0.62612 | 0.87255 | 0.59977 | 0.87110 |
| 1.2B | qwen-like 4.5d | 0.28004 | 0.29095 | 0.57006 | 0.05434 | 0.63658 | 0.85351 | 0.58820 | 0.89175 |
| 1.2B | qwen-like 3.0d | 0.42065 | 0.41932 | 0.57200 | 0.06234 | 0.64473 | 0.85149 | 0.58491 | 0.87557 |

## Status

| model | status | workspace | link | message |
| --- | --- | --- | --- | --- |
| olmoe3-275m-cx8-baseline-olmobase | finalized | ai2/olmo-instruct | [beaker](https://beaker.org/ex/01KWNC2HEFG1P706WE2MQN26AH) |  |
| olmoe3-275m-cx8-int-deep-olmobase | finalized | ai2/olmo-instruct | [beaker](https://beaker.org/ex/01KWNC53DAXCRFM4A07J8QFMA4) |  |
| olmoe3-275m-cx8-int-wide-olmobase | finalized | ai2/olmo-instruct | [beaker](https://beaker.org/ex/01KWNC4EY14WA5ATZA2KSWJFVJ) | finalized; result artifact not cached yet |
| olmoe3-275m-cx8-q3am-olmobase | finalized | ai2/olmo-instruct | [beaker](https://beaker.org/ex/01KWNC35WC375QGJ1KDZ0X64ES) |  |
| olmoe3-275m-cx8-q3td-olmobase | finalized | ai2/olmo-instruct | [beaker](https://beaker.org/ex/01KWNC3SSSR77GFQNZFEYFKFG9) |  |
| olmoe3-480m-cx8-baseline-olmobase | finalized | ai2/olmo-instruct | [beaker](https://beaker.org/ex/01KWNC2VJ296NADBKAAES8Q052) |  |
| olmoe3-480m-cx8-int-deep-olmobase | finalized | ai2/olmo-instruct | [beaker](https://beaker.org/ex/01KWNC5DAJ9EACPHRME4EFDR4R) | finalized; result artifact not cached yet |
| olmoe3-480m-cx8-int-wide-olmobase | finalized | ai2/olmo-instruct | [beaker](https://beaker.org/ex/01KWNC4S04TB82C6V4QZZTJ6KH) |  |
| olmoe3-480m-cx8-q3am-olmobase | finalized | ai2/olmo-instruct | [beaker](https://beaker.org/ex/01KWNC3FTGZ6941KQYPDHV925Z) |  |
| olmoe3-480m-cx8-q3td-olmobase | finalized | ai2/olmo-instruct | [beaker](https://beaker.org/ex/01KWNC44G1NF1GW32JXYJAEGGH) | finalized; result artifact not cached yet |
| olmoe3-810m-cx8-baseline-olmobase | finalized | ai2/OLMo-3-moe-experiments | [beaker](https://beaker.org/ex/01KWNAC2MT4K2H847P4JQBWHX0) |  |
| olmoe3-810m-cx8-q3am-olmobase | finalized | ai2/OLMo-3-moe-experiments | [beaker](https://beaker.org/ex/01KWNAD5S3W3SST3TM4X1SX2VM) |  |
| olmoe3-810m-cx8-q3td-olmobase | finalized | ai2/OLMo-3-moe-experiments | [beaker](https://beaker.org/ex/01KWNAEAG73CRPPP952KAX0EYW) |  |
| olmoe3-1p2b-cx8-baseline-olmobase | finalized | ai2/OLMo-3-moe-experiments | [beaker](https://beaker.org/ex/01KWNACAVEJPRAP8JK7PM85DZK) |  |
| olmoe3-1p2b-cx8-q3am-olmobase | finalized | ai2/OLMo-3-moe-experiments | [beaker](https://beaker.org/ex/01KWNADEPKBF1JY939N5JG2GTN) |  |
| olmoe3-1p2b-cx8-q3td-olmobase | finalized | ai2/OLMo-3-moe-experiments | [beaker](https://beaker.org/ex/01KWNAEJA903D4EVF65CAMB032) |  |

## Suite Aggregates

| suite | metric | direction | olmoe3-275m-cx8-baseline-olmobase | olmoe3-275m-cx8-int-deep-olmobase | olmoe3-275m-cx8-int-wide-olmobase | olmoe3-275m-cx8-q3am-olmobase | olmoe3-275m-cx8-q3td-olmobase | olmoe3-480m-cx8-baseline-olmobase | olmoe3-480m-cx8-int-deep-olmobase | olmoe3-480m-cx8-int-wide-olmobase | olmoe3-480m-cx8-q3am-olmobase | olmoe3-480m-cx8-q3td-olmobase | olmoe3-810m-cx8-baseline-olmobase | olmoe3-810m-cx8-q3am-olmobase | olmoe3-810m-cx8-q3td-olmobase | olmoe3-1p2b-cx8-baseline-olmobase | olmoe3-1p2b-cx8-q3am-olmobase | olmoe3-1p2b-cx8-q3td-olmobase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arc:bpb:olmo3base | primary_score:average | lower | 0.48128 | 0.51135 |  | 0.52096 | 0.51059 | 0.55602 |  | 0.56936 | 0.55226 |  | 0.58353 | 0.60392 | 0.58939 | 0.64111 | 0.62137 | 0.64056 |
| arc:mc:olmo3base | primary_score:average | higher | 0.25673 | 0.24662 |  | 0.23729 | 0.22781 | 0.25940 |  | 0.23816 | 0.25145 |  | 0.25448 | 0.25720 | 0.25176 | 0.26618 | 0.26338 | 0.47690 |
| arc:rc:olmo3base | primary_score:average | higher | 0.48170 | 0.51114 |  | 0.52117 | 0.51059 | 0.55538 |  | 0.56979 | 0.55205 |  | 0.58353 | 0.60392 | 0.58939 | 0.64111 | 0.62137 | 0.64056 |
| arc_challenge:bpb:olmo3base | accuracy:logprob | higher | 0.34471 | 0.37287 |  | 0.39334 | 0.38567 | 0.41297 |  | 0.42491 | 0.41724 |  | 0.43601 | 0.46331 | 0.43089 | 0.51792 | 0.49147 | 0.49829 |
| arc_challenge:mc_olmo3base | accuracy:logprob | higher | 0.25714 | 0.24324 |  | 0.22625 | 0.22162 | 0.27722 |  | 0.22548 | 0.25753 |  | 0.24170 | 0.26062 | 0.24595 | 0.25753 | 0.26371 | 0.40541 |
| arc_challenge:rc:olmo3base | accuracy:logprob | higher | 0.34556 | 0.37287 |  | 0.39334 | 0.38567 | 0.41212 |  | 0.42577 | 0.41724 |  | 0.43601 | 0.46331 | 0.43089 | 0.51792 | 0.49147 | 0.49829 |
| arc_easy:bpb:olmo3base | accuracy:logprob | higher | 0.61785 | 0.64983 |  | 0.64857 | 0.63552 | 0.69907 |  | 0.71380 | 0.68729 |  | 0.73106 | 0.74453 | 0.74790 | 0.76431 | 0.75126 | 0.78283 |
| arc_easy:mc:olmo3base | accuracy:logprob | higher | 0.25631 | 0.25000 |  | 0.24832 | 0.23401 | 0.24158 |  | 0.25084 | 0.24537 |  | 0.26726 | 0.25379 | 0.25758 | 0.27483 | 0.26305 | 0.54840 |
| arc_easy:rc:olmo3base | accuracy:logprob | higher | 0.61785 | 0.64941 |  | 0.64899 | 0.63552 | 0.69865 |  | 0.71380 | 0.68687 |  | 0.73106 | 0.74453 | 0.74790 | 0.76431 | 0.75126 | 0.78283 |
| basic_skills:bpb:olmo3base | primary_score:average | lower | 1.0743 | 1.0084 |  | 1.0503 | 1.0538 | 0.92320 |  | 0.84072 | 0.88683 |  | 0.79227 | 0.74689 | 0.74918 | 0.69630 | 0.70394 | 0.71369 |
| basic_skills:rc:olmo3base | primary_score:average | higher | 0.56539 | 0.59924 |  | 0.58178 | 0.57766 | 0.64139 |  | 0.66838 | 0.64018 |  | 0.69513 | 0.70809 | 0.70810 | 0.73806 | 0.75139 | 0.73438 |
| basic_skills_arithmetic:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 1.7088 | 1.4394 |  | 1.6472 | 1.6618 | 1.4331 |  | 1.0205 | 1.2383 |  | 0.99876 | 0.92788 | 0.89264 | 0.87840 | 0.76989 | 0.86919 |
| basic_skills_arithmetic:rc:olmo3base | accuracy:logprob | higher | 0.29990 | 0.38873 |  | 0.34384 | 0.31996 | 0.46705 |  | 0.54823 | 0.47755 |  | 0.56829 | 0.58739 | 0.60076 | 0.64374 | 0.66380 | 0.60840 |
| basic_skills_coding:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 1.0450 | 0.99475 |  | 1.0558 | 1.0344 | 0.95633 |  | 1.0130 | 0.97116 |  | 0.95831 | 1.0502 | 1.0810 | 1.0311 | 1.1110 | 1.0338 |
| basic_skills_coding:rc:olmo3base | accuracy:logprob | higher | 0.51976 | 0.54743 |  | 0.53458 | 0.53557 | 0.56719 |  | 0.58893 | 0.58992 |  | 0.61660 | 0.61858 | 0.62154 | 0.63834 | 0.63538 | 0.64328 |
| basic_skills_common_knowledge:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.62859 | 0.47014 |  | 0.43047 | 0.49411 | 0.43549 |  | 0.39568 | 0.41825 |  | 0.29995 | 0.27467 | 0.30343 | 0.23697 | 0.26087 | 0.21484 |
| basic_skills_common_knowledge:rc:olmo3base | accuracy:logprob | higher | 0.73481 | 0.80231 |  | 0.80810 | 0.76567 | 0.83028 |  | 0.88428 | 0.84185 |  | 0.90743 | 0.91418 | 0.91610 | 0.94214 | 0.94407 | 0.93635 |
| basic_skills_logical_reasoning:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.13260 | 0.14010 |  | 0.15587 | 0.15297 | 0.12639 |  | 0.11503 | 0.11800 |  | 0.09502 | 0.09396 | 0.09195 | 0.08030 | 0.08191 | 0.07335 |
| basic_skills_logical_reasoning:rc:olmo3base | accuracy:logprob | higher | 0.89177 | 0.92039 |  | 0.84884 | 0.89356 | 0.92308 |  | 0.92308 | 0.86941 |  | 0.94544 | 0.95349 | 0.89088 | 0.95259 | 0.96691 | 0.94007 |
| basic_skills_pattern:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 1.0463 | 1.1065 |  | 1.0216 | 1.0220 | 0.92131 |  | 0.96010 | 0.90474 |  | 0.98757 | 0.79721 | 0.83956 | 0.76111 | 0.76456 | 0.89652 |
| basic_skills_pattern:rc:olmo3base | accuracy:logprob | higher | 0.63109 | 0.64045 |  | 0.64607 | 0.65918 | 0.71536 |  | 0.70974 | 0.71536 |  | 0.73596 | 0.75655 | 0.76404 | 0.76592 | 0.78464 | 0.77528 |
| basic_skills_string_operations:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 1.8842 | 1.8992 |  | 1.9910 | 1.9574 | 1.6666 |  | 1.5400 | 1.6706 |  | 1.4140 | 1.3375 | 1.2865 | 1.1900 | 1.2354 | 1.1945 |
| basic_skills_string_operations:rc:olmo3base | accuracy:logprob | higher | 0.31501 | 0.29614 |  | 0.30927 | 0.29204 | 0.34537 |  | 0.35603 | 0.34701 |  | 0.39705 | 0.41838 | 0.45529 | 0.48564 | 0.51354 | 0.50287 |
| codex_humaneval:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.83557 | 0.83422 |  | 0.83522 | 0.84197 | 0.79467 |  | 0.76892 | 0.79918 |  | 0.75002 | 0.79146 | 0.78114 | 0.74839 | 0.76928 | 0.76447 |
| coqa:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.69321 | 0.66405 |  | 0.71113 | 0.67084 | 0.64629 |  | 0.60792 | 0.61388 |  | 0.54211 | 0.51645 | 0.54337 | 0.51198 | 0.47564 | 0.47596 |
| coqa:gen:olmo3base | f1:f1 | higher | 0.33542 | 0.38381 |  | 0.34656 | 0.38358 | 0.42984 |  | 0.43918 | 0.43291 |  | 0.51718 | 0.50682 | 0.53021 | 0.55144 | 0.58683 | 0.59752 |
| coqa:mc:olmo3base | accuracy:logprob | higher | 0.23850 | 0.25164 |  | 0.25305 | 0.26995 | 0.23897 |  | 0.23709 | 0.24038 |  | 0.25399 | 0.24460 | 0.25352 | 0.24319 | 0.27136 | 0.34977 |
| coqa:rc:olmo3base | accuracy:logprob | higher | 0.58592 | 0.63005 |  | 0.59484 | 0.63521 | 0.65634 |  | 0.67559 | 0.65728 |  | 0.72817 | 0.72911 | 0.73850 | 0.76808 | 0.79202 | 0.79906 |
| csqa:bpb:olmo3base | accuracy:logprob | higher | 0.50614 | 0.53890 |  | 0.53071 | 0.52826 | 0.57576 |  | 0.56511 | 0.57903 |  | 0.63227 | 0.62654 | 0.62735 | 0.66830 | 0.67486 | 0.67813 |
| csqa:mc_olmo3base | accuracy:logprob | higher | 0.19430 | 0.20090 |  | 0.20040 | 0.20440 | 0.19430 |  | 0.19420 | 0.20740 |  | 0.19780 | 0.20240 | 0.19170 | 0.20020 | 0.21010 | 0.40660 |
| csqa:rc:olmo3base | accuracy:logprob | higher | 0.50614 | 0.53890 |  | 0.53071 | 0.52826 | 0.57494 |  | 0.56511 | 0.57903 |  | 0.63227 | 0.62654 | 0.62735 | 0.66830 | 0.67486 | 0.67813 |
| drop:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 1.7109 | 1.7170 |  | 1.7545 | 1.7265 | 1.5829 |  | 1.5932 | 1.5137 |  | 1.5609 | 1.4073 | 1.4955 | 1.3562 | 1.2825 | 1.2888 |
| drop:gen:olmo3base | f1:drop_f1 | higher | 0.13981 | 0.17428 |  | 0.15655 | 0.16159 | 0.21943 |  | 0.21323 | 0.20340 |  | 0.24686 | 0.25651 | 0.24222 | 0.27136 | 0.29757 | 0.30091 |
| drop:mc:olmo3base | accuracy:logprob | higher | 0.26574 | 0.23048 |  | 0.25378 | 0.24244 | 0.23300 |  | 0.24811 | 0.22922 |  | 0.24118 | 0.22859 | 0.22166 | 0.25945 | 0.24748 | 0.25630 |
| drop:rc:olmo3base | accuracy:logprob | higher | 0.28526 | 0.28715 |  | 0.29030 | 0.29975 | 0.33501 |  | 0.34257 | 0.34761 |  | 0.37720 | 0.37594 | 0.39169 | 0.42191 | 0.44458 | 0.45466 |
| gsm8k:olmo3base | pass_at_1:exact_match | higher | 0.01905 | 0.01933 |  | 0.01895 | 0.02445 | 0.02436 |  | 0.02654 | 0.02644 |  | 0.03952 | 0.04103 | 0.04719 | 0.07534 | 0.08548 | 0.10462 |
| gsm_symb:olmo3base | primary_score:average | higher | 0.00898 | 0.00920 |  | 0.01000 | 0.00996 | 0.00983 |  | 0.00995 | 0.01024 |  | 0.01401 | 0.01327 | 0.01386 | 0.02683 | 0.02528 | 0.03121 |
| gsm_symbolic:olmo3base | pass_at_1:exact_match | higher | 0.00843 | 0.00873 |  | 0.00985 | 0.00937 | 0.01155 |  | 0.01337 | 0.01307 |  | 0.02245 | 0.02193 | 0.02167 | 0.05895 | 0.05320 | 0.07032 |
| gsm_symbolic:p1:olmo3base | pass_at_1:exact_match | higher | 0.00988 | 0.00958 |  | 0.01275 | 0.01075 | 0.01047 |  | 0.00783 | 0.00950 |  | 0.01112 | 0.01043 | 0.01130 | 0.01553 | 0.01503 | 0.01625 |
| gsm_symbolic:p2:olmo3base | pass_at_1:exact_match | higher | 0.00865 | 0.00930 |  | 0.00740 | 0.00975 | 0.00745 |  | 0.00865 | 0.00815 |  | 0.00845 | 0.00745 | 0.00860 | 0.00600 | 0.00760 | 0.00705 |
| hellaswag:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.85322 | 0.83413 |  | 0.84668 | 0.84030 | 0.80051 |  | 0.79003 | 0.79723 |  | 0.76574 | 0.75798 | 0.75951 | 0.74483 | 0.73524 | 0.73399 |
| hellaswag:rc:olmo3base | accuracy:logprob | higher | 0.45280 | 0.48660 |  | 0.47140 | 0.47930 | 0.55010 |  | 0.57390 | 0.55130 |  | 0.61820 | 0.63490 | 0.63490 | 0.66650 | 0.68670 | 0.68660 |
| jeopardy:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 1.0161 | 0.86781 |  | 0.95449 | 0.91178 | 0.71685 |  | 0.68969 | 0.74680 |  | 0.57766 | 0.55158 | 0.53394 | 0.47283 | 0.44584 | 0.44043 |
| jeopardy:gen:olmo3base | f1:f1 | higher | 0.09669 | 0.13374 |  | 0.11933 | 0.13120 | 0.23007 |  | 0.25606 | 0.21991 |  | 0.35316 | 0.37510 | 0.37935 | 0.43720 | 0.44846 | 0.44754 |
| jeopardy:mc:olmo3base | accuracy:logprob | higher | 0.26562 | 0.25846 |  | 0.24654 | 0.23796 | 0.25465 |  | 0.25370 | 0.26085 |  | 0.24320 | 0.23987 | 0.23939 | 0.25990 | 0.24845 | 0.47544 |
| jeopardy:rc:olmo3base | accuracy:logprob | higher | 0.44063 | 0.46781 |  | 0.45875 | 0.46018 | 0.54459 |  | 0.61040 | 0.55126 |  | 0.65188 | 0.67668 | 0.68145 | 0.71769 | 0.74249 | 0.75298 |
| lab_bench_dbqa:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 3.9078 | 3.8479 |  | 3.8609 | 3.9496 | 3.7480 |  | 3.6877 | 3.7897 |  | 3.6687 | 3.6499 | 3.6513 | 3.5215 | 3.4592 | 3.4034 |
| lab_bench_dbqa:olmo3base | accuracy:logprob | higher | 0.26538 | 0.27885 |  | 0.27885 | 0.28462 | 0.24423 |  | 0.28462 | 0.26923 |  | 0.27500 | 0.27308 | 0.29423 | 0.28846 | 0.27885 | 0.29038 |
| lab_bench_protocolqa:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 1.3072 | 1.2694 |  | 1.2863 | 1.2933 | 1.2651 |  | 1.2379 | 1.2507 |  | 1.2236 | 1.2022 | 1.2073 | 1.1948 | 1.1928 | 1.1658 |
| lab_bench_protocolqa:olmo3base | accuracy:logprob | higher | 0.25926 | 0.22222 |  | 0.26852 | 0.25000 | 0.25926 |  | 0.26852 | 0.27778 |  | 0.25926 | 0.23148 | 0.27778 | 0.25000 | 0.24074 | 0.25000 |
| lambada | greedy_accuracy:logprob | higher | 0.45449 | 0.47273 |  | 0.47448 | 0.47817 | 0.53503 |  | 0.55308 | 0.52785 |  | 0.59577 | 0.60916 | 0.59732 | 0.63652 | 0.64661 | 0.65981 |
| lambada:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.61420 | 0.59046 |  | 0.58766 | 0.58226 | 0.50510 |  | 0.48248 | 0.50613 |  | 0.43045 | 0.41668 | 0.42477 | 0.39208 | 0.37542 | 0.36522 |
| lambada:olmo3base | greedy_accuracy:logprob | higher | 0.45449 | 0.47273 |  | 0.47448 | 0.47817 | 0.53503 |  | 0.55288 | 0.52785 |  | 0.59558 | 0.60916 | 0.59732 | 0.63691 | 0.64661 | 0.65981 |
| mbpp:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 1.1539 | 1.1715 |  | 1.1767 | 1.1694 | 1.1335 |  | 1.1404 | 1.1771 |  | 1.1648 | 1.1772 | 1.1747 | 1.1641 | 1.2035 | 1.1644 |
| medmcqa:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 1.4138 | 1.3637 |  | 1.3884 | 1.3639 | 1.2677 |  | 1.2170 | 1.2536 |  | 1.1671 | 1.1537 | 1.1493 | 1.1101 | 1.0844 | 1.0891 |
| medmcqa:mc:olmo3base | accuracy:logprob | higher | 0.22854 | 0.29405 |  | 0.29978 | 0.24074 | 0.26321 |  | 0.32202 | 0.30791 |  | 0.30002 | 0.23117 | 0.23596 | 0.26225 | 0.32991 | 0.32728 |
| medmcqa:rc:olmo3base | accuracy:logprob | higher | 0.27349 | 0.27588 |  | 0.27660 | 0.28186 | 0.28998 |  | 0.30146 | 0.29811 |  | 0.30935 | 0.31150 | 0.32226 | 0.32297 | 0.33612 | 0.33517 |
| medqa_en:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.99692 | 0.95637 |  | 0.98542 | 0.97689 | 0.89706 |  | 0.86353 | 0.88904 |  | 0.81832 | 0.81318 | 0.79955 | 0.79129 | 0.75922 | 0.75625 |
| medqa_en:mc:olmo3base | accuracy:logprob | higher | 0.19953 | 0.20896 |  | 0.22310 | 0.20031 | 0.21288 |  | 0.21681 | 0.21524 |  | 0.21603 | 0.20974 | 0.19639 | 0.19953 | 0.21681 | 0.26159 |
| medqa_en:rc:olmo3base | accuracy:logprob | higher | 0.22859 | 0.23252 |  | 0.22702 | 0.22152 | 0.23645 |  | 0.25687 | 0.25216 |  | 0.27651 | 0.27416 | 0.29222 | 0.29301 | 0.30244 | 0.31265 |
| minerva_math:bpb:olmo3base | primary_score:average | lower | 0.78918 | 0.75207 |  | 0.77458 | 0.76818 | 0.70521 |  | 0.67488 | 0.69955 |  | 0.64066 | 0.62896 | 0.62314 | 0.59977 | 0.58820 | 0.58491 |
| minerva_math:olmo3base | primary_score:average | higher | 0.01449 | 0.01839 |  | 0.01862 | 0.01568 | 0.02572 |  | 0.02712 | 0.02643 |  | 0.03586 | 0.03659 | 0.03400 | 0.04342 | 0.05227 | 0.05117 |
| minerva_math_algebra:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.84352 | 0.80039 |  | 0.82251 | 0.82345 | 0.74688 |  | 0.71570 | 0.74514 |  | 0.67672 | 0.66434 | 0.65269 | 0.62552 | 0.61257 | 0.61145 |
| minerva_math_algebra:olmo3base | pass_at_1:minerva_math_flex | higher | 0.02001 | 0.01959 |  | 0.02211 | 0.01959 | 0.02591 |  | 0.02317 | 0.02275 |  | 0.03328 | 0.04149 | 0.03812 | 0.05097 | 0.05581 | 0.06676 |
| minerva_math_counting_and_probability:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.74333 | 0.71426 |  | 0.73258 | 0.72676 | 0.67351 |  | 0.64227 | 0.66632 |  | 0.61135 | 0.59897 | 0.59936 | 0.57149 | 0.55527 | 0.55738 |
| minerva_math_counting_and_probability:olmo3base | pass_at_1:minerva_math_flex | higher | 0.01635 | 0.02057 |  | 0.02268 | 0.01846 | 0.03323 |  | 0.02373 | 0.02795 |  | 0.02848 | 0.02954 | 0.03059 | 0.04378 | 0.04852 | 0.04378 |
| minerva_math_geometry:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.83466 | 0.78213 |  | 0.81003 | 0.80462 | 0.73865 |  | 0.69814 | 0.73165 |  | 0.67065 | 0.65988 | 0.65364 | 0.63173 | 0.61688 | 0.61867 |
| minerva_math_geometry:olmo3base | pass_at_1:minerva_math_flex | higher | 0.00887 | 0.01044 |  | 0.00992 | 0.01200 | 0.01566 |  | 0.01879 | 0.02766 |  | 0.03862 | 0.03288 | 0.02557 | 0.04436 | 0.04489 | 0.04958 |
| minerva_math_intermediate_algebra:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.83818 | 0.80332 |  | 0.83396 | 0.82148 | 0.75092 |  | 0.73122 | 0.75030 |  | 0.69319 | 0.68183 | 0.66713 | 0.65041 | 0.64032 | 0.63253 |
| minerva_math_intermediate_algebra:olmo3base | pass_at_1:minerva_math_flex | higher | 0.01107 | 0.01633 |  | 0.01301 | 0.01301 | 0.02464 |  | 0.02326 | 0.01827 |  | 0.02935 | 0.02436 | 0.02464 | 0.02464 | 0.03295 | 0.03212 |
| minerva_math_number_theory:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.85006 | 0.81376 |  | 0.83161 | 0.83044 | 0.76849 |  | 0.73984 | 0.76400 |  | 0.70855 | 0.69488 | 0.69160 | 0.66896 | 0.65778 | 0.65428 |
| minerva_math_number_theory:olmo3base | pass_at_1:minerva_math_flex | higher | 0.01296 | 0.01574 |  | 0.01713 | 0.01111 | 0.01898 |  | 0.02778 | 0.02824 |  | 0.03009 | 0.03056 | 0.03565 | 0.03009 | 0.04306 | 0.04769 |
| minerva_math_prealgebra:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.75042 | 0.71466 |  | 0.73306 | 0.72671 | 0.66677 |  | 0.63362 | 0.65750 |  | 0.59949 | 0.58772 | 0.58490 | 0.55874 | 0.54748 | 0.54421 |
| minerva_math_prealgebra:olmo3base | pass_at_1:minerva_math_flex | higher | 0.02210 | 0.02727 |  | 0.03129 | 0.02325 | 0.03416 |  | 0.05023 | 0.04277 |  | 0.06142 | 0.06803 | 0.06056 | 0.08037 | 0.10448 | 0.09127 |
| minerva_math_precalculus:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.66409 | 0.63594 |  | 0.65829 | 0.64381 | 0.59128 |  | 0.56337 | 0.58192 |  | 0.52468 | 0.51511 | 0.51264 | 0.49157 | 0.48712 | 0.47585 |
| minerva_math_precalculus:olmo3base | pass_at_1:minerva_math_flex | higher | 0.01007 | 0.01877 |  | 0.01419 | 0.01236 | 0.02747 |  | 0.02289 | 0.01740 |  | 0.02976 | 0.02930 | 0.02289 | 0.02976 | 0.03617 | 0.02701 |
| mmlu:bpb | primary_score:average | lower | 1.1226 | 1.0786 |  | 1.1003 | 1.0909 | 1.0128 |  | 0.97891 | 1.0007 |  | 0.93465 | 0.91689 | 0.91017 | 0.87666 | 0.85838 | 0.85970 |
| mmlu:humanities:mc:olmo3base | primary_score:average | higher | 0.25203 | 0.24957 |  | 0.24525 | 0.25562 | 0.24342 |  | 0.24994 | 0.24748 |  | 0.24676 | 0.27681 | 0.28071 | 0.27745 | 0.27345 | 0.39708 |
| mmlu:other:mc:olmo3base | primary_score:average | higher | 0.27936 | 0.25507 |  | 0.24902 | 0.25344 | 0.26339 |  | 0.25278 | 0.26450 |  | 0.24641 | 0.25815 | 0.26697 | 0.25816 | 0.28232 | 0.41681 |
| mmlu:rc:olmo3base | primary_score:average | higher | 0.31088 | 0.31928 |  | 0.31979 | 0.31615 | 0.33764 |  | 0.35194 | 0.34480 |  | 0.36509 | 0.37712 | 0.37780 | 0.39576 | 0.40214 | 0.40637 |
| mmlu:social_sciences:mc:olmo3base | primary_score:average | higher | 0.25019 | 0.27645 |  | 0.22624 | 0.25622 | 0.25411 |  | 0.23233 | 0.25039 |  | 0.23356 | 0.24505 | 0.27320 | 0.29003 | 0.31144 | 0.41539 |
| mmlu:stem:mc:olmo3base | primary_score:average | higher | 0.28807 | 0.25541 |  | 0.24668 | 0.26701 | 0.26828 |  | 0.22738 | 0.25132 |  | 0.24743 | 0.27680 | 0.26736 | 0.28097 | 0.27910 | 0.35747 |
| mmlu_abstract_algebra:mc:olmo3base | accuracy:logprob | higher | 0.25000 | 0.26000 |  | 0.26000 | 0.27000 | 0.27000 |  | 0.24000 | 0.19000 |  | 0.22000 | 0.23000 | 0.26000 | 0.25000 | 0.26000 | 0.19000 |
| mmlu_abstract_algebra:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.0277 | 0.97167 |  | 1.0414 | 1.0445 | 0.99639 |  | 0.97451 | 1.0565 |  | 0.88534 | 0.87385 | 0.88149 | 0.83556 | 0.84838 | 0.81143 |
| mmlu_abstract_algebra:rc:olmo3base | accuracy:logprob | higher | 0.25000 | 0.22000 |  | 0.21000 | 0.25000 | 0.23000 |  | 0.19000 | 0.23000 |  | 0.20000 | 0.31000 | 0.22000 | 0.23000 | 0.23000 | 0.28000 |
| mmlu_anatomy:mc:olmo3base | accuracy:logprob | higher | 0.24444 | 0.19259 |  | 0.18519 | 0.17778 | 0.28889 |  | 0.18519 | 0.21481 |  | 0.20741 | 0.31111 | 0.30370 | 0.21481 | 0.25926 | 0.37778 |
| mmlu_anatomy:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.81655 | 0.77346 |  | 0.78495 | 0.75623 | 0.68858 |  | 0.65446 | 0.67273 |  | 0.60346 | 0.60093 | 0.59550 | 0.57548 | 0.54850 | 0.54213 |
| mmlu_anatomy:rc:olmo3base | accuracy:logprob | higher | 0.37778 | 0.33333 |  | 0.33333 | 0.35556 | 0.38519 |  | 0.41481 | 0.39259 |  | 0.45185 | 0.44444 | 0.41481 | 0.46667 | 0.47407 | 0.48148 |
| mmlu_astronomy:mc:olmo3base | accuracy:logprob | higher | 0.32237 | 0.17105 |  | 0.20395 | 0.28947 | 0.25658 |  | 0.15789 | 0.24342 |  | 0.19079 | 0.25000 | 0.32895 | 0.30263 | 0.34211 | 0.42105 |
| mmlu_astronomy:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.0221 | 0.99042 |  | 1.0129 | 1.0127 | 0.92043 |  | 0.88115 | 0.88980 |  | 0.87050 | 0.83840 | 0.85251 | 0.79600 | 0.80093 | 0.81749 |
| mmlu_astronomy:rc:olmo3base | accuracy:logprob | higher | 0.34868 | 0.36184 |  | 0.41447 | 0.34868 | 0.38816 |  | 0.38158 | 0.39474 |  | 0.39474 | 0.43421 | 0.42763 | 0.46711 | 0.44079 | 0.47368 |
| mmlu_business_ethics:mc:olmo3base | accuracy:logprob | higher | 0.23000 | 0.30000 |  | 0.30000 | 0.35000 | 0.21000 |  | 0.33000 | 0.27000 |  | 0.30000 | 0.27000 | 0.25000 | 0.25000 | 0.29000 | 0.39000 |
| mmlu_business_ethics:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.4047 | 1.4458 |  | 1.3870 | 1.4323 | 1.3785 |  | 1.3713 | 1.3260 |  | 1.2221 | 1.2235 | 1.1929 | 1.1967 | 1.1202 | 1.2295 |
| mmlu_business_ethics:rc:olmo3base | accuracy:logprob | higher | 0.43000 | 0.41000 |  | 0.44000 | 0.41000 | 0.46000 |  | 0.49000 | 0.45000 |  | 0.53000 | 0.51000 | 0.48000 | 0.54000 | 0.54000 | 0.54000 |
| mmlu_clinical_knowledge:mc:olmo3base | accuracy:logprob | higher | 0.24528 | 0.21132 |  | 0.24528 | 0.23396 | 0.24528 |  | 0.21509 | 0.28679 |  | 0.21132 | 0.23774 | 0.32453 | 0.24151 | 0.26792 | 0.44906 |
| mmlu_clinical_knowledge:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.96115 | 0.93663 |  | 0.94338 | 0.92272 | 0.85116 |  | 0.83845 | 0.84500 |  | 0.81100 | 0.79019 | 0.77196 | 0.74958 | 0.74006 | 0.74234 |
| mmlu_clinical_knowledge:rc:olmo3base | accuracy:logprob | higher | 0.36981 | 0.38113 |  | 0.37736 | 0.36604 | 0.36981 |  | 0.40377 | 0.38113 |  | 0.39245 | 0.40377 | 0.43019 | 0.43396 | 0.41509 | 0.45283 |
| mmlu_college_biology:mc:olmo3base | accuracy:logprob | higher | 0.24306 | 0.25694 |  | 0.25694 | 0.29167 | 0.25000 |  | 0.25694 | 0.24306 |  | 0.23611 | 0.25694 | 0.20833 | 0.25000 | 0.27083 | 0.43056 |
| mmlu_college_biology:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.80631 | 0.74436 |  | 0.76355 | 0.75977 | 0.66504 |  | 0.64989 | 0.66391 |  | 0.61137 | 0.59459 | 0.59409 | 0.55786 | 0.53092 | 0.54461 |
| mmlu_college_biology:rc:olmo3base | accuracy:logprob | higher | 0.36806 | 0.40278 |  | 0.35417 | 0.39583 | 0.41667 |  | 0.43056 | 0.38889 |  | 0.40278 | 0.48611 | 0.47222 | 0.44444 | 0.52083 | 0.47917 |
| mmlu_college_chemistry:mc:olmo3base | accuracy:logprob | higher | 0.35000 | 0.17000 |  | 0.22000 | 0.33000 | 0.24000 |  | 0.21000 | 0.18000 |  | 0.27000 | 0.28000 | 0.25000 | 0.27000 | 0.16000 | 0.34000 |
| mmlu_college_chemistry:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.7411 | 1.6806 |  | 1.6972 | 1.6648 | 1.6139 |  | 1.5223 | 1.5774 |  | 1.4492 | 1.4683 | 1.4397 | 1.3602 | 1.4101 | 1.3912 |
| mmlu_college_chemistry:rc:olmo3base | accuracy:logprob | higher | 0.30000 | 0.26000 |  | 0.28000 | 0.32000 | 0.26000 |  | 0.26000 | 0.33000 |  | 0.34000 | 0.28000 | 0.26000 | 0.31000 | 0.38000 | 0.28000 |
| mmlu_college_computer_science:mc:olmo3base | accuracy:logprob | higher | 0.33000 | 0.34000 |  | 0.25000 | 0.33000 | 0.29000 |  | 0.20000 | 0.30000 |  | 0.27000 | 0.42000 | 0.28000 | 0.31000 | 0.33000 | 0.45000 |
| mmlu_college_computer_science:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.3900 | 1.2904 |  | 1.3901 | 1.2956 | 1.2673 |  | 1.2095 | 1.2405 |  | 1.1881 | 1.1470 | 1.1521 | 1.1128 | 1.1425 | 1.0822 |
| mmlu_college_computer_science:rc:olmo3base | accuracy:logprob | higher | 0.28000 | 0.28000 |  | 0.31000 | 0.31000 | 0.28000 |  | 0.30000 | 0.30000 |  | 0.30000 | 0.33000 | 0.31000 | 0.32000 | 0.27000 | 0.33000 |
| mmlu_college_mathematics:mc:olmo3base | accuracy:logprob | higher | 0.29000 | 0.22000 |  | 0.30000 | 0.34000 | 0.28000 |  | 0.21000 | 0.27000 |  | 0.37000 | 0.26000 | 0.26000 | 0.28000 | 0.32000 | 0.35000 |
| mmlu_college_mathematics:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.6901 | 1.6356 |  | 1.7111 | 1.6394 | 1.5670 |  | 1.5022 | 1.5344 |  | 1.4351 | 1.4554 | 1.3777 | 1.3208 | 1.3596 | 1.3191 |
| mmlu_college_mathematics:rc:olmo3base | accuracy:logprob | higher | 0.19000 | 0.14000 |  | 0.21000 | 0.20000 | 0.16000 |  | 0.20000 | 0.19000 |  | 0.22000 | 0.17000 | 0.23000 | 0.25000 | 0.20000 | 0.18000 |
| mmlu_college_medicine:mc:olmo3base | accuracy:logprob | higher | 0.32370 | 0.21965 |  | 0.20809 | 0.24277 | 0.20809 |  | 0.24277 | 0.22543 |  | 0.21965 | 0.23699 | 0.22543 | 0.24277 | 0.29480 | 0.41618 |
| mmlu_college_medicine:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.97960 | 0.92077 |  | 0.97128 | 0.93674 | 0.87030 |  | 0.85543 | 0.87615 |  | 0.82191 | 0.82083 | 0.81070 | 0.80183 | 0.76246 | 0.78623 |
| mmlu_college_medicine:rc:olmo3base | accuracy:logprob | higher | 0.28902 | 0.31214 |  | 0.28902 | 0.28902 | 0.32370 |  | 0.35260 | 0.34104 |  | 0.33526 | 0.37572 | 0.38150 | 0.37572 | 0.41618 | 0.36994 |
| mmlu_college_physics:mc:olmo3base | accuracy:logprob | higher | 0.27451 | 0.22549 |  | 0.23529 | 0.25490 | 0.28431 |  | 0.20588 | 0.20588 |  | 0.21569 | 0.17647 | 0.17647 | 0.22549 | 0.20588 | 0.34314 |
| mmlu_college_physics:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.5796 | 1.4909 |  | 1.6309 | 1.5484 | 1.4348 |  | 1.3644 | 1.4677 |  | 1.3024 | 1.3099 | 1.2502 | 1.2703 | 1.1483 | 1.1829 |
| mmlu_college_physics:rc:olmo3base | accuracy:logprob | higher | 0.21569 | 0.21569 |  | 0.18627 | 0.20588 | 0.25490 |  | 0.23529 | 0.23529 |  | 0.25490 | 0.20588 | 0.28431 | 0.24510 | 0.30392 | 0.28431 |
| mmlu_computer_security:mc:olmo3base | accuracy:logprob | higher | 0.25000 | 0.33000 |  | 0.26000 | 0.28000 | 0.20000 |  | 0.29000 | 0.29000 |  | 0.30000 | 0.29000 | 0.33000 | 0.30000 | 0.29000 | 0.57000 |
| mmlu_computer_security:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.1848 | 1.1447 |  | 1.1244 | 1.1534 | 1.0824 |  | 1.0261 | 1.0329 |  | 0.98815 | 0.93285 | 0.96471 | 0.91398 | 0.88166 | 0.88958 |
| mmlu_computer_security:rc:olmo3base | accuracy:logprob | higher | 0.42000 | 0.40000 |  | 0.37000 | 0.41000 | 0.44000 |  | 0.47000 | 0.47000 |  | 0.47000 | 0.51000 | 0.42000 | 0.54000 | 0.47000 | 0.47000 |
| mmlu_conceptual_physics:mc:olmo3base | accuracy:logprob | higher | 0.25957 | 0.29787 |  | 0.30638 | 0.19149 | 0.23404 |  | 0.26383 | 0.27660 |  | 0.26383 | 0.24681 | 0.28936 | 0.29362 | 0.35319 | 0.32766 |
| mmlu_conceptual_physics:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.96687 | 0.91776 |  | 0.92280 | 0.90953 | 0.84176 |  | 0.80986 | 0.84127 |  | 0.80529 | 0.79027 | 0.75727 | 0.73817 | 0.71759 | 0.71761 |
| mmlu_conceptual_physics:rc:olmo3base | accuracy:logprob | higher | 0.35745 | 0.40851 |  | 0.41277 | 0.41702 | 0.39574 |  | 0.44255 | 0.40426 |  | 0.42553 | 0.44255 | 0.45957 | 0.48936 | 0.48936 | 0.51489 |
| mmlu_econometrics:mc:olmo3base | accuracy:logprob | higher | 0.28070 | 0.29825 |  | 0.26316 | 0.19298 | 0.21053 |  | 0.25439 | 0.22807 |  | 0.21930 | 0.25439 | 0.23684 | 0.22807 | 0.28947 | 0.28070 |
| mmlu_econometrics:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.81179 | 0.75766 |  | 0.77746 | 0.75284 | 0.70272 |  | 0.66793 | 0.69785 |  | 0.62809 | 0.62435 | 0.61189 | 0.58231 | 0.57721 | 0.57503 |
| mmlu_econometrics:rc:olmo3base | accuracy:logprob | higher | 0.28070 | 0.29825 |  | 0.28070 | 0.26316 | 0.29825 |  | 0.27193 | 0.27193 |  | 0.30702 | 0.33333 | 0.31579 | 0.30702 | 0.30702 | 0.32456 |
| mmlu_electrical_engineering:mc:olmo3base | accuracy:logprob | higher | 0.28966 | 0.20000 |  | 0.23448 | 0.24138 | 0.24828 |  | 0.23448 | 0.24828 |  | 0.22759 | 0.24828 | 0.33793 | 0.32414 | 0.37931 | 0.46207 |
| mmlu_electrical_engineering:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.3630 | 1.3366 |  | 1.3518 | 1.3994 | 1.2668 |  | 1.2296 | 1.2225 |  | 1.1515 | 1.0980 | 1.0893 | 1.0881 | 1.0110 | 1.0561 |
| mmlu_electrical_engineering:rc:olmo3base | accuracy:logprob | higher | 0.28276 | 0.24828 |  | 0.28276 | 0.29655 | 0.31034 |  | 0.32414 | 0.36552 |  | 0.35172 | 0.37931 | 0.37241 | 0.38621 | 0.40000 | 0.35172 |
| mmlu_elementary_mathematics:mc:olmo3base | accuracy:logprob | higher | 0.23280 | 0.24603 |  | 0.28307 | 0.25661 | 0.24074 |  | 0.21693 | 0.24074 |  | 0.19841 | 0.23545 | 0.26190 | 0.26455 | 0.25661 | 0.29894 |
| mmlu_elementary_mathematics:rc:bpb | bits_per_byte:bits_per_byte | lower | 2.0162 | 1.9549 |  | 2.0229 | 1.9695 | 1.7840 |  | 1.7165 | 1.7756 |  | 1.6002 | 1.5446 | 1.5753 | 1.3448 | 1.3416 | 1.3581 |
| mmlu_elementary_mathematics:rc:olmo3base | accuracy:logprob | higher | 0.29894 | 0.30159 |  | 0.29365 | 0.31746 | 0.35185 |  | 0.35714 | 0.35185 |  | 0.43915 | 0.41270 | 0.42593 | 0.48148 | 0.48942 | 0.51323 |
| mmlu_formal_logic:mc:olmo3base | accuracy:logprob | higher | 0.31746 | 0.19841 |  | 0.19048 | 0.27778 | 0.25397 |  | 0.22222 | 0.21429 |  | 0.19841 | 0.23810 | 0.32540 | 0.29365 | 0.25397 | 0.35714 |
| mmlu_formal_logic:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.6101 | 1.4779 |  | 1.5396 | 1.4232 | 1.2222 |  | 1.2755 | 1.3154 |  | 1.1000 | 1.0813 | 1.0152 | 0.94198 | 0.90690 | 0.88736 |
| mmlu_formal_logic:rc:olmo3base | accuracy:logprob | higher | 0.27778 | 0.26984 |  | 0.34921 | 0.29365 | 0.34921 |  | 0.33333 | 0.24603 |  | 0.34127 | 0.32540 | 0.34921 | 0.37302 | 0.29365 | 0.34921 |
| mmlu_global_facts:mc:olmo3base | accuracy:logprob | higher | 0.19000 | 0.24000 |  | 0.19000 | 0.32000 | 0.35000 |  | 0.28000 | 0.21000 |  | 0.21000 | 0.26000 | 0.35000 | 0.35000 | 0.28000 | 0.25000 |
| mmlu_global_facts:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.6812 | 1.6239 |  | 1.6424 | 1.6229 | 1.5929 |  | 1.5484 | 1.5370 |  | 1.5234 | 1.5002 | 1.5244 | 1.4765 | 1.4242 | 1.4550 |
| mmlu_global_facts:rc:olmo3base | accuracy:logprob | higher | 0.25000 | 0.23000 |  | 0.27000 | 0.29000 | 0.24000 |  | 0.35000 | 0.30000 |  | 0.31000 | 0.32000 | 0.35000 | 0.36000 | 0.40000 | 0.39000 |
| mmlu_high_school_biology:mc:olmo3base | accuracy:logprob | higher | 0.20645 | 0.20645 |  | 0.20000 | 0.18710 | 0.30645 |  | 0.18387 | 0.24194 |  | 0.22903 | 0.23226 | 0.24194 | 0.28065 | 0.21613 | 0.40645 |
| mmlu_high_school_biology:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.88401 | 0.84615 |  | 0.84783 | 0.85048 | 0.75214 |  | 0.74869 | 0.75002 |  | 0.72242 | 0.69052 | 0.71115 | 0.65122 | 0.64618 | 0.63437 |
| mmlu_high_school_biology:rc:olmo3base | accuracy:logprob | higher | 0.33548 | 0.36129 |  | 0.36129 | 0.34839 | 0.41935 |  | 0.41935 | 0.40000 |  | 0.43548 | 0.44516 | 0.48710 | 0.51290 | 0.51935 | 0.51613 |
| mmlu_high_school_chemistry:mc:olmo3base | accuracy:logprob | higher | 0.26108 | 0.19212 |  | 0.17734 | 0.16749 | 0.26601 |  | 0.22167 | 0.22660 |  | 0.14778 | 0.24138 | 0.27586 | 0.22167 | 0.24631 | 0.34483 |
| mmlu_high_school_chemistry:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.3823 | 1.3841 |  | 1.3993 | 1.3678 | 1.2925 |  | 1.2374 | 1.2606 |  | 1.2025 | 1.1892 | 1.2046 | 1.1544 | 1.1319 | 1.1255 |
| mmlu_high_school_chemistry:rc:olmo3base | accuracy:logprob | higher | 0.22660 | 0.23645 |  | 0.20690 | 0.25616 | 0.23153 |  | 0.25616 | 0.22660 |  | 0.23645 | 0.26601 | 0.25616 | 0.28079 | 0.27094 | 0.30049 |
| mmlu_high_school_computer_science:mc:olmo3base | accuracy:logprob | higher | 0.24000 | 0.30000 |  | 0.29000 | 0.24000 | 0.21000 |  | 0.26000 | 0.25000 |  | 0.31000 | 0.33000 | 0.32000 | 0.30000 | 0.25000 | 0.29000 |
| mmlu_high_school_computer_science:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.2242 | 1.1730 |  | 1.2338 | 1.2254 | 1.0489 |  | 1.0071 | 1.0852 |  | 0.94825 | 0.91272 | 0.90810 | 0.88183 | 0.78934 | 0.82371 |
| mmlu_high_school_computer_science:rc:olmo3base | accuracy:logprob | higher | 0.40000 | 0.33000 |  | 0.33000 | 0.38000 | 0.37000 |  | 0.42000 | 0.41000 |  | 0.37000 | 0.44000 | 0.38000 | 0.44000 | 0.42000 | 0.41000 |
| mmlu_high_school_european_history:mc:olmo3base | accuracy:logprob | higher | 0.29091 | 0.23636 |  | 0.23030 | 0.21212 | 0.26667 |  | 0.22424 | 0.25455 |  | 0.25455 | 0.25455 | 0.27273 | 0.25455 | 0.24848 | 0.44848 |
| mmlu_high_school_european_history:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.79173 | 0.78099 |  | 0.78795 | 0.77377 | 0.70920 |  | 0.71736 | 0.71379 |  | 0.67386 | 0.66172 | 0.65838 | 0.63131 | 0.63355 | 0.62159 |
| mmlu_high_school_european_history:rc:olmo3base | accuracy:logprob | higher | 0.40606 | 0.40000 |  | 0.36970 | 0.36364 | 0.43030 |  | 0.38182 | 0.42424 |  | 0.37576 | 0.40000 | 0.46667 | 0.46061 | 0.51515 | 0.50303 |
| mmlu_high_school_geography:mc:olmo3base | accuracy:logprob | higher | 0.20202 | 0.21212 |  | 0.17677 | 0.18687 | 0.33838 |  | 0.18182 | 0.24242 |  | 0.18687 | 0.25253 | 0.28788 | 0.31818 | 0.30808 | 0.43939 |
| mmlu_high_school_geography:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.0482 | 0.99395 |  | 1.0159 | 0.98629 | 0.94558 |  | 0.90515 | 0.91324 |  | 0.86480 | 0.85395 | 0.84182 | 0.81092 | 0.81630 | 0.78078 |
| mmlu_high_school_geography:rc:olmo3base | accuracy:logprob | higher | 0.35859 | 0.35859 |  | 0.36364 | 0.37879 | 0.38889 |  | 0.41414 | 0.43434 |  | 0.42424 | 0.46465 | 0.43939 | 0.41919 | 0.44444 | 0.46970 |
| mmlu_high_school_government_and_politics:mc:olmo3base | accuracy:logprob | higher | 0.25907 | 0.27979 |  | 0.19171 | 0.25389 | 0.34197 |  | 0.18653 | 0.20207 |  | 0.20725 | 0.26425 | 0.29016 | 0.28497 | 0.28497 | 0.44560 |
| mmlu_high_school_government_and_politics:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.72093 | 0.66943 |  | 0.70215 | 0.69273 | 0.63438 |  | 0.61238 | 0.63078 |  | 0.58235 | 0.57895 | 0.56880 | 0.55636 | 0.53658 | 0.53244 |
| mmlu_high_school_government_and_politics:rc:olmo3base | accuracy:logprob | higher | 0.36269 | 0.38342 |  | 0.38860 | 0.35233 | 0.38860 |  | 0.38860 | 0.36269 |  | 0.42487 | 0.44560 | 0.46114 | 0.45078 | 0.46632 | 0.45596 |
| mmlu_high_school_macroeconomics:mc:olmo3base | accuracy:logprob | higher | 0.33333 | 0.30769 |  | 0.21538 | 0.32564 | 0.30256 |  | 0.20000 | 0.26923 |  | 0.21795 | 0.23077 | 0.22821 | 0.33077 | 0.33077 | 0.34872 |
| mmlu_high_school_macroeconomics:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.88888 | 0.86320 |  | 0.88039 | 0.87645 | 0.80382 |  | 0.78401 | 0.79131 |  | 0.75506 | 0.74166 | 0.72222 | 0.71102 | 0.66894 | 0.67695 |
| mmlu_high_school_macroeconomics:rc:olmo3base | accuracy:logprob | higher | 0.30513 | 0.32821 |  | 0.32308 | 0.30513 | 0.32564 |  | 0.35385 | 0.31026 |  | 0.33846 | 0.35897 | 0.35897 | 0.35897 | 0.36410 | 0.37692 |
| mmlu_high_school_mathematics:mc:olmo3base | accuracy:logprob | higher | 0.28519 | 0.27037 |  | 0.24815 | 0.23704 | 0.27407 |  | 0.25556 | 0.27778 |  | 0.25185 | 0.31481 | 0.21852 | 0.26667 | 0.25556 | 0.24444 |
| mmlu_high_school_mathematics:rc:bpb | bits_per_byte:bits_per_byte | lower | 2.5635 | 2.6288 |  | 2.5851 | 2.5576 | 2.4764 |  | 2.4583 | 2.5052 |  | 2.4069 | 2.4032 | 2.3182 | 2.2839 | 2.2658 | 2.2464 |
| mmlu_high_school_mathematics:rc:olmo3base | accuracy:logprob | higher | 0.16667 | 0.17778 |  | 0.20370 | 0.16667 | 0.16667 |  | 0.19259 | 0.20370 |  | 0.19259 | 0.23333 | 0.25185 | 0.21852 | 0.24815 | 0.22963 |
| mmlu_high_school_microeconomics:mc:olmo3base | accuracy:logprob | higher | 0.27731 | 0.18908 |  | 0.20168 | 0.28992 | 0.19328 |  | 0.22269 | 0.18908 |  | 0.23109 | 0.19748 | 0.23109 | 0.20588 | 0.24790 | 0.40336 |
| mmlu_high_school_microeconomics:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.95408 | 0.90169 |  | 0.90692 | 0.88710 | 0.82683 |  | 0.77568 | 0.79800 |  | 0.74270 | 0.74311 | 0.73778 | 0.69277 | 0.67330 | 0.68582 |
| mmlu_high_school_microeconomics:rc:olmo3base | accuracy:logprob | higher | 0.33613 | 0.34454 |  | 0.34874 | 0.35294 | 0.36975 |  | 0.41597 | 0.38655 |  | 0.40336 | 0.38235 | 0.44958 | 0.45378 | 0.45378 | 0.44958 |
| mmlu_high_school_physics:mc:olmo3base | accuracy:logprob | higher | 0.31126 | 0.21854 |  | 0.26490 | 0.21854 | 0.29801 |  | 0.20530 | 0.19868 |  | 0.23179 | 0.31126 | 0.23841 | 0.29139 | 0.26490 | 0.25166 |
| mmlu_high_school_physics:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.3913 | 1.3342 |  | 1.3753 | 1.3354 | 1.2778 |  | 1.2098 | 1.2473 |  | 1.2106 | 1.2024 | 1.1866 | 1.2138 | 1.1273 | 1.1451 |
| mmlu_high_school_physics:rc:olmo3base | accuracy:logprob | higher | 0.21192 | 0.25166 |  | 0.23841 | 0.27152 | 0.25166 |  | 0.26490 | 0.27152 |  | 0.31126 | 0.29801 | 0.27815 | 0.29139 | 0.30464 | 0.29139 |
| mmlu_high_school_psychology:mc:olmo3base | accuracy:logprob | higher | 0.25688 | 0.23853 |  | 0.18716 | 0.25505 | 0.24220 |  | 0.19266 | 0.22936 |  | 0.21468 | 0.21651 | 0.25688 | 0.23486 | 0.25505 | 0.46789 |
| mmlu_high_school_psychology:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.84796 | 0.79688 |  | 0.81178 | 0.80955 | 0.70078 |  | 0.67816 | 0.68754 |  | 0.62955 | 0.60840 | 0.59981 | 0.60665 | 0.55580 | 0.55509 |
| mmlu_high_school_psychology:rc:olmo3base | accuracy:logprob | higher | 0.42752 | 0.44220 |  | 0.44220 | 0.44954 | 0.50642 |  | 0.52477 | 0.50459 |  | 0.54312 | 0.55229 | 0.57982 | 0.58716 | 0.60183 | 0.62018 |
| mmlu_high_school_statistics:mc:olmo3base | accuracy:logprob | higher | 0.47685 | 0.38889 |  | 0.21759 | 0.43056 | 0.43056 |  | 0.20370 | 0.31944 |  | 0.20833 | 0.44444 | 0.22222 | 0.29630 | 0.31944 | 0.35648 |
| mmlu_high_school_statistics:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.2394 | 1.1874 |  | 1.2482 | 1.2541 | 1.1561 |  | 1.1062 | 1.1313 |  | 1.0645 | 1.0758 | 1.0699 | 1.0439 | 1.0322 | 1.0122 |
| mmlu_high_school_statistics:rc:olmo3base | accuracy:logprob | higher | 0.33796 | 0.30093 |  | 0.30093 | 0.25926 | 0.30556 |  | 0.32407 | 0.30093 |  | 0.29630 | 0.31019 | 0.31019 | 0.33333 | 0.37500 | 0.35648 |
| mmlu_high_school_us_history:mc:olmo3base | accuracy:logprob | higher | 0.26961 | 0.22059 |  | 0.23039 | 0.31373 | 0.22059 |  | 0.21569 | 0.27451 |  | 0.21078 | 0.27941 | 0.25000 | 0.28431 | 0.25490 | 0.40196 |
| mmlu_high_school_us_history:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.88961 | 0.84650 |  | 0.85871 | 0.86255 | 0.79141 |  | 0.77237 | 0.78217 |  | 0.74385 | 0.72117 | 0.72329 | 0.69882 | 0.67903 | 0.68581 |
| mmlu_high_school_us_history:rc:olmo3base | accuracy:logprob | higher | 0.27941 | 0.31373 |  | 0.30392 | 0.31373 | 0.36275 |  | 0.36765 | 0.35294 |  | 0.36275 | 0.35784 | 0.36765 | 0.37255 | 0.42647 | 0.40196 |
| mmlu_high_school_world_history:mc:olmo3base | accuracy:logprob | higher | 0.21941 | 0.22785 |  | 0.24473 | 0.23207 | 0.24473 |  | 0.30380 | 0.23207 |  | 0.26160 | 0.26160 | 0.23207 | 0.23629 | 0.27426 | 0.41772 |
| mmlu_high_school_world_history:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.90588 | 0.86538 |  | 0.87662 | 0.86214 | 0.81466 |  | 0.78892 | 0.82257 |  | 0.77602 | 0.75727 | 0.76473 | 0.74353 | 0.71975 | 0.72354 |
| mmlu_high_school_world_history:rc:olmo3base | accuracy:logprob | higher | 0.24473 | 0.29114 |  | 0.31224 | 0.27848 | 0.29114 |  | 0.31224 | 0.28270 |  | 0.32068 | 0.32911 | 0.34599 | 0.34177 | 0.34599 | 0.38819 |
| mmlu_human_aging:mc:olmo3base | accuracy:logprob | higher | 0.29596 | 0.33632 |  | 0.30045 | 0.22422 | 0.17489 |  | 0.30493 | 0.35426 |  | 0.28251 | 0.20179 | 0.32287 | 0.21076 | 0.21973 | 0.39910 |
| mmlu_human_aging:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.93369 | 0.91535 |  | 0.92644 | 0.91770 | 0.83982 |  | 0.81866 | 0.84330 |  | 0.75548 | 0.74638 | 0.76440 | 0.72388 | 0.71458 | 0.70631 |
| mmlu_human_aging:rc:olmo3base | accuracy:logprob | higher | 0.36771 | 0.36323 |  | 0.35874 | 0.35426 | 0.40359 |  | 0.38117 | 0.39462 |  | 0.42601 | 0.43498 | 0.43498 | 0.47085 | 0.45740 | 0.48430 |
| mmlu_human_sexuality:mc:olmo3base | accuracy:logprob | higher | 0.23664 | 0.29771 |  | 0.26718 | 0.21374 | 0.26718 |  | 0.26718 | 0.25954 |  | 0.24427 | 0.20611 | 0.25191 | 0.34351 | 0.38931 | 0.45038 |
| mmlu_human_sexuality:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.0968 | 1.0421 |  | 1.0960 | 1.0855 | 1.0158 |  | 0.95057 | 0.96670 |  | 0.89816 | 0.87579 | 0.87342 | 0.84191 | 0.83418 | 0.83736 |
| mmlu_human_sexuality:rc:olmo3base | accuracy:logprob | higher | 0.38931 | 0.39695 |  | 0.41985 | 0.38931 | 0.39695 |  | 0.42748 | 0.41221 |  | 0.41985 | 0.42748 | 0.44275 | 0.46565 | 0.46565 | 0.44275 |
| mmlu_international_law:mc:olmo3base | accuracy:logprob | higher | 0.23140 | 0.21488 |  | 0.28099 | 0.28926 | 0.23967 |  | 0.24793 | 0.23140 |  | 0.23140 | 0.33884 | 0.37190 | 0.33058 | 0.28099 | 0.55372 |
| mmlu_international_law:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.74297 | 0.71907 |  | 0.72104 | 0.72930 | 0.68135 |  | 0.66113 | 0.67501 |  | 0.63673 | 0.63070 | 0.62141 | 0.60799 | 0.60012 | 0.58911 |
| mmlu_international_law:rc:olmo3base | accuracy:logprob | higher | 0.30579 | 0.27273 |  | 0.28099 | 0.30579 | 0.31405 |  | 0.34711 | 0.32231 |  | 0.36364 | 0.36364 | 0.37190 | 0.37190 | 0.37190 | 0.38017 |
| mmlu_jurisprudence:mc:olmo3base | accuracy:logprob | higher | 0.24074 | 0.29630 |  | 0.26852 | 0.22222 | 0.24074 |  | 0.25000 | 0.28704 |  | 0.31481 | 0.26852 | 0.37037 | 0.26852 | 0.27778 | 0.35185 |
| mmlu_jurisprudence:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.0030 | 0.96758 |  | 0.97220 | 0.97827 | 0.91781 |  | 0.87585 | 0.91410 |  | 0.86677 | 0.81070 | 0.82557 | 0.80118 | 0.79069 | 0.76836 |
| mmlu_jurisprudence:rc:olmo3base | accuracy:logprob | higher | 0.26852 | 0.25926 |  | 0.28704 | 0.25926 | 0.27778 |  | 0.32407 | 0.30556 |  | 0.34259 | 0.35185 | 0.34259 | 0.37037 | 0.37037 | 0.37963 |
| mmlu_logical_fallacies:mc:olmo3base | accuracy:logprob | higher | 0.21472 | 0.30675 |  | 0.22086 | 0.23313 | 0.23313 |  | 0.23313 | 0.26994 |  | 0.19018 | 0.33129 | 0.20245 | 0.23926 | 0.26994 | 0.37423 |
| mmlu_logical_fallacies:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.0052 | 0.94603 |  | 0.97315 | 0.99510 | 0.87816 |  | 0.85952 | 0.87733 |  | 0.79543 | 0.76561 | 0.77677 | 0.74324 | 0.71143 | 0.72901 |
| mmlu_logical_fallacies:rc:olmo3base | accuracy:logprob | higher | 0.26380 | 0.29448 |  | 0.31902 | 0.25767 | 0.33129 |  | 0.33742 | 0.30061 |  | 0.35583 | 0.37423 | 0.39877 | 0.40491 | 0.39877 | 0.40491 |
| mmlu_machine_learning:mc:olmo3base | accuracy:logprob | higher | 0.31250 | 0.30357 |  | 0.23214 | 0.25000 | 0.25000 |  | 0.27679 | 0.32143 |  | 0.31250 | 0.21429 | 0.31250 | 0.33036 | 0.30357 | 0.35714 |
| mmlu_machine_learning:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.5272 | 1.4892 |  | 1.3948 | 1.5939 | 1.5750 |  | 1.4081 | 1.4696 |  | 1.3717 | 1.2322 | 1.2256 | 1.2105 | 1.1855 | 1.2314 |
| mmlu_machine_learning:rc:olmo3base | accuracy:logprob | higher | 0.23214 | 0.29464 |  | 0.19643 | 0.24107 | 0.25000 |  | 0.24107 | 0.30357 |  | 0.26786 | 0.24107 | 0.27679 | 0.28571 | 0.25000 | 0.27679 |
| mmlu_management:mc:olmo3base | accuracy:logprob | higher | 0.37864 | 0.17476 |  | 0.20388 | 0.26214 | 0.31068 |  | 0.16505 | 0.17476 |  | 0.23301 | 0.22330 | 0.22330 | 0.29126 | 0.18447 | 0.48544 |
| mmlu_management:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.0274 | 0.96280 |  | 0.99168 | 0.98058 | 0.91053 |  | 0.88013 | 0.90138 |  | 0.84427 | 0.84718 | 0.82072 | 0.77735 | 0.76931 | 0.75936 |
| mmlu_management:rc:olmo3base | accuracy:logprob | higher | 0.43689 | 0.46602 |  | 0.51456 | 0.46602 | 0.46602 |  | 0.46602 | 0.50485 |  | 0.53398 | 0.48544 | 0.51456 | 0.54369 | 0.61165 | 0.59223 |
| mmlu_marketing:mc:olmo3base | accuracy:logprob | higher | 0.27778 | 0.27778 |  | 0.28205 | 0.24359 | 0.22222 |  | 0.29060 | 0.23504 |  | 0.29487 | 0.28632 | 0.28205 | 0.26496 | 0.29487 | 0.52137 |
| mmlu_marketing:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.91714 | 0.87592 |  | 0.89112 | 0.88939 | 0.80803 |  | 0.79614 | 0.78527 |  | 0.74566 | 0.73445 | 0.73888 | 0.70036 | 0.66879 | 0.67678 |
| mmlu_marketing:rc:olmo3base | accuracy:logprob | higher | 0.42735 | 0.45726 |  | 0.42735 | 0.42308 | 0.50000 |  | 0.49145 | 0.52564 |  | 0.51709 | 0.55556 | 0.54701 | 0.56410 | 0.62393 | 0.59402 |
| mmlu_medical_genetics:mc:olmo3base | accuracy:logprob | higher | 0.28000 | 0.26000 |  | 0.28000 | 0.19000 | 0.27000 |  | 0.30000 | 0.35000 |  | 0.31000 | 0.28000 | 0.23000 | 0.25000 | 0.34000 | 0.49000 |
| mmlu_medical_genetics:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.1351 | 1.1101 |  | 1.1120 | 1.0866 | 0.96803 |  | 0.94167 | 0.96078 |  | 0.91965 | 0.88666 | 0.87539 | 0.83094 | 0.83398 | 0.84677 |
| mmlu_medical_genetics:rc:olmo3base | accuracy:logprob | higher | 0.37000 | 0.36000 |  | 0.39000 | 0.37000 | 0.48000 |  | 0.49000 | 0.47000 |  | 0.46000 | 0.50000 | 0.49000 | 0.50000 | 0.48000 | 0.49000 |
| mmlu_miscellaneous:mc:olmo3base | accuracy:logprob | higher | 0.23499 | 0.27714 |  | 0.25160 | 0.25543 | 0.24393 |  | 0.24010 | 0.26820 |  | 0.28352 | 0.25543 | 0.26181 | 0.23755 | 0.28991 | 0.44572 |
| mmlu_miscellaneous:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.1486 | 1.0703 |  | 1.0936 | 1.0860 | 0.96605 |  | 0.89771 | 0.94108 |  | 0.85586 | 0.81006 | 0.81533 | 0.76191 | 0.74580 | 0.76249 |
| mmlu_miscellaneous:rc:olmo3base | accuracy:logprob | higher | 0.40358 | 0.42018 |  | 0.43934 | 0.42529 | 0.51086 |  | 0.51469 | 0.50702 |  | 0.56066 | 0.56705 | 0.56577 | 0.61941 | 0.63091 | 0.60281 |
| mmlu_moral_disputes:mc:olmo3base | accuracy:logprob | higher | 0.19942 | 0.25434 |  | 0.26590 | 0.21676 | 0.20809 |  | 0.26012 | 0.25723 |  | 0.27457 | 0.26012 | 0.25434 | 0.34104 | 0.32081 | 0.40462 |
| mmlu_moral_disputes:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.94114 | 0.88644 |  | 0.90745 | 0.92761 | 0.86384 |  | 0.84462 | 0.85170 |  | 0.80591 | 0.79261 | 0.79625 | 0.76159 | 0.76311 | 0.75739 |
| mmlu_moral_disputes:rc:olmo3base | accuracy:logprob | higher | 0.24855 | 0.28035 |  | 0.25145 | 0.24855 | 0.28613 |  | 0.27746 | 0.28035 |  | 0.28613 | 0.32081 | 0.32370 | 0.33815 | 0.32370 | 0.33237 |
| mmlu_moral_scenarios:mc:olmo3base | accuracy:logprob | higher | 0.27933 | 0.23352 |  | 0.24581 | 0.24916 | 0.24358 |  | 0.25251 | 0.25922 |  | 0.23911 | 0.23017 | 0.24246 | 0.27598 | 0.24246 | 0.27263 |
| mmlu_moral_scenarios:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.15545 | 0.14321 |  | 0.13817 | 0.14246 | 0.13925 |  | 0.15387 | 0.13559 |  | 0.13388 | 0.13781 | 0.14511 | 0.14246 | 0.13447 | 0.14947 |
| mmlu_moral_scenarios:rc:olmo3base | accuracy:logprob | higher | 0.28380 | 0.24134 |  | 0.28268 | 0.27263 | 0.25251 |  | 0.23799 | 0.23799 |  | 0.26034 | 0.25587 | 0.23799 | 0.27263 | 0.24804 | 0.28156 |
| mmlu_nutrition:mc:olmo3base | accuracy:logprob | higher | 0.24837 | 0.23203 |  | 0.23856 | 0.21242 | 0.21242 |  | 0.25817 | 0.24510 |  | 0.23203 | 0.24510 | 0.26797 | 0.29739 | 0.28758 | 0.47059 |
| mmlu_nutrition:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.92423 | 0.89405 |  | 0.91173 | 0.90329 | 0.83125 |  | 0.81543 | 0.83001 |  | 0.77941 | 0.78475 | 0.78773 | 0.75657 | 0.77064 | 0.74439 |
| mmlu_nutrition:rc:olmo3base | accuracy:logprob | higher | 0.30392 | 0.35948 |  | 0.31046 | 0.29085 | 0.33333 |  | 0.35948 | 0.38235 |  | 0.37582 | 0.38562 | 0.36928 | 0.39869 | 0.40196 | 0.39216 |
| mmlu_philosophy:mc:olmo3base | accuracy:logprob | higher | 0.22830 | 0.27331 |  | 0.22186 | 0.27974 | 0.25723 |  | 0.24116 | 0.22508 |  | 0.26688 | 0.33441 | 0.29904 | 0.26367 | 0.26688 | 0.45338 |
| mmlu_philosophy:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.0305 | 0.99954 |  | 1.0183 | 1.0240 | 0.93739 |  | 0.93400 | 0.91492 |  | 0.86452 | 0.85344 | 0.87079 | 0.85527 | 0.81987 | 0.80859 |
| mmlu_philosophy:rc:olmo3base | accuracy:logprob | higher | 0.25723 | 0.27010 |  | 0.22508 | 0.26045 | 0.28296 |  | 0.30225 | 0.30547 |  | 0.35370 | 0.35370 | 0.33441 | 0.36334 | 0.38264 | 0.37299 |
| mmlu_prehistory:mc:olmo3base | accuracy:logprob | higher | 0.22222 | 0.25309 |  | 0.22222 | 0.24074 | 0.26543 |  | 0.21605 | 0.22840 |  | 0.23457 | 0.26852 | 0.27469 | 0.29321 | 0.25926 | 0.38272 |
| mmlu_prehistory:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.1162 | 1.0490 |  | 1.0483 | 1.0739 | 0.94939 |  | 0.90872 | 0.92856 |  | 0.85707 | 0.85531 | 0.85294 | 0.84429 | 0.79999 | 0.81471 |
| mmlu_prehistory:rc:olmo3base | accuracy:logprob | higher | 0.28704 | 0.33951 |  | 0.31790 | 0.30247 | 0.34568 |  | 0.39506 | 0.37963 |  | 0.40432 | 0.39506 | 0.41358 | 0.41975 | 0.45062 | 0.43827 |
| mmlu_professional_accounting:mc:olmo3base | accuracy:logprob | higher | 0.24823 | 0.21631 |  | 0.23759 | 0.26241 | 0.23404 |  | 0.26241 | 0.21986 |  | 0.24468 | 0.24113 | 0.26241 | 0.25177 | 0.30851 | 0.28723 |
| mmlu_professional_accounting:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.3070 | 1.2800 |  | 1.2841 | 1.2591 | 1.1747 |  | 1.1879 | 1.2285 |  | 1.1881 | 1.1726 | 1.1651 | 1.1536 | 1.1277 | 1.1156 |
| mmlu_professional_accounting:rc:olmo3base | accuracy:logprob | higher | 0.25532 | 0.29078 |  | 0.26596 | 0.25532 | 0.26950 |  | 0.27305 | 0.25177 |  | 0.24823 | 0.27660 | 0.25177 | 0.24823 | 0.25177 | 0.29078 |
| mmlu_professional_law:mc:olmo3base | accuracy:logprob | higher | 0.24120 | 0.22490 |  | 0.25619 | 0.24055 | 0.23338 |  | 0.25489 | 0.24967 |  | 0.23859 | 0.25815 | 0.24967 | 0.25684 | 0.26010 | 0.29335 |
| mmlu_professional_law:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.75074 | 0.72256 |  | 0.74443 | 0.73475 | 0.68246 |  | 0.66411 | 0.67533 |  | 0.63081 | 0.61968 | 0.61892 | 0.59779 | 0.58928 | 0.58529 |
| mmlu_professional_law:rc:olmo3base | accuracy:logprob | higher | 0.26532 | 0.25619 |  | 0.26271 | 0.27575 | 0.26728 |  | 0.27314 | 0.26076 |  | 0.27836 | 0.27966 | 0.28748 | 0.28357 | 0.27314 | 0.28227 |
| mmlu_professional_medicine:mc:olmo3base | accuracy:logprob | higher | 0.44853 | 0.38603 |  | 0.26838 | 0.35662 | 0.36765 |  | 0.18750 | 0.38971 |  | 0.16176 | 0.36029 | 0.19853 | 0.18015 | 0.31618 | 0.41912 |
| mmlu_professional_medicine:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.87540 | 0.84007 |  | 0.87509 | 0.85410 | 0.77672 |  | 0.74273 | 0.78025 |  | 0.70396 | 0.68700 | 0.66473 | 0.63731 | 0.62529 | 0.63007 |
| mmlu_professional_medicine:rc:olmo3base | accuracy:logprob | higher | 0.28309 | 0.33456 |  | 0.30882 | 0.27941 | 0.29044 |  | 0.31618 | 0.30882 |  | 0.36397 | 0.34559 | 0.34926 | 0.37132 | 0.37500 | 0.38603 |
| mmlu_professional_psychology:mc:olmo3base | accuracy:logprob | higher | 0.22386 | 0.24183 |  | 0.24837 | 0.25980 | 0.22386 |  | 0.25654 | 0.25490 |  | 0.25980 | 0.23366 | 0.28922 | 0.26307 | 0.26961 | 0.35294 |
| mmlu_professional_psychology:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.97442 | 0.92980 |  | 0.95435 | 0.93853 | 0.86529 |  | 0.84531 | 0.84594 |  | 0.80595 | 0.79025 | 0.78021 | 0.75090 | 0.75071 | 0.73525 |
| mmlu_professional_psychology:rc:olmo3base | accuracy:logprob | higher | 0.30719 | 0.30229 |  | 0.30882 | 0.30065 | 0.33007 |  | 0.31699 | 0.33333 |  | 0.33497 | 0.36275 | 0.36438 | 0.37582 | 0.38562 | 0.39542 |
| mmlu_public_relations:mc:olmo3base | accuracy:logprob | higher | 0.17273 | 0.30909 |  | 0.27273 | 0.20000 | 0.29091 |  | 0.30909 | 0.27273 |  | 0.30000 | 0.21818 | 0.32727 | 0.30000 | 0.27273 | 0.37273 |
| mmlu_public_relations:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.3659 | 1.3048 |  | 1.2823 | 1.3196 | 1.2174 |  | 1.1833 | 1.1936 |  | 1.1771 | 1.1289 | 1.1199 | 1.0875 | 1.1238 | 1.1232 |
| mmlu_public_relations:rc:olmo3base | accuracy:logprob | higher | 0.29091 | 0.32727 |  | 0.27273 | 0.26364 | 0.29091 |  | 0.33636 | 0.29091 |  | 0.38182 | 0.34545 | 0.28182 | 0.33636 | 0.35455 | 0.37273 |
| mmlu_security_studies:mc:olmo3base | accuracy:logprob | higher | 0.24082 | 0.33469 |  | 0.19184 | 0.36327 | 0.17959 |  | 0.20816 | 0.31837 |  | 0.18776 | 0.29796 | 0.26531 | 0.36735 | 0.39592 | 0.46531 |
| mmlu_security_studies:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.94699 | 0.89194 |  | 0.93997 | 0.94181 | 0.91845 |  | 0.88831 | 0.87139 |  | 0.86317 | 0.86634 | 0.85876 | 0.82607 | 0.83141 | 0.82111 |
| mmlu_security_studies:rc:olmo3base | accuracy:logprob | higher | 0.23265 | 0.27755 |  | 0.25714 | 0.26122 | 0.25306 |  | 0.27755 | 0.25714 |  | 0.24490 | 0.27755 | 0.26939 | 0.27347 | 0.26122 | 0.29796 |
| mmlu_sociology:mc:olmo3base | accuracy:logprob | higher | 0.21891 | 0.28856 |  | 0.21891 | 0.29353 | 0.21891 |  | 0.21891 | 0.22886 |  | 0.24378 | 0.24876 | 0.26368 | 0.27363 | 0.31343 | 0.46766 |
| mmlu_sociology:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.90016 | 0.85943 |  | 0.87817 | 0.88673 | 0.82659 |  | 0.79826 | 0.81190 |  | 0.77531 | 0.77348 | 0.77348 | 0.74094 | 0.74143 | 0.73282 |
| mmlu_sociology:rc:olmo3base | accuracy:logprob | higher | 0.27861 | 0.34826 |  | 0.31841 | 0.29851 | 0.30846 |  | 0.31343 | 0.33333 |  | 0.33333 | 0.32836 | 0.35323 | 0.35323 | 0.36318 | 0.36318 |
| mmlu_us_foreign_policy:mc:olmo3base | accuracy:logprob | higher | 0.30000 | 0.32000 |  | 0.28000 | 0.24000 | 0.24000 |  | 0.29000 | 0.31000 |  | 0.29000 | 0.32000 | 0.35000 | 0.33000 | 0.38000 | 0.49000 |
| mmlu_us_foreign_policy:rc:bpb | bits_per_byte:bits_per_byte | lower | 0.82994 | 0.78827 |  | 0.81651 | 0.81281 | 0.75342 |  | 0.74186 | 0.73257 |  | 0.68057 | 0.66799 | 0.68184 | 0.63216 | 0.63262 | 0.61494 |
| mmlu_us_foreign_policy:rc:olmo3base | accuracy:logprob | higher | 0.33000 | 0.30000 |  | 0.35000 | 0.31000 | 0.30000 |  | 0.33000 | 0.36000 |  | 0.36000 | 0.37000 | 0.36000 | 0.41000 | 0.40000 | 0.47000 |
| mmlu_virology:mc:olmo3base | accuracy:logprob | higher | 0.26506 | 0.24699 |  | 0.29518 | 0.21687 | 0.34940 |  | 0.27711 | 0.25904 |  | 0.25904 | 0.20482 | 0.23494 | 0.33133 | 0.31928 | 0.43373 |
| mmlu_virology:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.0241 | 0.98176 |  | 1.0268 | 1.0069 | 0.95599 |  | 0.91628 | 0.93213 |  | 0.86638 | 0.88401 | 0.87357 | 0.85832 | 0.84282 | 0.84550 |
| mmlu_virology:rc:olmo3base | accuracy:logprob | higher | 0.29518 | 0.31325 |  | 0.33133 | 0.33133 | 0.35542 |  | 0.31928 | 0.31325 |  | 0.34940 | 0.36747 | 0.36747 | 0.35542 | 0.37952 | 0.37952 |
| mmlu_world_religions:mc:olmo3base | accuracy:logprob | higher | 0.32164 | 0.30409 |  | 0.30994 | 0.31579 | 0.25731 |  | 0.32749 | 0.23392 |  | 0.29240 | 0.27485 | 0.30409 | 0.26901 | 0.34503 | 0.45029 |
| mmlu_world_religions:rc:bpb | bits_per_byte:bits_per_byte | lower | 1.5225 | 1.4473 |  | 1.4724 | 1.4294 | 1.2980 |  | 1.1846 | 1.2792 |  | 1.1327 | 1.0744 | 1.0804 | 0.98235 | 0.97189 | 0.98008 |
| mmlu_world_religions:rc:olmo3base | accuracy:logprob | higher | 0.35088 | 0.38012 |  | 0.37427 | 0.36257 | 0.42690 |  | 0.46784 | 0.39766 |  | 0.48538 | 0.57310 | 0.54971 | 0.61988 | 0.61404 | 0.62573 |
| mt_mbpp:bpb:olmo3base | primary_score:average | lower | 0.78799 | 0.78677 |  | 0.80045 | 0.78852 | 0.73351 |  | 0.71657 | 0.76157 |  | 0.70424 | 0.72097 | 0.71219 | 0.70079 | 0.70247 | 0.69784 |
| mt_mbpp_bash:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.77962 | 0.73412 |  | 0.76484 | 0.74239 | 0.68466 |  | 0.64429 | 0.69763 |  | 0.61712 | 0.62431 | 0.60671 | 0.61997 | 0.61597 | 0.60546 |
| mt_mbpp_c:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.64658 | 0.64546 |  | 0.67835 | 0.65000 | 0.61338 |  | 0.57708 | 0.63920 |  | 0.55113 | 0.59096 | 0.58529 | 0.56314 | 0.57337 | 0.59887 |
| mt_mbpp_cpp:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.64820 | 0.65938 |  | 0.68019 | 0.66600 | 0.62188 |  | 0.60277 | 0.66819 |  | 0.58248 | 0.60657 | 0.58212 | 0.55430 | 0.58998 | 0.58663 |
| mt_mbpp_csharp:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.52904 | 0.52103 |  | 0.53452 | 0.54241 | 0.48060 |  | 0.47397 | 0.52708 |  | 0.46688 | 0.47516 | 0.47215 | 0.47248 | 0.44422 | 0.47692 |
| mt_mbpp_go:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.67527 | 0.66779 |  | 0.68818 | 0.68396 | 0.64014 |  | 0.62154 | 0.65535 |  | 0.61215 | 0.62719 | 0.62823 | 0.60769 | 0.63414 | 0.61087 |
| mt_mbpp_haskell:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.89741 | 0.83880 |  | 0.87793 | 0.86386 | 0.73519 |  | 0.69826 | 0.74556 |  | 0.64046 | 0.62421 | 0.61545 | 0.59640 | 0.57769 | 0.57669 |
| mt_mbpp_java:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.49153 | 0.50037 |  | 0.51781 | 0.50043 | 0.44745 |  | 0.44442 | 0.48079 |  | 0.41907 | 0.43495 | 0.42362 | 0.43694 | 0.41649 | 0.44361 |
| mt_mbpp_javascript:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.73101 | 0.72951 |  | 0.73390 | 0.72465 | 0.67844 |  | 0.67125 | 0.71626 |  | 0.66831 | 0.70761 | 0.66816 | 0.65868 | 0.65378 | 0.65957 |
| mt_mbpp_matlab:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 1.0174 | 1.0212 |  | 1.0299 | 1.0142 | 0.95916 |  | 0.93965 | 0.98072 |  | 0.94850 | 0.94862 | 0.94041 | 0.92344 | 0.94086 | 0.92389 |
| mt_mbpp_php:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.60026 | 0.58896 |  | 0.61099 | 0.58641 | 0.55626 |  | 0.53481 | 0.56436 |  | 0.51432 | 0.52766 | 0.51382 | 0.50047 | 0.51962 | 0.51095 |
| mt_mbpp_python:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 1.1767 | 1.1894 |  | 1.1939 | 1.1879 | 1.1500 |  | 1.1545 | 1.1884 |  | 1.1809 | 1.1908 | 1.1892 | 1.1699 | 1.2099 | 1.1726 |
| mt_mbpp_r:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 1.0638 | 1.0782 |  | 1.1112 | 1.0983 | 1.0329 |  | 0.99599 | 1.0469 |  | 0.97944 | 1.0260 | 1.0265 | 0.98852 | 1.0133 | 0.99474 |
| mt_mbpp_ruby:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 1.1409 | 1.1636 |  | 1.1779 | 1.1735 | 1.1256 |  | 1.1170 | 1.1711 |  | 1.1190 | 1.1476 | 1.1422 | 1.1539 | 1.1415 | 1.1118 |
| mt_mbpp_rust:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.81854 | 0.85270 |  | 0.86221 | 0.84752 | 0.79092 |  | 0.78870 | 0.81906 |  | 0.77732 | 0.78545 | 0.79154 | 0.78447 | 0.78654 | 0.75989 |
| mt_mbpp_scala:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.92568 | 0.93132 |  | 0.91223 | 0.91123 | 0.83694 |  | 0.80878 | 0.88005 |  | 0.79617 | 0.81920 | 0.81453 | 0.81441 | 0.77267 | 0.76958 |
| mt_mbpp_swift:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.61369 | 0.60994 |  | 0.60256 | 0.58702 | 0.53132 |  | 0.53459 | 0.55021 |  | 0.52110 | 0.51536 | 0.51904 | 0.50180 | 0.48901 | 0.49693 |
| mt_mbpp_typescript:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.64006 | 0.64335 |  | 0.63091 | 0.62502 | 0.58492 |  | 0.57408 | 0.61577 |  | 0.57777 | 0.60496 | 0.58827 | 0.56706 | 0.56299 | 0.56437 |
| naturalqs:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 1.4738 | 1.4101 |  | 1.4397 | 1.4292 | 1.3142 |  | 1.2500 | 1.2880 |  | 1.1875 | 1.1879 | 1.1573 | 1.1081 | 1.0863 | 1.1203 |
| naturalqs:gen:olmo3base | f1:drop_f1 | higher | 0.07386 | 0.09509 |  | 0.08339 | 0.08862 | 0.12825 |  | 0.13718 | 0.12035 |  | 0.15601 | 0.17371 | 0.17629 | 0.20304 | 0.21762 | 0.21834 |
| naturalqs:mc:olmo3base | accuracy:logprob | higher | 0.25863 | 0.24512 |  | 0.24012 | 0.25213 | 0.25063 |  | 0.25063 | 0.25213 |  | 0.24262 | 0.24012 | 0.24912 | 0.25563 | 0.25163 | 0.34167 |
| naturalqs:rc:olmo3base | accuracy:logprob | higher | 0.25213 | 0.26963 |  | 0.26913 | 0.26413 | 0.31616 |  | 0.33317 | 0.30915 |  | 0.37719 | 0.38019 | 0.39320 | 0.43522 | 0.44372 | 0.43872 |
| olmobase:easy:code:bpb | primary_score:average | lower | 0.92581 | 0.93082 |  | 0.93744 | 0.93331 | 0.88722 |  | 0.87528 | 0.91262 |  | 0.87301 | 0.89655 | 0.88933 | 0.87110 | 0.89175 | 0.87557 |
| olmobase:easy:math:bpb | primary_score:average | lower | 0.78918 | 0.75207 |  | 0.77458 | 0.76818 | 0.70521 |  | 0.67488 | 0.69955 |  | 0.64066 | 0.62896 | 0.62314 | 0.59977 | 0.58820 | 0.58491 |
| olmobase:easy:qa:bpb | primary_score:average | lower | 1.0800 | 1.0432 |  | 1.0693 | 1.0545 | 0.98870 |  | 0.95864 | 0.97621 |  | 0.92123 | 0.90231 | 0.90398 | 0.87255 | 0.85351 | 0.85149 |
| olmobase:easy:qa:rc | primary_score:average | higher | 0.48352 | 0.50487 |  | 0.49771 | 0.50462 | 0.53942 |  | 0.55333 | 0.53874 |  | 0.58986 | 0.59611 | 0.60040 | 0.62612 | 0.63658 | 0.64473 |
| olmobase:gen | primary_score:average | higher | 0.34705 | 0.37450 |  | 0.34279 | 0.37451 | 0.43339 |  | 0.45025 | 0.41537 |  | 0.50610 | 0.51870 | 0.51889 | 0.55182 | 0.57006 | 0.57200 |
| olmobase:math | primary_score:average | higher | 0.01417 | 0.01564 |  | 0.01586 | 0.01670 | 0.01997 |  | 0.02120 | 0.02104 |  | 0.02979 | 0.03030 | 0.03169 | 0.04853 | 0.05434 | 0.06234 |
| olmobase:mcqa_non_stem | primary_score:average | higher | 0.28125 | 0.27677 |  | 0.27217 | 0.27657 | 0.27379 |  | 0.27359 | 0.27550 |  | 0.27120 | 0.27381 | 0.27766 | 0.28528 | 0.29095 | 0.41932 |
| olmobase:mcqa_stem | primary_score:average | higher | 0.24197 | 0.25321 |  | 0.25237 | 0.24018 | 0.24855 |  | 0.25207 | 0.25479 |  | 0.25859 | 0.24638 | 0.23549 | 0.25779 | 0.28004 | 0.42065 |
| piqa:bpb:olmo3base | accuracy:logprob | higher | 0.67682 | 0.68716 |  | 0.68063 | 0.69097 | 0.71545 |  | 0.72797 | 0.71600 |  | 0.73232 | 0.74864 | 0.73830 | 0.75898 | 0.77530 | 0.77095 |
| piqa:mc_olmo3base | accuracy:logprob | higher | 0.50920 | 0.49850 |  | 0.49800 | 0.50110 | 0.49980 |  | 0.50060 | 0.49920 |  | 0.49790 | 0.50210 | 0.50140 | 0.50530 | 0.50200 | 0.54100 |
| piqa:rc:olmo3base | accuracy:logprob | higher | 0.67682 | 0.68770 |  | 0.68063 | 0.69097 | 0.71545 |  | 0.72797 | 0.71600 |  | 0.73286 | 0.74864 | 0.73830 | 0.75952 | 0.77530 | 0.77040 |
| qasper_yesno:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.32103 | 0.30523 |  | 0.35778 | 0.29944 | 0.30723 |  | 0.29784 | 0.30808 |  | 0.28838 | 0.28036 | 0.29100 | 0.28903 | 0.27867 | 0.27270 |
| qasper_yesno:rc:olmo3base | accuracy:logprob | higher | 0.61442 | 0.67712 |  | 0.66144 | 0.65517 | 0.61442 |  | 0.59561 | 0.53292 |  | 0.69592 | 0.65204 | 0.61442 | 0.59248 | 0.63323 | 0.71160 |
| sciq:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.70516 | 0.64335 |  | 0.66578 | 0.66306 | 0.54966 |  | 0.51393 | 0.55169 |  | 0.47992 | 0.44106 | 0.45112 | 0.42679 | 0.39916 | 0.39737 |
| sciq:mc:olmo3base | accuracy:logprob | higher | 0.23700 | 0.26100 |  | 0.25500 | 0.26500 | 0.23900 |  | 0.25600 | 0.24800 |  | 0.27500 | 0.25700 | 0.22600 | 0.28000 | 0.31100 | 0.68000 |
| sciq:rc:olmo3base | accuracy:logprob | higher | 0.72000 | 0.76000 |  | 0.76100 | 0.75000 | 0.79300 |  | 0.83300 | 0.81300 |  | 0.84800 | 0.87200 | 0.86700 | 0.87300 | 0.89600 | 0.88900 |
| sciriff_yesno:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.37466 | 0.30268 |  | 0.31639 | 0.26383 | 0.27063 |  | 0.23756 | 0.24801 |  | 0.22217 | 0.24145 | 0.19963 | 0.19043 | 0.18696 | 0.18084 |
| sciriff_yesno:rc:olmo3base | accuracy:logprob | higher | 0.71492 | 0.71997 |  | 0.72314 | 0.74526 | 0.73957 |  | 0.74968 | 0.74526 |  | 0.75790 | 0.71997 | 0.80784 | 0.83881 | 0.80215 | 0.80847 |
| socialiqa:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 1.1343 | 1.1264 |  | 1.1179 | 1.1162 | 1.0927 |  | 1.0543 | 1.0682 |  | 1.0272 | 1.0222 | 1.0108 | 0.99032 | 0.99427 | 0.97946 |
| socialiqa:mc_olmo3base | accuracy:logprob | higher | 0.32520 | 0.33700 |  | 0.33030 | 0.32630 | 0.33340 |  | 0.33990 | 0.32680 |  | 0.33330 | 0.32730 | 0.33960 | 0.33760 | 0.34530 | 0.47410 |
| socialiqa:rc:olmo3base | accuracy:logprob | higher | 0.44831 | 0.46162 |  | 0.44575 | 0.45241 | 0.48362 |  | 0.46725 | 0.48158 |  | 0.49335 | 0.50665 | 0.50665 | 0.54299 | 0.53838 | 0.55834 |
| squad:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 0.54817 | 0.51581 |  | 0.58740 | 0.48192 | 0.45741 |  | 0.40219 | 0.39442 |  | 0.33000 | 0.30475 | 0.33095 | 0.31200 | 0.30754 | 0.30213 |
| squad:gen:olmo3base | f1:f1 | higher | 0.42053 | 0.43601 |  | 0.26218 | 0.47236 | 0.52980 |  | 0.56406 | 0.40112 |  | 0.66766 | 0.69259 | 0.67823 | 0.70366 | 0.72686 | 0.73224 |
| squad:mc:olmo3base | accuracy:logprob | higher | 0.25498 | 0.24123 |  | 0.25118 | 0.24265 | 0.24597 |  | 0.25024 | 0.25213 |  | 0.24645 | 0.24692 | 0.23697 | 0.25118 | 0.25687 | 0.53839 |
| squad:rc:olmo3base | accuracy:logprob | higher | 0.59052 | 0.60616 |  | 0.57915 | 0.64313 | 0.69336 |  | 0.69005 | 0.69526 |  | 0.76398 | 0.80806 | 0.79573 | 0.82607 | 0.83697 | 0.84218 |
| winogrande:bpb:olmo3base | bits_per_byte:bits_per_byte | lower | 1.1755 | 1.1632 |  | 1.1728 | 1.1745 | 1.1452 |  | 1.1305 | 1.1494 |  | 1.0922 | 1.0832 | 1.0871 | 1.0656 | 1.0481 | 1.0732 |
| winogrande:rc:olmo3base | accuracy:logprob | higher | 0.58450 | 0.58900 |  | 0.58940 | 0.59810 | 0.63660 |  | 0.64740 | 0.64130 |  | 0.70510 | 0.71140 | 0.72340 | 0.75820 | 0.76850 | 0.77070 |