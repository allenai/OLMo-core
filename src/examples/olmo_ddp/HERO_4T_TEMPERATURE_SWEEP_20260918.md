# 4T SFT temperature sweep and eval recovery

Scope: the two 4T decay → 100B MT → 100B LC → high-reasoning GPT-OSS SFT
lineages. SFT uses LR 5e-5 and two epochs, step3360. EMO is off during all
MT/LC/SFT. No new training or LR sweep is authorized by this change.

Evaluate each final checkpoint with temperatures 0, .2, .4, .6, .8, 1.0 on
Math500 (500), IFBench (300), HumanEval (164), AlpacaEval (805). This is 48
cells total: eight canonical T=.6 evaluations reused, plus 40 sweep cells.
Only temperature changes: same Think chat template, 32,768 generated-token
limit, top-p .95, one sample, seed1234, BF16 grouped/FLA inference, tokenizer,
prompt sets, scorer and GPT-4.1 Alpaca judge as the previous evaluations.
T=0 is greedy. Alpaca's API judge remains unchanged across temperatures.

Run the existing CPU-only 4T eval controller with
`HERO_SFT_TEMPERATURE_SWEEP=1`. Each model is independently gated on its final
training success receipt, successful conversion, weight/metadata integrity,
and tokenizer/chat-template checks. The previously authorized numerical-parity
waiver remains in force; this change does not claim numerical qualification.

T=.6 keeps canonical `posttrain-evals-r1` paths. Others use
`posttrain-temperature-20260918/tXX/BUNDLE`. Every worker is allocated, urgent,
one GPU, minimum runtime8h, timeout24h, autoResume. Nothing is written into
Beaker result datasets. Math/IFBench/Alpaca resume immutable saved responses;
HumanEval restarts in a fresh attempt directory with isolated Modal sandboxes.
Metrics hashes, full instance counts, overrun counts, unfinished-reasoning and
empty-final counts are recorded. Completed cells never relaunch automatically;
failed cells are reported for explicit reconciliation, not retry loops.

Recovery: timeout/OS failure in a stage tick is logged and retried at the next
poll without terminating the controller or skipping other stages. Durable
submission intents prevent duplicate launches after ambiguous API timeouts.
Existing PT/MT/LC jobs retain their old commit and canonical tracking receipts.
The one failed LC code job is retried separately with HTTP/1.1 and bounded
retry-all-errors for its Cloud SDK download; scoring/inference are unchanged.

The existing 12TB free-storage submission floor remains in place. No checkpoint
deletion, retention changes, new model copies for temperatures, or hero/PT
resumptions are part of this work.
