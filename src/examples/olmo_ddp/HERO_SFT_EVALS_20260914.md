# Small hero SFT exports and post-training evaluations

Scope: convert both saved epochs from the six completed September 14 SFT runs;
evaluate **only epoch 2 (step 1810)**. Three LRs (1e-5, 5e-5, 1e-4) for each
EMO/non-EMO arm. No training changes or manual checkpoint deletion.

## Conversion and inference metadata

`olmoe3_hero_sft_exports.py` submits export-only workers in the allocated MoE
evaluation workspace. Source training pin is
`2610a90ced51542c10848a7d82e9534f3ef65923`. The original conversion and strict
FP32 reference checks stay frozen; metadata is installed before qualification,
checksums and atomic publication. The first final export gates conversion fan-out.

The saved training tokenizer retains all vocabulary IDs and completed-turn token
sequences. Its inference generation prompt appends exactly one `<think>` after
the assistant header. The training system prompt is retained; we do not copy a
different model's identity or tool instructions. No extra BOS token is inserted.
EOS is `[100265, 100257]`, PAD is `100277`, context length is 65536.

References: `allenai/Olmo-3-7B-Think-SFT` at
`6ff857587e040d6d523a3d5f3a56e918f5401d66` and
`allenai/Olmo-Hybrid-Think-SFT-7B` at
`79d1f7f613a9f98169e6c9f00880ab2df4383860`.

## Evaluation recipe

All four tasks use chat prompting, zero shots, temperature 0.6, top-p 0.95,
disabled top-k, one sampled response, generation seed 1234, and up to 32768 new
tokens. Stop on IM_END or end-of-text, not on `</think>`. The full raw response is
saved, but only the final answer after `</think>` is scored. Unfinished reasoning
is an empty final answer, not silently counted as a completed answer.

| Suite | Instances | Scoring |
|---|---:|---|
| Math-500 | 500 | Boxed answer, native mathematical equivalence |
| IFBench | 300 | `allenai/IFBench_test2`, prompt/instruction strict and loose |
| AlpacaEval | 805 | GPT-4.1 weighted judge; raw and length-controlled win rate vs GPT-4-1106-preview reference outputs |
| HumanEval | 164 | Original HumanEval, whole-function chat answers, isolated execution pass@1 |

AlpacaEval uses the Olmo recipe's `weighted_alpaca_eval_gpt4.1` judge, sometimes
called v3, **not** the GPT-4-Turbo judge from standard AlpacaEval 2. Label results
accordingly. HumanEval is not HumanEval+, and IFBench is not the broader multiturn
suite. These are single-sample sanity-check evaluations, not low-variance pass@1
estimates from many samples. All dataset revisions are pinned in
`olmoe3_hero_sft_tasks.py`.

The fast BF16 grouped-MoE/FLA inference profile is the same provisional profile
used for the current hero base evaluations. Strict conversion/FP32 vLLM parity
does not establish exact numerical equivalence for that faster BF16 profile.

## Automation and validation

`olmoe3_hero_sft_eval_control.py` is a separate resource-free Phobos CPU controller.
It waits for each final export, runs frozen HF/vLLM qualification, then runs a
three-example smoke of **each** new chat/scoring task on the first final model.
Only a successful smoke releases the 24 full-suite jobs. Workers are urgent,
allocated (1h minimum runtime), Jupiter/Ceres, in
`ai2/OLMo-3-moe-experiments`; normally one GPU per benchmark, two for qualification.
No checkpoint or model payload goes into a Beaker results dataset.

Durable per-stage submission intents prevent duplicate submissions; failures stay
visible and are not automatically retried indefinitely. Results and raw/final
responses go under each final export's `posttrain-evals-r1` directory. A success
receipt requires complete instance counts and successful scoring, including the
Alpaca judge, not just completion of generation.

Code executes in a mount-free, credential-free remote Modal sandbox. A known-pass
and known-fail program test sandbox behavior before model generation. Only the
evaluation host receives the Modal credentials; model-generated code does not.
Alpaca workers receive the scoped `jacobm_HERO_SFT_OPENAI_API_KEY` Beaker secret.

Local checks: metadata round-trip tests, Think extraction, full-function code
extraction, prompt/sampling checks, native evaluator mock run, all dataset counts,
real GPT-4.1 annotation request, and synthetic length-correction compatibility.
The real model's GPU chat/scoring smoke remains the gate before full fan-out.
