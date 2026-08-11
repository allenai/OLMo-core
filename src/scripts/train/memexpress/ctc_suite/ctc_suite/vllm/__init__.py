"""vLLM speed-track eval backend for the CTC suite (records/ctc-suite-scaling-plan.md §4/§8).

See ``src/scripts/eval/ctc_suite/VLLM_WIRING_NOTES.md`` for the design, the parity gate, and
current PASS/FAIL status per (task, arm). This is a PARALLEL, NON-BLOCKING track: the native
torchrun evaluators (``eval_lc_native_docchunk*.py``, dispatched by ``run_rung_eval.py``'s default
``--backend native``) remain the default and the correctness authority. A (task, arm) only becomes
vLLM-eligible for the real Stage-5 sweep after its parity check passes (see
:mod:`scripts.eval.ctc_suite.vllm.orchestrate`).
"""
