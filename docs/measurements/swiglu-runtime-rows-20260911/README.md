# Runtime row capacity in the forward SwiGLU kernel

Historical prototype evidence, preserved from `5bfa0618f584`. The production
implementation was subsequently integrated in `307d20590` with an explicit
`row_specialization` choice. This probe describes the original unconditional
prototype; it is not a current runtime configuration guide. Current Core keeps
the static default, while the open-instruct profiles opt into dynamic rows.

This isolated change starts at Core290d2ca, which includes eager intermediate
rounding for scoring. It makes forward `rows` a runtime scalar with Triton
`do_not_specialize`, and obtains the unchanged row stride from `tl.num_programs(0)`.
The launch grid, masks, arithmetic modes and backward kernel remain unchanged.
Using runtime rows alone would leave a second shape key: capped grid size was also
passed as the `ROW_PROGRAMS` constexpr for capacities below16,384 rows.

An RTX4090 probe used hidden952, BF16 and twelve capacities spanning1 to31,007,
including tile and grid-cap boundaries. Both eager-rounding and fused modes produced
bitwise identical outputs to the original kernel, including untouched tail rows.
Each original mode generated12 JIT misses and12 cubins; each new mode generated1
miss and1 cubin. Repeated capacity sweeps generated none. Sum of first-call wall
seconds was1.565→0.0913 for fused mode and1.067→0.0875 for eager-rounding mode.
These are local kernel diagnostics, not model throughput measurements. Repeated
CUDA-stream launch times remain approximately0.010ms at small capacities and0.205ms
at31,007 rows; this includes host launch gaps and is not isolated GPU active time.

`result.json` retains raw per-capacity timings and both source hashes. `probe.py`
expects the original source exported as `/validation/swiglu_before_runtime_rows.py`,
imports the candidate from `olmo_core`, and writes its report under `/validation`.
For a fresh run, set a new empty `TRITON_CACHE_DIR` before starting Python. Export
the baseline with `git show 290d2ca:src/olmo_core/kernels/swiglu.py`.
The probe requires Torch/Triton and one CUDA GPU; it launches no optimizer/model job.

Validation:15 GPU kernel tests pass, including a new regression requiring a shared
binary across odd/aligned capacities and both sides of the capped-grid boundary.
Existing forward/backward and eager-rounding checks remain in that suite. Active
MILES runtime pins and running Beaker images were not changed. FLA compilation is
separate and this change does not claim to remove the complete historical score cost.

The broader compiled pairwise-activation suite is not clean in this runtime:
`test_compiled_pairwise_activation` fails its exact gradient comparison on6 of
2,097,152 BF16 elements (maximum0.00390625); four architecture-specific cases skip.
The identical failure reproduces on unchanged290d2ca with the same indices and values.
That test uses `ops/swiglu_pairwise`, not the modified valid-prefix kernel. It is a
pre-existing qualification limitation, not fixed or hidden by this patch.
