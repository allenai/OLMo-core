**Debug Fields (What They Mean)**  
From `OLMo-core/src/olmo_core/nn/moe/v2/block.py:649` and `OLMo-core/src/olmo_core/nn/moe/v2/block.py:1531`.

- `all_rank_output_diffs`: sync vs no-sync output difference per rank.
- `mae`: mean absolute error for that rank’s block output.
- `rel_mae`: `mae / mean(abs(sync_output))`.
- `max_abs`: max absolute elementwise diff.

- `num_dropped`: how many routed tokens were dropped by no-sync capacity.
- `local_kept_tokens`: tokens this rank kept from its own routed tokens after drop.
- `received_after_drop`: tokens this rank received in dispatch for local experts (`sum(dispatch_splits)`).
- `combined_tokens`: tokens this rank got back after combine (`sum(combine_splits)`).
- `zero_rows_after_local_unpermute`: rows that became all-zero after final local unpermute.
- `combine_vs_allowed_sum_abs`: `sum(abs(combine_splits - allowed_splits))`.
- `combine_vs_allowed_max_abs`: `max(abs(combine_splits - allowed_splits))`.
- `used_expected_combine_layout`: whether fallback layout reconstruction was used (now `False` with hard-fail path).

From the hard-fail diagnostic `OLMo-core/src/olmo_core/nn/moe/v2/block.py:1455`.

- `requested_sum`: total tokens requested before drop (`sum(requested_splits)`).
- `allowed_sum`: total tokens allowed after drop (`sum(allowed_splits)`).
- `dispatch_recv`: same as `received_after_drop`.
- `combine_recv`: same as `combined_tokens`.
- `local_kept_x_rows`: rows reconstructed from op-reported combine metadata.
- `local_kept_x_expected_layout_rows`: rows if reconstructed by deterministic contiguous layout.
- `dispatch_max_end`, `combine_max_end`: max `(offset + split)` in each phase (useful for offset sanity).

**What To Pay Attention To First (Priority Order)**

1. `combine_vs_allowed_sum_abs` must be `0`.  
2. `combined_tokens` must equal `allowed_sum` (and equal `local_kept_tokens`).  
3. `local_kept_x_rows` must equal `local_kept_tokens`.  
4. If `num_dropped == 0`, `all_rank_output_diffs` should be near zero (only tiny fp noise).  
5. `received_after_drop` can differ from `local_kept_tokens`; that alone is not a bug.

**How to Read Your Last Error**

- `num_dropped=0` on both ranks: not a capacity/drop issue.
- Rank 0 had huge `combine_vs_allowed_*` and `combined_tokens != local_kept_tokens`.
- Rank 1 had perfect metadata.
- So the failure is at combine metadata/output consistency on rank 0, not router/top-k/drop logic.