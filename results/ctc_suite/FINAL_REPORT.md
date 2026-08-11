# CTC-suite: Qwen3.5-4B — full attention vs document-chunked (with mask mixing)

**Model:** Qwen3.5-4B GDN-hybrid. **Eval:** vLLM (native chunked FlexAttention path), eval_size=500
per rung (scifact=300, obliq=data-capped). **Ladder:** token-budget rungs 2k/4k/8k/16k/32k, which
scale the number of documents per context (e.g. reorder: 12→27→57→116 docs), so the ladder is a
context-token-count (CTC) / N-scaling sweep parameterized by token budget.

**Two arms:**
- **dense** = per-task `-full` checkpoints, evaluated with full causal attention.
- **chunked** = per-task `-cmix` checkpoints (curriculum **mask-mixing** training), evaluated with
  document-chunked attention. This is the paper's *"+ mask mixing"* method — the improved chunked
  setting, **not** pure chunked — so chunked numbers here are the *best-case* chunked, and the
  full≫chunked gaps below are the residual that mask mixing cannot close.

## Headline result — reproduces the paper's CTC thesis on the 4B

The full-vs-chunked gap tracks the task's CTC complexity class, exactly as the paper predicts:

**Low-CTC / local-retrieval — chunked ≈ full (both high, gap ~flat with N):**
| task | metric | dense (2k/4k/8k) | chunked (2k/4k/8k) |
|---|---|---|---|
| niah | gold_id_f1 | 0.988 / 0.988 / 0.976 | 0.984 / 0.994 / 0.976 (→32k 0.970) |
| nq | gold_id_f1 | 0.904 / 0.930 / 0.864 | 0.964 / 0.942 / 0.896 |
| msmarco | gold_id_f1 | 0.961 / 0.939 / 0.917 | 0.897 / 0.902 / 0.868 |
| hotpotqa | gold_id_f1 | *(ckpt corrupt)* | 0.995 / 0.987 / 0.983 |
| cycle | cycle_f1 | 1.000 / 1.000 / 0.998 | 0.996 / 0.994 / 0.994 |
| rerank | mrr@10 | 0.989 / 0.971 / 0.960 | 0.980 / 0.966 / 0.950 |
| absence_gutenberg | set_f1 | 0.964 / 0.976 / 0.986 | 0.949 / 0.961 / 0.981 |
| outlier_amzn | set_f1 | 0.921 / 0.891 / 0.896 | 0.912 / 0.894 / 0.887 |

**High-CTC / global-comparison — full ≫ chunked, gap widening with N:**
| task | metric | dense (2k→) | chunked (2k→) | failure mode |
|---|---|---|---|---|
| qdmatch | pair_f1 | 0.999 flat | 0.650 / 0.760 / 0.668 / 0.547 / 0.333 | graceful decay |
| outlier (wiki) | set_f1 | 0.664 / 0.300 / 0.142 | 0.093 / 0.047 / 0.052 | graceful decay |
| textgroups | textgroups_f1 | 0.196 / 0.090 / 0.054 | 0.087 / 0.021 / 0.019 | graceful decay |
| reorder | kendall_tau | 0.747 / 0.600 / 0.262 | 0.471 / 0.214 / 0.045 | both decline (O(N²)) |
| strmatch | set_f1 | 0.999 flat | 0.003 / 0.003 / 0.000 | **mode-collapse** |

**Two distinct chunked failure modes** (both real, both on tasks that work under full attention):
- *Graceful decay* (outlier, textgroups, qdmatch): well-formed, per-example, accuracy falls as N grows.
- *Mode-collapse* (strmatch): under chunked masking the model abandons per-example reasoning and
  emits a near-constant attractor (90% identical output). strmatch = exact cross-document string
  matching, which chunking makes structurally impossible → the most extreme collapse (0.999→0.003).

**grouping / grouping_labeled** are the notable "no gap" case: chunked tracks dense *almost exactly*
(0.439/0.357/0.185/0.041 both arms). This is mask mixing doing its job — it fully closes the O(NM)
grouping gap at this N range (consistent with the paper's +0.07 mask-mixing gain on grouping).

## Paper reference-check (qualitative — axes differ)

The paper (Fig 4 / Table 2) sweeps N (doc count) directly at fixed points (N=20/50/100); we sweep a
token budget that scales N per-task. So this is a **qualitative** cross-check, not a
number-for-number reproduction:
- **Agrees:** O(N) retrieval tasks show a flat ~zero gap; O(NM)/O(N²) tasks show a widening full≫chunked
  gap; mask mixing helps but cannot close the high-CTC gap at scale — all reproduced here.
- **Absolute-number caveats:** our outlier(wiki) sits lower than the paper's (~0.05 chunked vs paper
  0.518 at N=100; our dense also lower) — attributable to task-variant/difficulty and the
  token-budget-vs-N axis, not a contradiction of the trend. Do not read the absolute cells as a
  literal match to Table 2; read the *pattern*.

## Excluded from the comparison (verified reasons, NOT chunking effects)
- **groups4, scifact** — the `-full` checkpoint **const-collapsed during training** (constant
  prediction under *full* attention: scifact/full emits `[1]` on 86% of examples). Fails the "works
  under full?" test → checkpoint issue, not a chunking finding. (scifact's `-cmix`/chunked checkpoint
  trained fine: 0.976/0.956/0.933.)
- **hotpotqa** — `-full` checkpoint corrupted on S3 (missing distcp shard). Chunked-only.
- **oolong** — no `-full` checkpoint exists. Chunked-only (0.628/0.523/0.390).
- **mathmatch, helmet_qa, helmet_summ** — near-floor in *both* arms → task too hard for this
  checkpoint, unrelated to chunking.
- **reorder-16k** — genuine model capability floor (only ~1% of responses are a valid 116-permutation;
  complete-but-wrong, not truncated). reorder 2k/4k are reliable.

## Eval-infrastructure fixes made during this run (all verified)
- **Chunked-vLLM was emitting `!!!!` garbage** on the GDN-hybrid → root-caused to
  `flex_attn_kv_block_size=32` not dividing the KV page size 528; fixed to 16 + a divisibility
  guardrail (`vllm_chunked_patch.py` / `run_vllm_eval.py`). This unblocked the entire chunked arm.
- **absence_gutenberg false 0.01** → grader required a leading `[` the chunked model drops; fixed
  `_parse_snippet_list` to reconstruct the array (recovered 0.007→0.949, ≈ dense).
- **Dense-arm retrieval artifacts** (niah 4k/8k, rerank 2k/4k read implausibly low) → recovered to
  full levels on a clean re-run with the fixed pipeline.
- **reorder truncation** at long rungs → added a `MAX_NEW_TOKENS` override (1024) after confirming
  via doc-count math; 16k then shown to be a real floor, not truncation.
- **Beaker/jupiter vLLM** was brought online for the 4B (was a hard blocker): the wall was a pip
  CUDA-toolkit incoherence, fixed with a real system toolkit + cu129 wheel (native on the 12.8 driver).

Full per-rung table: `results/ctc_suite/dense_vs_chunked_table.md`.
Grades on S3: `s3://ai2-llm/checkpoints/prasanns/_transfer/ctc_{dense,chunked}_results/<task>/rung_<N>.json`.
