# records/ — standalone writeups

Experiment diagnoses, plans, handoffs and how-tos. Code lives in `recipes/` (the canonical
data → train → eval pipeline) and `research/<family>/` (experiment families); records point at both.
Results, checkpoints and weights live outside git (node-local `/data`, weka) — records say where.

Areas: **pipeline** (data, training, eval, infra traps of the canonical CTC path) ·
**token_routing** (soft/pooled tokens, dev-loss grid, learned router, ds64) · **ffn_compute** (FFN
routing, FLOP scaling, drop-CPT) · **kv_route** (learned KV allocation / flexcompute) · **landmark**
(landmark / document-chunked attention).

| record | area | status | one line |
|---|---|---|---|
| [ctc-fast-suite-eval](ctc-fast-suite-eval.md) | pipeline | current | What a full CTC-suite eval costs; current path = olmo-eval `launch_ctc_suite.py` (r256k YaRN x2 + chunked prefill); setA IID build status. |
| [v3-eval-howto](v3-eval-howto.md) | pipeline | legacy | Running the v2/v3 eval ladders through the native launcher (v3 only for contra + outlier). |
| [ctc-suite-olmo-eval-handoff](ctc-suite-olmo-eval-handoff.md) | pipeline | reference | The 22-task suite on olmo-eval `prasann/ctc-suite`, HF dataset, ready-to-post PR body. |
| [ctc-final-suite](ctc-final-suite.md) | pipeline | frozen 2026-08-12 | The frozen 22-row CTC-suite roster; contradiction scores on `contradiction_iid` only. |
| [ctc-figure-settings-unification](ctc-figure-settings-unification.md) | pipeline | audit | Settings each CTC paper figure selects, and what unifying them on one task set would cost. |
| [contradiction-train-eval-non-iid](contradiction-train-eval-non-iid.md) | pipeline | diagnosed | The contradiction eval was never IID with training data — most of the gap to the paper. |
| [contradiction-data-and-base-hygiene](contradiction-data-and-base-hygiene.md) | pipeline | running list | Silently-wrong data/base artifacts that load and train fine but give garbage; what to use instead. |
| [document-chunked-marker-embeddings](document-chunked-marker-embeddings.md) | pipeline | must-read | Qwen marker rows are bit-identical; run `recipes/data/fix_marker_embeddings.py` on every new base. |
| [n100-chunked-marker-position-bug](n100-chunked-marker-position-bug.md) | pipeline | root-caused 2026-07-14 | The n=100 chunked collapse was marker embedding × position (incl. the NORM bug), not a mask result. |
| [ctc-suite-scaling-plan](ctc-suite-scaling-plan.md) | pipeline | superseded roster | Figure-4 plan (~25 tasks, 2k–32k, full vs chunked) behind `recipes/train/research_trainer/`. |
| [ctc-local-triton-nfs-cache-hang](ctc-local-triton-nfs-cache-hang.md) | pipeline | fixed | Local trainer "hang" at the dry-run forward = FLA/Triton JIT cache on NFS; keep it on `/data`. |
| [learned-token-router](learned-token-router.md) | token_routing | active | Learned linear token router for slot-less compaction (small eval sizes, flagged inline). |
| [devloss-grid-underperformers](devloss-grid-underperformers.md) | token_routing | audit 2026-09-21 | Every dev-loss-grid cell where a keep-scheme trails full by ΔCE > 0.05, with paired SE. |
| [ds64-scaling-plan](ds64-scaling-plan.md) | token_routing | done 2026-09-16 | Short-heavy 2k–64k data-scaling campaign, soft tokens vs dense; plan + status log. |
| [ds64-handoff](ds64-handoff.md) | token_routing | handoff | "Read me first" for the ds64 campaign: state, commands, open items. |
| [ds64-overnight-2026-09-14](ds64-overnight-2026-09-14.md) | token_routing | log | Overnight orchestrator log for the ds64 wave, 09-14 → 09-15. |
| [pooled-doc-kv-attention](pooled-doc-kv-attention.md) | token_routing | research log | Train-time pooled-document KV, full-attention inference: mechanisms, probes, dead ends. |
| [pooled-doc-kv-handoff](pooled-doc-kv-handoff.md) | token_routing | handoff 2026-08-31 | Operational pick-up doc for pooled-doc-KV: what is proven, what is dead, exact commands. |
| [soft-kv-slot-probe-handoff](soft-kv-slot-probe-handoff.md) | token_routing | parity reached 2026-09-08 | Eval-side slot-construction study; header-real slots reach parity, next = training arms. |
| [landmark-slot-probe](landmark-slot-probe.md) | token_routing | no-go | Landmark tokens as the soft-token slot: control already far from FULL, so the probe is void. |
| [outlier-slot-probe](outlier-slot-probe.md) | token_routing | answered | Can a dense model read a pooled outlier from its soft-token slot? (eval-side). |
| [outlier-richer-slot-probe](outlier-richer-slot-probe.md) | token_routing | answered | Cheaper-than-training richer slots for pooled outlier: none make it readable. |
| [outlier-realtoken-parity-probe](outlier-realtoken-parity-probe.md) | token_routing | complete | Whether a real-token subset per document reaches frozen-model parity on outlier. |
| [outlier-saliency-preview-probe](outlier-saliency-preview-probe.md) | token_routing | answered | Saliency-selected real tokens and a one-layer preview vs full parity on outlier. |
| [outlier-cc00-evaltime-probe](outlier-cc00-evaltime-probe.md) | token_routing | answered | Eval-time loss parity of the ds64 `cc00` trained checkpoint. |
| [trained-parity-winning-arms](trained-parity-winning-arms.md) | token_routing | complete | Eval-time CE parity on the ds64 campaign's winning arms. |
| [softdetach-cpt-plan](softdetach-cpt-plan.md) | token_routing | screening done, CPT arms run | Soft-detached pooling for continued pretraining (`research/token_routing/softdetach/`). |
| [softdetach-cpt-crossover](softdetach-cpt-crossover.md) | token_routing | diagnosis | Why `sd20` stops being Pareto-optimal at high FLOPs, and candidate tweaks. |
| [layer-soft-probe](layer-soft-probe.md) | token_routing | eval-only probe | Does the pooled soft token need to be applied at every layer? |
| [flop-scaling-ffn-kv-plan](flop-scaling-ffn-kv-plan.md) | ffn_compute | plan | Are FFN routing and pooled-KV compute-optimal vs dense? Grid design (`research/ffn_compute/flop_scaling/`). |
| [flop-scaling-report-2026-09-02](flop-scaling-report-2026-09-02.md) | ffn_compute | report | 93-run FLOP-scaling results: FFN routing and KV soft tokens vs dense, fitted laws. |
| [ffndrop-cpt-plan](ffndrop-cpt-plan.md) | ffn_compute | active | Random null-FFN CPT before routed-FFN SFT (`research/ffn_compute/ffndrop/`). |
| [kv-route-attention-allocation](kv-route-attention-allocation.md) | kv_route | partly run | Learned KV-cache allocation: KV keep, block skip, joint budget; trainer variants `kvroute`/`flexcompute`. |
| [landmark-packing-cp-task](landmark-packing-cp-task.md) | landmark | task | Context parallelism + sequence packing together for landmark attention. |
| [landmark-sparse-decode-task](landmark-sparse-decode-task.md) | landmark | task | Make top-k landmark decode computationally sparse, not just masked. |
| [multihop-gold-routing-experiment](multihop-gold-routing-experiment.md) | landmark | proposal | Can a model use two documents that never attend to each other? |
| [free-pad-probe-is-confounded](free-pad-probe-is-confounded.md) | landmark | void | The `free_pad_repeat` probe does not measure FREE-position capacity. |
| [n100-clean-results-2026-07-14](n100-clean-results-2026-07-14.md) | landmark | results | n=100 document-chunked after the marker fix: chunked .441, hier K50 .831, full .934. |
| [soft-detach-plus-compressive-landmark](soft-detach-plus-compressive-landmark.md) | landmark | design note | Whether soft-detached pooling and compressive landmark can be combined. |
| [eval_launches/](eval_launches/README.md) | pipeline | ledger | Eval launch ledger (one YAML per checkpoint, gitignored); `pull-evals` flips jobs to done. |
