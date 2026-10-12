# pooled_kv — eval-side and layer-wise probes of soft-token / pooled-doc KV slots

Probes that hold a trained (dense or soft-token) checkpoint fixed and ask which slot construction
reproduces full-attention answer loss. Operational state: `records/soft-kv-slot-probe-handoff.md`,
`records/pooled-doc-kv-handoff.md`, `records/pooled-doc-kv-attention.md`.

| path | role |
|---|---|
| `eval_side_slot_probe.py` (+ `oracle_meankv_probe.py`, `slot_ceiling_probe.py`, `collect_slot_results.py`, `make_slot_tracker.py`, `analyze_slot_rows.py`) | slot-construction study across the four tasks; the tracker page is rendered from `results/pooled_kv/slot_probe_results.csv` |
| `trained_parity_check.py`, `launch_trained_parity*.sh`, `run_trained_parity_local.sbatch` | eval-time CE parity of the ds64 winning arms (`records/trained-parity-winning-arms.md`); grades with `recipes/eval/legacy_native/ctc_eval` |
| `outlier_probe/` | outlier readability / realtoken / richer-slot / saliency probes (`records/outlier-*-probe.md`) |
| `landmark_probe/` | landmark tokens as the soft token (needs `convert_longctx_tasks_to_sft.py`, see note below) |
| `layer_probe/` | does pooling have to apply at every layer (`records/layer-soft-probe.md`); sbatch sweeps; `layer_probe/results.csv` is not in git |
| `pooledkv_eval_probe.py`, `output_equivalence_probe.py`, `mechanical_slot_fidelity.py`, `fit_summarizer_and_fidelity.py`, `kv_predictability_probe.py`, `flex_*.py`, `oom_bisect.py`, `bench_softtoken_throughput.py` | assorted fidelity / timing probes |
| `transfer_*.sh` | gantry weka -> S3 checkpoint staging (`ckpt_transfer_log` was not ported) |

Note: `landmark_probe/landmark_slot_probe.py` imports `convert_longctx_tasks_to_sft` from
`src/scripts/data`, which is not part of this port map; it will not import unless that module is present.
Probe result JSONs (`probe_results/`) and logs are not in git.
