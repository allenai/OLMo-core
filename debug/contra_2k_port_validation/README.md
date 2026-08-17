# contra_2k_port_validation

The end-to-end port-validation run for `ctc-eval`: reproduce a known pre-migration contradiction
@2k number (f1 0.9583, eval_size 488, `ctc-t0-contra-n20-full`) from the same checkpoint and data
through the ported eval path. The sbatch header documents the target number, its provenance, and
what each possible outcome would mean.

Lived in `run/` while the validation was live; moved here because `run/` holds the permanent
entry points (`data.sh`, `eval.sh`, `_env.sh`) and this is a one-off evidence artifact.
