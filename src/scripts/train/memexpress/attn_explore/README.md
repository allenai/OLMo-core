# attn_explore/ — mask-design experiments (local: mooney/cubbins)
Fast-proxy contradiction-n20 SFT on Qwen3-0.6B (+ Qwen3.5-0.8B) across attention structures:
dense / dilated / compressive / docchunk-mask-mix (curriculum mask-mixing is the default — see
attn-mask-mixing memory) / fast-landmark (local + Beaker twin). run_q06b_attn_explore_mooney.sbatch
is the parametrized multi-variant launcher; eval_q06b_contra_n20_native.sbatch +
eval_q4b_attn_explore_cubbins.sbatch (+ probe_train_memorization.py) are the eval side.
run_q06b_dense_contra_n20_local_mooney.sbatch is the reference local_env.sh retrofit.
