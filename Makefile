debug:
	torchrun --nproc_per_node=1 \
	src/scripts/train/lc_cont_train/OLMo2-7B-lc_anneal_tp4_synth.py \
		train \
		debug \
		--train_module.state_dict_load_opts='{flatten_optimizer_state_dict: true, strict: false}' \
		--trainer.max_duration.value=10_000_000_000 \


synth_baseline_10b_sc:
	python src/scripts/train/lc_cont_train/OLMo2-7B-lc_anneal_tp4_synth.py \
		launch \
		synth_baseline_10b_sc \
		ai2/augusta-google-1 \
		--launch.num_nodes=1 \
		--launch.priority=high \
		--launch.workspace=ai2/long-contexts \
		--train_module.state_dict_load_opts='{flatten_optimizer_state_dict: true, strict: false}' \
		--trainer.max_duration.value=10_000_000_000 \

synth_baseline_10b_lc:
	python src/scripts/train/lc_cont_train/OLMo2-7B-lc_anneal_tp4_synth.py \
		launch \
		synth_baseline_10b_lc \
		ai2/augusta-google-1 \
		--launch.num_nodes=4 \
		--launch.priority=high \
		--launch.workspace=ai2/long-contexts \
		--train_module.state_dict_load_opts='{flatten_optimizer_state_dict: true, strict: false}' \
		--trainer.max_duration.value=10_000_000_000 \


synth_target_10b_synth:
	python src/scripts/train/lc_cont_train/OLMo2-7B-lc_anneal_tp4_synth.py \
		launch \
		synth_target_10b_synth \
		ai2/augusta-google-1 \
		--launch.num_nodes=4 \
		--launch.priority=high \
		--launch.workspace=ai2/long-contexts \
		--train_module.state_dict_load_opts='{flatten_optimizer_state_dict: true, strict: false}' \
		--trainer.max_duration.value=10_000_000_000 \
