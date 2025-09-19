# default command explanations:

# the first name is the name appear in beaker
# for more details, do `python -m olmo_core.launch.beaker --help`

# basically it's running `src/examples/llm/train.py`
# the first config is a run name (used for save_folder, wandb name, etc)
# for more details, `python src/examples/llm/train.py olmo1B-pretrain-01 --dry-run`

# -- trainer.load_path if you want to load from another model

# when the config is a class, we could either use a json string or set individual value
# e.g., `--trainer.hard_stop='value: 100, unit: steps'` or 
#       `--trainer.hard_stop.value=100 --trainer.hard_stop.unit=steps`

##############################################################

runname="olmoe-pretrain-01"
python -m olmo_core.launch.beaker \
  	--name $runname \
	--gpus 4 \
    	--nodes 1 \
	--weka=oe-training-default \
      	--shared-filesystem \
	--workspace ai2/flex2 \
	--cluster ai2/jupiter \
	--allow-dirty \
	--priority urgent \
	--env-secret WANDB_API_KEY=SEWONM_WANDB_API_KEY \
	-- src/scripts/train/olmoe-1B-7B.py \
       		$runname \
		--save-folder="/weka/oe-training-default/$USER/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="/weka/oe-training-default/$USER/dataset-cache" \
		--trainer.no_checkpoints \
		--trainer.hard_stop='{value: 100, unit: steps}' \
		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.project="olmoe-modular"







