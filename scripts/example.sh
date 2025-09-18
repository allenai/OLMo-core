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

# more details: https://olmo-core.readthedocs.io/en/researcher-quick-start/guides/all_in_one_for_researchers.html

runname="olmo1B-pretrain-01"

python -m olmo_core.launch.beaker \
	--name $runname \
	--gpus 2 \
	--nodes 1 \
	--budget ai2/oe-base \
	--workspace ai2/flex2 \
	--cluster ai2/jupiter \
	--priority urgent \
	--preemptible \
	--allow-dirty \
	--weka=oe-training-default \
	-- src/examples/llm/train.py \
		$runname \
		--trainer.save_folder=/oe-training-default/sewonm/${runname} \
		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
		--trainer.callbacks.wandb='{enabled: true, entity: sewonm, project: olmo1B, name: $runname}' \
		--trainer.callbacks.lm_evaluator.enabled=false \
		--trainer.callbacks.downstream_evaluator.enabled=false \
		--trainer.no_checkpoints \
		--trainer.hard_stop='{value: 100, unit: steps}'

#	--trainer.callbacks.wandb='{enabled: true, entity=sewonm, project=olmo1B, name=$runname}' \

