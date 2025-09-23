from olmo_core.cookbook import (
    CookbookExperimentConfig,
    CookbookModelConfig,
    CookbookTrainerConfig,
    CookbookEvalConfig,
    CookbookWandbLoggingConfig,
    CookbookSourceConfig,
    launch_experiment,
)
from olmo_core.launchers import BeakerLauncherConfig


model_config = CookbookModelConfig(
    model_type="olmo3",                     # one of olmo2, olmo3, olmo3_lc, etc.
    model_size="1b",                        # this should be one of configs from a ladder
    checkpoint=None,                        # if you provide a checkpoint, it will resume from there
    tokenizer="allenai/dolma2-tokenizer",   # pull from HF using transformers library so we get eos/bos config, etc
    seed=2352,                              # both data and model seed
)

trainer_config = CookbookTrainerConfig(
    # one of the two following must be provided
    chinchilla_multiplier=5,
    max_tokens=None,
    # if the following are not provided, error out if not resuming from a checkpoint;
    # otherwise import from checkpoint
    batch_size=1024 * 4096,
    sequence_length=4096,
    scheduler="cosine",
    learning_rate=1.8e-3,
)

eval_config = CookbookEvalConfig(
    eval_interval=250,  # every 250 steps
    checkpoint_interval=1000, # every 1000 steps
    downstream_evaluators=["olmo2_dev_1b"], # by default empty list,  no downstream evals.
)

# users don't assign name or groups. the name is experiment_name from CookbookExperimentConfig,
# suffixed with a 8 hexidecimal character id. Group is the same of experiment_name, but with no suffix.
logging_config = CookbookWandbLoggingConfig(project="olmo3", entity="ai2-llm")

# this should set up environment for you, including pushing the right git keys,
# auth for aws, google, and beaker from your environment to beaker workspaces.
# only ai2 users should use this.
beaker_config = BeakerLauncherConfig(
    name="olmo3-1b-5xC-dclm-pstar-001-hlr-dolma2",
    budget="ai2/oe-training",
    workspace="ai2/oe-data",
    num_nodes=4,
    num_gpus=8,
    preemptible=True,
    priority="high",
    clusters="ai2/augusta-google-1"
)

cookbook_sources = [
    CookbookSourceConfig(
        name="social_life",
        target_ratio=0.01865278057032931,
        repetition_factor=2.0,
        paths=["s3://ai2-llm/preprocessed/dclm/baseline_topic_ft_lr05_ng2_n3M6_ova_20pct/allenai/dolma2-tokenizer/social_life/**/*.npy"]
    ),
    CookbookSourceConfig(
        name="software",
        target_ratio=0.00024003225357094727,
        repetition_factor=2.0,
        paths=["s3://ai2-llm/preprocessed/dclm/baseline_topic_ft_lr05_ng2_n3M6_ova_20pct/allenai/dolma2-tokenizer/software/**/*.npy"]
    ),
    CookbookSourceConfig(
        name="software_development",
        target_ratio=0.2546200267035965,
        repetition_factor=2.0,
        paths=["s3://ai2-llm/preprocessed/dclm/baseline_topic_ft_lr05_ng2_n3M6_ova_20pct/allenai/dolma2-tokenizer/software_development/**/*.npy"]
    ),
    # ...
    # could do more ofc.
]


experiment_config = CookbookExperimentConfig(
    experiment_name="my-first-experiment",
    model_config=model_config,
    trainer_config=trainer_config,
    eval_config=eval_config,
    logging_config=logging_config,
    beaker_config=beaker_config,
    sources=cookbook_sources,
)

# this will launch the experiment on beaker if beaker config is provided, otherwise run locally.
# should support a dry_run mode, and allow override of any config flag via CLI.
launch_experiment(experiment_config)
