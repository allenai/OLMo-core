from olmo_core.internal.experiment import CliContext, ExperimentConfig, main

from ..builder_common import common_build_experiment_config
from ..config_common import (
    TOKENIZER_CONFIG,
    OLMO_3_7B_MODEL_CONFIG,
    OLMO_3_SEQUENCE_LENGTH,
    OLMO_3_MICROANNEAL_BATCH_SIZE,
    OLMO_3_MICROANNEAL_START_LR,
    OLMO_3_MICROANNEAL_MAX_TOKENS,
    WEB_50PCT_STACK_EDU_PYTHON_50PCT_BASELINE_CONFIG,
)


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    return common_build_experiment_config(
        cli_context=cli_context,
        seq_length=OLMO_3_SEQUENCE_LENGTH,
        global_batch_size=OLMO_3_MICROANNEAL_BATCH_SIZE,
        max_tokens=OLMO_3_MICROANNEAL_MAX_TOKENS,
        lr=OLMO_3_MICROANNEAL_START_LR,  # final LR for pretrained model
        source_list=WEB_50PCT_STACK_EDU_PYTHON_50PCT_BASELINE_CONFIG,
        tokenizer_config=TOKENIZER_CONFIG,
        model_config=OLMO_3_7B_MODEL_CONFIG,
    )


if __name__ == "__main__":
    """
    Invoke this script directly to access the internal experiment CLI, which
    supports launch, train, dry_run, and other subcommands.

    Examples:
        To render the config and exit:
            python src/scripts/train/OLMo3/OLMo3-7B-midtraining.py dry_run debug_run ai2/augusta

        To launch a training run on Augusta w/ 8 nodes:
        python src/scripts/train/OLMo3/OLMo3-7B-midtraining.py launch my_run ai2/augusta \
            --launch.num_nodes=8 \
            --launch.priority=high
    """
    main(config_builder=build_experiment_config)
