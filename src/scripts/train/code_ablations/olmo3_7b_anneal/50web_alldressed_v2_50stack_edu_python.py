from olmo_core.internal.experiment import CliContext, ExperimentConfig, main

from ..builder_common import common_build_experiment_config, common_main
from ..config_common import (
    TOKENIZER_CONFIG,
    OLMO_3_7B_MODEL_CONFIG,
    OLMO_3_SEQUENCE_LENGTH,
    OLMO_3_MICROANNEAL_BATCH_SIZE,
    OLMO_3_MICROANNEAL_START_LR,
    OLMO_3_MICROANNEAL_MAX_TOKENS,
    OLMO_3_MICROANNEAL_LOAD_PATH,
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
        num_nodes=4,
        load_path=OLMO_3_MICROANNEAL_LOAD_PATH,
    )


if __name__ == "__main__":
    """
    Invoke this script directly to access the internal experiment CLI, which
    supports launch, train, dry_run, and other subcommands.

     uv run python \
        -d src.scripts.train.code_ablations.olmo3_7b_anneal.50web_alldressed_v2_50stack_edu_python \
        launch \
        my_run ai2/neptune --launch.num_nodes=8
    """
    common_main(config_builder=build_experiment_config, default_run_name=__name__)
