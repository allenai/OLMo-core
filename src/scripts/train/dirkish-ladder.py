import copy
from dataclasses import dataclass
from typing import Any, ClassVar, Dict

import olmo_eval.tasks

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import get_beaker_username, get_work_dir, get_root_dir
from olmo_core.internal.model_ladder import SubCmd, build_config
from olmo_core.io import join_path
from olmo_core.model_ladder import ModelLadder, ModelSize, RunDuration
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import AdamWConfig, OptimConfig, OptimGroupOverride
from olmo_core.optim.scheduler import CosWithWarmupAndLinearDecay
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import SchedulerCallback, DownstreamEvaluatorCallbackConfig


@dataclass
class DirkishModelLadder(ModelLadder):
    """
    Baseline OLMo model ladder using the current recommended architecture.
    """

    MBZ_SIZES: ClassVar[Dict[ModelSize, int]] = {
        # ===============================
        ModelSize.size_190M: 8 * 4096,
        ModelSize.size_370M: 8 * 4096,
        ModelSize.size_600M: 8 * 4096,
        ModelSize.size_760M: 8 * 4096,
        # ===============================
        ModelSize.size_1B: 8 * 4096,
        ModelSize.size_3B: 4 * 4096,
        ModelSize.size_7B: 2 * 4096,
        ModelSize.size_13B: 1 * 4096,
    }

    MODEL_OVERRIDES: ClassVar[Dict[ModelSize, Dict[str, Any]]] = {
        ModelSize.size_1B: dict(n_layers=16),  # need to scale down our actual 1B model
    }

    def _get_model_config(self, *, size: ModelSize) -> TransformerConfig:
        data_parallel_type = DataParallelType.hsdp
        return getattr(TransformerConfig, f"olmo2_{size}")(
            vocab_size=self.tokenizer.padded_vocab_size(),
            init_seed=self.init_seed,
            compile=True,
            dp_config=TransformerDataParallelConfig(
                name=data_parallel_type, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
            ),
            **self.MODEL_OVERRIDES.get(size, {}),
        )

    def get_optim_config(self) -> OptimConfig:
        # Calculate LR according to https://api.semanticscholar.org/CorpusID:270764838
        assert self.sequence_length in {2048, 4096}
        lr = 0.0047 * (self.model_size / 108000000) ** (-1 / 3)

        return AdamWConfig(
            lr=lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        )

    def get_rank_microbatch_size(self, *, size: ModelSize, gpu_type: str) -> int:
        if gpu_type.lower() in ("mps", "cpu"):
            return 4096
        result = self.MBZ_SIZES[size]
        if "b200" in gpu_type.lower():
            result *= 2
        return result

    def get_trainer_config(
        self,
        *,
        size: ModelSize,
        run_duration: RunDuration,
        gpu_type: str,
        dp_world_size: int,
    ) -> TrainerConfig:
        config = super().get_trainer_config(
            size=size,
            run_duration=run_duration,
            gpu_type=gpu_type,
            dp_world_size=dp_world_size)

        # monkey-patch oe-eval
        available_task_labels = list(olmo_eval.tasks.label_to_task_map.keys())
        for task_label in available_task_labels:
            bpb_task_label = task_label + "_bpb"
            if bpb_task_label in olmo_eval.tasks.label_to_task_map:
                continue
            task = olmo_eval.tasks.label_to_task_map[task_label]
            if not isinstance(task, tuple):
                continue
            if len(task) < 2:
                continue
            if not isinstance(task[1], dict):
                continue
            bpb_task = copy.deepcopy(task)
            bpb_task[1]["metric_type"] = "bpb"
            olmo_eval.tasks.label_to_task_map[bpb_task_label] = bpb_task

        config.callbacks['lm_evaluator'].enabled = False
        config.callbacks['downstream_evaluator'] = DownstreamEvaluatorCallbackConfig(
            tasks=[
                # OLMES Core 9 RC
                "arc_challenge_test_rc_5shot_bpb",
                "arc_easy_test_rc_5shot_bpb",
                "hellaswag_rc_5shot_bpb",
                "winogrande_val_rc_5shot_bpb", # Helpful after 750M-5xC scale
                "csqa_val_rc_5shot_bpb",
                "piqa_val_rc_5shot_bpb",
                "socialiqa_val_rc_5shot_bpb",

                # Too noisy to be worth tracking
                # "boolq_val_rc_5shot_bpb",
                # "openbookqa_test_rc_5shot_bpb",

                # MMLU RC BPB
                "mmlu_stem_val_rc_5shot_bpb",
                "mmlu_humanities_val_rc_5shot_bpb",
                "mmlu_social_sciences_val_rc_5shot_bpb",
                "mmlu_other_val_rc_5shot_bpb",
                "mmlu_stem_test_rc_5shot_bpb",
                "mmlu_humanities_test_rc_5shot_bpb",
                "mmlu_social_sciences_test_rc_5shot_bpb",
                "mmlu_other_test_rc_5shot_bpb",

                # OLMES Core 9 MC (BPB is included in these)
                "arc_challenge_test_mc_5shot_bpb",
                "arc_easy_test_mc_5shot_bpb",
                "hellaswag_rc_5shot_bpb",
                "csqa_val_mc_5shot_bpb",
                "piqa_val_mc_5shot_bpb",
                "socialiqa_val_mc_5shot_bpb",
                "winogrande_val_rc_5shot_bpb",

                # Too noisy to be worth tracking
                # "boolq_val_mc_5shot_bpb",
                # "openbookqa_test_mc_5shot_bpb",

                # MMLU MC BPB
                "mmlu_stem_val_mc_5shot_bpb",
                "mmlu_humanities_val_mc_5shot_bpb",
                "mmlu_social_sciences_val_mc_5shot_bpb",
                "mmlu_other_val_mc_5shot_bpb",
                "mmlu_stem_test_mc_5shot_bpb",
                "mmlu_humanities_test_mc_5shot_bpb",
                "mmlu_social_sciences_test_mc_5shot_bpb",
                "mmlu_other_test_mc_5shot_bpb",
            ],
            tokenizer=self.tokenizer,
            eval_interval=1000,
        )
        config.callbacks['checkpointer'].ephemeral_save_interval = 1000

        # Set a modified cosine schedule with decay to 0 at the end
        config.callbacks['lr_scheduler'] = SchedulerCallback(
            scheduler=CosWithWarmupAndLinearDecay(warmup_steps=round(self.model_size / self.get_global_batch_size()))
        )

        return config


def build_ladder(name: str, root_dir: str) -> DirkishModelLadder:
    save_folder = str(join_path(root_dir, f"checkpoints/{get_beaker_username().lower()}/ladder"))
    r = DirkishModelLadder(
        name=name,
        project="dirkish-ladder",
        mix_base_dir=root_dir,
        work_dir=get_work_dir(root_dir),
        save_folder=save_folder,
        sequence_length=4096
    )
    return r


if __name__ == "__main__":
    import sys

    usage = f"""
[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]{'|'.join(SubCmd)}[/] [i b]NAME SIZE RUN_DURATION CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]      Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use [b magenta]launch[/] or run it with torchrun.
[b magenta]train_single:[/]       Run the trainer on a single device (GPU, CPU, MPS). num_nodes is ignored.
[b magenta]dry_run:[/]     Pretty print the config to run and exit.

[b]Examples[/]
$ [i]python {sys.argv[0]} {SubCmd.launch} 1B 1xC ai2/pluto-cirrascale --launch.num_nodes=2[/]
    """.strip()

    try:
        script, cmd, name, size, run_duration, cluster, overrides = (
            sys.argv[0],
            SubCmd(sys.argv[1]),
            sys.argv[2],
            ModelSize(sys.argv[3]),
            RunDuration(sys.argv[4]),
            sys.argv[5],
            sys.argv[6:],
        )
    except (IndexError, ValueError):
        import rich

        rich.get_console().print(usage, highlight=False)
        sys.exit(1)

    cmd.prepare_environment()

    # Build ladder config.
    ladder = build_ladder(name, get_root_dir(cluster))
    ladder = ladder.merge(overrides, prefix="ladder")

    # Build run config.
    config = build_config(ladder, script, size, run_duration, cmd, cluster, overrides)
    config.data_loader.num_workers = 16

    # monkey patch the launch command
    script, command, size, run_duration, cluster, *overrides = config.launch.cmd
    config.launch.cmd = [script, command, name, size, run_duration, cluster, *overrides]

    config.ladder.validate()

    # Run the cmd.
    cmd.run(config)
