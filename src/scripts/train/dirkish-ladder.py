from dataclasses import dataclass
from typing import Any, ClassVar, Dict

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
        # TODO: may need to tune these
        # ===============================
        ModelSize.size_190M: 16 * 4096,
        ModelSize.size_370M: 16 * 4096,
        ModelSize.size_600M: 16 * 4096,
        ModelSize.size_760M: 16 * 4096,
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
        else:
            assert "h100" in gpu_type.lower()
            return self.MBZ_SIZES[size]

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

        # Need no in-loop evals
        config.callbacks['lm_evaluator'].enabled = False
        config.callbacks['downstream_evaluator'] = DownstreamEvaluatorCallbackConfig(
            tasks=[
                # OLMES
                "arc_challenge_mc_5shot_bpb",
                "arc_easy_mc_5shot_bpb",
                "boolq_mc_5shot_bpb",
                "csqa_mc_5shot_bpb",
                "hellaswag_mc_5shot_bpb",
                "openbookqa_mc_5shot_bpb",
                "piqa_mc_5shot_bpb",
                "socialiqa_mc_5shot_bpb",
                "winogrande_mc_5shot_bpb",
                # OLMES includes MMLU
                "mmlu_stem_mc_5shot",
                "mmlu_humanities_mc_5shot",
                "mmlu_social_sciences_mc_5shot",
                "mmlu_other_mc_5shot",
            ],
            tokenizer=self.tokenizer,
            eval_interval=2,
        )

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
    config.ladder.validate()

    # Run the cmd.
    cmd.run(config)
