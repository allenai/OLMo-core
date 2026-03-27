import argparse
import logging
from dataclasses import dataclass, field
from typing import Any, Dict

import torch

import olmo_core.distributed.utils as dist_utils
import olmo_core.io as io
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import *
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import (
    ModelLadder,
    Olmo3ModelConfigurator,
    TransformerSize,
    WSDSChinchillaRunConfigurator,
)
from olmo_core.train.callbacks.callback import Callback

log = logging.getLogger(__name__)

OVERLAP_OPEN_ID = 100274
OVERLAP_CLOSE_ID = 100275


NUM_STEPS_TO_CHECK = 50


@dataclass
class FirstBatchAlignmentCheck(Callback):
    """Check the first NUM_STEPS_TO_CHECK steps, then stop the job."""

    _steps_checked: int = field(default=0, init=False)
    _total_icl: int = field(default=0, init=False)
    _total_baseline: int = field(default=0, init=False)
    _total_fail: int = field(default=0, init=False)

    def pre_step(self, batch: Dict[str, Any]):
        if self._steps_checked >= NUM_STEPS_TO_CHECK:
            return

        input_ids = batch["input_ids"]
        n, seq_len = input_ids.shape
        special = torch.tensor([OVERLAP_OPEN_ID, OVERLAP_CLOSE_ID], device=input_ids.device)

        for i in range(n):
            row = input_ids[i]
            tok0 = row[0].item()
            has_special = torch.isin(row, special).any().item()
            if tok0 == OVERLAP_OPEN_ID:
                self._total_icl += 1
            elif not has_special:
                self._total_baseline += 1
            else:
                self._total_fail += 1
                if dist_utils.get_rank() == 0:
                    log.warning(
                        f"ALIGNMENT FAIL step={self._steps_checked} instance={i} "
                        f"first_token={tok0} first_20={row[:20].tolist()}"
                    )

        self._steps_checked += 1

    def post_step(self):
        if self._steps_checked < NUM_STEPS_TO_CHECK:
            return

        total = self._total_icl + self._total_baseline + self._total_fail
        if dist_utils.get_rank() == 0:
            log.warning(f"\n{'='*80}")
            log.warning(
                f"ALIGNMENT CHECK COMPLETE ({NUM_STEPS_TO_CHECK} steps, "
                f"{total} instances on rank 0)"
            )
            log.warning(f"  ICL-overlap (first_token={OVERLAP_OPEN_ID}): {self._total_icl}")
            log.warning(f"  Baseline (no special tokens):                 {self._total_baseline}")
            log.warning(f"  FAILED (special tokens but wrong start):      {self._total_fail}")
            result = "PASS" if self._total_fail == 0 else "FAIL"
            log.warning(f"  RESULT: {result}")
            log.warning(f"{'='*80}\n")

        self.trainer.cancel_run("Alignment check complete", no_sync=True)


class _InspectableLadder(ModelLadder):
    def _configure_trainer(self, size_spec, for_benchmarking=False):
        config = super()._configure_trainer(size_spec, for_benchmarking=for_benchmarking)
        config.add_callback("alignment_check", FirstBatchAlignmentCheck())
        return config


ICL_OVERLAP_PATHS = [
    "/weka/oe-training-default/ai2-llm/suffix-arrays/preprocessed/dolma2-0625-v01/icl-overlap-max-suffix-8192-eos-fix/allenai/dolma2-tokenizer/*.npy",
]

DOLMA2_BASELINE_PATHS = [
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2-fixed/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/arxiv/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/finemath-3plus/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/s2pdf_redacted/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/stack-edu/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/wikipedia/**/*.npy",
]


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        MixingInstanceSourceConfig(
            source_specs=[
                MixingInstanceSourceSpecConfig(
                    source=ConcatAndChunkInstanceSourceConfig(
                        sources=[
                            NumpyDocumentSourceConfig(
                                source_paths=ICL_OVERLAP_PATHS,
                                tokenizer=tokenizer,
                            ),
                        ],
                        sequence_length=args.sequence_length,
                    ),
                    ratio=0.5,
                    label="icl-overlap",
                ),
                MixingInstanceSourceSpecConfig(
                    source=ConcatAndChunkInstanceSourceConfig(
                        sources=[
                            NumpyDocumentSourceConfig(
                                source_paths=DOLMA2_BASELINE_PATHS,
                                tokenizer=tokenizer,
                            ),
                        ],
                        sequence_length=args.sequence_length,
                    ),
                    ratio=0.5,
                    label="baseline",
                ),
            ],
        ),
    ]

    ladder = _InspectableLadder(
        name=args.name,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=list(TransformerSize),
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=Olmo3ModelConfigurator(
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
        ),
        run_configurator=WSDSChinchillaRunConfigurator(
            chinchilla_multiple=args.chinchilla_multiple
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
    )
    return ladder


if __name__ == "__main__":
    main(configure_ladder)
