"""Verify final CPT checkpoints and measure both contradiction packing plans on CPU."""

import json
from pathlib import Path

from _qwen35_contradiction_256k_common import DATA_ROOT, _ARMS, build_contradiction_experiment

from olmo_core.internal.experiment import CliContext, SubCmd, prep
from olmo_core.utils import prepare_cli_environment


def main():
    """Fail before GPU submission if the inputs are missing or oversized."""
    prepare_cli_environment()
    root = Path(DATA_ROOT)
    metadata = json.loads((root / "metadata.json").read_text())
    print("CONTRADICTION_METADATA", json.dumps(metadata), flush=True)
    for arm in _ARMS:
        context = CliContext(
            script=__file__,
            cmd=SubCmd.prep,
            run_name=f"q35-{arm}-contra-prep",
            cluster="ai2/jupiter-cirrascale-2",
            overrides=[],
        )
        config = build_contradiction_experiment(context, arm=arm)
        checkpoint = Path(config.trainer.load_path)
        assert (checkpoint / ".metadata").is_file(), checkpoint
        print("CHECKPOINT_OK", arm, str(checkpoint), flush=True)
        source = config.dataset[0].build(config.data_loader.work_dir)
        if arm == "compressive":
            assert source._num_dropped == 0, source._num_dropped
        # Exercise the real prep path, including loader reshuffling and cache generation.
        prep(config)
        print(
            "PACKING_RESULT",
            json.dumps(
                dict(
                    arm=arm,
                    packed_windows=len(source),
                    sequence_length=source.sequence_length,
                    steps_per_epoch=len(source) // 4,
                    epoch_tail_windows=len(source) % 4,
                    total_steps=3 * (len(source) // 4),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
