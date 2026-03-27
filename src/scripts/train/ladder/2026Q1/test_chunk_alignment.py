"""Test that MixingInstanceSource preserves chunk boundaries."""

import numpy as np
import os
import tempfile

from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ConcatAndChunkInstanceSourceConfig,
    MixingInstanceSourceConfig,
    MixingInstanceSourceSpecConfig,
    NumpyDocumentSourceConfig,
)

SEQ_LEN = 8192
CHUNKS_PER_SHARD = 3
NUM_SHARDS = 2


DTYPE = np.uint32  # dolma2 vocab_size=100278 > uint16 max, so olmo-core infers uint32


def make_shards(directory: str, prefix: str, id_offset: int) -> None:
    """Write raw binary .npy files matching the data gen format (no numpy header)."""
    os.makedirs(directory)
    for shard_idx in range(NUM_SHARDS):
        tokens = np.empty(CHUNKS_PER_SHARD * SEQ_LEN, dtype=DTYPE)
        for chunk_idx in range(CHUNKS_PER_SHARD):
            val = id_offset + shard_idx * CHUNKS_PER_SHARD + chunk_idx
            tokens[chunk_idx * SEQ_LEN : (chunk_idx + 1) * SEQ_LEN] = val
        path = os.path.join(directory, f"shard-{shard_idx:03d}.npy")
        with open(path, "wb") as f:
            f.write(tokens.tobytes())


def test_single_source():
    """Verify ConcatAndChunkInstanceSource alone preserves boundaries."""
    print("=== Test: single source (no mixing) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "data")
        make_shards(data_dir, "data", id_offset=100)

        tokenizer = TokenizerConfig.dolma2()
        work_dir = os.path.join(tmpdir, "work")
        os.makedirs(work_dir)

        source = ConcatAndChunkInstanceSourceConfig(
            sources=[NumpyDocumentSourceConfig(
                source_paths=[os.path.join(data_dir, "*.npy")],
                tokenizer=tokenizer,
            )],
            sequence_length=SEQ_LEN,
        ).build(work_dir)

        failures = 0
        for i in range(len(source)):
            ids = np.array(source[i]["input_ids"])
            unique_vals = np.unique(ids)
            if len(unique_vals) != 1:
                print(f"  FAIL instance {i}: contains values {unique_vals}")
                failures += 1
            else:
                print(f"  OK   instance {i}: value={unique_vals[0]}")
        print(f"  Result: {len(source) - failures}/{len(source)} passed\n")
        return failures == 0


def test_mixing_instance_source():
    """Verify MixingInstanceSource preserves boundaries with 50/50 mix."""
    print("=== Test: MixingInstanceSource (50/50 mix) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        treatment_dir = os.path.join(tmpdir, "treatment")
        baseline_dir = os.path.join(tmpdir, "baseline")
        make_shards(treatment_dir, "treatment", id_offset=100)
        make_shards(baseline_dir, "baseline", id_offset=1000)

        tokenizer = TokenizerConfig.dolma2()
        work_dir = os.path.join(tmpdir, "work")
        os.makedirs(work_dir)

        source = MixingInstanceSourceConfig(
            source_specs=[
                MixingInstanceSourceSpecConfig(
                    source=ConcatAndChunkInstanceSourceConfig(
                        sources=[NumpyDocumentSourceConfig(
                            source_paths=[os.path.join(treatment_dir, "*.npy")],
                            tokenizer=tokenizer,
                        )],
                        sequence_length=SEQ_LEN,
                    ),
                    ratio=0.5,
                    label="treatment",
                ),
                MixingInstanceSourceSpecConfig(
                    source=ConcatAndChunkInstanceSourceConfig(
                        sources=[NumpyDocumentSourceConfig(
                            source_paths=[os.path.join(baseline_dir, "*.npy")],
                            tokenizer=tokenizer,
                        )],
                        sequence_length=SEQ_LEN,
                    ),
                    ratio=0.5,
                    label="baseline",
                ),
            ],
        ).build(work_dir)

        failures = 0
        treatment_count = 0
        baseline_count = 0
        for i in range(len(source)):
            ids = np.array(source[i]["input_ids"])
            unique_vals = np.unique(ids)
            if len(unique_vals) != 1:
                print(f"  FAIL instance {i}: contains values {unique_vals}")
                failures += 1
            else:
                val = unique_vals[0]
                label = "treatment" if val < 1000 else "baseline"
                if label == "treatment":
                    treatment_count += 1
                else:
                    baseline_count += 1
                print(f"  OK   instance {i}: value={val} ({label})")

        total = len(source)
        print(f"  Mix: {treatment_count} treatment, {baseline_count} baseline "
              f"({100*treatment_count/total:.0f}%/{100*baseline_count/total:.0f}%)")
        print(f"  Result: {total - failures}/{total} passed\n")
        return failures == 0


if __name__ == "__main__":
    ok1 = test_single_source()
    ok2 = test_mixing_instance_source()

    if ok1 and ok2:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        raise SystemExit(1)
