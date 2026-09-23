"""
CPU pre-flight for the setA dense SFT pair -- run before any GPU job (``launch_setA_sft.py prep``).

Fails if an input is missing or wrong; otherwise prints, per arm, exactly what training will see:

  * every task shard has ``metadata.json`` and NO document-marker ids (the dense arms must be
    marker-free to match olmo-eval's prompts);
  * per task, examples / tokens kept and dropped at each arm's window (``LongDocStrategy.exclude``),
    counted from the EOS-separated shard files themselves;
  * the base checkpoint exists;
  * packed windows and steps per epoch from the real packer, which also warms the packing cache.
"""

import json
import os
import sys
from glob import glob

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "sft_xlong256k"))

from _qwen35_setA_common import (  # noqa: E402
    ARMS,
    BASE_CHECKPOINT,
    DATA_ROOT,
    TASKS,
    arm_geometry,
    build_setA_experiment,
)

from olmo_core.data.document_chunk_landmark import reserved_ids  # noqa: E402
from olmo_core.internal.experiment import CliContext, SubCmd, prep  # noqa: E402
from olmo_core.utils import prepare_cli_environment  # noqa: E402


def scan_task(task: str, windows: dict, marker_ids: tuple, eos: int) -> dict:
    """Example lengths of one task shard (EOS-terminated), and a marker-id check."""
    lens = []
    markers = 0
    for path in sorted(glob(f"{DATA_ROOT}/{task}/token_ids_part_*.npy")):
        ids = np.fromfile(path, dtype=np.uint32)
        markers += int(np.isin(ids, marker_ids).sum())
        ends = np.flatnonzero(ids == eos)
        lens.append(np.diff(np.concatenate([[-1], ends])))
    lens = np.concatenate(lens) if lens else np.zeros(0, dtype=np.int64)
    out = {
        "examples": int(lens.size),
        "tokens": int(lens.sum()),
        "marker_tokens": markers,
        "longest": int(lens.max()) if lens.size else 0,
    }
    for arm, seq in windows.items():
        keep = lens <= seq
        out[arm] = {
            "kept": int(keep.sum()),
            "kept_tokens": int(lens[keep].sum()),
            "dropped": int((~keep).sum()),
        }
    return out


def main():
    """Fail before GPU submission if the inputs are missing, marker-wrapped or oversized."""
    prepare_cli_environment()
    ids = reserved_ids("qwen3_5")
    windows = {arm: spec["sequence_length"] for arm, spec in ARMS.items()}

    bad = []
    stats = {}
    for task in TASKS:
        meta_path = f"{DATA_ROOT}/{task}/metadata.json"
        if not os.path.exists(meta_path):
            bad.append(f"{task}: no metadata.json under {DATA_ROOT}")
            continue
        meta = json.load(open(meta_path))
        if meta.get("marker_set") != "qwen3_5":
            bad.append(f"{task}: marker_set {meta.get('marker_set')!r} != 'qwen3_5'")
        if meta.get("doc_markers", True):
            bad.append(f"{task}: built WITH document markers (need --no-doc-markers)")
        stats[task] = scan_task(task, windows, (ids.doc_start, ids.doc_end), ids.eos)
        if stats[task]["marker_tokens"]:
            bad.append(f"{task}: {stats[task]['marker_tokens']} marker tokens in the shard")
        print("TASK", task, json.dumps(stats[task]), flush=True)
    extra = sorted(set(os.listdir(DATA_ROOT)) - set(TASKS)) if os.path.isdir(DATA_ROOT) else []
    if extra:
        print(f"note: {DATA_ROOT} also holds {extra} -- not trained on", flush=True)
    if not os.path.exists(os.path.join(BASE_CHECKPOINT, ".metadata")):
        bad.append(f"base checkpoint missing: {BASE_CHECKPOINT}/.metadata")
    if bad:
        raise SystemExit("PREP FAILED:\n  " + "\n  ".join(bad))

    print(
        f"\n{'task':<15}{'examples':>10}{'tokens':>15}"
        + "".join(f"{'kept@' + a:>14}{'tok@' + a:>15}" for a in windows),
        flush=True,
    )
    for task, s in stats.items():
        print(
            f"{task:<15}{s['examples']:>10,}{s['tokens']:>15,}"
            + "".join(f"{s[a]['kept']:>14,}{s[a]['kept_tokens']:>15,}" for a in windows),
            flush=True,
        )
    for a in windows:
        print(
            f"TOTAL@{a}: {sum(s[a]['kept'] for s in stats.values()):,} examples, "
            f"{sum(s[a]['kept_tokens'] for s in stats.values()):,} tokens kept; "
            f"{sum(s[a]['dropped'] for s in stats.values()):,} dropped",
            flush=True,
        )

    for arm in ARMS:
        g = arm_geometry(arm)
        config = build_setA_experiment(
            CliContext(
                script=__file__,
                cmd=SubCmd.prep,
                run_name=f"q35-dense-ctc-setA-{arm}-prep",
                cluster="ai2/jupiter-cirrascale-2",
                overrides=[],
            ),
            arm=arm,
        )
        source = config.dataset[0].build(config.data_loader.work_dir)
        prep(config)  # the real prep path: packing + loader cache
        print(
            "PACKING_RESULT",
            json.dumps(
                dict(
                    arm=arm,
                    **g,
                    packed_windows=len(source),
                    steps_per_epoch=len(source) // g["windows_per_step"],
                    epoch_tail_windows=len(source) % g["windows_per_step"],
                )
            ),
            flush=True,
        )
    print("PREP OK", flush=True)


if __name__ == "__main__":
    main()
