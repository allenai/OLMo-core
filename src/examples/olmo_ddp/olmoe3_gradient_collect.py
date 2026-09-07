"""Collect bounded gradient diagnostics; raw gradient samples stay on Weka."""

import argparse
import json
import math
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/weka/olmo-3p5-checkpoints/production-profiling")


def write(path, data):
    """Publish a compact complete JSON artifact atomically."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def family(name):
    """Disentangle recurrent/attention and expert gradients in the report."""
    if "router" in name:
        return "router"
    if "routed_experts" in name:
        return "expert-down" if "w_down" in name else "expert-up-gate"
    if "attention" in name or "sequence_mixer" in name:
        for index in (7, 15, 23):
            if f"blocks.{index}." in name:
                return "full-attention"
        return "KDA"
    return "dense-other"


def summarize(run):
    """Use full-shard scalar norms on every rank, not sampled norm estimates."""
    provenance = json.loads((run / "provenance.json").read_text())
    gpus = provenance["gpus"]
    squared = defaultdict(float)
    parameters = defaultdict(float)
    reported, independent, rng = [], [], []
    for rank in range(gpus):
        data = json.loads((run / f"gradients-rank-{rank}.json").read_text())
        assert data["rank"] == rank and data["gpus"] == gpus
        reported.append(data["reported_norm"])
        independent.append(data["independent_norm"])
        rng.append(data.get("first_step_rng"))
        for row in data["parameters"]:
            if row["stage"] != "optimizer-intake":
                continue
            copies = (
                1
                if "Shard(" in row["placements"]
                else (gpus if row["group"] == "dp" else gpus // 8)
            )
            value = row["norm"] ** 2 / copies
            parameters[row["name"]] += value
            squared[family(row["name"])] += value
    assert max(independent) - min(independent) < 1e-8
    assert math.isclose(math.sqrt(sum(squared.values())), independent[0], rel_tol=1e-6)
    return {
        "run": run.name,
        "gpus": gpus,
        "reported_norm_min_max": [min(reported), max(reported)],
        "independent_norm": independent[0],
        "reported_vs_independent_relative": reported[0] / independent[0] - 1,
        "family_norms": {k: math.sqrt(v) for k, v in squared.items()},
        "parameter_norms": {k: math.sqrt(v) for k, v in parameters.items()},
        "provenance": provenance,
        "first_step_rng_by_rank": rng,
    }


def compare(reference, candidate, a, b):
    """Sample differences are explicitly distinct from full-vector comparisons."""
    stats = defaultdict(lambda: np.zeros(6, dtype=np.float64))
    parameter_stats = defaultdict(lambda: np.zeros(6, dtype=np.float64))
    for rank in range(8):
        with (
            np.load(reference / f"gradient-samples-rank-{rank}.npz") as aa,
            np.load(candidate / f"gradient-samples-rank-{rank}.npz") as bb,
        ):
            assert set(aa.files) == set(bb.files), "Gradient sample parameter sets differ"
            for key in aa.files:
                x, y = aa[key].astype(np.float64), bb[key].astype(np.float64)
                assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
                delta = y - x
                row = np.array(
                    [
                        np.dot(x, x),
                        np.dot(y, y),
                        np.dot(delta, delta),
                        np.dot(x, y),
                        np.max(np.abs(delta), initial=0),
                        x.size,
                    ]
                )
                name = key.split("/", 1)[1]
                for dest in (stats[family(name)], parameter_stats[name]):
                    dest[:4] += row[:4]
                    dest[4] = max(dest[4], row[4])
                    dest[5] += row[5]

    def finish(row):
        xx, yy, dd, xy, max_abs, n = row
        return {
            "sample_relative_l2": math.sqrt(dd / xx) if xx else None,
            "sample_cosine": xy / math.sqrt(xx * yy) if xx and yy else None,
            "sample_max_absolute": max_abs,
            "sample_count": int(n),
        }

    return {
        "reference": reference.name,
        "candidate": candidate.name,
        "full_norm_relative_change": b["independent_norm"] / a["independent_norm"] - 1,
        "rng_equal_ranks": {
            device: sum(
                ra is not None and rb is not None and ra.get(device) == rb.get(device)
                for ra, rb in zip(a["first_step_rng_by_rank"], b["first_step_rng_by_rank"])
            )
            for device in ("cpu", "cuda")
        },
        "family_full_norm_relative_change": {
            k: b["family_norms"][k] / v - 1 if v else None for k, v in a["family_norms"].items()
        },
        "sample_scope": "2048 fixed elements per optimizer shard, ranks 0..7; not full-vector errors",
        "family_samples": {k: finish(v) for k, v in stats.items()},
        "parameter_samples": {k: finish(v) for k, v in parameter_stats.items()},
    }


def main():
    """Watch only the explicitly requested diagnostics and publish compact results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--wait-seconds", type=int, default=28800)
    args = parser.parse_args()
    assert 60 <= args.wait_seconds <= 28800
    assert all(Path(name).name == name for name in args.names)
    output = Path(os.environ.get("RESULTS_DIR", "/results")) / "gradient-summaries"
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.wait_seconds
    done = {}
    compared = set()
    while len(done) < len(args.names):
        for name in args.names:
            if name in done:
                continue
            run = ROOT / name
            if not (run / "training-process-complete.json").is_file():
                continue
            provenance = json.loads((run / "provenance.json").read_text())
            if not all(
                (run / f"gradients-rank-{r}.json").is_file() for r in range(provenance["gpus"])
            ):
                continue
            summary = summarize(run)
            write(output / f"{name}.json", summary)
            small = output / name
            small.mkdir(exist_ok=True)
            for filename in (
                "metrics.jsonl",
                "initial-weights-sha256.json",
                "first-batch-sha256.json",
            ):
                shutil.copy2(run / filename, small / filename)
            memory = [json.loads(p.read_text()) for p in run.glob("full-run-memory-rank-*.json")]
            write(small / "memory.json", memory)
            done[name] = summary
            print(
                "GRADIENT_COLLECTED",
                name,
                json.dumps(
                    {
                        k: v
                        for k, v in summary.items()
                        if k not in ("parameter_norms", "provenance", "first_step_rng_by_rank")
                    }
                ),
                flush=True,
            )
        for name, candidate in done.items():
            if name in compared or "-baseline-timing" in name:
                continue
            reference = next(
                (
                    n
                    for n, a in done.items()
                    if a["gpus"] == candidate["gpus"] and "-baseline-timing" in n
                ),
                None,
            )
            if reference is None:
                continue
            comparison = compare(ROOT / reference, ROOT / name, done[reference], candidate)
            write(output / f"comparison-{name}.json", comparison)
            print(
                "GRADIENT_COMPARISON",
                name,
                json.dumps({k: v for k, v in comparison.items() if k != "parameter_samples"}),
                flush=True,
            )
            compared.add(name)
        if len(done) < len(args.names):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Incomplete gradient diagnostics: {set(args.names) - set(done)}"
                )
            print(
                "Waiting for gradient diagnostics", sorted(set(args.names) - set(done)), flush=True
            )
            time.sleep(30)


if __name__ == "__main__":
    main()
