"""Dump the first global batches of the Molmo2 Stage-1/Stage-2 data loaders, built exactly as the
scripts build them, for one data-parallel rank. The olmo-core tree is selected via PYTHONPATH and
the script is read from that same tree.

Usage:
    HF_HUB_OFFLINE=1 PYTHONPATH=<tree>/src python loader_parity.py <tree> {1,2} <rank> <n_batches> <out.json> [overrides...]
    python loader_parity.py --compare a.json b.json

Each record holds, per batch, every key's dtype/shape/sha256 (and for packed rows the ordered
example-id runs), so two trees can be compared tensor by tensor.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time


def _sha(x) -> dict:
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().contiguous()
        meta = {"dtype": str(arr.dtype), "shape": list(arr.shape)}
        data = arr.view(torch.uint8).numpy().tobytes() if arr.numel() else b""
    elif isinstance(x, np.ndarray):
        arr = np.ascontiguousarray(x)
        meta = {"dtype": str(arr.dtype), "shape": list(arr.shape)}
        data = arr.tobytes()
    else:
        meta = {"type": type(x).__name__}
        data = json.dumps(x, default=str, sort_keys=True).encode()
    meta["sha256"] = hashlib.sha256(data).hexdigest()[:16]
    return meta


def _load_script(tree: str, stage: int):
    path = f"{tree}/src/scripts/train/Molmo2-Stage{stage}.py"
    spec = importlib.util.spec_from_file_location(f"molmo2_stage{stage}_script", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, path


def build_loader(tree: str, stage: int, rank: int, world: int, overrides: list):
    from olmo_core.data.multimodal import MixtureDataLoader

    mod, script = _load_script(tree, stage)
    config = mod.build_config(script, f"loader-parity-s{stage}", overrides)
    if stage == 1:
        tokenizer = mod._load_tokenizer(config.model_size, config.init_from)
        datasets, weights, names = mod._build_mixture_sources(tokenizer, config)
        collator = config.collator.build()
        loader = MixtureDataLoader(
            datasets,
            weights,
            collator,
            dataset_names=names,
            work_dir="/tmp/claude-0/loader-parity",
            global_batch_size=config.global_batch_size,
            seed=config.data_seed,
            pack=config.pack_sequences,
            pack_max_crops=config.pack_max_crops if config.pack_sequences else None,
            pack_buffer_size=config.pack_buffer_size,
            pack_image_weight=config.pack_image_weight,
            prefetch_workers=config.data_prefetch_workers,
            dp_world_size=world,
            dp_rank=rank,
        )
    else:
        tokenizer = mod._load_tokenizer()
        datasets, weights, names = mod._build_mixture(tokenizer, config)
        collator = config.collator.build()
        loader = MixtureDataLoader(
            datasets,
            weights,
            collator,
            work_dir="/tmp/claude-0/loader-parity",
            global_batch_size=config.global_batch_size,
            seed=config.data_seed,
            pack=config.pack_sequences,
            pack_max_crops=config.pack_max_crops if config.pack_sequences else None,
            est_tokens_per_example=mod.EST_TOKENS_PER_EXAMPLE,
            prefetch_workers=mod.DATA_PREFETCH_WORKERS,
            dp_world_size=world,
            dp_rank=rank,
            dataset_names=names,
        )
    return loader, names


def dump(tree: str, stage: int, rank: int, n_batches: int, out: str, overrides: list):
    import olmo_core

    t0 = time.time()
    loader, names = build_loader(tree, stage, rank, 8, overrides)
    t_build = time.time() - t0
    loader.reshuffle(epoch=1)
    record = {
        "olmo_core": olmo_core.__file__,
        "stage": stage,
        "rank": rank,
        "names": names,
        "loader_attrs": {
            k: getattr(loader, k, None)
            for k in ("pack", "pack_max_crops", "pack_buffer_size", "pack_image_weight", "seed")
        },
        "total_batches": loader.total_batches,
        "build_seconds": round(t_build, 1),
        "batches": [],
    }
    it = iter(loader._iter_batches())
    for b in range(n_batches):
        t1 = time.time()
        batch = next(it)
        entry = {"keys": sorted(batch), "tensors": {k: _sha(v) for k, v in sorted(batch.items())}}
        if "example_ids" in batch:
            rows = []
            for row in batch["example_ids"].tolist():
                runs = []
                for v in row:
                    if not runs or runs[-1] != v:
                        runs.append(v)
                rows.append(runs)
            entry["example_id_runs"] = rows
        entry["seconds"] = round(time.time() - t1, 1)
        record["batches"].append(entry)
        print(f"batch {b}: {len(batch)} keys, {entry['seconds']}s", flush=True)
    close = getattr(it, "close", None)
    if close is not None:
        close()
    json.dump(record, open(out, "w"), indent=1, default=str)
    print(f"wrote {out} ({t_build:.0f}s build)")


def compare(a_path: str, b_path: str) -> int:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    ok = True
    for key in ("names", "loader_attrs", "total_batches"):
        if a[key] != b[key]:
            ok = False
            print(f"DIFF {key}: {a[key]} vs {b[key]}")
    for i, (ba, bb) in enumerate(zip(a["batches"], b["batches"])):
        if ba["keys"] != bb["keys"]:
            ok = False
            print(f"batch {i}: key sets differ: only A {sorted(set(ba['keys']) - set(bb['keys']))}, "
                  f"only B {sorted(set(bb['keys']) - set(ba['keys']))}")
        for k in sorted(set(ba["keys"]) & set(bb["keys"])):
            ta, tb = ba["tensors"][k], bb["tensors"][k]
            same = ta == tb
            ok &= same
            print(f"batch {i} {'ok  ' if same else 'DIFF'} {k:22s} {ta.get('dtype')} {ta.get('shape')} "
                  f"{ta['sha256']}{'' if same else ' vs ' + str(tb.get('shape')) + ' ' + tb['sha256']}")
    print("LOADER PARITY:", "IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    tree, stage, rank, n, out, *ov = sys.argv[1:]
    dump(tree, int(stage), int(rank), int(n), out, ov)
