"""
Stage full HF model checkpoints onto weka so eval jobs never download weights at run time.

The tokenizer-only sibling of this script is :mod:`stage_tokenizers_weka`; the reasoning is the
same but the stakes are higher. A 15 GB weight download that dies on a transient Hub 429 wastes an
allocated GPU node, and every job that wants the model pays the download again. Staging once to a
weka path that every eval job already mounts removes the dependency entirely.

Run once per model (CPU-only gantry job, weka-mounted)::

    gantry run --name stage-hils-models -w ai2/flex2 -b ai2/oe-other \\
      --cluster ai2/neptune,ai2/ceres,ai2/saturn,ai2/jupiter --gpus 0 --priority urgent \\
      --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \\
      --weka oe-training-default:/weka/oe-training-default \\
      --env HF_HUB_DISABLE_XET=1 --install true --timeout 0 --yes -- \\
      python src/scripts/data/stage_hf_models_weka.py \\
        --repos tencent/HiLS-Attention-7B,allenai/Olmo-3-1025-7B

Layout matches the tokenizer staging convention -- ``<root>/<repo id with '/' -> '__'>``, e.g.
``/weka/oe-training-default/amandab/hf_models/tencent__HiLS-Attention-7B``.
"""

import argparse
import json
import os
import shutil
import sys
import time

DEFAULT_ROOT = "/weka/oe-training-default/amandab/hf_models"

# Weights + config + tokenizer, but NOT the .bin duplicates some repos keep alongside safetensors
# (doubling the download for files nothing loads) and not the repo card assets.
ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
]


def local_dir(root: str, repo: str) -> str:
    """
    Map a Hub repo id to its staged directory.

    :param root: The staging root, e.g. ``/weka/oe-training-default/amandab/hf_models``.
    :param repo: The Hub repo id, e.g. ``tencent/HiLS-Attention-7B``.

    :returns: The absolute directory the model is staged to.
    """
    return os.path.join(root, repo.replace("/", "__"))


def stage(repo: str, root: str, attempts: int = 8) -> bool:
    """
    Download one model's Hub files into ``root``, retrying transient Hub failures.

    ``snapshot_download`` resumes what is already on disk, so a retry after a partial failure only
    re-fetches the missing shards. A single 429/503 aborts the whole call, which is why the retry
    loop is load-bearing rather than defensive.

    :param repo: The Hub repo id to fetch.
    :param root: The staging root directory.
    :param attempts: How many times to try before giving up.

    :returns: ``True`` if the model was staged successfully.
    """
    from huggingface_hub import snapshot_download

    dest = local_dir(root, repo)
    for attempt in range(1, attempts + 1):
        try:
            t0 = time.time()
            snapshot_download(repo, allow_patterns=ALLOW_PATTERNS, local_dir=dest)
            shutil.rmtree(os.path.join(dest, ".cache"), ignore_errors=True)
            print(f"SAVED {repo} -> {dest}  ({time.time() - t0:.0f}s)", flush=True)
            return True
        except Exception as e:  # noqa: BLE001 -- any Hub/network error is worth retrying
            print(f"  attempt {attempt}/{attempts} for {repo}: {type(e).__name__}: {e}", flush=True)
            if attempt < attempts:
                time.sleep(15 * attempt)
    print(f"GAVE UP on {repo}", flush=True)
    return False


def verify(repo: str, root: str) -> bool:
    """
    Check the staged copy is complete: config present, and every shard named by the safetensors
    index actually on disk.

    Deliberately does NOT instantiate the model. These are staged on a CPU node, and one of them
    (HiLS) cannot be built without its out-of-tree modeling code anyway -- so a load-based check
    would either be impossible or would prove something other than "the bytes arrived".

    :param repo: The Hub repo id that was staged.
    :param root: The staging root directory.

    :returns: ``True`` if the staged copy looks complete.
    """
    dest = local_dir(root, repo)
    cfg_path = os.path.join(dest, "config.json")
    if not os.path.exists(cfg_path):
        print(f"VERIFY FAILED {dest}: no config.json", flush=True)
        return False
    cfg = json.load(open(cfg_path))

    index_path = os.path.join(dest, "model.safetensors.index.json")
    if os.path.exists(index_path):
        shards = sorted(set(json.load(open(index_path))["weight_map"].values()))
    else:  # single-shard repo
        shards = ["model.safetensors"]
    missing = [s for s in shards if not os.path.exists(os.path.join(dest, s))]
    if missing:
        print(f"VERIFY FAILED {dest}: missing shards {missing}", flush=True)
        return False

    total = sum(os.path.getsize(os.path.join(dest, s)) for s in shards)
    print(
        f"VERIFIED {dest}: model_type={cfg.get('model_type')} "
        f"arch={cfg.get('architectures')} vocab={cfg.get('vocab_size')} "
        f"n_layer={cfg.get('num_hidden_layers')} max_pos={cfg.get('max_position_embeddings')} "
        f"| {len(shards)} shard(s), {total / 2**30:.1f} GiB",
        flush=True,
    )
    print(f"    files: {sorted(os.listdir(dest))}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="weka staging root.")
    ap.add_argument("--repos", required=True, help="comma-separated Hub repo ids to stage.")
    args = ap.parse_args()

    # Xet-backed downloads turn one 429 into a hard failure of the whole call; the plain HTTP path
    # is slower but survives the retry loop above.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    os.makedirs(args.root, exist_ok=True)
    print(f"staging {len(repos)} model(s) -> {args.root}\n", flush=True)

    staged = [r for r in repos if stage(r, args.root)]
    print(flush=True)
    verified = [r for r in staged if verify(r, args.root)]

    failed = [r for r in repos if r not in verified]
    if failed:
        print(f"\nFAILED: {failed}", flush=True)
        return 1
    print(f"\nAll {len(verified)} model(s) staged and verified under {args.root}.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
