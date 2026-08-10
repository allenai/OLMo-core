"""
Stage HF tokenizers onto weka so eval jobs never have to reach huggingface.co at run time.

Every native eval calls ``AutoTokenizer.from_pretrained(<hub repo id>)`` at startup. That is a
live network call: when the Hub is briefly unreachable -- or the node's HF cache is cold, which it
reliably is for a repo the cluster hasn't pulled before -- the job dies immediately with::

    OSError: We couldn't connect to 'https://huggingface.co' to load the files,
             and couldn't find them in the cached files.

This has cost whole sweeps (2026-08-10: 20 of 30 eval jobs, most of them minutes after launch, and
some after hours of GPU work when a preempted job auto-resumed into the outage). Staging the
tokenizers to a weka path that every job already mounts removes the dependency entirely.

Run once (or whenever a new model family is added), then point ``TOKENIZER`` at the staged copy::

    gantry run --name stage-tokenizers-weka -w ai2/flex2 -b ai2/oe-other \\
      --cluster ai2/jupiter --gpus 0 --priority urgent \\
      --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \\
      --weka oe-training-default:/weka/oe-training-default \\
      --env HF_HUB_DISABLE_XET=1 --install true --timeout 0 --yes -- \\
      python src/scripts/data/stage_tokenizers_weka.py

Layout: ``<root>/<repo id with '/' -> '__'>``, e.g.
``/weka/oe-training-default/amandab/tokenizers/Qwen__Qwen3.5-0.8B``. The eval runner resolves this
path itself (see ``run_beaker_multirung_eval.sh``), so the mapping must stay in sync with it.
"""

import argparse
import os
import sys
import time

# The tokenizers our evals actually load. Qwen3 and Qwen3.5 do NOT share a vocabulary (151936 vs
# 248320), and feeding a checkpoint the wrong one does not error -- it silently scores ~0 on every
# task (see OLMo-core 555bb5069). Keep both staged so neither family can fall back to the network.
DEFAULT_REPOS = [
    "Qwen/Qwen3.5-0.8B",  # Qwen3.5 family eval tokenizer (vocab 248320)
    "Qwen/Qwen3-4B",  # Qwen3 family eval tokenizer (vocab 151936)
]

DEFAULT_ROOT = "/weka/oe-training-default/amandab/tokenizers"


def local_dir(root: str, repo: str) -> str:
    """
    Map a Hub repo id to its staged directory.

    :param root: The staging root, e.g. ``/weka/oe-training-default/amandab/tokenizers``.
    :param repo: The Hub repo id, e.g. ``Qwen/Qwen3.5-0.8B``.

    :returns: The absolute directory the tokenizer is staged to.
    """
    return os.path.join(root, repo.replace("/", "__"))


# Copy the Hub's files VERBATIM rather than round-tripping through
# ``AutoTokenizer.save_pretrained``. Re-serializing rewrites tokenizer_config.json in whatever
# transformers version happens to be running, and the result is not portable: a copy saved by
# transformers 5.10 set ``extra_special_tokens`` to null, and the older transformers in the beaker
# image then died with "'list' object has no attribute 'keys'" while loading it. save_pretrained
# also drops vocab.json/merges.txt. Shipping the raw files means the staged copy is byte-identical
# to what the job would have downloaded itself.
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
]


def stage(repo: str, root: str, attempts: int = 6) -> bool:
    """
    Download one tokenizer's raw Hub files into ``root``, retrying transient Hub failures.

    A single 429/503 aborts the whole download, so the retry loop is not optional -- it is the
    reason this script exists rather than a one-line ``snapshot_download``.

    :param repo: The Hub repo id to fetch.
    :param root: The staging root directory.
    :param attempts: How many times to try before giving up.

    :returns: ``True`` if the tokenizer was staged successfully.
    """
    import shutil

    from huggingface_hub import snapshot_download

    dest = local_dir(root, repo)
    for attempt in range(1, attempts + 1):
        try:
            snapshot_download(repo, allow_patterns=TOKENIZER_FILES, local_dir=dest)
            # snapshot_download leaves a .cache/ dir of download metadata next to the files.
            shutil.rmtree(os.path.join(dest, ".cache"), ignore_errors=True)
            print(f"SAVED {repo} -> {dest}  ({sorted(os.listdir(dest))})", flush=True)
            return True
        except Exception as e:  # noqa: BLE001 -- any Hub/network error is worth retrying
            print(f"  attempt {attempt}/{attempts} for {repo}: {type(e).__name__}: {e}", flush=True)
            if attempt < attempts:
                time.sleep(10 * attempt)
    print(f"GAVE UP on {repo}", flush=True)
    return False


def verify(repo: str, root: str) -> bool:
    """
    Reload a staged tokenizer from disk to prove the copy is usable offline.

    Runs with ``HF_HUB_OFFLINE=1`` set, so a copy that still secretly reaches the Hub fails here
    rather than in a sweep six hours later.

    :param repo: The Hub repo id that was staged.
    :param root: The staging root directory.

    :returns: ``True`` if the staged copy loads.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    dest = local_dir(root, repo)
    try:
        tok = AutoTokenizer.from_pretrained(dest)
    except Exception as e:  # noqa: BLE001
        print(f"VERIFY FAILED {dest}: {type(e).__name__}: {e}", flush=True)
        return False
    print(
        f"VERIFIED {dest}: vocab={len(tok)} eos={tok.eos_token_id} pad={tok.pad_token_id}",
        flush=True,
    )
    print(f"    files: {sorted(os.listdir(dest))}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="weka staging root.")
    ap.add_argument(
        "--repos",
        default=",".join(DEFAULT_REPOS),
        help="comma-separated Hub repo ids to stage.",
    )
    args = ap.parse_args()

    # Xet-backed downloads turn one 429 into a hard failure of the whole call; the plain HTTP path
    # is slower but survives the retry loop above.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    os.makedirs(args.root, exist_ok=True)
    print(f"staging {len(repos)} tokenizer(s) -> {args.root}\n", flush=True)

    staged = [r for r in repos if stage(r, args.root)]
    print(flush=True)
    verified = [r for r in staged if verify(r, args.root)]

    failed = [r for r in repos if r not in verified]
    if failed:
        print(f"\nFAILED: {failed}", flush=True)
        return 1
    print(f"\nAll {len(verified)} tokenizer(s) staged and verified under {args.root}.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
