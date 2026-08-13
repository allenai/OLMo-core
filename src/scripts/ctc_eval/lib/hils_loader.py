"""
Load a HiLS-Attention HF checkpoint (``tencent/HiLS-Attention-7B`` and friends).

HiLS is a chunk-wise sparse attention published as `HiLS-Attention
<https://github.com/abertsch72/HiLS-Attention>`_ (upstream:
``Tencent-Hunyuan/HiLS-Attention``). The released checkpoint is an HF-format model, but it is
**not** loadable with plain ``AutoModelForCausalLM``:

* the repo ships **no** ``auto_map``, so ``trust_remote_code=True`` finds nothing to load;
* ``model_type`` is ``olmo_hils``, which no installed transformers version knows;
* the modeling code lives out-of-tree in the HiLS repo (``models/FlashHiLS/``) and pulls in
  ``tilelang`` (JIT CUDA kernels for the chunk-pool / sliding-window attention) and ``veomni``.

So the sequence is: put the HiLS repo on ``sys.path``, import its ``HiLSConfig`` +
``HiLSForCausalLM``, register them into the Auto* factories, and only then call
``from_pretrained``. That is what :func:`load_hils_model` does, and it is the single place that
knows it -- the eval harness and the smoke test both go through here so they cannot drift.

Set ``HILS_REPO`` to the checked-out HiLS repo (``hils_env_setup.sh`` does this).
"""

import json
import os
import sys
from typing import Any, Optional

# The two model families the HiLS repo implements, keyed by the ``model_type`` in config.json.
# ``olmo_hils`` is what the released 7B (CPT'd from allenai/Olmo-3-1025-7B) carries.
_MODULE_BY_MODEL_TYPE = {
    "olmo_hils": "models.FlashHiLS.modeling_olmo_hils",
    "qwen_hils": "models.FlashHiLS.modeling_qwen_hils",
}


def is_hils_checkpoint(path: str) -> bool:
    """
    Report whether a local checkpoint dir is a HiLS model.

    :param path: Directory containing ``config.json``.

    :returns: ``True`` if the config's ``model_type`` names a HiLS variant.
    """
    cfg_path = os.path.join(path, "config.json")
    if not os.path.exists(cfg_path):
        return False
    with open(cfg_path) as fh:
        return "hils" in str(json.load(fh).get("model_type", ""))


def hils_repo_path(explicit: Optional[str] = None) -> str:
    """
    Resolve the checked-out HiLS repo.

    :param explicit: An explicit path, or ``None`` to read ``$HILS_REPO``.

    :returns: The repo root.

    :raises RuntimeError: If no repo path is set or it does not look like the HiLS repo.
    """
    repo = explicit or os.environ.get("HILS_REPO", "")
    if not repo:
        raise RuntimeError(
            "HiLS checkpoint requested but $HILS_REPO is unset. Run hils_env_setup.sh "
            "(src/scripts/train/memexpress/hils_eval/) first, or pass --hils-repo."
        )
    if not os.path.isdir(os.path.join(repo, "models", "FlashHiLS")):
        raise RuntimeError(f"$HILS_REPO={repo} has no models/FlashHiLS -- not a HiLS checkout.")
    return repo


def register_hils(path: str, repo: Optional[str] = None) -> Any:
    """
    Put the HiLS repo on ``sys.path`` and register its config/model classes with transformers.

    :param path: The checkpoint dir (its ``model_type`` picks the OLMo vs Qwen implementation).
    :param repo: The HiLS repo root; defaults to ``$HILS_REPO``.

    :returns: The ``HiLSForCausalLM`` class that was registered.

    :raises RuntimeError: If the checkpoint's ``model_type`` is not a known HiLS variant.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    repo = hils_repo_path(repo)

    # tilelang JIT-compiles its kernels on first use and caches them under
    # $TILELANG_CACHE_DIR (default ~/.tilelang/cache). Under an 8-way `torchrun` every rank
    # imports the same kernels at the same moment against the same $HOME, so they race to write
    # the same cache entries. Give each local rank its own cache: the duplicated compile is a
    # one-off cost per job, while a corrupted shared cache is a mid-sweep failure that looks like
    # a model bug. MUST be set before the HiLS modules import tilelang, which is why it lives
    # here rather than in the caller.
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None and "TILELANG_CACHE_DIR" not in os.environ:
        os.environ["TILELANG_CACHE_DIR"] = f"/tmp/tilelang_cache_rank{local_rank}"
        print(f"[hils] TILELANG_CACHE_DIR={os.environ['TILELANG_CACHE_DIR']}", flush=True)

    # The HiLS modules import each other as top-level packages (``from ops...``, ``from utils...``),
    # so the repo ROOT has to be importable, not just models/.
    if repo not in sys.path:
        sys.path.insert(0, repo)

    with open(os.path.join(path, "config.json")) as fh:
        model_type = str(json.load(fh).get("model_type", ""))
    module_name = _MODULE_BY_MODEL_TYPE.get(model_type)
    if module_name is None:
        raise RuntimeError(
            f"{path} has model_type={model_type!r}, which is not a known HiLS variant "
            f"({sorted(_MODULE_BY_MODEL_TYPE)})."
        )

    import importlib

    from models.FlashHiLS.configuration_hils import HiLSConfig  # type: ignore[import-not-found]

    hils_cls = importlib.import_module(module_name).HiLSForCausalLM

    # register() raises if the model_type is already taken; a second call in the same process
    # (e.g. two rungs in one job) must be a no-op rather than a crash.
    try:
        AutoConfig.register(model_type, HiLSConfig)
    except ValueError:
        pass
    hils_cls.config_class = HiLSConfig
    try:
        AutoModelForCausalLM.register(HiLSConfig, hils_cls)
    except ValueError:
        pass
    print(f"[hils] registered {module_name}.HiLSForCausalLM for model_type={model_type}", flush=True)
    return hils_cls


def load_hils_model(
    path: str,
    *,
    device: Any,
    attn_implementation: Optional[str] = None,
    repo: Optional[str] = None,
    max_position_embeddings: Optional[int] = None,
) -> Any:
    """
    Build a HiLS model from a local checkpoint dir, ready for ``.generate()``.

    :param path: Local checkpoint dir (weka-staged; never a Hub id -- see
        ``stage_hf_models_weka.py`` for why eval jobs must not hit the Hub).
    :param device: Torch device to place the model on.
    :param attn_implementation: Override for the DENSE layers' attention kernel. The checkpoint
        ships ``flash_attention_3``, which needs FA3 to be installed; pass ``flash_attention_2``
        or ``sdpa`` where it is not. This does **not** change the HiLS sparse path, which is
        always the repo's tilelang kernels -- it only selects the kernel for the interleaved
        full-attention layers, so the choice is a speed/precision one, not a semantic one.
    :param repo: The HiLS repo root; defaults to ``$HILS_REPO``.
    :param max_position_embeddings: Raise the config's position budget (the released 7B ships
        131072; HoPE is designed to extrapolate past it, and the ladder's xlong rungs go
        further). ``None`` leaves the checkpoint value alone.

    :returns: The loaded model in eval mode.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    register_hils(path, repo)

    config = AutoConfig.from_pretrained(path)
    if max_position_embeddings is not None:
        prev = getattr(config, "max_position_embeddings", None)
        if prev is None or max_position_embeddings > prev:
            config.max_position_embeddings = max_position_embeddings
            print(f"[hils] max_position_embeddings {prev} -> {max_position_embeddings}", flush=True)
    if attn_implementation:
        config._attn_implementation = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        path,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation or config._attn_implementation,
    )
    model.to(device)
    model.eval()
    return model
