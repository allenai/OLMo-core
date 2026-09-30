"""Build the OLMo 3.5 small hero LM from its checkpoint config and load its master weights.

Tree-agnostic: it only uses public olmo-core config classes plus the legacy-key normalizer, so
the same module runs against the text pin, the ``vision`` baseline and our stack via PYTHONPATH.

Local (A100) mode: the two attention layers use the ``torch`` backend and KDA uses the FLA
kernels. The weights are identical to the production ``flash_4`` / kernel-fun path.
"""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Any, Dict, Optional

import torch

CKPT = "/weka/olmo-3p5-checkpoints/protected/olmo35-small-hero-20260907/emo/8T"


def load_config(checkpoint: str = CKPT) -> Dict[str, Any]:
    with open(os.path.join(checkpoint, "config.json")) as f:
        return json.load(f)


def normalized_model_config_dict(
    model: Dict[str, Any],
    *,
    emo: bool = False,
    attention_backend: Optional[str] = "torch",
    experimental_kernels: Optional[bool] = False,
) -> Dict[str, Any]:
    """Return a copy of ``config.json["model"]`` that decodes on current olmo-core."""
    from olmo_core.nn.hf.convert_checkpoint import _normalize_legacy_latent_moe_config

    model = copy.deepcopy(model)
    out = _normalize_legacy_latent_moe_config(model)
    if out is not None:
        model = out
    blocks = [model["block"], *(model.get("block_overrides") or {}).values()]
    for block in blocks:
        mixer = block.get("sequence_mixer") or block.get("attention") or {}
        if experimental_kernels is not None and "use_experimental_kernels" in mixer:
            mixer["use_experimental_kernels"] = experimental_kernels
        if attention_backend is not None and "backend" in mixer:
            mixer["backend"] = attention_backend
        router = block.get("routed_experts_router")
        if router is not None and not emo:
            router["emo"] = None
    return model


def build_model(
    config: Dict[str, Any],
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    **normalize_kwargs,
):
    from olmo_core.nn.transformer.config import OLMoDDPModelConfig

    model_cfg = OLMoDDPModelConfig.from_dict(
        normalized_model_config_dict(config["model"], **normalize_kwargs)
    )
    model = model_cfg.build(init_device="meta")
    model = model.to_empty(device=device)
    model = model.to(dtype)
    return model_cfg, model


def load_main_weights(model: torch.nn.Module, checkpoint: str = CKPT, prefix: str = "module.") -> Dict[str, Any]:
    """Load ``{prefix}{name}.main`` fp32 masters from the distributed checkpoint into ``model``.

    Returns a coverage report: parameters loaded, parameters without a key, unused keys.
    """
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint import FileSystemReader

    class _Reader(FileSystemReader):
        """The checkpoint's storage entries predate the ``transform_descriptors`` slot that newer
        torch readers expect; fill it in after every metadata read."""

        def read_metadata(self):
            metadata = super().read_metadata()
            for info in metadata.storage_data.values():
                if not hasattr(info, "transform_descriptors"):
                    try:
                        object.__setattr__(info, "transform_descriptors", None)
                    except AttributeError:
                        pass
            return metadata

    reader = _Reader(os.path.join(checkpoint, "model_and_optim"))
    metadata = reader.read_metadata()
    available = {
        k[len(prefix) : -len(".main")]: tuple(v.size)
        for k, v in metadata.state_dict_metadata.items()
        if k.startswith(prefix) and k.endswith(".main")
    }
    params = dict(model.named_parameters())
    missing = [n for n in params if n not in available]
    unused = [n for n in available if n not in params]
    mismatch = [n for n in params if n in available and int(torch.tensor(available[n]).prod()) != params[n].numel()]
    t0 = time.time()
    # Load in groups to bound host memory (fp32 masters total ~50 GB).
    names = [n for n in params if n in available and n not in mismatch]
    group: Dict[str, torch.Tensor] = {}
    group_bytes = 0
    loaded = 0

    def flush():
        nonlocal group, group_bytes, loaded
        if not group:
            return
        dist_cp.load(group, storage_reader=reader, no_dist=True)
        with torch.no_grad():
            for key, flat in group.items():
                name = key[len(prefix) : -len(".main")]
                params[name].copy_(flat.view(params[name].shape).to(params[name].dtype))
        loaded += len(group)
        group = {}
        group_bytes = 0

    for name in names:
        numel = params[name].numel()
        group[f"{prefix}{name}.main"] = torch.empty(numel, dtype=torch.float32)
        group_bytes += numel * 4
        if group_bytes > 8 * 2**30:
            flush()
    flush()
    return {
        "loaded": loaded,
        "missing": missing,
        "unused": unused,
        "mismatch": mismatch,
        "seconds": round(time.time() - t0, 1),
    }


def text_batch(
    config: Dict[str, Any],
    *,
    batch_size: int = 2,
    seq_len: int = 2048,
    file_index: int = 0,
    offset_tokens: int = 1_000_000,
    device: str = "cuda",
) -> torch.Tensor:
    """Deterministic token windows from the checkpoint's own pretraining data (uint32 numpy)."""
    import numpy as np

    with open(os.path.join(CKPT, "data_paths.txt")) as f:
        paths = [line.strip() for line in f if line.strip()]
    path = paths[file_index]
    dtype = np.uint16 if config["dataset"].get("dtype") == "uint16" else np.uint32
    mm = np.memmap(path, dtype=dtype, mode="r")
    rows = []
    for b in range(batch_size):
        start = offset_tokens + b * seq_len
        rows.append(torch.from_numpy(np.array(mm[start : start + seq_len], dtype=np.int64)))
    return torch.stack(rows).to(device)

def init_single_process_group(backend: str = "gloo"):
    """A world-size-1 process group, so routers with ``global_load_balancing`` can run locally
    without the DDP wrapper (global == local load balancing at world size 1)."""
    import torch.distributed as dist

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29571")
        dist.init_process_group(backend=backend, rank=0, world_size=1)
    group = dist.group.WORLD
    return group


def set_router_groups(model, group) -> int:
    n = 0
    for block in model.modules():
        router = getattr(block, "routed_experts_router", None)
        if router is not None and hasattr(router, "set_load_balancing_process_group"):
            router.set_load_balancing_process_group(group)
            n += 1
    return n
