"""Convert the released ``allenai/Olmo-Hybrid-7B`` into an olmo-core distcp base for the CTC suite.

    python convert_olmo_hybrid_base.py --hf-src <dir> --out <dir>

── WHY NOT ``src/examples/huggingface/convert_checkpoint_from_hf.py`` ─────────────────────────
That script's ``--model-arch`` table maps names to :class:`TransformerConfig` *classmethods*, and
its state conversion goes through the dense ``state_mapping`` templates, which assume one HF prefix
per layer. A hybrid has two (``linear_attn.*`` and ``self_attn.*``) with *different norm placements*
on each, so the template path cannot express it -- which is exactly why olmo-core carries dedicated
``HYBRID_*_KEY_MAP`` tables for the export direction.

This script is the import direction, and it is **derived by inverting those same tables** rather
than by retyping them. A hand-copied second table is a table that drifts: the two would agree on the
day it was written and silently disagree after any upstream rename, and a mis-mapped norm does not
crash -- it loads and produces plausible garbage.

── THE ONE PLACE INVERSION IS NOT ENOUGH ─────────────────────────────────────────────────────
The export tables emit the *newer* HF spellings (``linear_attn.out_proj`` / ``linear_attn.norm``),
but the released Olmo-Hybrid-7B checkpoint uses the *older* ones (``linear_attn.o_proj`` /
``linear_attn.o_norm``) -- verified against its safetensors header. Both are accepted below.

── VALIDATION ────────────────────────────────────────────────────────────────────────────────
``load_state_dict(strict=True)`` is the real check: it fails on any missing, unexpected, or
wrong-shaped tensor, so a mapping error cannot reach a checkpoint. The parameter total is asserted
against the released 7,430,870,688 first, because that catches a *config* error (wrong head_dim, a
stray RoPE, a missing gate) which strict loading alone would report as a confusing key mismatch.
Run ``olmo3_parity_check.py`` against the result afterwards for the functional check -- CE on real
prose must land ~2-3 nats, not near ln(vocab)=11.5.
"""

import argparse
import json
import os
from typing import Dict

import torch
from olmo_hybrid_configs import (  # type: ignore[import-not-found]
    OLMO_HYBRID_LAYER_TYPES,
    OLMO_HYBRID_VOCAB_SIZE,
    olmo_hybrid_7B_ctc,
)

from olmo_core.distributed.checkpoint import save_model_and_optim_state
from olmo_core.nn.hf.convert import (
    HYBRID_ATTN_LAYER_KEY_MAP,
    HYBRID_GDN_LAYER_KEY_MAP,
    HYBRID_SHARED_KEY_MAP,
)

#: Released parameter count; the config must reproduce it exactly before any weight is touched.
EXPECTED_PARAMS = 7_430_870_688

#: HF spellings the released checkpoint uses where the export tables emit the newer alias.
_HF_ALIASES = {
    "linear_attn.out_proj.weight": "linear_attn.o_proj.weight",
    "linear_attn.norm.weight": "linear_attn.o_norm.weight",
}


def _hf_to_olmo_key(olmo_suffix: str, hf_suffix: str, layer: int, hf_state: Dict) -> str:
    """Resolve one layer-local mapping to a concrete HF key, honouring the spelling aliases.

    :param olmo_suffix: Layer-local olmo-core key, e.g. ``attention.w_q.weight``.
    :param hf_suffix: Layer-local HF key from the export table.
    :param layer: Layer index.
    :param hf_state: The loaded HF state dict, used to pick whichever alias is present.

    :returns: The full HF key.

    :raises KeyError: If neither the table's spelling nor its alias is in the checkpoint.
    """
    for cand in (hf_suffix, _HF_ALIASES.get(hf_suffix, hf_suffix)):
        full = f"model.layers.{layer}.{cand}"
        if full in hf_state:
            return full
    raise KeyError(
        f"layer {layer}: neither 'model.layers.{layer}.{hf_suffix}' nor its alias is present "
        f"(olmo-core key would be blocks.{layer}.{olmo_suffix})"
    )


def build_olmo_state(hf_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map a HF ``olmo_hybrid`` state dict onto olmo-core keys.

    :param hf_state: State dict loaded from the released safetensors.

    :returns: The olmo-core state dict.

    :raises KeyError: If a mapped tensor is absent from the checkpoint.
    """
    state: Dict[str, torch.Tensor] = {}
    for olmo_key, hf_key in HYBRID_SHARED_KEY_MAP.items():
        state[olmo_key] = hf_state[hf_key]

    for i, layer_type in enumerate(OLMO_HYBRID_LAYER_TYPES):
        table = (
            HYBRID_GDN_LAYER_KEY_MAP
            if layer_type == "linear_attention"
            else HYBRID_ATTN_LAYER_KEY_MAP
        )
        for olmo_suffix, hf_suffix in table.items():
            state[f"blocks.{i}.{olmo_suffix}"] = hf_state[
                _hf_to_olmo_key(olmo_suffix, hf_suffix, i, hf_state)
            ]
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-src", required=True, help="local dir holding the released HF checkpoint")
    ap.add_argument("--out", required=True, help="destination base dir (model_and_optim/ created)")
    ap.add_argument("--vocab-size", type=int, default=OLMO_HYBRID_VOCAB_SIZE)
    args = ap.parse_args()

    config = olmo_hybrid_7B_ctc(vocab_size=args.vocab_size)
    if config.num_params != EXPECTED_PARAMS:
        raise SystemExit(
            f"config builds {config.num_params:,} params but the released checkpoint has "
            f"{EXPECTED_PARAMS:,} -- the architecture is wrong; fix olmo_hybrid_configs.py before "
            "converting (loading weights into a wrong-shaped model is the failure that produces a "
            "plausible-looking but meaningless base)."
        )
    print(f"[convert] config OK: {config.num_params:,} params", flush=True)

    from safetensors.torch import load_file

    shards = sorted(f for f in os.listdir(args.hf_src) if f.endswith(".safetensors"))
    if not shards:
        raise SystemExit(f"no .safetensors under {args.hf_src}")
    hf_state: Dict[str, torch.Tensor] = {}
    for s in shards:
        hf_state.update(load_file(os.path.join(args.hf_src, s)))
    print(f"[convert] loaded {len(hf_state)} HF tensors from {len(shards)} shard(s)", flush=True)

    state = build_olmo_state(hf_state)
    print(f"[convert] mapped {len(state)} olmo-core tensors", flush=True)

    # ``init_device``, not ``device`` -- TransformerConfig.build() has no ``device`` kwarg.
    model = config.build(init_device="cpu")
    # strict=True is the validation: any missing / unexpected / wrong-shaped tensor stops here
    # rather than becoming a base that trains to nonsense.
    model.load_state_dict(state, strict=True)
    print("[convert] load_state_dict(strict=True) OK", flush=True)

    os.makedirs(args.out, exist_ok=True)
    save_model_and_optim_state(os.path.join(args.out, "model_and_optim"), model)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"model": config.as_config_dict()}, f, indent=1)
    print(f"[convert] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
