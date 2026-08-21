"""Export an **Olmo-Hybrid-7B** (3:1 GDN:full-attention) olmo-core checkpoint to HF for vLLM.

Sibling of :mod:`export_noswa_to_hf`, which covers the full-attention (noswa) arm of the same
pair. Everything downstream is shared: the export lands a directory that
``debug/ctc_vllm_validation/run_vllm_eval_generic.py`` can serve directly, so a hybrid ladder is
evaluated with the same three-step prefill/vLLM/grade pipeline as every other arm.

    PYTHONPATH=src python debug/ctc_olmo_hybrid/export_hybrid_to_hf.py <ckpt-dir> <out-dir>

``olmo_core.nn.hf`` already knows how to do the conversion itself -- ``convert_checkpoint_to_hf``
auto-detects the hybrid layout and routes to ``save_hf_hybrid_model``, which writes
``model_type: olmo_hybrid`` + ``architectures: [OlmoHybridForCausalLM]`` and remaps the state dict
with ``HYBRID_{GDN,ATTN,SHARED}_KEY_MAP``. Those maps are the exact inverse of the ones
``src/scripts/train/memexpress/ctc_suite/convert_olmo_hybrid_base.py`` used to bring the released
``allenai/Olmo-Hybrid-7B`` weights *into* olmo-core, so the round trip is lossless by construction.

This wrapper exists for the five things that are NOT handled, each of which is silent rather than
loud:

0. **Two GDN tensors are exported under names no consumer reads.**
   ``HYBRID_GDN_LAYER_KEY_MAP`` spells the Gated-DeltaNet output projection and output norm
   ``linear_attn.out_proj`` / ``linear_attn.norm``; the released ``allenai/Olmo-Hybrid-7B``
   checkpoint, transformers 5.x ``OlmoHybridGatedDeltaNet`` and vLLM's
   ``olmo_gdn_linear_attn.py`` all spell them ``linear_attn.o_proj`` / ``linear_attn.o_norm``.
   ``convert_olmo_hybrid_base.py`` already knows this -- its ``_HF_ALIASES`` table lists exactly
   these two -- but only for the *import* direction, so the export half still writes the dead
   spelling. This is an olmo-core bug, patched here at the dict rather than in the library so the
   fix ships with the thing that needs it; the real fix is those two values in
   ``src/olmo_core/nn/hf/convert.py``.

   Only *half* of it fails loudly. vLLM's ``track_weights_loading`` exempts every parameter whose
   module has a ``process_weights_after_loading`` quant method -- which is every ``*Linear`` --
   so the unloaded ``o_proj`` is **not** reported and the model silently runs 24 of its 32 layers
   with a randomly initialized output projection. Only ``o_norm`` (an ``RMSNormGated``, not a
   Linear) raises. Do not "fix" just the one the traceback names.

1. **``AutoConfig`` cannot read the file we just wrote.** ``convert_checkpoint_to_hf`` finishes by
   re-opening ``config.json`` through ``AutoConfig`` to stamp ``max_position_embeddings`` and the
   token ids. ``olmo_hybrid`` only exists in transformers >= 5.x, but the conversion has to run in
   the env that has ``olmo_core`` + ``cached_path`` (``corpus-reasoning-olmo``, transformers
   4.57.6), where that call dies with ``KeyError: 'olmo_hybrid'`` *after* 30 minutes of loading.
   A tiny JSON shim stands in for ``AutoConfig`` so the same fixup happens without the class.
2. **No dtype is written.** ``save_hf_hybrid_model`` writes raw JSON rather than going through
   ``save_pretrained``, so the export carries no ``dtype`` field. vLLM's ``dtype="auto"`` then
   resolves against a *missing* config dtype and picks float16 -- a different rounding mode than
   the native evaluator, which builds at bfloat16. That is a numerics difference dressed up as a
   modeling difference, so the dtype is stamped explicitly.
3. **NoPE is written in the pre-5.x spelling.** Olmo-Hybrid's full-attention layers have no RoPE
   (``rope=None`` in :mod:`olmo_hybrid_configs`), and ``get_hybrid_hf_config`` renders that as a
   top-level ``rope_theta: null``. vLLM reads ``config.rope_parameters["rope_theta"]``, which is
   also how the released checkpoint spells it, so the key is normalized here.
4. **The tokenizer.** As in ``export_noswa_to_hf``: the embedded tokenizer must be the PATCHED
   dolma2 marker copy (``<|extra_id_1|>``/``<|extra_id_2|>``, ids 100266/100267, renamed
   ``<|box_start|>``/``<|box_end|>``) that the shards were tokenized with. A wrong tokenizer does
   not crash; it produces plausible, wrong numbers.

Not a concern, though ``get_hybrid_hf_config`` warns about it in general: the warning fires when
the **GDN** blocks are ``ReorderedNormTransformerBlock``, and in this family they are not.
``olmo_hybrid_configs.olmo_hybrid_7B_ctc`` builds GDN blocks as plain pre-norm
``TransformerBlockType.default`` and only the *attention* blocks as ``reordered_norm`` -- which is
exactly the split HF ``olmo_hybrid`` implements (GDN: ``input_layernorm`` +
``post_attention_layernorm``; attention: ``post_attention_layernorm`` +
``post_feedforward_layernorm``). The export is checked for that and refuses to run if a future
config flips it.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


class _JsonConfigShim:
    """Stand-in for ``AutoConfig`` that reads/writes ``config.json`` as plain JSON.

    ``convert_checkpoint_to_hf``'s final fixup only ever sets a handful of attributes and calls
    ``save_pretrained``; doing that against the file directly keeps the hybrid export working on a
    transformers that has never heard of ``olmo_hybrid``.
    """

    def __init__(self, data: Dict[str, Any]):
        object.__setattr__(self, "_data", data)

    @classmethod
    def from_pretrained(cls, path, **_kwargs) -> "_JsonConfigShim":
        """Load ``<path>/config.json``.

        :param path: Directory holding the exported ``config.json``.
        """
        with open(Path(path) / "config.json") as f:
            return cls(json.load(f))

    def save_pretrained(self, path, **_kwargs) -> None:
        """Write the (possibly mutated) config back to ``<path>/config.json``."""
        with open(Path(path) / "config.json", "w") as f:
            json.dump(self._data, f, indent=2)

    def __getattr__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, "_data")[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        object.__getattribute__(self, "_data")[name] = value


#: Berkeley-local patched dolma2 marker tokenizer (see module docstring, point 4).
DEFAULT_TOKENIZER = "/scratch/users/prasann/hf_models/Olmo-3-1025-7B-docchunk"


def _finalize_config(out: str, dtype: str) -> Dict[str, Any]:
    """Apply the two post-conversion config repairs (dtype, NoPE spelling).

    :param out: The exported HF directory.
    :param dtype: The dtype the weights were actually written at.

    :returns: The final config dict, for logging.
    """
    cfg_path = Path(out) / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    # (2) vLLM's dtype="auto" needs a config dtype or it silently picks float16.
    cfg["dtype"] = dtype

    # (3) NoPE: `rope_theta: null` -> `rope_parameters: {"rope_theta": null}`, which is both what
    # vLLM's OlmoHybridAttention reads and how the released checkpoint spells it.
    if cfg.get("rope_theta", "missing") is None and "rope_parameters" not in cfg:
        cfg.pop("rope_theta")
        cfg["rope_parameters"] = {"rope_theta": None}

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


#: Per-layer tensor names the HF/vLLM ``olmo_hybrid`` implementations look for, by layer type.
_EXPECTED_SUFFIXES = {
    "linear_attention": {
        "linear_attn.q_proj.weight", "linear_attn.k_proj.weight", "linear_attn.v_proj.weight",
        "linear_attn.g_proj.weight", "linear_attn.a_proj.weight", "linear_attn.b_proj.weight",
        "linear_attn.o_proj.weight", "linear_attn.o_norm.weight",
        "linear_attn.q_conv1d.weight", "linear_attn.k_conv1d.weight",
        "linear_attn.v_conv1d.weight", "linear_attn.A_log", "linear_attn.dt_bias",
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    },
    "full_attention": {
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "post_attention_layernorm.weight", "post_feedforward_layernorm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    },
}


def _audit_keys(out: str, layer_types: list) -> None:
    """Check the exported tensor names against what HF/vLLM actually read.

    Worth doing explicitly because the failure it catches is quiet: vLLM's loader skips a weight
    whose name it does not recognize, and its post-load "not initialized" check exempts every
    Linear, so a misnamed projection produces a model that loads cleanly and generates garbage.

    :param out: The exported HF directory.
    :param layer_types: ``config["layer_types"]``.

    :raises AssertionError: If any layer is missing an expected tensor, or carries an unknown one.
    """
    from safetensors import safe_open

    with safe_open(Path(out) / "model.safetensors", framework="pt") as f:
        keys = set(f.keys())

    problems = []
    for i, lt in enumerate(layer_types):
        prefix = f"model.layers.{i}."
        got = {k[len(prefix):] for k in keys if k.startswith(prefix)}
        want = _EXPECTED_SUFFIXES[lt]
        if got - want:
            problems.append(f"layer {i} ({lt}) has UNKNOWN tensors: {sorted(got - want)}")
        if want - got:
            problems.append(f"layer {i} ({lt}) is MISSING tensors: {sorted(want - got)}")
    for shared in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        if shared not in keys:
            problems.append(f"missing shared tensor {shared}")
    assert not problems, "exported state dict does not match olmo_hybrid:\n  " + "\n  ".join(
        problems[:12]
    )
    print(f"[export] key audit OK ({len(keys)} tensors)", flush=True)


def main(ckpt: str, out: str, tokenizer_id: str, max_seq_len: int, dtype: str) -> None:
    """Convert one olmo-core hybrid distcp checkpoint to an HF ``olmo_hybrid`` directory.

    :param ckpt: Checkpoint dir holding ``config.json`` + ``model_and_optim/``.
    :param out: Output HF directory.
    :param tokenizer_id: Path/id of the PATCHED dolma2 marker tokenizer to embed in the export.
    :param max_seq_len: Written to ``max_position_embeddings``; should match the run's ``--seq-len``.
    :param dtype: Weight dtype to write. bfloat16 matches the native evaluator.

    :raises AssertionError: If the checkpoint is not hybrid, or its GDN blocks are post-norm (the
        one layout HF ``olmo_hybrid`` cannot represent).
    """
    from olmo_core.nn.hf import convert_checkpoint  # noqa: WPS433 -- patched below
    from olmo_core.nn.hf.convert import HYBRID_GDN_LAYER_KEY_MAP

    # (0) Correct the two GDN key spellings (see module docstring). Mutated in place so
    # convert_hybrid_state_to_hf, which reads the module global, picks them up.
    for olmo_suffix, correct_hf in (
        ("attention.w_out.weight", "linear_attn.o_proj.weight"),
        ("attention.o_norm.weight", "linear_attn.o_norm.weight"),
    ):
        stale = HYBRID_GDN_LAYER_KEY_MAP[olmo_suffix]
        if stale != correct_hf:
            HYBRID_GDN_LAYER_KEY_MAP[olmo_suffix] = correct_hf
            print(f"[export] GDN key fix: {stale!r} -> {correct_hf!r}")

    # (1) Neutralize the AutoConfig round trip; transformers 4.x has no `olmo_hybrid`.
    convert_checkpoint.AutoConfig = _JsonConfigShim  # type: ignore[assignment]

    from olmo_core.config import DType
    from olmo_core.nn.hf import convert_checkpoint_to_hf, load_config

    experiment_config = load_config(ckpt)
    assert experiment_config is not None, "no experiment config in checkpoint"
    model_cfg = experiment_config["model"]

    blocks = model_cfg.get("block")
    assert isinstance(blocks, dict) and "gdn" in blocks, (
        "checkpoint is not a named-block hybrid (expected a 'gdn' block); "
        f"got block keys {list(blocks) if isinstance(blocks, dict) else type(blocks)}"
    )
    # The one genuine correctness blocker get_hybrid_hf_config only *warns* about.
    gdn_block_type = blocks["gdn"].get("name", "default")
    assert gdn_block_type == "default", (
        f"GDN blocks are '{gdn_block_type}' (post-norm); HF olmo_hybrid implements pre-norm for "
        "linear_attention layers, so this checkpoint cannot be exported faithfully."
    )

    # Nothing is ever forward-passed during conversion, so drop the flash backend requirement --
    # it lets the export run on a CPU-only allocation and can't change a single weight.
    for name, block in blocks.items():
        mixer = block.get("sequence_mixer", {})
        if mixer.get("type") == "attention" or mixer.get("name") == "default":
            if "backend" in mixer:
                mixer["backend"] = "torch"
                mixer["use_flash"] = False
                print(f"[export] block {name!r}: attention backend -> torch (weights-only)")

    convert_checkpoint_to_hf(
        original_checkpoint_path=ckpt,
        output_path=out,
        transformer_config_dict=model_cfg,
        tokenizer_config_dict=experiment_config.get("dataset", {}).get("tokenizer") or {},
        tokenizer_id=tokenizer_id,
        max_sequence_length=max_seq_len,
        dtype=DType(dtype),
        validate=False,
    )

    cfg = _finalize_config(out, dtype)
    _audit_keys(out, cfg["layer_types"])
    n_full = sum(1 for t in cfg["layer_types"] if t == "full_attention")
    n_lin = sum(1 for t in cfg["layer_types"] if t == "linear_attention")
    print(
        f"EXPORT-OK -> {out}\n"
        f"  arch={cfg['architectures']} model_type={cfg['model_type']} dtype={cfg['dtype']}\n"
        f"  layers: {n_lin} linear_attention / {n_full} full_attention\n"
        f"  rope_parameters={cfg.get('rope_parameters')} vocab_size={cfg['vocab_size']}\n"
        f"  eos={cfg['eos_token_id']} pad={cfg['pad_token_id']} "
        f"max_position_embeddings={cfg['max_position_embeddings']}",
        flush=True,
    )
    tok_files = sorted(p.name for p in Path(out).glob("token*"))
    print(f"  tokenizer files: {tok_files} (from {tokenizer_id})", flush=True)
    print(f"  weights: {os.path.getsize(Path(out) / 'model.safetensors') / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--max-seq-len", type=int, default=40960)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()
    main(args.ckpt, args.out, args.tokenizer, args.max_seq_len, args.dtype)
