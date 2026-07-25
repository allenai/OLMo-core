"""Export a trained olmo-core Qwen3-dense checkpoint -> HF format for eval with evaluate.py.

olmo-core's stock `convert_checkpoint_to_hf()` routes through `get_hf_config()`, which only
knows OLMo-family architectures (Olmo2/Olmo3/FlexOlmo/hybrid) and raises
`NotImplementedError: Block is not a ReorderedNormTransformerBlock` for a Qwen3 dense model.
The *weight* conversion, however, has a registered `qwen3` mapping (`convert_state_to_hf` keyed
on `model_type`). So we do the conversion ourselves: load the distcp weights into the olmo
model, then map them onto a real Qwen3 HF config built from --base-model.

Auto-selects the latest step<N> dir under the save folder.
"""

import argparse
import glob
import os
import re

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.io import join_path
from olmo_core.nn.hf.convert import convert_state_to_hf

from corpus_reasoning.lib.olmo_models import build_transformer_config, resolve_olmo_model


def latest_step_dir(save_folder: str) -> str:
    steps = []
    for d in glob.glob(os.path.join(save_folder, "step*")):
        m = re.search(r"step(\d+)$", d)
        if m and os.path.isdir(d):
            steps.append((int(m.group(1)), d))
    if not steps:
        raise SystemExit(f"no step<N> checkpoint dir under {save_folder}")
    return max(steps)[1]


def apply_rope_overrides(cfg, rope_theta: float, max_pos: int, label: str):
    """Stamp train-time RoPE/context overrides onto an HF config in place.

    The exported config is built from the *base* HF model, so a run that trained with an NTK
    ``rope_theta`` override (or beyond the base's ``max_position_embeddings``) would otherwise
    export with the base's values -- a silent train/eval RoPE mismatch.

    :param cfg: The HF config object to mutate.
    :param rope_theta: The run's ``--rope-theta``; 0 leaves the base value alone.
    :param max_pos: The run's context length; 0 leaves the base value alone.
    :param label: Config name used in the log line.
    """
    if rope_theta and rope_theta > 0:
        print(f"[export] {label}: rope_theta {getattr(cfg, 'rope_theta', None)} -> {rope_theta}",
              flush=True)
        cfg.rope_theta = rope_theta
    if max_pos and max_pos > 0:
        print(f"[export] {label}: max_position_embeddings "
              f"{getattr(cfg, 'max_position_embeddings', None)} -> {max_pos}", flush=True)
        cfg.max_position_embeddings = max_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-folder", required=True, help="olmo-core SFT save folder (contains step<N>/)")
    ap.add_argument("--hf-out", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Base",
                    help="HF base model the run started from (resolves builder/vocab/tokenizer/config)")
    ap.add_argument("--ckpt", default=None, help="explicit step dir; else latest under --save-folder")
    ap.add_argument("--peft-out", default=None,
                    help="for LoRA runs: also write a standalone PEFT adapter here "
                         "(default <hf-out>_adapter). Ignored for full-FT checkpoints.")
    # The HF config is built from --base-model, so any RoPE/context knob the RUN overrode at
    # train time (train_ctc_suite.py --rope-theta, the NTK context extension) is silently lost
    # here -- and `inv_freq` is recomputed from the stale theta, so the exported model's RoPE
    # would not match the one it was trained with. Long-context evals then read as a modeling
    # failure. Pass the SAME values the run trained with.
    ap.add_argument("--rope-theta", type=float, default=0.0,
                    help="override rope_theta in the exported HF config; MUST match the run's "
                         "--rope-theta (NTK context extension). 0 = keep the base value.")
    ap.add_argument("--max-position-embeddings", type=int, default=0,
                    help="override max_position_embeddings in the exported HF config; set to the "
                         "run's --seq-len for long-context checkpoints (the base caps at 32k, "
                         "which makes vLLM refuse/truncate longer prompts). 0 = keep the base.")
    args = ap.parse_args()

    ckpt = args.ckpt or latest_step_dir(args.save_folder)
    print(f"[export] olmo ckpt: {ckpt}\n[export] hf out: {args.hf_out}", flush=True)

    # LoRA checkpoints carry a sidecar (r/alpha/target/base_model) written by olmo_train.py.
    # When present, the distcp contains base + adapter params: we rebuild a matching skeleton,
    # emit a standalone PEFT adapter, then merge the adapter into the base for the HF export.
    from corpus_reasoning.lib import olmo_lora
    lora = olmo_lora.read_lora_sidecar(args.save_folder)
    if lora is not None:
        args.base_model = lora["base_model"]  # the base the run actually started from
        print(f"[export] LoRA checkpoint: r={lora['r']} alpha={lora['alpha']} "
              f"target={lora['target']} base={args.base_model}", flush=True)

    # 1) Build the olmo-core model skeleton and load the trained distcp weights (single process).
    spec = resolve_olmo_model(args.base_model)
    model_cfg = build_transformer_config(spec)
    model = model_cfg.build()
    if lora is not None:
        # Inject BEFORE to_empty so the adapter params get materialized and the strict
        # distcp load (base + adapter) populates them with the trained values.
        olmo_lora.inject_lora(model, r=lora["r"], alpha=lora["alpha"],
                              dropout=lora.get("dropout", 0.0), target=lora["target"])
    model.to_empty(device=torch.device("cpu"))
    model_and_optim_dir = join_path(ckpt, "model_and_optim")
    print(f"[export] loading distcp weights from {model_and_optim_dir}", flush=True)
    load_model_and_optim_state(model_and_optim_dir, model)

    if lora is not None:
        # Emit the standalone PEFT adapter from the trained adapter params (before merging),
        # then fold the adapter into the base weights for a normal full-model HF export.
        peft_out = args.peft_out or (args.hf_out.rstrip("/") + "_adapter")
        olmo_lora.write_peft_adapter(
            peft_out, olmo_lora.lora_peft_state_dict(model),
            base_model=args.base_model, r=lora["r"], alpha=lora["alpha"],
            dropout=lora.get("dropout", 0.0), target=lora["target"])
        n_merged = olmo_lora.merge_lora(model)
        print(f"[export] wrote PEFT adapter -> {peft_out}; merged {n_merged} adapters "
              f"into base weights", flush=True)

    olmo_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # --- hybrid Qwen3.5: inverse converter -> text-only Qwen3_5ForCausalLM (needs the
    # transformers-with-qwen3_5 shadow on PYTHONPATH). No olmo->HF mapping in olmo-core. ---
    if getattr(spec, "is_hybrid", False):
        import glob as _glob, json as _json, types as _types
        from huggingface_hub import snapshot_download
        from corpus_reasoning.lib.qwen35_to_hf import convert_qwen3_5_state_to_hf
        # Accept a local snapshot DIRECTORY as well as a hub id. On the Berkeley cluster the base
        # is staged node-local (/data/.../Qwen3.5-4B-Base) and is often absent from the HF cache
        # entirely, where snapshot_download() under HF_HUB_OFFLINE=1 either fails or stalls on NFS.
        # Every other from_pretrained() call here already accepts a path; only this one did not.
        snap = args.base_model if os.path.isdir(args.base_model) else snapshot_download(args.base_model)
        raw = _json.load(open(os.path.join(snap, "config.json")))
        text_ns = _types.SimpleNamespace(**raw["text_config"])
        print(f"[export] qwen3_5 inverse-convert {len(olmo_state)} tensors -> HF text decoder", flush=True)
        hf_state = {k: v.contiguous() for k, v in convert_qwen3_5_state_to_hf(olmo_state, text_ns).items()}
        import transformers
        TextCfg = transformers.AutoConfig.from_pretrained(args.base_model).text_config
        # Emit the trained output head explicitly (qwen35_to_hf writes lm_head.weight from
        # olmo w_out) and DON'T tie: tie_weights() would copy embed_tokens over the trained
        # head. Mark the config untied so the eval-time load also keeps the trained head.
        TextCfg.tie_word_embeddings = False
        apply_rope_overrides(TextCfg, args.rope_theta, args.max_position_embeddings,
                             "qwen3_5 text_config")
        hf_model = transformers.Qwen3_5ForCausalLM(TextCfg)
        missing, unexpected = hf_model.load_state_dict(hf_state, strict=False)
        missing = [m for m in missing if "rotary" not in m and "inv_freq" not in m]
        if missing or unexpected:
            raise SystemExit(f"[export] qwen3_5 state mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
        os.makedirs(args.hf_out, exist_ok=True)
        hf_model.save_pretrained(args.hf_out)
        AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.hf_out)
        print(f"[export] qwen3_5 DONE -> {args.hf_out}", flush=True)
        return

    # 2) Real Qwen3 HF config (model_type='qwen3' -> selects the qwen3 olmo->hf weight mapping).
    hf_config = AutoConfig.from_pretrained(args.base_model)
    # Before any weight mapping: the rotary `inv_freq` buffer is built from hf_config at
    # from_config() time below, so the override has to land here, not after.
    apply_rope_overrides(hf_config, args.rope_theta, args.max_position_embeddings, "hf_config")
    print(f"[export] converting {len(olmo_state)} tensors olmo-core -> HF (model_type={hf_config.model_type})", flush=True)
    hf_state = convert_state_to_hf(hf_config, olmo_state)

    # olmo-core ships a qwen3 norm override for the *from_hf* direction only; the *to_hf*
    # converter (MODEL_TYPE_SPECIFIC_OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS) has no qwen3 entry, so
    # it falls back to OLMo2 post-norm key names. Remap to Qwen3 pre-norm names -- the exact
    # inverse of the validated from_hf qwen3 mapping:
    #   olmo attention_norm   -> HF input_layernorm          (pre-attention)
    #   olmo feed_forward_norm -> HF post_attention_layernorm (pre-MLP)
    if hf_config.model_type == "qwen3":
        remapped = {}
        for k, v in hf_state.items():
            if ".post_attention_layernorm.weight" in k:
                k = k.replace(".post_attention_layernorm.weight", ".input_layernorm.weight")
            elif ".post_feedforward_layernorm.weight" in k:
                k = k.replace(".post_feedforward_layernorm.weight", ".post_attention_layernorm.weight")
            remapped[k] = v
        hf_state = remapped

    hf_state = {k: v.contiguous() for k, v in hf_state.items()}

    # 3) Instantiate Qwen3ForCausalLM from config, load mapped weights, save.
    # Build on CPU with real init (NOT meta/empty) so non-persistent buffers like the rotary
    # `inv_freq` -- which aren't in the state dict -- are correctly computed; load_state_dict
    # (assign=False) then overwrites the trained weights in place, leaving those buffers intact.
    hf_model = AutoModelForCausalLM.from_config(hf_config)
    missing, unexpected = hf_model.load_state_dict(hf_state, strict=False)
    # tie_word_embeddings: lm_head.weight may be tied to embed_tokens and absent from state.
    missing = [m for m in missing if "lm_head" not in m]
    if missing or unexpected:
        raise SystemExit(f"[export] state_dict mismatch: missing={missing} unexpected={unexpected}")
    hf_model.tie_weights()

    os.makedirs(args.hf_out, exist_ok=True)
    hf_model.save_pretrained(args.hf_out)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.hf_out)
    print(f"[export] DONE -> {args.hf_out}", flush=True)


if __name__ == "__main__":
    main()
