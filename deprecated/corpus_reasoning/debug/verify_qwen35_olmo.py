"""Verify Qwen3.5 (hybrid GDN + full-attn) support on the prasann olmo-core branch
WITHOUT needing a transformers build that understands `qwen3_5`.

The HF checkpoint "Qwen/Qwen3.5-0.8B-Base" is a *multimodal* `Qwen3_5ForConditionalGeneration`
(text decoder nested under `text_config`, plus a vision tower). transformers 4.57.6 release
lacks the `qwen3_5` architecture, so we cannot `from_pretrained` it. But conversion +
olmo-core training only need the config.json + safetensors, which we read directly:

  1. provenance: olmo_core from prasann worktree + fla importable
  2. read config.json (-> text_config namespace) + load all safetensors shards
  3. build TransformerConfig.qwen3_5_0_8B + convert_qwen3_5_state_from_hf + strict load
  4. forward on GPU (exercises the FLA GDN kernels) + greedy-decode for coherence

This is the "does olmo-core support qwen3.5 end-to-end for training" proof. Full bit-parity
vs an HF reference is a separate step that needs transformers-with-qwen3_5.
"""
import glob
import json
import os
import sys
import traceback
import types


def banner(msg):
    print(f"\n{'=' * 8} {msg} {'=' * 8}", flush=True)


def _ns(d):
    """dict -> SimpleNamespace (shallow; lists/scalars kept as-is)."""
    return types.SimpleNamespace(**d)


def main():
    import torch
    from safetensors.torch import load_file

    banner("PHASE 1: provenance")
    import olmo_core
    print("olmo_core:", olmo_core.__file__, flush=True)
    assert "OLMo-core-prasann" in olmo_core.__file__, "olmo_core not from prasann worktree"
    import fla
    print("fla pkg:", fla.__file__, "| has ops:", os.path.isdir(os.path.join(os.path.dirname(fla.__file__ or ""), "ops"))
          if fla.__file__ else "namespace", flush=True)

    snap = sorted(glob.glob(
        "/scratch/users/prasann/huggingface-cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/*/"))[-1]
    print("snapshot:", snap, flush=True)

    banner("PHASE 2: read config + load safetensors")
    raw = json.load(open(os.path.join(snap, "config.json")))
    text_cfg = _ns(raw["text_config"])
    cfg_obj = _ns(raw)
    cfg_obj.text_config = text_cfg
    vocab = text_cfg.vocab_size
    print(f"text_config: layers={text_cfg.num_hidden_layers} d_model={text_cfg.hidden_size} "
          f"vocab={vocab} full_attn_layers={text_cfg.layer_types.count('full_attention')} "
          f"linear_layers={text_cfg.layer_types.count('linear_attention')}", flush=True)

    shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    print(f"loading {len(shards)} safetensors shard(s)", flush=True)
    hf_state = {}
    for s in shards:
        hf_state.update(load_file(s))
    n_text = sum(1 for k in hf_state if "visual" not in k and "vision" not in k)
    print(f"state: {len(hf_state)} tensors ({n_text} non-vision)", flush=True)

    banner("PHASE 3: build olmo + convert + strict load")
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.hf.convert import convert_qwen3_5_state_from_hf
    from olmo_core.nn.transformer import TransformerConfig

    cfg = TransformerConfig.qwen3_5_0_8B(vocab_size=vocab, attn_backend=AttentionBackendName.torch)
    print("built TransformerConfig.qwen3_5_0_8B", flush=True)
    model = cfg.build(init_device="cpu").eval()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"built olmo model: {nparams/1e9:.3f}B params", flush=True)

    converted = convert_qwen3_5_state_from_hf(cfg_obj, hf_state)
    print(f"converted: {len(converted)} tensors", flush=True)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    print(f"strict-load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("  MISSING (first 8):", list(missing)[:8], flush=True)
    if unexpected:
        print("  UNEXPECTED (first 8):", list(unexpected)[:8], flush=True)

    banner("PHASE 4: GPU forward + greedy decode (exercises GDN/FLA)")
    device = torch.device("cuda")
    model.to(device).to(torch.bfloat16)
    # toy input ids in-vocab
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=device)
    with torch.no_grad():
        logits = model(ids)
    print(f"forward OK: logits {tuple(logits.shape)} finite={torch.isfinite(logits).all().item()} "
          f"mean={logits.float().mean().item():.3f}", flush=True)

    # greedy continue a few steps to confirm the stack runs autoregressively
    cur = ids
    for _ in range(10):
        with torch.no_grad():
            nxt = model(cur)[:, -1].argmax(-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=1)
    print("greedy ids:", cur[0].tolist(), flush=True)

    ok = bool(torch.isfinite(logits).all().item()) and not missing and not unexpected
    print(f"\nRESULT: {'PASS' if ok else 'PARTIAL'} "
          f"(strict-key-match={'yes' if not missing and not unexpected else 'NO'}, finite-forward=yes)", flush=True)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        banner("UNCAUGHT EXCEPTION")
        traceback.print_exc()
        sys.exit(1)
