"""Probe Qwen3.5-0.8B cache structure to plan serial-prefill+batched-decode.

Loads the model, runs prefill on 2 short prompts independently (no batching),
and dumps:
  - Top-level past_key_values class
  - Per-layer attention type (softmax vs GDN linear-attention)
  - Per-layer cache representation (tensor shapes / fields)
  - Whether the cache supports clean per-example access for stacking

Goal: figure out how to merge B per-example caches into a batched cache.
"""
from __future__ import annotations

import argparse
import torch


def _load(base_model, lora):
    import types
    from corpus_reasoning.eval.evaluate import load_hf_model
    args = types.SimpleNamespace(
        base_model=base_model, lora_path=lora, backend="chunked-sdpa",
        language_model_only=False, attn_impl_override=None,
    )
    return load_hf_model(args)


def _describe(obj, depth=0, prefix=""):
    """Recursively describe shape/dtype/type of nested tensor structures."""
    pad = "  " * depth
    if isinstance(obj, torch.Tensor):
        return f"{prefix}Tensor shape={tuple(obj.shape)} dtype={obj.dtype}"
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return f"{prefix}{type(obj).__name__}[empty]"
        # Heads-up if all elements are tensors with same shape
        if all(isinstance(x, torch.Tensor) for x in obj):
            shapes = {tuple(x.shape) for x in obj}
            if len(shapes) == 1:
                return (f"{prefix}{type(obj).__name__}[{len(obj)}] of "
                        f"Tensor shape={list(shapes)[0]} dtype={obj[0].dtype}")
        out = [f"{prefix}{type(obj).__name__}[{len(obj)}]:"]
        for i, x in enumerate(obj[:3]):
            out.append(_describe(x, depth + 1, f"{pad}  [{i}] "))
        if len(obj) > 3:
            out.append(f"{pad}  ... ({len(obj)-3} more)")
        return "\n".join(out)
    if isinstance(obj, dict):
        out = [f"{prefix}dict[{len(obj)}]:"]
        for k, v in list(obj.items())[:5]:
            out.append(_describe(v, depth + 1, f"{pad}  {k!r}: "))
        return "\n".join(out)
    return f"{prefix}{type(obj).__name__}: {repr(obj)[:80]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--lora", required=True)
    args = ap.parse_args()

    model, tokenizer, doc_start_id, doc_end_id = _load(args.base_model, args.lora)
    print(f"\n=== Model class ===")
    print(f"  type(model) = {type(model).__name__}")
    print(f"  num decoder layers = {len(model.model.layers)}")
    for i, layer in enumerate(model.model.layers):
        # Enumerate ALL submodule names so we find both softmax (self_attn)
        # and GDN linear-attention modules (different name).
        children = [(n, type(m).__name__) for n, m in layer.named_children()]
        print(f"    layer {i:2d}: {type(layer).__name__}  children={children}")

    # Tiny forward
    text = "Hello, this is a probe."
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    print(f"\n=== Forward(input_ids shape={tuple(ids.shape)}, use_cache=True) ===")
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    pkv = out.past_key_values
    print(f"\n=== past_key_values ===")
    print(f"  top-level type: {type(pkv).__name__}")
    print(f"  module: {type(pkv).__module__}")
    print(f"  has __len__: {hasattr(pkv, '__len__')}; len = "
          f"{len(pkv) if hasattr(pkv, '__len__') else 'N/A'}")
    # Common attributes
    for attr in ("layers", "key_cache", "value_cache", "linear_attn_cache",
                 "conv_states", "ssm_states", "recurrent_state", "_seen_tokens"):
        if hasattr(pkv, attr):
            v = getattr(pkv, attr)
            print(f"  attr .{attr}: {type(v).__name__}", end="")
            if isinstance(v, (list, tuple)):
                print(f" len={len(v)}")
            elif isinstance(v, torch.Tensor):
                print(f" shape={tuple(v.shape)}")
            else:
                print()

    # Per-layer cache access via pkv.layers — exhaustive introspection
    print(f"\n=== Per-layer cache (one of each kind, full introspection) ===")
    seen_classes = set()
    if hasattr(pkv, "layers"):
        for i, lc in enumerate(pkv.layers):
            cls_name = type(lc).__name__
            if cls_name in seen_classes:
                continue
            seen_classes.add(cls_name)
            print(f"\n  pkv.layers[{i}]: {cls_name}")
            print(f"    module: {type(lc).__module__}")
            print(f"    __dict__: keys={list(vars(lc).keys()) if hasattr(lc, '__dict__') else 'n/a'}")
            if hasattr(lc, "__dict__"):
                for k, v in vars(lc).items():
                    if isinstance(v, torch.Tensor):
                        print(f"      .{k}: Tensor shape={tuple(v.shape)} dtype={v.dtype}")
                    elif isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
                        print(f"      .{k}: {type(v).__name__}[{len(v)}] shape={tuple(v[0].shape)} dtype={v[0].dtype}")
                    else:
                        repr_short = repr(v)[:100]
                        print(f"      .{k}: {type(v).__name__}  repr={repr_short}")
            print(f"    public methods/attrs:")
            for name in dir(lc):
                if name.startswith("_"):
                    continue
                try:
                    v = getattr(lc, name)
                except Exception:
                    continue
                if callable(v):
                    print(f"      .{name}(): method")
                elif isinstance(v, torch.Tensor):
                    print(f"      .{name}: Tensor shape={tuple(v.shape)} dtype={v.dtype}")

    # Also dump the actual GDN attention module to see the state-shape spec
    print(f"\n=== GDN layer module spec (layer 0) ===")
    gdn = model.model.layers[0].linear_attn
    print(f"  type: {type(gdn).__name__}")
    print(f"  attrs of interest:")
    for name in ("num_heads", "head_dim", "head_v_dim", "head_k_dim",
                 "key_dim", "value_dim", "expand_v", "expand_k",
                 "conv_size", "use_short_conv", "d_model", "hidden_size"):
        if hasattr(gdn, name):
            print(f"    .{name} = {getattr(gdn, name)}")

    # Pkv-level methods (batch stacking?)
    print(f"\n=== DynamicCache top-level methods ===")
    for name in dir(pkv):
        if name.startswith("_"):
            continue
        v = getattr(pkv, name, None)
        if callable(v):
            print(f"  .{name}()")

    # Try also: get_seq_length / total_seq_len
    print(f"\n=== Cache metadata ===")
    for attr in ("get_seq_length", "_seen_tokens", "seen_tokens"):
        if hasattr(pkv, attr):
            v = getattr(pkv, attr)
            if callable(v):
                try:
                    print(f"  pkv.{attr}() = {v()}")
                except Exception as e:
                    print(f"  pkv.{attr}() ERROR {type(e).__name__}: {e}")
            else:
                print(f"  pkv.{attr} = {v}")


if __name__ == "__main__":
    main()
