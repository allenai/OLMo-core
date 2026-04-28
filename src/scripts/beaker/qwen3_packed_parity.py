"""
Reproduce the Qwen3 packed intra-doc attention parity bug in OLMo-core.

Solo (per-doc OLMo-core forward) is the reference (HF doesn't build intra-doc
masks from position_ids alone). We compare:

    packed[:, :la]      vs solo_a   (doc_a slice)
    packed[:, la:la+lb] vs solo_b   (doc_b slice)

per block, where packed = cat([doc_a, doc_b]) is run with
doc_lens=[[la, lb]], max_doc_lens=[max(la, lb)].

Three reports:
  1. Per-block packed-with-doc_lens vs solo.
  2. Block_0 attention submodule drill (w_q/w_k/w_v/q_norm/k_norm/rope/attn).
  3. Sanity: packed-without-doc_lens vs solo (doc_a slice should still match).
"""

import logging
import os

import torch

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import prepare_cli_environment

import transformers

log = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-4B-Base"
LA = 10
LB = 12
SEED = 0


def make_hook(store, name):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        store[name] = hidden.detach()

    return hook


def make_input_hook(store, name):
    def hook(_module, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        store[name] = x.detach()

    return hook


def register_block_hooks(model, store):
    handles = []
    for key in sorted(model.blocks.keys(), key=int):
        block = model.blocks[key]
        i = int(key)
        handles.append(block.register_forward_hook(make_hook(store, f"block_{i:02d}")))
        handles.append(
            block.attention.register_forward_hook(make_hook(store, f"block_{i:02d}.attn"))
        )
        handles.append(
            block.feed_forward.register_forward_hook(make_hook(store, f"block_{i:02d}.ffn"))
        )
    handles.append(model.lm_head.norm.register_forward_hook(make_hook(store, "final_norm")))
    return handles


def remove_handles(handles):
    for h in handles:
        h.remove()


def diff_summary(a, b):
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item()


def report_per_block(packed_store, solo_a_store, solo_b_store, n_layers, la, lb):
    log.info("=" * 88)
    log.info("Report 1: per-block packed-with-doc_lens vs solo")
    log.info(
        "%-22s %12s %12s %12s %12s",
        "name",
        "max(a)",
        "mean(a)",
        "max(b)",
        "mean(b)",
    )
    log.info("-" * 88)
    for i in range(n_layers):
        for suffix in (".attn", ".ffn", ""):
            name = f"block_{i:02d}{suffix}"
            packed = packed_store[name]
            mx_a, mn_a = diff_summary(packed[:, :la], solo_a_store[name])
            mx_b, mn_b = diff_summary(packed[:, la : la + lb], solo_b_store[name])
            log.info("%-22s %12.3e %12.3e %12.3e %12.3e", name, mx_a, mn_a, mx_b, mn_b)
    name = "final_norm"
    packed = packed_store[name]
    mx_a, mn_a = diff_summary(packed[:, :la], solo_a_store[name])
    mx_b, mn_b = diff_summary(packed[:, la : la + lb], solo_b_store[name])
    log.info("%-22s %12.3e %12.3e %12.3e %12.3e", name, mx_a, mn_a, mx_b, mn_b)


def drill_block0(model, doc_a, doc_b, packed, doc_lens, max_doc_lens, la):
    log.info("=" * 88)
    log.info("Report 2: block_0 attention submodule drill (packed[:, :la] vs solo_a)")
    log.info("%-22s %12s %12s", "name", "max", "mean")
    log.info("-" * 88)

    block0 = model.blocks[sorted(model.blocks.keys(), key=int)[0]]
    attn = block0.attention

    targets = []
    for child_name in ("w_q", "w_k", "w_v", "q_norm", "k_norm", "rope", "w_out"):
        sub = getattr(attn, child_name, None)
        if sub is None or not isinstance(sub, torch.nn.Module):
            continue
        targets.append((child_name, sub))
    targets.append(("attn", attn))
    targets.append(("block", block0))

    solo_in: dict = {}
    solo_out: dict = {}
    pkd_in: dict = {}
    pkd_out: dict = {}

    def attach(store_in, store_out):
        handles = []
        for name, mod in targets:
            handles.append(mod.register_forward_pre_hook(make_input_hook(store_in, name)))
            handles.append(mod.register_forward_hook(make_hook(store_out, name)))
        return handles

    h = attach(solo_in, solo_out)
    with torch.no_grad():
        model(doc_a)
    remove_handles(h)

    h = attach(pkd_in, pkd_out)
    with torch.no_grad():
        model(packed, doc_lens=doc_lens, max_doc_lens=max_doc_lens)
    remove_handles(h)

    for name, _ in targets:
        if name not in solo_in or name not in pkd_in:
            continue
        if pkd_in[name].dim() < 2 or solo_in[name].dim() < 2:
            continue
        mx_in, mn_in = diff_summary(pkd_in[name][:, :la], solo_in[name])
        mx_out, mn_out = diff_summary(pkd_out[name][:, :la], solo_out[name])
        log.info("%-22s %12.3e %12.3e", f"{name}.in", mx_in, mn_in)
        log.info("%-22s %12.3e %12.3e", f"{name}.out", mx_out, mn_out)


def main() -> None:
    prepare_cli_environment()
    hf_token = os.environ.get("HF_TOKEN")
    device = torch.device("cuda")
    dtype = torch.bfloat16

    log.info("Loading HF config %s", MODEL_ID)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=hf_token, torch_dtype=dtype, attn_implementation="flash_attention_2"
    )
    hf_config = hf_model.config

    log.info("Building OLMo-core qwen3_4B (vocab=%d)", hf_config.vocab_size)
    olmo_config = TransformerConfig.qwen3_4B(
        vocab_size=hf_config.vocab_size,
        attn_backend=AttentionBackendName.flash_2,
    )
    olmo_model = olmo_config.build(init_device="cpu")

    log.info("Loading converted HF weights")
    converted_state = convert_state_from_hf(hf_config, hf_model.state_dict(), model_type="qwen3")
    olmo_model.load_state_dict(converted_state)
    olmo_model.to(device=device, dtype=dtype).eval()

    del hf_model
    torch.cuda.empty_cache()

    torch.manual_seed(SEED)
    doc_a = torch.randint(0, hf_config.vocab_size, (1, LA), device=device)
    doc_b = torch.randint(0, hf_config.vocab_size, (1, LB), device=device)
    packed = torch.cat([doc_a, doc_b], dim=1)
    doc_lens = torch.tensor([[LA, LB]], device=device, dtype=torch.int32)
    max_doc_lens = torch.tensor([max(LA, LB)], device=device, dtype=torch.int32)
    log.info("doc_a=%s", doc_a.tolist())
    log.info("doc_b=%s", doc_b.tolist())

    n_layers = len(olmo_model.blocks)

    solo_a_store: dict = {}
    h = register_block_hooks(olmo_model, solo_a_store)
    with torch.no_grad():
        olmo_model(doc_a)
    remove_handles(h)

    solo_b_store: dict = {}
    h = register_block_hooks(olmo_model, solo_b_store)
    with torch.no_grad():
        olmo_model(doc_b)
    remove_handles(h)

    packed_store: dict = {}
    h = register_block_hooks(olmo_model, packed_store)
    with torch.no_grad():
        olmo_model(packed, doc_lens=doc_lens, max_doc_lens=max_doc_lens)
    remove_handles(h)

    report_per_block(packed_store, solo_a_store, solo_b_store, n_layers, LA, LB)

    drill_block0(olmo_model, doc_a, doc_b, packed, doc_lens, max_doc_lens, LA)

    log.info("=" * 88)
    log.info("Report 3: sanity — packed without doc_lens vs solo")
    log.info("%-22s %12s %12s %12s %12s", "name", "max(a)", "mean(a)", "max(b)", "mean(b)")
    log.info("-" * 88)
    nodoc_store: dict = {}
    h = register_block_hooks(olmo_model, nodoc_store)
    with torch.no_grad():
        olmo_model(packed)
    remove_handles(h)
    for i in range(n_layers):
        name = f"block_{i:02d}"
        packed_h = nodoc_store[name]
        mx_a, mn_a = diff_summary(packed_h[:, :LA], solo_a_store[name])
        mx_b, mn_b = diff_summary(packed_h[:, LA : LA + LB], solo_b_store[name])
        log.info("%-22s %12.3e %12.3e %12.3e %12.3e", name, mx_a, mn_a, mx_b, mn_b)


if __name__ == "__main__":
    main()
