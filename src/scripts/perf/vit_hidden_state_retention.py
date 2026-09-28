"""Does collecting all ViT block outputs actually retain extra memory?  (Answer: no.)

Run with::

    PYTHONPATH=$PWD/src python src/scripts/perf/vit_hidden_state_retention.py

`VisionTransformer.forward` returns the output of every block, but the connector consumes
only `vit_layers=(24, 18)`. That looks like 23 layers of retained activations waiting to be
reclaimed, and it was on the shortlist of remaining Stage-2 memory levers.

It is not a lever. Per-block activation checkpointing saves each block's *input* for
recomputation, and block i's input is block (i-1)'s output -- so every block output is
already pinned by the autograd graph. The Python list adds only a reference. The measured
delta is exactly 0, with activation checkpointing on *or* off (with AC off the successor
block's backward needs the same tensor anyway).

Two controls are included, because a probe that always prints 0 proves nothing: halving the
block count does move the number (so the counter is sensitive), and the AC-on vs AC-off
totals differ ~20x (so the graph being measured is the real one -- and incidentally that
20x is why turning vision AC off OOMs at every useful crop budget).
"""
import torch

from olmo_core.nn.vision.config import VisionEncoderConfig
from olmo_core.nn.vision.image_vit import VisionTransformer

N = 8
cfg = VisionEncoderConfig(
    image_default_input_size=(28, 28),
    image_patch_size=4,
    image_emb_dim=64,
    image_num_heads=4,
    image_num_key_value_heads=4,
    image_num_layers=N,
    image_head_dim=16,
    image_mlp_dim=128,
    image_num_pos=50,
    use_cls_token=True,
)
vit = VisionTransformer(cfg, init_device="cpu")
vit.apply_activation_checkpointing()
print("blocks:", len(vit.blocks))

x = torch.randn(2, 49, 4 * 4 * 3)


def count_saved(select):
    saved = []

    class H(torch.autograd.graph.saved_tensors_hooks):
        def __init__(self):
            super().__init__(lambda t: (saved.append(tuple(t.shape)), t)[1], lambda t: t)

    with H():
        hs = vit(x)
        out = torch.cat([hs[i] for i in select], dim=-1)
        out.sum().backward()
    vit.zero_grad(set_to_none=True)
    elems = sum(torch.Size(s).numel() for s in saved)
    return len(saved), elems


a_n, a_e = count_saved(list(range(N)))
b_n, b_e = count_saved([N - 1, N // 2])
print(f"caller consumes ALL {N} layers : {a_n} saved tensors, {a_e} elems")
print(f"caller consumes only 2 layers  : {b_n} saved tensors, {b_e} elems")
print(f"delta                          : {a_e - b_e} elems " f"({100*(a_e-b_e)/max(a_e,1):.1f}%)")

# --- positive control: the counter must be able to see a real change ---
print()
cfg4 = VisionEncoderConfig(
    image_default_input_size=(28, 28),
    image_patch_size=4,
    image_emb_dim=64,
    image_num_heads=4,
    image_num_key_value_heads=4,
    image_num_layers=4,
    image_head_dim=16,
    image_mlp_dim=128,
    image_num_pos=50,
    use_cls_token=True,
)
vit4 = VisionTransformer(cfg4, init_device="cpu")
vit4.apply_activation_checkpointing()
vit = vit4
c_n, c_e = count_saved([3, 2])
print(f"CONTROL 4 blocks instead of 8  : {c_n} saved tensors, {c_e} elems")
print("counter is sensitive:", c_e != b_e)

# --- second control: AC off, all layers vs two ---
print()
vit_noac = VisionTransformer(cfg, init_device="cpu")
vit = vit_noac
d_n, d_e = count_saved(list(range(N)))
e_n, e_e = count_saved([N - 1, N // 2])
print(f"AC OFF, all {N} layers         : {d_n} saved tensors, {d_e} elems")
print(f"AC OFF, only 2 layers          : {e_n} saved tensors, {e_e} elems")
print(f"AC OFF delta                   : {d_e - e_e} elems")
