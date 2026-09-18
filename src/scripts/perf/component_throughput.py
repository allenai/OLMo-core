"""Separate the LM's dense and attention FLOPs, then bound the ViT's expected time share."""
from olmo_core.nn.vision.multimodal import MultimodalLMConfig
cfg = MultimodalLMConfig.molmo2_4B(); m = cfg.build(init_device="meta")
S = 16384
f1, f2 = m.num_flops_per_token(S), m.num_flops_per_token(S // 2)
B = (f1 - f2) / (S - S // 2)        # attention coefficient (per token, per unit S)
A = f1 - B * S                      # dense part per token
print("LM dense   : %.4e FLOPs/token" % A)
print("LM attn    : %.4e FLOPs/token at S=%d  (%.1f%% of the LM)" % (B*S, S, 100*B*S/f1))

# With the pad fix, attention only computes within-example causal blocks.
# ~n examples of ~len each; cost ~ sum(len_i^2) vs the full S^2.
def attn_pf(n_ex, real_tok):
    L = real_tok / n_ex
    return B * n_ex * L * L          # B has units FLOPs/token/token

dense_pf = A * S                     # dense runs on all 16384 padded tokens
d, nl = cfg.vision.image_emb_dim, cfg.vision.image_num_layers
P, POOLED = 729, 144
def vit_pf(nc):
    return nc*P*(2*m._n_vision_params + 4*nl*P*d)*3 + nc*POOLED*6*m._n_connector_params

print()
print("%-9s %9s %9s %9s %9s" % ("arm","LM dense","LM attn","ViT","total PF"))
arms = [("crops=25", 24, 4, 5046, 0.60), ("crops=80", 79, 13, 16180, 1.03)]
vals = {}
for name, nc, nex, real, sec in arms:
    a = attn_pf(nex, real); v = vit_pf(nc); t = dense_pf + a + v
    vals[name] = (dense_pf, a, v, t, sec)
    print("%-9s %9.4f %9.4f %9.4f %9.4f" % (name, dense_pf/1e15, a/1e15, v/1e15, t/1e15))

# Solve time = (dense+attn)/r_lm + vit/r_vit  across the two arms.
(d1,a1,v1,t1,s1) = vals["crops=25"]; (d2,a2,v2,t2,s2) = vals["crops=80"]
lm1, lm2 = d1+a1, d2+a2
# two equations, two unknowns (1/r_lm, 1/r_vit)
det = lm1*v2 - lm2*v1
inv_lm = (s1*v2 - s2*v1)/det
inv_vit = (lm1*s2 - lm2*s1)/det
print()
print("implied LM  throughput: %.3f PF/s" % (1/inv_lm/1e15))
print("implied ViT throughput: %.3f PF/s  (%.1fx less efficient)" % (1/inv_vit/1e15, inv_vit/inv_lm))
print()
for name,(dd,aa,vv,tt,ss) in vals.items():
    vt = vv*inv_vit
    print("%-9s predicted ViT time share: %.1f%%  (FLOP share %.1f%%)" % (
        name, 100*vt/ss, 100*vv/tt))
