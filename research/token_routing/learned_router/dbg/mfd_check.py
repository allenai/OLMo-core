"""Debug: markers-follow-doc soft_keep vs plain at init gates (niah, 2 train rows)."""
import sys, os, math, torch, types
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import router_lib as RL, train_e2e_router as E, oracle.oracle as O
_pr = E.TR.prep_row
E.TR.prep_row = lambda m, r, ids, dev, route_markers=False: _pr(m, r, ids, dev, route_markers=False)
a = types.SimpleNamespace(task="niah", ckpt=sys.argv[1], ckpt_format="hf", train_root=sys.argv[2] + "/router_train_stg", val_root=sys.argv[2] + "/router_val_stg",
                          tokenizer=sys.argv[3], n_train=2, n_val=1, val_extra_from_train=0)
model, tok, ids, vocab, prepped, dev = O.setup(a, ["train"])
model._pooled_soft_tokens["keep_token_mask_markers"] = True
for p in prepped["train"]:
    lpf = E.full_logprobs(model, p).to(dev).float(); cef = float(F.nll_loss(lpf, p["tgt"]))
    N = p["feats"]["idx"].numel()
    gen = torch.Generator().manual_seed(0)
    la = torch.full((N,), math.log(0.95 / 0.05), device=dev)
    la = la + E.calibrate_offset(la, 1.0, 0.67)
    z = E.keep_dropout(RL.hard_concrete(la, 0.67, gen, straight_through=False, st_clamp=True), 0.15, gen)
    for follow in (False, True):
        p["feats"]["mk_follow"] = follow
        lg = E.relaxed_logits(model, p, z)
        ce = float(F.cross_entropy(lg, p["tgt"]))
        sk = RL.soft_keep_row(p["feats"], z, p["T"])
        print(f"follow={follow} CE {ce:.4f} (full {cef:.4f}) z mean {float(z.mean()):.3f} min {float(z.min()):.3f} | marker sk {sk[p['feats']['mk_idx']].mean().item():.3f} min {sk[p['feats']['mk_idx']].min().item():.3f}", flush=True)
    # batch path as in training
    for follow in (False, True):
        p["feats"]["mk_follow"] = follow
        lg = E.relaxed_logits_batch(model, [p], [z], ids.eos)[0]
        print("batch follow", follow, float(F.cross_entropy(lg, p["tgt"])))
    z1 = torch.ones(N, device=dev)
    p["feats"]["mk_follow"] = True
    print("z=1 follow CE", float(F.cross_entropy(E.relaxed_logits(model, p, z1), p["tgt"])))
