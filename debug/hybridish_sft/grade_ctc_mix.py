"""Grade a checkpoint on the 2k-32k CTC mix, per task, with each task's own spec.

One model, one mix, per-task numbers -- which is what separates "the arm is better" from "this task
is learnable at all". The 275M runs showed the task axis dominating the model axis ~28x, so per-task
reporting is the point, not a nicety.

Grading uses the suite's parser+scorer with gold taken from the EXAMPLE (never from parsing the gold
text -- `parse` returns a set and `score` subscripts its second argument). parse_rate is reported
beside every score: a low score at low parse_rate is a decoding failure, not a capability one.

**Shard instance i is NOT source row i.** The converter drops rows over its length cap (5.1% of the
32k mix, concentrated in absence/xabsence at 35k-84k tokens), so the shards are a subsequence of the
source JSONL. Zipping them positionally pairs each generation with a different example's gold from
the first drop onward -- which scored the entire 8-task table against near-random gold and read as
"these tasks are unlearnable". This script therefore REQUIRES the `src_index.json` sidecar and
refuses to run without it.

The guard against a recurrence is `--self-check`: the gold answer already lives in the shard, in the
span the label mask marks, so scoring the SHARD's own gold against the MAPPED example's gold must
return 1.0. It needs no generation and it fails loudly on any misalignment. It is on by default.
"""
import argparse, inspect, json, math, os, sys, time
from collections import defaultdict
import numpy as np, torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
DDP_ON = WORLD > 1
if DDP_ON:
    # 4h timeout, not the default. Grading generates up to --max-new tokens per example and each
    # rank waits at an all_reduce for the others; after SFT the model emits full-length answers
    # (the base model hit EOS almost immediately), so the post-training grade is far slower than
    # the baseline one. On the 2x-slower 7:1 arm that overran the default and NCCL aborted the
    # collective -- which looked like a crash in grading rather than a timeout.
    from datetime import timedelta
    dist.init_process_group("nccl", timeout=timedelta(hours=4))
    torch.cuda.set_device(LOCAL_RANK)

def p0(*args, **kw):
    """Print on rank 0 only -- 8 ranks echoing the same line makes logs unreadable."""
    if RANK == 0: print(*args, **kw)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--shards", required=True)
ap.add_argument("--src-jsonl", required=True); ap.add_argument("--plugin", required=True)
ap.add_argument("--ctcsrc", required=True)
ap.add_argument("--eval-per-task", type=int, default=40)
ap.add_argument("--train-n", type=int, default=3000)
ap.add_argument("--bs", type=int, default=1); ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--ssmax", default="position")
ap.add_argument("--max-new", type=int, default=24); ap.add_argument("--grade-bs", type=int, default=2)
ap.add_argument("--max-len", type=int, default=32768)
ap.add_argument("--tag", default="mt"); ap.add_argument("--grade-base", action="store_true")
ap.add_argument("--no-self-check", action="store_true",
                help="skip the gold-alignment assertion (don't -- it is the guard that catches a "
                     "shard/source misalignment before it becomes a results table)")
ap.add_argument("--out-dir",
                default="/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/hybridish_sft/results")
ap.add_argument("--bands", default="4096,16384",
                help="token-length cut points for the per-band breakdown; '' disables it")
a = ap.parse_args()

sys.path.insert(0, a.plugin); sys.path.insert(0, a.ctcsrc)
import transformers_plugin
for fn in ("register", "register_config"):
    try: getattr(transformers_plugin, fn)()
    except Exception: pass
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers_plugin import modeling_mainline_ladder as _MLM
_C = _MLM.MainlineLadderDynamicCache
if not hasattr(_C, "get_query_offset"):
    def _qo(self, layer_idx=0):
        k = self.key_cache[layer_idx] if layer_idx < len(self.key_cache) else None
        return 0 if k is None else k.shape[2]
    _C.get_query_offset = _qo
if not hasattr(_C, "get_mask_sizes"):
    def _ms(self, cache_position, layer_idx):
        off = self.get_query_offset(layer_idx); return (off + cache_position.shape[0], off)
    _C.get_mask_sizes = _ms
from ctc.tasks import load_all; from ctc.format import registry
load_all()
T2S = {"nq":"retrieval","msmarco":"retrieval","hotpotqa":"cot_retrieval","qdmatch_nq":"qdmatch",
       "qdmatch_hpqa":"qdmatch","outlier_review":"outlier","contra_fever":"contradiction"}
def spec_for(t): return registry.get(T2S.get(t, t))

def score_any(sp, parsed, ex):
    """Call a spec's scorer with whatever its SECOND ARGUMENT actually wants.

    The specs are not uniform: `retrieval.score(parsed, example_or_gold)` accepts the example, but
    the other seven take `gold: Sequence[int]`. Passing the example dict to those iterates its KEYS
    -- `int('documents')` -> ValueError -- so every one of them scores 0.0. Verified: with this
    dispatch, a gold answer self-scores 1.0 on all 8 tasks; without it, only nq does.
    """
    params = list(inspect.signature(sp.score).parameters)
    second = params[1] if len(params) > 1 else ""
    if "example" in second:
        return sp.score(parsed, ex)
    for k in ("gold_doc_indices", "gold_pairs", "gold_order", "answers"):
        v = ex.get(k)
        if v:
            return sp.score(parsed, v)
    raise KeyError(f"no gold field on example for spec {sp.name}")

meta = json.load(open(f"{a.shards}/metadata.json")); EOS = meta["eos_token_id"]
tok_files = sorted([f for f in os.listdir(a.shards) if f.startswith("token_ids_part_")])
msk_files = sorted([f for f in os.listdir(a.shards) if f.startswith("labels_mask_")])
ids = np.concatenate([np.fromfile(f"{a.shards}/{f}", dtype=np.uint32) for f in tok_files])
msk = np.concatenate([np.fromfile(f"{a.shards}/{f}", dtype=np.bool_) for f in msk_files])
b = np.flatnonzero(ids == EOS) + 1
inst = list(zip(np.concatenate([[0], b[:-1]]).tolist(), b.tolist()))
SRC = [json.loads(l) for l in open(a.src_jsonl)]
IDX_PATH = f"{a.shards}/src_index.json"
if not os.path.exists(IDX_PATH):
    raise SystemExit(
        f"missing {IDX_PATH}. Shard instance i is not source row i whenever the converter dropped a "
        "row, so grading without this map scores each generation against another example's gold. "
        "Build it with debug/hybridish_sft/build_src_index.py."
    )
SRC_ROW = json.load(open(IDX_PATH))["src_index"]
assert len(SRC_ROW) == len(inst), f"src_index has {len(SRC_ROW)} entries for {len(inst)} instances"
def ex_for(i): return SRC[SRC_ROW[i]]
p0(f"[data] {len(inst)} shard instances, {len(SRC)} src rows, "
   f"{len(SRC) - len(inst)} dropped by the length cap", flush=True)

# hold out the first --eval-per-task of each task; everything else is eligible for training
by_task = defaultdict(list)
for i in range(len(inst)): by_task[ex_for(i).get("_task", "?")].append(i)
eval_idx, train_pool = {}, []
for t, idxs in sorted(by_task.items()):
    fit = [i for i in idxs if (inst[i][1] - inst[i][0]) <= a.max_len]
    eval_idx[t] = fit[: a.eval_per_task]
    train_pool += fit[a.eval_per_task:]
rng = np.random.default_rng(0); rng.shuffle(train_pool)
train_idx = train_pool[: a.train_n]
p0(f"[data] tasks: { {t: len(v) for t, v in eval_idx.items()} }", flush=True)
p0(f"[data] train {len(train_idx)}", flush=True)

dev = f"cuda:{LOCAL_RANK}" if DDP_ON else "cuda"
cfg = AutoConfig.from_pretrained(a.ckpt); cfg.use_cache = False
model = AutoModelForCausalLM.from_pretrained(a.ckpt, config=cfg, dtype=torch.bfloat16).to(dev)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
tok = AutoTokenizer.from_pretrained(a.ckpt)
p0(f"[model] {sum(p.numel() for p in model.parameters()):,} params  layers={cfg.num_hidden_layers}", flush=True)

if a.ssmax != "none":
    from safetensors import safe_open
    sc = {}
    with safe_open(f"{a.ckpt}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            if "ssmax_scale" in k: sc[int(k.split(".layers.")[1].split(".")[0])] = f.get_tensor(k).float().to(dev)
    assert sc, "ssmax requested but checkpoint has none"
    def mk(s):
        def hook(m, i, out):
            H = s.numel(); sh = list(out.shape)
            hax = next(j for j in (1, 2) if sh[j] == H); tax = 2 if hax == 1 else 1
            T = sh[tax]; fs = [1,1,1,1]; fs[hax] = H
            ts = [1,1,1,1]; ts[tax] = T
            pos = torch.arange(1, T+1, device=out.device, dtype=torch.float32).clamp(min=2)
            return (out.float() * (s.view(*fs) * torch.log(pos).view(*ts))).to(out.dtype)
        return hook
    for i, s in sc.items(): model.model.layers[i].self_attn.q_norm.register_forward_hook(mk(s))
    p0(f"[ssmax] re-attached on layers {sorted(sc)} (RECONSTRUCTION)", flush=True)

if DDP_ON:
    model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=False)
    p0(f"[ddp] world_size={WORLD}")
CORE = model.module if DDP_ON else model


def loss_on_answer(x, y):
    """Loss computed only where labels exist.

    A full-sequence lm_head at 32768 x 100352 vocab is ~12GB once cross_entropy upcasts to fp32 --
    that OOMed the 2B arm and dominated step time, for a loss covering a few dozen answer tokens.
    Every CTC answer is a contiguous span at the END of the instance, so the backbone runs over the
    whole context (which is the point of the task) and only the trailing slice is projected.
    """
    lab = y[0]
    pos = torch.nonzero(lab != -100, as_tuple=False)
    if pos.numel() == 0:
        return None
    first = int(pos[0])
    k = lab.shape[0] - first                      # trailing positions carrying labels
    h = CORE.model(input_ids=x, use_cache=False).last_hidden_state
    logits = CORE.lm_head(h[:, first - 1 : -1, :])        # predicts tokens [first : end]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                           lab[first:].reshape(-1), ignore_index=-100)


def batch(idxs):
    items = [inst[i] for i in idxs]
    L = max(e - s for s, e in items)
    x = np.full((len(items), L), EOS, np.int64); y = np.full((len(items), L), -100, np.int64)
    for r, (s, e) in enumerate(items):
        k = e - s; x[r,:k] = ids[s:e]; y[r,:k] = np.where(msk[s:e], ids[s:e].astype(np.int64), -100)
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

@torch.no_grad()
def grade_all():
    CORE.eval(); out = {}
    for t, idxs_all in sorted(eval_idx.items()):
        idxs = idxs_all[RANK::WORLD] if DDP_ON else idxs_all   # each rank grades a disjoint slice
        sp = spec_for(t); scores = []; parsed = 0; per_band = defaultdict(list)
        for i in range(0, len(idxs), a.grade_bs):
            chunk = idxs[i:i + a.grade_bs]
            for gi in chunk:
                s, e = inst[gi]; m = msk[s:e]; cut = int(np.argmax(m))
                x = torch.from_numpy(ids[s:s+cut].astype(np.int64)).unsqueeze(0).to(dev)
                g = CORE.generate(x, max_new_tokens=a.max_new, do_sample=False,
                                   pad_token_id=EOS, eos_token_id=EOS, use_cache=True)
                text = tok.decode(g[0, cut:], skip_special_tokens=True)
                ex = ex_for(gi); nd = len(ex.get("documents", []) or [])
                p = sp.parse(text, nd)
                if p is None: scores.append(0.0); per_band[band_of(gi)].append(0.0); continue
                parsed += 1
                try: v = float(score_any(sp, p, ex)[sp.primary_metric])
                except Exception as exc:
                    print(f"[grade] {t} scoring error {type(exc).__name__}: {exc}", flush=True); v = 0.0
                scores.append(v); per_band[band_of(gi)].append(v)
        if DDP_ON:
            # gather per-rank sums so the reported mean is over the FULL eval set, not one shard
            buf = torch.tensor([float(np.sum(scores)), float(parsed), float(len(idxs))], device=dev)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            tot, par, n_ = buf.tolist()
        else:
            tot, par, n_ = float(np.sum(scores)), float(parsed), float(len(idxs))
        bands = {}
        for bname in sorted(set(list(per_band) + ([] if not BANDS else []))):
            v = per_band[bname]
            if DDP_ON:
                bb = torch.tensor([float(np.sum(v)), float(len(v))], device=dev)
                dist.all_reduce(bb, op=dist.ReduceOp.SUM); sv, sn = bb.tolist()
            else:
                sv, sn = float(np.sum(v)), float(len(v))
            if sn: bands[bname] = {"score": sv / sn, "eval_size": int(sn)}
        out[t] = (tot / max(n_, 1), par / max(n_, 1), int(n_), sp.primary_metric, bands)
    CORE.train(); return out

def self_check():
    """Score the shard's OWN gold against the mapped example's gold. Must be 1.0.

    The answer span is already in the shard -- it is what the label mask marks -- so this needs no
    generation and costs nothing. If the instance/source map is off by even one row, the gold of a
    different example is being scored and this collapses toward 0, which is exactly the failure that
    produced an entire near-zero results table while every part looked individually reasonable.
    """
    bad, tot = [], 0
    for t, idxs in sorted(eval_idx.items()):
        sp = spec_for(t); sc = []
        for gi in idxs:
            s_, e_ = inst[gi]; m = msk[s_:e_]; cut = int(np.argmax(m))
            gold_text = tok.decode(ids[s_+cut:e_].astype(np.int64).tolist(), skip_special_tokens=True)
            ex = ex_for(gi); nd = len(ex.get("documents", []) or [])
            pr = sp.parse(gold_text, nd)
            if pr is None: sc.append(0.0); continue
            try: sc.append(float(score_any(sp, pr, ex)[sp.primary_metric]))
            except Exception: sc.append(0.0)
        mean = float(np.mean(sc)) if sc else 0.0
        tot += 1
        p0(f"[self-check] {t:<14} gold-vs-gold {mean:.4f}  (must be 1.0)", flush=True)
        if mean < 0.99: bad.append((t, mean))
    if bad:
        raise SystemExit(
            "gold does not self-score on: " + ", ".join(f"{t}={v:.3f}" for t, v in bad) +
            ". The shard instances and the source rows are misaligned, so every number this "
            "script would print is a generation scored against some other example's gold."
        )
    p0(f"[self-check] PASS on all {tot} tasks -- instance/source map is correct", flush=True)

if not a.no_self_check:
    self_check()

BANDS = [int(x) for x in a.bands.split(",") if x.strip()]
def band_of(i):
    L = inst[i][1] - inst[i][0]
    for b in BANDS:
        if L <= b: return f"<={b//1024}k"
    return f">{BANDS[-1]//1024}k"

base = grade_all() if a.grade_base else {}
for t, (f, pr, nn, mt, bd) in sorted(base.items()):
    p0(f"[BASE] {a.tag} {t:<14} {mt} {f:.4f}  parse {pr:.3f}  eval_size={nn}", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9,0.95), eps=1e-8)
# optimizer steps PER RANK: DDP shards the data, so each rank walks 1/WORLD of it
total = ((len(train_idx)//max(WORLD,1))//a.bs)*a.epochs
warm = max(1, int(0.03*total)); step = 0; t0 = time.time()
for ep in range(a.epochs):
    my_idx = train_idx[RANK::WORLD] if DDP_ON else train_idx
    for i in range(0, len(my_idx)-a.bs+1, a.bs):
        for g in opt.param_groups: g["lr"] = a.lr*min(1.0,(step+1)/warm)*(1-step/max(total,1))
        x, y = batch(my_idx[i:i+a.bs])
        loss = loss_on_answer(x, y)
        if loss is None:
            step += 1; continue
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 50 == 0:
            p0(f"[train] {step}/{total} loss {loss.detach().item():.4f} {(time.time()-t0)/step:.2f}s/step", flush=True)
fin = grade_all()
for t, (f, pr, nn, mt, bd) in sorted(fin.items()):
    b0 = base.get(t, (float('nan'),))[0]
    bs = "  ".join(f"{k}:{v['score']:.3f}(n={v['eval_size']})" for k, v in sorted(bd.items()))
    p0(f"[FINAL] {a.tag} {t:<14} {mt} {b0:.4f}->{f:.4f}  parse {pr:.3f}  eval_size={nn}  | {bs}",
       flush=True)
if RANK == 0:
  os.makedirs(a.out_dir, exist_ok=True)
  json.dump({"tag":a.tag,"ckpt":a.ckpt,"train_n":len(train_idx),"steps":step,
           "base":{t:{"score":v[0],"parse":v[1],"eval_size":v[2],"metric":v[3],"by_band":v[4]} for t,v in base.items()},
           "final":{t:{"score":v[0],"parse":v[1],"eval_size":v[2],"metric":v[3],"by_band":v[4]} for t,v in fin.items()}},
          open(f"{a.out_dir}/{a.tag}.json","w"), indent=2)
if DDP_ON: dist.destroy_process_group()
p0("MT_DONE", flush=True)
