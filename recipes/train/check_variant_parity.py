"""
Parity check for ``recipes/train/variants.py`` -- no GPU, configs + meta-device models only.

Three checks, each a hard assertion:

1. **Refactor safety** (old vs new ``train_ctc_suite``): for every case, ``resolve_plan`` (model
   config, instance source, curriculum) and ``build_train_module_config`` produce the same
   ``as_dict()`` at the pre-refactor commit (``--old-root``) and now. Run in subprocesses so each
   tree imports its own ``olmo_core``.
2. **Cross-recipe parity** (new ``train_ctc_suite`` vs the setA path, same options): attention
   mixer config (minus the backend), ``document_chunk_attention``, the state ``enable_*`` adds to a
   meta-device model (every attribute added/changed vs that recipe's own dense model, on every
   submodule, plus new parameter names/shapes), the variant's callbacks, the optimizer's
   param-group structure (built on the meta model), ``microbatch_sort_pad_id``, the base-load mode
   and the data override. The recipes' intended differences (batch, LR, CP, AC, backend...) are
   printed, not asserted.
3. **Dense setA unchanged**: ``build_setA_experiment(...).as_dict()`` for every arm equals the
   pre-refactor build (timestamps normalized), and setA refuses the arms it cannot run.

Run (compute node; see recipes/README.md)::

    PYTHONPATH=src python recipes/train/check_variant_parity.py --old-root <633919f00 tree>
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

#: case -> (train_ctc_suite argv, setA variant or None (setA refuses it), setA --variant-opts)
CASES: Dict[str, Tuple[List[str], Optional[str], str]] = {
    "full": (["--variant", "full", "--pack"], None, ""),
    "chunked": (["--variant", "chunked", "--pack"], "chunked", ""),
    "chunked-mix": (
        ["--variant", "chunked-mix", "--pack", "--mix-start-p", "0.7"],
        "chunked-mix",
        "mix_start_p=0.7",
    ),
    "sparselandmark": (["--variant", "sparselandmark"], "sparselandmark", ""),
    "pooledkv": (["--variant", "pooledkv", "--pooled-keep-prob", "0.2"], "pooledkv", ""),
    "ffnmoe": (
        ["--variant", "ffnmoe", "--pack", "--ffn-moe-target", "0.3", "--ffn-moe-start-layer", "4"],
        "ffnmoe",
        "ffn_moe_target=0.3,ffn_moe_start_layer=4",
    ),
    "softtoken": (["--variant", "softtoken", "--st-gold-blind"], "softtoken", ""),
    "kvroute": (
        ["--variant", "kvroute", "--pack", "--kv-route-target", "0.4"],
        "kvroute",
        "kv_route_target=0.4",
    ),
    # No --block-skip-target: Transformer.enable_block_skip passes compact=/capacity_frac= that
    # BlockSkipHolder on this branch does not accept (TypeError in BOTH recipes, pre-existing), so
    # the block-skip router cannot be built here at all. Config-level parity still covers it.
    "flexcompute": (
        ["--variant", "flexcompute", "--pack", "--kv-route-target", "0.4"],
        "flexcompute",
        "kv_route_target=0.4",
    ),
    "flexcompute-joint": (
        ["--variant", "flexcompute", "--pack", "--kv-route-off", "--flex-joint-target", "0.3"],
        "flexcompute",
        "kv_route_off=true,flex_joint_target=0.3",
    ),
    "ffndrop": (
        ["--variant", "full", "--pack", "--ffn-drop-max-rate", "0.3", "--ffn-drop-layer-prob", "0.1"],
        "ffndrop",
        "ffn_drop_max_rate=0.3,ffn_drop_layer_prob=0.1",
    ),
}
COMMON = ["--task", "parity", "--run-name", "parity", "--model-family", "qwen3_5"]
COMMON += ["--model-scale", "4b", "--seq-len", "32768", "--seed", "0"]
WORLD = 8
#: The setA 32k arm: 4 grad-accum forwards per step. Both recipes get this horizon.
SETA_ARM, ACCUM, TOTAL_STEPS = "32k", 4, 100
TOTAL_CALLS = ACCUM * TOTAL_STEPS
TS = re.compile(r"\d{8}T\d{4,6}([+-]\d{4})?")


def _norm(obj: Any) -> Any:
    return json.loads(TS.sub("<TS>", json.dumps(obj, default=str, sort_keys=True)))


def _load(path: str, name: str):
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _stub_beaker() -> None:
    """amandab's builder asks Beaker for the root dir and user name; answer offline."""
    import olmo_core.internal.common as common

    common.get_beaker_username = lambda: "parity"  # type: ignore[assignment]
    common.get_root_dir = lambda cluster: "/weka/oe-training-default/ai2-llm"  # type: ignore


# ----------------------------------------------------------------------------------------------
# per-tree builders
# ----------------------------------------------------------------------------------------------


def ctc_module(root: str):
    return _load(f"{root}/recipes/train/research_trainer/train_ctc_suite.py", "train_ctc_suite")


def ctc_build(root: str, case: str, shard: str):
    mod = ctc_module(root)
    sys.argv = ["train_ctc_suite.py", "--data", shard, *COMMON, *CASES[case][0]]
    opts = mod.parse_args()
    plan = mod.resolve_plan(opts, WORLD)
    tm = mod.build_train_module_config(opts, WORLD)
    return mod, opts, plan, tm


def setA_module(root: str):
    _stub_beaker()
    mod = _load(f"{root}/recipes/train/setA/_qwen35_setA_common.py", "_qwen35_setA_common")
    amandab = sys.modules["_qwen35_xlong5_dolci25_256k_common"]
    amandab.get_root_dir = lambda cluster: "/weka/oe-training-default/ai2-llm"  # type: ignore
    # The Beaker launch config needs a git checkout + Beaker API; it is not built by train and is
    # unchanged by this refactor, so both trees skip it (the builder handles None).
    amandab.build_launch_config = lambda **kw: None  # type: ignore
    return mod


def setA_build(mod, arm: str, **kw):
    from olmo_core.internal.experiment import CliContext, SubCmd

    ctx = CliContext("x.py", SubCmd.train, "parity", "ai2/jupiter-cirrascale-2", [])
    return mod.build_setA_experiment(ctx, arm=arm, **kw)


def dump(root: str, what: str, arg: str, shard: str) -> Dict[str, Any]:
    """Subprocess body: the as_dicts one tree produces."""
    for k in [k for k in os.environ if k.startswith("SETA_")]:
        del os.environ[k]
    if what == "ctc":
        _, _, plan, tm = ctc_build(root, arg, shard)
        return _norm(
            {
                "model": plan["model_config"].as_dict(),
                "source": plan["instance_source_config"].as_dict(),
                "curriculum": plan["curriculum"],
                "train_module": tm.as_dict(),
            }
        )
    return _norm(setA_build(setA_module(root), arg).as_dict())


def run_dump(root: str, what: str, arg: str, shard: str, scratch: str) -> Dict[str, Any]:
    out = os.path.join(scratch, f"{os.path.basename(root)}-{what}-{arg}.json")
    env = dict(os.environ, PYTHONPATH=f"{root}/src", PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, os.path.abspath(__file__), "--dump", what, arg, root, shard, out]
    r = subprocess.run(cmd, env=env, cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"dump {what}/{arg} in {root} failed:\n{r.stdout[-3000:]}\n{r.stderr[-5000:]}")
    with open(out) as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------
# summaries
# ----------------------------------------------------------------------------------------------


def _summ(v: Any) -> Any:
    import torch

    if isinstance(v, torch.Tensor):
        return f"tensor{tuple(v.shape)}:{v.dtype}"
    if isinstance(v, torch.nn.Module):
        return f"module:{type(v).__name__}"
    if isinstance(v, dict):
        return {str(k): _summ(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_summ(x) for x in v]
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return re.sub(r"0x[0-9a-f]+", "ADDR", repr(v))[:300]


def model_state(model) -> Dict[str, Any]:
    """Every submodule's non-child attributes + parameter shapes / requires_grad."""
    state: Dict[str, Any] = {}
    for name, mod in model.named_modules():
        attrs = {
            k: _summ(v)
            for k, v in vars(mod).items()
            if k not in ("_modules", "_parameters", "_buffers") and not k.endswith("hooks")
        }
        attrs["__hooks__"] = len(getattr(mod, "_forward_pre_hooks", {})) + len(
            getattr(mod, "_forward_hooks", {})
        )
        state[f"mod:{name}"] = attrs
    for n, p in model.named_parameters():
        state[f"param:{n}"] = [list(p.shape), bool(p.requires_grad)]
    return state


def added_state(variant: Dict[str, Any], dense: Dict[str, Any]) -> Dict[str, Any]:
    """What the variant added or changed relative to the same recipe's dense model."""
    out: Dict[str, Any] = {}
    for key, val in variant.items():
        if key.startswith("param:"):
            if dense.get(key) != val:
                out[key] = val
            continue
        base = dense.get(key, {})
        diff = {k: v for k, v in val.items() if base.get(k, "<absent>") != v}
        if diff:
            out[key] = diff
    for key in dense:
        if key.startswith("param:") and key not in variant:
            out[key] = "<removed>"
    return out


def group_structure(optim_cfg, model) -> List[Dict[str, Any]]:
    names = {id(p): n for n, p in model.named_parameters()}
    groups = optim_cfg.build_groups(model, strict=True)
    out = []
    for g in groups if isinstance(groups, list) and groups and isinstance(groups[0], dict) else []:
        pn = sorted(names[id(p)] for p in g["params"])
        out.append(
            {
                "n": len(pn),
                "names_sha": hashlib.sha1("\n".join(pn).encode()).hexdigest()[:12],
                "opts": _norm({k: v for k, v in g.items() if k != "params"}),
            }
        )
    return sorted(out, key=lambda d: (d["n"], d["names_sha"]))


def attn_mixer(cfg) -> Dict[str, Any]:
    blk = cfg.block
    mixer = blk["attn"].sequence_mixer if isinstance(blk, dict) else blk.sequence_mixer
    d = _norm(mixer.as_dict())
    d.pop("backend", None)  # flash_2 (research trainer) vs flash_3 (setA): intended
    return d


def flat_diff(a: Any, b: Any, prefix: str = "") -> List[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for k in sorted(set(a) | set(b)):
            out += flat_diff(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{prefix}.{k}")
        return out
    return [] if a == b else [f"{prefix}: {a!r} vs {b!r}"]


# ----------------------------------------------------------------------------------------------
# the checks
# ----------------------------------------------------------------------------------------------


def check_refactor(new_root: str, old_root: str, shard: str, scratch: str) -> None:
    print("== 1. train_ctc_suite: pre-refactor vs now (resolve_plan + train-module as_dict)")
    for case in CASES:
        old = run_dump(old_root, "ctc", case, shard, scratch)
        new = run_dump(new_root, "ctc", case, shard, scratch)
        diff = flat_diff(old, new)
        assert not diff, f"{case}: train_ctc_suite config changed by the refactor:\n" + "\n".join(
            diff[:20]
        )
        print(f"  {case:18s} identical ({len(json.dumps(new))} chars of config)")


def check_setA_dense(new_root: str, old_root: str, shard: str, scratch: str) -> None:
    print("== 3. setA dense build: pre-refactor vs now (config.as_dict(), timestamps normalized)")
    for arm in ("32k", "256k", "256k-2node"):
        old = run_dump(old_root, "setA", arm, shard, scratch)
        new = run_dump(new_root, "setA", arm, shard, scratch)
        diff = flat_diff(old, new)
        assert not diff, f"setA {arm}: dense config changed:\n" + "\n".join(diff[:20])
        print(f"  {arm:10s} byte-identical ({len(json.dumps(new, sort_keys=True))} chars)")


def check_cross(new_root: str, shard: str) -> None:
    print("== 2. train_ctc_suite path vs setA path, same variant options")
    import torch  # noqa: F401

    sys.path.insert(0, f"{new_root}/recipes/train")
    import variants as V

    setA = setA_module(new_root)
    dense_setA = setA_build(setA, SETA_ARM, variant="dense", variant_opts="")
    m = dense_setA.model.build(init_device="meta")
    setA_dense_state = model_state(m)
    dense_setA_cbs = set(dense_setA.trainer.callbacks)
    _, _, ctc_dense_plan, _ = ctc_build(new_root, "full", shard)
    ctc_dense_state = model_state(ctc_dense_plan["model_config"].build(init_device="meta"))

    intended_printed = False
    for case, (_, setA_name, setA_opts) in CASES.items():
        if setA_name is None:
            continue
        mod, opts, plan, tm = ctc_build(new_root, case, shard)
        variant = V.get_variant(setA_name)
        spec = f"{setA_opts};total_steps={TOTAL_STEPS}" if setA_opts else f"total_steps={TOTAL_STEPS}"
        spec = spec.replace(",", ";")
        if variant.setA_blocker:
            try:
                setA_build(setA, SETA_ARM, variant=setA_name, variant_opts=spec)
            except ValueError as exc:
                print(f"  {case:18s} setA REFUSES (as intended): {str(exc)[:110]}...")
                continue
            raise AssertionError(f"setA accepted {setA_name} despite setA_blocker")
        if variant.cp_blocker:
            try:
                setA_build(setA, "256k", variant=setA_name, variant_opts=spec)
                raise AssertionError(f"setA accepted {setA_name} on a CP arm despite cp_blocker")
            except ValueError:
                pass
        # chunked-mix: give the research trainer the same anneal horizon as setA (steps x accum).
        if case == "chunked-mix":
            opts._mix_total_forwards = TOTAL_CALLS
            plan["model_config"] = mod.build_model_config(opts)
        cfg_s = setA_build(setA, SETA_ARM, variant=setA_name, variant_opts=spec)
        cfg_c = plan["model_config"]

        # attention config + document_chunk_attention
        assert attn_mixer(cfg_c) == attn_mixer(cfg_s.model), (
            case,
            flat_diff(attn_mixer(cfg_c), attn_mixer(cfg_s.model)),
        )
        assert _norm(cfg_c.document_chunk_attention) == _norm(
            cfg_s.model.document_chunk_attention
        ), (case, cfg_c.document_chunk_attention, cfg_s.model.document_chunk_attention)

        # enable_* state on meta models (each vs its own recipe's dense model)
        mc = cfg_c.build(init_device="meta")
        vctx = mod.variant_context(opts, rank=0, total_calls=TOTAL_CALLS, calls_per_step=ACCUM)
        for v, o in mod.active_variants(opts):
            v.post_build_model(mc, o, vctx)
        ms = cfg_s.model.build(init_device="meta")
        hooks = setA.setA_train_hooks()
        hooks.post_model(cfg_s, ms)
        add_c = added_state(model_state(mc), ctc_dense_state)
        add_s = added_state(model_state(ms), setA_dense_state)
        d = flat_diff(add_c, add_s)
        assert not d, f"{case}: enable_* state differs:\n" + "\n".join(d[:20])
        n_new = sum(1 for k in add_c if k.startswith("param:"))

        # callbacks
        cbs_c = {}
        for v, o in mod.active_variants(opts):
            cbs_c.update(v.extra_callbacks(o, vctx))
        cbs_s = {k: cfg_s.trainer.callbacks[k] for k in set(cfg_s.trainer.callbacks) - dense_setA_cbs}
        cbs_s.pop("flop_meter")  # every arm in both recipes (seq_len differs: intended)
        assert sorted(cbs_c) == sorted(cbs_s), (case, sorted(cbs_c), sorted(cbs_s))
        for k in cbs_c:
            assert _summ(cbs_c[k]) == _summ(cbs_s[k]), (case, k, cbs_c[k], cbs_s[k])

        # optimizer param-group structure (built on the meta models)
        gc, gs = group_structure(tm.optim, mc), group_structure(cfg_s.train_module.optim, ms)
        assert gc == gs, (case, gc, gs)

        # train-module / load / data pieces
        assert tm.microbatch_sort_pad_id == cfg_s.train_module.microbatch_sort_pad_id, case
        assert (cfg_s.trainer.load_path is None) == variant.new_params, case
        src_c, src_s = plan["instance_source_config"], cfg_s.dataset[0]
        if variant.data_path == "landmark":
            for f in ("mem_freq", "mem_id", "pad_id", "sequence_length", "num_landmarks"):
                assert getattr(src_c, f) == getattr(src_s, f), (case, f)
            assert type(src_c) is type(src_s), case
        print(
            f"  {case:18s} match: attn mixer, doc_chunk_attention, enable_* state "
            f"({len(add_c)} added/changed keys, {n_new} new params), callbacks {sorted(cbs_c)}, "
            f"{len(gc)} optim groups"
        )
        if not intended_printed:
            intended_printed = True
            tm_c = _norm(tm.as_dict())
            tm_s = _norm(cfg_s.train_module.as_dict())
            tm_c["optim"].pop("group_overrides")
            tm_s["optim"].pop("group_overrides")
            print("  intended recipe differences (train module, research trainer vs setA 32k):")
            for line in flat_diff(tm_c, tm_s):
                print(f"    {line[:160]}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--dump":
        what, arg, root, shard, out = sys.argv[2:7]
        res = dump(root, what, arg, shard)
        with open(out, "w") as f:
            json.dump(res, f)
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old-root", required=True, help="tree of the pre-refactor commit")
    ap.add_argument("--new-root", default=os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    ap.add_argument("--skip-refactor", action="store_true", help="skip check 1 (already passed)")
    args = ap.parse_args()
    scratch = tempfile.mkdtemp(prefix="variant_parity_")
    shard = os.path.join(scratch, "shard")
    os.makedirs(shard)
    with open(os.path.join(shard, "metadata.json"), "w") as f:
        json.dump(
            {"num_instances": 20000, "max_example_len": 8000, "marker_set": "qwen3_5",
             "num_tokens": 100_000_000}, f)
    if not args.skip_refactor:
        check_refactor(args.new_root, args.old_root, shard, scratch)
    check_setA_dense(args.new_root, args.old_root, shard, scratch)
    check_cross(args.new_root, shard)
    print("ALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
