"""
Where does prefill reuse lose the prompt? Three experiments that name the layer, not the symptom.

`parity_probe.py` established that reusing a shared prefix changes the model's output. This script
asks *why*, and separates the two candidate explanations that a logit delta cannot distinguish:

* **bf16 kernel noise.** The two paths feed the model different sequence lengths (one 3.6k-token
  forward vs a 2.0k prefill plus a 1.6k suffix), and flash-attention / chunked kernels reduce in a
  different order at different lengths. That produces small, semantically meaningless deltas.
* **a structurally dropped state.** These checkpoints are Qwen3.5 hybrids -- ``block_pattern`` is
  ``['gdn','gdn','gdn','attn']``, so 24 of 32 layers are GatedDeltaNet, which carries a *recurrent*
  state and a conv window rather than a KV cache. ``prefix_cache.py`` does snapshot and restore that
  state, but :meth:`GatedDeltaNet.forward` only consumes it when ``T == 1``::

      use_precomputed = cache is not None and cache.has_state and T_og == 1

  A suffix is a multi-token call, so it takes the ``else`` branch: the conv windows are overwritten
  from the suffix's own inputs and ``dispatch_chunk_gated_delta_rule`` is called **without**
  ``initial_state``. The restored state is discarded before it is ever read.

The three experiments:

1. ``ablate`` -- run the suffix twice, once with the restored GDN state and once with it wiped. If
   the logits are bit-identical, the restored state provably has no effect and the second bullet is
   the answer. This is the decisive one, and it needs no second checkpoint.
2. ``layers`` -- hook every block and compare the last position's hidden state, plain vs reused,
   layer by layer. Says where the error enters and how it grows.
3. ``synthetic`` -- build small random-init models from the checkpoint's own config with
   ``block_pattern`` forced to ``['attn']`` and to ``['gdn']``, and run the same comparison. The
   attention-only model is the control the suite has no checkpoint for: if reuse is exact there and
   broken on the GDN-only model, the mechanism is settled with nothing else varying.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

MIN_PREFIX = 64


def build_parser() -> argparse.ArgumentParser:
    """:returns: The argument parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--rows", type=int, default=4, help="rows of one corpus group to use")
    ap.add_argument("--query-position", default="both")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--experiments",
        default="synthetic,ablate,layers",
        help="comma list: synthetic, ablate, layers",
    )
    return ap


# ── a generation-module stand-in, so a random-init model can be driven like a checkpoint ─────────


class _StubGM:
    """
    The two methods :mod:`ctc.eval.prefix_cache` needs, over a bare model.

    ``prepare_inference_cache`` mirrors
    :meth:`TransformerGenerationModule.prepare_inference_cache` exactly -- attention blocks get a KV
    cache, recurrent mixers get a conv+recurrent state cache. Reimplemented rather than imported so
    a model built from a config, with no checkpoint behind it, can be driven by the same code.

    :param model: A built :class:`~olmo_core.nn.transformer.Transformer`.
    """

    def __init__(self, model):
        self.model = model

    def prepare_inference_cache(self, batch_size: int, max_seq_len: int) -> None:
        """
        :param batch_size: Inference batch size.
        :param max_seq_len: Cache length.
        """
        from olmo_core.nn.attention import Attention

        for block in self.model.blocks.values():
            mixer = block.attention
            if isinstance(mixer, Attention):
                if mixer.kv_cache_manager is None:
                    mixer.init_kv_cache_manager(batch_size, max_seq_len)
                else:
                    mixer.kv_cache_manager.reset(batch_size, max_seq_len)
            elif hasattr(mixer, "init_state_cache"):
                mixer.init_state_cache(batch_size, max_seq_len)


def _plain_last_logits(torch, gm, ids, device, max_length):
    """
    :returns: Logits for the last token of ``ids`` after a full prefill.
    """
    with torch.no_grad():
        gm.prepare_inference_cache(1, max_length)
        leftpad = torch.zeros(1, dtype=torch.int32, device=device)
        out = gm.model(
            torch.tensor([list(ids)], device=device), logits_to_keep=1, cache_leftpad=leftpad
        )
    return out[0, -1].detach().float().cpu().clone()


def _shared_last_logits(torch, gm, prefills, device, max_length):
    """
    :returns: One logit vector per prompt, via the shared-prefix path.
    """
    from ctc.eval.prefix_cache import generate_group_with_shared_prefix

    captured: List[Any] = []
    generate_group_with_shared_prefix(
        gm,
        prefills,
        device=device,
        max_length=max_length,
        decode_fn=lambda logits: captured.append(logits[0, -1].detach().float().cpu().clone())
        or "",
        min_prefix=MIN_PREFIX,
    )
    return captured


# ── 1. synthetic: same code, one architectural knob ──────────────────────────────────────────────


def experiment_synthetic(args) -> Dict[str, Any]:
    """
    Run the reuse comparison on small random-init models that differ only in ``block_pattern``.

    Built from the checkpoint's own model config so every other field -- head counts, norms, RoPE,
    conv width -- is the suite's, not an invention. Weights are random, which is fine: the question
    is whether two ways of computing the *same function* agree, and that does not need a trained
    model. It does need the real kernels, hence a GPU.

    :param args: Parsed arguments.

    :returns: One entry per architecture.
    """
    import torch

    from olmo_core.nn.transformer import TransformerConfig

    base = json.loads((Path(args.ckpt) / "config.json").read_text())["model"]
    prefix_len, suffix_len, rows = 512, 32, 4
    torch.manual_seed(0)

    results = []
    for label, pattern in (
        ("attn-only", ["attn"]),
        ("gdn-only", ["gdn"]),
        ("hybrid (the suite's)", ["gdn", "gdn", "gdn", "attn"]),
    ):
        spec = dict(base)
        spec["n_layers"] = 4
        spec["vocab_size"] = 4096
        spec["block_pattern"] = pattern
        model = TransformerConfig.from_dict(spec).build(init_device="cuda")
        # Random but *initialised*: an unbuilt parameter buffer is whatever was in memory, and
        # comparing two NaN logit vectors would report agreement or disagreement at random.
        model.init_weights(max_seq_len=2048, device=torch.device("cuda"))
        # init_weights leaves the parameters in fp32; flash-attn accepts only fp16/bf16, so the
        # attention arm dies at its first forward without this cast.
        model.to(torch.bfloat16)
        model.eval()
        gm = _StubGM(model)

        shared = torch.randint(0, 4096, (prefix_len,)).tolist()
        prefills = [
            shared + torch.randint(0, 4096, (suffix_len,)).tolist() for _ in range(rows)
        ]

        plain = [_plain_last_logits(torch, gm, ids, "cuda", 2048) for ids in prefills]
        reused = _shared_last_logits(torch, gm, prefills, "cuda", 2048)
        diffs = [float((a - b).abs().max()) for a, b in zip(plain, reused)]
        identical = [bool(torch.equal(a, b)) for a, b in zip(plain, reused)]

        results.append(
            {
                "arch": label,
                "block_pattern": pattern,
                "max_abs_logit_diff": max(diffs),
                "rows_bitwise_identical": sum(identical),
                "rows": rows,
                "argmax_agrees": sum(
                    int(int(a.argmax()) == int(b.argmax())) for a, b in zip(plain, reused)
                ),
            }
        )
        print(f"[localize] synthetic {label:<22} max|dlogit|={max(diffs):.4g}  "
              f"bitwise identical {sum(identical)}/{rows}", flush=True)
        del model, gm
        torch.cuda.empty_cache()
    return {"experiment": "synthetic", "prefix_len": prefix_len, "results": results}


# ── 2. ablate: is the restored GDN state read at all? ────────────────────────────────────────────


def experiment_ablate(backend, prefills, args) -> Dict[str, Any]:
    """
    Wipe the restored recurrent state and see whether the suffix's logits move.

    The comparison is between "reuse the prefix, restoring GDN state as ``prefix_cache`` does" and
    "reuse the prefix, then throw the GDN state away". If those two agree bit for bit, the restore
    is decorative: the GDN layers rebuild from zero either way, and the shared prefix is invisible
    to them.

    :param backend: A loaded native backend.
    :param prefills: Tokenized prompts of one corpus group.
    :param args: Parsed arguments.

    :returns: The comparison.
    """
    import torch

    from ctc.eval.prefix_cache import (
        longest_common_token_prefix,
        restore_recurrent_states,
        rewind_kv_cursor,
        snapshot_recurrent_states,
    )

    gm, device, max_length = backend.gm, backend.device, args.max_length
    shared = longest_common_token_prefix(prefills)

    with torch.no_grad():
        gm.prepare_inference_cache(1, max_length)
        leftpad = torch.zeros(1, dtype=torch.int32, device=device)
        gm.model(
            torch.tensor([prefills[0][:shared]], device=device),
            logits_to_keep=1,
            cache_leftpad=leftpad,
        )
        snapshots = snapshot_recurrent_states(gm.model)

        restored, wiped, suffix_only = [], [], []
        for ids in prefills:
            suffix = list(ids[shared:])

            rewind_kv_cursor(gm.model, shared)
            restore_recurrent_states(snapshots)
            out = gm.model(torch.tensor([suffix], device=device), logits_to_keep=1)
            restored.append(out[0, -1].detach().float().cpu().clone())

            rewind_kv_cursor(gm.model, shared)
            restore_recurrent_states(snapshots)
            # Throw away exactly what the restore just put back.
            for cache, _ in snapshots:
                cache.conv_state_q.zero_()
                cache.conv_state_k.zero_()
                cache.conv_state_v.zero_()
                cache.recurrent_state = None
                cache.has_state = False
            out = gm.model(torch.tensor([suffix], device=device), logits_to_keep=1)
            wiped.append(out[0, -1].detach().float().cpu().clone())

            # And what the model would say if it had never seen the prefix at all -- the reference
            # for "the GDN layers behave as if the prefix did not exist".
            suffix_only.append(_plain_last_logits(torch, gm, suffix, device, max_length))

    per_row = []
    for i, (a, b, c) in enumerate(zip(restored, wiped, suffix_only)):
        per_row.append(
            {
                "row": i,
                "restored_vs_wiped_max_abs": float((a - b).abs().max()),
                "restored_vs_wiped_identical": bool(torch.equal(a, b)),
                "restored_vs_suffix_only_max_abs": float((a - c).abs().max()),
            }
        )
    return {
        "experiment": "ablate",
        "prefix_len": shared,
        "rows": len(prefills),
        "all_restored_equal_wiped": all(r["restored_vs_wiped_identical"] for r in per_row),
        "per_row": per_row,
    }


# ── 3. layers: where does the divergence enter ───────────────────────────────────────────────────


def experiment_layers(backend, prefills, args) -> Dict[str, Any]:
    """
    Compare the last position's hidden state after every block, plain vs reused.

    :param backend: A loaded native backend.
    :param prefills: Tokenized prompts of one corpus group.
    :param args: Parsed arguments.

    :returns: Per-block divergence for the first row of the group, with each block's mixer type.
    """
    import torch

    from ctc.eval.prefix_cache import (
        longest_common_token_prefix,
        restore_recurrent_states,
        rewind_kv_cursor,
        snapshot_recurrent_states,
    )
    from olmo_core.nn.attention import Attention

    gm, device, max_length = backend.gm, backend.device, args.max_length
    shared = longest_common_token_prefix(prefills)
    ids = list(prefills[0])
    captured: Dict[str, Any] = {}
    handles = []

    def hook(name):
        def fn(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor[0, -1].detach().float().cpu().clone()

        return fn

    for name, block in gm.model.blocks.items():
        handles.append(block.register_forward_hook(hook(name)))

    try:
        with torch.no_grad():
            gm.prepare_inference_cache(1, max_length)
            leftpad = torch.zeros(1, dtype=torch.int32, device=device)
            gm.model(
                torch.tensor([ids], device=device), logits_to_keep=1, cache_leftpad=leftpad
            )
            plain = dict(captured)
            captured.clear()

            gm.prepare_inference_cache(1, max_length)
            gm.model(
                torch.tensor([ids[:shared]], device=device),
                logits_to_keep=1,
                cache_leftpad=leftpad,
            )
            snapshots = snapshot_recurrent_states(gm.model)
            captured.clear()
            rewind_kv_cursor(gm.model, shared)
            restore_recurrent_states(snapshots)
            gm.model(torch.tensor([ids[shared:]], device=device), logits_to_keep=1)
            reused = dict(captured)
    finally:
        for handle in handles:
            handle.remove()

    rows = []
    for name, block in gm.model.blocks.items():
        if name not in plain or name not in reused:
            continue
        a, b = plain[name], reused[name]
        rows.append(
            {
                "block": name,
                "mixer": "attn" if isinstance(block.attention, Attention) else "gdn",
                "max_abs_diff": float((a - b).abs().max()),
                "rel_diff": float((a - b).norm() / max(1e-9, float(a.norm()))),
            }
        )
    return {"experiment": "layers", "prefix_len": shared, "prompt_tokens": len(ids), "blocks": rows}


def _group_prefills(backend, spec, path, rows, query_position):
    """
    :returns: Tokenized prompts for the first corpus group of ``path``, at most ``rows`` of them.
    """
    from ctc.eval.prefill import build_prefills

    examples = []
    target = None
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            key = row.get("corpus_id", "__none")
            if target is None:
                target = key
            if key != target:
                break
            examples.append(row)
            if len(examples) >= rows:
                break

    backend.query_position = query_position
    backend._prefill = None
    prompts = [spec.build_prompt(ex, query_position=query_position) for ex in examples]
    return build_prefills(backend.prefill_for(spec.name), prompts, examples)


def main() -> int:
    """:returns: 0."""
    args = build_parser().parse_args()
    wanted = [e.strip() for e in args.experiments.split(",") if e.strip()]
    reports = []

    # Each experiment is independent evidence, so one failing must not take the others down with
    # it: the first run of this script lost the decisive GDN ablation because the synthetic models
    # -- a nice-to-have control -- hit a dtype error three lines into the job.
    def attempt(name, fn):
        try:
            reports.append(fn())
        except Exception as e:  # noqa: BLE001 - a broken control must not hide a working probe
            import traceback

            traceback.print_exc()
            reports.append({"experiment": name, "failed": f"{type(e).__name__}: {e}"})
            print(f"[localize] {name} FAILED: {type(e).__name__}: {e}", flush=True)

    if "synthetic" in wanted:
        attempt("synthetic", lambda: experiment_synthetic(args))

    if {"ablate", "layers"} & set(wanted):
        import ctc.tasks
        from ctc.eval.backends.native import NativeBackend
        from ctc.format import registry

        ctc.tasks.load_all()
        spec = registry.get(args.spec)
        backend = NativeBackend(
            Path(args.ckpt),
            tokenizer=args.tokenizer,
            attn="full",
            max_length=args.max_length,
            query_position=args.query_position,
        )
        prefills = _group_prefills(backend, spec, args.data, args.rows, args.query_position)
        print(f"[localize] {len(prefills)} prompts from one corpus group", flush=True)

        if "ablate" in wanted:

            def run_ablate():
                report = experiment_ablate(backend, prefills, args)
                print(
                    f"[localize] ablate: restored state == wiped state for all rows? "
                    f"{report['all_restored_equal_wiped']}",
                    flush=True,
                )
                for row in report["per_row"]:
                    print(
                        f"[localize]   row{row['row']}  restored-vs-wiped max|d|="
                        f"{row['restored_vs_wiped_max_abs']:.4g}  restored-vs-suffix-only "
                        f"max|d|={row['restored_vs_suffix_only_max_abs']:.4g}",
                        flush=True,
                    )
                return report

            attempt("ablate", run_ablate)

        if "layers" in wanted:

            def run_layers():
                report = experiment_layers(backend, prefills, args)
                for row in report["blocks"]:
                    print(
                        f"[localize] block {row['block']:>3} [{row['mixer']}]  "
                        f"max|d|={row['max_abs_diff']:.4g}  rel={row['rel_diff']:.4g}",
                        flush=True,
                    )
                return report

            attempt("layers", run_layers)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ckpt": args.ckpt, "reports": reports}, indent=2) + "\n")
    print(f"[localize] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
