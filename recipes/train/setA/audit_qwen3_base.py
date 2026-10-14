"""
Audit the Qwen3-4B-Base olmo-core base for the setA Qwen3 arm (``launch_setA_sft.py base-audit``).

The base is the existing conversion ``ctc_suite/bases/qwen3-4b-base-trainedmark`` (2026-07-23,
``convert_hf_to_olmo.py`` with HF-parity validation, then ``fix_marker_embeddings.py`` with
trained-delimiter donor rows). This job re-establishes both facts from the weights, on CPU:

1. **Marker audit, before and after** -- ``fix_marker_embeddings.audit`` (cosine AND norm gates) on
   the HF ``Qwen/Qwen3-4B-Base`` embedding (before) and on the olmo-core base (after), for
   ``<|box_start|>``/``<|box_end|>`` (the setA markers) and for all four repaired rows.
2. **Provenance** -- every other tensor checked against HF ``Qwen/Qwen3-4B-Base``: the embedding
   (all rows except the repaired ones), the untied ``lm_head`` (= HF's tied embedding), and every
   block's attention / norm / feed-forward weights, bit-exact after bf16 -> fp32. A mismatch fails.

It also writes a copy of the Qwen3-4B-Base tokenizer whose generation prompt closes the empty
thinking block (``--nothink-out``): see ``make_nothink_template``.
"""

import argparse
import glob
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "recipes", "data"))
sys.path.insert(0, os.path.join(REPO, "src"))

DEFAULT_BASE = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/bases/"
    "qwen3-4b-base-trainedmark/model_and_optim"
)
DEFAULT_NOTHINK = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/hf_tokenizers/Qwen3-4B-Base-nothink"
)

#: The stock Qwen3 generation-prompt branch, and its replacement. Stock: ``assistant\\n`` unless
#: ``enable_thinking=False``. Replacement: the empty thinking block unless ``enable_thinking=True``,
#: i.e. the generation prompt ends ``<think>\\n\\n</think>\\n\\n`` by default, as Qwen3.5-0.8B's does.
STOCK_GEN = "{%- if enable_thinking is defined and enable_thinking is false %}"
NOTHINK_GEN = "{%- if not (enable_thinking is defined and enable_thinking is true) %}"


def make_nothink_template(template: str) -> str:
    """:returns: ``template`` whose generation prompt closes the empty thinking block by default."""
    assert template.count(STOCK_GEN) == 1, "unexpected Qwen3 chat template"
    return template.replace(STOCK_GEN, NOTHINK_GEN)


def hf_to_olmo_pairs(n_layers: int):
    """(olmo key, HF key) for every tensor of a Qwen3 dense model (olmo-core's HF converter map)."""
    pairs = [
        ("model.embeddings.weight", "model.embed_tokens.weight"),
        ("model.lm_head.norm.weight", "model.norm.weight"),
        ("model.lm_head.w_out.weight", "model.embed_tokens.weight"),  # HF ties; olmo-core does not
    ]
    for i in range(n_layers):
        b, h = f"model.blocks.{i}", f"model.layers.{i}"
        pairs += [
            (f"{b}.attention.w_q.weight", f"{h}.self_attn.q_proj.weight"),
            (f"{b}.attention.w_k.weight", f"{h}.self_attn.k_proj.weight"),
            (f"{b}.attention.w_v.weight", f"{h}.self_attn.v_proj.weight"),
            (f"{b}.attention.w_out.weight", f"{h}.self_attn.o_proj.weight"),
            (f"{b}.attention.q_norm.weight", f"{h}.self_attn.q_norm.weight"),
            (f"{b}.attention.k_norm.weight", f"{h}.self_attn.k_norm.weight"),
            (f"{b}.attention_norm.weight", f"{h}.input_layernorm.weight"),
            (f"{b}.feed_forward_norm.weight", f"{h}.post_attention_layernorm.weight"),
            (f"{b}.feed_forward.w1.weight", f"{h}.mlp.gate_proj.weight"),
            (f"{b}.feed_forward.w2.weight", f"{h}.mlp.down_proj.weight"),
            (f"{b}.feed_forward.w3.weight", f"{h}.mlp.up_proj.weight"),
        ]
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--hf", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--nothink-out", default=DEFAULT_NOTHINK)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    import fix_marker_embeddings as fme
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from torch.distributed.checkpoint import FileSystemReader
    from transformers import AutoTokenizer

    from olmo_core.data.document_chunk_landmark import reserved_ids
    from olmo_core.distributed.checkpoint import load_keys

    ids = reserved_ids("qwen3")
    local = snapshot_download(args.hf, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    hf_cfg = json.load(open(os.path.join(local, "config.json")))
    print(f"HF {args.hf}: {local}  rope_theta={hf_cfg['rope_theta']} "
          f"max_pos={hf_cfg['max_position_embeddings']} tie={hf_cfg['tie_word_embeddings']}",
          flush=True)
    hf = {}
    for path in sorted(glob.glob(os.path.join(local, "*.safetensors"))):
        with safe_open(path, "pt") as f:
            for k in f.keys():
                hf[k] = f.get_tensor(k)

    report = {"base": args.base, "hf": args.hf}
    marker_sets = {
        "doc": ["doc_start", "doc_end"],
        "all_repaired": ["doc_start", "doc_end", "landmark", "pad"],
    }
    emb_hf = hf["model.embed_tokens.weight"].float()
    emb_olmo = next(iter(load_keys(args.base, ["model.embeddings.weight"]))).float()
    for which, emb in (("before_hf", emb_hf), ("after_olmo_base", emb_olmo)):
        for ms_name, ms in marker_sets.items():
            print(f"\n=== marker audit: {which} [{ms_name}] ===", flush=True)
            rep = fme.audit(emb, ids, fme._marker_ids(ids, ms))
            fme.print_audit(rep)
            report[f"{which}/{ms_name}"] = rep

    # ---- provenance: everything but the repaired rows is HF Qwen3-4B-Base, bit-exact ----
    keys = set(FileSystemReader(args.base).read_metadata().state_dict_metadata)
    pairs = hf_to_olmo_pairs(hf_cfg["num_hidden_layers"])
    missing = [o for o, _ in pairs if o not in keys]
    extra = sorted(k for k in keys if k.startswith("model.") and k not in {o for o, _ in pairs})
    print(f"\nolmo keys: {len(keys)}; expected model tensors {len(pairs)}; missing {missing[:5]}; "
          f"unexpected model.* {extra[:5]}", flush=True)
    repaired = [ids.doc_start, ids.doc_end, ids.landmark, ids.pad]
    keep = torch.ones(emb_hf.shape[0], dtype=torch.bool)
    keep[repaired] = False
    mismatches = []
    by_key = dict(pairs)
    olmo_keys = [o for o, _ in pairs if o in keys]
    for start in range(0, len(olmo_keys), 24):
        chunk = olmo_keys[start:start + 24]
        for o, t in zip(chunk, load_keys(args.base, chunk)):
            ref = hf[by_key[o]].float()
            t = t.float()
            if o in ("model.embeddings.weight", "model.lm_head.w_out.weight"):
                assert t.shape[0] >= ref.shape[0], (o, t.shape, ref.shape)
                t_cmp, ref_cmp = t[: ref.shape[0]][keep], ref[keep]
                if t.shape[0] > ref.shape[0]:
                    print(f"  {o}: {t.shape[0] - ref.shape[0]} padded rows past HF", flush=True)
            else:
                t_cmp, ref_cmp = t, ref
            if t_cmp.shape != ref_cmp.shape or not torch.equal(t_cmp, ref_cmp):
                diff = (t_cmp - ref_cmp).abs().max().item() if t_cmp.shape == ref_cmp.shape else None
                mismatches.append((o, diff))
    lm_markers_equal_hf = bool(
        torch.equal(next(iter(load_keys(args.base, ["model.lm_head.w_out.weight"])))[
            [ids.doc_start, ids.doc_end]].float(), emb_hf[[ids.doc_start, ids.doc_end]]))
    report["provenance"] = {
        "tensors_compared": len(olmo_keys),
        "missing": missing,
        "unexpected": extra,
        "mismatches": mismatches,
        "lm_head_marker_rows_untouched": lm_markers_equal_hf,
        "embedding_rows": int(emb_olmo.shape[0]),
    }
    print("PROVENANCE", json.dumps(report["provenance"]), flush=True)

    # ---- the eval tokenizer: Qwen3-4B-Base with the empty thinking block closed ----
    if args.nothink_out:
        tok = AutoTokenizer.from_pretrained(args.hf)
        tok.chat_template = make_nothink_template(tok.chat_template)
        os.makedirs(args.nothink_out, exist_ok=True)
        tok.save_pretrained(args.nothink_out)
        back = AutoTokenizer.from_pretrained(args.nothink_out)
        user = [{"role": "user", "content": "Q?"}]
        p = back.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        full = back.apply_chat_template(user + [{"role": "assistant", "content": "[1]"}],
                                        tokenize=False)
        assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), p
        assert full.startswith(p) and full[len(p):] == "[1]<|im_end|>\n", full
        assert back("x")["input_ids"] == tok("x")["input_ids"]
        report["nothink_tokenizer"] = args.nothink_out
        print(f"wrote {args.nothink_out}: generation prompt ends {p[-30:]!r}", flush=True)

    ok = (not missing and not mismatches and report["after_olmo_base/doc"]["audit_pass"])
    report["ok"] = ok
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        json.dump(report, open(args.out_json, "w"), indent=2)
    print("AUDIT", "OK" if ok else "FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
