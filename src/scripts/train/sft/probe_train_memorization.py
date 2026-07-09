"""
Memorization probe for the docchunk attn-explore checkpoints.

Loads N training instances straight from the docdense npy shard (the EXACT bytes the model
trained on), splits each at the labels_mask boundary into (prompt, gold answer), prefills the
prompt through the docchunk generation path (chunked mask applied at prefill via
enable_document_chunk_attention + config.json cross_doc_mode), greedily decodes, and compares
with the memorized gold answer.

If the model reproduces its training answers -> the eval machinery (loading/mask/decode) is
sound, and a low held-out f1 is a genuine generalization result, not an eval bug.
"""

import argparse
import json
import time

import numpy as np
import torch

EOS_TOKEN_ID = 151643
DOC_START_ID = 151648
DOC_END_ID = 151649


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-dir", required=True, help="docdense shard dir (token_ids_part_*.npy)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--dense", action="store_true", help="full-attention model: skip chunk mask")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from olmo_core.config import DType
    from olmo_core.generate.generation_module.config import GenerationConfig
    from olmo_core.generate.generation_module.transformer import (
        TransformerGenerationModuleConfig,
    )

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    device = torch.device("cuda:0")

    # ---- load training instances from the raw shard ----
    ids = np.fromfile(f"{args.data_dir}/token_ids_part_000000.npy", dtype=np.uint32)
    mask = np.fromfile(f"{args.data_dir}/labels_mask_part_000000.npy", dtype=np.bool_)
    assert len(ids) == len(mask)
    # Instances are EOS-terminated and concatenated.
    eos_pos = np.where(ids == EOS_TOKEN_ID)[0]
    starts = np.concatenate([[0], eos_pos[:-1] + 1])
    ends = eos_pos + 1  # inclusive of EOS
    print(f"[probe] shard has {len(starts)} instances")

    instances = []
    for s, e in zip(starts[: args.n], ends[: args.n]):
        inst_ids = ids[s:e].astype(np.int64)
        inst_mask = mask[s:e]
        ans_pos = np.where(inst_mask)[0]
        assert len(ans_pos) > 0, "no supervised tokens in instance"
        p0 = ans_pos[0]
        prompt = inst_ids[:p0].tolist()
        gold = inst_ids[p0:].tolist()
        # strip trailing EOS off gold for text comparison
        while gold and gold[-1] == EOS_TOKEN_ID:
            gold = gold[:-1]
        instances.append((prompt, gold))

    t0 = time.time()
    gen_cfg = GenerationConfig(
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=151645,
        max_length=args.max_length,
        use_cache=True,
    )
    gm = TransformerGenerationModuleConfig(
        gen_cfg, float8_config=None, dtype=DType("bfloat16"), compile_model=False
    ).build(checkpoint_dir=args.model_path, device=device)
    if not args.dense:
        gm.model.enable_document_chunk_attention(
            doc_start_id=DOC_START_ID,
            doc_end_id=DOC_END_ID,
            eos_id=EOS_TOKEN_ID,
            mode="chunked",
        )
    print(f"[probe] built {args.model_path} in {time.time() - t0:.1f}s")

    n_exact = 0
    records = []
    for i, (prompt, gold) in enumerate(instances):
        gm.prepare_inference_cache(1, args.max_length)
        leftpad = torch.zeros(1, dtype=torch.int32, device=device)
        with torch.no_grad():
            logits = gm.model(
                torch.tensor([prompt], device=device), logits_to_keep=1, cache_leftpad=leftpad
            )
            nxt = int(logits[0, -1].argmax().item())
            out = []
            for _ in range(args.max_new_tokens):
                if nxt == EOS_TOKEN_ID:
                    break
                out.append(nxt)
                with torch.no_grad():
                    logits = gm.model(torch.tensor([[nxt]], device=device), logits_to_keep=1)
                nxt = int(logits[0, -1].argmax().item())
        gold_txt = tok.decode(gold, skip_special_tokens=True).strip()
        gen_txt = tok.decode(out, skip_special_tokens=True).strip()
        exact = gen_txt == gold_txt
        n_exact += exact
        # token-level prefix agreement
        agree = 0
        for a, b in zip(out, gold):
            if a != b:
                break
            agree += 1
        records.append(
            {"i": i, "exact": exact, "prefix_agree": agree, "gold_len": len(gold),
             "gold": gold_txt[:300], "gen": gen_txt[:300]}
        )
        print(f"[{i}] exact={exact} prefix_agree={agree}/{len(gold)}")
        print(f"    gold: {gold_txt[:200]}")
        print(f"    gen : {gen_txt[:200]}")

    print(f"[probe] exact-match {n_exact}/{len(instances)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(records, f, indent=2)


if __name__ == "__main__":
    main()
