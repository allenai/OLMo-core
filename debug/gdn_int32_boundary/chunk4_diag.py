"""
Is the chunk_size=4 chunked-prefill divergence a bug or a bf16 greedy near-tie?

Re-runs test_generation_module_chunked_prefill_matches_one_shot's setup (12-token prompt, greedy,
24 tokens) for chunk sizes None/3/4/5 in bf16 and fp32, and prints where each continuation first
leaves the one-shot one, the one-shot top-2 logit margin at that step, and the max |logit delta|
over the prefill-determined first step. A bug shows up as a large delta at step 0 (the prefill
output) or in fp32; a near-tie as a late divergence at a tiny margin, bf16 only.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join("src", "test", "generate", "generation_module", "transformer"))
from generation_module_test import small_hybrid_gdn_transformer_config  # noqa: E402

from olmo_core.config import DType  # noqa: E402
from olmo_core.generate.generation_module import TransformerGenerationModule  # noqa: E402
from olmo_core.generate.generation_module.config import GenerationConfig  # noqa: E402
from olmo_core.utils import seed_all  # noqa: E402

dev = torch.device("cuda")
for dtype in (DType.bfloat16, DType.float32):
    for fuse in (False, True):
        seed_all(0)
        cfg = small_hybrid_gdn_transformer_config(use_flash=dtype == DType.bfloat16, dtype=dtype)
        cfg.apply(lambda c: setattr(c, "fuse_qkv", fuse) if hasattr(c, "fuse_qkv") else None)
        model = cfg.build()
        ids = torch.randint(2, 500, (1, 12), device=dev)
        mask = torch.ones(1, 12, device=dev, dtype=torch.bool)

        def run(cs):
            seed_all(0)
            gm = TransformerGenerationModule(
                model=model,
                generation_config=GenerationConfig(
                    max_length=24,
                    do_sample=False,
                    eos_token_id=1,
                    pad_token_id=0,
                    use_cache=True,
                    prefill_chunk_size=cs,
                ),
                device=dev,
            )
            out, logits, _ = gm.generate_batch(
                ids, attention_mask=mask, return_logits=True, completions_only=False
            )
            return out[0], logits[0].float()

        ref_ids, ref_lg = run(None)
        for cs in (3, 4, 5):
            got_ids, got_lg = run(cs)
            n = min(len(ref_ids), len(got_ids))
            div = next((i for i in range(n) if ref_ids[i] != got_ids[i]), None)
            first_gen = 12  # logits row for the first generated token
            d0 = (got_lg[first_gen - 1] - ref_lg[first_gen - 1]).abs().max().item()
            dmax = (got_lg[: n - 1] - ref_lg[: n - 1]).abs().max().item()
            if div is not None:
                top2 = ref_lg[div - 1].topk(2).values
                margin = (top2[0] - top2[1]).item()
                where = f"diverges at token {div} (gen step {div - 12}), one-shot top-2 margin {margin:.4f}"
            else:
                where = "identical ids"
            print(
                f"{str(dtype):<16} fuse={fuse!s:<5} chunk={cs}: {where}; "
                f"|dlogit| first-gen step {d0:.4f}, max over shared prefix {dmax:.4f}",
                flush=True,
            )
