"""CPU check that an online-router arm's exact trainer flags wire through: argparse (train_ctc_suite.parse_args,
including its cross-flag checks) -> SoftTokenOptions -> SoftToken.post_build_model on the real 4B config (meta
device) -> the router (features, gold sidecar) -> extra_callbacks. ``--dry-run`` alone stops before the variant
hooks, which is how a flag check slipped through once (2026-10-22: every online launch died in argparse).

    python research/token_routing/online_router/check_variant_wiring.py --data <shard dir> -- <arm flags...>
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

from recipes.train import variants as V  # noqa: E402
from recipes.train.research_trainer import train_ctc_suite as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a, rest = ap.parse_known_args()
    arm = rest[1:] if rest[:1] == ["--"] else rest
    sys.argv = ["train_ctc_suite.py", "--task", "mt4", "--data", a.data, "--variant", "softtoken",
                "--model-family", "qwen3_5", "--model-scale", "4b", "--seq-len", "10240", "--global-batch", "16",
                "--micro-batch-instances", "2", "--no-compile", "--lr", "2e-06"] + arm
    opts = T.parse_args()
    variant, vopts = T.variant_and_options(opts)
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer import TransformerConfig

    cfg = TransformerConfig.qwen3_5_4B(vocab_size=T.FAMILY_VOCAB_SIZE["qwen3_5"],
                                       attn_backend=AttentionBackendName("flash_2"))
    model = cfg.build(init_device="meta")
    ctx = V.VariantContext(model_family="qwen3_5", seed=0, rank=0, total_calls=1000, calls_per_step=4,
                           seq_len=10240, vocab_size=T.FAMILY_VOCAB_SIZE["qwen3_5"], data_dir=a.data,
                           meta=T.read_shard_metadata(a.data, 10240, "qwen3_5"),
                           tokenizer_id=T.FAMILY_TOKENIZER["qwen3_5"])
    variant.post_build_model(model, vopts, ctx)
    r = model._online_token_router
    cbs = variant.extra_callbacks(vopts, ctx)
    assert r is not None and "online_router" in cbs, (r, cbs)
    pst = model._pooled_soft_tokens
    assert pst["keep_token_rule"] == "online" and pst["drop_slots"], pst["keep_token_rule"]
    print(f"OK arm={' '.join(arm)}\n   router cfg={r.cfg}\n   n_features={r.n_features} "
          f"gold_table={'yes' if r.gold is not None else 'no'} ({len(r.gold or {})} fps) callbacks={list(cbs)}")


if __name__ == "__main__":
    main()
