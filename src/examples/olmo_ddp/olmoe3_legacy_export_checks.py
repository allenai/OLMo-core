"""CPU regression checks for legacy/shared/split attention export backends."""

import copy

from olmoe3_hero_4t_eval_policy import portable_attention_backends


def main():
    """Retain architecture and gain policy while avoiding Flash4 construction."""
    for gain in (None, False, True):
        attention = {
            "_CLASS_": "olmo_core.nn.attention.AttentionConfig",
            "type": "attention",
            "backend": "flash_4",
            "n_heads": 16,
            "n_kv_heads": 8,
            "head_dim": 128,
            "use_head_qk_norm": True,
            "scalable_softmax": True,
        }
        if gain is not None:
            attention["qk_norm_per_head_gains"] = gain
        kda = {"type": "kimi_delta_attention", "expand_v": 2.0, "allow_neg_eigval": True}
        original = {"block": {"sequence_mixer": kda}, "block_overrides": {
            str(i): {"sequence_mixer": copy.deepcopy(attention)} for i in (4, 9, 14)
        }}
        actual = portable_attention_backends(copy.deepcopy(original))
        assert actual["block"] == original["block"]
        for block in actual["block_overrides"].values():
            mixer = block["sequence_mixer"]
            assert mixer.pop("backend") == "torch"
            assert mixer.pop("use_flash") is False
            assert mixer == {k: v for k, v in attention.items() if k != "backend"}
    print("LEGACY_SHARED_SPLIT_EXPORT_BACKENDS_PASSED", flush=True)


if __name__ == "__main__":
    main()
