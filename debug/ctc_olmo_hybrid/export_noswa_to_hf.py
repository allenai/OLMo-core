"""Export a noswa (full-attention) OLMo3-family checkpoint to HF for vLLM serving.

Thin wrapper over ``src/examples/huggingface/convert_checkpoint_to_hf.py``'s machinery with one
patch: the converter maps a sliding-window-free olmo3 model onto the **Olmo2** HF class (correct —
that IS the full-attention arch), but ``Olmo2Config._rope_scaling_validation`` predates modern
rope dicts and rejects our YaRN config ``{rope_type: yarn, factor: 8, ...}``. vLLM's rope parser
accepts exactly that dict, so the validation is no-op'd here rather than dumbing down the config.

    PYTHONPATH=src python debug/ctc_olmo_hybrid/export_noswa_to_hf.py <ckpt-dir> <out-dir>

⚠ ``--tokenizer`` IS NOT COSMETIC. The tokenizer written into the export is the one vLLM
detokenizes with, so it must be the PATCHED dolma2 copy whose ``<|extra_id_1|>``/``<|extra_id_2|>``
slots (100266/100267) are renamed ``<|box_start|>``/``<|box_end|>`` -- the same copy the shards
were tokenized with. It defaults to the Berkeley path; on Beaker pass the weka copy
(``/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_olmo3/tokenizer``). Neither path
exists on the other cluster, which is why this is an argument and not a constant.
"""

import argparse

from transformers.models.olmo2.configuration_olmo2 import Olmo2Config

# Neutralize the legacy validator (see module docstring).
Olmo2Config._rope_scaling_validation = lambda self: None  # type: ignore[method-assign]

from olmo_core.config import DType  # noqa: E402
from olmo_core.nn.hf import convert_checkpoint_to_hf, load_config  # noqa: E402

#: Berkeley-local patched dolma2 marker tokenizer (mirrors
#: ``scripts/train/memexpress/ctc_suite/olmo3_configs._OLMO3_TOKENIZER_CANDIDATES[0]``).
DEFAULT_TOKENIZER = "/scratch/users/prasann/hf_models/Olmo-3-1025-7B-docchunk"


def main(ckpt: str, out: str, tokenizer_id: str, max_seq_len: int, dtype: str) -> None:
    """Convert one olmo-core distcp checkpoint to an HF (Olmo2) directory.

    :param ckpt: Checkpoint dir holding ``config.json`` + ``model_and_optim/``.
    :param out: Output HF directory.
    :param tokenizer_id: Path/id of the PATCHED dolma2 marker tokenizer to embed in the export.
    :param max_seq_len: Written to ``max_position_embeddings``; should match the run's ``--seq-len``.
    :param dtype: Weight dtype to write. See ``--dtype``.
    """
    experiment_config = load_config(ckpt)
    assert experiment_config is not None, "no experiment config in checkpoint"
    convert_checkpoint_to_hf(
        original_checkpoint_path=ckpt,
        output_path=out,
        transformer_config_dict=experiment_config["model"],
        tokenizer_config_dict=experiment_config.get("dataset", {}).get("tokenizer") or {},
        tokenizer_id=tokenizer_id,
        max_sequence_length=max_seq_len,
        dtype=DType(dtype),
        validate=False,
    )
    print(f"EXPORT-OK -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--max-seq-len", type=int, default=40960)
    # ⚠ bfloat16 IS THE PARITY CHOICE, not a size optimization (though it is that too: 29GB -> 15GB).
    # The distcp holds fp32 master weights, so an unspecified dtype writes an fp32 HF config --
    # and vLLM's dtype="auto" then DOWNCASTS float32 to float16, a different rounding mode than the
    # native evaluator, which builds the model at DType("bfloat16")
    # (corpus_reasoning/eval/eval_lc_native_docchunk.py). Any vLLM-vs-native comparison has to hold
    # dtype fixed or the delta is partly numerics. Pass --dtype float32 to reproduce the older
    # local exports under /data/prasann/hf_exports/.
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()
    main(args.ckpt, args.out, args.tokenizer, args.max_seq_len, args.dtype)
