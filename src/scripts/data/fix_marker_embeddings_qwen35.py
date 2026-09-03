"""
Qwen3.5 sibling of ``fix_marker_embeddings.py``: repair the untrained reserved-token embedding rows
of a converted Qwen3.5 base (marker set ``qwen3_5``) before any document-chunk / landmark training.

Same diagnosis and cure as the Qwen3 original (see its docstring and
``records/document-chunked-marker-embeddings.md`` / ``records/n100-chunked-marker-position-bug.md``):
the reserved rows -- ``<|box_start|>``/``<|box_end|>`` and the landmark/pad ids past the real vocab
-- are untrained in the base conversion, mutually near-identical and far below a trained row's norm,
which RMSNorm amplifies into full-strength noise on marker-dense data. Each marker is seeded from a
real trained delimiter row plus jitter, then asserted distinguishable AND in-distribution in norm.

Ids come from ``RESERVED_IDS["qwen3_5"]`` -- never retyped. Run on a machine holding the distcp
base (e.g. cubbins /data ctc_suite/bases), then stage the repaired copy to weka via the usual
S3 two-step::

    python src/scripts/data/fix_marker_embeddings_qwen35.py \
      --base /data/prasann/ctc_suite/bases/q35-4b-base-modelonly/model_and_optim \
      --out  /data/prasann/ctc_suite/bases/q35-4b-base-markerfix \
      --model-scale 4b
"""

import argparse
import os

import torch

from olmo_core.data.document_chunk_landmark import (
    RESERVED_IDS,  # canonical ids -- never retype
)
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.nn.transformer import TransformerConfig

FAMILY = "qwen3_5"
VOCAB_SIZE = 248320  # Qwen3.5 embedding-matrix rows (matches FAMILY_VOCAB_SIZE in train_ctc_suite)
TOKENIZER = "Qwen/Qwen3.5-0.8B-Base"

FACTORIES = {
    "0.8b": TransformerConfig.qwen3_5_0_8B,
    "2b": TransformerConfig.qwen3_5_2B,
    "4b": TransformerConfig.qwen3_5_4B,
    "9b": TransformerConfig.qwen3_5_9B,
    "27b": TransformerConfig.qwen3_5_27B,
}

#: Trained delimiter donors, same choices as the Qwen3 script (delimiters already read as
#: "a boundary is here", which is the markers' job).
DONOR_STRINGS = {"doc_start": "«", "doc_end": "»", "landmark": "§", "pad": "¶"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="source model_and_optim distcp dir")
    ap.add_argument(
        "--out", required=True, help="destination dir (model_and_optim/ written under it)"
    )
    ap.add_argument("--model-scale", default="4b", choices=sorted(FACTORIES))
    ap.add_argument("--seed", type=int, default=34521)
    ap.add_argument("--jitter", type=float, default=0.1)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    ids = RESERVED_IDS[FAMILY]
    tok = AutoTokenizer.from_pretrained(os.environ.get("FIX_TOKENIZER", TOKENIZER))  # FIX_TOKENIZER: local/weka copy when the Hub is offline or rate-limited

    model = FACTORIES[args.model_scale](vocab_size=VOCAB_SIZE).build(init_device="cpu")
    load_model_and_optim_state(args.base, model)

    emb = model.embeddings.weight.data
    trained = emb[: ids.real_vocab_size].float()
    median_norm = trained.norm(dim=-1).median()
    std = trained.std()

    g = torch.Generator().manual_seed(args.seed)
    targets = {
        ids.doc_start: "doc_start",
        ids.doc_end: "doc_end",
        ids.landmark: "landmark",
        ids.pad: "pad",
    }
    print(
        f"trained-row median norm={median_norm:.4f}  (markers must land near this, NOT far below)"
    )
    for tid, name in targets.items():
        donor_str = DONOR_STRINGS[name]
        donor_ids = tok.encode(donor_str, add_special_tokens=False)
        if len(donor_ids) != 1 or donor_ids[0] >= ids.real_vocab_size or donor_ids[0] == ids.eos:
            raise SystemExit(
                f"donor {donor_str!r} for {name} is not a single trained token: {donor_ids}"
            )
        donor = donor_ids[0]
        before = emb[tid].float().norm()
        vec = emb[donor].float() + torch.randn(emb.shape[1], generator=g) * (std * args.jitter)
        emb[tid] = vec.to(emb.dtype)
        print(
            f"  {tid} {name:10s} <- {donor_str!r} (id {donor})   norm {before:.4f} -> "
            f"{emb[tid].float().norm():.4f}"
        )

    id_list = list(targets)
    for i in range(len(id_list)):
        for j in range(i + 1, len(id_list)):
            cos = torch.nn.functional.cosine_similarity(
                emb[id_list[i]].float()[None], emb[id_list[j]].float()[None]
            ).item()
            assert abs(cos) < 0.9, f"markers {id_list[i]}/{id_list[j]} too similar (cos={cos:.4f})"
    for tid, name in targets.items():
        ratio = (emb[tid].float().norm() / median_norm).item()
        assert (
            0.5 < ratio < 2.0
        ), f"marker {name} norm is {ratio:.2f}x the trained-row median -- out of distribution."
    print("markers are distinguishable AND in-distribution in norm")

    save_model_and_optim_state(f"{args.out}/model_and_optim", model)
    print(f"wrote fixed base -> {args.out}/model_and_optim")


if __name__ == "__main__":
    main()
