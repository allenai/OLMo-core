"""Response loss of native multimodal checkpoints on training samples vs held-out data, per source.

Diagnoses a late-training drop in vision evals: training-sample loss falling while held-out loss
rises points at memorization; both rising points at interference. Run with eight torchrun ranks on
one node; every checkpoint is a native step directory (``config.json`` + ``model_and_optim``).

Training samples are drawn from the stage-1 v3 sources as the merged MT+LC stage builds them (one
annotation per example, document layout, dolma2 at the pinned revision); held-out samples from the
alignment validation splits and the image-disjoint caption/transcript validation selection. The
same indices (seed 0) are scored for every checkpoint. Writes one JSON per checkpoint:
``{source: {"split": ..., "examples": n, "mean_ce": ..., "token_weighted_ce": ...}}``.

    torchrun --nproc-per-node=8 src/scripts/heldout_vision_loss.py \\
        --output-dir <dir> --checkpoint <step dir> [--checkpoint <step dir> ...]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from olmo_core.data import TokenizerConfig
from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig
from olmo_core.data.multimodal.collator import MultimodalCollatorConfig
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.eval.multimodal_checkpoint import (
    build_model_and_module_config,
    checkpoint_state_dir,
    response_ce_by_example,
)
from olmo_core.internal.vision_alignment import _set_annotation_sampling, _visual_sources
from olmo_core.internal.vision_alignment_data import (
    DEFAULT_ALIGNMENT_ARTIFACT_ROOT,
    build_stage1_v3_sources,
    build_stage1_v3_validation_sources,
)
from olmo_core.train import prepare_training_environment, teardown_training_environment

log = logging.getLogger(__name__)

SEQUENCE_LENGTH = 8192
TRAIN_SOURCES = (
    "pixmo_caption",
    "pixmo_transcript",
    "pixmo_points_v2",
    "pixmo_count_v2",
    "cosyn_point_v2",
    "olmocr_documents",
    "text_rich_doc",
    "plot_qa",
)
HELDOUT_SOURCES = (
    "v3_long_caption",
    "v3_transcript",
    "pixmo_points_basic",
    "count_numeric",
    "cosyn_point",
    "ocr_document",
)


def _datasets(artifact_root: str):
    train = build_stage1_v3_sources("joint", SEQUENCE_LENGTH, artifact_root)
    train = {name: train[name] for name in TRAIN_SOURCES}
    heldout = dict(_visual_sources("joint", SEQUENCE_LENGTH, artifact_root, split="validation"))
    heldout.update(build_stage1_v3_validation_sources(SEQUENCE_LENGTH, artifact_root))
    heldout = {name: heldout[name] for name in HELDOUT_SOURCES}
    sources = {
        **{f"train/{k}": v for k, v in train.items()},
        **{f"heldout/{k}": v for k, v in heldout.items()},
    }
    _set_annotation_sampling(sources, "one")
    mixture = MultimodalMixtureConfig(
        tokenizer=TokenizerConfig.dolma2(),
        sources=sources,
        model_vocab_size=100352,
        target_loss_mass={name: 1.0 for name in sources},
        tokenizer_revision="5292e5d6c0f40b67cc765fe41bec991cf4345b5c",
    )
    tokenizer, token_ids = mixture.build_tokenizer()
    return mixture.build_sources(tokenizer, token_ids), tokenizer


def _score(train_module, datasets, examples: int, pad_token_id: int) -> dict:
    collate = MultimodalCollatorConfig(pad_token_id=pad_token_id).build()
    model = train_module.model
    device = next(model.parameters()).device
    rank, world = get_rank(), get_world_size()
    out = {}
    for name, dataset in datasets.items():
        indices = np.random.default_rng(0).choice(
            len(dataset), min(examples, len(dataset)), replace=False
        )
        records = []
        for position, index in enumerate(indices):
            if position % world != rank:
                continue
            example = dataset.get(int(index), 0) if hasattr(dataset, "get") else dataset[int(index)]
            batch = collate([example])
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.no_grad():
                logits = model(
                    inputs["input_ids"],
                    **{
                        k: v
                        for k, v in inputs.items()
                        if k not in ("input_ids", "labels", "loss_masks")
                    },
                )
            (record,) = response_ce_by_example(
                {"labels": inputs["labels"], "loss_masks": inputs["loss_masks"]}, logits
            )
            records.append((record["windows"]["all"], record["response_tokens"]))
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        flat = [r for part in gathered for r in part]
        ces = np.array([c for c, _ in flat])
        tokens = np.array([t for _, t in flat])
        out[name] = {
            "split": name.split("/")[0],
            "examples": len(flat),
            "mean_ce": float(ces.mean()),
            "se": float(ces.std(ddof=1) / np.sqrt(len(ces))) if len(ces) > 1 else None,
            "token_weighted_ce": float((ces * tokens).sum() / tokens.sum()),
        }
        if rank == 0:
            log.info("%s: %s", name, out[name])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--artifact-root", default=DEFAULT_ALIGNMENT_ARTIFACT_ROOT)
    args = parser.parse_args()
    prepare_training_environment()
    try:
        datasets, tokenizer = _datasets(args.artifact_root)
        for checkpoint in args.checkpoint:
            out_path = args.output_dir / f"{checkpoint.parent.name}__{checkpoint.name}.json"
            if out_path.exists():
                continue
            config = json.loads((checkpoint / "config.json").read_text())
            model, module_config, _ = build_model_and_module_config(
                config,
                ep_degree=None,
                max_sequence_length=SEQUENCE_LENGTH,
                rank_batch_size=SEQUENCE_LENGTH,
            )
            train_module = module_config.build(model, eval_only=True)
            train_module.load_state_dict_direct(
                checkpoint_state_dir(checkpoint),
                process_group=dist.group.WORLD,
                load_optim_state=False,
            )
            train_module.model.eval()
            result = _score(train_module, datasets, args.examples, tokenizer.pad_token_id)
            if get_rank() == 0:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps({"checkpoint": str(checkpoint), "sources": result}, indent=1)
                )
                log.info("wrote %s", out_path)
            del train_module, model
            torch.cuda.empty_cache()
            dist.barrier()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
