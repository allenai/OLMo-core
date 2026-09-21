"""Audit the saved dense recipe and prepare the shared-vector Dolci25 mixture on CPU."""

import copy
import json
import logging
import sys
from pathlib import Path

import torch
from _qwen3_sharedvec_33344_common import BASE_CHECKPOINT, build_experiment_config

from olmo_core.data import LongDocStrategy
from olmo_core.data.composable import PackingInstanceSourceConfig
from olmo_core.distributed.checkpoint import load_keys
from olmo_core.internal.experiment import CliContext, SubCmd
from olmo_core.utils import prepare_cli_environment


def main():
    """Check provenance, build both packers, and save the measured comparison."""
    prepare_cli_environment()
    logging.getLogger().setLevel(logging.INFO)
    run_name = sys.argv[1]
    ctx = CliContext(__file__, SubCmd.prep, run_name, "ai2/jupiter-cirrascale-2", [])
    config = build_experiment_config(ctx, arm="sharedvec-dolci25-tokenmatch")
    save_dir = Path(config.trainer.save_folder)
    assert not save_dir.exists(), f"Fresh training destination already exists: {save_dir}"
    dense_path = Path(
        "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/"
        "q4b-dense-5task-dolci25-32k-nocpt/step10700/config.json"
    )
    dense = json.loads(dense_path.read_text())
    report = {"run_name": run_name, "dense_reference": str(dense_path), "cpt": {}}
    for name, path in (("sharedvec", BASE_CHECKPOINT), ("dense", dense["trainer"]["load_path"])):
        path = Path(path)
        assert path.is_dir(), path
        cpt = json.loads((path.parent / "config.json").read_text())
        report["cpt"][name] = {
            "checkpoint": str(path),
            "global_batch_size": cpt["data_loader"]["global_batch_size"],
            "step": int(path.parent.name.removeprefix("step")),
            "dataset": cpt["dataset"],
            "model": cpt["model"],
        }
    shared_cpt = report["cpt"]["sharedvec"]
    attn = shared_cpt["model"]["block"]["sequence_mixer"]
    assert attn["name"] == "shared_vector_landmark"
    assert attn["mem_freq"] == 63 and attn["vec_dim"] == 32
    for name, cpt in report["cpt"].items():
        cpt["token_slots"] = cpt["step"] * cpt["global_batch_size"]
        print(
            "CPT_BUDGET",
            name,
            cpt["step"],
            cpt["global_batch_size"],
            cpt["token_slots"],
            flush=True,
        )

    # The landmark row was trained during CPT; audit it without modifying the checkpoint.
    emb = next(iter(load_keys(BASE_CHECKPOINT, ["model.embeddings.weight"]))).float()
    marker = emb[151860]
    median_norm = emb[:151643].norm(dim=-1).median().item()
    norm_ratio = marker.norm().item() / median_norm
    report["landmark_embedding"] = {
        "norm_ratio_to_trained_median": norm_ratio,
        "identical_to_unused_neighbor": bool(torch.equal(marker, emb[151861])),
    }
    assert (
        torch.isfinite(marker).all()
        and not report["landmark_embedding"]["identical_to_unused_neighbor"]
    )
    assert 0.5 < norm_ratio < 2.0, report["landmark_embedding"]
    del emb, marker
    print("LANDMARK_EMBEDDING", report["landmark_embedding"], flush=True)

    dataset_config = config.dataset[0]
    work_dir = config.data_loader.work_dir
    packed = dataset_config.build(work_dir)
    mixture = packed.source
    dense_mix_config = copy.deepcopy(dataset_config.source)
    dense_mix_config.source_specs[1].max_repetition_factor = 1.0
    dense_mixture = dense_mix_config.build(work_dir)[0]
    assert (
        mixture.fingerprint == dense_mixture.fingerprint
    ), "Dolci repetition ceiling changed sampling"
    report["mixture"] = []
    for source in mixture.sampled_sources:
        lengths = [e - s for s, e in source.get_document_offsets()]
        record = {
            "label": source.label,
            "tokens": source.num_tokens,
            "fraction_before_packing": source.num_tokens / mixture.num_tokens,
            "source_tokens": source.source.num_tokens,
            "documents": len(lengths),
            "sharedvec_retained_tokens": sum(n for n in lengths if n <= 32823),
            "dense_retained_tokens": sum(n for n in lengths if n <= 32768),
            "sharedvec_dropped_documents": sum(n > 32823 for n in lengths),
            "dense_dropped_documents": sum(n > 32768 for n in lengths),
        }
        report["mixture"].append(record)
    assert abs(report["mixture"][0]["fraction_before_packing"] - 0.75) < 0.001
    assert abs(report["mixture"][1]["fraction_before_packing"] - 0.25) < 0.001
    dense_packed = PackingInstanceSourceConfig(
        sources=[dense_mix_config],
        sequence_length=32768,
        tokenizer=dense_mix_config.source_specs[1].source.tokenizer,
        long_doc_strategy=LongDocStrategy.exclude,
    ).build(work_dir)
    report["sharedvec_instances"] = packed.num_instances
    report["dense_instances"] = dense_packed.num_instances
    report["instance_ratio"] = packed.num_instances / dense_packed.num_instances
    report["config"] = config.as_config_dict()
    output = Path("/results")
    output.mkdir(exist_ok=True)
    (output / "prep-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PREP_REPORT",
        json.dumps({k: v for k, v in report.items() if k not in ("config", "cpt")}),
        flush=True,
    )
    print("PREP_OK", flush=True)


if __name__ == "__main__":
    main()
