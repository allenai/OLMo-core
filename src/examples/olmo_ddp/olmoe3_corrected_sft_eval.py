"""Correct-tokenizer exports and the frozen four-suite, three-temperature eval grid."""

import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import olmoe3_corrected_sft_plan as p
from olmoe3_lr_sweep_watch import atomic_json


def sha(path):
    """Stream large weight hashes rather than allocating another model-sized buffer."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def convert(r):
    """Map weights with the existing converter, then validate the real inference tokenizer."""
    from olmoe3_corrected_sft_data import tokenizer_check
    from olmoe3_hero_4t_eval_policy import (
        portable_attention_backends,
        structural_check,
        validate_export,
    )
    from olmoe3_hero_sft_metadata import install_metadata

    root = r.hf.parent
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "conversion.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if r.hf.exists():
        validate_export(r.hf, hash_weights=True)
        tokenizer_check(r.hf)
        return
    native = r.root / f"step{r.end}"
    p.base.validate_checkpoint(native, r.end, r.batch, r.gpus)
    # Owned immutable hardlinks avoid copying hundreds of GB or modifying the source.
    raw = root / "olmo-core"
    raw.mkdir(exist_ok=True)
    for f in sorted(native.rglob("*")):
        assert not f.is_symlink()
        dest = raw / f.relative_to(native)
        if f.is_dir():
            dest.mkdir(exist_ok=True)
        elif f.is_file():
            if not dest.exists():
                os.link(f, dest)
            assert os.path.samefile(f, dest)
    atomic_json(
        root / "source-ready.json",
        dict(
            passed=True,
            source=str(native),
            step=r.end,
            source_metadata_sha256=sha(native / ".metadata.json"),
            policy="owned immutable hardlinks; no source writes or deletions",
        ),
    )
    sys.path.insert(0, "/tmp/hero-conversion-source/src/examples/olmo_ddp")
    import hero_hf_convert as converter
    import torch
    from hero_hf_reference_ops import install, self_test
    from olmo_core.config import DType
    from olmo_core.nn.hf.convert_checkpoint import convert_checkpoint_to_hf, load_config

    torch.set_num_threads(8)
    self_test()
    install()
    config = load_config(raw)
    model_config = portable_attention_backends(converter.reference_config(config["model"]))
    output = root / "hf.partial"
    if not (root / "weights-written.json").exists():
        # The converter owns an unpublished output; retrying must not treat a partial export as final.
        assert not output.exists(), "Incomplete export requires scoped operator inspection"
        os.environ["OLMO_USE_TORCH_GROUPED_MM"] = "0"
        os.environ["OLMO_HF_MOE_CORE_REFERENCE"] = "1"
        convert_checkpoint_to_hf(
            raw,
            output,
            model_config,
            config["dataset"]["tokenizer"],
            tokenizer_id=str(r.data / "train/tokenizer"),
            dtype=DType.bfloat16,
            max_sequence_length=65536,
            device=torch.device("cpu"),
            validation_device=torch.device("cuda"),
            validate=False,
        )
        atomic_json(root / "weights-written.json", dict(passed=True, source=str(native)))
    install_metadata(output, r.data / "train/tokenizer")
    cfg = json.loads((output / "config.json").read_text())
    assert cfg["use_rope"] is False and cfg.get("rope_theta") is None
    cfg["rope_parameters"] = {"rope_theta": None}
    atomic_json(output / "config.json", cfg)
    tokenizer_check(output)
    proof = structural_check(root, full=True, precise=False)
    proof.update(
        step=r.end,
        tokens=r.end * r.batch,
        run=r.as_dict(),
        source_ready_sha256=sha(root / "source-ready.json"),
        output_sha256={f.name: sha(f) for f in output.iterdir() if f.is_file()},
    )
    atomic_json(output / "_HERO_CONVERSION_SUCCESS.json", proof)
    output.rename(r.hf)
    atomic_json(root / "conversion-success.json", proof)
    validate_export(r.hf, hash_weights=True)
    print("CORRECTED_SFT_CONVERSION_SUCCESS", r.run_id, flush=True)


def evaluate(r, bundle, temperature):
    """Change temperature only, retaining prompt/scorer/max-length/seed parity."""
    assert bundle in p.BUNDLES and temperature in p.TEMPERATURES
    os.environ.update(
        HERO_SFT_TEMPERATURE=str(temperature),
        HERO_SFT_RESUME="1",
        HERO_SFT_OUTPUT_GROUP=f"posttrain-t{round(temperature*10):02d}",
        HERO_SFT_EVAL_GROUP=p.CAMPAIGN,
        QKGAIN_SFT_STEP=str(r.end),
    )
    # Before task definitions import, including multiprocessing child execution.
    from olmoe3_corrected_sft_data import tokenizer_check
    from transformers import AutoTokenizer
    from vllm.tokenizers import get_tokenizer
    from olmo_core.nn.hf.config import _register_olmo3moe_auto_classes

    _register_olmo3moe_auto_classes()
    tok = tokenizer_check(r.hf)
    vt = get_tokenizer(str(r.hf), trust_remote_code=True, local_files_only=True)
    for text in ("9078563412", "def f():\n    return 123456789\n", "Café 日本語 🙂"):
        assert vt.encode(text, add_special_tokens=False) == tok.encode(
            text, add_special_tokens=False
        )
    assert AutoTokenizer.from_pretrained(r.hf).chat_template == vt.chat_template
    import olmoe3_hero_sft_plan as plan
    import olmoe3_hero_sft_convert as conversion
    import olmoe3_hero_sft_eval as evaluation

    plan.DATA = r.data
    plan.find_run = lambda _: SimpleNamespace(run_id=r.run_id, arm="emo", smoke=False)
    conversion.export_root = lambda _: p.EVAL / r.run_id
    sys.argv = [evaluation.__file__, "--run", r.run_id, "--bundle", bundle]
    evaluation.main()
    output = r.hf.parent / os.environ["HERO_SFT_OUTPUT_GROUP"] / bundle
    task = evaluation.TASKS[bundle][0]
    rows = [json.loads(f.read_text()) for f in (output / "responses" / task).glob("*.json")]
    lengths = [row["output_metadata"]["num_tokens"] for row in rows]
    assert len(rows) == evaluation.TASKS[bundle][1]
    atomic_json(
        output / "overruns.json",
        dict(
            instances=len(rows),
            temperature=temperature,
            mean_output_tokens=sum(lengths) / len(lengths),
            at_token_limit=sum(n >= 32768 for n in lengths),
            unclosed_reasoning=sum(not row["reasoning_closed"] for row in rows),
            empty_final=sum(not row["final_response"].strip() for row in rows),
        ),
    )
