"""Unified training dispatcher.

Reads a YAML config and launches the matching trainer:
  - `attn_smoothing` block OR `attention_pattern: hierarchical_anchor`
                                          ->  scripts/train/train_chunked_smoothed.py
  - `standard_attention: false` explicit  ->  scripts/train/train_chunked_fast.py
  - anything else                          ->  axolotl.cli.train

For the axolotl path, if `datasets[0].path` points at unified JSONL, it is
converted to alpaca format on the fly and cached under data/.cache/, and the
YAML is rewritten (under /tmp/) to point at the cached file before launch.

A config with `backend: olmo-core` instead routes to the olmo-core full-FT path
(scripts/train/olmo_train.py via torchrun): the unified JSONL is tokenized into
NumpyPaddedFSLDataset format and cached, the HF base is converted to an olmo-core
checkpoint (cached), and the axolotl-style batch/epoch/warmup knobs are mapped onto
olmo-core. Qwen3 *dense* base models only, full fine-tuning only.

Usage:
    python scripts/train/train.py configs/foo.yml
    NUM_GPUS=2 python scripts/train/train.py configs/foo.yml
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
Path(os.environ["TRITON_CACHE_DIR"]).mkdir(exist_ok=True)

from corpus_reasoning.lib.io import load_jsonl, save_jsonl
from corpus_reasoning.lib.data_format import build_prompt_parts

CACHE_DIR = Path("data/.cache")


def num_gpus() -> int:
    if "NUM_GPUS" in os.environ:
        return int(os.environ["NUM_GPUS"])
    try:
        return len(subprocess.check_output(["nvidia-smi", "-L"]).decode().splitlines())
    except Exception:
        return 1


# HPQA and NQ are treated as retrieval benchmarks throughout this codebase
# (eval expects [doc_id] outputs). Training with task=qa silently produces a
# model that won't match the eval harness, so reject it at the dispatcher.
_RETRIEVAL_ONLY_DATASET_PREFIXES = ("hotpotqa_", "hpqa_", "nq_")


def assert_retrieval_for_hpqa_nq(cfg: dict, cfg_path: Path) -> None:
    if cfg.get("task", "retrieval") != "qa":
        return
    datasets = cfg.get("datasets") or []
    if not datasets:
        return
    name = Path(datasets[0]["path"]).name
    if name.startswith(_RETRIEVAL_ONLY_DATASET_PREFIXES):
        raise AssertionError(
            f"{cfg_path.name}: task=qa is disabled for HPQA / NQ datasets "
            f"(dataset={name}). Use task=retrieval — these benchmarks are "
            f"evaluated as retrieval, so a QA-trained model won't match the "
            f"eval harness."
        )


def is_unified_jsonl(path: Path) -> bool:
    try:
        with path.open() as f:
            first = f.readline()
        return bool(first) and "documents" in json.loads(first)
    except (OSError, ValueError):
        return False


def convert_to_alpaca_cached(src: Path, task, qp, use_titles, bd, ad,
                             cot_mode="label") -> Path:
    """Convert a unified JSONL to alpaca format, caching under data/.cache/.

    If the first example carries a `_task` field, prompt building is
    dispatched per-example using `_task` and `_cot_mode` (multi-task combined
    files; see `scripts/data/build_combined_unified.py`). Otherwise, the
    YAML-level `task` / `cot_mode` are used for every example.

    Cache key includes source mtime+size plus all prompt-building options, so
    edits or option changes invalidate the cache automatically.
    """
    st = src.stat()
    # Detect multi-task tagging by peeking at the first record.
    multitask = False
    with src.open() as f:
        first_line = f.readline()
    if first_line:
        try:
            if "_task" in json.loads(first_line):
                multitask = True
        except ValueError:
            pass
    tag = "multitask" if multitask else task
    key = f"{src.resolve()}|{st.st_mtime_ns}|{st.st_size}|{tag}|{qp}|{use_titles}|{bd}|{ad}|{cot_mode}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{src.stem}_alpaca_{tag}_q{qp}_{h}.jsonl"
    if cached.exists():
        print(f"[cache] reusing {cached}")
        return cached
    print(f"[cache] converting {src} -> {cached.name}  (tag={tag}, qp={qp}, titles={use_titles}, cot_mode={cot_mode})")
    out = []
    for ex in load_jsonl(src):
        ex_task = ex.get("_task", task) if multitask else task
        ex_cot  = ex.get("_cot_mode", cot_mode) if multitask else cot_mode
        instr, inp, outp = build_prompt_parts(
            ex, task=ex_task, query_position=qp, use_titles=use_titles,
            before_dummy=bd, after_dummy=ad, cot_mode=ex_cot,
        )
        out.append({"instruction": instr, "input": inp, "output": outp})
    save_jsonl(str(cached), out)
    return cached


def prepare_axolotl_yaml(cfg_path: Path, cfg: dict) -> Path:
    """If the config points at unified JSONL, convert + cache and return a rewritten YAML.

    Otherwise returns `cfg_path` unchanged so pre-converted alpaca files pass through.
    """
    data_path = Path(cfg["datasets"][0]["path"])
    if not is_unified_jsonl(data_path):
        return cfg_path
    cached = convert_to_alpaca_cached(
        data_path,
        task=cfg.get("task", "retrieval"),
        qp=cfg.get("query_position", "after"),
        use_titles=not cfg.get("no_titles", False),
        bd=cfg.get("before_dummy", 0),
        ad=cfg.get("after_dummy", 0),
        cot_mode=cfg.get("cot_mode", "label"),
    )
    new_cfg = dict(cfg)
    new_cfg["datasets"] = [{"path": str(cached), "type": "alpaca", "ds_type": "json"}]
    out = Path("/tmp") / f"{cfg_path.stem}__{cached.stem[-16:]}.yml"
    out.write_text(yaml.safe_dump(new_cfg, sort_keys=False))
    print(f"[cache] rewrote config -> {out}")
    return out


def fix_tokenizer_after_fullft(cfg: dict) -> None:
    """Re-save the base-model tokenizer after axolotl full-FT so vLLM can load it.

    Axolotl's full-FT output strips added_tokens_decoder, which breaks vLLM's
    tokenizer loading. Re-save directly from the base model as the last step.
    """
    if cfg.get("adapter"):
        return
    base_model = cfg.get("base_model")
    output_dir = cfg.get("output_dir")
    if not (base_model and output_dir and Path(output_dir).is_dir()):
        return
    print(f"[post] saving tokenizer from {base_model} to {output_dir}")
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(base_model).save_pretrained(output_dir)


# Env-overridable so the same dispatcher works on slurm (/scratch/...) and on VESSL (/mnt/corpus/...).
# Set CORPUS_OLMO_CKPT_ROOT=/mnt/corpus/olmo_ckpts on VESSL; we derive the converted-cache subdir.
_OLMO_CKPT_ROOT = Path(os.environ.get("CORPUS_OLMO_CKPT_ROOT", "/scratch/users/prasann/olmo_ckpts"))
CONVERTED_CKPT_ROOT = _OLMO_CKPT_ROOT / "converted"


def _visible_gpu_count() -> int:
    """GPUs this process may actually use, honoring clean-GPU pinning.

    The job wrapper pins a clean subset via CUDA_VISIBLE_DEVICES / NEED (horton hands out
    a rogue-occupied GPU in every allocation), so prefer those over `nvidia-smi -L`, which
    counts all physical GPUs regardless of the pin.
    """
    if os.environ.get("NEED"):
        return int(os.environ["NEED"])
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return len([x for x in cvd.split(",") if x.strip() != ""])
    return num_gpus()


def ensure_converted_ckpt(base_model: str, vocab_size: int) -> Path:
    """Convert the HF base -> olmo-core distcp format if not already cached.

    Runs convert_hf_to_olmo.py as a SINGLE-PROCESS subprocess (it needs one GPU for
    validate=True and must not run inside torchrun). Cache key = sanitized id + vocab.
    """
    sanitized = base_model.replace("/", "_")
    out = CONVERTED_CKPT_ROOT / f"{sanitized}_{vocab_size}"
    if (out / "model_and_optim").is_dir():
        print(f"[olmo] reusing converted base ckpt {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[olmo] converting {base_model} -> {out} (single-process, validate)")
    rc = subprocess.call([
        sys.executable, "scripts/train/convert_hf_to_olmo.py",
        "--base-model", base_model, "--out", str(out),
    ])
    if rc != 0 or not (out / "model_and_optim").is_dir():
        sys.exit(f"[olmo] base-model conversion failed (rc={rc})")
    return out


def run_olmo_core(cfg: dict, cfg_path: Path) -> int:
    """Tokenize + convert + launch olmo-core full-FT for a `backend: olmo-core` config."""
    # Heavy deps (transformers / olmo-core) only imported on this path.
    from corpus_reasoning.data.tokenize_unified_for_olmo import tokenize_unified_for_olmo
    from corpus_reasoning.lib.olmo_models import resolve_olmo_model

    base_model = cfg["base_model"]
    data_path = cfg["datasets"][0]["path"]
    seq_len = int(cfg["sequence_len"])
    task = cfg.get("task", "retrieval")

    # Custom attention masking (FlexAttention chunked / modified_swa). When set, the data
    # must be tokenized with document-boundary markers (--wrap-docs) so the per-token
    # chunk structure can be recovered at train time. See scripts/lib/olmo_flex_attention.
    attn_pattern = cfg.get("attention_pattern", "standard")
    attn_window = int(cfg.get("attn_window", cfg.get("token_window_w", 512)))
    # Gold-document gradient masking (scripts/lib/olmo_gold_grad_mask): detach K/V of
    # distractor-doc tokens so only ground-truth docs drive the update. Needs doc markers
    # to recover doc structure, so it forces wrap_docs even under standard attention.
    gold_grad_mask = bool(cfg.get("gold_grad_mask", False))
    wrap_docs = (attn_pattern != "standard") or gold_grad_mask
    # Eager FlexAttention materializes the full (B,H,S,S) score matrix -> OOM at long ctx;
    # the compiled (fused) kernel does not. Default ON whenever a custom pattern is used.
    compile_flex = bool(cfg.get("compile_flex", wrap_docs))
    # Mask mixing: fraction of examples tokenized without doc markers (-> plain causal
    # under the mask). Baked at tokenize time; mirrors HF `standard_mix_prob`.
    standard_mix_prob = float(cfg.get("standard_mix_prob", 0.0))
    mix_seed = int(cfg.get("mix_seed", 42))
    if standard_mix_prob > 0.0 and not wrap_docs:
        sys.exit("[olmo] standard_mix_prob>0 requires a non-standard attention_pattern "
                 "(there are no doc masks to mix out under standard attention).")
    # Mix curriculum (runtime): p(standard) anneals mix_start_p -> mix_end_p over training.
    # Requires a custom pattern (something to collapse) and is mutually exclusive with the
    # static tokenize-time mix (that bakes markerless examples; the curriculum needs all
    # examples marker-wrapped so it can choose per-forward).
    mix_start_p = float(cfg.get("mix_start_p", 0.0))
    mix_end_p = float(cfg.get("mix_end_p", 0.0))
    if (mix_start_p > 0.0 or mix_end_p > 0.0):
        if not wrap_docs:
            sys.exit("[olmo] mix_start_p/mix_end_p require a non-standard attention_pattern.")
        if standard_mix_prob > 0.0:
            sys.exit("[olmo] mix curriculum (mix_start_p/mix_end_p) is mutually exclusive "
                     "with static standard_mix_prob>0; set standard_mix_prob: 0.")

    # 1) Validate model (Qwen3 dense only); resolve builder + vocab.
    spec = resolve_olmo_model(base_model)  # raises with guidance on qwen2/hybrid/MoE
    print(f"[olmo] {base_model} -> builder={spec.builder} vocab={spec.vocab_size}")

    # 2) Batch/CP knob mapping (axolotl-style YAML -> olmo-core).
    g = _visible_gpu_count()
    cp = int(cfg.get("cp_degree", 1))
    if g < 1:
        sys.exit("[olmo] no GPUs visible")
    if g % cp != 0:
        sys.exit(f"[olmo] cp_degree={cp} must divide num_gpus={g}")
    dp = g // cp
    mbs = int(cfg.get("micro_batch_size", 1))
    gas = int(cfg.get("gradient_accumulation_steps", 1))
    rank_microbatch_tokens = seq_len * mbs
    global_batch_tokens = seq_len * mbs * gas * dp
    # olmo-core enforces these at runtime; check up front to fail before grabbing GPUs.
    assert global_batch_tokens % (rank_microbatch_tokens * dp) == 0
    assert (mbs * gas * dp) >= 1

    # 3) Tokenize + cache the dataset (CPU, in-process).
    tokens, label_mask, meta = tokenize_unified_for_olmo(
        data_path, base_model, task=task,
        query_position=cfg.get("query_position", "after"),
        use_titles=not cfg.get("no_titles", False),
        before_dummy=cfg.get("before_dummy", 0),
        after_dummy=cfg.get("after_dummy", 0),
        cot_mode=cfg.get("cot_mode", "label"),
        seq_len=seq_len,
        # axolotl-style: false = loss only on completion (default); true = loss on every token.
        train_on_inputs=bool(cfg.get("train_on_inputs", False)),
        wrap_docs=wrap_docs,
        standard_mix_prob=standard_mix_prob, mix_seed=mix_seed,
        emit_gold_mask=gold_grad_mask,
    )

    save_folder = cfg.get("output_dir") or str(
        _OLMO_CKPT_ROOT / cfg.get("wandb_name", cfg_path.stem))
    # Args shared by the dry-run gate and the real torchrun launch (script + flags, no
    # interpreter — the dry-run prepends sys.executable, torchrun supplies its own).
    script_args = [
        "scripts/train/olmo_train.py",
        "--base-model", base_model, "--tokens", tokens, "--label-mask", label_mask,
        "--meta", meta, "--seq-len", str(seq_len),
        "--global-batch-tokens", str(global_batch_tokens),
        "--rank-microbatch-tokens", str(rank_microbatch_tokens),
        "--epochs", str(cfg.get("num_epochs", 3)),
        "--max-steps", str(cfg.get("max_steps", 0)),
        "--lr", str(cfg.get("learning_rate", 1e-5)),
        "--warmup-fraction", str(cfg.get("warmup_ratio", 0.1)),
        "--warmup-steps", str(cfg.get("warmup_steps", 0)),
        "--cp-degree", str(cp),
        "--ac-mode", str(cfg.get("ac_mode", "none")),
        "--ac-budget", str(cfg.get("ac_budget", 0.5)),
        "--ac-block-interval", str(cfg.get("ac_block_interval", 2)),
        # Default to bf16_sr -- full-bf16 master+moments with stochastic rounding. Empirically at
        # parity with fp32-master baseline (NQ recall 0.500 vs 0.498, contradiction f1 0.955 vs
        # 0.948) at substantially lower memory. Set precision_arm: baseline in the YAML to opt out.
        "--arm", str(cfg.get("precision_arm", "bf16_sr")),
        "--attn-pattern", str(attn_pattern),
        "--attn-window", str(attn_window),
        *(["--compile-flex"] if compile_flex else []),
        # Mix curriculum (runtime): anneal p(standard) from mix_start_p -> mix_end_p over
        # training. Uses marker-wrapped data with static standard_mix_prob=0 (above).
        "--mix-start-p", str(cfg.get("mix_start_p", 0.0)),
        "--mix-end-p", str(cfg.get("mix_end_p", 0.0)),
        "--mix-seed", str(mix_seed),
        *(["--gold-grad-mask"] if gold_grad_mask else []),
        "--save-folder", save_folder,
        "--wandb-project", cfg.get("wandb_project", "corpus-reasoning"),
        "--wandb-name", cfg.get("wandb_name", cfg_path.stem),
    ]
    # Default: model-only checkpoint (no optimizer state). Set `save_optim_state: true` in the
    # config for a full resumable checkpoint (~3x larger).
    if cfg.get("save_optim_state", False):
        script_args.append("--save-optim")

    # LoRA: freeze the base, train only injected adapters (olmo_train.py handles the rest;
    # see scripts/lib/olmo_lora.py). Flags flow to both the dry-run gate and the launch.
    if cfg.get("lora", False):
        script_args += [
            "--lora",
            "--lora-r", str(cfg.get("lora_r", 16)),
            "--lora-alpha", str(cfg.get("lora_alpha", 32)),
            "--lora-dropout", str(cfg.get("lora_dropout", 0.05)),
            "--lora-target", str(cfg.get("lora_target", "all_linear")),
        ]

    # 4) Cheap single-process dataset gate (catches dtype/eos-segmentation errors before
    #    grabbing N GPUs; a wrong eos -> "Failed to produce any documents").
    print(f"=== {cfg_path.name}  (olmo-core dry-run gate) ===")
    rc = subprocess.call([sys.executable, *script_args, "--dry-run"])
    if rc != 0:
        sys.exit(f"[olmo] dry-run dataset gate failed (rc={rc})")
    if os.environ.get("OLMO_DRY_RUN"):
        print("[olmo] OLMO_DRY_RUN set — stopping after tokenize + dataset gate.")
        return 0

    # 5) Convert base weights (expensive) only after the cheap gates pass.
    load_path = ensure_converted_ckpt(base_model, spec.vocab_size)

    port = os.environ.get("MASTER_PORT", "0")
    cmd = ["torchrun", "--nproc_per_node", str(g), "--rdzv_backend", "c10d",
           "--rdzv_endpoint", f"localhost:{port}",
           *script_args, "--load-path", str(load_path)]
    print(f"=== {cfg_path.name}  (olmo-core, {g} GPUs, cp={cp}, dp={dp}) ===")
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python scripts/train/train.py <config.yml>")
    cfg_path = Path(sys.argv[1])
    cfg = yaml.safe_load(cfg_path.read_text())
    assert_retrieval_for_hpqa_nq(cfg, cfg_path)

    if cfg.get("backend") == "olmo-core":
        sys.exit(run_olmo_core(cfg, cfg_path))

    n = num_gpus()

    # Chunked-trainer opt-in:
    #   - explicit `standard_attention: false`, OR
    #   - `attention_pattern` set to anything other than "standard".
    # Everything else routes to axolotl (standard causal full FT / LoRA).
    pattern_name = cfg.get("attention_pattern")
    chunked = (cfg.get("standard_attention", True) is False) or (
        pattern_name is not None and pattern_name != "standard"
    )
    # Smoothed trainer is required for features that the fast trainer doesn't
    # implement: attn_smoothing, the hierarchical_anchor per-layer SDPA wrapper,
    # and longlora (S^2-Attn via custom attn function + modules_to_save). All
    # three plug into the SDPA attention slot the fast trainer can't reach.
    longlora_enabled = bool((cfg.get("longlora") or {}).get("enabled", False))
    needs_smoothed = (
        chunked
        and (
            cfg.get("attn_smoothing") is not None
            or pattern_name == "hierarchical_anchor"
        )
    ) or longlora_enabled
    if needs_smoothed:
        trainer_script = "scripts/train/train_chunked_smoothed.py"
        kind = "smoothed"
    elif chunked:
        trainer_script = "scripts/train/train_chunked_fast.py"
        kind = "chunked"
    else:
        trainer_script = None
        kind = "axolotl"
    # Pass MASTER_PORT through as --main_process_port so multiple sbatch
    # jobs sharing a node don't collide on accelerate's default 29500.
    port_args = []
    mp = os.environ.get("MASTER_PORT")
    if mp:
        port_args = ["--main_process_port", str(mp)]
    if trainer_script:
        cmd = ["accelerate", "launch", "--num_processes", str(n), *port_args,
               trainer_script, str(cfg_path)]
    else:
        launch_cfg = prepare_axolotl_yaml(cfg_path, cfg)
        cmd = ["accelerate", "launch", "--num_processes", str(n), *port_args,
               "-m", "axolotl.cli.train", str(launch_cfg)]

    print(f"=== {cfg_path.name}  ({kind}, {n} GPUs) ===")
    print("+", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc == 0 and kind == "axolotl":
        fix_tokenizer_after_fullft(cfg)
    sys.exit(rc)


if __name__ == "__main__":
    main()
