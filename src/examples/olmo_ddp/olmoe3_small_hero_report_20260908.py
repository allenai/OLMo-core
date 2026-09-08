"""Build a live W&B report without modifying any training run or its logged history.

Run with isolated dependencies: uv run --no-project --with wandb-workspaces python
src/examples/olmo_ddp/olmoe3_small_hero_report_20260908.py --output-dir <directory>
Add --publish after inspecting the generated inventory and report specification.
"""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import wandb
import wandb_workspaces.reports.v2 as wr

ENTITY = "ai2-llm"
PROJECT = "olmo3p5-hero"
TOKEN = "throughput/total tokens"
LOSS = "train/CE loss"
HERO_IDS = ("lasc1m2x", "aqb1droj")
COLORS = {"lasc1m2x": "#2563eb", "aqb1droj": "#ea580c"}


def inventory_run(run, *, min_step=0):
    """Retain only report-relevant metadata; never serialize secrets or full configs."""
    config = run.config
    summary = dict(run.summary.items())
    keys = sorted(k for k in summary if not any(word in k for word in ("/block ", "/layer ")))
    first = None
    if summary.get(TOKEN) is not None and summary.get(LOSS) is not None:
        # Scan pages cover STEP ranges, not returned-row counts. Large pages avoid
        # hundreds of thousands of empty windows before a resumed run's first step.
        first = next(
            (
                row
                for row in run.scan_history(
                    keys=[TOKEN, LOSS, "_step"], page_size=10_000, min_step=min_step
                )
                if row.get(TOKEN) is not None and row.get(LOSS) is not None
            ),
            None,
        )
    evaluator = config.get("trainer", {}).get("callbacks", {}).get("lm_evaluator", {})
    return {
        "id": run.id,
        "name": run.name,
        "url": run.url,
        "state_at_inventory": run.state,
        "created_at": run.created_at,
        "first_loss_row": first,
        "last_step": summary.get("_step"),
        "last_tokens": summary.get(TOKEN),
        "last_loss": summary.get(LOSS),
        "keys": keys,
        "model": {k: config.get("model", {}).get(k) for k in ("d_model", "n_layers", "vocab_size")},
        "batch_tokens": config.get("data_loader", {}).get("global_batch_size"),
        "sequence_length": config.get("dataset", {}).get("sequence_length"),
        "mix": config.get("dataset", {}).get("mix"),
        "tokenizer": config.get("dataset", {}).get("tokenizer"),
        "lm_eval_interval": evaluator.get("eval_interval"),
        "lm_eval_duration": evaluator.get("eval_duration"),
    }


def make_inventory():
    """Include every OLMo25 restart, even crashed/empty records, and both hero histories."""
    api = wandb.Api(timeout=60)
    heroes = [inventory_run(api.run(f"{ENTITY}/{PROJECT}/{rid}")) for rid in HERO_IDS]
    dense = []
    for run in api.runs(f"{ENTITY}/olmo3", filters={"group": "OLMo25"}):
        record = inventory_run(run)
        assert record["model"]["d_model"] == 4096 and record["model"]["n_layers"] == 32
        dense.append(record)
        print("INVENTORY", run.id, record["last_tokens"], flush=True)
    dense.sort(key=lambda r: r["created_at"])
    dense_keys = set().union(*(set(r["keys"]) for r in dense))
    shared = set(heroes[0]["keys"]) & set(heroes[1]["keys"]) & dense_keys
    shared_lm = sorted(k for k in shared if k.startswith("eval/lm/") and k.endswith("/CE loss"))
    assert len(shared_lm) == 11, shared_lm
    assert all(r["first_loss_row"] is not None for r in heroes)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "heroes": heroes,
        "dense": dense,
        "shared_lm": shared_lm,
        "shared_token_limit_at_creation": min(r["last_tokens"] for r in heroes),
    }


def runsets(inv, dense=True):
    """Explicit IDs, independent restart segments, stable colors; no state filters or averaging."""
    result = [
        wr.Runset(
            entity=ENTITY,
            project=PROJECT,
            name="Small EMO" if rid == HERO_IDS[0] else "Small non-EMO",
            filters=f"Metric('name') == '{rid}'",
            run_settings={rid: wr.RunSettings(color=COLORS[rid])},
        )
        for rid in HERO_IDS
    ]
    if dense:
        ids = [r["id"] for r in inv["dense"]]
        result.append(
            wr.Runset(
                entity=ENTITY,
                project="olmo3",
                name="OLMo 3 7B / OLMo25 (all restart segments)",
                filters=f"Metric('name') in {ids!r}",
                run_settings={rid: wr.RunSettings(color="#737373") for rid in ids},
            )
        )
    return result


def plot(metric, title, *, upper=None, smooth=False, expression=None):
    """Use logged absolute token counts, raw evals, visible loss smoothing and no run aggregation."""
    return wr.LinePlot(
        title=title,
        x=TOKEN,
        y=[metric],
        title_x="Tokens seen (absolute, not optimizer steps)",
        title_y=metric,
        range_x=(0, upper),
        xaxis_format=".3s",
        aggregate=False,
        groupby=None,
        ignore_outliers=False,
        smoothing_type="exponentialTimeWeighted" if smooth else "none",
        smoothing_factor=0.7 if smooth else 0,
        smoothing_show_original=True,
        max_runs_to_show=100,
        point_visualization_method="bucketing-gorilla",
        custom_expressions=[expression] if expression else None,
        layout=wr.Layout(w=12, h=8),
    )


def grid(sets, panels):
    """Lay out two readable charts per row without sharing mutable runset objects."""
    for index, panel in enumerate(panels):
        panel.layout = wr.Layout(x=(index % 2) * 12, y=(index // 2) * 8, w=12, h=8)
    return wr.PanelGrid(runsets=copy.deepcopy(sets), panels=panels)


def make_blocks(inv):
    """Build an ongoing comparison plus an explicitly timestamped common-token zoom."""
    all_sets = runsets(inv)
    hero_sets = runsets(inv, dense=False)
    overlap = inv["shared_token_limit_at_creation"]
    dense_max = max(r["last_tokens"] or 0 for r in inv["dense"])
    nonempty = sum(r["first_loss_row"] is not None for r in inv["dense"])
    blocks = [
        wr.P(
            "Live comparison: Small EMO (blue), Small non-EMO (orange), OLMo 3 7B / OLMo25 "
            "restart segments (gray). Panels read the original runs directly; their histories "
            "will continue updating. No copied/derived training runs were created."
        ),
        wr.P(
            f"Inventory: {inv['created_at']}. Both hero IDs retain history across Beaker resumes. "
            f"All {len(inv['dense'])} OLMo25 records are included ({nonempty} with loss history; "
            f"coverage through {dense_max / 1e12:.3f}T tokens). Failed/crashed statuses are NOT "
            "filtered out. Gray segments may overlap after retries; they are not averaged, "
            "spliced into an inferred trajectory, or treated as independent training seeds."
        ),
        wr.TableOfContents(),
        wr.H1("How to interpret this comparison"),
        wr.P(
            "The controlled comparison is EMO vs non-EMO: the same 0.794B-active / 12.496B-total "
            "16-layer small model, 64 B300 GPUs, 16,777,216-token batch, LR 1.1e-3, 2,000-step "
            "warmup, and qualified BF16 optimizations. OLMo 3 7B is a dense, much larger-active "
            "reference, not a controlled EMO ablation or compute-matched baseline."
        ),
        wr.P(
            "Small configuration: d_model=1024, latent=512, 14 KDA + 2 full-attention layers, "
            "Q/KV heads=8/4, 512 routed experts, top-16 plus one shared expert. EMO pools "
            "range from 16 to 512 when enabled. Both use PP1 / EP1 / DP64, microbatch 4, "
            "gradient accumulation 8, synchronous full-state checkpoints and no activation "
            "recomputation. Stable WSD trunks initially stop around 3T and remain continuable "
            "to 14T; a final decay is not yet selected."
        ),
        wr.P(
            "Both use the Dolma2 tokenizer family (100,278 tokens before embedding padding) "
            "and 8,192-token training context. Data differ: Dolma3p5_14t for the heroes vs "
            "OLMo-mix-0625 for OLMo25; the baseline batch is 4,194,304 tokens. Therefore train "
            "CE levels also reflect data-mixture differences. Match held-out metric names "
            "and tokens for the more useful quality comparison; do not infer equal compute. "
            "Eval harness versions and exact examples have not been byte-for-byte matched."
        ),
        wr.P(
            "All plots use throughput/total tokens, not W&B step or wall time. Training CE "
            "has moderate time-weighted smoothing with original data visible; held-out evals "
            "are unsmoothed. No per-layer/per-block charts, cross-domain averages, RULER+, "
            "or invented missing metrics. Hero in-loop LM evals occur every 1,000 steps; "
            "baseline LM evals typically every 10,000 steps."
        ),
        wr.H1("Training loss"),
        grid(
            all_sets,
            [
                plot(LOSS, "Training CE — all recorded tokens", smooth=True),
                plot(
                    LOSS,
                    f"Training CE — common window at creation (0–{overlap / 1e9:.1f}B)",
                    upper=overlap,
                    smooth=True,
                ),
            ],
        ),
        wr.H2("Live EMO ablation (auto-expanding token range)"),
        grid(hero_sets, [plot(LOSS, "Small EMO vs non-EMO — training CE", smooth=True)]),
        wr.H1("Shared held-out LM evaluation"),
        wr.P(
            "These are the 11 shared logged CE-loss metrics (lower is better), shown "
            "individually. Axes are capped at the common hero-token limit at report creation "
            "so later dense-model results are not mistaken for matched-budget results. "
            "Remove the x-axis cap in the UI as the heroes progress, or rerun the report builder."
        ),
        grid(
            all_sets,
            [plot(k, k.split("/")[2], upper=overlap) for k in inv["shared_lm"]],
        ),
        wr.H1("Optimization health"),
        grid(
            all_sets,
            [
                plot("optim/LR (group 0)", "Learning rate — schedules differ"),
                plot("optim/total grad norm", "Total gradient norm — scale/model dependent"),
                plot("optim/step skipped", "Skipped optimizer step (0/1)"),
            ],
        ),
        wr.H1("Aggregate MoE behavior — small models only"),
        wr.P(
            "Only the existing model-level aggregates are shown. These are logger-defined "
            "aggregates, not a new average across layers, and their reduction/normalization "
            "must be respected. EMO pool masking can affect their interpretation."
        ),
        grid(
            hero_sets,
            [
                plot("train/load balancing loss", "Load-balancing auxiliary loss"),
                plot("train/router Z loss", "Router Z auxiliary loss"),
                plot("train/load imbalance", "Aggregate load imbalance"),
                plot("train/global load imbalance", "Aggregate global load imbalance"),
            ],
        ),
        wr.H1("Small-run performance and operational health"),
        wr.P(
            "These panels compare only the hardware-matched heroes. TPS is logged per GPU, "
            "not job-total TPS. MFU is the current logger's estimate, not a cross-hardware "
            "benchmark; startup, evaluation and checkpoint overhead can depress samples."
        ),
        grid(
            hero_sets,
            [
                plot("throughput/device/TPS", "Per-GPU tokens/s", smooth=True),
                plot("throughput/device/MFU", "Logged per-GPU MFU (%)"),
                plot("throughput/device/TFLOPs_per_GPU", "TFLOP/s per GPU"),
                plot("gpu_memory/GPU active mem (GiB)", "Active GPU memory (GiB)"),
                plot("checkpoint/save_duration_s", "Full-state checkpoint save time (s)"),
                plot("throughput/device/data loading (%)", "Data-loading fraction (%)"),
            ],
        ),
        wr.H1("Downstream evals — dense reference only"),
        wr.P(
            "The hero runs do not currently log downstream task accuracies. The following "
            "baseline-only panels are context, not a three-way comparison. Only the logged "
            "accuracy-v2 variants are selected; no blending with the older metric variants."
        ),
    ]
    dense_keys = set().union(*(set(r["keys"]) for r in inv["dense"]))
    selected = [
        "eval/downstream/arc_challenge_test_mc_5shot_fast (accuracy v2)",
        "eval/downstream/arc_easy_test_mc_5shot_fast (accuracy v2)",
        "eval/downstream/mmlu_stem_test_mc_5shot_fast (length-normalized accuracy v2)",
        "eval/downstream/mmlu_humanities_test_mc_5shot_fast (length-normalized accuracy v2)",
        "eval/downstream/mmlu_social_sciences_test_mc_5shot_fast (length-normalized accuracy v2)",
        "eval/downstream/mmlu_other_test_mc_5shot_fast (length-normalized accuracy v2)",
    ]
    blocks.append(
        grid([all_sets[-1]], [plot(k, k.split("/")[-1]) for k in selected if k in dense_keys])
    )
    blocks.extend([wr.H1("Run inventory and provenance")])
    rows = [
        "| Model | Run | First loss tokens (B) | Last tokens (B), snapshot |",
        "|---|---|---:|---:|",
    ]
    for label, records in [("Small", inv["heroes"]), ("OLMo25", inv["dense"])]:
        for r in records:
            first = r["first_loss_row"]
            start = f"{first[TOKEN] / 1e9:.3f}" if first else "—"
            end = f"{r['last_tokens'] / 1e9:.3f}" if r["last_tokens"] is not None else "—"
            rows.append(f"| {label} | [{r['name']}]({r['url']}) | {start} | {end} |")
    blocks.append(wr.MarkdownBlock("\n".join(rows)))
    blocks.append(
        wr.P(
            "Source selection follows the provided script: ai2-llm/olmo3, group OLMo25. "
            "All selected configs were verified as d_model=4096, n_layers=32. Empty attempts "
            "remain in the inventory but naturally draw no line. The small source IDs are "
            "lasc1m2x and aqb1droj, including history before and after cadence resumes. "
            "No source-run names, tags, configs, metrics, or access permissions were modified."
        )
    )
    return blocks


def main():
    """Inventory and validate locally, optionally publish/update exactly one report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-inventory", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output_dir / "run_inventory.json"
    inv = json.loads(inventory_path.read_text()) if args.reuse_inventory else make_inventory()
    # SDK scans can include metadata-only rows with null requested metrics.
    # Repair inventories produced before the explicit non-null row check.
    api = None
    for r in inv["heroes"] + inv["dense"]:
        first = r["first_loss_row"]
        if first is not None and (first.get(TOKEN) is None or first.get(LOSS) is None):
            api = api or wandb.Api(timeout=60)
            project = PROJECT if r["id"] in HERO_IDS else "olmo3"
            repaired = inventory_run(
                api.run(f"{ENTITY}/{project}/{r['id']}"), min_step=int(first.get("_step") or 0)
            )
            r["first_loss_row"] = repaired["first_loss_row"]
    inventory_path.write_text(json.dumps(inv, indent=2) + "\n")
    receipt = args.output_dir / "report_receipt.json"
    report = (
        wr.Report.from_url(json.loads(receipt.read_text())["url"])
        if receipt.exists()
        else wr.Report(
            entity=ENTITY,
            project=PROJECT,
            title="OLMo 3.5 Small: EMO vs non-EMO, with OLMo 3 7B pretraining reference",
            description="Live token-aligned hero comparison, shared held-out LM evals and aggregate MoE health.",
            width="fluid",
        )
    )
    report.blocks = make_blocks(inv)
    model = report._to_model()
    (args.output_dir / "report_spec.json").write_text(model.model_dump_json(indent=2) + "\n")
    grids = [b for b in report.blocks if isinstance(b, wr.PanelGrid)]
    assert all(p.max_runs_to_show >= len(inv["dense"]) + 2 for g in grids for p in g.panels)
    assert all(
        p.x == TOKEN and not p.aggregate and p.groupby is None for g in grids for p in g.panels
    )
    assert not any("/block " in str(p.y) or "/layer " in str(p.y) for g in grids for p in g.panels)
    print("REPORT_VALIDATED", len(grids), "grids", sum(len(g.panels) for g in grids), "panels")
    if args.publish:
        report.save()
        # Retain the exact URL before optional read-back, so a read timeout does
        # not create another report on the next explicitly requested update.
        receipt.write_text(
            json.dumps(
                {"url": report.url, "saved_at": datetime.now(timezone.utc).isoformat()}, indent=2
            )
            + "\n"
        )
        saved = wr.Report.from_url(report.url)
        assert len(saved.blocks) == len(report.blocks)
        print("REPORT_SAVED", report.url, flush=True)


if __name__ == "__main__":
    main()
