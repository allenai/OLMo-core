"""
Stage — Results source.

Loads the central ``results.csv`` from the results-hub repo and the column
metadata from its ``schema.py`` (facets, numeric, help text), and writes
``outputs/results_table.json`` for the renderer's "Results" tab. Also copies the
raw CSV into ``outputs/`` so the site's download link works.

If the results-hub checkout isn't present, this writes an empty payload so the
rest of the site still builds.

Usage:
    python -m viz.collect_results
    RESULTS_HUB_DIR=/path/to/results-hub python viz/collect_results.py
"""

import csv
import json
import os
import shutil
import sys

try:
    from . import config
except ImportError:
    import config

# Columns shown by default in the table (others available via the column picker).
DEFAULT_VISIBLE = [
    "date_eval_ran", "who_ran", "eval_name", "eval_context_length",
    "eval_data_quantity", "metric_name", "metric_value", "model_type",
    "attention_type", "pipeline_stage", "model_train_data_hparams", "git_commit",
]


def _load_schema_cols():
    """Import the results-hub schema for rich column metadata; fall back to the
    CSV header with sensible defaults if it isn't importable."""
    if config.RESULTS_HUB_DIR not in sys.path:
        sys.path.insert(0, config.RESULTS_HUB_DIR)
    try:
        import schema  # results-hub/schema.py
        return [
            {"name": c.name, "help": c.help, "facet": c.facet, "numeric": c.numeric,
             "enum": c.enum or []}
            for c in schema.COLUMNS
        ], schema.FACETS
    except Exception as e:
        print(f"[collect_results] schema.py not importable ({e}); inferring from CSV header")
        return None, None


def _cols_from_header(header):
    facets = ["who_ran", "eval_name", "metric_name", "model_type",
              "attention_type", "pipeline_stage", "chat_template", "eval_context_length"]
    numeric = ["metric_value", "eval_context_length", "eval_data_quantity",
               "landmark_top_k_fixed_val", "landmark_top_k_percentage",
               "landmark_nonselected_percentage"]
    cols = [{"name": h, "help": h, "facet": h in facets, "numeric": h in numeric, "enum": []}
            for h in header]
    return cols, [h for h in header if h in facets]


def main():
    config.ensure_out_dir()
    rows, header = [], []
    if os.path.isfile(config.RESULTS_CSV):
        with open(config.RESULTS_CSV, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            rows = list(reader)
        print(f"[collect_results] {len(rows)} rows from {config.RESULTS_CSV}")
        shutil.copyfile(config.RESULTS_CSV, os.path.join(config.OUT_DIR, "results.csv"))
    else:
        print(f"[collect_results] no results.csv at {config.RESULTS_CSV} — empty Results tab")

    cols, facets = _load_schema_cols()
    if cols is None:
        cols, facets = _cols_from_header(header)
    # keep only columns that actually appear in the CSV (schema may be ahead of data)
    if header:
        present = set(header)
        cols = [c for c in cols if c["name"] in present]
        facets = [f for f in facets if f in present]

    payload = {
        "rows": rows,
        "cols": cols,
        "facets": facets,
        "default_visible": [c for c in DEFAULT_VISIBLE if not header or c in set(header)],
        "source_csv": config.RESULTS_CSV,
        "n": len(rows),
    }
    with open(config.RESULTS_JSON, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"[collect_results] wrote {config.RESULTS_JSON} ({len(rows)} rows, {len(cols)} cols)")


if __name__ == "__main__":
    main()
