"""
Stage 2b — Collect CPT + Tülu + contradiction "mixing experiment" eval results.

Reads the per-run eval JSONs in ``<corpus-reasoning>/outputs/eval_results/*_lc.json``
(override with ``CR_EVAL_RESULTS_DIR``), parses each run's data mix / LR / batch
out of its filename, and emits ``outputs/mixing_results.json`` — an additive
section consumed by the renderer (a sorted RULER-vs-contradiction table).

Each eval JSON has keys:
    ruler_avg_recall : float
    ruler            : per-subtask dict
    contradiction    : dict with f1, exact_match, parse_rate

ΔRULER is computed relative to the same model's ``*-cpt-base`` (no-SFT) row.

Usage:
    python -m viz.collect_mixing
"""

import glob
import json
import os
import re

try:
    from . import config
except ImportError:
    import config


def _eval_results_dir() -> str:
    return os.environ.get(
        "CR_EVAL_RESULTS_DIR",
        os.path.join(config.CR_REPO_ROOT, "outputs", "eval_results"),
    )


def _parse_lr(tokens):
    """Find an ``lrNeM`` token -> "N e-M" (e.g. lr5e5 -> 5e-5)."""
    for t in tokens:
        m = re.match(r"^lr(\d+)e(\d+)$", t)
        if m:
            return f"{m.group(1)}e-{m.group(2)}"
    return None


def _parse_batch(tokens):
    """Find a batch token: ``bsNm`` (1M-token batch) or a lone ``Nm`` token."""
    for t in tokens:
        m = re.match(r"^bs(\d+)([mk])$", t, re.IGNORECASE)
        if m:
            return f"{m.group(1)}{m.group(2).upper()}"
    for t in tokens:
        m = re.match(r"^(\d+)([mk])$", t, re.IGNORECASE)
        if m:
            return f"{m.group(1)}{m.group(2).upper()}"
    return None


def _pct(digits: str):
    """
    Turn a percent token's digits into a percentage.

    Two-digit / plain integers are the percent directly (``15`` -> 15,
    ``100`` -> 100). A 3+-digit token with a leading zero is a decimal
    fraction (``075`` -> 0.075 -> 7.5%), used by the fine-grained mixes.
    """
    if len(digits) >= 3 and digits[0] == "0":
        return round(float("0." + digits) * 100, 3)
    return int(digits)


def _task_recipe(name: str, cfg: dict) -> str:
    """A short human description of which SFT task(s) a run trains on, from its name."""
    n = name.lower()
    if cfg.get("is_base"):
        return "CPT base (no SFT)"
    if "5task" in n or re.search(r"-5[pw]\d", n) or re.search(r"-5p\d", n):
        return "5-task mix"
    if "nqonly" in n:
        return "NQ-only"
    if "both" in n:
        return "NQ + contra"
    if "cpt100" in n or re.search(r"-cpt100", n):
        return "CPT-only control"
    has_nq = cfg.get("nq") is not None or "nq" in n
    has_contra = (cfg.get("contra") is not None and cfg.get("contra") > 0) or "contra" in n
    parts = []
    if has_contra:
        parts.append("contra")
    if has_nq:
        parts.append("NQ")
    return " + ".join(parts) if parts else "SFT"


def friendly_label(name: str, cfg: dict) -> str:
    """
    Build a human-readable label from a parsed run config.

    e.g. ``q4b-lr1e5-c85-nq075-contra075`` -> "4B · CPT 85% · contra + NQ · lr 1e-5".
    The raw run-name is preserved separately (``cfg['raw']``) for tooltips/subtext.
    """
    if cfg.get("is_base"):
        return f"{cfg.get('model','?')} — CPT base (pre-SFT)"
    bits = [cfg.get("model", "?")]
    if cfg.get("cpt") is not None:
        bits.append(f"CPT {cfg['cpt']}%")
    bits.append(_task_recipe(name, cfg))
    if cfg.get("proxy"):
        bits.append(cfg["proxy"])
    if cfg.get("lr"):
        bits.append(f"lr {cfg['lr']}")
    return " · ".join(str(b) for b in bits if b)


def parse_config(name: str) -> dict:
    """
    Parse a run name (filename minus ``_lc.json``) into a config row.

    Returns a dict with: name, model, is_base, cpt, tulu, contra, nq (percents or
    None), lr, batch, raw (always the original name for unknown configs).
    """
    tokens = name.split("-")
    model = "4B" if name.startswith("q4b") else ("0.6B" if name.startswith("q06b") else "?")

    out = {
        "name": name,
        "model": model,
        "is_base": False,
        "cpt": None,
        "tulu": None,
        "contra": None,
        "nq": None,
        "lr": _parse_lr(tokens),
        "batch": _parse_batch(tokens),
        "proxy": None,
        "raw": name,
    }

    # Fast-proxy validation probes trained on a fraction of the tokens: pNN.
    for t in tokens:
        m = re.match(r"^p(\d+)$", t)
        if m:
            out["proxy"] = f"{int(m.group(1))}% tok"
            break

    # CPT base = no SFT (e.g. q4b-cpt-base).
    if "cpt" in tokens and "base" in tokens:
        out["is_base"] = True
        return out

    cpt = tulu = contra = nq = None
    for t in tokens:
        m = re.match(r"^cpt(\d+)$", t)  # 100% CPT control: cpt100
        if m:
            cpt = _pct(m.group(1))
            continue
        m = re.match(r"^f(\d+)$", t)  # cpt_frac: f85 -> 85% CPT
        if m:
            cpt = _pct(m.group(1))
            continue
        m = re.match(r"^c(\d+)$", t)  # explicit cpt %: c55, c70
        if m:
            cpt = _pct(m.group(1))
            continue
        m = re.match(r"^t(\d+)$", t)  # tülu %: t30
        if m:
            tulu = _pct(m.group(1))
            continue
        m = re.match(r"^contra(\d+)$", t)  # contradiction %: contra15, contra075
        if m:
            contra = _pct(m.group(1))
            continue
        m = re.match(r"^nq(\d+)$", t)  # NQ-retrieval %: nq15, nq075
        if m:
            nq = _pct(m.group(1))
            continue

    out["nq"] = nq
    if cpt is not None:
        if tulu is None:
            tulu = 0
        if contra is None:
            contra = max(0, 100 - cpt - tulu - (nq or 0))
        out["cpt"], out["tulu"], out["contra"] = cpt, tulu, contra
    elif contra is not None or nq is not None:
        # Runs that encode only SFT-task % (e.g. q06b-contra50-lr1e5) imply the
        # remaining fraction is CPT.
        tulu = tulu or 0
        contra = contra or 0
        cpt = max(0, 100 - contra - tulu - (nq or 0))
        out["cpt"], out["tulu"], out["contra"] = cpt, tulu, contra

    # Default LR for SFT runs that don't encode one — our default sweep LR.
    if out["lr"] is None:
        out["lr"] = "5e-5"

    return out


# Hypothesis-organized narrative over the contradiction CPT-mix runs. Each
# section names the runs (eval-JSON basenames minus ``_lc.json``) and which
# metric columns matter; the renderer flags any RULER < ``ruler_ok`` as degraded.
# Runs whose JSON is missing at build time are dropped automatically.
HYPOTHESIS_SECTIONS = [
    {
        "id": "setup",
        "title": "Setup — the forgetting problem",
        "hypothesis": (
            "A long-context CPT model (Qwen3-4B, dolma3longmino) reaches RULER ≈ 0.79–0.82, "
            "but naive SFT on a target task at the default LR catastrophically forgets retrieval. "
            "Goal: instruction-tune the target tasks (contradiction, NQ, oolong, rerank, outlier) "
            "WITHOUT degrading RULER. RULER eval = the 2 smallest contexts (L1024 / L2048), size 100; "
            "keep RULER ≥ ~0.80."
        ),
        "columns": ["model", "lr", "cpt", "contra_f1", "ruler"],
        "runs": ["q4b-cpt-base", "q4b-cpt100-lr5e5-12m", "q4b-cptmix-f85-lr5e5"],
        "takeaway": (
            "Forgetting is real and severe: contra SFT at the default lr5e-5 (c85 mix) learns the task "
            "(contra 0.89) but wipes RULER to 0.03. Everything below is about avoiding that collapse."
        ),
    },
    {
        "id": "h1",
        "title": "H1 — Learning rate controls forgetting",
        "hypothesis": (
            "It is the learning rate, not the task, that drives forgetting: contra-only SFT at lr5e-5 "
            "collapses RULER; lr1e-5 preserves it."
        ),
        "columns": ["model", "lr", "cpt", "contra_f1", "ruler"],
        "runs": [
            "q4b-cptmix-f85-lr5e5",
            "q4b-f85-lr1e5",
            "q4b-p10-contra-lr1e5",
            "q06b-p10-contra-lr5e5",
            "q06b-p10-contra-lr1e5",
            "q06b-p50-contra-lr5e5",
            "q06b-p50-contra-lr1e5",
        ],
        "takeaway": (
            "On 4B, dropping lr5e-5→1e-5 takes RULER 0.03→0.81 at equal contra quality. "
            "(The 0.6B model barely moves either way — see H4.)"
        ),
    },
    {
        "id": "h2",
        "title": "H2 — Mixing CPT data preserves RULER",
        "hypothesis": (
            "Mixing a high fraction of CPT data (0.85) back into SFT preserves RULER while still "
            "learning the task."
        ),
        "columns": ["cpt", "lr", "contra_f1", "nq_f1", "ruler"],
        "runs": [
            "q4b-cpt100-lr5e5-12m",
            "q4b-f85-lr1e5",
            "q4b-lr1e5-c85-nq075-contra075",
            "q4b-lr1e5-c85-nq075-contra075-96M",
        ],
        "takeaway": (
            "c85 + NQ+contra at lr1e-5 keeps RULER at base (0.82) with contra 0.95 / NQ 0.96 — "
            "matching the pure-CPT control (0.81)."
        ),
    },
    {
        "id": "h3",
        "title": "H3 — NQ retrieval intrinsically forgets; high CPT rescues it",
        "hypothesis": (
            "NQ-retrieval SFT intrinsically forgets RULER (more than contradiction does); a high-CPT "
            "mix rescues it."
        ),
        "columns": ["cpt", "nq_f1", "contra_f1", "ruler"],
        "runs": [
            "q4b-p10-nqonly-lr1e5",
            "q4b-lr1e5-c85-nq15",
            "q4b-lr1e5-c85-nq075-contra075-96M",
        ],
        "takeaway": (
            "NQ-only collapses RULER to 0.45 (and even c85-nq15 only reaches 0.55), but the full c85 "
            "NQ+contra split preserves RULER at 0.82 — NQ is the dangerous task, and CPT mixing "
            "neutralizes it."
        ),
    },
    {
        "id": "h4",
        "title": "H4 — Proxy validity: less DATA is a good proxy, a smaller MODEL is not",
        "hypothesis": (
            "10% of the tokens is a valid cheap forgetting proxy for the full 4B run, but the 0.6B "
            "model is too robust to serve as a proxy for the 4B forgetting frontier."
        ),
        "columns": ["model", "proxy", "lr", "nq_f1", "contra_f1", "ruler"],
        "runs": [
            "q4b-p10-nqonly-lr1e5",
            "q4b-p10-contra-lr1e5",
            "q4b-p10-both-lr1e5",
            "q06b-p10-nqonly-lr1e5",
            "q06b-p10-contra-lr1e5",
            "q06b-p10-both-lr1e5",
            "q06b-p50-nqonly-lr1e5",
            "q06b-p50-contra-lr1e5",
            "q06b-p50-both-lr1e5",
        ],
        "takeaway": (
            "At 10% tokens the 4B model still reveals the frontier (NQ-only RULER 0.45 vs contra 0.79), "
            "so 10% data is a good proxy. But the 0.6B model's RULER stays ~0.65–0.70 regardless of "
            "task / LR / token-budget — it masks the collapse, so a smaller model CANNOT proxy the "
            "4B forgetting frontier."
        ),
    },
    {
        "id": "h5",
        "title": "H5 — 5-task mix + weighting",
        "hypothesis": (
            "Adding oolong, msmarco-rerank, outlier: an even 5-way split starves the hard tasks "
            "(contra/rerank/outlier); a weighted split (contra 2x, rerank/outlier 1.5x) gives them more."
        ),
        "columns": ["cpt", "lr", "contra_f1", "nq_f1", "oolong", "rerank", "outlier", "ruler"],
        "runs": [
            "q4b-5p10-c070",
            "q4b-5p10-c025",
            "q4b-5w10-c085",
            "q4b-5w10-c040-lr5e6",
            "q4b-lr1e5-5task-c85",
        ],
        "takeaway": (
            "Even-split (5p10) leaves contra at 0.09–0.25; the weighted mix (5w10) and the full-token "
            "5task-c85 lift contra (0.21 / 0.88) while keeping the other four. At 10% tokens (~146 steps) "
            "contra can't fully learn — the proxy measures forgetting, the full run measures learning."
        ),
    },
    {
        "id": "h6",
        "title": "H6 — Min-CPT & LR frontier for the 5-task mix",
        "hypothesis": (
            "RULER falls as the CPT fraction drops; a lower LR (5e-6) buys back RULER at lighter CPT."
        ),
        "columns": ["cpt", "lr", "contra_f1", "nq_f1", "ruler"],
        "runs": [
            "q4b-5w10-c085",
            "q4b-5w10-c040-lr5e6",
            "q4b-5p10-c070",
            "q4b-5p10-c025",
        ],
        "takeaway": (
            "5w10: c085→RULER 0.765; 5p10: c070→0.791, c025→0.598. lr5e-6 at c040 holds RULER "
            "at 0.719 (vs the lr1e-5 trend ~0.66) — there is a CPT/LR frontier to tune in the full run. "
            "(Some lighter-CPT 5w10 runs may still be completing.)"
        ),
    },
    {
        "id": "h7",
        "title": "H7 — Generalization: NQ → other retrieval",
        "hypothesis": (
            "NQ-style retrieval SFT transfers to held-out SINGLE-query retrieval but collapses on "
            "MULTI-query HotpotQA (output-format mismatch)."
        ),
        "columns": ["nq_f1", "gen_fiqa", "gen_scifact", "gen_msmarco", "gen_hpqa"],
        "runs": [
            "q4b-5p10-c070",
            "q4b-5p10-c025",
            "q4b-5w10-c085",
            "q4b-5w10-c040-lr5e6",
        ],
        "takeaway": (
            "gen_fiqa 0.46–0.63, gen_scifact 0.32–0.66, gen_msmarco 0.70–0.73 ≈ "
            "in-distribution NQ (~0.62), but gen_hpqa collapses to ~0.00–0.14 — a multi-query "
            "output-format mismatch, not a retrieval failure."
        ),
    },
]


def collect() -> dict:
    """Build the mixing-results payload from the eval-result JSONs."""
    eval_dir = _eval_results_dir()
    rows = []
    for path in sorted(glob.glob(os.path.join(eval_dir, "*_lc.json"))):
        name = os.path.basename(path)[: -len("_lc.json")]
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        cfg = parse_config(name)
        cfg["disp"] = friendly_label(name, cfg)
        contra = d.get("contradiction", {}) or {}
        nq = d.get("nq", {}) or {}
        cfg["ruler"] = d.get("ruler_avg_recall")
        cfg["contra_f1"] = contra.get("f1")
        cfg["contra_em"] = contra.get("exact_match")
        cfg["parse_rate"] = contra.get("parse_rate")
        cfg["nq_f1"] = nq.get("f1")
        # 5-task suite primaries (oolong / msmarco-rerank / outlier).
        cfg["oolong"] = d.get("oolong_primary")
        cfg["rerank"] = d.get("rerank_primary")
        cfg["outlier"] = d.get("outlier_primary")
        # NQ-retrieval generalization probes (held-out single/multi-query retrieval).
        for g in ("gen_hpqa", "gen_fiqa", "gen_msmarco", "gen_scifact"):
            gd = d.get(g) or {}
            cfg[g] = gd.get("f1") if isinstance(gd, dict) else None
        rows.append(cfg)

    # ΔRULER vs the same model's CPT base.
    base_ruler = {r["model"]: r["ruler"] for r in rows if r.get("is_base")}
    for r in rows:
        b = base_ruler.get(r["model"])
        r["d_ruler"] = (
            (r["ruler"] - b) if (b is not None and r.get("ruler") is not None and not r["is_base"]) else None
        )

    # Sort: 4B before 0.6B; within a model, base first then RULER descending.
    model_order = {"4B": 0, "0.6B": 1}

    def sort_key(r):
        return (
            model_order.get(r["model"], 9),
            0 if r["is_base"] else 1,
            -(r["ruler"] if r.get("ruler") is not None else -1),
        )

    rows.sort(key=sort_key)

    caption = (
        "Winning recipe: lr 1e-5 + ~15% total SFT (split across tasks) + CPT 0.85 + enough "
        "tokens → contradiction 0.95, NQ 0.96, RULER preserved (0.82). NQ-the-task specifically "
        "costs RULER; contradiction doesn't."
    )

    # Drop any section rows whose eval JSON is missing at build time.
    present = {r["name"] for r in rows}
    sections = [
        {**s, "runs": [n for n in s["runs"] if n in present]} for s in HYPOTHESIS_SECTIONS
    ]
    sections = [s for s in sections if s["runs"]]

    return {
        "caption": caption,
        "base_ruler": base_ruler,
        "rows": rows,
        "sections": sections,
        "ruler_ok": 0.80,
        "ruler_ref": 0.82,
        "source": "outputs/eval_results/*_lc.json",
    }


def main():
    config.ensure_out_dir()
    payload = collect()
    out_path = os.path.join(config.OUT_DIR, "mixing_results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[collect_mixing] wrote {out_path} ({len(payload['rows'])} runs)")


if __name__ == "__main__":
    main()
