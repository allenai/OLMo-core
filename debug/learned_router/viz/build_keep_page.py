"""Build a self-contained HTML page showing which tokens the learned routers keep vs gold_fl20p8_noslot.

Reads keep_masks.json / keep_stats.json (written by extract_keep_masks.py) and writes one HTML file.

    python debug/learned_router/viz/build_keep_page.py --out <path>.html
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# qualitative read per entry (from inspecting the masks; eval_size 16 at 2k, 8 at 32k)
NOTES = {
    "nq@2k": "Sensible core: gold is kept (99% of gold tokens) and every doc opening survives, including titles. "
             "It drops the ' [' of 'Document [1]:' but keeps the digit, which is harmless. Unlike the rule it keeps every "
             "marker and never keeps document ends; the rest of its non-gold budget is spread in mid-document stripes "
             "that look like filler rather than a meaningful rule.",
    "niah@2k": "Worrying: it keeps only about 66% of the needle document (whole in 1 of 16 test rows), so the needle "
               "sentence is often truncated. This is the v4 + keep-dropout router, the one that matched on one seed "
               "and lost on another.",
    "textgroups@2k": "The strange one. v5 drops non-gold passage numbers but keeps the 'Passage' boilerplate, and a "
                     "learned hole at end offsets 10-11 deletes words inside gold passages ('frozen', 'lavish') that the "
                     "query asks to count. Switch to F0 to see the whole-document rule: gold whole, non-gold documents "
                     "kept or dropped as units by length bucket.",
    "rerank@2k": "Keeps gold and doc openings; drops the ' [' of the id header like nq. Non-gold budget again goes to "
                 "scattered mid-document tokens.",
    "scifact@2k": "Worrying: gold is only 56% kept and never whole. The weights are dominated by very large (±25-34) "
                  "single-offset one-hots, so the keep pattern inside each document is a fixed comb, not a content rule.",
    "strmatch@2k": "Sensible: gold kept whole; in every other document it drops the 'String' word but keeps the id "
                   "('…1:'), which is exactly what a string-match answer needs.",
    "grouping@2k": "No gold set. Keeps markers and the 'Document [1](Title:' header, drops title words in a "
                   "checkerboard pattern, and keeps a band at 30-40% of each abstract with no obvious reason.",
    "contradiction@2k": "Sensible: gold 95% kept; drops the 'Claim' word and keeps the id in other documents.",
    "textgroups@32k": "The 32k failure (v5's worst row: +1.09 nats vs the rule's +0.09). Every long non-gold document "
                      "shrinks to its markers plus a stray 'age' token, and inside gold it drops the counted noun "
                      "'garden'. F0 instead keeps gold whole plus ~25% of other documents whole and drops 71% entirely "
                      "(+0.14 on this row).",
}

REGION_ROWS = [
    ("gold_body", "gold body tokens"),
    ("docs_body_kept_whole_gold", "gold docs kept whole"),
    ("nongold_id_region_j<8", "non-gold first 8 tokens (id)"),
    ("nongold_rest_j>=8_all", "non-gold rest of body"),
    ("markers_nongold", "non-gold markers"),
    ("docs_fully_dropped_incl_markers_nongold", "non-gold docs dropped entirely"),
]


def slim_stats(e):
    reg = e["regions"]
    arms = ["router", "bar"] + (["f0"] if any(k.startswith("f0/") for k in reg) else [])
    rows = []
    for key, label in REGION_ROWS:
        vals = {a: (reg.get(f"{a}/{key}") or {}).get("frac") for a in arms}
        n = (reg.get(f"router/{key}") or {}).get("n_tokens")
        if all(v is None for v in vals.values()):
            continue
        rows.append({"label": label, "n": n, **vals})
    dec = {a: [(reg.get(f"{a}/nongold_rest_j>=8_decile{i}") or {}).get("frac") for i in range(10)] for a in arms}
    w = e.get("router_weights", {})
    out = {"arms": arms, "rows": rows, "deciles": dec, "eval_size": e.get("eval_size"), "target": e.get("target_T2T"),
           "w_pos": w.get("top10_positive", [])[:6], "w_neg": w.get("top10_negative", [])[:6], "variant": w.get("variant")}
    if "f0_weights" in e:
        f = e["f0_weights"]
        out["f0_pos"] = [x for x in f.get("top10_positive", []) if x[1]][:6]
        out["f0_neg"] = [x for x in f.get("top10_negative", []) if x[1]][:6]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    masks = json.load(open(os.path.join(HERE, "keep_masks.json")))["entries"]
    stats = json.load(open(os.path.join(HERE, "keep_stats.json")))["entries"]
    data = []
    for e in masks:
        rows = []
        for r in e["rows"]:
            rows.append({k: r[k] for k in ("row", "T", "n_docs", "gold_docs", "answer", "T2T", "grid_dce", "docs")}
                        | {"q": r["prompt_before_docs"].split("<|im_start|>user\n")[-1][-700:],
                           "after": r["prompt_after_docs"].split("<|im_end|>")[0][-300:]})
        data.append({"id": e["id"], "task": e["task"], "rung": e["rung"], "target": e["target_T2T"],
                     "router_file": os.path.basename(e["router_file"]), "has_f0": "f0_router_file" in e,
                     "rows": rows, "stats": slim_stats(stats[e["id"]]), "note": NOTES.get(e["id"], "")})
    tpl = open(os.path.join(HERE, "keep_page_template.html")).read()
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write(html)
    print(f"wrote {a.out} ({len(html) / 1e6:.2f} MB, {len(data)} entries)")


if __name__ == "__main__":
    main()
