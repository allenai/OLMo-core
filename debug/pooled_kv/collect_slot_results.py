"""
Collect every eval-side slot-construction probe (Beaker logs + local sneetches/horton logs) into one
CSV, `results/pooled_kv/slot_probe_results.csv`, one row per (task, rung, checkpoint, source,
construction, keep, G): answer CE, delta vs full attention, top-1 agreement, KL, exact-answer rate,
compaction ratio, and the implied train-time speedup (1 / compaction, validated by
bench_softtoken_throughput: 3.5x at 0.355, 11.9x at 0.10) and activation-memory saving (1 - c).
Constructions that need the dense forward's own K/V (oracle mean-KV, fitted slot) are ceilings and
carry no train-time saving of their own; they are flagged.

    python debug/pooled_kv/collect_slot_results.py
"""

import csv
import json
import os
import re
import subprocess

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
OUT = f"{REPO}/results/pooled_kv"
ENV = dict(os.environ, PATH="/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:" + os.environ.get("PATH", ""))

# (source label, task, rung, checkpoint label, kind, beaker experiment id | local log path)
SOURCES = [
    ("beaker", "contradiction", "32k", "dense 56M (ladder)", "slot", "01M20YEJ995B06CSRYSNQS5DDW"),
    ("beaker", "oolong", "32k", "dense 80M (ladder)", "slot", "01M20YFQZFZ1YBQXCY6JM9X5AE"),
    ("beaker", "nq", "32k", "dense 48M (ladder)", "slot", "01M2162XMKDCWJ3GVCWFJQR96X"),
    ("beaker", "outlier", "32k", "dense 160M (ladder)", "slot", "01M2166T2QBWZX0QD5VYXTYBNF"),
    ("beaker", "contradiction", "32k", "dense 56M (ladder)", "oracle", "01M20Z4DWPGQNP176TZHBBYEHE"),
    ("beaker", "oolong", "32k", "dense 80M (ladder)", "oracle", "01M20Z5R7MJRKQP0SDJ5SNQDF5"),
    ("beaker", "nq", "32k", "dense 48M (ladder)", "oracle", "01M2163N012NX6J3NS003CWSPT"),
    ("beaker", "outlier", "32k", "dense 160M (ladder)", "oracle", "01M2167JH8D5WXKB348NCT662M"),
    ("beaker", "contradiction", "32k", "dense 56M (ladder)", "ceiling", "01M210R84TYFDSXYF9PZYVMFNV"),
    ("beaker", "oolong", "32k", "dense 80M (ladder)", "ceiling", "01M210SDGGF8G0G8JQF49SY25G"),
    ("beaker", "nq", "32k", "dense 48M (ladder)", "ceiling", "01M2164GTC9XFSZA1K12R5QVF6"),
    ("beaker", "outlier", "32k", "dense 160M (ladder)", "ceiling", "01M2168CQP92M2HNG99AAF124D"),
    ("sneetches", "contradiction", "32k", "dense 56M (ladder)", "slot", "/net/sneetches/data/prasann/slot_probe/slot_contradiction_32768.log"),
    ("sneetches", "contradiction", "32k", "dense 56M (ladder)", "oracle", "/net/sneetches/data/prasann/slot_probe/oracle_contradiction_32768.log"),
    ("sneetches", "oolong", "32k", "dense 80M (ladder)", "slot", "/net/sneetches/data/prasann/slot_probe/slot_oolong_32768.log"),
    ("sneetches", "oolong", "32k", "dense 80M (ladder)", "oracle", "/net/sneetches/data/prasann/slot_probe/oracle_oolong_32768.log"),
    ("sneetches", "contradiction", "8k", "dense 56M (ladder)", "slot", "/net/sneetches/data/prasann/slot_probe/slot_contradiction_8192.log"),
    ("sneetches", "contradiction", "8k", "dense 56M (ladder)", "oracle", "/net/sneetches/data/prasann/slot_probe/oracle_contradiction_8192.log"),
    ("sneetches", "oolong", "8k", "dense 80M (ladder)", "slot", "/net/sneetches/data/prasann/slot_probe/slot_oolong_8192.log"),
    ("sneetches", "oolong", "8k", "dense 80M (ladder)", "oracle", "/net/sneetches/data/prasann/slot_probe/oracle_oolong_8192.log"),
    ("sneetches", "contradiction", "32k", "dense 56M (ladder)", "ceiling", "/net/sneetches/data/prasann/slot_probe/ceiling_contradiction_32768.log"),
    ("sneetches", "oolong", "32k", "dense 80M (ladder)", "ceiling", "/net/sneetches/data/prasann/slot_probe/ceiling_oolong_32768.log"),
    ("sneetches", "contradiction", "8k", "dense 56M (ladder)", "ceiling", "/net/sneetches/data/prasann/slot_probe/ceiling_contradiction_8192.log"),
    ("sneetches", "oolong", "8k", "dense 80M (ladder)", "ceiling", "/net/sneetches/data/prasann/slot_probe/ceiling_oolong_8192.log"),
    ("horton", "contradiction", "32k", "s5 2k-trained 4B (local)", "slot", "/net/horton/data/prasann/slot_probe/contra_s5_32768.log"),
    ("horton", "contradiction", "8k", "s5 2k-trained 4B (local)", "slot", "/net/horton/data/prasann/slot_probe/contra_s5_8192.log"),
    ("beaker", "contradiction", "32k", "Qwen3-4B dense 56M (pure attention)", "slot", "01M214G6XMJ62A91BA5ZT4H1PR"),
    ("beaker", "contradiction", "32k", "Qwen3-4B dense 56M (pure attention)", "oracle", "01M217TQZECQXZGBTFJZ712QMZ"),
    ("beaker", "contradiction", "32k", "Qwen3-4B dense 56M (pure attention)", "ceiling", "01M217VR74HJXMAN3TCN0QJS4E"),
    ("beaker", "oolong", "32k", "Qwen3-4B dense 80M (pure attention)", "slot", "01M217WJGYH236Q6X16JPB1NA8"),
    ("beaker", "oolong", "32k", "Qwen3-4B dense 80M (pure attention)", "oracle", "01M217XGKD91751NSBNNMN3J84"),
    ("beaker", "oolong", "32k", "Qwen3-4B dense 80M (pure attention)", "ceiling", "01M217YAYERS22875JPFWY4PMQ"),
    ("beaker", "contradiction", "32k", "dense 56M (ladder)", "pooledkv", "01M215860C2NMTS9A7BBVSENZ0"),
    ("beaker", "oolong", "32k", "dense 80M (ladder)", "pooledkv", "01M2158X5K2VRJZ45MM28R6S6S"),
    ("beaker", "nq", "32k", "dense 48M (ladder)", "pooledkv", "01M21658V40RG6S2DW1B2CGC0W"),
    ("beaker", "outlier", "32k", "dense 160M (ladder)", "pooledkv", "01M2169HPQ4DGM25KEB02FAW4J"),
    ("beaker", "contradiction", "32k", "dense 56M (ladder)", "policy", "01M2183MZJVTJYV7KBD2J0ZXWE"),
    ("beaker", "oolong", "32k", "dense 80M (ladder)", "policy", "01M2184TV0VNN7133Y93WQC05C"),
    ("beaker", "nq", "32k", "dense 48M (ladder)", "policy", "01M2185V4XWRQWH27XAQ0ZKSX4"),
    ("beaker", "outlier", "32k", "dense 160M (ladder)", "policy", "01M2187549JW0XYXQAP3P4V46N"),
    ("beaker", "nq", "32k", "dense 48M (ladder)", "policy-hardneg", "01M218SJE5KETF99DAND6GR5EV"),
    ("beaker", "contradiction", "32k", "dense 56M (ladder)", "gdn-nowrite", "01M218XKAHWSBCBJ34AC07MD7T"),
    ("beaker", "oolong", "32k", "dense 80M (ladder)", "gdn-nowrite", "01M218YDV9CGGRAHY4VNQC1H7Q"),
    ("beaker", "nq", "32k", "dense 48M (ladder)", "gdn-nowrite", "01M218Z99QR0FQVS9RYEYD1PN8"),
    ("beaker", "outlier", "32k", "dense 160M (ladder)", "gdn-nowrite", "01M21903R27BPJZF2KRPVXPZF3"),
    # v2 (2026-09-08 afternoon, sneetches): prefix-real headers, gold-neighbour policies, gdn-nowrite, per-row dumps
    ("sneetches", "oolong", "32k", "dense 80M (ladder)", "v2", "/net/sneetches/data/prasann/slot_probe/v2_oolong_32768.log"),
    ("sneetches", "contradiction", "32k", "dense 56M (ladder)", "v2", "/net/sneetches/data/prasann/slot_probe/v2_contradiction_32768.log"),
    # v3: all-pooled oolong slot-bias sweep, slot RoPE position (start/end vs centre)
    ("sneetches", "oolong", "32k", "dense 80M (ladder)", "v3-k0bias", "/net/sneetches/data/prasann/slot_probe/v3_oolong_k0bias_32768.log"),
    ("sneetches", "oolong", "32k", "dense 80M (ladder)", "v3-slotpos", "/net/sneetches/data/prasann/slot_probe/v3_oolong_slotpos_32768.log"),
    ("sneetches", "contradiction", "32k", "dense 56M (ladder)", "v3-slotpos", "/net/sneetches/data/prasann/slot_probe/v3_contradiction_slotpos_32768.log"),
    # v4: cheapest parity (header real at keep 1/36, 0), leak-free neighbour runs, left/right neighbour
    ("sneetches", "contradiction", "32k", "dense 56M (ladder)", "v4", "/net/sneetches/data/prasann/slot_probe/v4_contradiction_32768.log"),
    # v5: cheaper headers (content-only, last-2, boundary-only, fraction of docs)
    ("sneetches", "oolong", "32k", "dense 80M (ladder)", "v5-header", "/net/sneetches/data/prasann/slot_probe/v5_oolong_32768.log"),
    ("sneetches", "contradiction", "32k", "dense 56M (ladder)", "v5-header", "/net/sneetches/data/prasann/slot_probe/v5_contradiction_32768.log"),
    # Beaker confirmations on the eval-bundle rows (2026-09-08 13:25): header real, neighbour runs
    ("beaker", "contradiction", "32k", "dense 56M (ladder)", "header", "01M21B3PRZGWQNJJTP7SN5SPT4"),
    ("beaker", "oolong", "32k", "dense 80M (ladder)", "header", "01M21B4F4VAER7E8MJZC1F0G05"),
    ("beaker", "nq", "32k", "dense 48M (ladder)", "nbr-runs", "01M21B581CNK4WTQN8EEDM7B06"),
    ("beaker", "outlier", "32k", "dense 160M (ladder)", "nbr-runs", "01M21B612ZEK7TFXC8ERARDW19"),
]

ROW = re.compile(r"^(?P<name>full|soft .*?|k=.*?|G=.*?|pooledKV .*?)\s{2,}(?P<ce>[0-9.]+)\s+(?P<top1>[0-9.]+)\s+(?P<kl>[0-9.]+)\s+(?P<correct>[0-9.]+)\s+(?P<comp>[0-9.]+)")


def read_text(src):
    if src.startswith("/"):
        return open(src).read() if os.path.exists(src) else ""
    try:
        return subprocess.run(["beaker", "experiment", "logs", src], env=ENV, capture_output=True, text=True, timeout=300).stdout
    except Exception:  # noqa: BLE001
        return ""


def parse_last_table(text):
    """The LAST printed table (cumulative means) and the row count it covers."""
    lines = [re.sub(r"^\S+Z ", "", ln.rstrip()) for ln in text.splitlines()]  # strip beaker timestamps
    rows_done = 0
    for ln in lines:
        m = re.search(r"\[probe\] row (\d+)/(\d+)", ln)
        if m:
            rows_done = int(m.group(1))
    starts = [i for i, ln in enumerate(lines) if ln.startswith("config ")]
    if not starts:
        return rows_done, []
    table = []
    for ln in lines[starts[-1] + 1:]:
        m = ROW.match(ln)
        if not m:
            if table:
                break
            continue
        table.append(m.groupdict())
    return rows_done, table


def classify(name):
    """-> (construction, keep, G, bias, ceiling?)"""
    keep = re.search(r"k=([0-9.]+)", name)
    keep = float(keep.group(1)) if keep else 1.0
    G = re.search(r"G=(\d+)", name)
    G = int(G.group(1)) if G else 1
    n = name
    if n == "full":
        return "full attention", 1.0, 1, "", False
    if "fitted" in n:
        return "fitted log-mass slot (k*, v*, c)", keep, G, "fitted c", True
    if "oracle meanKV" in n:
        bias = n.split("oracle meanKV")[1].strip() or "no-bias"
        return "oracle mean K/V slot", keep, G, bias, True
    pre = re.search(r"prefix=([\w.@-]+)", n)
    pre_tag = f", doc header real ({pre.group(1)})" if pre else ""
    n = n.replace(pre.group(0), "").replace("  ", " ").strip() if pre else n
    sp = re.search(r"slotpos=(\w+)", n)
    if sp:
        pre_tag += f", slot at doc {sp.group(1)}"
        n = n.replace(sp.group(0), "").replace("  ", " ").strip()
    if "gdn-nowrite" in n:
        pol = re.search(r"policy=([\w+]+)", n)
        pol_tag = f", keep policy {pol.group(1)}" if pol else ""
        return f"soft token, attention-only (no GDN write){pol_tag}{pre_tag}", keep, G, "no-bias", False
    if n.startswith("pooledKV"):
        bias = n.split()[-1]
        return "pooled K/V attention (all tokens kept, GDN intact)", keep, G, bias, True
    if "meanEmb" in n:
        bias = n.split("meanEmb")[1].strip() or "no-bias"
        return "soft token (mean embedding)", keep, G, bias, False
    if n.startswith("soft "):
        rest = re.sub(r"^soft k=[0-9.]+\s*", "", n)
        pol = re.search(r"policy=([\w+]+)", rest)
        if pol:
            return f"soft token, keep policy {pol.group(1)}{pre_tag}", keep, G, rest.replace(pol.group(0), "").strip() or "no-bias", False
        return f"soft token (mean embedding){pre_tag}", keep, G, rest, False
    return n, keep, G, "", False


def main():
    os.makedirs(OUT, exist_ok=True)
    out = []
    for source, task, rung, ckpt, kind, src in SOURCES:
        rows_done, table = parse_last_table(read_text(src))
        if not table:
            continue
        full = next((float(r["ce"]) for r in table if r["name"] == "full"), None)
        for r in table:
            cons, keep, G, bias, ceil = classify(r["name"])
            comp = float(r["comp"])
            out.append({"task": task, "rung": rung, "checkpoint": ckpt, "source": source, "probe": kind, "rows": rows_done,
                        "construction": cons, "keep": keep, "G": G, "bias": bias, "ceiling_only": int(ceil),
                        "answer_ce": float(r["ce"]), "full_ce": full, "delta_ce": (float(r["ce"]) - full) if full is not None else None,
                        "top1_agree": float(r["top1"]), "kl": float(r["kl"]), "exact_answer": float(r["correct"]),
                        "compaction": comp, "train_speedup_x": (1.0 / comp) if comp > 0 else None, "act_memory_saving_pct": 100 * (1 - comp),
                        "raw_name": r["name"]})
        print(f"{source:9} {task:13} {rung:3} {kind:8} rows={rows_done:2d} configs={len(table)}")
    cols = ["task", "rung", "checkpoint", "source", "probe", "rows", "construction", "keep", "G", "bias", "ceiling_only",
            "answer_ce", "full_ce", "delta_ce", "top1_agree", "kl", "exact_answer", "compaction", "train_speedup_x", "act_memory_saving_pct", "raw_name"]
    with open(f"{OUT}/slot_probe_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(out)
    print(f"wrote {OUT}/slot_probe_results.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
