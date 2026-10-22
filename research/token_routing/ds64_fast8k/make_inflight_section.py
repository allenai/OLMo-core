"""'In flight' section of the tokdrop_lr artifact: every training run of tokdrop_lr / learned_router that is queued or
running right now (ops registry joined to the live `ops status` snapshot), grouped by what it tests.

    python research/token_routing/ds64_fast8k/make_inflight_section.py --page PAGE.html
"""

import argparse
import collections
import html
import json
import re
import subprocess

OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"


def what(rid):
    if "densepf" in rid:
        return "dense through the soft-token path (pad-free wall-clock reference)"
    if re.search(r"-dense-(s|1|sk|ld)", rid) or re.search(r"-dense-\d", rid):
        if re.search(r"-dense-(sk|ld)", rid):
            return "dense + train-time layer skip"
        return "dense curve (LR %s)" % re.search(r"lr([0-9.e-]+)", rid).group(1)
    m = re.search(r"-(otr|rnd|fl|dh)(\d+)([ML])(\d+)(\w*?)-", rid)
    if m:
        kind = {"otr": "online router", "rnd": "random", "fl": "first/last", "dh": "hidden-state-change rule"}[m.group(1)]
        where = "split after block %s" % m.group(4) if m.group(3) == "M" else "input drop (features from block %s)" % m.group(4)
        tag = {"x00": ", exploration 0", "x30": ", exploration 0.3", "x50": ", exploration 0.5",
               "de16": ", full-context step every 16", "de0": ", no full-context steps", "sd1": ", seed 1",
               "dh": ", target = hidden-state change", "mix": ", target = gnorm/dh rank mix",
               "gxh": ", target = grad x activation"}.get(m.group(5), (", " + m.group(5)) if m.group(5) else "")
        return f"{kind}, keep {int(m.group(2))}%, {where}{tag}"
    if re.search(r"-(sk3|sk8|ldl20)-", rid):
        return "fixed rule + train-time layer skip"
    if re.search(r"-(chk16|chk64|kv33|flk64|ck64ds|ohdr33)-", rid):
        return "ds64 per-task winning rule (verbatim recipe)"
    return "fixed keep rule"


def mix_of(rid):
    if "mt11" in rid:
        return "11-task"
    if re.match(r"f8k-(contradiction|nq|outlier|oolong)-", rid):
        return "single task (" + rid.split("-")[1] + ")"
    return "4-task"


def build():
    runs = []
    for exp in ("tokdrop_lr", "learned_router"):
        out = subprocess.run([OPS, "runs", "--experiment", exp, "--json"], capture_output=True, text=True).stdout
        runs += json.loads(out) if out.strip().startswith("[") else []
    st = subprocess.run([OPS, "status"], capture_output=True, text=True).stdout
    live = {}
    for line in st.splitlines():
        f = line.split()
        if len(f) > 4 and f[4] in ("PENDING", "RUNNING"):
            live[f[0].rstrip("…")] = f[4]

    def state(rid):
        for k, v in live.items():
            if rid == k or rid.startswith(k):
                return v
        return None

    groups = collections.OrderedDict()
    for r in sorted(runs, key=lambda r: r["run_id"]):
        if r.get("kind") != "train" or r.get("final_state"):
            continue
        sv = state(r["run_id"])
        if not sv:
            continue
        groups.setdefault((what(r["run_id"]), mix_of(r["run_id"])), []).append((r["run_id"], sv))
    h = '<!--INFLIGHT-BEGIN-->\n<section id="inflight">\n  <h2>In flight (queued or training)</h2>\n'
    h += ('  <p>Every training run of this line of work that has not finished, from the run registry (tokdrop_lr and '
          'learned_router) joined to the live job snapshot. Results move into the tables and plots above as evals land.</p>\n')
    h += '  <div class="tbl"><table>\n    <tr><th>what</th><th>mix</th><th class="n">runs</th><th>run names</th></tr>\n'
    n = 0
    for (w, mix), lst in groups.items():
        n += len(lst)
        names = "<br>".join(f'<span class="mono">{html.escape(r)}</span> <span class="chip run">{s.lower()}</span>' for r, s in lst)
        h += f'    <tr><td>{w}</td><td>{mix}</td><td class="n">{len(lst)}</td><td>{names}</td></tr>\n'
    h += '  </table></div>\n</section>\n<!--INFLIGHT-END-->'
    return h, n, len(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True)
    a = ap.parse_args()
    h, n, g = build()
    s = open(a.page).read()
    if "<!--INFLIGHT-BEGIN-->" in s:
        s = re.sub(r"<!--INFLIGHT-BEGIN-->.*?<!--INFLIGHT-END-->", lambda m: h, s, flags=re.S)
    else:
        s = s.replace('<section id="predictors"', h + '\n\n<section id="predictors"', 1)
    open(a.page, "w").write(s)
    print(f"{n} in-flight runs in {g} groups")


if __name__ == "__main__":
    main()
