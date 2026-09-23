"""
Freeze olmo-eval's CTC ROSTER (the authority on which rungs exist, which spec grades each row and
where the scorer reads gold) into ``olmo_eval_roster.json`` for the IID audit.

Parsed with ``ast`` rather than imported: importing olmo_eval pulls in every task module, and one
unrelated huggingface_hub incompatibility breaks the whole package. The spec extras come from the
vendored ctc, which imports standalone and is the code that grades.

    python debug/ctc_sft_setA/dump_olmo_eval_roster.py --olmo-eval ../olmo-eval
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--olmo-eval", required=True, help="olmo-eval repo root")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "olmo_eval_roster.json"))
    args = ap.parse_args()

    pkg = os.path.join(args.olmo_eval, "src/olmo_eval/evals/tasks/ctc_suite")
    tree = ast.parse(open(os.path.join(pkg, "__init__.py")).read())
    env: dict = {}
    roster_node = None
    for node in tree.body:
        tgt = node.targets[0] if isinstance(node, ast.Assign) else (
            node.target if isinstance(node, ast.AnnAssign) else None)
        if not isinstance(tgt, ast.Name):
            continue
        if tgt.id == "ROSTER":
            roster_node = node.value
            continue
        # every other module-level constant the rows may reference (_LADDER_*, _SUB500_*, ...),
        # evaluated in order against the ones before it; anything non-constant is skipped
        try:
            env[tgt.id] = eval(compile(ast.Expression(node.value), "<roster>", "eval"),  # noqa: S307
                               {"__builtins__": {"dict": dict, "tuple": tuple, "list": list, "range": range}}, env)
        except Exception:
            pass

    safe = {"__builtins__": {"dict": dict, "tuple": tuple, "list": list, "range": range}}

    def ev(n):
        # Row fields reference module constants (_LADDER_*, _SUB500_*) and comprehensions over them.
        return eval(compile(ast.Expression(n), "<roster>", "eval"), safe, env)  # noqa: S307

    sys.path.insert(0, os.path.join(pkg, "_vendor"))
    from ctc.format import registry
    import ctc.tasks as T

    T.load_all()
    out = {}
    for k, v in zip(roster_node.keys, roster_node.values):
        kw = {a.arg: a.value for a in v.keywords}
        spec = ev(kw["spec"])
        row = {"subset": ev(kw["subset"]), "spec": spec,
               "rungs": list(ev(kw["rungs"])) if "rungs" in kw else list(env["_LADDER_2K_32K"]),
               "rung_alias": ev(kw["rung_alias"]) if "rung_alias" in kw else {},
               "eval_size": ev(kw["eval_size"]) if "eval_size" in kw else {}}
        try:
            sp = registry.get(spec)
            row.update(spec_registered=True,
                       gold_field=sp.extra.get("gold_field", "gold_doc_indices"),
                       score_takes_example=bool(sp.extra.get("score_takes_example")))
        except KeyError:
            row.update(spec_registered=False)
        out[ast.literal_eval(k)] = row
    commit = subprocess.run(["git", "-C", args.olmo_eval, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    json.dump({"hf_dataset": env["HF_DATASET"], "rung_tokens": env["RUNG_TOKENS"],
               "olmo_eval_commit": commit, "roster": out}, open(args.out, "w"), indent=1)
    for k, r in out.items():
        print(f"{k:<20} {r['subset']:<18} {r['spec']:<14} {r['rungs'][0]}..{r['rungs'][-1]:<6}"
              f" registered={r['spec_registered']} gold={r.get('gold_field')}"
              f"{' (whole example)' if r.get('score_takes_example') else ''}")
    print(f"wrote {args.out} (olmo-eval {commit})")


if __name__ == "__main__":
    main()
