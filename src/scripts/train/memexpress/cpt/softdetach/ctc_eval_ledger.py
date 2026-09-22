"""Write the run-evals ledger (records/eval_launches/<date>_sdcpt-q35-4b_ctcbench.yaml) from the
wave TSV (run, task, job name, experiment id) and collect results from the Beaker logs.

    python .../ctc_eval_ledger.py ledger <wave.tsv>     # ledger YAML + README row + EVAL_LEDGER rows
    python .../ctc_eval_ledger.py collect <wave.tsv>    # f1 per rung from `[ladder:task@rung] f1=` lines

Base/CPT checkpoints scored with --prompt-format raw on the 2k-32k figure rungs only (Prasann,
2026-09-21: downstream ranking vs dev loss; no xlong / YaRN / OOD ladders for raw CPT models).
"""
import json, os, re, subprocess, sys, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../../../.."))
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
DATE = "2026-09-21"
LEDGER = f"{REPO}/records/eval_launches/{DATE}_sdcpt-q35-4b_ctcbench.yaml"
CKPT = lambda run: f"{W}/ctc_suite/bases/q35-4b-base-markerfix" if run.startswith("q35-4b-base") else f"{W}/ctc_suite/ckpts/{run}"
ARM = lambda run: "base" if run.startswith("q35-4b-base") else run.split("-")[3]
BUDGET = lambda run: "0M" if run.startswith("q35-4b-base") else run.rsplit("-u", 1)[1]

def rows(tsv):
    out = []
    for line in open(tsv):
        p = line.rstrip("\n").split("\t")
        if len(p) == 4:
            out.append(dict(run=p[0], task=p[1].strip("[]"), job=p[2], ex=p[3]))
    return out

def ledger(tsv):
    rs = rows(tsv)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    runs = sorted({r["run"] for r in rs}, key=lambda x: (ARM(x), int(BUDGET(x)[:-1])))
    y = [f"# {DATE} downstream CTC-bench on the soft-detach CPT arms (raw prompt format, 2k-32k only).",
         "# Rules 1-3 of run-evals (xlong, YaRN, OOD ladders) deliberately NOT applied: these are raw CPT",
         "# checkpoints, the question is the ranking across arms at matched FLOPs vs dev loss.",
         "checkpoints:"]
    for run in runs:
        y += [f"  - run_name: {run}", f"    ckpt: {CKPT(run)}", "    model_type: dense",
              f"    model_subtype: {ARM(run)}-u{BUDGET(run)}",
              f"    attention_type: {'full' if ARM(run) in ('dense', 'base') else 'full (trained with pooled soft tokens, ' + ARM(run) + ')'}",
              f"    pipeline_stage: {'base' if ARM(run) == 'base' else 'CPT'}", "    chat_template: raw",
              f"    model_slug: sdcpt-q35-4b-{ARM(run)}-{BUDGET(run)}",
              "    training_description: >", "      Qwen3.5-4B marker-repaired base, CPT on dolma3_longmino 64k rows"
              f" (8 rows/step), arm {ARM(run)} at {BUDGET(run)} tokens; see records/softdetach-cpt-plan.md"]
    y += ["common:", "  who_ran: <git user>", f"  date_eval_ran: {DATE}", "  eval_version: v2",
          f"  eval_set_weka_pointer: {W}/_eval_bundle_eval500_v2_clean", f"  git_commit: {head}",
          "  cluster: ai2/jupiter-cirrascale-2", "  priority: urgent", "  tokenizer: inferred from checkpoint config (Qwen3.5)",
          "  prompt_format: raw", "  query_position: both", "  landmark_top_k_fixed_val: \"\"",
          "passes:", "  - name: base-figure", "    eval_tag: base", "    tasks: [contra, nq, oolong]",
          "    rungs_per_task: {contra: [2k, 8k, 16k, 32k], nq: [3k, 8k, 16k, 32k], oolong: [8k, 16k, 32k]}",
          "    yarn_factor: null",
          "    decoding_hparams_other: \"temp=0.0 top_p=1.0 max_length=40960 batch_size=2 ngpu=2 max_test=500 prompt_format=raw\"",
          "    eval_command: \"RUN=<run> TASK=<task> VARIANT=dense CKPT=<ckpt> EVAL_OUT_DIR=" + W + "/softdetach_cpt/ctc_eval/<run> PROMPT_FORMAT='raw' QUERY_POSITION='both' MAX_TEST=500 MAX_LENGTH=40960 BATCH_SIZE=2 NGPU=2 LADDER_VERSION=v2 bash src/scripts/train/memexpress/singletask_ladder/run_beaker_multirung_eval.sh\"",
          "    jobs:"]
    for r in rs:
        y += [f"      - run: {r['run']}", f"        task: {r['task']}", f"        beaker_experiment_id: {r['ex']}",
              f"        beaker_job_name: {r['job']}", "        status: submitted", "        result_json: \"\"",
              "        rungs_ingested: []", "        pulled_at: \"\"", "        notes: \"\""]
    open(LEDGER, "w").write("\n".join(y) + "\n")
    readme = f"{REPO}/records/eval_launches/README.md"
    row = (f"| {DATE} | sdcpt-q35-4b (base, dense 16M-128M, sd20 64M/128M, sfl20 64M) | run-root model_and_optim | "
           f"CTC-bench contra/nq/oolong 2k-32k, raw prompt, {len(rs)} jobs (submitted) | [yaml]({os.path.basename(LEDGER)}) |\n")
    if os.path.basename(LEDGER) not in open(readme).read():
        open(readme, "a").write(row)
    ev = f"{HERE}/EVAL_LEDGER.tsv"
    with open(ev, "a") as f:
        for r in rs:
            f.write(f"CTC\t{r['run']}\t{ARM(r['run'])}\t{r['ex']}\t{datetime.datetime.now():%Y-%m-%d %H:%M}\t{r['task']} raw 2k-32k\n")
    print(f"ledger {LEDGER}: {len(rs)} jobs; README row + EVAL_LEDGER rows appended")

def collect(tsv):
    pat = re.compile(r"\[ladder:(\w+)@(\w+)\] f1=([0-9.]+) \(n=(\d+)")
    out = []
    for r in rows(tsv):
        log = subprocess.run(["beaker", "experiment", "logs", r["ex"]], capture_output=True, text=True, timeout=120).stdout
        state = "FAILED" if "Traceback" in log else ("done" if "=== done" in log or "ALL DONE" in log else "running")
        for m in pat.finditer(log):
            out.append(dict(run=r["run"], arm=ARM(r["run"]), budget=BUDGET(r["run"]), task=m.group(1), rung=m.group(2),
                            f1=float(m.group(3)), eval_size=int(m.group(4)), ex=r["ex"], state=state))
        if not pat.search(log):
            out.append(dict(run=r["run"], arm=ARM(r["run"]), budget=BUDGET(r["run"]), task=r["task"], rung=None, f1=None,
                            eval_size=None, ex=r["ex"], state=state))
    json.dump(out, open(f"{HERE}/ctc_eval_results.json", "w"), indent=1)
    for o in out:
        print(f"{o['arm']:7s} {o['budget']:5s} {o['task']:7s} {str(o['rung']):4s} f1={o['f1']} n={o['eval_size']} {o['state']}")

if __name__ == "__main__":
    {"ledger": ledger, "collect": collect}[sys.argv[1]](sys.argv[2])
