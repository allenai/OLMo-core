"""Tests for opslib.backfill on a tiny fake repo with one ledger of each weird shape."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from opslib import backfill, registry, results

B1, B2, B3 = "01M227SWE3DH9W66T54XT4HT7S", "01M227V5ZCS5TAMBGEPPGC1GJ5", "01M1GDT9PMJ7WEDNWPQ7A22GQJ"
WARN = "   warnings.warn(msg, UserWarning)\nSecret 'X' not configured in beaker workspace ai2/flex2"

FILES = {
    # headerless ds64 shape, quoted multi-line notes cell, comment lines, one relaunch of the same run
    "debug/ds64/LAUNCH_LEDGER.tsv": (
        f'ds64\t4b\tcontradiction\tdense\t16M\tds64-c-dense-u16M\tai2/jupiter-cirrascale-2\t2026-09-08 21:46\tLAUNCHED\t"{B1}{WARN}"\n'
        "# a comment line\n#   continued\n"
        f'ds64\t4b\tcontradiction\thdr03\t16M\tds64-c-hdr03-u16M\tai2/jupiter-cirrascale-2\t2026-09-08 21:47\tLAUNCHED\t"{B2}{WARN}"\n'
    ),
    "debug/ds64/orchestrator_ds64_state.json": json.dumps({"runs": {
        "ds64-c-dense-u16M": {"ex": B1, "state": "DONE", "eval": "E1"},
        "ds64-c-hdr03-u16M": {"ex": B2, "state": "FAILED"},
        "ds64-c-orphan-u16M": {"ex": B3, "state": "DONE"}},
        "evals": {"ds64-c-dense-u16M": {"ex": "01M228QFMFGPVA858VQDP92VTT", "state": "DONE"}}}),
    # header + junk job_id ('beaker35') with the real id in notes, note rows to skip, slurm rows
    "debug/flop_scaling/LAUNCH_LEDGER.tsv": (
        "job_id\ttask\tarm\tbudget\tstage\trun_name\tcluster\tlaunched\tstatus\tnotes\n"
        "3492749\tcontradiction\tffnmoe\tsmoke\t-\tfs-smoke-ffnmoe\tsneetches\t2026-09-02 07:40\tFAILED\trelaunched\n"
        f'beaker35\tnq\tkv17\t16M\t-\tfs35-nq-kv17-s16M\tai2/jupiter-cirrascale-2\t2026-09-01 23:45\tLAUNCHED\t"{B3}{WARN}"\n'
        "PIVOT\t-\t-\tnote\t-\tQwen3.5-4B track\t-\t2026-09-01 23:20\t-\tsome prose\n"
        "3493055\tnq\t-\tkvdata35\t-\tnq nmix -> *_mk\tmooney\t2026-09-01 23:24\tDONE\tdata job\n"
    ),
    # ctc_modelscale: headerless, widths 3, 4, 9 (lane), 9 (header layout), 10 (vessl)
    "debug/ctc_modelscale/LAUNCH_LEDGER.tsv": (
        "2026-08-12T09:47:26-07:00\tctcms-a-full-08b\tcontradiction\tfull\t0.8b\t40960\tlaneA\tpreemptive_high\t110071\n"
        "2026-08-12T16:44:00-07:00\tctcms-b-full-4b-vsl\toolong\tfull\t4b\t40960\tvessl\tbetelgeuse\tjob-kx083mcvmuw4\tMOVED\n"
        "2026-08-21T01:00:17-07:00\tctcms-c-full-2b\tcontradiction\tfull\t2b\t40960\tjupiter\t01M0HNBEDDA9MS0ZV21PPY76KA\tCANCELLED redundant\n"
        "2026-08-21T22:06:04-07:00\tctcms-d-cmix-08b\t01M0KXS4YX7V6HR1NGY0P2XGDQ\n"
        "2026-08-22T03:54:58-07:00\tctcfam-e-full\t01M0MHQZZPGHZMPM6MFG1MYHJC\tmoved-to-saturn\n"
    ),
    # header ledger with slurm- prefixed ids and status-only CANCELLED rows
    "debug/ctc_olmo_hybrid/LAUNCH_LEDGER.tsv": (
        "launched_at\trun_name\ttask\tarm\tscale\tseq_len\tcluster\texperiment_id\tnotes\n"
        "2026-08-17T14:54:33-07:00\tctc-olmo3-nq-local\tnq\tfull\t7b\t40960\tberkeleynlp-horton\tslurm-3459936\twave2\n"
        "2026-08-17T15:06:40-07:00\tctc-olmo3-nq-local(3459936)\t-\tCANCELLED\t-\t-\t-\t-\tswapped\n"
        "2026-08-17T15:06:40-07:00\tgroup-of-things\t-\tCANCELLED\t-\t-\t-\t-\tnot a run\n"
    ),
    # eval ledger keyed by checkpoint + rung
    "debug/ctc_crossfamily/launches/LAUNCH_LEDGER.tsv": (
        "launched_at\tckpt\ttask\tladder\tarm\trung\texperiment_id\n"
        "2026-08-15T02:02:18-07:00\tctc-olmo3-contra-full-20260727T020006-0700\tcontradiction\tc_iid\tfull\t2560\t01M02AGQDM9F29B8MZJD6PE21H\n"
    ),
    # softdetach headerless EVAL ledger (comment header, 5 and 6 columns)
    "src/scripts/train/memexpress/cpt/softdetach/EVAL_LEDGER.tsv": (
        "# dev set ... columns: EVAL<TAB>run...\n"
        "EVAL\tsdcpt-dense-u32M\tdense\t01M33QYAEANM8PF41CZKYA5DST\t2026-09-21 22:03\n"
        "CTC\tsdcpt-dense-u16M\tdense\t01M33WAYQ9E46QYBMFCVR2DMA3\t2026-09-21 23:24\tcontra raw 2k-32k\n"
    ),
    "src/scripts/train/memexpress/cpt/softdetach/LAUNCH_LEDGER.tsv": (
        "run\tarm\tbudget\tcluster\tlaunched\tstate\tnote\n"
        f'sdcpt-dense-u32M\tdense\t32M\tai2/jupiter-cirrascale-2\t2026-09-21 21:44\tFAILED(SyntaxError)\t"01M33PKBRTTRE7FNS5EM0S27QQ{WARN}"\n'
    ),
    "results/ds64/harvest/runs/ds64-c-dense-u16M/provenance.json": json.dumps({"git_commit": "788aaeda"}),
    "results/ds64/results.csv": (
        "run,task,arm,f1_2k,f1_8k,eval_ex\nds64-c-dense-u16M,contradiction,dense,0.929,,E1\n"
    ),
    "results/flop_scaling/results35.csv": "task,arm,f1_2k,f1_8k,run\noutlier,dense,,0.505,prior-dense-outlier-16M\n",
    "results/flop_scaling/results_scale.csv": "task,arm,f1_8k,run\noutlier,dense,0.505,prior-dense-outlier-16M\n",
    "results/ctc_suite/all_results.jsonl": "\n".join(json.dumps({
        "task": "contradiction", "metric_name": "set_f1", "metric_value": v, "rung_tokens": 2048, "eval_size": 488,
        "cot_label": "no-cot", "provenance": {"ckpt_path": "/d/ckpts/ctc-t0", "git_commit": "e57", "eval_backend": "native"}})
        for v in (0.5, 0.95)) + "\n",
}
HUB = ("id,who_ran,record_added,eval_name,eval_version,eval_context_length,eval_data_quantity,metric_name,"
       "metric_value,weka_model_location,model_train_data_hparams,git_commit,other_notes,quality_flag\n"
       "a1,PrasannS,2026-07-01T08:28:46Z,contradiction,,2k,488,f1,0.001,,q4b-base-dense,f7ef9f94,,small_eval_set\n"
       "a2,amandab,2026-07-01T08:28:46Z,contradiction,,2k,488,f1,0.5,,someone-else,f7ef9f94,,\n"
       "a3,prasann,2026-07-02T08:28:46Z,nq,v3,8k,,f1,0.4,/weka/ckpts/q4b-x/step100,,,note,no_yarn_ge256k\n")


class BackfillTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.repo = root / "OLMo-core"
        for rel, text in FILES.items():
            p = self.repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        (root / "results-hub").mkdir()
        (root / "results-hub" / "results.csv").write_text(HUB)
        self.state = root / "state"
        self._env = os.environ.get("OPS_STATE_DIR")
        os.environ["OPS_STATE_DIR"] = str(self.state)

    def tearDown(self):
        if self._env is None:
            os.environ.pop("OPS_STATE_DIR", None)
        else:
            os.environ["OPS_STATE_DIR"] = self._env
        self.tmp.cleanup()

    def test_helpers(self):
        self.assertEqual(backfill.job_id_from("beaker35", f"{B3}   warnings"), B3)
        self.assertEqual(backfill.job_id_from("slurm-3459936"), "3459936")
        self.assertIsNone(backfill.job_id_from("beaker35", "-"))
        self.assertEqual(backfill.cluster_for("laneA preemptive_high", "110071"), "lambda")
        self.assertEqual(backfill.cluster_for("sneetches", "3492749"), "slurm")
        self.assertEqual(backfill.cluster_for("whatever", B1), "beaker")
        self.assertEqual(backfill.final_state_for("SIGSEGV@830"), "failed")
        self.assertEqual(backfill.final_state_for("CANCELLED@51"), "cancelled")
        self.assertIsNone(backfill.final_state_for("LAUNCHED"))
        self.assertEqual(backfill.experiment_for("debug/ctc_modelscale/gapfill/LAUNCH_LEDGER.tsv"), "ctc_modelscale")
        self.assertEqual(backfill.experiment_for("debug/new_thing/LAUNCH_LEDGER.tsv"), "new_thing")

    def test_ledgers(self):
        res = backfill.backfill_ledgers(repo=self.repo)
        runs = registry.load_runs()
        self.assertEqual(len(res["ledgers"]), 7)
        # ds64: ledger + orchestrator state + provenance
        dense = runs["ds64-c-dense-u16M"]
        self.assertEqual((dense["job_id"], dense["cluster"], dense["final_state"]), (B1, "beaker", "done"))
        self.assertEqual(dense["git_commit"], "788aaeda")
        self.assertEqual(dense["launched_at"], "2026-09-09T04:46:00Z")
        self.assertEqual(runs["ds64-c-hdr03-u16M"]["final_state"], "failed")
        self.assertEqual(runs["ds64-c-orphan-u16M"]["state_source"], "debug/ds64/orchestrator_ds64_state.json")
        self.assertEqual(runs["ds64-c-dense-u16M__eval"]["parent_run"], "ds64-c-dense-u16M")
        # flop_scaling: junk job id resolved from notes; note row skipped; data row kept under a synthetic id
        self.assertEqual(runs["fs35-nq-kv17-s16M"]["job_id"], B3)
        self.assertEqual(runs["fs-smoke-ffnmoe"]["final_state"], "failed")
        self.assertIn("flop_scaling-job-3493055", runs)
        self.assertFalse(any("PIVOT" in r or "Qwen3.5" in r for r in runs))
        # ctc_modelscale mixed widths
        self.assertEqual(runs["ctcms-a-full-08b"]["cluster"], "lambda")
        self.assertEqual(runs["ctcms-b-full-4b-vsl"]["job_id"], "job-kx083mcvmuw4")
        self.assertEqual(runs["ctcms-c-full-2b"]["final_state"], "cancelled")
        self.assertEqual(runs["ctcms-d-cmix-08b"]["job_id"], "01M0KXS4YX7V6HR1NGY0P2XGDQ")
        self.assertEqual(runs["ctcfam-e-full"]["notes"], "moved-to-saturn")
        # olmo_hybrid: status-only cancel applies to the known run only
        self.assertEqual(runs["ctc-olmo3-nq-local"]["final_state"], "cancelled")
        self.assertEqual(runs["ctc-olmo3-nq-local"]["job_id"], "3459936")
        self.assertNotIn("group-of-things", runs)
        # eval ledgers
        ev = runs["ctc-olmo3-contra-full__eval_contradiction_full_r2560"]
        self.assertEqual((ev["kind"], ev["parent_run"]), ("eval", "ctc-olmo3-contra-full"))
        self.assertEqual(runs["sdcpt-dense-u16M__ctc__01M33WAYQ9E46QYBMFCVR2DMA3"]["kind"], "eval")
        self.assertEqual(runs["sdcpt-dense-u32M"]["final_state"], "failed")
        for r in runs.values():
            self.assertIn(r.get("cluster"), registry.CLUSTERS, r)
            self.assertTrue(r.get("experiment"), r)

    def test_ledgers_idempotent_and_dry_run(self):
        dry = backfill.backfill_ledgers(repo=self.repo, dry_run=True)
        self.assertFalse(registry.runs_path().exists())
        first = backfill.backfill_ledgers(repo=self.repo)
        self.assertEqual(dry["events_new"], first["events_new"])
        n = len(registry.runs_path().read_text().splitlines())
        self.assertEqual(backfill.backfill_ledgers(repo=self.repo)["events_new"], 0)
        self.assertEqual(len(registry.runs_path().read_text().splitlines()), n)

    def test_results(self):
        res = backfill.backfill_results(repo=self.repo)
        rows = results.load()
        self.assertEqual(res["rows_new"], len(rows))
        by = {(r["run_id"], r["task"], str(r["rung"])): r for r in rows}
        ds = by[("ds64-c-dense-u16M", "contradiction", "2k")]
        self.assertEqual((ds["eval_job_id"], ds["eval_size"]), ("E1", None))
        self.assertIn("eval_size_missing", ds["flags"])
        self.assertIn("provenance_incomplete", ds["flags"])
        self.assertEqual(sum(r["run_id"] == "prior-dense-outlier-16M" for r in rows), 1)  # deduped across CSVs
        ctc = by[("ctc-t0", "contradiction", "2048")]
        self.assertEqual(ctc["value"], 0.95)  # last line wins, as in results_io
        self.assertIn("eval_size<500", ctc["flags"])
        hub = [r for r in rows if r["source_json"].startswith("results-hub:")]
        self.assertEqual(len(hub), 2)  # amandab filtered out
        self.assertEqual(by[("q4b-x", "nq@v3", "8k")]["flags"],
                         ["backfill", "eval_size_missing", "no_yarn_ge256k", "provenance_incomplete"])
        self.assertEqual(backfill.backfill_results(repo=self.repo)["rows_new"], 0)


if __name__ == "__main__":
    unittest.main()
