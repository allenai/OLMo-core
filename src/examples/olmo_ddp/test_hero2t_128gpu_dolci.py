"""CPU checks for the pending hero's 128-GPU MT → LC → Dolci-only handoff."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import olmoe3_corrected_sft as worker
import olmoe3_corrected_sft_plan as sft
import olmoe3_dolci_hero_plan as hero

BASELINE = "444025b6de8013025c351d9c0015b760ef866217"


class HeroResizeTest(unittest.TestCase):
    """Exercise run selection, recipe preservation, rank gates and torchrun dispatch."""

    def test_only_full_dolci_descendant(self):
        future = [r for r in sft.runs() if r.future_parent]
        self.assertEqual(len(sft.runs()), 13)
        self.assertEqual([r.dataset for r in future], ["dolci-think"])
        r = future[0]
        self.assertEqual((r.gpus, r.nodes, r.source_gpus), (128, 16, 128))
        self.assertEqual((r.batch, r.microbatch, r.epochs, r.lr), (8388608, 65536, 2, 5e-5))
        self.assertEqual(r.batch // (r.gpus * r.microbatch), 1)
        for dataset in ("gptoss-medium", "gptoss-high"):
            with self.assertRaises(StopIteration):
                sft.find_run(f"{sft.CAMPAIGN}-2t3to1-non-emo-{dataset}")

    def test_mt_lc_batch_and_parent_chain(self):
        mt, lc = hero.run("hero2t-mt"), hero.run("hero2t-lc")
        for r in (mt, lc):
            self.assertEqual((r.gpus, r.nodes, r.batch, r.end), (128, 16, 16777216, 5961))
            self.assertEqual(r.end * r.batch, 100008984576)
        self.assertEqual(mt.batch // (mt.gpus * mt.microbatch), 4)
        self.assertEqual(lc.batch // (lc.gpus * lc.microbatch), 2)
        self.assertEqual(mt.source, hero.run("hero").root / "step120000")
        self.assertEqual(lc.source, mt.root / "step5961")
        self.assertEqual(
            next(r for r in sft.runs() if r.future_parent).source, lc.root / "step5961"
        )

    def test_completed_controls_are_unchanged(self):
        source = subprocess.check_output(
            ["git", "show", f"{BASELINE}:src/examples/olmo_ddp/olmoe3_corrected_sft_plan.py"],
            text=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline_sft.py"
            path.write_text(source)
            spec = importlib.util.spec_from_file_location("baseline_sft", path)
            old = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {spec.name: old}):
                spec.loader.exec_module(old)
                with (
                    patch.object(old.Run, "end", new_callable=PropertyMock, return_value=5402),
                    patch.object(sft.Run, "end", new_callable=PropertyMock, return_value=5402),
                ):
                    before = {r.run_id: r.as_dict() for r in old.runs() if not r.future_parent}
                    after = {r.run_id: r.as_dict() for r in sft.runs() if not r.future_parent}
                    self.assertEqual(len(before), 12)
                    self.assertEqual(before, after)

    def test_future_parent_requires_128_rank_receipt(self):
        r = next(r for r in sft.runs() if r.future_parent)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "step5961"
            source.mkdir()
            (source / ".metadata.json").write_text("{}")
            proof = source.parent / "audit/success.json"
            proof.parent.mkdir()
            with patch.object(sft.Run, "source", new_callable=PropertyMock, return_value=source):
                self.assertFalse(r.parent_ready())
                for gpus in (64, 128):
                    proof.write_text(
                        json.dumps({"passed": True, "smoke": False, "step": 5961, "gpus": gpus})
                    )
                    if gpus == 64:
                        with self.assertRaises(AssertionError):
                            r.parent_ready()
                    else:
                        self.assertTrue(r.parent_ready())

    def test_sft_dispatches_16_nodes_and_records_128_ranks(self):
        runtime = SimpleNamespace(verify_runtime=Mock())
        resolve = Mock(return_value=("leader-job", "leader-host"))
        topology = SimpleNamespace(validate_topology=Mock(return_value={"all_pairs_nvlink": True}))
        backend = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            r = SimpleNamespace(
                root=root / "run",
                nodes=16,
                gpus=128,
                batch=8388608,
                run_id="test-dolci",
                end=2,
                source=root / "parent",
            )
            r.root.mkdir()

            def train(command, **kwargs):
                self.assertIn("--nnodes=16", command)
                self.assertEqual(kwargs["env"]["QKGAIN_LOAD"], str(r.source))
                final = r.root / "step2"
                final.mkdir()
                (final / ".metadata.json").write_text("{}")

            env = {
                "BEAKER_REPLICA_COUNT": "16",
                "BEAKER_ASSIGNED_GPU_COUNT": "8",
                "BEAKER_REPLICA_RANK": "0",
                "BEAKER_EXPERIMENT_ID": "experiment",
                "BEAKER_JOB_ID": "job",
            }
            modules = {
                "olmoe3_hero_decay_runtime": runtime,
                "olmoe3_profile_node": SimpleNamespace(resolve_ready_leader=resolve),
                "olmoe3_profile_topology": topology,
            }
            with (
                patch.dict(os.environ, env),
                patch.dict(sys.modules, modules),
                patch.object(worker.p, "ROOT", root),
                patch("beaker.Beaker.from_env", return_value=backend),
                patch.object(worker.subprocess, "check_output", return_value="topology"),
                patch.object(worker.subprocess, "run", side_effect=train),
                patch.object(worker.p.base, "validate_checkpoint") as validate,
            ):
                worker.node(r)
                validate.assert_called_once_with(r.root / "step2", 2, 8388608, 128)
            proof = json.loads((r.root / "audit/success.json").read_text())
            self.assertEqual(proof["gpus"], 128)
            self.assertTrue(proof["passed"])
            self.assertEqual(resolve.call_args.args[-1], 16)

    def test_sft_rejects_eight_node_allocation_for_128_gpu_run(self):
        modules = {
            "olmoe3_hero_decay_runtime": SimpleNamespace(verify_runtime=Mock()),
            "olmoe3_profile_node": SimpleNamespace(resolve_ready_leader=Mock()),
            "olmoe3_profile_topology": SimpleNamespace(validate_topology=Mock()),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {"BEAKER_REPLICA_COUNT": "8"}),
            self.assertRaises(AssertionError),
        ):
            worker.node(SimpleNamespace(nodes=16))


if __name__ == "__main__":
    unittest.main()
