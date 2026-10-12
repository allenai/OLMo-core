"""CPU-only checks: python -m unittest discover -s src/examples/olmo_ddp -p test_hero_mt_plan.py."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import olmoe3_hero_mt_plan as plan

PRIOR_MT20 = (
    plan.MOUNT
    / "production-hero-small-midtrain/olmo35-small-2t-mt20-20260912"
    / "olmo35-small-2t-mt20-20260912-emo/audit/session-01M2BASTVCD4P7F4VX7FD0SS5N-0.json"
)
PRIOR_SOURCE = (
    plan.MOUNT
    / "production-fixed-pt-decays/olmo35-small-12t-plus400bdecay-20261006-emo"
    / "step739342/config.json"
)
MIX = (
    Path(__file__).resolve().parents[2]
    / "olmo_core/data/source_mixtures/OLMo3-32B-midtraining-modelnamefilter.yaml"
)

try:
    from olmo_core.optim.scheduler import CosWithWarmup
except ImportError:
    CosWithWarmup = None


class HeroMTPlanTest(unittest.TestCase):
    """Pin the mt20 recipe, the isolated child identity and the checkpoint gate."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def checkpoint(self, path, step, gpus=64):
        """Write synthetic structural metadata, without model tensors or pickle."""
        for directory in [path / "model_and_optim", path / "train", path / "resume_audit"]:
            directory.mkdir(parents=True, exist_ok=True)
        (path / ".metadata.json").write_text("{}")
        (path / "model_and_optim/.metadata").write_text("metadata")
        for rank in range(gpus):
            row = dict(step=step, tokens=step * plan.BATCH, gpus=gpus, rank=rank)
            (path / f"resume_audit/rank{rank}.json").write_text(json.dumps(row))
            (path / f"train/rank{rank}.pt").write_bytes(b"trainer")
        return path

    def test_recipe_totals(self):
        self.assertEqual((plan.END, plan.END * plan.BATCH), (5961, 100_008_984_576))
        self.assertEqual((plan.LR, plan.WARMUP, plan.SMOKE_END), (1.1e-3 * 0.2, 2000, 2))
        self.assertEqual(plan.SEED, 1_387_822_106)
        run = plan.find_run("olmo35-small-12p4t-mt20-20261012-emo")
        self.assertEqual(run.as_dict()["parent_tokens"], 12_398_362_624_000)
        self.assertEqual(run.as_dict()["total_seen_tokens"], 12_498_371_608_576)
        self.assertEqual((run.as_dict()["source_gpus"], run.as_dict()["gpus"]), (128, 64))

    @unittest.skipUnless(PRIOR_MT20.is_file(), "2T mt20 EMO session record is not mounted")
    def test_recipe_matches_2t_mt20_emo_run(self):
        prior = json.loads(PRIOR_MT20.read_text())
        current = plan.runs()[0].as_dict()
        for key in (
            "batch_tokens",
            "gpus",
            "nodes",
            "lr",
            "warmup",
            "schedule",
            "requested_mt_tokens",
            "mt_steps",
            "mt_tokens",
            "optimizer_reset_at_mt_start",
            "data_seed",
            "mixture_sha256",
        ):
            with self.subTest(key=key):
                self.assertEqual(current[key], prior[key])
        self.assertEqual((prior["arm"], current["arm"]), ("emo", "emo"))

    def test_isolated_child_identity(self):
        (run,) = plan.runs()
        self.assertTrue(run.emo)
        self.assertEqual(run.root, plan.ROOT / run.run_id)
        self.assertEqual(run.prefix, "mt20-after-12p4t/emo")
        self.assertTrue(plan.SOURCE.is_relative_to(plan.MOUNT / "protected"))
        self.assertFalse(run.root.is_relative_to(plan.MOUNT / "protected"))
        registration = run.registration()
        self.assertEqual(
            {k: registration[k] for k in ("run_id", "lineage_id", "checkpoint_root")},
            dict(run_id=run.run_id, lineage_id=run.run_id, checkpoint_root=str(run.root)),
        )
        self.assertEqual(registration["deletion_mode"], "apply")
        self.assertGreaterEqual(registration["min_local_checkpoints"], 2)
        self.assertGreaterEqual(registration["delete_grace_seconds"], 3600)
        with self.assertRaises(StopIteration):
            plan.find_run("olmo35-small-12p4t-mt20-20261012-non-emo")

    def test_checkpoint_gate(self):
        good = self.checkpoint(self.root / "step2", 2)
        plan.validate_checkpoint(good, 2)
        with self.assertRaisesRegex(ValueError, "Wrong checkpoint audit"):
            plan.validate_checkpoint(good, 500)
        (good / "train/rank63.pt").unlink()
        with self.assertRaisesRegex(ValueError, "rank63"):
            plan.validate_checkpoint(good, 2)
        alias = self.root / "alias"
        alias.symlink_to(self.checkpoint(self.root / "step500", 500), target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "real checkpoint"):
            plan.validate_checkpoint(alias, 500)
        ephemeral = self.checkpoint(self.root / "step1000", 1000)
        (ephemeral / ".metadata.json").write_text(json.dumps({"ephemeral": True}))
        with self.assertRaisesRegex(ValueError, "complete"):
            plan.validate_checkpoint(ephemeral, 1000)
        wide = self.checkpoint(self.root / "step1500", 1500, gpus=128)
        with self.assertRaisesRegex(ValueError, "Wrong checkpoint audit"):
            plan.validate_checkpoint(wide, 1500)
        plan.validate_checkpoint(wide, 1500, plan.SOURCE_GPUS)
        (wide / "train/rank127.pt").unlink()
        with self.assertRaisesRegex(ValueError, "rank127"):
            plan.validate_checkpoint(wide, 1500, plan.SOURCE_GPUS)
        with self.assertRaisesRegex(ValueError, "Wrong checkpoint audit"):
            plan.validate_checkpoint(good, 2, plan.SOURCE_GPUS)

    @unittest.skipUnless(plan.SOURCE.is_dir(), "protected 12.4T hero checkpoint is not mounted")
    def test_source_is_the_non_decayed_128_gpu_hero_checkpoint(self):
        alias = plan.MOUNT / "protected/olmo35-small-hero-20260907/emo/12.4T"
        self.assertEqual(alias.resolve(), plan.SOURCE)
        plan.validate_checkpoint(plan.SOURCE, plan.PT_STEP, plan.SOURCE_GPUS)
        self.assertEqual(plan.SOURCE_GPUS, 128)
        config = json.loads((plan.SOURCE / "config.json").read_text())
        self.assertEqual(config["run_name"], plan.PARENT)
        scheduler = config["train_module"]["scheduler"]
        self.assertTrue(scheduler["_CLASS_"].endswith(".ConstantWithWarmup"))
        self.assertEqual(config["train_module"]["optim"]["lr"], 1.1e-3)
        self.assertEqual(config["data_loader"]["global_batch_size"], plan.BATCH)
        model = config["model"]
        routers = [
            block["routed_experts_router"]
            for block in [model["block"], *model.get("block_overrides", {}).values()]
            if block.get("routed_experts_router") is not None
        ]
        self.assertTrue(routers)
        self.assertTrue(all(router.get("emo") for router in routers))
        if PRIOR_SOURCE.is_file():
            self.assertEqual(model, json.loads(PRIOR_SOURCE.read_text())["model"])

    def test_eval_pins_match_2t_mt20(self):
        self.assertEqual(plan.EVAL_ROOT, plan.MOUNT / "scratch" / plan.CAMPAIGN)
        self.assertEqual(plan.CORE_REF, "b1fd2c9746e88baeb20e372bdca340d788d0f7e5")
        self.assertEqual(plan.HELPER_REF, "3542d99ebeb6d156887ed1eaf2871cff5f07666d")

    @unittest.skipUnless(plan.MOUNT.is_mount(), "verified_copy refuses non-Weka storage")
    def test_verified_copy(self):
        source = self.checkpoint(self.root / "src/step2", 2)
        (source / "model_and_optim/__0_0.distcp").write_bytes(b"weights")
        destination = self.root / "eval/step2/olmo-core"
        receipt = plan.verified_copy(source, destination, 2)
        self.assertTrue(receipt["all_file_hashes_verified"])
        self.assertFalse(destination.with_name("olmo-core.partial").exists())
        self.assertEqual((destination / "model_and_optim/__0_0.distcp").read_bytes(), b"weights")
        self.assertEqual(plan.verified_copy(source, destination, 2), receipt)
        with self.assertRaises(AssertionError):
            plan.verified_copy(source, destination, 500)

    @unittest.skipUnless(
        (plan.ROOT / "olmo35-small-12p4t-mt20-20261012-emo" / f"step{plan.END}").is_dir(),
        "final MT checkpoint is not mounted",
    )
    def test_final_checkpoint_is_ready(self):
        self.assertTrue(plan.ready(plan.runs()[0], plan.END))

    def test_mixture_is_the_mt20_mixture(self):
        self.assertEqual(hashlib.sha256(MIX.read_bytes()).hexdigest(), plan.MIX_SHA256)

    @unittest.skipIf(CosWithWarmup is None, "olmo_core is not importable")
    def test_cosine_schedule(self):
        schedule = CosWithWarmup(warmup=plan.WARMUP, alpha_f=0.0)
        self.assertEqual(schedule.get_lr(plan.LR, 0, plan.END), 0.0)
        self.assertEqual(schedule.get_lr(plan.LR, plan.WARMUP, plan.END), plan.LR)
        self.assertEqual(schedule.get_lr(plan.LR, plan.END, plan.END), 0.0)
        middle = plan.WARMUP + (plan.END - plan.WARMUP) // 2
        self.assertAlmostEqual(schedule.get_lr(plan.LR, middle, plan.END), plan.LR / 2, places=6)


if __name__ == "__main__":
    unittest.main()
