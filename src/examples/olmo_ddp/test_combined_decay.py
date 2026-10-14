"""CPU checks for schedule continuity, state transfer, reshard auditing and safe resumes."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import olmoe3_combined_decay_plan as p
import torch
from olmoe3_combined_decay_reshard import sample_hash, verify_halved_dp


class CombinedDecayTest(unittest.TestCase):
    """Exercise state transitions and failure cases using tiny synthetic native checkpoints."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [
            patch.object(p, "ROOT", self.root / "runs"),
            patch.object(p, "PROTECTED", self.root / "protected"),
            patch.object(p.fixed, "CHILD_ROOT", self.root / "fixed"),
        ]
        for item in self.patches:
            item.start()
        self.addCleanup(self.tmp.cleanup)
        for item in self.patches:
            self.addCleanup(item.stop)
        self.source = self.checkpoint(self.root / "source", 834466, 128)
        self.plan = p.make_plan(self.source, "test-combined")

    def checkpoint(self, path, step, gpus, child=False):
        """Make structural checkpoint metadata without deserializable model payloads."""
        for name in ["model_and_optim", "train", "resume_audit"]:
            (path / name).mkdir(parents=True, exist_ok=True)
        p.write_json(path / ".metadata.json", dict(ephemeral=False))
        (path / "model_and_optim/.metadata").touch()
        cfg = dict(
            run_name="source",
            model={"block": {"routed_experts_router": {"emo": {"on": True}}}},
            train_module=dict(
                optim=dict(lr=p.LR),
                scheduler={"_CLASS_": "olmo_core.optim.scheduler.ConstantWithWarmup"},
            ),
            data_loader=dict(global_batch_size=p.BATCH, seed=928543231),
            dataset={"mix": "PT"},
        )
        if child:
            cfg["run_name"] = self.plan["run_name"]
            cfg["train_module"]["scheduler"] = dict(
                _CLASS_="olmo_core.optim.scheduler.WSD", decay=self.plan["decay_steps"]
            )
            p.write_json(
                path / "combined-decay.json",
                dict(plan_sha256=p.digest(self.plan), phase=p.checkpoint_phase(self.plan, step)),
            )
        p.write_json(path / "config.json", cfg)
        for rank in range(gpus):
            p.write_json(
                path / "resume_audit" / f"rank{rank}.json",
                dict(step=step, tokens=step * p.BATCH, gpus=gpus, rank=rank),
            )
            (path / "train" / f"rank{rank}.pt").touch()
        return path

    def test_plan_and_continuous_schedule(self):
        p.validate_plan(self.plan)
        self.assertEqual((self.plan["source_gpus"], self.plan["gpus"]), (128, 64))
        self.assertEqual((self.plan["pt_steps"], self.plan["mt_steps"]), (35763, 5961))
        self.assertEqual((self.plan["switch_step"], self.plan["end_step"]), (870229, 876190))
        self.assertEqual(p.learning_rate(self.plan, 834466), p.LR)
        self.assertEqual(p.learning_rate(self.plan, 876190), 0)
        self.assertAlmostEqual(p.learning_rate(self.plan, 870229), p.LR / 7)
        points = [p.learning_rate(self.plan, s) for s in [870228, 870229, 870230]]
        self.assertGreater(points[0], points[1])
        self.assertGreater(points[1], points[2])
        self.assertFalse(Path(self.plan["save_folder"]).exists())

    def test_tampered_plan_rejected(self):
        for field, value in [
            ("gpus", 128),
            ("mt_steps", 6000),
            ("optimizer_reset_at_switch", True),
            ("warmup_at_switch", 2000),
            ("mt_mixture_sha256", "bad"),
        ]:
            with self.subTest(field=field):
                plan = dict(self.plan, **{field: value})
                with self.assertRaises(ValueError):
                    p.validate_plan(plan)

    def states(self):
        """Example PT trainer state and untouched MT loader state."""
        saved = dict(
            global_step=self.plan["switch_step"],
            global_train_tokens_seen=self.plan["switch_step"] * p.BATCH,
            global_train_petaflops=123.5,
            world_size=64,
            epoch=3,
            rng={"torch": [1, 2, 3]},
            callbacks={"checkpoint": {"step": 870229}},
            data_loader=dict(
                seed=self.plan["pt_data_seed"],
                tokens_processed=100,
                batches_processed=9,
                dataset_fingerprint="pt",
                epoch=3,
            ),
        )
        fresh = dict(
            seed=p.MT_SEED,
            tokens_processed=0,
            batches_processed=0,
            dataset_fingerprint="mt",
            dataset_fingerprint_version="v1",
            epoch=None,
        )
        return saved, fresh

    def test_boundary_changes_only_data_and_epoch(self):
        saved, fresh = self.states()
        before, fresh_before = copy.deepcopy(saved), copy.deepcopy(fresh)
        result = p.transition_state(self.plan, saved, fresh)
        self.assertEqual(saved, before)
        self.assertEqual(fresh, fresh_before)
        for key in saved.keys() - {"data_loader", "epoch"}:
            self.assertEqual(result[key], saved[key])
        self.assertEqual(result["epoch"], 1)
        self.assertEqual(result["data_loader"]["epoch"], 1)
        self.assertEqual(result["data_loader"]["tokens_processed"], 0)
        self.assertEqual(result["data_loader"]["seed"], p.MT_SEED)

    def test_boundary_cannot_repeat_after_first_mt_step(self):
        saved, fresh = self.states()
        for field, value in [
            ("global_step", 870230),
            ("world_size", 128),
            ("global_train_tokens_seen", 0),
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.transition_state(self.plan, dict(saved, **{field: value}), fresh)
        for key, value in [
            ("seed", 7),
            ("tokens_processed", p.BATCH),
            ("dataset_fingerprint", "pt"),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                p.transition_state(self.plan, saved, dict(fresh, **{key: value}))

    def test_resume_before_at_and_after_switch(self):
        root = Path(self.plan["save_folder"])
        self.assertEqual(p.select_checkpoint(self.plan), (834466, str(self.source)))
        for step in [834476, 870229, 870239]:
            path = self.checkpoint(root / f"step{step}", step, 64, child=True)
            self.assertEqual(p.select_checkpoint(self.plan), (step, str(path)))
        (root / "step875000-tmp").mkdir()
        (root / "step876190").mkdir()
        self.assertEqual(p.select_checkpoint(self.plan)[0], 870239)
        (root / "step870239/train/rank63.pt").unlink()
        with self.assertRaises(ValueError):
            p.select_checkpoint(self.plan)

    def test_wrong_checkpoint_identity_rejected(self):
        path = self.checkpoint(
            Path(self.plan["save_folder"]) / "step870229", 870229, 64, child=True
        )
        p.write_json(path / "combined-decay.json", dict(plan_sha256="wrong", phase="pt"))
        with self.assertRaises(ValueError):
            p.select_checkpoint(self.plan)

    def test_protection_is_idempotent_and_survives_source_unlink(self):
        path = self.checkpoint(
            Path(self.plan["save_folder"]) / "step870229", 870229, 64, child=True
        )
        p.protect_checkpoint(self.plan, 870229)
        p.protect_checkpoint(self.plan, 870229)
        target = Path(self.plan["protected_folder"]) / "step870229/olmo-core"
        self.assertTrue((target / "config.json").samefile(path / "config.json"))
        (path / "config.json").unlink()
        self.assertTrue((target / "config.json").exists())
        p.protect_checkpoint(self.plan, 870229)

    def reshard_case(self):
        """Build a tiny enlarged shard and its two independent old-rank audit records."""
        local = torch.arange(1024, dtype=torch.float32)
        replicated = torch.arange(16, dtype=torch.bfloat16)
        shard = SimpleNamespace(
            placements=[SimpleNamespace(is_shard=lambda dim: dim == 0)], to_local=lambda: local
        )
        trainer = SimpleNamespace(
            train_module=SimpleNamespace(optim=SimpleNamespace(states={"moment": shard}))
        )
        actual = dict(
            step=834466,
            tokens=834466 * p.BATCH,
            rank=17,
            gpus=64,
            loss_history=[1.2],
            norm_history=[0.3],
            tensors={"moment": sample_hash(local), "model": sample_hash(replicated)},
        )
        for half, rank in enumerate([34, 35]):
            saved = dict(
                actual,
                rank=rank,
                gpus=128,
                tensors={
                    "moment": sample_hash(local[half * 512 : (half + 1) * 512]),
                    "model": sample_hash(replicated),
                },
            )
            p.write_json(self.root / "resume_audit" / f"rank{rank}.json", saved)
        return trainer, actual, local

    def test_halving_verifies_both_source_shards(self):
        trainer, actual, _ = self.reshard_case()
        proof = verify_halved_dp(trainer, self.root, actual)
        self.assertEqual(proof["source_ranks"], [34, 35])
        self.assertEqual(proof["old_optimizer_shards_verified"], 2)

    def test_halving_rejects_corruption_in_either_half(self):
        for index in [0, 512]:
            with self.subTest(index=index):
                trainer, actual, local = self.reshard_case()
                local[index] += 1
                with self.assertRaises(AssertionError):
                    verify_halved_dp(trainer, self.root, actual)


if __name__ == "__main__":
    unittest.main()
