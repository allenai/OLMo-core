"""Small local tests proving raw-copy cleanup gates fail closed."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import hero_hf_cleanup as subject


class CleanupChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hero-cleanup-test-")
        self.addCleanup(self.temp.cleanup)
        self.scratch = Path(self.temp.name) / "scratch"
        self.root = self.scratch / "emo" / "step6000"
        self.raw, self.hf = self.root / "olmo-core", self.root / "hf"
        self.raw.mkdir(parents=True)
        self.hf.mkdir()
        (self.raw / "payload").write_bytes(b"raw")
        (self.hf / "model.safetensors").write_bytes(b"converted")
        self.original = Path(self.temp.name) / "training-original"
        self.original.write_bytes(b"original")
        self.patches = [
            patch.object(subject, "SCRATCH", self.scratch),
            patch.object(subject, "prepare_scratch", lambda: None),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.record(
            "download-success.json",
            {"step": 6000, "raw_path": str(self.raw), "files": 1, "bytes": 3},
        )
        conversion = {
            "passed": True,
            "arm": "emo",
            "step": 6000,
            "output": str(self.hf),
            "output_sha256": {"model.safetensors": hashlib.sha256(b"converted").hexdigest()},
        }
        self.record("conversion-success.json", conversion)
        self.record("hf/_HERO_CONVERSION_SUCCESS.json", conversion)
        conversion_hash = hashlib.sha256(
            (self.hf / "_HERO_CONVERSION_SUCCESS.json").read_bytes()
        ).hexdigest()
        self.record(
            "vllm-parity-success.json",
            {
                "passed": True,
                "model": str(self.hf),
                "conversion_sha256": conversion_hash,
            },
        )
        self.record("metrics.json", {"tasks": ["toy"]})
        self.record(
            "eval-smoke-success.json",
            {
                "passed": True,
                "conversion_sha256": conversion_hash,
                "parity_sha256": hashlib.sha256(
                    (self.root / "vllm-parity-success.json").read_bytes()
                ).hexdigest(),
                "model": str(self.hf),
                "metrics": str(self.root / "metrics.json"),
                "metrics_sha256": hashlib.sha256(
                    (self.root / "metrics.json").read_bytes()
                ).hexdigest(),
            },
        )

    def record(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def test_accepts_only_raw_copy(self):
        result = subject.cleanup("emo", 6000)
        self.assertTrue(result["passed"])
        self.assertFalse(self.raw.exists())
        self.assertTrue((self.hf / "model.safetensors").is_file())
        self.assertEqual(self.original.read_bytes(), b"original")

    def test_missing_eval_refuses(self):
        (self.root / "eval-smoke-success.json").unlink()
        with self.assertRaises(RuntimeError):
            subject.cleanup("emo", 6000)
        self.assertTrue(self.raw.exists())

    def test_stale_receipt_refuses(self):
        conversion = json.loads((self.root / "conversion-success.json").read_text())
        conversion["revision"] = "new revision not yet evaluated"
        self.record("conversion-success.json", conversion)
        self.record("hf/_HERO_CONVERSION_SUCCESS.json", conversion)
        with self.assertRaises(RuntimeError):
            subject.cleanup("emo", 6000)
        self.assertTrue(self.raw.exists())

    def test_diagnostic_receipt_refuses(self):
        smoke = json.loads((self.root / "eval-smoke-success.json").read_text())
        smoke["diagnostic_only"] = True
        self.record("eval-smoke-success.json", smoke)
        with self.assertRaises(RuntimeError):
            subject.cleanup("emo", 6000)
        self.assertTrue(self.raw.exists())

    def test_modified_output_refuses(self):
        (self.hf / "model.safetensors").write_bytes(b"tampered")
        with self.assertRaises(RuntimeError):
            subject.cleanup("emo", 6000)
        self.assertTrue(self.raw.exists())

    def test_mismatched_precision_refuses(self):
        smoke = json.loads((self.root / "eval-smoke-success.json").read_text())
        smoke["precise"] = True
        self.record("eval-smoke-success.json", smoke)
        with self.assertRaisesRegex(RuntimeError, "precision profiles disagree"):
            subject.cleanup("emo", 6000)
        self.assertTrue(self.raw.exists())

    def test_unknown_precision_refuses(self):
        conversion = json.loads((self.root / "conversion-success.json").read_text())
        conversion["inference_profile"] = "unqualified_recipe"
        self.record("conversion-success.json", conversion)
        with self.assertRaisesRegex(RuntimeError, "Unknown inference precision"):
            subject.cleanup("emo", 6000)
        self.assertTrue(self.raw.exists())

    def test_link_to_original_refuses(self):
        (self.raw / "payload").unlink()
        (self.raw / "payload").symlink_to(self.original)
        with self.assertRaises(RuntimeError):
            subject.cleanup("emo", 6000)
        self.assertEqual(self.original.read_bytes(), b"original")

    def test_unlisted_step_refuses(self):
        with self.assertRaises(ValueError):
            subject.cleanup("emo", 42)
        self.assertTrue(self.raw.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
