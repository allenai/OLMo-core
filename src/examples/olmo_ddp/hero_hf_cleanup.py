"""Delete only a qualified conversion's owned raw scratch copy, never its source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from hero_hf_download import SCRATCH, TARGETS, prepare_scratch, write_json


def checked_file(root: Path, name: str) -> Path:
    """Require an ordinary file strictly inside the given, non-symlinked root."""
    path = root / name
    if path.resolve() != path or not path.is_relative_to(root) or not path.is_file():
        raise RuntimeError(f"Unsafe or missing qualification artifact: {path}")
    return path


def cleanup(arm: str, step: int) -> dict:
    """Fail closed unless all conversion/eval evidence and output hashes still match."""
    if arm not in ("emo", "non-emo") or step not in TARGETS.values():
        raise ValueError("Checkpoint is outside the approved campaign allowlist")
    prepare_scratch()
    root = SCRATCH / arm / f"step{step}"
    if root.resolve() != root:
        raise RuntimeError("Symlink in scratch path")
    source = json.loads(checked_file(root, "download-success.json").read_text())
    conversion = json.loads(checked_file(root, "conversion-success.json").read_text())
    parity = json.loads(checked_file(root, "vllm-parity-success.json").read_text())
    smoke = json.loads(checked_file(root, "eval-smoke-success.json").read_text())
    profile = conversion.get("inference_profile", "bf16")
    if profile not in ("bf16", "fp32-linear64-recurrent-v1"):
        raise RuntimeError("Unknown inference precision qualification")
    precise = profile == "fp32-linear64-recurrent-v1"
    if bool(parity.get("precise")) != precise or bool(smoke.get("precise")) != precise:
        raise RuntimeError("Conversion, native parity, and eval precision profiles disagree")
    hf, raw = root / "hf", root / "olmo-core"
    if (
        conversion.get("passed") is not True
        or parity.get("passed") is not True
        or smoke.get("passed") is not True
        or parity.get("diagnostic_only")
        or smoke.get("diagnostic_only")
        or conversion.get("arm") != arm
        or conversion.get("step") != step
        or conversion.get("output") != str(hf)
        or parity.get("model") != str(hf)
        or smoke.get("model") != str(hf)
        or source.get("step") != step
        or source.get("raw_path") != str(raw)
    ):
        raise RuntimeError("Conversion/parity/eval provenance is incomplete or inconsistent")
    if json.loads(checked_file(hf, "_HERO_CONVERSION_SUCCESS.json").read_text()) != conversion:
        raise RuntimeError("HF output's conversion receipt disagrees with external receipt")
    conversion_hash = hashlib.sha256(
        checked_file(hf, "_HERO_CONVERSION_SUCCESS.json").read_bytes()
    ).hexdigest()
    parity_hash = hashlib.sha256(
        checked_file(root, "vllm-parity-success.json").read_bytes()
    ).hexdigest()
    if (
        parity.get("conversion_sha256") != conversion_hash
        or smoke.get("conversion_sha256") != conversion_hash
        or smoke.get("parity_sha256") != parity_hash
    ):
        raise RuntimeError("Stale or mismatched conversion/parity/eval receipt chain")
    hashes = conversion.get("output_sha256", {})
    if not hashes or not any(name.endswith(".safetensors") for name in hashes):
        raise RuntimeError("No verified HF model weights")
    for name, expected in hashes.items():
        path = checked_file(hf, name)
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise RuntimeError(f"HF output changed after qualification: {path}")
    metric = checked_file(root, str(Path(smoke["metrics"]).relative_to(root)))
    if hashlib.sha256(metric.read_bytes()).hexdigest() != smoke["metrics_sha256"]:
        raise RuntimeError("Eval metrics changed after qualification")
    if (root / "cleanup-success.json").is_file():
        if raw.exists():
            raise RuntimeError("Raw files reappeared after a previously recorded cleanup")
        return json.loads((root / "cleanup-success.json").read_text())
    if raw.resolve() != raw or raw.is_symlink() or not raw.is_dir():
        raise RuntimeError("Raw target is missing or unsafe")
    files = []
    for path in raw.rglob("*"):
        if path.is_symlink() or path.resolve() != path:
            raise RuntimeError(f"Symlink inside raw cleanup target: {path}")
        if path.is_file():
            st = path.stat()
            if st.st_nlink != 1:
                raise RuntimeError(f"Raw file is not an independent copy: {path}")
            files.append((str(path.relative_to(raw)), st.st_size))
    total = sum(size for _, size in files)
    if len(files) != source["files"] or total != source["bytes"]:
        raise RuntimeError("Raw inventory changed since staging; refusing cleanup")
    intent = {
        "arm": arm,
        "step": step,
        "target": str(raw),
        "kept_hf": str(hf),
        "files": len(files),
        "bytes": total,
        "started_at_unix": time.time(),
        "scope": "owned scratch copy only; original checkpoint and HF bucket untouched",
    }
    write_json(root / "cleanup-intent.json", intent)
    tombstone = root / "olmo-core.deleting"
    if tombstone.exists() or tombstone.is_symlink():
        raise RuntimeError("Earlier deletion needs inspection; no new deletion attempted")
    os.rename(raw, tombstone)
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError("Require symlink-attack-resistant directory removal")
    shutil.rmtree(tombstone)
    result = {**intent, "passed": True, "completed_at_unix": time.time()}
    write_json(root / "cleanup-success.json", result)
    print("HERO_RAW_SCRATCH_CLEANUP_SUCCESS " + json.dumps(result), flush=True)
    return result


def main() -> None:
    """Cleanup requires the explicit arm/step allowlist and all completed gates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("emo", "non-emo"), required=True)
    parser.add_argument("--step", type=int, choices=tuple(TARGETS.values()), required=True)
    args = parser.parse_args()
    cleanup(args.arm, args.step)


if __name__ == "__main__":
    main()
