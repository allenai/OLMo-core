"""
Run the CI pytest suite on each locally-built image on Beaker (2 GPUs each), across the GPU types
each image supports (H100, B200, B300), collect the JUnit results, and verify that every test
executed (passed or failed) on at least one run — except tests that need more than 2 GPUs, which
cannot run on a 2-GPU node.

Each run covers a different subset for real: e.g. the ``flash_3`` backend tests only execute on the
CUDA-12.8 images and ``flash_4`` only on the ``-fa4`` (CUDA-13) images, so no single image covers
everything but the *union* should. CUDA-12.8 images (sm_90/100) are only launched on H100/B200, not
B300 (sm_103); CUDA-13 images run on all three.

Two phases; the analysis phase is standalone so it can be re-run on already-downloaded results.

Launch + collect + analyze (needs Beaker access)::

    python src/scripts/beaker/run_ci_tests_on_images.py --date 2026-07-28
    python src/scripts/beaker/run_ci_tests_on_images.py --date 2026-07-28 --gpus h100 b200

Analyze only, on a directory of ``<image>@<gpu>.xml`` JUnit files::

    python src/scripts/beaker/run_ci_tests_on_images.py --analyze-dir /path/to/junit-xmls

Requires a Beaker token (``beaker account login`` or ``BEAKER_TOKEN``) for the launch phase.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# The image matrix, by label -> tag stem (the '<date>' suffix is appended at runtime). Mirrors the
# Makefile targets in `beaker-image-<label>`.
IMAGE_TAG_STEMS: Dict[str, str] = {
    "cu128": "tch2100cu128",
    "cu128-rma": "tch2100cu128-rma",
    "cu130": "tch2110cu130",
    "cu130-fa4": "tch2110cu130-fa4",
    "cu130-rma": "tch2110cu130-rma",
    "cu130-fa4-rma": "tch2110cu130-fa4-rma",
}

NUM_GPUS = 2

# GPU targets. At AI2 each cluster is homogeneous, so selecting the cluster selects the GPU type;
# GPU_TYPE_NAMES is just for display. (A `gpu_types` constraint isn't needed and its name format
# differs from these, so we don't set one.)
GPU_TYPE_NAMES: Dict[str, str] = {
    "h100": "NVIDIA H100 80GB HBM3",
    "b200": "NVIDIA B200",
    "b300": "NVIDIA B300",
}
DEFAULT_CLUSTERS_BY_GPU: Dict[str, List[str]] = {
    "h100": ["ai2/jupiter", "ai2/ceres"],
    "b200": ["ai2/titan"],
    "b300": ["ai2/holmes"],
}


def compatible_gpus(image_label: str) -> List[str]:
    """GPUs an image can run on. CUDA-12.8 images (sm_90/100) can't run on B300 (sm_103)."""
    return ["h100", "b200"] if image_label.startswith("cu128") else ["h100", "b200", "b300"]


# A test skipped on *every* image is a coverage gap UNLESS it needs more GPUs than we launch with.
# Skip reasons like "requires four GPUs" / "Requires 4 GPUs" identify those; "Requires multiple GPUs"
# (>= 2) does NOT count — those run on our 2-GPU node, so if such a test skipped everywhere that is a
# real gap worth surfacing.
_WORD_TO_INT = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


def _required_gpu_count(skip_reason: str) -> Optional[int]:
    """Parse ``N`` from a skip reason that needs N GPUs (e.g. 'requires four GPUs'), else None."""
    if not skip_reason:
        return None
    m = re.search(r"(\d+|two|three|four|five|six|seven|eight)\s+gpus?", skip_reason.lower())
    if not m:
        return None
    tok = m.group(1)
    return int(tok) if tok.isdigit() else _WORD_TO_INT.get(tok)


@dataclass
class Outcome:
    """A test's outcome on one image."""

    status: str  # "passed" | "failed" | "skipped" | "error"
    skip_reason: str = ""


def parse_junit(path: Path) -> Dict[str, Outcome]:
    """Parse a JUnit XML into ``{nodeid: Outcome}``."""
    outcomes: Dict[str, Outcome] = {}
    root = ET.parse(path).getroot()
    for case in root.iter("testcase"):
        classname = case.get("classname", "")
        name = case.get("name", "")
        nodeid = f"{classname}::{name}" if classname else name
        skipped = case.find("skipped")
        failure = case.find("failure")
        error = case.find("error")
        if skipped is not None:
            outcomes[nodeid] = Outcome(
                "skipped", skipped.get("message", "") or (skipped.text or "")
            )
        elif failure is not None:
            outcomes[nodeid] = Outcome("failed")
        elif error is not None:
            outcomes[nodeid] = Outcome("error")
        else:
            outcomes[nodeid] = Outcome("passed")
    return outcomes


@dataclass
class CoverageReport:
    per_image: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )  # label -> {passed, failed, ...}
    failures: Dict[str, List[str]] = field(default_factory=dict)  # label -> [nodeid failed/errored]
    genuine_gaps: List[str] = field(default_factory=list)  # skipped on every image, <=2 GPUs
    excluded_gaps: List[Tuple[str, int]] = field(default_factory=list)  # (nodeid, gpus needed)

    @property
    def ok(self) -> bool:
        return not self.genuine_gaps and not any(self.failures.values())


def analyze(results: Dict[str, Dict[str, Outcome]]) -> CoverageReport:
    """
    :param results: ``{image_label: {nodeid: Outcome}}``.
    :returns: a coverage report over the union of all tests seen across images.
    """
    report = CoverageReport()
    all_nodeids: set[str] = set()
    for label, outcomes in results.items():
        all_nodeids.update(outcomes)
        counts: Dict[str, int] = {}
        fails: List[str] = []
        for nodeid, o in outcomes.items():
            counts[o.status] = counts.get(o.status, 0) + 1
            if o.status in ("failed", "error"):
                fails.append(nodeid)
        report.per_image[label] = counts
        report.failures[label] = sorted(fails)

    for nodeid in sorted(all_nodeids):
        # Did the test actually execute (pass or fail) on any image?
        if any(
            results.get(label, {}).get(nodeid, Outcome("absent")).status in ("passed", "failed")
            for label in results
        ):
            continue
        # Skipped/absent on every image. Is it excluded because it needs > NUM_GPUS GPUs?
        needed = None
        for label in results:
            oc = results.get(label, {}).get(nodeid)
            if oc is not None and oc.status == "skipped":
                needed = needed or _required_gpu_count(oc.skip_reason)
        if needed is not None and needed > NUM_GPUS:
            report.excluded_gaps.append((nodeid, needed))
        else:
            report.genuine_gaps.append(nodeid)
    return report


def print_report(report: CoverageReport) -> None:
    print("\n================ Per-run outcomes (image@gpu) ================")
    for label in sorted(report.per_image):
        c = report.per_image[label]
        print(
            f"  {label:22s} passed={c.get('passed', 0):5d} failed={c.get('failed', 0):3d} "
            f"error={c.get('error', 0):3d} skipped={c.get('skipped', 0):5d}"
        )

    any_fail = any(report.failures.values())
    if any_fail:
        print("\n================ Failures ================")
        for label, fails in report.failures.items():
            for nodeid in fails:
                print(f"  ✗ [{label}] {nodeid}")

    if report.excluded_gaps:
        print(f"\n================ Excluded (need > {NUM_GPUS} GPUs) ================")
        for nodeid, n in report.excluded_gaps:
            print(f"  - {nodeid}  (needs {n} GPUs)")

    print("\n================ Coverage ================")
    if report.genuine_gaps:
        print(f"  ✗ {len(report.genuine_gaps)} test(s) skipped on ALL images (coverage gap):")
        for nodeid in report.genuine_gaps:
            print(f"      {nodeid}")
    else:
        print("  ✓ Every test ran on at least one image (excluding >2-GPU tests).")
    print()


# --------------------------------------------------------------------------------------------------
# Beaker launch + collect
# --------------------------------------------------------------------------------------------------
def _beaker_user() -> str:
    out = subprocess.check_output(["beaker", "account", "whoami", "--format=json"], text=True)
    import json

    return json.loads(out)[0]["name"]


def launch_and_collect(
    date: str,
    out_dir: Path,
    images: List[str],
    gpus: List[str],
    clusters_by_gpu: Dict[str, List[str]],
    workspace: str,
    priority: str,
) -> None:
    from beaker.types import BeakerWorkload

    from olmo_core.launch.beaker import BeakerEnvSecret, BeakerLaunchConfig
    from olmo_core.train.callbacks.beaker import BEAKER_RESULT_DIR
    from olmo_core.utils import generate_uuid, prepare_cli_environment

    prepare_cli_environment()
    user = _beaker_user()
    out_dir.mkdir(parents=True, exist_ok=True)

    # pytest over the whole tree; JUnit XML lands in the Beaker result dataset. continue-on-collection
    # so a module that hard-imports a backend absent on this image doesn't abort the run.
    pytest_cmd = (
        "pytest -v --color=yes --continue-on-collection-errors "
        f"-o junit_family=xunit2 --junitxml={BEAKER_RESULT_DIR}/junit.xml src/test/ ; "
        'echo "pytest exit: $?"'
    )

    # (image, gpu) matrix, skipping GPUs an image can't run on (cu128 has no B300 kernels).
    jobs = [(img, gpu) for img in images for gpu in compatible_gpus(img) if gpu in gpus]
    print(f"[launch] {len(jobs)} job(s) across {len(images)} image(s) x GPUs {gpus}:")
    for img, gpu in jobs:
        print(f"    {img}@{gpu}  -> {GPU_TYPE_NAMES[gpu]}  on {clusters_by_gpu[gpu]}")

    workloads: Dict[str, BeakerWorkload] = {}
    for img, gpu in jobs:
        label = f"{img}@{gpu}"
        image = f"{user}/olmo-core-{IMAGE_TAG_STEMS[img]}-{date}"
        cfg = BeakerLaunchConfig(
            name=f"olmo-core-imgtest-{img}-{gpu}-{generate_uuid()[:6]}",
            budget="ai2/oe-other",
            cmd=["bash", "-lc", pytest_cmd],
            task_name=f"test-{img}-{gpu}",
            workspace=workspace,
            beaker_image=image,
            clusters=clusters_by_gpu[gpu],
            priority=priority,
            num_nodes=1,
            num_gpus=NUM_GPUS,
            shared_filesystem=True,
            torchrun=False,  # tests spawn their own distributed processes
            env_secrets=[BeakerEnvSecret(name="HF_TOKEN", secret="HF_TOKEN")],
        )
        print(f"[launch] {label}: image={image}")
        workloads[label] = cfg.launch(follow=False)

    print(f"\n[launch] {len(workloads)} job(s) submitted. Waiting for completion...")
    results: Dict[str, Dict[str, Outcome]] = {}
    for label, wl in workloads.items():
        xml = _await_and_fetch_junit(wl, out_dir / f"{label}.xml")
        if xml is not None:
            results[label] = parse_junit(xml)
        else:
            print(f"[collect] WARNING: could not fetch results for {label}; skipping in analysis.")

    if not results:
        print(
            "[collect] No results collected. Fetch the JUnit XMLs manually and re-run with "
            "--analyze-dir."
        )
        sys.exit(1)

    report = analyze(results)
    print_report(report)
    sys.exit(0 if report.ok else 1)


def _await_and_fetch_junit(workload, dest: Path) -> Optional[Path]:
    """Await a workload, then fetch junit.xml from its result dataset via the Beaker CLI.

    Best-effort: the reliable path is to let the jobs finish, download each result dataset's
    junit.xml as ``<label>.xml``, and re-run with ``--analyze-dir``. Adjust the ``beaker`` CLI
    invocations below if your CLI version differs.
    """
    import json

    exp_id = getattr(workload.experiment, "id", None) or workload.experiment.name

    # Poll until the experiment's job has exited.
    while True:
        try:
            status = subprocess.check_output(
                ["beaker", "experiment", "get", exp_id, "--format=json"], text=True
            )
            jobs = json.loads(status)[0].get("jobs", [])
            state = (jobs[-1] if jobs else {}).get("status", {})
            if state.get("finalized") or state.get("exited") is not None:
                break
        except Exception as e:  # noqa: BLE001
            print(f"[collect] poll error for {exp_id}: {e}")
            return None
        time.sleep(30)

    try:
        result_ds = subprocess.check_output(
            ["beaker", "experiment", "results", exp_id, "--format=json"], text=True
        )
        ds_id = json.loads(result_ds)[0]["id"]
        subprocess.check_call(
            [
                "beaker",
                "dataset",
                "fetch",
                ds_id,
                "--prefix",
                "junit.xml",
                "--output",
                str(dest.parent),
            ]
        )
        fetched = dest.parent / "junit.xml"
        if fetched.exists():
            fetched.rename(dest)
            return dest
    except Exception as e:  # noqa: BLE001
        print(f"[collect] fetch error for {exp_id}: {e}")
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", help="Image date suffix, e.g. 2026-07-28 (required to launch).")
    p.add_argument(
        "--analyze-dir",
        type=Path,
        help="Analyze existing *.xml JUnit files (named <image>@<gpu>.xml) in this dir.",
    )
    p.add_argument(
        "--images", nargs="+", default=list(IMAGE_TAG_STEMS), choices=list(IMAGE_TAG_STEMS)
    )
    p.add_argument(
        "--gpus",
        nargs="+",
        default=list(GPU_TYPE_NAMES),
        choices=list(GPU_TYPE_NAMES),
        help="GPU types to test on (default: all).",
    )
    p.add_argument(
        "--b300-clusters",
        nargs="+",
        default=DEFAULT_CLUSTERS_BY_GPU["b300"],
        help=f"Cluster(s) hosting B300 GPUs (default: {DEFAULT_CLUSTERS_BY_GPU['b300']}).",
    )
    p.add_argument("--workspace", default="ai2/OLMo-core", help="Beaker workspace to launch in.")
    p.add_argument(
        "--priority",
        default="normal",
        choices=["low", "normal", "high", "urgent"],
        help="Beaker job priority (default: normal).",
    )
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/olmo-core-image-tests"))
    args = p.parse_args()

    if args.analyze_dir is not None:
        results = {}
        for xml in sorted(args.analyze_dir.glob("*.xml")):
            results[xml.stem] = parse_junit(xml)
        if not results:
            print(f"[analyze] no JUnit XMLs found in {args.analyze_dir}.")
            sys.exit(1)
        report = analyze(results)
        print_report(report)
        sys.exit(0 if report.ok else 1)

    if not args.date:
        p.error("--date is required to launch (or use --analyze-dir).")
    clusters_by_gpu = dict(DEFAULT_CLUSTERS_BY_GPU)
    clusters_by_gpu["b300"] = args.b300_clusters
    launch_and_collect(
        args.date,
        args.out_dir,
        args.images,
        args.gpus,
        clusters_by_gpu,
        args.workspace,
        args.priority,
    )


if __name__ == "__main__":
    main()
