"""Fail early on a degraded eight-GPU NVLink node, before a costly profile compiles."""

import argparse
import json
import re
import socket
import subprocess
from pathlib import Path


def validate_topology(output: str, expected_gpus: int = 8) -> dict:
    """Require every off-diagonal GPU link to be an active NVLink connection.

    This check is deliberately specific to our full eight-B300-node speed tests,
    not PCIe systems or jobs using a subset of a host's GPUs.
    """
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    rows = {}
    for line in output.splitlines():
        fields = line.split()
        if fields and re.fullmatch(r"GPU\d+", fields[0]):
            rows[int(fields[0][3:])] = fields[1 : expected_gpus + 1]
    if set(rows) != set(range(expected_gpus)):
        raise ValueError(f"Expected GPU0..GPU{expected_gpus - 1}, got {sorted(rows)}")
    bad_links = []
    for source, links in rows.items():
        if len(links) != expected_gpus:
            raise ValueError(f"Incomplete topology row for GPU{source}: {links}")
        for target, link in enumerate(links):
            if source == target:
                if link != "X":
                    raise ValueError(f"Unexpected GPU{source} diagonal: {link}")
            elif not re.fullmatch(r"NV[1-9]\d*", link):
                bad_links.append([source, target, link])
    if bad_links:
        raise ValueError(f"Degraded local GPU topology; non-NVLink pairs: {bad_links}")
    return {"gpus": expected_gpus, "gpu_link_matrix": rows, "all_pairs_nvlink": True}


def main():
    """Save read-only topology evidence, and reject disconnected GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True, timeout=30)
    inventory = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,power.limit",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=30,
    )
    report = {"hostname": socket.getfqdn(), "inventory": inventory, "raw_topology": topology}
    try:
        report.update(validate_topology(topology))
    except ValueError as error:
        report["error"] = str(error)
        raise
    finally:
        (args.output_dir / "topology.json").write_text(json.dumps(report, indent=2))
        print("PROFILE_TOPOLOGY", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
