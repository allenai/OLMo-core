import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import google.auth
from beaker import Job, Priority
from google.cloud.compute_v1.services.instances.client import InstancesClient

from olmo_core.exceptions import BeakerInsufficientResourcesError

log = logging.getLogger(__name__)


@dataclass
class HostMetadata:
    block: str
    """
    The ID of the block in which the running instance is located. Instances within the same block
    experience low network latency.
    """

    subblock: str
    """
    The ID of the sub-block in which the running instance is located. Instances in the same sub-block
    experience lower network latency than instances in the same block.
    """

    machine: str
    """
    The ID of the machine ('host' according to Google) on which the running instance is located.
    Instances on the same machine experience the lowest possible network latency.
    """


def get_hosts_metadata_from_gcp(
    zone: str, *, credentials_path: Optional[Path] = None
) -> dict[str, HostMetadata]:
    if credentials_path:
        credentials, project_id = google.auth.load_credentials_from_file(str(credentials_path))
    else:
        credentials, project_id = google.auth.default()

    if project_id != "h100-cluster-owner":
        raise RuntimeError(
            f"Expected credentials for 'h100-cluster-owner', got credentials for {project_id}"
        )

    client = InstancesClient(credentials=credentials)
    instance_pages = client.list(project=project_id, zone=zone).pages
    return {
        f"{instance.name}.reviz.ai2.in": HostMetadata(
            block=instance.resource_status.physical_host_topology.block,
            subblock=instance.resource_status.physical_host_topology.subblock,
            machine=instance.resource_status.physical_host_topology.host,
        )
        for page in instance_pages
        for instance in page.items
    }


def get_occupied_beaker_hosts(
    hosts_metadata: dict[str, HostMetadata], beaker_cluster: str, beaker_priority: Priority
) -> set[str]:
    from olmo_core.internal.common import get_beaker_client

    beaker = get_beaker_client()
    assert beaker is not None

    occupied_hosts = set()

    cluster = beaker.cluster.get(beaker_cluster)
    nodes = beaker.cluster.nodes(cluster)
    for node in sorted(nodes, key=lambda node: node.hostname):
        host = node.hostname
        assert host not in occupied_hosts, f"Host {host} is somehow already in occupied hosts"
        if host not in hosts_metadata:
            log.warning(f"No metadata found for beaker host {host}")
            continue

        if node.cordoned is not None:
            # Treat cordonned node as occupied since it might be uncordonned later.
            occupied_hosts.add(host)
            continue

        jobs = beaker.job.list(node=node)
        for job in jobs:
            if (
                job.is_running
                and job.execution is not None
                and (resources := job.execution.spec.resources) is not None
                and resources.gpu_count is not None
                and resources.gpu_count > 0
                and not _is_job_preemptible(job, beaker_priority)
            ):
                occupied_hosts.add(host)
                break

    return occupied_hosts


def _is_job_preemptible(job: Job, desired_priority: Priority) -> bool:
    if not job.is_preemptible:
        return False

    assert job.priority is not None
    # Priorities are sorted highest to lowest by default
    sorted_priorities = list(Priority)
    return sorted_priorities.index(desired_priority) < sorted_priorities.index(job.priority)


def get_hostname_constraints(
    hosts_metadata: dict[str, HostMetadata],
    num_execution_units: int,
    num_hosts_per_task: int,
    num_tasks: int,
    occupied_hosts: Optional[set[str]] = None,
) -> list[list[str]]:
    if num_hosts_per_task % num_execution_units != 0:
        raise ValueError(
            "Number of execution units must be a divisor of number of hosts in a task "
            "(since a task runs one or more execution units)."
        )
    num_hosts_per_exec_unit = (num_tasks * num_hosts_per_task) // num_execution_units

    occupied_hosts = occupied_hosts or set()

    hosts_by_block: defaultdict[str, list[str]] = defaultdict(list)
    available_hosts_by_block: defaultdict[str, list[str]] = defaultdict(list)
    for host, metadata in hosts_metadata.items():
        hosts_by_block[metadata.block].append(host)
        if host not in occupied_hosts:
            available_hosts_by_block[metadata.block].append(host)

    # Sort blocks in descending order of number of available hosts
    sorted_blocks = sorted(
        available_hosts_by_block.keys(),
        key=lambda block: len(available_hosts_by_block[block]),
        reverse=True,
    )

    if len(available_hosts_by_block[sorted_blocks[0]]) > num_hosts_per_task * num_tasks:
        # If all the tasks can be fulfilled in a single block, let's do that!
        # Moreover, include occupied hosts since they might free up later.
        return [list(hosts_by_block[sorted_blocks[0]]) for _ in range(num_tasks)]

    hosts_per_task: list[list[str]] = []
    block_idx = 0
    for _ in range(num_tasks):
        task_hosts: list[str] = []

        while block_idx < len(sorted_blocks) and len(task_hosts) < num_hosts_per_task:
            block = sorted_blocks[block_idx]
            num_block_hosts = len(available_hosts_by_block[block])

            if num_block_hosts < num_hosts_per_exec_unit:
                # We have exhausted this block, move onto the next one
                block_idx += 1
                continue

            needed_hosts = num_hosts_per_task - len(task_hosts)
            assert needed_hosts % num_hosts_per_exec_unit == 0

            # Take the nearest multiple of num_hosts_per_exec_unit as the number of hosts to give to the current task
            host_count_from_block = min(
                needed_hosts,
                num_block_hosts // num_hosts_per_exec_unit * num_hosts_per_exec_unit,
            )
            assert host_count_from_block > 0

            block_hosts = available_hosts_by_block[block]
            task_hosts.extend(block_hosts[:host_count_from_block])

            available_hosts_by_block[block] = block_hosts[host_count_from_block:]

        hosts_per_task.append(task_hosts)

    if len(hosts_per_task) < num_tasks:
        raise BeakerInsufficientResourcesError(
            f"Could only satisfy {len(hosts_per_task)} out of {num_tasks} tasks"
        )
    if len(hosts_per_task[-1]) < num_hosts_per_task:
        raise BeakerInsufficientResourcesError(
            f"Could not satisfy task number {len(hosts_per_task) - 1}, only got {len(hosts_per_task[-1])} hosts"
        )

    return hosts_per_task


def get_beaker_hostname_constraints(
    num_nodes: int,
    num_execution_units: int,
    beaker_task_count: int,
    gcp_zone: str,
    *,
    beaker_cluster: str,
    beaker_priority: Priority,
    gcp_credentials_path: Optional[Path] = None,
) -> list[list[str]]:
    if beaker_cluster != "ai2/augusta-google-1":
        raise ValueError(
            "Only Augusta is supported. Making this work for other clusters probably would be a bad idea..."
        )

    if beaker_priority != Priority.urgent:
        log.warning(
            "Host selection depends on the cluster having nodes with jobs running at lower priority. "
            "It is relatively unlikely to work on non-urgent priorities."
        )

    assert num_nodes > 0
    assert num_nodes % num_execution_units == 0
    assert num_nodes % beaker_task_count == 0
    beaker_num_hosts_per_task = num_nodes // beaker_task_count

    hosts_metadata = get_hosts_metadata_from_gcp(gcp_zone, credentials_path=gcp_credentials_path)

    occupied_hosts = get_occupied_beaker_hosts(hosts_metadata, beaker_cluster, beaker_priority)

    return get_hostname_constraints(
        hosts_metadata,
        num_execution_units,
        beaker_num_hosts_per_task,
        beaker_task_count,
        occupied_hosts,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_nodes", type=int, help="Total number of nodes")
    parser.add_argument(
        "--num-execution-units",
        type=int,
        default=1,
        help="Number of `execution units`. An `execution unit` is abstraction for any node-using entity of which 1 or more copies are run that requires its nodes come from the same block (e.g., a model replica).",
    )
    parser.add_argument(
        "--task-count",
        type=int,
        default=1,
        help="Number of beaker (pre-replication) tasks this job is being spread across",
    )

    # Beaker-related settings
    parser.add_argument(
        "--priority",
        type=Priority,
        default=Priority.normal,
        help="Desired beaker job priority.",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="ai2/augusta-google-1",
        help="The beaker cluster. This defaults to and is assumed to be Augusta for now.",
    )

    # GCP-related settings
    parser.add_argument(
        "--zone",
        type=str,
        default="us-central1-b",
        help="The GCP zone where the Augusta nodes are located.",
    )
    parser.add_argument(
        "--credentials-path",
        type=Path,
        help="The path to GCP credetials.",
    )
    args = parser.parse_args()

    print(
        get_beaker_hostname_constraints(
            args.num_nodes,
            args.num_execution_units,
            args.task_count,
            args.zone,
            beaker_cluster=args.cluster,
            beaker_priority=args.priority,
        )
    )


if __name__ == "__main__":
    main()
