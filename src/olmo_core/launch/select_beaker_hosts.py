import argparse
import concurrent.futures
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import google.auth
from beaker import Beaker, Job, Node, Priority
from google.cloud.compute_v1.services.instances.client import InstancesClient

from olmo_core.exceptions import BeakerInsufficientResourcesError
from olmo_core.utils import prepare_cli_environment

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
        if instance.status == "RUNNING"
    }


def node_is_occupied(beaker: Beaker, node: Node, beaker_priority: Priority) -> Tuple[Node, bool]:
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
            return node, True
    return node, False


def get_occupied_beaker_hosts(
    hosts_metadata: dict[str, HostMetadata], beaker_cluster: str, beaker_priority: Priority
) -> set[str]:
    from olmo_core.internal.common import get_beaker_client

    beaker = get_beaker_client()
    assert beaker is not None

    occupied_hosts = set()

    cluster = beaker.cluster.get(beaker_cluster)
    nodes = beaker.cluster.nodes(cluster)
    missing_metadata_count = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for node in sorted(nodes, key=lambda node: node.hostname):
            host = node.hostname
            assert host not in occupied_hosts, f"Host {host} is somehow already in occupied hosts"
            if host not in hosts_metadata:
                log.warning(f"No metadata found for beaker host {host}")
                missing_metadata_count += 1
                continue

            if node.cordoned is not None:
                # Treat cordoned node as occupied since it might be uncordoned later.
                occupied_hosts.add(host)
                continue

            futures.append(executor.submit(node_is_occupied, beaker, node, beaker_priority))

        for future in concurrent.futures.as_completed(futures):
            node, is_occupied = future.result()
            if is_occupied:
                occupied_hosts.add(node.hostname)

    if missing_metadata_count > 1:
        log.warning(
            f"Could not find metadata for {missing_metadata_count} hosts; "
            "these hosts will be ignored when selecting hosts."
        )

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
            f"Could not satisfy task {len(hosts_per_task)} of {num_tasks}. "
            f"Only found {len(hosts_per_task[-1])} elegible hosts with the constraint that "
            f"all {num_hosts_per_exec_unit} hosts for an execution unit must be in the same block, "
            f"but {num_hosts_per_task} are required."
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
    hosts_metadata: Optional[dict[str, HostMetadata]] = None,
) -> list[list[str]]:
    if beaker_cluster != "ai2/augusta":
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

    if hosts_metadata is None:
        hosts_metadata = get_hosts_metadata_from_gcp(
            gcp_zone, credentials_path=gcp_credentials_path
        )

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
        default="ai2/augusta",
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
    prepare_cli_environment()

    hosts_metadata = get_hosts_metadata_from_gcp(
        args.zone,
        credentials_path=args.credentials_path,
    )

    for task, hosts in enumerate(
        get_beaker_hostname_constraints(
            args.num_nodes,
            args.num_execution_units,
            args.task_count,
            args.zone,
            beaker_cluster=args.cluster,
            beaker_priority=args.priority,
            gcp_credentials_path=args.credentials_path,
            hosts_metadata=hosts_metadata,
        )
    ):
        print(f"Task {task+1}/{args.task_count}, found {len(hosts)} elegible hosts:")
        for host in hosts:
            metadata = hosts_metadata[host]
            print(f"  {host} - block: {metadata.block}, subblock: {metadata.subblock}")


if __name__ == "__main__":
    main()
