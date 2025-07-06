import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import google.auth
from beaker import Job, Priority
from google.cloud.compute_v1.services.instances.client import InstancesClient

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


def _is_job_preemptible(job: Job, desired_priority: Priority) -> bool:
    if not job.is_preemptible:
        return False

    assert job.priority is not None
    # Priorities are sorted highest to lowest by default
    sorted_priorities = list(Priority)
    return sorted_priorities.index(desired_priority) < sorted_priorities.index(job.priority)


def get_host_name_constraints(
    hosts_metadata: dict[str, HostMetadata],
    num_hosts_per_replica: int,
    num_hosts_per_task: int,
    num_tasks: int,
) -> list[list[str]]:
    hosts_by_block: defaultdict[str, list[str]] = defaultdict(list)
    for host, metadata in hosts_metadata.items():
        hosts_by_block[metadata.block].append(host)

    hosts_per_task: list[list[str]] = []
    for block, hosts in sorted(hosts_by_block.items(), key=lambda item: len(item[1]), reverse=True):
        # We want no model replicas to have nodes from different blocks.
        # To make this possible, within a task, either 1) every host must be from the same block or
        # 2) we must have exactly as many hosts as the task size, and the number of hosts from each block
        # must be a multiple of the model replica size.

        if len(hosts) > num_hosts_per_task:
            # Use all the hosts that we desire! They are all from the same block

            for _ in range(0, len(hosts), num_hosts_per_task):
                hosts_per_task.append(list(hosts))
        else:
            # We must constrain the hosts so that we have exactly as many hosts as the task size. This
            # forces the tasks to be on those specific hosts

            # Only keep a multiple of replica_size number of hosts from each block, to avoid having replicas
            # go across block boundaries
            host_count_from_block = (len(hosts) // num_hosts_per_replica) * num_hosts_per_replica

            used_hosts_count = 0

            if len(hosts_per_task) > 0 and len(hosts_per_task[-1]) < num_hosts_per_task:
                # Last task needs more hosts, fill them up from here
                needed_hosts = num_hosts_per_task - len(hosts_per_task[-1])
                assert needed_hosts % num_hosts_per_replica == 0

                # Take the nearest multiple of num_hosts_per_replica as the number of hosts to give to that task
                used_hosts_count = min(
                    needed_hosts,
                    host_count_from_block // num_hosts_per_replica * num_hosts_per_replica,
                )
                assert used_hosts_count % num_hosts_per_replica == 0

                hosts_per_task[-1].extend(hosts[:used_hosts_count])

            # Deal with when last host didn't get filled up

            # Put remaining hosts into next task
            assert host_count_from_block - used_hosts_count <= num_hosts_per_task
            assert (host_count_from_block - used_hosts_count) % num_hosts_per_replica == 0
            hosts_per_task.append(list(hosts[used_hosts_count:host_count_from_block]))

    hosts_per_task = hosts_per_task[:num_tasks]

    if len(hosts_per_task) < num_tasks:
        raise RuntimeError(f"Could only satisfy {len(hosts_per_task)} tasks")
    if len(hosts_per_task[-1]) < num_hosts_per_task:
        raise RuntimeError(
            f"Could not satisfy task number {len(hosts_per_task) - 1}, only got {len(hosts_per_task[-1])} hosts"
        )

    return hosts_per_task


def get_beaker_host_name_constraints(
    num_nodes: int,
    num_model_replica_nodes: int,
    beaker_task_count: int,
    gcp_zone: str,
    *,
    beaker_cluster: str,
    beaker_priority: Priority,
    gcp_credentials_path: Optional[Path] = None,
    remove_occupied_hosts: bool = False,
) -> list[list[str]]:
    if beaker_cluster != "augusta-google-1":
        raise ValueError(
            "Only Augusta is supported. Making this work for other clusters probably would be a bad idea..."
        )

    if beaker_priority != Priority.urgent:
        log.warning(
            "This script depends on cluster having nodes with jobs running at lower priority."
            "It is relatively unlikely to work on non-urgent priorities."
        )

    assert num_nodes > 0
    assert num_nodes % num_model_replica_nodes == 0
    assert num_nodes % beaker_task_count == 0
    beaker_num_hosts_per_task = num_nodes // beaker_task_count

    if beaker_task_count != 1:
        raise NotImplementedError("Multiple task are not (fully) supported")

    machines_metadata = get_hosts_metadata_from_gcp(gcp_zone, credentials_path=gcp_credentials_path)

    # Remove hosts with jobs running on equal or higher priority
    if remove_occupied_hosts:
        from olmo_core.internal.common import get_beaker_client

        beaker = get_beaker_client()
        assert beaker is not None

        cluster = beaker.cluster.get(beaker_cluster)
        jobs = beaker.job.list(cluster=cluster)

        for job in jobs:
            if job.node is None:
                continue

            host = beaker.node.get(job.node).hostname
            if host not in machines_metadata:
                continue

            if (
                job.is_running
                and job.execution
                and job.execution.spec.resources.gpu_count > 0
                and _is_job_preemptible(job, beaker_priority)
            ):
                del machines_metadata[host]

    return get_host_name_constraints(
        machines_metadata, num_model_replica_nodes, beaker_num_hosts_per_task, beaker_task_count
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("num-nodes", type=int, required=True, help="Total number of nodes")
    parser.add_argument(
        "num-model-replica-nodes", type=int, required=True, help="Number of nodes in each replica"
    )
    parser.add_argument(
        "--task-count",
        type=int,
        required=True,
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
        default="augusta-google-1",
        help="The beaker cluster. This defaults to and is assumed to be Augusta for now.",
    )
    parser.add_argument(
        "--allow-occupied-hosts",
        action="store_false",
        dest="remove_occupied_hosts",
        help="If set, allow selecting hosts with equal/higher priority GPU jobs.",
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
        required=True,
        help="The path to GCP credetials.",
    )
    args = parser.parse_args()

    print(
        get_beaker_host_name_constraints(
            args.num_nodes,
            args.num_model_replica_nodes,
            args.task_count,
            args.zone,
            beaker_cluster=args.cluster,
            beaker_priority=args.priority,
            remove_occupied_hosts=args.remove_occupied_hosts,
        )
    )


if __name__ == "__main__":
    main()
