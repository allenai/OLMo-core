import argparse
from collections import defaultdict

from google.cloud.compute_v1.services.instances.client import InstancesClient
from google.cloud.compute_v1.types import ResourceStatusPhysicalHostTopology


def get_machines_metadata() -> dict[str, ResourceStatusPhysicalHostTopology]:
    client = InstancesClient()
    instance_pages = client.list(project="h100-cluster-owner", zone="us-central1-b").pages
    return {
        instance.name: instance.resource_status.physical_host_topology
        for page in instance_pages
        for instance in page.items
    }


def get_host_name_constraints(
    num_nodes: int,
    num_model_replica_nodes: int,
    beaker_task_count: int,
    skip_urgent_nodes: bool = True,
):
    assert num_nodes % num_model_replica_nodes == 0
    assert num_nodes % beaker_task_count == 0

    assert num_nodes % beaker_task_count == 0
    beaker_task_size = num_nodes // beaker_task_count

    # assert beaker_task_count == 1, "Other task counts not supported"

    machines_metadata = get_machines_metadata()
    machines_metadata = {
        f"{host}.reviz.ai2.in": metadata for host, metadata in machines_metadata.items()
    }

    # Remove hosts with jobs running on urgent priority
    if skip_urgent_nodes:
        from olmo_core.internal.common import get_beaker_client

        beaker = get_beaker_client()
        assert beaker is not None

        cluster = beaker.cluster.get("augusta-google-1")
        jobs = beaker.job.list(cluster=cluster)

        for job in jobs:
            if job.node is None:
                continue

            host = beaker.node.get(job.node).hostname
            if host not in machines_metadata:
                continue

            if (
                job.is_running and job.execution and job.execution.spec.resources.gpu_count > 0
            ) and (not job.is_preemptible or job.priority == "urgent"):
                del machines_metadata[host]

    hosts_by_block: defaultdict[str, list[str]] = defaultdict(list)
    for host, metadata in machines_metadata.items():
        hosts_by_block[metadata.block].append(host)

    hosts_per_task: list[list[str]] = []
    for block, hosts in sorted(hosts_by_block.items(), key=lambda item: len(item[1]), reverse=True):
        # If beaker_task_size is too big, then we might specify every host we need for things to work.
        # If beaker_task_size is small, then we might specify every host we need for things to work.
        # Within a task, either every host must be from the same block or we must have exactly as many hosts as the task size!

        if len(hosts) > beaker_task_size:
            # Use all the hosts that we desire! They are all from the same block

            for _ in range(0, len(hosts), beaker_task_size):
                hosts_per_task.append(list(hosts))
        else:
            # We must constrain the hosts so that we have exactly as many hosts as the task size. This
            # forces the tasks to be on those specific hosts

            # Only keep a multiple of replica_size number of hosts from each block, to avoid having replicas
            # go across block boundaries
            host_count_from_block = (
                len(hosts) // num_model_replica_nodes
            ) * num_model_replica_nodes

            used_hosts_count = 0

            if len(hosts_per_task) > 0 and len(hosts_per_task[-1]) < beaker_task_size:
                # Last task needs more hosts, fill them up from here
                needed_hosts = beaker_task_size - len(hosts_per_task[-1])
                assert needed_hosts % num_model_replica_nodes == 0

                # Take the nearest multiple of num_model_replica_nodes as the number of hosts to give to that task
                used_hosts_count = min(
                    needed_hosts,
                    host_count_from_block // num_model_replica_nodes * num_model_replica_nodes,
                )

                hosts_per_task[-1].extend(hosts[:used_hosts_count])

            # Put remaining hosts into next task
            assert host_count_from_block - used_hosts_count <= beaker_task_size
            assert (host_count_from_block - used_hosts_count) % num_model_replica_nodes == 0
            hosts_per_task.append(list(hosts[used_hosts_count:host_count_from_block]))

    hosts_per_task = hosts_per_task[:beaker_task_count]

    if len(hosts_per_task) < beaker_task_count:
        raise RuntimeError(f"Could only satisfy {len(hosts_per_task)} tasks")
    if len(hosts_per_task[-1]) < beaker_task_size:
        raise RuntimeError(
            f"Could not satisfy task number {len(hosts_per_task) - 1}, only got {len(hosts_per_task[-1])} hosts"
        )

    return hosts_per_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_nodes", type=int, help="Total number of nodes")
    parser.add_argument("num_model_replica_nodes", type=int, help="Number of nodes in each replica")
    parser.add_argument(
        "beaker_task_count",
        type=int,
        help="Number of beaker (pre-replication) tasks this job is being spread across",
    )
    parser.add_argument(
        "--ignore-urgent",
        action="store_false",
        dest="skip_urgent_nodes",
        help="Number of beaker (pre-replication) tasks this job is being spread across",
    )
    args = parser.parse_args()

    print(
        get_host_name_constraints(
            args.num_nodes,
            args.num_model_replica_nodes,
            args.beaker_task_count,
            skip_urgent_nodes=args.skip_urgent_nodes,
        )
    )


if __name__ == "__main__":
    main()
