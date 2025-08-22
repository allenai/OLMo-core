from collections import Counter

import pytest

from olmo_core.exceptions import BeakerInsufficientResourcesError
from olmo_core.launch.select_beaker_hosts import HostMetadata, get_host_name_constraints


def create_hosts_metadata(num_hosts_by_block: list[int]):
    return {
        f"block{i}_host{j}": HostMetadata(block=str(i), subblock="", machine="")
        for i, block_hosts in enumerate(num_hosts_by_block)
        for j in range(block_hosts)
    }


def test_get_host_name_constraints_one_super_block():
    num_hosts_by_block = [20]
    num_hosts_per_replica = 4
    num_tasks = 1
    num_hosts_per_task = 16

    hosts_metadata = create_hosts_metadata(num_hosts_by_block)
    host_name_constraints = get_host_name_constraints(
        hosts_metadata, num_hosts_per_replica, num_hosts_per_task, num_tasks
    )

    assert len(host_name_constraints) == num_tasks
    assert len(host_name_constraints[0]) >= num_hosts_per_task


def test_get_host_name_constraints_one_insufficient_block():
    num_hosts_by_block = [10]
    num_hosts_per_replica = 4
    num_tasks = 1
    num_hosts_per_task = 16

    hosts_metadata = create_hosts_metadata(num_hosts_by_block)

    with pytest.raises(BeakerInsufficientResourcesError) as exc_info:
        get_host_name_constraints(
            hosts_metadata, num_hosts_per_replica, num_hosts_per_task, num_tasks
        )
        assert exc_info.match("Could not satisfy task number 0, only got 10 hosts")


def test_get_host_name_constraints_multiple_blocks():
    num_hosts_by_block = [5, 13]
    num_hosts_per_replica = 4
    num_tasks = 1
    num_hosts_per_task = 16

    hosts_metadata = create_hosts_metadata(num_hosts_by_block)
    host_name_constraints = get_host_name_constraints(
        hosts_metadata, num_hosts_per_replica, num_hosts_per_task, num_tasks
    )

    assert len(host_name_constraints) == num_tasks
    assert len(host_name_constraints[0]) >= num_hosts_per_task
    blocks_host_count = Counter(host.split("_")[0] for host in host_name_constraints[0])
    assert blocks_host_count["block0"] == 4
    assert blocks_host_count["block1"] == 12


def test_get_host_name_constraints_multiple_insufficient_blocks():
    num_hosts_by_block = [5, 11]
    num_hosts_per_replica = 4
    num_tasks = 1
    num_hosts_per_task = 16

    hosts_metadata = create_hosts_metadata(num_hosts_by_block)

    with pytest.raises(BeakerInsufficientResourcesError) as exc_info:
        get_host_name_constraints(
            hosts_metadata, num_hosts_per_replica, num_hosts_per_task, num_tasks
        )
        assert exc_info.match("Could not satisfy task number 0, only got 12 hosts")
