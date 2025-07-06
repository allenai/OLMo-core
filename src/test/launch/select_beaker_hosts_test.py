from olmo_core.launch.select_beaker_hosts import get_host_name_constraints, HostMetadata



def test_host_name_constraints():
    num_hosts_by_block = [2, 4, 6, 8]
    num_hosts_per_replica = 4
    num_tasks = 1
    num_hosts_per_task = 16

    hosts_metadata = {
        f"block{i}_host{j}": HostMetadata(block=str(i), subblock="", machine="")
        for i, block_hosts in enumerate(num_hosts_by_block)
        for j in range(block_hosts) 
    }

    get_host_name_constraints(hosts_metadata, num_hosts_per_replica, num_hosts_per_task, num_tasks)
