import argparse
import sys

import requests
import torch.distributed as dist
from urllib3.exceptions import MaxRetryError, NameResolutionError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rank", type=int, help="Worker number")
    parser.add_argument("world_size", type=int, help="Total number of workers")
    parser.add_argument("master_addr", help="Hostname of worker 0")
    parser.add_argument("--master_port", type=int, default=29501, help="Port for TCPStore")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (outside of GCP)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the whole list of node ids in order on rank 0 to stderr",
    )
    args = parser.parse_args()

    # Create or connect to the store
    store = dist.TCPStore(
        host_name=args.master_addr,
        port=args.master_port,
        world_size=args.world_size,
        is_master=(args.rank == 0),
    )

    if args.debug:
        host_id = "Hello"
    else:
        try:
            response = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/attributes/physical_host",
                headers={"Metadata-Flavor": "Google"},
            )
            assert response.status_code == 200
            host_id = response.text.strip()
            # block = host_id.strip("/").split("/")[0]
            # sub_block = host_id.strip("/").split("/")[1]
            # machine = host_id.strip("/").split("/")[2]
        except requests.exceptions.ConnectionError as e:
            # Unwrap the exception
            e = e.args[0]
            if not isinstance(e, MaxRetryError):
                raise
            e = e.reason
            if not isinstance(e, NameResolutionError):
                raise
            # Seems we called this outside of GCP, so we do nothing and just print our original rank.
            print(args.rank)
            sys.exit(0)

    # Find the index of our host id
    store.set(f"node_{args.rank}_hostid", host_id)
    store.wait([f"node_{i}_hostid" for i in range(args.world_size)])
    all_hostids = [store.get(f"node_{i}_hostid").decode("UTF-8") for i in range(args.world_size)]
    assert host_id in all_hostids

    all_blocks = [hostid.strip("/").split("/")[0] for hostid in all_hostids]
    all_subblocks = [hostid.strip("/").split("/")[1] for hostid in all_hostids]
    all_machines = [hostid.strip("/").split("/")[2] for hostid in all_hostids]

    rank0_block = all_blocks[0]
    rank0_subblock = all_subblocks[0]
    rank0_machine = all_machines[0]

    # Rank 0 needs to remain rank 0.
    ranks = list(range(args.world_size))
    # Sort so that rank 0 blocks come first, then so that blocks are together, lastly so that shorter ranks are first
    ranks.sort(
        key=lambda rank: (
            all_blocks[rank] != rank0_block,
            all_subblocks[rank] != rank0_subblock,
            all_machines[rank] != rank0_machine,
            all_blocks[rank],
            all_subblocks[rank],
            all_machines[rank],
            rank
        )
    )
    assert ranks[0] == 0

    if args.verbose and args.rank == 0:
        for rank in ranks:
            print(
                f"Initial rank: {rank}, Final rank: {ranks.index(rank)}, Hostids: {all_hostids[rank]}",
                file=sys.stderr,
            )
    print(ranks.index(args.rank))

    # Make sure we're all done before exiting
    store.set(f"node_{args.rank}_done", host_id)
    store.wait([f"node_{i}_done" for i in range(args.world_size)])


if __name__ == "__main__":
    main()
