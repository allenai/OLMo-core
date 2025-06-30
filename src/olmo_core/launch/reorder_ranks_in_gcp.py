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

    # Get our own host id
    if args.debug:
        import socket

        host_id = f"{socket.gethostname()}_{args.rank}"
    else:
        try:
            response = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/attributes/physical_host",
                headers={"Metadata-Flavor": "Google"},
            )
            assert response.status_code == 200
            host_id = response.text.strip()
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
    all_host_ids = [store.get(f"node_{i}_hostid").decode("UTF-8") for i in range(args.world_size)]
    assert len(set(all_host_ids)) == len(all_host_ids)
    assert host_id in all_host_ids

    # hostid is of the form "/block/subblock/machine", extract each component from all ranks.
    all_blocks = [hostid.strip("/").split("/")[0] for hostid in all_host_ids]
    all_subblocks = [hostid.strip("/").split("/")[1] for hostid in all_host_ids]
    all_machines = [hostid.strip("/").split("/")[2] for hostid in all_host_ids]

    rank0_block = all_blocks[0]
    rank0_subblock = all_subblocks[0]
    rank0_machine = all_machines[0]

    ranks = list(range(args.world_size))
    # We want that 1) rank 0 remains rank 0, 2) blocks are grouped up, subblocks are grouped
    # up within blocks, and machines are grouped up within subblocks.
    # To achieve this, sort ranks so that the rank 0 block, subblock and machine are first
    # by sorting on making its block, subblock and machine falsey in the sorting key.
    # After this separation, sort so that blocks are together, subblocks are together
    # within blocks, and machines are together within subblocks.
    # Lastly, if 2 ranks are on the same block + subblock + machine, make the lowest
    # rank first.
    ranks.sort(
        key=lambda rank: (
            all_blocks[rank] != rank0_block,
            all_subblocks[rank] != rank0_subblock,
            all_machines[rank] != rank0_machine,
            all_blocks[rank],
            all_subblocks[rank],
            all_machines[rank],
            rank,
        )
    )
    assert ranks[0] == 0

    if args.verbose and args.rank == 0:
        for rank in ranks:
            if args.verbose and host_id == rank0_host_id:
        for i in all_host_ids:
            print(i, file=sys.stderr)print(
                f"Initial rank: {rank}, Final rank: {ranks.index(rank)}, Hostids: {all_host_ids[rank]}",
                file=sys.stderr,
            )
    print(ranks.index(args.rank))

    # Make sure we're all done before exiting
    store.set(f"node_{args.rank}_done", host_id)
    store.wait([f"node_{i}_done" for i in range(args.world_size)])


if __name__ == "__main__":
    main()
