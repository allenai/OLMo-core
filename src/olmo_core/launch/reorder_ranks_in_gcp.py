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
    rank0_host_id = all_host_ids[0]
    all_host_ids.sort()
    # Rank 0 needs to remain rank 0, so we reshuffle around it
    rank0_index = all_host_ids.index(rank0_host_id)
    all_host_ids = all_host_ids[rank0_index:] + all_host_ids[:rank0_index]
    print(all_host_ids.index(host_id))

    # Make sure we're all done before exiting
    store.set(f"node_{args.rank}_done", host_id)
    store.wait([f"node_{i}_done" for i in range(args.world_size)])


if __name__ == "__main__":
    main()
