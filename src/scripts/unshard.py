import argparse

from olmo_core.distributed.checkpoint import unshard_checkpoint
from olmo_core.utils import prepare_cli_environment, LogFilterType


def main():
    prepare_cli_environment(LogFilterType.all_ranks)

    parser = argparse.ArgumentParser(description='Unshard a checkpoint')
    parser.add_argument('directory', help='directory containing the checkpoint')
    parser.add_argument('-o', '--output', help='output directory', default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = f"{args.directory.rstrip('/')}_unsharded"

    unshard_checkpoint(args.directory, args.output, optim=True)


if __name__ == '__main__':
    main()