"""
Example of how to train a transformer language model.
"""

from olmo_core.train import prepare_training_environment

BACKEND = "nccl"


def main():
    pass


if __name__ == "__main__":
    prepare_training_environment(backend=BACKEND)
    main()
