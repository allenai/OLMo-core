from olmo_core.train.utils import prepare_training_environment

BACKEND = "nccl"


def main():
    pass


if __name__ == "__main__":
    prepare_training_environment(backend=BACKEND)
    main()
