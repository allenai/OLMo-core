import logging

from olmo_core.utils import prepare_cli_environment

log = logging.getLogger("olmo_core.logging_test")


def main():
    import datasets
    import transformers

    print("Starting test...")
    log.debug("Debug message!")
    log.info("Info message!")
    log.warning("Warning message!")
    log.error("Error message!")
    print("End test. If you didn't see any log messages, something is wrong.")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
