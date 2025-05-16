import logging

from olmo_core.utils import prepare_cli_environment

log = logging.getLogger("olmo_core.logging_test")


def main():
    print("Starting test...")
    log.debug("Debug message!")
    log.info("Info message!")
    log.info("Warning message!")
    log.info("Error message!")
    print("End test. If you didn't see any log messages, something is wrong.")


if __name__ == "__main__":
    prepare_cli_environment()
