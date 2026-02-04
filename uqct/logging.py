import logging
import sys


def setup_logging(level=logging.INFO):
    """
    Sets up the logging configuration for the project.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


_initialized = False


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with the given name.
    Automatically configures logging if it hasn't been configured yet.
    """
    global _initialized
    if not _initialized:
        setup_logging()
        _initialized = True
    return logging.getLogger(name)
