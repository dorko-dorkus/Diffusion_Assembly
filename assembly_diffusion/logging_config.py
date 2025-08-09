import logging
import os

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured for console output.

    The logger uses a :class:`~logging.StreamHandler` with a timestamped format
    and honours the ``LOG_LEVEL`` environment variable to control verbosity.
    Subsequent calls with the same ``name`` return the existing logger without
    adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    return logger
