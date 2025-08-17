import json
import logging
import sys
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - trivial
        record_dict: Dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Include bound context if present
        for key in ("wo", "asset", "rule_hash"):
            if hasattr(record, key):
                record_dict[key] = getattr(record, key)
        return json.dumps(record_dict)


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that supports context binding."""

    def bind(self, **kwargs: Any) -> "ContextLogger":
        context = {**self.extra, **kwargs}
        return ContextLogger(self.logger, context)

    def process(self, msg: Any, kwargs: Dict[str, Any]):
        extra = kwargs.setdefault("extra", {})
        extra.update(self.extra)
        return msg, kwargs


def get_logger(**context: Any) -> ContextLogger:
    """Return a context-aware logger for the ``loto`` namespace."""
    logger = logging.getLogger("loto")
    return ContextLogger(logger, context)


def init_logging(verbosity: int = 0) -> logging.Logger:
    """Initialise JSON logging for CLI applications.

    Parameters
    ----------
    verbosity: int
        0 -> WARNING, 1 -> INFO, 2+ -> DEBUG
    """

    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO

    logger = logging.getLogger("loto")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


__all__ = ["get_logger", "init_logging"]
