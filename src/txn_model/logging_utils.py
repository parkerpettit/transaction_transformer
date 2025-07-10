import logging
import time
from contextlib import contextmanager


def configure_logging(name: str = "txn_model", level: int = logging.INFO, log_file: str | None = None) -> logging.Logger:
    """Configure and return a logger with standard formatting."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


@contextmanager
def log_duration(logger: logging.Logger, message: str):
    """Context manager to log the duration of a code block."""
    start = time.perf_counter()
    logger.info(f"{message}...")
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.info(f"{message} completed in {duration:.2f}s")

