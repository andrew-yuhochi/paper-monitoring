# Logging configuration for the paper-monitoring project.
# Call setup_logging() once at startup (pipeline.py or seed.py entry points).
# Writes to both stderr and a rotating file at data/logs/pipeline.log.

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_dir: Path, log_level: str) -> None:
    """Configure the root logger with a rotating file handler and a stderr stream handler.

    Args:
        log_dir: Directory where pipeline.log will be written. Created if absent.
        log_level: Logging level string (e.g. "INFO", "DEBUG", "WARNING").
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(log_format)

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Rotating file handler — max 5 MB per file, keep 3 backups
    file_handler = RotatingFileHandler(
        log_dir / "pipeline.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    # Stream handler — writes to stderr
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
