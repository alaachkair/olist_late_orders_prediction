import logging
import sys
from pathlib import Path


def setup_logging(log_file: str = "logs/service.log"):
    """
    Configure logging to both console and file.
    """
    # Create logs directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("olist_service")
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
