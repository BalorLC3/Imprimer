import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Create (or reuse) a logger with:
    - Console output (INFO+)
    - Rotating file output (DEBUG+)
    """

    # Ensure log directory exists (no error if already created)
    Path("logs").mkdir(exist_ok=True)

    logger = logging.getLogger(name)

    # If the logger was already configured, reuse it
    # (prevents duplicate messages from multiple handlers)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Capture everything
    logger.propagate = False  # Don't pass logs to root logger

    # Common format for both console and file
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    # Console handler → only show important messages
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    # File handler → keep full debug history, rotate to save space
    file = RotatingFileHandler(
        "logs/app.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=3,  # keep last 3 logs
    )
    file.setLevel(logging.DEBUG)
    file.setFormatter(formatter)

    # Attach handlers to logger
    logger.addHandler(console)
    logger.addHandler(file)

    return logger
