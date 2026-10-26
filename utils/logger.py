import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


def get_logger(name: str = "priceoracle", level: int = logging.INFO) -> logging.Logger:
    """Return a console logger, configuring it only once per name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
