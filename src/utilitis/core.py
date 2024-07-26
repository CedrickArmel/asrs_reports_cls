"""This module defines the core utilitis for the project."""
import logging


def get_logger(name: str) -> logging.Logger:
    """
    Template for getting a logger.

    Args:
        name: Name of the logger.

    Returns: Logger.
    """

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s | %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger(name)
    return logger
