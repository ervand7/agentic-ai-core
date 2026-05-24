"""Logging setup for the application."""

import logging


def setup_logging() -> None:
    """
    Configure a simple production-style logger.
    In real deployments, logs are usually shipped to a centralized platform.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
