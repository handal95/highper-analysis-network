import sys
import logging
import structlog


def init_logger():
    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=logging.INFO)
    structlog.configure(
        logger_factory=structlog.stdlib.LoggerFactory())