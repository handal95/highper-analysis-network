import sys
import logging
import structlog
from app.run import run


def set_logger():
    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=logging.INFO
    )
    structlog.configure(
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


if __name__ == '__main__':
    set_logger()
    run()
