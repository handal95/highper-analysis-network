import structlog 


logger = structlog.get_logger()


def run():
    log = logger.bind()
    log.info("Step 1 >> Data Preparing")

    log.info("Step 2 >> Feature Engineering")

    log.info("Step 3 >> Model Generation")

    log.info("Step 4 >> Model Setting")

    log.info("Step 5 >> Model Evaluation")

    log.info("Step 6 >> Output")
