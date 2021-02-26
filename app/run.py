import structlog
from app.exc.exc import HANException
from app.data.prepare import prepare_data


logger = structlog.get_logger()


def run():
    # TODO: Load config
    target_dataset = "fashion_mnist"
    
    try:
       logger.info("Step 1 >> Data Preparing")
       prepare_data(target_dataset)

       logger.info("Step 2 >> Feature Engineering")

       logger.info("Step 3 >> Model Generation")

       logger.info("Step 4 >> Model Setting")

       logger.info("Step 5 >> Model Evaluation")

       logger.info("Step 6 >> Output")

    except HANException as e:
        logger.warn(f"Custom Exception raised : {e.message}")
        return
