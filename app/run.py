import structlog
from app.exc.exc import HANException
from app.data.prepare import prepare_data


logger = structlog.get_logger()


def run():
    # TODO: Load config
    target_dataset = "fashion_mnist"
    
    try:
       logger.info("Step 1 >> Data Preparation")
       prepare_data(target_dataset)
       logger.info(" - 1.1 : Data Collection ")
       logger.info(" - 1.2 : Data Cleaning")
       logger.info(" - 1.3 : Data Augmentation")


       logger.info("Step 2 >> Feature Engineering")

       logger.info(" - 2.1 : Feature Selection")
       logger.info(" - 2.2 : Feature Extraction")
       logger.info(" - 2.3 : Feature Construction")

       logger.info("Step 3 >> Model Generation")
       logger.info(" - 3.1 : Neural Architecture")
       logger.info(" - 3.2 : Architecture optimization")

       logger.info("Step 4 >> Model Setting")

       logger.info("Step 5 >> Model Evaluation")

       logger.info("Step 6 >> Output")

    except HANException as e:
        logger.warn(f"Custom Exception raised : {e.message}")
        return
