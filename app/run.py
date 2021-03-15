import os
import structlog
from app.exc.exc import HANException
from app.data.prepare import prepare_data
from app.data.clean import clean_data
from app.model.model import build_model


logger = structlog.get_logger()


def run():
    # TODO: Load config
    config = {
        'DATASET': 'pubg-finish-placement-prediction',
        'DATASET_PATH': os.path.join('D:', 'datasets', 'pubg-finish-placement-prediction'),
        'DATASET_TYPE': 'CSV',
        'TARGET_LABEL': 'winPlacePerc',
        'SPLIT_RATE': 0.8,
        'MODEL': 'CatBoost'
    }

    try:
        logger.info("Step 1 >> Data Preparation")
        logger.info(" - 1.1 : Data Collection ")
        dataset = prepare_data(config)

        logger.info(" - 1.2 : Data Cleaning")
        dataset = clean_data(config, dataset)

        logger.info(" - 1.3 : Data Augmentation")

        logger.info("Step 2 >> Feature Engineering")
        logger.info(" - 2.1 : Feature Selection")
        logger.info(" - 2.2 : Feature Extraction")
        logger.info(" - 2.3 : Feature Construction")

 
        logger.info("Step 3 >> Model Generation")
        logger.info(" - 3.1 : Neural Architecture")
        # model = build_model(config)

        logger.info(" - 3.2 : Architecture optimization")

        logger.info("Step 4 >> Model Setting")

        logger.info("Step 5 >> Model Evaluation")

        logger.info("Step 6 >> Output")

    except HANException as e:
        logger.warn(f"Custom Exception raised : {e.message}")
        return
