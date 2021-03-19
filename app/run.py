import os
import structlog
from app.exc.exc import HANException
from app.data.prepare import prepare_data
from app.data.clean import clean_empty_label, clean_duplicate
from app.data.check import check_null_values, check_cardinal_values
from app.model.model import build_model


logger = structlog.get_logger()


def run():
    # TODO: Load config
    DATASET = 'house-prices-advanced-regression-techniques'
    config = {
        'DATASET': DATASET,
        'DATASET_PATH': os.path.join('D:', 'datasets', DATASET),
        'DATASET_TYPE': 'CSV',
        'INDEX_LABEL': 'Id',
        'TARGET_LABEL': 'SalePrice',
        'SPLIT_RATE': 0.8,
        'CLEAN_EMPTY_LABEL': True,
        'CLEAN_DUPLICATE': True,
        'CARDINAL_THRESHOLD': [6, 9],
        'MODEL': 'CatBoost'
    }

    try:
        logger.info("Step 1 >> Data Preparation")
        logger.info(" - 1.1 : Data Collection ")
        dataset = prepare_data(config)

        logger.info(" - 1.2 : Data Cleaning")
        dataset = clean_duplicate(config, dataset)
        dataset = clean_empty_label(config, dataset)

        logger.info(" - 1.3 : Data Check")
        dataset = check_null_values(config, dataset)
        dataset = check_cardinal_values(config, dataset)

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
