import os
import structlog
from app.exc.exc import HANException
from app.data.prepare import prepare_data
from app.data.clean import clean_empty_label, clean_duplicate, clean_null_column
from app.data.check import check_cardinal_values, check_skewness_kurtosis
from app.feature.split import split_label_feature, split_train_valid, shuffle_train_data
from app.feature.select import select_feature
from app.feature.encoding import one_hot_encoding
from app.feature.scale import scale_feature
from app.model.model import build_model, fit_model, estimate_model


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
        'CLEAN_EMPTY_LABEL': True,
        'CLEAN_DUPLICATE': True,
        'CLEAN_NULL_THRESHOLD': 0.5,
        'CARDINAL_THRESHOLD': [5, 9],
        'SPLIT_RATE': 0.8,
        'DATA_SHUFFLE': True,
        'MODEL': 'XGBoost'
    }

    try:
        logger.info("Step 1 >> Data Preparation")
        logger.info(" - 1.1 : Data Collection ")
        dataset = prepare_data(config)

        logger.info(" - 1.2 : Data Cleaning")
        dataset = clean_duplicate(config, dataset)
        dataset = clean_empty_label(config, dataset)
        dataset = clean_null_column(config, dataset)

        logger.info(" - 1.3 : Data Check")
        dataset = check_cardinal_values(config, dataset)
        dataset = check_skewness_kurtosis(config, dataset)
        logger.info(" - 1.3 : Data Augmentation")

        logger.info("Step 2 >> Feature Engineering")
        logger.info(" - 2.1 : Feature Selection")
        dataset = select_feature(config, dataset)
        dataset = one_hot_encoding(config, dataset)
        # dataset = check_cardinal_values(config, dataset)

        logger.info(" - 2.2 : Feature Extraction")
        logger.info(" - 2.3 : Feature Construction")
        logger.info(" - 2.4 : Feature Scaling")

        dataset = split_label_feature(config, dataset)
        # dataset = scale_feature(config, dataset)

        # dataset = shuffle_train_data(config, dataset)

 
        logger.info("Step 3 >> Model Generation")
        logger.info(" - 3.1 : Neural Architecture")
        model = build_model(config)

        logger.info(" - 3.2 : Architecture optimization")

        logger.info("Step 4 >> Model Setting")
        logger.info(" - 4.0 : Data Split")
        train, valid = split_train_valid(config, dataset)

        logger.info("Step 5 >> Model Evaluation")
        model = fit_model(model, train, valid)

        logger.info("Step 6 >> Output")
        estimate_model(model)

    except HANException as e:
        logger.warn(f"Custom Exception raised : {e.message}")
        return
