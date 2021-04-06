import os
from app.utils.logger import Logger
from app.exc.exc import HANException, QuitException
from app.data.prepare import prepare_data, prepare_meta
from app.data.analyze import analize_dataset, analize_feature
from app.data.clean import clean_empty_label, clean_duplicate, clean_null_column
from app.data.check import check_cardinal_values, check_skewness_kurtosis, check_data
from app.feature.split import split_label_feature, split_train_valid, shuffle_train_data
from app.feature.select import select_feature
from app.feature.encoding import one_hot_encoding
from app.feature.scale import scale_feature
from app.model.model import build_model, fit_model, estimate_model


logger = Logger()

def run():
    # TODO: Load config
    DATASET = 'house-prices-advanced-regression-techniques'
    config = {
        'DATASET': DATASET,
        'DATASET_PATH': os.path.join('D:', 'datasets', DATASET),
        'DESCRIPT_PATH': 'description.txt',
        'DATASET_TYPE': 'CSV',
        'INDEX_LABEL': 'Id',
        'TARGET_LABEL': ['SalePrice'],
        'CLEAN_EMPTY_LABEL': True,
        'CLEAN_DUPLICATE': True,
        'CLEAN_NULL_THRESHOLD': 0.5,
        'CARDINAL_THRESHOLD': [5, 9],
        'SPLIT_RATE': 0.8,
        'DATA_SHUFFLE': True,
        'MODEL': 'XGBoost',
        'options': {
            'FIX_COLUMN_INFO': True,
            'FIX_COLUMN_AUTO': True,
        },
        'info': None,
        'meta': dict(),
    }

    try:
        logger.log("Step 1 >> Data Preparation")
        logger.log("- 1 : Data Collection ", level=1)
        dataset = prepare_data(config)
        metaset = prepare_meta(config, dataset)

        dataset = analize_dataset(config, dataset, metaset)

        dataset = analize_feature(config, dataset, metaset)

        logger.log("- 2 : Data Cleaning")
        # dataset = clean_duplicate(config, dataset)
        # dataset = clean_empty_label(config, dataset)
        dataset['train'] = clean_null_column(config, dataset['train'])

        logger.log("- 3 : Data Check")
        dataset['train'] = check_cardinal_values(config, dataset['train'])
        dataset['train'] = check_skewness_kurtosis(config, dataset['train'])

        logger.log("- 4 : Data Augmentation")

        logger.log("Step 2 >> Feature Engineering")
        logger.log("- 2.1 : Feature Selection")
        dataset['train'] = select_feature(config, dataset['train'])
        dataset['train'] = one_hot_encoding(config, dataset['train'])
        # dataset = check_cardinal_values(config, dataset)

        logger.log("- 2.2 : Feature Extraction")
        logger.log("- 2.3 : Feature Construction")
        logger.log("- 2.4 : Feature Scaling")

        dataset = split_label_feature(config, dataset)
        dataset = scale_feature(config, dataset)

        # dataset = shuffle_train_data(config, dataset)

 
        logger.log("Step 3 >> Model Generation")
        logger.log("- 3.1 : Neural Architecture")
        model = build_model(config)

        logger.log("- 3.2 : Architecture optimization")

        logger.log("Step 4 >> Model Setting")
        logger.log("- 4.0 : Data Split")
        # train, valid = split_train_valid(config, dataset)

        logger.log("Step 5 >> Model Evaluation")
        model = fit_model(model, dataset)

        logger.log("Step 6 >> Output")
        estimate_model(model)


    except QuitException:
        logger.warn(f'Quit!')
        return
    except KeyboardInterrupt:
        logger.warn(f'Abort!')
        return
    except HANException as e:
        logger.warn(f"Custom Exception raised : {e.message}")
        return
