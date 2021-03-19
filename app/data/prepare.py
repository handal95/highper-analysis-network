import os
import structlog
import pandas as pd
import numpy as np
from tensorflow import keras



logger = structlog.get_logger()


def prepare_data(config):
    logger.info(f" - Prepare dataset [ {config['DATASET']} ]")

    dataset = {'train': None, 'valid': None,'test': None}
    if config['DATASET_TYPE'] == 'CSV':
        dataset = load_csv_dataset(config, dataset)

    describe_data(dataset['train'])

    return dataset


def load_csv_dataset(config, dataset):
    logger.info(f" - '{config['DATASET']}' is now loading...")

    dataset['train'] = read_csv(config, 'train')
    dataset['test'] = read_csv(config, 'test')
    dataset['valid'] = read_csv(config, 'valid')
    dataset['train'] = pd.concat([dataset['train'], dataset['valid']], axis=0)

    return dataset


def read_csv(config, category):
    file_path = os.path.join(config['DATASET_PATH'], f"{category}.csv")
    try:
        csv_file = pd.read_csv(file_path, index_col = config['INDEX_LABEL'])
        csv_file.describe()
        logger.info(f"   - {category:5} shape : {csv_file.shape}")
        return csv_file
    except FileNotFoundError:
        return None


def describe_data(data):
    describe = data.describe(percentiles=[.03, .25, .50, .75, .97]).T
    logger.info(f" - DATA describe \n{describe}")
