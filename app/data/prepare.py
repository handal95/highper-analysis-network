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
        csvfile = load_csv_dataset(config, dataset)
        dataset = pd.concat(csvfile, axis=0)

    describe_data(dataset)

    return dataset


def load_csv_dataset(config, dataset):
    logger.info(f" - '{config['DATASET']}.csv' is now loading...")

    train_set = read_csv(config, 'train')
    test_set = read_csv(config, 'test')
    valid_set = read_csv(config, 'valid')

    return [train_set, test_set, valid_set]
    

def read_csv(config, category):
    file_path = os.path.join(config['DATASET_PATH'], f"{category}.csv")
    try:
        csv_file = pd.read_csv(file_path, index_col = config['INDEX_LABEL'])            
        csv_file["_SET_"] = category
        logger.info(f"   - {category:5} shape : {csv_file.shape}")

        return csv_file
    except FileNotFoundError:
        return None


def describe_data(dataset):
    describe = dataset.describe(percentiles=[.03, .25, .50, .75, .97]).T
    logger.info(f" - DATA describe, len: {len(dataset)} \n{describe}")
