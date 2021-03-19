import os
import structlog
import pandas as pd
import numpy as np
from tensorflow import keras



logger = structlog.get_logger()


def prepare_data(config):
    logger.info(f" - Prepare dataset [ {config['DATASET']} ]")

    dataset = None
    if config['DATASET_TYPE'] == 'CSV':
        dataset = load_csv_dataset(config)

    (train, valid, test) = dataset
    describe_data(train)

    return dataset


def load_csv_dataset(config):
    logger.info(f" - '{config['DATASET']}' is now loading...")

    test_csv = read_csv(config, 'test')
    train_csv = read_csv(config, 'train')
    valid_csv = read_csv(config, 'valid')
    output_csv = read_csv(config, 'output')

    return (train_csv, valid_csv, test_csv)


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


"""
def train_valid_split(config, data_path):
    logger.info(f" - splitting valid set split_rate [{config['SPLIT_RATE']}]")
    
    train = pd.read_csv(data_path['train'])
    split_idx = int(len(train) * config['SPLIT_RATE'])
    train_data = train[:split_idx]
    valid_data = train[split_idx:]

    return train_data, valid_data

"""