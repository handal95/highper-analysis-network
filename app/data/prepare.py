import os
import structlog
import pandas as pd
from tensorflow import keras



logger = structlog.get_logger()


def prepare_data(config):
    logger.info(f" - Prepare dataset [ {config['DATASET']} ]")

    dataset = None
    if config['DATASET_TYPE'] == 'CSV':
        dataset = load_csv_dataset(config)

    return dataset


def load_csv_dataset(config):
    logger.info(f" - '{config['DATASET']}' is now loading...")

    data_path = {
        'train': os.path.join(config['DATASET_PATH'], 'train.csv'),
        'valid': os.path.join(config['DATASET_PATH'], 'valid.csv'),
        'test' : os.path.join(config['DATASET_PATH'], 'test.csv'),
        'output': os.path.join(config['DATASET_PATH'], 'output.csv')
    }

    test_csv = pd.read_csv(data_path['test'])
    train_csv = pd.read_csv(data_path['train'])
    valid_csv = pd.read_csv(data_path['valid']) if os.path.exists(data_path['valid']) else None

    describe = train_csv.describe(
        percentiles=[.03, .25, .50, .75, .97]
    ).T 
    logger.info(f" - DATA describe \n{describe}")

    return (train_csv, valid_csv, test_csv)
    
"""
def train_valid_split(config, data_path):
    logger.info(f" - splitting valid set split_rate [{config['SPLIT_RATE']}]")
    
    train = pd.read_csv(data_path['train'])
    split_idx = int(len(train) * config['SPLIT_RATE'])
    train_data = train[:split_idx]
    valid_data = train[split_idx:]

    return train_data, valid_data

"""
def data_label_split(config, dataset, is_train=False):
    try:
        label = dataset[config['TARGET_LABEL']]
        dataset = dataset.drop(columns=config['TARGET_LABEL'])
    except:
        if is_train:
            return (dataset, None)
        else:
            raise

    return (dataset, label)
    