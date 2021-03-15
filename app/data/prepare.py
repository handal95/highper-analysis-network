import os
import structlog
import pandas as pd
from tensorflow import keras



logger = structlog.get_logger()


def prepare_data(config):
    logger.info(f" - Prepare dataset [ {config['DATASET']} ]")

    dataset = load_custom_dataset(config)
    train_set, valid_set, test_set = data_label_split(config, dataset)

    return train_set, valid_set, test_set


def load_custom_dataset(config):
    logger.info(f" - '{config['DATASET']}' is now loading...")

    if config['DATASET_TYPE'] == 'CSV':
        data_path = {
            'train': os.path.join(config['DATASET_PATH'], 'train.csv'),
            'valid': os.path.join(config['DATASET_PATH'], 'valid.csv'),
            'test' : os.path.join(config['DATASET_PATH'], 'test.csv'),
            'output': os.path.join(config['DATASET_PATH'], 'output.csv')
        }

        test_data = pd.read_csv(data_path['test'])
        if os.path.exists(data_path['valid']):
            train_data = pd.read_csv(data_path['test'])
            valid_data = pd.read_csv(data_path['test'])
        else:
            train_data, valid_data = train_valid_split(config, data_path)

        describe = train_data.describe(
            percentiles=[.03, .25, .50, .75, .97]
        ).T 
        logger.info(f" - DATA describe \n{describe}")

    logger.info(
        f" - '{config['DATASET']}' is loaded \n"
        f" Train {len(train_data)} rows, "
        f" Valid {len(valid_data)} rows, "
        f" Test  {len(test_data)} rows"
    )
    return (train_data, valid_data, test_data)
    

def train_valid_split(config, data_path):
    logger.info(f" - splitting valid set split_rate [{config['SPLIT_RATE']}]")
    
    train = pd.read_csv(data_path['train'])
    split_idx = int(len(train) * config['SPLIT_RATE'])
    train_data = train[:split_idx]
    valid_data = train[split_idx:]

    return train_data, valid_data


def data_label_split(config, dataset):
    logger.info(f" - splitting label from data [{config['TARGET_LABEL']}]")
    (train, valid, test) = dataset

    train_label = train[config['TARGET_LABEL']]
    train_data = train.drop(columns=config['TARGET_LABEL'])

    valid_label = valid[config['TARGET_LABEL']]
    valid_data = valid.drop(columns=config['TARGET_LABEL'])

    test_label = None
    test_data = test

    return (train_data, train_label), (valid_data, valid_label), (test_data, test_label)

