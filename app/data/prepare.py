import os
import structlog
import pandas as pd
from tensorflow import keras



logger = structlog.get_logger()


def prepare_data(config):
    default_list = [
        'fashion_mnist'
    ]

    logger.info(f" - Prepare dataset [ {config['DATASET']} ]")

    if config['DATASET'] in default_list:
        dataset = load_default_dataset(config['DATASET'])
    else:
        dataset = load_custom_dataset(config)


def load_default_dataset(config):
    logger.info(f" - '{config['DATASET']}' is now loading...")
    load_dataset = {
        "fashion_mnist" : load_fashion_mnist
    }[target_dataset]

    (train_data, train_labels), (test_data, test_labels) = load_dataset()
    logger.info(f" - '{target_dataset}' is loaded")
    return (train_data, train_labels), (test_data, test_labels)


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

    logger.info(
        f" - '{config['DATASET']}' is loaded \n"
        f" Train {len(train_data)} rows, "
        f" Valid {len(valid_data)} rows, "
        f" Test  {len(test_data)} rows"
    )
    return train_data, valid_data, test_data
    

def train_valid_split(config, data_path):
    logger.info(f" - splitting valid set split_rate [{config['SPLIT_RATE']}]")
    
    train = pd.read_csv(data_path['train'])
    split_idx = int(len(train) * config['SPLIT_RATE'])
    train_data = train[:split_idx]
    valid_data = train[split_idx:]

    return train_data, valid_data



def load_fashion_mnist():
    return keras.datasets.fashion_mnist.load_data()
