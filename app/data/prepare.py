import structlog
from tensorflow import keras


logger = structlog.get_logger()


def prepare_data(target_dataset):
    default_list = [
        'fashion_mnist'
    ]

    logger.info(f" - Prepare dataset [ {target_dataset} ]")

    if target_dataset in default_list:
        load_default_dataset(target_dataset)
    else:
        logger.warn("load non-default dataset still not developed")


def load_default_dataset(target_dataset):
    logger.info(f" - '{target_dataset}' is now loading...")
    load_dataset = {
        "fashion_mnist" : load_fashion_mnist
    }[target_dataset]

    (train_data, train_labels), (test_data, test_labels) = load_dataset()
    logger.info(f" - '{target_dataset}' is loaded")
    return (train_data, train_labels), (test_data, test_labels)


def load_fashion_mnist():
    return keras.datasets.fashion_mnist.load_data()
