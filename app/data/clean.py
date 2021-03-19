import structlog


logger = structlog.get_logger()


def clean_empty_label(config, dataset):
    if not config['CLEAN_EMPTY_LABEL']:
        logger.info(f" - Skipping Duplicate data Cleaning...")
        return dataset
    
    logger.info(f" - Empty labled data Cleaning...")
    (train_data, valid_data, test_data) = dataset

    train_data = remove_empty_label(config, train_data, 'train data')
    valid_data = remove_empty_label(config, valid_data, 'valid data')
    
    return dataset


def clean_duplicate(config, dataset):
    if not config['CLEAN_DUPLICATE']:
        logger.info(f" - Skipping Duplicate data Cleaning...")
        return dataset

    logger.info(f" - Duplicate data Cleaning...")
    (train_data, valid_data, test_data) = dataset

    train_data = remove_duplicate(train_data, 'train data')
    valid_data = remove_duplicate(valid_data, 'valid data')
    test_data = remove_duplicate(test_data, 'test data')

    return (train_data, valid_data, test_data)


def remove_duplicate(dataset, name):
    if dataset is None:
        return None

    raw_length = len(dataset)
    dataset = dataset.drop_duplicates()
    removed_lines = len(dataset) - raw_length

    cleaning_log('duplicated data', name, removed_lines)

    return dataset


def remove_empty_label(config, dataset, name):
    if dataset is None:
        return None

    raw_length = len(dataset)
    dataset = dataset[dataset['SalePrice'] != None]
    removed_lines = len(dataset) - raw_length

    cleaning_log('empty labeled data', name, removed_lines)

    return dataset


def cleaning_log(target, name, removed_lines=0):
    if removed_lines > 0:
        logger.info(f"    - {removed_lines} {target} removed from {name}")
    else:
        logger.info(f"    - {name:10} doesn't have any {target}")


def clean_data(config, dataset):
    logger.info(f" - Cleaning dataset [ {config['DATASET']} ]")
    (train_data, valid_data, test_data) = dataset
    
    DATA_LEN  = len(train_data)
    THRESHOLD_RATE = 0.1
    for col in train_data.columns:
        if train_data[col].dtype != 'object':
            continue

        unique_count = len(train_data[col].unique())
        logger.info(f"Norminal column [{col}] {unique_count}/{DATA_LEN}")
        # if len(train_data[col].unique()) >= threshold:
        #     train_data = train_data.drop(columns=col)
        #     test_data = test_data.drop(columns=col)
        #     if valid_data:
        #         valid_data = valid_data.drop(columns=col)
        # else:
        #     logger.info(f"Norminal column [{col}] is not dropped"
        #                 f"{unique_count}/{DATA_LEN}")
            

    return (train_data, valid_data, test_data)
