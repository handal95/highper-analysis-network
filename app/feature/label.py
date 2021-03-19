import structlog

logger = structlog.get_logger()

def split_label_feature(config, dataset):
    logger.info(f"   - Splitting target label")
    (train, valid, test) = dataset
    
    (train_data, train_label) = split_label(config, train, 'train')
    (valid_data, valid_label) = split_label(config, valid, 'valid')
    (test_data, test_label) = split_label(config, test, 'test')

    return ((train_data, train_label), (valid_data, valid_label), (test_data, test_label))


def split_label(config, dataset, name):
    if dataset is None:
        return (None, None)

    try:
        label = dataset[config['TARGET_LABEL']]
        dataset = dataset.drop(columns=config['TARGET_LABEL'])
        logger.info(f"        {name:5} Label shape {label.shape}, Dataset shape {dataset.shape}")
    except:
        if name == 'test':
            logger.info(f"        {name:5} doesn't have [{config['TARGET_LABEL']}] label column")
            return (dataset, None)
        else:
            raise

    return (dataset, label)
        