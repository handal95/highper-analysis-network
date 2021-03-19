import structlog

logger = structlog.get_logger()

def split_label_feature(config, dataset):
    logger.info(f"   - Splitting target label")
    
    (train_data, train_label) = split_label(config, dataset['train'], 'train')
    (test_data, test_label) = split_label(config, dataset['test'], 'test')

    return ((train_data, train_label), (test_data, test_label))


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
        

def split_train_valid(config, dataset):
    logger.info(f" - splitting valid set split_rate [{config['SPLIT_RATE']}]")
    
    (train, test) = dataset
    split_idx = int(len(train) * config['SPLIT_RATE'])
    train_data = train[:split_idx]
    valid_data = train[split_idx:]

    return train_data, valid_data