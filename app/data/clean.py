import structlog


logger = structlog.get_logger()


def clean_duplicate(config, dataset):
    if not config['CLEAN_DUPLICATE']:
        logger.info(f" - Skipping Duplicate data Cleaning...")
        return dataset

    raw_length = len(dataset)
    dataset = dataset.drop_duplicates()
    removed_lines = len(dataset) - raw_length

    cleaning_log('Duplicated data', removed_lines)

    return dataset


def clean_empty_label(config, dataset):
    if not config['CLEAN_EMPTY_LABEL']:
        logger.info(f" - Skipping Duplicate data Cleaning...")
        return dataset
    
    logger.info(f" - Empty labled data Cleaning...")

    raw_length = len(dataset)
    dataset = dataset[dataset[config['TARGET_LABEL']] != None]
    removed_lines = len(dataset) - raw_length

    cleaning_log('Empty label data', removed_lines)

    return dataset



def cleaning_log(target, removed_lines=0):
    if removed_lines > 0:
        logger.info(f"    - {removed_lines} {target} removed")
    else:
        logger.info(f"    - Dataset doesn't have any {target}")
