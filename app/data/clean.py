import structlog
import numpy as np
import pandas as pd


logger = structlog.get_logger()


def clean_duplicate(config, dataset):
    if not config['CLEAN_DUPLICATE']:
        logger.info(f" - Skipping Duplicate data Cleaning...")
        return dataset

    raw_length = len(dataset)
    dataset = dataset.drop_duplicates()
    removed_lines = len(dataset) - raw_length

    logger.info(f"    - Cleaning Duplicated data, {removed_lines} removed")

    return dataset


def clean_empty_label(config, dataset):
    if not config['CLEAN_EMPTY_LABEL']:
        logger.info(f" - Skipping Duplicate data Cleaning...")
        return dataset
    
    raw_length = len(dataset)
    dataset = dataset[dataset[config['TARGET_LABEL']] != None]
    removed_lines = len(dataset) - raw_length

    logger.info(f"    - Cleaning Empty labeled data, {removed_lines} removed")

    return dataset


def clean_null_column(config, dataset):
    logger.info(f" - Check NA value column...")

    na_info_data = []
    removed_col = []
    for col in dataset.columns.drop(config["TARGET_LABEL"]):
        na_count = dataset[col].isna().sum()
        if na_count > 0:
            na_percent = np.round((100 * (na_count)/len(dataset)), 2)
            na_removed = na_percent > config['CLEAN_NULL_THRESHOLD'] * 100   
            na_info ={
                'Features' : col,
                f'NA (count)': na_count,
                f'NA (%)': na_percent,
                f'remove': na_removed
            }
            if na_removed:
                removed_col.append(col)
            na_info_data.append(na_info)

    dataset.drop(columns=removed_col)
    data_frame = pd.DataFrame(na_info_data, index=None).sort_values(by=f'NA (count)', ascending=False)
    logger.info(f"\n{data_frame}")

    return dataset
