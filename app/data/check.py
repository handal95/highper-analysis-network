import structlog
import numpy as np
import pandas as pd

logger = structlog.get_logger()


def check_null_values(config, dataset):
    logger.info(f" - Check Null value data...")
    (train_data, valid_data, test_data) = dataset

    train_null = get_null_values(train_data, 'train')
    test_null = get_null_values(test_data, 'test')

    null_display = pd.merge(train_null, test_null, how='outer', on='Features')
    logger.info(f"\n{null_display}")


def get_null_values(dataset, name):
    if dataset is None:
        return None

    na_info_data = []
    for col in dataset.columns:
        na_count = dataset[col].isna().sum()
        if na_count > 0:
            na_percent = np.round((100 * (na_count)/len(dataset)), 2)            
            na_info ={
                'Features' : col,
                f'NA_{name} (count)': na_count,
                f'NA_{name} (%)': na_percent
            }
            na_info_data.append(na_info)

    data_frame = pd.DataFrame(na_info_data, index=None).sort_values(by=f'NA_{name} (count)', ascending=False)

    return data_frame
